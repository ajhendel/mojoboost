"""Shared controls for multicore CPU dispatch.

Two environment variables govern every parallel loop in the CPU backend
(histogram accumulation, split scanning, bin fitting, bin transform,
gradient generation, row partitioning, prediction), for reproducible
benchmarking and for tests that must force one path:

- `MOJOTREES_NUM_WORKERS`: `1` forces the serial path; `N > 1` forces the
  parallel path with the work chunked into at most N tasks regardless of
  input size; `0`, unset, or unparsable means auto (task count derived from
  the machine's core count once the work is large enough to amortize
  scheduling).
- `MOJOTREES_PARALLEL_MIN_OPS`: integer override of the auto-mode work
  threshold (`PARALLEL_MIN_OPS`), compared against the caller's op estimate
  (conventionally items * rows touched).

`binning.mojo` adds one more of the same kind:
`MOJOTREES_BINNING_SELECT_MIN_ROWS` chooses between the two ways a quantile
fit resolves its order statistics (rank selection or a full sort). Both
resolve the same values, so it moves no edge and no bin.

`apple_cpu_policy.mojo` adds four more, all scheduling-only and all
documented there: `MOJOTREES_CPU_TASKS_PER_CORE`, `MOJOTREES_CPU_CORE_POOL`,
`MOJOTREES_CPU_FEATURE_GROUP`, and `MOJOTREES_CPU_COMPACT_MIN_ROWS`. The two
above override them: an explicit worker count bypasses the grain floor and
the core cap alike.

The GPU backend adds its own variables, documented where they are read:
`MOJOTREES_GPU_HIST_STRATEGY` (gpu_tiling.mojo),
`MOJOTREES_GPU_SPLIT_STRATEGY` (train_gpu.mojo; `device` moves per-node
split selection onto the accelerator, `host`, unset, or unrecognized keeps
the host scan), `MOJOTREES_GPU_VERIFY_ROWS` (gpu_active_rows.mojo; `1`
makes every device row partition verify its left count against the
grower's histogram count, one host synchronization per split, off by
default), `MOJOTREES_GPU_TRACE` and `MOJOTREES_GPU_STAGING_SLOTS`
(gpu_runtime.mojo).

Four dispatch shapes are provided. All of them keep every floating-point
summation order independent of the task count, so every result is
bit-identical to the serial path on every machine and at every worker
setting:

- `dispatch_feature_ranges` hands each task a contiguous half-open range of
  independent units (conventionally features). A unit's accumulation runs
  start to finish inside one task, so no sum is ever reassociated, and a
  kernel that wants to interleave two units for instruction-level
  parallelism gets them in the same call rather than in two.
- `dispatch_features` is the one-unit-at-a-time form of the same thing,
  written in terms of it.
- `dispatch_feature_rows` splits by feature *and*, when there are fewer
  features than tasks, by rows within each feature. Elementwise callers
  only: it exists so that a matrix with four features over five million
  rows can use more than four workers, which neither of the two above can
  do. With features to spare it is `dispatch_feature_ranges` verbatim.
- `dispatch_rows` splits a row range into contiguous ascending blocks.
  Callers use it only for elementwise work (gradients, predictions, bin
  lookups, sibling subtraction) and for counting passes, where disjoint
  in-order blocks reproduce the serial result exactly. It is deliberately
  not used for histogram accumulation, which would need a cross-block
  reduction.

Task count comes from the workload shape rather than from the item count.
One task per item balances perfectly but pays a scheduling event per item,
which dominates on wide inputs, so auto mode caps the fan-out at
`TASKS_PER_CORE` tasks per core and chunks the rest. Which cores are counted,
and how the fan-out reacts to a machine whose cores are not all the same
speed, is `apple_cpu_policy.mojo`'s decision; this module only applies the
grain rule and the split.

Ranges are split evenly rather than by a ceiling-divided chunk: task `w` of
`n` takes `[w * items // n, (w + 1) * items // n)`. Two task counts that
differ by one item never leave a trailing task empty, which a ceiling chunk
does (6 tasks over 10 features leaves the sixth with nothing to do), and the
spread makes the remainder land on different tasks instead of all on the
last. Which task runs which unit changes nothing about the result.

Nesting. These dispatches are not reentrant-aware: a `sync_parallelize`
inside a task would oversubscribe the machine with no scheduler to arbitrate
it. Every caller in this package dispatches from the single-threaded part of
its stage, and a kernel that runs inside a task uses the serial helper
instead (`Histogram.reset` rather than a dispatching zero pass, for
example). Anything new that runs inside a task must do the same.
"""

from max.algorithm import sync_parallelize
from std.os import getenv

from .apple_cpu_policy import DEFAULT_TASKS_PER_CORE, cpu_profile

# Serial-vs-parallel crossover measured at 25k-50k ops on Apple M4, AMD
# Zen4, and Neoverse-N2 (bench/bench_threshold.mojo); 1 << 16 sits above
# all three with 1.2-1.6x parallel speedup at exactly this size.
comptime PARALLEL_MIN_OPS = 1 << 16

# Auto-mode fan-out per core. The value, the core pool it multiplies, and the
# reasoning behind both now live in `apple_cpu_policy.mojo`; this name stays
# because benchmarks and tests print and bound it. Sweep it with
# bench/bench_profile.mojo (or `MOJOTREES_CPU_TASKS_PER_CORE`) on an idle
# machine before treating it as tuned.
comptime TASKS_PER_CORE = DEFAULT_TASKS_PER_CORE


def _env_int(name: String, default: Int) -> Int:
    var s = getenv(name)
    if s.byte_length() == 0:
        return default
    try:
        var n = Int(s)
        if n < 0:
            return default
        return n
    except:
        return default


def env_num_workers() -> Int:
    """0 = auto, 1 = serial, N > 1 = force N-way chunked parallelism."""
    return _env_int("MOJOTREES_NUM_WORKERS", 0)


def env_parallel_min_ops() -> Int:
    return _env_int("MOJOTREES_PARALLEL_MIN_OPS", PARALLEL_MIN_OPS)


def plan_tasks(n_items: Int, total_ops: Int) -> Int:
    """How many parallel tasks to split `n_items` units of work into.

    Returns 1 for the serial path. The result is a scheduling decision only:
    every caller produces the same answer at every task count, so this can be
    tuned freely without changing any output.

    `total_ops` is the caller's work estimate in *histogram-op equivalents*:
    one accumulate of a gradient, a hessian, and a count into a scattered bin.
    Stages whose per-row work is cheaper or dearer than that scale their
    estimate accordingly (see the callers), because a single count of "rows"
    or "features" says nothing about how long a task will run, and the whole
    point of the threshold is to compare work against scheduling overhead.

    Two limits apply in auto mode. Below one grain of work the whole loop
    stays serial, and above it the task count is capped so that no task holds
    less than a grain: fanning 100k cheap ops across 40 workers costs far more
    in scheduling than the 2.5k ops each one would run. An explicit
    `MOJOTREES_NUM_WORKERS` bypasses both, so tests can force the parallel
    path at any size.

    The per-core ceiling comes from `apple_cpu_policy`, which decides how many
    cores to count on a machine whose cores are not all the same speed. Its
    default is every physical core times `TASKS_PER_CORE`, which is what this
    line has always computed.
    """
    if n_items <= 1:
        return 1
    var workers = env_num_workers()
    if workers == 1:
        return 1
    var n_tasks = workers
    if workers == 0:
        var grain = env_parallel_min_ops()
        if total_ops < grain:
            return 1
        n_tasks = cpu_profile().max_auto_tasks()
        var by_grain = total_ops // grain
        if by_grain < n_tasks:
            n_tasks = by_grain
        if n_tasks < 1:
            n_tasks = 1
    if n_tasks > n_items:
        n_tasks = n_items
    return n_tasks


def dispatch_feature_ranges[FuncType: def (Int, Int) -> None](
    func: FuncType, n_features: Int, total_ops: Int
) raises:
    """Run `func(start, end)` over contiguous ranges covering
    [0, n_features), under the env contract above.

    `func` must be safe to run concurrently for disjoint ranges: each feature
    writes only its own output range, so ranges of them do too. `total_ops`
    is the work estimate compared against the auto-mode threshold,
    conventionally n_features * rows_touched plus whatever else the kernel
    does per feature (see `apple_cpu_policy.derive_accumulation_plan`, which
    counts the zeroing pass a histogram build also runs).

    The serial path is a single `func(0, n_features)` call, which is the same
    shape a task gets, so there is one body to test rather than two. Handing
    a kernel a range rather than one feature at a time is what lets it
    interleave two features in one inner loop; it changes nothing about which
    feature accumulates what, only how many are in flight.
    """
    var n_tasks = plan_tasks(n_features, total_ops)
    if n_tasks <= 1:
        if n_features > 0:
            func(0, n_features)
        return

    # Even split: task w takes [w * n // tasks, (w + 1) * n // tasks). With
    # n_tasks <= n_features (plan_tasks clamps it) no task comes out empty,
    # and the remainder is spread instead of landing entirely on the last.
    def do_range(w: Int) {imm}:
        func(
            (w * n_features) // n_tasks,
            ((w + 1) * n_features) // n_tasks,
        )

    sync_parallelize(do_range, n_tasks)


def dispatch_features[FuncType: def (Int) -> None](
    func: FuncType, n_features: Int, total_ops: Int
) raises:
    """Run `func(f)` for f in [0, n_features) under the env contract above.

    The one-feature-at-a-time form of `dispatch_feature_ranges`, and written
    in terms of it so there is a single split rule to reason about. Use this
    unless the kernel has something to gain from seeing several features at
    once.
    """

    def do_range(start: Int, end: Int) {imm}:
        for f in range(start, end):
            func(f)

    dispatch_feature_ranges(do_range, n_features, total_ops)


def dispatch_feature_rows[FuncType: def (Int, Int, Int) -> None](
    func: FuncType, n_features: Int, n_rows: Int, total_ops: Int
) raises:
    """Run `func(f, row_start, row_end)` over tiles covering
    [0, n_features) x [0, n_rows), under the env contract above.

    For an elementwise kernel: one where the value written at (f, r) depends
    on that cell alone. `dispatch_features` splits such a kernel by feature
    and nothing else, so a matrix with four features cannot use more than
    four workers however many rows it has. Four features by five million rows
    is a real shape (a few engineered signals over a long history), and on
    that shape feature-only splitting leaves most of the machine idle.

    So the split falls back to rows only when it has to. With at least as
    many features as tasks, this *is* `dispatch_feature_ranges`, called
    rather than reimplemented: one task per contiguous run of features,
    exactly the schedule that shape has always had, and no new tasks to pay
    for. Below that, each feature's rows are cut into enough blocks to reach
    the task count, and the tiles are dispatched as one flat list.

    Tiling cannot move a result. Every tile writes only its own
    `(f, [row_start, row_end))` cells, reads only the matching input cells,
    and computes each from that cell alone, so the output is the same bytes
    at every task count and on every machine. That is a property of the
    kernel, not of the schedule, which is why this is restricted to
    elementwise callers: a kernel that reduces across rows (a column
    minimum, a rank, a histogram) needs a combine step and must not use this.

    `total_ops` is the usual work estimate in histogram-op equivalents. It is
    compared against the auto-mode threshold once, over the whole matrix, so
    a shape stays serial here on exactly the same grounds it would elsewhere.
    """
    var n_tasks = plan_tasks(n_features * n_rows, total_ops)
    if n_tasks <= 1:
        for f in range(n_features):
            if n_rows > 0:
                func(f, 0, n_rows)
        return

    if n_features >= n_tasks:
        # Enough features to keep every task busy: the established schedule,
        # reached through the established call so there is one split rule.
        def do_range(start: Int, end: Int) {imm}:
            for f in range(start, end):
                func(f, 0, n_rows)

        dispatch_feature_ranges(do_range, n_features, total_ops)
        return

    # Blocks per feature, rounded up, so `n_features * blocks >= n_tasks` and
    # no worker is left without a tile. Clamped to `n_rows` because a block
    # cannot be shorter than one row.
    var per_feature = (n_tasks + n_features - 1) // n_features
    if per_feature > n_rows:
        per_feature = n_rows
    if per_feature < 1:
        return
    var chunk = (n_rows + per_feature - 1) // per_feature
    # Recount from the chunk, as `plan_row_blocks` does: ceiling division can
    # leave a trailing block empty, and an empty tile is a task that costs
    # scheduling and does nothing.
    var n_blocks = (n_rows + chunk - 1) // chunk

    def do_tile(t: Int) {imm}:
        var f = t // n_blocks
        var b = t - f * n_blocks
        var r0 = b * chunk
        var r1 = r0 + chunk
        if r1 > n_rows:
            r1 = n_rows
        func(f, r0, r1)

    sync_parallelize(do_tile, n_features * n_blocks)


@fieldwise_init
struct RowBlocks(Copyable, Movable):
    """A split of [0, n_rows) into `n_blocks` contiguous ascending blocks.

    Made explicit, rather than hidden inside a dispatch call, because a
    two-pass parallel algorithm has to size its per-block scratch before it
    runs the first pass: the row partitioner counts per block, prefix-sums
    the counts, then scatters. `n_blocks == 1` is the serial plan.
    """

    var n_blocks: Int
    var chunk: Int
    var n_rows: Int

    @always_inline
    def start(self, b: Int) -> Int:
        return b * self.chunk

    @always_inline
    def end(self, b: Int) -> Int:
        var e = (b + 1) * self.chunk
        return self.n_rows if e > self.n_rows else e


def plan_row_blocks(n_rows: Int, total_ops: Int) -> RowBlocks:
    """Choose the block split for a row range under the env contract."""
    if n_rows <= 0:
        return RowBlocks(0, 1, 0)
    var n_tasks = plan_tasks(n_rows, total_ops)
    if n_tasks <= 1:
        return RowBlocks(1, n_rows, n_rows)
    var chunk = (n_rows + n_tasks - 1) // n_tasks
    # Ceiling division can leave trailing blocks empty (10 rows over 4 tasks
    # gives chunk 3, and block 3 would be empty); recount from the chunk so
    # every block has work and `n_blocks` is exact.
    return RowBlocks((n_rows + chunk - 1) // chunk, chunk, n_rows)


def run_row_blocks[FuncType: def (Int) -> None](
    blocks: RowBlocks, func: FuncType
) raises:
    """Run `func(b)` for each block id in `blocks`, in parallel when the plan
    has more than one block. `func` reads its own range via
    `blocks.start(b)` / `blocks.end(b)`."""
    if blocks.n_blocks <= 0:
        return
    if blocks.n_blocks == 1:
        func(0)
        return
    sync_parallelize(func, blocks.n_blocks)


def dispatch_rows[FuncType: def (Int, Int) -> None](
    func: FuncType, n_rows: Int, total_ops: Int
) raises:
    """Run `func(start, end)` over contiguous blocks covering [0, n_rows).

    Blocks are disjoint and each one walks its rows in ascending order, so
    elementwise callers get bit-identical results at every task count. The
    serial path is a single `func(0, n_rows)` call, which is the same shape a
    block gets, so there is only one body to test.
    """
    var blocks = plan_row_blocks(n_rows, total_ops)

    def do_block(b: Int) {imm}:
        func(blocks.start(b), blocks.end(b))

    run_row_blocks(blocks, do_block)
