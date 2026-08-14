"""Shared controls for multicore CPU dispatch.

Two environment variables govern every parallel loop in the CPU backend
(histogram accumulation, split scanning, bin fitting, bin transform,
gradient generation, row partitioning, prediction), for reproducible
benchmarking and for tests that must force one path:

- `MOJOBOOST_NUM_WORKERS`: `1` forces the serial path; `N > 1` forces the
  parallel path with the work chunked into at most N tasks regardless of
  input size; `0`, unset, or unparsable means auto (task count derived from
  the machine's core count once the work is large enough to amortize
  scheduling).
- `MOJOBOOST_PARALLEL_MIN_OPS`: integer override of the auto-mode work
  threshold (`PARALLEL_MIN_OPS`), compared against the caller's op estimate
  (conventionally items * rows touched).

The GPU backend adds its own variables, documented where they are read:
`MOJOBOOST_GPU_HIST_STRATEGY` (gpu_tiling.mojo),
`MOJOBOOST_GPU_SPLIT_STRATEGY` (train_gpu.mojo; `device` moves per-node
split selection onto the accelerator, `host`, unset, or unrecognized keeps
the host scan), `MOJOBOOST_GPU_VERIFY_ROWS` (gpu_active_rows.mojo; `1`
makes every device row partition verify its left count against the
grower's histogram count, one host synchronization per split, off by
default), `MOJOBOOST_GPU_TRACE` and `MOJOBOOST_GPU_STAGING_SLOTS`
(gpu_runtime.mojo).

Two dispatch shapes are provided. Both keep every floating-point summation
order independent of the task count, so every result is bit-identical to the
serial path on every machine and at every worker setting:

- `dispatch_features` splits independent units (conventionally features)
  across tasks. A unit's accumulation runs start to finish inside a single
  task, so no sum is ever reassociated.
- `dispatch_rows` splits a row range into contiguous ascending blocks.
  Callers use it only for elementwise work (gradients, predictions, bin
  lookups) and for counting passes, where disjoint in-order blocks reproduce
  the serial result exactly. It is deliberately not used for histogram
  accumulation, which would need a cross-block reduction.

Task count comes from the workload shape rather than from the item count.
One task per item balances perfectly but pays a scheduling event per item,
which dominates on wide inputs, so auto mode caps the fan-out at
`TASKS_PER_CORE` tasks per physical core and chunks the rest.
"""

from max.algorithm import sync_parallelize
from std.os import getenv
from std.sys.info import num_physical_cores

# Serial-vs-parallel crossover measured at 25k-50k ops on Apple M4, AMD
# Zen4, and Neoverse-N2 (bench/bench_threshold.mojo); 1 << 16 sits above
# all three with 1.2-1.6x parallel speedup at exactly this size.
comptime PARALLEL_MIN_OPS = 1 << 16

# Auto-mode fan-out per physical core, applied on top of the grain rule in
# `plan_tasks`. Above 1 the extra tasks absorb the jitter of unequal per-item
# cost (features differ in bin occupancy, blocks in cache behaviour); far
# above it the scheduling events cost more than the imbalance they hide. 4 is
# a starting point, not a measured optimum: sweep it with
# bench/bench_profile.mojo on an idle machine before treating it as tuned.
comptime TASKS_PER_CORE = 4


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
    return _env_int("MOJOBOOST_NUM_WORKERS", 0)


def env_parallel_min_ops() -> Int:
    return _env_int("MOJOBOOST_PARALLEL_MIN_OPS", PARALLEL_MIN_OPS)


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
    `MOJOBOOST_NUM_WORKERS` bypasses both, so tests can force the parallel
    path at any size.
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
        n_tasks = TASKS_PER_CORE * num_physical_cores()
        var by_grain = total_ops // grain
        if by_grain < n_tasks:
            n_tasks = by_grain
        if n_tasks < 1:
            n_tasks = 1
    if n_tasks > n_items:
        n_tasks = n_items
    return n_tasks


def dispatch_features[FuncType: def (Int) -> None](
    func: FuncType, n_features: Int, total_ops: Int
) raises:
    """Run `func(f)` for f in [0, n_features) under the env contract above.

    `func` must be safe to run concurrently for distinct f (each feature
    writes only its own output range). `total_ops` is the work estimate
    compared against the auto-mode threshold, conventionally
    n_features * rows_touched.
    """
    var n_tasks = plan_tasks(n_features, total_ops)
    if n_tasks <= 1:
        for f in range(n_features):
            func(f)
        return

    var chunk = (n_features + n_tasks - 1) // n_tasks

    def do_chunk(w: Int) {imm}:
        var f = w * chunk
        var end = f + chunk
        if end > n_features:
            end = n_features
        while f < end:
            func(f)
            f += 1

    sync_parallelize(do_chunk, n_tasks)


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
