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

# Auto-mode fan-out per physical core. Above 1 the extra tasks absorb the
# jitter of unequal per-item cost; far above it the scheduling events cost
# more than the imbalance they hide. Measured flat between 2 and 8 on Apple
# M4 (bench/bench_profile.mojo), so 4 is the midpoint of the flat region.
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
    every caller must produce the same answer at every task count, so this
    can be tuned freely without changing any output.
    """
    if n_items <= 1:
        return 1
    var workers = env_num_workers()
    if workers == 1:
        return 1
    if workers == 0 and total_ops < env_parallel_min_ops():
        return 1
    var n_tasks = workers
    if workers == 0:
        n_tasks = TASKS_PER_CORE * num_physical_cores()
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


def dispatch_rows[FuncType: def (Int, Int) -> None](
    func: FuncType, n_rows: Int, total_ops: Int
) raises:
    """Run `func(start, end)` over contiguous blocks covering [0, n_rows).

    Blocks are disjoint and each one walks its rows in ascending order, so
    elementwise callers get bit-identical results at every task count. The
    serial path is a single `func(0, n_rows)` call, which is also the shape
    a block gets, so there is only one body to test.
    """
    if n_rows <= 0:
        return
    var n_tasks = plan_tasks(n_rows, total_ops)
    if n_tasks <= 1:
        func(0, n_rows)
        return

    var chunk = (n_rows + n_tasks - 1) // n_tasks
    # Ceiling division can make the last tasks empty (e.g. 10 rows, 4 tasks,
    # chunk 3 leaves task 3 with nothing); recompute so every task has work.
    n_tasks = (n_rows + chunk - 1) // chunk

    def do_block(w: Int) {imm}:
        var start = w * chunk
        var end = start + chunk
        if end > n_rows:
            end = n_rows
        func(start, end)

    sync_parallelize(do_block, n_tasks)
