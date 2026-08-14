"""Shared controls for multicore per-feature dispatch.

Two environment variables govern every per-feature parallel loop (histogram
accumulation, bin fitting, bin transform), for reproducible benchmarking and
for tests that must force one path:

- `MOJOBOOST_NUM_WORKERS`: `1` forces the serial path; `N > 1` forces the
  parallel path with features chunked into at most N tasks regardless of
  input size; `0`, unset, or unparsable means auto (one task per feature
  when the work is large enough to amortize scheduling).
- `MOJOBOOST_PARALLEL_MIN_OPS`: integer override of the auto-mode work
  threshold (`PARALLEL_MIN_OPS`), compared against features * rows.
"""

from max.algorithm import sync_parallelize
from std.os import getenv

# Serial-vs-parallel crossover measured at 25k-50k ops on Apple M4, AMD
# Zen4, and Neoverse-N2 (bench/bench_threshold.mojo); 1 << 16 sits above
# all three with 1.2-1.6x parallel speedup at exactly this size.
comptime PARALLEL_MIN_OPS = 1 << 16


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


def dispatch_features[FuncType: def (Int) -> None](
    func: FuncType, n_features: Int, total_ops: Int
) raises:
    """Run `func(f)` for f in [0, n_features) under the env contract above.

    `func` must be safe to run concurrently for distinct f (each feature
    writes only its own output range). `total_ops` is the work estimate
    compared against the auto-mode threshold, conventionally
    n_features * rows_touched.
    """
    var workers = env_num_workers()
    if n_features <= 1 or workers == 1 or (
        workers == 0 and total_ops < env_parallel_min_ops()
    ):
        for f in range(n_features):
            func(f)
    elif workers == 0:
        sync_parallelize(func, n_features)
    else:
        var n_tasks = workers
        if n_tasks > n_features:
            n_tasks = n_features
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
