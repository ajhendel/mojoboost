"""Apple-silicon CPU tuning policy.

A pure policy layer for the multicore CPU backend, in the same shape as
`apple_gpu_policy.mojo`: detected machine facts and a workload shape in, a
plan out. It allocates nothing, dispatches nothing, and touches no dataset,
so every decision below can be exercised and asserted on any machine.

Nothing here changes a result. Every quantity it produces is a scheduling or
a memory-traffic decision: how many tasks a loop is split into, whether an
accumulation gathers its gradients into a scratch buffer first, how many
features one inner loop interleaves. The kernels in `histogram.mojo` and the
dispatch shapes in `parallel.mojo` keep each feature's summation inside one
task and each row block in ascending order whatever this module answers, so
the output is bit-identical at every setting (see the invariants in
`handoffs/performance_13_apple_cpu.md`).

What this module encodes
------------------------
- **Asymmetric cores, handled conservatively.** Apple silicon reports a mix
  of performance and efficiency cores, and `num_physical_cores()` counts both.
  A barrier-synchronized fan-out finishes when its slowest task finishes, so
  a task landing on an efficiency core sets the pace. The two portable
  responses are to over-decompose (more tasks than cores, so a fast core can
  take a second chunk) and to shrink the pool to the performance cores. The
  default is the first, unchanged from what this repository already shipped,
  because the second throws away real cores on the strength of an assumption
  and no measurement here justifies it. `MOJOTREES_CPU_CORE_POOL=performance`
  selects the second for a benchmark that wants to measure the difference.
  Neither path pins a thread: no portable affinity control is used, and none
  is assumed to exist.
- **Cache-sized inner loops.** How many features one accumulation loop may
  interleave is bounded by an assumed L1 data-cache size, so the interleaved
  group's histogram slices stay resident. The bound is derived from
  `n_bins` and the histogram's bytes-per-cell, not from a fixed feature count.
- **Work estimates that include zeroing.** A tree node's histogram build
  costs one write of the whole `n_features * n_bins` output plus one
  accumulate per (active feature, row). On a small node the first term
  dominates, so an estimate counting only rows sends a multi-megabyte write
  down the serial path. `derive_accumulation_plan` reports the two terms
  separately, and each dispatch is sized by the term it actually runs.

What this module deliberately does NOT encode
---------------------------------------------
- **No development-machine constants.** No M-series part number appears
  anywhere in this file, and no per-chip table is consulted. Everything is
  derived from what `std.sys.info` reports on the running machine, plus the
  three assumptions named below, which are conservative floors rather than
  the development Mac's values.
- **No claim that any of it is measured.** Every constant here is a starting
  point with a stated mechanism, not a tuned optimum.
  `bench/apple/cpu_plan.json` lists the sweep that would turn each one into
  a measured value, and the handoff lists which claims need profiler or
  disassembly evidence before anyone repeats them.
- **No affinity, no QoS class, no thread pinning, no core-type
  detection beyond the counts `std.sys.info` reports.**
- **No Python.** The policy is consumed by Mojo hot paths; a Python
  performance policy would be unreachable from them.

Assumed constants
-----------------
Three numbers are not reported by any portable API and are assumed:
`ARM_CACHE_LINE_BYTES`, `GENERIC_CACHE_LINE_BYTES`, and `ASSUMED_L1D_BYTES`.
Each is set to a conservative floor, because underestimating a cache costs
some locality while overestimating it produces a plan that thrashes. They are
named, not inlined, so a machine that turns out to differ has one place to
change and the handoff has something to point a measurement at.

Environment contract (all optional, all scheduling-only)
--------------------------------------------------------
- `MOJOTREES_CPU_TASKS_PER_CORE`: positive integer, auto-mode fan-out per
  core. Default `DEFAULT_TASKS_PER_CORE`.
- `MOJOTREES_CPU_CORE_POOL`: `all` (default) counts every physical core;
  `performance` counts only the reported performance cores.
- `MOJOTREES_CPU_FEATURE_GROUP`: `1` or `2`, how many features one
  accumulation loop interleaves. Default `DEFAULT_FEATURE_GROUP`, further
  clamped by the L1 estimate.
- `MOJOTREES_CPU_COMPACT_MIN_ROWS`: row count below which a subset
  accumulation skips the gradient/hessian gather. Default
  `DEFAULT_COMPACT_MIN_ROWS`.

`MOJOTREES_NUM_WORKERS` and `MOJOTREES_PARALLEL_MIN_OPS` keep their meanings
from `parallel.mojo` and override everything here: an explicit worker count
still bypasses the grain floor and the core cap.
"""

from std.sys.info import (
    CompilationTarget,
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
    simd_width_of,
)
from std.os import getenv

# Auto-mode fan-out per core, applied on top of the grain rule in
# `parallel.plan_tasks`. Above 1 the extra tasks absorb the jitter of unequal
# per-item cost (features differ in bin occupancy, blocks in cache behaviour)
# and give a performance core somewhere to go when an efficiency core is
# still working; far above it the scheduling events cost more than the
# imbalance they hide. 4 is a starting point, not a measured optimum.
comptime DEFAULT_TASKS_PER_CORE = 4

# Core pools. `all` is the default and is what this repository has always
# used; `performance` exists so a benchmark can measure whether excluding the
# efficiency cores from a barrier-synchronized fan-out is worth the cores it
# gives up. Neither pins anything.
comptime CORE_POOL_ALL = 0
comptime CORE_POOL_PERFORMANCE = 1

# How many features one accumulation inner loop interleaves. Two independent
# features give the out-of-order engine two independent read-modify-write
# chains into two disjoint histogram slices, which is what hides the
# store-to-load latency of consecutive rows landing in the same bin, and it
# halves the number of times the shared row index and gradient pair are
# loaded. Above two the resident slices start to crowd L1 and the register
# pressure rises; `plan_feature_group` clamps by the cache estimate anyway.
comptime DEFAULT_FEATURE_GROUP = 2
comptime MAX_FEATURE_GROUP = 2

# Below this many rows a node's gradients and hessians are small enough to
# stay in cache across every feature's pass, so gathering them into a
# contiguous scratch buffer first would add a pass without removing any
# misses. Above it the gather is paid once instead of once per feature.
comptime DEFAULT_COMPACT_MIN_ROWS = 256

# Gathering is only worth a pass when more than one feature will read the
# result; with a single active feature the gather reads exactly what the
# accumulation would have read.
comptime COMPACT_MIN_FEATURES = 2

# One histogram cell is a gradient, a hessian, and a count.
comptime HISTOGRAM_BYTES_PER_CELL = 24

# Assumed, not reported. Apple silicon uses a 128-byte line; 64 is the
# portable floor elsewhere. Only used for locality arithmetic and reporting,
# never for correctness.
comptime ARM_CACHE_LINE_BYTES = 128
comptime GENERIC_CACHE_LINE_BYTES = 64

# Resolved once at compile time rather than queried per call: it is a
# property of the target, it costs nothing to fold, and it keeps every
# function below non-raising, which is what lets `parallel.plan_tasks` stay
# non-raising for its non-raising callers.
comptime TARGET_HAS_NEON = CompilationTarget.has_neon()
comptime ASSUMED_CACHE_LINE_BYTES = (
    ARM_CACHE_LINE_BYTES if TARGET_HAS_NEON else GENERIC_CACHE_LINE_BYTES
)

# Assumed, not reported: a conservative L1 data cache floor. Apple's
# efficiency cores have the smaller L1 of the two core types, and this is
# meant to be at or below it, so a group sized against it stays resident on
# whichever core type the task lands on.
comptime ASSUMED_L1D_BYTES = 64 * 1024

# Fraction of the assumed L1 an interleaved feature group may claim. The rest
# is left for the row index stream, the gradient pairs, and the binned column
# the loop is walking, all of which are streaming through the same cache.
comptime L1_GROUP_DIVISOR = 2


def _env_int(name: String, default: Int) -> Int:
    """Deliberately a private copy of `parallel._env_int` rather than an
    import of it: `parallel.mojo` imports this module, and a policy that
    imported back would make the two mutually dependent. Ten lines of
    duplication buys a one-way dependency."""
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


def _positive_or(value: Int, default: Int) -> Int:
    return value if value > 0 else default


def env_tasks_per_core() -> Int:
    """Auto-mode fan-out per core; 0 or unparsable means the default."""
    return _positive_or(
        _env_int("MOJOTREES_CPU_TASKS_PER_CORE", DEFAULT_TASKS_PER_CORE),
        DEFAULT_TASKS_PER_CORE,
    )


def env_core_pool() -> Int:
    """Which cores auto mode counts. Anything unrecognized means `all`, so a
    typo loses the experiment rather than the parallelism."""
    var s = getenv("MOJOTREES_CPU_CORE_POOL")
    if s == "performance" or s == "PERFORMANCE" or s == "p":
        return CORE_POOL_PERFORMANCE
    return CORE_POOL_ALL


def env_feature_group() -> Int:
    return _env_int("MOJOTREES_CPU_FEATURE_GROUP", DEFAULT_FEATURE_GROUP)


def env_compact_min_rows() -> Int:
    return _env_int("MOJOTREES_CPU_COMPACT_MIN_ROWS", DEFAULT_COMPACT_MIN_ROWS)


def core_pool_name(pool: Int) -> String:
    if pool == CORE_POOL_PERFORMANCE:
        return String("performance")
    return String("all")


@fieldwise_init
struct CpuProfile(Copyable, Movable):
    """Everything the CPU policy is allowed to know about the machine.

    Only the four counts are read from the system. The rest are the assumed
    constants above, carried on the profile so a caller can print exactly
    what a decision was made against.
    """

    var physical_cores: Int
    var logical_cores: Int

    var performance_cores: Int
    """Reported performance cores, sanitized into `[1, physical_cores]`. On a
    symmetric machine this equals `physical_cores`, which is how
    `asymmetric` is decided; no core-type detection beyond that is done."""

    var float64_lanes: Int
    """Compile-time SIMD width for Float64 on this target."""

    var has_neon: Bool
    var cache_line_bytes: Int
    var l1d_bytes: Int

    @staticmethod
    def detect() -> CpuProfile:
        """The running machine, with every count sanitized. A machine that
        reports nonsense (zero cores, more performance cores than physical)
        degrades to the conservative reading rather than producing a plan
        with a zero or negative task count."""
        var physical = _positive_or(num_physical_cores(), 1)
        var logical = _positive_or(num_logical_cores(), physical)
        var perf = _positive_or(num_performance_cores(), physical)
        if perf > physical:
            perf = physical
        return CpuProfile(
            physical,
            logical,
            perf,
            simd_width_of[DType.float64](),
            TARGET_HAS_NEON,
            ASSUMED_CACHE_LINE_BYTES,
            ASSUMED_L1D_BYTES,
        )

    @staticmethod
    def synthetic(physical_cores: Int, performance_cores: Int) -> CpuProfile:
        """A constructed profile, for exercising the rules below on a machine
        that is not the one being planned for. Never used as a fallback:
        `detect` always reads the running machine. The SIMD width and the
        cache assumptions stay the compiling target's, because those are
        properties of the build rather than of the invented core layout."""
        var physical = _positive_or(physical_cores, 1)
        var perf = _positive_or(performance_cores, physical)
        if perf > physical:
            perf = physical
        return CpuProfile(
            physical,
            physical,
            perf,
            simd_width_of[DType.float64](),
            TARGET_HAS_NEON,
            ASSUMED_CACHE_LINE_BYTES,
            ASSUMED_L1D_BYTES,
        )

    def asymmetric(self) -> Bool:
        """Whether the machine reports fewer performance cores than physical
        cores, which is the only signal used to recognize a big.LITTLE-style
        layout. It is a fact about the report, not a chip identification."""
        return self.performance_cores < self.physical_cores

    def efficiency_cores(self) -> Int:
        return self.physical_cores - self.performance_cores

    def dispatch_cores(self) -> Int:
        """Cores auto mode plans against, under `MOJOTREES_CPU_CORE_POOL`."""
        if env_core_pool() == CORE_POOL_PERFORMANCE:
            return _positive_or(self.performance_cores, 1)
        return _positive_or(self.physical_cores, 1)

    def max_auto_tasks(self) -> Int:
        """Ceiling on the task count auto mode may choose. The grain rule in
        `parallel.plan_tasks` applies on top of this and usually binds
        first."""
        return _positive_or(env_tasks_per_core() * self.dispatch_cores(), 1)

    def describe(self) -> String:
        """One line for benchmark headers and bug reports."""
        var neon = String("yes") if self.has_neon else String("no")
        return String(
            "cores=",
            self.physical_cores,
            " (perf ",
            self.performance_cores,
            ", eff ",
            self.efficiency_cores(),
            ", logical ",
            self.logical_cores,
            ") pool=",
            core_pool_name(env_core_pool()),
            " tasks_per_core=",
            env_tasks_per_core(),
            " max_auto_tasks=",
            self.max_auto_tasks(),
            " f64_lanes=",
            self.float64_lanes,
            " neon=",
            neon,
            " assumed_line=",
            self.cache_line_bytes,
            " assumed_l1d=",
            self.l1d_bytes,
        )


def cpu_profile() -> CpuProfile:
    """The running machine. Cheap enough to call once per histogram build,
    which is where every caller calls it; it is not cached, because this
    project has no once-initialized global facility and a stale cache would
    hide an environment change a benchmark just made."""
    return CpuProfile.detect()


def feature_slice_bytes(n_bins: Int) -> Int:
    """Bytes one feature's histogram slice occupies across the three
    arrays."""
    return n_bins * HISTOGRAM_BYTES_PER_CELL


def plan_feature_group(profile: CpuProfile, n_bins: Int) -> Int:
    """How many features one accumulation inner loop may interleave.

    Interleaving is only useful while every slice in the group stays in L1:
    the whole point is that the two read-modify-write chains do not stall on
    memory. The group is therefore bounded by the assumed L1 budget, and by
    the environment knob, and by `MAX_FEATURE_GROUP`.
    """
    var wanted = env_feature_group()
    if wanted > MAX_FEATURE_GROUP:
        wanted = MAX_FEATURE_GROUP
    if wanted < 1:
        wanted = 1
    var slice_bytes = feature_slice_bytes(n_bins)
    if slice_bytes <= 0:
        return 1
    var budget = profile.l1d_bytes // L1_GROUP_DIVISOR
    var fits = budget // slice_bytes
    if fits < wanted:
        wanted = fits
    if wanted < 1:
        wanted = 1
    return wanted


@fieldwise_init
struct AccumulationPlan(Copyable, Movable):
    """The scheduling shape of one histogram build.

    Three dispatches can run: the excluded features' slices are zeroed, the
    node's gradients and hessians are optionally gathered, and the active
    features are zeroed and accumulated. Each carries its own work estimate,
    because sizing all three by the row count is what used to send a
    multi-megabyte zeroing pass down the serial path on a small node.
    """

    var group_width: Int
    """Features one inner loop interleaves; 1 is the plain per-feature
    loop."""

    var compact_rows: Bool
    """Whether to gather the node's (gradient, hessian) pairs into a
    contiguous scratch buffer before the feature loop. False for the
    full-dataset builder, whose rows are already contiguous."""

    var active_ops: Int
    """Work estimate for the zero-and-accumulate dispatch, in histogram-op
    equivalents: one zeroed cell and one accumulated row each count as one."""

    var excluded_ops: Int
    """Work estimate for zeroing the slices of features this build does not
    accumulate (feature subsampling leaves them zero)."""

    var gather_ops: Int
    """Work estimate for the gradient/hessian gather, zero when it is
    skipped. Two indirect loads and two sequential stores per row, counted
    as two ops."""

    def total_ops(self) -> Int:
        return self.active_ops + self.excluded_ops + self.gather_ops

    def describe(self) -> String:
        var compact = String("yes") if self.compact_rows else String("no")
        return String(
            "group_width=",
            self.group_width,
            " compact_rows=",
            compact,
            " active_ops=",
            self.active_ops,
            " excluded_ops=",
            self.excluded_ops,
            " gather_ops=",
            self.gather_ops,
        )


def derive_accumulation_plan(
    profile: CpuProfile,
    n_features: Int,
    n_active: Int,
    n_bins: Int,
    n_rows_touched: Int,
    rows_are_indirect: Bool,
) -> AccumulationPlan:
    """Plan one histogram build.

    `n_active` is how many features will be accumulated (all of them unless
    the caller passed a feature subset), `n_rows_touched` how many rows each
    of them walks, and `rows_are_indirect` whether those rows arrive through
    a row-id list (a tree node) rather than as the whole contiguous dataset.

    The gather is chosen when it can pay for itself: it costs one pass over
    the node's rows and saves the remaining features an indirect load of the
    gradient and the hessian per row, so it wants at least two active
    features and enough rows that the pass is not pure overhead.
    """
    var active = n_active if n_active > 0 else 0
    var rows = n_rows_touched if n_rows_touched > 0 else 0
    var bins = n_bins if n_bins > 0 else 0
    var excluded = n_features - active
    if excluded < 0:
        excluded = 0

    var compact = (
        rows_are_indirect
        and active >= COMPACT_MIN_FEATURES
        and rows >= env_compact_min_rows()
    )
    var group = plan_feature_group(profile, bins)
    if group > active and active > 0:
        group = active
    if group < 1:
        group = 1

    return AccumulationPlan(
        group,
        compact,
        active * (bins + rows),
        excluded * bins,
        2 * rows if compact else 0,
    )


def subtract_ops(n_cells: Int) -> Int:
    """Work estimate for a sibling subtraction over `n_cells` histogram
    cells. Each cell is a load, a subtract, and a store in each of three
    independent streams, so it is counted as three ops rather than one: the
    pass is memory-bound and a single-op estimate keeps it serial well past
    the point where the streams saturate one core."""
    return 3 * n_cells


def describe_policy(profile: CpuProfile, plan: AccumulationPlan) -> String:
    return String(profile.describe(), " | ", plan.describe())
