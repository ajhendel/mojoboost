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
  The width is then floored to a rung of the ladder `histogram.mojo`
  instantiates its accumulation kernel at (1, 2, 4, 8, 16), because a width
  with no instantiation cannot run.
- **A group width the schedule can actually deliver, and can afford.** The
  interleave only happens if one task holds a whole group, so the dispatch is
  over *groups* rather than over features. Before that, `parallel.plan_tasks`
  fanned 50 features over 40 tasks, 30 of which held a single feature and
  therefore could not interleave anything, so a nominal width of 2 ran at an
  effective 1.25 and the knob that changed the task count silently changed the
  width too.
  Dispatching over groups makes the width real and makes its cost real with
  it: the task count is now divided by exactly the factor the memory traffic
  is. `TASK_BALANCE_FACTOR` is the price of that, a floor on how many groups
  the width must leave per core, because a width chosen from bytes alone would
  have put 13 tasks on a 10-core machine and idled seven of them through the
  second barrier round. Both numbers are reported by `describe_cpu_policy`,
  since a width without its utilization is half a decision.
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

`ASSUMED_L1D_BYTES` stays at its 64 KiB floor, and that is a decision rather
than an oversight. The development M4's L1 data cache is widely reported to be
larger, and raising the constant would widen the interleave a rung at 255
bins. Two things stop it. The number is not read from any API this project
calls, so raising it would replace a stated assumption with a remembered
figure; and there is no CPU equivalent of the device report that
`gpu_split_policy._is_observed_m4` guards its hardware-specific threshold
with, because nothing portable in `std.sys.info` identifies the part. Guessing
the part from core counts is exactly the per-chip table the section above
refuses. So the floor stands, the ladder is selected against it, and
`MOJOTREES_CPU_FEATURE_GROUP` is the way to run the wider rungs the floor
declines to choose. A measurement that resolves this belongs in
`bench/apple/cpu_plan.json`, not in a constant nobody verified.

Environment contract (all optional, all scheduling-only)
--------------------------------------------------------
- `MOJOTREES_CPU_TASKS_PER_CORE`: positive integer, auto-mode fan-out per
  core. Default `DEFAULT_TASKS_PER_CORE`.
- `MOJOTREES_CPU_CORE_POOL`: `all` (default) counts every physical core;
  `performance` counts only the reported performance cores.
- `MOJOTREES_CPU_FEATURE_GROUP`: a rung of the ladder (`1`, `2`, `4`, `8`,
  `16`), how many features one accumulation loop interleaves. Unset means the
  derived width: `DEFAULT_FEATURE_GROUP` clamped by the L1 estimate, by the
  active feature count, and by the core count. An explicit request overrides
  all three, because the estimate is the thing the knob exists to test; an
  off-ladder value (`3`, `32`, a word) is *refused*, not rounded, because no
  kernel is instantiated at it.
- `MOJOTREES_CPU_COMPACT_MIN_ROWS`: row count below which a subset
  accumulation skips the gradient/hessian gather. Default
  `DEFAULT_COMPACT_MIN_ROWS`.

`MOJOTREES_NUM_WORKERS` and `MOJOTREES_PARALLEL_MIN_OPS` keep their meanings
from `parallel.mojo` and override everything here: an explicit worker count
still bypasses the grain floor and the core cap.

Resolving the environment once
------------------------------
Every function above reads its variable with `getenv` at the moment it is
asked, and `cpu_profile()` re-detects the machine on every call. That is one
`getenv` per question and three core-count queries per detection, and the
tree grower asks these questions once per dispatch, several times per node,
tens of thousands of times per fit. `ResolvedCpuPolicy.resolve()` answers all
of them once and hands back a value that can be carried for the length of a
fit; `parallel.DispatchSettings` wraps it with the two variables that module
owns. Nothing about a plan changes: the resolved forms below compute the same
answers from the same numbers, and the unresolved forms are kept and still
read the environment live, so a test that flips a variable mid-process and
calls the unresolved form sees the flip exactly as it always did.

A resolved value is a snapshot. It does not observe a later `setenv`, and
there is deliberately no global cache and no invalidation hook to forget:
whoever resolves it owns it, and dropping it and resolving again is the
whole of the invalidation story. A caller that must see a mid-process change
either resolves again or uses the unresolved form.
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

# How many features one accumulation inner loop interleaves. N independent
# features give the out-of-order engine N independent read-modify-write chains
# into N disjoint histogram slices, which is what hides the store-to-load
# latency of consecutive rows landing in the same bin, and -- the reason this
# ladder exists at all -- it divides by N the number of times the node's
# gradient and hessian stream is walked. Feature-major accumulation reads that
# stream once per group, so at 50 features and 1M rows a width of 1 streams
# 50 * 16 MB = 800 MB of gradient traffic per node histogram against the
# 50 MB of binned data it actually needs; a width of 4 streams 208 MB.
# That ratio is why LightGBM's `force_row_wise` builder reads each row's
# gradient exactly once (`src/io/multi_val_dense_bin.hpp`), and widening this
# is the cheapest available probe of how much of the gap that accounts for.
#
# The rungs are 1, 2, 4, 8, 16, matching the GPU histogram family's ladder in
# `gpu_tiling.HIST_FEATURE_GROUP_LADDER` for the same reason: the accumulation
# kernel is instantiated once per width, and a width with no instantiation
# cannot run. `MAX_FEATURE_GROUP` is the top of the ladder, not a policy
# ceiling; `plan_feature_group` is where the width is actually decided, and it
# is bounded by the L1 estimate rather than by a hand-set number.
#
# Nothing here is measured. Wider is strictly less gradient traffic and
# strictly more live histogram slices and register pressure, and where those
# two cross on any particular core is a benchmark's answer, not this comment's.
#
# And wider is NOT free even before the hardware is consulted, which is the
# correction that produced `TASK_BALANCE_FACTOR` below. A group is the dispatch
# unit, so widening divides the task count by exactly the factor it divides the
# traffic by. Byte counts cannot see that, and a plan chosen from byte counts
# alone would have shipped 13 tasks on a 10-core machine here.
comptime FEATURE_GROUP_LADDER = 5
"""Rungs in the feature-group ladder, counting 1."""

comptime MAX_FEATURE_GROUP = 16
comptime DEFAULT_FEATURE_GROUP = MAX_FEATURE_GROUP
"""What the policy asks for before the clamps: the widest rung there is. The
L1 estimate, the core count, and the active feature count are what actually
choose, which is the point of deriving a width instead of naming one."""

# Groups per dispatch core the resolved width must leave, when the feature
# count allows it. The balance rule, and the reason width is not chosen from
# memory traffic alone.
#
# The arithmetic. `T` equal-cost tasks over `C` cores under a barrier finish
# in `ceil(T / C)` rounds, so utilization is `T / (C * ceil(T / C))`. That is
# 100% only when C divides T, and over all `T >= F * C` its worst case is at
# `T = F * C + 1`, one task past a whole number of rounds, giving
# `(F * C + 1) / ((F + 1) * C)`. On ten cores that is 70% at F = 2, 77% at
# F = 3 and 82% at F = 4; as C grows it tends to `F / (F + 1)`. So the factor
# buys a floor on how much of the machine the last round can waste, and the
# floor is not steep: going from 2 to 4 buys twelve points of guarantee.
#
# Why 2 and not 4. The floor is a guarantee, not the typical case, and at the
# shape this lane exists for it is not what binds: 50 features on 10 cores at
# width 2 is 25 groups, 3 rounds, 25/30 = 83%. Raising the factor to 4 would
# demand 40 groups and force width 1 on that same shape, which is to say it
# would revert the lane on the only shape it was written for. 3 would demand
# 30 groups and force width 1 as well. 2 is therefore the largest factor that
# leaves any interleave at all at 50 features on this machine, and that is the
# whole of its derivation: it is a boundary, not an optimum.
#
# It is unmeasured in both directions. Two things it does not model: cores here
# are not equal (a task on an efficiency core sets the pace of its round, so
# the real cost of a partial round is worse than the fraction suggests), and
# the accumulation may be memory-bound well before every core is busy (in which
# case idle cores cost less than this rule prices them at). Only an interleaved
# wall-time A/B settles it. `MOJOTREES_CPU_FEATURE_GROUP` is how it is swept,
# and `bench/apple/cpu_plan.json` states the comparison that would move it.
comptime TASK_BALANCE_FACTOR = 2

# Below this many rows a node's gradients and hessians are small enough to
# stay in cache across every feature's pass, so gathering them into a
# contiguous scratch buffer first would add a pass without removing any
# misses. Above it the gather is paid once instead of once per feature.
comptime DEFAULT_COMPACT_MIN_ROWS = 256

# Gathering is only worth a pass when more than one feature will read the
# result; with a single active feature the gather reads exactly what the
# accumulation would have read.
comptime COMPACT_MIN_FEATURES = 2

# One host histogram cell: Float64 gradient + Float64 hessian + Int count per
# (feature, bin), the layout of `Histogram` in histogram.mojo on a 64-bit
# target. The one definition; histogram_cache_policy.mojo imports it.
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


def is_feature_group_width(group: Int) -> Bool:
    """Whether `group` is a rung of the ladder, so `histogram.mojo` has an
    accumulation kernel instantiated at it. 3 is not, and is refused rather
    than rounded, for the same reason `gpu_tiling.is_feature_group_width`
    refuses it: a width with no instantiation cannot run, and silently running
    a different one loses whatever experiment asked for it."""
    var rung = 1
    for _ in range(FEATURE_GROUP_LADDER):
        if rung == group:
            return True
        rung *= 2
    return False


def feature_group_floor(n: Int) -> Int:
    """The widest rung at or below `n`, never below 1.

    Rounding *down* is the only safe direction for every clamp below: a rung
    above the bound would violate the bound the clamp was expressing, while a
    rung below it merely leaves some of the budget unused. `feature_group_floor(5)`
    is 4, which is what the L1 estimate at 255 bins comes to.
    """
    var best = 1
    var rung = 1
    for _ in range(FEATURE_GROUP_LADDER):
        if rung <= n:
            best = rung
        rung *= 2
    return best


def feature_group_count(n_active: Int, group: Int) -> Int:
    """How many groups `n_active` features split into at this width.

    The dispatch unit of a histogram build. Stated here rather than derived at
    each call site because it is the number that has to match between the
    policy, the kernel's group loop, and the task split: a task holds whole
    groups, so a group is the smallest thing that can be scheduled, and the
    last group owning fewer than `group` features is the ragged tail the
    kernel handles with the same code path as a full one.
    """
    if n_active <= 0 or group <= 0:
        return 0
    return (n_active + group - 1) // group


def env_feature_group() raises -> Int:
    """The requested interleave width, or 0 when nothing is requested.

    Zero rather than `DEFAULT_FEATURE_GROUP` because the two answers mean
    different things downstream: an explicit request bypasses the derived
    clamps (it exists to test the estimate those clamps encode) while the
    default does not.

    Raises on anything that is not a rung. That includes an unparsable value,
    which the other knobs in this file quietly ignore. The difference is
    deliberate and narrow: the others are scheduling numbers where any value
    produces some valid plan, whereas this one names a kernel instantiation,
    and a benchmark arm that asked for 3 and silently got 2 has recorded a
    result under the wrong label. That has happened in this repository before,
    with the GPU split strategy, and `gpu_active_rows.set_feature_group`
    refuses for the same reason.
    """
    var s = getenv("MOJOTREES_CPU_FEATURE_GROUP")
    if s.byte_length() == 0:
        return 0
    try:
        var n = Int(s)
        if not is_feature_group_width(n):
            raise Error("off the ladder")
        return n
    except:
        raise Error(
            'MOJOTREES_CPU_FEATURE_GROUP must be 1, 2, 4, 8, or 16: those are'
            ' the widths an accumulation kernel is instantiated at. Got "',
            s,
            '"',
        )


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


@fieldwise_init
struct ResolvedCpuPolicy(Copyable, Movable):
    """One reading of the machine and of this module's four variables.

    The same four questions `env_tasks_per_core`, `env_core_pool`,
    `env_feature_group`, and `env_compact_min_rows` answer, plus one
    `CpuProfile.detect()`, resolved together so a fit pays for them once
    instead of once per dispatch. Every method here is the resolved twin of a
    free function above and computes the identical answer from the identical
    numbers; the difference is only when the environment was read.

    Held by value and copied freely: it is five machine counts and four small
    integers, so passing it down a call chain costs nothing worth measuring
    and no lifetime has to be reasoned about. It is a snapshot, so a
    `setenv` after `resolve()` is not observed by it; that is the documented
    contract rather than a defect, and it is why the unresolved functions are
    still here for anything that wants a live read.
    """

    var profile: CpuProfile
    var tasks_per_core: Int
    var core_pool: Int
    var feature_group: Int
    var compact_min_rows: Int

    @staticmethod
    def resolve() raises -> ResolvedCpuPolicy:
        """Detect the machine and read all four variables, once.

        Raises for an off-ladder `MOJOTREES_CPU_FEATURE_GROUP`, at the one
        point the snapshot is taken, rather than once per node."""
        return ResolvedCpuPolicy(
            CpuProfile.detect(),
            env_tasks_per_core(),
            env_core_pool(),
            env_feature_group(),
            env_compact_min_rows(),
        )

    @staticmethod
    def of(profile: CpuProfile) raises -> ResolvedCpuPolicy:
        """A policy around an already-detected profile, reading the four
        variables now. For a caller that has a `CpuProfile` in hand (a
        synthetic one in a test, or one it detected itself) and wants the
        resolved methods without a second detection."""
        return ResolvedCpuPolicy(
            profile.copy(),
            env_tasks_per_core(),
            env_core_pool(),
            env_feature_group(),
            env_compact_min_rows(),
        )

    def dispatch_cores(self) -> Int:
        """`CpuProfile.dispatch_cores` against the resolved pool."""
        if self.core_pool == CORE_POOL_PERFORMANCE:
            return _positive_or(self.profile.performance_cores, 1)
        return _positive_or(self.profile.physical_cores, 1)

    def max_auto_tasks(self) -> Int:
        """`CpuProfile.max_auto_tasks` against the resolved knobs."""
        return _positive_or(
            _positive_or(self.tasks_per_core, DEFAULT_TASKS_PER_CORE)
            * self.dispatch_cores(),
            1,
        )

    def feature_group_for(self, n_bins: Int, n_active: Int) raises -> Int:
        """`plan_feature_group` against the resolved knobs.

        Takes `n_active` because the width is no longer a property of the
        cache estimate alone: the balance rule bounds it by how many groups
        the dispatch can still spread over the cores.
        """
        return _plan_group(
            self.profile.l1d_bytes,
            self.dispatch_cores(),
            self.feature_group,
            n_bins,
            n_active,
        )

    def describe(self) -> String:
        """One line naming the snapshot, for benchmark headers. Distinct from
        `CpuProfile.describe`, which re-reads the environment to print it and
        so would report a live value next to a resolved plan."""
        return String(
            "cores=",
            self.profile.physical_cores,
            " (perf ",
            self.profile.performance_cores,
            ", eff ",
            self.profile.efficiency_cores(),
            ") pool=",
            core_pool_name(self.core_pool),
            " tasks_per_core=",
            self.tasks_per_core,
            " max_auto_tasks=",
            self.max_auto_tasks(),
            " feature_group=",
            self.feature_group,
            " compact_min_rows=",
            self.compact_min_rows,
        )


def feature_slice_bytes(n_bins: Int) -> Int:
    """Bytes one feature's histogram slice occupies across the three
    arrays."""
    return n_bins * HISTOGRAM_BYTES_PER_CELL


def _cache_group(l1d_bytes: Int, n_bins: Int) -> Int:
    """The cache clamp over raw integers, so the resolved snapshot and the
    live path cannot drift. `cache_feature_group` is the profile-shaped
    spelling of it."""
    var slice_bytes = feature_slice_bytes(n_bins)
    if slice_bytes <= 0:
        return 1
    var budget = l1d_bytes // L1_GROUP_DIVISOR
    return feature_group_floor(budget // slice_bytes)


def _schedule_group(cores: Int, n_active: Int) -> Int:
    """The balance clamp over raw integers; see `schedule_feature_group`."""
    if cores < 1:
        return 1
    var wanted_groups = TASK_BALANCE_FACTOR * cores
    if wanted_groups < 1:
        wanted_groups = 1
    return feature_group_floor(n_active // wanted_groups)


def _plan_group(
    l1d_bytes: Int, cores: Int, requested: Int, n_bins: Int, n_active: Int
) raises -> Int:
    """The one copy of the width rule, with every input already read.

    Both entry points land here: `plan_feature_group` after a `getenv`, and
    `ResolvedCpuPolicy.feature_group_for` from a snapshot taken once per fit.
    That matters more than it looks. A cached policy that answered a width
    the live path would refuse, or refused one the live path would answer,
    would put the accumulation kernel and the plan that sized its buffers at
    different widths, and the two are not separately checked anywhere.
    """
    if requested > 0:
        return requested
    var group = DEFAULT_FEATURE_GROUP
    var by_cache = _cache_group(l1d_bytes, n_bins)
    if by_cache < group:
        group = by_cache
    var by_schedule = _schedule_group(cores, n_active)
    if by_schedule < group:
        group = by_schedule
    var by_active = feature_group_floor(n_active)
    if by_active < group:
        group = by_active
    return group if group >= 1 else 1


def cache_feature_group(profile: CpuProfile, n_bins: Int) -> Int:
    """The widest rung whose group of histogram slices fits the L1 budget.

    Interleaving is only useful while every slice in the group stays in L1:
    the whole point is that the N read-modify-write chains do not stall on
    memory. `L1_GROUP_DIVISOR` leaves the rest of the assumed cache for the
    row index stream, the gradient pairs, and the N binned columns the loop
    walks, all of which stream through the same cache.

    At the default shape this is the arithmetic that decides everything:
    `feature_slice_bytes(255)` is 6120, the budget is `65536 / 2 = 32768`,
    32768 / 6120 = 5 slices fit, and `feature_group_floor(5)` is 4. Every one
    of those numbers is an assumption or a floor, none is measured, and
    `ASSUMED_L1D_BYTES` in particular is a portable floor rather than this
    machine's cache; the module docstring says why it stays that way.
    """
    return _cache_group(profile.l1d_bytes, n_bins)


def dispatch_rounds(n_tasks: Int, cores: Int) -> Int:
    """Barrier rounds `n_tasks` equal-cost tasks take on `cores` cores.

    The crude model the balance rule is priced with, written down so it can be
    disagreed with: greedy scheduling, equal task costs, equal cores, one
    barrier at the end. Real cores here are not equal, so this understates the
    cost of a partial round rather than overstating it.
    """
    if n_tasks <= 0 or cores <= 0:
        return 0
    return (n_tasks + cores - 1) // cores


def dispatch_utilization_percent(n_tasks: Int, cores: Int) -> Int:
    """Core-seconds used over core-seconds elapsed, as a percentage.

    `n_tasks / (cores * dispatch_rounds(...))`, truncated. 13 tasks on 10
    cores is 65: two rounds, seven cores idle through the second. 25 on 10 is
    83, 40 on 10 is 100. This is the number a memory-traffic table cannot see,
    and reporting a width without it is how a change that halves traffic and
    halves the task count gets recorded as a win.
    """
    var rounds = dispatch_rounds(n_tasks, cores)
    if rounds <= 0:
        return 0
    return (100 * n_tasks) // (cores * rounds)


def schedule_feature_group(profile: CpuProfile, n_active: Int) -> Int:
    """The widest rung that still leaves `TASK_BALANCE_FACTOR` groups per core.

    A group is the dispatch unit, so `n_active` features at width N offer
    `feature_group_count(n_active, N)` units of parallelism and no more.
    Widening divides the task count by the same factor it divides the gradient
    traffic by, and under a barrier that is a real and predictable loss, so the
    width is bounded by what the schedule can still balance.

    Worked, on the shape this lane exists for: 50 active features on a 10-core
    profile. The rule asks for `2 * 10 = 20` groups, `50 // 20 = 2`, and 2 is a
    rung, so the width is 2 and the dispatch is 25 groups in 3 rounds at 83%
    utilization. Width 4 would have been 13 groups in 2 rounds at 65%, and
    width 8 seven groups in one round at 70%. The rungs unlock at
    `n_active >= width * TASK_BALANCE_FACTOR * cores`, which on this machine is
    40 features for width 2, 80 for width 4, 160 for width 8, and 320 for 16.

    `dispatch_cores()` and not `max_auto_tasks()`: the 4x over-decomposition is
    itself a balance device against unequal cores, and counting it here as well
    would demand 400 groups and cap the width at 1 on every shape this project
    cares about. The two floors are for the same problem and applying both
    would double-charge for it.

    Fewer features than `TASK_BALANCE_FACTOR * cores` gives 1, which is what
    the dispatch already did on such a shape and gives up nothing measurable:
    there was never a wide group for a task to hold.
    """
    return _schedule_group(profile.dispatch_cores(), n_active)


def plan_feature_group(
    profile: CpuProfile, n_bins: Int, n_active: Int
) raises -> Int:
    """How many features one accumulation inner loop interleaves.

    An explicit `MOJOTREES_CPU_FEATURE_GROUP` is returned as asked, having
    been checked against the ladder. It deliberately bypasses all three clamps
    below: those encode an assumed cache size and an unmeasured guess about
    what parallelism is worth, and a knob that could not exceed them could not
    be used to test them. It may exceed `n_active`, which costs nothing: the
    kernel's tail group owns only the features that remain.

    Otherwise the width is the narrowest of what the cache estimate allows,
    what the balance rule leaves schedulable, and what `n_active` features can
    fill, floored to a rung throughout. The three are independent bounds and
    which one binds depends on the shape: at 255 bins and 50 features on ten
    cores the balance rule binds at 2 while the cache estimate would have
    allowed 4, and at 32 bins the cache estimate allows 16 while the balance
    rule still says 2.

    Whatever it returns, the result is bit-identical: the width changes how
    many features share one walk of the rows, never the order in which any one
    feature's bins are summed.
    """
    return _plan_group(
        profile.l1d_bytes,
        profile.dispatch_cores(),
        env_feature_group(),
        n_bins,
        n_active,
    )



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
    loop. Always a rung of the ladder, because the kernel is instantiated per
    rung."""

    var group_count: Int
    """Groups the active features split into, and therefore the number of
    units the accumulation dispatches over. A task takes whole groups, which
    is what stops the task splitter from handing a task fewer features than
    the width it was told to interleave."""

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
            " group_count=",
            self.group_count,
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
) raises -> AccumulationPlan:
    """Plan one histogram build.

    `n_active` is how many features will be accumulated (all of them unless
    the caller passed a feature subset), `n_rows_touched` how many rows each
    of them walks, and `rows_are_indirect` whether those rows arrive through
    a row-id list (a tree node) rather than as the whole contiguous dataset.

    The gather is chosen when it can pay for itself: it costs one pass over
    the node's rows and saves the remaining features an indirect load of the
    gradient and the hessian per row, so it wants at least two active
    features and enough rows that the pass is not pure overhead.

    Raises only for an off-ladder `MOJOTREES_CPU_FEATURE_GROUP`; every
    workload shape, including a degenerate one, produces a plan.
    """
    return _derive_plan(
        profile.l1d_bytes,
        profile.dispatch_cores(),
        env_feature_group(),
        env_compact_min_rows(),
        n_features,
        n_active,
        n_bins,
        n_rows_touched,
        rows_are_indirect,
    )


def derive_accumulation_plan_with(
    policy: ResolvedCpuPolicy,
    n_features: Int,
    n_active: Int,
    n_bins: Int,
    n_rows_touched: Int,
    rows_are_indirect: Bool,
) raises -> AccumulationPlan:
    """`derive_accumulation_plan` against an already-resolved policy.

    Identical arithmetic on identical inputs; the only difference is that the
    two variables come from the snapshot rather than from a fresh `getenv`,
    and no `CpuProfile.detect()` happens here at all.
    """
    return _derive_plan(
        policy.profile.l1d_bytes,
        policy.dispatch_cores(),
        policy.feature_group,
        policy.compact_min_rows,
        n_features,
        n_active,
        n_bins,
        n_rows_touched,
        rows_are_indirect,
    )


def _derive_plan(
    l1d_bytes: Int,
    dispatch_cores: Int,
    feature_group: Int,
    compact_min_rows: Int,
    n_features: Int,
    n_active: Int,
    n_bins: Int,
    n_rows_touched: Int,
    rows_are_indirect: Bool,
) raises -> AccumulationPlan:
    """The one copy of the plan arithmetic, with every variable already
    read. `dispatch_cores` is here because the width now depends on how many
    groups the schedule can balance, not only on what L1 holds."""
    var active = n_active if n_active > 0 else 0
    var rows = n_rows_touched if n_rows_touched > 0 else 0
    var bins = n_bins if n_bins > 0 else 0
    var excluded = n_features - active
    if excluded < 0:
        excluded = 0

    var compact = (
        rows_are_indirect
        and active >= COMPACT_MIN_FEATURES
        and rows >= compact_min_rows
    )
    var group = _plan_group(
        l1d_bytes, dispatch_cores, feature_group, bins, active
    )

    return AccumulationPlan(
        group,
        feature_group_count(active, group),
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
    return subtract_ops_for_planes(n_cells, 3)


def subtract_ops_for_planes(n_cells: Int, n_planes: Int) -> Int:
    """`subtract_ops` for a subtraction that touches `n_planes` of the three.

    The constant-hessian specialization in `histogram.mojo` subtracts the
    gradient and count planes and derives the hessian plane from the count it
    just wrote, so its pass runs two of the three read-modify-write streams
    rather than three. The estimate is the traffic, so it has to follow: a
    two-plane pass that told `parallel.plan_tasks` it was a three-plane one
    would clear the crossover at two thirds of the size that actually
    saturates a core.

    Nothing about this can move a value. The subtraction is elementwise over
    disjoint ascending blocks, so the task count changes which core writes a
    cell and nothing else, which is the guarantee `parallel.mojo` states for
    `dispatch_rows` and the reason a work estimate is allowed to be a
    judgment call at all. A plane count below one is clamped to one, so a
    degenerate caller still gets a positive estimate rather than a zero that
    would read as no work.
    """
    var planes = n_planes if n_planes > 0 else 1
    return planes * n_cells


# What one scored split candidate costs, in the histogram-op equivalents
# `parallel.plan_tasks` compares against its threshold. A histogram op is a
# scattered read-modify-write into three L1-resident arrays; a split
# candidate is two Float64 divisions, an L1 soft-threshold on each child, and
# a handful of compares. A division is the term that dominates and it is the
# one instruction on this class of core that is neither single-cycle nor well
# pipelined, so a candidate is counted as several ops rather than one.
#
# 8 is an order-of-magnitude reading of that instruction mix, not a
# measurement: nothing in this repository has timed a split candidate against
# a histogram op. It matters only through the threshold, so getting it wrong
# moves the size at which the scan starts fanning out and moves nothing else.
# bench/bench_threshold.mojo is the harness that would settle it.
comptime SPLIT_CANDIDATE_OPS = 8


def split_scan_ops(n_active: Int, n_bins: Int, two_sided: Bool) -> Int:
    """Work estimate for one node's split scan over `n_active` features.

    Every ordinary bin of every scanned feature is a threshold, and a feature
    with a reserved missing bin scores each threshold twice, once with the
    missing rows left and once with them right (`two_sided`). The totals pass
    that precedes each feature's scan is one streaming read of the same
    slice, cheap next to the candidates, and is counted as one op per bin.

    Categorical features are not this shape at all: their candidate count is
    a sorted partition search over categories rather than a walk over bins.
    They are estimated here as if they were ordinal, which under-counts a
    wide categorical node. That only ever makes the scan more conservative
    about fanning out.
    """
    var active = n_active if n_active > 0 else 0
    var bins = n_bins if n_bins > 0 else 0
    var per_bin = 2 * SPLIT_CANDIDATE_OPS if two_sided else SPLIT_CANDIDATE_OPS
    return active * bins * (per_bin + 1)


def describe_cpu_policy(profile: CpuProfile, plan: AccumulationPlan) -> String:
    """One line carrying both halves of the width decision.

    The utilization figure is appended here rather than in
    `AccumulationPlan.describe` because it needs the core count, which the plan
    does not carry. It is the number that makes a wide group's cost visible
    next to its benefit; a benchmark header that prints only the width is
    reporting half of the trade.
    """
    var cores = profile.dispatch_cores()
    return String(
        profile.describe(),
        " | ",
        plan.describe(),
        " rounds=",
        dispatch_rounds(plan.group_count, cores),
        " utilization=",
        dispatch_utilization_percent(plan.group_count, cores),
        "%",
    )
