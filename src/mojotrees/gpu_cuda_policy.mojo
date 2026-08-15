"""NVIDIA/CUDA backend specialization policy.

What one backend adds on top of the portable contract, and nothing else.
`gpu_portability.mojo` says what the shared GPU source is allowed to
require of *any* device MAX opens; `gpu_tiling.mojo` says how a launch
geometry is arithmetically derived from bounds; this module says which
bounds a CUDA device actually reports, and what follows from the two
attributes Metal refuses and CUDA answers.

It is a pure policy layer, in the shape of `apple_gpu_policy.mojo`:
reported numbers and a dataset shape in, a launch plan out. It opens no
`DeviceContext`, reads no attribute, allocates nothing, launches nothing,
and reads no environment. Every decision below is therefore exercisable on
a machine with no accelerator, which is the only way any of it could be
exercised here at all.

**No hardware validation is claimed for any of this.** This repository has
never executed a GPU kernel on an NVIDIA device.
`gpu_backend_policy.backend_support(API_CUDA)` is `SUPPORT_PORTABLE`, every
CUDA row in `docs/GPU_VALIDATION.md` reads "not run", and every function
here that would select a specialized kernel variant routes through
`require_specializations_allowed`, which refuses on an unexercised backend
unless `MOJOTREES_GPU_BACKEND_UNVALIDATED=1` acknowledges it.

What this module encodes
------------------------
Only differences that are (a) reported by the device through an attribute
Mojo's `DeviceContext.get_attribute` exposes, or (b) documented properties
of the CUDA programming model that the shared source is already subject to.
Nothing is encoded from a chip name, a compute capability number, a
marketing tier, or a guess about throughput.

- **Subgroup width.** CUDA answers `DeviceAttribute.WARP_SIZE`; Metal
  refuses it (`docs/GPU_VALIDATION.md`, the M4 capture). A *reported* width
  is used, as the granularity a threadgroup width is rounded to. An
  unreported one stays 0, meaning unknown, and the portable
  `gpu_tiling.WARP_GRANULARITY` rounding applies instead. `CUDA_WARP_LANES`
  below is never substituted for a missing report; it is only used to
  refuse a reported value that cannot be a CUDA warp, which would mean the
  attribute was misread.
- **Occupancy bounds.** CUDA answers
  `MAX_THREADS_PER_MULTIPROCESSOR` and `MAX_BLOCKS_PER_MULTIPROCESSOR`;
  Metal refuses both, which is why `gpu_tiling.TARGET_BLOCKS_PER_SM` is a
  fixed 8 and why `apple_gpu_policy.resident_blocks_per_core` estimates
  residency from the per-*block* shared-memory limit instead. That estimate
  is not reproduced here: the per-multiprocessor shared-memory pool is not
  a member of `DeviceAttribute` at all, so dividing the per-block limit by
  a per-block footprint answers a different question than the one
  residency asks. This module derives residency from the two attributes
  that do answer it and reports which of them bound the result, and falls
  back to the portable target when neither is reported, so an unreported
  device plans exactly as it does today.
- **Static threadgroup memory.** CUDA caps *statically* declared shared
  memory at 48 KiB per block; going above it requires an opt-in dynamic
  allocation the shared source does not take (its histogram planes are
  `stack_allocation[..., AddressSpace.SHARED]`, which is static by
  construction). The shipping kernels request 3 KiB, so this ceiling is
  inert today and is checked so that stays a fact rather than an
  assumption.
- **Packed-bin alignment.** The portable packed window is four one-byte
  cells per 32-bit word (`gpu_histogram_specializations.PACK_LANES`), and
  that arithmetic is not re-derived here. What is added is the CUDA memory
  transaction granularity the window is measured against, so a benchmark
  can say whether a body of packed loads covers whole sectors or fragments
  of them.
- **Allocation and transfer.** Unified memory does not follow from the API
  name on CUDA: a discrete card, an integrated part, and managed memory
  are indistinguishable from "cuda". It therefore has to be reported, and
  `DeviceReport.unified_memory` defaults False, which routes every buffer
  through the staged copy `unified_memory_policy.plan_session_routes`
  already selects for a non-unified device.
- **Streams and synchronization.** One in-order queue. That is the model
  `gpu_runtime.HazardTracker` rests on and the only one this package uses;
  nothing here launches on a second stream, and asking for concurrent
  queues is a capability error rather than a silently serialized launch.

What this module deliberately does NOT encode
---------------------------------------------
- **No strategy preference.** `preferred_strategy` returns
  `STRATEGY_AUTO` on every shape. NVIDIA device-memory atomics are widely
  held to be cheaper than the tiled reduction at low contention, and that
  is exactly the kind of claim this repository has no measurement for on
  any NVIDIA part. `StrategyInputs` reports what such a rule would key on
  and carries `TILED_PREFERENCE_UNMEASURED` where the threshold would go,
  in the shape `apple_gpu_policy.CrossoverInputs` uses for the CPU/GPU
  crossover it also declines to invent.
- **No Float64.** `gpu_portability.require_device_float64` refuses it on
  every backend including this one. NVIDIA has Float64; Apple silicon does
  not, and one source takes the weakest backend's floor. Relaxing that for
  CUDA is a specialization needing the same evidence as any other, and it
  is not taken here.
- **No compute-capability branch.** `DeviceContext.compute_capability()`
  exists and nothing below reads it. A capability number would be a proxy
  for a generation, and branching on a generation is what
  `apple_gpu_policy.mojo` explains at length that this project does not do.
- **No register-pressure bound.** `MAX_REGISTERS_PER_BLOCK` is reported and
  is carried on `DeviceReport` for diagnostics, but registers *per thread*
  for the compiled kernel are not queryable from here, so no occupancy
  bound is derived from it. `Occupancy.registers_bound_known` says so.
- **No second geometry engine.** The tile arithmetic is
  `gpu_tiling.resolve_tiling`, called with CUDA-derived bounds. The
  baseline every plan is compared against is `gpu_tiling.derive_tiling`,
  called rather than reproduced. The launch gate is
  `gpu_portability.require_histogram_launchable`, called rather than
  restated.

Backend-neutral section
-----------------------
The section marked "Backend-neutral" below (`DeviceReport`, `Occupancy`,
`BackendLaunchPlan`, `BackendSpecialization`, and their helpers) is not
CUDA-specific and `gpu_amd_policy.mojo` imports it from here rather than
carrying a second copy. It lives in this file because this lane owns only
`gpu_cuda_policy.mojo` and `gpu_amd_policy.mojo`; its home is
`gpu_portability.mojo`, and `handoffs/remaining_11_gpu_backends.md` carries
the ready-to-apply patch that relocates it there. Until that lands, an
importer should read the neutral names as belonging to the portability
layer that happens to be spelled here.

Where this module sits
----------------------
Above `gpu_tiling.mojo`, `gpu_histogram_specializations.mojo`,
`gpu_backend_policy.mojo`, `gpu_portability.mojo`, and
`unified_memory_policy.mojo`; below anything that launches. That direction
is fixed: none of those may import this module, or the layering closes into
a cycle. Nothing in the shipping path imports it yet, which
`docs/GPU_BACKEND_SPECIALIZATIONS.md` records as the honest status and the
handoff carries the call-site patches for.

LightGBM difference: LightGBM ships a separate CUDA device type with its
own kernels, so its CUDA behavior can diverge from its OpenCL behavior
arbitrarily. mojotrees has one source and one `gpu` device value, so a
CUDA difference has to be expressible as a bound handed to the shared
arithmetic. Everything in this module is such a bound.
"""

from .apple_gpu_policy import (
    API_CUDA,
    APPLE_GEN_UNKNOWN,
    GpuProfile,
    api_name,
    partial_budget_bytes,
)
from .gpu_backend_policy import backend_support, support_name
from .gpu_histogram_specializations import (
    PACK_LANES,
    BinStorageDescriptor,
    DeviceHistogramCapabilities,
    KernelFeatures,
    PackedLoadWindow,
    bin_capacity_for,
    plan_packed_window_for,
)
from .gpu_portability import (
    REQ_IN_ORDER_QUEUE,
    BackendContract,
    contract_from_profile,
    describe_contract,
    histogram_capabilities,
    kernel_shared_request,
    require_histogram_launchable,
    require_specializations_allowed,
)
from .gpu_tiling import (
    BYTES_PER_PARTIAL_CELL,
    FALLBACK_MAX_THREADS_PER_BLOCK,
    FALLBACK_SHARED_MEMORY_PER_BLOCK,
    FALLBACK_SM_COUNT,
    STRATEGY_ATOMIC,
    STRATEGY_AUTO,
    STRATEGY_TILED,
    TARGET_BLOCK_THREADS,
    TARGET_BLOCKS_PER_SM,
    WARP_GRANULARITY,
    DeviceCaps,
    HistogramTiling,
    clamp_block_threads,
    derive_tiling,
    resolve_tiling,
    strategy_name,
)
from .unified_memory_policy import SessionMemoryPlan, plan_session_routes


# =====================================================================
# Backend-neutral. Relocation target: gpu_portability.mojo. See the
# module docstring and handoffs/remaining_11_gpu_backends.md.
# =====================================================================


# A reported attribute this backend did not answer. Zero rather than a
# negative sentinel because `DeviceContext.get_attribute` returning a
# nonpositive value and raising are the same fact to a policy layer, and
# `gpu_tiling._attribute_or` already collapses them that way.
comptime ATTR_UNREPORTED = 0


def _reported_or(value: Int, default: Int) -> Int:
    """A reported attribute, or `default` when it is absent or nonsense.

    The same sanitizing `gpu_tiling._attribute_or` applies at the query and
    `apple_gpu_policy._positive_or` applies at the profile, restated here
    only because both are private to their modules.
    """
    if value > 0:
        return value
    return default


def _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


def _bool_text(value: Bool) -> String:
    if value:
        return String("true")
    return String("false")


@fieldwise_init
struct DeviceReport(Copyable, Movable):
    """Every device attribute this package knows how to ask for, as this
    device answered it.

    One field per line of `bench/bench_gpu_validation._report_device`, which
    is the one place in this repository that queries the full set, so a
    caller holding an open `DeviceContext` fills this in mechanically and
    the policy layer never queries anything itself. Zero means the backend
    refused the query or answered nonsense; Metal refuses six of these and
    CUDA answers all of them, which is the whole reason this struct exists
    alongside `gpu_tiling.DeviceCaps` rather than replacing it.

    `DeviceCaps` stays the three attributes the portable geometry needs.
    This is those three plus the ones only some backends answer, and it
    projects down to a `DeviceCaps` and to a `GpuProfile` so nothing
    downstream has to learn a fourth shape for the same device.
    """

    var multiprocessor_count: Int
    var warp_size: Int
    """Lanes in lockstep, as reported. 0 on any backend that refuses the
    query, which is Metal. Never defaulted to an architectural constant:
    see `subgroup_width`."""

    var max_threads_per_block: Int
    var max_threads_per_multiprocessor: Int
    var max_blocks_per_multiprocessor: Int
    var max_shared_memory_per_block: Int
    var max_registers_per_block: Int
    """Reported for diagnostics only. Registers per thread for the compiled
    kernel are not queryable from here, so no occupancy bound is derived
    from this; see `Occupancy.registers_bound_known`."""

    var max_grid_dim_x: Int
    var max_grid_dim_y: Int
    """What this device allows. The bound the shared source is written to
    is the tightest of the three backends, which lives on
    `gpu_portability.BackendContract.max_grid_dim_y`; a device reporting
    more does not widen it, because one source targets all three."""

    var memory_budget_bytes: Int
    """Device memory the policy may plan against, or 0 for unreported, in
    which case the partial buffer falls back to the portable ceiling."""

    var unified_memory: Bool
    """Whether the budget above is shared with the host. It does not follow
    from the API name outside Metal, so it has to be reported: a discrete
    card, an integrated part, CUDA managed memory, and an XNACK-enabled HIP
    device are all "cuda" or "hip" to a name query. False is the
    conservative answer and the one that keeps every buffer on the staged
    copy route."""

    @staticmethod
    def unreported() -> DeviceReport:
        """A device that answered nothing. Every derivation below falls
        back to the portable constant for it, so a plan built from this is
        the plan `gpu_tiling.derive_tiling` already produces."""
        return DeviceReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)

    @staticmethod
    def queried(
        multiprocessor_count: Int = 0,
        warp_size: Int = 0,
        max_threads_per_block: Int = 0,
        max_threads_per_multiprocessor: Int = 0,
        max_blocks_per_multiprocessor: Int = 0,
        max_shared_memory_per_block: Int = 0,
        max_registers_per_block: Int = 0,
        max_grid_dim_x: Int = 0,
        max_grid_dim_y: Int = 0,
        memory_budget_bytes: Int = 0,
        unified_memory: Bool = False,
    ) -> DeviceReport:
        """A report from a caller that queried some attributes and not
        others. Every argument defaults to unreported, so a call site adds
        attributes as it learns how to read them without the policy layer
        changing shape."""
        return DeviceReport(
            multiprocessor_count,
            warp_size,
            max_threads_per_block,
            max_threads_per_multiprocessor,
            max_blocks_per_multiprocessor,
            max_shared_memory_per_block,
            max_registers_per_block,
            max_grid_dim_x,
            max_grid_dim_y,
            memory_budget_bytes,
            unified_memory,
        )

    def caps(self) -> DeviceCaps:
        """The three attributes the portable geometry plans from, sanitized
        exactly as `gpu_tiling.query_device_caps` sanitizes them, so a plan
        derived from this report and the baseline it is compared against see
        the same device."""
        return DeviceCaps(
            _reported_or(self.multiprocessor_count, FALLBACK_SM_COUNT),
            _reported_or(
                self.max_threads_per_block, FALLBACK_MAX_THREADS_PER_BLOCK
            ),
            _reported_or(
                self.max_shared_memory_per_block,
                FALLBACK_SHARED_MEMORY_PER_BLOCK,
            ),
        )

    def profile(self, api: Int) -> GpuProfile:
        """This report as the `GpuProfile` the device-policy stack already
        consumes.

        `apple_generation` is `APPLE_GEN_UNKNOWN` because it is meaningful
        only on Metal and `GpuProfile.from_reported` only parses it there;
        `synthetic` is False because a report is a reading. A caller that
        built this from `DeviceReport.unreported()` is not holding a
        reading and should say so by constructing the profile itself, the
        same caveat `apple_histogram_policy.profile_from_caps` carries.
        """
        return GpuProfile(
            api,
            APPLE_GEN_UNKNOWN,
            _reported_or(self.multiprocessor_count, FALLBACK_SM_COUNT),
            _reported_or(
                self.max_threads_per_block, FALLBACK_MAX_THREADS_PER_BLOCK
            ),
            _reported_or(
                self.max_shared_memory_per_block,
                FALLBACK_SHARED_MEMORY_PER_BLOCK,
            ),
            _reported_or(self.memory_budget_bytes, 0),
            self.unified_memory,
            False,
        )

    def answered(self) -> Int:
        """How many of the ten numeric attributes this device answered.

        Reported so a diagnostic can say "this backend answered four of ten"
        rather than leaving the reader to infer it from zeroes, and so a
        capture on new hardware can be compared against the Metal capture in
        `docs/GPU_VALIDATION.md`, which answers five.
        """
        var n = 0
        if self.multiprocessor_count > 0:
            n += 1
        if self.warp_size > 0:
            n += 1
        if self.max_threads_per_block > 0:
            n += 1
        if self.max_threads_per_multiprocessor > 0:
            n += 1
        if self.max_blocks_per_multiprocessor > 0:
            n += 1
        if self.max_shared_memory_per_block > 0:
            n += 1
        if self.max_registers_per_block > 0:
            n += 1
        if self.max_grid_dim_x > 0:
            n += 1
        if self.max_grid_dim_y > 0:
            n += 1
        if self.memory_budget_bytes > 0:
            n += 1
        return n


# --- Occupancy ---------------------------------------------------------

comptime RESIDENCY_PORTABLE_TARGET = 0
"""Neither per-multiprocessor attribute was reported, so the fixed
`gpu_tiling.TARGET_BLOCKS_PER_SM` stands. This is what Metal gets, and it
is what every plan gets today."""

comptime RESIDENCY_BY_BLOCKS = 1
"""`MAX_BLOCKS_PER_MULTIPROCESSOR` bound the result."""

comptime RESIDENCY_BY_THREADS = 2
"""`MAX_THREADS_PER_MULTIPROCESSOR / block_threads` bound the result."""

comptime RESIDENCY_BY_BOTH = 3
"""Both were reported and agreed on the same number."""


def residency_source_name(source: Int) -> String:
    if source == RESIDENCY_BY_BLOCKS:
        return String("max_blocks_per_multiprocessor")
    if source == RESIDENCY_BY_THREADS:
        return String("max_threads_per_multiprocessor")
    if source == RESIDENCY_BY_BOTH:
        return String("both_multiprocessor_limits")
    return String("portable_target")


@fieldwise_init
struct Occupancy(Copyable, Movable):
    """How many threadgroups this policy expects resident on one
    multiprocessor, and which reported bound produced that number."""

    var blocks_per_multiprocessor: Int
    var source: Int
    var by_blocks: Int
    """The `MAX_BLOCKS_PER_MULTIPROCESSOR` bound, or 0 when unreported."""

    var by_threads: Int
    """`MAX_THREADS_PER_MULTIPROCESSOR // block_threads`, or 0 when the
    attribute was unreported."""

    var shared_bound_known: Bool
    """Always False. The per-multiprocessor shared-memory pool is not a
    member of `DeviceAttribute` at all (`docs/GPU_VALIDATION.md` records
    that), so the shared-memory limit on residency cannot be computed from
    anything this package can query. Carried as a field rather than left
    implicit because a residency figure that silently omits a bound reads
    like one that considered it."""

    var registers_bound_known: Bool
    """Always False, and for the same kind of reason:
    `MAX_REGISTERS_PER_BLOCK` is reported but registers *per thread* for the
    compiled kernel are not, and the quotient is what a register bound
    needs."""

    def is_portable_target(self) -> Bool:
        """Whether this is the fixed target rather than a derived bound,
        which is the case a plan must be able to distinguish: it means the
        occupancy figure carries no device information at all."""
        return self.source == RESIDENCY_PORTABLE_TARGET


def resident_blocks_from_reported(
    max_threads_per_multiprocessor: Int,
    max_blocks_per_multiprocessor: Int,
    block_threads: Int,
) raises -> Occupancy:
    """Threadgroups resident on one multiprocessor, from the two attributes
    that answer the question.

    The tightest of the reported bounds, floored at one, and
    `gpu_tiling.TARGET_BLOCKS_PER_SM` when neither was reported. That
    fallback is what makes this a conservative addition: a device that
    answers nothing plans exactly as `derive_tiling` plans for it today.

    Deliberately not `apple_gpu_policy.resident_blocks_per_core`'s rule,
    which divides the per-*block* shared-memory limit by a per-block
    footprint. On a backend that reports these two attributes that estimate
    answers a different question, and on one that does not, this falls back
    to the same fixed target `derive_tiling` uses rather than to an estimate
    built from a limit that was never a pool.
    """
    if block_threads < 1:
        raise Error("a threadgroup needs a positive width")

    var by_blocks = 0
    if max_blocks_per_multiprocessor > 0:
        by_blocks = max_blocks_per_multiprocessor
    var by_threads = 0
    if max_threads_per_multiprocessor > 0:
        by_threads = max_threads_per_multiprocessor // block_threads

    if by_blocks < 1 and by_threads < 1:
        return Occupancy(
            TARGET_BLOCKS_PER_SM,
            RESIDENCY_PORTABLE_TARGET,
            by_blocks,
            by_threads,
            False,
            False,
        )

    var resident: Int
    var source: Int
    if by_blocks < 1:
        resident = by_threads
        source = RESIDENCY_BY_THREADS
    elif by_threads < 1:
        resident = by_blocks
        source = RESIDENCY_BY_BLOCKS
    elif by_blocks < by_threads:
        resident = by_blocks
        source = RESIDENCY_BY_BLOCKS
    elif by_threads < by_blocks:
        resident = by_threads
        source = RESIDENCY_BY_THREADS
    else:
        resident = by_blocks
        source = RESIDENCY_BY_BOTH

    # A threadgroup wider than a whole multiprocessor's thread budget still
    # launches; it just runs one at a time. Flooring here keeps the tile
    # arithmetic's `target_blocks >= 1` precondition rather than inventing
    # parallelism.
    if resident < 1:
        resident = 1
    return Occupancy(resident, source, by_blocks, by_threads, False, False)


# --- Partial-histogram strategy inputs ---------------------------------

comptime TILED_PREFERENCE_UNMEASURED = -1
"""`StrategyInputs.min_conflict_for_tiled` when no measurement supports
preferring the tiled reduction over device atomics at any contention, which
is every case on every backend. Negative disables, in the shape
`apple_gpu_policy.CROSSOVER_DISABLED` uses for the CPU/GPU threshold it
also declines to invent."""


def atomic_conflict_degree(strategy: Int, n_tiles: Int) raises -> Int:
    """Threadgroups that fold into the same output histogram cell under a
    resolved strategy.

    A structural property of the two flushes, not a measurement. Under
    `STRATEGY_ATOMIC` every one of a feature's row tiles fetch-adds into
    that feature's output slice, so the degree is the tile count. Under
    `STRATEGY_TILED` each tile owns its own slot in the partial buffer and
    nothing else writes it, so the degree is one and the second kernel sums
    the slots in ascending order. That difference is the whole reason the
    tiled path exists, and it is the quantity a measured strategy rule
    would key on.

    `STRATEGY_AUTO` is a request rather than a resolution and has no
    degree, matching `gpu_tiling.launches_for_strategy`.
    """
    if n_tiles < 1:
        raise Error("a launch covers a positive number of row tiles")
    if strategy == STRATEGY_ATOMIC:
        return n_tiles
    if strategy == STRATEGY_TILED:
        return 1
    raise Error(
        "an unresolved strategy has no atomic conflict degree; resolve it"
        " with derive_tiling first"
    )


@fieldwise_init
struct StrategyInputs(Copyable, Movable):
    """What a measured partial-histogram strategy rule would key on, and
    the threshold it would produce.

    Reporting only. Nothing reads `min_conflict_for_tiled` to choose a
    strategy: `gpu_tiling.resolve_tiling` resolves `STRATEGY_AUTO` by tile
    count, which is the rule that ships on every backend, and
    `preferred_strategy` below returns `STRATEGY_AUTO` so that rule keeps
    deciding. This struct exists so the benchmark that would justify a
    backend-specific rule has a defined set of inputs to regress against
    and a defined place to put its answer.
    """

    var api: Int
    var n_tiles: Int
    var n_slots: Int
    var n_bins: Int
    var block_threads: Int
    var resident_blocks: Int
    var atomic_conflict_degree: Int
    """Under the resolved strategy: see the function of the same name."""

    var partial_cells: Int
    """`n_tiles * n_slots * n_bins` under the tiled strategy, 0 under the
    atomic one, which is what `HistogramTiling.partial_cells` reports."""

    var partial_bytes: Int
    """`partial_cells * BYTES_PER_PARTIAL_CELL`: the device memory the
    tiled strategy trades for the atomic traffic it removes. The other
    half of the trade a rule would weigh."""

    var min_conflict_for_tiled: Int
    """Conflict degree at or above which the tiled reduction would be
    preferred. `TILED_PREFERENCE_UNMEASURED` in every case this module can
    construct."""


def strategy_inputs(
    api: Int,
    tiling: HistogramTiling,
    n_slots: Int,
    n_bins: Int,
    resident_blocks: Int,
) raises -> StrategyInputs:
    """The strategy-rule inputs implied by a resolved geometry."""
    if n_slots < 1 or n_bins < 1:
        raise Error("strategy inputs need positive features and bins")
    return StrategyInputs(
        api,
        tiling.n_tiles,
        n_slots,
        n_bins,
        tiling.block_threads,
        resident_blocks,
        atomic_conflict_degree(tiling.strategy, tiling.n_tiles),
        tiling.partial_cells,
        tiling.partial_cells * BYTES_PER_PARTIAL_CELL,
        TILED_PREFERENCE_UNMEASURED,
    )


def preferred_strategy(inputs: StrategyInputs) -> Int:
    """Which accumulation strategy this backend would rather run.

    `STRATEGY_AUTO` on every shape, on every backend, which means "the
    portable rule in `gpu_tiling.resolve_tiling` decides". It is a function
    rather than a constant so a measured backend rule has one place to be
    installed, and so a caller that consults it is already written to be
    told "no preference".

    Do not fill this in from reasoning about which backend's atomics are
    fast. Fill it in from a recorded sweep, cite the record where
    `device_policy.CrossoverEvidence.evidence_id` cites one, and set
    `min_conflict_for_tiled` on the inputs at the same time. The inputs are
    taken rather than ignored so that installation is a change to this
    body alone.
    """
    return STRATEGY_AUTO


# --- The resolved plan -------------------------------------------------


@fieldwise_init
struct BackendLaunchPlan(Copyable, Movable):
    """A resolved histogram launch for one backend, one reported device,
    and one node shape, beside the portable plan it would otherwise be.

    `tiling` is the geometry to launch and is the same `HistogramTiling`
    every launch site in this package already takes, so consuming a plan
    costs a call site nothing but the field access. `baseline` is what
    `gpu_tiling.derive_tiling` would have launched for the same node on the
    same device, carried so a caller can compare, a benchmark can report
    both, and a backend plan that misbehaves can be dropped for the
    baseline at the call site without re-deriving anything.
    """

    var api: Int
    var block_threads: Int
    var block_width_granularity: Int
    """The multiple `block_threads` was rounded to: the reported subgroup
    width when the device answered `WARP_SIZE`, and
    `gpu_tiling.WARP_GRANULARITY` otherwise."""

    var subgroup_width: Int
    """Reported, or 0 for unknown. Nothing divides by it."""

    var occupancy: Occupancy
    var target_blocks: Int
    """`multiprocessor_count * occupancy.blocks_per_multiprocessor`: the
    threadgroups this plan wants device-wide, which is the occupancy bound
    handed to `resolve_tiling`."""

    var shared_bytes_per_block: Int
    """What the kernel this build compiled really requests, from
    `gpu_portability.kernel_shared_request`, not the `n_bins * 12` model."""

    var amortize_bins: Int
    """The bin width a threadgroup's partial histogram really occupies, as
    handed to `resolve_tiling`. `n_bins` unless the specialized kernels are
    compiled in, in which case it is the instantiated capacity."""

    var partial_cell_limit: Int
    var tiling: HistogramTiling
    var baseline: HistogramTiling
    var strategy: StrategyInputs

    var selected: KernelFeatures
    """The specialized kernel variants this plan actually chose, which is
    what `gpu_portability.require_specializations_allowed` gates on and is
    deliberately not the same as the set compiled into the build.

    A plan from this layer selects the bin-capacity kernel exactly when it
    is compiled in, because its shared-memory footprint and its tile
    amortization already assume that narrower partial histogram. It never
    selects packed loads or batched leaves: choosing those is the
    specialization ladder's job (`apple_histogram_policy.mojo`), and a
    second selector here would be a second ladder.
    """

    def matches_baseline(self) -> Bool:
        """Whether this plan's geometry is the portable one. True whenever
        the device reported nothing beyond the three attributes
        `DeviceCaps` already carries, which is the property that makes this
        module a conservative addition rather than a behavior change."""
        return (
            self.tiling.strategy == self.baseline.strategy
            and self.tiling.block_threads == self.baseline.block_threads
            and self.tiling.n_tiles == self.baseline.n_tiles
            and self.tiling.rows_per_tile == self.baseline.rows_per_tile
        )

    def gpu_launches(self) raises -> Int:
        """Kernel launches this node costs, from the resolved strategy. The
        number `hybrid_leaf_scheduler.LeafWork` needs and must not guess."""
        return self.tiling.launches()


@fieldwise_init
struct BackendSpecialization(Copyable, Movable):
    """The whole per-backend descriptor, for a diagnostic record, a bug
    report, or `docs/GPU_BACKEND_SPECIALIZATIONS.md`.

    Every field is either a reported number, a documented property of the
    backend's programming model, or a constant already defined lower in the
    stack. None of them is a measurement, and `support` is the field that
    says so: it is `SUPPORT_PORTABLE` for CUDA and HIP alike, because this
    repository has never run a kernel on either.
    """

    var api: Int
    var support: Int
    var subgroup_width: Int
    var subgroup_width_reported: Bool
    var launch_granularity: Int
    var static_shared_ceiling_bytes: Int
    var reported_shared_per_block: Int
    var coalescing_bytes: Int
    """The memory transaction granularity a packed load is measured
    against: 32-byte sectors on CUDA, 64-byte cache lines on HIP."""

    var native_wide_load_bytes: Int
    """The widest single load the backend offers (16 bytes on both), which
    the shared source does not emit. `PACK_LANES` is what it does emit, and
    the gap is what a packed-load specialization would be reaching for."""

    var pack_alignment_bytes: Int
    var concurrent_queues: Bool
    var unified_memory_inferable: Bool
    """Whether unified memory follows from the API name. True on Metal
    alone; on CUDA and HIP it has to be reported."""

    var device_float64_permitted: Bool
    var specializations_compiled_in: Bool


def describe_specialization(spec: BackendSpecialization) -> String:
    """One line naming the backend and every bound this policy applies to
    it."""
    return String(
        "backend=",
        api_name(spec.api),
        " support=",
        support_name(spec.support),
        " subgroup_width=",
        spec.subgroup_width,
        " (reported=",
        _bool_text(spec.subgroup_width_reported),
        ") granularity=",
        spec.launch_granularity,
        " static_shared_ceiling=",
        spec.static_shared_ceiling_bytes,
        " reported_shared=",
        spec.reported_shared_per_block,
        " coalescing=",
        spec.coalescing_bytes,
        " native_wide_load=",
        spec.native_wide_load_bytes,
        " pack_alignment=",
        spec.pack_alignment_bytes,
        " concurrent_queues=",
        _bool_text(spec.concurrent_queues),
        " unified_inferable=",
        _bool_text(spec.unified_memory_inferable),
        " float64=",
        _bool_text(spec.device_float64_permitted),
        " specialized_kernels=",
        _bool_text(spec.specializations_compiled_in),
    )


def describe_plan(plan: BackendLaunchPlan) -> String:
    """One line for benchmark output and bug reports: which backend the
    plan was made for, what bounded it, and whether it differs from the
    portable geometry at all."""
    return String(
        "backend=",
        api_name(plan.api),
        " strategy=",
        strategy_name(plan.tiling.strategy),
        " threads=",
        plan.block_threads,
        "/",
        plan.block_width_granularity,
        " resident=",
        plan.occupancy.blocks_per_multiprocessor,
        " (",
        residency_source_name(plan.occupancy.source),
        ") target_blocks=",
        plan.target_blocks,
        " tiles=",
        plan.tiling.n_tiles,
        " rows_per_tile=",
        plan.tiling.rows_per_tile,
        " amortize_bins=",
        plan.amortize_bins,
        " shared=",
        plan.shared_bytes_per_block,
        " partial=",
        plan.tiling.partial_cells,
        "/",
        plan.partial_cell_limit,
        " conflict=",
        plan.strategy.atomic_conflict_degree,
        " matches_baseline=",
        _bool_text(plan.matches_baseline()),
    )


# --- Shared derivations, parameterized by backend ----------------------
#
# The three functions below take the backend-specific numbers as arguments
# so `gpu_amd_policy.mojo` can run the identical arithmetic against its own
# constants. They are the reason there is no second copy of the block-width
# rule, the shared-memory gate, or the plan assembly.


def derive_block_threads_for(
    report: DeviceReport, n_rows: Int, granularity: Int
) raises -> Int:
    """Threads per threadgroup: the portable target, bounded by the rows
    available, clamped to the reported device maximum, and rounded down to
    `granularity`.

    The row bound is the shape-driven part and it is not an Apple rule
    despite `apple_gpu_policy.derive_block_threads` also applying it: a
    256-thread block scanning 40 rows leaves seven eighths of its lanes with
    nothing to accumulate on any device. What is backend-specific is only
    `granularity`, which is the reported subgroup width when the device
    answered `WARP_SIZE` and the portable `gpu_tiling.WARP_GRANULARITY`
    otherwise.

    When the width was not reported this delegates to
    `gpu_tiling.clamp_block_threads`, so an unreported device gets the
    identical width the shipping path gives it, minus the row bound.
    """
    if granularity < 1:
        raise Error("a threadgroup width rounds to a positive granularity")
    var maximum = _reported_or(
        report.max_threads_per_block, FALLBACK_MAX_THREADS_PER_BLOCK
    )
    var threads = TARGET_BLOCK_THREADS
    if n_rows > 0 and n_rows < threads:
        threads = n_rows
    if granularity == WARP_GRANULARITY:
        return clamp_block_threads(threads, maximum)
    if threads > maximum:
        threads = maximum
    threads = (threads // granularity) * granularity
    if threads < granularity:
        threads = granularity
    # A device whose reported maximum is narrower than its reported
    # subgroup width is self-inconsistent; the maximum is the one that
    # would actually fail the launch, so it wins.
    if threads > maximum:
        threads = maximum
    if threads < 1:
        raise Error(
            "this device reports a maximum threadgroup width of ",
            maximum,
            ", which cannot hold a threadgroup",
        )
    return threads


def require_shared_within_ceiling(
    api: Int, shared_bytes: Int, ceiling: Int
) raises:
    """Refuse a statically declared threadgroup allocation the backend's
    programming model will not accept.

    A *model* limit rather than a device one, and separate from the
    reported per-block limit for that reason: CUDA caps statically declared
    shared memory at 48 KiB whatever the device advertises, and going above
    it requires an opt-in dynamic allocation the shared source does not
    take, because its histogram planes are
    `stack_allocation[..., AddressSpace.SHARED]`, which is static by
    construction. The shipping kernels request 3 KiB, so nothing reaches
    this today; it is checked so that stays a fact.
    """
    if shared_bytes < 1:
        raise Error(
            "a threadgroup histogram needs positive threadgroup memory; got ",
            shared_bytes,
        )
    if shared_bytes <= ceiling:
        return
    raise Error(
        "the compiled kernel statically declares ",
        shared_bytes,
        " bytes of threadgroup memory per block, above the ",
        ceiling,
        " bytes the ",
        api_name(api),
        " programming model allows a static declaration; the shared GPU"
        " source allocates its histogram planes statically and does not"
        " take the dynamic-shared-memory opt-in",
    )


def require_shared_reported_fits(
    api: Int, report: DeviceReport, shared_bytes: Int
) raises:
    """Refuse an allocation larger than this device advertises per block.

    The device half of the same question. Skipped, not assumed, when the
    device did not answer `MAX_SHARED_MEMORY_PER_BLOCK`: refusing on an
    unreported attribute would refuse every launch on a backend that
    answers nothing, and `gpu_portability.require_launch_geometry` checks
    the same bound against the sanitized `DeviceCaps` at launch time.
    """
    if report.max_shared_memory_per_block <= 0:
        return
    if shared_bytes <= report.max_shared_memory_per_block:
        return
    raise Error(
        "the compiled kernel requests ",
        shared_bytes,
        " bytes of threadgroup memory per block and this ",
        api_name(api),
        " device reports only ",
        report.max_shared_memory_per_block,
    )


def derive_plan_for(
    api: Int,
    report: DeviceReport,
    granularity: Int,
    static_shared_ceiling: Int,
    n_rows: Int,
    n_slots: Int,
    n_bins: Int,
    compiled: KernelFeatures,
    requested_strategy: Int = STRATEGY_AUTO,
    max_partial_cells: Int = 0,
) raises -> BackendLaunchPlan:
    """Resolve one node's histogram launch for a backend.

    `compiled` is what this build linked in, which is the distinction
    `gpu_portability.require_specializations_allowed` turns on: linking the
    batched module compiles its kernels whether or not anything selects
    them. What this plan *selects* is derived from `compiled` and reported
    on `BackendLaunchPlan.selected`, and it is that set, not this argument,
    that the unexercised-backend gate is applied to.

    Pure host arithmetic. The tile arithmetic is
    `gpu_tiling.resolve_tiling`, called with three bounds this layer
    derives differently from the portable path:

    - threadgroups wanted device-wide is
      `multiprocessor_count * resident`, with residency from the two
      per-multiprocessor attributes rather than from the fixed target,
      whenever the device reported them;
    - a tile amortizes the bin width the partial histogram really occupies,
      which is the kernel's instantiated capacity when the specialized
      kernels are compiled in and `n_bins` otherwise;
    - the partial buffer is bounded by the reported device memory budget as
      well as by any buffer already allocated.

    Everything else, including the `MOJOTREES_GPU_ROW_TILE`,
    `MOJOTREES_GPU_BLOCK_THREADS`, and `MOJOTREES_GPU_HIST_STRATEGY`
    overrides, is the shared rule and reaches this plan because it is the
    shared rule that runs.

    `n_rows` of zero plans at one row, matching
    `GpuActiveRows.range_tiling` and `apple_histogram_policy`, so an empty
    node still gets a launchable geometry.

    Raises on a shape that cannot be planned (nonpositive features or bins,
    a bin count past `MAX_BINS` through `bin_capacity_for`), on a
    threadgroup allocation the backend's model or the device refuses, and,
    through `require_specializations_allowed`, on a build whose specialized
    kernel variants are being selected on a backend nobody has run.
    """
    if n_slots < 1:
        raise Error("a histogram plan needs at least one active feature")
    if n_rows < 0:
        raise Error("a node cannot own a negative number of rows")
    var rows = n_rows
    if rows < 1:
        rows = 1

    var caps = report.caps()
    var baseline = derive_tiling(
        caps, rows, n_slots, n_bins, requested_strategy, max_partial_cells
    )

    var profile = report.profile(api)
    var contract = contract_from_profile(profile)

    # What this plan chooses, which is what the unexercised-backend gate
    # applies to. The bin-capacity kernel is chosen exactly when it is
    # compiled in, because the shared footprint and the amortization width
    # below already assume it; packed loads and batched leaves are the
    # specialization ladder's to choose and are never chosen here.
    var selected = KernelFeatures(
        compiled.specialized_bin_kernels, False, False
    )
    require_specializations_allowed(contract, selected)

    var shared_bytes = kernel_shared_request(n_bins, compiled)
    require_shared_within_ceiling(api, shared_bytes, static_shared_ceiling)
    require_shared_reported_fits(api, report, shared_bytes)

    var block_threads = derive_block_threads_for(report, rows, granularity)
    var occupancy = resident_blocks_from_reported(
        report.max_threads_per_multiprocessor,
        report.max_blocks_per_multiprocessor,
        block_threads,
    )
    var target_blocks = (
        caps.sm_count * occupancy.blocks_per_multiprocessor
    )
    if target_blocks < 1:
        target_blocks = 1

    # The width a threadgroup's partial histogram really occupies. The
    # shipping kernels allocate the full `MAX_BINS` width whatever `n_bins`
    # is, but the baseline amortizes against `n_bins`, and staying equal to
    # the baseline wherever no specialized kernel exists is the
    # conservative direction: a wider amortization figure would cut the
    # tile count on every device, which is a behavior change nothing here
    # has measured.
    var amortize_bins = n_bins
    if compiled.specialized_bin_kernels:
        amortize_bins = bin_capacity_for(n_bins)

    var partial_cell_limit = (
        partial_budget_bytes(profile) // BYTES_PER_PARTIAL_CELL
    )
    if max_partial_cells > 0 and max_partial_cells < partial_cell_limit:
        partial_cell_limit = max_partial_cells
    if partial_cell_limit < 1:
        partial_cell_limit = 1

    var tiling = resolve_tiling(
        rows,
        n_slots,
        n_bins,
        block_threads,
        amortize_bins,
        target_blocks,
        partial_cell_limit,
        requested_strategy,
    )

    var reported_width = 0
    if report.warp_size > 0:
        reported_width = report.warp_size
    var inputs = strategy_inputs(
        api,
        tiling,
        n_slots,
        n_bins,
        occupancy.blocks_per_multiprocessor,
    )
    return BackendLaunchPlan(
        api,
        block_threads,
        granularity,
        reported_width,
        occupancy^,
        target_blocks,
        shared_bytes,
        amortize_bins,
        partial_cell_limit,
        tiling^,
        baseline^,
        inputs^,
        selected^,
    )


# =====================================================================
# End backend-neutral section. Everything below is CUDA.
# =====================================================================


# The warp width every CUDA architecture has shipped. Used **only** to
# refuse a reported value that cannot be a CUDA warp, which would mean the
# attribute was misread or the API code is wrong. It is never substituted
# for an unreported width: an unreported width is unknown, and
# `subgroup_width` returns 0 for it.
comptime CUDA_WARP_LANES = 32

# Statically declared shared memory per block, which is what the shared
# source allocates. Raising it needs the dynamic-shared-memory opt-in and a
# `cudaFuncAttributeMaxDynamicSharedMemorySize` call the shared source does
# not make.
comptime CUDA_STATIC_SHARED_CEILING_BYTES = 48 * 1024

# The memory transaction granularity a global load is served at. The unit a
# packed-bin body is measured in, not a threshold anything is compared
# against here.
comptime CUDA_SECTOR_BYTES = 32

# The widest single load CUDA offers (a 128-bit `int4`). The shared source
# emits none: `gpu_histogram_specializations.pack4_bins` packs four one-byte
# cells into a 32-bit word, and that portable implementation is the one that
# runs. Recorded so the gap a packed-load specialization would close is
# stated rather than implied.
comptime CUDA_NATIVE_WIDE_LOAD_BYTES = 16


def require_cuda(api: Int) raises:
    """Refuse an API code this module has no business planning for.

    A capability error rather than a silent fallback, and it refuses
    `API_UNKNOWN` too: a device that did not name its API might be CUDA and
    might not, and planning CUDA bounds for it would apply a 48 KiB static
    shared ceiling and a 32-lane rounding granularity to a device nobody
    identified. The portable path already covers an unidentified device;
    an operator who knows it is CUDA declares it with
    `MOJOTREES_GPU_BACKEND=cuda`, which `device_policy.env_declared_api`
    reads.
    """
    if api == API_CUDA:
        return
    raise Error(
        "gpu_cuda_policy plans for the cuda backend only; got api code ",
        api,
        " (",
        api_name(api),
        "). An unidentified device takes the portable path in"
        " gpu_tiling.mojo; declare MOJOTREES_GPU_BACKEND=cuda if this"
        " device is known to be NVIDIA",
    )


def cuda_contract(report: DeviceReport) raises -> BackendContract:
    """The portability contract for the device this report describes.

    `gpu_portability.contract_from_profile`'s answer, not a second one: the
    grid bound, the primitive set, and the Float64 refusal are properties of
    the shared source and belong to that layer. What this adds is only that
    the profile is built from a CUDA report, so `unified_memory` on the
    contract is what the device said rather than what the API name implies.
    """
    return contract_from_profile(report.profile(API_CUDA))


# --- Subgroup width ----------------------------------------------------


def subgroup_width(report: DeviceReport) -> Int:
    """Lanes in lockstep on this device, or 0 for unknown.

    The reported `WARP_SIZE` and nothing else. `CUDA_WARP_LANES` is not
    substituted for a missing report, for the reason
    `gpu_histogram_specializations.DeviceHistogramCapabilities` gives: 0
    means unknown, nothing in this package divides by a width, and a
    confident wrong width is worse than an honest missing one.
    """
    if report.warp_size > 0:
        return report.warp_size
    return 0


def require_subgroup_width_plausible(report: DeviceReport) raises:
    """Refuse a reported width that cannot be a CUDA warp.

    Not a device check: every CUDA architecture that has shipped has a
    32-lane warp, so a positive report of anything else means the attribute
    was misread, the report was filled in from the wrong device, or the API
    code is wrong. Unreported passes, because unknown is a legitimate
    answer and the portable rounding granularity covers it.
    """
    if report.warp_size <= 0:
        return
    if report.warp_size == CUDA_WARP_LANES:
        return
    raise Error(
        "this device reported a subgroup width of ",
        report.warp_size,
        " for the cuda backend, and every CUDA architecture has a ",
        CUDA_WARP_LANES,
        "-lane warp; the attribute or the backend code is wrong",
    )


def require_subgroup_width_known(report: DeviceReport, what: String) raises:
    """Refuse a specialization that needs a subgroup width on a device that
    did not report one.

    Nothing in this package needs one today, which is why this has no
    caller: no kernel here performs a warp shuffle, a ballot, or a
    warp-level reduction, and `gpu_tiling.WARP_GRANULARITY` is a rounding
    granularity rather than a width claim. It exists so the first
    warp-level specialization has a gate to pass rather than a constant to
    reach for.
    """
    if subgroup_width(report) > 0:
        return
    raise Error(
        "'",
        what,
        "' needs the subgroup width and this device did not report"
        " WARP_SIZE; nothing in this package assumes a width, and"
        " gpu_tiling.WARP_GRANULARITY is a launch-rounding granularity"
        " rather than a device width",
    )


def block_width_granularity(report: DeviceReport) -> Int:
    """The multiple a threadgroup width is rounded to on this device: the
    reported subgroup width, and `gpu_tiling.WARP_GRANULARITY` when the
    device did not report one.

    A reported 32 is finer than the portable 64, so a narrow node can get a
    32-thread block on CUDA where the portable rule floors at 64. That is a
    real difference and it rests entirely on the device having answered the
    query; with no answer this returns the portable granularity and
    `derive_block_threads_for` delegates to `clamp_block_threads`, which is
    the shipping rule unchanged.
    """
    var width = subgroup_width(report)
    if width > 0:
        return width
    return WARP_GRANULARITY


# --- Shared memory, atomics, queue -------------------------------------


def static_shared_ceiling() -> Int:
    """Statically declared threadgroup memory per block CUDA allows."""
    return CUDA_STATIC_SHARED_CEILING_BYTES


def require_shared_memory_supported(
    report: DeviceReport, n_bins: Int, compiled: KernelFeatures
) raises:
    """Refuse a threadgroup allocation this backend or this device will not
    accept, for the kernel this build compiled.

    Both halves in one call: the 48 KiB static ceiling the CUDA programming
    model imposes whatever the device advertises, and the per-block limit
    the device reported. What is checked is
    `gpu_portability.kernel_shared_request`, the footprint the compiled
    kernel really has, not the `n_bins * 12` model whose own docstring
    records that it is optimistic below 256 bins.
    """
    var shared_bytes = kernel_shared_request(n_bins, compiled)
    require_shared_within_ceiling(
        API_CUDA, shared_bytes, CUDA_STATIC_SHARED_CEILING_BYTES
    )
    require_shared_reported_fits(API_CUDA, report, shared_bytes)


def require_in_order_queue(contract: BackendContract) raises:
    """Refuse a backend that does not promise an in-order device queue.

    The property `gpu_runtime.HazardTracker` rests on: device work never
    needs a host synchronization to observe earlier device work, so the
    only required synchronizations are the two host-side hazards it tracks.
    CUDA promises it for a single stream, which is the only queue model this
    package uses.
    """
    if contract.provides(REQ_IN_ORDER_QUEUE):
        return
    raise Error(
        "the ",
        api_name(contract.api),
        " backend does not promise an in-order device queue, which the"
        " synchronization model in gpu_runtime.mojo depends on",
    )


def concurrent_queues_available() -> Bool:
    """Whether this package can launch on more than one device queue.

    False. `DeviceContext` is used here through `enqueue_create_buffer`,
    `enqueue_create_host_buffer`, `enqueue_copy`, `enqueue_memset`, and
    `synchronize`, all against one context per session
    (`gpu_runtime.GpuSession` owns exactly one). CUDA streams exist; no
    abstraction in this repository reaches them, and claiming overlap that
    the code does not implement would make `gpu_runtime.PhaseCounters`
    attribute time to a concurrency that never happened.
    """
    return False


def require_concurrent_queues(what: String) raises:
    """Refuse a caller that needs two device queues to overlap.

    The clear failure for a future overlap specialization, so it fails at
    the policy boundary with a message naming what it wanted rather than
    quietly running serialized and reporting a speedup that was never
    available.
    """
    raise Error(
        "'",
        what,
        "' needs concurrent device queues; this package holds one"
        " DeviceContext per session and issues every copy and launch on"
        " its single in-order queue (see gpu_runtime.mojo)",
    )


# --- Allocation and transfer -------------------------------------------


def unified_memory_inferable() -> Bool:
    """Whether unified memory follows from the API name on this backend.

    False. A discrete card, an integrated part, and CUDA managed memory are
    all "cuda" to a name query, so `gpu_portability.contract_for` sets
    unified for Metal alone and `contract_from_profile` takes it from a
    report instead. `DeviceReport.unified_memory` is that report and it
    defaults False, which is the conservative answer: it keeps every buffer
    on the staged copy route and tightens nothing.
    """
    return False


def allocation_plan(report: DeviceReport) raises -> SessionMemoryPlan:
    """The per-role transfer routes a session on this device gets.

    `unified_memory_policy.plan_session_routes`' answer, from the reported
    unified-memory flag. Called rather than reproduced: that module owns
    the route vocabulary, the evidence ladder, and the refusals, and a
    second route table here that disagreed with it by one role would be
    worse than none.

    With `unified_memory` False, which is what an unreported CUDA device
    carries, every role resolves to the staged copy the GPU histogram
    builder already requires (`ROUTE_COPY_STAGED`), so this changes nothing
    about what a CUDA session does today and only says why.
    """
    return plan_session_routes(report.unified_memory)


def partial_budget_for(report: DeviceReport) -> Int:
    """Bytes the partial-histogram buffer may claim on this device.

    `apple_gpu_policy.partial_budget_bytes`' rule, which is not Apple-
    specific despite where it lives: a fraction of the reported budget,
    the tighter fraction when memory is unified, capped by the portable
    ceiling, and the whole portable ceiling when no budget was reported.
    A CUDA device that reports a budget and is not unified takes the
    discrete fraction; one that reports nothing plans exactly as
    `gpu_tiling.PARTIAL_BUDGET_BYTES` says today.
    """
    return partial_budget_bytes(report.profile(API_CUDA))


# --- Packed bin loads --------------------------------------------------


def pack_alignment_bytes() -> Int:
    """The alignment the portable packed window requires: four one-byte
    cells per 32-bit word. Not a CUDA number; carried so the descriptor can
    state it beside the wider load CUDA offers and does not get."""
    return PACK_LANES


def native_wide_load_bytes() -> Int:
    """The widest single load CUDA offers, which the shared source does not
    emit."""
    return CUDA_NATIVE_WIDE_LOAD_BYTES


def coalescing_bytes() -> Int:
    """The transaction granularity a global load is served at."""
    return CUDA_SECTOR_BYTES


def plan_packed_window(
    storage: BinStorageDescriptor,
    first_row: Int,
    count: Int,
    rows_are_contiguous_run: Bool,
) raises -> PackedLoadWindow:
    """The packed-load window for a node on this backend.

    `gpu_histogram_specializations.plan_packed_window_for`'s answer,
    unmodified. The window arithmetic is a property of the bin matrix's
    layout and not of the device, so there is nothing for a backend policy
    to change about it, and re-deriving it here is exactly how a second
    copy would come to read the wrong bytes. What this module contributes
    is `packed_body_transactions` below, which measures the window the
    shared planner produced against this backend's transaction size.
    """
    return plan_packed_window_for(
        storage, first_row, count, rows_are_contiguous_run
    )


def packed_body_transactions(window: PackedLoadWindow) raises -> Int:
    """Memory transactions the packed body of one window costs on this
    backend, at `coalescing_bytes` per transaction.

    A diagnostic, not a decision: nothing selects the packed path on it.
    It is reported so a benchmark that does enable the packed path can say
    whether the body covered whole sectors or fragments of them, which is
    the quantity that would distinguish a packed win from a packed wash.
    An unusable window costs nothing, because its body is empty.
    """
    if not window.usable:
        return 0
    var body_bytes = window.body_quads * PACK_LANES
    return _ceil_div(body_bytes, CUDA_SECTOR_BYTES)


# --- The CUDA entry points ---------------------------------------------


def cuda_specialization(
    report: DeviceReport, compiled: KernelFeatures
) raises -> BackendSpecialization:
    """The whole CUDA descriptor for one reported device and one build."""
    require_subgroup_width_plausible(report)
    return BackendSpecialization(
        API_CUDA,
        backend_support(API_CUDA),
        subgroup_width(report),
        report.warp_size > 0,
        block_width_granularity(report),
        CUDA_STATIC_SHARED_CEILING_BYTES,
        report.max_shared_memory_per_block,
        CUDA_SECTOR_BYTES,
        CUDA_NATIVE_WIDE_LOAD_BYTES,
        PACK_LANES,
        concurrent_queues_available(),
        unified_memory_inferable(),
        False,
        compiled.any(),
    )


def cuda_histogram_capabilities(
    report: DeviceReport,
) raises -> DeviceHistogramCapabilities:
    """The capability record `apple_histogram_policy.derive_histogram_plan`
    consumes, for a CUDA device that reported a subgroup width.

    `gpu_portability.histogram_capabilities` builds it from a contract,
    where the width is always 0 because the contract carries no reading.
    This substitutes the reported width when the device answered
    `WARP_SIZE`, which is the one field a CUDA report can improve on. Every
    other field is the contract's: `wide_byte_loads` stays False on every
    backend because it is a measurement rather than a specification and no
    measurement exists.
    """
    require_subgroup_width_plausible(report)
    var caps = histogram_capabilities(cuda_contract(report))
    caps.subgroup_width = subgroup_width(report)
    return caps^


def derive_cuda_plan(
    report: DeviceReport,
    n_rows: Int,
    n_slots: Int,
    n_bins: Int,
    compiled: KernelFeatures,
    requested_strategy: Int = STRATEGY_AUTO,
    max_partial_cells: Int = 0,
) raises -> BackendLaunchPlan:
    """Resolve one node's histogram launch on a CUDA device.

    `derive_plan_for` with this backend's granularity rule and static
    shared-memory ceiling. Everything the plan reports about how it was
    bounded is on the returned value, including the portable plan it would
    otherwise have been, so a caller that does not trust it can launch
    `plan.baseline` instead without re-deriving anything.
    """
    require_subgroup_width_plausible(report)
    return derive_plan_for(
        API_CUDA,
        report,
        block_width_granularity(report),
        CUDA_STATIC_SHARED_CEILING_BYTES,
        n_rows,
        n_slots,
        n_bins,
        compiled,
        requested_strategy,
        max_partial_cells,
    )


def require_cuda_launchable(
    report: DeviceReport,
    plan: BackendLaunchPlan,
    grid_x: Int,
    n_bins: Int,
    compiled: KernelFeatures,
) raises:
    """The whole gate for one resolved CUDA launch.

    `gpu_portability.require_histogram_launchable` for the bin count, the
    primitives the resolved strategy needs, the build's specializations
    against what this backend has run, and the geometry against what the
    device reported; then this backend's own two additions, which that
    layer does not carry because they are not portable facts: the in-order
    queue the synchronization model depends on, and the 48 KiB static
    shared-memory ceiling the CUDA programming model imposes whatever the
    device advertises.

    Raises on the first failure, naming it. `grid_x` is passed rather than
    derived because the axis carries the feature count in
    `histogram_gpu.mojo`, the active feature count under feature
    subsampling, and a leaf-slot count in `gpu_leaf_batching.mojo`.
    """
    require_cuda(plan.api)
    var contract = cuda_contract(report)
    require_in_order_queue(contract)
    require_shared_memory_supported(report, n_bins, compiled)
    require_histogram_launchable(
        contract,
        report.caps(),
        plan.tiling,
        grid_x,
        n_bins,
        compiled,
        plan.selected,
    )


def describe_cuda(
    report: DeviceReport, compiled: KernelFeatures
) raises -> String:
    """One line pairing the backend contract with this module's descriptor,
    for a diagnostic record or a bug report."""
    return String(
        describe_contract(cuda_contract(report)),
        " | ",
        describe_specialization(cuda_specialization(report, compiled)),
        " | attributes_answered=",
        report.answered(),
        "/10",
    )
