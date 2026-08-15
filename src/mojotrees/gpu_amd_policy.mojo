"""AMD/HIP backend specialization policy.

The HIP half of the pair `gpu_cuda_policy.mojo` opens. Same shape, same
layering, same refusals: reported numbers and a dataset shape in, a launch
plan out; no `DeviceContext` opened, no attribute queried, nothing
allocated, nothing launched, no environment read. Every decision below is
exercisable on a machine with no accelerator, which is the only way any of
it could be exercised here at all.

**No hardware validation is claimed for any of this.** This repository has
never executed a GPU kernel on an AMD device.
`gpu_backend_policy.backend_support(API_HIP)` is `SUPPORT_PORTABLE`, every
HIP row in `docs/GPU_VALIDATION.md` reads "not run", and every function
here that would select a specialized kernel variant routes through
`gpu_portability.require_specializations_allowed`, which refuses on an
unexercised backend unless `MOJOTREES_GPU_BACKEND_UNVALIDATED=1`
acknowledges it.

What is genuinely different about HIP
-------------------------------------
Three things, and only these are encoded:

1. **The wavefront is not one number.** CDNA parts execute 64 lanes in
   lockstep and RDNA parts execute 32. That single fact is why
   `gpu_tiling.WARP_GRANULARITY` is 64 rather than 32 in the first place:
   its own comment says 64 is AMD's wavefront and a multiple of the 32-wide
   warp elsewhere, so one constant is legal on all three backends. It also
   means this backend cannot have an architectural width the way CUDA can,
   so `require_subgroup_width_plausible` here admits two values where the
   CUDA one admits a single value, and neither module substitutes either
   for a missing report.

   The consequence for the launch is the opposite of CUDA's. On CUDA a
   reported 32-lane warp lets a narrow node take a *finer* block than the
   portable 64-lane rounding allows. On HIP the portable granularity is
   already the wider of the two wavefronts, so a reported width can only
   ever leave the rounding alone (64) or refine it (32 on RDNA); it can
   never ask for a block the portable rule would have refused.

2. **Local data share is 64 KiB per workgroup.** Both CDNA and RDNA
   allocate a workgroup at most 64 KiB of LDS, and the shared source's
   histogram planes are `stack_allocation[..., AddressSpace.SHARED]`, which
   is a static declaration. That is a different ceiling from CUDA's 48 KiB
   static cap, which is the only reason the ceiling is a parameter of the
   shared `require_shared_within_ceiling` rather than a constant inside it.
   The shipping kernels request 3 KiB, so this is inert today and is
   checked so that stays a fact rather than an assumption.

3. **Coalescing is measured in 64-byte cache lines**, not CUDA's 32-byte
   sectors. Nothing decides on it; it is the unit `packed_body_transactions`
   reports a packed window's body in, so a benchmark that enables the packed
   path can say whether the body covered whole lines or fragments of them.

What this module deliberately does NOT encode
---------------------------------------------
- **No `gfx` architecture parse.** `DeviceContext.arch_name()` returns
  something like a `gfx` identifier on this backend and nothing here reads
  it. Mapping `gfx` numbers to wavefront widths would be a generation
  branch, which `apple_gpu_policy.mojo` explains at length that this
  project does not take, and it would be a worse answer than the one
  already available: the wavefront is directly queryable as
  `DeviceAttribute.WARP_SIZE`, so a device that answers tells us, and a
  device that refuses is honestly unknown.
- **No strategy preference.** `gpu_cuda_policy.preferred_strategy` returns
  `STRATEGY_AUTO` on every shape and every backend, and this module reuses
  it rather than shipping a second opinion. AMD device-memory atomics and
  the tiled reduction have never been compared on any AMD part from this
  repository.
- **No Float64.** `gpu_portability.require_device_float64` refuses it on
  every backend including this one. CDNA has strong Float64; Apple silicon
  has none, and one source takes the weakest backend's floor.
- **No second geometry engine.** The tile arithmetic is
  `gpu_tiling.resolve_tiling`, reached through
  `gpu_cuda_policy.derive_plan_for`. The baseline every plan is compared
  against is `gpu_tiling.derive_tiling`. The launch gate is
  `gpu_portability.require_histogram_launchable`. All three are called, not
  reproduced.

Why this module imports from `gpu_cuda_policy.mojo`
---------------------------------------------------
`DeviceReport`, `Occupancy`, `BackendLaunchPlan`, `BackendSpecialization`,
`StrategyInputs`, and the derivations over them are backend-neutral: they
describe *any* device that answers the attributes Metal refuses, and both
of this lane's backends are such devices. They are imported from
`gpu_cuda_policy.mojo` rather than copied because a second copy of the
occupancy rule or the plan assembly is exactly the duplicate policy engine
this work exists to avoid. Their home is `gpu_portability.mojo`, which is
another lane's file; `handoffs/remaining_11_gpu_backends.md` carries the
ready-to-apply patch that relocates them there and re-points both imports.
Until that lands, read the neutral names as belonging to the portability
layer that happens to be spelled in the CUDA file.

Where this module sits
----------------------
Above `gpu_tiling.mojo`, `gpu_histogram_specializations.mojo`,
`gpu_backend_policy.mojo`, `gpu_portability.mojo`,
`unified_memory_policy.mojo`, and `gpu_cuda_policy.mojo`; below anything
that launches. None of those may import this module. Nothing in the
shipping path imports it yet, which `docs/GPU_BACKEND_SPECIALIZATIONS.md`
records as the honest status and the handoff carries the call-site patches
for.

LightGBM difference: LightGBM reaches AMD hardware through its OpenCL
`gpu` device type, sharing kernels with every other OpenCL device.
mojotrees has one source for Metal, CUDA, and HIP alike, so an AMD
difference has to be expressible as a bound handed to the shared
arithmetic. Everything in this module is such a bound.
"""

from .apple_gpu_policy import (
    API_HIP,
    api_name,
    partial_budget_bytes,
)
from .gpu_backend_policy import backend_support
from .gpu_cuda_policy import (
    BackendLaunchPlan,
    BackendSpecialization,
    DeviceReport,
    StrategyInputs,
    _bool_text,
    _ceil_div,
    derive_plan_for,
    describe_specialization,
    preferred_strategy,
    require_shared_reported_fits,
    require_shared_within_ceiling,
)
from .gpu_histogram_specializations import (
    PACK_LANES,
    BinStorageDescriptor,
    DeviceHistogramCapabilities,
    KernelFeatures,
    PackedLoadWindow,
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
)
from .gpu_tiling import STRATEGY_AUTO, WARP_GRANULARITY
from .unified_memory_policy import SessionMemoryPlan, plan_session_routes


# The two wavefront widths AMD GPUs execute in. Used **only** to refuse a
# reported value that can be neither, which would mean the attribute was
# misread or the backend code is wrong. Neither is ever substituted for an
# unreported width: an unreported width is unknown, and `subgroup_width`
# returns 0 for it.
comptime AMD_WAVEFRONT_CDNA = 64
comptime AMD_WAVEFRONT_RDNA = 32

# Local data share a workgroup may statically declare, on both CDNA and
# RDNA. `gpu_tiling.WARP_GRANULARITY` is 64 for the CDNA wavefront; this is
# the other AMD number the shared source is subject to, and it is the one
# that differs from CUDA's 48 KiB static cap.
comptime AMD_LDS_PER_WORKGROUP_BYTES = 64 * 1024

# The cache line a global load is served in. The unit a packed-bin body is
# measured in, not a threshold anything is compared against here.
comptime AMD_CACHE_LINE_BYTES = 64

# The widest single load HIP offers (`global_load_dwordx4`, 128 bits). The
# shared source emits none: `gpu_histogram_specializations.pack4_bins` packs
# four one-byte cells into a 32-bit word, and that portable implementation
# is the one that runs. Recorded so the gap a packed-load specialization
# would close is stated rather than implied.
comptime AMD_NATIVE_WIDE_LOAD_BYTES = 16


def require_amd(api: Int) raises:
    """Refuse an API code this module has no business planning for.

    A capability error rather than a silent fallback, and it refuses
    `API_UNKNOWN` too: a device that did not name its API might be AMD and
    might not, and planning HIP bounds for it would apply a 64 KiB LDS
    ceiling and a wavefront-width admissibility rule to a device nobody
    identified. The portable path already covers an unidentified device; an
    operator who knows it is AMD declares it with
    `MOJOTREES_GPU_BACKEND=hip`, which `device_policy.env_declared_api`
    reads (it accepts the `rocm` spellings too, through
    `apple_gpu_policy.parse_api`).
    """
    if api == API_HIP:
        return
    raise Error(
        "gpu_amd_policy plans for the hip backend only; got api code ",
        api,
        " (",
        api_name(api),
        "). An unidentified device takes the portable path in"
        " gpu_tiling.mojo; declare MOJOTREES_GPU_BACKEND=hip if this device"
        " is known to be AMD",
    )


def amd_contract(report: DeviceReport) raises -> BackendContract:
    """The portability contract for the device this report describes.

    `gpu_portability.contract_from_profile`'s answer, not a second one: the
    grid bound, the primitive set, and the Float64 refusal are properties of
    the shared source and belong to that layer. What this adds is only that
    the profile is built from a HIP report, so `unified_memory` on the
    contract is what the device said rather than what the API name implies,
    which matters more here than anywhere: an XNACK-enabled part and an
    accelerated processing unit are both unified and both spell their API
    "hip", and a discrete card spells it the same way.
    """
    return contract_from_profile(report.profile(API_HIP))


# --- Subgroup width ----------------------------------------------------


def subgroup_width(report: DeviceReport) -> Int:
    """Lanes in lockstep on this device, or 0 for unknown.

    The reported `WARP_SIZE` and nothing else. Neither wavefront constant is
    substituted for a missing report, and on this backend that restraint is
    not a stylistic preference: the two AMD families disagree by a factor of
    two, so a guess is as likely to be wrong as right.
    """
    if report.warp_size > 0:
        return report.warp_size
    return 0


def require_subgroup_width_plausible(report: DeviceReport) raises:
    """Refuse a reported width that can be neither AMD wavefront.

    Not a device check: AMD parts execute 64 lanes (CDNA and the GCN parts
    before it) or 32 (RDNA), so a positive report of anything else means the
    attribute was misread, the report was filled in from the wrong device,
    or the API code is wrong. Unreported passes, because unknown is a
    legitimate answer and the portable rounding granularity covers it.
    """
    if report.warp_size <= 0:
        return
    if report.warp_size == AMD_WAVEFRONT_CDNA:
        return
    if report.warp_size == AMD_WAVEFRONT_RDNA:
        return
    raise Error(
        "this device reported a subgroup width of ",
        report.warp_size,
        " for the hip backend, and AMD wavefronts are ",
        AMD_WAVEFRONT_CDNA,
        " lanes on CDNA or ",
        AMD_WAVEFRONT_RDNA,
        " on RDNA; the attribute or the backend code is wrong",
    )


def require_subgroup_width_known(report: DeviceReport, what: String) raises:
    """Refuse a specialization that needs a wavefront width on a device that
    did not report one.

    Nothing in this package needs one today, which is why this has no
    caller: no kernel here performs a cross-lane shuffle, a ballot, or a
    wavefront-level reduction, and `gpu_tiling.WARP_GRANULARITY` is a
    rounding granularity rather than a width claim. It exists so the first
    cross-lane specialization has a gate to pass rather than a constant to
    reach for, and on this backend reaching for a constant would mean
    picking between 32 and 64 without evidence.
    """
    if subgroup_width(report) > 0:
        return
    raise Error(
        "'",
        what,
        "' needs the wavefront width and this device did not report"
        " WARP_SIZE; AMD wavefronts are ",
        AMD_WAVEFRONT_CDNA,
        " lanes on CDNA and ",
        AMD_WAVEFRONT_RDNA,
        " on RDNA and nothing here picks between them without a report",
    )


def block_width_granularity(report: DeviceReport) -> Int:
    """The multiple a workgroup width is rounded to on this device: the
    reported wavefront, and `gpu_tiling.WARP_GRANULARITY` when the device
    did not report one.

    The portable granularity is 64, which is the CDNA wavefront, so an
    unreported AMD device rounds to a legal multiple on either family (64 is
    a multiple of 32). A reported 32 refines that for an RDNA part, letting a
    narrow node take a 32-lane workgroup where the portable rule floors at
    64. That refinement rests entirely on the device having answered the
    query.
    """
    var width = subgroup_width(report)
    if width > 0:
        return width
    return WARP_GRANULARITY


def wavefront_matches_granularity(report: DeviceReport) -> Bool:
    """Whether the portable rounding granularity is this device's wavefront.

    True on a CDNA part and on any device that did not report a width, false
    on RDNA. Reported rather than acted on: a workgroup rounded to 64 on a
    32-lane part is still a legal launch and still fully occupied, it is
    simply rounded coarser than it needed to be. This is the one line a
    capture on new AMD hardware should be read against, because it is where
    the portable constant and the device stop agreeing.
    """
    return block_width_granularity(report) == WARP_GRANULARITY


# --- Shared memory, atomics, queue -------------------------------------


def static_shared_ceiling() -> Int:
    """Local data share a workgroup may statically declare on this
    backend."""
    return AMD_LDS_PER_WORKGROUP_BYTES


def require_shared_memory_supported(
    report: DeviceReport, n_bins: Int, compiled: KernelFeatures
) raises:
    """Refuse a workgroup LDS allocation this backend or this device will
    not accept, for the kernel this build compiled.

    Both halves in one call: the 64 KiB per-workgroup LDS ceiling whatever
    the device advertises, and the per-block limit the device reported.
    What is checked is `gpu_portability.kernel_shared_request`, the
    footprint the compiled kernel really has, not the `n_bins * 12` model
    whose own docstring records that it is optimistic below 256 bins.
    """
    var shared_bytes = kernel_shared_request(n_bins, compiled)
    require_shared_within_ceiling(
        API_HIP, shared_bytes, AMD_LDS_PER_WORKGROUP_BYTES
    )
    require_shared_reported_fits(API_HIP, report, shared_bytes)


def require_in_order_queue(contract: BackendContract) raises:
    """Refuse a backend that does not promise an in-order device queue.

    The property `gpu_runtime.HazardTracker` rests on: device work never
    needs a host synchronization to observe earlier device work, so the only
    required synchronizations are the two host-side hazards it tracks. HIP
    promises it for a single stream, which is the only queue model this
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

    False, for the same reason it is false on CUDA: `DeviceContext` is used
    here through `enqueue_create_buffer`, `enqueue_create_host_buffer`,
    `enqueue_copy`, `enqueue_memset`, and `synchronize`, all against one
    context per session (`gpu_runtime.GpuSession` owns exactly one). HIP
    streams exist; no abstraction in this repository reaches them.
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
        " DeviceContext per session and issues every copy and launch on its"
        " single in-order queue (see gpu_runtime.mojo)",
    )


# --- Allocation and transfer -------------------------------------------


def unified_memory_inferable() -> Bool:
    """Whether unified memory follows from the API name on this backend.

    False, and less inferable here than anywhere: an XNACK-enabled discrete
    card, an accelerated processing unit with genuinely shared memory, and
    an ordinary discrete card all spell their API "hip".
    `gpu_portability.contract_for` therefore sets unified for Metal alone
    and `contract_from_profile` takes it from a report instead.
    `DeviceReport.unified_memory` is that report and it defaults False,
    which is the conservative answer: it keeps every buffer on the staged
    copy route and tightens nothing.
    """
    return False


def allocation_plan(report: DeviceReport) raises -> SessionMemoryPlan:
    """The per-role transfer routes a session on this device gets.

    `unified_memory_policy.plan_session_routes`' answer, from the reported
    unified-memory flag. Called rather than reproduced: that module owns the
    route vocabulary, the evidence ladder, and the refusals, and a second
    route table here that disagreed with it by one role would be worse than
    none.

    With `unified_memory` False, which is what an unreported HIP device
    carries, every role resolves to the staged copy the GPU histogram
    builder already requires (`ROUTE_COPY_STAGED`), so this changes nothing
    about what a HIP session does today and only says why.
    """
    return plan_session_routes(report.unified_memory)


def partial_budget_for(report: DeviceReport) -> Int:
    """Bytes the partial-histogram buffer may claim on this device.

    `apple_gpu_policy.partial_budget_bytes`' rule, which is not
    Apple-specific despite where it lives: a fraction of the reported
    budget, the tighter fraction when memory is unified, capped by the
    portable ceiling, and the whole portable ceiling when no budget was
    reported. The unified fraction is the one that matters on this backend,
    because a HIP device can honestly report unified memory where a Metal
    device always does and a CUDA device usually does not.
    """
    return partial_budget_bytes(report.profile(API_HIP))


# --- Packed bin loads --------------------------------------------------


def pack_alignment_bytes() -> Int:
    """The alignment the portable packed window requires: four one-byte
    cells per 32-bit word. Not an AMD number; carried so the descriptor can
    state it beside the wider load HIP offers and does not get."""
    return PACK_LANES


def native_wide_load_bytes() -> Int:
    """The widest single load HIP offers, which the shared source does not
    emit."""
    return AMD_NATIVE_WIDE_LOAD_BYTES


def coalescing_bytes() -> Int:
    """The cache line a global load is served in."""
    return AMD_CACHE_LINE_BYTES


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
    to change about it, and re-deriving it here is exactly how a second copy
    would come to read the wrong bytes. What this module contributes is
    `packed_body_transactions` below, which measures the window the shared
    planner produced against this backend's cache line.
    """
    return plan_packed_window_for(
        storage, first_row, count, rows_are_contiguous_run
    )


def packed_body_transactions(window: PackedLoadWindow) raises -> Int:
    """Memory transactions the packed body of one window costs on this
    backend, at `coalescing_bytes` per transaction.

    A diagnostic, not a decision: nothing selects the packed path on it. It
    is reported so a benchmark that does enable the packed path can say
    whether the body covered whole cache lines or fragments of them, which
    is the quantity that would distinguish a packed win from a packed wash.
    The count is half CUDA's for the same window, because the line is twice
    the sector; that difference is the reason each backend reports its own
    rather than sharing one number.

    An unusable window costs nothing, because its body is empty.
    """
    if not window.usable:
        return 0
    var body_bytes = window.body_quads * PACK_LANES
    return _ceil_div(body_bytes, AMD_CACHE_LINE_BYTES)


# --- The HIP entry points ----------------------------------------------


def amd_specialization(
    report: DeviceReport, compiled: KernelFeatures
) raises -> BackendSpecialization:
    """The whole HIP descriptor for one reported device and one build."""
    require_subgroup_width_plausible(report)
    return BackendSpecialization(
        API_HIP,
        backend_support(API_HIP),
        subgroup_width(report),
        report.warp_size > 0,
        block_width_granularity(report),
        AMD_LDS_PER_WORKGROUP_BYTES,
        report.max_shared_memory_per_block,
        AMD_CACHE_LINE_BYTES,
        AMD_NATIVE_WIDE_LOAD_BYTES,
        PACK_LANES,
        concurrent_queues_available(),
        unified_memory_inferable(),
        False,
        compiled.any(),
    )


def amd_histogram_capabilities(
    report: DeviceReport,
) raises -> DeviceHistogramCapabilities:
    """The capability record `apple_histogram_policy.derive_histogram_plan`
    consumes, for a HIP device that reported a wavefront width.

    `gpu_portability.histogram_capabilities` builds it from a contract,
    where the width is always 0 because the contract carries no reading.
    This substitutes the reported width when the device answered
    `WARP_SIZE`, which is the one field a HIP report can improve on. Every
    other field is the contract's: `wide_byte_loads` stays False on every
    backend because it is a measurement rather than a specification and no
    measurement exists.
    """
    require_subgroup_width_plausible(report)
    var caps = histogram_capabilities(amd_contract(report))
    caps.subgroup_width = subgroup_width(report)
    return caps^


def derive_amd_plan(
    report: DeviceReport,
    n_rows: Int,
    n_slots: Int,
    n_bins: Int,
    compiled: KernelFeatures,
    requested_strategy: Int = STRATEGY_AUTO,
    max_partial_cells: Int = 0,
) raises -> BackendLaunchPlan:
    """Resolve one node's histogram launch on a HIP device.

    `gpu_cuda_policy.derive_plan_for` with this backend's granularity rule
    and LDS ceiling. Everything the plan reports about how it was bounded is
    on the returned value, including the portable plan it would otherwise
    have been, so a caller that does not trust it can launch `plan.baseline`
    instead without re-deriving anything.
    """
    require_subgroup_width_plausible(report)
    return derive_plan_for(
        API_HIP,
        report,
        block_width_granularity(report),
        AMD_LDS_PER_WORKGROUP_BYTES,
        n_rows,
        n_slots,
        n_bins,
        compiled,
        requested_strategy,
        max_partial_cells,
    )


def amd_strategy_inputs(plan: BackendLaunchPlan) raises -> StrategyInputs:
    """What a measured HIP strategy rule would key on, off a resolved plan.

    The plan already carries these; this is the accessor that makes that
    explicit at a call site that wants only the strategy question, and it is
    a copy rather than a second derivation. `preferred_strategy` over them
    returns `STRATEGY_AUTO` on this backend as on every other, because
    nothing here has compared AMD device-memory atomics against the tiled
    reduction on any AMD part.
    """
    return plan.strategy.copy()


def amd_preferred_strategy(plan: BackendLaunchPlan) raises -> Int:
    """Which accumulation strategy this backend would rather run for a
    resolved plan.

    `gpu_cuda_policy.preferred_strategy`'s answer, which is
    `STRATEGY_AUTO` on every shape and every backend: the portable rule in
    `gpu_tiling.resolve_tiling` decides, and it already did, so a caller
    that consults this and gets `STRATEGY_AUTO` should launch
    `plan.tiling.strategy` unchanged. It is reached through the shared
    function rather than answered here so that a measured AMD rule and a
    measured NVIDIA rule cannot come to disagree about what "no preference"
    means.
    """
    return preferred_strategy(amd_strategy_inputs(plan))


def require_amd_launchable(
    report: DeviceReport,
    plan: BackendLaunchPlan,
    grid_x: Int,
    n_bins: Int,
    compiled: KernelFeatures,
) raises:
    """The whole gate for one resolved HIP launch.

    `gpu_portability.require_histogram_launchable` for the bin count, the
    primitives the resolved strategy needs, the build's specializations
    against what this backend has run, and the geometry against what the
    device reported; then this backend's own two additions, which that layer
    does not carry because they are not portable facts: the in-order queue
    the synchronization model depends on, and the 64 KiB per-workgroup LDS
    ceiling whatever the device advertises.

    Raises on the first failure, naming it. `grid_x` is passed rather than
    derived because the axis carries the feature count in
    `histogram_gpu.mojo`, the active feature count under feature
    subsampling, and a leaf-slot count in `gpu_leaf_batching.mojo`.
    """
    require_amd(plan.api)
    var contract = amd_contract(report)
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


def describe_amd(
    report: DeviceReport, compiled: KernelFeatures
) raises -> String:
    """One line pairing the backend contract with this module's descriptor,
    for a diagnostic record or a bug report."""
    return String(
        describe_contract(amd_contract(report)),
        " | ",
        describe_specialization(amd_specialization(report, compiled)),
        " | wavefront_matches_granularity=",
        _bool_text(wavefront_matches_granularity(report)),
        " | attributes_answered=",
        report.answered(),
        "/10",
    )
