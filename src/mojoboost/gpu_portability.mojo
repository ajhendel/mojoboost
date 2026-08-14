"""The contract the one GPU source is checked against, per backend.

`docs/GPU_VALIDATION.md` opens with the design commitment this module
exists to defend: there is no CUDA file, no HIP file, and no Metal file.
`histogram_gpu.mojo`, `gpu_active_rows.mojo`, `train_gpu.mojo`, and
`gpu_predict.mojo` are compiled for whichever device MAX opens, and they
reach for a small, fixed set of device primitives to do it. That set is the
portability contract, and until this module it was written down nowhere:
each kernel simply used what it needed, and the only thing any backend was
checked for was three integers from `gpu_tiling.query_device_caps`.

What the shipping kernels actually require
------------------------------------------
Read off the imports of every module under `src/mojoboost/gpu_*.mojo` and
`histogram_gpu.mojo`, not assumed:

- `barrier()` from `max.gpu.sync`, a whole-threadgroup barrier reached under
  uniform control flow.
- `stack_allocation[..., AddressSpace.SHARED]`, a *statically* sized
  threadgroup allocation. The shipping histogram kernels allocate three
  `MAX_BINS`-wide Int32 planes, which is 3 KiB per threadgroup whatever
  `n_bins` is.
- `Atomic.fetch_add` on Int32, in threadgroup memory (the per-block partial
  histogram) and in device memory (the atomic strategy's flush). Int32 and
  only Int32: there is no float atomic anywhere in this package, which is
  the property that makes a histogram bit-identical run to run.
- A two-dimensional grid. `grid.x` carries features or leaf slots and
  `grid.y` carries row tiles or classes. Nothing in this package launches a
  `grid.z`, and no portable bound for one has ever been established here.
- An in-order device queue, which is what `gpu_runtime.mojo`'s hazard model
  rests on: device work never needs a host synchronization to observe
  earlier device work.

Every one of those is specified by Metal, CUDA, and HIP alike. That is why
one source is a reasonable commitment. It is not evidence that the source
*runs* on all three: Metal is the only backend this repository has executed,
and `gpu_backend_policy.backend_support` is where that difference is
recorded.

What this module refuses
------------------------
Two kinds of thing, both clearly rather than silently:

1. **A launch the reported device cannot run.** The threadgroup memory the
   kernels really request against what the device advertises, the block
   width against `MAX_THREADS_PER_BLOCK`, `grid.y` against the smallest
   portable bound, and the bin count against `MAX_BINS`. The existing guard
   in `gpu_tiling.derive_tiling` compares `n_bins * 12` against the reported
   shared memory, which its own docstring records as optimistic: it is the
   footprint of a kernel sized to its bin count, and the kernels that ship
   are sized to `MAX_BINS`. `kernel_shared_request` here is the footprint
   the compiled kernel actually has, so a device that would pass that guard
   and then fail to launch is refused first.
2. **A specialization on a backend nobody has run.** Handled by
   `gpu_backend_policy.require_backend_exercised`, with the loud override
   documented there.

What this module deliberately does not do
-----------------------------------------
- **No subgroup width.** `BackendContract.subgroup_width` is 0 on every
  backend in the table, meaning unknown, which is the only honest value:
  Metal rejects the `WARP_SIZE` query, and this repository has never read
  the attribute on CUDA or HIP. Nothing divides by it. `launch_granularity`
  beside it is `gpu_tiling.WARP_GRANULARITY`, a rounding granularity chosen
  to be a multiple of every width the supported backends use, which is a
  different kind of statement and is not a claim about any device.
- **No hardware detection.** The API code is an argument, from
  `device_policy.env_declared_api()` or from a reported profile. Nothing
  here reads an operating system name or a chip string.
- **No Float64 on device, anywhere.** Apple GPUs have no Float64, so the
  shared source carries gradients as Float32 and accumulates in fixed-point
  Int32. That makes CUDA and HIP inherit Apple's floor, which is a real cost
  and is recorded as `device_float64_permitted = False` on every backend
  rather than left implicit. Relaxing it for a backend that has Float64 is a
  specialization, and it needs the same evidence as any other.
- **No policy.** Which geometry to launch is `gpu_tiling.derive_tiling` and
  `apple_histogram_policy.derive_histogram_plan`; which device to train on
  is `device_policy.decide_device`. This module answers only whether what
  they chose can be launched on the backend in front of them.

Where this module sits
----------------------
Above `gpu_tiling.mojo` and `gpu_histogram_specializations.mojo`, whose
constants and footprint arithmetic it consumes rather than restates, and
below every module that launches a kernel. That direction is fixed: nothing
in `gpu_tiling.mojo` or `gpu_histogram_specializations.mojo` may import this
module, or the layering closes into a cycle.
"""

from .apple_gpu_policy import (
    API_CUDA,
    API_HIP,
    API_METAL,
    API_UNKNOWN,
    GpuProfile,
    api_name,
)
from .gpu_backend_policy import (
    SUPPORT_EXERCISED,
    backend_support,
    require_backend_covered,
    require_backend_exercised,
    support_name,
)
from .gpu_histogram_specializations import (
    BYTES_PER_PLANE_CELL,
    MAX_BINS,
    PLANES_PER_HISTOGRAM,
    DeviceHistogramCapabilities,
    KernelFeatures,
    bin_capacity_for,
    kernel_shared_bytes,
    unspecialized_kernel_shared_bytes,
)
from .gpu_tiling import (
    MAX_GRID_DIM_Y,
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    WARP_GRANULARITY,
    DeviceCaps,
    HistogramTiling,
    strategy_name,
)


# --- The primitive set --------------------------------------------------

comptime REQ_THREADGROUP_BARRIER = 0
comptime REQ_SHARED_STATIC_ALLOC = 1
comptime REQ_SHARED_INT32_ATOMIC_ADD = 2
comptime REQ_GLOBAL_INT32_ATOMIC_ADD = 3
comptime REQ_GRID_2D = 4
comptime REQ_IN_ORDER_QUEUE = 5
comptime N_REQUIREMENTS = 6


def _bool_text(value: Bool) -> String:
    """How a Bool is spelled in a described contract. Local rather than
    imported: `device_policy.mojo` and `apple_histogram_policy.mojo` each
    have their own three-line spelling of this, and importing either would
    point this module's dependency the wrong way."""
    if value:
        return String("true")
    return String("false")


def requirement_name(requirement: Int) -> String:
    if requirement == REQ_THREADGROUP_BARRIER:
        return String("threadgroup-barrier")
    if requirement == REQ_SHARED_STATIC_ALLOC:
        return String("static-threadgroup-allocation")
    if requirement == REQ_SHARED_INT32_ATOMIC_ADD:
        return String("threadgroup-int32-atomic-add")
    if requirement == REQ_GLOBAL_INT32_ATOMIC_ADD:
        return String("device-int32-atomic-add")
    if requirement == REQ_GRID_2D:
        return String("two-dimensional-grid")
    if requirement == REQ_IN_ORDER_QUEUE:
        return String("in-order-queue")
    return String("unknown")


# Grid axes this package launches on. `grid.x` carries features or leaf
# slots, `grid.y` carries row tiles or classes, and nothing carries a
# `grid.z`: `gpu_histogram_specializations.mojo` records why an earlier
# batched draft that wanted one was removed rather than given a guessed
# bound.
comptime GRID_AXES = 2

# Threadgroup memory a device must advertise before the shipping histogram
# kernels can launch at all: their three `MAX_BINS`-wide Int32 planes, in
# the terms `gpu_histogram_specializations.kernel_shared_bytes` states them,
# so this constant cannot drift away from that function's answer at
# `MAX_BINS`. Every supported backend advertises far more, which is why the
# optimistic guard in `gpu_tiling.derive_tiling` has never been observed to
# fail; checking it here is what keeps that a fact rather than an
# assumption.
comptime MIN_SHARED_MEMORY_PER_BLOCK = (
    PLANES_PER_HISTOGRAM * BYTES_PER_PLANE_CELL * MAX_BINS
)


# --- The contract -------------------------------------------------------


@fieldwise_init
struct BackendContract(Copyable, Movable):
    """What one backend promises, and how far this repository has taken it.

    Every Bool below is a property the corresponding API specifies, not a
    property anyone here measured. `support` is the field that carries what
    was measured, and it is `SUPPORT_EXERCISED` for Metal only.
    """

    var api: Int
    var support: Int

    var subgroup_width: Int
    """Lanes that execute in lockstep, or 0 for unknown, which is what every
    entry in the table carries. Metal refuses the query; nobody has run the
    query on CUDA or HIP from this repository. A caller that did read the
    attribute may substitute its own value, and nothing in this package will
    divide by it either way."""

    var launch_granularity: Int
    """`gpu_tiling.WARP_GRANULARITY`. The multiple a threadgroup width is
    rounded to, chosen because 64 is AMD's wavefront and a multiple of the
    32-wide warp elsewhere. A rounding rule, not a width claim."""

    var max_grid_dim_y: Int
    """The smallest portable `grid.y` bound, which is CUDA's 65535. Metal and
    HIP allow more; the tightest bound applies everywhere because one source
    launches on all three."""

    var grid_axes: Int

    var threadgroup_barrier: Bool
    var shared_static_alloc: Bool
    var shared_int32_atomic_add: Bool
    var global_int32_atomic_add: Bool
    var in_order_queue: Bool

    var device_float64_permitted: Bool
    """Whether the shared source may emit Float64 device arithmetic. False
    on every backend, including those that have Float64: Apple silicon does
    not, and one source means the weakest backend sets the floor. See the
    module docstring."""

    var unified_memory: Bool
    """Whether the device budget is shared with the host. `contract_for`
    sets it True for Metal alone, that being the one API where it follows
    from the API and nothing else. It does not follow from CUDA or HIP: an
    integrated part, CUDA managed memory, and an XNACK-enabled HIP device
    are all unified in ways an API name cannot distinguish from a discrete
    card, so on those backends the value has to come from a report.
    `contract_from_profile` takes it from one."""

    def provides(self, requirement: Int) raises -> Bool:
        """Whether this backend promises one of the primitives the shipping
        kernels use."""
        if requirement == REQ_THREADGROUP_BARRIER:
            return self.threadgroup_barrier
        if requirement == REQ_SHARED_STATIC_ALLOC:
            return self.shared_static_alloc
        if requirement == REQ_SHARED_INT32_ATOMIC_ADD:
            return self.shared_int32_atomic_add
        if requirement == REQ_GLOBAL_INT32_ATOMIC_ADD:
            return self.global_int32_atomic_add
        if requirement == REQ_GRID_2D:
            return self.grid_axes >= 2
        if requirement == REQ_IN_ORDER_QUEUE:
            return self.in_order_queue
        raise Error(
            "no such device requirement: ",
            requirement,
            "; expected 0 through ",
            N_REQUIREMENTS - 1,
        )

    def is_exercised(self) -> Bool:
        """Whether this repository has run a kernel on this backend."""
        return self.support == SUPPORT_EXERCISED


def contract_for(api: Int) raises -> BackendContract:
    """The contract for one backend.

    Raises for an API code outside the covered set, which is the clear
    failure an unsupported build or a mis-plumbed backend code deserves.
    `API_UNKNOWN` is covered and takes the portable floor: it is the value
    every launch on today's path carries, because nothing in the shipping
    code reads an API name before it launches.

    The five primitive Bools are True on every covered backend, including
    the unknown one. That is deliberate and it is what keeps this gate a
    conservative addition rather than a behavior change: the kernels already
    use all five unconditionally on every device MAX opens, and Metal
    demonstrates that at least one real device honors them. Setting them
    False for the unknown backend would refuse the path that ships today.
    """
    require_backend_covered(api)
    return BackendContract(
        api,
        backend_support(api),
        0,
        WARP_GRANULARITY,
        MAX_GRID_DIM_Y,
        GRID_AXES,
        True,
        True,
        True,
        True,
        True,
        False,
        api == API_METAL,
    )


def contract_from_profile(profile: GpuProfile) raises -> BackendContract:
    """The contract for the device a profile describes.

    The profile's `unified_memory` wins over the API-derived default,
    because a profile built by `GpuProfile.from_reported` carries what a
    device said about itself and this table carries only what an API name
    implies. Everything else comes from the API: a device cannot report its
    way out of `grid.y` being bounded by the tightest backend when one
    source targets all of them.
    """
    var contract = contract_for(profile.api)
    contract.unified_memory = profile.unified_memory
    return contract^


def describe_contract(contract: BackendContract) -> String:
    """One line for a diagnostic record or a bug report."""
    return String(
        "backend=",
        api_name(contract.api),
        " support=",
        support_name(contract.support),
        " subgroup_width=",
        contract.subgroup_width,
        " granularity=",
        contract.launch_granularity,
        " max_grid_y=",
        contract.max_grid_dim_y,
        " grid_axes=",
        contract.grid_axes,
        " float64=",
        _bool_text(contract.device_float64_permitted),
        " unified=",
        _bool_text(contract.unified_memory),
    )


# --- Specialization capabilities ---------------------------------------


def histogram_capabilities(
    contract: BackendContract,
) -> DeviceHistogramCapabilities:
    """The capability record `apple_histogram_policy.derive_histogram_plan`
    consumes, built from a backend contract.

    The one way a non-Metal device reaches the existing specialization
    planner. Until now `DeviceHistogramCapabilities.portable()` was the only
    constructor anything called, so every device planned as if it had
    reported nothing, and the specialization layer was reachable in practice
    only through the Apple-named module that constructs the profile.

    `wide_byte_loads` stays False on every backend. It is the one field that
    is a measurement rather than a specification, no measurement exists on
    any backend, and the portable `pack4_bins` implementation beside it
    produces the identical integers.
    """
    return DeviceHistogramCapabilities(
        contract.subgroup_width,
        False,
        contract.unified_memory,
    )


def kernel_shared_request(
    n_bins: Int, compiled: KernelFeatures
) raises -> Int:
    """Threadgroup memory one block of the kernel this build compiled really
    requests, at `n_bins` bins.

    Not `gpu_tiling.shared_bytes_for`, which models `n_bins * 12`: that is
    the footprint of a kernel sized to its bin count, and its own docstring
    records that the kernels which ship allocate three `MAX_BINS`-wide Int32
    planes whatever `n_bins` is. The two agree only at 256 bins, and below
    it the model is optimistic by up to 2688 bytes, so a device advertising
    less than 3 KiB would pass the modeled guard and then fail to launch.

    Which of the two is true is a property of the build, which is what
    `KernelFeatures.specialized_bin_kernels` records, so this takes the
    build's compiled variants rather than assuming either. It is the
    compiled set and never the selected one: an uninstantiated kernel cannot
    be the one that allocates.
    """
    if compiled.specialized_bin_kernels:
        return kernel_shared_bytes(bin_capacity_for(n_bins))
    return unspecialized_kernel_shared_bytes()


def require_specializations_allowed(
    contract: BackendContract, selected: KernelFeatures
) raises:
    """Refuse a specialized kernel variant a plan *selected* on a named
    backend this repository has never executed.

    `selected`, not compiled-in. The distinction matters and getting it
    wrong would break the shipping path: `histogram_gpu.build_kernel_features`
    reports `batched_leaf_kernel = True` because linking that module
    instantiates the batched kernels, which is a fact about compilation.
    Selecting them is a separate decision that `apple_histogram_policy` makes
    only when `SPEC_LEVEL_BATCHED` was asked for by name, never from `auto`.
    A caller passes what its resolved plan chose, so the gate fires on the
    choice rather than on the link.

    Metal passes. CUDA and HIP need `MOJOBOOST_GPU_BACKEND_UNVALIDATED=1`.
    An unidentified backend is not refused, for the reason
    `gpu_backend_policy.require_backend_exercised` documents at length.
    """
    if not selected.any():
        return
    if selected.specialized_bin_kernels:
        require_backend_exercised(
            contract.api, String("bin-capacity specialized histogram kernels")
        )
    if selected.packed_bin_loads:
        require_backend_exercised(
            contract.api, String("packed four-byte bin loads")
        )
    if selected.batched_leaf_kernel:
        require_backend_exercised(
            contract.api, String("batched multi-leaf histogram kernels")
        )


def require_device_float64(contract: BackendContract) raises:
    """Refuse Float64 device arithmetic.

    Every backend, unconditionally, and it is a function rather than a
    comment so a future Float64 specialization has one place to be gated
    from. Apple silicon has no Float64 and one source targets all three
    backends, so the shared kernels carry Float32 gradients and fixed-point
    Int32 accumulation everywhere. A backend that has Float64 may only use
    it behind a variant that reproduces the fixed-point integers exactly,
    and that variant does not exist.
    """
    if contract.device_float64_permitted:
        return
    raise Error(
        "the shared GPU source does not emit Float64 device arithmetic on"
        " any backend, including ",
        api_name(contract.api),
        ": Apple silicon has no Float64 and one source takes the weakest"
        " backend's floor; gradients are Float32 and accumulation is"
        " fixed-point Int32",
    )


# --- Launch gates -------------------------------------------------------


def required_primitives(strategy: Int) raises -> List[Int]:
    """The primitives one resolved histogram strategy needs.

    Both strategies build a threadgroup partial histogram with Int32 atomics
    under a barrier. They differ in the flush: the atomic strategy folds
    every partial into the output with device-memory atomics, and the tiled
    strategy writes each partial to its own slot, which nothing else writes,
    and reduces them in a second kernel. That is the whole reason the tiled
    path is the one to reach for when device atomics are contended, and it
    is why the two lists differ by exactly one entry.

    `STRATEGY_AUTO` is a request rather than a resolution and has no list,
    matching `gpu_tiling.launches_for_strategy`.
    """
    var needed: List[Int] = [
        REQ_THREADGROUP_BARRIER,
        REQ_SHARED_STATIC_ALLOC,
        REQ_SHARED_INT32_ATOMIC_ADD,
        REQ_GRID_2D,
        REQ_IN_ORDER_QUEUE,
    ]
    if strategy == STRATEGY_ATOMIC:
        needed.append(REQ_GLOBAL_INT32_ATOMIC_ADD)
        return needed^
    if strategy == STRATEGY_TILED:
        return needed^
    raise Error(
        "an unresolved strategy has no primitive requirements; resolve it"
        " with derive_tiling first"
    )


def require_primitives(contract: BackendContract, strategy: Int) raises:
    """Refuse a strategy whose primitives the backend does not promise."""
    var needed = required_primitives(strategy)
    for i in range(len(needed)):
        if not contract.provides(needed[i]):
            raise Error(
                "the ",
                api_name(contract.api),
                " backend does not provide '",
                requirement_name(needed[i]),
                "', which the ",
                strategy_name(strategy),
                " histogram strategy requires",
            )


def require_bins_supported(n_bins: Int) raises:
    """Refuse a bin count outside what the GPU histogram kernels index.

    The kernels allocate `MAX_BINS`-wide threadgroup planes and index them
    by a UInt8 bin, so this is a hard structural limit rather than a tuning
    choice. `device_policy.MAX_GPU_BINS` refuses the same value earlier, at
    device-selection time; this is the backstop for a caller that reached a
    launch without passing through it.
    """
    if n_bins < 1:
        raise Error("GPU histogram needs a positive bin count")
    if n_bins > MAX_BINS:
        raise Error(
            "GPU histogram supports at most ",
            MAX_BINS,
            " bins; got ",
            n_bins,
        )


def require_device_can_host_kernels(
    contract: BackendContract, caps: DeviceCaps
) raises:
    """Refuse a device that cannot host the shipping kernels at all.

    One question, asked once when a session opens rather than per launch:
    does this device advertise the `MIN_SHARED_MEMORY_PER_BLOCK` the
    unspecialized histogram kernels allocate, and a threadgroup at least one
    launch granularity wide? A device that fails either cannot run any
    histogram this package builds, at any bin count and any tiling, so
    finding that out at the first launch of a training run wastes the
    upload that preceded it.

    Every backend MAX supports advertises far more than this. The check is
    cheap and its failure message is specific, which is worth more than the
    assumption on hardware nobody here has opened.
    """
    require_backend_covered(contract.api)
    if caps.max_shared_memory_per_block < MIN_SHARED_MEMORY_PER_BLOCK:
        raise Error(
            "this ",
            api_name(contract.api),
            " device reports ",
            caps.max_shared_memory_per_block,
            " bytes of threadgroup memory per block; the GPU histogram"
            " kernels allocate ",
            MIN_SHARED_MEMORY_PER_BLOCK,
            " (three ",
            MAX_BINS,
            "-wide Int32 planes) at every bin count",
        )
    if caps.max_threads_per_block < contract.launch_granularity:
        raise Error(
            "this ",
            api_name(contract.api),
            " device reports a maximum threadgroup width of ",
            caps.max_threads_per_block,
            ", below the ",
            contract.launch_granularity,
            " launch granularity every kernel here rounds to",
        )


def require_launch_geometry(
    contract: BackendContract,
    caps: DeviceCaps,
    grid_x: Int,
    grid_y: Int,
    block_threads: Int,
    shared_bytes: Int,
) raises:
    """Refuse a launch the reported device cannot run.

    The four bounds every kernel in this package is subject to, checked
    against what the device advertised rather than against a constant:

    - `grid_x` positive. No portable upper bound is checked because none has
      been established here: CUDA's `grid.x` limit is over two billion and
      the other backends allow at least that, so a feature or leaf-slot
      count that exceeded it would have failed for other reasons long
      before.
    - `grid_y` within `contract.max_grid_dim_y`, the tightest of the three
      backends. `derive_tiling` already clamps its tile count to it; a
      caller packing something else onto that axis (leaf batches, classes)
      may not have.
    - `block_threads` positive and within the device maximum. Not required
      to be a multiple of `launch_granularity`: rounding is
      `gpu_tiling.clamp_block_threads`' job, and a caller that deliberately
      launched a narrower block is launching something the device can run.
    - `shared_bytes` within the device's advertised per-block threadgroup
      memory. What the compiled kernel really requests, from
      `kernel_shared_request`, not the `n_bins * 12` model. The
      device-level version of the same question, asked once at session open
      rather than per launch, is `require_device_can_host_kernels`.

    Every message names the value, the bound, and the backend, so a failure
    on hardware nobody here has is diagnosable from the text alone.
    """
    require_backend_covered(contract.api)
    if grid_x < 1:
        raise Error("a launch needs a positive grid.x; got ", grid_x)
    if grid_y < 1:
        raise Error("a launch needs a positive grid.y; got ", grid_y)
    if grid_y > contract.max_grid_dim_y:
        raise Error(
            "grid.y is ",
            grid_y,
            ", above the portable bound of ",
            contract.max_grid_dim_y,
            " that one source targeting metal, cuda, and hip is subject to",
        )
    if block_threads < 1:
        raise Error(
            "a launch needs a positive threadgroup width; got ",
            block_threads,
        )
    if block_threads > caps.max_threads_per_block:
        raise Error(
            "threadgroup width ",
            block_threads,
            " is above the ",
            caps.max_threads_per_block,
            " this device reports for ",
            api_name(contract.api),
        )
    if shared_bytes < 1:
        raise Error(
            "a threadgroup histogram needs positive threadgroup memory; got ",
            shared_bytes,
        )
    if shared_bytes > caps.max_shared_memory_per_block:
        raise Error(
            "the compiled kernel requests ",
            shared_bytes,
            " bytes of threadgroup memory per block and this device reports"
            " only ",
            caps.max_shared_memory_per_block,
            " on ",
            api_name(contract.api),
        )


def require_histogram_launchable(
    contract: BackendContract,
    caps: DeviceCaps,
    tiling: HistogramTiling,
    grid_x: Int,
    n_bins: Int,
    compiled: KernelFeatures,
    selected: KernelFeatures = KernelFeatures.none(),
) raises:
    """The whole gate for one resolved histogram launch.

    Everything a caller holding a `HistogramTiling` needs to check, in one
    call: the bin count, the primitives the resolved strategy needs, the
    variants the plan selected against what the backend has run, and the
    geometry against what the device reported. `grid_x` is passed rather
    than derived because the axis carries the feature count in
    `histogram_gpu.mojo`, the active feature count under feature
    subsampling, and a leaf-slot count in `gpu_leaf_batching.mojo`.

    The two `KernelFeatures` are different questions and must not be the
    same value. `compiled` is what the build instantiated, which is what
    decides the threadgroup footprint; `selected` is what the plan chose,
    which is what the validation gate applies to. Passing `compiled` for
    both would refuse the shipping path, because linking `histogram_gpu`
    instantiates the batched kernels whether or not anything selects them.

    `selected` defaults to `KernelFeatures.none()`, which is the
    conservative answer for a caller that does not yet track what its plan
    chose: the geometry and primitive gates still apply and the
    specialization gate is inert. A caller that knows should say so, since
    an unstated selection cannot be checked.

    Raises on the first failure, naming it. Returns nothing on success: it
    is a gate, not a plan, and the plan it gates is `tiling`.
    """
    require_bins_supported(n_bins)
    require_primitives(contract, tiling.strategy)
    require_specializations_allowed(contract, selected)
    require_launch_geometry(
        contract,
        caps,
        grid_x,
        tiling.n_tiles,
        tiling.block_threads,
        kernel_shared_request(n_bins, compiled),
    )


def describe_launch(
    contract: BackendContract,
    tiling: HistogramTiling,
    grid_x: Int,
    n_bins: Int,
    compiled: KernelFeatures,
) raises -> String:
    """One line pairing a checked launch with the backend it was checked
    against, for benchmark output and bug reports. `compiled` is the build's
    variants, because what it reports is the threadgroup footprint."""
    return String(
        describe_contract(contract),
        " strategy=",
        strategy_name(tiling.strategy),
        " grid=",
        grid_x,
        "x",
        tiling.n_tiles,
        " block=",
        tiling.block_threads,
        " bins=",
        n_bins,
        " shared=",
        kernel_shared_request(n_bins, compiled),
    )


def porting_note(api: Int) raises -> String:
    """The one thing most likely to differ first when this source is run on
    a backend for the first time.

    Not a warning about correctness and not a prediction: each of these is a
    documented property of the API that the shared source currently has no
    reason to branch on, written down so the person holding the hardware
    knows where to look when something does differ.
    `docs/NVIDIA_GPU.md` and `docs/AMD_GPU.md` carry the long form.
    """
    require_backend_covered(api)
    if api == API_METAL:
        return String(
            "threadgroup memory is the scarce resource and the device"
            " budget is system RAM shared with the host dataset"
        )
    if api == API_CUDA:
        return String(
            "grid.y is capped at 65535, which is the bound the whole"
            " package is written to, and static threadgroup memory is"
            " capped at 48 KiB without an opt-in the shared source does"
            " not take"
        )
    if api == API_HIP:
        return String(
            "the wavefront is 64 lanes on CDNA and 32 on RDNA, which is"
            " why no width is assumed anywhere and the launch granularity"
            " is a multiple of both"
        )
    return String(
        "the device did not name its API, so it takes the portable floor:"
        " every bound is the tightest of the three backends"
    )


# `gpu_tiling.mojo` and `gpu_histogram_specializations.mojo` sit beneath
# this module and must stay there, or the layering closes into a cycle.
# This module reads their constants so the portability contract and the
# geometry it gates cannot drift into disagreeing about `MAX_GRID_DIM_Y`,
# `MAX_BINS`, or the threadgroup footprint.
