"""The per-backend matrix: what each API HAS, and what this package USES.

Written 2026-08-19, when the project's priority was restated. Portability is
the commitment (Apple Silicon, NVIDIA and AMD all work) and single-source
compilation is only the means. Multiple paths, vendor-specific kernels and
toggles are all permitted; the one thing refused is dropping a vendor. See
`docs/GPU_PORTABILITY.md`.

Before this module the table was one column. `gpu_portability.contract_for`
constructed a `BackendContract` in which every field except `unified_memory`
and `support` was the same literal for every API: `device_float64_permitted`
False everywhere, the five primitives True everywhere, `max_grid_dim_y` fixed
at CUDA's bound everywhere. That is a *floor*, correct for one source and
wrong for a project that now wants each backend as fast as that backend can
go. This module is the floor turned into a matrix.

TWO TABLES, AND THEY MUST NOT BE MERGED
---------------------------------------
`BackendCapability` answers **what does this backend have**. Float64,
subgroup width, `grid.y` bound. It is a property of the API or the silicon,
verifiable against a vendor's specification, and not a decision anyone here
gets to make.

`BackendPolicy` answers **what does this package use**. Whether Float64 is
emitted, which histogram strategy is preferred, how much of the device budget
a partial-histogram buffer may claim. It is a decision, it is tunable, and it
should be driven by measurement.

Keeping them apart is the whole point. Merged, "Metal cannot" and "we chose
not to" become the same fact, and that confusion is exactly how Apple's
missing Float64 came to bind NVIDIA and AMD, which have it. A policy may read
a capability. A capability may never read a policy.

THE FOURTH COLUMN IS DERIVED, NEVER TYPED
-----------------------------------------
`safe_capabilities()` is the portable floor: for each row, the weakest value
across every covered backend, computed here rather than written down. A
hand-maintained "default" column drifts, and the way it drifts is by
permitting something no real backend actually has. `API_UNKNOWN` resolves to
it.

That fourth column is not a fallback today. It is what everything uses.
Nothing on the shipping launch path reads an API name before it launches, so
`API_UNKNOWN` is the value every launch carries. **A matrix changes no
behavior until that plumbing exists**, and this module deliberately does not
pretend otherwise: see `matrix_is_reachable`.

EVERY ROW IS TAGGED NUMERIC OR SCHEDULING, AND THAT TAG IS THE POINT
--------------------------------------------------------------------
A `DIVERGENCE_SCHEDULING` row changes launch geometry, tile shape, occupancy
or buffer budget.

**The tag is conditional, and the condition must be stated rather than
assumed.** A block count is a summation order. A partial-histogram
replication factor decides how many partial sums combine and in what
sequence. Those are scheduling decisions that would be *numeric* decisions
under a floating-point accumulator, because float addition is neither
associative nor exact.

They are inert here for one reason: `gpu_portability` records that every
accumulation in this package is Int32 through integer atomics, with no float
atomic anywhere, and integer addition is associative and exact. Reorder that
reduction all you like.

So the honest statement of the rule is: **a row is NUMERIC if it changes the
sequence or the precision of the arithmetic.** Geometry rows escape that only
while the accumulator is integral. The day anyone adds a float atomic or a
float partial-histogram flush, block count and replication factor become
numeric rows and this classification is wrong until it is redone.
`require_integer_accumulation` exists so that day fails loudly.

A `DIVERGENCE_NUMERIC` row changes arithmetic. Float64, accumulator width,
anything a backend may fuse or round differently.

The tag makes bit-identity a mechanical mode rather than a guess:

    identity mode = pin every NUMERIC row to the safe column,
                    leave every SCHEDULING row per-vendor.

So a build can be as fast as its silicon on geometry and still produce the
same model everywhere, and the cost of that guarantee becomes a number to
measure instead of an argument to have. `policy_for(api, identity=True)` is
that mode.

WHAT THE TAG DOES NOT COVER
---------------------------
Compiler behavior. The 2026-08-18 CUDA defect was a multiply-add the backend
contracted into an FMA where Metal did not, and it disagreed with Metal on
1841 of 3000 rows. No runtime table can toggle that: contraction happens in
codegen, and the fix was to move the multiply to the host so the kernel had a
plain add and nothing left to fuse. Rows here describe capability and policy.
They do not describe what a compiler will do to an expression, and a future
reader should not expect identity mode to catch that class.

LAYERING
--------
This module imports API codes and the two portable geometry constants and
nothing else. `gpu_portability.mojo` imports *this*; this must never import
`gpu_portability`, or the layering closes into a cycle.
"""

from std.os import getenv

from .apple_gpu_policy import (
    API_CUDA,
    API_HIP,
    API_METAL,
    API_UNKNOWN,
    api_name,
)
from .gpu_tiling import (
    MAX_GRID_DIM_Y,
    WARP_GRANULARITY,
)

comptime GRID_AXES = 2
"""Mirrors `gpu_portability.GRID_AXES`, and must not drift from it. It is
defined here rather than imported because `gpu_portability` imports this
module and the reverse would close the layering into a cycle. The two are
checked against each other by `gpu_portability.contract_for`, which passes
this value into the field that constant used to fill."""


# --- Row taxonomy -------------------------------------------------------
#
# Two orthogonal tags. `ROW_*` says who decides; `DIVERGENCE_*` says what
# disagreeing about it would cost.

comptime ROW_CAPABILITY = 0
"""The backend has it or does not. Not ours to choose."""

comptime ROW_POLICY = 1
"""We use it or do not. Ours to choose, and to measure."""

comptime DIVERGENCE_SCHEDULING = 0
"""Changing this row cannot change the model. Geometry, occupancy, budget.
Safe to vary per vendor even under identity mode, because accumulation is
Int32 and integer addition is associative."""

comptime DIVERGENCE_NUMERIC = 1
"""Changing this row changes arithmetic, so two backends that disagree about
it produce different models. Pinned to the safe column under identity
mode."""


comptime N_BACKENDS = 4
"""API_UNKNOWN, API_METAL, API_CUDA, API_HIP. The safe column is
API_UNKNOWN's, and it is derived from the other three."""


def divergence_name(kind: Int) raises -> String:
    """For diagnostics and for the row table printed by `describe_matrix`."""
    if kind == DIVERGENCE_SCHEDULING:
        return "scheduling"
    if kind == DIVERGENCE_NUMERIC:
        return "numeric"
    raise Error("no such divergence kind: ", kind)


# --- Column one: what the backend HAS -----------------------------------


@fieldwise_init
struct BackendCapability(Copyable, Movable):
    """What one API specifies, per row. No decisions live here.

    Every field is a property of the API or the silicon that a vendor's
    documentation would confirm. Nothing here was measured by this
    repository, and `gpu_backend_policy.backend_support` remains the only
    field anywhere that carries what actually ran.
    """

    var api: Int

    var float64: Bool
    """DIVERGENCE_NUMERIC. Whether the API offers Float64 device arithmetic
    at all. False on Metal, which is the fact that used to bind everyone.
    True on CUDA and HIP. Note for whoever spends this: consumer NVIDIA parts
    run Float64 at roughly a sixty-fourth of their Float32 rate, and the only
    NVIDIA device this project has executed on is a consumer RTX 5090, so
    having it and wanting it are different questions."""

    var subgroup_width: Int
    """DIVERGENCE_SCHEDULING. Lanes in lockstep, or 0 for unknown. Metal
    refuses the query, so 0 there is a real answer rather than a gap. 32 on
    CUDA, 64 on HIP, both from the vendors' programming models rather than
    from any query run here."""

    var max_grid_dim_y: Int
    """DIVERGENCE_SCHEDULING. The `grid.y` bound the API specifies. CUDA's
    65535 is the tightest, which is why it was the value every backend
    carried before this matrix existed."""

    var grid_axes: Int
    """DIVERGENCE_SCHEDULING, and read this one carefully because the name
    invites the wrong reading. It is **the axes this package launches on**,
    which is `gpu_portability.GRID_AXES` and is 2 on every column, not the
    axes an API offers, which is 3 everywhere. `grid.x` carries features or
    leaf slots, `grid.y` carries row tiles or classes, and nothing carries a
    `grid.z`.

    It is therefore closer to a policy row than a capability row, and it sits
    here only because `BackendContract` has always carried it in this
    position. If a backend ever wants a third axis, this is the row to split
    into a real capability and a real policy first."""

    var launch_granularity: Int
    """DIVERGENCE_SCHEDULING. The multiple a threadgroup width rounds to.
    A rounding rule, not a width claim."""

    var threadgroup_barrier: Bool
    var shared_static_alloc: Bool
    var shared_int32_atomic_add: Bool
    var global_int32_atomic_add: Bool
    var in_order_queue: Bool
    """The five primitives the shipping kernels use unconditionally. True on
    every covered backend including the unknown one, which is deliberate:
    the kernels already use all five on every device MAX opens, and Metal
    demonstrates at least one real device honors them. Setting any False for
    the unknown backend would refuse the path that ships today."""

    var api_implies_unified_memory: Bool
    """DIVERGENCE_SCHEDULING, and read the name carefully. This is whether
    the API *name alone* implies a shared host/device budget, which is true
    of Metal and of nothing else. An integrated CUDA part, CUDA managed
    memory and an XNACK-enabled HIP device are all unified in ways an API
    name cannot distinguish from a discrete card, so on those backends the
    answer must come from a device report. `contract_from_profile` already
    lets a report win, and that behavior is unchanged."""


def capabilities_for(api: Int) raises -> BackendCapability:
    """The capability column for one API.

    This is the matrix. Rows are the fields of `BackendCapability`, columns
    are the branches below, and `API_UNKNOWN` takes the derived floor rather
    than a fourth set of literals.
    """
    if api == API_METAL:
        return BackendCapability(
            API_METAL,
            False,              # float64: Apple silicon has none
            0,                  # subgroup_width: Metal refuses the query
            MAX_GRID_DIM_Y,     # Metal allows more; see safe_capabilities
            GRID_AXES,
            WARP_GRANULARITY,
            True, True, True, True, True,
            True,               # unified memory follows from the API here
        )
    if api == API_CUDA:
        return BackendCapability(
            API_CUDA,
            True,               # float64: present, and slow on consumer parts
            32,                 # warp
            MAX_GRID_DIM_Y,     # 65535, and CUDA is where that bound comes from
            GRID_AXES,
            WARP_GRANULARITY,
            True, True, True, True, True,
            False,
        )
    if api == API_HIP:
        return BackendCapability(
            API_HIP,
            True,               # float64: present
            64,                 # wavefront
            MAX_GRID_DIM_Y,
            GRID_AXES,
            WARP_GRANULARITY,
            True, True, True, True, True,
            False,
        )
    if api == API_UNKNOWN:
        return safe_capabilities()
    raise Error(
        "no capability column for API code ",
        api,
        "; covered codes are ",
        API_UNKNOWN,
        " (unknown), ",
        API_METAL,
        " (metal), ",
        API_CUDA,
        " (cuda), ",
        API_HIP,
        " (hip)",
    )


def safe_capabilities() raises -> BackendCapability:
    """The fourth column, derived from the other three.

    For each row, the weakest value any covered backend offers. Computed
    rather than typed, because a hand-maintained default column drifts, and
    it drifts by permitting something no real backend has.

    `subgroup_width` resolves to 0 because Metal reports none, and 0 means
    unknown rather than one lane. Nothing in this package divides by it.
    """
    var metal = capabilities_for(API_METAL)
    var cuda = capabilities_for(API_CUDA)
    var hip = capabilities_for(API_HIP)

    var float64 = metal.float64 and cuda.float64 and hip.float64

    var width = metal.subgroup_width
    if cuda.subgroup_width < width:
        width = cuda.subgroup_width
    if hip.subgroup_width < width:
        width = hip.subgroup_width

    var grid_y = metal.max_grid_dim_y
    if cuda.max_grid_dim_y < grid_y:
        grid_y = cuda.max_grid_dim_y
    if hip.max_grid_dim_y < grid_y:
        grid_y = hip.max_grid_dim_y

    var axes = metal.grid_axes
    if cuda.grid_axes < axes:
        axes = cuda.grid_axes
    if hip.grid_axes < axes:
        axes = hip.grid_axes

    return BackendCapability(
        API_UNKNOWN,
        float64,
        width,
        grid_y,
        axes,
        WARP_GRANULARITY,
        metal.threadgroup_barrier and cuda.threadgroup_barrier
        and hip.threadgroup_barrier,
        metal.shared_static_alloc and cuda.shared_static_alloc
        and hip.shared_static_alloc,
        metal.shared_int32_atomic_add and cuda.shared_int32_atomic_add
        and hip.shared_int32_atomic_add,
        metal.global_int32_atomic_add and cuda.global_int32_atomic_add
        and hip.global_int32_atomic_add,
        metal.in_order_queue and cuda.in_order_queue and hip.in_order_queue,
        False,
    )


# --- Column two: what this package USES ---------------------------------


@fieldwise_init
struct BackendPolicy(Copyable, Movable):
    """What this package chooses to do on one backend.

    Every field is a decision, so every field is allowed to change on a
    measurement. Nothing here is a property of the silicon; that is
    `BackendCapability`.
    """

    var api: Int

    var identity_mode: Bool
    """Whether every NUMERIC row was pinned to the safe column. When True the
    model this build produces is intended to be bit-identical to the model
    any other backend produces under the same flag, and SCHEDULING rows are
    still free to differ per vendor."""

    var emit_float64: Bool
    """DIVERGENCE_NUMERIC. Whether the shared source may emit Float64 device
    arithmetic. False under identity mode on every backend. Otherwise it
    follows the capability, which is the change this matrix exists to make:
    Apple's floor no longer binds a backend that has Float64.

    Before spending it, note that accumulation here is already fixed-point
    Int32, which is both faster than Float64 and exact, so there is no
    histogram precision to recover. If Float64 earns a place it is as an
    accuracy lever elsewhere in the objective."""

    var max_grid_dim_y: Int
    """DIVERGENCE_SCHEDULING. Free to be the backend's real bound even under
    identity mode, because a different grid shape reorders an integer sum and
    integer addition is associative."""

    var subgroup_width: Int
    """DIVERGENCE_SCHEDULING. As reported, 0 when unknown."""

    var launch_granularity: Int
    """DIVERGENCE_SCHEDULING."""


def policy_for(api: Int, identity: Bool) raises -> BackendPolicy:
    """What to do on `api`, under or outside identity mode.

    The one place the NUMERIC/SCHEDULING tag turns into behavior. Every
    numeric row is taken from `safe_capabilities()` when `identity` is set
    and from the backend's own column when it is not; every scheduling row
    is taken from the backend's own column either way.

    `identity=True` is therefore not "turn off the optimizations". It is
    "pin the arithmetic, keep the geometry", which is a much smaller and much
    better defined thing to give up.
    """
    var have = capabilities_for(api)
    var floor = safe_capabilities()

    var float64 = have.float64
    if identity:
        float64 = floor.float64

    return BackendPolicy(
        api,
        identity,
        float64,
        have.max_grid_dim_y,
        have.subgroup_width,
        have.launch_granularity,
    )


def numeric_rows_agree(left: BackendPolicy, right: BackendPolicy) -> Bool:
    """Whether two backends' policies can produce the same model.

    Compares the NUMERIC rows only, which is the whole content of the claim.
    Two policies that disagree on `max_grid_dim_y` may still be bit-identical;
    two that disagree on `emit_float64` cannot be.

    This is a necessary condition and not a sufficient one. It says the
    *policy* does not force divergence. It says nothing about compiler
    contraction, about `libm`, or about any of the ways two backends remain
    free to differ once the table has been satisfied. See the module
    docstring on the 2026-08-18 FMA defect.
    """
    return left.emit_float64 == right.emit_float64


comptime IDENTITY_ENV = "MOJOTREES_BIT_IDENTITY"
"""The one switch that turns identity mode on. Off by default, because under
the 2026-08-19 priority speed and portability come first and bit-identity is
subordinate to both."""


def identity_mode_requested() -> Bool:
    """Whether the operator asked for bit-identical models across backends.

    `MOJOTREES_BIT_IDENTITY=1` and nothing else. Off by default.

    What it does is narrow on purpose. It pins every NUMERIC row to the safe
    column and leaves every SCHEDULING row per-backend, so a build in identity
    mode still picks its own grid shape, its own threadgroup width and its own
    buffer budget. It is not "turn off the optimizations", and the distinction
    is the reason the row tags exist.

    What it cannot do is make two compilers agree. See the module docstring on
    the 2026-08-18 FMA defect: contraction is a codegen decision and no runtime
    flag reaches it.
    """
    return getenv(IDENTITY_ENV) == "1"


def require_integer_accumulation(uses_float_atomics: Bool) raises:
    """Fail loudly if the invariant the SCHEDULING tag rests on is broken.

    Every `DIVERGENCE_SCHEDULING` row in this module is safe to vary per
    vendor, under identity mode included, on the strength of one fact: every
    accumulation in this package is Int32 through integer atomics, so a
    different block count or replication factor reorders an exact associative
    sum and cannot move a bit.

    Introduce a float atomic and that stops being true. Block count becomes a
    summation order, replication factor becomes a rounding sequence, and rows
    tagged scheduling here quietly become numeric rows that identity mode is
    no longer pinning. Nothing would fail; the models would simply stop
    matching, which is the worst way to find out.

    So the invariant is a call rather than a comment. A caller that adds a
    float accumulation path passes True here and gets a clear error pointing
    at the reclassification it now owes.
    """
    if uses_float_atomics:
        raise Error(
            "backend_matrix classifies grid shape, block count and"
            " replication factor as DIVERGENCE_SCHEDULING, which is only"
            " sound while every accumulation is Int32 through integer"
            " atomics. A float accumulation path makes each of those a"
            " summation order, so they are numeric rows now and identity"
            " mode is not pinning them. Reclassify before shipping this",
        )


def matrix_is_reachable(api: Int) -> Bool:
    """Whether a per-backend column can actually be selected on this path.

    False for `API_UNKNOWN`, which is what every shipping launch carries
    today because nothing on the launch path reads an API name before it
    launches. Until that plumbing exists, every caller of `policy_for` gets
    the safe column no matter what silicon it is running on, and a matrix
    that nobody can reach is a matrix that changes nothing.

    Exposed rather than commented so the gap is visible from a diagnostic
    instead of from a docstring, and so the eventual plumbing has one
    predicate to flip.
    """
    return api != API_UNKNOWN


def _bool_text(value: Bool) -> String:
    """How a Bool is spelled in a described policy. Local rather than
    imported, for the same reason `gpu_portability.mojo` keeps its own: this
    module sits below that one and importing it would close the layering into
    a cycle."""
    if value:
        return String("true")
    return String("false")


def describe_policy(policy: BackendPolicy) -> String:
    """One line for a diagnostic record or a bug report."""
    return String(
        "backend=",
        api_name(policy.api),
        " identity=",
        _bool_text(policy.identity_mode),
        " emit_float64=",
        _bool_text(policy.emit_float64),
        " max_grid_y=",
        policy.max_grid_dim_y,
        " subgroup_width=",
        policy.subgroup_width,
        " granularity=",
        policy.launch_granularity,
        " reachable=",
        _bool_text(matrix_is_reachable(policy.api)),
    )
