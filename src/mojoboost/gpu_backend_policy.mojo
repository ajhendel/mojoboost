"""Which GPU backend is in front of us, and what this repository has
actually done on it.

Two questions, kept apart on purpose:

1. **Is this backend one the shared GPU source targets?** Metal, CUDA, and
   HIP are, and so is a device whose API this process could not name, which
   takes the portable floor. Anything outside that set is refused rather
   than guessed at.
2. **Has this repository ever executed the shipping kernels on it?** Only
   Metal, on one Apple M4. `docs/GPU_VALIDATION.md` is the record, and every
   CUDA and HIP row in it reads "not run".

Collapsing those two into one "supported" flag is how a status table starts
claiming NVIDIA behavior. They are separate values here, and the second one
gates nothing about ordinary training: the whole design commitment of this
project is one portable source, so the portable kernels run on whatever
device MAX opens. What the second value gates is *specialization*. A kernel
variant, a vendor intrinsic, or a launch shape that only pays off on a
particular architecture may not be selected on a backend nobody has run,
because there is no measurement that could have justified it and no record
that it produces the same integers.

Where this module sits
----------------------
The leaf of the portability layer. It imports the API vocabulary from
`apple_gpu_policy.mojo`, which is where `API_METAL`, `API_CUDA`, `API_HIP`,
and `API_UNKNOWN` are defined and parsed today, and nothing else. It opens
no device, reads no device attribute, and does not detect which backend it
is on: the API code is an argument, because the two places that can answer
that question already do so and neither should be duplicated here.

- `device_policy.env_declared_api()` reads `MOJOBOOST_GPU_BACKEND`, the
  operator's declaration of the API name.
- `apple_gpu_policy.GpuProfile.from_reported` parses whatever an open
  `DeviceContext` reported, which is authoritative when it exists.

`gpu_portability.mojo` is the layer above: it turns a backend code from
either source into the launch contract the shared kernels are checked
against.

Environment contract, following `MOJOBOOST_GPU_TRANSFER_UNPROVEN` in
`unified_memory_policy.mojo`, which is the same shape of gate:

- `MOJOBOOST_GPU_BACKEND_UNVALIDATED=1` acknowledges that a specialized path
  is being selected on a backend this repository has never executed, and
  runs it anyway. Deliberately loud, and required for every specialization
  outside Metal until a validation record exists. Any number measured under
  it has to be reported with it.

LightGBM difference: LightGBM ships separate OpenCL (`gpu`) and CUDA
(`cuda`) device types, so its user chooses the backend. mojoboost has one
source for all three APIs and one `gpu` device value, so the backend is
detected rather than chosen, and this module is where the consequences of
that choice are written down.
"""

from std.os import getenv

from .apple_gpu_policy import (
    API_CUDA,
    API_HIP,
    API_METAL,
    API_UNKNOWN,
    api_name,
)


# --- Support levels ---------------------------------------------------

# An API code this module has no contract for. Not a device that reported
# nothing (that is `API_UNKNOWN`, which is covered); an out-of-range value,
# which is a caller bug and is refused rather than defaulted.
comptime SUPPORT_UNSUPPORTED = 0

# The shared GPU source targets this backend and the portable contract in
# `gpu_portability.mojo` covers it, but this repository has never executed a
# kernel on it. CUDA, HIP, and an unidentified device are all here.
comptime SUPPORT_PORTABLE = 1

# This repository has run the shipping kernels on this backend and recorded
# the run in `docs/GPU_VALIDATION.md`. Metal only, on one Apple M4, which is
# a statement about a device class and not about every device in it.
comptime SUPPORT_EXERCISED = 2


def support_name(level: Int) -> String:
    if level == SUPPORT_EXERCISED:
        return String("exercised")
    if level == SUPPORT_PORTABLE:
        return String("portable-untested")
    return String("unsupported")


def backend_support(api: Int) -> Int:
    """How far this repository has taken a backend.

    `SUPPORT_EXERCISED` for Metal and nothing else, because nothing else has
    run. `API_UNKNOWN` is `SUPPORT_PORTABLE` rather than unsupported: a
    device that did not answer an API-name query is still a device MAX
    opened, and refusing it would refuse every launch on the path that
    ships today, where nobody sets `MOJOBOOST_GPU_BACKEND` and no reported
    API name reaches the policy layer.
    """
    if api == API_METAL:
        return SUPPORT_EXERCISED
    if api == API_CUDA or api == API_HIP or api == API_UNKNOWN:
        return SUPPORT_PORTABLE
    return SUPPORT_UNSUPPORTED


def backend_is_covered(api: Int) -> Bool:
    """Whether the portable contract covers this backend at all."""
    return backend_support(api) != SUPPORT_UNSUPPORTED


def backend_is_exercised(api: Int) -> Bool:
    """Whether this repository has a record of running on this backend.

    Never a performance claim and never a claim about a particular device:
    see the status table in `docs/GPU_VALIDATION.md` for what was run, on
    what, and when.
    """
    return backend_support(api) == SUPPORT_EXERCISED


def backend_is_identified(api: Int) -> Bool:
    """Whether anything told us which API this is.

    False for `API_UNKNOWN`, which is the value every launch carries today:
    nothing in the shipping code reads an API name before it launches, and
    `MOJOBOOST_GPU_BACKEND` is unset on an ordinary run. It is the reason
    `require_backend_exercised` cannot refuse an unidentified device.
    """
    return api != API_UNKNOWN and backend_is_covered(api)


def env_ack_unvalidated() -> Bool:
    """Whether `MOJOBOOST_GPU_BACKEND_UNVALIDATED=1` acknowledges running a
    specialized path on a backend with no validation record."""
    return getenv("MOJOBOOST_GPU_BACKEND_UNVALIDATED") == "1"


def require_backend_covered(api: Int) raises:
    """Refuse an API code the portable contract does not cover.

    The clear failure for an unsupported build or device: a caller that
    hands a code outside `API_UNKNOWN`, `API_METAL`, `API_CUDA`, `API_HIP`
    gets an error naming what it passed, not a silent fallback to the
    portable floor. Nothing here can be reached by an ordinary run, which
    is the point; it catches a mis-plumbed backend code at the boundary
    instead of inside a kernel launch.
    """
    if backend_is_covered(api):
        return
    raise Error(
        "no GPU backend contract for API code ",
        api,
        "; the covered codes are ",
        API_UNKNOWN,
        " (unknown), ",
        API_METAL,
        " (metal), ",
        API_CUDA,
        " (cuda), and ",
        API_HIP,
        " (hip)",
    )


def require_backend_exercised(api: Int, what: String) raises:
    """Refuse a specialization on a *named* backend this repository has
    never run.

    `what` names the thing being asked for, so the error says which
    specialization was refused rather than only that one was.

    The escape hatch is `MOJOBOOST_GPU_BACKEND_UNVALIDATED=1`, and it exists
    for the same reason `MOJOBOOST_GPU_TRANSFER_UNPROVEN=1` does in
    `unified_memory_policy.mojo`: the evidence that would lift this gate can
    only be produced by running the thing the gate blocks, so without an
    override the gate is unsatisfiable by construction. A run that reaches a
    specialization through it is not a validated run and any number it
    produces has to be reported with the flag.

    **An unidentified backend is not refused**, and that is the honest limit
    of this gate rather than a hole left open by accident. `API_UNKNOWN` is
    the value every launch carries today: nothing in the shipping code reads
    an API name before it launches, and `MOJOBOOST_GPU_BACKEND` is unset on
    an ordinary run. So an unidentified device is indistinguishable here from
    the Apple part this repository was developed on, and refusing it would
    refuse every run that ships, including every Metal one. The gate is
    exactly as strong as the backend identification reaching it; plumbing
    the reported API name into the policy layer is what makes it bite, and
    that patch request is in `handoffs/connect_20_gpu_portability.md`.
    """
    require_backend_covered(api)
    if backend_is_exercised(api):
        return
    if not backend_is_identified(api):
        return
    if env_ack_unvalidated():
        return
    raise Error(
        "refusing '",
        what,
        "' on the ",
        api_name(api),
        " backend: this repository has never executed a GPU kernel on it"
        " (see docs/GPU_VALIDATION.md), so no measurement supports"
        " specializing for it; set MOJOBOOST_GPU_BACKEND_UNVALIDATED=1 to"
        " run it anyway and report that flag with any number it produces",
    )


def describe_backend(api: Int) -> String:
    """One line for a diagnostic record or a bug report."""
    return String(
        "backend=",
        api_name(api),
        " support=",
        support_name(backend_support(api)),
    )
