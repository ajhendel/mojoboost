"""The device-policy contract: one authoritative decision engine in Mojo.

Every question "should this training run go to the accelerator, and if not,
why not" is answered here and nowhere else. Before this module the answer
existed twice: once in `device.mojo` (the vocabulary, the availability
probe, and a size heuristic) and once again, larger and with a different
set of rules, in `python/mojotrees/device_selection.py`. Two engines that
answer the same question will disagree eventually, and the disagreement is
invisible to the user until a run lands on a backend they did not expect.

So this module owns all of it:

- the device vocabulary (`cpu`, `gpu`, `auto`) and its codes,
- what the GPU training path supports (objectives, outputs, row and bin
  limits),
- the memory estimate for one GPU training session,
- how detected hardware capabilities are turned into a plan,
- the crossover evidence table and the rule that reads it,
- the fallback semantics for unknown hardware,
- the refusal an explicit `gpu` request gets when it cannot run,
- which transfer route each device buffer is on, by carrying the plan
  unified_memory_policy.mojo resolves,
- how much of the one-time startup cost the process has already paid, by
  carrying the `SessionState` initialization.mojo defines.

The last two are carried, not decided. A route is chosen by
unified_memory_policy.mojo against its own evidence ladder and a session
state is observed by initialization.mojo; this module folds both into the
decision so that one report answers "where does this run, on what hardware,
moving bytes how, having already paid what". Neither one may select a
backend: a transfer route and a warm session are performance facts, and
`crossover_rules()` is the only place a performance fact is allowed to
change a device.

`device.mojo` is a thin compatibility facade over this module.
`python/mojotrees/device_selection.py` is reduced to extracting plain
workload metadata from `X` and `y` and formatting the decision this module
returns. Neither one decides anything.

Callers that are not Mojo
-------------------------
`decide_device` takes structs, which no CPython binding, C API, or CLI can
construct. `decide_device_report` and `decide_device_report_reported` are
the flat forms of the same call: plain scalars in, the serialized decision
out. They are marshallers and nothing else, and they exist so that each of
those callers does not grow its own. `resolve_device_full` is the raising
form for a trainer that knows its whole workload, as against `resolve_device`,
which knows only the shape and says so.

The contract
------------
`DeviceRequest` is the serializable input: the requested device, the shape
(rows, features, outputs, bins), the objective code, and the flags that
change what the GPU path can do (sparse input, categorical features,
missing values, validation use). `DeviceCapabilities` is the other input:
what the build and the device in front of us can do, either detected here
or injected by a caller that already has a `DeviceContext` open.

`DeviceDecision` is the serializable output: the selected backend, the
ordered blocking reasons, the warnings, the memory estimate, the policy
version, and the identifier of the evidence a GPU selection rests on.
`DeviceDecision.serialize()` renders it as `key=value` lines, which is the
form the bindings hand to Python: Python parses those lines and formats
them, it does not recompute them.

Three requested devices, and what each one means
------------------------------------------------
- `cpu`: the dependable path. Float64 throughout, every objective, every
  entry point. It always resolves to itself.
- `gpu`: an explicit request. It runs on the accelerator or it raises.
  There is no silent fallback, because a fallback turns "my GPU run" into
  "a CPU run that took the same wall clock and I never knew".
- `auto`: the GPU only when the GPU path covers the workload and evidence
  says the GPU is the faster choice for that shape on that device. With no
  such evidence it chooses the CPU and says so.

Why `auto` is the CPU everywhere today
--------------------------------------
`crossover_rules()` is empty. Nothing in this repository has measured a
workload size where GPU training beats CPU training: the one end-to-end
measurement that exists (Apple M4, bench/bench_train_gpu.mojo) came out
slower than the CPU trainer, and no NVIDIA or AMD device has run this code
at all (docs/GPU_VALIDATION.md). A threshold invented here would be a
performance claim with nothing under it, so the table ships empty and
`auto` conservatively resolves to the CPU. Adding a rule is a benchmarking
result, not a code change: it carries its `evidence` identifier, and
`POLICY_VERSION` is bumped with it.

`MOJOTREES_AUTO_MIN_CELLS` is the escape hatch for running that benchmark:
an integer cell count (`n_rows * n_features`) at or above which `auto`
selects the GPU, `0` meaning "whenever the GPU path covers the workload",
unset or negative meaning the heuristic is off. A run that reaches the GPU
through it is reported with `EVIDENCE_ENV` and a warning, never as a
validated choice.

`MOJOTREES_DISABLE_GPU=1` makes this module report that no accelerator is
present: `gpu` raises and `auto` chooses the CPU on a machine that does
have one. It exists to exercise the unavailable-GPU path and to pin a
mixed fleet to the CPU backend.

Unknown hardware
----------------
A device that reports nothing gets `GpuProfile.generic()`, the conservative
portable profile in apple_gpu_policy.mojo. It is deliberately not
Apple-shaped and not NVIDIA-shaped. Nothing in this module infers hardware
from an operating system name or from a marketing chip string: when a
caller has an open `DeviceContext`, the reported attributes are
authoritative and arrive through `DeviceCapabilities.from_profile`; when
nobody has opened one, the profile is the portable fallback and the
decision says so through `PROFILE_FALLBACK`. `MOJOTREES_GPU_BACKEND` is
honored only as an operator's declaration of the API name, is recorded as
`PROFILE_DECLARED`, and never overrides a reported capability.

Availability is a build property
--------------------------------
Mojo resolves `has_accelerator()` at compile time, so a binary built where
an accelerator was present reports one as available. On a redistributed
build (a wheel) a `gpu` request therefore fails when the device is opened
rather than when it is resolved. `WARN_BUILD_TIME_AVAILABILITY` marks every
decision that rests on that comptime answer, and `MOJOTREES_DISABLE_GPU=1`
is the way to pin such a build to the CPU.

LightGBM difference: LightGBM spells this `device_type` and takes `cpu`,
`gpu`, or `cuda`, with no `auto`. mojotrees has a single portable GPU
backend rather than separate OpenCL and CUDA ones, so the value is `gpu`
for every accelerator, and `auto` is an addition.
"""

from std.os import getenv
from std.sys import has_accelerator

from .apple_gpu_policy import (
    API_UNKNOWN,
    BYTES_PER_PARTIAL_CELL,
    CROSSOVER_DISABLED,
    GpuProfile,
    api_name,
    apple_generation_name,
    parse_api,
    partial_budget_bytes,
)
# `CUSTOM` only. The other objective codes were imported to spell out the
# built-in list here, and that list now comes from objective_registry.mojo,
# so naming them again would be re-establishing the duplicate by hand.
# `CUSTOM` stays because `_collect_blocks` refuses it by name and gives a
# reason specific to it.
from .boosting import CUSTOM
from .initialization import SessionState, warmup_level_name

# The one table of objective facts. Imported rather than restated: this
# module used to carry its own copy of the built-in objective list, its own
# copy of which objectives have a device-side derivative kernel, and its own
# `LAMBDARANK = 7`, which made three lists that had to be edited together and
# nothing that would notice if they were not.
#
# Safe to import: objective_registry.mojo imports only `boosting` (which this
# module already imports) and `metrics` (which imports nothing local), so it
# reaches neither `model.mojo` nor the GPU kernel stack and closes no cycle.
from .objective_registry import (
    LAMBDARANK,
    objective_gradients_on_device,
    objective_is_builtin,
)

# Aliased on the way in. unified_memory_policy.mojo has its own
# `block_reason_name` and its own `EVIDENCE_NONE`, and both mean something
# different here: a transfer-route block is not a device block, and its
# evidence ladder is not the crossover evidence identifier. Importing them
# under their own names would shadow this module's, which is how a report
# ends up naming a route refusal as the reason a GPU request was denied.
from .unified_memory_policy import (
    SessionMemoryPlan,
    evidence_name as transfer_evidence_name,
    plan_session_routes,
    retire_event_name,
    route_name as transfer_route_name,
    transfer_block_name,
    transfer_role_name,
)


# --- Mirrors. NOT pinned by any test in this repository. ---
#
# Three constants, all from histogram_gpu.mojo, copied rather than imported
# because importing that module pulls the whole `max.gpu.*` kernel stack into
# a layer that has to stay compilable and testable on a machine with no
# accelerator. Same reason objective_registry.mojo does not import
# gpu_objectives_native.mojo.
#
# The objective mirrors that used to sit here are gone: `LAMBDARANK` and the
# built-in objective list now come from objective_registry.mojo, which is the
# one table of objective facts and which this module can import without
# closing a cycle.
#
# There is no `tests/parallel/test_device_policy.mojo`. Nothing asserts that
# the three below still equal their source, so each is a copy that can drift
# silently, and a drifted `MAX_GPU_ROWS` or `MAX_GPU_BINS` admits a workload
# the kernels cannot index. handoffs/connect_05_device_policy.md specifies
# that test; it is unwritten and unrun, and this comment says so rather than
# claiming a guarantee that does not exist.

# `MAX_ROWS` in histogram_gpu.mojo: the histogram and partition kernels
# index rows as Int32, which is also where the fixed-point accumulator
# stops being exact.
comptime MAX_GPU_ROWS = Int(Int32.MAX)

# `max_bins must be in [2, 256]` in binning.mojo, and `MAX_BINS` in
# histogram_gpu.mojo, which reserves shared memory for that many.
comptime MIN_GPU_BINS = 2
comptime MAX_GPU_BINS = 256

# --- End mirrors. ---


comptime CPU_DEVICE = 0
comptime GPU_DEVICE = 1
comptime AUTO_DEVICE = 2

# `DeviceDecision.selected_device` when the request was refused. An
# explicit `gpu` that cannot run selects nothing; it does not select the
# CPU, because falling back is exactly what this policy refuses to do.
comptime NO_DEVICE = -1

# `DeviceRequest.objective` when the caller did not declare one. The
# objective gates are skipped for it and the decision carries
# `WARN_INCOMPLETE_REQUEST`, which is how the compatibility entry point in
# device.mojo keeps its four-argument signature without this module
# inventing an objective the caller never named.
comptime OBJECTIVE_UNSPECIFIED = -2

# `DeviceRequest.n_bins` when the caller did not declare one. Same
# treatment: the bin-limit gate is skipped, the histogram terms of the
# memory estimate are unknown, and the estimate is marked partial.
comptime BINS_UNSPECIFIED = 0

# Cells (`n_rows * n_features`) at or above which `auto` chooses the GPU
# with no rule behind it. Negative disables the heuristic, which is the
# default: see the module docstring for why there is no measured crossover
# to ship. Deliberately the same sentinel apple_gpu_policy.mojo reports on
# `CrossoverInputs.min_cells`.
comptime AUTO_MIN_CELLS = CROSSOVER_DISABLED

# Bumped whenever a crossover rule is added, removed, or retuned, or
# whenever a gate below changes what it admits. A decision carries it so a
# report from one release can be told apart from a report from another.
comptime POLICY_VERSION = 1

# `DeviceDecision.evidence_id` values that are not a crossover rule name.
comptime EVIDENCE_NONE = String("none")
comptime EVIDENCE_EXPLICIT = String("explicit-request")
comptime EVIDENCE_ENV = String("MOJOTREES_AUTO_MIN_CELLS")


# --- Where a capability profile came from -----------------------------

comptime PROFILE_NONE = 0
"""No accelerator, so no profile was built."""

comptime PROFILE_FALLBACK = 1
"""`GpuProfile.generic()`: nobody has opened a device, so the conservative
portable profile stands in. Not Apple-shaped and not NVIDIA-shaped."""

comptime PROFILE_DECLARED = 2
"""An operator named the API through `MOJOTREES_GPU_BACKEND`. The numbers
are still the portable fallback; only the API name is theirs."""

comptime PROFILE_REPORTED = 3
"""Read from an open `DeviceContext`. Authoritative."""

comptime PROFILE_SYNTHETIC = 4
"""A named fixture (`apple_synthetic`), never a reading. Only tests and
benchmarks should produce this."""


def profile_source_name(source: Int) -> String:
    if source == PROFILE_NONE:
        return String("none")
    if source == PROFILE_FALLBACK:
        return String("fallback")
    if source == PROFILE_DECLARED:
        return String("declared")
    if source == PROFILE_REPORTED:
        return String("reported")
    if source == PROFILE_SYNTHETIC:
        return String("synthetic")
    return String("unknown")


# --- Blocking reasons -------------------------------------------------
#
# A block is something that will actually fail on the GPU path. Each one
# corresponds to a rule enforced somewhere else: the trainer's guards in
# train_gpu.mojo and boosting.mojo, the binner's range in binning.mojo, or
# the kernels' indexing limits in histogram_gpu.mojo. Codes are stable and
# travel into serialized reports, so they are appended to, never renumbered.

comptime BLOCK_NO_ACCELERATOR = 1
comptime BLOCK_GPU_DISABLED_ENV = 2
comptime BLOCK_SPARSE_INPUT = 3
comptime BLOCK_CUSTOM_OBJECTIVE = 4
comptime BLOCK_RANKING_OBJECTIVE = 5
comptime BLOCK_UNKNOWN_OBJECTIVE = 6
comptime BLOCK_VALIDATION_SET = 7
comptime BLOCK_ROW_LIMIT = 8
comptime BLOCK_BIN_LIMIT = 9
comptime BLOCK_OUTPUT_LIMIT = 10
comptime BLOCK_MEMORY_BUDGET = 11


def block_reason_name(code: Int) -> String:
    if code == BLOCK_NO_ACCELERATOR:
        return String("no-accelerator")
    if code == BLOCK_GPU_DISABLED_ENV:
        return String("gpu-disabled-env")
    if code == BLOCK_SPARSE_INPUT:
        return String("sparse-input")
    if code == BLOCK_CUSTOM_OBJECTIVE:
        return String("custom-objective")
    if code == BLOCK_RANKING_OBJECTIVE:
        return String("ranking-objective")
    if code == BLOCK_UNKNOWN_OBJECTIVE:
        return String("unknown-objective")
    if code == BLOCK_VALIDATION_SET:
        return String("validation-set")
    if code == BLOCK_ROW_LIMIT:
        return String("row-limit")
    if code == BLOCK_BIN_LIMIT:
        return String("bin-limit")
    if code == BLOCK_OUTPUT_LIMIT:
        return String("output-limit")
    if code == BLOCK_MEMORY_BUDGET:
        return String("memory-budget")
    return String("unknown-block")


# --- Warnings ---------------------------------------------------------
#
# A warning never changes the selected backend. It records something the
# reader should know about how much the decision is worth.

comptime WARN_BUILD_TIME_AVAILABILITY = 1
comptime WARN_UNKNOWN_HARDWARE = 2
comptime WARN_SYNTHETIC_CAPABILITIES = 3
comptime WARN_UNIFIED_MEMORY_BUDGET = 4
comptime WARN_INCOMPLETE_REQUEST = 5
comptime WARN_EXPLICIT_GPU_UNMEASURED = 6
comptime WARN_ENV_THRESHOLD_UNVALIDATED = 7
comptime WARN_HOST_GRADIENT_PATH = 8
comptime WARN_MEMORY_BUDGET_UNKNOWN = 9
comptime WARN_COLD_SESSION = 10
comptime WARN_UNPROVEN_TRANSFER_ROUTE = 11
comptime WARN_KERNEL_RETIREMENT_ROUTE = 12


def warning_name(code: Int) -> String:
    if code == WARN_BUILD_TIME_AVAILABILITY:
        return String("build-time-availability")
    if code == WARN_UNKNOWN_HARDWARE:
        return String("unknown-hardware")
    if code == WARN_SYNTHETIC_CAPABILITIES:
        return String("synthetic-capabilities")
    if code == WARN_UNIFIED_MEMORY_BUDGET:
        return String("unified-memory-budget")
    if code == WARN_INCOMPLETE_REQUEST:
        return String("incomplete-request")
    if code == WARN_EXPLICIT_GPU_UNMEASURED:
        return String("explicit-gpu-unmeasured")
    if code == WARN_ENV_THRESHOLD_UNVALIDATED:
        return String("env-threshold-unvalidated")
    if code == WARN_HOST_GRADIENT_PATH:
        return String("host-gradient-path")
    if code == WARN_MEMORY_BUDGET_UNKNOWN:
        return String("memory-budget-unknown")
    if code == WARN_COLD_SESSION:
        return String("cold-session")
    if code == WARN_UNPROVEN_TRANSFER_ROUTE:
        return String("unproven-transfer-route")
    if code == WARN_KERNEL_RETIREMENT_ROUTE:
        return String("kernel-retirement-route")
    return String("unknown-warning")


# --- How the decision was reached -------------------------------------

comptime DECISION_EXPLICIT_CPU = 0
comptime DECISION_EXPLICIT_GPU = 1
comptime DECISION_GPU_REFUSED = 2
comptime DECISION_AUTO_CPU_BLOCKED = 3
comptime DECISION_AUTO_CPU_NO_EVIDENCE = 4
comptime DECISION_AUTO_CPU_BELOW_EVIDENCE = 5
comptime DECISION_AUTO_CPU_BELOW_ENV_THRESHOLD = 6
comptime DECISION_AUTO_GPU_EVIDENCE = 7
comptime DECISION_AUTO_GPU_ENV_THRESHOLD = 8


def decision_name(code: Int) -> String:
    if code == DECISION_EXPLICIT_CPU:
        return String("explicit-cpu")
    if code == DECISION_EXPLICIT_GPU:
        return String("explicit-gpu")
    if code == DECISION_GPU_REFUSED:
        return String("gpu-refused")
    if code == DECISION_AUTO_CPU_BLOCKED:
        return String("auto-cpu-blocked")
    if code == DECISION_AUTO_CPU_NO_EVIDENCE:
        return String("auto-cpu-no-evidence")
    if code == DECISION_AUTO_CPU_BELOW_EVIDENCE:
        return String("auto-cpu-below-evidence")
    if code == DECISION_AUTO_CPU_BELOW_ENV_THRESHOLD:
        return String("auto-cpu-below-env-threshold")
    if code == DECISION_AUTO_GPU_EVIDENCE:
        return String("auto-gpu-evidence")
    if code == DECISION_AUTO_GPU_ENV_THRESHOLD:
        return String("auto-gpu-env-threshold")
    return String("unknown-decision")


# --- Vocabulary -------------------------------------------------------


def parse_device(name: String) raises -> Int:
    """Device code for a public device name ("cpu", "gpu", or "auto").

    Names are canonical lowercase here. The Python wrapper lowercases what
    the user passes before calling in, which is how LightGBM treats
    `device_type`."""
    if name == "cpu":
        return CPU_DEVICE
    if name == "gpu":
        return GPU_DEVICE
    if name == "auto":
        return AUTO_DEVICE
    raise Error(
        "unknown device '", name, "'; expected 'cpu', 'gpu', or 'auto'"
    )


def device_name(device: Int) raises -> String:
    """Public device name for a device code."""
    if device == CPU_DEVICE:
        return String("cpu")
    if device == GPU_DEVICE:
        return String("gpu")
    if device == AUTO_DEVICE:
        return String("auto")
    if device == NO_DEVICE:
        return String("none")
    raise Error("unknown device code ", device)


# --- What the GPU training path supports ------------------------------


def is_builtin_objective(objective: Int) -> Bool:
    """Whether `objective` is one of the built-in single-output objectives
    the boosted trainers implement.

    The same membership `_check_objective` in boosting.mojo tests before it
    raises "unknown objective", expressed as a predicate so the device
    policy can answer without raising. `CUSTOM` is excluded (its gradients
    come from a caller-supplied callable) and so is `LAMBDARANK` (its
    gradients come from query groups) and `MULTICLASS` (one tree per class
    per round, through its own trainer).

    Delegates to `objective_is_builtin` in objective_registry.mojo, which is
    the one table of objective facts. Kept as a name here because it is the
    spelling the gates below read and because a device question should not
    have to know which module the answer lives in; it holds no list of its
    own."""
    return objective_is_builtin(objective)


def gpu_trains_objective(objective: Int) -> Bool:
    """Whether `train_gpu` covers this objective.

    Every built-in objective, which is what `train_gpu` accepts: it runs
    the same `_check_objective` the CPU trainer does and then grows trees
    on the device.

    Deliberately narrower than "does any GPU trainer exist for this". It
    does not for `LAMBDARANK`, which is CPU only; it does for `CUSTOM`
    (`train_custom_gpu`) and for multiclass (`train_multiclass_gpu`), but
    neither is reachable through the `device` setting, because each is its
    own entry point. This predicate answers what the device vocabulary can
    route, which is the only question this module is asked.
    `objective_backends` in objective_registry.mojo answers the wider one.

    `OBJECTIVE_UNSPECIFIED` answers True: a caller that did not name an
    objective is not asserting an unsupported one, and the decision carries
    `WARN_INCOMPLETE_REQUEST` so the gap is visible rather than silently
    resolved either way."""
    if objective == OBJECTIVE_UNSPECIFIED:
        return True
    return is_builtin_objective(objective)


def gpu_objective_is_device_resident(objective: Int) -> Bool:
    """Whether this objective's gradients are generated on the device.

    The objectives with a closed-form per-row derivative the kernels in
    gpu_objectives_native.mojo implement, which today is every built-in
    one. When it is False, `train_gpu` still trains the objective, but it
    fills gradients on the host and uploads them once per round, which is
    slower and worth reporting; that report is `WARN_HOST_GRADIENT_PATH`,
    never a block.

    Delegates to `objective_gradients_on_device` in objective_registry.mojo,
    which is the one table of objective facts and which
    `gpu_objectives_native.supports_device_objective` already defers to. The
    three predicates therefore now agree by construction rather than by
    three lists being edited together.

    `OBJECTIVE_UNSPECIFIED` answers True for the same reason
    `gpu_trains_objective` does: a caller that did not name an objective is
    not asserting one without a kernel, and the gap is reported through
    `WARN_INCOMPLETE_REQUEST` rather than resolved silently either way."""
    if objective == OBJECTIVE_UNSPECIFIED:
        return True
    return objective_gradients_on_device(objective)


def gpu_supports_outputs(n_outputs: Int) -> Bool:
    """Whether the complete GPU training path covers this many trees per
    boosting round. `n_outputs` is 1 for single-output training and for
    binary classification, and the class count beyond that.

    Every workload the device vocabulary routes is covered: multiclass
    grows one tree per class per round through `train_multiclass_gpu`, on
    the same device-resident builder the single-output trainer uses. The
    check stays as the one place to reject a future workload the GPU path
    does not implement, which is why the engine still consults it rather
    than assuming coverage."""
    return n_outputs >= 1


# --- Environment ------------------------------------------------------


def build_has_accelerator() -> Bool:
    """Whether an accelerator was present when this build was compiled.

    Mojo resolves `has_accelerator()` at compile time, so this is a
    property of the binary, not of the machine running it. See the module
    docstring on redistributed builds."""
    comptime if has_accelerator():
        return True
    else:
        return False


def gpu_disabled_by_env() -> Bool:
    """Whether `MOJOTREES_DISABLE_GPU=1` pins this process to the CPU."""
    return getenv("MOJOTREES_DISABLE_GPU") == "1"


def gpu_available() -> Bool:
    """True when training can run on an accelerator: one was present when
    this build was compiled and `MOJOTREES_DISABLE_GPU=1` is not set."""
    if not build_has_accelerator():
        return False
    return not gpu_disabled_by_env()


def env_auto_min_cells() -> Int:
    """The `auto` size threshold in cells. Unset, negative, or unparsable
    means disabled, in which case `auto` never selects the GPU on size
    alone."""
    var s = getenv("MOJOTREES_AUTO_MIN_CELLS")
    if s.byte_length() == 0:
        return AUTO_MIN_CELLS
    try:
        return Int(s)
    except:
        return AUTO_MIN_CELLS


def env_declared_api() -> Int:
    """The GPU API an operator named through `MOJOTREES_GPU_BACKEND`, or
    `API_UNKNOWN`.

    A declaration, not a detection: it names the API for reporting and for
    scoping a crossover rule, and it never supplies a capability number.
    Nothing here reads an operating system name or a marketing chip string
    to guess at hardware."""
    var s = getenv("MOJOTREES_GPU_BACKEND")
    if s.byte_length() == 0:
        return API_UNKNOWN
    return parse_api(s)


def _bool_text(value: Bool) -> String:
    """How a Bool is spelled in the serialized decision."""
    if value:
        return String("true")
    return String("false")


# --- Reason lists -----------------------------------------------------


struct ReasonList(Copyable, Movable):
    """An ordered list of (stable code, prose) pairs.

    Two parallel lists rather than a list of pairs so the serialized form
    stays flat and a consumer that only matches on codes never has to parse
    the prose."""

    var codes: List[Int]
    var messages: List[String]

    def __init__(out self):
        self.codes = List[Int]()
        self.messages = List[String]()

    def add(mut self, code: Int, var message: String):
        self.codes.append(code)
        self.messages.append(message^)

    def count(self) -> Int:
        return len(self.codes)

    def is_empty(self) -> Bool:
        return len(self.codes) == 0

    def first_message(self) raises -> String:
        if len(self.messages) == 0:
            raise Error("no reasons recorded")
        return self.messages[0].copy()


# --- Workload and request ---------------------------------------------


struct DeviceRequest(Copyable, Movable):
    """One training run, as the policy needs to see it.

    Every field is plain data. Python builds one of these by reading `X`
    and `y`, which is inspection, not policy: nothing here needs the
    dataset itself, and nothing about the decision depends on values it
    could not serialize.

    - `requested_device`: `CPU_DEVICE`, `GPU_DEVICE`, or `AUTO_DEVICE`.
    - `n_rows`, `n_features`: the shape of the training matrix.
    - `n_outputs`: trees grown per boosting round, 1 for single-output
      training and for binary classification, the class count beyond that.
    - `n_bins`: the estimator's `max_bin`, or `BINS_UNSPECIFIED`.
    - `objective`: a boosting.mojo objective code, or
      `OBJECTIVE_UNSPECIFIED`.
    - `sparse`: the input is a sparse matrix. There is no sparse GPU
      kernel, so this is a hard block.
    - `categorical`: the run declares categorical features. The GPU
      grower routes them (`split.is_categorical` in train_gpu.mojo), so
      this is reported, not blocked.
    - `has_missing`: the run uses LightGBM's `use_missing` handling. The
      GPU grower carries a missing bin per feature, so this is reported,
      not blocked.
    - `uses_validation`: the run has an eval set. Validation metrics are
      scored on the CPU, so a run with one trains there too.
    """

    var requested_device: Int
    var n_rows: Int
    var n_features: Int
    var n_outputs: Int
    var n_bins: Int
    var objective: Int
    var sparse: Bool
    var categorical: Bool
    var has_missing: Bool
    var uses_validation: Bool

    def __init__(
        out self,
        requested_device: Int,
        n_rows: Int,
        n_features: Int,
        n_outputs: Int = 1,
        n_bins: Int = BINS_UNSPECIFIED,
        objective: Int = OBJECTIVE_UNSPECIFIED,
        sparse: Bool = False,
        categorical: Bool = False,
        has_missing: Bool = False,
        uses_validation: Bool = False,
    ):
        self.requested_device = requested_device
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_bins = n_bins
        self.objective = objective
        self.sparse = sparse
        self.categorical = categorical
        self.has_missing = has_missing
        self.uses_validation = uses_validation

    def cells(self) -> Int:
        """`n_rows * n_features`, the size measure the crossover rules and
        `MOJOTREES_AUTO_MIN_CELLS` are written in."""
        return self.n_rows * self.n_features

    def bins_known(self) -> Bool:
        return self.n_bins != BINS_UNSPECIFIED

    def objective_known(self) -> Bool:
        return self.objective != OBJECTIVE_UNSPECIFIED

    def is_complete(self) -> Bool:
        """Whether the caller declared everything the policy can gate on.
        An incomplete request still gets a decision; it gets
        `WARN_INCOMPLETE_REQUEST` with it."""
        return self.bins_known() and self.objective_known()


# --- Capabilities -----------------------------------------------------


struct DeviceCapabilities(Copyable, Movable):
    """What the build and the device can do, as one serializable value.

    Never a probe once constructed, so a caller can inject a machine it
    does not have and the engine cannot tell the difference. That is what
    makes every gate below testable without an accelerator.

    - `gpu_available`: training can run on an accelerator.
    - `built_with_accelerator`: an accelerator was present at compile
      time, which is a different question (see the module docstring).
    - `disabled_by_env`: `MOJOTREES_DISABLE_GPU=1` was set.
    - `profile`: the hardware capabilities, from apple_gpu_policy.mojo.
    - `profile_source`: one of the `PROFILE_*` codes, which is how a
      reader tells a reading from a fallback.
    - `max_rows`, `min_bins`, `max_bins`: the kernel and binner limits.
      Fields rather than constants so a build that widens one can say so.
    - `auto_min_cells`: the `MOJOTREES_AUTO_MIN_CELLS` value in effect.
      Negative means the heuristic is off.
    - `session`: how much of the one-time startup cost this process has
      already paid, from initialization.mojo. Reported and warned on, never
      selected on: see `SessionState`.
    - `transfer`: the per-role transfer routes in effect, from
      unified_memory_policy.mojo. The staged copy for every role in every
      context this repository controls.

    The last two are why the environment is read *here* and not in
    `decide_device`: both of them fold an environment variable into a value,
    and keeping every such read on this side of the boundary is what lets
    `decide_device` stay pure and injectable.
    """

    var gpu_available: Bool
    var built_with_accelerator: Bool
    var disabled_by_env: Bool
    var profile: GpuProfile
    var profile_source: Int
    var max_rows: Int
    var min_bins: Int
    var max_bins: Int
    var auto_min_cells: Int
    var session: SessionState
    var transfer: SessionMemoryPlan

    def __init__(
        out self,
        gpu_available: Bool,
        built_with_accelerator: Bool,
        disabled_by_env: Bool,
        var profile: GpuProfile,
        profile_source: Int,
        auto_min_cells: Int,
        var session: SessionState,
        var transfer: SessionMemoryPlan,
        max_rows: Int = MAX_GPU_ROWS,
        min_bins: Int = MIN_GPU_BINS,
        max_bins: Int = MAX_GPU_BINS,
    ):
        self.gpu_available = gpu_available
        self.built_with_accelerator = built_with_accelerator
        self.disabled_by_env = disabled_by_env
        self.profile = profile^
        self.profile_source = profile_source
        self.max_rows = max_rows
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.auto_min_cells = auto_min_cells
        self.session = session^
        self.transfer = transfer^

    @staticmethod
    def detect(
        var session: SessionState = SessionState.cold(),
    ) raises -> DeviceCapabilities:
        """Capabilities for the build and process running right now.

        Opens no device. The hardware profile is therefore the portable
        fallback (`PROFILE_FALLBACK`), or `PROFILE_DECLARED` when an
        operator named the API through `MOJOTREES_GPU_BACKEND`. A caller
        that already has a `DeviceContext` open should read its attributes
        and call `from_profile` instead, which is the only way to reach
        `PROFILE_REPORTED`, and should hand its own `SessionState` rather
        than taking the cold default.

        Raises for an unparsable `MOJOTREES_GPU_TRANSFER`, which is
        `plan_session_routes`' rule and the same one this module applies to
        `device`: a misspelled knob must not resolve to a different path.
        """
        var built = build_has_accelerator()
        var disabled = gpu_disabled_by_env()
        var available = built and not disabled
        var declared = env_declared_api()
        var profile = GpuProfile.generic()
        var source: Int = PROFILE_NONE
        if available:
            source = PROFILE_FALLBACK
            if declared != API_UNKNOWN:
                source = PROFILE_DECLARED
                profile = GpuProfile(
                    declared,
                    profile.apple_generation,
                    profile.core_count,
                    profile.max_threads_per_block,
                    profile.max_shared_memory_per_block,
                    profile.memory_budget_bytes,
                    profile.unified_memory,
                    profile.synthetic,
                )
        var transfer = plan_session_routes(profile.unified_memory)
        return DeviceCapabilities(
            available,
            built,
            disabled,
            profile^,
            source,
            env_auto_min_cells(),
            session^,
            transfer^,
        )

    @staticmethod
    def from_profile(
        var profile: GpuProfile,
        profile_source: Int = PROFILE_REPORTED,
        var session: SessionState = SessionState.cold(),
    ) raises -> DeviceCapabilities:
        """Capabilities carrying a profile a caller already has.

        `PROFILE_REPORTED` is the default because the intended caller is
        one that read the attributes off an open `DeviceContext`. Pass
        `PROFILE_SYNTHETIC` for a fixture; the source is recorded and
        surfaces as `WARN_SYNTHETIC_CAPABILITIES`, and it never changes a
        gate.

        That same caller is the one that can answer `session` truthfully: it
        holds the context, so `context_open` is True for it and cold for
        everybody else. The default stays cold because guessing warm on
        behalf of a caller that did not say so reports a paid cost that was
        not paid.
        """
        var built = build_has_accelerator()
        var disabled = gpu_disabled_by_env()
        var transfer = plan_session_routes(profile.unified_memory)
        return DeviceCapabilities(
            built and not disabled,
            built,
            disabled,
            profile^,
            profile_source,
            env_auto_min_cells(),
            session^,
            transfer^,
        )

    @staticmethod
    def unavailable() raises -> DeviceCapabilities:
        """A machine with no accelerator. The conservative answer, and the
        one an injected fixture wants when it is testing the CPU path.

        Reads no environment at all, including the transfer knob: a fixture
        asking what the CPU path does should not be affected by a variable
        about device buffers.
        """
        return DeviceCapabilities(
            False,
            False,
            False,
            GpuProfile.generic(),
            PROFILE_NONE,
            AUTO_MIN_CELLS,
            SessionState.cold(),
            SessionMemoryPlan.staged(),
        )

    def memory_budget_known(self) -> Bool:
        return self.profile.memory_budget_bytes > 0


# --- Memory estimate --------------------------------------------------


@fieldwise_init
struct MemoryEstimate(Copyable, Movable):
    """What one GPU training session is estimated to allocate.

    One term per buffer `GpuHistogramBuilder.__init__` creates in
    histogram_gpu.mojo:

        binned matrix     n_rows * n_features * 1     uint8
        leaf ids          n_rows * 4                  int32
        gradients         n_rows * 4 * n_outputs      float32
        hessians          n_rows * 4 * n_outputs      float32
        histograms        n_features * n_bins * 12    3 int32 planes
        feature ids       n_features * 4              int32

    plus, on the host, two pinned float32 staging planes of `n_rows` and
    one pinned copy of the histogram buffer.

    Invariants. No test asserts them today; the handoff specifies the one
    that should, and marks it unwritten. They are stated here because the
    memory gate below compares this estimate against a device budget, and
    invariant 4 in particular is what makes that comparison sound:

    1. Every term is nonnegative.
    2. `device_bytes()` is the sum of the six device terms and nothing
       else, and `host_bytes()` the sum of the two host terms.
    3. `upper_bound_bytes()` equals `device_bytes() + partial_budget`, so
       it is never below `device_bytes()`.
    4. The estimate is nondecreasing in each of `n_rows`, `n_features`,
       `n_outputs`, and `n_bins`. No term may ever shrink as a workload
       grows, because the memory gate compares against a budget and a
       shrinking estimate would admit a run that does not fit.
    5. `bins_known` False means the histogram terms are zero rather than
       guessed. An estimate with `bins_known` False must never be used to
       block a run, because it is a lower bound on an unknown quantity.

    This is an estimate and it is labeled one everywhere it appears. It
    counts the training buffers, not the allocator's own overhead, and the
    `n_outputs` factor on the gradient planes is an upper bound that
    assumes every class plane is resident at once.

    The tiled accumulation strategy also allocates a partial-histogram
    buffer whose size depends on device attributes. `partial_budget_bytes`
    in apple_gpu_policy.mojo derives the ceiling from the reported memory
    budget, tighter when memory is unified, capped by the portable 64 MiB
    ceiling, and that ceiling is what `upper_bound_bytes()` adds.
    """

    var binned_matrix_bytes: Int
    var leaf_id_bytes: Int
    var gradient_bytes: Int
    var hessian_bytes: Int
    var histogram_bytes: Int
    var feature_id_bytes: Int
    var host_staging_bytes: Int
    var host_readback_bytes: Int
    var partial_budget: Int
    var bins_known: Bool

    def device_bytes(self) -> Int:
        """Device allocations excluding the tiled partial buffer."""
        return (
            self.binned_matrix_bytes
            + self.leaf_id_bytes
            + self.gradient_bytes
            + self.hessian_bytes
            + self.histogram_bytes
            + self.feature_id_bytes
        )

    def upper_bound_bytes(self) -> Int:
        """Device allocations with the partial-histogram budget included."""
        return self.device_bytes() + self.partial_budget

    def host_bytes(self) -> Int:
        """Pinned host staging buffers."""
        return self.host_staging_bytes + self.host_readback_bytes


def estimate_gpu_memory(
    request: DeviceRequest, profile: GpuProfile
) raises -> MemoryEstimate:
    """The `MemoryEstimate` for one request on one device.

    See `MemoryEstimate` for the terms and the invariants. Raises for a
    shape with no rows or no features, which is not a workload."""
    if request.n_rows < 1 or request.n_features < 1:
        raise Error(
            "a workload needs at least one row and one feature; got n_rows=",
            request.n_rows,
            ", n_features=",
            request.n_features,
        )
    var planes = request.n_outputs
    if planes < 1:
        planes = 1
    var histogram_bytes: Int = 0
    var readback_bytes: Int = 0
    if request.bins_known():
        var cells = request.n_features * request.n_bins
        histogram_bytes = cells * BYTES_PER_PARTIAL_CELL
        readback_bytes = histogram_bytes
    return MemoryEstimate(
        request.n_rows * request.n_features,
        request.n_rows * 4,
        request.n_rows * 4 * planes,
        request.n_rows * 4 * planes,
        histogram_bytes,
        request.n_features * 4,
        request.n_rows * 8,
        readback_bytes,
        partial_budget_bytes(profile),
        request.bins_known(),
    )


# --- Crossover evidence -----------------------------------------------


struct CrossoverEvidence(Copyable, Movable):
    """One benchmark-derived rule saying "the GPU wins from here up".

    A rule is a claim about measured performance, so it carries the
    measurement with it. `evidence_id` cites where the numbers live (a
    document section, a benchmark file, a commit) and is what a decision
    reports; `measured_on` names the device they came from. A rule without
    an evidence identifier is not a rule, and `crossover_rules()` is empty
    for exactly that reason.

    Scope narrows a rule to what was actually measured. `api` and
    `apple_generation` limit it to one device family or generation,
    `objective` to one objective, and `min_rows`, `min_features`, and
    `min_cells` are the thresholds themselves. `API_UNKNOWN`,
    `OBJECTIVE_UNSPECIFIED`, and a zero minimum do not constrain. A rule
    matches only when every set field matches, so widening a rule to
    hardware nobody measured takes a deliberate edit.
    """

    var name: String
    var evidence_id: String
    var measured_on: String
    var api: Int
    var apple_generation: Int
    var objective: Int
    var min_rows: Int
    var min_features: Int
    var min_cells: Int
    var max_outputs: Int
    """Trees per round the measurement covered. Zero does not constrain."""

    def __init__(
        out self,
        var name: String,
        var evidence_id: String,
        var measured_on: String,
        api: Int = API_UNKNOWN,
        apple_generation: Int = 0,
        objective: Int = OBJECTIVE_UNSPECIFIED,
        min_rows: Int = 0,
        min_features: Int = 0,
        min_cells: Int = 0,
        max_outputs: Int = 0,
    ) raises:
        if evidence_id.byte_length() == 0:
            raise Error(
                "a crossover rule needs an evidence identifier; cite the"
                " benchmark that measured it"
            )
        self.name = name^
        self.evidence_id = evidence_id^
        self.measured_on = measured_on^
        self.api = api
        self.apple_generation = apple_generation
        self.objective = objective
        self.min_rows = min_rows
        self.min_features = min_features
        self.min_cells = min_cells
        self.max_outputs = max_outputs

    def matches(
        self, caps: DeviceCapabilities, request: DeviceRequest
    ) -> Bool:
        """Whether this rule covers the (device, workload) pair."""
        if self.api != API_UNKNOWN and caps.profile.api != self.api:
            return False
        if (
            self.apple_generation != 0
            and caps.profile.apple_generation != self.apple_generation
        ):
            return False
        if (
            self.objective != OBJECTIVE_UNSPECIFIED
            and request.objective != self.objective
        ):
            return False
        if request.n_rows < self.min_rows:
            return False
        if request.n_features < self.min_features:
            return False
        if request.cells() < self.min_cells:
            return False
        if self.max_outputs != 0 and request.n_outputs > self.max_outputs:
            return False
        return True


def crossover_rules() raises -> List[CrossoverEvidence]:
    """The benchmark-derived crossover rules, in priority order.

    Empty, and the module docstring says why: no measurement in this
    repository has found a shape where GPU training beats CPU training,
    and the only device that has ever run the GPU trainer end to end came
    out slower. Do not add a rule from reasoning. Add one from a recorded
    sweep, cite it in `evidence_id`, and bump `POLICY_VERSION`.
    """
    return List[CrossoverEvidence]()


# --- The engine -------------------------------------------------------


def _collect_blocks(
    request: DeviceRequest,
    caps: DeviceCapabilities,
    memory: MemoryEstimate,
) raises -> ReasonList:
    """Everything that will actually fail if this workload goes to the GPU.

    Ordered from the cheapest and most fundamental to the most
    workload-specific, so the first entry is the one worth putting in a
    refusal message. Availability short-circuits: on a machine with no
    accelerator, nothing else about the workload matters.
    """
    var blocks = ReasonList()

    if not caps.gpu_available:
        if caps.disabled_by_env:
            blocks.add(
                BLOCK_GPU_DISABLED_ENV,
                String(
                    "MOJOTREES_DISABLE_GPU=1 pins this process to the CPU"
                    " backend, so no accelerator is available"
                ),
            )
        else:
            blocks.add(
                BLOCK_NO_ACCELERATOR,
                String("no accelerator is available to this build"),
            )
        return blocks^

    # An impossible shape is not a block. `estimate_gpu_memory` raises for
    # it before this function is reached, because a workload with no rows
    # or no features is a caller error and not something the CPU path
    # would have run either.

    if request.sparse:
        blocks.add(
            BLOCK_SPARSE_INPUT,
            String(
                "sparse input trains on the CPU; there is no sparse GPU"
                " histogram kernel"
            ),
        )

    if request.objective == CUSTOM:
        # `train_custom_gpu` does exist, and it grows trees on the device
        # while calling the gradient callback on the host. The `device`
        # vocabulary does not route to it: `fit` and `fit_multiclass` send
        # a custom objective to `train_custom`, and `train_gpu` itself
        # rejects the code. So this block is about the path the request
        # will actually take, not a claim that no GPU custom trainer
        # exists. Reaching that trainer is an explicit call, not a device
        # setting.
        blocks.add(
            BLOCK_CUSTOM_OBJECTIVE,
            String(
                "a custom objective's gradients come from a caller-supplied"
                " callable, and the device setting routes it to train_custom"
                " on the CPU; call train_custom_gpu directly to grow its"
                " trees on the device"
            ),
        )
    elif request.objective == LAMBDARANK:
        blocks.add(
            BLOCK_RANKING_OBJECTIVE,
            String(
                "objective 'lambdarank' takes its gradients from query"
                " groups and trains on the CPU only"
            ),
        )
    elif not gpu_trains_objective(request.objective):
        blocks.add(
            BLOCK_UNKNOWN_OBJECTIVE,
            String(
                "objective code ",
                request.objective,
                " is not one the built-in trainers implement, so no backend"
                " covers it",
            ),
        )

    if request.uses_validation:
        blocks.add(
            BLOCK_VALIDATION_SET,
            String(
                "validation metrics are scored on the CPU, so a run with an"
                " eval set trains there too"
            ),
        )

    if not gpu_supports_outputs(request.n_outputs):
        blocks.add(
            BLOCK_OUTPUT_LIMIT,
            String(
                "the GPU path does not cover ",
                request.n_outputs,
                " trees per boosting round",
            ),
        )

    if request.n_rows > caps.max_rows:
        blocks.add(
            BLOCK_ROW_LIMIT,
            String(
                request.n_rows,
                " rows is past the ",
                caps.max_rows,
                " the GPU kernels can index",
            ),
        )

    if request.bins_known() and (
        request.n_bins < caps.min_bins or request.n_bins > caps.max_bins
    ):
        blocks.add(
            BLOCK_BIN_LIMIT,
            String(
                "max_bin=",
                request.n_bins,
                " is outside the [",
                caps.min_bins,
                ", ",
                caps.max_bins,
                "] the binner and the GPU histogram kernels support",
            ),
        )

    # Invariant 5 on `MemoryEstimate`: an estimate built without a bin
    # count is a lower bound on an unknown quantity, so it may never
    # block. A budget of zero means unreported, which is not a budget.
    if caps.memory_budget_known() and memory.bins_known:
        if memory.device_bytes() > caps.profile.memory_budget_bytes:
            blocks.add(
                BLOCK_MEMORY_BUDGET,
                String(
                    "the estimated ",
                    memory.device_bytes(),
                    " bytes of training buffers does not fit the reported ",
                    caps.profile.memory_budget_bytes,
                    " byte budget",
                ),
            )

    return blocks^


def _collect_warnings(
    request: DeviceRequest, caps: DeviceCapabilities
) raises -> ReasonList:
    """Everything a reader should know about how much this decision is
    worth. Nothing here changes the selected backend.

    Everything below the availability check is about the accelerator, so
    an explicit `cpu` request collects none of it: a machine that happens
    to have a GPU should not narrate its capabilities at every CPU run.
    The incomplete-request warning is the exception, because a caller that
    under-described its workload wants to know that whatever device it got.
    """
    var warnings = ReasonList()

    if not request.is_complete():
        var missing = String("")
        if not request.objective_known():
            missing += String(" an objective")
        if not request.bins_known():
            if missing.byte_length() > 0:
                missing += String(" and")
            missing += String(" a bin count")
        warnings.add(
            WARN_INCOMPLETE_REQUEST,
            String(
                "the caller did not declare",
                missing,
                ", so the gates that depend on it were skipped and the"
                " memory estimate is partial",
            ),
        )

    if not caps.gpu_available or request.requested_device == CPU_DEVICE:
        return warnings^

    if caps.built_with_accelerator:
        warnings.add(
            WARN_BUILD_TIME_AVAILABILITY,
            String(
                "accelerator availability is resolved at compile time, so a"
                " redistributed build reports the device its builder had;"
                " set MOJOTREES_DISABLE_GPU=1 to pin such a build to the CPU"
            ),
        )

    if caps.profile_source == PROFILE_FALLBACK:
        warnings.add(
            WARN_UNKNOWN_HARDWARE,
            String(
                "no device attributes were read, so the conservative"
                " portable profile is in use and no crossover rule can be"
                " scoped to this hardware"
            ),
        )
    elif caps.profile_source == PROFILE_DECLARED:
        warnings.add(
            WARN_UNKNOWN_HARDWARE,
            String(
                "MOJOTREES_GPU_BACKEND declared api '",
                api_name(caps.profile.api),
                "' but no device attributes were read, so the capability"
                " numbers are still the portable fallback",
            ),
        )
    elif caps.profile_source == PROFILE_SYNTHETIC or caps.profile.synthetic:
        warnings.add(
            WARN_SYNTHETIC_CAPABILITIES,
            String(
                "these capabilities were constructed rather than read from a"
                " device, so they describe a fixture and not the hardware in"
                " front of you"
            ),
        )

    if not caps.memory_budget_known():
        warnings.add(
            WARN_MEMORY_BUDGET_UNKNOWN,
            String(
                "no device memory budget was reported, so memory cannot"
                " block this run and the partial-histogram budget falls back"
                " to the portable ceiling"
            ),
        )
    elif caps.profile.unified_memory:
        warnings.add(
            WARN_UNIFIED_MEMORY_BUDGET,
            String(
                "device memory is host memory on this backend, so the budget"
                " shown is shared with the dataset the host is holding"
            ),
        )

    if request.objective_known() and (
        not gpu_objective_is_device_resident(request.objective)
    ):
        warnings.add(
            WARN_HOST_GRADIENT_PATH,
            String(
                "objective code ",
                request.objective,
                " has no device-side derivative kernel, so its gradients are"
                " filled on the host and uploaded once per boosting round",
            ),
        )

    if caps.session.paid_nothing():
        warnings.add(
            WARN_COLD_SESSION,
            String(
                "no device context or kernel has been created in this"
                " process, so a GPU run here also pays context creation and"
                " first-launch kernel creation; that cost is startup, not"
                " training, and comparing it against a warm CPU run measures"
                " the wrong thing (warm-up level ",
                warmup_level_name(caps.session.warmup_level),
                ")",
            ),
        )

    # The transfer plan never blocks. A role that cannot take the requested
    # route falls back to the staged copy inside unified_memory_policy.mojo,
    # and the allocation site is what refuses; what belongs in a device
    # decision is that the run is not on the shipped route.
    if caps.transfer.any_unproven():
        warnings.add(
            WARN_UNPROVEN_TRANSFER_ROUTE,
            String(
                "MOJOTREES_GPU_TRANSFER_UNPROVEN=1 put at least one buffer on"
                " transfer route '",
                transfer_route_name(caps.transfer.requested),
                "', which has no installed evidence; any number measured"
                " under this must be reported with the flag",
            ),
        )
    if caps.transfer.needs_kernel_retirement():
        warnings.add(
            WARN_KERNEL_RETIREMENT_ROUTE,
            String(
                "at least one buffer is on a route the kernels read directly,"
                " so the host cannot refill it until every kernel that read"
                " it has retired and the staging ring's overlap does not"
                " apply; this is a synchronization change, not only an"
                " allocation one",
            ),
        )

    return warnings^


struct DeviceDecision(Copyable, Movable):
    """The decision, everything it rested on, and how to serialize it.

    `selected_device` is `CPU_DEVICE` or `GPU_DEVICE` when a backend was
    chosen, and `NO_DEVICE` when the request cannot run, in which case
    `blocked` is True and `message` holds what `raise_if_blocked` raises.

    `blocking_reasons` holds the reasons the GPU path is unavailable for
    this workload, whatever was requested. It is populated even when
    `cpu` was asked for and the answer was never in doubt, because "why
    would the GPU not have worked here" is a question a report should be
    able to answer without being asked twice.
    """

    var request: DeviceRequest
    var capabilities: DeviceCapabilities
    var selected_device: Int
    var blocked: Bool
    var decision_code: Int
    var message: String
    var blocking_reasons: ReasonList
    var warnings: ReasonList
    var memory: MemoryEstimate
    var policy_version: Int
    var evidence_id: String

    def __init__(
        out self,
        var request: DeviceRequest,
        var capabilities: DeviceCapabilities,
        selected_device: Int,
        blocked: Bool,
        decision_code: Int,
        var message: String,
        var blocking_reasons: ReasonList,
        var warnings: ReasonList,
        var memory: MemoryEstimate,
        var evidence_id: String,
    ):
        self.request = request^
        self.capabilities = capabilities^
        self.selected_device = selected_device
        self.blocked = blocked
        self.decision_code = decision_code
        self.message = message^
        self.blocking_reasons = blocking_reasons^
        self.warnings = warnings^
        self.memory = memory^
        self.policy_version = POLICY_VERSION
        self.evidence_id = evidence_id^

    def validated(self) -> Bool:
        """Whether a GPU selection rests on benchmark-derived evidence.
        False for every CPU selection, and False for a GPU reached through
        `MOJOTREES_AUTO_MIN_CELLS` or requested explicitly."""
        if self.selected_device != GPU_DEVICE:
            return False
        return self.decision_code == DECISION_AUTO_GPU_EVIDENCE

    def raise_if_blocked(self) raises:
        """Raise when the request cannot run, and return otherwise. This
        is what turns a report into the refusal an explicit `gpu` gets."""
        if self.blocked:
            raise Error(self.message)

    def serialize(self) raises -> String:
        """The decision as `key=value` lines.

        The wire format between this module and any formatter, Python
        included. One key per line, `=` separating key from value, values
        free of newlines. Repeated keys are lists in order: `block` for
        each blocking reason, `warning` for each warning, each rendered as
        `<code>:<name>:<message>`.

        A consumer that only cares about the answer reads `selected`.
        A consumer that wants to explain it reads the rest.
        """
        var out = String("policy_version=", self.policy_version, "\n")
        out += String("evidence_id=", self.evidence_id, "\n")
        out += String(
            "requested=", device_name(self.request.requested_device), "\n"
        )
        out += String("selected=", device_name(self.selected_device), "\n")
        out += String("blocked=", _bool_text(self.blocked), "\n")
        out += String(
            "decision=", decision_name(self.decision_code), "\n"
        )
        out += String("message=", self.message, "\n")
        out += String("validated=", _bool_text(self.validated()), "\n")

        out += String("n_rows=", self.request.n_rows, "\n")
        out += String("n_features=", self.request.n_features, "\n")
        out += String("n_outputs=", self.request.n_outputs, "\n")
        out += String("cells=", self.request.cells(), "\n")
        out += String("n_bins=", self.request.n_bins, "\n")
        out += String(
            "bins_known=", _bool_text(self.request.bins_known()), "\n"
        )
        out += String("objective=", self.request.objective, "\n")
        out += String(
            "objective_known=",
            _bool_text(self.request.objective_known()),
            "\n",
        )
        out += String("sparse=", _bool_text(self.request.sparse), "\n")
        out += String(
            "categorical=", _bool_text(self.request.categorical), "\n"
        )
        out += String(
            "has_missing=", _bool_text(self.request.has_missing), "\n"
        )
        out += String(
            "uses_validation=",
            _bool_text(self.request.uses_validation),
            "\n",
        )

        out += String(
            "gpu_available=",
            _bool_text(self.capabilities.gpu_available),
            "\n",
        )
        out += String(
            "built_with_accelerator=",
            _bool_text(self.capabilities.built_with_accelerator),
            "\n",
        )
        out += String(
            "disabled_by_env=",
            _bool_text(self.capabilities.disabled_by_env),
            "\n",
        )
        out += String(
            "profile_source=",
            profile_source_name(self.capabilities.profile_source),
            "\n",
        )
        out += String("api=", api_name(self.capabilities.profile.api), "\n")
        out += String(
            "apple_generation=",
            apple_generation_name(
                self.capabilities.profile.apple_generation
            ),
            "\n",
        )
        out += String(
            "core_count=", self.capabilities.profile.core_count, "\n"
        )
        out += String(
            "max_threads_per_block=",
            self.capabilities.profile.max_threads_per_block,
            "\n",
        )
        out += String(
            "max_shared_memory_per_block=",
            self.capabilities.profile.max_shared_memory_per_block,
            "\n",
        )
        out += String(
            "memory_budget_bytes=",
            self.capabilities.profile.memory_budget_bytes,
            "\n",
        )
        out += String(
            "unified_memory=",
            _bool_text(self.capabilities.profile.unified_memory),
            "\n",
        )
        out += String("max_rows=", self.capabilities.max_rows, "\n")
        out += String("min_bins=", self.capabilities.min_bins, "\n")
        out += String("max_bins=", self.capabilities.max_bins, "\n")
        out += String(
            "auto_min_cells=", self.capabilities.auto_min_cells, "\n"
        )

        out += String(
            "session_context_open=",
            _bool_text(self.capabilities.session.context_open),
            "\n",
        )
        out += String(
            "session_kernels_ready=",
            _bool_text(self.capabilities.session.kernels_ready),
            "\n",
        )
        out += String(
            "session_warmup_level=",
            warmup_level_name(self.capabilities.session.warmup_level),
            "\n",
        )

        out += String(
            "transfer_requested=",
            transfer_route_name(self.capabilities.transfer.requested),
            "\n",
        )
        out += String(
            "transfer_all_default=",
            _bool_text(self.capabilities.transfer.all_default()),
            "\n",
        )
        out += String(
            "transfer_ack_unproven=",
            _bool_text(self.capabilities.transfer.any_unproven()),
            "\n",
        )
        out += String(
            "transfer_unified_memory=",
            _bool_text(self.capabilities.transfer.unified_memory),
            "\n",
        )
        # One `transfer` line per buffer role, in role order, always all
        # present, rendered `<role>:<requested>:<selected>:<reason>:<evidence>
        # :<retire_on>`. A repeated key is a list in order, the same
        # convention `block` and `warning` use below, so a consumer that only
        # wants the headline reads `transfer_all_default` and one that wants
        # to explain a per-role fallback reads these.
        for i in range(len(self.capabilities.transfer.decisions)):
            var d = self.capabilities.transfer.decisions[i].copy()
            out += String(
                "transfer=",
                transfer_role_name(d.role),
                ":",
                transfer_route_name(d.requested),
                ":",
                transfer_route_name(d.selected),
                ":",
                transfer_block_name(d.reason),
                ":",
                transfer_evidence_name(d.evidence_level),
                ":",
                retire_event_name(d.contract.retire_event),
                "\n",
            )

        out += String(
            "memory_device_bytes=", self.memory.device_bytes(), "\n"
        )
        out += String(
            "memory_upper_bound_bytes=",
            self.memory.upper_bound_bytes(),
            "\n",
        )
        out += String("memory_host_bytes=", self.memory.host_bytes(), "\n")
        out += String(
            "memory_partial_budget_bytes=", self.memory.partial_budget, "\n"
        )
        out += String(
            "memory_estimate_complete=",
            _bool_text(self.memory.bins_known),
            "\n",
        )
        out += String(
            "memory_binned_matrix_bytes=",
            self.memory.binned_matrix_bytes,
            "\n",
        )
        out += String(
            "memory_leaf_id_bytes=", self.memory.leaf_id_bytes, "\n"
        )
        out += String(
            "memory_gradient_bytes=", self.memory.gradient_bytes, "\n"
        )
        out += String(
            "memory_hessian_bytes=", self.memory.hessian_bytes, "\n"
        )
        out += String(
            "memory_histogram_bytes=", self.memory.histogram_bytes, "\n"
        )
        out += String(
            "memory_feature_id_bytes=", self.memory.feature_id_bytes, "\n"
        )

        for i in range(self.blocking_reasons.count()):
            out += String(
                "block=",
                self.blocking_reasons.codes[i],
                ":",
                block_reason_name(self.blocking_reasons.codes[i]),
                ":",
                self.blocking_reasons.messages[i],
                "\n",
            )
        for i in range(self.warnings.count()):
            out += String(
                "warning=",
                self.warnings.codes[i],
                ":",
                warning_name(self.warnings.codes[i]),
                ":",
                self.warnings.messages[i],
                "\n",
            )
        return out^


def decide_device(
    request: DeviceRequest, caps: DeviceCapabilities
) raises -> DeviceDecision:
    """Resolve one request against one set of capabilities.

    Pure: it opens no device, reads no environment (the environment was
    already folded into `caps`), and touches no dataset. Every input is a
    value a caller could have serialized, which is what makes the whole
    policy testable on a machine with no accelerator and injectable with
    hardware nobody here owns.

    Never raises for a GPU request that cannot run. The decision carries
    `blocked` and `message` instead, and `raise_if_blocked` turns it into
    the refusal. That split is what lets a caller ask "what would
    device='gpu' do here" without handling an exception.

    Raises only for a device code outside the vocabulary and for a shape
    with no rows or no features, both of which are caller errors rather
    than policy outcomes.
    """
    if (
        request.requested_device != CPU_DEVICE
        and request.requested_device != GPU_DEVICE
        and request.requested_device != AUTO_DEVICE
    ):
        raise Error("unknown device code ", request.requested_device)

    var memory = estimate_gpu_memory(request, caps.profile)
    var blocks = _collect_blocks(request, caps, memory)
    var warnings = _collect_warnings(request, caps)

    # Filled in by one of the branches below, then handed to the single
    # constructor at the end. One construction site rather than nine keeps
    # the ownership transfers of `blocks`, `warnings`, and `memory` in one
    # place, where they can be read against the fields they land in.
    var selected: Int = CPU_DEVICE
    var blocked: Bool = False
    var code: Int = DECISION_EXPLICIT_CPU
    var message = String("")
    var evidence: String = EVIDENCE_NONE

    if request.requested_device == CPU_DEVICE:
        message = String(
            "device='cpu' was requested, and the CPU path covers every"
            " objective and every input"
        )
    elif request.requested_device == GPU_DEVICE:
        if not blocks.is_empty():
            selected = NO_DEVICE
            blocked = True
            code = DECISION_GPU_REFUSED
            message = String(
                "device 'gpu' requested but ",
                blocks.first_message(),
                "; use device 'cpu' or 'auto'",
            )
        else:
            selected = GPU_DEVICE
            code = DECISION_EXPLICIT_GPU
            evidence = EVIDENCE_EXPLICIT
            message = String(
                "device='gpu' was requested and nothing blocks it, so"
                " training runs on the accelerator; an explicit request"
                " never falls back to the CPU"
            )
            warnings.add(
                WARN_EXPLICIT_GPU_UNMEASURED,
                String(
                    "device='gpu' was requested explicitly, so this run"
                    " rests on no crossover measurement and may be slower"
                    " than the CPU"
                ),
            )
    elif not blocks.is_empty():
        code = DECISION_AUTO_CPU_BLOCKED
        message = String(
            "auto chose the CPU because the GPU path cannot run this"
            " workload: ",
            blocks.first_message(),
        )
    elif caps.auto_min_cells >= 0:
        # An operator's threshold outranks the rule table: it exists so the
        # crossover benchmark can reach the GPU on a machine no rule covers.
        evidence = EVIDENCE_ENV
        if request.cells() >= caps.auto_min_cells:
            selected = GPU_DEVICE
            code = DECISION_AUTO_GPU_ENV_THRESHOLD
            message = String(
                "MOJOTREES_AUTO_MIN_CELLS=",
                caps.auto_min_cells,
                " and this workload has ",
                request.cells(),
                " cells, so the size heuristic selects the GPU",
            )
            warnings.add(
                WARN_ENV_THRESHOLD_UNVALIDATED,
                String(
                    "MOJOTREES_AUTO_MIN_CELLS is the knob for running the"
                    " crossover benchmark, not a validated threshold; this"
                    " GPU choice rests on no measurement"
                ),
            )
        else:
            code = DECISION_AUTO_CPU_BELOW_ENV_THRESHOLD
            message = String(
                "MOJOTREES_AUTO_MIN_CELLS=",
                caps.auto_min_cells,
                " and this workload has only ",
                request.cells(),
                " cells, so auto keeps the CPU",
            )
    else:
        var rules = crossover_rules()
        var matched = -1
        for i in range(len(rules)):
            if rules[i].matches(caps, request):
                matched = i
                break
        if matched >= 0:
            selected = GPU_DEVICE
            code = DECISION_AUTO_GPU_EVIDENCE
            evidence = rules[matched].evidence_id.copy()
            message = String(
                "crossover rule '",
                rules[matched].name,
                "' covers this device and workload, so auto selects the GPU",
            )
        elif len(rules) > 0:
            code = DECISION_AUTO_CPU_BELOW_EVIDENCE
            message = String(
                "none of the ",
                len(rules),
                " crossover rule(s) in policy version ",
                POLICY_VERSION,
                " covers this device and workload, so auto keeps the CPU",
            )
        else:
            code = DECISION_AUTO_CPU_NO_EVIDENCE
            message = String(
                "the crossover table (policy version ",
                POLICY_VERSION,
                ") is empty: no benchmark has established a workload size"
                " where GPU training beats CPU training, so auto"
                " conservatively keeps the CPU. Set MOJOTREES_AUTO_MIN_CELLS"
                " to run that benchmark, or device='gpu' to force the"
                " accelerator",
            )

    return DeviceDecision(
        request.copy(),
        caps.copy(),
        selected,
        blocked,
        code,
        message^,
        blocks^,
        warnings^,
        memory^,
        evidence^,
    )


def resolve_device(
    device: Int, n_rows: Int, n_features: Int, n_outputs: Int = 1
) raises -> Int:
    """Resolve a requested device to the backend that will actually run:
    `CPU_DEVICE` or `GPU_DEVICE`, never `AUTO_DEVICE`.

    The narrow entry point the trainers use, in terms of the engine above:
    it builds a request the caller has only partly described (no objective,
    no bin count, no input flags), detects capabilities, and raises rather
    than returning a refusal. Callers that can describe the whole workload
    should build a `DeviceRequest` and call `decide_device` directly, which
    is what gets them the memory gate, the objective gate, and a report.
    """
    var request = DeviceRequest(device, n_rows, n_features, n_outputs)
    var caps = DeviceCapabilities.detect()
    var decision = decide_device(request, caps)
    decision.raise_if_blocked()
    return decision.selected_device


# --- Flat entry points ------------------------------------------------
#
# `decide_device` takes structs, which is right for a Mojo caller and wrong
# for every other caller this policy has. A CPython binding builds arguments
# out of `PythonObject`s and cannot construct a `DeviceRequest`; a C API and
# a CLI are in the same position. Without a flat seam each of them grows its
# own argument marshalling, and the one that gets it slightly wrong is a
# second policy nobody meant to write.
#
# So the flat forms live here, beside the engine, and every one of them is a
# marshaller: it normalizes sentinels, builds the two structs, and calls
# `decide_device`. None of them decides anything, and none of them may.


def _normalized_bins(n_bins: Int) -> Int:
    """A bin count off a flat boundary, with "undeclared" normalized.

    Anything nonpositive is undeclared. The boundary carries plain ints, so
    an undeclared value needs a value, and Python sends `-1`
    (`_BINS_UNSPECIFIED` in python/mojotrees/device_selection.py). Zero
    arrives from a caller that left a field default. Neither is a bin count
    the binner would accept, and treating either as one would put a fabricated
    number into the memory estimate.
    """
    if n_bins <= 0:
        return BINS_UNSPECIFIED
    return n_bins


def _normalized_objective(objective: Int) -> Int:
    """An objective code off a flat boundary, with "undeclared" normalized.

    Below `-1` is undeclared. `-1` is deliberately excluded: it is the
    multiclass marker and a real code, so folding it into the undeclared
    sentinel would silently skip the objective gate for every multiclass run.
    """
    if objective < -1:
        return OBJECTIVE_UNSPECIFIED
    return objective


def capabilities_from_reported(
    reported_api: String,
    reported_arch: String,
    core_count: Int,
    max_threads_per_block: Int,
    max_shared_memory_per_block: Int,
    memory_budget_bytes: Int = 0,
    context_open: Bool = True,
    kernels_ready: Bool = False,
    warmup_level: Int = 0,
) raises -> DeviceCapabilities:
    """Capabilities from attributes a caller read off an open `DeviceContext`.

    The call site apple_gpu_policy.mojo's `GpuProfile.from_reported`
    docstring promises and this module's `PROFILE_REPORTED` describes, in one
    function, so an owner of a `DeviceContext` needs no knowledge of how the
    two layers fit together: query the attributes, pass them here, hand the
    result to `decide_device`.

    `context_open` defaults True because a caller with attributes to report
    has a context open by construction. `kernels_ready` defaults False
    because having a context says nothing about whether any kernel has been
    created, and reporting an unpaid cost as paid is the failure
    `SessionState` exists to prevent.
    """
    var profile = GpuProfile.from_reported(
        reported_api,
        reported_arch,
        core_count,
        max_threads_per_block,
        max_shared_memory_per_block,
        memory_budget_bytes,
    )
    return DeviceCapabilities.from_profile(
        profile^,
        PROFILE_REPORTED,
        SessionState(context_open, kernels_ready, warmup_level),
    )


def decide_device_report(
    device: String,
    n_rows: Int,
    n_features: Int,
    n_outputs: Int = 1,
    n_bins: Int = BINS_UNSPECIFIED,
    objective: Int = OBJECTIVE_UNSPECIFIED,
    sparse: Bool = False,
    categorical: Bool = False,
    has_missing: Bool = False,
    uses_validation: Bool = False,
) raises -> String:
    """The whole contract across a flat boundary: workload in, serialized
    decision out.

    This is the entry point `python/mojotrees/device_selection.py` calls
    `decide_device` and the one that moves it off its `"narrow"` contract.
    The exact binding is in handoffs/connect_05_device_policy.md.

    Never raises for a workload it refuses. A refusal is `blocked=true` in the
    returned lines with `message=` saying why, so a caller can ask "what would
    `device='gpu'` do here" without handling an exception, and the caller that
    wants the exception calls `resolve_device_full` instead. It raises only
    for a device name outside the vocabulary, a shape with no rows or no
    features, and an unparsable `MOJOTREES_GPU_TRANSFER`: caller and operator
    errors, not policy outcomes.

    Capabilities are detected, so the hardware profile is the portable
    fallback and the decision says so through `profile_source=fallback`. A
    caller holding an open `DeviceContext` should use
    `decide_device_report_reported`, which is the same function over
    attributes it actually read.
    """
    var request = DeviceRequest(
        parse_device(device),
        n_rows,
        n_features,
        n_outputs,
        _normalized_bins(n_bins),
        _normalized_objective(objective),
        sparse,
        categorical,
        has_missing,
        uses_validation,
    )
    var caps = DeviceCapabilities.detect(SessionState.from_env())
    return decide_device(request, caps).serialize()


def decide_device_report_reported(
    device: String,
    n_rows: Int,
    n_features: Int,
    n_outputs: Int,
    n_bins: Int,
    objective: Int,
    sparse: Bool,
    categorical: Bool,
    has_missing: Bool,
    uses_validation: Bool,
    reported_api: String,
    reported_arch: String,
    core_count: Int,
    max_threads_per_block: Int,
    max_shared_memory_per_block: Int,
    memory_budget_bytes: Int = 0,
    context_open: Bool = True,
    kernels_ready: Bool = False,
    warmup_level: Int = 0,
) raises -> String:
    """`decide_device_report` for a caller that has read the device.

    The same engine over `PROFILE_REPORTED` capabilities instead of the
    portable fallback, which is what makes the memory gate, the unified-memory
    warning, and any future hardware-scoped crossover rule mean anything. The
    decision it returns carries `profile_source=reported`, so a report from
    this entry point can be told from one that guessed.
    """
    var request = DeviceRequest(
        parse_device(device),
        n_rows,
        n_features,
        n_outputs,
        _normalized_bins(n_bins),
        _normalized_objective(objective),
        sparse,
        categorical,
        has_missing,
        uses_validation,
    )
    var caps = capabilities_from_reported(
        reported_api,
        reported_arch,
        core_count,
        max_threads_per_block,
        max_shared_memory_per_block,
        memory_budget_bytes,
        context_open,
        kernels_ready,
        warmup_level,
    )
    return decide_device(request, caps).serialize()


def resolve_device_full(
    device: Int,
    n_rows: Int,
    n_features: Int,
    n_outputs: Int = 1,
    n_bins: Int = BINS_UNSPECIFIED,
    objective: Int = OBJECTIVE_UNSPECIFIED,
    sparse: Bool = False,
    categorical: Bool = False,
    has_missing: Bool = False,
    uses_validation: Bool = False,
) raises -> Int:
    """Resolve a fully described workload to `CPU_DEVICE` or `GPU_DEVICE`,
    raising the refusal when it cannot run.

    What `resolve_device` should have been, and what a trainer that knows its
    own objective, bin count, and input flags should call. `resolve_device`
    describes only the shape, so it skips the objective gate, the bin-limit
    gate, the sparse and validation blocks, and the memory gate, and the
    decision it produces carries `WARN_INCOMPLETE_REQUEST` saying so. That is
    not a second policy, it is the same engine asked a smaller question, but
    the smaller question is one an explicit `device='gpu'` can pass and then
    fail deeper in, which is precisely the outcome the refusal exists to
    prevent.

    The exact call-site edits for `model.mojo` and `trainset.mojo` are in
    handoffs/connect_05_device_policy.md. They are one lane over, so they are
    a patch request rather than an edit.
    """
    var request = DeviceRequest(
        device,
        n_rows,
        n_features,
        n_outputs,
        _normalized_bins(n_bins),
        _normalized_objective(objective),
        sparse,
        categorical,
        has_missing,
        uses_validation,
    )
    var caps = DeviceCapabilities.detect(SessionState.from_env())
    var decision = decide_device(request, caps)
    decision.raise_if_blocked()
    return decision.selected_device


def describe_decision(decision: DeviceDecision) raises -> String:
    """One line for benchmark output and bug reports.

    The prose report belongs to whoever is formatting for a human;
    `serialize()` is what they should parse. This is the terse form for a
    log line that has one line to spend."""
    var out = String(
        "device ",
        device_name(decision.request.requested_device),
        " -> ",
        device_name(decision.selected_device),
        " (",
        decision_name(decision.decision_code),
        ") policy=",
        decision.policy_version,
        " evidence=",
        decision.evidence_id,
        " rows=",
        decision.request.n_rows,
        " features=",
        decision.request.n_features,
        " api=",
        api_name(decision.capabilities.profile.api),
        " profile=",
        profile_source_name(decision.capabilities.profile_source),
        " session=",
        decision.capabilities.session.describe(),
    )
    if not decision.capabilities.transfer.all_default():
        out += String(
            " transfer=",
            transfer_route_name(decision.capabilities.transfer.requested),
        )
    if decision.capabilities.transfer.any_unproven():
        out += String(" ack_unproven=1")
    if not decision.blocking_reasons.is_empty():
        out += String(
            " blocked_by=",
            block_reason_name(decision.blocking_reasons.codes[0]),
        )
    return out^
