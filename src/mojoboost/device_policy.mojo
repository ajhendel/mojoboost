"""The device-policy contract: one authoritative decision engine in Mojo.

Every question "should this training run go to the accelerator, and if not,
why not" is answered here and nowhere else. Before this module the answer
existed twice: once in `device.mojo` (the vocabulary, the availability
probe, and a size heuristic) and once again, larger and with a different
set of rules, in `python/mojoboost/device_selection.py`. Two engines that
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
- the refusal an explicit `gpu` request gets when it cannot run.

`device.mojo` is a thin compatibility facade over this module.
`python/mojoboost/device_selection.py` is reduced to extracting plain
workload metadata from `X` and `y` and formatting the decision this module
returns. Neither one decides anything.

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

`MOJOBOOST_AUTO_MIN_CELLS` is the escape hatch for running that benchmark:
an integer cell count (`n_rows * n_features`) at or above which `auto`
selects the GPU, `0` meaning "whenever the GPU path covers the workload",
unset or negative meaning the heuristic is off. A run that reaches the GPU
through it is reported with `EVIDENCE_ENV` and a warning, never as a
validated choice.

`MOJOBOOST_DISABLE_GPU=1` makes this module report that no accelerator is
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
decision says so through `PROFILE_FALLBACK`. `MOJOBOOST_GPU_BACKEND` is
honored only as an operator's declaration of the API name, is recorded as
`PROFILE_DECLARED`, and never overrides a reported capability.

Availability is a build property
--------------------------------
Mojo resolves `has_accelerator()` at compile time, so a binary built where
an accelerator was present reports one as available. On a redistributed
build (a wheel) a `gpu` request therefore fails when the device is opened
rather than when it is resolved. `WARN_BUILD_TIME_AVAILABILITY` marks every
decision that rests on that comptime answer, and `MOJOBOOST_DISABLE_GPU=1`
is the way to pin such a build to the CPU.

LightGBM difference: LightGBM spells this `device_type` and takes `cpu`,
`gpu`, or `cuda`, with no `auto`. mojoboost has a single portable GPU
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
from .boosting import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    CUSTOM,
    FAIR,
    GAMMA,
    HUBER,
    L1,
    MAPE,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    TWEEDIE,
)


# --- Mirrors. Pinned by tests/parallel/test_device_policy.mojo. ---
#
# Copied rather than imported because importing their home modules would
# close an import cycle through this one: `ranking.mojo` imports
# `model.mojo`, which imports `device.mojo`, which imports this module, and
# `histogram_gpu.mojo` pulls the whole GPU kernel stack into a layer that
# has to stay compilable and testable on a machine with no accelerator.
# The handoff records the pinning test that asserts each mirror still
# equals its source.

# `LAMBDARANK` in ranking.mojo: objective code 7, which continues the
# registry in boosting.mojo but whose gradients come from query groups.
comptime LAMBDARANK = 7

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
comptime EVIDENCE_ENV = String("MOJOBOOST_AUTO_MIN_CELLS")


# --- Where a capability profile came from -----------------------------

comptime PROFILE_NONE = 0
"""No accelerator, so no profile was built."""

comptime PROFILE_FALLBACK = 1
"""`GpuProfile.generic()`: nobody has opened a device, so the conservative
portable profile stands in. Not Apple-shaped and not NVIDIA-shaped."""

comptime PROFILE_DECLARED = 2
"""An operator named the API through `MOJOBOOST_GPU_BACKEND`. The numbers
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
    gradients come from query groups)."""
    return (
        objective == SQUARED_ERROR
        or objective == BINARY_LOGISTIC
        or objective == POISSON
        or objective == HUBER
        or objective == QUANTILE
        or objective == L1
        or objective == GAMMA
        or objective == TWEEDIE
        or objective == MAPE
        or objective == FAIR
        or objective == CROSS_ENTROPY
    )


def gpu_trains_objective(objective: Int) -> Bool:
    """Whether `train_gpu` covers this objective.

    Every built-in objective, which is what `train_gpu` accepts: it runs
    the same `_check_objective` the CPU trainer does and then grows trees
    on the device. `CUSTOM` and `LAMBDARANK` are not built-in and are
    trained through `train_custom` and the ranker, both of which are host
    paths.

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

    This predicate is the authoritative one.
    `gpu_objectives_native.supports_device_objective` still carries its own
    copy of the list; the handoff specifies the one-line edit that makes it
    delegate here, which is what collapses the duplicate."""
    if objective == OBJECTIVE_UNSPECIFIED:
        return True
    return is_builtin_objective(objective)


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
    """Whether `MOJOBOOST_DISABLE_GPU=1` pins this process to the CPU."""
    return getenv("MOJOBOOST_DISABLE_GPU") == "1"


def gpu_available() -> Bool:
    """True when training can run on an accelerator: one was present when
    this build was compiled and `MOJOBOOST_DISABLE_GPU=1` is not set."""
    if not build_has_accelerator():
        return False
    return not gpu_disabled_by_env()


def env_auto_min_cells() -> Int:
    """The `auto` size threshold in cells. Unset, negative, or unparsable
    means disabled, in which case `auto` never selects the GPU on size
    alone."""
    var s = getenv("MOJOBOOST_AUTO_MIN_CELLS")
    if s.byte_length() == 0:
        return AUTO_MIN_CELLS
    try:
        return Int(s)
    except:
        return AUTO_MIN_CELLS


def env_declared_api() -> Int:
    """The GPU API an operator named through `MOJOBOOST_GPU_BACKEND`, or
    `API_UNKNOWN`.

    A declaration, not a detection: it names the API for reporting and for
    scoping a crossover rule, and it never supplies a capability number.
    Nothing here reads an operating system name or a marketing chip string
    to guess at hardware."""
    var s = getenv("MOJOBOOST_GPU_BACKEND")
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
        `MOJOBOOST_AUTO_MIN_CELLS` are written in."""
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
    - `disabled_by_env`: `MOJOBOOST_DISABLE_GPU=1` was set.
    - `profile`: the hardware capabilities, from apple_gpu_policy.mojo.
    - `profile_source`: one of the `PROFILE_*` codes, which is how a
      reader tells a reading from a fallback.
    - `max_rows`, `min_bins`, `max_bins`: the kernel and binner limits.
      Fields rather than constants so a build that widens one can say so.
    - `auto_min_cells`: the `MOJOBOOST_AUTO_MIN_CELLS` value in effect.
      Negative means the heuristic is off.
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

    def __init__(
        out self,
        gpu_available: Bool,
        built_with_accelerator: Bool,
        disabled_by_env: Bool,
        var profile: GpuProfile,
        profile_source: Int,
        auto_min_cells: Int,
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

    @staticmethod
    def detect() -> DeviceCapabilities:
        """Capabilities for the build and process running right now.

        Opens no device. The hardware profile is therefore the portable
        fallback (`PROFILE_FALLBACK`), or `PROFILE_DECLARED` when an
        operator named the API through `MOJOBOOST_GPU_BACKEND`. A caller
        that already has a `DeviceContext` open should read its attributes
        and call `from_profile` instead, which is the only way to reach
        `PROFILE_REPORTED`.
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
        return DeviceCapabilities(
            available,
            built,
            disabled,
            profile^,
            source,
            env_auto_min_cells(),
        )

    @staticmethod
    def from_profile(
        var profile: GpuProfile, profile_source: Int = PROFILE_REPORTED
    ) -> DeviceCapabilities:
        """Capabilities carrying a profile a caller already has.

        `PROFILE_REPORTED` is the default because the intended caller is
        one that read the attributes off an open `DeviceContext`. Pass
        `PROFILE_SYNTHETIC` for a fixture; the source is recorded and
        surfaces as `WARN_SYNTHETIC_CAPABILITIES`, and it never changes a
        gate."""
        var built = build_has_accelerator()
        var disabled = gpu_disabled_by_env()
        return DeviceCapabilities(
            built and not disabled,
            built,
            disabled,
            profile^,
            profile_source,
            env_auto_min_cells(),
        )

    @staticmethod
    def unavailable() -> DeviceCapabilities:
        """A machine with no accelerator. The conservative answer, and the
        one an injected fixture wants when it is testing the CPU path."""
        return DeviceCapabilities(
            False,
            False,
            False,
            GpuProfile.generic(),
            PROFILE_NONE,
            AUTO_MIN_CELLS,
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

    Invariants, which the pinning test in the handoff asserts:

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
                    "MOJOBOOST_DISABLE_GPU=1 pins this process to the CPU"
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
        blocks.add(
            BLOCK_CUSTOM_OBJECTIVE,
            String(
                "a custom objective's gradients come from a caller-supplied"
                " callable on the host, so it trains through train_custom on"
                " the CPU"
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
                " set MOJOBOOST_DISABLE_GPU=1 to pin such a build to the CPU"
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
                "MOJOBOOST_GPU_BACKEND declared api '",
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
        `MOJOBOOST_AUTO_MIN_CELLS` or requested explicitly."""
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
        out += String("bins_known=", _bool_text(self.request.bins_known()), "\n")
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
                "MOJOBOOST_AUTO_MIN_CELLS=",
                caps.auto_min_cells,
                " and this workload has ",
                request.cells(),
                " cells, so the size heuristic selects the GPU",
            )
            warnings.add(
                WARN_ENV_THRESHOLD_UNVALIDATED,
                String(
                    "MOJOBOOST_AUTO_MIN_CELLS is the knob for running the"
                    " crossover benchmark, not a validated threshold; this"
                    " GPU choice rests on no measurement"
                ),
            )
        else:
            code = DECISION_AUTO_CPU_BELOW_ENV_THRESHOLD
            message = String(
                "MOJOBOOST_AUTO_MIN_CELLS=",
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
                " conservatively keeps the CPU. Set MOJOBOOST_AUTO_MIN_CELLS"
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
    )
    if not decision.blocking_reasons.is_empty():
        out += String(
            " blocked_by=",
            block_reason_name(decision.blocking_reasons.codes[0]),
        )
    return out^
