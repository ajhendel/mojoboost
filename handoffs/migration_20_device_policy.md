# Handoff: Mojo is authoritative for device choice (task 20)

The device decision now has one implementation, and it is in Mojo. This
file is the exact wiring for whoever integrates it, plus the list of what
this lane knowingly left broken because the files involved were not its to
edit.

Nothing was committed or staged. No test was written or run, no build was
run, and no Mojo, Python, or Pixi command was executed: this lane was
static reasoning only, so **none of the code below has been compiled**.
Treat the first `pixi run mojo` invocation as the real review.

## Files this lane owns

| File | State |
|---|---|
| `src/mojoboost/device_policy.mojo` | New. The whole contract and the decision engine. |
| `src/mojoboost/device.mojo` | Rewritten as a thin compatibility facade. 166 lines to 120, all delegation. |
| `src/mojoboost/apple_gpu_policy.mojo` | Docstring edits only. No behavior change, no signature change. |
| `python/mojoboost/device_selection.py` | Rewritten. 1,461 lines to about 640, and the decision engine is gone. |
| `handoffs/migration_20_device_policy.md` | This file. |

Nothing else was touched.

## The shape of it

```
apple_gpu_policy.mojo    leaf. GpuProfile, tuning plan, partial budget.
        |                imports nothing, opens nothing.
        v
device_policy.mojo       the contract and the engine. Imports only
        |                apple_gpu_policy and boosting (both CPU-pure),
        |                so the whole policy compiles and runs on a
        |                machine with no accelerator.
        v
device.mojo              compatibility facade. Decides nothing.
        |
        v
model.mojo, params.mojo, trainset.mojo, bindings   unchanged callers
```

`device_policy.mojo` deliberately imports nothing from the GPU stack.
Pulling `histogram_gpu.mojo` or `gpu_objectives_native.mojo` in for two
constants and one predicate would have put the whole kernel stack behind
`params.mojo`, which imports `device.mojo`, and would have made the policy
untestable without the `max.gpu` package resolving. The four values that
would have come from there are mirrored instead, in a marked block, which
is the convention `apple_gpu_policy.mojo` already established for its
`gpu_tiling.mojo` mirrors.

## What is authoritative where

| Question | One implementation |
|---|---|
| Device vocabulary and codes | `device_policy.parse_device` / `device_name` |
| Objectives the GPU path trains | `device_policy.gpu_trains_objective` |
| Objectives with device-side gradients | `device_policy.gpu_objective_is_device_resident` |
| Outputs per round the GPU covers | `device_policy.gpu_supports_outputs` |
| Row, bin, and memory limits | `device_policy.DeviceCapabilities` fields + `_collect_blocks` |
| Memory estimate | `device_policy.estimate_gpu_memory` |
| Hardware capabilities | `apple_gpu_policy.GpuProfile` |
| Partial-histogram budget | `apple_gpu_policy.partial_budget_bytes` |
| Crossover evidence | `device_policy.crossover_rules` |
| Explicit-GPU refusal | `device_policy.decide_device` |
| Selected backend | `device_policy.decide_device` |

Python owns exactly two things: reading `X` and `y` into a `Workload`, and
rendering a `DeviceReport`.

---

# 1. The binding

One new function. `bindings/_mojoboost.mojo` is not this lane's file, so
here it is in full.

The objective crosses as a **code**, not a name. The estimators already
hold the code (`_objective_code()` in `python/mojoboost/__init__.py`), and
a code sidesteps two traps a name would walk into: `objective_from_name`
in `params.mojo` *raises* for `lambdarank` (it needs query groups, so it
is not a parameter-string objective), which would turn the ranker's block
into an exception, and it returns `MULTICLASS` (-1), which is a tree count
rather than an objective code.

Add to the imports at the top:

```mojo
from mojoboost.device_policy import (
    BINS_UNSPECIFIED,
    DeviceCapabilities,
    DeviceRequest,
    OBJECTIVE_UNSPECIFIED,
    decide_device as mojo_decide_device,
)
```

Add the function, next to the existing `resolve_device`:

```mojo
def decide_device(
    device: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    n_outputs: PythonObject,
    n_bins: PythonObject,
    objective: PythonObject,
    sparse: PythonObject,
    categorical: PythonObject,
    has_missing: PythonObject,
    uses_validation: PythonObject,
) raises -> PythonObject:
    """Resolve a device request and return the serialized decision.

    Returns the `key=value` lines `DeviceDecision.serialize` produces. It
    does not raise for a workload the GPU path refuses: that refusal is
    `blocked=true` in the lines, and the Python wrapper turns it into a
    `DeviceUnavailableError`. It raises only for a device name outside the
    vocabulary and for a shape with no rows or no features, both of which
    are caller errors rather than policy outcomes.

    Undeclared fields arrive as sentinels, because the boundary carries
    plain ints: a negative `n_bins` means no bin count, and an `objective`
    below -1 means no objective (-1 is the multiclass marker and is a real
    value). The four flags are 0 or 1, so nothing here converts a Python
    bool, which is the convention `_parse_goss` already follows.
    """
    var bins = Int(py=n_bins)
    if bins < 0:
        bins = BINS_UNSPECIFIED
    var obj = Int(py=objective)
    if obj < 0:
        # Both the "undeclared" sentinel and MULTICLASS land here.
        # Multiclass is a tree count, and `n_outputs` already carries it.
        obj = OBJECTIVE_UNSPECIFIED
    var request = DeviceRequest(
        parse_device(String(py=device)),
        Int(py=n_rows),
        Int(py=n_features),
        Int(py=n_outputs),
        bins,
        obj,
        Int(py=sparse) != 0,
        Int(py=categorical) != 0,
        Int(py=has_missing) != 0,
        Int(py=uses_validation) != 0,
    )
    var caps = DeviceCapabilities.detect()
    var decision = mojo_decide_device(request, caps)
    return PythonObject(decision.serialize())
```

Register it beside the others in the module builder:

```mojo
        m.def_function[decide_device]("decide_device")
```

`resolve_device` and `gpu_available` stay exactly as they are. They still
work, they still run the same engine, and `python/mojoboost/__init__.py`
still calls `resolve_device` (see section 2).

## 1b. Optional: `objective_code`, for the name-only explain path

`explain_device_choice(X, y, objective="poisson")` is a public entry point
where the caller has a name and no code. `Workload` will resolve it
through an `objective_code` binding when one exists, and otherwise treats
the objective as undeclared and skips that gate. Adding it is three lines,
and the name table stays in `params.mojo`:

```mojo
def objective_code(name: PythonObject) raises -> PythonObject:
    """The objective code a public objective name denotes. Raises for a
    name the trainer does not implement, which the Python caller catches
    and reports as an undeclared objective."""
    return PythonObject(objective_from_name(String(py=name)))
```

with `from mojoboost.params import objective_from_name` and
`m.def_function[objective_code]("objective_code")`.

Caveats worth a comment at that call site: it raises for `lambdarank` and
returns `MULTICLASS` (-1) for `multiclass`. Both are handled downstream,
because `_code_for_objective_name` returns None on any exception and the
`decide_device` binding folds any negative code into
`OBJECTIVE_UNSPECIFIED`. The consequence is that a name-only caller does
not get the ranking block; a caller that has the code does. That is
acceptable because the estimators, which are the path that matters, always
have the code.

## Reported capabilities instead of the fallback profile

`DeviceCapabilities.detect()` opens no device, so its profile is
`PROFILE_FALLBACK` (or `PROFILE_DECLARED` when `MOJOBOOST_GPU_BACKEND` is
set). That is deliberate: opening a `DeviceContext` to answer "which
backend" would cost the context creation on every call, including every
CPU run.

The moment a caller already has one open, it should hand the real numbers
over. Four lines, in whichever file owns the open context (`gpu_runtime`
or `train_gpu`, neither of which is this lane's):

```mojo
var caps = query_device_caps(ctx)          # gpu_tiling.mojo, already exists
var profile = GpuProfile.from_reported(
    String("metal"),                        # or the API this build targets
    String(""),                             # architecture string, when readable
    caps.sm_count,
    caps.max_threads_per_block,
    caps.max_shared_memory_per_block,
    0,                                      # memory budget, when readable
)
var policy_caps = DeviceCapabilities.from_profile(profile^)
```

That is the only way to reach `PROFILE_REPORTED`, and it is what turns the
memory gate on: `_collect_blocks` refuses to block on memory while the
budget is unreported, because a zero budget is not a budget.

**Do not** substitute an operating-system name or a marketing chip string
for a reported capability. `MOJOBOOST_GPU_BACKEND` is honored only as an
operator's declaration of the API, is recorded as `PROFILE_DECLARED`, and
supplies no numbers. The Python layer used to guess Metal from
`Darwin` + `arm64` and read `machdep.cpu.brand_string` through a
subprocess; both are gone and neither should come back.

---

# 2. Estimator integration

`python/mojoboost/__init__.py` is not this lane's file. Today it calls
`_mojoboost.resolve_device(...)` in `_resolve_device` (line 1284) and that
**keeps working unchanged** through the facade. Nothing here is required
for correctness; all of it is upgrade.

## 2a. The minimum, if you want the richer gates

Replace the body of `_resolve_device` so it passes the whole workload:

```python
    def _resolve_device(self, n_rows, n_features, n_outputs, workload=None):
        """The backend that will actually run, "cpu" or "gpu". Names are
        case-insensitive, as LightGBM treats `device_type`. Raises
        ValueError for an unknown `device` and RuntimeError when "gpu" is
        requested but unavailable or unsupported; "gpu" never falls back to
        the CPU."""
        from .device_selection import (
            DeviceUnavailableError,
            Workload,
            select_device,
        )

        device = self._resolve_alias("device", "device_type", "cpu")
        if workload is None:
            workload = Workload(n_rows, n_features, n_classes=n_outputs)
        try:
            return select_device(device, workload).resolved
        except DeviceUnavailableError as exc:
            raise RuntimeError(str(exc)) from None
```

`select_device` raises `ValueError` for a name outside `DEVICES` with the
same message shape the estimator raises today, so the explicit
`_DEVICES` check above it can go.

Note the `n_classes=n_outputs` spelling: `Workload.n_outputs` derives
trees-per-round from the class count (1 for 1 or 2 classes), and every
caller in `__init__.py` already passes trees-per-round, so passing it as
`n_classes` round-trips correctly for 1 and for 3-or-more and is only
ambiguous at exactly 2, where both spellings give 1.

## 2b. The three call sites, with the full workload

The workload each one should build, with `objective_code` coming from the
estimator's own `_objective_code()` (or the ranker's `_LAMBDARANK`), which
is the value the native contract wants:

```python
    workload = Workload(
        n_rows,
        n_features,
        objective_code=self._objective_code(),
        n_classes=n_outputs,
        max_bin=int(self.max_bin),
        sparse=False,
        categorical=bool(cat_buf),
        has_missing=bool(self.use_missing),
        has_eval_set=eval_set is not None,
    )
```

| Line (current file) | Caller | Differences from the above |
|---|---|---|
| 2428 | `MojoBoostRegressor._fit` | as written |
| 2825 | `MojoBoostClassifier._fit` | `n_classes=len(self.classes_)` |
| 3390 | `MojoBoostRanker._fit` | `objective_code=_LAMBDARANK` (7); the ranker has no `_objective_code` |

`objective` (the display name) can be passed alongside for a nicer
explanation, but it changes no gate.

Two payoffs, both of which delete a Python guard:

- The ranker's `if device != "cpu": raise` at line 3391 becomes
  redundant: `BLOCK_RANKING_OBJECTIVE` refuses `lambdarank` on the GPU
  natively, with a better message.
- The eval-set guard at line 1510 and the custom-objective guard at line
  2497 likewise become redundant: `BLOCK_VALIDATION_SET` and
  `BLOCK_CUSTOM_OBJECTIVE` cover them.

Delete those three Python guards only after the binding above is landed
and a test asserts the native refusal, not before. Until then they are the
enforcement and the native blocks are a second opinion.

The sparse estimator's guard at line 2068 (`if self.device == "gpu": raise`)
should stay as is: it fires before a `Workload` exists.

## 2c. Custom objectives

`_fit_custom` (line 2484) trains through `train_custom`, and
`_objective_code()` already returns `_CUSTOM` (6) for a callable
objective. Pass it: `BLOCK_CUSTOM_OBJECTIVE` then refuses the GPU
natively, with a message that explains why rather than just refusing.
That makes the Python guard at line 2497 redundant, per section 2b.

---

# 3. Policy version and evidence

`POLICY_VERSION` is `1`. Bump it when, and only when:

- a rule is added to, removed from, or retuned in `crossover_rules()`,
- a gate in `_collect_blocks` changes what it admits or refuses,
- a term of `estimate_gpu_memory` changes,
- a block or warning code changes meaning (adding a new code does not
  count, because codes are appended and never renumbered).

Do not bump it for a message reword or a docstring edit. The version
exists so a report from one release can be told apart from a report from
another, which stops working if it changes for reasons that do not change
decisions.

`evidence_id` is what a GPU selection rests on, and it takes one of:

| Value | Meaning |
|---|---|
| `none` | No GPU was selected, or the selection rests on nothing |
| `explicit-request` | The user asked for `device='gpu'` |
| `MOJOBOOST_AUTO_MIN_CELLS` | The operator's size knob, which is not a measurement |
| a rule's own id | A benchmark. This is the only validated case. |

`DeviceDecision.validated()` is True only for the last one.

## Installing a measured crossover rule

`crossover_rules()` returns an empty list, and it should keep returning one
until a sweep exists. To install a rule:

1. Run the sweep (section 6) and record it in `docs/GPU_VALIDATION.md`
   with a section anchor.
2. Add one `CrossoverEvidence` to `crossover_rules()`, scoped to exactly
   what was measured: the `api` that was measured, the
   `apple_generation` when that is what varied, the `objective` when only
   one was swept, and the thresholds. Leave every other field at its
   non-constraining default rather than widening the rule.
3. Put the document anchor in `evidence_id`, and the device in
   `measured_on`.
4. Bump `POLICY_VERSION`.

The constructor raises for an empty `evidence_id`, which is the mechanism:
a rule cannot be added without citing something.

---

# 4. Memory-estimate invariants

`MemoryEstimate` documents five, and they are what a pinning test should
assert:

1. Every term is nonnegative.
2. `device_bytes()` is the sum of the six device terms and nothing else;
   `host_bytes()` is the sum of the two host terms.
3. `upper_bound_bytes() == device_bytes() + partial_budget`, so it is
   never below `device_bytes()`.
4. The estimate is **nondecreasing** in each of `n_rows`, `n_features`,
   `n_outputs`, and `n_bins`. This is the load-bearing one: the memory
   gate compares the estimate against a budget, and a term that shrank as
   a workload grew would admit a run that does not fit.
5. `bins_known == False` means the histogram terms are zero rather than
   guessed, and such an estimate must never block a run, because it is a
   lower bound on an unknown quantity. `_collect_blocks` enforces this by
   requiring both `caps.memory_budget_known()` and `memory.bins_known`
   before it adds `BLOCK_MEMORY_BUDGET`.

The terms mirror the buffers `GpuHistogramBuilder.__init__` allocates in
`histogram_gpu.mojo`. If that constructor grows a buffer, this estimate
has to grow a term, and `POLICY_VERSION` has to move.

---

# 5. Fallback semantics

Four of them, and they are distinct:

**Unknown hardware.** A device that reports nothing gets
`GpuProfile.generic()`: 16 cores, 1024 threads per block, 16 KiB of
threadgroup memory, no memory budget, not unified. Low rather than
typical, for the reason `gpu_tiling.mojo` gives. It is not Apple-shaped
and not NVIDIA-shaped, it carries `synthetic=True`, and the decision says
`profile_source=fallback` and warns `unknown-hardware`.

**Unknown memory budget.** `memory_budget_bytes == 0` means unreported.
Memory cannot block, `partial_budget_bytes` falls back to the portable 64
MiB ceiling, and the decision warns `memory-budget-unknown`. A zero budget
is never treated as a budget of zero.

**Incomplete request.** No objective, no bin count, or both: the gates
that need them are skipped, the histogram terms of the estimate are zero,
`bins_known` is False, and the decision warns `incomplete-request`. This
is the state `resolve_device` (the four-argument compatibility entry
point) always produces, which is why it behaves exactly as the old
`resolve_device` did.

**`auto` with no evidence.** The CPU. Not a guess, not a heuristic, not a
threshold: `DECISION_AUTO_CPU_NO_EVIDENCE`, with a message that names
`MOJOBOOST_AUTO_MIN_CELLS` as the way to run the benchmark that would
change it.

And one thing that is explicitly **not** a fallback: an explicit
`device='gpu'` that cannot run. It selects `NO_DEVICE`, sets
`blocked=True`, and `raise_if_blocked()` raises. It never resolves to the
CPU, at any layer, for any reason.

---

# 6. Validation and benchmark commands, for later

None of these were run. They are what this lane would have run.

## Focused compile and unit check

```sh
nice -n 19 tools/with_build_lock.sh pixi run mojo run -I src \
    tests/parallel/test_device_policy.mojo
```

That file does not exist yet. It is the main piece of follow-up work, and
it should cover, all with injected `DeviceCapabilities` so it passes on a
CPU-only machine:

- the four mirrors in `device_policy.mojo` still equal their sources
  (`LAMBDARANK` in `ranking.mojo`, `MAX_ROWS` and `MAX_BINS` in
  `histogram_gpu.mojo`, the `[2, 256]` range in `binning.mojo`);
- every `_collect_blocks` branch, one workload each;
- explicit `gpu` raising for each block, and never resolving to the CPU;
- `auto` resolving to the CPU with the table empty, on capabilities that
  claim a GPU;
- the `MOJOBOOST_AUTO_MIN_CELLS` threshold in both directions;
- the five memory invariants above, including a monotonicity sweep;
- `serialize()` round-tripping through the Python `_parse_decision`;
- `resolve_device` reproducing the pre-change behavior exactly, which is
  the compatibility claim this lane is making.

## The existing device test

```sh
nice -n 19 tools/with_build_lock.sh pixi run mojo run -I src \
    tests/test_device.mojo
```

This one exists and **must still pass unchanged**. It imports
`AUTO_DEVICE`, `CPU_DEVICE`, `GPU_DEVICE`, `device_name`,
`env_auto_min_cells`, `gpu_available`, `parse_device`, and
`resolve_device` from `mojoboost.device`, and the facade keeps all eight
defined. If it fails, the facade is wrong, not the test.

## The crossover sweep that would justify a rule

```sh
MOJOBOOST_AUTO_MIN_CELLS=0 nice -n 19 tools/with_build_lock.sh \
    pixi run mojo run -I src bench/bench_train_gpu.mojo
```

`MOJOBOOST_AUTO_MIN_CELLS=0` is what makes `auto` reach the GPU at all.
Sweep rows and features across at least two orders of magnitude, record
both backends' wall clock per shape, and only then consider section 3.

---

# 7. What this lane knowingly broke

Both are in files it was not allowed to edit. Neither is a surprise; both
are the direct consequence of deleting the Python decision engine, which
was the task.

## 7a. `python/tests/parallel/test_device_selection.py`

54 tests, and most of them will fail. They test the Python engine: they
construct `Capabilities` fixtures and assert which device the Python
policy picks, which is exactly the thing that no longer exists. The
symbols they import that are gone:

`CROSSOVER_RULES`, `RULES_VERSION`, `GPU_OBJECTIVES`, `MAX_GPU_ROWS`,
`MIN_GPU_BINS`, `MAX_GPU_BINS`, `Capabilities`, `CrossoverRule`,
`MemoryEstimate`, `detect_capabilities`, `estimate_gpu_memory`.

They should be rewritten, not restored, and they split cleanly in two:

- The extraction and formatting tests keep their shape. `Workload.from_data`
  against numpy, pandas, a list of rows, and a duck-typed sparse matrix;
  `_shape_of` and `_count_classes` edge cases; `DeviceReport.explanation`
  and `to_dict()` against a hand-written `fields` dict, which is now the
  cheap way to test the renderer with no extension at all.
- The policy tests move to Mojo, into the `test_device_policy.mojo` above,
  where injected `DeviceCapabilities` does what injected `Capabilities`
  used to.

One new Python test is worth adding: `native_contract()` returns
`"full"`, which is the assertion that the binding in section 1 landed.

## 7b. `docs/DEVICE_SELECTION.md`

Stale. It documents `Capabilities`, `CrossoverRule`, `detect_capabilities`,
`estimate_gpu_memory`, and the `rules`/`rules_version` parameters of
`select_device` and `explain_device_choice`, none of which exist now. The
user-facing API that survives is `Workload`, `DeviceReport`,
`DeviceUnavailableError`, `select_device`, `explain_device_choice`, and
`native_contract`.

## 7c. Optional: `src/mojoboost/__init__.mojo`

Not broken, just incomplete. It re-exports the device vocabulary from
`.device` and still compiles. If the package should expose the contract to
Mojo users directly, add:

```mojo
from .device_policy import (
    DeviceCapabilities,
    DeviceDecision,
    DeviceRequest,
    MemoryEstimate,
    decide_device,
    describe_decision,
    estimate_gpu_memory,
)
```

---

# 8. Python that becomes deletable

Nothing in `device_selection.py` is dead today, but one class is dead the
moment section 1 lands:

**`_NarrowNativePolicy`** and its `CONTRACT_NARROW` constant. It exists
only because `decide_device` is not bound yet. It runs no policy of its
own (it calls the native `resolve_device`, which runs the same engine),
but it can only report the backend and the refusal text, so every report
it produces is marked `contract="narrow"` and `complete=False`.

Deleting it is one edit: remove the class, remove `CONTRACT_NARROW`,
remove the `resolve` branch of `_policy()`, and make a missing
`decide_device` raise `NativePolicyUnavailable`. Then
`DeviceReport.complete` reduces to the `memory_estimate_complete` check
and `DeviceReport.contract` can go too.

Also deletable at that point, from `python/mojoboost/__init__.py` and only
after a test asserts the native equivalent: the ranker's CPU-only guard
(line 3391), the eval-set guard (line 1510), and the custom-objective
guard (line 2497). See section 2b.

---

# 9. Review checklist for whoever compiles this first

In rough order of how likely each is to be the thing that breaks:

1. `device.mojo` aliases an imported `comptime` (`comptime CPU_DEVICE =
   _CPU_DEVICE`). If Mojo rejects that, restate the three integer literals
   in the facade with a comment pointing at the source of truth.
2. `comptime EVIDENCE_NONE = String("none")` follows the
   `comptime SUPPORTED_KEYS = String(...)` precedent in `params.mojo`,
   but the three evidence constants are also assigned into a local
   (`var evidence = EVIDENCE_NONE`) and transferred with `^`. If that
   fights the comptime-ness, make them plain `def` functions returning a
   `String`.
3. `ReasonList.add` takes `var message: String` and appends with `^`.
4. `decide_device` builds its message into a local and transfers
   `message^`, `blocks^`, `warnings^`, `memory^`, `evidence^` into one
   constructor call at the end. There is exactly one construction site, so
   an ownership error will show up once, not nine times.
5. `String(...)` with mixed literal and `Int` arguments is used throughout,
   which is the idiom `apple_gpu_policy.describe_policy` already uses.
6. `estimate_gpu_memory` raises before `_collect_blocks` runs, so an
   impossible shape is an error and not a block. That is why there is no
   `BLOCK_INVALID_SHAPE`.

## Behavior compatibility claim

For the four-argument `resolve_device` path, this change is intended to be
a no-op:

| Request | Before | After |
|---|---|---|
| `cpu` | `CPU_DEVICE` | `CPU_DEVICE`, `DECISION_EXPLICIT_CPU` |
| `gpu`, no accelerator | raise | raise, `BLOCK_NO_ACCELERATOR` |
| `gpu`, accelerator | `GPU_DEVICE` | `GPU_DEVICE`, `DECISION_EXPLICIT_GPU` |
| `auto`, threshold unset | `CPU_DEVICE` | `CPU_DEVICE`, `DECISION_AUTO_CPU_NO_EVIDENCE` |
| `auto`, cells >= threshold | `GPU_DEVICE` | `GPU_DEVICE`, `DECISION_AUTO_GPU_ENV_THRESHOLD` |
| `auto`, cells < threshold | `CPU_DEVICE` | `CPU_DEVICE`, `DECISION_AUTO_CPU_BELOW_ENV_THRESHOLD` |
| unknown device code | raise | raise |

The old `gpu_supports(n_outputs)` returned `n_outputs >= 1` and so never
refused anything; `gpu_supports_outputs` is the same predicate with the
same docstring rationale, so multiclass keeps reaching the GPU. If the
table above does not hold, the facade is wrong.
