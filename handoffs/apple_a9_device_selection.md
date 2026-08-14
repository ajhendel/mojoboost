# Handoff: explainable automatic device selection (Apple task A9)

Policy and report layer for `device="auto"` and `explain_device_choice(X)`.
Implemented, tested, and not yet wired into anything. This file is the
exact wiring for whoever integrates it.

## Files this lane owns

| File | State |
|---|---|
| `python/mojoboost/device_selection.py` | New. The whole policy, 1,461 lines with docstrings, standard library only. |
| `python/tests/parallel/test_device_selection.py` | New. 54 tests, all injecting fake capabilities. |
| `docs/DEVICE_SELECTION.md` | New. The user-facing document. |
| `handoffs/apple_a9_device_selection.md` | This file. |

Nothing else was touched. No estimator, no binding, no Mojo module, no
`python/mojoboost/__init__.py`, no `README.md`, no `pixi.toml`.

## Focused test

```sh
nice -n 19 tools/with_build_lock.sh pixi run -e pytest pytest -q \
    python/tests/parallel/test_device_selection.py
```

Result: `54 passed, 3 warnings in 2.66s`. The three warnings are the
existing `DeprecationWarning: builtin type Model has no __module__
attribute` from importing the compiled extension, not from this lane.

`git diff --check --` on the four assigned paths reports nothing.

The test file loads the module through `from mojoboost import
device_selection` and falls back to loading the file directly when the
package cannot be imported, because the package `__init__` imports the
compiled extension and this layer needs nothing from it. On this machine
the package import path is the one that ran.

## What the module provides

```python
from mojoboost.device_selection import (
    Capabilities, CrossoverRule, DeviceReport, DeviceUnavailableError,
    MemoryEstimate, Reason, Workload,
    CROSSOVER_RULES, RULES_VERSION,
    detect_capabilities, estimate_gpu_memory, explain_device_choice,
    select_device,
)
```

- `select_device(device, workload, capabilities=None, rules=None,
  rules_version=None)` returns a `DeviceReport` whose `resolved` is
  `"cpu"` or `"gpu"`. Raises `ValueError` for an unknown device name and
  `DeviceUnavailableError` (subclass of `RuntimeError`) when explicit
  `"gpu"` cannot run.
- `explain_device_choice(X, y=None, device="auto", capabilities=None,
  rules=None, rules_version=None, **workload_kwargs)` never raises for an
  unsupported GPU request; the report carries `would_raise` and `error`,
  and `report.raise_if_unsupported()` converts it back.
- `DeviceReport.to_dict()` / `.to_json()` for structure,
  `.explanation` / `str(report)` for prose.

`CROSSOVER_RULES` is empty and `RULES_VERSION` is 1. See
`docs/DEVICE_SELECTION.md` for why, and for the procedure that would add
a rule.

## Central integration required

None of this is done. Each item is an edit to a file this lane may not
touch.

### 1. Export from `python/mojoboost/__init__.py`

Two lines, both in the shared-hotspot file:

```python
# with the other imports at the top of the module
from . import device_selection  # noqa: F401  (submodule, imported for
                                # `mojoboost.device_selection` access)

# in __all__, alongside "gpu_available"
    "explain_device_choice",
```

and re-export the function itself:

```python
from .device_selection import explain_device_choice  # noqa: F401
```

`device_selection` imports nothing from the package at module scope (the
extension probe is a function-local import inside `_native_gpu_available`),
so this cannot create a cycle.

### 2. Replace the body of `_Base._resolve_device`

Currently at `python/mojoboost/__init__.py:1275`. It validates the name
and delegates to `_mojoboost.resolve_device`. The replacement runs the
Python policy first and passes the concrete result to the native layer:

```python
    def _resolve_device(self, n_rows, n_features, n_outputs, **workload):
        """The backend that will actually run, "cpu" or "gpu"."""
        device = self._resolve_alias("device", "device_type", "cpu")
        report = device_selection.select_device(
            device,
            device_selection.Workload(
                n_rows=n_rows,
                n_features=n_features,
                n_classes=n_outputs if n_outputs > 1 else 1,
                max_bin=self.max_bin,
                **workload
            ),
        )
        self._device_report = report
        return report.resolved
```

Notes for whoever applies it:

- **Keep passing the resolved concrete name to the native layer.** The
  estimators already do this (`_parse_device` in
  `bindings/_mojoboost.mojo` says so). It is what keeps the two policies
  from disagreeing: a `"gpu"` chosen here is requested as `"gpu"`
  natively and therefore runs or raises on native terms too. Passing
  `"auto"` through would discard this policy entirely, because the native
  `auto` gate is only `MOJOBOOST_AUTO_MIN_CELLS`.
- `select_device` raises `ValueError` for an unknown name and
  `DeviceUnavailableError` for a refused GPU, which is exactly what the
  current method raises (`ValueError` and `RuntimeError`). No caller
  needs to change.
- The `n_classes` mapping above reproduces the current call convention:
  the classifier already passes `1 if n_classes == 2 else n_classes`, so
  `n_outputs > 1` means genuine multiclass.
- The call sites are `__init__.py:2408` (regressor), `:2802`
  (classifier), and `:3366` (ranker). The ranker should pass
  `objective="lambdarank"` and the classifier `objective="binary"` or
  `"multiclass"`; the regressor should pass its own `self.objective` and
  `custom_objective=callable(self.objective)`.
- Pass `has_eval_set=eval_set is not None` where the call site knows it.
  Today the eval_set restriction is enforced later, at
  `__init__.py:1501`, with a message the policy layer reproduces; moving
  the check earlier makes the failure arrive before any binning work.

### 3. Feed the report to `_record_fit`

`_record_fit` (at `:1836`) sets `self.device_`. Adding
`self.device_report_ = self._device_report` there gives a fitted
estimator the full explanation of the device it used, which is the thing
a support ticket actually needs. Optional, but cheap.

### 4. Module-level `explain_device_choice`

The public spelling users will reach for is
`mojoboost.explain_device_choice(X, y, device="auto")`. Once item 1 is
done that is a re-export and nothing more. Consider also
`MojoBoostRegressor.explain_device_choice(X, y)` as a thin method that
fills `objective`, `max_bin`, and `custom_objective` from the estimator's
own parameters, so the user does not restate them.

### 5. Documentation

- `README.md` device paragraph: add a pointer to
  `docs/DEVICE_SELECTION.md`. Not done here (shared hotspot).
- `python/mojoboost/__init__.py:215-221` already describes the device
  vocabulary and points at `src/mojoboost/device.mojo`. Add the new
  document beside it.
- `docs/LIGHTGBM_PARITY.md` records `device_type` parity. `auto` is a
  mojoboost addition with no LightGBM equivalent; if the parity table
  wants a row for `explain_device_choice`, it belongs in the
  "extensions" section, not the parity section. Not edited here.

## Drift found while reading, not fixed

Report these to whoever owns the files; this lane may not edit them.

1. **Multiclass GPU coverage is documented two ways.**
   `src/mojoboost/device.mojo` says `gpu_supports` returns True for every
   `n_outputs >= 1` and that multiclass now trains on the device through
   `train_multiclass_gpu`. The classifier docstring at
   `python/mojoboost/__init__.py:2632` still says "Multiclass training is
   CPU-only, so `device="gpu"` raises for 3 or more classes". One of the
   two is stale. This lane followed `device.mojo`, since that is what
   `resolve_device` actually executes, and made it a capability flag
   (`Capabilities.supports_multiclass`, default True) so flipping the
   answer is a one-line change rather than a rewrite.
2. **`device.mojo`'s objective list may be narrower than reality.** The
   module docstring names squared error, binary logistic, poisson, huber,
   quantile, and L1, but `train_gpu` calls the shared `_check_objective`
   and `train_custom_gpu` exists, which suggests wider coverage than the
   docstring claims. Because of that uncertainty, an objective outside
   the documented set is treated as soft uncertainty here: it keeps
   `auto` on the CPU but never blocks an explicit `"gpu"`. If someone
   confirms the real set, update `GPU_OBJECTIVES` in
   `device_selection.py` and the docstring in `device.mojo` together.
3. **Custom objectives.** `train_custom_gpu` exists in Mojo, but the
   Python estimator hard-refuses any device other than CPU for a custom
   objective (`_fit_custom`, `:2477`). This lane mirrors the Python
   behavior, which is what users hit. If the estimator ever wires
   `train_custom_gpu` up, flip
   `Capabilities.supports_custom_objective`.

## Deliberate design decisions

- **The rule table ships empty.** The task said never to invent crossover
  thresholds, and there is nothing to ship: the only end-to-end GPU
  training measurement in the repository is slower than the CPU trainer,
  and `docs/GPU_VALIDATION.md` still lists every CUDA and HIP row as not
  run. `test_shipped_rule_table_is_empty` fails if a rule ever appears,
  which is the reminder to bring evidence with it.
- **`CrossoverRule` refuses to be constructed without `evidence`.** A
  rule is a performance claim; the citation is a required field, not a
  convention.
- **Hard blocks versus soft uncertainty.** Refusing a run the native
  layer would have accepted is as wrong as promising one it would not.
  Only the gates that something already enforces are hard.
- **`MOJOBOOST_AUTO_MIN_CELLS` is parsed exactly as `device.mojo` parses
  it**, including treating unset, negative, and unparsable identically.
  `test_env_parsing_matches_the_native_rules` pins that, and it is the
  test to update if the native parsing ever changes.
- **Memory is an estimate and never silently decides anything.** It
  blocks only when a budget is known and the estimate exceeds it, and it
  is labeled an estimate in every rendering.
- **Capabilities are injected data, never probes.** That is what let this
  lane test CUDA and HIP behavior on a machine with neither.

## Risks

- The estimator integration in item 2 changes where a GPU refusal is
  raised (earlier, before binning) and its message text. Any existing
  test asserting the exact wording of a device error will need its string
  updated; the codes stay stable, the prose does not.
- `detect_capabilities()` runs one short `sysctl` subprocess on macOS to
  read the chip name, guarded by try/except and a 2 second timeout. If
  even that is unwanted in a library, pass `Capabilities` explicitly or
  drop `_detect_chip` to `platform.processor()` everywhere; the chip name
  only scopes crossover rules and none exist yet.
- Backend detection on Linux is a filesystem heuristic
  (`/proc/driver/nvidia/version`, `/sys/module/amdgpu`). It reports None
  rather than guessing when neither is present, and
  `MOJOBOOST_GPU_BACKEND` overrides it. It has never run on a Linux GPU
  machine, because none is available here.
- A sibling lane in this round added `src/mojoboost/apple_gpu_policy.mojo`,
  a Mojo launch-plan policy that also derives a partial-histogram budget
  (a fraction of the reported budget on unified memory, under the same
  portable 64 MiB ceiling). This module's `MemoryEstimate` still adds the
  flat `PARTIAL_BUDGET_BYTES` cap as its upper bound, which stays a true
  upper bound either way but is looser than what that policy would
  actually allocate on Apple silicon. If both land, have
  `estimate_gpu_memory` take the partial budget as a parameter and feed
  it the number that policy computes, rather than duplicating the
  fraction here.
- The memory model tracks the buffers in
  `histogram_gpu.mojo:465-497`. A change to those allocations should
  update `estimate_gpu_memory` and
  `test_memory_components_follow_the_gpu_buffers` together. The
  multiclass gradient scaling is a stated upper bound, not a reading of
  `train_multiclass_gpu`.
