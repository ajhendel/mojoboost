# Device selection

`device="auto"` has to answer one question, "CPU or GPU for this run", and
it has to be able to say why. This document describes the policy, the
evidence rule behind it, and the report it produces.

The policy lives in `python/mojotrees/device_selection.py`. The device
vocabulary it implements is the one in `src/mojotrees/device.mojo`, which
is the authority; this layer adds explanation, Python-level feature gates,
a memory estimate, and a versioned table of measured crossovers.

## The three values

| Value | Behavior |
|---|---|
| `"cpu"` | The default and the dependable path. Always resolves to itself. Float64 throughout, every objective, every input. |
| `"gpu"` | An explicit request. It runs on the accelerator or it raises. There is no fallback. |
| `"auto"` | The policy below. It picks the GPU only when the GPU path covers the workload and a validated crossover rule says the GPU is faster for that shape on that backend. Otherwise the CPU. |

Names are case insensitive, as LightGBM treats `device_type`. Anything
outside the three raises `ValueError`.

Why `"gpu"` never falls back: a silent fallback turns "my GPU run" into "a
CPU run that took the same wall clock and I never knew". A refusal is
information; a fallback destroys it.

## What `auto` currently does, and why

**`auto` resolves to the CPU on every machine and every workload, unless
`MOJOTREES_AUTO_MIN_CELLS` is set.**

`CROSSOVER_RULES` is empty. Nothing in this repository has measured a
workload size where GPU training beats CPU training. The one end-to-end
measurement that exists is on an Apple M4 (`bench/bench_train_gpu.mojo`)
and it came out slower than the CPU trainer, and no NVIDIA or AMD device
has ever executed this code at all (see `docs/GPU_VALIDATION.md`, where
every CUDA and HIP row still reads **not run**).

A threshold written from reasoning rather than measurement would be a
performance claim with no evidence under it. So the table ships empty,
`auto` conservatively chooses the CPU, and the report says exactly that
rather than implying the GPU was evaluated and lost.

```text
device='auto' resolved to CPU.

Device      accelerator available, metal, Apple M4
Workload    1,000,000 rows x 100 features, objective 'regression', 1 output(s) per round, max_bin=255, dense
Memory      107.1 MiB device, 171.1 MiB including the tiled partial-histogram budget, 7.9 MiB pinned host (estimate)
Budget      16.0 GiB
Rules       version 1, 0 rule(s), none matched

Why
  [no-validated-rule] the crossover table (version 1) is empty: no
  benchmark has established a workload size where GPU training beats CPU
  training, so auto conservatively keeps the CPU. Set
  MOJOTREES_AUTO_MIN_CELLS to run that benchmark, or device='gpu' to force
  the accelerator
```

## Hard blocks and soft uncertainty

Two different things keep a workload off the GPU, and conflating them
would either refuse runs that work or promise runs that do not.

A **hard block** is something that will actually fail. Explicit `"gpu"`
raises on it and `"auto"` takes the CPU:

| Block | Reason code | Enforced by |
|---|---|---|
| No accelerator for this build | `no-accelerator` | `gpu_available` in `src/mojotrees/device.mojo` |
| `MOJOTREES_DISABLE_GPU=1` | `gpu-disabled-env` | the same function |
| Sparse input | `unsupported-feature` | `_fit_sparse` in `python/mojotrees/__init__.py` |
| A custom objective callable | `unsupported-feature` | `_fit_custom` in the same file |
| An `eval_set` (validation is scored on the CPU) | `unsupported-feature` | `_fit_with_metrics` in the same file |
| `lambdarank` | `unsupported-feature` | the ranker's `fit` |
| Multiclass on a build whose GPU path lacks it | `unsupported-feature` | `gpu_supports` in `device.mojo` |
| More rows than the kernels can index | `workload-limit` | `MAX_ROWS` in `src/mojotrees/histogram_gpu.mojo` |
| `max_bin` outside [2, 256] | `workload-limit` | `MAX_BINS` there and the binner |
| The memory estimate does not fit the budget | `insufficient-memory` | the estimate below |

**Soft uncertainty** is a workload nobody has measured or documented as
covered, most often an objective outside the set `device.mojo` names
(squared error, binary logistic, poisson, huber, quantile, L1). It never
blocks an explicit `"gpu"` request, because refusing a run the native
layer would have accepted is its own kind of lie. It does keep `"auto"` on
the CPU, since choosing the GPU on an uncharacterized path is exactly the
guess `auto` exists to not make. An unidentifiable backend is soft for the
same reason: no crossover rule can be scoped to a device nobody can name.

## Environment variables

| Variable | Effect |
|---|---|
| `MOJOTREES_DISABLE_GPU=1` | Reports no accelerator. `"gpu"` raises, `"auto"` takes the CPU on a machine that does have one. |
| `MOJOTREES_AUTO_MIN_CELLS` | Cells (`n_rows * n_features`) at or above which `auto` selects the GPU. `0` means "whenever the GPU path covers the workload". Unset, negative, or unparsable means the heuristic is off, which is the default. |
| `MOJOTREES_GPU_BACKEND` | Names the backend when detection cannot. Only scopes crossover rules; it never enables or disables anything. |

`MOJOTREES_AUTO_MIN_CELLS` is parsed here exactly as `env_auto_min_cells`
parses it in `device.mojo`, so the two layers cannot disagree about what a
given environment does. It is the knob for running the crossover benchmark
that would justify a default. A GPU choice reached through it is reported
with `validated == False` and a warning saying the choice rests on no
measurement.

## The crossover rule table

```python
RULES_VERSION = 1
CROSSOVER_RULES = ()
```

A `CrossoverRule` is a claim about measured performance, so it carries the
measurement with it. `evidence` is required and the constructor refuses a
rule without it.

| Field | Meaning |
|---|---|
| `name` | Short identifier, shown in the report. |
| `evidence` | Where the numbers live: a document section, a benchmark file, a commit. Required. |
| `measured_on` | The device the numbers came from. |
| `backend`, `chip` | Scope. Unset means the rule is not limited that way. |
| `objectives` | The objectives that were benchmarked. |
| `min_rows`, `min_features`, `min_cells` | The thresholds themselves. |
| `max_classes` | Upper bound on classes the measurement covered. |
| `speedup` | What was actually seen, for the record. |

A rule matches only when every field that is set matches, so widening a
rule to hardware nobody measured takes a deliberate edit rather than an
oversight.

### Adding a rule

Adding a rule is a benchmarking result, not a code change:

1. Run the sweep. `pixi run gpu-validate` for the phase breakdown and
   `bench/bench_train_gpu.mojo` for end-to-end training, on the device the
   rule will claim, following the procedure in `docs/GPU_VALIDATION.md`.
   Pin CPU threading (`MOJOTREES_NUM_WORKERS`, `MOJOTREES_PARALLEL_MIN_OPS`)
   so the CPU side of the comparison is reproducible.
2. Find the crossover with `MOJOTREES_AUTO_MIN_CELLS`, which exists so the
   sweep needs no rebuild.
3. Record the output in the record section of `docs/GPU_VALIDATION.md`,
   loss numbers next to throughput numbers.
4. Add the rule scoped to what was measured, cite that record in
   `evidence`, and bump `RULES_VERSION`.

Do not add a rule from reasoning, from another project's numbers, or from
a single shape. `python/tests/parallel/test_device_selection.py` asserts
that the shipped table is empty; that test failing is the reminder to
bring evidence.

## The memory estimate

Every report carries an estimate of what one GPU training session would
allocate, derived term by term from the buffers
`GpuHistogramBuilder.__init__` creates in `src/mojotrees/histogram_gpu.mojo`:

| Term | Bytes | Buffer |
|---|---|---|
| `binned_matrix` | `n_rows * n_features` | `bins_dev`, uint8 |
| `leaf_ids` | `n_rows * 4` | `leaf_dev`, int32 |
| `gradients` | `n_rows * 4 * n_outputs` | `grad_dev`, float32 |
| `hessians` | `n_rows * 4 * n_outputs` | `hess_dev`, float32 |
| `histograms` | `n_features * n_bins * 12` | `out_dev`, three int32 planes |
| `feature_ids` | `n_features * 4` | `feat_dev`, int32 |

Plus, pinned on the host, two float32 staging planes of `n_rows` and one
copy of the histogram buffer.

The tiled accumulation strategy also allocates a partial-histogram buffer
whose size comes from device attributes read at runtime, so it cannot be
computed host-side without a device. `PARTIAL_BUDGET_BYTES` in
`src/mojotrees/gpu_tiling.mojo` caps it at 64 MiB, and that cap is what
`upper_bound_bytes` adds.

It is an estimate and it is labeled one everywhere it appears. It counts
training buffers, not allocator overhead, and the `n_outputs` factor on
the gradient planes is an upper bound that assumes every class plane is
resident at once. It blocks a run only when a device memory budget is
known and the estimate exceeds it. When the budget is unknown the report
says so and memory is not a factor. On a unified memory backend the budget
shown is installed host RAM, which the report also says.

## The report

`select_device(device, workload, capabilities=None, rules=None)` returns a
`DeviceReport` and raises `DeviceUnavailableError` (a `RuntimeError`
subclass) when an explicit `"gpu"` cannot run.

`explain_device_choice(X, y=None, device="auto", **workload_kwargs)` is
the same policy in a form that never raises: a request that would fail
comes back with `would_raise=True` and the message in `error`, so "what
would `device='gpu'` do here" is answerable without try/except.
`report.raise_if_unsupported()` turns it back into the raise.

```python
from mojotrees.device_selection import explain_device_choice

report = explain_device_choice(X, y, device="auto")
print(report)                            # the prose explanation
report.to_dict()["resolved"]             # "cpu" or "gpu"
[r.code for r in report.reasons]         # the stable reason codes
report.memory.upper_bound_bytes          # what a GPU run would allocate
report.validated                         # a rule backed this GPU choice
```

Report fields:

| Field | Meaning |
|---|---|
| `requested` | The normalized request, `"cpu"`, `"gpu"`, or `"auto"`. |
| `resolved` | `"cpu"`, `"gpu"`, or None when the request cannot run. |
| `reasons` | Ordered `Reason(code, message)` records, the decision itself. |
| `capabilities`, `workload`, `memory` | Everything the decision rested on. |
| `rules_version`, `rules_considered`, `matched_rule` | Which table was consulted and what it said. |
| `validated` | True only for a GPU chosen by a rule with evidence. |
| `would_raise`, `error` | Set instead of raising, in `explain_device_choice`. |
| `warnings` | Choices that are legitimate but unbacked, such as an explicit GPU request. |

`to_dict()` and `to_json()` are JSON-serializable, which is what a support
ticket or a CI log wants. `explanation` renders the same content as prose
and `str(report)` is that explanation.

Reason codes are stable strings, safe to match on: `explicit-cpu`,
`explicit-gpu`, `no-accelerator`, `gpu-disabled-env`,
`unsupported-feature`, `unsupported-objective`, `workload-limit`,
`insufficient-memory`, `unvalidated-path`, `no-validated-rule`,
`rule-matched`, `below-rule-threshold`, `env-threshold`,
`below-env-threshold`.

## Injecting capabilities

`Capabilities` is data, never a probe, so a caller describes a machine it
does not have and the policy cannot tell the difference. That is how the
tests cover CUDA and HIP devices nobody here owns, and it is how a user
can ask "would a bigger GPU change this answer".

```python
from mojotrees.device_selection import Capabilities, Workload, select_device

caps = Capabilities(
    gpu_available=True,
    backend="cuda",
    chip="NVIDIA L4",
    device_memory_bytes=24 * 1024**3,
)
report = select_device("auto", Workload(5_000_000, 200), capabilities=caps)
```

`detect_capabilities()` fills one in from the real environment. It asks
the compiled extension whether an accelerator is available, reads the
three environment variables, and identifies the backend and chip when the
platform will name them. Backend detection is a heuristic and says so
through `backend_source`; `MOJOTREES_GPU_BACKEND` overrides it. Anything
it cannot determine comes back None and the report prints "unknown"
rather than a plausible value.

Note that accelerator availability is a property of the build, not of the
running machine: Mojo resolves `has_accelerator()` at compile time. A
redistributed wheel built where a device was present reports one as
available, so a `"gpu"` request there fails when the device is opened
rather than when it is resolved. `build_has_accelerator` records what the
build claimed, and `MOJOTREES_DISABLE_GPU=1` is the way to pin such a
build to the CPU.

## Status

This module is the policy and report layer. It is not yet wired into the
estimators; `MojoTreesRegressor(device="auto")` still resolves through
`_resolve_device`, which calls the native `resolve_device` directly. The
exact wiring, including why the estimators must keep passing a resolved
concrete device name to the native layer rather than `"auto"`, is in
`handoffs/apple_a9_device_selection.md`.
