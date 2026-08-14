# Connect 06: GPU prediction and validation through the Python bindings

Lane scope: `src/mojoboost/gpu_predict.mojo`, `bindings/_mojoboost.mojo`,
this handoff. Nothing was committed by this lane. Nothing outside the owned
files was edited; everything else is a patch request below.

## 1. Implementations found

One authoritative GPU prediction implementation, already complete, already
tested, and connected to nothing that Python can reach.

| Where | What it is | State before this lane |
| --- | --- | --- |
| `src/mojoboost/gpu_predict.mojo` | `GpuPredictor`, `_predict_kernel`, `_leaf_kernel`, `_response_kernel`, `_metric_kernel`, `FlatEnsemble`, `flatten_*`, four one-shot `predict_*_gpu` helpers | The authoritative implementation. Batch scoring, response transforms, leaf ordinals, and a resident validation set with a device-side metric reduction are all present. |
| `src/mojoboost/model.mojo` | `Model.predict_batch` / `MulticlassModel.predict_batch` | The one place that chooses between the host walk and `GpuPredictor`, via `resolve_device`. Reachable from Mojo only. |
| `src/mojoboost/train_gpu.mojo` | `GpuValidScorer` trait, `_HostValidScorer`, `_DeviceValidScorer`, `device_loss_metric` | Resident validation scoring across a training run, inside the Mojo trainer. Selected by `MOJOBOOST_GPU_VALID_SCORING`, defaulting to the host. |
| `src/mojoboost/gpu_fused_round.mojo` | imports `NODE_STRIDE`, `_append_tree`, `_leaf_kernel` from `gpu_predict` | Already a consumer of the same flat layout, not a second one. |
| `bindings/_mojoboost.mojo` | `predict`, `predict_raw`, `predict_proba`, `predict_range`, `predict_proba_range`, `predict_leaf`, `predict_leaf_multiclass`, `predict_*_csr` | Every one of them a host row-at-a-time walk. **No import of `gpu_predict` existed.** |

No duplicate predictor, no second flat ensemble, no alternate device policy
was found. The gap was entirely one of connection: `device="gpu"` was a
training-only setting because prediction never left the host.

## 2. Call path, before and after

Before:

```
estimator.predict(X)
  -> _mojoboost.predict_range(model, x_addr, ..., out_addr)
     -> Model.predict_range(row, rng)  per row      [host only, device ignored]
```

After (the established path is still there, unchanged):

```
estimator.predict(X, device=...)
  -> _mojoboost.predict_batch(model, x_addr, n_rows, n_features, params, out_addr)
     -> _predict_device(params, ...)                 [gpu_predict_support + resolve_device]
     -> Model.predict_batch(features, n_rows, rng, raw, device)
        -> GPU: predict_gpu / predict_raw_gpu        [gpu_predict.GpuPredictor]
        -> CPU: the host walk it always had
     <- returns the name of the backend that ran
```

Validation, driven from Python:

```
handle = _mojoboost.gpu_validation_open(model, x_addr, n, f, params)
   -> GpuPredictor(n_features, n_outputs).set_validation(bins, y, w)
   -> reset_validation([model.base_score])
per round:
   _mojoboost.gpu_validation_accumulate(handle, model, start, stop)
      -> accumulate_booster_rounds -> upload the round's trees, accumulate_round
   _mojoboost.gpu_validation_metric(handle, metric_code, objective_code)
      -> validation_host_metric -> _response_kernel + _metric_kernel
```

## 3. Connections completed

### `src/mojoboost/gpu_predict.mojo`

- **Capability and refusal record.** `GpuPredictSupport` and
  `gpu_predict_support(n_rows, n_features, n_outputs, n_bins, sparse)`.
  The block codes and names are `device_policy.mojo`'s `BLOCK_*` and
  `block_reason_name`, not a second vocabulary. Gates: availability
  (`gpu_available` / `gpu_disabled_by_env`), dense-only, the Int32 row
  ceiling (`MAX_GPU_ROWS`), the UInt8 bin range (`MIN_GPU_BINS` /
  `MAX_GPU_BINS`), and `gpu_supports_outputs`. Nonpositive `n_bins` means
  undeclared, the same rule `_normalized_bins` applies to every other flat
  boundary into the policy.
- **Host metric codes on the device.** `device_metric_code`,
  `device_metric_is_error`, `device_metric_matches_host`, and
  `validation_host_metric`. These translate `metrics.mojo`'s metric codes
  (the ones `eval_metric` already takes) into this module's `METRIC_*`
  kernels, so no caller carries a second numbering. Six of twenty-one
  built-in metrics have a kernel; the rest raise and name themselves.
  `device_metric_matches_host` is True only for `l2` and `l1`, and the
  docstring records why the log losses and the error rates are not
  (the `_F32_PROB_FLOOR` clamp; discrete threshold and argmax flips).
- **Batched leaf indices.** `leaf_indices_gpu`, `leaf_indices_multiclass_gpu`,
  the one-shot forms of `GpuPredictor.leaf_indices`, matching the existing
  `predict_*_gpu` helpers in shape and behavior.
- **Round accumulation.** `accumulate_rounds`, `accumulate_booster_rounds`,
  `accumulate_multiclass_rounds`: turn "the iterations this model gained"
  into one upload and N `accumulate_round` calls, with zero base scores
  because `reset_validation` owns the base score. Empty slice is a no-op.

### `bindings/_mojoboost.mojo`

Fourteen new module functions, all registered in `PyInit__mojoboost`, plus
one new Python type.

| Function | Contract |
| --- | --- |
| `predict_batch(model, x_addr, n_rows, n_features, params, out_addr)` | Single-output scores into a float64 buffer of length `n_rows`. Returns the backend name that ran, or `None` for an empty batch. |
| `predict_proba_batch(...)` | Multiclass, row-major `[r * n_classes + k]`, buffer `n_rows * n_classes`. Same return. |
| `predict_leaf_batch(...)` | Leaf ordinals `[r * n_iterations + i]`. Returns `None` for an empty batch or an empty range. |
| `predict_leaf_multiclass_batch(...)` | Leaf ordinals, round-major within a row, column `i * n_classes + k`. |
| `gpu_predict_capability(params)` | `[supported, block_code, reason_name, message]`. Asks without requesting. |
| `gpu_validation_open(model, x_addr, n_rows, n_features, params)` | Opaque `GpuValidation` handle, matrix binned by the model's own mapper, raw scores seeded with `booster.base_score`. |
| `gpu_validation_open_multiclass(...)` | Same, seeded with `booster.base_scores`. |
| `gpu_validation_shape(handle)` | `[n_rows, n_outputs]`. |
| `gpu_validation_reset(handle, base_addr)` | Reseed from a float64 buffer of length `n_outputs`. |
| `gpu_validation_accumulate(handle, model, start, stop)` | Fold `[start, stop)` into the resident scores. |
| `gpu_validation_accumulate_multiclass(handle, model, start, stop)` | Same, whole rounds of `n_classes` trees. |
| `gpu_validation_metric(handle, metric, objective)` | Device metric value; `metric` is `eval_metric`'s code, `objective` selects the link (softmax for a multiclass handle). |
| `gpu_validation_metric_matches_host(metric)` | `[has_kernel, matches_host]`. |
| `gpu_validation_raw(handle, out_addr)` | Copy the resident raw scores home, `[r * n_outputs + k]`. |

`params` keys, exactly:

- dense score/leaf: `device` (`"cpu"`, `"gpu"`, `"auto"`), `start`, `stop`,
  and `raw_score` (int flag, score entry points only).
- `gpu_predict_capability`: `n_rows`, `n_features`, `n_outputs`, `n_bins`
  (0 or -1 when unknown), `sparse` (int flag).
- `gpu_validation_open*`: `y_addr`, `weight_addr` (0 for unweighted).

Device residency without device pointers: `GpuValidation` holds the
`GpuPredictor`; nothing about a `DeviceBuffer` crosses the boundary. Scores
leave only as a metric value or as a copy into a caller-owned float64
buffer. The handle does not retain the model: each accumulate copies the
trees it needs, so a model can be updated, copied, or dropped underneath a
live handle. Device memory is released when Python drops the handle.

Model ownership, feature-count checks, missing and categorical routing, and
multiclass shapes are all preserved because none of them were reimplemented:
binning stays `BinMapper.transform` on the host, routing stays the kernel's
single rule, and shapes come from `Model` / `MulticlassModel`.

## 4. Duplicates fused or quarantined

- **Fused.** `predict_leaf` and `predict_leaf_multiclass` had their bodies
  extracted into `_leaf_host` and `_leaf_multiclass_host`, and the new
  device-aware entry points fall back to exactly those functions rather than
  to a second copy of the ordinal-table walk. Behavior and signatures are
  unchanged.
- **Not created.** No `src/mojoboost/gpu_validation.mojo`. `GpuPredictor`
  already is the resident validation implementation and `train_gpu.mojo`
  already drives it; a new module would have been the duplicate this lane
  exists to avoid. The Python-facing glue is a thin handle in the bindings.
- **Not created.** No `bindings/gpu_prediction.mojo`. See the layout note in
  section 6: the bindings directory is mid-migration to multiple files owned
  by other lanes, and `build.sh` compiles `_mojoboost.mojo` alone today.
- **Device selection is not duplicated.** The bindings ask
  `resolve_device` (device.mojo) once and hand the concrete code to
  `Model.predict_batch`, which resolves a concrete code to itself. The
  established convention in `_parse_device`'s docstring ("Python passes an
  already-resolved device; the trainer resolves it again, so the policy in
  device.mojo stays the only one") is the one followed.

## 5. Remaining disconnections

1. **No estimator calls any of this yet.** That is Task 07 (section 7).
   Until then the new entry points are reachable but unused, and
   `device="gpu"` still has no effect on `MojoBoostRegressor.predict`.
2. **`contrib` (TreeSHAP) has no device path.** `gpu_predict.mojo`'s
   docstring documents why (per-node weight vectors, a different kernel and
   memory budget). `predict_contrib*` stay host-only and take no device.
3. **Sparse prediction has no device path, by design.** The three
   `predict_*_csr` entry points now refuse an explicit `device="gpu"`
   through `gpu_predict_support` instead of ignoring the key. They still
   default to the CPU when no `device` key is present, which is what every
   current caller means.
4. **`resolve_device_full` is not used here.** The device-policy lane added
   flat entry points (`resolve_device_full`, `decide_device_report`) that
   carry the sparse, bin, objective, and memory gates. They are shaped for
   *training*: the memory estimate sizes histograms and gradient buffers,
   and the objective gates refuse objectives that predict perfectly well on
   the device. Using them for prediction would produce false refusals, so
   this lane used the narrow `resolve_device` plus a prediction-scoped
   capability record. See the patch request in section 6.
5. **Validation buffers are per predictor, not pooled.** A second handle on
   the same matrix re-uploads it. This is the same limit `GpuPredictor`'s
   session constructor already documents.
6. **`unified_memory_policy.mojo` refers to `GpuPredictor.upload_validation`**,
   a name that does not exist (it is `set_validation`). Documentation only,
   in another lane's file. Patch request below.

## 6. Cross-lane patch requests (exact)

**R1 — `bindings/build.sh` (not owned; needed by several lanes, not by
this one).** The new `bindings/*_bindings.mojo` files import
`binding_support`, which the current command cannot resolve:

```sh
pixi run mojo build --emit shared-lib -I src -I bindings \
    bindings/_mojoboost.mojo -o python/mojoboost/_mojoboost.so
```

This lane deliberately does not depend on it: every symbol it added lives in
`bindings/_mojoboost.mojo`. If the project standardizes on the multi-file
layout, the whole `# -- GPU prediction` and `# -- resident validation
scoring` sections move to `bindings/gpu_prediction.mojo` unchanged, and the
only edit needed here is the import block plus the `PyInit` registrations.

**R2 — `src/mojoboost/device_policy.mojo` (device-policy lane).** Add a
prediction-scoped flat resolver so prediction stops borrowing the narrow
training one:

```mojo
def resolve_predict_device(
    device: Int, n_rows: Int, n_features: Int, n_outputs: Int,
    n_bins: Int = BINS_UNSPECIFIED, sparse: Bool = False,
) raises -> Int:
```

It should apply availability, sparse, row, bin, and output gates and skip
the objective gates and the *training* memory estimate (histograms,
gradients, hessians), which a prediction batch never allocates. When it
exists, `gpu_predict_support` in `gpu_predict.mojo` should delegate its
gates to it and keep only the record shape, and `_predict_device` in the
bindings should call it instead of `gpu_predict_support` +
`resolve_device`.

**R3 — `src/mojoboost/__init__.mojo` (public-exports lane).** Add the new
public symbols to the `from .gpu_predict import (...)` block, in the
existing alphabetical order:

```mojo
    GpuPredictSupport,
    accumulate_booster_rounds,
    accumulate_multiclass_rounds,
    accumulate_rounds,
    device_metric_code,
    device_metric_is_error,
    device_metric_matches_host,
    gpu_predict_support,
    leaf_indices_gpu,
    leaf_indices_multiclass_gpu,
    validation_host_metric,
```

**R4 — `src/mojoboost/model.mojo` (model lane).** `Model.predict_batch`
and `MulticlassModel.predict_batch` have no leaf-index form, so the leaf
entry points in the bindings have to bin and dispatch themselves. Add:

```mojo
    def leaf_indices_batch(
        self, features: List[Float64], n_rows: Int, rng: IterationRange,
        device: Int = CPU_DEVICE,
    ) raises -> List[Int]:
```

routing `GPU_DEVICE` to `leaf_indices_gpu` (or
`leaf_indices_multiclass_gpu`) and the CPU to the existing per-row walk.
`predict_leaf_batch` in the bindings then collapses to one call, and the
`_leaf_host` fallback can go.

**R5 — `src/mojoboost/unified_memory_policy.mojo` (unified-memory lane).**
Line ~944 names `GpuPredictor.upload_validation`; the method is
`set_validation`. Documentation-only rename.

**R6 — `src/mojoboost/train_gpu.mojo` (GPU trainer lane), optional.**
`_DeviceValidScorer.observe` builds a one-tree list and calls
`upload_ensemble` + `accumulate_round(0)` inline. That is exactly
`accumulate_rounds(self.predictor, round_trees, 1, learning_rate)` now.
Swapping it removes the second copy of the zero-base-score rule. Behavior
identical; do it only if the lane wants it.

## 7. Exact estimator calls Task 07 must add

All of these are in `python/mojoboost/__init__.py`, which this lane does not
own. `_addr`, `_out_buffer`, `_finish`, and `self._model` are the existing
helpers.

**Dense scores** — replace the `predict_range` call in the estimator's
`predict` (and the classifier's `predict_proba`) with:

```python
out = _out_buffer(n_rows)
ran = _mojoboost.predict_batch(
    self._model, _addr(Xb), n_rows, self.n_features_in_,
    {
        "device": self.device,          # "cpu" | "gpu" | "auto", already lowercased
        "start": start,
        "stop": stop,
        "raw_score": int(bool(raw_score)),
    },
    _addr(out),
)
# `ran` is "cpu" or "gpu", or None when n_rows == 0. Record it if the
# estimator reports where prediction ran; do not assume it.
```

Guard `n_rows == 0` before the call the way `_predict_leaf` already does:
the policy refuses to size a workload with no rows, and the entry point
returns `None` without touching the buffer.

Multiclass is `predict_proba_batch` with the same dict and an
`n_rows * n_classes` buffer. `pred_leaf` is `predict_leaf_batch` /
`predict_leaf_multiclass_batch` with the same dict minus `raw_score`.

**Refusals.** An explicit `device="gpu"` raises from the extension with the
policy's message. Nothing needs to be reimplemented on the Python side. To
*ask* rather than request:

```python
supported, code, reason, message = _mojoboost.gpu_predict_capability({
    "n_rows": n_rows, "n_features": self.n_features_in_,
    "n_outputs": self.n_classes_ if self._multiclass else 1,
    "n_bins": self.max_bin, "sparse": int(_arrays.is_sparse(X)),
})
```

**Sparse.** Pass the device into the sparse params dict so an explicit
`gpu` is refused rather than silently served on the host:

```python
params = buffers.params()
params["device"] = self.device
_mojoboost.predict_csr(self._model, params, _addr(out))
```

**Resident validation** (for a Python-driven early-stopping loop):

```python
handle = _mojoboost.gpu_validation_open(
    model, _addr(Xv), n_valid, n_features,
    {"y_addr": _addr(yv), "weight_addr": _addr(wv) if wv is not None else 0},
)
n_done = 0
for _ in range(n_rounds):
    added = _mojoboost.booster_update(model, dataset, params)
    _mojoboost.gpu_validation_accumulate(handle, model, n_done, n_done + added)
    n_done += added
    has_kernel, matches = _mojoboost.gpu_validation_metric_matches_host(metric_code)
    if has_kernel and matches:
        score = _mojoboost.gpu_validation_metric(handle, metric_code, objective_code)
    else:
        raw = _out_buffer(n_valid * n_outputs)
        _mojoboost.gpu_validation_raw(handle, _addr(raw))
        score = _mojoboost.eval_metric(metric_code, {..., "pred_addr": _addr(raw)})
```

The `matches` branch is the important one: a stopping decision made from a
metric the device defines differently is a different run. Default to the
host score unless `matches_host` is true, which is the same rule
`device_loss_metric` follows inside `train_gpu`.

## 8. Fallbacks preserved

- Every pre-existing entry point keeps its signature and its host walk.
  `predict_range`, `predict_leaf`, and their siblings are untouched behavior,
  so an estimator that has not moved over is bit-identical to before.
- The new entry points with `device="cpu"` run `Model.predict_batch`'s host
  branch, not a new code path.
- `device="auto"` still resolves to the CPU, because `crossover_rules()` is
  empty and no measurement says otherwise.
- The sparse guard is inert until a caller starts sending a `device` key.

## 9. Serialization and public-API effects

- **No serialization change.** No model field was added, removed, or
  reinterpreted. `FlatEnsemble` is built from a `Booster` at call time and is
  never persisted; `GpuValidation` holds no model state.
- Leaf ordinals are identical on both backends (`_append_tree` assigns a
  leaf its rank among its tree's leaves in node order, which is exactly what
  `Tree.leaf_ordinals` counts), so a saved model reports the same ordinals
  whichever device reads it.
- **Public Mojo API:** new symbols in `gpu_predict.mojo` are not re-exported
  from `src/mojoboost/__init__.mojo` until R3 lands.
- **Extension surface:** fourteen new functions and one new type
  (`GpuValidation`). Nothing was renamed or removed, so an older wrapper
  against a newer extension keeps working.

## 10. Risks

1. **Nothing here has been compiled or run.** No build, no test, no
   benchmark. Every claim about behavior is a reading of the code.
2. **Float32.** Device scores accumulate in Float32 and agree with the host
   to Float32 tolerance, not bit for bit. Routing is exact (bins are
   integers), so a row reaches the same leaf on both backends and only the
   sum of the leaf values it collects rounds differently. An estimator that
   asserts exact equality between devices will fail; that property is
   `gpu_predict.mojo`'s documented contract, not something introduced here.
3. **`BinMapper.transform` vs `bin_row`.** The batch path bins with
   `transform` and the established path with `bin_row`. They should agree by
   construction (same edges, same rule), but nothing in this lane verified
   it, and a disagreement would move rows between leaves rather than round
   them. This is the single highest-value check for the test lane.
4. **Concurrent churn.** `device_policy.mojo`, `objective_registry.mojo`,
   and `metrics.mojo` are all being edited by other lanes right now. This
   lane imports `BLOCK_*`, `MAX_GPU_ROWS`, `MIN_GPU_BINS`, `MAX_GPU_BINS`,
   `BINS_UNSPECIFIED`, `block_reason_name`, `gpu_available`,
   `gpu_disabled_by_env`, `gpu_supports_outputs`, `metric_canonical_name`,
   and six `METRIC_*` constants. All were present on disk at the end of this
   lane; a rename in any of them breaks the build in `gpu_predict.mojo`.
5. **New import edge.** `gpu_predict.mojo` now imports `device_policy.mojo`
   and `objective_registry.mojo`. Neither imports `gpu_predict`, and
   `device_policy`'s docstring records that it deliberately avoids importing
   the GPU stack, so the new edge points the safe way (kernels depend on
   policy). If a future edit makes `device_policy` import a GPU module, that
   edge closes a cycle.
6. **`GpuValidation` as a Python type.** It is `Movable, Writable` with no
   `py_init`, matching `max/sys/_hal/_mojo_module/device.mojo`'s `Device`
   and this module's `Model` / `Dataset`. It is constructed only by
   `gpu_validation_open*`. If `add_type` requires `Copyable` in this
   toolchain version, the type will not compile: `GpuPredictor` owns
   `DeviceBuffer`s and is `Movable` only.
7. **Empty batches.** `decide_device` raises for a workload with no rows
   whatever the device, so the batch entry points return `None` for
   `n_rows == 0` before resolving. Task 07 must not treat `None` as an
   error.
8. **Another lane committed this lane's in-progress work.** Commit `860b1cf`
   ("Integrate training and interoperability subsystems") contains an
   intermediate state of both owned files. This lane committed nothing. The
   final state on disk is the intended one; the delta is uncommitted.

## 11. Smallest later focused commands — ALL UNRUN

None of these were run. They are listed smallest first.

```sh
# 1. Does the extension still compile with the new imports and type?
bindings/build.sh

# 2. The one existing test over this module, unchanged by this lane.
pixi run mojo run -I src tests/parallel/test_gpu_predict.mojo

# 3. Formatting of the two owned files only.
pixi run mojo format src/mojoboost/gpu_predict.mojo bindings/_mojoboost.mojo
```

The check worth writing next, once (3) is clean: a test that predicts the
same dense matrix through `Model.predict_batch` with `CPU_DEVICE` and
through `predict_range`'s per-row path and asserts exact equality. That
pins risk 3, needs no accelerator, and is the assumption every new entry
point rests on.
