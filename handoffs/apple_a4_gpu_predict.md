# Apple lane A4, GPU prediction and validation scoring

Status. Implemented and run on an Apple M4 GPU. Nothing central was edited.

Files this lane owns and changed.

- `src/mojoboost/gpu_predict.mojo` (new)
- `tests/parallel/test_gpu_predict.mojo` (new)
- `handoffs/apple_a4_gpu_predict.md` (this file)

Focused test command and result.

```
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/parallel/test_gpu_predict.mojo
```

9 tests run, 9 passed, 0 failed, 0 skipped, 56.3 s wall on an M4 with an
accelerator present. Every test walks real kernels; none of them skipped.
`git diff --check -- src/mojoboost/gpu_predict.mojo
tests/parallel/test_gpu_predict.mojo handoffs/apple_a4_gpu_predict.md` is
clean (the three files are untracked, so the check is vacuous today, and it
will stay clean once they are added).

## What the module provides

`GpuPredictor` is a device-resident tree walker. Construct it once for an
ensemble shape, upload an ensemble, then use either of two paths.

Batch scoring, which is what a Python `predict` call needs.

- `raw_scores(data, rng) -> List[Float64]`, row-major `[r * n_outputs + k]`
- `response_scores(data, rng, response) -> List[Float64]`, the same with the
  objective's inverse link applied on the device
- `leaf_indices(data, rng) -> List[Int]`, per-tree leaf ordinals, row-major
  within a row and round-major within an iteration

Resident validation scoring, which is what early stopping needs.

- `set_validation(data, target, weight=[])` uploads bins, labels, weights,
  and allocates the running raw-score vector
- `reset_validation(base_scores)` seeds the raw scores with the ensemble's
  per-output base score
- `accumulate_round(iteration=0)` adds one iteration of the uploaded
  ensemble into the resident raw scores, so round `i` costs one tree per
  output rather than `i` of them
- `score_validation(rng)` rewrites the resident raw scores from a whole
  ensemble, the finished-model counterpart
- `validation_metric(metric, response) -> Float64` and
  `validation_error(metric, response)` reduce on the device
- `validation_raw() -> List[Float64]` is the escape hatch, so any host
  metric in `metrics.mojo` can score the device's scores

Free helpers.

- `flatten_booster`, `flatten_multiclass`, `flatten_trees` build the
  `FlatEnsemble` the device walks
- `response_for_objective(objective)` maps a built-in objective to the
  `RESPONSE_*` code matching `Booster.response`
- `predict_gpu`, `predict_raw_gpu`, `predict_proba_gpu` are one-shot
  convenience wrappers that build a predictor per call

Constants a caller needs. `RESPONSE_IDENTITY`, `RESPONSE_SIGMOID`,
`RESPONSE_EXP`, `RESPONSE_SOFTMAX`, `METRIC_L2`, `METRIC_L1`,
`METRIC_BINARY_LOG_LOSS`, `METRIC_MULTICLASS_LOG_LOSS`,
`METRIC_BINARY_ACCURACY`, `METRIC_MULTICLASS_ACCURACY`.

## Design notes that constrain integration

Trees cross to the device as flat arrays. One Int32 array holds every node
of every tree at `NODE_STRIDE = 8` entries per node, with child links
rebased to absolute indices and every tree's categorical bitsets
concatenated into one `UInt64` pool. `tree_root[t]` is tree `t`'s root, and
trees stay round-major, so iteration `i` of output `k` is tree
`i * n_outputs + k`, exactly the layout `MulticlassBooster` documents. A
single-output ensemble is `n_outputs = 1` and falls out of the same code.

Routing is written a third time against the flat layout and matches
`Tree.goes_left` and `_partition_kernel` term for term. Categorical nodes
test set membership in the pool, a row in the node's missing bin follows the
node's default direction, everything else compares against the threshold
bin. The routing test in the lane's test file trains a model with both a
categorical feature and a feature carrying missing values, asserts the grown
trees actually contain both kinds of node, and then compares every row's
prediction against the host.

Precision. Apple GPUs have no Float64, so leaf values, base scores, and the
raw accumulator are Float32. Agreement with the CPU predictor is to Float32
tolerance, not bit-exact, which is the same contract `histogram_gpu.mojo`
already ships. Routing itself is exact because bins are integers, so a row
reaches the same leaf on both backends and only the sum of the leaf values
rounds differently. The tests use `atol=1e-4` for scores of order 1, which
is loose for Float32 accumulation and far tighter than any misrouted row.

Determinism. One thread sums one row's trees in ascending iteration order,
so no reduction order can vary and repeated runs agree bit for bit (the test
asserts exact equality across two runs). The metric reduction is a
shared-memory tree reduction at a fixed block size followed by a host sum of
the per-block partials in ascending order in Float64, so it is deterministic
too.

Binning stays on the host. This is deliberate and it is the one thing that
must not be quietly changed during integration. Bin edges are Float64 and a
routing decision is discrete, so searching Float32 edges could put a row in
a different bin and move its prediction by a whole leaf value rather than by
a rounding step. `BinMapper.transform` runs host-side and only the resulting
`BinnedMatrix` is uploaded.

Metric definitions match `metrics.mojo` term for term with one documented
exception. The device log losses clamp probabilities at `1e-7` rather than
`1e-15`, because Float32 cannot hold `1 - 1e-15` apart from `1`. The two
agree wherever the clamp does not engage, which is everywhere a probability
is not already numerically certain.

Buffer reuse. Batch buffers grow to the largest batch seen and are then
reused. Because a device transfer is sized by the destination buffer rather
than by the batch, `upload_bins` stages through a pinned host buffer, the
same pattern `GpuHistogramBuilder.stage_gradients` uses. A smaller batch
after a larger one is covered by a test.

## Central integration required

None of the following was done by this lane. Each item lists the exact edit.

### 1. Module export

`src/mojoboost/__init__.mojo`. Add `gpu_predict` alongside the other GPU
modules so `from mojoboost.gpu_predict import ...` resolves the same way
`histogram_gpu` and `train_gpu` do. Follow whatever form that file already
uses for `train_gpu`.

### 2. Test registration

`pixi.toml`. Append to the `test` and `test-gpu` task strings, after
`tests/test_gpu_training.mojo`.

```
&& mojo run -I src tests/parallel/test_gpu_predict.mojo
```

The file skips (passing) with a printed `skipped: no accelerator` on
CPU-only machines, so it is safe in the CPU CI matrix.

### 3. Model-level entry points

`src/mojoboost/model.mojo`. Prediction on a raw matrix currently goes one
row at a time through `Model.predict`. Add batched device forms next to
them. Suggested signatures, matching the existing `device` vocabulary in
`device.mojo`.

```mojo
from .gpu_predict import (
    GpuPredictor,
    RESPONSE_SOFTMAX,
    flatten_booster,
    flatten_multiclass,
    response_for_objective,
)
from .device import CPU_DEVICE, GPU_DEVICE, resolve_device

def predict_batch(
    self,
    features: List[Float64],
    n_rows: Int,
    rng: IterationRange,
    raw_score: Bool = False,
    device: Int = CPU_DEVICE,
) raises -> List[Float64]
```

Body. Resolve the device with `resolve_device(device, n_rows,
self.mapper.n_features, 1)`. Bin once with `self.mapper.transform(features,
n_rows)`. On `CPU_DEVICE`, loop rows through the existing
`booster.predict_raw_bins_range` / `predict_bins_range`. On `GPU_DEVICE`,
build a `GpuPredictor(self.mapper.n_features, 1)`, `upload_ensemble(
flatten_booster(self.booster))`, then `raw_scores(data, rng)` or
`response_scores(data, rng, response_for_objective(self.booster.objective))`.

`MulticlassModel` gets the same shape with `n_outputs = booster.n_classes`,
`flatten_multiclass`, and `RESPONSE_SOFTMAX` for probabilities. Its raw form
is `raw_scores`.

Note for whoever writes this. `resolve_device` currently raises for `gpu`
only on availability and `gpu_supports`; both already pass for prediction.
No change to `device.mojo` is needed for prediction, and `AUTO_DEVICE` for
prediction should keep resolving to the CPU until a crossover benchmark
exists, exactly as training does. Do not add a prediction size heuristic
without a measurement behind it.

### 4. C bindings

`bindings/_mojoboost.mojo`. Add two functions and register them in the
`def_function` block next to `predict_range` and `predict_proba_range`.

```mojo
def predict_range_device(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    start: PythonObject,
    stop: PythonObject,
    raw_score: PythonObject,
    device: PythonObject,      # "cpu" | "gpu" | "auto"
    out_addr: PythonObject,
) raises -> PythonObject

def predict_proba_range_device(...)   # same, MulticlassModel, n_classes wide
```

Each one parses the device name with `parse_device` (already imported for
`resolve_device`), calls the new `Model` / `MulticlassModel` batch method,
and stores the result into the caller's preallocated Float64 buffer with the
existing `_store` / `unsafe_store` pattern. The output layouts are
unchanged, `[r]` for single output and `[r * n_classes + k]` for multiclass,
so no Python-side shape logic changes.

```
m.def_function[predict_range_device]("predict_range_device")
m.def_function[predict_proba_range_device]("predict_proba_range_device")
```

Do not change the signature of the existing `predict_range` or
`predict_proba_range`. Adding a parameter to those would be an ABI break for
any caller pinned to the current extension.

### 5. Python estimator API

`python/mojoboost/__init__.py`.

`MojoBoostRegressor.predict` and `MojoBoostClassifier.predict` /
`predict_proba` take a new keyword.

```python
def predict(self, X, raw_score=False, start_iteration=0, num_iteration=None,
            pred_leaf=False, pred_contrib=False, validate_features=False,
            device=None):
```

Semantics. `device=None` keeps the fitted estimator's behaviour, which is
the CPU. A string goes through the existing `self._resolve_device(n_rows,
self.n_features_in_, n_outputs)`, so `"gpu"` raises without an accelerator
and `"auto"` resolves to the CPU under the current policy, and the names are
lowercased the way LightGBM treats `device_type`. Validate against
`_DEVICES`.

Routing inside `predict`. The dense path replaces the `_mojoboost
.predict_range(...)` call with `_mojoboost.predict_range_device(...)`,
passing the resolved name. Three cases must still refuse the GPU with a
clear error rather than falling back.

- `pred_contrib=True`. Contributions have no device kernel (see follow-ups).
- `pred_leaf=True` can be wired to `GpuPredictor.leaf_indices`, but only
  after `_predict_leaf` is taught the device path; until then refuse.
- Sparse input. `_arrays.is_sparse(X)` already routes to `_sparse_scores`,
  and there is no sparse GPU predictor. Refuse with the same wording the
  fit path uses at `MojoBoostRegressor` for `device="gpu"` on sparse input.

`predict_proba` on the classifier forwards its own `device` to the
multiclass binding and keeps the `[n_rows, n_classes]` reshape unchanged.
The binary classifier's two-column probability path is built from the
single-output response scores and needs no separate kernel.

Docstring edits. The module docstring's device paragraph (near
"Estimators take `device=\"cpu\"`") should say that prediction now takes its
own `device` and that the ensemble is unchanged either way, so a model
fitted on the CPU can be predicted on the GPU and the reverse.

### 6. Training-validation wiring

`src/mojoboost/custom_metric.mojo` holds the one metric training loop,
`train_with_callbacks`, plus `_base_scores`, `_update_valid_raw`, and
`_eval_round`. The device path replaces the middle one and, when the metric
is one the device implements, the third.

Suggested shape, keeping the CPU path byte-identical when no device is
requested.

1. Before the loop, per validation set, build one `GpuPredictor(
   data.n_features, n_outputs)`, call `set_validation(valid.data,
   valid.target, valid.weight)`, then `reset_validation([base_score])` (or
   the per-class base scores for the softmax loop).
2. Per round, after the round's trees are grown, call
   `upload_ensemble(flatten_trees(round_trees, zeros, n_outputs,
   learning_rate))` where `round_trees` is that round's one tree per output,
   then `accumulate_round()`. That replaces `_update_valid_raw`.
3. Scoring. If every metric in the suite maps to a `METRIC_*` code, call
   `validation_metric(code, response_for_objective(objective))` and append
   to the history, which replaces the body of `_eval_round`. Otherwise call
   `validation_raw()` once per round and hand that vector to the existing
   host evaluator unchanged. `validation_raw` returns exactly the vector
   `_eval_round` already passes as `valid_raw[v]`, so the fallback is a
   one-line substitution.
4. `_StopState`, `observe`, and `exhausted` are untouched. Early stopping
   reads the history, not the scores.

Two decisions for whoever does this.

- The device metric values differ from the host ones at Float32 tolerance.
  Early stopping compares a metric against its own running best with
  `min_delta`, so the comparison is self-consistent, but a run stopped on
  device metrics can pick a different `best_iteration` than the same run
  stopped on host metrics when two rounds are within Float32 noise. Either
  document that, or score on the host from `validation_raw()` and keep the
  device only for the accumulation. Scoring on the host is the conservative
  default and still removes the per-round tree walk, which is the expensive
  part.
- `python/mojoboost/__init__.py` currently raises `"validation metrics are
  scored on the CPU; use device='cpu' or device='auto'"` when an `eval_set`
  is combined with a non-CPU device. That guard, and the matching paragraph
  in the module docstring near line 115, come out only once the above is
  wired and tested.

### 7. Documentation

- `docs/` and `README.md`. The device section says GPU covers training. Add
  that prediction and validation scoring have a device path, that the
  ensemble is device independent, and that binning stays on the host.
- `docs/LIGHTGBM_PARITY.md`. No parity claim changes. LightGBM's
  `device_type` covers training only, so a prediction-time `device` is a
  mojoboost addition and belongs in the differences section next to `auto`.

## CPU-equivalence tests

Already in `tests/parallel/test_gpu_predict.mojo`, all passing.

1. `test_raw_and_response_match_cpu`. Whole-ensemble raw and response scores
   for a logistic model against `predict_raw_bins` and `predict_bins`, row
   by row, plus a check that the sigmoid really was applied.
2. `test_iteration_ranges_match_cpu`. Slices against
   `predict_raw_bins_range`, that `[0, k)` and `[k, n)` sum to the full
   score, that an empty range past 0 is zero, and that `[0, 0)` is the base
   score alone.
3. `test_missing_and_categorical_routing_match_cpu`. Categorical set
   membership and missing-bin default direction, with an assertion that the
   grown trees really contain both kinds of node before the comparison runs.
4. `test_leaf_indices_match_cpu`. Leaf ordinals over a slice against
   `leaf_indices_bins`, exact integer equality.
5. `test_multiclass_proba_matches_cpu`. Softmax probabilities and raw
   per-class scores against `predict_proba_bins` and `predict_raw_bins`,
   probabilities summing to one, and a sliced range against
   `predict_proba_bins_range`.
6. `test_incremental_validation_and_metrics`. The trainer loop written the
   way a trainer would write it, resident raw scores against a host-computed
   running raw vector, device `l2` and `l1` against `metrics.mojo`, that
   scoring twice does not disturb the raw scores, and that
   `score_validation` over the whole ensemble reproduces the incremental
   result.
7. `test_weighted_and_classification_metrics`. Weighted and unweighted
   binary log loss against `metrics.mojo`.
8. `test_multiclass_validation_metric`. Multiclass log loss against
   `metrics.mojo`.
9. `test_prediction_is_deterministic_and_buffers_are_reused`. Bit-exact
   equality across two runs, and a smaller batch through buffers sized for a
   larger one.

To add once integration lands, in the central suites rather than here.

- `python/tests`. `predict(X, device="gpu")` against `predict(X)` on a
  fitted regressor and classifier, `predict_proba` likewise, and the three
  refusals (contributions, leaf until wired, sparse) asserting the error
  message rather than a silent fallback. These belong with the other
  estimator tests and should skip when `_mojoboost.gpu_available()` is
  false.
- `tests/test_backend_equivalence.mojo`. A `Model.predict_batch` CPU versus
  GPU comparison once the model-level entry point exists, which is the
  natural home for it since that file already owns cross-backend claims.
- The metric training loop. A `train_with_callbacks` run with and without
  the device path, asserting the same `best_iteration` on a problem whose
  rounds are not within Float32 noise of each other.

## Interaction with the concurrent Apple lanes

Read-only inspection of the other lanes' landed files at the end of this
lane's work. Nothing below was edited here, and none of it changes the code
this lane shipped; it is sequencing information for whoever integrates.

### A5, `gpu_runtime.mojo`, the seam that already exists

That lane's resource registry reserves two slots for this one.

```
comptime RES_VALID_BINS = 9    # "the validation matrix"
comptime RES_VALID_SCORE = 10  # "its device-side scores"
```

The comment there says the reservation exists "so a prediction path can
share one tracker with training", which is exactly this module. The mapping
is direct. `GpuPredictor.valid_bins_dev` is `RES_VALID_BINS`.
`valid_raw_dev` and `valid_resp_dev` are both `RES_VALID_SCORE`, and the
hazard between them is a genuine one worth tracking, since
`validation_metric` reads the raw buffer and writes the response buffer
while `accumulate_round` writes the raw buffer. The batch buffers
(`bins_dev`, `out_dev`, `resp_dev`) have no reserved slot and would need
either new ids or an explicit decision that batch prediction sits outside
the training session's tracker.

One divergence to resolve, and this lane's choice is the weaker one.
`GpuPredictor` holds its own `DeviceContext`, following
`GpuHistogramBuilder`. A5's session and A3's `GpuObjectiveState` both take
the context as a parameter instead of holding it, so one context can drive
every buffer in a run. Reconciling means changing `GpuPredictor.__init__` to
accept a `DeviceContext` and dropping the field, which is a small edit to
this module and should be made when A5 lands rather than kept as is. Until
then, a training run that also predicts opens two contexts.

### A3, `gpu_objectives_native.mojo`, complementary rather than duplicate

That lane's `GpuObjectiveState` also keeps device-resident Float32 raw
scores in the row-major `[r * n_classes + k]` layout this module uses, and
its `_update_raw_kernel` also advances them by `learning_rate * value[leaf]`
per tree. The two are not redundant, and the reason matters enough to write
down before someone collapses them.

A3 updates the *training* rows. Those rows were partitioned during growth,
so every one of them already sits in a leaf on the device and the tree's
`value` array is a direct lookup. No traversal, no feature reads.

This module updates the *validation* rows. They were never partitioned, so
there is no leaf assignment to read and the tree has to be walked. That is
what `accumulate_round` does.

So the correct integration keeps both, and specifically does not use this
module's walk for training rows, which would throw away the leaf array that
makes A3's update free. The shared layout is deliberate and useful:
`validation_raw()` here and `host_raw` there return the same shape, so the
host metric path can take either without a reshape.

### A6, `apple_gpu_policy.mojo`

This module imports `derive_block_threads` and `query_device_caps` from
`gpu_tiling.mojo`, and A6 restates that policy (`TARGET_BLOCK_THREADS`,
`WARP_GRANULARITY`, the fallback capability constants) in its own file. If
A6's policy supersedes `gpu_tiling`'s, the two imports at the top of
`gpu_predict.mojo` move and nothing else here changes. The one constraint
this module places on any tiling policy is that `REDUCE_BLOCK` stays a power
of two and stays the `block_dim` of the metric launch, because the
shared-memory reduction halves a compile-time-sized array. The walk kernels
have no such constraint and will take whatever block size a policy hands
them.

### A2, `gpu_split_search.mojo`

No conflict, one observation. That lane encodes categorical sets as 16-bit
words (`CAT_WORDS = MAX_SPLIT_BINS // 16`) inside its split record, while
this module carries the 64-bit `CAT_BITSET_WORDS` pool that
`categorical.mojo` and `Tree.cat_bitset` already use. Both are correct for
their own kernel. It does mean the repository now holds three flat encodings
of categorical membership, and that is worth one owner and one conversion
point eventually rather than three independent ones.

### A1, `gpu_active_rows.mojo`

Unaffected. This module never reads or writes the leaf-assignment array and
never depends on row order, so order-preserving row compaction cannot change
a prediction. The only future coupling would be fusing prediction with
compacted training rows, at which point the compaction's row mapping would
have to be applied before scores are written back in the caller's order.

## Follow-ups this lane deliberately did not take

- Feature contributions (`contrib.mojo`, TreeSHAP). They walk every node of
  every tree carrying a per-node weight vector rather than following one
  root-to-leaf path, so they need a different kernel, a different memory
  budget, and per-thread scratch proportional to tree depth. Genuinely
  incompatible with the walk kernel here.
- Device-side binning. Would remove the host pass over `n_rows *
  n_features`, but only with a Float32 edge search that provably agrees with
  the Float64 one. See the precision note above. This is the largest
  remaining win for `predict(device="gpu")` on wide raw matrices, and it is
  a correctness problem before it is a performance one.
- Sparse prediction. `sparse.mojo` and `model_sparse.mojo` have no device
  path, and CSR row walks are a different kernel shape.
- Multi-batch streaming for matrices larger than device memory. The
  predictor sizes one batch at a time today; a caller chunks its own rows.

## Risks

- No performance claim is made. This lane measured correctness only. The
  test file trains on the CPU and predicts on the GPU on datasets of one to
  two thousand rows, which is far below any plausible crossover, so nothing
  here says GPU prediction is faster than the host walk. A crossover
  benchmark belongs with the A8 benchmark lane and should land before any
  `auto` policy is taught to pick the GPU for prediction.
- Two kernel-argument aliasing rules bit during development and will bite
  anyone extending the module. `out` cannot name a function argument (it is
  a soft keyword), and the same pointer cannot be passed twice to
  `enqueue_function`, which is why the response transform always writes to a
  separate buffer rather than in place.
- `upload_ensemble` reallocates the ensemble buffers on every call. That is
  correct and cheap for the per-round validation path (a round's flat form
  is kilobytes) but it is not something to call inside a per-batch loop.
- The Float32 metric tolerance is the one place where the device path can
  change a training run's outcome rather than just its speed. See the
  decision noted in the wiring section above.
