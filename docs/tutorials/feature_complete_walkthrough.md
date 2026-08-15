# A walk through everything mojotrees can do

One dataset, one session, and every feature the Python API exposes, in the
order you would meet them. Validation, early stopping, callbacks,
inspection, model IO, continued training, leaf indices, feature
contributions, sparse input, categorical features, and choosing a backend.

Where a feature is missing, partial, or refuses a combination, this
document says so where you would have reached for it, with the workaround
if there is one. A tutorial that only shows the happy path is a sales
document, and it costs the reader an afternoon to find out.

**Status of this document.** The code here was written against the source
of `python/mojotrees` and against `docs/LIGHTGBM_PARITY.md`. It has not
been executed end to end as one script by the pass that wrote it, so treat
it as a careful reading rather than as a transcript. Turning it into a
script that CI runs is tracked in `handoffs/task20_compatibility.md`.
Anything you find that disagrees with the code is a bug in this file.

Compatibility rules for everything shown here, including which of it is
guaranteed to survive the next release, are in
[docs/COMPATIBILITY_POLICY.md](../COMPATIBILITY_POLICY.md).

## 0. Setup

The extension module has to be built before the package imports.

```console
$ pixi run build-python
```

numpy is optional throughout; plain lists of rows work everywhere. This
walkthrough uses numpy because the shapes are easier to read.

```python
import numpy as np
import mojotrees as mb
from mojotrees import MojoTreesRegressor, MojoTreesClassifier, MojoTreesRanker

rng = np.random.default_rng(0)
n, p = 4000, 12
X = rng.normal(size=(n, p))
y = X[:, 0] * 2.0 + X[:, 1] ** 2 - 0.5 * X[:, 2] + rng.normal(scale=0.1, size=n)

X_train, X_valid = X[:3000], X[3000:]
y_train, y_valid = y[:3000], y[3000:]
```

## 1. A first model

Defaults are LightGBM's, deliberately, so a comparison measures the
implementation and not the configuration.

```python
model = MojoTreesRegressor().fit(X_train, y_train)
pred = model.predict(X_valid)
```

`num_leaves=31`, `learning_rate=0.1`, `n_estimators=100`,
`min_data_in_leaf=20`, and `max_bin=255` are LightGBM's values. Two are
mojotrees's own and worth knowing: `lambda_l2` defaults to 1.0 and
`min_child_hess` to 1e-3.

`X` may hold `NaN`, which is the missing-value marker. It may not hold
infinities, and `y` and `sample_weight` must be finite. That matches
LightGBM's scikit-learn wrapper, which validates with
`force_all_finite="allow-nan"`.

## 2. Validation sets

Every estimator takes LightGBM's validation arguments.

```python
model = MojoTreesRegressor(n_estimators=200).fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_names=["holdout"],
    eval_metric=["l2", "l1"],
)
```

`eval_set` takes a list of `(X, y)` pairs, or one bare pair, or the
separate `eval_X=` and `eval_y=` arguments. Unnamed sets are `valid_0`,
`valid_1`, and so on. `eval_sample_weight` takes one weight vector per
set.

`eval_metric` takes built-in names, callables, or a mix. The names resolve
through `python/mojotrees/_eval.py` to codes that
`src/mojotrees/metrics.mojo` computes, so a named metric agrees with the
Mojo API by construction rather than by a second Python implementation.

| Task | Names |
|---|---|
| Regression | `l2`, `rmse`, `l1`, `quantile`, `huber`, `mape`, `fair`, `poisson`, `gamma`, `gamma_deviance`, `tweedie`, `cross_entropy`, `kullback_leibler` |
| Binary | `binary_logloss`, `binary_error`, `auc`, `average_precision` |
| Multiclass | `multi_logloss`, `multi_error` |
| Ranking | `ndcg`, `map` |

LightGBM's aliases work too, so `mse`, `regression_l2`, `mae`,
`l2_root`, `xentropy`, `kldiv`, and the rest land on the canonical name.
With no `eval_metric` at all, the objective's own loss is scored.

A callable is `f(y_true, y_pred) -> float`, called once per metric per set
per round, and it receives **raw** scores in `y_pred`, exactly as
LightGBM's `feval` does. That is log-odds for the binary classifier and
one row-major block of `n_classes_` per row for the softmax one.

```python
def top_decile_error(y_true, y_pred):
    k = max(1, len(y_true) // 10)
    idx = np.argsort(y_pred)[-k:]
    return float(np.mean((y_true[idx] - y_pred[idx]) ** 2))

model = MojoTreesRegressor(n_estimators=200).fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=["l2", ("top_decile", top_decile_error, False)],
)
```

The three-tuple form gives a callable a name and a direction, `True` for
higher-is-better.

**Not available.** A callable metric is handed unweighted predictions, so
combining one with `eval_sample_weight` raises rather than quietly
dropping the weights. Use a built-in name when you need weighted
validation, or fold the weights into the callable yourself and pass
`eval_sample_weight=None`.

Results land on `evals_result_`.

```python
history = model.evals_result_["valid_0"]["l2"]
# history[i] is the score after i trees; history[0] is the base score alone
```

`evals_result_[name][metric][0]` is the base-score-only model, so index
`i` is the score after `i` trees. That indexing is part of the contract.

**A difference from LightGBM.** `best_score_` here is one number, the best
value of the primary metric. LightGBM's is a nested dict of every set's
every metric. The whole grid is in `evals_result_`, so nothing is lost,
but code that indexes `best_score_["valid_0"]["l2"]` needs rewriting.
Section 5.4 of the compatibility policy records that this is not settled.

## 3. Early stopping

```python
model = MojoTreesRegressor(n_estimators=2000).fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=["l2", "l1"],
    early_stopping_rounds=25,
    min_delta=1e-5,
    primary_metric="l2",
)

model.best_iteration_   # where the primary metric peaked
model.n_iter_           # how many rounds actually trained
model.stopped_early_    # whether patience ran out
model.best_score_       # the primary metric's best value
```

Training stops once a watched metric has gone `early_stopping_rounds`
rounds without improving by more than `min_delta`, and the ensemble is
truncated to the best round of `primary_metric`, which takes an index or a
name, on the first validation set.

Truncation is the mechanism, and it is why `predict()` needs no
bookkeeping to honor `best_iteration_`. The trees the model still holds
**are** the best iteration, so `num_iteration=None` predicting with
`best_iteration_` iterations is structural rather than a convention.

With `early_stopping_rounds=0` nothing stops and nothing is truncated, but
every value is still recorded and `best_iteration_` still reports where
the metric peaked. That is the difference between `best_iteration_` and
`n_iter_`.

**Requires.** An `eval_set`. Asking for early stopping without one raises
and says so.

## 4. Callbacks

LightGBM's four factories are here, importable from the top level or from
`mojotrees.callback`.

```python
from mojotrees import log_evaluation, record_evaluation, reset_parameter

recorded = {}
model = MojoTreesRegressor(n_estimators=100).fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=["l2"],
    callbacks=[
        log_evaluation(period=10),
        record_evaluation(recorded),
        reset_parameter(learning_rate=lambda i: 0.1 * (0.99 ** i)),
    ],
)
```

A callback is any callable taking one `CallbackEnv`, a namedtuple of
`(model, params, iteration, begin_iteration, end_iteration,
evaluation_result_list)`. Callbacks split into before-iteration and
after-iteration groups by a `before_iteration` attribute and run in
ascending `order` within each group, which is LightGBM's contract.

A before-iteration callback may change nine hyperparameters, which is what
`reset_parameter` uses: `learning_rate`, `num_leaves`, `max_depth`,
`min_data_in_leaf`, `min_sum_hessian_in_leaf`, `lambda_l1`, `lambda_l2`,
`feature_fraction`, and `feature_fraction_bynode`. The scikit-learn
spellings work as well, so a schedule written for `reg_alpha` or `eta`
lands on the right parameter.

Raise `EarlyStopException` from a callback to stop training. Any other
exception propagates with its own type and leaves the estimator unfitted,
which is a guarantee about the estimator's state and not only about the
exception.

A callback costs one crossing of the Python boundary per phase per round
and nothing per row. With no callbacks the bridge does not cross the
boundary at all and the model is unchanged to the bit.

**Not available.** Callbacks need an `eval_set`, because the hook lives in
the trainer that scores validation metrics. They run for the regressor and
the binary classifier. The softmax multiclass trainer and the LambdaRank
trainer **refuse** a callback list rather than ignoring it, which is the
right failure but it is still a gap: a multiclass run cannot have a
learning-rate schedule today.

## 5. Inspecting a fitted model

```python
model.feature_importances_          # importance_type="split" by default
MojoTreesRegressor(importance_type="gain").fit(X_train, y_train).feature_importances_

booster = model.booster_
booster.num_trees()
booster.current_iteration()
booster.num_model_per_iteration()
booster.num_feature()
booster.feature_name()
text = booster.model_to_string()    # the whole model, in mojotrees's format
```

`booster_` is the same model object the functional API returns, so there
is one model type in this package rather than one per door.

Structured inspection lives in its own module.

```python
from mojotrees.inspection import dump_model, trees_to_dataframe

dump = dump_model(model)             # the schema, as a dict
frame = trees_to_dataframe(model)    # one row per node, LightGBM's columns
```

`dump_model` gives you LightGBM's `Booster.dump_model()` shape, and the
module also carries `trees_to_records` (the same rows without pandas),
`split_values`, `get_split_value_histogram`, `leaf_index_of`,
`raw_scores`, `parse_model_string`, and `booster_of`. What the dump
contains is stated normatively in
[docs/MODEL_INSPECTION_SCHEMA.md](../MODEL_INSPECTION_SCHEMA.md), which is
what a consumer should read rather than either implementation.

Two version numbers travel in the dump and they answer different
questions. `dump_format_version` says what the dump's keys mean, and
`model_format_version` says which optional facts a model of that vintage
can carry at all. Branch on the capability flags rather than assuming: a
model read from a file written before v4 reports `has_split_gain: false`,
because those formats dropped the gains and a fitted tree cannot recompute
them, and a v1 or v2 model reports `has_node_count: false`.

**Two caveats.** `trees_to_dataframe` needs pandas, which mojotrees does
not depend on; `trees_to_records` is the dependency-free form. And as of
this writing the inspection names are not re-exported from the package
top level, so `mojotrees.inspection` is a submodule import rather than one
of the guaranteed import paths. Section 8.1 of the compatibility policy
carries that as an open question for the first release, not as a
statement that the module is unstable.

## 6. Saving, loading, and pickling

Three ways to move a model, carrying three different amounts.

```python
model.save("model.mbst")
reloaded = MojoTreesRegressor.load("model.mbst")

import pickle
blob = pickle.dumps(model)
same_estimator = pickle.loads(blob)

text = model.booster_.model_to_string()
booster = mb.Booster(model_str=text)
```

| Path | Carries | Does not carry |
|---|---|---|
| `pickle` | The whole estimator: hyperparameters, fitted attributes, feature names, split gains | Nothing a fitted estimator holds |
| `save` and `load` | The model, its split gains, and its feature names. `n_features_in_` and `best_iteration_` are recomputed from it | Hyperparameters, `device_`, `evals_result_` |
| `model_to_string` | The model, its split gains, and its feature names | The training set, the parameter object |

The file format is versioned and currently v4. Floats travel as IEEE-754
bit patterns, so a round trip is bit-exact and there is no locale,
precision, or endianness pitfall. Every release reads every file any
earlier release wrote. An older release does **not** read a newer file,
and says so rather than misparsing.

It is mojotrees's format, not LightGBM's. The two are not interchangeable
in either direction.

## 7. Continued training

This is where the two doors differ, so it is worth being exact.

```python
train_set = mb.Dataset(X_train, label=y_train)
booster = mb.train({"objective": "regression"}, train_set, num_boost_round=50)
booster.update(num_iteration=50)     # 50 more rounds on the same data
booster.num_trees()                  # 100

# or resume from a booster you already have
more = mb.train(
    {"objective": "regression"},
    train_set,
    num_boost_round=50,
    init_model=booster,
)
```

`Dataset` owns the data and its binning. Binning is the expensive part of
starting a run, so a dataset is binned once, by `construct()` or by the
first `train` that uses it, and every later run on it reuses those bins.
`init_model` copies the model first, so the booster you passed in still
holds the trees it held.

Continued training resumes from the raw scores the existing trees produce.
That covers the single-output objectives and softmax multiclass.

**Not available, three ways.**

1. **The estimators cannot continue training.** `fit` has no `init_model`
   argument, and the `Booster` an estimator holds on `booster_` has no
   training set behind it, so it can do everything except `update()`. Use
   the functional API when you need to resume.
2. **Ranking cannot continue.** LambdaRank gradients are computed within a
   query from state the fitted ensemble does not carry, so a ranking
   booster refuses `update()` rather than appending trees that would be
   wrong.
3. **`train()` does not report.** `valid_sets` are registered on the
   returned booster for `eval_valid()`, but there is no per-round history
   and no early stopping in the functional API. Those live on the
   estimators' `fit`. If you need both resumption and early stopping in
   one run today, you have to drive the rounds yourself.

`Dataset` is immutable once constructed. There is no `set_label`,
`set_field`, or `set_categorical_feature`, because bin edges are fitted
from the data and from the categorical declaration and changing either
afterwards would leave the binned matrix describing data the dataset no
longer holds. Every field is a constructor argument.

## 8. Leaf indices and feature contributions

```python
leaves = model.predict(X_valid, pred_leaf=True)
# (n_valid, num_iteration) for the regressor, the ranker, and binary
# (n_valid, num_iteration * n_classes) for softmax, column i*n_classes+k

contrib = model.predict(X_valid, pred_contrib=True)
# (n_valid, n_features + 1); the last column is the expected value
raw = model.predict(X_valid, raw_score=True)
assert np.allclose(contrib.sum(axis=1), raw)   # an identity, not a fit
```

Contributions are exact TreeSHAP Shapley values, so every row's entries
sum to that row's raw score exactly. The sum is a mathematical identity
rather than a normalization, which is the difference between this and a
split-gain heuristic.

A leaf is named by its ordinal within its own tree, in
`[0, num_leaves)`, in node order. The numbering is fixed once a tree is
grown and survives `save`, `load`, and pickle. It is mojotrees's own
numbering and not LightGBM's, and the two agree only by coincidence, so
leaf ids used as a categorical feature downstream cannot be transferred
between the two libraries.

**Not available, and why.**

- `raw_score=True` with `pred_leaf=True` raises. They ask for different
  dtypes and different shapes. LightGBM lets `pred_leaf` win silently,
  which is not recoverable from the output.
- `raw_score=True` with `pred_contrib=True` raises. Contributions always
  explain the raw score whatever the link, so the flag would be either
  redundant or a lie.
- A model loaded from a v1 or v2 file raises on `pred_contrib`. Those
  formats predate per-node covers, which are the background weighting the
  exact contributions condition on, and they cannot be recovered from a
  fitted tree. Refit, or re-save from a current build, which writes v4.
- Neither flag takes sparse input. See the next section.

## 9. Sparse input

```python
from scipy import sparse

Xs = sparse.random(4000, 200, density=0.01, format="csr", random_state=0)
ys = rng.normal(size=4000)

model = MojoTreesRegressor().fit(Xs, ys)
pred = model.predict(Xs)
```

Any SciPy sparse matrix or array is accepted and stays sparse. Nothing is
densified at any point, including a 200k by 5k matrix that would not fit
dense. Whatever format you pass is converted to the one that side of the
boundary wants, CSC to fit because histogram accumulation is
feature-oriented and CSR to predict because prediction is row-oriented,
and a non-canonical matrix is copied before its indices are sorted, so
your matrix is never mutated.

An implicit zero is the numerical value 0.0 and not a missing value, which
matches LightGBM's default `zero_as_missing=false`. A sparse fit equals
the dense fit of the same matrix with the gaps filled with zeros.
Explicitly stored zeros mean the same as the gaps. `NaN` is still the
missing marker wherever it is stored.

**Not available for sparse input.** Each of these raises rather than
densifying behind your back.

| Wanted | State |
|---|---|
| `device="gpu"` | No sparse GPU kernel exists |
| A Python objective callback | Not wired through the sparse path |
| `eval_set` and early stopping | Not wired through; the Mojo API has `train_sparse_with_valid` |
| Ranking | Not wired through |
| `pred_leaf`, `pred_contrib` | Not wired through |
| `start_iteration`, `num_iteration` slicing | Not wired through |

The workaround for all of them is `X.toarray()`, and the reason it is not
done for you is that densifying a matrix chosen for its sparsity is the
one thing the caller cannot afford.

`zero_as_missing=true` is not implemented and no alias accepts it.

## 10. Categorical features

```python
import pandas as pd

frame = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(p)])
frame["city"] = pd.Categorical(rng.choice(["a", "b", "c"], size=len(frame)))

model = MojoTreesClassifier().fit(frame, (y_train > 0).astype(int))
model.categorical_feature_
```

`categorical_feature` is LightGBM's parameter, and `"auto"`, its default
and LightGBM's, means every pandas `category` column and nothing else. On
any other input that is no columns. You may instead pass a sequence of
feature names, column indices, or a mix.

Those columns are split by category set rather than by threshold, with no
one-hot expansion. Missing codes, which are negative or `NaN`, unseen
codes, and dropped codes all route right. A pandas `category` column is
encoded by its labels and the mapping is kept on the fitted estimator, so
a prediction frame that orders or extends its categories differently still
lands on the categories the model was fitted with.

`max_cat_to_onehot`, `max_cat_threshold`, `cat_smooth`, `cat_l2`, and
`min_data_per_group` are LightGBM's categorical hyperparameters with
LightGBM's defaults, and have no effect unless some feature is
categorical.

**Refusals worth knowing.** Leaving a pandas `category` column out of an
explicit `categorical_feature` raises rather than feeding its codes to the
numerical scan. Category codes on any non-pandas input must be whole
numbers below 2\*\*31, `NaN`, or negative. On sparse input only explicit
indices can be categorical, because a sparse frame has no `category`
dtype to read.

## 11. Choosing a backend

```python
mb.gpu_available()                       # can this build train on an accelerator

model = MojoTreesRegressor(device="cpu").fit(X_train, y_train)     # the default
model.device_                            # the backend that actually ran
```

`device` takes `"cpu"`, `"gpu"`, or `"auto"`. `"cpu"` is the default and
the dependable backend. `"gpu"` raises when no accelerator is available or
when the GPU path does not cover the workload, rather than falling back
silently, so a GPU run that returns is a GPU run. `"auto"` picks a backend
for you and **currently always picks the CPU**. Fitting records what ran
on `device_`.

CPU parallelism is controlled by two environment variables.
`MOJOTREES_NUM_WORKERS` is 1 for serial, N above 1 to force chunked
dispatch, and 0 or unset for automatic. `MOJOTREES_PARALLEL_MIN_OPS`
overrides the built-in threshold below which dispatch stays serial.
Neither changes a result. Each feature owns its output slice during
histogram accumulation, so the worker count is a speed control and not a
numerical one.

**Not available on the GPU today.** Each raises rather than falling back.

| Wanted | State |
|---|---|
| `device="gpu"` with an `eval_set` | Validation is scored on the CPU |
| `device="gpu"` for 3 or more classes | Multiclass training is CPU-only |
| `device="gpu"` with sparse input | No sparse GPU kernel |
| `device="auto"` selecting the GPU off an Apple M4 | The policy exists; the only measured crossover rules are M4 rules, so elsewhere it resolves to the CPU |

**And the honest part.** GPU training has been run on exactly one device,
an Apple M4 through Metal, where correctness and repeat-run determinism
pass and it is slower than the four-worker CPU path at every shape
measured. No NVIDIA and no AMD hardware has ever executed this code, not
in CI and not on a laptop. There is one GPU source and no per-vendor
files, which is a reason to expect portability and is not evidence of it.
`docs/GPU_VALIDATION.md` is the record and says what a backend has to
produce before any claim is made about it.

## 12. Classification and ranking, briefly

The classifier picks binary or softmax from the labels, which may be of
any single comparable type.

```python
clf = MojoTreesClassifier(class_weight="balanced").fit(X_train, y_train > 0)
clf.classes_
clf.predict_proba(X_valid)          # (n, n_classes), columns in classes_ order
clf.predict(X_valid)                # labels from classes_, the argmax of the above
clf.predict(X_valid, raw_score=True)  # log-odds, shape (n,), for binary
```

`class_weight` takes `None`, `"balanced"`, or a dict from label to
weight. LightGBM's binary-only `scale_pos_weight` is `class_weight={1: w}`
here, and its `is_unbalance` is `"balanced"` up to a constant factor.
Weighting is not calibration: a class-weighted model's probabilities are
probabilities under the reweighted sample.

The classifier takes no `objective`. Custom objectives are single-output
only, so pass yours to `MojoTreesRegressor` and apply your own link to the
raw predictions. `objective=` is accepted on the classifier solely so that
passing one raises that message instead of a bare `TypeError`.

The ranker takes query groups.

```python
group = mb.group_from_query_ids(query_ids)     # per-query row counts
rnk = MojoTreesRanker(ndcg_eval_at=10).fit(X_train, rel_train, group=group)
scores = rnk.predict(X_valid)                  # sort within a query, descending
mb.ndcg_score(scores, rel_valid, valid_group, at=10)
```

Scores are comparable within a query and not between queries. `score()`
accepts `sample_weight` for signature compatibility and ignores it,
because a weighted NDCG has no LightGBM definition to match.

## 13. A custom objective

```python
def logcosh(raw, y):
    d = raw - y
    return np.tanh(d), 1.0 - np.tanh(d) ** 2

model = MojoTreesRegressor(objective=logcosh, base_score="mean").fit(X_train, y_train)
```

The callback is called once per boosting round over whole arrays, never
per row, through three reused float64 buffers.

Three deliberate differences from LightGBM. The framework applies
`sample_weight` **after** the callback returns, so your callback never
sees weights and a custom objective matches the equivalent built-in one
bit-exactly. Gradients and hessians are validated every round for length,
finiteness, and non-negative hessian, where LightGBM does not check.
`base_score` defaults to 0.0, and `base_score="mean"` is the opt-in that
matches a built-in objective's starting point; it is resolved on the Mojo
side, because Python's sequential sum differs from Mojo's by one unit in
the last place and that alone would break bit-exactness.

Predictions from a custom objective are raw scores. Apply your own link.

**Not available.** A Python objective callback cannot be combined with an
`eval_set`, so a custom objective gets no validation metrics and no early
stopping from the Python API. The Mojo API pairs them in
`train_custom_with_metrics`. It also cannot take sparse input.

Measure before reaching for it. The measured cost on an M4 was about 8.9
ms per round at 100k by 20 over 100 rounds, roughly 36 percent, and about
0.81 ms per round at 20k by 10. `bench/bench_custom_objective.py` is the
harness. When the objective sits on a hot path, write it in Mojo against
`src/mojotrees/objective.mojo` instead.

## 14. What this walkthrough could not show

Collected in one place, because a reader deciding whether to adopt
mojotrees should not have to assemble this list from thirteen sections.

| Missing | Where it bites | Workaround |
|---|---|---|
| Inspection names at the package top level | `import mojotrees; mojotrees.dump_model` | `from mojotrees.inspection import dump_model`, which works today |
| `trees_to_dataframe` without pandas | Dependency-free environments | `trees_to_records`, the same rows as dicts |
| Continued training from an estimator | `fit` has no `init_model` | The functional API, `Dataset` plus `train(init_model=...)` |
| Early stopping in the functional API | `train()` keeps no per-round history | The estimators' `fit` |
| Continued training for ranking | LambdaRank state is not in the ensemble | None; retrain |
| Callbacks for softmax and ranking | No learning-rate schedule for multiclass | None; the trainers refuse rather than ignore |
| Validation with a Python objective callback | No early stopping for a custom objective | The Mojo API's `train_custom_with_metrics` |
| Validation, slicing, leaves, and contributions for sparse input | Sparse workflows | `X.toarray()`, when you can afford it |
| GPU for multiclass, sparse, or any run with an `eval_set` | Those runs raise on `device="gpu"` | The CPU path, which is the dependable one |
| `device="auto"` ever choosing the GPU | Nothing today; it always picks the CPU | Ask for `device="gpu"` explicitly |
| Any NVIDIA or AMD GPU result | Every claim about non-Apple GPUs | None. Nobody has run it |
| `zero_as_missing=true` | Datasets whose zeros mean absent | None; no alias accepts it |
| A LightGBM-shaped `best_score_` | Code that indexes it as a dict | `evals_result_`, which holds the whole grid |
| A published wheel for any platform | Installation anywhere | Build from source with `pixi run build-wheel` on macOS arm64 |
| `check_estimator` compliance | Strict scikit-learn integration | None; the suite has not been run, so the package is scikit-learn style and claims nothing more |

None of the above is a promise about when it lands. The repository's
planning files are not published and this table is a description of today.
