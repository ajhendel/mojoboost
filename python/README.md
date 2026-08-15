# mojotrees

Gradient boosted decision trees in [Mojo](https://www.modular.com/mojo),
with a scikit-learn style Python API backed by a CPython extension module
built from the same Mojo code.

mojotrees is a from-scratch GBDT library in the LightGBM family. It uses
histogram-based split finding, leaf-wise (best-first) tree growth, and
defaults matched to LightGBM. Objectives include squared error, binary
logistic, Poisson, and multiclass softmax, with sample weights and
bit-exact model save/load.

> **Experimental public alpha.** The feature surface is broad and training
> works end to end, but this is not yet a production replacement for
> LightGBM or XGBoost. Treat every capability according to the evidence in
> [docs/LIGHTGBM_PARITY.md](https://github.com/mojotrees/mojotrees/blob/main/docs/LIGHTGBM_PARITY.md)
> and
> [docs/GPU_VALIDATION.md](https://github.com/mojotrees/mojotrees/blob/main/docs/GPU_VALIDATION.md),
> report failures, and do not rely on unvalidated hardware or parameter
> combinations in production. The parity contract scores each capability on
> seven independent axes rather than one, because several things in the
> repository are implemented and reachable by nobody; a row saying
> `supported` is the one that means what it sounds like.

## Installing

The first public alpha is on PyPI:

```sh
pip install --pre mojotrees
```

The current `0.1.0a2` wheel supports **CPython 3.14 on Apple Silicon running
macOS 26 or newer**. It includes its runtime dependencies and requires no
Mojo, MAX, Pixi, or compiler. Unsupported interpreters and platforms fail
cleanly with `No matching distribution found`.

| State | What you type | Status today |
|---|---|---|
| Published alpha | `pip install --pre --only-binary=:all: mojotrees` | **Available:** `0.1.0a2`, CPython 3.14, Apple Silicon, macOS 26+ |
| A release wheel file | `pip install ./mojotrees-<version>-<tags>.whl` | **Available** from the release workflow |
| Source checkout with Pixi | `git clone`, `pixi install`, `pixi run build-python` | **Available** for contributors |

```sh
git clone https://github.com/mojotrees/mojotrees.git
cd mojotrees
pixi install
pixi run build-python
PYTHONPATH=python python -c "import mojotrees; print(mojotrees.__version__)"
```

[pixi](https://pixi.sh) resolves the pinned Mojo and MAX versions, so no
separate Mojo or MAX installation is needed. The published wheel bundles
the runtime libraries it links and needs no toolchain at all.

mojotrees publishes no source distribution, deliberately, so
`pip install mojotrees` can never turn into a Mojo compile on a machine with
no Mojo toolchain. It resolves to a wheel that matches your machine or it
refuses with "no matching distribution found".

The whole picture, with the wheel filename per platform, the first five
minutes step by step, and every error an install can produce, is in
[docs/INSTALLATION.md](https://github.com/mojotrees/mojotrees/blob/main/docs/INSTALLATION.md).

## The first five minutes

Import and diagnostics, a tiny regression, a validation set with early
stopping, a bit-exact save and load, and what each device value does on your
machine, in one standard-library-only script that prints each result.

```sh
python examples/install_smoke.py                     # installed package
PYTHONPATH=python python examples/install_smoke.py   # source checkout
```

The diagnostics it prints first are what an installation bug report needs,
and they are one call on their own.

```python
import mojotrees

mojotrees.show_versions()      # or build_info() for the same facts as a dict
```

That includes the one thing no other call can tell you: whether a GPU path
was compiled into this build. Availability is decided when the extension is
compiled rather than on the machine that runs it, so one wheel carries one
answer to everybody who installs it, and `gpu_available()` alone returns
`False` both for a build without a GPU path and for a GPU build with
`MOJOTREES_DISABLE_GPU=1` set.

## Usage

```python
from mojotrees import MojoTreesRegressor, MojoTreesClassifier

model = MojoTreesRegressor(num_leaves=31, n_estimators=100).fit(X, y)
pred = model.predict(X)          # numpy in/out when numpy is available
model.save("model.mbst")
model = MojoTreesRegressor.load("model.mbst")

clf = MojoTreesClassifier().fit(X, labels)   # binary or multiclass by labels
proba = clf.predict_proba(X)
```

`fit` accepts `sample_weight`. numpy is optional; plain Python sequences
work without it. Install with the `numpy` extra to pull it in.

Native LightGBM names are canonical (`min_data_in_leaf`, `min_child_hess`,
`lambda_l1`, `lambda_l2`, `bagging_fraction`, `bagging_freq`, `boosting`,
`device`). For easy migration from `LGBMRegressor` and `LGBMClassifier`,
their scikit-learn spellings are accepted too: `min_child_samples`,
`min_child_weight`, `reg_alpha`, `reg_lambda`, `subsample`,
`subsample_freq`, `boosting_type`, and `device_type`. Conflicting values
raise instead of silently choosing one.

`boosting="goss"` trains with Gradient-based One-Side Sampling instead of
on every row: each round keeps the `top_rate` share of rows with the
largest gradient magnitude, samples `other_rate` of the rest, and scales
the sampled rows to compensate. `goss_seed` makes the sample reproducible,
`goss_warmup_rounds` overrides LightGBM's automatic
`int(1 / learning_rate)` full-data rounds, and GOSS cannot be combined with
row bagging.

## Dataset, Booster, and train()

LightGBM's functional API is here too, over the same trainer:

```python
import mojotrees as mb

train_set = mb.Dataset(X, label=y)            # binned once, reused
booster = mb.train({"objective": "regression", "num_leaves": 31},
                   train_set, num_boost_round=100)

booster.predict(X_test)
booster.eval(train_set, "training")
booster.update(50)                            # 100 + 50 == the 150-round model
booster.save_model("model.mbst")
```

A `Dataset` carries `label`, `weight`, `group`, `init_score`,
`feature_name`, and `categorical_feature`, and is immutable once
constructed: LightGBM's `set_*` mutators would invalidate the bin edges
fitted from the data, so fields are constructor arguments instead. The
estimators hold the same `Booster` on `booster_`, and a model trained
through `train()` predicts identically to one trained through an
estimator with the same parameters.

Continued training covers the single-output objectives and softmax
multiclass; ranking raises, because LambdaRank gradients need per-query
state that the fitted ensemble does not carry.

## scikit-learn conventions

`get_params`, `set_params`, `fit`, `predict`, `predict_proba`, and `score`
are there, and a fitted estimator carries `n_features_in_`,
`feature_names_in_`, `classes_`, `feature_importances_`,
`best_iteration_`, and `device_`. `clone`, `Pipeline`, `GridSearchCV`, and
`cross_val_score` work, and estimators pickle. scikit-learn itself is
optional: nothing imports it except the `__sklearn_tags__` hook it calls.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("scale", StandardScaler()), ("gbdt", MojoTreesRegressor())])
search = GridSearchCV(pipe, {"gbdt__num_leaves": [15, 31]}, cv=3).fit(X, y)
```

Class labels may be of any single comparable type; they are sorted onto
`classes_` and `predict` returns them. `X` may contain `NaN`, the
missing-value marker, but not infinities, and `y` and `sample_weight` must
be finite, which is how LightGBM's own scikit-learn wrapper validates.
Pickling keeps the whole estimator; `save()`/`load()` keeps the model
alone, without the class labels, hyperparameters, feature names, or split
gains.

## Categorical features

`categorical_feature` names the columns whose integer codes are unordered
categories. It takes column indices, column names, a mix of the two, `None`
for no categorical feature, or LightGBM's default `"auto"`: every pandas
`category` column of `X`, and nothing else.

```python
model = MojoTreesRegressor(categorical_feature=["city"]).fit(df, y)
model.categorical_feature_          # [0]
```

Those columns are split by category set rather than by threshold, with no
one-hot expansion. A pandas `category` column is encoded by its labels and
the table is kept on the fitted estimator, so the same label reaches the
same category whatever a later frame numbers it; leaving such a column out
of an explicit `categorical_feature` raises rather than feeding its codes to
the numerical scan. Elsewhere the codes are whole numbers below 2**31, and a
negative code or `NaN` means missing: missing, unseen, and dropped
categories all route right. Pickling keeps the label tables;
`save()`/`load()` keeps the category tables but not the labels, so a loaded
model takes codes. `max_cat_to_onehot`, `max_cat_threshold`, `cat_smooth`,
`cat_l2`, and `min_data_per_group` are LightGBM's categorical
hyperparameters, with LightGBM's defaults.

## Prediction options

`predict` and `predict_proba` take LightGBM's prediction keywords:

```python
model.predict(X, raw_score=True)              # before the inverse link
model.predict(X, num_iteration=10)            # the first 10 iterations
model.predict(X, start_iteration=10)          # everything after them
model.predict(X, pred_leaf=True)              # leaf ordinals, one per tree
model.predict(X, pred_contrib=True)           # exact TreeSHAP contributions
model.predict(X, validate_features=True)      # names must match, not warn
```

`pred_contrib` returns one column per feature plus an expected-value
column, shape `(n_samples, n_features + 1)`, or
`(n_samples, n_classes * (n_features + 1))` in class-major blocks for the
multiclass classifier. Each row (each class block, for multiclass) sums to
that row's raw score exactly: these are TreeSHAP Shapley values, not a
split-gain heuristic. `raw_score` cannot be combined with it, since
contributions always explain the raw score.

`raw_score` is a no-op for the regressor and the ranker, which have no
inverse link; the binary classifier returns log-odds of shape
`(n_samples,)` and the multiclass classifier pre-softmax scores of shape
`(n_samples, n_classes)`, both as LightGBM does.

The iteration bounds clamp as LightGBM's do: a negative start becomes 0, a
start past the end selects nothing, and `num_iteration=None` or a value
<= 0 means every iteration from the start on. `None` therefore predicts
with `best_iteration_` iterations, which after early stopping is every tree
the model kept. The base score belongs to iteration 0, so `[0, k)` and
`[k, n)` sum to the whole raw score, and `num_iteration=k` reproduces a
`k`-round fit.

`pred_leaf` returns integers, `(n_samples, num_iteration)` for the
single-output estimators and `(n_samples, num_iteration * n_classes)` for
the multiclass classifier, whose column `i * n_classes + k` is class k's
tree in iteration i. Leaves are numbered per tree in node order, in
`[0, num_leaves)`, stably across `save`/`load` and pickling; the numbering
is mojotrees's own, not LightGBM's leaf id.

`raw_score=True` and `pred_leaf=True` together raise rather than letting one
win silently.

`check_estimator`'s full suite has not been run, so this is scikit-learn
style rather than a compliance claim. Known deviations: shared
hyperparameters are forwarded through `**kwargs`, so `inspect.signature`
does not list them (`get_params()` does), and `best_iteration_` is always
set, where LightGBM sets it only when early stopping ran.

## Validation sets and early stopping

Every estimator takes them, in LightGBM's spelling:

```python
model = MojoTreesRegressor(n_estimators=500).fit(
    X, y,
    eval_set=[(X_valid, y_valid)],   # or (X_valid, y_valid), or eval_X/eval_y
    eval_names=["holdout"],          # valid_0, valid_1, ... by default
    eval_sample_weight=[w_valid],
    eval_metric=["l2", "l1"],
    early_stopping_rounds=20,
)
model.best_iteration_        # where the primary metric peaked
model.n_iter_                # rounds actually trained
model.best_score_            # the primary metric's best value
model.evals_result_["holdout"]["l2"]   # index 0 is the base score alone
model.stopped_early_
```

`eval_metric` takes LightGBM's metric names (`l2`, `rmse`, `l1`,
`quantile`, `huber`, `mape`, `fair`, `poisson`, `gamma`, `gamma_deviance`,
`tweedie`, `cross_entropy`, `kullback_leibler`, `binary_logloss`,
`binary_error`, `auc`, `average_precision`, `multi_logloss`, `multi_error`,
`ndcg`, `map`, and their aliases), callables, or both, and defaults to the
objective's own loss. A name has to make sense for the model being fitted:
the regressor takes the regression metrics, the classifier the binary or
multiclass ones, the ranker `ndcg` and `map`. Predictions reach a built-in
metric through the objective's own inverse link, so `l2` on a poisson model
scores expected counts and `binary_logloss` scores probabilities. A name is computed by
`src/mojotrees/metrics.mojo`, so it agrees with the Mojo API by
construction, and `eval_sample_weight` weights it. A callable is
`f(y_true, y_pred) -> float`, called once per metric per validation set per
round with raw scores in `y_pred`; declare its direction with
`("name", f, True)`. Callables are handed unweighted predictions, so
combining one with `eval_sample_weight` raises rather than dropping the
weights quietly.

`early_stopping_rounds` stops once a watched metric has gone that many
rounds without improving by more than `min_delta`, and rolls the ensemble
back to the best round of `primary_metric` on the first validation set. It
needs an `eval_set` and says so otherwise. With `early_stopping_rounds=0`
nothing stops and nothing is rolled back, but every value is still
recorded.

The classifier encodes validation labels through the `classes_` it
recorded, so a label absent from training raises. The ranker takes
`eval_group`, one group array per validation set, and rejects
`eval_sample_weight`, since NDCG has no weighted LightGBM definition to
match. Validation is scored on the CPU, so `device="gpu"` with an
`eval_set` raises rather than falling back.

## Model inspection

The ensemble as structured data, in LightGBM's shapes, at the top level of
the package:

```python
import mojotrees as mb

schema = mb.dump_model(model)                  # the documented dump schema
frame = mb.trees_to_dataframe(model)           # one pandas row per node
records = mb.trees_to_records(model)           # the same, as dicts, no pandas
hist = mb.get_split_value_histogram(model, "age", bins=10)
```

`model` is a fitted estimator, a `Booster`, or the text
`Booster.model_to_string()` produces, so a model read back from a file can
be inspected without refitting. `feature_names=` names the features of a
model that carries none. Every key of the dump is documented in
[docs/MODEL_INSPECTION_SCHEMA.md](https://github.com/mojotrees/mojotrees/blob/main/docs/MODEL_INSPECTION_SCHEMA.md);
the two to branch on are `has_split_gain` and `has_node_count`.

One thing worth knowing before you build on it. A model this version wrote
carries per-node split gains, because model format v4 serializes them, so
`split_gain` is a measured number on every internal node and
`has_split_gain` is `True`. A model read back from a file an older version
wrote carries none, and every node reports `split_gain: None` with
`has_split_gain: False`. Branch on the flag rather than on the value: a
`None` means "this file never had it", and a `0.0` means a split whose
measured gain was zero. `trees_to_dataframe` and `trees_to_records` inherit
whichever of the two the dump was built from. The state of the native dump
seam is in
[docs/INTEGRATION_INVENTORY.md](https://github.com/mojotrees/mojotrees/blob/main/docs/INTEGRATION_INVENTORY.md).

## Cross-validation

LightGBM's `cv`, over the same trainer, returning the same
`{metric-mean, metric-stdv}` history:

```python
history = mb.cv({"objective": "regression", "num_leaves": 31},
                mb.Dataset(X, label=y),
                num_boost_round=100, nfold=5,
                early_stopping_rounds=10)
```

`folds`, `stratified`, `shuffle`, `metrics`, `feval`, `fpreproc`,
`init_model`, `eval_train_metric`, and `return_cvbooster` are all there, and
a caller-supplied splitter works as well as a fold count. Each fold is
binned from its own training rows rather than sliced out of one constructed
`Dataset`, so fold binning cannot leak across the split.

## Device selection

```python
from mojotrees import MojoTreesRegressor, gpu_available

model = MojoTreesRegressor(device="auto").fit(X, y)
model.device_          # the backend that actually ran: "cpu" or "gpu"
```

To ask what a device would do before committing to it, without handling an
exception:

```python
report = mb.explain_device_choice(X, y, device="gpu")
print(report)                  # the resolution and the reasons behind it
report.would_raise             # True when a fit would have raised
report.to_dict()               # JSON-serializable, for a log or a ticket
```

The backend in a report is the native engine's answer, the same one `fit`
would get. How much of the rest crosses depends on the extension you have,
and `report.contract` says which: `"full"` means `decide_device` answered
and the blocking reasons, warnings, memory estimate, and evidence
identifier are the engine's own, and `"narrow"` means only the older
`resolve_device` was there, in which case the report names the gates that
were skipped rather than guessing at them. An extension built from this
source registers `decide_device`, so `"full"` is what a current build
reports.

`device="cpu"` is the default and the dependable backend. `device="gpu"`
raises when no accelerator is available or when the GPU path does not
cover the workload, rather than falling back silently. Sparse input, an
`eval_set`, and a Python objective callback are the workloads it does not
cover today, and each says so by name. `device="auto"` picks for you and
currently always picks the CPU, because no benchmark has established a
workload size where end-to-end GPU training wins and no crossover threshold
ships enabled. `gpu_available()` reports whether this build can train on an
accelerator, which is decided when the extension is compiled rather than on
the machine that runs it.

The device is a training choice rather than part of the model, so saved
models are identical either way and a loaded estimator carries no
`device_`.

## SciPy sparse input

`fit` and `predict` accept any SciPy sparse matrix or array and keep it
sparse. Nothing is densified at any point.

```python
from scipy import sparse
from mojotrees import MojoTreesRegressor

X = sparse.random(100_000, 500, density=0.01, format="csr")
model = MojoTreesRegressor().fit(X, y)      # converted to CSC to fit
pred = model.predict(X)                     # converted to CSR to predict
```

Whatever format you pass is converted to the one that side of the boundary
wants, and a matrix that is not in SciPy's canonical form is copied before
its indices are sorted, so your matrix is never mutated.

An implicit zero is the numerical value 0.0, not a missing value: this is
LightGBM's default `zero_as_missing=false`, so a sparse fit equals the dense
fit of the same matrix with its gaps filled with zeros. Explicitly stored
zeros mean the same thing as the gaps. `NaN` is still the missing marker
wherever it is stored.

scipy is not a dependency: the wrapper duck-types the CSC/CSR interface and
imports nothing from scipy.

Not available for sparse input, each raising rather than densifying quietly:
`device="gpu"`, a Python objective callback, `eval_set` and early stopping,
ranking, and `pred_leaf` / `pred_contrib` / iteration slicing at predict
time.

## Platform support

The first wheel target is macOS on Apple silicon, with Linux x86_64 and
aarch64 after it. A wheel bundles the Mojo runtime libraries it needs, so no
Mojo or MAX installation is required at runtime. Intel Macs, Windows, and
free-threaded Python are out of scope.

The declared interpreter is CPython 3.14, which is the only one anything has
run on rather than a toolchain requirement; 3.13 is expected but unproven and
3.12 and earlier are blocked by an entry point added in 3.13, all worked
through in
[docs/PYTHON_SUPPORT.md](https://github.com/mojotrees/mojotrees/blob/main/docs/PYTHON_SUPPORT.md).
The Linux platform tag is likewise unsettled: the default `linux_x86_64` and
`linux_aarch64` tags are rejected by every index, and a `manylinux` tag needs
a measured glibc floor first.

The CPython 3.14 Apple Silicon wheel has now been published and clean-install
validated. Every target, its artifact name, and the evidence behind its
status is in
[docs/PLATFORM_MATRIX.md](https://github.com/mojotrees/mojotrees/blob/main/docs/PLATFORM_MATRIX.md),
where the rule is that a platform counts as validated when hardware ran the
artifact and somebody wrote down what happened.

For unsupported targets, build from source with [pixi](https://pixi.sh)
using the instructions above.

## When something goes wrong

| What you see | What it means |
|---|---|
| `No matching distribution found for mojotrees` | PyPI has no wheel matching your Python, operating system, and architecture. Use the supported target or build from source |
| `Requires-Python >=3.14` in pip's output | Your interpreter is older than the declared floor |
| `... is not a supported wheel on this platform` | The wheel's tags do not describe your machine. Do not force it |
| `ImportError: cannot import name '_mojotrees' from 'mojotrees'` | Source checkout without a built extension. Run `pixi run build-python` |
| `ImportError: ... Library not loaded: @rpath/libKGENCompilerRTShared.dylib` | The MAX runtime libraries were not found. Run through `pixi run`, or install a self-contained wheel |
| `RuntimeError: device 'gpu' requested but no accelerator is available` | This build has no GPU path. Availability is fixed when the extension is compiled, not at runtime |
| `RuntimeError: validation metrics are scored on the CPU` | An `eval_set` with `device="gpu"`. Use `device="cpu"` or `"auto"` |
| `RuntimeError: sparse input trains on the CPU` | There is no sparse GPU kernel. Use `device="cpu"`, `"auto"`, or densify |
| `device="auto"` chose the CPU and said nothing | Expected. The crossover table is empty, so `auto` keeps the CPU everywhere |

Each case, with the full message and what to do about it, is in
[docs/INSTALLATION.md](https://github.com/mojotrees/mojotrees/blob/main/docs/INSTALLATION.md#when-something-goes-wrong).
Installation problems are in scope for the
[bug report template](https://github.com/mojotrees/mojotrees/issues/new?template=bug_report.yml);
accelerator results belong in the
[hardware validation template](https://github.com/mojotrees/mojotrees/issues/new?template=hardware_validation.yml).

## Links

- Source, benchmarks against LightGBM, and the native Mojo API:
  [github.com/mojotrees/mojotrees](https://github.com/mojotrees/mojotrees)
- [Installation](https://github.com/mojotrees/mojotrees/blob/main/docs/INSTALLATION.md)
- [LightGBM parity contract](https://github.com/mojotrees/mojotrees/blob/main/docs/LIGHTGBM_PARITY.md)
- [Capability levels](https://github.com/mojotrees/mojotrees/blob/main/docs/CAPABILITY_LEVELS.md),
  the seven words the parity contract scores against
- [Integration inventory](https://github.com/mojotrees/mojotrees/blob/main/docs/INTEGRATION_INVENTORY.md),
  what is written here and reachable by nobody
- [GPU validation record](https://github.com/mojotrees/mojotrees/blob/main/docs/GPU_VALIDATION.md)
- [Device selection policy](https://github.com/mojotrees/mojotrees/blob/main/docs/DEVICE_SELECTION.md)
- [Contributing](https://github.com/mojotrees/mojotrees/blob/main/CONTRIBUTING.md)

## License

Apache-2.0
