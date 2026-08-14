# LightGBM parity contract

Contract version: 1
Audited against: **LightGBM 4.7.0** (the version pinned in the `bench` pixi
environment; `lightgbm.__version__` reported `4.7.0` when this was written)
Audited: 2026-08-14, mojoboost at commit `6190f88` plus the working tree of
that day

Several features landed in the working tree while this audit was being
written, so treat the statuses as a snapshot rather than as news: run
`python3 tools/check_parity.py` and re-read the rows you care about before
relying on one. The check is what keeps the snapshot from rotting silently;
it cannot tell that something moved from `deferred` to `supported`, which is
a judgment for whoever moved it.

This file is the authoritative statement of what mojoboost does and does not
do relative to LightGBM's public interface. It is a contract, not a wish
list: a row saying `supported` means the named behavior exists, is reachable
from a public entry point, and has a test that would fail if it stopped
working. `tools/check_parity.py` enforces the parts of that claim a script
can check.

## How to use this document

- **Before implementing** a LightGBM feature, find its row. If it already
  says `supported` or `partial`, strengthen what is there instead of adding
  a second way to do the same thing.
- **When a public contract changes**, update the row in the same change that
  updates the code, the bindings, the docs, and the serialization format.
- **When you change a row**, run `python3 tools/check_parity.py`. It fails if
  a row that must stay `supported` was deleted or downgraded, if a cited file
  no longer exists, or if a named public symbol disappeared.

## Status vocabulary

| Status | Meaning |
|---|---|
| `supported` | Reachable from a public entry point, tested, and semantically equivalent to LightGBM except where the notes say otherwise |
| `partial` | Exists but does not cover LightGBM's full surface (fewer inputs, one backend, Mojo only, and so on). The notes say what is missing |
| `different` | Deliberately not LightGBM's design. The notes say why, and what mojoboost does instead |
| `deferred` | Agreed to be worth doing, out of scope for v1. The notes say why, not merely that it is absent |
| `unsupported` | Not planned. The notes say why not |

`deferred` and `unsupported` are not interchangeable. Nothing difficult and
high value is marked `unsupported` to keep the matrix clean; the expensive
items (feature contributions, sparse input, a real distributed transport,
non-Apple GPU validation) are `deferred` with a named reason and, where one
exists, a task number in `CLAUDE_CODE_FEATURE_COMPLETE_TASKS.txt`.

## v1 scope

**In v1.** A user should be able to migrate a LightGBM tabular workload that
uses the scikit-learn estimators, without dropping to a fork of their code:

- the three scikit-learn style estimators, LightGBM's parameter names, and
  the aliases LightGBM's own estimators use
- the objectives and metrics ordinary tabular work uses (regression family,
  binary, multiclass, ranking, Poisson, quantile, MAE, Huber, custom)
- validation sets, evaluation metrics, and early stopping
- all four prediction modes: response, raw score, leaf index, and feature
  contributions
- continued training from a fitted model, and initial scores
- native categorical features and missing values
- CSR/CSC input without densification
- Dataset-level and Booster-level APIs for users who do not want an
  estimator
- cross validation
- model inspection, dumping, and editing
- CPU parity on every objective, GPU coverage for the objectives that share
  the per-row gradient/Hessian interface
- Linux and macOS wheels

**Out of v1, deferred.** Real distributed transports and the Dask
integration built on them, unbiased LambdaRank, DART and random-forest
boosting, linear trees, and CEGB. Each is a subsystem, and each is listed
with its reason below rather than dropped.

**Never.** LightGBM's file-based CLI surface (`config`, `data`, `valid`,
`output_model`, `task`, `convert_model`, and the parameters that only feed
it), its deprecated aliases, and any parameter whose only purpose is to
select between implementations mojoboost does not have.

---

## 1. Top-level Python package symbols

Every name in `lightgbm.__all__` for 4.7.0, verified by importing the pinned
package.

| LightGBM symbol | Status | Notes | mojoboost |
|---|---|---|---|
| `LGBMRegressor` | supported | `MojoBoostRegressor`. Named for the library, not for LightGBM; behavior and parameter names match where documented | `python/mojoboost/__init__.py` |
| `LGBMClassifier` | supported | `MojoBoostClassifier`. Binary and softmax multiclass, chosen from the label count | `python/mojoboost/__init__.py` |
| `LGBMRanker` | supported | `MojoBoostRanker`, LambdaRank | `python/mojoboost/__init__.py` |
| `LGBMModel` | different | No shared public base class. `_Base` holds the shared hyperparameters but is private, because a bare `LGBMModel` with `objective=` selecting the task is a second way to spell what the three estimators already do | `python/mojoboost/__init__.py` |
| `Booster` | deferred | The fitted model is an opaque handle behind the estimators. A Booster-level API (`update`, `eval`, `dump_model`, iteration control) is v1 work, task 7 | `src/mojoboost/boosting.mojo` |
| `Dataset` | partial | A Mojo `Dataset` exists (`src/mojoboost/dataset.mojo`: construction, labels, weights, groups, `train_dataset`, `update_dataset`), but it is not exported from the package and has no Python surface, so no user can reach it. Binning otherwise happens inside `fit`. v1 work, task 7 | `src/mojoboost/dataset.mojo` |
| `train` | deferred | The functional training entry point arrives with `Booster`/`Dataset`, task 7 | — |
| `cv` | deferred | Needs `Dataset` and the callback system first. v1 work, task 15 | — |
| `CVBooster` | deferred | With `cv`, task 15 | — |
| `early_stopping` | partial | Early stopping exists, as `fit(early_stopping_rounds=..., min_delta=...)` rather than as a callback object. `first_metric_only` has no equivalent; every metric flagged for early stopping is watched | `python/mojoboost/__init__.py`, `src/mojoboost/custom_metric.mojo` |
| `log_evaluation` | deferred | Needs the callback system, task 3. mojoboost prints nothing during training today | — |
| `record_evaluation` | partial | `evals_result_` is populated by `fit` directly, without a callback | `python/mojoboost/__init__.py` |
| `reset_parameter` | deferred | Per-iteration parameter schedules need the callback system, task 3 | — |
| `EarlyStopException` | deferred | Part of the callback contract, task 3 | — |
| `EvalResult` | deferred | Part of the callback contract, task 3 | — |
| `register_logger` | unsupported | mojoboost has no logging layer to redirect. Training is silent by design; adding a logger to redirect is not a goal | — |
| `plot_importance` | unsupported | Plotting belongs in the caller's plotting library. `feature_importances_` is the data; matplotlib is not a dependency mojoboost will take | — |
| `plot_metric` | unsupported | Same reason. `evals_result_` is the data | — |
| `plot_split_value_histogram` | deferred | The plot is out of scope, but the underlying `get_split_value_histogram` data is model inspection and is v1 work, task 14 | — |
| `plot_tree` / `create_tree_digraph` | deferred | The graphviz rendering is out of scope; the structured tree dump it renders is v1 work, task 14 | — |
| `Sequence` | deferred | The incremental-data protocol is part of the `Dataset` work, tasks 7 and 10 | — |
| `DaskLGBMRegressor` / `DaskLGBMClassifier` / `DaskLGBMRanker` | deferred | Dask estimators are a thin layer over real distributed training, which mojoboost does not have yet. Building them on the in-process prototype would be a distribution claim with nothing behind it. Task 17, after task 16 | `docs/distributed.md` |

## 2. Estimator constructor parameters

Every parameter of `lightgbm.LGBMModel.__init__` in 4.7.0.

| LightGBM parameter | Status | Notes |
|---|---|---|
| `boosting_type` | partial | Accepted as `boosting` (LightGBM's native name) with `boosting_type` as an alias. `gbdt` and `goss` only; `dart` and `rf` are deferred, see section 7 |
| `num_leaves` | supported | Same default (31) |
| `max_depth` | supported | Same default (-1). Leaf-wise growth is preserved under the limit |
| `learning_rate` | supported | Same default (0.1) |
| `n_estimators` | supported | Same default (100) |
| `subsample_for_bin` | deferred | mojoboost bins from every row rather than a sample. Correct but slower on very large data; the sampled binner is a performance item, not a semantic one |
| `objective` | partial | Regressor: `regression`, `huber`, `quantile`, `mae`/`regression_l1`, or a callable. Classifier: rejected, the task comes from the labels. Missing objectives are listed in section 8 |
| `class_weight` | deferred | Per-class weighting is expressible today only by expanding it into `sample_weight`. v1 work, task 11 |
| `min_split_gain` | unsupported | LightGBM's `min_gain_to_split`. Not implemented; see section 7 |
| `min_child_weight` | supported | Alias for `min_child_hess` (LightGBM's `min_sum_hessian_in_leaf`), default 1e-3 |
| `min_child_samples` | supported | Alias for `min_data_in_leaf`, default 20 |
| `subsample` | supported | Alias for `bagging_fraction` |
| `subsample_freq` | supported | Alias for `bagging_freq` |
| `colsample_bytree` | partial | mojoboost's name is `feature_fraction`; `colsample_bytree` itself is **not** accepted as an alias yet, though `feature_fraction_bynode` is spelled natively. Adding the two `colsample_*` spellings is a v1 gap |
| `reg_alpha` | supported | Alias for `lambda_l1` |
| `reg_lambda` | supported | Alias for `lambda_l2`. Note the default differs: mojoboost defaults to 1.0, LightGBM to 0.0. Documented in the README defaults table |
| `random_state` | different | mojoboost has no single global seed. Each sampler takes its own (`bagging_seed`, `feature_fraction_seed`, `goss_seed`), and every stream is counter-based, so a draw depends only on its seed and index and never on history. One global seed would reintroduce the ordering dependence that design removes |
| `n_jobs` | different | Thread count is `MOJOBOOST_NUM_WORKERS`, an environment variable, not a model parameter, because it changes how a model is computed and never what it equals. Same rule as the GPU tiling knobs |
| `importance_type` | supported | `split` (default) and `gain`, LightGBM's two types |
| `**kwargs` (arbitrary core parameters) | different | Unknown keyword arguments raise. LightGBM forwards them to the C++ config, which silently accepts typos of parameters it does not know |

Additional mojoboost constructor parameters that LightGBM spells only as
core parameters (`use_missing`, `categorical_feature`, `max_cat_to_onehot`,
`max_cat_threshold`, `cat_smooth`, `cat_l2`, `min_data_per_group`,
`interaction_constraints`, `monotone_constraints`, `bagging_seed`,
`top_rate`, `other_rate`, `goss_seed`, `feature_fraction`,
`feature_fraction_bynode`, `feature_fraction_seed`, `max_bin`, `device`)
are covered in section 7.

**Alias rule, intentionally different.** Setting a parameter and its alias
to different non-default values raises. LightGBM warns and keeps one. A typo
that silently trains a different model is worse than a failed call.

## 3. Estimator `fit` and `predict`

| LightGBM argument | Status | Notes |
|---|---|---|
| `fit(X, y)` | supported | numpy, pandas, or plain sequences |
| `fit(sample_weight=)` | supported | Weighted gradients, Hessians, and base scores; zero-weight rows drop out |
| `fit(init_score=)` | deferred | Starting from an existing score vector is v1 work, task 6 |
| `fit(group=)` | supported | Ranker only, LightGBM's `group` array. `group_from_query_ids` builds it from a query id column |
| `fit(eval_set=)` | supported | List of `(X, y)` pairs. Requires `eval_metric`. Single-output only: a multiclass classifier with `eval_set` raises rather than scoring something it cannot score |
| `fit(eval_names=)` | supported | Names for the validation sets, used as `evals_result_` keys |
| `fit(eval_sample_weight=)` | different | Validation rows are never weighted, matching `train_with_valid`. A weighted validation metric is the caller's to compute inside their `eval_metric` |
| `fit(eval_init_score=)` | deferred | With `init_score`, task 6 |
| `fit(eval_metric=)` | partial | Caller-supplied callables, tuples, or dicts. LightGBM's built-in metric **names** (`"auc"`, `"rmse"`, ...) are not accepted as strings yet, though the Mojo library implements several of them; wiring the built-in names through is a v1 gap. The direction (`higher_is_better`) is declared up front rather than returned per call, because early stopping needs it before the first evaluation |
| `fit(feature_name=)` | partial | Feature names are read from a pandas frame's columns into `feature_names_in_` and checked at predict time. An explicit `feature_name=` argument, and carrying names into the model file, are not there |
| `fit(categorical_feature=)` | supported | Accepted as `categorical_feature` on the constructor (LightGBM's name) rather than on `fit`, because scikit-learn's clone contract keeps hyperparameters on the estimator. Indices, column names, or `"auto"` (the default, meaning every pandas `category` column). One difference: a `category` column left out of an explicit list raises, where LightGBM quietly feeds its codes to the numerical scan |
| `fit(callbacks=)` | deferred | The callback system is task 3 |
| `fit(init_model=)` | deferred | Continued training is task 6 |
| `predict(X)` | supported | Response scale, matching LightGBM's default |
| `predict(raw_score=)` | supported | Scores on the link scale. The objectives without a link (squared error, huber, quantile, L1) predict raw either way |
| `predict(start_iteration=)` / `predict(num_iteration=)` | supported | A slice of the ensemble, LightGBM's pair. `num_iteration=None` means every iteration the model kept |
| `predict(pred_leaf=)` | supported | Leaf index per tree. Combining it with `raw_score` raises, where LightGBM silently lets one win |
| `predict(pred_contrib=)` | partial | Exact TreeSHAP contributions exist in Mojo and are exported (`predict_contrib`, `predict_contrib_multiclass`, `src/mojoboost/contrib.mojo`). No estimator argument reaches them yet |
| `predict(validate_features=)` | supported | Same flag. A name mismatch raises either way, as scikit-learn already refuses to predict through one; the flag turns the one-sided cases from warnings into errors |
| `score(X, y)` | supported | R^2 for the regressor, accuracy for the classifier, mean NDCG for the ranker |

## 4. Fitted attributes

| LightGBM attribute | Status | Notes |
|---|---|---|
| `n_features_in_` | supported | |
| `feature_names_in_` | supported | Set when `X` carried string column names |
| `n_features_` | different | LightGBM's pre-scikit-learn spelling of `n_features_in_`. Not duplicated |
| `classes_` / `n_classes_` | supported | Classifier. Labels of any single comparable type, sorted |
| `feature_importances_` | supported | Both kinds computed at fit time. A model read back with `load()` reports zero gain importance and warns, because gains are not in the file |
| `best_iteration_` | different | Always set, to the number of iterations kept. LightGBM sets it only when early stopping ran |
| `best_score_` | partial | A single float, the primary metric's best value on the first validation set. LightGBM's is a nested dict over sets and metrics |
| `evals_result_` | supported | `{valid_name: {metric_name: [values]}}`. Index 0 is the base-score-only model, so entry `i` is the score after `i` trees; LightGBM starts at the first iteration |
| `booster_` | deferred | With the Booster API, task 7 |
| `objective_` | deferred | The resolved objective name is not exposed; `objective` is echoed back as given |
| `n_estimators_` / `n_iter_` | different | `best_iteration_` reports the kept iteration count. Two more names for the same number are not added |
| `feature_name_` | deferred | With the Booster API, task 7 |
| `device_` (mojoboost) | different | Not a LightGBM attribute. Records which backend actually ran, because `device="auto"` makes that a runtime outcome |
| `stopped_early_` (mojoboost) | different | Not a LightGBM attribute. True when early stopping fired, which `best_iteration_ < n_estimators` alone does not distinguish from objective convergence |

## 5. Booster and Dataset APIs

The whole of `lightgbm.Booster` and `lightgbm.Dataset` is `deferred` to
task 7 except where noted. They are grouped rather than listed
method-by-method because none of them is reachable today and the reason is
the same for all: mojoboost's fitted model is an opaque handle and its
dataset is built inside `fit`.

| LightGBM API group | Status | Notes |
|---|---|---|
| `Booster` construction, `update`, `rollback_one_iter`, `current_iteration`, `reset_parameter` | deferred | Incremental training control, task 7 with task 6 |
| `Booster.predict` with all prediction modes | partial | Response, raw score, leaf index, and iteration ranges are reachable through the estimators' `predict`; feature contributions are Mojo only. Section 3 has the detail. The Booster object itself is task 7 |
| `Booster.save_model` / `model_to_string` / `model_from_string` | different | mojoboost has `save()`/`load()` and a versioned text format (`src/mojoboost/serialize.mojo`) that stores floats as raw bit patterns so a round trip predicts bit-exactly. The format is mojoboost's own and is not LightGBM-readable; a converter is not planned |
| `Booster.dump_model` / `trees_to_dataframe` | deferred | Structured model inspection, task 14 |
| `Booster.feature_importance` | supported | Reachable as `feature_importances_`; both `split` and `gain` |
| `Booster.num_feature` / `num_trees` / `num_model_per_iteration` | partial | `num_trees`, `num_iterations`, `n_features` exist in the extension module (`bindings/_mojoboost.mojo`) and back `best_iteration_` and `n_features_in_`; they are not public Python methods |
| `Booster.eval` / `eval_train` / `eval_valid` / `add_valid` | partial | Validation scoring happens inside `fit` via `eval_set`. Post-hoc evaluation on a new set needs the Booster object, task 7 |
| `Booster.get_leaf_output` / `set_leaf_output` / `shuffle_models` / `refit` | deferred | Model editing, task 14 (`refit` also needs task 6) |
| `Booster.get_split_value_histogram` | deferred | Model inspection, task 14 |
| `Booster.lower_bound` / `upper_bound` | deferred | Model inspection, task 14 |
| `Booster.free_dataset` / `set_train_data_name` | different | No user-visible dataset object to free or name |
| `Booster.set_network` / `free_network` | deferred | Distributed training, task 16 |
| `Dataset` construction, `construct`, `subset`, `create_valid`, `set_reference` | partial | Construction and reuse exist in Mojo (`src/mojoboost/dataset.mojo`); nothing is exported or reachable from Python. Task 7 |
| `Dataset.set_field` / `get_field` and the typed accessors (`label`, `weight`, `group`, `init_score`, `position`) | partial | Labels, weights, and groups exist on the Mojo `Dataset`; `init_score` and `position` do not. Task 7; `init_score` also task 6, `position` also depends on unbiased LambdaRank |
| `Dataset.save_binary` / `get_data` / `feature_num_bin` / `add_features_from` | deferred | Task 7 |
| `Dataset.set_categorical_feature` / `set_feature_name` | partial | Both are constructor parameters on the estimators instead (`categorical_feature`; feature names come from the frame) |

## 6. Data inputs

| LightGBM input | Status | Notes |
|---|---|---|
| 2-D numpy array | supported | Converted to column-major float64 |
| pandas `DataFrame` | supported | Column names recorded as `feature_names_in_` |
| Python lists / sequences | supported | Works with numpy absent, which is also how the wheel is smoke-tested |
| `scipy.sparse` CSR | deferred | Rejected explicitly rather than densified silently. The Mojo library has CSC/CSR structures and sparse histogram builders (`src/mojoboost/sparse.mojo`, `src/mojoboost/histogram_sparse.mojo`, `src/mojoboost/tree_sparse.mojo`), but they are not exported from the package, not reachable from Python, and have no tests. v1 work, task 9 |
| `scipy.sparse` CSC | deferred | Same |
| `Dataset` from a file path | unsupported | Part of the file-based CLI surface mojoboost does not implement |
| `Sequence` / batched construction | deferred | Task 10, with the `Dataset` object |
| pyarrow tables and arrays | deferred | Task 10 |
| polars frames | deferred | Task 10 |
| `datatable` frames | unsupported | LightGBM's own support is legacy; not worth matching |
| NaN as missing | supported | `use_missing`, LightGBM's parameter, on by default. A reserved bin, a per-node default direction, and the same direction at predict time |
| `zero_as_missing` | deferred | Only meaningful together with sparse input, task 9 |
| Infinities in `X` | different | Rejected. LightGBM's own scikit-learn wrapper validates with `force_all_finite="allow-nan"`, which permits infinities into the C++ binner; mojoboost refuses them rather than binning them as extreme finite values by accident |

## 7. Core parameters

All 141 canonical parameter names in LightGBM 4.7.0, from
`lightgbm.basic._ConfigAliases`. Aliases are omitted; section 2 lists the
aliases mojoboost accepts.

### Core and objective

| Parameter | Status | Notes |
|---|---|---|
| `objective` | partial | Section 8 |
| `boosting` | partial | `gbdt`, `goss`. `dart` and `rf` deferred |
| `data_sample_strategy` | partial | `bagging` and `goss` are both implemented, selected through `boosting="goss"` (LightGBM 3.x spelling) rather than through this newer parameter |
| `num_iterations` | supported | Spelled `n_estimators` in Python, `BoosterParams.n_rounds` in Mojo |
| `learning_rate` | supported | |
| `num_leaves` | supported | |
| `tree_learner` | different | mojoboost has one grower. The distributed prototype is selected by calling `train_distributed`, not by a string |
| `num_threads` | different | `MOJOBOOST_NUM_WORKERS`, see section 2 |
| `device_type` | supported | Accepted as `device` with `device_type` as an alias. Values differ: mojoboost has `cpu`, `gpu`, `auto`; LightGBM has `cpu`, `gpu`, `cuda`. One portable GPU backend, so no vendor split, and `auto` is an addition |
| `seed` | different | Per-sampler seeds, section 2 |
| `deterministic` | different | Always on. Determinism is a property of the implementation (counter-based sampling, fixed-point GPU histogram reduction), not a toggle |
| `num_class` | supported | Inferred from the labels in Python; an explicit argument to `train_multiclass`/`fit_multiclass` in Mojo |

### Learning control

| Parameter | Status | Notes |
|---|---|---|
| `max_depth` | supported | |
| `min_data_in_leaf` | supported | |
| `min_sum_hessian_in_leaf` | supported | Spelled `min_child_hess` |
| `bagging_fraction` | supported | |
| `pos_bagging_fraction` / `neg_bagging_fraction` | deferred | Class-conditional bagging for unbalanced binary data. v1 work with `is_unbalance`, task 11 |
| `bagging_freq` | supported | |
| `bagging_seed` | supported | |
| `bagging_by_query` | different | Always on for the ranker. A half-sampled query would be normalized against a maxDCG no served ranking ever had, so mojoboost does not offer the row-sampling variant |
| `feature_fraction` | supported | |
| `feature_fraction_bynode` | supported | |
| `feature_fraction_seed` | supported | |
| `extra_trees` / `extra_seed` | deferred | Randomized split thresholds. v1 work, task 12 |
| `early_stopping_round` | supported | `fit(early_stopping_rounds=)`, and `train_with_valid` / `train_with_metrics` in Mojo |
| `early_stopping_min_delta` | supported | `fit(min_delta=)`. Same strict-improvement rule as LightGBM |
| `first_metric_only` | different | Every metric flagged for early stopping is watched, and the ensemble is truncated to the best round of the **primary** metric on the **first** validation set. LightGBM truncates to the best iteration of whichever pair ran out of patience first, which makes the kept model depend on scheduling |
| `max_delta_step` | partial | Fixed at LightGBM's Poisson value (`poisson_max_delta_step`, 0.7) inside the Poisson objective. Not a user parameter for other objectives |
| `lambda_l1` | supported | LightGBM's `ThresholdL1` soft thresholding, applied to split gains and leaf values |
| `lambda_l2` | supported | Default differs (1.0 here, 0.0 in LightGBM); documented in the README |
| `linear_lambda` | unsupported | Only meaningful with `linear_tree` |
| `min_gain_to_split` | unsupported | Not implemented. A gain floor is cheap to add and is v1 work, task 12; listed here as unsupported because today a caller who sets it gets nothing |
| `drop_rate` / `max_drop` / `skip_drop` / `xgboost_dart_mode` / `uniform_drop` / `drop_seed` | deferred | DART parameters. DART itself is deferred, see `boosting` |
| `top_rate` / `other_rate` | supported | GOSS. Same defaults, same `\|grad * hess\|` importance, same warmup rule |
| `min_data_per_group` | supported | Categorical |
| `max_cat_threshold` | supported | Categorical |
| `cat_l2` | supported | Categorical |
| `cat_smooth` | supported | Categorical |
| `max_cat_to_onehot` | supported | Categorical |
| `top_k` | deferred | Voting-parallel only; with task 16 |
| `monotone_constraints` | supported | Per-feature -1/0/1. The guarantee holds at any feature value, not only on the training data |
| `monotone_constraints_method` | different | One method. LightGBM's `basic`/`intermediate`/`advanced` choice is an artifact of three implementations; mojoboost's bounds propagation is the exact one |
| `monotone_penalty` | deferred | Depth-scaled penalty on constrained splits. Low value relative to the constraint itself |
| `feature_contri` | deferred | Per-feature gain multipliers. v1 work, task 12 |
| `forcedsplits_filename` | unsupported | File-based CLI surface |
| `refit_decay_rate` | deferred | With `refit`, task 14 |
| `cegb_tradeoff` / `cegb_penalty_split` / `cegb_penalty_feature_lazy` / `cegb_penalty_feature_coupled` | deferred | Cost-effective gradient boosting. A whole subsystem, and rare in practice |
| `path_smooth` | deferred | Leaf-value smoothing toward the parent. v1 work, task 12 |
| `interaction_constraints` | supported | LightGBM's per-branch allowed-feature rule, including the sharp edge that a feature in no group is never split on |
| `verbosity` | different | Training is silent. There is no logging layer to turn up |
| `input_model` / `output_model` / `saved_feature_importance_type` / `snapshot_freq` | unsupported | File-based CLI surface. `save()`/`load()` cover the model itself |
| `use_quantized_grad` / `num_grad_quant_bins` / `quant_train_renew_leaf` / `stochastic_rounding` | deferred | Quantized-gradient training. Interesting for the GPU path; nothing depends on it today |

### Dataset and IO

| Parameter | Status | Notes |
|---|---|---|
| `max_bin` | supported | Default 255 |
| `max_bin_by_feature` | deferred | Per-feature bin counts. Straightforward once the binner takes a vector; low demand |
| `min_data_in_bin` | partial | Enforced for categorical features (`min_data_per_group` governs the sorted search); the numerical binner has no minimum-population rule |
| `data_random_seed` | different | Binning is deterministic and reads every row, so there is no sampling seed to set |
| `bin_construct_sample_cnt` | deferred | See `subsample_for_bin` in section 2 |
| `is_enable_sparse` | deferred | With sparse input, task 9 |
| `enable_bundle` | deferred | Exclusive Feature Bundling. v1 work, task 13 |
| `use_missing` | supported | |
| `zero_as_missing` | deferred | With sparse input, task 9 |
| `feature_pre_filter` | different | mojoboost does not drop features during binning, so there is nothing to disable. A constant feature simply never yields a positive-gain split |
| `pre_partition` | deferred | Distributed, task 16 |
| `two_round` / `header` / `label_column` / `weight_column` / `group_column` / `ignore_column` / `parser_config_file` / `precise_float_parser` / `forcedbins_filename` / `save_binary` | unsupported | File-based CLI surface: mojoboost takes arrays, not files with headers |
| `categorical_feature` | supported | Indices, names, or `"auto"`. pandas `category` columns are label-encoded by the estimator, and the encoding is kept for prediction; the model file carries the category tables but not the labels, so a model read back from disk takes integer codes |
| `data` / `valid` / `config` / `task` / `convert_model` / `convert_model_language` / `output_result` | unsupported | File-based CLI surface |
| `histogram_pool_size` | different | Histogram memory is pooled per grower without a user cap |

### Objective-specific

| Parameter | Status | Notes |
|---|---|---|
| `boost_from_average` | different | Always on for the built-in objectives, always off for custom ones (the framework does not know the link). LightGBM makes it a toggle |
| `is_unbalance` / `scale_pos_weight` | deferred | Unbalanced-binary shortcuts. v1 work, task 11 |
| `sigmoid` | partial | Supported for LambdaRank. Not exposed for the binary objective, which uses the standard logistic |
| `alpha` | supported | Huber transition point and quantile level, LightGBM's meanings |
| `fair_c` | deferred | With the `fair` objective, task 11 |
| `poisson_max_delta_step` | different | Fixed at LightGBM's default 0.7 rather than exposed |
| `tweedie_variance_power` | deferred | With the `tweedie` objective, task 11 |
| `lambdarank_truncation_level` | supported | Default 30, and also the maxDCG cutoff, as in LightGBM |
| `lambdarank_norm` | supported | Default on |
| `label_gain` | different | Fixed at LightGBM's default `2^i - 1` for labels 0..30, which is also why labels outside that range are rejected. A user-supplied gain vector is deferred |
| `lambdarank_position_bias_regularization` | deferred | Part of unbiased LambdaRank, which is out of v1 |
| `objective_seed` | different | No objective in mojoboost draws random numbers |
| `reg_sqrt` | deferred | `sqrt`-transformed regression. Rare |
| `multi_error_top_k` / `auc_mu_weights` | deferred | With the `multi_error` and `auc_mu` metrics, task 11 |

### Metric

| Parameter | Status | Notes |
|---|---|---|
| `metric` | partial | Section 9 |
| `metric_freq` / `is_provide_training_metric` | different | Metrics are evaluated every round on the validation sets only; the training set is not scored automatically |
| `eval_at` | partial | `ndcg_eval_at`, a single cutoff for the ranker's `score`. LightGBM takes a list; `ndcg_score` takes any cutoff, and `ndcg_at_cutoffs` in Mojo takes several at once |

### Network, GPU, and prediction

| Parameter | Status | Notes |
|---|---|---|
| `num_machines` / `local_listen_port` / `time_out` / `machine_list_filename` / `machines` | deferred | No transport exists. `src/mojoboost/collective.mojo` defines the contract a transport would implement; `docs/distributed.md` states what is and is not built. Task 16 |
| `gpu_platform_id` / `gpu_device_id` / `gpu_device_id_list` / `gpu_use_dp` / `num_gpu` | different | One portable backend, one device, Float64 host arithmetic with a fixed-point device reduction. There is no OpenCL platform to select, and no double-precision toggle because the reduction is integer |
| `force_col_wise` / `force_row_wise` | different | mojoboost builds histograms feature-major, always. The choice exists in LightGBM to trade off multi-threading strategies; here the parallel dispatch is governed by `MOJOBOOST_PARALLEL_MIN_OPS` |
| `predict_raw_score` | supported | Section 3 |
| `predict_leaf_index` | supported | Section 3 |
| `predict_contrib` | partial | Mojo only, section 3 |
| `num_iteration_predict` / `start_iteration_predict` | supported | Section 3 |
| `pred_early_stop` / `pred_early_stop_freq` / `pred_early_stop_margin` | unsupported | Early-exit prediction trades accuracy for latency with no error bound. Not planned |
| `predict_disable_shape_check` | different | Shape and feature-name checks are always on. Disabling them turns a caught error into a silently wrong prediction |

## 8. Objectives

Every objective name accepted by LightGBM 4.7.0, verified by training a
1-round model with each name against the pinned install.

| LightGBM objective | Status | mojoboost | Evidence |
|---|---|---|---|
| `regression` (l2) | supported | `SQUARED_ERROR`, Python `objective="regression"` | `src/mojoboost/boosting.mojo`, `tests/test_mojoboost.mojo` |
| `regression_l1` / `mae` | supported | `L1`, with LightGBM's `RenewTreeOutput` leaf-value replacement | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `huber` | supported | `HUBER`, `alpha` is the transition point. No leaf renewal, as in LightGBM | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `quantile` | supported | `QUANTILE`, `alpha` is the level, with weighted-percentile leaf renewal | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `poisson` | supported | `POISSON`, exp link, log-mean base score, `poisson_max_delta_step` in the Hessian. Reachable from Mojo; the Python regressor does not list it yet, which is a v1 gap | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `binary` | supported | `BINARY_LOGISTIC` | `src/mojoboost/boosting.mojo`, `tests/test_mojoboost.mojo` |
| `multiclass` (softmax) | supported | `train_multiclass` / `fit_multiclass`, one tree per class per round | `src/mojoboost/boosting.mojo`, `tests/test_multiclass_model.mojo` |
| `lambdarank` | supported | `train_ranker` / `fit_ranker` / `MojoBoostRanker` | `src/mojoboost/ranking.mojo`, `tests/test_ranking.mojo` |
| custom (callable) | different | Single output only, weights applied by the framework, gradients validated every round, base score explicit. See the README section on custom objectives for each difference and why | `src/mojoboost/objective.mojo`, `tests/test_custom_objective.mojo` |
| `mape` | deferred | v1 work, task 11 |  — |
| `fair` | deferred | v1 work, task 11 | — |
| `gamma` | deferred | v1 work, task 11 | — |
| `tweedie` | deferred | v1 work, task 11, with `tweedie_variance_power` | — |
| `cross_entropy` | deferred | v1 work, task 11 | — |
| `cross_entropy_lambda` | deferred | v1 work, task 11 | — |
| `multiclassova` | deferred | One-vs-rest multiclass. v1 work, task 11 | — |
| `rank_xendcg` | deferred | Out of v1 with the rest of the unbiased/alternative ranking objectives | — |

GPU coverage: `train_gpu` covers every single-output objective above that
shares the per-row gradient/Hessian interface, `train_custom_gpu` covers
custom objectives, and `train_multiclass_gpu` covers softmax. LambdaRank is
CPU only.

## 9. Metrics

Every metric name accepted by LightGBM 4.7.0, verified the same way.

| LightGBM metric | Status | Notes |
|---|---|---|
| `l2` | supported | Mojo `rmse` reports the root; the squared value is the training loss used for early stopping. Not selectable by name from Python |
| `rmse` | supported | `src/mojoboost/metrics.mojo`, `tests/test_metrics.mojo`. Not selectable by name from Python |
| `binary_logloss` | supported | Same file. Not selectable by name from Python |
| `multi_logloss` | supported | Same file. Not selectable by name from Python |
| `auc` | supported | Rank-based, with scikit-learn's tie handling. Not selectable by name from Python |
| `ndcg` | supported | Any cutoff, per query, averaged; `ndcg_score` from Python. An all-zero-label query counts as 1.0, as in LightGBM |
| `binary_error` / `multi_error` | partial | `binary_accuracy` and `multiclass_accuracy` are the complements. The LightGBM spellings are not provided |
| `l1` | deferred | Task 11 |
| `quantile` / `mape` / `huber` / `fair` / `poisson` / `gamma` / `gamma_deviance` / `tweedie` | deferred | Task 11, with the matching objectives |
| `map` | deferred | Task 11 |
| `average_precision` | deferred | Task 11 |
| `auc_mu` | deferred | Task 11 |
| `cross_entropy` / `cross_entropy_lambda` / `kullback_leibler` | deferred | Task 11 |
| custom metrics (`feval`) | supported | Mojo: `MetricSuite`, several metrics with a declared direction and early-stopping flag. Python: `fit(eval_metric=...)` with callables. Differences from `feval` are listed at the top of `src/mojoboost/custom_metric.mojo` |

**Selecting a built-in metric by name is the gap.** The metrics marked
`supported` above are implemented and tested in Mojo, but from Python a user
passes a callable. Accepting `eval_metric="auc"` is v1 work.

## 10. Callbacks

| LightGBM callback | Status | Notes |
|---|---|---|
| callback protocol (`CallbackEnv`, ordering, `before_iteration`) | deferred | Task 3. Nothing accepts a callback today |
| `early_stopping` | partial | As `fit` arguments, section 1 |
| `log_evaluation` | deferred | Task 3 |
| `record_evaluation` | partial | `evals_result_` without a callback |
| `reset_parameter` | deferred | Task 3 |
| `EarlyStopException` | deferred | Task 3 |

## 11. Distributed modes

| LightGBM mode | Status | Notes |
|---|---|---|
| data parallel | partial | Designed and prototyped: row partitioning, local histograms, all-reduce, globally consistent splits, identical trees on every rank, deterministic failure agreement. Every rank runs in one process (`LocalCollective`), so nothing has crossed a network. `src/mojoboost/distributed.mojo`, `src/mojoboost/collective.mojo`, `tests/test_distributed.mojo`, `docs/distributed.md` |
| feature parallel | deferred | Section 2 of `docs/distributed.md` explains why data parallel comes first |
| voting parallel | deferred | Same |
| a real transport (MPI, sockets, gRPC) | deferred | Task 16. The collective contract exists so a transport can be added without touching tree logic. **No distributed performance or scaling claim is made anywhere** |
| Dask integration | deferred | Task 17, after a transport exists |
| distributed GPU | deferred | Out of v1 |

## 12. GPU and accelerators

| Capability | Status | Notes |
|---|---|---|
| GPU histogram construction | supported | Portable kernel, 2D grid, fixed-point Int32 reduction, two combine strategies that are bit-identical to each other. `src/mojoboost/histogram_gpu.mojo`, `tests/test_gpu_strategies.mojo` |
| End-to-end GPU training | supported | `train_gpu`, device-resident binned matrix and leaf assignments. `src/mojoboost/train_gpu.mojo`, `tests/test_gpu_training.mojo` |
| GPU multiclass | partial | `train_multiclass_gpu` exists, but `fit_multiclass` and the Python classifier still resolve multiclass to the CPU |
| CUDA (NVIDIA) validation | deferred | The source targets it and `tests/test_gpu_portability.mojo` pins the launch limits CUDA imposes, but **no NVIDIA device has run this code**. `docs/GPU_VALIDATION.md` holds the procedure and the status table |
| HIP (AMD) validation | deferred | Same |
| GPU speed vs CPU | different | `auto` ships with its size heuristic disabled and always chooses the CPU, because no benchmark on any device has found a crossover. Shipping a threshold first would be a performance claim with nothing behind it |

## 13. Packaging and distribution

| LightGBM property | Status | Notes |
|---|---|---|
| PyPI wheels | deferred | The wheel builds and validates locally (`pixi run test-wheel`); nothing has been uploaded |
| macOS arm64 wheel | supported | Self-contained: the four MAX runtime dylibs are bundled with an `@loader_path` rpath and re-signed. `packaging/build_wheel.sh`, `packaging/test_wheel.sh` |
| macOS x86-64 wheel | deferred | No such machine here |
| manylinux wheel | deferred | Task 18. The Mojo toolchain runs on Linux in CI, so this is packaging work rather than a port |
| Windows wheel | deferred | Task 18 |
| Python version range | different | `requires-python = ">=3.14"`, one interpreter. LightGBM ships 3.9 through 3.13 |
| conda package | deferred | Task 18 |
| R package | deferred | Not started in this repository |
| C API | deferred | LightGBM's `c_api.h` is a public interface that other language bindings use. Nothing stable is committed here yet |
| source build from a clean checkout | supported | `pixi install && pixi run test` |

---

## Known gaps in this contract

Findings from the audit that are not LightGBM parity items but that make the
matrix less trustworthy than it looks. They are recorded here rather than
quietly fixed, because each is somebody's in-flight work:

1. **The sparse modules are unreachable and untested.**
   `src/mojoboost/sparse.mojo`, `src/mojoboost/histogram_sparse.mojo`, and
   `src/mojoboost/tree_sparse.mojo` are not exported from
   `src/mojoboost/__init__.mojo` and no test file imports them. The sparse
   rows above are `deferred` for that reason, not `partial`.
2. **Built-in metric names are not selectable from Python**, though the
   metrics exist in Mojo. This is the largest single mismatch between what
   is implemented and what a LightGBM user can reach.
3. **`colsample_bytree` and `colsample_bynode` are not accepted** as aliases
   even though the rest of the scikit-learn spellings are.
4. **`poisson` is missing from the Python regressor's objective table** even
   though the objective is implemented and tested.
5. **TreeSHAP and the Mojo `Dataset` are one export away from being
   reachable.** `src/mojoboost/contrib.mojo` is exported but has no Python
   argument; `src/mojoboost/dataset.mojo` is not exported at all. Both rows
   above say `partial` for that reason: the algorithm existing is not the
   same as a user being able to call it.

## Enforcement

`tools/check_parity.py` (stdlib only, no build required) checks:

- every status cell in this file uses the vocabulary above
- every repository path cited in this file exists
- a fixed inventory of rows still says `supported`, so a supported row
  cannot be deleted or downgraded without the check failing
- the public Python symbols those rows depend on still exist, by importing
  `mojoboost` when the extension is built and by parsing
  `python/mojoboost/__init__.py` otherwise
- the public Mojo symbols those rows depend on are still exported from
  `src/mojoboost/__init__.mojo`
- every Mojo test suite cited here is run by a pixi task. A suite that is
  cited as evidence but never executed is not evidence; the exception list
  in the script must match the "Known gaps" section exactly, and it is
  currently empty

Run it with `python3 tools/check_parity.py` or `pixi run check-parity`. It
runs in CI on every push and pull request.

## Changelog

- **v1 (2026-08-14)**: first audit, against LightGBM 4.7.0 and mojoboost at
  `6190f88`.
