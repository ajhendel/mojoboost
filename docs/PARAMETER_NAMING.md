# mojotrees canonical parameter names (user-facing layer)

Rule: one canonical name per parameter, always an EXISTING name from LightGBM,
XGBoost, CatBoost, or sklearn; every other vendor's name is an accepted alias.
Canonical is the scikit-learn spelling wherever the three vendors' own sklearn
wrappers share one, because that is the vocabulary an estimator user expects,
and the clearest existing vendor name otherwise (max_leaves and device are the
latter kind). Values still match LightGBM stock defaults.
Internal struct fields, LightGBM model I/O, and check_parity keep LightGBM's
names on the wire. Docs and error messages name the canonical spelling FIRST
and the vendor spelling the user typed in parentheses after it, for example
`max_leaves (num_leaves) must be at least 2`; where the typed spelling is
unknowable, they name the canonical alone and list the vendor spellings that
reach it. Value strings are case-insensitive.

This table is the determination. `tools/api_snapshot.py` reads the OURS column
out of it to fill `python.parameter_aliases.<alias>.canonical`, so a parameter
missing a row here is recorded as `null` and reported to users by its wire name
instead. `docs/COMPATIBILITY_POLICY.md` section 4.1 states the same rule as a
release contract and records why LightGBM's spellings are deliberately not the
canonical ones.

Behavior fix: subsample < 1 implies subsample_freq = 1 unless subsample_freq is
set explicitly (no LightGBM silent no-op).

Legend: OURS = canonical; other columns = the vendor's name (accepted as alias);
"-" = the vendor has no such parameter.

| OURS                       | LightGBM                  | XGBoost                    | CatBoost                       | sklearn HGB            | why                                   |
|----------------------------|---------------------------|----------------------------|--------------------------------|------------------------|---------------------------------------|
| n_estimators               | num_iterations            | n_estimators               | iterations                     | max_iter               | sklearn/XGBoost word                  |
| learning_rate              | learning_rate             | eta / learning_rate        | learning_rate                  | learning_rate          | unanimous                             |
| max_leaves                 | num_leaves                | max_leaves                 | max_leaves                     | max_leaf_nodes         | "max" says what it is                 |
| max_depth                  | max_depth                 | max_depth                  | depth                          | max_depth              | 3 of 4                                |
| min_child_samples          | min_data_in_leaf          | -                          | min_data_in_leaf               | min_samples_leaf       | LightGBM sklearn name                 |
| min_child_weight           | min_sum_hessian_in_leaf   | min_child_weight           | -                              | -                      | XGBoost short form                    |
| subsample                  | bagging_fraction          | subsample                  | subsample                      | -                      | 3 of 4                                |
| subsample_freq             | bagging_freq              | (implicit 1)               | (implicit 1)                   | -                      | kept for LightGBM users; default 1    |
| colsample_bytree           | feature_fraction          | colsample_bytree           | rsm                            | -                      | rsm is opaque                         |
| colsample_bynode           | feature_fraction_bynode   | colsample_bynode           | -                              | max_features           | XGBoost                               |
| reg_lambda                 | lambda_l2                 | reg_lambda                 | l2_leaf_reg                    | l2_regularization      | XGBoost / LightGBM-sklearn            |
| reg_alpha                  | lambda_l1                 | reg_alpha                  | -                              | -                      | same                                  |
| min_split_gain             | min_gain_to_split         | gamma                      | -                              | -                      | gamma is opaque                       |
| max_bin                    | max_bin                   | max_bin                    | border_count                   | max_bins               | 3 of 4                                |
| min_data_in_bin            | min_data_in_bin           | -                          | -                              | -                      | only one exists                       |
| boosting_type              | boosting                  | booster                    | boosting_type                  | -                      | values gbdt/dart/goss/rf + plain (=gbdt) + ordered (not implemented yet) |
| grow_policy                | (implicit leaf-wise)      | grow_policy                | grow_policy                    | (implicit depth-wise)  | values below                          |
|   = lossguide              | (leaf-wise, default)      | lossguide (max_leaves)     | Lossguide                      | -                      | alias: leafwise                       |
|   = depthwise              | -                         | depthwise (default)        | Depthwise                      | (default)              |                                       |
|   = symmetrictree          | -                         | -                          | SymmetricTree (default)        | -                      | alias: oblivious, symmetric           |
| objective                  | objective                 | objective                  | loss_function                  | loss                   | all three vendors' loss names accepted|
| metric                     | metric                    | eval_metric                | eval_metric / custom_metric    | scoring                | LightGBM                              |
| device (cpu \| gpu)        | device_type               | device / tree_method         | task_type (CPU/GPU)            | -                      | XGBoost 2.x                           |
| random_state               | seed                      | random_state / seed        | random_seed                    | random_state           | sklearn convention                    |
| n_jobs                     | num_threads               | n_jobs / nthread           | thread_count                   | -                      | sklearn convention                    |
| early_stopping_rounds      | early_stopping_round      | early_stopping_rounds      | early_stopping_rounds / od_wait| n_iter_no_change       | 3 of 4                                |
| verbose                    | verbosity                 | verbosity                  | verbose / logging_level        | verbose                | 3 of 4                                |
| categorical_features       | categorical_feature       | (enable_categorical)       | cat_features                   | categorical_features   | sklearn                               |
| max_cat_to_onehot          | max_cat_to_onehot         | max_cat_to_onehot          | one_hot_max_size               | -                      | 2 of 4                                |
| monotone_constraints       | monotone_constraints      | monotone_constraints       | monotone_constraints           | monotonic_cst          | unanimous                             |
| monotone_penalty           | monotone_penalty          | -                          | -                              | -                      | LightGBM only, so its name stands; long form monotone_constraints_penalty is ours and is accepted as an alias |
| interaction_constraints    | interaction_constraints   | interaction_constraints    | -                              | interaction_cst        | unanimous                             |
| random_strength            | -                         | -                          | random_strength                | -                      | CatBoost only                         |
| bootstrap_type             | -                         | -                          | bootstrap_type                 | -                      | CatBoost only (Bayesian/Bernoulli/MVS/Poisson) |
| bagging_temperature        | -                         | -                          | bagging_temperature            | -                      | CatBoost only                         |
| leaf_estimation_iterations | -                         | -                          | leaf_estimation_iterations     | -                      | CatBoost only                         |
| score_function             | -                         | -                          | score_function                 | -                      | CatBoost only                         |
| max_ctr_complexity         | -                         | -                          | max_ctr_complexity             | -                      | CatBoost only                         |
| linear_tree                | linear_tree               | -                          | -                              | -                      | LightGBM only                         |
| extra_trees                | extra_trees               | -                          | -                              | -                      | LightGBM only                         |
| path_smooth                | path_smooth               | -                          | -                              | -                      | LightGBM only                         |
| top_rate / other_rate      | top_rate / other_rate     | -                          | -                              | -                      | GOSS, LightGBM only                   |
| drop_rate / skip_drop ...  | same                      | rate_drop / skip_drop      | -                              | -                      | DART; XGBoost names aliased           |
| cegb_*                     | cegb_*                    | -                          | -                              | -                      | LightGBM only                         |

Not covered here (kept as LightGBM spells them, aliases added when a vendor has
one): min_data_per_group, cat_smooth, cat_l2, max_cat_threshold,
feature_pre_filter, bin_construct_sample_cnt, forcedsplits, refit_decay_rate,
num_class (XGBoost num_class, CatBoost classes_count).
