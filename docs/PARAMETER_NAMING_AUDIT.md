# Parameter naming audit

Read-only audit of every user-facing string that names a parameter. Nothing
in this document was applied. It is a list to apply, ordered so the top of
the table is worth more than the bottom.

Scope of the sweep. Every `raise` in `python/mojotrees/*.py` (28 modules,
199 raises whose message text mentions a parameter name), the estimator
class docstring and per-parameter documentation in
`python/mojotrees/sklearn.py`, `python/README.md`, and the parameter-facing
raises in `src/mojotrees/params.mojo`.

---

## 1. The rule

mojotrees accepts parameter spellings from LightGBM, XGBoost, CatBoost and
scikit-learn. The naming rule has always been coherent. It has been
understated, which is what makes the surface read as a hodgepodge when it is
not. Stated properly, there are four layers and each one has exactly one
right spelling.

**Canonical is the scikit-learn spelling.** It is the vocabulary all three
vendors' own sklearn wrappers share, and it is what an estimator user
expects. The table in `docs/PARAMETER_NAMING.md` is the list, one canonical
name per parameter, always a name some vendor already uses.

**Aliases are every vendor spelling, and they must work either way.** A
LightGBM, XGBoost or CatBoost script runs through this estimator unchanged.
Setting two spellings of one parameter to different non-default values
raises, where LightGBM warns and keeps one.

**Wire, model files and parity checks use LightGBM keys**, because that is
the wire format. This covers the native layer, the parameter string in
`src/mojotrees/params.mojo`, `params=` dicts, `RESETTABLE` slot names, the
`Dataset` and `Booster` and `train()` argument names in
`python/mojotrees/basic.py`, saved models, and `tools/check_parity.py`.

**Growth-mode defaults mirror the vendor whose tree they are**, which is
LANE_RULES rule 9. `grow_policy='symmetrictree'` mirrors CatBoost,
`lossguide` mirrors LightGBM.

### Why LightGBM is not canonical

The decision was taken explicitly and is recorded here so it is not
relitigated. Making LightGBM's spellings canonical would make the
CatBoost-mode surface read wrong, because `depth`, `SymmetricTree` and
`bootstrap_type` are CatBoost's words and there is no LightGBM word to
replace them with. It would also gain nothing, because LightGBM users are
already served by aliases that work either way. The wire keeps LightGBM's
spellings for the reason it always has, which is that the model file and the
parity harness are LightGBM's format.

### The rule a contributor follows

> Every error message and every docstring names the **canonical** spelling
> first, and the **vendor spelling the user typed** in parentheses.

Examples of the form.

- `max_leaves (num_leaves) must be at least 2`
- `subsample (bagging_fraction)=0.8 cannot be set beside bootstrap_type='MVS'`
- `categorical_features (categorical_feature) index 7 is out of range`

Where the typed spelling is unknowable, name the canonical alone and list
the vendor spellings that reach it, so the user can find their own word.
Section 4 says where that is and what it would take to fix.

Three surfaces are exempt because their argument names **are** the wire.
`src/mojotrees/params.mojo`, the `params=` dict keys, and the
`Dataset`/`Booster`/`train()` argument names in `basic.py`. A message on
those surfaces naming a LightGBM key is correct, not a violation. The
exemption stops at CatBoost and XGBoost keys, which are neither canonical
nor wire, and which those surfaces do accept.

---

## 2. Canonical determination, and the snapshot field finding

### 2.1 The finding

**`compatibility/api_snapshot.json` records the field `python.parameter_aliases.<alias>.canonical`, and its value is not the canonical name. It is the wire name.** Every consumer of that table inherits the confusion.

The evidence is three-part and unambiguous.

The field is derived by `tools/api_snapshot.py:alias_pairs` (line 495) from
the **first** argument of each `self._resolve_alias(...)` call site, and the
local variable that holds it is named `canonical`.

`_resolve_alias` names that same argument `primary`, and its own docstring
at `python/mojotrees/sklearn.py:1859` says what it is.

> `primary` is the name that holds the stock default and the value the
> native layer is sent, LightGBM's spelling, because that is the wire (see
> the class docstring). It is not necessarily the canonical user-facing
> name. `num_leaves` is the primary and `max_leaves` is the canonical name
> resolved onto it.

The snapshot then disagrees with `docs/PARAMETER_NAMING.md` on exactly the
rows where wire and canonical differ.

| alias | snapshot `canonical` | PARAMETER_NAMING.md canonical | agree |
|---|---|---|---|
| `max_leaves`, `max_leaf_nodes` | `num_leaves` | `max_leaves` | no |
| `min_child_samples`, `min_samples_leaf` | `min_data_in_leaf` | `min_child_samples` | no |
| `min_child_weight`, `min_sum_hessian_in_leaf` | `min_child_hess` | `min_child_weight` | no |
| `subsample` | `bagging_fraction` | `subsample` | no |
| `subsample_freq` | `bagging_freq` | `subsample_freq` | no |
| `colsample_bytree`, `rsm` | `feature_fraction` | `colsample_bytree` | no |
| `colsample_bynode`, `max_features` | `feature_fraction_bynode` | `colsample_bynode` | no |
| `reg_lambda`, `l2_leaf_reg`, `l2_regularization` | `lambda_l2` | `reg_lambda` | no |
| `reg_alpha` | `lambda_l1` | `reg_alpha` | no |
| `min_split_gain`, `gamma` | `min_gain_to_split` | `min_split_gain` | no |
| `categorical_features`, `cat_features` | `categorical_feature` | `categorical_features` | no |
| `monotone_constraints_penalty` | `monotone_penalty` | `monotone_penalty` (LightGBM only) | yes |
| `iterations`, `num_iterations`, `max_iter`, `num_boost_round` | `n_estimators` | `n_estimators` | yes |
| `depth` | `max_depth` | `max_depth` | yes |
| `border_count`, `max_bins` | `max_bin` | `max_bin` | yes |
| `device_type`, `task_type` | `device` | `device` | yes |
| `seed`, `random_seed` | `random_state` | `random_state` | yes |
| `num_threads`, `nthread`, `thread_count` | `n_jobs` | `n_jobs` | yes |
| `early_stopping_round`, `od_wait`, `n_iter_no_change` | `early_stopping_rounds` | `early_stopping_rounds` | yes |
| `verbosity` | `verbose` | `verbose` | yes |
| `eta`, `shrinkage_rate` | `learning_rate` | `learning_rate` | yes |
| `one_hot_max_size` | `max_cat_to_onehot` | `max_cat_to_onehot` | yes |
| `monotonic_cst` | `monotone_constraints` | `monotone_constraints` | yes |
| `interaction_cst` | `interaction_constraints` | `interaction_constraints` | yes |
| `rate_drop` | `drop_rate` | `drop_rate` | yes |

Eleven of the forty five aliases resolve to a snapshot `canonical` that is
not the canonical name. The other thirty four coincide, and **that is what
makes the field dangerous rather than merely wrong**. A reader who spot
checks `depth`, `seed` or `verbosity` concludes the field means what it
says, then trusts it on `subsample` and gets the wire.

### 2.2 What to do about it

The field is load-bearing and renaming it is a snapshot schema change, so
this is a recommendation and not a correction to apply blind. Two options.

1. Rename the field to `primary` in `tools/api_snapshot.py:alias_pairs`,
   in `compatibility/SNAPSHOT_SCHEMA.md` section 3.2, and in the drift
   classifier patterns at `tools/api_snapshot.py:1787` and `:1791`. Add a
   second derived field `canonical` read from `docs/PARAMETER_NAMING.md`,
   so the snapshot records both and a divergence between them is a
   detectable defect rather than an invisible one.
2. If option 1 is too large, at minimum correct
   `compatibility/SNAPSHOT_SCHEMA.md` section 3.2, which currently
   describes the field as "First argument. The guaranteed spelling". It is
   not the guaranteed user-facing spelling. Both spellings are guaranteed;
   this one is the one sent to the native layer.

Option 1 is preferred because the snapshot is what an external consumer
reads, and a table labeled `canonical` is the natural place to look up
"what should my error message say".

### 2.3 A second instance of the same misnaming

`python/mojotrees/callback.py:165` defines `canonical_reset_key(key)`. It
returns `_RESET_ALIASES.get(key, key)`, and `_RESET_ALIASES` maps the
scikit-learn spellings **to** the LightGBM ones (`reg_lambda` to
`lambda_l2`, `colsample_bytree` to `feature_fraction`). The function named
"canonical" returns the wire name, for the same reason and with the same
consequence. Its own docstring is accurate ("The primary name of a
resettable parameter"), so only the function name is wrong. This one is
public API surface, so renaming it is a deprecation, not an edit.

### 2.4 Determination

For the purposes of this audit, **canonical is the OURS column of
`docs/PARAMETER_NAMING.md`**, which is the scikit-learn spelling wherever
scikit-learn has one and the clearest vendor name otherwise. The snapshot's
`canonical` field is not evidence of canonicity and was not used as such.

Two scoping calls made from the code rather than from taste.

**`eval_metric` is canonical, not a violation.** It is a `fit()` argument
(`api_snapshot.json` records it in `MojoTreesRegressor.methods.fit`), not
the `metric` params key, and `eval_metric` is what LightGBM's, XGBoost's and
CatBoost's own sklearn wrappers call the fit argument. The `metric` row of
`PARAMETER_NAMING.md` governs the params dict, which is the wire. All twenty
`eval_metric` messages in `_fit_args.py`, `cv.py`, `dask.py` and `_eval.py`
are therefore correct as written.

**`basic.py`'s `categorical_feature` argument is correct as written.**
`Dataset(categorical_feature=...)` is LightGBM's own API and takes no alias,
so its six messages (lines 592, 600, 605, 613, 618, 623) name the only
spelling that surface has. They are not in the violation table.

---

## 3. Violating sites

Ordered by how likely a user is to hit the message, most likely first. Apply
from the top.

### 3.1 Error messages

| # | File | Line | Current | Corrected |
|---|---|---|---|---|
| 1 | `python/mojotrees/sklearn.py` | 1556 | `f"unknown categorical_feature {spec!r}; expected 'auto', None, or a sequence of feature names or indices"` | `f"unknown categorical_features {spec!r}; expected 'auto', None, or a sequence of feature names or indices"` (add the typed spelling per section 4) |
| 2 | `python/mojotrees/sklearn.py` | 1488 | `"categorical_feature must be 'auto', None, or a sequence of feature names or indices, got {spec!r}"` | `"categorical_features must be 'auto', None, or a sequence of feature names or indices, got {spec!r}"` |
| 3 | `python/mojotrees/sklearn.py` | 1570 | `f"columns {labels} have pandas categorical dtype but are not in categorical_feature; list them, cast them to a numeric dtype, or leave categorical_feature at 'auto'"` | `f"columns {labels} have pandas categorical dtype but are not in categorical_features; list them, cast them to a numeric dtype, or leave categorical_features at 'auto'"` |
| 4 | `python/mojotrees/sklearn.py` | 1588 | `f"categorical_feature index {index} is out of range for {n_features} features"` | `f"categorical_features index {index} is out of range for {n_features} features"` |
| 5 | `python/mojotrees/sklearn.py` | 1508 | `f"categorical_feature name {entry!r} is not a column of X; X has {known}"` | `f"categorical_features name {entry!r} is not a column of X; X has {known}"` |
| 6 | `python/mojotrees/sklearn.py` | 1502 | `f"categorical_feature names {entry!r}, but X carries no feature names; pass column indices, or fit on a pandas DataFrame"` | `f"categorical_features names {entry!r}, but X carries no feature names; pass column indices, or fit on a pandas DataFrame"` |
| 7 | `python/mojotrees/sklearn.py` | 1517 | `f"categorical_feature entry {entry!r} is neither a feature name nor an index"` | `f"categorical_features entry {entry!r} is neither a feature name nor an index"` |
| 8 | `python/mojotrees/sklearn.py` | 1522 | `"categorical_feature entries must be whole feature indices, got {entry!r}"` | `"categorical_features entries must be whole feature indices, got {entry!r}"` |
| 9 | `python/mojotrees/sklearn.py` | 1496 | `f"categorical_feature entry {entry!r} is a bool, not a feature name or an index"` | `f"categorical_features entry {entry!r} is a bool, not a feature name or an index"` |
| 10 | `python/mojotrees/sklearn.py` | 1528 | `f"categorical_feature lists feature {index} twice"` | `f"categorical_features lists feature {index} twice"` |
| 11 | `python/mojotrees/sklearn.py` | 3508 | `f"tree_method={self.tree_method!r} is a different split search, not a spelling of device: mojotrees searches a histogram, which is XGBoost's 'hist'. Use device='cpu' or device='gpu'."` | `f"device (tree_method={self.tree_method!r}) names a different split search, not a spelling of device: mojotrees searches a histogram, which is XGBoost's 'hist'. Use device='cpu' or device='gpu'."` |
| 12 | `python/mojotrees/sklearn.py` | 3518 | `f"unknown tree_method {self.tree_method!r}; expected 'hist', 'gpu_hist', or 'auto'. device='cpu' and device='gpu' are the canonical spellings."` | `f"unknown device (tree_method={self.tree_method!r}); expected 'hist', 'gpu_hist', or 'auto'. device='cpu' and device='gpu' are the canonical spellings."` |
| 13 | `python/mojotrees/sklearn.py` | 1735 | `f"boosting={boosting!r} is not available {where}; it trains dense, single-output models on the CPU without eval_set or a callable objective"` | `f"boosting_type={boosting!r} is not available {where}; it trains dense, single-output models on the CPU without eval_set or a callable objective"` (with the typed spelling per section 4) |
| 14 | `python/mojotrees/sklearn.py` | 2229 | `f"bagging_fraction={self.bagging_fraction} cannot be set beside bootstrap_type={self.bootstrap_type!r}: bagging_fraction IS CatBoost's Bernoulli bootstrap under mojotrees's name, so this asks for two bootstrap types at once. Use subsample for the MVS rate, or drop bootstrap_type."` | `f"subsample (bagging_fraction)={self.bagging_fraction} cannot be set beside bootstrap_type={self.bootstrap_type!r}: subsample IS CatBoost's Bernoulli bootstrap under mojotrees's name, so this asks for two bootstrap types at once. Use subsample for the MVS rate, or drop bootstrap_type."` |
| 15 | `python/mojotrees/callback.py` | 170 | `f"{key!r} cannot be reset during training; the trainer re-reads only " + ", ".join(RESETTABLE)` | `f"{key!r} cannot be reset during training; the trainer re-reads only " + ", ".join(_canonical_resettable())` where the helper maps each `RESETTABLE` wire name to its canonical spelling and appends the wire name in parentheses. A user who typed `reg_lambda` currently gets a list containing `lambda_l2` and not their own word |
| 16 | `python/mojotrees/_multi_target.py` | 163 | `f"{name} is not honored by a multi-target fit: {mechanism} reaches boosting.train or alternate_boosting, and multi_target.train_multi_rmse is neither. Refused rather than accepted and dropped."` | `f"{canonical_of(name)} ({name}) is not honored by a multi-target fit: ..."`, unchanged otherwise. `name` is already the exact spelling the user typed, so this site needs only the canonical added in front. It is the model site for the rule, see section 4 |
| 17 | `python/mojotrees/sklearn.py` | 2671 | `"auto_learning_rate=True with an explicit l2_leaf_reg (lambda_l2, reg_lambda, l2_regularization) would do nothing: CatBoost's derivation is gated on l2_leaf_reg being unset (options_helper.cpp:280), so naming it pins the rate back to the constant. Drop one of the two"` | `"auto_learning_rate=True with an explicit reg_lambda (lambda_l2, l2_leaf_reg, l2_regularization) would do nothing: CatBoost's derivation is gated on l2_leaf_reg being unset (options_helper.cpp:280), so naming it pins the rate back to the constant. Drop one of the two"`. The second `l2_leaf_reg` is a citation of CatBoost's own source and stays |
| 18 | `src/mojotrees/params.mojo` | 1284 | `"one_hot_max_size is supported by the Mojo API and by the scikit-learn estimator only, not by parameter strings: ..."` | `"max_cat_to_onehot (one_hot_max_size) is supported by the Mojo API and by the scikit-learn estimator only, not by parameter strings: ..."`. Rest unchanged. CatBoost's spelling is neither canonical nor wire, so the wire exemption does not cover it |
| 19 | `src/mojotrees/params.mojo` | 1149 | `"logging_level '", value, "' cannot be honored here: ..."` | `"verbose (logging_level) '", value, "' cannot be honored here: ..."`. Rest unchanged. Same reason as row 18 |

Correct as written, listed so a future sweep does not reopen them.
`sklearn.py:1797` (`unknown boosting_type`), `sklearn.py:3524` (names
`device` and `tree_method` both), `sklearn.py:2385` (`boosting_type='ordered'`),
`sklearn.py:2173` (`random_state`), `params.mojo:1117` (`n_jobs`),
`params.mojo:1536` and `:1544` (both name `tree_method` and then `device`,
and `:1544` even uses the word "canonical"), and every `eval_metric` and
`basic.py` categorical message for the reasons in section 2.4.

### 3.2 Docstrings

| # | File | Line | Current | Corrected |
|---|---|---|---|---|
| D1 | `python/README.md` | 109 to 115 | "Native LightGBM names are canonical (`min_data_in_leaf`, `min_child_hess`, `lambda_l1`, `lambda_l2`, `bagging_fraction`, `bagging_freq`, `boosting`, `device`). For easy migration from `LGBMRegressor` and `LGBMClassifier`, their scikit-learn spellings are accepted too: ..." | "The scikit-learn spellings are canonical (`min_child_samples`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `subsample`, `subsample_freq`, `boosting_type`, `device`), and they are what every error message and docstring names. Every vendor spelling is accepted as an alias and works either way, so a LightGBM, XGBoost or CatBoost script runs unchanged (`min_data_in_leaf`, `min_child_hess`, `lambda_l1`, `lambda_l2`, `bagging_fraction`, `bagging_freq`, `boosting`, `device_type`). LightGBM's spellings are what the native layer, the model files and `tools/check_parity.py` use on the wire. Conflicting values raise instead of silently choosing one." **This paragraph currently states the opposite of the policy and is the single highest-value correction in the document** |
| D2 | `python/mojotrees/sklearn.py` | 823 | "`categorical_feature` is LightGBM's parameter of the same name (the plural `categorical_features` is accepted as an alias)." | "`categorical_features` is scikit-learn's spelling (LightGBM's singular `categorical_feature` and CatBoost's `cat_features` are accepted as aliases)." The paragraph presents the vendor spelling as primary and the canonical one as the alias, which is the rule inverted |
| D3 | `python/README.md` | 182, 188, 196 | "`categorical_feature` names the columns ...", `MojoTreesRegressor(categorical_feature=["city"])`, "leaving such a column out of an explicit `categorical_feature` raises" | Same three with `categorical_features`, and a first-mention parenthetical "(LightGBM's `categorical_feature`)". Leave `model.categorical_feature_` at line 189 alone, see section 5 |
| D4 | `python/README.md` | 97 | `model = MojoTreesRegressor(num_leaves=31, n_estimators=100).fit(X, y)` | `model = MojoTreesRegressor(max_leaves=31, n_estimators=100).fit(X, y)`. The headline example is the first line of mojotrees a user ever runs and it currently teaches the alias |
| D5 | `python/mojotrees/sklearn.py` | 449, 451, 459 | "`num_leaves` stays a hard bound", "at the default `num_leaves=31`", "`num_leaves` does not bind" | "`max_leaves` stays a hard bound", "at the default `max_leaves=31`", "`max_leaves` does not bind". Line 435 already introduces the pair correctly, so the bullets need no parenthetical |
| D6 | `python/mojotrees/sklearn.py` | 497 | "`bootstrap_type` and `bagging_fraction` are two values of one CatBoost enum and cannot both be set" | "`bootstrap_type` and `subsample` are two values of one CatBoost enum and cannot both be set". Line 468 already introduces the pair |
| D7 | `python/mojotrees/sklearn.py` | 550 to 551 | "it needs a source of per-tree randomness, `bagging_fraction < 1` with `bagging_freq > 0` or `feature_fraction < 1`" | "it needs a source of per-tree randomness, `subsample < 1` with `subsample_freq > 0` or `colsample_bytree < 1`" |
| D8 | `python/mojotrees/sklearn.py` | 854 | "- `min_gain_to_split` (alias `min_split_gain`) is the gain a split must clear to be taken at all." | "- `min_split_gain` (LightGBM's `min_gain_to_split`) is the gain a split must clear to be taken at all." Inverted, same defect as D2 |
| D9 | `python/mojotrees/sklearn.py` | 859 | "LightGBM needs `min_data_in_leaf` of at least 2 for it, and so does mojotrees" | "LightGBM needs `min_child_samples` (`min_data_in_leaf`) of at least 2 for it, and so does mojotrees" |
| D10 | `python/mojotrees/sklearn.py` | 839 | "leaving such a column out of an explicit `categorical_feature` raises" | "leaving such a column out of an explicit `categorical_features` raises" |
| D11 | `python/mojotrees/sklearn.py` | 6291 | "`bagging_fraction` samples whole queries rather than rows, LightGBM's `bagging_by_query=true` behavior" | "`subsample` (LightGBM's `bagging_fraction`) samples whole queries rather than rows, LightGBM's `bagging_by_query=true` behavior". `MojoTreesRanker`'s docstring is a separate class docstring and does not inherit `_Base`'s introduction |
| D12 | `python/mojotrees/sklearn.py` | 811, 813 | "`learning_rate` and `lambda_l2` default to `None` for this", "an unset `l2_leaf_reg`, and 'unset' there is provenance" | "`learning_rate` and `reg_lambda` (`lambda_l2`) default to `None` for this". Line 813's `l2_leaf_reg` is a citation of CatBoost's gate and stays |
| D13 | `python/mojotrees/sklearn.py` | 863 | "- `monotone_penalty` (alias `monotone_constraints_penalty`) discounts the gain" | "- `monotone_penalty` (CatBoost's `monotone_constraints_penalty`) discounts the gain". `monotone_penalty` is correct as canonical, only the word "alias" is doing the D8 inversion |
| D14 | `python/mojotrees/sklearn.py` | 2137, 2141, 2143, 2147, 2153, 2154 | `_resolve_bootstrap` docstring uses `bagging_fraction` six times and `bagging_freq` once | Lowest priority of the docstring set because it is a private method and its argument really is the wire value. Add one sentence at the top, "`bagging_fraction` here is the wire member behind the canonical `subsample`", rather than renaming six mentions that are accurately describing the member |
| D15 | `python/mojotrees/sklearn.py` | 423 | "The canonical name is what this docstring and every error message use." | Keep the sentence. It is the statement of the rule and it is the right one. It is currently false, which is what this audit is for. Apply rows 1 to 19 and D1 to D14 and it becomes true |

`sklearn.py` docstring mentions that are correct as written and must not be
"fixed": lines 435, 436, 468, 469, 481, 482, 523, 557, 559, 565, 573, 579,
580, 736, 1764, 1769, 1862, 1949, 1986, 3467, 4988, 5704. Every one of them
already has the form "canonical (vendor's spelling)" or is explaining the
wire on purpose.

---

## 4. Can the code know which spelling the user typed?

The rule as stated assumes it always can. It does not always can, and the
answer differs by surface. This matters because a message that names only
the canonical spelling to a user who typed the vendor one is its own failure
mode, arguably worse than today's inconsistency, since the user cannot find
in their own script the parameter the error is about.

### 4.1 Where it can, today, with no new machinery

**Every alias is a separate constructor keyword stored unmodified on the
estimator, and an unset alias is `None`.** `api_snapshot.json`
`python.shared_estimator_parameters` lists all one hundred and thirty three,
including every alias. So `getattr(self, alias) is not None` is an exact
test of "the user typed this spelling", available at any `_Base` method.

Sites that already exploit it.

- `_multi_target.py:150` iterates `_UNHONORED` by attribute and interpolates
  `{name}`, so its message names the exact spelling typed. This is the model
  site. It needs only the canonical added in front.
- `sklearn.py:2229` fires on `float(self.bagging_fraction) != 1.0`, which
  can only be true if the user typed `bagging_fraction`, so naming
  `bagging_fraction` is correct provenance. It is missing the canonical, not
  the typed name.
- `sklearn.py:1774` loops over `("boosting_type", "booster")` and holds the
  alias name in a local, which is why its conflict message at 1780 can
  interpolate `{alias}`.
- `sklearn.py:2485` `_l2_named()` tests all four spellings, so it knows one
  was typed but discards which. Changing the `or` chain to return the name
  instead of a bool is a three-line change and would let row 17's message
  name the typed spelling.
- `_resolve_alias` itself (`sklearn.py:1849`) has both `primary` and `alias`
  in scope, which is why its own conflict message is already the best in the
  codebase.

### 4.2 Where it cannot, and what would make it able

**The runtime has no alias table.** `compatibility/SNAPSHOT_SCHEMA.md`
section 3.2 says so outright, "There is no alias table in
`python/mojotrees/__init__.py`. The pairs are expressed as calls". The table
exists only as `_resolve_alias` call sites, recovered at build time by an
AST walk in `tools/api_snapshot.py`. So a method that holds a resolved value
cannot ask "which spelling produced this" even though the attributes are all
still on `self`, because it has no map from primary to the list of aliases
to test.

This is what blocks rows 1 to 10 and 13 from naming the typed spelling.
`_resolve_categorical` (`sklearn.py:1548`) resolves three spellings into one
`spec` and then raises on the value, at which point the provenance is gone.
`_resolve_boosting` (`sklearn.py:1773`) collapses three spellings into one
string before `_refuse_alternate_boosting` reads it.

**What would make it able.** One module-level dict in `sklearn.py`, primary
to the tuple of aliases, plus one helper.

```python
def _typed_spelling(self, primary):
    """The spelling the user actually typed, or `primary`."""
    for alias in _ALIASES.get(primary, ()):
        if getattr(self, alias, None) is not None:
            return alias
    return primary
```

The cost is that the dict is a second place the alias pairs live, which is
exactly what `SNAPSHOT_SCHEMA.md` 3.2 says the current design avoids. Two
ways to pay it without reintroducing the drift.

1. Generate the dict from the `_resolve_alias` call sites at import time by
   the same AST walk `tools/api_snapshot.py` already does. Correct, and it
   puts a parse of the module's own source in the import path, which is a
   startup cost `docs/STARTUP_LATENCY.md` would have to price.
2. Write the dict by hand and have `tools/api_snapshot.py` assert it equals
   the derived `parameter_aliases` table. Free at runtime, and the drift is
   caught by the snapshot job that already runs in CI. **This is the
   recommended option.** It is also the natural place to record the canonical
   spelling, which section 2.2 option 1 wants anyway, so one table serves
   both fixes.

### 4.3 Where it cannot even in principle at that site

**Free functions that receive resolved values.** `_fit_args.py`, `_eval.py`,
`cv.py`, `dask.py`, `_dask_runtime.py` and `_validation.py` take values, not
an estimator, so no amount of table would help at the raise itself. None of
their messages is in the violation table, so nothing is blocked today, but a
future message there must either take the typed name as an argument or name
the canonical alone and list the vendor spellings.

**`src/mojotrees/params.mojo` splits in two.** Inside `_parse_params`'s token
loop the typed key is literally in scope as `key`, and several messages
already interpolate it (`:538`, `:547`, `:556`, `:1137`, `:1573`, `:1584`).
Rows 18 and 19 are in that loop and can name the typed spelling for free.
After the loop, the post-parse validators receive a `TrainConfig` plus
`Bool` flags (`saw_lambda_l2` at `:865`, `saw_learning_rate` at `:864`,
`saw_leaf_estimation_iterations` at `:866`, threaded through `:1609`,
`:1649`, `:1846`, `:1909`). A `Bool` records that some spelling was seen and
not which. Making `params.mojo:1940` able to name the typed spelling means
carrying a `String` that is the typed key and empty when unset, in place of
each `Bool`. That is a mechanical change to about ten flags and their four
signatures, and it is not needed for any row in the table above, so it is
recorded as an OPEN item rather than proposed.

### 4.4 The answer

The codebase **can** know the typed spelling at every estimator-level site
in the violation table, because every alias is a live attribute on `self`
and unset means `None`. It **does not** know today at eleven of them,
because the alias table exists only as call sites and not as runtime data.
One hand-written dict guarded by the existing snapshot job closes all
eleven. The parameter-string surface knows inside its parse loop and forgets
afterwards, and the free-function modules never knew. Until the dict lands,
rows 1 to 10 and 13 should name the canonical alone and, where the message
has room, append the vendor spellings that reach it, so a user who typed
`cat_features` still finds their word in the text.

---

## 5. Out of scope, recorded so it is not lost

**`categorical_feature_` is the fitted attribute** (`api_snapshot.json`
`python.fitted_attributes`), spelled the wire way while its parameter is
canonical the sklearn way. Renaming a fitted attribute is a breaking change
under compatibility policy 4.3 and needs a deprecation cycle, so it is not
in the table. It should be decided, because `categorical_features=[...]` in
and `categorical_feature_` out is the kind of asymmetry the naming policy
exists to prevent.

**`compatibility/api_snapshot.json` itself is generated** and must not be
hand-edited. Section 2.2's fix is to `tools/api_snapshot.py` and
`compatibility/SNAPSHOT_SCHEMA.md`, after which the snapshot is regenerated.

**Nothing in this audit changes a fitted model, a wire key, or a default.**
Every row is text. The two `params.mojo` rows change message text only and
neither touches a parsed key, so no parameter string that works today stops
working.
