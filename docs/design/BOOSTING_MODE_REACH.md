# Which boosting mode and training control reaches which surface

Written 2026-08-17 by reading the source at head (`perf-round-2`, tip
`c775959`). Nothing here was measured and nothing here was trained. Every row
names the line that decides it, so any one row can be checked without
re-deriving the table.

**Line numbers drift and the symbol names do not.** This was read in the shared
checkout with nine lanes live in it, and `bindings/_mojotrees.mojo` was edited
by another lane part way through the read: `_parse_boosting` moved nine lines
while the file was open. Every citation below was refreshed against the file as
it stood at 09:41, and every one of them also names the function or the symbol,
which is the durable half. If a number does not land where it says, grep the
name.

This is the boosting-modes half of the sweep `GROWTH_POLICY_REACH.md` did for
GPU switches. It was commissioned on the belief that DART, random forest,
early stopping and EFB were "built but unreachable and never run". **That
belief is stale.** Four of the six families are reachable, honored and
covered by tests today, and the two files that said otherwise
(`boosting_dart.mojo`, `boosting_rf.mojo`) were both carrying a
reachability section that described a state the library left some time ago.
Those have been corrected in place, and the corrections quote what they used
to say.

What the sweep did find is three real defects, all of the accept-then-ignore
shape, and a large set of second-layer entry points that exist, are correct,
and have no binding.

## The vocabulary

- **REACHABLE AND HONORED.** A public surface can select it and the trainer
  behind that surface applies it.
- **REACHABLE AND SILENTLY IGNORED.** A wrong answer. A surface accepts the
  value and the trainer behind it does something else without saying so.
- **UNREACHABLE.** Implemented, correct as far as reading can establish, and
  no public surface can select it. The missing piece is named per row: a
  binding, a parser entry, a dispatch branch, or a refusal that fires early.
- **REFUSED BY NAME.** The surface raises and says what to do instead.
  Acceptable, and the house style.
- **ABSENT.** Not implemented.

There are five surfaces, and they do not agree with each other. That
disagreement is where every defect below lives.

| surface | door | reads the `boosting` key |
| --- | --- | --- |
| estimators | `MojoTreesRegressor.fit` etc. -> `_mojotrees.fit` | **yes**, `bindings/_mojotrees.mojo:2082` |
| `mojotrees.train` / `Dataset` / `Booster.update` / `mojotrees.cv` | `basic.train` -> `train_dataset*`, `booster_update*` | **no** |
| parameter string | `params.parse_params` | refuses dart/goss/rf by name, `params.mojo:706-716` |
| C API | `capi/mojotrees_capi.mojo` -> `parse_params` | as the parameter string |
| CLI | `cli/mojotrees_cli.mojo` -> `parse_params` | as the parameter string |

## The matrix

Growth policy is collapsed to one column wherever the honoring trainer calls
`tree.grow_tree`, because that function dispatches leaf-wise, depth-wise and
oblivious internally (`tree.mojo:3159-3160`, and `_grow_oblivious_levels` at
`tree.mojo:2262` takes the same `bundling: BundledMatrix` argument the other
two do). A mode that reaches `grow_tree` reaches all three policies, and none
of the parameters below is read by a grower at all except `enable_bundle`.

| parameter | estimators, CPU | estimators, GPU | `mojotrees.train` / `cv` | parameter string, CLI, C API | verdict |
| --- | --- | --- | --- | --- | --- |
| `boosting='dart'` | **HONORED**, all 3 policies | REFUSED BY NAME | **SILENTLY IGNORED** | REFUSED BY NAME | defect 1 |
| `drop_rate`, `max_drop`, `skip_drop`, `xgboost_dart_mode`, `drop_seed` | **HONORED** | n/a, mode refused | ignored with the mode | REFUSED BY NAME | with the mode |
| `uniform_drop` | **HONORED**, default diverges | n/a | ignored with the mode | REFUSED BY NAME | defect 2 |
| `boosting='rf'` | **HONORED**, all 3 policies | REFUSED BY NAME | **SILENTLY IGNORED**, and the rate is forced to 1.0 on the way | REFUSED BY NAME | defect 1, worse form |
| `early_stopping_rounds` (fit argument) | **HONORED** | REFUSED BY NAME (`BLOCK_VALIDATION_SET`) | ABSENT from the surface | REFUSED BY NAME | ok |
| `early_stopping_rounds` (constructor, no `eval_set`) | **SILENTLY IGNORED** | same | n/a | n/a | defect 3 |
| `eval_set` and the validation path | **HONORED** | REFUSED BY NAME | ABSENT from the surface | n/a | ok |
| `linear_tree` | **HONORED** | REFUSED BY NAME (`BLOCK_LINEAR_TREE`) | HONORED | REFUSED, Mojo-API only | ok |
| `enable_bundle` and its 6 knobs | **HONORED**, all 3 policies, dense and sparse | REFUSED BY NAME (`BLOCK_FEATURE_BUNDLING`) | HONORED | **HONORED** | ok |
| `boost_from_average` | **HONORED** | **HONORED** | HONORED on both dense arms, refused by name on sparse | ABSENT from the key list | ok |
| `langevin`, `diffusion_temperature`, `model_shrink_rate`, `model_shrink_mode` | REFUSED BY NAME | REFUSED BY NAME | REFUSED BY NAME | ABSENT | ok, and see below |

### Citations, row by row

**`boosting='dart'` on the estimators.** `sklearn._resolve_boosting`
(`sklearn.py:1733`) resolves the three spellings, `_params` writes
`"boosting"` at `sklearn.py:3061` and the six `drop_*` keys at
`sklearn.py:3074-3079`. `bindings/_mojotrees.mojo:1787 _parse_boosting` builds
`DartParams.enable(...)` from all six. `bindings/_mojotrees.mojo:2209` is the
dart/rf fork of `fit`, which calls
`alternate_boosting.fit_boosting` (`alternate_boosting.mojo:1725`) ->
`train_boosting` (1307) -> `train_dart` (540) -> `_dart_rounds` (443), which
calls `select_drop`, `dart_begin_round`, `dart_commit_round` in
`boosting_dart.mojo` and `tree.grow_tree` for the tree.
`fold_weights_into_trees` (408) folds the per-tree weights into node values and
the returned `Booster` carries a shrinkage factor of 1.0, which is why no
model-format change was ever needed. Tests:
`tests/test_alternate_boosting.mojo` (5 cases) and
`python/tests/test_params.py:185 test_dart_and_rf_train_through_the_estimator`,
which asserts the dart predictions differ from the gbdt ones.

**`boosting='rf'`.** Same route, `AlternateBoostingParams.rf()` ->
`boosting_rf.train_rf` (`boosting_rf.mojo:1006`) ->
`RfBooster.to_booster`. `sklearn._params` forces `learning_rate = 1.0` under
`rf` (`sklearn.py:2953`), matching LightGBM's `rf.hpp`.

**Both refused on the GPU** at `bindings/_mojotrees.mojo:2215-2218`, and
refused beside `boosting_type='ordered'`, `random_strength`,
`leaf_estimation_iterations`, `boost_from_average=false`, `ctr` and
`linear_tree` in the six guards at `bindings/_mojotrees.mojo:2083-2208`. On
the Python side `sklearn._refuse_alternate_boosting` (`sklearn.py:1688`)
refuses them beside sparse input (5031), `eval_set` (5076), a callable
objective (5095), `tree_learner` (5098), a multiclass classifier (5817, 5819),
sparse again (5916) and the ranker (6375).

**`enable_bundle`.** Honored by every CPU trainer that takes a
`BoosterParams`: `boosting.mojo` 2441 / 3238 / 3857 / 4278,
`alternate_boosting.mojo` 483 / 798 / 925 / 1194, `boosting_rf.mojo` 723 /
1406, and `boosting_sparse.prepare_bundling_csc` reached from
`model_sparse.mojo` 122 / 205. Refused by name by `check_bundling_honored` in
`train_gpu.mojo:1576` and `train_gpu_sparse.mojo:185`, by
`device_policy.mojo:2369` (`BLOCK_FEATURE_BUNDLING`, read before the crossover
table so `device='auto'` takes the CPU rather than failing), and by the six
entry points that pass `unbundled=` to `_parse_params`
(`bindings/_mojotrees.mojo` 2470, 2585, 2836, 2913, 3396, 5106). It is a
parameter-string key too (`params.mojo:1462`), so the CLI and the C API can
select it.

**`linear_tree`.** `bp.linear.is_active()` fork at
`bindings/_mojotrees.mojo:2248`, routed to `custom_metric.fit_with_metrics`
with a metric that costs nothing. Refused on the GPU at 2254-2257 and by
`device_policy.mojo:2380` (`BLOCK_LINEAR_TREE`), and under dart or rf by
`alternate_boosting._refuse_linear` (391) ->
`linear_tree.check_linear_tree_unconnected` (998).

**`boost_from_average`.** `sklearn.py:3148` sends it with a `_set` companion;
`tree_parameters_extra.mojo:1842` holds it;
`boosting.mojo:539` and `3039` and `3255` thread it into `_base_score`;
`train_gpu.mojo:4047` and `5980` thread it on the device;
`boosting._refuse_boost_from_average` (1268) is the refusal for the arms that
do not. CatBoost's per-objective resolution is
`auto_learning_rate.catboost_boost_from_average_default` (300), which returns
True for RMSE, MAE, Quantile and MAPE and False for everything else including
Logloss, read from `options_helper.cpp:353-374`. **The divergence this
parameter was reported as being is closed at head**; the estimator has the
parameter, both devices honor it, and only the sparse arm refuses it.

**`langevin` and friends.** `sklearn._check_langevin` (2199), called from
`_params` (2848), so every surface that builds a params dict refuses them.
`False` / `0.0` is accepted because that is what an untouched fit already is.
The mechanism in `src/mojotrees/langevin.mojo` is complete and tested; what is
missing is a `LangevinParams` and a `ModelShrinkParams` field on
`BoosterParams`, plus a leaf-sum call site in `tree.mojo`. This is the one
family where the refusal text is accurate, current, and names the blocker.

## The three defects

### Defect 1. `mojotrees.train` accepts `boosting='dart'` / `'rf'` and trains gbdt

**Reachable and silently ignored. This is a wrong answer and it is the same
shape as the two found earlier today.**

`python/mojotrees/basic.py:_Config` (176) constructs a real estimator from the
params dict, so `boosting='dart'` survives validation, and
`_Config.binding_params` (334) calls `self.base._params(...)` directly, which
writes `"boosting": "dart"` onto the wire. `Booster._build` (1129) then calls
`_mojotrees.train_dataset` / `train_dataset_multiclass` /
`train_dataset_ranker`, and `Booster.update` (1146) calls
`booster_update` / `booster_update_multiclass`. **None of those five bindings
calls `_parse_boosting`.** `params["boosting"]` is read at exactly one place
in the whole extension, `bindings/_mojotrees.mojo:1794`, inside
`_parse_boosting`, whose only caller is `fit` at 2082.

So:

```python
ds = mojotrees.Dataset(X, label=y)
bst = mojotrees.train({"objective": "regression", "boosting": "dart",
                       "drop_rate": 0.5}, ds, 100)
```

trains a plain GBDT ensemble and reports nothing. `mojotrees.cv` has the same
hole, because `cv.py:174` imports `train` from `basic.py`.

`'rf'` is worse than `'dart'` here, and the extra harm comes from a line that
is correct on its own. `sklearn._params` sets `learning_rate = 1.0` whenever
the resolved mode is `rf` (`sklearn.py:2953`), on the correct reasoning that a
forest averages and `boosting_rf` refuses any other rate. On the `fit` route
that is right. On this route the mode is dropped and the rate is not, so

```python
mojotrees.train({"objective": "regression", "boosting": "rf",
                 "learning_rate": 0.05}, ds, 100)
```

trains a **boosted** ensemble at learning rate **1.0**: not the forest asked
for, and not the rate asked for either. Neither value the user typed survives.

The severity argument matters because this is the route the benchmark uses.
`bench/real_data` trains through `mojotrees.train(params, Dataset)`, which is
stated at `bindings/_mojotrees.mojo:4885` and again at 5002. No current arm
sets `boosting`, so no published number is affected. What it means is that the
one route with a benchmark pointed at it is the route with no mode dispatch.

**Fix, and it is not this lane's file.** The narrow, safe form is a refusal,
not a wiring: none of those five trainers has a dart or forest round loop, so
the honest answer is the sentence `_refuse_alternate_boosting` already writes.
Exact text below.

### Defect 2. `uniform_drop`'s default diverges from LightGBM, ungated

`boosting_dart.DEFAULT_UNIFORM_DROP` is `False`, which is LightGBM's own
default, and `tools/check_parity.py:1538` gates on exactly that
(`STOCK_COMPTIME_DEFAULTS` against `LIGHTGBM_STOCK["uniform_drop"] = False` at
1476). The gate is green.

**No Python fit reads that constant.** `sklearn.py:1081` ships
`uniform_drop=True` in the estimator signature and
`bindings/_mojotrees.mojo:1800` passes the wire value explicitly on every
fit, so `DartParams.enable`'s default is never taken. The divergence is
documented in prose (`sklearn.py:526`, "uniform_drop defaults to True here
(LightGBM defaults to False)") and has no entry in
`check_parity.STOCK_DIVERGENCES`, where `enable_bundle`'s does, and
`uniform_drop` is not in `STOCK_PYTHON_SIGNATURE` either. So the gate that
exists to catch a silent convergence or divergence is watching the wrong copy
of the number.

This is the rule-7 NULL shape: a green parity check on a value no user meets.
It also breaks the standing rule that the lossguide path mirrors LightGBM,
which is the path `boosting='dart'` runs on.

The two rules are not cosmetically different. `uniform_drop=True` draws each
iteration independently at the same rate; `False` scales that rate by
`w_i / mean(w)`, so the iterations that currently weigh most are the likeliest
to be hidden, and under DART's own normalization the recent iterations weigh
least. The two select systematically different drop sets from round two
onward.

### Defect 3. A constructor `early_stopping_rounds` with no `eval_set` is dropped

`_fit_args._check_eval_arguments` (377) refuses `eval_set=None` beside a
positive `early_stopping_rounds`, `eval_metric`, `eval_sample_weight` or
`callbacks`, by name, with the argument to pass instead. That is correct and
covers the `fit(early_stopping_rounds=20)` case.

It reads the **fit argument only**. The estimator also carries
`early_stopping_rounds` (`sklearn.py:985`), documented at `sklearn.py:559` as
"sets the default for `fit`'s argument of the same name", and that default is
resolved inside `_fit_with_metrics` (3849-3868), which only runs when there is
an `eval_set`. So

```python
MojoTreesRegressor(n_estimators=1000, early_stopping_rounds=20).fit(X, y)
```

trains all 1000 rounds and says nothing, while the same number passed to
`fit()` raises. One value, two behaviors, and the silent one is the one a
scikit-learn user reaches through `GridSearchCV`, which sets constructor
parameters and calls `fit(X, y)`.

Lowest severity of the three: nothing is mistrained, a user simply does not
get the early stopping they configured. It is still accept-and-ignore.

## Unreachable: implemented, correct, no binding

Every row here is Mojo-API reachable and Python-unreachable, and in every case
the missing piece is a binding rather than an algorithm. None of them can be
selected today, so wiring any of them cannot move a bit in any existing fit.

| entry point | what is missing | first-user exposure if wired |
| --- | --- | --- |
| `alternate_boosting.fit_boosting_multiclass` (1787) | a dart/rf fork in `fit_multiclass`; `sklearn._refuse_alternate_boosting` refuses at 5817 first | multiclass DART and forests; covered by `tests/test_alternate_boosting.mojo:179`, so it has run |
| `train_dart_with_valid` (732), `train_dart_multiclass_with_valid` (1145) | a dart branch in `fit_with_metrics`; refused at `sklearn.py:5076` first | DART early stopping. Untested. `DartBestState` exists because truncation is wrong under DART, and nothing has ever exercised the snapshot-and-restore |
| `train_dart_more` (620), `train_dart_multiclass_more` (1065) | `booster_update` never reads `boosting` | continued DART. Its own docstring says `50 + 50` is not `100`, because the weight vector cannot be read back off a folded model |
| `train_rf_more` (`boosting_rf`) | same | continued forests; recomputes the fit constant from `target`, exact only on the training data |
| `boosting_rf.train_forest*` and `RfBooster` / `RfMulticlassBooster` | no binding at all; `train_rf` is what `fit` reaches | iteration ranges on a forest, a forest randomized by GOSS, a forest randomized by `pos_bagging_fraction` / `neg_bagging_fraction`. The bridged `Booster` divides by the whole tree count whatever the range, which is why the second layer exists |
| `boosting_dart.dart_weights_are_uniform` | **no caller anywhere, tests included** | none. Dead. The fold means no serializer ever sees a weight vector, so the question has no asker |
| `langevin.mojo` in full | `LangevinParams` and `ModelShrinkParams` fields on `BoosterParams` (`boosting.mojo`), a leaf-sum call site (`tree.mojo`) | refused by name today, which is the right state until the trainer change lands |
| reading a LightGBM `average_output` model | `lgbm_model_io.mojo:808` refuses on the header line | see the decline note below |

## Two declines that carry no price (LANE_RULES rule 6)

**`params.mojo:706-716`, the parameter-string refusal of dart / goss / rf.**
The stated reason is that selecting one "needs a parameter bundle (DartParams,
GossParams) or the random-forest loop, which a whitespace-separated string
cannot carry any more than it can carry `bagging_fraction`". That reasoning is
falsified nine lines above it in the same docstring, which explains that
`ordered` is accepted precisely because its four knobs are scalars a string can
carry. DART's six knobs are scalars too, and all six are already in
`_MOJO_API_ONLY` (`params.mojo:187-188`), which is the list of names the parser
knows and declines. `rf` needs no bundle at all. So the decline prices nothing
and the barrier it names is not the barrier. ASSERTED, and therefore an open
item. The CLI and C API are the surfaces it costs.

**`lgbm_model_io.mojo:808`, the `average_output` refusal.** "mojotrees sums
them, so a random forest cannot be converted." mojotrees now trains an
averaged forest and bridges it to a summing `Booster` with base score 0 and
rate `1 / T` (`boosting_rf.RfBooster.to_booster`), and the tree count an
importer would need is known once the trees are read. So the stated
impossibility is not one. Priced at nothing, ASSERTED, open. Low value.

## Recommendation per item: WIRE, REMOVE, or LEAVE

DART's and RF's drop rules, normalization, `max_drop` semantics and
`uniform_drop` semantics were verified against LightGBM's `dart.hpp` and
`rf.hpp` on 2026-08-15, function by function, with the citations still in the
code. That is banked work and it argues against removal everywhere below.

| item | recommendation | reason |
| --- | --- | --- |
| `boosting='dart'` / `'rf'` on `mojotrees.train` | **WIRE the refusal, not the mode** | Five trainers with no dart or forest loop. A refusal costs six lines and removes a wrong answer; wiring the mode into five round loops is a trainer project |
| `uniform_drop` default | **WIRE**, flip the estimator default to `False` | Restores LightGBM parity on the path that claims it, and makes the gate watch the number a user meets. Moves bits, but only for a `boosting='dart'` fit, which no benchmark arm and no default fit reaches |
| constructor `early_stopping_rounds` with no `eval_set` | **WIRE the refusal** | One line in `_check_eval_arguments`'s caller, and it makes one parameter behave one way |
| `boosting='dart'` / `'rf'` under `device='auto'` | **WIRE a policy block** | See below. A resolved default should not turn a valid configuration into an error |
| DART itself, all of `boosting_dart.mojo` | **LEAVE** | Reachable, honored, tested, LightGBM-verified. Nothing to do |
| RF itself, `train_rf` layer | **LEAVE** | Same |
| `train_dart_with_valid` and the DART early-stopping trio | **LEAVE unwired, ready-to-wire** | Correct on reading and never executed. See "what I did not wire" |
| `fit_boosting_multiclass` | **LEAVE unwired, ready-to-wire** | Has actually run in `tests/test_alternate_boosting.mojo`, so it is the strongest candidate of the unreachable set |
| `train_*_more` continuations | **LEAVE unwired** | Each carries a documented inexactness (`50 + 50` is not `100`; the forest constant is recomputed). Wiring these ships a caveat, not a feature |
| `boosting_rf.train_forest*` second layer | **LEAVE** | It is the escape hatch the bridge's two known losses point at. Cheap to carry, no maintenance surface |
| `dart_weights_are_uniform` | **REMOVE**, owner's call | Dead with no caller and a justification that is false. Marked in place |
| `langevin` family | **LEAVE refused** | The refusal names the blocker and the blocker is real |
| `enable_bundle`, `linear_tree`, `boost_from_average`, `eval_set` | **LEAVE** | Reachable, honored, refused by name where not |

### Not a defect but adjacent: `device='auto'` with dart or rf

`device_policy.mojo` has 21 block codes (558-631) and none of them is
`boosting='dart'` or `'rf'`. `device_selection.Workload` has no field for the
mode either. The pattern for exactly this case exists three times over:
`BLOCK_LINEAR_TREE`, `BLOCK_FEATURE_BUNDLING`, `BLOCK_VALIDATION_SET`, each
read before the crossover table so that `auto` takes the CPU without a shape
being compared.

Without one, `device='auto'` on a shape the policy sends to the accelerator
resolves to `"gpu"`, and then
`bindings/_mojotrees.mojo:2206` raises "boosting='dart' and boosting='rf'
train on the CPU only; set device='cpu'". A caller who typed no device gets an
error where the CPU fit was available. The default is `device="cpu"`
(`sklearn.py:1046`), so this needs `device='auto'` to be typed, which is why it
is filed here rather than as defect 4.

Fixing it is a `POLICY_VERSION` bump (5 added the three above, 8 added
`BLOCK_RANDOM_STRENGTH`, so this is 9), a field on `DeviceRequest`, a field on
the Python `Workload`, and one argument at `sklearn._resolve_device`'s call
site. Four files, none of them this lane's.

## What was verified against LightGBM and still holds

Re-read at head 2026-08-17, all four confirmed still true of the code:

- Normalization **multiplies** the shrinkage rather than replacing it.
  `dart_normalization` returns `learning_rate / (k + 1)` for the new tree and
  `k / (k + 1)` for each dropped iteration, matching `DroppingTrees`'s final
  two assignments and `Normalize`.
- `max_drop` **caps the rate first and then breaks ascending**, so the set that
  survives the cap is the earliest iterations drawn, not the hardest drawn.
  `select_drop` steps 2 and 4.
- `uniform_drop=False` is weight-proportional and is LightGBM's default, and
  under it LightGBM's own cap expression (`max_drop * inv_average_weight /
  sum_weight`) comes out far above 1 so the cap never binds. Reproduced as
  written with the oddity named in the comment rather than corrected in
  silence.
- LightGBM's DART **silently ignores early stopping**. mojotrees refuses the
  pair by name instead (`sklearn.py:5076`), which is better behavior and is
  also why `train_dart_with_valid` has no caller.

## What could not be verified without a compiler

- That the three files edited in this pass still compile. The edits are
  docstrings, comments and two `raise Error` message strings; no signature,
  no control flow and no literal changed. The two message strings are
  multi-argument `raise Error(...)` calls whose argument count and types are
  unchanged.
- Whether `dart_weights_are_uniform` is referenced from a generated or
  reflected surface rather than by name. The grep covered `*.mojo`, `*.py` and
  `*.md` outside `.claude/`, and found only its own definition.
- Whether the dask distributed path can reach dart or rf. `CAPABILITIES` in
  `_dask_runtime.py` has no dart or rf name and `_shared_capabilities`
  (`dask.py:1526`) asks only about goss, so a mode would be dropped rather than
  negotiated. It appears moot because `provider.distributed_worker_train` does
  not exist anywhere in the extension, which would make
  `native_runtime_status().available` False and every distributed fit raise
  before a mode mattered, but that is inferred from a grep and not traced.
- Every claim in this file about what a trainer *does* is from reading the call
  graph. Nothing here was run.

## Appendix: the exact edits, for the lanes that own the files

This lane owns `boosting_dart.mojo`, `boosting_rf.mojo`, `efb.mojo` and this
file, and made only documentation and refusal-message corrections in the first
three. Everything below is a CORRECTION or a FIX in somebody else's file, quoted
so it can be applied verbatim rather than re-derived. None of them is a
suggestion.

### FIX 1, defect 1. `bindings/_mojotrees.mojo`, five entry points

Add, immediately after each of these five functions reads its `Dataset` /
`Model` handle and before it calls `_parse_params`:

```mojo
    # LightGBM's `boosting`, refused rather than dropped. `_parse_boosting` is
    # called by `fit` and by nothing else, and this trainer has no dropout and
    # no forest round loop, so a `boosting` of dart or rf here would train a
    # plain gbdt ensemble under a parameter that named another algorithm. It
    # also loses a second value on the way: `sklearn._params` forces
    # `learning_rate = 1.0` when the mode is rf, so a dropped `rf` leaves a
    # boosted fit at rate 1.0 rather than at the rate the caller typed.
    var _mode = parse_boosting(String(py=params["boosting"]))
    if _mode == BOOSTING_DART or _mode == BOOSTING_RF:
        raise Error(
            "boosting='",
            boosting_name(_mode),
            "' is not available on this entry point: dart and rf run"
            " alternate_boosting's own round loops, which only"
            " _mojotrees.fit reaches. Train through an estimator's fit()"
            " on dense, single-output, CPU input, or set boosting='gbdt'",
        )
```

The five are `train_dataset` (4874), `train_dataset_multiclass` (5047),
`train_dataset_ranker` (5091), `booster_update` (5122) and
`booster_update_multiclass` (5205). `boosting_name` needs adding to the
existing `from mojotrees.alternate_boosting import (...)` block near line 272,
beside `parse_boosting`, which is already imported.

A cheaper single-site alternative, if the orchestrator prefers Python: refuse in
`python/mojotrees/basic.py:_Config.__init__`, after the estimator is built, with
`self.base._refuse_alternate_boosting("through mojotrees.train()")`. That covers
`train`, `Booster.update` and `mojotrees.cv` in one line because all three go
through `_Config`, and it reuses the message that already exists. It does NOT
cover a direct `_mojotrees.train_dataset` call, which is why the native version
above is the complete one.

### FIX 2, defect 2. `python/mojotrees/sklearn.py:1081`

```
-        uniform_drop=True,
+        uniform_drop=False,
```

and at `sklearn.py:525-527`:

```
-    `skip_drop`, `xgboost_dart_mode`, and `drop_seed` are LightGBM's
-    parameters of those names. `uniform_drop` defaults to True here (LightGBM
-    defaults to False); both drop rules follow LightGBM's `dart.hpp`. "rf" is random
+    `skip_drop`, `xgboost_dart_mode`, `uniform_drop`, and `drop_seed` are
+    LightGBM's parameters of those names, at LightGBM's own defaults. Both
+    drop rules follow LightGBM's `dart.hpp`; `uniform_drop=False`, the
+    default, draws each iteration at a rate proportional to what it currently
+    weighs, and `True` draws every iteration at the same rate. "rf" is random
```

**This moves bits, and only for a `boosting='dart'` fit.** No default fit and no
benchmark arm sets the mode, so nothing recorded changes; a user who explicitly
selected dart and did not name `uniform_drop` gets a different drop set from
round two onward. It needs the real-data gate under LANE_RULES rule 3 only if a
dart arm is ever added, and it should be announced in the changelog as a default
change rather than a fix.

If the owner would rather keep `True`, then the CORRECTION instead is to add the
divergence to the gate that is supposed to see it, in
`tools/check_parity.py:1497 STOCK_DIVERGENCES`:

```python
    "uniform_drop": (
        False,
        True,
        "the native default (boosting_dart.DEFAULT_UNIFORM_DROP) is "
        "LightGBM's False, but the estimator signature ships True and "
        "bindings/_mojotrees.mojo:_parse_boosting passes the wire value on "
        "every fit, so no Python fit reads the native default. Registered "
        "2026-08-17 after the comptime gate was found to be watching a "
        "constant no user meets",
    ),
```

Keeping `True` with no entry here is the one option that is not available,
because that is the state the gate reads as parity.

### FIX 3, defect 3. `python/mojotrees/sklearn.py`

In each `fit` that calls `_check_eval_arguments`, pass the constructor value as
the fallback so one parameter behaves one way:

```
-        _check_eval_arguments(
-            eval_set,
-            eval_metric,
-            eval_sample_weight,
-            early_stopping_rounds,
-            callbacks,
-        )
+        _check_eval_arguments(
+            eval_set,
+            eval_metric,
+            eval_sample_weight,
+            # The constructor value too, not just the fit argument. A
+            # positive `early_stopping_rounds` on the estimator with no
+            # eval_set was silently dropped while the same number passed to
+            # fit() raised, and the silent path is the one GridSearchCV
+            # takes. Corrected 2026-08-17.
+            early_stopping_rounds or self.early_stopping_rounds,
+            callbacks,
+        )
```

Call sites: `sklearn.py:5017`, `5771`, `6354`. No native change.

### CORRECTION 1. `src/mojotrees/histogram.mojo`, two stale claims

The one-word change both of these were waiting for has landed:
`boosting_rf._multiclass_rf_gradients` declares `raises` at
`boosting_rf.mojo:1453`, and `boosting._fill_softmax_grad_hess` calls
`check_derivative_precision()` because of it. Both comments still say it does
not. At `histogram.mojo:598-605`, in `derivative_precision_narrows`:

```
-    `float64` selects Float64 derivatives; `float32`, unset, and anything
-    else select the narrowing default. It does not raise, because two of its
-    three callers sit in contexts that cannot
-    (`boosting._fill_softmax_grad_hess` is reached from
-    `boosting_rf._multiclass_rf_gradients`, which is not this lane's file to
-    add a `raises` to). A typo is diagnosed by
-    `check_derivative_precision` instead, which is called from every entry
-    that can raise; see there for the one gap that leaves and the one word
-    that closes it.
+    `float64` selects Float64 derivatives; `float32`, unset, and anything
+    else select the narrowing default. It does not raise, because it is a
+    predicate and its callers want an answer rather than an exception. A typo
+    is diagnosed by `check_derivative_precision` instead, which is now called
+    from every derivative site without exception; see there.
```

At `histogram.mojo:620-627`, in `check_derivative_precision`:

```
-    Called from `ConstHessianSettings.resolve()` (once per fit) and from
-    `boosting.fill_grad_hess` (once per round). **The one path it does not
-    cover** is a multiclass or random-forest fit that resolves no snapshot
-    and never reaches `fill_grad_hess`, because its only derivative site is
-    `boosting._fill_softmax_grad_hess`, which cannot raise while
-    `boosting_rf._multiclass_rf_gradients` does not declare `raises`. That
-    is a one-word change in a file this lane does not own, and the report
-    names it.
+    Called from `ConstHessianSettings.resolve()` (once per fit), from
+    `boosting.fill_grad_hess` (once per round) and from
+    `boosting._fill_softmax_grad_hess` (once per class per round).
+    **There is no longer a path it does not cover.** The gap was a multiclass
+    or random-forest fit that resolves no snapshot and never reaches
+    `fill_grad_hess`; it closed when `boosting_rf._multiclass_rf_gradients`
+    declared `raises`, which let `_fill_softmax_grad_hess` call this. Both
+    landed before 2026-08-17 and these two docstrings were the last things
+    still describing the hole.
```

### CORRECTION 2. `src/mojotrees/device_policy.mojo:2373`

```
-                "enable_bundle is applied by the dense CPU trainers in"
-                " boosting.mojo, which build a bundled matrix before they"
-                " grow; the GPU trainers build their histograms from the"
-                " unbundled binned matrix and cannot apply it"
+                "enable_bundle is applied by every CPU trainer that takes a"
+                " BoosterParams -- boosting.mojo, alternate_boosting.mojo"
+                " (dart), boosting_rf.mojo (rf) and boosting_sparse.mojo"
+                " (CSC) -- each of which builds a bundled matrix before it"
+                " grows; the GPU trainers build their histograms from the"
+                " unbundled binned matrix and cannot apply it"
```

### CORRECTION 3. `src/mojotrees/params.mojo:693-716`, an unpriced decline

The docstring's reason for refusing dart / goss / rf in a parameter string is
falsified by the paragraph above it about `ordered`. Replace the last paragraph
before the code with:

```
-    `dart`, `goss` and `rf` are implemented, but selecting one means handing
-    a trainer a `DartParams`, a `GossParams` or the RF loop, which a
-    whitespace-separated string cannot carry, exactly as it cannot carry
-    `bagging_fraction`. They are refused with the Mojo API sentence rather
-    than as unknown values.
+    `dart`, `goss` and `rf` are implemented and reachable from the Python
+    estimators, and are refused here because this surface has no parser
+    entries for them, NOT because a string cannot express them. That
+    distinction was recorded wrong until 2026-08-17: the old text said the
+    modes "need a parameter bundle ... which a whitespace-separated string
+    cannot carry", which the `ordered` paragraph above disproves. DART's six
+    knobs are scalars, all six are already listed in `_MOJO_API_ONLY`, and
+    `rf` needs no bundle at all; `goss`'s four are scalars too. So this is an
+    OPEN item under LANE_RULES rules 4 and 6 rather than a settled
+    impossibility, and what it costs is the CLI and the C API, which are the
+    two surfaces that reach the library only through this parser.
```

Also correct the refusal message itself, which tells a Python user something
false:

```
-            "' is supported by the Mojo API only, not by parameter strings:"
-            " it needs a parameter bundle (DartParams, GossParams) or the"
-            " random-forest loop, which a whitespace-separated string cannot"
-            " carry any more than it can carry bagging_fraction",
+            "' is not supported by parameter strings, which have no parser"
+            " entries for it. dart and rf are reachable from the Python"
+            " estimators (boosting_type='dart' / 'rf' on a dense,"
+            " single-output CPU fit) and from the Mojo API"
+            " (alternate_boosting.fit_boosting); goss is reachable from"
+            " both as well",
```

### CORRECTION 4. `src/mojotrees/lgbm_model_io.mojo:808-813`, an unpriced decline

Low value, listed for completeness. "mojotrees sums them, so a random forest
cannot be converted" states an impossibility that
`boosting_rf.RfBooster.to_booster` disproves: base score 0 and rate `1 / T`
make a summing `Booster` compute a mean, and the tree count is known once the
trees are read. Suggested replacement, as a priced decline rather than a
prohibition:

```
-                    "this LightGBM model averages its trees (boosting='rf');"
-                    " mojotrees sums them, so a random forest cannot be"
-                    " converted"
+                    "this LightGBM model averages its trees (boosting='rf');"
+                    " mojotrees can represent that (boosting_rf.RfBooster."
+                    "to_booster bridges an averaged forest to a summing"
+                    " Booster at base score 0 and rate 1/T) but this reader"
+                    " does not build the bridge, because the rate depends on"
+                    " the tree count and the header is parsed before the"
+                    " trees are. Not refused on principle; unimplemented"
```

### CORRECTION 5. `tests/test_gpu_refusals.mojo:182`, a stale docstring

Not an assertion, so nothing fails, but the sentence is one of the copies of the
claim corrected everywhere else.

```
-    Bundling is applied by the dense CPU trainers in boosting.mojo, which fit
-    a plan once per training call and grow every tree on the bundled matrix.
+    Bundling is applied by every CPU trainer that takes a BoosterParams --
+    boosting.mojo, alternate_boosting.mojo (dart), boosting_rf.mojo (rf) and
+    boosting_sparse.mojo (CSC) -- each of which fits a plan once per training
+    call and grows every tree on the bundled matrix.
```

The assertion in that test is `assert_raises(contains="enable_bundle")`, which
both of the corrected refusal messages in `efb.mojo` still satisfy, so this
lane's message edits cannot fail it.

### Not a defect, but the same shape. `device_policy.mojo` has no dart/rf block

Described in the body above. `POLICY_VERSION` 9, a `DeviceRequest` field, a
`Workload` field in `python/mojotrees/device_selection.py`, and one argument at
`sklearn._resolve_device`'s call site. Four files, none of them this lane's, and
it needs the owner's decision on whether the policy should learn about training
modes at all.
