# Remaining 02 handoff, random-forest boosting mode

What this session did, in the three files it owns, and nothing else.

| File | Change |
|---|---|
| `src/mojoboost/boosting_rf.mojo` | new. The whole of `boosting='rf'`: parameters, validation, the round loop, the averaged model, multiclass, validation-set early stopping, continued training, and the `Booster` bridge |
| `docs/RANDOM_FOREST_MODE.md` | new. The specification: LightGBM's semantics, what is refused, what differs and why, what is unverified |
| `handoffs/remaining_02_rf.md` | this file |

No other file was edited. No test was written, nothing was compiled, nothing
was run, nothing was committed. Static reasoning only, as the brief required,
so every claim below is a reading of the code and not a measurement.

**State of the shared checkout.** Concurrent lanes are live and were mid-edit
throughout. `src/mojoboost/alternate_boosting.mojo`, `boosting_dart.mojo`,
`cegb.mojo`, `linear_tree.mojo`, `quantized_gradient.mojo` and the docs
beside them appeared during this session; `lgbm_model_io.mojo`, `tree.mojo`,
and `trainset.mojo` were modified by other lanes while this file was being
written. Nothing here touches any of them.

**The one hard constraint this session had to meet.**
`src/mojoboost/alternate_boosting.mojo` (another lane's file) already
contains

```mojo
from .boosting_rf import train_rf, train_rf_more
```

and calls both, against a `BoosterParams` and an ordinary `Booster`, before
`boosting_rf.mojo` existed. Its docstring also names `boosting_rf.is_forest`
and `boosting_rf.check_rf_params`. That contract was treated as fixed:
`train_rf`, `train_rf_more`, and `is_forest` in this module have exactly the
signatures that file calls, so it compiles unchanged. Section 1.2 spells out
the contract that is now load-bearing between the two lanes.

---

## 1. What exists

### 1.1 The algorithm layer

`train_forest` is random-forest mode. The distinction the whole module rests
on, and the one the brief warned about, is that this is not GBDT with row
bagging:

- gradients and hessians are computed **once**, at the constant raw score
  `base_score`, and every tree fits those same numbers
  (LightGBM's `RF::Boosting()`, called from `Init` and not per iteration);
- the model is an **average**, and `learning_rate` is refused rather than
  applied (LightGBM's `shrinkage_rate_ = 1.0f`);
- the base score is folded into every tree's node values
  (LightGBM's `Tree::AddBias`), so a tree predicts on the raw scale alone;
- no round is skipped and no run stops early, because the tree count is the
  denominator of the average.

| Symbol | What it is |
|---|---|
| `RfParams` | `n_estimators`, `TreeParams`, and the three row samplers. No `learning_rate` field, deliberately |
| `RfParams.default()` | 100 trees, `bagging_fraction=0.8`, `bagging_freq=1`, `bagging_seed=3`. LightGBM's own defaults do not satisfy its own rf check, so a working default is supplied |
| `RfParams.from_booster_params` | the `(BoosterParams, BaggingParams)` pair the mode surface arrives with |
| `check_rf_params` | `RF::Init`'s check: an objective rf can serve, the sampler ranges, sampler exclusivity, the feature fractions, and at least one source of per-tree randomization |
| `rf_randomizer_name` | which source satisfies it, in LightGBM's order, or `""` |
| `check_rf_objective` | refuses `CUSTOM` and `LAMBDARANK` by name |
| `check_rf_init_score` | refuses `init_score` |
| `check_rf_learning_rate` | refuses any rate but 1.0 |
| `is_rf_boosting` | LightGBM's two spellings, `rf` and `random_forest` |
| `RfBooster` | the averaged single-output model, with LightGBM's slice semantics |
| `RfMulticlassBooster` | the averaged softmax model, round-major |
| `train_forest`, `train_forest_more`, `train_forest_with_valid` | single-output training, continuation, early stopping |
| `train_forest_multiclass`, `..._more`, `..._with_valid` | the softmax counterparts |

### 1.2 The `boosting='rf'` surface, and the cross-lane contract

| Symbol | Signature | Called by |
|---|---|---|
| `train_rf` | `(data, target, objective, params: BoosterParams, sample_weight=[], alpha=0.9, bagging=BaggingParams.disabled(), init_score=[]) raises -> Booster` | `alternate_boosting.train_boosting` |
| `train_rf_more` | `(mut booster: Booster, data, target, params: BoosterParams, sample_weight=[], alpha=0.9, bagging=BaggingParams.disabled(), init_score=[]) raises -> Int` | `alternate_boosting.train_boosting_more` |
| `is_forest` | `(booster: Booster) -> Bool` | named in `alternate_boosting`'s docstring; `train_rf_more` enforces it |

`train_rf` is `train_forest` followed by `RfBooster.to_booster()`, which
returns a `Booster` with `base_score = 0.0` and `learning_rate = 1 / T`. A
`Booster` computes `base_score + sum_i(learning_rate * tree_i)`, so that is
the forest's mean exactly, and an rf model therefore serializes, predicts,
dumps, and explains through paths that need no change at all.

`train_rf_more` takes the ensemble by reference because growing a forest
rescales every tree already in it: after appending it sets
`booster.learning_rate = 1 / new_T`. It refuses to touch an ensemble
`is_forest` rejects, since rewriting the shrinkage of a boosted model would
be a silent corruption.

Two facts the other lane's file should be read against:

1. Its docstring at line 72 attributes the learning-rate refusal to
   `boosting_rf.check_rf_params`. The rule is live, in
   `check_rf_learning_rate`, which `train_rf` and `train_rf_more` both call
   first. Documentation nit in that file, not a behavior gap.
2. Its docstring says the forest's folded factor is `1 / K`. It is, and
   `is_forest` tests exactly `booster.learning_rate == 1.0 / Float64(T)`,
   which is exact across a `serialize` round trip because that format stores
   raw IEEE-754 bit patterns.

### 1.3 Reuse rather than reimplementation

Nothing in the objective layer was copied. `boosting_rf.mojo` imports and
calls `_base_score`, `_check_objective`, `_check_sample_weight`, `_check_goss`,
`_check_class_bagging`, `_fill_grad_hess`, `_renew_leaf_values`,
`_fill_softmax_grad_hess`, `_softmax_inplace`, `_mean_loss`,
`_multiclass_mean_loss`, `_multiclass_goss_select`, `objective_renews_leaves`,
`renewal_alpha`, and `renewal_weights` from `boosting.mojo`, plus the samplers
from `bagging.mojo`, `goss.mojo`, and `sampling.mojo`, and `grow_tree` from
`tree.mojo`. `boosting_sparse.mojo` sets the precedent for importing the
underscore-prefixed names across modules in this package.

The single point worth noticing: LightGBM's rf `residual_getter` is
`label[i] - init_score`, a constant, where GBDT's is `label[i] - score[i]`.
`boosting._renew_leaf_values` already takes the raw-score vector as an
argument, so passing it the constant vector *is* LightGBM's rf renewal and
there is no second renewal path anywhere.

Three small duplications remain, each because the original is private to
another module and this lane owns neither. Each is a patch below:
`_same_signs` (5.5), `_class_log_priors` (5.5), `_objective_response`, which
routes through an empty `Booster` rather than re-casing the inverse links
(5.5).

One dependency was deliberately not taken. `check_rf_objective` refuses
`lambdarank` by its integer code, spelled `_LAMBDARANK = 7` rather than
imported from `ranking.mojo`: that module imports `model.mojo`, and patch 4.2
asks `model.fit` to reach a boosting mode, which would close the loop
`model -> boosting_rf -> ranking -> model` the moment it is applied. If
`ranking.LAMBDARANK` ever moves to a leaf module, import it and delete the
constant.

---

## 2. What is inert, and why

`boosting='rf'` is not reachable from any user-facing surface.

| Surface | State | Blocking file (not owned) |
|---|---|---|
| Mojo API | **reachable**: `alternate_boosting.train_boosting(..., AlternateBoostingParams.rf())`, and `train_forest` directly | none |
| `src/mojoboost/__init__.mojo` | no export. `boosting_rf` and `alternate_boosting` are both absent | `__init__.mojo` |
| parameter strings (`parse_params`) | `boosting` and `boosting_type` are in `_MOJO_API_ONLY`, so a string naming either is refused | `params.mojo` |
| C ABI, CLI | inherit the parameter-string refusal | `capi/`, `cli/` |
| Python estimators | `_BOOSTING_TYPES = ("gbdt", "goss")`, so `boosting='rf'` raises `unknown boosting` | `python/mojoboost/__init__.py` |
| bindings | `fit` calls `model.fit`, which has no mode argument | `bindings/_mojoboost.mojo`, `model.mojo` |
| multiclass rf | implemented, but `alternate_boosting` refuses multiclass for both alternate modes | `alternate_boosting.mojo` |
| rf with a validation set | `train_forest_with_valid` exists; the mode surface has no `train_boosting_with_valid` | `alternate_boosting.mojo` |
| GPU | `train_gpu` has its own round loop and no mode argument | `train_gpu.mojo` |
| sparse | `boosting_sparse` has its own round loop and no mode argument | `boosting_sparse.mojo` |
| LightGBM model import | `average_output` is refused by name | `lgbm_model_io.mojo` |
| model files | no `average_output` flag; the bridge's `1 / T` carries it implicitly | `serialize.mojo` |

Nothing was half-wired to shorten that list. An accepted `boosting='rf'` that
trained a summed ensemble, or a saved model that silently lost its averaging,
would be worse than a mode that is still refused.

---

## 3. Default-behavior invariants

Everything below is a claim about code paths, not a measurement.

1. **No existing training path changes.** No file outside this lane was
   edited. `boosting.train`, `train_gpu`, `train_sparse`, and the ranker are
   untouched, and `boosting_rf.mojo` has no module-level side effects.
2. **`alternate_boosting.mojo` compiles against this module** if and only if
   the three signatures in 1.2 hold. They were written from its call sites.
3. **`AlternateBoostingParams()` is GBDT**, so every existing caller of
   `train_boosting` keeps its behavior; rf is reached only by naming it.
4. **A forest never reaches `boosting.train_more`.**
   `alternate_boosting.train_boosting_more` dispatches rf to `train_rf_more`,
   which checks `is_forest`. A GBDT ensemble handed to `train_rf_more` is
   refused rather than rescaled.
5. **`RfParams` cannot express a shrunk forest.** There is no
   `learning_rate` field, so the only way a rate reaches rf is through
   `BoosterParams` at the mode surface, where `check_rf_learning_rate`
   refuses anything but 1.0.
6. **An unrandomized forest cannot be trained.** `check_rf_params` refuses
   it, and every entry point in the module calls `check_rf_params` before
   growing anything.
7. **Serialization is unchanged.** No format version, no new section, no new
   field. The bridge writes an ordinary `Booster`.

---

## 4. READY-TO-APPLY INTEGRATION PATCHES

Each is a single edit outside this lane's ownership, listed so it can be
applied mechanically. None has been applied. None has been compiled.

### 4.1 `src/mojoboost/__init__.mojo`, export the mode

- **Target file / symbol**: `src/mojoboost/__init__.mojo`, the import block.
- **Edit**: add, in the alphabetical position after the `from .boosting import (...)` block:

```mojo
from .boosting_rf import (
    RF_SHRINKAGE,
    RfBooster,
    RfMulticlassBooster,
    RfParams,
    check_rf_init_score,
    check_rf_learning_rate,
    check_rf_objective,
    check_rf_params,
    is_forest,
    is_rf_boosting,
    rf_randomizer_name,
    train_forest,
    train_forest_more,
    train_forest_multiclass,
    train_forest_multiclass_more,
    train_forest_multiclass_with_valid,
    train_forest_with_valid,
    train_rf,
    train_rf_more,
)
```

- **Call site**: none; this makes `from mojoboost import train_rf` work.
- **State flow**: none.
- **Errors**: none added.
- **Ownership**: `__init__.mojo` is unowned by this lane and is edited by
  most lanes in this round. Apply after the others, and expect a merge.
- **Fallback**: without it, `boosting_rf` is importable as
  `mojoboost.boosting_rf` by submodule path only.
- **Serialization effect**: none.
- **Public API effect**: adds 19 Mojo symbols. The API snapshot, if the round
  has one, has to be regenerated.
- **Dependency**: none. Apply first.
- **Validation, UNRUN**: `pixi run test` compiles the package; a module that
  fails to import fails every test file.

### 4.2 `src/mojoboost/model.mojo`, a mode on `fit`

- **Target file / symbol**: `model.fit`, and `model.fit_multiclass`.
- **Signature**: append one argument, last so every positional caller is
  unaffected:

```mojo
def fit(
    ...,
    categorical_features: List[Int] = [],
    boosting: AlternateBoostingParams = AlternateBoostingParams(),
) raises -> Model:
```

- **Call site**: inside, replace the CPU branch

```mojo
    else:
        booster = train(data, target, objective, params, sample_weight, alpha, bagging, goss)
```

with

```mojo
    else:
        booster = train_boosting(
            data, target, objective, params, boosting,
            sample_weight, alpha, bagging, goss,
        )
```

- **State flow**: the mode travels from the caller to
  `alternate_boosting.train_boosting`, which dispatches. `AlternateBoostingParams()`
  is GBDT, so the default path is `train` with the arguments it takes today.
- **Errors**: a non-GBDT mode with `device=GPU_DEVICE` must raise rather than
  fall back. Add, before `resolve_device`:

```mojo
    if boosting.mode != BOOSTING_GBDT and device != CPU_DEVICE:
        raise Error(
            "boosting='", boosting_name(boosting.mode),
            "' is CPU only: the GPU trainer carries its own round loop",
        )
```

- **Ownership**: `model.mojo` is not owned by this lane. It is also imported
  by `alternate_boosting.mojo` (for `Model`), so the import must go the other
  way: `model.mojo` importing `alternate_boosting` closes a cycle. Resolve by
  moving `fit_boosting` out of `alternate_boosting.mojo` into `model.mojo` as
  this patch, and deleting it there.
- **Fallback**: leave `fit_boosting` in `alternate_boosting.mojo` as the
  wrapper it already is. It bins with the same `fit_bins` and returns the
  same `Model`, so nothing is lost except one entry point.
- **Serialization effect**: none.
- **Public API effect**: one new defaulted argument on two functions.
- **Dependency**: 4.1.
- **Validation, UNRUN**: `fit(..., boosting=AlternateBoostingParams.rf())`
  with `learning_rate=1.0` and `bagging_fraction=0.8` returns a `Model` whose
  `booster.base_score` is 0.0 and whose `booster.learning_rate` is
  `1 / n_estimators`.

### 4.3 `src/mojoboost/params.mojo`, accept `boosting=rf` in a parameter string

- **Target file / symbol**: `TrainConfig`, `parse_params`, `_MOJO_API_ONLY`,
  `SUPPORTED_KEYS`, `_validate`.
- **Signature**: add one field, `var boosting: Int`, defaulted to
  `BOOSTING_GBDT` in `TrainConfig.__init__`.
- **Edit**:
  1. remove `boosting boosting_type` from `_MOJO_API_ONLY`;
  2. add `boosting` to `SUPPORTED_KEYS`;
  3. add the branch, before the `_is_mojo_api_only` fallback:

```mojo
        elif key == "boosting" or key == "boosting_type" or key == "boost":
            # `goss` stays Mojo-API-only: selecting it means handing the
            # trainer a GossParams, which a parameter string cannot carry.
            if value == "gbdt" or value == "gbrt":
                config.boosting = BOOSTING_GBDT
            elif is_rf_boosting(value):
                config.boosting = BOOSTING_RF
            else:
                raise Error(
                    "boosting '", value,
                    "' is supported by the Mojo API only, through"
                    " AlternateBoostingParams",
                )
```

  4. in `_validate`, add the two rules a parameter string can check:

```mojo
    if config.boosting == BOOSTING_RF:
        check_rf_learning_rate(config.booster.learning_rate)
        if config.booster.tree.feature_fraction >= 1.0:
            raise Error(
                "boosting='rf' from a parameter string needs"
                " feature_fraction < 1: bagging_fraction is Mojo-API-only,"
                " so feature_fraction is the only randomization a parameter"
                " string can set"
            )
```

- **Call site**: `capi/mojoboost_capi.mojo` and `cli/` read `TrainConfig` and
  call `fit`; both must pass `AlternateBoostingParams` built from
  `config.boosting`, which is 4.2.
- **State flow**: string to `TrainConfig.boosting` to
  `AlternateBoostingParams` to `train_boosting`.
- **Errors**: `learning_rate` defaults to 0.1, so `boosting=rf` alone now
  fails with the learning-rate message. That is correct and matches
  LightGBM's refusal, but it makes the shortest working string
  `boosting=rf learning_rate=1.0 feature_fraction=0.7`. Say so in the error.
- **Ownership**: `params.mojo` is not owned by this lane.
- **Fallback**: leave `boosting` in `_MOJO_API_ONLY`. The message it already
  produces is accurate.
- **Serialization effect**: none.
- **Public API effect**: `parse_params` accepts two new keys and one new
  value. `params_names_mojo_api_only` must stop reporting `boosting` and
  start reporting `boosting=goss` and `boosting=dart` specifically, so the C
  ABI's status code stays right.
- **Dependency**: 4.1, 4.2.
- **Validation, UNRUN**: `parse_params("boosting=rf learning_rate=1.0
  feature_fraction=0.7")` succeeds; `parse_params("boosting=rf")` raises with
  `learning_rate` named; `parse_params("boosting=dart")` raises with
  `AlternateBoostingParams` named.

### 4.4 `bindings/_mojoboost.mojo`, carry the mode across the boundary

- **Target file / symbol**: a new `_parse_boosting`, and the five `fit`
  entry points that already call `_parse_goss`.
- **Signature**:

```mojo
def _parse_boosting(params: PythonObject) raises -> AlternateBoostingParams:
    """The boosting mode from the params dict. `boosting` arrives as an int
    code so the boundary carries no string conversion; the codes are
    `alternate_boosting`'s and are never serialized."""
    return AlternateBoostingParams.from_mode(Int(py=params["boosting_mode"]))
```

  `AlternateBoostingParams.from_mode` does not exist; it is one static method
  on that lane's struct, or `AlternateBoostingParams.named(...)` with a
  string if the extra conversion is acceptable.
- **Call site**: in `fit`, add `boosting=_parse_boosting(params)` to the
  `mojo_fit` call (4.2 gives `fit` the argument).
- **State flow**: Python dict to `AlternateBoostingParams` to `model.fit`.
- **Errors**: an unknown code raises from `boosting_name`, which is what
  `AlternateBoostingParams.validate` already calls.
- **Ownership**: `bindings/` is not owned by this lane.
- **Fallback**: omit, and the Python estimators keep refusing `rf` (4.5 then
  cannot be applied either).
- **Serialization effect**: none.
- **Public API effect**: none at the Python level until 4.5.
- **Dependency**: 4.1, 4.2.
- **Validation, UNRUN**: `pixi run -e pytest test-estimators` still passes
  with `boosting_mode` defaulted to 0 for every existing test.

### 4.5 `python/mojoboost/__init__.py`, accept `boosting='rf'`

- **Target file / symbol**: `_BOOSTING_TYPES`, `_Base._resolve_boosting`,
  and the params dict built at the end of the validation method.
- **Edit**:

```python
_BOOSTING_TYPES = ("gbdt", "goss", "rf")

_BOOSTING_CODES = {"gbdt": 0, "goss": 1, "dart": 2, "rf": 3}
```

  and in the validation method, beside the existing `goss` block:

```python
        if boosting == "rf":
            # A forest averages its trees and applies no shrinkage; LightGBM
            # forces shrinkage to 1 and ignores learning_rate, which means
            # the number the user set is not the number the model trains
            # with. See src/mojoboost/boosting_rf.mojo.
            if float(self.learning_rate) != 1.0:
                raise ValueError(
                    "boosting='rf' requires learning_rate=1.0: a random "
                    "forest averages its trees and applies no shrinkage"
                )
            randomized = (
                int(bagging_freq) > 0 and float(bagging_fraction) < 1.0
            ) or float(self.feature_fraction) < 1.0
            if not randomized:
                raise ValueError(
                    "boosting='rf' needs a source of per-tree randomness: "
                    "set bagging_fraction < 1 with bagging_freq > 0, or "
                    "feature_fraction < 1"
                )
```

  and in the returned dict: `"boosting_mode": _BOOSTING_CODES[boosting],`.
- **Call site**: `_resolve_boosting` already resolves the
  `boosting` / `boosting_type` alias pair and is the only place that does; no
  second copy of the rule.
- **State flow**: estimator attribute to params dict to `_parse_boosting`.
- **Errors**: the two above, raised at `fit` from the same method that
  already raises for GOSS plus bagging. Both mirror `check_rf_params` and
  `check_rf_learning_rate`, which raise again natively; the Python copies
  exist to name the estimator parameter rather than the Mojo one, which is
  what the GOSS block already does.
- **Ownership**: `python/` is not owned by this lane.
- **Fallback**: omit, and `boosting='rf'` keeps raising `unknown boosting`,
  which is honest.
- **Serialization effect**: none. A fitted rf model saves through the same
  `Booster.model_to_string`.
- **Public API effect**: `boosting='rf'` and `boosting_type='rf'` become
  accepted values on all three estimators. `best_iteration_` and the
  `start_iteration` / `num_iteration` prediction slice keep GBDT semantics
  and are wrong for a forest until 4.6; **do not apply 4.5 without 4.6 or
  without documenting the slice as unsupported for rf**.
- **Dependency**: 4.1, 4.2, 4.4.
- **Validation, UNRUN**: `MojoBoostRegressor(boosting="rf",
  learning_rate=1.0, bagging_fraction=0.8, bagging_freq=1,
  n_estimators=20).fit(X, y).predict(X)` runs, and its predictions are the
  mean of the 20 trees rather than their sum, which is checkable against
  `predict(..., start_iteration=0, num_iteration=20)`.

### 4.6 `src/mojoboost/boosting.mojo` and `serialize.mojo`, the averaging flag

This is the patch the bridge exists in place of, and the only one that makes
an rf model self-describing.

- **Target file / symbol**: `boosting.Booster`, one new field; `serialize`,
  format v5.
- **Signature**:

```mojo
struct Booster(Copyable, Movable):
    var trees: List[Tree]
    var base_score: Float64
    var learning_rate: Float64
    var objective: Int
    var monotone: MonotoneConstraints
    var average_output: Bool     # new, defaulted False in __init__
```

  and in `predict_raw_bins_range`, after the existing loop:

```mojo
        if self.average_output and not rng.is_empty():
            # LightGBM divides a sliced rf prediction by the slice length,
            # not by the ensemble size (GBDT::Predict, num_iteration_for_pred_).
            s /= Float64(rng.n_iterations())
```

  with `learning_rate` staying 1.0 for an averaged model and the base score
  staying inside the trees.
- **Call site**: `RfBooster.to_booster` sets `average_output=True` and
  `learning_rate=1.0` instead of `1 / T`; `is_forest` becomes
  `booster.average_output`; `train_rf_more` stops rewriting the rate.
- **State flow**: the flag travels with the model into every predictor.
- **Errors**: `train_more` must refuse an averaged ensemble outright, in
  `boosting.mojo`, rather than relying on `alternate_boosting` to route
  around it: `if booster.average_output: raise Error(...)`.
- **Ownership**: `boosting.mojo` and `serialize.mojo` are not owned by this
  lane, and `boosting.mojo` is the most contended file in the round.
- **Fallback**: the bridge, which is what ships today. It is exact for a
  full-model prediction and wrong only for slices.
- **Serialization effect**: **format v5**. One optional token after the
  monotone section, written only when the flag is set, so a GBDT model
  serializes to exactly the bytes it does today and v1 to v4 files load as
  `average_output=False`. `CURRENT_FORMAT_VERSION` in `serialize.mojo` and
  `MODEL_FORMAT_VERSION` in `model_dump.mojo` both move to 5.
- **Public API effect**: `model_dump` and `python/mojoboost/inspection.py`
  gain a field. `gpu_predict.flatten_booster` must carry the flag or refuse
  an averaged model; `contrib.mojo` needs the divisor too, or exact
  contributions on an rf model will be `T` times too large.
- **Dependency**: apply after 4.1 and before 4.5, or slices lie.
- **Validation, UNRUN**: save and reload a forest; `predict_raw_bins_range`
  over `[0, T)` equals `predict_raw_bins`, and over `[0, k)` equals the mean
  of the first `k` trees; a v4 file still loads and predicts identically.

### 4.7 `src/mojoboost/lgbm_model_io.mojo`, import a LightGBM forest

- **Target file / symbol**: the `average_output` rejection near the reader's
  header block (`if line == "average_output":`), and the module docstring
  line that says "mojoboost sums trees".
- **Edit**: replace the raise with a flag, and build the `Booster` with
  `learning_rate = 1 / num_iterations` and `base_score = 0.0` (the bridge),
  or with `average_output=True` once 4.6 exists.
- **Call site**: the same function that builds the `Booster` from the parsed
  trees.
- **State flow**: file flag to model.
- **Errors**: keep every other rejection. A LightGBM rf file with
  `num_tree_per_iteration > 1` is a multiclass forest and needs the
  multiclass reader path.
- **Ownership**: not owned by this lane, and modified by another lane during
  this session.
- **Fallback**: the current rejection, which names the construct.
- **Serialization effect**: none on writing until 4.6; note that the module
  docstring's exactness claim ("`learning_rate = 1.0` and `base_score = 0.0`
  round-trips bit-exactly") does **not** cover `1 / T`, so a re-dumped
  imported forest is exact only to a few units in the last place until 4.6
  makes the rate 1.0 again.
- **Public API effect**: `boosting='rf'` LightGBM models become loadable.
- **Dependency**: 4.6 for the exact round trip.
- **Validation, UNRUN**: a LightGBM file with `average_output` loads, and its
  prediction on one row equals the mean of its per-tree leaf values.

### 4.8 `src/mojoboost/alternate_boosting.mojo`, the three gaps

- **Target file / symbol**: `train_boosting`, `train_boosting_more`, and the
  module docstring.
- **Edits**, all inside that lane's file:
  1. **multiclass**: add `train_boosting_multiclass` dispatching
     `BOOSTING_RF` to a new adapter here, `train_rf_multiclass`, mirroring
     `train_rf`: `train_forest_multiclass(...)` then
     `RfMulticlassBooster.to_multiclass_booster()`. The algorithm is already
     written and tested by inspection only; nothing new is needed in
     `boosting_rf.mojo`.
  2. **validation sets**: add `train_boosting_with_valid` dispatching
     `BOOSTING_RF` to `train_forest_with_valid(...).to_booster()`. Unlike
     DART, truncating a forest to its best size is exact: the trees are
     independent, so the first `k` are the forest `n_estimators = k` would
     have grown. The docstring's reason for having no
     `train_dart_with_valid` does not apply to rf and should say so.
  3. **docstring**: line 72 attributes the learning-rate refusal to
     `check_rf_params`; it is `check_rf_learning_rate`.
- **Ownership**: that lane's file. This is a request, not an edit.
- **Fallback**: multiclass rf and rf early stopping stay reachable through
  `boosting_rf` directly.
- **Serialization effect**: none.
- **Public API effect**: none until 4.4 and 4.5.
- **Dependency**: none.
- **Validation, UNRUN**: a two-class softmax forest of 10 rounds has 20
  trees, and `to_multiclass_booster().learning_rate` is 0.1.

### 4.9 `docs/LIGHTGBM_PARITY.md` and `tools/check_parity.py`

- **Target file / symbol**: the `boosting_type` row in section 2, the v1
  scope paragraph that lists "DART and random-forest boosting" as deferred,
  and section 0's capability table.
- **Edit**: the row becomes

```
| `boosting_type` | partial | Accepted as `boosting` (LightGBM's native name) with `boosting_type` as an alias. `gbdt`, `goss`, and `rf`; `dart` is deferred, see section 7 |
```

  and a section 0 row is added:

```
| Random-forest boosting | deferred | yes | yes | no | no | no | n/a | n/a | `src/mojoboost/boosting_rf.mojo`, reached from `alternate_boosting.train_boosting`. No test file references it, `boosting='rf'` is rejected by the Python estimators and by `parse_params`, and no model file records that its trees are averaged |
```

- **Ownership**: not owned by this lane, and `check_parity.py` has its own CI
  job (`pixi run check-parity`).
- **Dependency**: apply with 4.5, not before; the parity row must not claim
  reachability the estimators do not have.
- **Validation, UNRUN**: `pixi run check-parity`.

---

## 5. Focused validation to run next, all UNRUN

Nothing below has been executed. In the order that finds the most for the
least, and with `tests/` unowned by this lane, so these are file requests.

### 5.1 `tests/test_rf.mojo`, the invariants that need no LightGBM

1. **Refusals.** `check_rf_params` raises for a forest with no randomizer,
   for two row samplers at once, and for `feature_fraction_bynode < 1` as the
   only source. `check_rf_objective` raises for `CUSTOM` and `LAMBDARANK`.
   `check_rf_learning_rate(0.1)` raises. `check_rf_init_score([0.0])` raises.
2. **The forest is not a boosted ensemble.** Train a forest and a GBDT
   ensemble on the same data with the same bagging and seed. Their trees must
   differ from round 1 onward, because the second round's gradients differ.
3. **Averaging.** `RfBooster.predict_raw_bins` equals the mean of
   `tree.predict_bins` over the trees, and `to_booster().predict_raw_bins`
   equals it to within a few units in the last place (exactly, when
   `n_estimators` is a power of two).
4. **Independence.** `train_forest` with `n_estimators=100` produces the same
   100 trees as `train_forest(50)` followed by `train_forest_more(50)`. This
   is the sharpest test in the file: it fails if any seeded draw reads a
   relative round index.
5. **Truncation.** `train_forest_with_valid` truncated to `k` trees predicts
   exactly what `train_forest` with `n_estimators=k` predicts.
6. **Renewal.** For `objective=L1` on data with a single split, each leaf's
   value is the median of the labels in it, not the median of the residuals.
   This is the check that the constant-residual renewal and the bias addition
   compose correctly.
7. **Bias placement.** A forest of one tree predicts what a single grown tree
   plus the base score predicts.
8. **`is_forest`.** True for `to_booster()` output with at least one tree,
   false for `boosting.train` output, false for an empty ensemble.
9. **Multiclass.** A `k`-class forest of `r` rounds holds `r * k` trees, and
   `predict_proba_bins` sums to 1.

### 5.2 Differential, against LightGBM

Not possible for tree identity: the RNG differs by design (section 9 of the
doc). What can be compared is the distribution and the schedule, which is
what `tools/check_parity.py` does for the parameter surface, plus one
end-to-end number: a forest's training loss on a fixed dataset should land
within sampling noise of LightGBM's for the same `bagging_fraction`,
`num_leaves`, and `num_iterations`. That is a benchmark, not a test.

### 5.3 What would falsify the design

- If `_renew_leaf_values` is ever changed to take a running score rather than
  a vector, rf renewal silently becomes GBDT renewal. 5.1.6 catches it.
- If `grow_tree` ever mutates its `grad` argument, every tree after the first
  in a forest is grown on corrupted gradients. Nothing catches that today
  except 5.1.4.
- If `Tree.value` ever stops holding internal node values, `_add_bias` biases
  only the leaves and contributions disagree with predictions.

### 5.4 Compilation

Nothing in this lane has been compiled. The likeliest compile failures, in
order: the underscore imports from `boosting.mojo` (precedented in
`boosting_sparse.mojo`, so low risk), `var tree: Tree` predeclared before a
branch in `_rf_rounds_multiclass`, and the `.copy()` calls on `TreeParams` and
`BaggingParams` in `RfParams.from_booster_params`.

### 5.5 Fusion requests inside `boosting.mojo`

Three duplications exist only because this lane owns one file. Each is a
one-line move:

1. `_base_score` to `base_score`, public, so `boosting_sparse`,
   `alternate_boosting`, and `boosting_rf` stop importing an underscore name.
   Three importers already do.
2. `Booster.response` to a free `objective_response(objective, raw)` that
   `Booster.response` then calls. `boosting_rf._objective_response` builds an
   empty `Booster` per call to reach it today.
3. `class_log_priors(labels, n_classes, weights)` lifted out of
   `train_multiclass`, which computes it inline.
   `boosting_rf._class_log_priors` is the second copy.
4. `_same_signs` made public. `boosting_rf._same_signs` is the second copy.

None changes behavior; all four are pure extractions.
