# Task 15 handoff: cross-validation and the CVBooster layer

Files added by this lane, and the only ones it touched:

- `python/mojoboost/cv.py`
- `python/tests/parallel/test_cv.py`
- `handoffs/task15_cv.md` (this file)

Nothing central or shared was edited. Everything below is a request on
another lane, in the order it has to land.

## 1. Package exports (owner: `python/mojoboost/__init__.py`)

`cv.py` stands on its own today, so `from mojoboost.cv import cv, CVBooster`
already works. LightGBM users reach for `mojoboost.cv(...)`, so the two names
belong at the top level next to `train`.

Add to the import block, after `from .basic import Booster, Dataset, train`:

```python
from .cv import CVBooster, cv
```

and add `"CVBooster"` and `"cv"` to `__all__`, alphabetically among the
existing entries (between `"Booster"` and `"Dataset"` for `CVBooster`, and
next to `"train"` for `cv`).

**Import-order note.** `cv.py` imports `.basic` at module scope and reaches
back into the package (`from . import _metric_specs`, `from . import callback
as _callback`) only from inside function bodies, which is the pattern
`basic.py` already uses. So the line above is safe wherever it sits after
`basic` is imported. Do not move those inner imports to module scope; that
does create a cycle.

**Name clash.** The package would then hold both a submodule
`mojoboost.cv` and a function `mojoboost.cv`. The function wins in the
package namespace, which is what a caller wants, and
`import mojoboost.cv as cv_module` still reaches the module. LightGBM has
exactly this shape (`lightgbm.cv` the function, `lightgbm.basic` etc. the
modules) so it is familiar, but it is worth a line in the module docstring.

## 2. `Dataset` / `Booster` API changes wanted

None are required: `cv.py` runs against the interface as it stands today.
Each item below removes a documented limitation, in priority order.

### 2.1 `Booster.update()` for LambdaRank (blocks per-round ranking cv)

`Booster._require_trainable` raises `NotImplementedError` for
`_eval.RANKING`, so a ranking fold cannot be grown a round at a time. `cv()`
therefore trains each ranking fold once at the full round count
(`_RetrainFold`), reports a **single history entry**
(`results["iterations"] == [num_boost_round]`), and **refuses**
`early_stopping_rounds=` and `callbacks=` for ranking rather than accepting
them and quietly ignoring them.

What is needed, in `src/mojoboost/boosting.mojo` and the binding:

- a `train_ranker_more`-shaped entry point matching `train_more` /
  `train_multiclass_more`, taking the fitted ensemble, the dataset (whose
  `group` the trainer already reads), and the round count;
- a `booster_update_ranker` binding beside `booster_update` and
  `booster_update_multiclass` in `bindings/_mojoboost.mojo`;
- `Booster._require_trainable` losing the ranking branch, and `update()`
  dispatching to it on `_task == _eval.RANKING`.

The hard part is the one `basic.py` already names: LambdaRank gradients are
computed within a query from state the fitted ensemble does not carry. It is
recoverable from the raw scores plus the group vector, which is what
`train_more` recomputes for the other objectives, so this is a matter of
recomputing per-query ranks rather than of new state.

Once it lands, `cv.py` needs one change: drop the `incremental` branch and
always build `_IncrementalFold`. `_RetrainFold` and the two
`NotImplementedError`s go with it, and the ranking test that asserts
`results["iterations"] == [5]` becomes `[1, 2, 3, 4, 5]`.

### 2.2 A training-state handle, so a per-round history is not quadratic

`Booster.update()` recomputes the training raw scores from the model on
every call ("one call of `update(60)` costs less than 60 calls of
`update()`"). `cv()` needs a value per round, so it makes R calls of
`update(1)` per fold, and the rescoring pass grows with the tree count: the
history costs O(R^2) predictions per fold on top of O(R) training. The
estimators do not pay this, because `train_with_callbacks` keeps the score
buffer inside the Mojo loop.

Two shapes would fix it, either is fine:

- **(a) A resumable handle.** `Booster.update()` keeps the raw-score buffer
  between calls and updates it with each new tree instead of recomputing it,
  invalidating it whenever the model is replaced or loaded. Purely internal;
  no Python signature changes; `cv.py` needs no change at all.
- **(b) Expose the callback loop on `Booster`.** A
  `Booster.grow(rounds, on_iteration=...)` that runs
  `train_with_callbacks`'s loop and calls back once per round. `cv.py` would
  then need a third `FoldModel` implementation that drives all folds through
  one hook, which is more work here but is the only shape that also gives
  the folds a *synchronized* stop inside the Mojo loop.

(a) is the smaller change and keeps this layer as it is. Prefer it.

### 2.3 Multi-metric `Booster.eval`

`Booster.eval(data, name, metric=None)` scores one metric and predicts the
whole dataset to do it. `cv()` scores M metrics on 2 sides of F folds every
round, so an M-metric run predicts M times where once would do.

Wanted: `metric=` accepting a sequence, returning one tuple per metric from
a single prediction pass, with the scalar form unchanged. That is a
backwards-compatible widening of an existing argument, and it is the only
place `cv.py` pays for extra metrics.

`cv.py` would then replace the per-metric loop in `_score` with one call.

### 2.4 `Booster.reset_parameter` (unblocks `reset_parameter()` in cv)

A `reset_parameter()` callback changes hyperparameters between rounds. The
fold boosters have no way to be told, so `CVBooster.reset_parameter` raises
`NotImplementedError` rather than letting a schedule look like it ran.

`_Config.binding_params` rebuilds the parameter dict from `self.base` on
every call, so the plumbing is nearly there: what is missing is a public
`Booster.reset_parameter(mapping)` that writes the canonical names in
`mojoboost.callback.RESETTABLE` onto `self._config.base` (through the same
alias resolution `_Base._resolve_alias` uses) and is honored by the next
`update()`. The validation already exists in
`mojoboost.callback.canonical_reset_key`.

`cv.py` would then have `CVBooster.reset_parameter` forward to every fold
instead of raising.

### 2.5 `show_stdv` in `log_evaluation` (owner: `python/mojoboost/callback.py`)

`log_evaluation(period, show_stdv)` documents `show_stdv` as having no
effect because "there is no `cv` here yet". There is now, and the deviation
it wants to format is in the history.

LightGBM's `cv` hands callbacks a 5-tuple
`(data_name, metric_name, mean, is_higher_better, stdv)`; this package's
`log_evaluation` unpacks exactly four, so `cv.py` passes 4-tuples
`("cv_agg", "valid l2", mean, False)` and the deviation is reachable only
from the returned history.

Wanted, in `callback.py`:

- `log_evaluation`'s formatter tolerating both widths (LightGBM's
  `_format_eval_result` does the same test), printing
  `cv_agg's valid l2: 0.31 + 0.02` when a fifth element is present and
  `show_stdv` is set;
- the same tolerance in `record_evaluation`, which currently unpacks four
  and would raise on a 5-tuple.

Once both accept five, `cv.py` appends the standard deviation to each
`env_results` entry in `cv()` (one line, at the `("cv_agg", key, ...)`
comprehension) and the `show_stdv` note in its module docstring goes away.

### 2.6 `init_model` across differently binned datasets (found by running it)

LightGBM's `cv(init_model=...)` starts every fold from an existing model.
That cannot work here, and the reason only surfaced when the suite ran:

```
Exception: continued training needs the dataset the model was trained on:
this one is binned differently
```

raised by `_mojoboost.booster_update` (through `Booster.update`, through
`basic._continue`). A cv fold bins itself over its own rows, by design, so
its bin edges never match the ones the passed-in model was trained with and
**every** fold is refused. `cv()` now refuses `init_model` up front with that
reason instead of letting the bindings say it once per fold.

To lift it, continued training has to stop requiring bin-identical data.
`Booster.predict` already scores arbitrary rows through the model's own
mapper, and `basic.py`'s own docstring leans on that ("mojoboost predicts a
validation set through the model's own mapper"), so the raw scores
`train_more` needs can be produced for a differently binned dataset. What is
needed is for the trainer to take the starting raw scores through the
model's mapper rather than assert that the two binnings agree:

- in `bindings/_mojoboost.mojo`, `booster_update` (and
  `booster_update_multiclass`) drop the binning-equality guard and score the
  new dataset through the model's mapper to seed the raw scores;
- keep a guard on feature count and on the categorical declaration, which
  are real mismatches; binning alone is not.

`cv.py` would then delete the `init_model` refusal and the `_IncrementalFold`
/ `_RetrainFold` constructors already thread the argument through, so nothing
else here changes.

### 2.7 Sparse folds (low priority)

`Dataset` takes dense data only, so `_take_rows` rejects a sparse matrix
with a message saying so. If `Dataset` grows sparse support, `_take_rows`
needs a CSR row-slice branch and nothing else.

## 3. What this lane guarantees today

Behavior worth knowing about when reviewing or documenting it:

- **No preprocessing leakage.** Folds are built from the *raw* matrix and
  the raw columns, never from a constructed `Dataset`, so each fold bins
  itself over its own rows. `fpreproc(dtrain, dvalid, params)` is called
  after the split, once per fold. A fold whose training rows intersect its
  own held-out rows is rejected, including one the caller passed in.
- **Folds**: explicit `(train_index, test_index)` pairs, scikit-learn
  splitters (anything with `.split`, called as `split(X, y, groups)`),
  deterministic shuffled K-fold seeded by `random.Random(seed)` (no numpy
  needed), label-stratified folds for `binary` and `multiclass`, and
  whole-query folds for `lambdarank`. A query straddling a ranking fold is
  reported, not trained on.
- **Metrics**: everything `eval_metric=` accepts (names, callables,
  `(name, func, higher, use_for_early_stopping)` tuples, dicts, lists), plus
  LightGBM's separate `feval=`, which composes with `metrics=`. Built-ins go
  through `Booster.eval`, so a cv number is the number `eval()` reports.
- **Early stopping** is a consensus: the rule runs on the across-fold mean,
  so the folds stop together, the history is truncated to the winning round,
  and `CVBooster.best_iteration` records it.
- **`results["iterations"]`** is this layer's addition: the round number each
  history entry belongs to. It exists because §2.1 means the length of a
  history list is not always the round count.

## 4. Suggested follow-up lanes (not started here)

- `docs/LIGHTGBM_PARITY.md` needs a `cv` / `CVBooster` row, and
  `tools/check_parity.py` may need to learn the new names. That file is
  shared, so this lane did not touch it.
- The `train()` docstring in `basic.py` says "there is no per-round history
  and no early stopping here yet: those live on the estimators' `fit()`".
  With `cv()` in, that sentence wants "or on `cv()`".
- `bench/` has no cv benchmark. §2.2's cost is the thing to measure, and the
  fix is worth measuring against.
