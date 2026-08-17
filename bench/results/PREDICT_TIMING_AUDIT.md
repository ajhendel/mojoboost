# Predict timing audit, 2026-08-17

**Subject.** The `predict s` column of the real-data report, as published for run
`20260817T185124Z-posthistfix`, scenario `dense_regression` on the real
YearPredictionMSD split, 51,630 held-out rows, 90 features, 100 trees, 10
threads, arms `mojotrees` / `mojotrees_catboost_mode` / `mojotrees_depthwise` /
`lightgbm` / `catboost` / `xgboost`.

**Verdict up front.** The column was not comparable across engines. One of six
figures was a cache read rather than a prediction. The adapter is fixed. **The
XGBoost predict numbers from that run, batch and row alike, are withdrawn, and
the column is not publishable for that run at all**, because the ranking it
carries was decided by the withdrawn figure.

## The rule this is written under

**A number whose magnitude is implausible for the work it claims to do is a
measurement bug until proven otherwise.** It is never first evidence that we,
or a peer, are three hundred times faster than a mature library at the same
task. The burden is on the number, and the way it discharges the burden is a
named mechanism plus a stored sample that shows the mechanism.

The second rule, which this audit exists to satisfy, is the protocol one, and
it says that a speed figure never travels without its run id, its shape, and
its arm set. A
`predict s` column that does not also say what each engine's predict call was
handed is a figure travelling without its shape.

And the process fact, recorded rather than softened. **This column was
published for one run before anybody checked it.** The implausibility was
visible in the published table itself, at a glance, without opening a record,
at 0.00015 s against LightGBM's 0.047 s for the same rows and the same trees.

## What was measured

| arm | published median | warmup sample | measured samples | ratio |
| --- | --- | --- | --- | --- |
| xgboost | 0.00015 s | 0.0093 s | 0.00018 / 0.00014 / 0.00013 s | **50x to 70x** |
| catboost | 0.0063 s | 0.0066 s | 0.0068 / 0.0063 / 0.0061 s | 1.0x |
| lightgbm | 0.047 s | 0.0486 s | 0.0495 / 0.0470 / 0.0462 s | 1.0x |
| mojotrees gpu | 0.093 s | 0.0948 s | 0.0918 / 0.0929 / 0.0919 s | 1.0x |
| mojotrees cpu | 0.098 s | 0.1085 s | 0.0954 / 0.0976 / 0.0993 s | 1.1x |

Source: `bench/real_data/results/20260817T185124Z-posthistfix/records/*.json`,
field `phases.predict_batch`, which stores the warmup call and the measured
calls separately for exactly this reason.

`cpu_s` settles it beyond the wall clock. XGBoost's warmup call spent 0.053 s of
CPU across ten threads; its measured calls spent **0.0004 s**. Four hundred
microseconds of CPU cannot visit the roughly five million tree nodes that
51,630 rows over 100 trees requires. No work happened.

## What each engine times, before and after

| engine | what the timed call is handed | what the call does | cached? |
| --- | --- | --- | --- |
| mojotrees | `test["X"]`, the harness's held-out matrix | `Booster.predict` bins the matrix, then walks the ensemble over row blocks into a per-call output buffer | no |
| lightgbm | `test["X"]` | `Booster.predict` marshals the array into `LGBM_BoosterPredictForMat`, which walks raw thresholds; no Dataset is built | no |
| catboost | `test_matrix`, the same held-out rows (a frame only where the scenario declares categoricals) | `CatBoost.predict` builds whatever internal container it needs, per call; the probability-column slice is inside the timing too | no |
| xgboost, BEFORE | a `DMatrix` built **once, above the loop** | `Booster.predict` on that DMatrix | **YES** |
| xgboost, AFTER | `test["X"]` | `_predict` builds a fresh `DMatrix` inside the timed call, then `Booster.predict` | no |

Three of the four were already measuring the same thing, being handed the
harness's own container, paying their own conversion inside the timing, and
carrying nothing between repeats. XGBoost was the exception on both counts at
once. It was the
only arm predicting from a library container built and paid for outside the
timing, and the only arm whose library caches the answer on that container.

## The XGBoost figure, settled

The figure is **(c), an artifact**. Not (a) work counted elsewhere and not (b)
a genuinely fast call. Evidence follows, in order of strength.

1. **XGBoost documents the cache, in the installed library.**
   `xgboost/core.py`, in `Booster.inplace_predict`'s docstring, reads *"Unlike
   :py:meth:`predict` method, inplace prediction does not cache the prediction
   result."* The contrast is the statement. `Booster.predict` caches, keyed on
   the DMatrix it was handed. Installed version 3.4.0.
2. **The adapter handed the same DMatrix to every repeat.**
   `bench/real_data/engines.py` built `dtest = self.module.DMatrix(test["X"])`
   above the loop and closed over it. `measure.repeat(fn, times, warmup=1)`
   calls `fn` once as warmup and then `times` more, so the warmup call filled
   the cache entry and every measured call read it back.
3. **The stored samples show exactly that shape and only on that arm.** Warmup
   0.0093 s then 0.00018 / 0.00014 / 0.00013 s, repeated across all three
   repeat records (`007`, `008`, `009`). Every other arm's warmup and measured
   samples agree within a few percent, which is what a call doing its work
   every time looks like.
4. **The arithmetic.** 0.0004 s of CPU for about 5 million node visits is
   roughly 0.08 ns per visit, which is under one cycle. It is not a prediction.

The honest XGBoost figure for that shape is somewhere at or above its warmup
sample, **about 0.009 s of scoring plus the DMatrix construction the other arms
pay the equivalent of**, and it is not yet measured. It has to be re-run.

`predict_row` on the same arm was distorted by the same mechanism but is not
visibly wrong (6e-05 s against LightGBM's 5e-05 s), because one row of scoring
is below the ctypes call overhead on every arm, which makes a cache read and a
real prediction look alike. It was fixed anyway. **A defect that is invisible
at one size is still a defect.**

## What changed

All of it in `bench/real_data/engines.py`, one file.

1. **`PREDICT_PHASE_RULE`**, a new module constant carrying the three-clause
   rule and the history that produced it. The rule has three clauses. The
   timed callable is handed the held-out matrix in the harness's own
   container. No state that could hold a previous answer survives between
   repeats. And the call returns a materialized score per held-out row.
2. **`_materialized_predictions(engine, out, n_rows)`**, a new helper called
   once per predict phase, **after** `measure.repeat` returns, so it adds
   nothing to any measured sample. It raises `EngineError` if the returned
   object is not an array or does not have one row per held-out row. This
   turns "no engine returns a lazy handle" from four separate assumptions
   about four libraries into a checked property of the column. Wired into all
   five batch predict sites, which are mojotrees dense, mojotrees sparse,
   lightgbm, catboost, and xgboost.
3. **`XGBoostEngine._predict` now takes the matrix and builds the DMatrix
   itself**, inside the timed call, and the hoisted `dtest` and `drow` are
   gone. `predict_batch` and `predict_row` both go through it.
4. **Notes on every arm's record** stating what its predict phase contains, so
   the record answers the question without a reader having to open the adapter.

Nothing was made slower to even things up. XGBoost still uses its own fastest
correct path for the values it produces; it just pays for the conversion the
other three pay for, and it is no longer allowed to answer from a cache.

## Why `predict` on a fresh DMatrix, and not `inplace_predict`

`inplace_predict` is the other way to get an uncached call, and it is a
different entry point. Every `predictions_sha256` this arm has recorded came
from `Booster.predict` on a DMatrix. **A fix to a timing has no business moving
the values whose accuracy is being compared**, and a bit-level change in this
arm's predictions would have propagated into the accuracy comparison and the
determinism digests. A fresh DMatrix defeats the cache while leaving the
numeric path alone. The cache cannot survive it, because the previous DMatrix
is dropped when the timed call returns and XGBoost prunes expired cache entries
on the next lookup.

## What a reader must know to trust the column, once it is re-run

- **Every arm's figure includes that library's own conversion of the held-out
  matrix.** That is the definition of the column, and it is the definition
  precisely because the alternative, pre-converting per engine, is what broke.
- **Our figure includes binning the held-out matrix; LightGBM's does not**,
  because LightGBM predicts against raw thresholds. This is a real
  architectural difference in the predict path and not a harness artifact, but
  it is not a like-for-like inner loop, and our number should not be read as
  "our tree walk is 2x LightGBM's tree walk".
- **The `mojotrees gpu` rows' predict figures are CPU predictions.** The
  binding calls `Model.predict_batch` with an explicit `CPU_DEVICE`
  (`bindings/_mojotrees.mojo`), so the `cpu` and `gpu` rows measuring within
  noise of each other, 0.098 s and 0.093 s, is the expected result and not a
  finding about the accelerator.
- **`path: "dmatrix"` and `path: "pool"` in a record describe the TRAINING
  ingest**, not the predict phase. CatBoost does not predict from its Pool
  here.
- **`scenarios.PHASE_SHAPE` says nothing about the predict phases.** It
  documents `ingest`, `binning`, `train` and `e2e` per engine, which is why a
  predict-phase asymmetry had no place to be declared and no place to be
  caught. Adding `predict_batch` and `predict_row` to `PHASE_SHAPE` is the
  structural fix and it is out of this lane's edit scope. It is an OPEN item.

## Open items, outside this lane's edit scope

1. **`scenarios.PHASE_SHAPE` needs predict-phase entries per engine**, and
   `selfcheck.py` already asserts every engine appears in `PHASE_SHAPE`, so the
   check that would have caught this is one key away from existing.
2. **`bench/apple/suite.py` builds an `xgb.QuantileDMatrix` and an
   `xgb.DMatrix`** and should be checked for the same reuse pattern before any
   figure from it travels. Not read in this lane.
3. **A peer is rewriting `src/mojotrees/predict.mojo` in the working tree**
   (554 lines added, a raw unbinned predict path, `raw_predict_enabled`,
   `predict_raw_batch`, `oblivious_raw_plan`). If that lands, our predict phase
   stops binning the held-out matrix and this column changes meaning. Any
   published `predict s` figure must name the commit, not just the run id.
4. **The run must be repeated** for the column to be publishable. The train,
   binning and accuracy columns of `20260817T185124Z-posthistfix` are
   untouched by this and stand.
