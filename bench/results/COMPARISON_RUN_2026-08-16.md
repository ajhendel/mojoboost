# The single comparison run, 2026-08-16

Taken on `9de2ad4` (harness fixes at `dcc99ee`), Apple M4, 10 threads, five
repeats, arms round-interleaved. Rules registered in advance in
`COMPARISON_RUN_PLAN.md`; nothing below was decided after the numbers existed.

**Read this first, in order.**

1. **The machine was on battery.** `report.py` says so on every block: "This
   machine was on battery. Timings below are not comparable with mains-powered
   runs." That is a provenance fact about all three tables and it was not
   noticed until the tables were rendered.
2. **No canary baseline exists, and the run straddles a CPU shift.** Details in
   the canary section. The CPU probe was 7.7 percent slower after the run than
   before, which exceeds the protocol's own 5 percent straddle bar. The GPU
   probe moved 2.1 percent.
3. **Five repeats, not twelve**, because the canary refused the window. Ruled in
   advance.
4. Accuracy sits beside every speed number, as required. Two accuracy gates
   failed and both are named.

Provenance: `stale_sources` 0, extension sha256
`028670ccea6426e9b05fe11060d78099b4416e5ccdc914d52ec73573c34d4aec`, package
precompile exit 0 with zero errors, `mojo 1.0.0 (ed45d567)`, `lightgbm 4.7.0`,
`catboost` per the pinned bench environment.

---

## Block A -- the decision row. Synthetic, 799,110 train rows x 100 features

`dense_regression --tier large --variant synthetic`, primary metric RMSE.
**verify.py: 19 pass, 0 fail, 0 warn, 3 declared skips.**

| arm | train s, median [min, max] | RMSE | vs LightGBM |
| --- | --- | --- | --- |
| **mojotrees GPU** | **3.619 [3.580, 3.701]** | 0.307782 | **1.31x faster, 1.04% better** |
| CatBoost defaults | 3.755 [3.139, 4.428] | 0.305335 | 1.26x faster |
| LightGBM stock+det | 4.749 [4.508, 5.698] | 0.311028 | -- |
| mojotrees CPU | 10.077 [9.066, 11.162] | 0.307782 | 2.12x slower |
| us in CatBoost's shape, CPU | 14.427 [12.581, 15.373] | 0.303973 | 3.04x slower |

**mojotrees on the GPU is faster than the comparator and more accurate than it,
on the decision row.** The margin over LightGBM is resolved rather than
inferred: our slowest repeat (3.701 s) is faster than LightGBM's fastest
(4.508 s), so no overlap and the spread does not have to be argued about.
`verify.py`'s differential gate agrees on the accuracy side, RMSE better by
1.043 percent and MAE better by 0.590 percent.

**Against CatBoost the result is INDISTINGUISHABLE and must not be read as a
win.** CatBoost's range [3.139, 4.428] contains our median. Medians differ by
3.6 percent, which is below the smallest effect this repository has ever
resolved.

Also on this block: all five repeats bit-identical for every arm including the
GPU; the GPU arm's `backend_proof` names `train_gpu` with 100 device transfers
and 200 host synchronizations; GPU and CPU prediction digests **differ** while
`max |gpu - cpu|` is 3.19e-06 against a 0.001 limit, so the matching RMSE is
rounding and not a silent CPU fallback.

## Block B -- real data, 463,715 train rows x 90 features

`dense_regression --tier large --variant auto` -> UCI YearPredictionMSD,
pinned and checksum-verified. **verify.py: 42 pass, 1 fail, 1 warn, 3 skips.**

| arm | train s, median [min, max] | RMSE |
| --- | --- | --- |
| CatBoost defaults | 1.854 [1.796, 2.148] | 9.23102 |
| LightGBM stock+det | 2.572 [2.539, 2.600] | 9.10086 |
| mojotrees GPU | 3.332 [3.241, 3.359] | 9.10607 |
| mojotrees CPU | 4.975 [4.783, 5.079] | 9.10383 |
| us in CatBoost's shape, CPU | 7.537 [7.382, 7.878] | 9.21605 |

**We lose here, resolved and not marginal.** LightGBM is 1.30x faster than our
best backend and the ranges do not overlap. Accuracy is a tie: 9.10383 against
9.10086, a gap of 0.03 percent.

**FAIL, and it is CatBoost's**: `baseline dense_regression/catboost/cpu`, RMSE
9.23102 against a 10.8525 baseline, improvement 0.1494 where 0.15 is required.
CatBoost misses the harness's learned-anything floor on this dataset by 0.0006.

**WARN**: `device_agreement`, `max |gpu - cpu| = 9.67` against a 0.001 limit,
with the primary metric agreeing. Recorded as consistent with the device split
search's documented near-tie divergence. Worth contrasting with Block A, where
the same check returned 3.19e-06 on synthetic data of a similar size: same
code, four million times the divergence.

## Block C -- high-cardinality categorical, 799,110 train rows x 15 features

`high_cardinality_categorical --tier standard`, primary metric AUC.
**verify.py: 18 pass, 1 fail, 0 warn, 3 skips.**

| arm | train s, median [min, max] | AUC | avg precision |
| --- | --- | --- | --- |
| **mojotrees CPU** | **2.437 [2.334, 2.506]** | 0.841028 | 0.421925 |
| CatBoost defaults | 3.686 [3.649, 3.803] | 0.844852 | -- |
| LightGBM stock+det | 5.621 [5.609, 5.651] | 0.865957 | 0.479591 |

**The largest speed win in the run and the only failed accuracy gate of ours.**
2.31x faster than LightGBM and 1.51x faster than CatBoost, with no range
overlap against either, so both are resolved.

**FAIL**: `differential .../average_precision`, 0.421925 against LightGBM's
0.479591, worse by 12.02 percent where the limit is 10 percent. AUC passes at a
2.49 percent gap against a 5 percent limit. **The two metrics disagree about
whether this row is acceptable, and the stricter one is the one that fails**, so
this row is a speed win bought with accuracy and must be published as one.

The CatBoost-mode arm has **no row here**: `grow_policy=oblivious` is
implemented for numerical thresholds only, and a level of an oblivious tree
shares one split while a categorical is searched as per-node category
partitions. Declared in `MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT`. CatBoost
does not hit this because a CTR turns the categorical numeric first.

---

## The canary

**No baseline was recorded, and that is the correct output rather than a gap.**
Both readings tripped `calibration_warning` -- the repeats disagreed by more
than 3 percent, so the instrument declined to be recorded, twice.

| | cpu_ms (min) | cpu_spread | gpu_ms (min) | gpu_spread |
| --- | --- | --- | --- | --- |
| before the run | 198.861 | 6.6% | 225.266 | 7.2% |
| after the run | 214.125 | 9.8% | 230.009 | 1.7% |

**The CPU probe moved 7.7 percent across the run and the GPU probe moved 2.1
percent.** The protocol's straddle threshold is 5 percent of the smaller
reading, so **the CPU side straddles and the GPU side does not**. The rule that
would discard a straddling run takes effect "the moment a baseline exists", and
none does, so this is reported rather than applied. It is the reader's to
weigh.

Two facts, offered as facts and not as interpretation. The two pre-run minima,
taken 45 minutes apart, agree to 0.5 percent (199.86 and 198.86), so the floor
this machine delivers is reproducible even where the spread is not. And the
round-interleaved arm order means a drift over a block lands on every arm
rather than on the arm that happens to sort last -- which is why the between-arm
comparisons above survive a shift the absolute seconds do not.

A1 capture: no mojo, no pixi, no build, no lane at any point. WindowServer, VS
Code, Chrome and three `claude` processes throughout; load 1.8 before, 4.67
after. **The interference is the desktop, not the campaign.**

## What this run does not say

- **Nothing about CatBoost's Ordered boosting.** Confirmed from source rather
  than assumed: `UpdateBoostingTypeOption` hard-sets Plain when
  `(learnSampleCount >= 50000 || IterationCount < 500)`, and at 100 estimators
  the second clause fires at every tier. No arm passed `boosting_type=Ordered`.
- **Nothing about `random_strength`.** The CatBoost-mode arm cannot carry it
  through this harness: its per-tree scale is computed only by the dense `fit`
  entry point, and the harness trains through `train(params, Dataset)`.
- **Nothing about mains-powered performance.** See the battery note.
- **Nothing about `multiclass` or `sparse_highdim`**, which are not in this run.
- **"Us in CatBoost's shape" is not CatBoost.** Twelve keys: symmetric trees at
  depth 6, Cosine scoring, MVS at 0.8, and CatBoost's regularization. No
  `random_strength`, no CTR. `CATBOOST_UNMATCHABLE` carries the rest.

## The headline, in one sentence each

**We beat LightGBM on the decision row, on the GPU, on speed and accuracy at
once -- and lose to it on real data at two thirds the rows.** The categorical
row is the fastest thing in the run and fails an accuracy gate. The CPU backend
is 2.1x behind the comparator at 1M x 100 and is the clearest single target.
