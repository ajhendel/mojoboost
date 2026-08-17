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

**SUPERSEDED, in part.** Every CatBoost figure here was taken against the
arm identified as `cb-default`, a hand-written dict of what someone believed
CatBoost's defaults to be. That arm has since been rebuilt as `cb-shipped`,
taking its values from CatBoost's own `get_all_params()` read-back, and
`check_catboost_arm` now diffs the two resolved dicts key by key and fails on
any matchable difference. **Read a CatBoost number here as `cb-default` and do
not compare it with a `cb-shipped` number.** The CatBoost-mode arm is separately
superseded twice over: it gained `random_strength` after this run, and lost the
`max_bin` and (temporarily) `random_strength` keys during it -- see the block
notes.

For the record, the read-back for `dense_regression/catboost/cpu` in this run's
own artifact confirms the parameters were what the harness intended:
`learning_rate 0.10000000149011612` (0.1 in float32, so CatBoost's automatic
learning rate never fired), `iterations 100`, `depth 6`, `l2_leaf_reg 3`,
`bootstrap_type MVS`, `subsample 0.8`, `random_strength 1`, `score_function
Cosine`, `boosting_type Plain`, `border_count 254`. The last two independently
confirm two things read from CatBoost's source earlier the same day: Plain at
our tier, and 254 thresholds against our 255 bins being the same granularity.

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

**That WARN was characterized after the run, from the saved prediction arrays,
and it is not a near-tie divergence.** Nothing was retrained; this is arithmetic
over artifacts already on disk.

| | value |
| --- | --- |
| test rows | 51,630 |
| rows differing by more than 1e-06 | **51,630 -- all of them, 100.0000%** |
| rows differing by more than 0.1 | 45,424 (87.98%) |
| rows differing by more than 1.0 | 8,361 (16.19%) |
| median absolute difference | 0.4601 |
| mean absolute difference | 0.5753 |
| CPU prediction range | 1976.30 to 2009.24 |

**A near-tie divergence moves the rows a differently-routed split sends
elsewhere. This moves every row in the test set.** The label is a year, so
predictions live near 2000, and the median disagreement is 0.46 of a year.

Normalizing removes the part of the gap that is just target scale and leaves
the part that is real. Against each block's own RMSE, the largest CPU/GPU
disagreement is **1.06 (Block B) against 1.04e-05 (Block A)** -- still five
orders of magnitude, so the effect survives normalization and is not an
artifact of comparing a target near 2000 with one near 0.3.

**RESOLVED, 2026-08-16, by a three-arm interleaved window. The divergence is
NOT path-linked.** The paragraph that stood here said the decision row's GPU
number was "not shown to be wrong and not shown to be right". That is now
superseded by a measurement.

The same shape was run three ways in one window -- CPU, GPU under AUTO (which
the split-policy threshold sends to the host scan), and GPU forced onto the
resident device plane:

| comparison | max | median | rows differing |
| --- | --- | --- | --- |
| GPU AUTO vs CPU | 9.6719 | 0.4601 | 100.00% |
| GPU forced device vs CPU | 9.6719 | 0.4601 | 100.00% |
| **GPU AUTO vs GPU forced device** | **0.0000** | **0.0000** | **0.00%** |

**Two entirely different code paths -- one issuing 15,100 dispatches with 3,100
host histogram downloads, the other a resident plane issuing 100 transfers --
produce bit-identical predictions, and both miss the CPU by exactly the same
amount.** So the divergence lives in what they share: the fixed-point Int32
histogram under the same scales. It is not the split search, not the launch
structure, and not the near-tie resolution the WARN was recorded under.

**What that means for the decision row.** The mechanism is shared by every GPU
fit, so it is present in Block A too -- and Block A's own check measures it
there at 3.19e-06. The same mechanism therefore produces 1.04e-05 relative
error on well-conditioned synthetic data and 1.06 relative on 90 correlated
audio features. **Block A's GPU number stands, and its risk is characterized
rather than unbounded**, which is a stronger statement than the one it replaces.
The open question is no longer "is the decision row valid" but "what data
conditions amplify fixed-point histogram error by five orders", and that is a
bounded question about one component.

Incidentally, two device paths agreeing to the bit across entirely different
launch structures is a strong correctness signal for both.

## A cross-unit exposure that applies to every table above

Round-interleaving protects a comparison when the arms share hardware: drift
lands on all of them and surfaces as spread. **It does not protect a CPU arm
against a GPU arm**, because two units do not respond to a thermal window the
same way. This is why the protocol reports `canary_cpu_ratio` and
`canary_gpu_ratio` separately and never averages them -- Session III measured
the CPU degrading 2.2x in a window where the GPU degraded 1.5x.

This run's canary moved the CPU +7.7 percent and the GPU +2.1 percent, a
**differential of about 5.6 percent between the two units**. Every
mojotrees-GPU-against-LightGBM comparison here is cross-unit and carries it:

- **Block A's headline, 1.31x, survives comfortably**: a 5.6 percent
  differential against a 31 percent margin.
- **Block B's loss, 1.30x, likewise survives** and is not an artifact.
- **A margin near or below ~10 percent in a cross-unit comparison in this
  window would not be resolvable at all**, and one was found the same day: a
  follow-up window measured the forced device plane at 2.251 s against
  LightGBM's 2.510 s on this dataset -- 1.115x with disjoint ranges -- in a
  window whose canary moved the CPU +7.5 percent and the GPU -4.5 percent, in
  opposite directions, for a combined ~12 percent against an 11.5 percent
  margin. **That result is not reported as a win here, because the correction
  dissolves it rather than shrinking it.** It remains unestablished.

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

**CAUSE FOUND, 2026-08-16, from LightGBM's source: we are not running the same
number of categorical bins, and `max_bin` is not the limit it looks like.**
`LightGBM/src/io/bin.cpp:461` reads `while (used_cnt < cut_cnt || num_bin_ <
max_bin)` -- a **disjunction**, with `cut_cnt` set at 99 percent row coverage.
So `max_bin` is a **lower** bound on categorical bins, not an upper one, and
LightGBM keeps taking categories until coverage is satisfied however many that
is. LightGBM's own loader documentation concedes the point in passing, saying
that `max_bin` "may be ignored with a large number of categories".

**The bin counts here were an estimate and are now MEASURED, 2026-08-17. The
estimate was wrong in three ways and the correction makes the cause stronger,
not weaker.** What stood here said LightGBM "runs roughly 991 and 19,801
categorical bins against our 254", which named two columns and guessed both
counts. Every record in this harness carries the per-feature bin counts in
`model.bins.counts`, so this never needed estimating. Read off
`bench/real_data/results/20260817T102725Z-cat1m/records.json`, on the five
categorical columns at cardinalities 8, 64, 1000, 20,000 and 200,000:

| arm | bins on the five categorical columns |
|---|---|
| mojotrees | 9, 65, **255, 255, 255** |
| LightGBM | 9, 65, **989, 19,433, 15,952** |
| CatBoost | 1, 1, 1, 1, 1 (each becomes a CTR numeric column) |

So **three** of our columns are truncated, not two; the counts are 989 and
19,433 rather than 991 and 19,801; and the 200,000-level column gets **15,952**
bins from LightGBM, not "roughly 100,000" as `bench/real_data/thresholds.json`
still says. That last number matters for the remedy, because LightGBM never exceeds
`UInt16` on this scenario either, so an argument against widening our bin index
on the ground that 200,000 exceeds `UInt16` is arguing against a target nobody
is at. In row-coverage terms we resolve 26.5, 1.8 and 10.8 percent of rows on
the three wide columns where LightGBM resolves 98.9, 98.1 and 44.5 percent.

**And the size of the cause is now computed rather than asserted.** Because this
scenario's target is an explicit sum of per-level effects, an oracle can be
built for any binning. Give every resolved level its exact effect, give every
dropped level the count-weighted mean of the pool it fell into, keep everything
else perfect. Fitted on the train split, scored on the test split, with this
harness's own `quality.average_precision`:

| | average precision |
|---|---|
| oracle, full latent | 0.563319 |
| oracle at LightGBM's bin counts | **0.552770** |
| oracle at our 254 bins | **0.442548** |
| oracle with the three wide columns contributing nothing | 0.419854 |
| measured, LightGBM | 0.479591 |
| measured, mojotrees | 0.421925 |

**The bin ceiling costs 19.9 percent of oracle average precision where the
measured gap is 12.02 percent, so the named cause more than fully accounts for
the failure and nothing else has to be invoked.** Full working in
`docs/design/ACCURACY_GAP.md` section 7.

**That changes what this row means.** It is not "our categorical accuracy is
worse"; it is "we resolved 254 categories where the comparator resolved
thousands, and lost 12 percent of average precision doing it". A named,
measured, and fixable cause, with a second contributing defect already
identified beside it: CTRs computed from the binned bucket rather than from raw
category codes, which is why a lane measured our CTRs as null in both
directions. The speed win in this row was partly bought with resolution, and the
next measurement of it should not be read against this one.

**One caution on the CTR remedy, added 2026-08-17 and measured.** The raw-code
CTR source landed and this row did not move by a single bit, because our shipped
`lossguide` default sends `ctr="off"` on every policy but `symmetrictree`
(`python/mojotrees/sklearn.py`), so the scenario never called the fixed path.
More importantly, **CatBoost runs CTRs here and scores 0.440563 average
precision, which is 8.1 percent behind LightGBM and barely above our own
254-bin oracle of 0.442548.** A CTR resolves every level, but as a noisy
one-dimensional statistic in 255 numeric bins; thousands of set-splittable
categorical bins resolve fewer levels as a partition the tree can cut
arbitrarily, and on this generator the second is worth more. **So a CTR default
is unlikely to clear the 10 percent gate on its own.** Widening the categorical
bin ceiling is the fix for this row. `docs/design/ACCURACY_GAP.md` section 7.4
ranks the three candidates.

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
