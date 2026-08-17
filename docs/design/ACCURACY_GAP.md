# Why CatBoost is more accurate than our leaf-wise default, priced

Written 2026-08-17, sections 1 and 2 re-measured the same day with
`bench/real_data/decompose.py`, which replaced the hand arithmetic they were
first built on. This is a reading and arithmetic lane. **Nothing was built,
nothing was compiled, nothing was fitted, no benchmark was run.** Every number
below is either read off a stored artifact in this repository, read off the
source, or computed by arithmetic over stored predictions and the generator,
with the method shown.

---

## 0. Verified, computed, estimated. The three words are not interchangeable

**Measured** means a number that exists in a stored artifact, with a path.

**Computed** means numpy arithmetic run in this session over (a) prediction
arrays already saved under `bench/real_data/results/*/predictions/` and (b) the
target regenerated from `bench/real_data/generators.py`, which is deterministic
counter-based splitmix64 and therefore reproduces the exact bits the harness
trained on. **The reconstruction is verified, not assumed.** The regenerated
train label mean is `1.674292928620779` at the large tier against the record's
`1.674292928620779`, `1.678177` at the standard tier against the record's
`1.6781773243440532`, and `0.10032661` on the categorical scenario against the
record's `0.10032661335736007`. Three independent scenarios, exact agreement.

**Estimated** means arithmetic on top of the first two, or a judgment. Every
estimate is labeled inline, every time.

**UNKNOWN** means the question was asked and the artifacts do not answer it.
Section 9 collects those.

---

## 1. The gap is 1.7 percent of RMSE and 28.8 percent of the model's own error

The dense regression target is

```
signal = 3*x0 + 2*x1*x2 + 1.5*sin(6*x3) - 2*(x4 > 0.7) + 0.8*x5**2
y      = signal + 0.30 * standard_normal
```

(`bench/real_data/generators.py`, `dense_regression`). The noise term is exactly
`0.30 * N(0,1)`, so **the Bayes-optimal RMSE is known**, and it is not a
parameter of the comparison. Computed on the held-out rows it is **0.298252** at
the standard tier and **0.299662** at the large tier.

Almost all of every arm's RMSE is that floor. Subtracting it changes the shape
of the problem completely. Excess RMSE below is `sqrt(mean((prediction -
signal)**2))`, the model's distance from the true conditional mean, computed
from the stored prediction arrays.

**Standard tier, 159,649 train x 50 features, 40,351 test, 100 trees**
(`bench/real_data/results/20260817T124906Z-postflip/`)

| arm | RMSE | excess RMSE | excess vs CatBoost |
|---|---|---|---|
| CatBoost, its own defaults | 0.305663 | **0.067886** | 1.00x |
| ours, symmetric, GPU | 0.307693 | 0.076429 | 1.13x |
| ours, symmetric, CPU | 0.308262 | 0.079487 | 1.17x |
| **ours, leaf-wise, CPU (shipped)** | 0.310775 | **0.087419** | **1.29x** |
| ours, leaf-wise, GPU (shipped) | 0.310847 | 0.087404 | 1.29x |
| LightGBM stock+det | 0.313535 | 0.096805 | 1.43x |
| XGBoost, its own defaults | 0.324012 | 0.126543 | 1.86x |
| ours, depth-wise, GPU (XGBoost mirror) | 0.324934 | 0.132711 | 1.96x |
| ours, depth-wise, CPU (XGBoost mirror) | 0.325803 | 0.133668 | 1.97x |

**Large tier, 799,110 train x 100 features, 200,890 test, 100 trees**
(`bench/real_data/results/20260817T110847Z-dense1mfixed/`)

| arm | RMSE | excess RMSE |
|---|---|---|
| ours, symmetric, GPU | 0.303271 | **0.046428** |
| ours, symmetric, CPU | 0.303431 | 0.047635 |
| CatBoost | 0.303468 | 0.048337 |
| ours, depth-wise, GPU (the OLD isolation arm) | 0.307367 | 0.069152 |
| ours, depth-wise, CPU (the OLD isolation arm) | 0.307403 | 0.069240 |
| **ours, leaf-wise (shipped)** | 0.307782 | **0.070303** |
| LightGBM stock+det | 0.311028 | 0.084417 |

**The reframing, stated once.** Our shipped default is 1.70 percent behind
CatBoost on RMSE at the standard tier and **28.8 percent behind on the error the
model is actually responsible for**, which is 66 percent more of it in MSE. It
is 2.86 percent ahead of LightGBM on RMSE and **9.7 percent ahead on excess
error**. Both statements are the same measurement seen through two lenses, and
the excess lens is the one that moves when a mechanism moves. Every price in
section 3 is quoted in it, and section 2 splits it into bias and variance.

**Three numbers describe that one gap and they are all correct, so a reader who
meets one of them alone will misquote it.** Against CatBoost at the standard
tier our shipped arm is behind by **28.8 percent on excess RMSE**, **65.8
percent on excess MSE measured directly as `mean((prediction - signal)**2)`**,
and **70.4 percent on excess MSE obtained by subtracting the floor from the
total, `rmse**2 - floor**2`**. The first two are the same quantity under a
square root. The last two differ by the cross term `2*mean(e * noise)`, which is
about a tenth of a percent of each arm's excess and therefore negligible per arm
but not negligible in the RATIO, because it lands on a numerator and a
denominator that differ by a factor of 1.7. **This document uses the direct
measurement throughout**, which is what `decompose.py` computes and the only one
of the three that decomposes into bias and variance. Quoting the subtraction
figure is not wrong, but it is a different lens and must be labeled.

One caution about the RMSE lens that this makes concrete. On a target with a
0.30 noise floor, a mechanism that halves the model's error moves RMSE by about
0.9 percent. **A 1 percent RMSE budget on this generator is a 50 percent
model-error budget**, which is why the budget looks generous and the ordering
looks tight at the same time.

### 1.1 Which gate is actually failing

Two different bars are in play and they disagree.

- **The harness gate.** `bench/real_data/thresholds.json`,
  `scenarios.dense_regression.differential`, is `relative, max_worse 0.03`
  **against the comparator, which is LightGBM**. Our leaf-wise arm is 0.86
  percent *better* than LightGBM. **This gate passes and is not close.**
- **The standing directive**, 1 percent relative to the better peer. CatBoost is
  the better peer at 0.305663, so 1 percent is 0.308720 and we sit at 0.310775.
  **This is the bar we miss, by 0.67 percentage points.**

The harness has no CatBoost gate. That is a real gap, because the arm that decides the
directive is a peer column with no threshold attached to it.

---

## 2. The gap is variance. CatBoost buys it with bias, and at the large tier we already have less bias than it does

Before attributing anything to a parameter it is worth knowing what kind of
error we are attributing.

The residual `e = prediction - signal` is decomposed by conditional means on
each of the six signal-carrying features. **Each component has the sampling
floor `n_bins * var(e) / n` subtracted**, because a conditional mean over many
cells of a pure-noise residual is not zero and quoting it as bias would be
wrong. Bin counts are chosen for about 40 to 50 rows per bin, which is 1000 bins
at the standard tier and 4000 at the large tier.

**Every number in this section is now produced by `bench/real_data/decompose.py`
rather than by hand.** Until 2026-08-17 this section was arithmetic done once in
one session and never re-run. The tool reproduces every excess-MSE figure below
to six significant digits from the stored predictions, which confirms it is
reading the same models, and it **disagreed with the hand split of that excess
into bias and variance by 27 to 38 percent at both tiers.** The tables here are
the tool's. What the disagreement cost is set out at the end of this section: the
conclusion survives unchanged, one headline ratio does not.

**The resolution of the decomposition changes the answer, the first pass got it
wrong, and the second pass was wrong about having fixed it.** At 40 bins the
systematic share reads 3 to 8 percent for every arm. That is an artifact. The
bias on the step term `-2*(x4 > 0.7)` lives in a narrow window around the bin
edge the model actually cut at, and a wide conditional mean averages the
residual's two opposite-signed spikes together and cancels most of it. The
estimate therefore rises with resolution. **It does not converge, at either
tier, at any resolution these test sets can support.**

Systematic part of excess MSE against resolution, our leaf-wise arm.

| rows per bin | bins | standard tier, systematic | large tier, systematic |
|---|---|---|---|
| ~1000 | 40 | 0.000424 (5.6%) | 0.000162 (3.3%) |
| ~200 | 200 / 1004 | 0.000814 (10.6%) | 0.000320 (6.5%) |
| 45 to 50 | 897 / 4018 | 0.002192 (28.7%) | 0.000765 (15.5%) |
| 20 to 25 | 2018 / 8036 | 0.002587 (33.8%) | 0.000966 (19.5%) |

**Non-convergence here is a property of the method against a step
discontinuity, not a defect in it.** To resolve bias concentrated in a window of
width `w` the bins must be narrower than `w`, and `w` is the distance between
0.7 and the bin edge the model cut at, which is unknown and can be arbitrarily
small. The floor is set by the test set: at 200,890 held-out rows, 25 rows per
bin is about 1/8000 of the range and going finer makes the subtracted sampling
term a large fraction of the raw estimate. So **every bias figure in this
document is a lower bound and every variance figure is an upper bound**, both by
an unknown factor, and separately a conditional mean on one feature cannot see
bias living in an interaction of two. The tables quote 45 to 50 rows per bin,
which is the finest resolution where the floor subtraction is still clean.

**Standard tier, 159,649 train and 40,351 held-out rows, at 1000 bins**, run
`20260817T124906Z-postflip`.

| arm | excess MSE | systematic | share | of which x4 | variance |
|---|---|---|---|---|---|
| ours, leaf-wise CPU | 0.007642 | 0.002192 | 28.7% | **0.001540** | 0.005450 |
| ours, leaf-wise GPU | 0.007639 | 0.002200 | 28.8% | **0.001543** | 0.005439 |
| LightGBM | 0.009371 | 0.003528 | 37.6% | **0.002868** | 0.005843 |
| CatBoost | 0.004608 | 0.002116 | 45.9% | **0.000261** | 0.002492 |
| ours, symmetric GPU | 0.005841 | 0.003345 | 57.3% | 0.001588 | 0.002497 |
| ours, symmetric CPU | 0.006318 | 0.003623 | 57.3% | 0.001651 | 0.002695 |
| XGBoost | 0.016013 | 0.001068 | 6.7% | 0.000118 | 0.014945 |
| ours, depth-wise (XGB mirror) | 0.017612 | 0.002517 | 14.3% | 0.001557 | 0.015096 |

**And the large tier, 799,110 train and 200,890 held-out rows, at 4018 bins**,
run `20260817T110847Z-dense1mfixed`, which is the cleaner of the two and the one
the conclusion rests on.

| arm | excess MSE | systematic | share | of which x4 | variance |
|---|---|---|---|---|---|
| **ours, leaf-wise (shipped)** | 0.004942 | 0.000765 | 15.5% | 0.000304 | **0.004177** |
| ours, symmetric GPU | 0.002156 | 0.001354 | 62.8% | 0.000313 | 0.000801 |
| ours, symmetric CPU | 0.002269 | 0.001461 | 64.4% | 0.000312 | 0.000808 |
| **CatBoost, its own rate** | 0.002336 | 0.001480 | 63.4% | 0.000347 | **0.000856** |
| LightGBM | 0.007126 | 0.002960 | 41.5% | 0.002513 | 0.004166 |
| ours, depth-wise isolation | 0.004782 | 0.001271 | 26.6% | 0.000322 | 0.003511 |

Two things in that table were not true of the hand version and are worth
naming before the arithmetic. **Our leaf-wise CPU and GPU arms are identical to
nine digits of excess MSE at the large tier**, which is what device agreement is
supposed to mean and is here confirmed on accuracy rather than on a tolerance.
And **our symmetric GPU arm is not merely close to CatBoost at the large tier,
it is ahead of it on all three quantities**: lower excess (0.002156 against
0.002336), lower variance (0.000801 against 0.000856) and lower bias (0.001354
against 0.001480). That is the strongest single argument in this document for
the shipped-default question, and it is an argument the hand table already
contained without drawing it.

One row of the hand version is missing here and one label is weaker. **CatBoost
at a learning rate matched to ours (0.1), which the hand table put at 0.004608
excess**, is not present in `dense1mfixed` and could not be re-measured; it is
cited from the hand version and marked accordingly wherever section 3 uses it.
The `lr 0.5` and `lr 0.1` labels are also dropped, because the stored records do
not carry a resolved learning rate per row and the tool cannot confirm them. The
rate each arm actually used is section 3.3's subject and is established there.

**This table is the answer to the brief's question, and it is one line.**

Against CatBoost at the large tier the gap is 0.002606 of excess MSE, and

- the **bias** difference is 0.000765 - 0.001480 = **-0.000715. We have 48
  percent LESS bias than CatBoost.**
- the **variance** difference is 0.004177 - 0.000856 = **+0.003321. We have 4.9
  times CatBoost's variance** at this resolution, and see the warning below
  about quoting that multiple at all.

**CatBoost is not fitting the function better than we are. It is fitting it
worse and averaging better, and the averaging wins by more than the fitting
loses.** Our symmetric arm reproduces the same trade against our own leaf-wise
arm (variance 0.000801 against 0.004177, bias 0.001354 against 0.000765), which
means the trade is a property of the configuration and not of CatBoost's code.
**Our symmetric arm in fact completes the trade further than CatBoost does**, to
lower bias and lower variance than CatBoost reaches, which is why it is the
better arm at the large tier.

**Four consequences, and the first is a correction of this document's own first
draft.**

1. **It is not true that our error is all variance.** It is at least 15.5
   percent bias at the large tier and at least 28.8 percent at the standard
   tier, and most of that bias is one feature, `x4`, the step. Section 3.10 is
   about that. The first draft said 7.7 percent and that number came from a
   40-bin decomposition that could not resolve the term.
2. **It IS true that the gap is variance, and this is the one part of the
   section that does not depend on resolution at all.** At the large tier the
   variance difference is 127 percent of the gap and the bias difference is
   minus 27 percent. More importantly, **at every resolution from 40 bins to
   8036 our bias is lower than CatBoost's and our variance is higher**, a range
   of 200x in bin width, so the sign of both differences is settled even though
   neither magnitude is. **Every recommendation in section 5 that reduces
   variance survives, and any that would reduce bias at the cost of variance is
   pointing the wrong way.**
3. **The x4 border term is NOT a CatBoost advantage at the large tier.** Every
   arm sits between 0.000304 and 0.000347 on it, ours the lowest of all four
   engines. Only LightGBM is an outlier, at 0.002513, eight times everyone else.
   Section 3.10 shows the standard tier reverses this, which is what makes it a
   lottery rather than a mechanism. **At the standard tier the reversal is
   sharper than the hand version reported**: our three arms sit at 0.001540,
   0.001588 and 0.001557, within 3 percent of each other despite three different
   growth policies, against CatBoost's 0.000261 and XGBoost's 0.000118. A term
   that is invariant to growth policy across a 3x spread in total excess is
   upstream of the tree, and section 3.10 identifies what it is upstream of.
4. **The room above us is bounded and it is worth knowing by how much.** If our
   leaf-wise arm kept its own bias and took CatBoost's variance, its excess MSE
   would be 0.001621, better than CatBoost's 0.002336 and 31 percent below it.
   **There is more than the gap available here**, which is the strongest reason
   to attack variance rather than to copy the configuration wholesale.

**And one warning about the multiple, which is the figure this section most
invites a reader to quote.** "We have N times CatBoost's variance" is not a
converged quantity and no single value of N is defensible. It reads 2.2 at 40
bins, 2.8 at 1004, 4.9 at 4018 and 7.8 at 8036, rising monotonically, because
CatBoost's bias estimate grows faster with resolution than ours does and each
arm's variance is the remainder. The hand version quoted **7.3 times** as
settled; that value is reachable near 8000 bins and is not what the document's
own stated resolution of 4000 produces. **Quote the sign, the ordering and the
share of the gap, all three of which are stable. Do not quote the multiple
without its resolution beside it.**

The step term also settles a prediction. `-2*(x4 > 0.7)` is the one term the
growth-attribution reading names as the one leaf-wise growth should be best at.
At the standard tier our bias there is 0.001540 against CatBoost's 0.000261, **a
factor of 5.9 worse**; at the large tier the two are level. **In neither case is
leaf-wise growth better at it**, and section 3.10 shows the reason has nothing
to do with growth. The re-measurement strengthens this: our leaf-wise, symmetric
and depth-wise arms agree on the term to within 3 percent, so growth policy does
not move it at all.

A negative result, recorded because it rules a hypothesis out. Marginal
dependence on the noise features was measured the same way, 20 features at 20
bins with the same floor subtracted, and it is at or below the floor for every
arm (mean recovered variance 2.2e-07 for us, 3.6e-07 for LightGBM, 9.5e-07 for
CatBoost, against excess MSE in the thousandths). **No arm has a first-order
dependence on a noise feature.** This does not rule out overfitting to
high-order interactions among noise features, which is what a deep tree
produces and which no one-dimensional test can see, so it constrains the
mechanism without identifying it.

A third check, which rules out the simplest reading of "100 trees at 0.1 is
under-converged". Regressing the true signal on each arm's prediction gives a
slope of 1.00259 for our leaf-wise CPU arm and 1.00003 for CatBoost, and the
predicted standard deviation is 1.7320 against the signal's 1.7387. **The
leaf-wise arms are under-dispersed by 0.26 percent and rescaling them by the
fitted slope removes 0.1 percent of their excess error.** They are not globally
shrunk toward the mean. Whatever the learning rate is doing here, it is not a
scalar shrinkage that a scalar could undo.

---

## 3. The mechanisms, priced

Every price is quoted as a change in excess RMSE at the LARGE tier, 799,110 x
100, because that is where two runs exist at two learning rates over identical
data and the isolations are cleanest.

### 3.1 Growth shape, isolated cleanly. Worth 1.5 percent

`bench/real_data/results/20260817T110847Z-dense1mfixed/` carries a depth-wise
arm whose resolved parameter dict is, read off the record,

```
num_leaves 31, max_depth -1, learning_rate 0.1, min_data_in_leaf 20,
min_child_hess 0.001, lambda_l1 0.0, grow_policy "depthwise", n_estimators 100
```

which is the leaf-wise arm's dict **with `grow_policy` changed and nothing
else**. That is a single-variable isolation of growth order, and it is the only
one in the repository.

| | excess RMSE | |
|---|---|---|
| leaf-wise | 0.070303 | |
| depth-wise, same everything else | 0.069240 (CPU), 0.069152 (GPU) | **-1.5%** |

**Present in our leaf-wise arm.** No, by definition.
**Cost.** Free at the parameter level. At the large tier the depth-wise GPU arm
was also 11 percent faster (3.245 s against 3.659 s); at the standard tier the
depth-wise arm was 39 percent slower (1.167 s against 0.839 s). Mixed.
**Price.** **-1.5 percent of excess error. Verified, single-variable.**

**This is the finding that resizes the growth-shape story.** The full CatBoost
shape-and-regularization bundle is worth 25 percent (section 3.2). Growth order
on its own is worth 1.5 percent. Section 4 takes that further.

**Warning about the arm's name.** `mojotrees_depthwise` STOPPED being this
isolation on 2026-08-17 at about 07:42, when `bench/real_data/scenarios.py`
rebuilt it as an XGBoost mirror carrying `learning_rate 0.3, num_leaves 64,
max_depth 6, lambda_l2 1.0, min_child_hess 1.0, min_data_in_leaf 1`. The
standard-tier depth-wise numbers in section 1 (0.324934, excess 0.132711) are
that arm and are **not** a growth-order isolation. Any series that puts the
0.069152 and the 0.132711 in one column is wrong.

### 3.2 The CatBoost shape-and-regularization bundle at matched learning rate. Worth 25 percent

`bench/real_data/results/20260816T181134Z-decision-1m/` ran the same large-tier
data with **every arm pinned at `learning_rate = 0.1`**, including CatBoost and
including the symmetric arm. That run is the matched-rate control that the
2026-08-17 runs are not.

| arm, all at lr 0.1 | RMSE | excess RMSE |
|---|---|---|
| ours, symmetric (CatBoost mode) | 0.303973 | **0.052572** |
| CatBoost | 0.305335 | 0.059163 |
| ours, leaf-wise | 0.307782 | 0.070303 |
| LightGBM | 0.311028 | 0.084417 |

The symmetric arm's dict differs from the leaf-wise arm's in twelve keys
(`grow_policy symmetrictree`, `max_depth 6`, `num_leaves 64`,
`min_data_in_leaf 1`, `min_child_hess 0.0`, `lambda_l2 3.0`,
`score_function cosine`, `random_strength 1.0`,
`leaf_estimation_iterations 1`, `max_cat_to_onehot 2`, `bootstrap_type MVS`,
`subsample 0.8`).

**Price of the whole bundle, at matched rate. -25.2 percent of excess error
(0.070303 to 0.052572). Verified as a bundle, unattributed within it.**

Note also that at matched rate **our symmetric arm beats CatBoost** (0.052572
against 0.059163, 11 percent better). Whatever CatBoost has that we do not, at
lr 0.1 it is worth less than the twelve keys we already copied.

### 3.3 CatBoost's automatic learning rate. Worth 9 percent on our symmetric arm and 18 percent on CatBoost

Verified from `src/mojotrees/auto_learning_rate.mojo`, which transcribes
`options_helper.cpp` `TAutoLRParamsGuesser` with its coefficient table, and
confirmed by arithmetic. For CPU, RMSE, `use_best_model=false`,
`boost_from_average=true`, the coefficients are `(0.158, -4.287, -0.813,
2.571)` and

```
lr = round6(min(exp(A log N + B) * exp(C log T + D) / exp(C log 1000 + D), 0.5))
```

- 20,000 rows, 100 iterations gives **0.427309**, which matches CatBoost's own
  recorded 0.4273 in catalog A37. The transcription is correct.
- 159,649 rows gives a raw 0.593307, **capped at 0.5**.
- 799,110 rows gives a raw 0.765229, **capped at 0.5**.
- 463,715 rows (YearPredictionMSD) gives a raw 0.702178, **capped at 0.5**.

`catboost_readback.json` in all three 2026-08-17 runs reads back
`learning_rate = 0.5`, so **at every tier this project measures, CatBoost's
default rate is the 0.5 cap and our shipped lossguide default is 0.1, a factor
of five, at an identical 100 trees.**

Prices, from the two large-tier runs over identical data.

| | at lr 0.1 | at lr 0.5 | price |
|---|---|---|---|
| our symmetric arm | 0.052572 | 0.047635 (CPU) | **-9.4%** |
| CatBoost | 0.059163 | 0.048337 | **-18.3%** |

**Present in our leaf-wise arm.** No. `src/mojotrees/auto_learning_rate.mojo`
is built, reachable and enabled by default under `grow_policy=oblivious` only;
under `lossguide` and `depthwise` the flat 0.1 stands, because those mirror
LightGBM and LightGBM has no such feature. The docstring says so and
`params._default_auto_learning_rate` implements it.
**Cost.** Free. Two `log` and three `exp` calls, once per fit.
**Price on the leaf-wise arm. UNKNOWN and it is the largest unpriced term in
this document.** The direction is not obvious and there is a stored
counterexample. The XGBoost mirror runs `lr 0.3` with 64 leaves and
`min_child_hess 1.0` and is the **worst** arm in the standard-tier table
(excess 0.132711). A high rate helps a heavily constrained tree and hurts a
loosely constrained one. Our lossguide default is 31 unbounded-depth leaves at
`lambda_l2 = 0`, which is the loose end.

### 3.4 `lambda_l2`. Worth 1.6 percent on the leaf-wise arm, free, and already implicated on two other scenarios

A clean isolation exists and nobody has read it as one.
`bench/results/cpu_accuracy_2026-08-16/records_prestock.json` ran the standard
tier before the stock-defaults flip, when `BASE_PARAMS` still carried
`lambda_l2 = 1.0`. Our arm's dict there is the current one **plus that key**,
and the two runs share a bin digest, `26c6dcd770b6bf4a...`, so the data and the
binning are identical.

| lambda_l2 | RMSE, CPU | RMSE, GPU | excess RMSE, CPU (estimate) |
|---|---|---|---|
| 1.0 | 0.310405 | 0.310355 | 0.086006 |
| 0.0 (shipped today) | 0.310775 | 0.310847 | 0.087419 (measured directly) |

The excess column at `lambda_l2 = 1.0` is an **estimate**, computed as
`sqrt(RMSE^2 - Bayes^2)` because the predictions from that run are not on disk
in a form this lane read. The same formula reproduces the directly measured
0.087419 as 0.087332, so it is good to about 0.1 percent.

**Price. -1.6 percent of excess error at `lambda_l2 = 1.0`. Estimate, from a
matched-data two-point comparison.** The comparison spans a day of extension
rebuilds by six lanes, so it is not a single-variable comparison in time.

**And this is not the largest evidence for it.**
`bench/results/cpu_float32_lambda0_2026-08-16/RESULTS.md` already records, as an
unlooked-for finding, that the flip to stock `lambda_l2 = 0` cost accuracy on
two other scenarios and cost us more than it cost LightGBM.

| scenario, metric | | lambda 1 | lambda 0 |
|---|---|---|---|
| imbalanced_binary, average precision | ours | 0.013600 | 0.010218 (0.75x) |
| | LightGBM | 0.013264 | 0.012352 (0.93x) |
| multiclass, multi_logloss | ours | 0.18763 | 0.62112 (3.31x worse) |
| | LightGBM | 0.18698 | 0.43625 (2.33x worse) |

**`lambda_l2 = 0` is the single default in this library with recorded accuracy
damage on three separate scenarios, and it is there because we mirror
LightGBM.** CatBoost ships 3.0.

**Present in our leaf-wise arm.** Yes, at 0.0
(`python/mojotrees/sklearn.py`, `_LAMBDA_L2 = 0.0`).
**Cost.** Free. It is a denominator term already in the leaf-value expression.

### 3.5 MVS bootstrap at 0.8

CatBoost's default `bootstrap_type` is MVS at `subsample 0.8`, confirmed in
`catboost_readback.json`. Our symmetric arm carries it; our leaf-wise arm does
not. It is a variance reducer and section 2 says variance is the entire gap.

**Present in our leaf-wise arm.** No.

**Reachability, corrected.** `CATBOOST_UNMATCHABLE["row_sampling"]` in
`bench/real_data/scenarios.py` reads as though the accelerator refuses the
bundle. **It is stale, and the correction matters for the recommendation.**
`src/mojotrees/train_gpu.mojo`, in the docstring of the device-gradient
eligibility function around line 960, states it in bold, quoted here in full. MVS "solves its keep
threshold from this round's per-row gradient magnitudes and then drops rows,
which the device round has neither the magnitudes nor a compaction step for, so
it is `ROUND_MVS_HOST_MAGNITUDES` there -- and under AUTO that resolves to the
host-gradient arm, where `sampling.bootstrap_round` draws it exactly and the
trees are still grown on the device. **So MVS is honored on a GPU fit rather
than dropped or refused**; what it costs is the device derivative kernel, not
the sampler." The symmetric GPU arm ran five repeats today and did not raise.

**Consequences of that correction, both ways.**

- MVS is matched on both backends, so it is **not** available as an explanation
  of either the CatBoost gap or the CPU-against-GPU symmetric divergence. It
  falls out of both.
- **Its cost on the GPU is not a speedup, it is a demotion.** Turning MVS on
  takes a GPU fit off the device round and onto host-computed gradients. The
  observed shape is consistent, and at the large tier the symmetric arm, which
  carries MVS, runs **17.072 s on the GPU against 9.089 s on the CPU**, the only
  arm in the run where the accelerator is slower than the host. Attributing all
  of that to MVS would be wrong, since the oblivious level path is in there too,
  but a mechanism that forces the host derivative arm is the obvious first
  suspect and it is a real cost.

  **Both seconds carry their run and both are PRE-FLIP.** Run
  `bench/real_data/results/20260817T110847Z-dense1mfixed/`, arm
  `mojotrees_catboost_mode`, 799,110 rows x 100 features x 100 trees, symmetric
  depth 6, 10 threads, five repeats, `dense_regression` tier `large`, comparator
  `stock+det@v2`. Both figures are the medians of their five repeats, 17.0719 of
  a 17.0322 to 17.8772 spread on the gpu and 9.0892 of an 8.8939 to 11.0279
  spread on the cpu, and the run's quiet-box status is not established. It was
  taken before the four symmetric GPU switches flipped to default on later the
  same day, so the GPU side of this contrast is a stale absolute for the arm that
  ships today. **No post-flip large-tier symmetric run has been filed**, so
  nothing in `bench/real_data/results/` supersedes it and the honest statement is
  that the post-flip absolute at this shape is unrecorded. The direction of the
  argument is unaffected either way, because the flips move the GPU number
  downward and the CPU number not at all, so the demotion this paragraph prices
  can only be smaller than it looks here.

**Cost, restated.** Negative on the CPU, roughly 20 percent fewer rows in the
histogram accumulation per tree. **Materially positive on the GPU**, because it
gives up the device round. Refused by entry point on multiclass and sparse.

**Price. UNKNOWN.** No isolation exists. **Estimate, and labeled as one, of
-3 to -8 percent of excess error**, reasoned from three things and worth no
more than that. The gap is variance-dominated, subsampling is the textbook
variance reducer with no bias cost at these row counts, and it is one of the
twelve keys in a bundle worth 25 percent.

### 3.6 Cosine split scoring against L2 gain

Our symmetric arm scores splits with Cosine, our leaf-wise arm with the L2 gain.

**Present in our leaf-wise arm.** No.
**Cost.** Estimated free. It is a different closed-form functional over the same
histogram, not extra passes.
**Price. UNKNOWN.** No isolation. The one adjacent measurement is a fixture,
recorded in `src/mojotrees/tree_parameters_extra.mojo` around line 2000, showing
a symmetric depth-6 fit at 4,000 x 12 moving by `max|diff| = 1.142` when Cosine
is turned on. That establishes that the setting takes effect. It says nothing
about the sign of its effect on held-out error.

### 3.7 `random_strength = 1.0`, and a correction to the premise

The brief's premise was that `params.mojo` applies CatBoost's `random_strength`
default only on the CPU, so a GPU symmetric fit runs at 0.0 and the parameter
cannot explain the GPU numbers. **Half of that is right and the conclusion does
not follow.**

What is verified, from `src/mojotrees/params.mojo`
`_apply_catboost_mode_defaults`, is that the **mode default** carries a
`config.device == CPU_DEVICE` guard. But that branch fires only when the caller
named nothing (`not saw_random_strength`). **The harness names it.** The record
in every symmetric cell carries `random_strength: 1.0` in the resolved dict, and
`python/mojotrees/sklearn.py` sends `random_strength_set: 1` beside it, so the
mode default is skipped and the explicit 1.0 is what reaches the trainer on both
devices. The CPU-only guard is therefore **not** the reason a GPU symmetric fit
might run unnoised.

What is separately verified is that a GPU fit **used** to accept the setting and
apply nothing. `src/mojotrees/tree_parameters_extra.mojo`, in the
`check_scalars` refusal block, records a measurement at 4,000 x 12, symmetric,
depth 6:

```
cpu  random_strength=1     max|diff vs plain| = 1.078
cpu  score_function=Cosine max|diff vs plain| = 1.142
gpu  random_strength=1     max|diff vs plain| = 0.000
gpu  score_function=Cosine max|diff vs plain| = 0.000
```

and then records that **both were re-allowed on 2026-08-17 on evidence that the
device model now moves**, once the oblivious level launch staged the noise plane
and passed the score function.

**Present in our leaf-wise arm.** No, it is 0.0.
**Cost.** Estimated free.
**Price. UNKNOWN, and the one contrast available points the wrong way for it.**
At both tiers the symmetric GPU arm beats the symmetric CPU arm (0.046428
against 0.047635 at large, 0.076429 against 0.079487 at standard). If the CPU
arm is noised and the GPU arm is not, that is `random_strength = 1.0` costing 2.5
to 4 percent of excess error. **The contrast is confounded** with the fixed-point
Int32 histogram divergence that `bench/results/COMPARISON_RUN_2026-08-16.md`
characterizes at length, and this lane cannot separate them.
`src/mojotrees/gpu_resident_round.mojo` was modified at 08:55 on 2026-08-17,
**after** the `postflip` run at 08:50, so whether that run's GPU arm was noised
is UNKNOWN.

### 3.8 Ordered boosting and CTRs. Both absent from the comparison entirely

**Ordered boosting is not in play at any tier this project measures.** Verified
from CatBoost's source and recorded in `COMPARISON_RUN_2026-08-16.md`:
`UpdateBoostingTypeOption` hard-sets `Plain` when `(learnSampleCount >= 50000 ||
IterationCount < 500)`, and at 100 estimators the second clause fires
unconditionally. `catboost_readback.json` confirms `boosting_type Plain` on
every cell. **Price on the dense gap. Zero, verified.**

**CTRs are not in play on `dense_regression` either.** There are no categorical
columns. `catboost_readback.json` for the categorical scenario carries
`max_ctr_complexity 1` and `one_hot_max_size 2`, so they are in play there and
section 7 handles it.

### 3.9 Leaf value estimation

`leaf_estimation_method` resolves to `Newton` on CatBoost and mojotrees has no
such parameter (`CATBOOST_UNMATCHABLE["leaf_estimation_method"]`). For RMSE the
Hessian is 1 and Newton and gradient descent on the leaf coincide, so **estimated
price on this scenario, zero**. `leaf_estimation_iterations` resolves to 1 under
RMSE, which our symmetric arm already matches. Neither is a candidate here.

### 3.10 Border placement. Large, measurable, and NOT a CatBoost advantage

CatBoost bins with `GreedyLogSum` and we bin equal-frequency
(`CATBOOST_UNMATCHABLE["border_placement"]`, which calls it "open and unclosable
by any parameter"). Same border count, different positions. This was raised as
the strongest remaining candidate for the whole gap. **It is a much bigger term
than anyone expected and it does not point the way it was expected to.**

**First, a factual correction to the premise.** The features on this scenario
are **uniform on [0,1)**, not standard normal. `generators._stream` returns
`uniform(...)`, verified numerically on feature 0: min 0.000000, max 0.999997,
mean 0.500189, standard deviation 0.287968, quartiles 0.25227 / 0.49892 /
0.74914. Only the label's noise term is normal. So the tail-density reasoning
that motivates comparing equal-frequency against a recursive median does not
apply here, and on a uniform marginal the two rules place borders in nearly the
same places. **That is exactly why the term turns out to be large anyway, and
the reason is not the marginal at all.**

#### The binning floor, computed

For a fixed border set, the best any tree ensemble can do is the conditional
mean of the signal given the bin vector. Because the target is additive plus one
product of independent features, that oracle can be built exactly. Fitted on the
train split, scored on the test split, standard tier, 254 borders per feature.

| border rule | binning-floor MSE | as RMSE |
|---|---|---|
| uniform width | 3.921e-03 | 0.062617 |
| **equal-frequency (ours)** | **2.598e-03** | 0.050967 |
| recursive median (proxy for GreedyLogSum) | 1.332e-03 | 0.036494 |
| equal-frequency with one border forced onto x4 = 0.7 | **6.685e-05** | 0.008176 |

For scale, our leaf-wise arm's whole excess MSE is 7.642e-03 and the gap to
CatBoost is 3.034e-03. **So the binning floor is 34 percent of our total error
and 86 percent of the gap. The mechanism is real and it is not small.**

The last row is the finding. **Moving a single border onto x4 = 0.7 drops the
floor by a factor of 39.** The entire binning floor on this generator is one
border, and it is the border next to `-2.0 * (x4 > 0.7)`, the only exact
discontinuity in the target. Everything else is smooth and 255 bins resolve it
to nothing, since the within-bin variance of `3*x0` at a bin width of 1/255 is
1.15e-05, three orders below the gap.

The recursive-median proxy is **my** implementation of "recursive median split",
built to the catalog's four-word description, not a transcription of
`GreedyLogSum`. Its 1.332e-03 says only that a different rule lands its border
somewhere else. It is a draw in a lottery, not a rule that is better.

#### What actually happened, measured on the stored predictions

The oracle says where the room is. The predictions say who took it. Residual
`prediction - signal` averaged in twenty 0.002-wide bins across
`x4` in [0.68, 0.72], converted to a contribution to total excess MSE.

**Standard tier, 159,649 rows.** The window column is the ±0.02 estimator above;
the full column is section 2's converged 1000-bin whole-range estimator, which is
the tighter lower bound and the one the shares use.

| arm | window contribution | full x4 bias | share of that arm's excess MSE | residual peak below / above 0.7 |
|---|---|---|---|---|
| XGBoost | 1.610e-04 | 2.37e-04 | 1.5% | -0.14 / +0.19 |
| CatBoost | 4.186e-04 | **5.51e-04** | 12.0% | -0.24 / +0.30 |
| **ours, leaf-wise** | 1.412e-03 | **2.242e-03** | **29.3%** | -0.77 / +0.40 |
| ours, symmetric | 1.591e-03 | 2.350e-03 | 40.2% | -0.81 / +0.42 |
| LightGBM | 3.973e-03 | **3.882e-03** | **41.4%** | -1.13 / +0.90 |

**Our disadvantage against CatBoost from this one border is 1.691e-03 of excess
MSE, which is 55.7 percent of the entire 3.034e-03 gap at this tier.** That is
by far the largest single attributed term in this document, and it is the reason
this section exists rather than being a footnote.

**Large tier, 799,110 rows. The advantage reverses and the term nearly
vanishes for everyone except LightGBM.**

| arm | window contribution | full x4 bias | share of that arm's excess MSE |
|---|---|---|---|
| **ours, leaf-wise** | 1.153e-04 | **5.96e-04** | 12.1% |
| ours, symmetric, lr 0.1 | 1.206e-04 | 6.26e-04 | 22.6% |
| ours, symmetric, lr 0.5 | n/a | 5.98e-04 | 27.7% |
| CatBoost, at its own rate 0.5 | 1.154e-04 | **6.69e-04** | 28.6% |
| CatBoost, at matched rate 0.1 | 2.928e-04 | 7.61e-04 | 21.7% |
| LightGBM | 1.735e-03 | **2.764e-03** | 38.8% |

**At the large tier our border lands better than every other engine's**, ours at
5.96e-04 against CatBoost's 6.69e-04 and LightGBM's 2.764e-03. We gain 7.3e-05
against CatBoost here where we lost 1.691e-03 at the standard tier. The term is
the same size in absolute MSE for us at both tiers within a factor of four, and
what moved by a factor of four is CatBoost's side of it, in our favor.

#### The verdict, and why it is not "implement GreedyLogSum"

Three readings, in order of how much weight they carry.

1. **The term is a lottery and we have already won one of the two draws.** The
   same two binners, on the same generator, at two row counts, put us behind
   CatBoost by 9.9e-04 at one tier and ahead by 1.8e-04 at the other. A
   systematic advantage does not change sign with the sample size. What changes
   with the sample size is where the empirical 254th quantile of a uniform
   column falls relative to 0.700000.
2. **LightGBM is the worst arm on this term at both tiers**, 42 percent of its
   excess MSE at the standard tier and 24 percent at the large tier, and
   LightGBM's binner is the one ours is modeled on. Two implementations of the
   same rule land far apart. That is the definition of a term that is not about
   the rule.
3. **The term is an artifact of the generator.** `-2.0 * (x4 > 0.7)` is an exact
   axis-aligned discontinuity at a round number. Real targets very rarely
   contain one, and `bench/real_data/results/20260816T180302Z-decision`, which
   is the real-data cell, has no analogue of it. **A binning change tuned to
   this term would be tuning to the synthetic data**, which is the failure mode
   section 4 already caught the breadth reading in.

**So do not implement `GreedyLogSum`.** It would cost a new binning mode, a
`feature_border_type` parameter across four surfaces, a bit-moving change to
every fit, and a re-anchor of the accuracy gate, in exchange for a different
ticket in the same lottery. The measured evidence is that our current ticket is
already the better of the two at the tier with more rows.

**What IS worth doing, and it is cheap.** The finding generalizes past this
generator in one direction only. **Where a feature's marginal has an atom or a
sharp density change, a border placed on it is worth up to a factor of 39 in
the binning floor**, and no equal-frequency rule looks for one. That is not
`GreedyLogSum`, it is a candidate-border refinement. After the quantile borders
are chosen, snap a border to a detected discontinuity in the empirical CDF if
one lies inside a bin. It is local to `binning.fit_bins`, it does not need a new
mode, and it would help the equal-frequency rule on exactly the shape where
equal-frequency is blind. **UNPRICED on real data and it must be measured there
before it is built**, precisely because this generator's discontinuity is
artificial. Logged as a candidate, not as a recommendation.

**Cost of the mechanism, for the table.** Not free. Any of these is bit-moving
on every fit and must be judged against our own accuracy anchor.

### 3.10a The noise streams, recorded for completeness

Both engines add `random_strength` noise and the draws differ by construction, since
CatBoost keys per-document draws on thread-block position, we key on
`(seed, tree, site, feature, bin)`, and
`WIRE_NOTE_oblivious_level_noise.md` records that our two backends had to be
brought onto one keying before they even agreed with each other. Same
distribution, different numbers, no parameter closes it. **Not a candidate for
the gap and not priceable**: it is a difference in a random draw, so it has no
expected sign, and section 3.7 already records that the one contrast available
suggests the regularizer costs us rather than helps.

### 3.11 The table

Prices are in excess MSE where a share of the gap is quoted and in excess RMSE
where a percentage change is quoted, and each row says which.

| mechanism | in our leaf-wise default | cost to add | price | evidence |
|---|---|---|---|---|
| the twelve-key CatBoost bundle, matched rate | no | 2.7x to 4.7x slower on GPU | **-25.2% of excess RMSE** | verified, bundle |
| **border placement next to the x4 step** | equal-frequency | bit-moving on every fit | **55.7% of the standard-tier gap AGAINST us, 2.8% of the large-tier gap FOR us** | verified, both directions |
| automatic learning rate 0.1 to 0.5 | no | free | -9.4% on symmetric, -18.3% on CatBoost; **UNKNOWN on lossguide** | verified on the other arm |
| MVS at 0.8 | no | faster on CPU, **loses the device round on GPU** | **estimate -3 to -8%** | none |
| `lambda_l2` 0.0 to 1.0 | 0.0 | free | **-1.6% of excess RMSE** | matched-data two-point |
| growth order, leaf-wise to depth-wise | no | mixed | **-1.5% of excess RMSE** | verified, single-variable |
| Cosine scoring | no | free | UNKNOWN | none |
| `random_strength` 1.0 | no | free | UNKNOWN, one contrast says **+2.5 to +4% worse** | confounded |
| ordered boosting | no | n/a | **zero, verified** | CatBoost source |
| CTRs | no | n/a | zero on this scenario | no categoricals |
| Newton leaf values | n/a | n/a | zero for RMSE | analytic |
| noise stream keying | different by construction | n/a | no expected sign | section 3.10a |

Nothing here sums to 25.2 percent and it should not be read as if it could. The
items are not additive, three of the twelve bundle keys are unpriced, and the
border term is inside the bundle comparison as well as outside it.

**The one line to take away from the table.** The largest single ATTRIBUTED term
in the standard-tier gap is border placement at 55.7 percent, and it reverses at
the large tier, so it is a lottery rather than a mechanism. The largest DURABLE
term is the variance the bundle buys, which section 2 puts at a factor of at
least 2.2 and at 4.9 at its stated resolution, the sign being settled and the
multiple not.
The largest ACTIONABLE term is `lambda_l2`, because it is the only entry that is
free, reaches every entry point, reduces variance, and has recorded damage on
three scenarios.

---

## 4. The breadth reading, tested against the real dataset, and rejected as a general claim

**A CONFOUND IN THIS SECTION'S OWN TABLE, found and then RESOLVED on
2026-08-17. Read this before the section, because the section's conclusion
survives but its evidence did not support it as written.**

The table below sorts arms into "low breadth" and "maximal breadth". Every
low-breadth arm carries **31 leaves** and every maximal-breadth arm carries
**64**, verified from `20260816T180302Z-decision`'s own records: `mojotrees`
and `lightgbm` at `num_leaves = 31`, `mojotrees_catboost_mode` at
`max_depth = 6` which is 64, and CatBoost at its own depth 6. So in this table
BREADTH AND LEAF BUDGET ARE THE SAME AXIS and it cannot separate them. The
section attributes the inverted ordering to "a property of the data" when a
simpler rival was never examined: on a target where most features are noise, a
64-leaf tree overfits more than a 31-leaf one. That is the same unmatched
comparison this repository spent 2026-08-17 finding in its own harness keying.

**So it was re-measured at MATCHED leaf budgets**, on the same real dataset,
463,715 train by 90 features, 51,630 held out, 100 trees, GPU, with every other
parameter pinned identically: `learning_rate = 0.1`, `lambda_l2 = 1.0`,
`min_data_in_leaf = 20`, `max_bin = 255`, one seed. Pinning matters and is not
decoration: `grow_policy='symmetrictree'` silently supplies CatBoost-mode
defaults including `learning_rate = 0.03`, so an unpinned comparison measures
the rate as much as the policy.

    arm                     leaves    fit        rmse
    lossguide                   31    1.991 s    9.10851
    symmetric  depth 5          32    1.842 s    9.27566
    lossguide                   64    2.997 s    9.05392
    symmetric  depth 6          64    2.696 s    9.21316

**THE CONCLUSION SURVIVES AND IS NOW BETTER SUPPORTED.** At a matched leaf
budget symmetric is about 8 to 10 percent FASTER and 1.2 to 1.8 percent WORSE
on rmse, in both pairs. Under rule 9 that trade fails badly: the exchange rate
is at most about 1 percent of accuracy for at least 2x speed, and this is more
accuracy for a tenth of the speed. **The leaf-wise default stands, and it now
stands on a comparison that holds the leaf budget fixed rather than on one that
confounded it.**

Two further facts the re-measurement produced, both worth their own line.

A SYNTHETIC PROBE POINTED THE OTHER WAY AND DID NOT TRANSFER. On synthetic
200,000 rows by 90 features, symmetric beat lossguide on BOTH axes at matched
leaves, 0.553 s and rmse 0.30760 against 0.648 s and 0.31562. On the real
dataset the accuracy ordering reverses completely. Whatever the generator
rewards, this dataset does not, and a defaults decision taken on the synthetic
probe alone would have been wrong.

`num_leaves = 64` IS MORE ACCURATE THAN 31 ON THIS DATASET under lossguide,
9.05392 against 9.10851, for 1.5x the fit. That is a separate open question
about our shipped `num_leaves = 31` and it is NOT settled here, because one
dataset at one seed decides nothing about a default. It is recorded so that
nobody reads the table above as evidence for 31.

`docs/design/GPU_GROWTH_ATTRIBUTION.md` section 6.2 records that accuracy
improves monotonically with growth breadth across four policies, offers a
mechanism in the generator's interaction-heavy target, and closes by saying the
claim is "one synthetic dataset" and that "the same run's real-data companion at
463,715 rows by 90 features is not in this file and would have to be read before
any such claim". **That file was right to hedge and this section reads the
companion.**

`bench/real_data/results/20260816T180302Z-decision/`, UCI YearPredictionMSD,
463,715 train x 90 features, 100 trees, five repeats, **every arm at
`learning_rate = 0.1`** including CatBoost's, which was `cb-default` at that
date.

| arm | RMSE | growth breadth |
|---|---|---|
| LightGBM | 9.100863 | low |
| **ours, leaf-wise CPU** | **9.103829** | low |
| ours, leaf-wise GPU | 9.106065 | low |
| ours, symmetric CPU | 9.216053 | maximal |
| CatBoost | 9.231016 | maximal |

**The ordering is exactly inverted.** The two maximal-breadth arms are 1.24 and
1.43 percent WORSE than the two low-breadth arms, and the two engines' arms
still track each other faithfully (9.216053 against 9.231016, 0.16 percent
apart) which means the mirror is honest in both directions and the reversal is
a property of the data, not of the implementation.

Three conclusions.

1. **The breadth ladder is a property of `dense_regression`, not of the
   algorithms.** The generator puts 94 of 100 features at pure noise and builds
   a target whose docstring says it is "deliberately not a sum of univariate
   terms". That is a shape chosen to punish concentrated growth, and it does.
2. **Section 3.1's clean isolation already said the same thing quantitatively.**
   Growth order alone, everything else held, is worth 1.5 percent of excess
   error on the synthetic data where breadth is supposed to win. It is not the
   mechanism.
3. **The real-data comparison is at matched lr 0.1 and CatBoost's shipped rate
   is 0.5**, so it prices the shape and the regularization but not the rate.
   Section 9 lists the missing measurement.

`GPU_GROWTH_ATTRIBUTION.md` does not need correcting. It stated its scope
honestly and it named the missing read. This is the missing read, and the answer
is that the ladder does not generalize.

---

## 5. What the shipped leaf-wise default should change, ranked

Ranked by expected accuracy per unit of speed and per unit of risk, not by size.

### R1. Set a nonzero `lambda_l2` default under `lossguide`

**The change.** `_LAMBDA_L2` in `python/mojotrees/sklearn.py` and the matching
native default move from 0.0 to a nonzero value. 1.0 is the measured point; 3.0
is CatBoost's and is unmeasured on our leaf-wise arm.

**Why first.** It is the only candidate with recorded accuracy damage on
**three** scenarios rather than one. It costs nothing at any tier, on any
device, in any trainer. It reaches every entry point, unlike `bootstrap_type`.
And the reason it is 0.0 is not a measurement, it is the mirror.

**A correction to this recommendation's own first draft, 2026-08-17.** It said
"it does not change the tree shape, so no speed argument touches it." The first
clause is false. `lambda_l2` sits in the denominator of the gain,
`GL**2/(HL + lambda_l2) + GR**2/(HR + lambda_l2) - G**2/(H + lambda_l2)`
(`src/mojotrees/split.mojo`), so it changes which candidate wins and therefore
can change the tree. The conclusion survives on a better argument: **the
`+ lambda_l2` term is unconditional in both the gain and the leaf value, so a
nonzero value changes no instruction count, adds no pass and adds no kernel.**
At `lambda_l2 = 0` that addition is still executed. Nothing is skipped at zero
that has to be paid for at one. The shape effect is real but bounded and points
the cheap way: a positive lambda makes small-hessian candidates less
attractive, so under a `num_leaves` cap the leaf count is unchanged and under
`min_gain_to_split` a few marginal splits may be rejected, which is fewer nodes
rather than more.

**And one interaction worth naming, because it was invisible at zero.**
`src/mojotrees/split.mojo` proves that at `lambda_l2 = 0` the Cosine and L2
score functions have the same argmax. Cosine is therefore **provably inert at
our old default** and becomes a live choice at a positive lambda. That is a
second reason the zero was load-bearing in a way nobody intended: it silently
neutralized a scoring option this library advertises.

**Expected size.** -1.6 percent of excess error on dense regression at 1.0
(measured), plus the recorded recovery of 0.75x to 1.00x on
`imbalanced_binary` average precision and 3.31x to 1.00x on multiclass logloss.
The last two are the bigger prizes and they are already on disk.

**The confirming experiment.** One interleaved window, no new code, current
extension. `lambda_l2` in `{0.0, 1.0, 3.0, 10.0}` as an arm axis, on
`dense_regression` at standard and large tiers, `imbalanced_binary`, and
`multiclass`, three repeats, mojotrees CPU only. Twelve cells per scenario.
Report excess RMSE, not RMSE, on `dense_regression`. **Decision rule registered
in advance. Ship the value that is best or tied-best on all four scenarios; if
no value is, ship the best on `multiclass` logloss, because that is where the
damage is largest.**

### R2. Measure the learning rate against the tree budget under `lossguide`, then decide

**The change.** No parameter change yet. This is the measurement that ranks
second because the thing it prices is the largest unpriced term in the
document, and because a wrong answer here is expensive.

**Why not just turn the derivation on.** `src/mojotrees/auto_learning_rate.mojo`
is built, tested and one flag away from firing under `lossguide`. The
temptation is to enable it. Three things argue for measuring first. The rate it
would produce at our tiers is the 0.5 cap, a factor of five. Section 2 shows
our error is variance, and raising the rate raises variance. And the stored
counterexample is severe. The XGBoost mirror at lr 0.3 with 64 leaves is the
worst arm in the standard-tier table at excess 0.132711, 1.9x our leaf-wise arm.
**A high rate rewards a heavily constrained tree. Our lossguide default is the
loosest tree in the comparison.**

**And section 2 says the rate is a bias knob here, which is bad news for this
recommendation and worth stating plainly.** Decomposed at the large tier, moving
CatBoost from lr 0.1 to lr 0.5 cut its bias from 0.002935 to 0.001804, a 39
percent reduction, and moved its variance by 6 percent (0.000565 to 0.000532).
Our symmetric arm shows the same signature, bias 0.002245 down to 0.001640 with
variance flat. **The rate is buying bias reduction, which means those two arms
were under-converged at lr 0.1 and the rate fixed it.**

**OPEN, 2026-08-17. That paragraph is the one part of the rate argument the
re-measurement could not check, and it is quoted from the hand version as a
matched pair.** The lr 0.1 arms are absent from `dense1mfixed`, so the tool
cannot re-decompose either side, and the lr 0.5 side alone disagrees with the
hand version enough to matter: CatBoost's bias at its own rate re-measures to
0.001480 rather than 0.001804 and its variance to 0.000856 rather than 0.000532.
Substituting one side and keeping the other would mix two methods and would flip
the sign of the variance move, so **the pair stands or falls together and it
needs a run that carries both rates before it can be relied on.** What does not
depend on it is the conclusion drawn next.

Our leaf-wise arm's bias re-measures to 0.000765, the lowest of any arm in the
table and 48 percent below CatBoost's at its own rate. **There is very little
bias for a rate increase to buy on our arm, and section 2 says variance is what
we cannot afford more of.**

**Expected size.** UNKNOWN, and the plausible range spans the sign. The prior
this document ends with is that it is small and possibly negative under
`lossguide`, which is the opposite of what the -9.4 percent on the symmetric arm
suggests when read without the decomposition. **This is exactly why R2 is a
measurement and not a change.**

**The confirming experiment.** `learning_rate` in `{0.1, 0.2, 0.3, 0.5}` crossed
with `lambda_l2` in `{0.0, 1.0, 3.0}`, `lossguide`, `n_estimators = 100`, on
`dense_regression` at both tiers **and on YearPredictionMSD**, three repeats.
Twelve cells per dataset. **Add one row that separates the two hypotheses that
a rate sweep confounds, namely `learning_rate 0.1, n_estimators 500`.** If that row is
much better than every 100-tree row, the arm is under-converged and the answer
is trees, not rate. If it is not, the arm is at its capacity and the rate is
trading bias for variance inside a fixed budget. **The real dataset is not
optional in this experiment**, because section 4 shows this generator disagrees
with it about tree shape and there is no reason to assume it agrees about rate.

### R3. Make MVS at 0.8 the `lossguide` default, after one measurement, and widen its reach first

**The change.** `bootstrap_type = MVS`, `subsample = 0.8` under `lossguide`.

**Why third, and why it is now a split decision by device.** Section 2 says the
whole gap is variance, and subsampling is the textbook variance reducer. On the
CPU it should also be faster, roughly 20 percent fewer rows in the histogram
accumulation. **On the GPU it is not free and the correction in section 3.5 is
why.** MVS is honored on a GPU fit, but it resolves to
`ROUND_MVS_HOST_MAGNITUDES` and takes the fit off the device round and onto
host-computed gradients. That is a speed cost on the backend this project is
fastest on. It is also still refused by entry point on multiclass and sparse.

**So the recommendation is narrower than "make MVS the default".** It is to make
it the default **under `lossguide` on the CPU** if the measurement supports it,
and leave the GPU default at no sampling until either the device round grows a
compaction step or a measurement shows the accuracy is worth the demotion.
**A default whose cost differs in sign between two backends should not be one
default.**

**Expected size.** Estimate, -3 to -8 percent of excess error. Speed effect,
estimated negative on CPU and positive on GPU, both unmeasured.

**The confirming experiment.** `subsample` in `{1.0, 0.8, 0.6}` at
`bootstrap_type = MVS`, `lossguide`, on `dense_regression` at both tiers and on
YearPredictionMSD, **mojotrees CPU and GPU, both**, three repeats, with wall
clock recorded on a quiet box because this one has a speed claim attached and
because the GPU arm's cost is the thing being decided. **Run it after R1**, so
the sampling is measured on top of whatever `lambda_l2` ships, not underneath
the value it is replacing.

### What is deliberately NOT recommended

**Do not flip the default growth policy to symmetric.** Three independent
reasons, each sufficient.

1. **It loses on real data.** Section 4. On YearPredictionMSD at matched rate,
   symmetric is 1.24 percent worse than leaf-wise and CatBoost is 1.43 percent
   worse than LightGBM.
2. **It costs 2.7x to 4.7x on the GPU, PRE-FLIP.** From the same records that
   carry the accuracy. Standard tier, symmetric GPU 2.238 s against leaf-wise GPU
   0.839 s. Large tier, 17.072 s against 3.659 s, both medians of five repeats in
   run `bench/real_data/results/20260817T110847Z-dense1mfixed/` at 799,110 rows x
   100 features x 100 trees, symmetric depth 6, 10 threads, comparator
   `stock+det@v2`. On the CPU it is 1.2x to 1.7x. These are from runs whose
   quiet-box status is not established and should be read as ratios, not as
   absolute seconds.

   **The 4.7x is measured on a symmetric arm that no longer ships, and the
   corrected ratio is UNRESOLVED.** Later on 2026-08-17 four GPU switches that
   reach only the symmetric path flipped to default on, so the numerator of this
   ratio moved and the denominator did not. **No post-flip large-tier run with
   both arms in one window has been filed**, and the post-flip symmetric
   absolutes in circulation, around 10.36 s and around 9.8 s, come from source
   comments and a commit message rather than from an artifact, and were taken
   against a different baseline in a different window, so they may not be divided
   into 3.659 s to manufacture a new ratio. See
   `bench/results/FIGURE_PROVENANCE.md`. What still stands is the sign, and one
   bound. Within the single window where both arms were read together, symmetric
   was 4.666x slower, and the largest speedup the flips have been credited with
   anywhere is the 2.20x combined arm, so dividing one by the other leaves about
   **2.1x, and that is an ESTIMATE across two windows rather than a measurement**,
   offered only to show that the recorded numbers do not reach a reversal. Reason
   1 and reason 3 do not depend on the multiple at all. **What is no longer
   supportable is the specific figure 4.7x**, and the sentence "no plausible
   correction closes 4.7x" was removed from this item on 2026-08-17 evening
   because a correction of exactly that kind had already landed.
3. **The mechanism it would buy is available more cheaply.** The gap is
   variance. R1 and R3 attack variance at zero and negative cost.

**Do not enable `random_strength` under `lossguide`.** The only contrast this
lane could find says it costs 2.5 to 4 percent of excess error, and that
contrast is confounded. There is no evidence for it and one weak signal against.

**Do not implement `GreedyLogSum` or any other border rule.** Section 3.10 in
full. The short form is that the term is large, it reverses sign between the two
tiers, LightGBM's implementation of our own rule is the worst arm on it at both
tiers, and the whole thing hangs on one synthetic discontinuity that the real
dataset has no analogue of. **That is a bit-moving change to every fit,
re-anchored against our own accuracy gate, in exchange for a different ticket in
the same lottery.** The generalizable version, snapping a border to a detected
discontinuity in the empirical CDF, is logged in 3.10 as an unpriced candidate
and must be measured on real data before it is built.

---

## 6. The mirroring rule, engaged directly

The rule, quoted from `src/mojotrees/params.mojo`
`_apply_catboost_mode_defaults`, is

> `grow_policy=oblivious` is CatBoost's symmetric tree and mirrors CatBoost
> exactly, `lossguide` mirrors LightGBM, and anything of our own is opt-in and
> named as ours.

**R1 breaks it.** LightGBM's stock `lambda_l2` is 0.0 and R1 ships a nonzero
value under `lossguide`. R2 would break it too if it ended in a rate other than
0.1. R3 breaks it, since LightGBM's `bagging_fraction` default is 1.0.

The argument for breaking it, stated as an argument and not as a preference.

**First, the rule is already not what the code does, in the direction that
matters most.** `docs/LIGHTGBM_PARITY.md` section "Decided 2026-08-16" states
that `symmetrictree` is "the default policy" and that the shipped tree budgets
are 72 under `lossguide` and 360 under `symmetrictree`. **None of that is in the
code.** `python/mojotrees/sklearn.py` defaults `grow_policy="lossguide"` and
`n_estimators=100`; `boosting.BoosterParams.default()` returns
`(100, 0.1, TreeParams.default())`; `tree.grow_tree` defaults
`grow_policy = GROW_LEAFWISE`. The same doc already carves the tree budget out
of the mirror explicitly. So the shipped position is that the mirror covers
per-parameter defaults and not the budget, and the documented position is a
different set of defaults again. **A rule that the documentation and the code
already disagree about is not a rule a decision has to be smuggled past.**
Section 8 records this as a discrepancy rather than fixing it, because a lane
may be mid-flight on it.

**Second, the mirror's explainability value is real and R1 does not spend much
of it.** The value is that a user who knows LightGBM can predict what we will
do. `lambda_l2` is a scalar with an identical meaning in all four engines and it
is the second most-tuned parameter in gradient boosting. A user surprised by
`lambda_l2 = 1.0` reads one line of release notes and is unsurprised. That is a
much smaller cost than a user surprised by a different tree shape, which is why
the "do not flip growth" recommendation stands even though growth is where the
biggest number is.

**Third, and this is the load-bearing part, the mirror is currently importing a
defect.** `lambda_l2 = 0.0` is not a considered choice of ours. It arrived
because LightGBM ships it, and
`bench/results/cpu_float32_lambda0_2026-08-16/RESULTS.md` records that when it
arrived it cost us 25 percent of average precision on `imbalanced_binary` and
3.3x on multiclass logloss, **more than it cost LightGBM on the same data**.
That last clause is the whole argument. A mirror is a good default when the two
implementations respond to a parameter the same way. Here they demonstrably do
not, and that file's own reading is that "LightGBM has something in that path
that we do not". **Mirroring a parameter across an unmatched implementation is
not fidelity, it is copying a number out of context.**

**The narrowest honest form of the rule, proposed.** Mirror the mirrored
engine's default **except where a recorded measurement on our own code shows
the mirrored value costs accuracy on our implementation**, and require the
exception to name the artifact. That keeps every unmeasured default at the
mirror, which is where the explainability comes from, and it lets three
recorded scenarios of damage overrule one line of `config.h`. Under that rule R1
qualifies today, R3 does not qualify yet and R2 does not qualify until it is
measured. That ordering is not a coincidence; it is the rule doing its job.

---

## 7. The categorical verdict. Still live, cause confirmed quantitatively, and the fix is not the one on the roadmap

### 7.1 It is live, and it has not moved

`bench/real_data/results/20260817T102725Z-cat1m/`, taken 2026-08-17 at 06:29,
`high_cardinality_categorical` at 799,110 train x 15 features, five repeats.

| arm | AUC | average precision | logloss |
|---|---|---|---|
| LightGBM | 0.865957 | **0.479591** | 0.231176 |
| CatBoost | 0.845317 | 0.440563 | 0.242530 |
| CatBoost, lossguide | 0.843789 | 0.434240 | 0.243733 |
| **mojotrees** | 0.841028 | **0.421925** | 0.245947 |

- The AP gap against LightGBM is **12.02 percent** against a 10 percent limit.
  **FAIL, unchanged to the byte from the 2026-08-16 record.**
- The AUC gap is 0.024929 absolute against a 0.05 limit. **PASS.**
- **CatBoost also loses to LightGBM here, by 8.1 percent of AP.** LightGBM is
  the outlier on this scenario, not us. Our gap to CatBoost is 4.2 percent.

### 7.2 The cause, verified from the artifact rather than argued

The per-feature bin counts are in the records and nobody appears to have read
them. Columns 10 to 14 are the categoricals at cardinalities
`(8, 64, 1000, 20000, 200000)`.

| arm | bins on the five categorical columns |
|---|---|
| mojotrees | 9, 65, **255, 255, 255** |
| LightGBM | 9, 65, **989, 19433, 15952** |
| CatBoost | 1, 1, 1, 1, 1 (they become CTR numeric columns) |

Verified structural cause, read off the source. A bin index is a byte
(`BinnedMatrix.bins` is `List[UInt8]`, `binning.mojo::BinnedMatrix`),
`binning.mojo::MAX_BINS` is 256, and a categorical node's split set is a fixed
four-word bitset (`categorical.mojo::CAT_MAX_BINS`, `64 * CAT_BITSET_WORDS`).
`categorical._keep_most_frequent` keeps `max_bins - 1 = 254` codes by descending
count and drops the rest into `UNKNOWN_BIN`. So the three wide columns are
resolved at **26.5 percent, 1.8 percent and 10.8 percent of rows** where
LightGBM resolves **98.9, 98.1 and 44.5 percent**. Those coverages are computed
from the generator's own draws.

### 7.3 How much of the gap the ceiling explains. Computed, not argued

The scenario's target is an explicit sum of per-level effects, so an **oracle**
can be built for any binning. Give every resolved level its exact effect, give
every dropped level the count-weighted mean effect of the pool it fell into, keep
everything else perfect, and score it. That is the best any model could do given
the representation. Fitted on the train split, scored on the test split, using
the harness's own `quality.average_precision` and `quality.auc`.

| | AP | AUC |
|---|---|---|
| oracle, full latent (Bayes ceiling) | 0.563319 | 0.895411 |
| oracle at LightGBM's bin counts | **0.552770** | 0.892220 |
| oracle at our 254 bins | **0.442548** | 0.850736 |
| oracle with the three wide columns contributing nothing | 0.419854 | 0.842461 |
| measured, LightGBM | 0.479591 | 0.865957 |
| measured, mojotrees | 0.421925 | 0.841028 |

**The bin ceiling costs 0.110222 of oracle AP, 19.9 percent relative. The
measured gap is 0.057666, 12.02 percent relative. The ceiling more than fully
accounts for the observed failure and nothing else needs to be invoked.**

Two further readings from the same table.

- **Our measured 0.421925 sits almost exactly on the "wide columns contribute
  nothing" oracle of 0.419854**, and far below our own 254-bin oracle of
  0.442548. We are realizing roughly 9 percent of the headroom our 254 bins
  make available. This is consistent with a ceiling that leaves the retained
  levels individually rare and individually unprofitable to split on, and it is
  not by itself evidence of a second defect, because LightGBM realizes only 87
  percent of its own much larger oracle.
- **The interaction term is not implicated.** The double-centered joint effect
  is over categorical columns 0 and 1, at cardinalities 8 and 64, which we bin
  exactly at 9 and 65. Nothing about the interaction is lost to the ceiling.

### 7.4 What this says about the fix, and it is not what the roadmap says

`docs/LIGHTGBM_PARITY.md` names an ordered target statistic as the remedy, on
the structural ground that a CTR reads the raw code and so has no bin ceiling.
That ground is correct. **The size of the prize is not what the roadmap
implies, and there is a measured counterexample sitting in the same table.**

**CatBoost runs CTRs on this scenario, with `max_ctr_complexity 1` and
`one_hot_max_size 2` read back from its own resolved parameters, and it scores
0.440563. That is 8.1 percent behind LightGBM's 0.479591 and it is barely above
our own 254-bin oracle of 0.442548.** A CTR turns a wide column into one numeric
column of 255 bins, which resolves every level but resolves it as a noisy
one-dimensional statistic; thousands of set-splittable bins resolve fewer levels
but resolve them as a partition the tree can cut arbitrarily. On this generator
the second is worth more.

So, ranked for the categorical failure.

1. **A CTR default under `lossguide` will not close this gate.** Estimate, from
   CatBoost's measured 0.440563 as the best available proxy for what a
   well-implemented CTR achieves here, it moves us from 0.421925 to somewhere
   near 0.44, which is 8 percent behind LightGBM and **still fails a 10 percent
   relative gate only narrowly if at all**. It is worth doing on its own merits.
   It is not the fix for this row. Note also that our shipped `lossguide`
   default sends `ctr = "off"` (`python/mojotrees/sklearn.py`, `ctr_rule` is
   `"off"` on every policy but `symmetrictree`), which is why the 2026-08-17
   rerun reproduced 0.421925 to the byte after the CTR raw-code lane landed. The
   CTR fix landed and this scenario never called it.
2. **Widening the categorical bin ceiling is the fix, and the parity doc already
   prices the work.** `UInt16` bins plus an eight-word or growable set would take
   us to 65,535, which covers columns 12 and 13 outright. The parity doc's own
   objection, that 200,000 exceeds `UInt16` anyway, is answered by the oracle
   table. At LightGBM's own 15,952 bins on that column the oracle is 0.552770,
   so **matching LightGBM does not require exceeding `UInt16`**, because
   LightGBM does not exceed it either on the column that motivated the
   objection. The remaining coverage on the 200,000-level column is 44.5 percent
   against our 10.8, and both are far from complete.
3. **A rarity break is worth measuring and is nearly free.** LightGBM's
   categorical admission loop is `while (used_cnt < cut_cnt || num_bin_ <
   max_bin)` at 99 percent row coverage, and it separately drops categories
   under `min_data_in_bin`. That second rule is why its 200,000-level column
   gets 15,952 bins and not more. **Spending our 254 slots by LightGBM's rule
   rather than by pure descending count is a change inside
   `categorical._keep_most_frequent` with no ABI, no bitset and no storage
   consequence.** UNKNOWN whether it helps, because the two rules coincide when
   the budget binds hard, which it does here.

**The confirming experiment for the whole row**, one window, three arms
interleaved on `high_cardinality_categorical` at the standard tier, three
repeats. The arms are our shipped `ctr="off"`, ours at `ctr="auto"`, and LightGBM. Primary
readout average precision against the 10 percent relative gate. That single run
settles whether a CTR default closes the gate and it retires the withdrawn
figure the parity doc is currently carrying an empty hole for.

---

## 8. Corrections to standing claims

**8.1 The categorical bin counts are wrong in two files and are correctable
from the artifact.** Both `bench/results/COMPARISON_RUN_2026-08-16.md` and
`docs/LIGHTGBM_PARITY.md` say LightGBM runs "roughly 991 and 19,801" categorical
bins against our 254. The record carries the actual counts. LightGBM runs
**989, 19,433 and 15,952** on **three** columns, not two, and our side is 255
bins holding 254 codes. Both files have been corrected. The substantive claim
they attach to the number is unaffected and is strengthened by section 7.3.

`bench/real_data/thresholds.json` carries a third version of the same error,
"254 bins against roughly 100,000 on one column and 254 against 20,000 on
another". LightGBM never reaches 100,000 bins on this scenario; it reaches
15,952 on the 200,000-level column. **That file is under `bench/real_data/` and
belongs to another lane, so it is reported here and not edited.**

**8.2 The documented shipped defaults are not the shipped defaults.**
`docs/LIGHTGBM_PARITY.md`, "Decided 2026-08-16", states that `symmetrictree` is
"the default policy" and that the tree budgets are 72 under `lossguide` and 360
under `symmetrictree`. Verified against the code on 2026-08-17:
`python/mojotrees/sklearn.py` defaults `grow_policy="lossguide"` and
`n_estimators=100`; `_LAMBDA_L2 = 0.0` and `_LEARNING_RATE = 0.1`;
`boosting.BoosterParams.default()` is `(100, 0.1, TreeParams.default())`;
`tree.grow_tree` defaults `grow_policy = GROW_LEAFWISE`. **Reported and not
edited**, because a run in this session's window is named `postflip` and a lane
may be landing exactly this. Whoever owns it should reconcile the two.

**8.3 The `mojotrees_depthwise` arm changed identity mid-day and the two
identities are in adjacent result directories.** Recorded in section 3.1. The
arm's own `MOJOTREES_DEPTHWISE_CLAIMS` string in `scenarios.py` already says
this correctly; this note exists so that a reader of the two result sets does
not build the series anyway.

**8.4 A premise about `random_strength` needed refining rather than
correcting.** Section 3.7. The CPU-only guard in `params.mojo` is real and
applies only to the mode default, which the harness bypasses by naming the
parameter. The reason a GPU symmetric fit may have run unnoised is a different
one, it was measured and fixed on 2026-08-17, and whether the fix was in the
binary for a given run is UNKNOWN.

**8.5 `CATBOOST_UNMATCHABLE["row_sampling"]` is stale on the accelerator.** It
reads as though a CatBoost-mode arm on the GPU refuses the bootstrap bundle.
`src/mojotrees/train_gpu.mojo` documents the opposite in bold around line 960
and the symmetric GPU arm ran five repeats today without raising. MVS routes to
the host-gradient arm and the trees still grow on the device, so **MVS is matched
on both backends and is not an explanation for either the peer gap or the
CPU-against-GPU divergence**. Section 3.5 carries the correction and its
consequence for R3. **That file is under `bench/real_data/` and belongs to
another lane, so it is reported here and not edited.**

**8.6 `CATBOOST_UNMATCHABLE["border_placement"]` is accurate as written and its
importance was understated by everyone, including this document's first draft,
which estimated it at under 0.5 percent.** It is worth 32.7 percent of the
standard-tier gap. The entry's own words, "open and unclosable by any
parameter", are correct about parameters and were read as though they also meant
small. Section 3.10 prices it and recommends against acting on it, for reasons
that are about the direction of the effect rather than its size.

---

## 9. What could not be determined

1. **The learning rate's effect on our `lossguide` arm.** The largest unpriced
   term in the document. No run has ever varied `learning_rate` on any mojotrees
   arm. R2 is the experiment.
2. **How CatBoost's shipped configuration performs on YearPredictionMSD.** The
   only real-data run with all arms, `20260816T180302Z-decision`, pinned every
   arm at lr 0.1, so it measured `cb-default` and not `cb-shipped`. CatBoost's
   own rate on that dataset would be the 0.5 cap. **Section 4's reversal is
   therefore a matched-rate result and does not establish that CatBoost's
   shipped configuration loses on real data.** One rerun of that cell against
   `cb-shipped` resolves it, and it is the single highest-value missing
   measurement in this whole area.
3. **The within-bundle attribution of the 25.2 percent.** Twelve keys move at
   once. Three of them are unpriced individually and no isolation exists for any
   of `subsample`, `score_function`, `num_leaves 64 / max_depth 6`,
   `min_data_in_leaf 1` or `min_child_hess 0.0`.
4. **Whether `random_strength` helps or hurts.** The only contrast is confounded
   with the fixed-point histogram divergence and with an extension rebuild.
5. **Whether the `postflip` run's GPU symmetric arm applied `random_strength`
   and Cosine.** `gpu_resident_round.mojo` was modified after that run started.
   The records carry no resolved-on-device parameter read-back, which is the
   thing that would answer it. A `dump_model` read-back of the settings a GPU
   fit actually applied, recorded beside the CatBoost read-back, would close
   this class of question permanently.
6. **Whether overfitting to high-order noise-feature interactions is the
   variance mechanism.** Section 2 rules out first-order dependence and cannot
   see the rest. A run with the 94 noise columns deleted, against the same 100
   trees, would separate "the model is fitting noise features" from "the model
   is fitting the noise in the label".
7. **Whether `ordered_boosting_small` says anything.** The scenario is defined in
   `scenarios.py` and has **never been run**. No records directory contains it.
8. **Whether a discontinuity-snapping border refinement helps on real data.**
   Section 3.10's one generalizable candidate. On this generator it is worth up
   to a factor of 39 in the binning floor; on a target with no atom in any
   feature marginal it is worth nothing. The measurement is a binning-floor
   oracle computed on YearPredictionMSD's 90 columns, which needs no fit at all
   and which this lane did not have the real feature matrix loaded to do.
9. **Whether MVS on the GPU is worth losing the device round.** Section 3.5. The
   accuracy side is unmeasured and the speed side is confounded with the
   oblivious level path in the only arm that carries both.
10. **Anything about the speed numbers quoted in section 5.** They come from runs
   whose quiet-box status is not established and at least one comparison run in
   this repository was taken on battery. They are used only as ratios and only
   where the ratio is large.
