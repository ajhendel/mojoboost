# CatBoost mechanism catalog

> **Design draft from source reading. Nothing here is measured.** Every claim
> is a reading of CatBoost's source or documentation, not a benchmark and not
> an implementation report. Items marked **verify** have NOT been checked
> against CatBoost source and must be before any lane builds them.
>
> Split from a single reviewer draft; `OBLIVIOUS.md` is Part B of the same
> document. Each lane updates its own entry here before writing code, marked
> "verified from source" or not.

# CatBoost catalog, and an oblivious grow policy for mojotrees: design draft

Status: design draft, 2026-08-16, written by the review session from reading
CatBoost source (cited per item) and LightGBM/XGBoost source read earlier
today. Nothing here is implemented, run, or measured on mojotrees. Every
number is estimated or a derived bound unless it says measured. Items marked
"verify" were written from the CatBoost paper and docs and must be checked
against source before a lane builds them.

Sources read: `catboost/cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp`,
`.../split_properties_helper.cpp`, `catboost/private/libs/algo/greedy_tensor_search.cpp`
(all `master`, Aug 2026). Comparator for anything in this document that
changes the model: CatBoost at its defaults on the same Mac (CPU-only there),
reported beside LightGBM `stock+det`; never instead of it.

## Part A. Catalog: what CatBoost does, whether we should, and who owns it

| # | CatBoost mechanism (source) | What it is | Moves bits? | Ours to take? | Owner | Rule |
|---|---|---|---|---|---|---|
| A1 | Per-level physical row reordering (`MakeSplit`: "segmented sort; segmented gather for each stats + indices; update part offsets and sizes") | After each level's split, rows are physically sorted so every leaf's rows are contiguous; gathers coalesce | No (order-preserving version) | Yes, GPU, if the histogram kernel is gather-bound | GPU | (2) trade: a sort per split vs coalesced reads; behind a switch, A/B after the decomposition probe |
| A2 | Small-leaf-first with sibling subtraction (`if (firstLeaf.Size < secondLeaf.Size)`; `SubstractHistograms`) | Build the smaller child, derive the larger | — | Already ours, both backends | — | — |
| A3 | `random_strength` (`scoreStDev = RandomStrength * derivativesStDevFromZero * modelSizeDecrease`; `SetBestScore(randSeed + taskIdx, ...)`) | Seeded noise added to candidate scores before argmax; a regularizer LightGBM lacks | Only when > 0 | Yes, all growth modes, default 0 | CPU (`split.mojo`), GPU search kernel for parity | (3) when on; deterministic under the seed and across workers |
| A4 | Bayesian bootstrap / `bootstrap_type` (**verified from source**, see A4 note) | Row weighting instead of row dropping: every row kept, each given weight `(-log(U + 1e-100)) ** bagging_temperature`, redrawn once per tree | Only when on | Yes for CatBoost mode (their default sampler). Built in `sampling.mojo`, off by default. **Device cost, checked:** per-row weights disable the constant-hessian plane (`round_has_constant_hessian`), so the device accumulates three planes and the Int16 gradient-staging arm loses its better half (staged bytes per visit 7 -> 9 at the default group, not 9 -> 7). Not free on the device even with no device code written. | CPU (`sampling.mojo`) | (3) when on; **carries a constant-hessian exclusion**, see the note |
| A5 | Ordered target statistics for categoricals (CTRs; "verify" against `catboost/private/libs/algo/` CTR code) | For each of several random permutations, a categorical value in row i is replaced by (sum of targets of earlier rows with that category + prior) / (count + 1); plus counters and pairwise combinations; test time uses all rows | Yes (feature values change) | Yes; it is CatBoost's real accuracy edge on high-cardinality columns and independent of tree shape | CPU (`binning.mojo`, categorical, dataset) | (3); design section B2 first, then lanes |
| A6 | `leaf_estimation_iterations` (`gradient_walker.h::FastGradientWalker`; defaults in `catboost_options.cpp::GetEstimationMethodDefaults`) -- **verified from source** | Re-estimate leaf values 1..k times, derivatives recomputed at the current leaf value each pass | Only when > 1 | Built on the CPU, opt-in, default 1. **DEVICE: not cheap, structural** (GPU orchestrator, 2026-08-16): each extra iteration needs per-leaf grad/hess sums after the previous raw-score update, so a second reduction per leaf per iteration inside the tree (more launches, against the oblivious command-buffer budget) or leaf estimation moved out of the device round (more host trips). **Correction:** an earlier version of this row said CatBoost defaults to 10 for logloss AND multiclass. Logloss is 10; **multiclass is 1**. The 10 in that block is the unreachable Gradient slot while the default method is Newton, and CatBoost's own docs agree. | CPU built; device design decision before any device lane | (3) when > 1; see A6 notes |
| A7 | Ordered boosting (`BodyTailArr`, tail derivatives only) | Derivatives for row i from a model that never saw i | Yes | No for now; large machinery, matters most on small data | — | design note only |
| A8 | Symmetric (oblivious) trees (`numScoreBlocks = 1` for `SymmetricTree`; leaf index = split-condition bits; depth default 6) | One split per level for all leaves | New mode | Yes, opt-in `grow_policy=oblivious`, both backends | CPU search+schedule; GPU cross-leaf reduce + level partition | Part B. **CPU half BUILT.** Shape/numbering/aggregation **verified from source**; the per-leaf min-child rule is **NOT verified, it is ours** (see below) |

## A8, verified from source

Status: the CPU half of A8 is implemented on `lane/oblivious-cpu`
(`growth_policy.GROW_OBLIVIOUS`, `split.find_best_split_shared`,
`tree._grow_oblivious_levels`). The lane read CatBoost `master` (Aug 2026)
before building.

**Marking, and read this before quoting anything below.** The tree SHAPE, the
leaf numbering, the cross-leaf aggregation and the parameter defaults are
**verified from source** and each cites the file it came from. The per-leaf
min-child rule is **NOT verified and cannot be**, because the parameter does
not exist in CatBoost's symmetric mode; it has its own heading below and is
marked as our decision. Three further corrections to the draft follow.

Files read:

- `catboost/private/libs/algo/greedy_tensor_search.cpp`
  (`GreedyTensorSearchOblivious`, `GreedyTensorSearchDepthwise`,
  `FindBestCandidate`, `SelectBestCandidate`, `CalcScoreStDev`)
- `catboost/private/libs/algo/scoring.cpp` (`UpdateScores`, `UpdateSplitScore`)
- `catboost/private/libs/algo/score_calcers.h` / `.cpp`
  (`TL2ScoreCalcer`, `TCosineScoreCalcer`)
- `catboost/private/libs/algo_helpers/online_predictor.h` (`CalcAverage`)
- `catboost/private/libs/algo/index_calcer.cpp`
  (`splitWeight = 1 << depth`, `GetRedundantSplitIdx`)
- `catboost/private/libs/algo/tensor_search_helpers.cpp` (`SetBestScore`)
- `catboost/private/libs/algo/rand_score.h`
- `catboost/cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp`
  (`numScoreBlocks`, `IsTerminalLeaf`)
- `catboost/cuda/methods/kernel/score_calcers.cuh`
- `catboost/libs/model/model.h`
- `catboost/private/libs/options/oblivious_tree_options.cpp`,
  `catboost/private/libs/options/catboost_options.cpp`
- `catboost/python-package/catboost/core.py` (`min_data_in_leaf` doc contract)

**Confirmed.** One split per level applied to every leaf, bounded by
`for curDepth < MaxDepth` ending in one `currentSplitTree.AddSplit(bestSplit)`;
leaf index is the bit pattern (`splitWeight = 1 << splitParams.Depth`);
`depth` defaults to 6, `l2_leaf_reg` to 3.0, `random_strength` to 1.0,
`grow_policy` to `SymmetricTree`; `numScoreBlocks = 1` for `SymmetricTree`
against `leavesToVisit.size()` for the other policies, which is the GPU
statement of "one score per candidate, not one per leaf"; the score is
accumulated with the leaf loop OUTSIDE and the candidate index inside
(`UpdateScores`: `for leaf ... CalcScoresForLeaf(... updateSplitScoreClosure)`);
the model is stored as `TreeSplits` / `TreeSizes` / `TreeStartOffsets` plus a
`[treeIndex][leafId]` leaf-value table.

### The per-leaf min-child rule is OURS. NOT verified from CatBoost source.

**There is nothing in CatBoost to verify it against, and this row must never
be read as saying otherwise.** `OBLIVIOUS.md` B2 said, marked *verify*, that
"CatBoost scores leaves that fail as zero contribution". That sentence has no
referent. CatBoost's `SymmetricTree` has no `min_data_in_leaf`: its own
documentation scopes the parameter to Depthwise and Lossguide
("This parameter is used only for Depthwise and Lossguide growing policies",
`catboost/python-package/catboost/core.py`), the CPU code reads it only in
`GreedyTensorSearchDepthwise` and `FindBestCandidate` where it gates whether
an *already existing* leaf is expanded rather than whether a candidate's
*child* is admissible, and the CUDA searcher switches even that off for
symmetric trees (`IsTerminalLeaf`:
`const bool checkLeafSize = Options.Policy != EGrowPolicy::SymmetricTree`).
CatBoost also has no `min_sum_hessian_in_leaf` / `min_child_weight` parameter
at all (`min_child_samples` is a synonym for `min_data_in_leaf` and nothing
else). So CatBoost never had to define what happens when a leaf fails these
tests, and we are choosing a behavior they had no need to define.

**The rule we chose:** a leaf that fails `min_data_in_leaf` or
`min_child_hess` at a candidate contributes zero to that candidate's summed
gain and does not veto it. **Why this one, and it is not a parity argument:**
the GPU device implements zero-contribution-without-veto, and host and device
must grow the same tree. Everything else is corroboration rather than
justification -- vetoing would let one narrow leaf of sixteen decide a whole
level, and CatBoost's nearest analogue is a zero *guard* rather than a zero
*penalty* (`CalcAverage` returns 0 for a child of zero weight, so an empty
child contributes nothing and the candidate survives). Do not cite a CatBoost
file for this rule. There is no such file.

The count of leaves that contributed zero at the chosen candidate is recorded
(`growth_policy.SharedSplitAudit`, and `MOJOTREES_OBLIVIOUS_TRACE=1` for a
line per level), because a split that was legal for one leaf of sixteen and
one that was legal for all sixteen are different objects.

### Corrections 2 to 4, which ARE from source

**Correction 2.** The summed score is not a sum of *gains*. `TL2ScoreCalcer`
sums `sumDer^2 / (sumWeight + l2)` over every child of every leaf and never
subtracts a parent term; the parent term comes off once per level as
`gain = score - scoreBeforeSplit` in `SelectBestCandidate`. Summing per-leaf
gains (what we do, and what LightGBM's formula spells) differs from that by
the sum of the per-leaf parent scores, which is constant across the
candidates of one level, so the two have the same argmax. Separately,
CatBoost's CPU **default** score function is `Cosine`, not `L2`, and Cosine is
not a sum at all: it is `Scores[i][0] / sqrt(Scores[i][1])` over two
cross-leaf accumulators. A10 already says we keep LightGBM's gain; this row
now says explicitly that doing so puts us on CatBoost's `L2` and not on its
default.

**Correction 3.** B3 said `num_leaves` "is ignored under oblivious and says
so". CatBoost does not ignore it: `catboost_options.cpp` overwrites
`MaxLeaves` with `1 << MaxDepth` when it is defaulted and raises
`"max_leaves option works only with lossguide tree growing"` when the user set
anything else. We ignore it because `TreeParams` cannot tell a defaulted
`num_leaves` from an explicit one, and we say so in `growth_policy.mojo`,
`TreeParams`, and `_check_oblivious`. Divergence, recorded.

**Correction 4.** B2's tie rule ("ascending feature then bin, as today") is
not CatBoost's default behavior. With `random_strength` at its default of 1.0,
noise is drawn TWICE, once in `SetBestScore` (best bin within a candidate) and
again in `SelectBestCandidate` (best candidate across features), so exact ties
effectively do not occur. Only at `random_strength = 0` does it degenerate to
first-in-iteration-order under a strict `>`, which is our rule. Our default is
`random_strength = 0`, so we are on the degenerate branch by default and B2's
statement is right for that branch and wrong as a description of CatBoost.

**Not implemented, deliberately.** `greedy_tensor_search.cpp` deletes a level
whose every leaf produced an empty child and stops growth
(`GetRedundantSplitIdx(GetIsLeafEmpty(...))`). That rule cannot fire under our
search: a chosen split needs a strictly positive summed gain, and a leaf with
an empty child contributes either zero (it failed a minimum) or exactly 0.0
(both minima off, and the candidate's two terms then equal the parent term),
so a level in which every leaf had an empty child sums to 0.0 and is never
chosen. Building it would be building a gate that cannot open.
| A9 | `Depthwise`, `Lossguide` (priority queue by gain, `MaxLeaves`) | Their versions of ours | — | Already ours | — | — |
| A10 | `score_function` (`Cosine` is the CPU default for **every** grow policy, `L2`, and the GPU-only `SolarL2`/`NewtonL2`/`NewtonCosine`/`LOOL2`/`SatL2`) -- **verified from source**, see the A10 note | Split scoring alternatives. Cosine is a RATIO of two cross-leaf accumulators, not a sum | Only when set to Cosine | Built on the CPU, opt-in, default `SCORE_L2` (= today's LightGBM gain). **Headline finding: at `lambda_l2 = 0` Cosine is provably a no-op on the argmax** -- it degenerates to `sqrt` of the L2 sum. Its entire difference from L2 is a function of `lambda_l2`, which mojotrees stock now sets to 0 | CPU (`split.mojo`) | (3) when set to Cosine; see the A10 note |
| A11 | MVS, `bootstrap_type=MVS` (**verified from source**, see the A11 note) | Minimal Variance Sampling: gradient-magnitude-weighted Poisson sampling with an inverse-probability weight, so large-gradient rows are kept certainly at weight 1 and small-gradient rows are kept with probability `g/mu` at weight `mu/g`. **This, not Bayesian, is what CatBoost actually runs on the CPU** for every objective we benchmark. Redrawn once per tree at the `PerTree` default. | Only when on | Yes. Extends A4's module, opt-in, our default does not move. **It is a row *dropper*, not only a reweighter**: a zero weight removes the row from CatBoost's score fold outright (`SetControlNoZeroWeighted`), so it touches fewer rows than a full pass. That is a speed side effect this lane does NOT claim and did NOT measure. **Carries A4's constant-hessian exclusion** for the same reason A4 does. | CPU (`sampling.mojo`, beside A4) | (3) when on |

### A4 note: Bayesian bootstrap, verified from source

Status: **verified from CatBoost source**, `master`, 2026-08-16. Files read:

- `catboost/private/libs/algo/tensor_search_helpers.cpp` -- the draw
  (`GenerateBayessianWeight`), the per-tree fill (`GenerateRandomWeights`),
  the product with the user's weights (`CalcWeightedData`), and the dispatch
  over `bootstrap_type` (`Bootstrap`).
- `catboost/private/libs/options/bootstrap_options.h` and `.cpp` -- the option
  surface, the defaults, and `TBootstrapConfig::Validate`.
- `catboost/private/libs/options/oblivious_tree_options.cpp` -- the
  `sampling_frequency` default.
- `catboost/private/libs/algo/greedy_tensor_search.cpp` -- the `DoBootstrap`
  call sites, which is where per-tree versus per-level is decided.
- `util/random/common_ops.h` -- `GenRandReal1`'s range.

What that settled:

1. **The draw.** `GenerateBayessianWeight` is
   `powf(-FastLogf(rand.GenRandReal1() + 1e-100), baggingTemperature)`: an
   exponential(1) variate raised to the temperature. At the default
   temperature of 1.0 the weights are exactly exponential(1), mean 1.
2. **`bagging_temperature` defaults to 1.0** and `Validate` requires `>= 0`.
   **Zero is not "off"**: `GenerateRandomWeights` early-returns filling every
   weight with 1, which is the unbootstrapped model. Larger temperatures
   stretch the distribution, so the regularizer strengthens as it rises.
3. **`subsample` is refused beside it** ("bayesian bootstrap doesn't support
   'subsample' option"): every row is kept, so there is no fraction.
4. **Once per tree** under the default `sampling_frequency=PerTree`; the
   per-level schedule exists but is not the default and is not implemented
   here.
5. **The weight is a sample weight.** `CalcWeightedData` multiplies the
   derivatives by `SampleWeights` and then does `SampleWeights[i] *=
   learnWeights[i]`, so the effective per-row weight is the product of the
   draw and the user's weight, and the leaf denominators carry it.

Consequence for mojotrees, which is the part that is not a transcription:
point 5 means **a bootstrapped fit has a per-row hessian**, so it must not
declare `histogram.CONSTANT_HESSIAN`. `boosting.round_has_constant_hessian`
cannot see a bootstrap configuration -- its three inputs are the objective,
the sample weights and the GOSS parameters, and its signature is a GPU-visible
contract no CPU lane may widen -- so the exclusion is expressed on the
sampler's side as `sampling.bayesian_bootstrap_varies_hessian` with
`sampling.check_bayesian_bootstrap_hessian_declaration` as the guard a round
loop must call. Widening the predicate itself is an open cross-campaign item;
see the lane report.

Not implemented: `sampling_unit=Group` (one weight per query group, which
belongs with the ranker), `sampling_frequency=PerTreeLevel`, and the
Bernoulli/MVS/Poisson bootstrap types. `canonical_bootstrap_type` refuses
Bernoulli by name because mojotrees already has that draw as
`bagging_fraction`.

### A6 notes: `leaf_estimation_iterations`, verified from source

Read 2026-08-16 from `github.com/catboost/catboost` at `master`. Files:
`catboost/private/libs/options/catboost_options.cpp`
(`GetEstimationMethodDefaults`, `SetLeavesEstimationDefault`),
`catboost/private/libs/options/oblivious_tree_options.cpp`,
`catboost/private/libs/algo/approx_calcer/gradient_walker.h`
(`FastGradientWalker`, the loop itself),
`catboost/private/libs/algo/approx_calcer.cpp`
(`CalcLeafValuesSimple`, `CalcLeafDersSimple`, `CalcLeafDeltasSimple`),
`catboost/private/libs/algo_helpers/online_predictor.h`
(`TSum`, `ScaleL2Reg`, `CalcDeltaNewtonBody`, `CalcDeltaGradient`),
`catboost/private/libs/algo_helpers/approx_updater_helpers.cpp`
(`NormalizeLeafValues`), `catboost/private/libs/algo/train.cpp`.

**The per-objective defaults.** `GetEstimationMethodDefaults` sets
`defaultNewtonIterations` *and* `defaultGradientIterations` for every loss, and
`SetLeavesEstimationDefault` then picks whichever belongs to that loss's
default `leaf_estimation_method`. The number in the unselected slot is dead
unless the caller overrides the method.

| CatBoost loss | default method | effective default iterations |
|---|---|---|
| `RMSE`, `MultiRMSE`, `Huber`, `Cox`, `QueryRMSE`, `Lq`, `SurvivalAft`, `Focal`, `LambdaMart`, `YetiRank` | Newton | 1 |
| `Logloss`, `CrossEntropy`, `MultiLogloss`, `MultiCrossEntropy` | Newton | **10** |
| `Poisson` | Newton | **10** |
| `PairLogit` | Newton | **10** |
| `Expectile` | Newton | **5** |
| `MultiClass`, `MultiClassOneVsAll` | Newton | **1** |
| `MAE`, `MAPE`, `RMSPE`, `Quantile`, `GroupQuantile`, `MultiQuantile` | **Exact** (via `useExact`) | 1 |
| `LogCosh` | Exact | 1 |
| `Tweedie` | Newton | 1 on CPU, **20** on GPU |
| `QuerySoftMax` | Gradient | **100** |
| `StochasticFilter` | Gradient | **100** |
| `PairLogitPairwise` | CPU Gradient / GPU Newton | CPU **50**, GPU 1 |
| `QueryCrossEntropy` (GPU only) | Newton | **10** |

**The claim we were handed was half wrong.** "10 Newton steps for logloss and
multiclass by default" is right for `Logloss`/`CrossEntropy` and **wrong for
`MultiClass`**, which defaults to 1. The MultiClass block does contain a 10,
but in the `defaultGradientIterations` slot, unreachable while the default
method is Newton. CatBoost's own documentation agrees with the source
("Multiclassification mode -- One Newton iteration"); the folklore does not.
`Poisson` at 10 is the one genuinely surprising row and was not in the claim
at all.

**Mechanism.** `FastGradientWalker` calls `leafUpdaterFunc` then
`approxUpdaterFunc` per iteration; the approx is mutated in place and the next
iteration's `CalcLeafDersSimple` differentiates at the mutated point, so
derivatives are fully recomputed each pass. Leaf *weights* are not: `TSum`'s
`SetZeroDers` clears `SumDer`/`SumDer2` and deliberately leaves `SumWeights`,
which is accumulated only when `recalcLeafWeights = (iterationIdx == 0)`.

**Damping.** There is none inside the loop: no learning rate, no `1/(1+k)`.
The Newton step is `SumDer / (-SumDer2 + l2_leaf_reg * sumAllWeights /
allDocCount)` -- note the L2 is scaled by the *mean sample weight*, which
LightGBM's and ours is not. `learning_rate` is applied exactly once, by
`NormalizeLeafValues`, to the accumulated leaf value after the last iteration,
so raising the iteration count does not compound the rate.

**Line search.** `leaf_estimation_backtracking` defaults to `AnyImprovement`
and is disabled at 1 iteration
(`*haveBacktrackingObjective = leavesEstimationIterations > 1 && ...`), which
also selects the trivial walker branch. On CPU a rejected halving *consumes an
iteration*, so `leaf_estimation_iterations=10` there is between 1 and 10
accepted steps. Armijo is GPU-only. GPU has an explicit `if (Iterations == 1)`
early return.

**Exact.** `useExact` switches `MAE`/`MAPE`/`RMSPE`/`Quantile`/
`GroupQuantile`/`MultiQuantile` to `ELeavesEstimation::Exact` on a single host
without `approx_on_full_history` and without monotone constraints, pinning
both counts to 1; a `CB_ENSURE` refuses `Exact` for any other loss. Exact is a
closed-form weighted-quantile leaf refit -- it is our `_renew_leaf_values` and
LightGBM's `RenewTreeOutput`. **CatBoost reaching for a closed form on exactly
the objectives LightGBM renews is the evidence that leaf renewal and leaf
estimation are one mechanism, not two**: exact where a closed form exists,
iterative where it does not. They therefore compose by exclusion, and
mojotrees refuses the combination rather than applying both.

**What mojotrees built.** `ExtraTreeParams.leaf_estimation_iterations`,
default **1**, honored by `boosting.train`, `boosting.train_more` and
`boosting.train_with_valid` through `boosting._estimate_leaf_values`; refused
by name everywhere else. Newton only, no Gradient method, no line search, and
`lambda_l2` is *not* rescaled by mean weight -- the iteration solves our leaf
problem more exactly rather than changing which problem it is. **The
per-objective CatBoost numbers above are recorded in
`boosting.catboost_leaf_estimation_iterations` and read by nothing; our
default stays 1 for every objective, which is LightGBM stock.**

### A10 note: `score_function`, verified from source

Status: **verified from CatBoost source**, `master`, read 2026-08-16 by
`lane/score-function`. Every claim below cites the file it came from; the two
paragraphs headed "OURS" are our decisions and cite nothing, because there is
nothing to cite.

Files read:

- `catboost/private/libs/algo/score_calcers.h` -- `IPointwiseScoreCalcer`,
  `TCosineScoreCalcer`, `TL2ScoreCalcer`, `MakePointwiseScoreCalcer`.
- `catboost/private/libs/algo/score_calcers.cpp` -- `AddLeafPlain` /
  `AddLeafOrdered` for both calcers.
- `catboost/libs/helpers/short_vector_ops.h` --
  `NSimdOps::UpdateScoreBinKernelPlain` / `...Ordered`, which is what
  `TCosineScoreCalcer::AddLeafPlain` actually runs.
- `catboost/private/libs/algo_helpers/online_predictor.h` -- `CalcAverage`,
  `ScaleL2Reg`.
- `catboost/private/libs/algo/calc_score_cache.h` -- `TBucketStats`.
- `catboost/private/libs/options/oblivious_tree_options.cpp` -- the default
  and the CPU restriction.
- `catboost/private/libs/options/catboost_options.cpp` -- the Lossguide
  default override and its enclosing scope.
- `catboost/private/libs/options/enums.h` -- `EScoreFunction`.
- `catboost/private/libs/algo/scoring.cpp` -- `CalculateNonPairwiseScore`
  (where the regularizer is scaled and set), `UpdateScores` (the monotone
  branch).
- `catboost/private/libs/algo/leafwise_scoring.cpp` --
  `CalcScoresForOneCandidate` dispatch and `CalcScoreWithoutSplit`.
- `catboost/private/libs/algo/greedy_tensor_search.cpp` --
  `SelectBestCandidate`, `GreedyTensorSearchOblivious`'s
  `scoreBeforeSplit`, `CalcBestScoreAndCandidate`.

#### 1. What Cosine computes

`TCosineScoreCalcer` keeps **two** accumulators per candidate, initialized
`{0, 1e-100}` (`score_calcers.h`, `SetSplitsCount`), and its final value is

```
score[i] = Scores[i][0] / sqrt(Scores[i][1])
```

Each child of each leaf contributes through `AddLeaf`:

```
Scores[i][0] += leafApprox * leafStats.SumWeightedDelta
Scores[i][1] += leafApprox * leafApprox * leafStats.SumWeight
```

with `leafApprox = CalcAverage(SumWeightedDelta, SumWeight, scaledL2) =
SumWeightedDelta / (SumWeight + scaledL2)`, and **0 when `SumWeight <= 0`**
(`online_predictor.h`: `count > 0 ? 1./(count + reg) : 0`). `TL2ScoreCalcer`
keeps only the first accumulator and returns it directly. So Cosine's
numerator IS the L2 score, and the whole difference is the division by
`sqrt` of a second accumulator.

Written as a cosine, which is where the name comes from: with `d` the vector
of per-child steps and `g` the vector of per-child derivative sums, the
numerator is `<d, g>` and the denominator is `||d||_W`, so the score is
`<d, g> / ||d||_W` -- the projection of the proposed step onto the gradient,
which is the cosine of the angle between them up to the constant `||g||`.

#### 2. How it differs from L2, exactly, and where `lambda` lives

Translated into mojotrees terms (our leaf output is `out = -G/(H + lambda)`,
so CatBoost's `SumWeightedDelta = -G` and their `SumWeight` sits where our
`H` sits):

```
num = sum_c   -out_c * G_c        =  sum_c  G_c^2 / (H_c + lambda)
den = sum_c  out_c^2 * H_c        =  sum_c  G_c^2 * H_c / (H_c + lambda)^2
cosine = num / sqrt(den)          L2 = num
```

**Set `lambda = 0` and `num` and `den` become the same expression.** Then
`cosine = num / sqrt(num) = sqrt(num) = sqrt(L2)`, and `sqrt` is strictly
increasing on the non-negative reals, so **the two score functions have
exactly the same argmax.** This holds for any number of children, so it holds
for the depth-wise search and for the oblivious level search alike.

Equivalently, and this is the cleaner statement: for a *single* child,
`num_c / sqrt(den_c) = |G_c| / sqrt(H_c)`, in which `lambda` has cancelled
completely. **Cosine is L2 with the regularizer left in the leaf-value
estimate and taken back out of the score normalization.** That is the whole
mechanism.

#### 3. The `lambda_l2 = 0` consequence, which is the point of this row

`bench/results/LANE_RULES.md` asks whether a ratio score function has a
denominator that `lambda` was damping. It does, and the answer runs the
opposite way from the failure that shape usually produces:

- **mojotrees stock is now `lambda_l2 = 0`. At that value Cosine cannot
  change which split is chosen.** It is not a small effect at stock, it is
  the zero effect, provably and not empirically.
- The `gain > 0` admission test is unaffected too: `num > parent_num` if and
  only if `sqrt(num) > sqrt(parent_num)`, so a candidate admitted under L2 is
  admitted under Cosine and vice versa.
- Cosine is therefore reachable as a *difference* only through one of:
  `lambda_l2 > 0`; a leaf-wise (`num_leaves`-bounded) priority queue, where
  gains from **different parents** are compared and `sqrt(a) - sqrt(p)` does
  reorder against `a - p` even at `lambda_l2 = 0`; a non-zero
  `min_gain_to_split`, `feature_contri`, CEGB cost or `random_strength`,
  each of which is an absolute amount or a multiplier applied to a gain whose
  units Cosine has changed.
- The denominator's own small-value hazard is the reverse of the usual one:
  `den -> 0` forces `num -> 0` through the same `SumWeight <= 0` guard, and
  where our L2 gain blows up like `G^2/H` as `H -> 0`, Cosine blows up like
  `|G|/sqrt(H)`, which is *less* singular. The `1e-100` seed on the
  denominator is a divide-by-zero guard, not a regularizer; `min_child_hess`
  is what actually keeps `H` off the floor and it is unchanged.

#### 4. The CPU default really is Cosine, and the grow-policy story

`oblivious_tree_options.cpp:22` -- `ScoreFunction("score_function",
EScoreFunction::Cosine)`. That is the constructor default and it is not
conditioned on anything.

`oblivious_tree_options.cpp:143` -- `CB_ENSURE(TaskType == GPU ||
EqualToOneOf(ScoreFunction, Cosine, L2), "Only Cosine and L2 score functions
are supported for CPU.")`. So on CPU the enum has exactly two reachable
values, and `MakePointwiseScoreCalcer` and
`leafwise_scoring.cpp:530-551` both refuse anything else by name.

**Per grow policy, on CPU:**

| grow policy | score function used | where |
|---|---|---|
| `SymmetricTree` (default) | `score_function`, i.e. **Cosine** by default | `greedy_tensor_search.cpp:685` builds the calcer from `ObliviousTreeOptions->ScoreFunction`; `numScoreBlocks = 1`, so one num/den pair is accumulated across **every leaf of the level** and one ratio is taken per candidate |
| `Depthwise` | `score_function`, i.e. **Cosine** by default | `leafwise_scoring.cpp:530`, one num/den pair per (leaf, candidate) |
| `Lossguide` | `score_function`, i.e. **Cosine** by default | same call site |

**The claim that Lossguide defaults to `NewtonL2` is GPU-only.** The
`ScoreFunction.SetDefault(NewtonL2 / L2)` block at
`catboost_options.cpp:980-991` sits inside `if (TaskType ==
ETaskType::GPU) {` opened at line 949. On CPU that block is not reached, and
`NewtonL2` would be refused by the `CB_ENSURE` above anyway. There is no
per-policy and no per-loss score-function default on CPU.

#### 5. The parent term, which is not what a reader expects

CatBoost never subtracts a per-candidate parent score inside the calcer.
`SelectBestCandidate` (`greedy_tensor_search.cpp:955`) computes
`gain = score - scoreBeforeSplit` and then multiplies by the per-feature
weight. Under `SymmetricTree`, `scoreBeforeSplit` starts at `0.0` for the
root level and is then set to the previous level's winning **score**
(line 1214) -- which is precisely the current level's cross-leaf unsplit
score, because the current level's leaves are exactly the children the
previous split made. Under `Depthwise`/`Lossguide` it is
`CalcScoreWithoutSplit` (`leafwise_scoring.cpp:555`), which runs the same
calcer over the leaf's own totals with an empty second child.

Two consequences:

1. With no feature weights set (the default) `scoreBeforeSplit` is a constant
   across the candidates of one level, so it cancels out of the argmax
   entirely. It matters only for the `bestScore == MINIMAL_SCORE` stop, for
   `feature_weights` / `penalties_coefficient`, and for the leaf-wise queue.
2. **Under Lossguide it does NOT cancel**, because gains from different
   parents are compared against each other. That is the one place where
   Cosine changes the tree even at `lambda_l2 = 0`.

#### 6. What mojotrees built

`split.SCORE_L2` (= 0, the default) and `split.SCORE_COSINE` (= 1), as a new
trailing `score_function` parameter on `split.find_best_split` and
`split.find_best_split_shared`.

**The default path is byte-for-byte the path it was.** Nothing was deleted
from `_split_gain`, which is untouched; the Cosine arithmetic lives in three
new `@always_inline` helpers (`_cosine_out`, `_cosine_unsplit`,
`_cosine_pair`) that the default never calls, behind one `if cosine:` on a
value that is loop-invariant for the whole node. The two extra accumulator
planes the oblivious path needs are allocated with length 0 when Cosine is
off. The golden fixtures are untouched and no bit moves at the default.

Faithful to source: the two-accumulator shape, the `num += -out * G` /
`den += out^2 * H` contributions, the `H <= 0 -> out = 0` guard, the `1e-100`
denominator seed, the use of the *finished and clamped* child output as
`leafApprox` when `max_delta_step` / `path_smooth` / monotone constraints are
active (`scoring.cpp:711-720` calls `AddLeaf` with the monotonized leaf
value), and the parent term computed by the same calcer over the node's own
totals.

Deliberate divergences, each recorded here rather than smoothed over:

- **No mean-weight rescale of `lambda`.** CatBoost uses
  `scaledL2Regularizer = l2_leaf_reg * (sumAllWeights / docCount)`
  (`scoring.cpp:749`, `ScaleL2Reg`). Ours is the raw `lambda_l2`, as A6 already
  records for leaf estimation. Under Cosine this is a *bigger* divergence than
  under L2, because Cosine's entire difference from L2 is a function of
  `lambda`; a mojotrees Cosine run at `lambda_l2 = 3` is not a CatBoost
  `l2_leaf_reg = 3` run unless the mean sample weight is 1.
- **L1 composes.** CatBoost has no `lambda_l1` in the split score. Ours feeds
  the same soft-thresholded `T(G)` into Cosine that it feeds into L2, so the
  two score functions differ only in the ratio and not in the input.
- **Ordered boosting's `AddLeafOrdered` is not implemented.** It computes
  `leafApprox` from `(SumDelta, Count)` while still accumulating against
  `(SumWeightedDelta, SumWeight)`, so under ordered boosting Cosine is *not*
  `sqrt(L2)` even at `lambda = 0`. mojotrees has no ordered boosting (A7), so
  the plain kernel is the only one that applies.
- **Addend order is ours.** `UpdateScoreBinKernelPlain` accumulates the true
  (right) child before the false (left) one and seeds the denominator before
  either. We seed the denominator, then add left, then add right, and in the
  oblivious path fold leaves in ascending `hists` order. Fixed by the loops,
  not by the scheduler, so the value is identical at every
  `MOJOTREES_NUM_WORKERS`.
- **Monotone constraints.** CatBoost *projects* leaf values by isotonic
  regression and scores the projection; mojotrees clamps to the node's
  interval and *rejects* a violating candidate. That divergence predates this
  lane. Under Cosine a rejected or illegal leaf of an oblivious level
  contributes its **unsplit** `num`/`den` terms, which is the exact
  generalization of what the L2 path already does (a leaf contributing `0.0`
  to a sum of `child - parent` differences is arithmetically identical to a
  leaf contributing its parent terms to both accumulators). See below.
- **Categorical features are refused under Cosine.** A categorical
  candidate's set search (`categorical.mojo`) scores partitions with the L2
  gain internally and only its winner reaches `find_best_split`; rescoring
  that winner under Cosine would mix two score functions inside one argmax.
  Refused by name rather than half-applied, exactly as `random_strength` and
  `extra_trees` are.

**OURS, not verified, and not verifiable:** the rule that an illegal leaf of
an oblivious level contributes its unsplit terms. CatBoost's `SymmetricTree`
has no `min_data_in_leaf` and no `min_child_hess` (see the A8 note), so it
never had to define the case. The rule is chosen because it is the unique
generalization that reduces to the existing L2 behavior: under L2,
`sum over legal leaves of (child_score - parent_score)` equals
`(sum over legal of child_score + sum over illegal of parent_score) - sum over
all of parent_score`, identically. Cosine's ratio makes the second spelling
the only one that can be written, and it is the one that agrees with L2 term
for term.

**OURS, and worth a lane of its own:** with Cosine available,
`random_strength`'s noise is finally dimensionally consistent with the score
it is added to. `tree_parameters_extra.mojo` already records that mojotrees
scales the noise by a gradient RMS (CatBoost's `derivativesStDevFromZero`)
and adds it to a gain of units `gradient^2 / hessian`, which is CatBoost's
`score_function=L2` pairing and not the one CatBoost ships. Cosine's value has
units of a gradient. `score_function=cosine` plus `random_strength > 0` is the
first configuration in which mojotrees reproduces CatBoost's *actual* default
regularizer, and A3's useful range of `random_strength` is the documented one
only in that pairing.

#### 7. Supersedes

The A8 note's Correction 2 says Cosine "is NOT this and NOT implemented
here", and `split.find_best_split_shared`'s docstring said the same. Both
were true when written. Cosine is now implemented, opt-in, default off; the
rest of Correction 2 (that our per-leaf-gain sum and CatBoost's `TL2ScoreCalcer`
sum have the same argmax) is unaffected and still holds.

### A11 note: MVS, verified from source

Status: **verified from CatBoost source**, `master`, 2026-08-16. Everything in
this note is marked either **verified from source** with the file it came from,
or **NOT verified, ours** with a heading of its own. Nothing here is measured
and this lane took no timings.

Files read:

- `catboost/private/libs/algo/mvs.cpp` and `mvs.h` -- the whole algorithm:
  `TMvsSampler::GenSampleWeights`, `TMvsSampler::CalculateThreshold`,
  `TMvsSampler::GetLambda`, `GetSingleProbability`,
  `CalculateMeanGradValue`, `CalculateLastIterMeanLeafValue`, and the
  `BlockSize = 8192` member.
- `catboost/private/libs/options/catboost_options.cpp`
  (`SetNotSpecifiedOptionsToDefaults`) -- where MVS becomes the default and
  where `subsample` becomes 0.8.
- `catboost/private/libs/options/bootstrap_options.h` and `.cpp` -- the option
  surface, the constructor defaults, and `TBootstrapConfig::Validate`.
- `catboost/private/libs/options/option.h` -- `TOption::SetDefault`, which is
  what makes the ordering of the defaulting block matter.
- `catboost/private/libs/options/enum_helpers.cpp` --
  `IsMultiClassOnlyMetric`, `IsMultiRegressionObjective`.
- `catboost/private/libs/algo/tensor_search_helpers.cpp` -- the `Bootstrap`
  dispatch, the `performRandomChoice = false` it sets for MVS, and
  `CalcWeightedData`.
- `catboost/private/libs/algo/greedy_tensor_search.cpp` -- the three
  `DoBootstrap` call sites and what `leafValues` is when it arrives.
- `catboost/private/libs/algo/calc_score_cache.cpp` --
  `TCalcScoreFold::Sample` and `SetControlNoZeroWeighted`, which is where a
  zero weight becomes a dropped row.

#### 1. MVS really is the CPU default, under four conditions. Verified.

`SetNotSpecifiedOptionsToDefaults` opens with the comment
`// TODO(nikitxskv): Support MVS for GPU.` and then:

    if (bootstrapType.NotSet()) {
        if (!IsMultiClassOnlyMetric(lossFunction)
            && !IsMultiRegressionObjective(lossFunction)
            && TaskType == ETaskType::CPU
            && ObliviousTreeOptions->BootstrapConfig->GetSamplingUnit() == ESamplingUnit::Object)
        {
            bootstrapType.SetDefault(EBootstrapType::MVS);
        }
    }

So MVS is the default when all four hold: the user did not set
`bootstrap_type`, the loss is not multiclass-only, the loss is not a
multi-regression objective, `task_type=CPU`, and `sampling_unit=Object` (the
`sampling_unit` default, `bootstrap_options.h`). **Both objectives this project
benchmarks satisfy every one of them.** `Logloss` is binary-class-compatible so
`IsMultiClassOnlyMetric` is false (`enum_helpers.cpp`: classification-only AND
multiclass-compatible AND NOT binary-compatible); `RMSE` is not a
classification metric at all. **The CatBoost we are compared against on the CPU
is running MVS, not the Bayesian bootstrap of A4.** A4's row weighting is what
CatBoost falls back to for `MultiClass`, `MultiClassOneVsAll`, the
`MultiRMSE` family, `sampling_unit=Group`, and every GPU fit -- those land on
the constructor default, `BootstrapType("type", EBootstrapType::Bayesian)`.

**This supersedes one sentence in the A4 note.** That note ends "Not
implemented: ... the Bernoulli/MVS/Poisson bootstrap types", and
`sampling.canonical_bootstrap_type` refuses `"mvs"` by name. Both were correct
when written. MVS is implemented now and that arm changes.

**Read this before quoting any existing CatBoost peer row.** The harness's
CatBoost arm is not "CatBoost with a bootstrap we have not built yet" -- it is
already running MVS, today, in every number we have published beside it.
`bench/real_data/scenarios.py::CATBOOST_LEFT_AT_STOCK` records
`"bootstrap_type": "MVS"` and `"subsample": 0.8` as the resolved stock values,
and section 2 above is why: 0.8 is the value MVS installs, not the 0.66 the
`TBootstrapConfig` constructor suggests, so anyone reading 0.66 out of the
header and assuming that is the peer's rate is wrong by a fifth of the rows.
Every CatBoost row we have therefore came from a fit that saw about 80 percent
of the rows per tree, gradient-weighted, while the LightGBM and mojotrees rows
beside it saw all of them. That is a difference in what was computed and not
only in how fast it arrived, and it applies retroactively to the rows already
published.

`CATBOOST_UNMATCHABLE["row_sampling"]` in the same file says "mojotrees's
`bagging_fraction` is not an emulation of it and the CatBoost-mode arm does not
try". That sentence goes stale the moment this lane lands: the CatBoost-mode
arm can now try, because MVS itself is reachable. That file is not this lane's
to edit and the correction is handed to the orchestrator as glue.

#### 2. `subsample = 0.8`, and only under MVS. Verified.

Immediately after the block above:

    if (subsample.IsSet()) {
        CB_ENSURE(bootstrapType != EBootstrapType::Bayesian, ...);
    } else {
        if (bootstrapType == EBootstrapType::MVS) {
            subsample.SetDefault(0.8);
        }
    }

0.8 is exactly as briefed. Note it is **not** the config's own default: the
constructor is `TakenFraction("subsample", 0.66f)`, so 0.66 is what Bernoulli
and Poisson get and 0.8 is a value MVS installs over it.

**A consequence of `TOption::SetDefault` that is easy to get wrong.** That
method is `DefaultValue = value; if (!IsSetFlag) Value = DefaultValue;` -- it
overwrites, it does not first-write-wins. So a *later* `SetDefault` in the same
function beats an earlier one. Three losses use this: for `QueryCrossEntropy`,
`YetiRankPairwise` and `PairLogitPairwise`, the switch further down does
`bootstrapType.SetDefault(EBootstrapType::Bernoulli)` and
`GetTakenFraction().SetDefault(0.5)`, which lands on top of the MVS/0.8 pair
set above. **For those three losses the CPU default is Bernoulli at 0.5, not
MVS at 0.8.** All three are pairwise, and `Bootstrap` skips MVS entirely under
`isPairwiseScoring` anyway, so the two facts agree.

#### 3. What MVS computes. Verified, `mvs.cpp`.

Per row `i`, over the approx dimensions `d`:

    g_i   = sqrt( sum_d der[d][i]^2 + lambda )          # regularized magnitude
    p_i   = (g_i > mu) ? 1 : g_i / mu                   # GetSingleProbability
    keep  = (prng.GenRandReal1() < p_i)
    w_i   = keep ? 1 / p_i : 0                          # SampleWeights[i]

`mu` is chosen so that `sum_i p_i` equals the target sample size. That is the
whole method: it is Poisson sampling with inclusion probability proportional to
the (regularized) gradient magnitude, capped at 1, plus the Horvitz-Thompson
inverse-probability weight that makes the sampled gradient sum unbiased. Rows
above the threshold are kept certainly at weight exactly 1; rows below it are
kept with probability `g/mu` and, when kept, amplified by `mu/g`. **This is
GOSS's family, better calibrated**: GOSS keeps a fixed top fraction and gives
every surviving small row one shared amplification `(1-top)/other`, where MVS
solves for the threshold that hits the requested rate and gives each small row
its own amplification.

`mu` is solved by `CalculateThreshold`, a recursive quickselect over the
`g_i`: pivot on the first candidate, three-way partition, compute the sample
size the pivot would imply
(`sumOfSmall / pivot + nLarge + nMiddle`), recurse into the large side if that
overshoots and the small side if it undershoots, and when a side runs out solve
the linear equation exactly. It is O(n) expected and allocates one `double` per
row per block.

**The solve is per block of 8192 rows, not global.** `BlockSize = 8192` is a
`const ui32` member of `TMvsSampler`, `blockParams.SetBlockSize(BlockSize)`,
and each block calls `CalculateThreshold(..., SampleRate * blockSize)` on its
own candidates. So `mu` differs block to block and the sampling rate is hit per
block rather than over the dataset. Two things follow. First, the block size is
a **constant**, not a thread count, so CatBoost's own MVS is thread-count
stable here. Second, any port that derives its block size from a worker count
would not be, which is why ours does not.

#### 4. `bagging_temperature` beside MVS: silently ignored, NOT refused. Verified.

The brief said `bagging_temperature` belongs to Bayesian and not to MVS. That
is right about behavior and wrong about enforcement, and the difference is
worth recording because it is a trap. `TBootstrapConfig::Validate` refuses the
parameter for `No` ("you shoudn't provide bootstrap options if bootstrap is
disabled") and for Bernoulli ("bagging temperature available for bayesian
bootstrap only"), but the `MVS` arm of that switch checks **only**
`GetSamplingUnit() == ESamplingUnit::Object`. So CatBoost accepts
`bootstrap_type=MVS, bagging_temperature=5`, never reads the temperature (the
MVS arm of `Bootstrap` does not pass it to `TMvsSampler` at all), and
`TBootstrapConfig::Save` drops it from the serialized config. A user who sets
it gets no error and no effect.

**Ours refuses it instead**, and that is a deliberate divergence, recorded
here: a knob that is accepted and does nothing is the shape of a silent wrong
answer, and this project has already paid for one of those.

#### 5. The denominator, and why `lambda_l2 = 0` is NOT the risk here

**First, the correction, because the brief conflated two parameters.** MVS's
`lambda` is `mvs_reg`, a parameter of the bootstrap config
(`MvsReg("mvs_reg", Nothing())`). It is **not** `l2_leaf_reg`, and it is not
our `lambda_l2`. The stock-defaults change that moved `lambda_l2` to 0 does not
reach MVS at all, and no amount of `lambda_l2` would have damped anything
described below.

**Second, the risk is real anyway, and it is exactly the shape the brief
named.** `mu` is a bare denominator in `GetSingleProbability`, and
`CalculateThreshold` has two more:

    // overshoot, no larger side left:
    return (sumOfSmallCurrent + sumOfSmallUpdate + sumOfMiddle) / (sampleSize - numberOfLargeCurrent);
    // undershoot, no smaller side left:
    return sumOfSmallCurrent / (sampleSize - numberOfLargeCurrent - numberOfMiddle - numberOfLargeUpdate);

The first denominator is safe: that branch is only reached when the estimated
sample size exceeds `sampleSize`, and the estimate is `>= numberOfLargeCurrent`,
so `numberOfLargeCurrent < sampleSize` strictly. The second is **not**. In that
branch `estimated = sumSmall/pivot + nLargeCur + nLargeUpd + nMiddle
<= sampleSize`, so the denominator equals `(sampleSize - estimated) +
sumSmall/pivot`, which is zero exactly when the estimate lands on `sampleSize`
and `sumSmall` is zero -- and the numerator is that same zero `sumSmall`. Two
reachable degenerate shapes:

- **All candidates zero.** If every row in a block has zero derivative *and*
  `lambda == 0`, every `g_i` is 0, the pivot is 0, `sumSmall/pivot` is `0/0`,
  the estimate is NaN, `NaN > sampleSize` is false so control falls to the
  undershoot branch, and it returns `0 / (0.8*n - n)` = **negative zero**. Then
  every probability is `0.0 / -0.0` = NaN, `NaN > epsilon` is false, and every
  row is assigned weight 0. **The tree gets no rows at all, with no exception
  and no warning.**
- **Exact tie on the boundary.** Positive `mu`, no strictly-smaller candidate
  ever accumulated, and `0.8 * blockSize` landing exactly on the integer
  `nLarge + nMiddle`. Gives `0/0` = NaN directly. Rarer, but reachable whenever
  a block's magnitudes tie heavily.

**What stands in front of the first one is `mvs_reg`, and its default is not a
number.** `MvsReg` defaults to `Nothing()`, and `TMvsSampler::GetLambda` then
derives it from the data:

    const double mean = (!leafValues.empty())
        ? CalculateLastIterMeanLeafValue(leafValues)   // mean over leaves of the
                                                       // L2 norm of the previous
                                                       // tree's leaf-value vector
        : CalculateMeanGradValue(derivatives, SampleCount, ...);  // mean |g|
    return mean * mean;

So on the first tree lambda is the squared mean gradient magnitude, and on
every later tree it is the squared mean leaf-value norm of the tree before it
(`ctx->LearnProgress->LeafValues` is what `DoBootstrap` passes, and
`leafValues.back()` is the last iteration). **That auto-lambda is the damper.**
It is positive whenever anything is nonzero, and it floors every `g_i` at
`sqrt(lambda) > 0`. A user who writes `mvs_reg=0` explicitly removes it and
re-opens the first shape.

##### The defect, stated on its own line, because it is worse than a NaN

**`mvs_reg = 0` on a zero-derivative block is a wrong answer with no signal.**
Not a crash, not an infinity, not a NaN that propagates into a score somebody
would notice. The chain is: pivot 0 makes the estimate NaN, `NaN > sampleSize`
is false so control takes the undershoot branch, the branch returns
`0 / (0.8n - n)` which is **negative zero**, `GetSingleProbability` then
computes `0.0 / -0.0` = NaN, and the very next line is
`if (probability > std::numeric_limits<double>::epsilon())` -- which is
**false for NaN**, so the `else` runs and writes `SampleWeights[i] = 0`. Every
row in the block gets weight zero. `SetControlNoZeroWeighted` then drops all of
them, the tree is built on nothing, and the user gets a model that trained to
nothing with no exception, no warning, and no NaN anywhere in the output to
look at. The NaN is consumed by a comparison that swallows it. Verified by
reading; not reproduced, because this lane runs nothing.

##### The degenerate-threshold guard is OURS. NOT verified from CatBoost source.

There is nothing to verify it against: CatBoost does not guard this and
produces the silent all-zero-weight block above. **Our rule, in two parts:**

1. **`mvs_reg = 0` is refused at validation**, by name, with a message that
   says what the auto value would have been instead. It is cheap -- one
   comparison per fit -- and there is no legitimate use for it: its only
   reachable outcomes are the auto-lambda's behavior (when no block is
   degenerate) or a dead tree. Accepting a setting whose best case is
   indistinguishable from the default and whose worst case is a silently empty
   model is not a knob, it is a trap.
2. **The threshold is guarded anyway**, because part 1 does not close the
   second degenerate shape (the exact tie on the boundary, which happens at
   positive lambda). If the solved threshold is not finite, or is not strictly
   positive, the block is treated as carrying no information to sample on and
   **every row in it is kept at weight exactly 1.0**. That is the correct limit
   -- as `mu` falls to zero every `p_i` rises to 1 -- and it is the only choice
   that cannot silently produce an empty tree.

Both are divergences from CatBoost and are recorded as such. The count of
blocks that took the guard is exposed on the result so a test can **prove** the
branch was reached rather than assume it.

#### 6. MVS drops rows, and this lane does not price that. Verified.

`Bootstrap` sets `performRandomChoice = false` for MVS alone, and
`TCalcScoreFold::Sample` then calls `SetControlNoZeroWeighted(objectCount,
fold.SampleWeights.data())` instead of `SetSampledControl`. Rows whose MVS
weight is zero are therefore compacted out of the score fold and never visited
by the split search -- MVS is a genuine row *dropper* and not only a
reweighter, and at `subsample=0.8` it visits roughly 80 percent of the rows.

**That is a speed side effect and this lane does not claim it.** A sampler that
touches fewer rows looks faster for reasons that have nothing to do with
whether the sampler is any good, the orchestrator is the only one who takes
timings, and this lane took none. The honest statement of the bound is
arithmetic and nothing more: at `subsample = s`, the expected number of rows in
the histogram pass is `s * n` rather than `n`, against an added `O(n)` per tree
for the magnitudes, the per-block quickselect, and one uniform draw per row.
Whether that nets out positive on this machine is not knowable from here.

#### 7. MVS weights are hessians, exactly as A4's are. Verified.

`Bootstrap` calls `CalcWeightedData` for MVS on the same line it calls it for
Bayesian, and that function does
`sampleWeightedDerivativesData[z] = weightedDerivativesData[z] *
sampleWeightsData[z]` and then `SampleWeights[i] *= learnWeights[i]`. So an MVS
weight multiplies the row's derivatives and rides in the leaf denominators
exactly as a sample weight does. **A fit with MVS on therefore has a per-row
hessian and must not declare `histogram.CONSTANT_HESSIAN`**, for precisely the
reason the A4 note gives, and it carries the same exclusion by the same
mechanism: the predicate `boosting.round_has_constant_hessian` cannot see a
bootstrap configuration and its signature is a GPU-visible contract, so the
guard lives on the sampler's side.

#### 8. Schedule and determinism

**Schedule, verified.** `IsSamplingPerTree` gates the call: at the
`sampling_frequency=PerTree` default `DoBootstrap` runs once before the tree
(`greedy_tensor_search.cpp`, the `MapTensorSearchStart` block), and under
`PerTreeLevel` it runs once per level inside both the oblivious and the
depthwise loops. Only the per-tree schedule is ported, as with A4, which is why
the draw is keyed by tree index and not by (tree, level).

**Determinism, and where we diverge.** CatBoost draws
`randSeed = rand->GenRand()` once from the fit's shared RNG and then gives
block `b` a `TRestorableFastRng64(randSeed + b)` advanced 10 steps, drawing
sequentially inside the block. Its block size is the fixed 8192, so its own
weights do not move with the thread count -- but they do depend on the whole
fit's RNG consumption history, which is a stream. **Ours is keyed, never
streamed**, in the shape this campaign requires and the shape A4 already uses:
row `r` of tree `t` reads a counter-based splitmix64 stream derived from
`(seed, tree index)` at offset `r`, and nothing advances. The threshold solve
is likewise a function of the block's contents and the fixed block constant
only. Consequently no weight, no threshold and no kept-row set depends on
`MOJOTREES_NUM_WORKERS`, on a buffer length, or on how many draws came before.
The distribution and the per-row transform match CatBoost; the individual draws
for a given seed do not, and that is the same trade already recorded for
bagging, GOSS, feature sampling and A4.

#### 9. What this should do to accuracy, stated as an expectation and not a result

**Expected direction.** MVS is an unbiased estimator of the full gradient sums
(that is what the `1/p` weight buys), with the inclusion probabilities chosen
to minimize the variance of those sums at a fixed expected sample size -- the
"minimal variance" in the name. Against the alternatives:

- **Against no sampling**, at a fixed tree budget, it can only add variance to
  each split's statistics. Expect training loss to be neutral-to-slightly-worse
  and held-out error to be neutral-to-slightly-better, because the added
  variance acts as a regularizer. Our benches are fixed-iteration, so the usual
  argument for sampling -- more trees in the same wall clock -- is invisible in
  our harness and must not be smuggled in.
- **Against uniform row bagging at the same rate** (`bagging_fraction=0.8`,
  CatBoost's Bernoulli), expect MVS to be better, and this is the one direction
  the method's own construction actually argues for: uniform sampling is the
  worst case of the same estimator family, since it ignores the magnitudes that
  determine each row's contribution to the variance.
- **Against our GOSS**, expect MVS to be better at an equal kept fraction, for
  the reason in section 3: same family, calibrated threshold, per-row rather
  than shared amplification.

**What would falsify it.** Any of these on `bench/real_data` against
stock+det: MVS at 0.8 losing to `bagging_fraction=0.8` on the same seeds, or
MVS losing to no sampling on held-out error at the same tree count. The gate is
rule (3), the real-data gate, and this lane does not run it.

**Unknown, and stated as unknown.** Whether the per-block threshold at 8192
rows is the right block for our row counts is not something reading CatBoost
settles. At 50k rows that is six blocks and the per-block rate is a decent
approximation of the global one; at a few thousand rows it is one block and the
question does not arise; the regime where it would matter is a block count high
enough for the per-block thresholds to diverge, and no source read tells us
where that is.
