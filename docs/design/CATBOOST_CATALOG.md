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
| A10 | Score functions (`Cosine` default on CPU symmetric trees, `L2`, pairwise) | Split scoring alternatives | Only if changed | No; keep LightGBM's gain | — | — |
| A11 | `feature_border_type` / `border_count` (`library/cpp/grid_creator/binarization.cpp`, `catboost/libs/data/quantization.cpp`) | CatBoost's float quantization: seven border-selection algorithms, `GreedyLogSum` by default, bounded by `border_count` thresholds | New mode; bit-moving on the arm that selects it, exact on the arm that does not | Yes, opt-in `border_type`, default stays the LightGBM/mojotrees quantile fit | CPU (`binning.mojo`) | A11 note. **BUILT**, five of seven types, `binning.fit_bins(border_type=...)` |
| A12 | `one_hot_max_size` (`catboost/private/libs/options/cat_feature_options.cpp`, `catboost/private/libs/algo/greedy_tensor_search.cpp`) | CatBoost's categorical one-hot threshold, default 2, `<=` on the count of real categories seen on learn | New mode; bit-moving only when set | Yes, opt-in `CategoricalParams.one_hot_max_size`, default off | CPU (`categorical.mojo`) | A12 note. **BUILT**. It also settles the owed `max_cat_to_onehot` off-by-one, see A12 |

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

### A11 note: border selection, verified from source

Status: **verified from CatBoost source**, `master`, read 2026-08-16. Files:

- `library/cpp/grid_creator/binarization.h` -- `EBorderSelectionType`, the
  `NSplitSelection::TQuantization` / `TFeatureValues` shapes,
  `NImpl::EPenaltyType`, `NImpl::EOptimizationType`.
- `library/cpp/grid_creator/binarization.cpp` -- `MakeBinarizer` (:114) mapping
  each enum value to a binarizer, `Penalty<MaxSumLog>` / `Penalty<MinEntropy>`
  (:174-186), `IFeatureBin` / `TFeatureBin` / `TWeightedFeatureBin`
  (:1320-1497), `GreedySplit` (:1500-1520), `TGreedyBinarizer::BestSplit`
  (:1677-1712), `GenerateMedianBorders` (:1046-1063), `TMedianBinarizer`
  (:1201), `TMedianPlusUniformBinarizer` (:1224-1260), `TUniformBinarizer`
  (:1262-1317), `RegularBorder` (:698-731), `NSplitSelection::BestSplit`
  (:74-112) for NaN removal, `SetQuantization` (:887-905) for the negative-zero
  purge.
- `catboost/libs/data/quantization.cpp` -- `CalcQuantizationAndNanMode`
  (:235-346), which is where `border_count` meets `nan_mode`.
- `catboost/private/libs/options/data_processing_options.cpp` (:14-19) and
  `catboost/private/libs/options/binarization_options.cpp` -- the defaults and
  the option names.
- `catboost/private/libs/options/restrictions.h` (:10) -- `GetMaxBinCount()`.
- `catboost/libs/model/model_build_helper.cpp` (:100-105) and
  `catboost/libs/model/model.cpp` (:181-194) -- what a *trained model's*
  borders are, which is the correction below.

**The defaults, verified.** `data_processing_options.cpp:14` constructs
`float_features_binarization` as
`TBinarizationOptions(GreedyLogSum, type == GPU ? 128 : 254, ENanMode::Min, 200000)`.
So on CPU: `border_type = GreedyLogSum`, `border_count = 254`,
`nan_mode = Min`, and `dev_max_subset_size_for_build_borders = 200000`. GPU
drops the budget to 128. `border_count <= GetMaxBinCount() = 65535`
(`binarization_options.cpp::Validate`, `restrictions.h:10`).

**`border_count` counts thresholds. Confirmed, and the existing note is
right.** `CalcQuantizationAndNanMode` passes its `nonNanValuesBorderCount`
straight in as `maxBordersCount`, and `GreedySplit` terminates at
`splits.size() == maxBordersCount + 1` bins, so the budget is
`border_count` borders and `border_count + 1` bins. CatBoost's 254 and this
repo's `max_bin = 255` are the same 255-bin budget. `bench/real_data/scenarios.py`
`CATBOOST_UNMATCHABLE["binning_budget"]` is correct on that point.

**Correction, and it is the important one. The "picks well under its budget"
sentence in that same note is wrong, and the number it cites is not a border
count.** From the source, `GreedyLogSum` produces exactly
`min(border_count, distinct_value_count - 1)` borders and nothing in between.
`GreedySplit`'s loop is `while (splits.size() <= maxBordersCount && splits.top().CanSplit())`;
`CanSplit()` is false only when a bin holds a single distinct value, and an
unsplittable bin scores `-inf` (`CalcSplitScore` returns `-inf` at
`splitPos == BinStart || splitPos == BinEnd`), so it sinks to the bottom of
the max-heap. The top being unsplittable therefore means *every* bin is
unsplittable, which happens only once there is one bin per distinct value.
Either the budget runs out first, or every level gets its own bin. There is no
third outcome, and a uniform 20,000-row column has 20,000 distinct values, so
it must fill 254. A faithful port of `TFeatureBin` + `GreedySplit` run over
uniform, lognormal and coarse-level columns at budgets 254 / 32 / 8 agrees
exactly with `min(budget, distinct - 1)` on every case
(**derived from source, corroborated by simulation, not measured against
CatBoost itself**).

Where 125, 113 and 101 come from instead: `TCommonModelBuilderHelper::ProcessSplitsSet`
clears every float feature's border list (`model_build_helper.cpp:102-103`,
`for (auto& feature : FloatFeatures) feature.Borders.clear();`) and
`TModelTrees::ProcessSplitsSet` then repopulates it from the set of splits the
built trees actually use (`model.cpp:191-194`,
`FloatFeatures.at(internalFloatIndex).Borders.push_back(split.FloatFeature.Split)`).
A trained model's `get_borders()` is a **used-split count**, not the
quantization grid, and it decreases across features exactly the way a
used-split count does. To read the grid, quantize a `Pool` and read the
borders off the quantized pool, never off a fitted model. **That note in
`scenarios.py` should be corrected; this lane did not edit the bench file
because it does not own it, and the literal replacement is in the lane report.**

**Consequence that matters for accuracy work.** Because the border *count* rule
is `min(budget, distinct - 1)` on both sides, CatBoost's `GreedyLogSum` and
this repo's `GreedyFindBin` port agree on *how many* bins a column gets at
`min_data_in_bin = 1`. They differ only in *where* the borders sit:
equal-frequency quantiles here, recursive median maximizing the sum of log bin
populations there. That is a much narrower gap than "one fills the budget and
one does not" suggested.

**The algorithm, exactly.** `Penalty<MaxSumLog>(w) = -log(w + 1e-8)` and
`Penalty<MinEntropy>(w) = w * log(w + 1e-8)` (`binarization.cpp:174-186`).
A bin `[s, e)` scores a cut at `p` as
`Penalty(L + R) - Penalty(L) - Penalty(R)` where `L`, `R` are the two sides'
*observation* weights, and `-inf` when `p` is at either end
(`TWeightedFeatureBin::CalcSplitScore`, :1454-1471; the unweighted
`TFeatureBin::CalcSplitScore` at :1398-1407 is the same expression with unit
weights). Only **two** cut positions are ever considered per bin, the two ends
of the run holding the bin's median observation
(`UpdateBestSplitProperties`, :1409-1424 and :1473-1493), and ties go left
(`BestSplit = scoreLeft >= scoreRight ? lb : ub`). `GreedySplit` (:1500-1520)
keeps the bins in a max-heap on that score, repeatedly splits the best one, and
emits `0.5f * v[s-1] + 0.5f * v[s]` as each non-first bin's left border
(`IFeatureBin::LeftBorder`, :1357-1371). So it is a recursive median split
prioritized by penalty gain, not an exhaustive search: `MaxLogSum` and
`MinEntropy` are the exhaustive DP versions of the same two objectives
(`TExactBinarizer`, `EOptimizationType::Exact`).

**The two forms agree.** CatBoost runs `TFeatureBin` over the raw sorted
observation array when the column is dense (`TGreedyBinarizer::BestSplit`
:1703-1712) and `TWeightedFeatureBin` over grouped distinct levels with
cumulative counts when it is sparse (:1686-1702). A port of both, compared on
uniform / coarse-level / skewed / near-binary columns at budgets 254, 32 and 8,
produced **identical** border sets in every case (**derived, corroborated by
simulation**). mojotrees implements the grouped form, because the repo already
has level-and-count machinery and the grouped array is the shorter one.

**NaN costs one border, on both sides.** `CalcQuantizationAndNanMode`
(`quantization.cpp:322-345`) decrements `nonNanValuesBorderCount` when the
column has any NaN, selects borders over the non-NaN values only, and then
inserts `lowest()` at the front (`nan_mode=Min`, the default) or `max()` at the
back (`nan_mode=Max`). Total borders stays `border_count`. LightGBM does the
same thing by a different route: `bin.cpp:394-397` calls `FindBinWithZeroAsOneBin`
with `max_bin - 1` and then `bin_upper_bound_.push_back(NaN)`. mojotrees already
matches that (`n_ordinary = max_bins - 1 if reserve`). **Not a divergence.** The
side NaN lands on *is* one: CatBoost's default sends it to bin 0 on the left,
LightGBM and mojotrees send it to the last bin on the right. Out of this lane's
scope, recorded for whoever owns `nan_mode`.

**Border-selection subsampling.** CatBoost fits borders on a subset capped at
`dev_max_subset_size_for_build_borders = 200000`
(`data_processing_options.cpp:18`, threaded through
`quantization.cpp:135/183/2653`). mojotrees's `bin_construct_sample_cnt`
defaults to LightGBM's 200,000, which is the same number by coincidence of
values rather than of rule; the sampling *schemes* were not compared and this
lane does not claim they match.

**Negative zero.** `SetQuantization` (:897-902) removes `-0.0f` from the border
set, commented "BestSplit might add negative zeros". mojotrees's midpoint
arithmetic is Float64 and its `emit_quantile_edges` strict-increase filter
already collapses a `-0.0` / `0.0` pair, so nothing was ported for this.

#### What mojotrees built for A11

`binning.fit_bins(..., border_type=BORDER_QUANTILE)`. The default is
`BORDER_QUANTILE`, which is the existing LightGBM `GreedyFindBin` port, reached
by the same instructions as before this lane: the new work sits behind
`if border_type != BORDER_QUANTILE` inside the per-feature body, so the default
arm pays one predictable branch per *feature* and nothing per row.

| `border_type` | CatBoost name | Built? |
|---|---|---|
| `BORDER_QUANTILE` (0) | none, this is ours | default, unchanged |
| `BORDER_GREEDY_LOG_SUM` (1) | `GreedyLogSum`, CatBoost's default | yes |
| `BORDER_GREEDY_MIN_ENTROPY` (2) | `GreedyMinEntropy` | yes |
| `BORDER_UNIFORM` (3) | `Uniform` | yes |
| `BORDER_MEDIAN` (4) | `Median` | yes |
| `BORDER_UNIFORM_AND_QUANTILES` (5) | `UniformAndQuantiles` | yes |
| `BORDER_MIN_ENTROPY` (6) | `MinEntropy` | **refused by name** |
| `BORDER_MAX_LOG_SUM` (7) | `MaxLogSum` | **refused by name** |

The two refused ones are `TExactBinarizer`, the exhaustive dynamic program at
`binarization.cpp:192-694`: roughly five hundred lines of banded DP with nine
solver modes, `O(distinct * bins)` time and `(bins - 2) * distinct` `size_t`
of scratch. They are refused with that reason rather than approximated,
because an approximate `MinEntropy` is `GreedyMinEntropy`, which is already
here under its own name.

Divergences from CatBoost in the built arms, all deliberate:

1. **Float64 throughout.** CatBoost's borders are `float`; the midpoint
   `0.5f * a + 0.5f * b` and the uniform step are single precision. mojotrees
   computes in Float64 and its edges are Float64. Strictly more precise, never
   bit-identical to CatBoost.
2. **Heap ties are broken by ascending bin start.** `std::priority_queue`'s
   order among equal scores is unspecified in C++, so CatBoost has no behavior
   here to match. A total order is required by this project's rule that a fit
   be reproducible across `MOJOTREES_NUM_WORKERS` and machines, and ties can
   change the output when they fall on the last split the budget allows.
3. **`initialBorders` is not ported.** It is CatBoost's mechanism for snapping
   new borders onto a previously quantized grid (`RegularBorder` :719-724,
   `LeftBorder` :1362-1367). mojotrees has no quantized-pool concept to snap to.
4. **The sparse `DefaultValue` path is not ported.** `TDefaultQuantizedBin` and
   `GenerateMedianBordersWithDefaultValue` exist for CatBoost's sparse columns;
   `sparse.fit_bins_csc` does not take `border_type` at all, so a sparse fit is
   always the quantile fit. Stated, not hidden.
5. **`min_data_in_bin` is not applied to the CatBoost arms.** It is LightGBM's
   parameter and CatBoost has no analogue; the CatBoost arms ignore it rather
   than inventing an interaction. A fit that sets both raises.

Cost, **derived bound, not measured, no timing was taken by this lane**: the
CatBoost arms sort each column, `O(n log n)`, where the default arm resolves a
few hundred ranks in `O(n)` by bucket selection above `SELECT_MIN_ROWS`. The
opt-in arms are expected to be the slower ones and nothing about the 2.86x
default result is claimed to carry over to them.

### A12 note: `one_hot_max_size`, verified from source, and the `max_cat_to_onehot` boundary

Status: **verified from CatBoost source and from LightGBM source**, both
`master`, read 2026-08-16.

- `catboost/private/libs/options/cat_feature_options.cpp:232-233, 267-268` --
  `OneHotMaxSize("one_hot_max_size", 2)`, and the ceiling
  `OneHotMaxSizeLimit = (CPU ? GetMaxBinCount() : 256)` enforced as
  `CB_ENSURE(OneHotMaxSize.Get() <= OneHotMaxSizeLimit, ...)`.
- `catboost/private/libs/algo/greedy_tensor_search.cpp:177-186`
  (`AddOneHotFeatures`), `:465-470` (`AddSimpleCtrs`), `:501-525`
  (`AddTreeCtrs`).
- `catboost/private/libs/algo/learn_context.cpp:72-74` and
  `catboost/private/libs/algo/data.cpp:150-153` -- the same threshold decides
  whether CTRs are computed at all.
- LightGBM `src/treelearner/feature_histogram.cpp:183` and `:435`, and
  `src/io/bin.cpp:440-470`.

**CatBoost's rule, exactly.** `AddOneHotFeatures` emits one-hot candidates for
a categorical feature when

```
onLearnOnlyCount = quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx).OnLearnOnly
if (onLearnOnlyCount > oneHotMaxSize) || (onLearnOnlyCount <= 1): skip
```

so the comparison is **`<=`** on the count of **real distinct categories seen
on the learn set**, with a lower guard of at least 2. The default is 2, so
CatBoost one-hots binary categoricals and nothing wider unless asked. There is
no dummy or unknown category in that count.

**What happens above the threshold.** Not "a wider split": the feature switches
to **CTRs**, target statistics. `AddSimpleCtrs` (:469) and `AddTreeCtrs` (:524)
both `return` early on `<= oneHotMaxSize`, which is the exact complement of
`AddOneHotFeatures`'s condition. So the two candidate generators partition the
categorical features between them with no overlap and no gap: at or below the
threshold a feature is one-hot only, above it is CTR only. The threshold is
load-bearing beyond candidate generation as well, since
`CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn() > one_hot_max_size` is
what decides whether CTRs are computed at all (`data.cpp:150-153`) and whether
the permutation machinery is needed (`learn_context.cpp:72-74`). A dataset
whose every categorical fits under the threshold costs CatBoost no CTR pass.

**LightGBM's rule, exactly, and it is not the same shape.**
`feature_histogram.cpp:183` is

```cpp
bool use_onehot = meta_->num_bin <= meta_->config->max_cat_to_onehot;
```

and `num_bin` for a categorical feature is `kept_categories + 1`, because
`bin.cpp:456-459` pushes a dummy bin for NaN/unknown at index 0 and starts
`num_bin_ = 1` before the loop that appends the real categories. The one-vs-rest
scan then runs over `[1 - offset, num_bin - offset)`, so bin 0 is never itself
a one-hot candidate. LightGBM's default is 4, therefore LightGBM one-hots up to
**3** real categories.

#### The `max_cat_to_onehot` boundary question, answered

The `wide-categorical-bins` lane is blocked on this. The answer is that
**there are two different correct boundaries and they must not be collapsed
onto one comparison**:

- For a parameter named `max_cat_to_onehot`, which is LightGBM's name and
  carries LightGBM's default of 4, the parity-correct test is
  **`n_categories + 1 <= max_cat_to_onehot`**, equivalently
  `n_categories <= max_cat_to_onehot - 1`. Real categories, plus one for the
  bin-0 dummy that `num_bin` counts and mojotrees's `n_categories` does not.
- For a parameter named `one_hot_max_size`, which is CatBoost's name and
  carries CatBoost's default of 2, the correct test is
  **`2 <= n_categories <= one_hot_max_size`** on real categories, with no
  `+ 1` anywhere.

**So the off-by-one that was owed is real and it is confirmed here.**
`categorical.find_best_categorical_split` tests
`n_categories <= cat.max_cat_to_onehot`, which is CatBoost's shape wearing
LightGBM's name and default, and it one-hots 4 categories where LightGBM
one-hots 3. This lane did **not** change that test, because changing it moves
bits on the default path for every categorical fit and that is the
`wide-categorical-bins` lane's call to make, not a side effect of an opt-in
CatBoost knob. What this lane did is give the CatBoost boundary its own
parameter under CatBoost's name, so that fixing the LightGBM one cannot
silently take the CatBoost one with it.

#### What mojotrees built for A12

`CategoricalParams.one_hot_max_size`, a sixth field with a **default of `-1`
meaning off**, so `CategoricalParams(4, 32, 10.0, 10.0, 100)` still constructs
and still behaves exactly as before. At `-1` the one-hot decision is the
existing `n_categories <= max_cat_to_onehot`. At `>= 0` it becomes CatBoost's
`n_categories <= one_hot_max_size`, with `n_categories < 2` already returning
no split above, which is CatBoost's `<= 1` guard.

Divergence, deliberate and unavoidable: above the threshold CatBoost builds
CTRs and mojotrees runs its many-vs-many sorted partition search. mojotrees has
no CTR machinery (catalog A5, unbuilt), so `one_hot_max_size` here selects
CatBoost's *boundary* and not CatBoost's *other side*. A fit with
`one_hot_max_size = 2` is CatBoost-shaped in which features get one-hot and
LightGBM-shaped in what the rest get. Anyone reading a comparison against
CatBoost has to hold both halves.
