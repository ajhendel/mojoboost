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
| A4 | Bayesian bootstrap / `bootstrap_type` (**verified from source**, see A4 note) | Row weighting instead of row dropping: every row kept, each given weight `(-log(U + 1e-100)) ** bagging_temperature`, redrawn once per tree | Only when on | Yes for CatBoost mode (their default sampler). Built in `sampling.mojo`, off by default. **Device cost, checked:** per-row weights disable the constant-hessian plane (`round_has_constant_hessian`), so the device accumulates three planes and the Int16 gradient-staging arm loses its better half (staged bytes per visit 7 -> 9 at the default group, not 9 -> 7). Not free on the device even before any device code was written. **Device plane now live** (weight-plane-device lane): `GpuObjectiveState.refresh_weights` rewrites the per-row weight plane once per tree and `GpuHistogramBuilder.refresh_objective_weights` refuses one beside a constant-hessian declaration. The weight is applied in the gradient kernel, where the CPU applies it, so the histogram accumulation gains no multiply and no fetch. Unreached: no trainer constructs a `BayesianBootstrapParams` yet. | CPU (`sampling.mojo`), device plane in `gpu_objectives_native.mojo` | (3) when on; **carries a constant-hessian exclusion**, see the note |
| A5 | Ordered target statistics for categoricals (CTRs; "verify" against `catboost/private/libs/algo/` CTR code) | For each of several random permutations, a categorical value in row i is replaced by (sum of targets of earlier rows with that category + prior) / (count + 1); plus counters and pairwise combinations; test time uses all rows | Yes (feature values change) | Yes; it is CatBoost's real accuracy edge on high-cardinality columns and independent of tree shape | CPU (`binning.mojo`, categorical, dataset) | (3); design section B2 first, then lanes |
| A6 | `leaf_estimation_iterations` (`gradient_walker.h::FastGradientWalker`; defaults in `catboost_options.cpp::GetEstimationMethodDefaults`) -- **verified from source** | Re-estimate leaf values 1..k times, derivatives recomputed at the current leaf value each pass | Only when > 1 | Built on the CPU and now on the device, opt-in, default 1. **DEVICE: built 2026-08-16**, the first of the two shapes the earlier survey named -- a per-iteration device reduction inside the tree, launches accepted, host kept out of the loop (`gpu_objectives_native.GpuLeafEstimator`). `3 * (k - 1)` launches per tree plus one round trip; at `k = 3` that is 278 -> 284 leaf-wise and 62 -> 68 oblivious, which crosses the 64-command-buffer knee and is why the same six launches cost proportionally more on the oblivious schedule. The earlier "structural, decide before any device lane" deferral is discharged, not still open. **Correction:** an earlier version of this row said CatBoost defaults to 10 for logloss AND multiclass. Logloss is 10; **multiclass is 1**. The 10 in that block is the unreachable Gradient slot while the default method is Newton, and CatBoost's own docs agree. | CPU built; device built | (3) when > 1; see A6 notes |
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

**What mojotrees built on the device**, 2026-08-16. `train_gpu`,
`train_gpu_with_valid` and their session entry points now honor the parameter
too, on both of `_train_gpu_rounds`'s arms. The host-objective arm's raw
scores are a `List[Float64]`, so that arm calls
`boosting._estimate_leaf_values` itself: same fold, same order, same bits as
the CPU trainer. The device-objective arm's raw scores live on the device and
is where the new code is.

*Which of the two shapes.* The earlier survey named two and deferred the
choice: a per-iteration device reduction inside the tree (more launches), or
leaf estimation lifted out of the device round (more host trips). The first
was taken. `docs/GPU_PORTABILITY.md` section 6.1.1 separates the two costs --
a launch or copy count predicts ordering hazard, a *round-trip* count predicts
seconds -- and the second shape converts a fixed launch cost into `k - 1`
round trips per tree, which at a hundred rounds and `k = 10` is nine hundred
waits on a plane whose whole design is one wait per tree.

*What it costs.* `GpuLeafEstimator.estimate` enqueues three launches per extra
iteration, independent of the leaf count: shift the raw scores by each leaf's
current value, run `_grad_hess_kernel` (the round's own derivative kernel,
unmodified, so the objectives and the weight multiplier keep one definition on
this backend) over the shifted scores, then one threadgroup per leaf to reduce
that leaf's `G` and `H` and take the step in place. Plus one device-to-host
copy and one synchronization per tree, which is unavoidable in either shape
because `Tree.value` is a host list; the point is that it is one and not
`k - 1`. At `k = 3` a leaf-wise tree goes from 278 launches to 284 (+2.2%,
entirely inside the 14-17 microsecond regime) and a depth-6 oblivious tree
from 62 to 68 (+9.7%, and it **crosses** the 64-command-buffer knee where the
per-launch cost roughly doubles). Added launches therefore cost proportionally
more on the oblivious schedule, and the crossing is the reason rather than the
count.

*What moves between iterations, and what does not.* The structure is fixed and
**no histogram is rebuilt**. Row membership is the leaf ranges the grower left
in the active-row permutation and does not move. What moves is two numbers per
leaf, `G` and `H`, and only because the point they are evaluated at moved --
`raw[r] + v`, with `v` the value the leaf currently holds, unshrunk, exactly
as on the host and exactly as in CatBoost's own walker.

*Agreement.* Node-identical to `boosting._estimate_leaf_values` to Float32,
not to the bit: the device sums a leaf's rows in Float32 in a strided
threadgroup reduction where the host sums them in Float64 sequentially in
ascending row index. That is the trade this plane already makes everywhere.
`tests/test_gpu_leaf_estimation.mojo` asserts it on one tree handed to both
implementations, and asserts the `k = 1` path is bit-identical to a fit with
the parameter absent on both arms.

*What the GPU trainers refuse rather than ignore.* The multiclass GPU
trainers (class `k`'s softmax derivative reads every class's raw score,
including trees this round has not grown yet -- and CatBoost defaults
`MultiClass` to 1 anyway) and `train_custom_gpu` (a `GradHessFn` is called
once per round over the whole row set; there is no per-leaf, shifted-score
call shape to ask it for). The renewing objectives and GOSS are refused by
the same `boosting._check_leaf_estimation_config` the CPU trainer calls.
