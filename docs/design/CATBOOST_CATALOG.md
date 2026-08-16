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

> **Superseded in part, 2026-08-16.** "A10 already says we keep LightGBM's
> gain" was true when written and is not now: Cosine is implemented, opt-in,
> default off, and A10 carries the note. The rest of Correction 2 stands
> unchanged. And the practical consequence is smaller than this paragraph
> implies -- at `lambda_l2 = 0`, which is mojotrees stock as of this round,
> Cosine and L2 have the **same argmax by derivation**, because Cosine's
> numerator IS the L2 sum and its denominator collapses onto it. Being on
> CatBoost's `L2` rather than its default is a difference of zero at our
> settings. See A10 section 3.

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
| A12 | Automatic `learning_rate` (`catboost/libs/train_lib/options_helper.cpp`: `UpdateLearningRate`, `TAutoLRParamsGuesser`) -- **verified from source**, see the A12 note | The rate is fitted from the train row count and the iteration count off a 20-row coefficient table, not taken from the 0.03 constant, whenever the user set none of `learning_rate`, `leaf_estimation_method`, `leaf_estimation_iterations`, `l2_leaf_reg` | Yes when on | Yes, and it is bucket C as much as B: our harness pins the rate on the CatBoost arm *because* of this, so "defaults vs defaults" is not currently a real comparison | CPU (`auto_learning_rate.mojo`, new file) | (3) when on; **BUILT, off by default**, no existing default changed |
| A13 | Stochastic Gradient Langevin Boosting: `langevin`, `diffusion_temperature` (**verified from source**, see the A13/A14 note) | Seeded Gaussian noise added to every row's derivative once per bootstrap, at scale `sqrt(2 / (learning_rate * diffusion_temperature))`; a second, differently scaled draw is added to each leaf's derivative sum at leaf estimation | Only when on | Yes, `langevin.mojo`, off by default. Per-row half built and wired by glue; the leaf-sum half is built and **not** wired (it needs a leaf-sum call site in `tree.mojo`) | CPU (`langevin.mojo`) | (3) when on; deterministic under the seed, across `MOJOTREES_NUM_WORKERS`, and across machines. **Carries no constant-hessian exclusion**, and that is a checked claim, not an omission: see `langevin_varies_hessian` |
| A14 | Model shrinkage: `model_shrink_rate`, `model_shrink_mode` (**verified from source**, see the A13/A14 note) | At the top of every iteration after the first, every accumulated raw score is multiplied by `1 - rate * learning_rate` (Constant) or `1 - rate / iteration` (Decreasing); the products are folded back into the leaf values of the already-grown trees at the end of the fit | Only when on | Yes, `langevin.mojo`, off by default. Built as a deferred fold (strictly less work than CatBoost's per-round rescale of the model, and exact) | CPU (`langevin.mojo`) | (1) as built, off; (3) when on. Refused beside continued training and beside `init_score`, both of which CatBoost also refuses |
| A15 | `feature_border_type` / `border_count` (`library/cpp/grid_creator/binarization.cpp`, `catboost/libs/data/quantization.cpp`) | CatBoost's float quantization: seven border-selection algorithms, `GreedyLogSum` by default, bounded by `border_count` thresholds | New mode; bit-moving on the arm that selects it, exact on the arm that does not | Yes, opt-in `border_type`, default stays the LightGBM/mojotrees quantile fit | CPU (`binning.mojo`) | A15 note. **BUILT**, five of seven types, `binning.fit_bins(border_type=...)` |
| A16 | `one_hot_max_size` (`catboost/private/libs/options/cat_feature_options.cpp`, `catboost/private/libs/algo/greedy_tensor_search.cpp`) | CatBoost's categorical one-hot threshold, default 2, `<=` on the count of real categories seen on learn | New mode; bit-moving only when set | Yes, opt-in `CategoricalParams.one_hot_max_size`, default off | CPU (`categorical.mojo`) | A16 note. **BUILT**. It also settles the owed `max_cat_to_onehot` off-by-one, see A12 |
| A18 | `QueryRMSE` (`catboost/private/libs/algo_helpers/error_functions.h::TQueryRmseError::CalcDersForQueries` and its private `CalcQueryAvrg`) -- **verified from source**, see the A18/A19/A20 note | Squared error with the query's weighted mean residual removed, so only *within-query* level is learned. `Der1 = w_i (t_i - a_i - avg)`, `Der2 = -w_i`, `avg = sum_j w_j (t_j - a_j) / sum_j w_j`. `IsStoreExpApprox` false | Yes -- it is a new objective | Yes. It is the cheapest of the three CatBoost ranking losses and the only one whose hessian stays constant under unit weights | CPU (`catboost_ranking.mojo`, new file) | A18/A19/A20 note. **BUILT, off by default, unreached.** (3) when on. Deterministic: no RNG at all |
| A19 | `PairLogit` + its automatic pair generation (`error_functions.h::TPairLogitError::CalcDersForQueries`; `catboost/private/libs/pairs/util.cpp::GeneratePairLogitPairs` / `GenerateBruteForce`; `catboost/private/libs/target/data_providers.cpp::GeneratePairs`) -- **verified from source** | Pairwise logistic loss over (winner, loser) pairs inside a group. `p = e^{a_lose} / (e^{a_lose} + e^{a_win})`, `Der1[win] += c p`, `Der1[lose] -= c p`, `Der2[both] += c p (p-1)`. With `max_pairs` unset, pairs are *every* label-distinct pair in the group, weight = group weight, no RNG | Yes -- a new objective, and it is the first mojotrees objective whose hessian is pairwise by construction | Yes. It is also the gradient kernel `YetiRank` reuses | CPU (`catboost_ranking.mojo`) | A18/A19/A20 note. **BUILT, off by default, unreached.** (3) when on. **Carries a constant-hessian exclusion.** Deterministic: default pair generation draws nothing |
| A20 | `YetiRank` (`catboost/private/libs/algo/yetirank_helpers.cpp`: `UpdatePairsForYetiRank`, `GenerateYetiRankPairsForQuery`, `TYetiRankPairWeightsCalcer::{AddNoise,CalcWeightsClassic}`; defaults in `catboost/private/libs/options/loss_description.cpp::GetYetiRankPermutations` / `GetYetiRankDecay`) -- **verified from source** | A *sampled* pairwise scheme, not a closed form: each round, per group, draw `permutations`=10 noisy rankings of the current scores, and in each ranking charge each adjacent pair `0.15 * decay^k * |rel_i - rel_j|`. Average the pair weights over the 10 draws and feed them to the `PairLogit` gradient | Yes, and it is the only objective in the catalog whose gradients depend on an RNG stream | Yes. It is CatBoost's flagship ranking loss and the one a CatBoost ranking comparison will be run under | CPU (`catboost_ranking.mojo`) | A18/A19/A20 note. **BUILT, off by default, unreached.** (3) when on. **Carries a constant-hessian exclusion.** Deterministic under `(seed, iteration, query, permutation, doc)` keying, across `MOJOTREES_NUM_WORKERS` and machines. CatBoost's own stream is NOT nameable |
| A21 | `group_id` and the two ranking eval metrics `NDCG`, `PFound` (`catboost/libs/data/objects.cpp::CheckGroupIds`; `catboost/libs/data/target.cpp::CheckGroupWeights`; `catboost/libs/metrics/dcg.cpp` + `catboost/libs/metrics/metric.cpp::TDcgMetric`; `catboost/libs/metrics/pfound.h::TPFoundCalcer` + `metric.cpp::TPFoundMetric`; tie rule in `catboost/libs/metrics/doc_comparator.h`) -- **verified from source**, see the A21 note | The grouping contract (contiguous, *not* sorted; loud failure when violated; weights constant inside a group) and the two metrics a CatBoost ranking run is actually judged by. CatBoost's default `NDCG` is **not** LightGBM's: `type=Base` (raw relevance, not `2^l - 1`), no truncation, and an adversarial tie rule | No (metrics move no bits) | Yes. Without them we cannot score a CatBoost ranking run at all, which is bucket C | CPU (`catboost_ranking.mojo`); `ranking.RankGroups` reused unchanged | A21 note. **BUILT, off by default, unreached.** (3) as a metric selection. Deterministic: no RNG |

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

### A12 note: automatic `learning_rate` from the data, verified from source

Status: written before any code by the `auto-learning-rate` lane, from
CatBoost `master` (Aug 2026). Every claim in this section is **verified from
source** and cites the file and line it came from; the two places where I am
inferring rather than reading are marked **not verified** in bold.

Why it is in bucket C as well as B: our benchmark harness pins
`learning_rate` on the CatBoost arm precisely because CatBoost otherwise
derives it and ours does not. Until we can derive the same number, a
"defaults vs defaults" comparison against CatBoost is not a comparison of
defaults at all -- it is our fixed 0.1 against their fitted value, and any
accuracy gap it produces is an artifact of the harness.

### Where it lives

**Not in `catboost_options.cpp`.** The brief pointed at
`catboost/private/libs/options/catboost_options.cpp`
(`SetNotSpecifiedOptionsToDefaults`); that file contains no automatic
learning-rate code at all (`grep -i learning` over the whole 1225-line file
returns only unrelated error strings and the parameter *name* at line 1149).
The derivation is a **data-dependent** default, not a static one, so it lives
with the other data-dependent defaults:

- `catboost/libs/train_lib/options_helper.cpp`, `UpdateLearningRate` (lines
  269-288) -- the gate.
- Same file, `TAutoLRParamsGuesser` (lines 177-266) -- the coefficient table
  and the formula.
- Same file, `SetDataDependentDefaults` (lines 403-435) -- the single call
  site, line 418.
- Same file, local `static double Round(double, int)` (lines 15-18).

Static defaults for the same parameters, for contrast:
`catboost/private/libs/options/boosting_options.cpp` line 10
(`LearningRate("learning_rate", 0.03)`), line 13
(`IterationCount("iterations", 1000)`), line 17
(`BoostFromAverage("boost_from_average", false)`).

### The formula

`TAutoLRParamsGuesser::GetLearningRate` (options_helper.cpp:252-262):

```
customIterationConstant  = exp(C * log(iterationCount) + D)
defaultIterationConstant = exp(C * log(1000)           + D)
defaultLearningRate      = exp(A * log(learnObjectCount) + B)
return Round(Min(defaultLearningRate * customIterationConstant
                 / defaultIterationConstant, 0.5), 6)
```

`A, B, C, D` are `DatasetSizeCoeff, DatasetSizeConst, IterCountCoeff,
IterCountConst` on `TLearningRateCoefficients` (options_helper.cpp:116-122),
whose own comment states the closed form:

    learning_rate = exp(B + A log size + C log iter - C log 1000)

**`D` cancels analytically and does not cancel in floating point.** The
comment above drops it; the code computes two separate `exp`s that both carry
it and divides. We reproduce the code's ordering, not the comment's algebra,
so that a value we print can be compared with CatBoost's without a
"which rounding" argument. Keeping `D` in the table is therefore deliberate
even though it is algebraically dead.

Inputs, all four of them, and nothing else:

1. `learnObjectCount` -- **train** row count, `trainDataMetaInfo.ObjectCount`
   (options_helper.cpp:411). Not the test rows, not the feature count.
2. `iterationCount` -- `BoostingOptions->IterationCount`
   (options_helper.cpp:272), i.e. `iterations`, default 1000.
3. The loss function, collapsed to one of four `ETargetType` values by
   `GetTargetType` (options_helper.cpp:181-194): `Logloss` covers
   `Logloss`, `MultiLogloss` and `MultiCrossEntropy`; `MultiClass` covers
   `MultiClass` only; `RMSE` covers `RMSE` only; **everything else is
   `Unknown`**.
4. Three flags that select which coefficient row is used: task type
   (CPU/GPU), `use_best_model`, `boost_from_average`.

`Round` is local (options_helper.cpp:15-18) and is
`round(x * 10^6) / 10^6` with C's `round`, i.e. **ties away from zero**, not
banker's rounding. The `Min(..., 0.5)` cap is applied *before* rounding.
There is no lower cap.

`Min` is applied to the whole product, so a long run with a negative `C`
shrinks the rate and a short run raises it, capped at 0.5.

### Where the flags come from

`SetDataDependentDefaults` resolves both flags *before* it calls
`UpdateLearningRate` (options_helper.cpp:415-418), so the auto-LR sees
resolved values, never "not set":

- `use_best_model`: `UpdateUseBestModel` (options_helper.cpp:100-113) sets it
  true when it is unset, a test set exists, and the test target is
  non-constant or there are test pairs; and forces it false with a warning
  when there is no test set. **So on a plain fit with no eval set,
  `use_best_model` is false**, which is the row a benchmark arm lands on.
- `boost_from_average`: `AdjustBoostFromAverageDefaultValue`
  (options_helper.cpp:353-374). Returns immediately if the user set it.
  Otherwise sets it **true** for `RMSE`, `MAE`, `Quantile`, `MAPE`,
  `MultiQuantile`, `MultiRMSE`, `MultiRMSEWithMissingValues` on a single host
  with no `continueFromModel`; then forces it **false** if either the train or
  the test pool carries a baseline. `Logloss` and `MultiClass` are not on that
  list, so their default is the static `false` from
  `boosting_options.cpp:17`.

Consequence worth stating plainly, because it decides which coefficient rows
a defaults-vs-defaults benchmark ever touches: with no eval set and no
user override, **binary lands on `(Logloss, CPU, useBestModel=False,
boostFromAverage=False)` and regression on `(RMSE, CPU, useBestModel=False,
boostFromAverage=True)`.**

### When it fires, and when it does not

`UpdateLearningRate` (options_helper.cpp:276-287) requires **all four** of
these to be unset by the user:

- `learning_rate`
- `leaf_estimation_method`
- `leaf_estimation_iterations`
- `l2_leaf_reg`

Setting *any one of the last three* silently pins `learning_rate` to the
constant 0.03. That is a genuinely surprising coupling and it is not
documented anywhere I could find: a user who sets `l2_leaf_reg` also
un-sets the automatic learning rate. It is presumably there because the
coefficients were fitted with the other three at their defaults.

Then `NeedToUpdate` (options_helper.cpp:246-249) requires that the
`(targetType, taskType, useBestModel, boostFromAverage)` key exists in the
table. It does **not** for:

- Any loss outside `{Logloss, MultiLogloss, MultiCrossEntropy, MultiClass,
  RMSE}` -- so MAE, Quantile, MAPE, Poisson, Tweedie, LambdaRank, every
  ranking loss, and every custom loss keep 0.03.
- `MultiClass` with `boost_from_average = true`. The table has only two
  MultiClass rows per task type and both are `EBoostFromAverage::False`
  (options_helper.cpp:207-210 CPU, 230-233 GPU). Reachable only by the user
  setting `boost_from_average` explicitly, since the adjuster never sets it
  for MultiClass.

Otherwise it assigns and logs `Learning rate set to <value>`
(options_helper.cpp:284-285).

### The coefficient table, transcribed

`TAutoLRParamsGuesser::TAutoLRParamsGuesser` (options_helper.cpp:197-244),
as `{A, B, C, D}`:

| target | task | use_best_model | boost_from_average | A | B | C | D |
|---|---|---|---|---|---|---|---|
| Logloss | CPU | true | true | 0.246 | -5.127 | -0.451 | 0.978 |
| Logloss | CPU | false | true | 0.408 | -7.299 | -0.928 | 2.701 |
| Logloss | CPU | true | false | 0.247 | -5.158 | -0.435 | 0.934 |
| Logloss | CPU | false | false | 0.427 | -7.525 | -0.917 | 2.63 |
| MultiClass | CPU | true | false | 0.02 | -2.364 | -0.382 | 0.924 |
| MultiClass | CPU | false | false | 0.051 | -2.889 | -0.845 | 2.928 |
| RMSE | CPU | true | true | 0.157 | -4.062 | -0.61 | 1.557 |
| RMSE | CPU | false | true | 0.158 | -4.287 | -0.813 | 2.571 |
| RMSE | CPU | true | false | 0.189 | -4.383 | -0.623 | 1.439 |
| RMSE | CPU | false | false | 0.178 | -4.473 | -0.76 | 2.133 |
| Logloss | GPU | true | true | 0.04 | -3.226 | -0.488 | 0.758 |
| Logloss | GPU | false | true | 0.427 | -7.316 | -0.907 | 2.354 |
| Logloss | GPU | true | false | -0.085 | -2.055 | -0.414 | 0.427 |
| Logloss | GPU | false | false | -0.055 | -3.01 | -0.896 | 2.366 |
| MultiClass | GPU | true | false | 0.101 | -2.95 | -0.437 | 1.136 |
| MultiClass | GPU | false | false | 0.204 | -4.144 | -0.833 | 2.889 |
| RMSE | GPU | true | true | 0.108 | -3.525 | -0.285 | 0.058 |
| RMSE | GPU | false | true | 0.131 | -4.114 | -0.597 | 1.693 |
| RMSE | GPU | true | false | 0.051 | -3.001 | -0.449 | 0.859 |
| RMSE | GPU | false | false | 0.047 | -3.034 | -0.591 | 1.554 |

20 rows: 4 Logloss + 2 MultiClass + 4 RMSE per task type. Note the sign of
`A` for GPU Logloss without `boost_from_average` -- the fitted rate there
*decreases* with dataset size, unlike every CPU row.

These are **fitted constants** in CatBoost's sense: a regression of a good
learning rate on `log(rows)` and `log(iterations)`, run offline by the
CatBoost authors over their benchmark suite. The source carries no derivation
and no citation for them. **They are numbers we copy, not numbers we can
re-derive**, and that is the honest description of what this lane ships.

### Float32 narrowing

`BoostingOptions->LearningRate` is `TOption<float>`
(`catboost/private/libs/options/boosting_options.h:26`), so the `double`
returned by `GetLearningRate` is narrowed to float32 on assignment at
options_helper.cpp:284. Everything downstream (`ApplyLearningRate`,
`algo_train.cpp:59,66,178,187,350`) uses that float. Our implementation
reproduces the narrowing, behind a field, so a printed value matches
CatBoost's rather than being right in the 9th decimal and different.

### Two things I did NOT establish

- **Not verified:** whether the CatBoost Python package can reach
  `SetDataDependentDefaults` on a path that skips this (e.g. `fit` with a
  pre-quantized pool, or `cv`). I read the C++ default resolution only; I did
  not trace the Python entry points. If a benchmark harness ever *relies* on
  auto-LR firing, that trace is required first.
- **Not verified:** why the last-three-parameters gate exists. The reasoning
  in this entry ("the coefficients were fitted with them at defaults") is my
  inference from the shape of the table, not a comment in the source.

### What mojotrees built, and the mapping

`src/mojotrees/auto_learning_rate.mojo`, **off by default**, reachable
through `AutoLearningRateParams` and the `auto_learning_rate=true` parameter
key. It changes no existing default: `BoosterParams.default()` is still
`learning_rate = 0.1` and `parse_params` still returns 0.1 unless the key is
given.

Reachability, and where it stops. The derivation needs the **train row
count**, which no parameter parser has, so `parse_params` records the request
on `TrainConfig.auto_learning_rate` and `TrainConfig.resolved_learning_rate(
n_rows)` applies it. That method returns `booster.learning_rate` untouched
whenever the derivation is off, so every caller can route through it
unconditionally. Two surfaces do: `capi/mojotrees_capi.mojo` (at the point it
first has `n_rows`) and `cli/mojotrees_cli.mojo` (after the table is read).
The Python mapping surface does **not** carry the key at all, so it is
refused there as an unknown parameter rather than accepted and dropped --
which is the repo's refuse-rather-than-ignore rule, not an oversight. Wiring
it into `bindings/_mojotrees.mojo` means threading a row count through
fourteen `_parse_params` call sites and adding the key to the Python default
mapping; that is a separate change and has not been made.

`parse_params` is stricter than CatBoost on purpose. CatBoost, handed
`learning_rate=0.05 l2_leaf_reg=5`, silently keeps 0.05 and prints nothing;
handed `l2_leaf_reg=5` alone with the rate unset, it silently reverts to the
constant 0.03. Here `auto_learning_rate=true` is an explicit act, so
combining it with `learning_rate=`, `lambda_l2=` or
`leaf_estimation_iterations=` is an error naming the gate
(`_enable_auto_learning_rate`). That is a deliberate divergence from
CatBoost's behavior and is the only one on this surface.

Target-type mapping from mojotrees objective codes
(`objective_registry.mojo`), which is **ours, not CatBoost's**, because the
objective sets do not coincide:

| mojotrees objective | CatBoost target type | why |
|---|---|---|
| `SQUARED_ERROR` | RMSE | same loss up to the sqrt, which does not change the minimizer |
| `BINARY_LOGISTIC` | Logloss | same loss |
| `CROSS_ENTROPY` | Logloss | CatBoost's `CrossEntropy` is the soft-label Logloss and `GetTargetType` maps `MultiCrossEntropy` to Logloss; the single-output `CrossEntropy` case is **our decision**, CatBoost's `GetTargetType` does not list plain `CrossEntropy` and would return Unknown for it |
| `MULTICLASS` | MultiClass | same loss |
| everything else | Unknown -> no change | matches CatBoost, which leaves 0.03 |

The `CROSS_ENTROPY` row is the one deviation and it is marked. If we wanted
strict parity we would map it to Unknown; mapping it to Logloss is a
judgement that the fitted coefficients transfer to the soft-label case
because the derivative is identical.

`boost_from_average` has no mojotrees equivalent as a switch -- our
`init_score` plays that role -- so
`catboost_boost_from_average_default(objective)` reproduces
`AdjustBoostFromAverageDefaultValue` for the objectives we have
(`SQUARED_ERROR`, `L1`, `QUANTILE`, `MAPE` -> true, everything else false)
and the caller may override it. `MultiQuantile`, `MultiRMSE` and
`MultiRMSEWithMissingValues` are on CatBoost's list and we do not have them.

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

### A13/A14 note: Langevin boosting and model shrinkage, verified from source

Status: **verified from CatBoost source**, `master`, 2026-08-16, read by
`lane/langevin-model-shrink`. Every claim in this section cites the file and
the function it came from. Where this lane's implementation diverges from
CatBoost the divergence has its own paragraph and says so in the first
sentence. Nothing here is measured; this lane took no timings and ran no
benchmark.

Files read:

- `catboost/private/libs/algo_helpers/langevin_utils.h` and `.cpp` --
  `CalcLangevinNoiseRate`, `AddLangevinNoiseToDerivatives` (both overloads),
  `AddLangevinNoiseToLeafDerivativesSum`, `AddLangevinNoiseToLeafNewtonSum`.
- `catboost/private/libs/algo/greedy_tensor_search.cpp` -- `DoBootstrap`, the
  only per-row noise call site, and its three callers.
- `catboost/private/libs/algo/approx_calcer.cpp` -- `CalcApproxDeltaSimple`
  and `CalcLeafValuesSimple`, the two leaf-sum noise call sites.
- `catboost/private/libs/algo/train.cpp` -- `TrainOneIteration` (the shrink
  itself) and `ScaleAllApproxes` (what it multiplies).
- `catboost/libs/train_lib/train_model.cpp` -- the end-of-fit fold of the
  shrink history into the leaf values, and the two refusals.
- `catboost/private/libs/options/boosting_options.h` and `.cpp` -- the option
  surface, the defaults, and `TBoostingOptions::Validate`.
- `catboost/private/libs/options/catboost_options.cpp` --
  `SetNotSpecifiedOptionsToDefaults` (the coupling) and the posterior-sampling
  refusals.
- `catboost/private/libs/options/enums.h` -- `EModelShrinkMode`.
- `catboost/private/libs/options/restrictions.h` -- `CB_THREAD_LIMIT`.
- `catboost/private/libs/algo_helpers/online_predictor.h` -- `ScaleL2Reg`.
- `util/random/normal.h` -- `StdNormalDistribution`.

### A13. What Langevin actually does

**The option surface.** `boosting_options.cpp` constructs
`Langevin("langevin", false)` and `DiffusionTemperature("diffusion_temperature",
0.0f)`. Both are plain `TOption`, so both exist on CPU and GPU;
`model_shrink_rate` and `model_shrink_mode` beside them are `TCpuOnlyOption`.
`TBoostingOptions::Validate` enforces exactly one thing about the temperature:
`CB_ENSURE(DiffusionTemperature >= 0.0, "Diffusion temperature should be
non-negative")`. `Save` writes the Langevin pair into the trained model's
option blob only when `Langevin` is true (`if (Langevin) { SaveFields(options,
Langevin, DiffusionTemperature); }`).

**The noise scale.**
`langevin_utils.cpp::CalcLangevinNoiseRate(diffusionTemperature, learningRate)`
is one line:

    return sqrt(2.0 / learningRate / diffusionTemperature);

Read that twice. `diffusion_temperature` is in the **denominator**, so it is
an inverse temperature: raising `diffusion_temperature` *lowers* the noise.
The name says the opposite of what the arithmetic does, and a user who reads
the name and turns the knob up to get more exploration gets less. At
CatBoost's own defaults once Langevin is on -- `diffusion_temperature = 1e4`,
`learning_rate = 0.03` -- the coefficient is `sqrt(2 / 300) = 0.0816`
(derived, one division and one square root, not measured). The learning rate
is also in the denominator, so lowering the learning rate *raises* the noise:
the pair `(learning_rate, diffusion_temperature)` sets the noise jointly and
neither one alone is "the noise knob".

**Where the per-row noise is added, and to what.**
`greedy_tensor_search.cpp::DoBootstrap`, and nowhere else:

    Bootstrap(...);
    if (ctx->Params.BoostingOptions->Langevin) {
        for (auto& bodyTail : fold->BodyTailArr) {
            AddLangevinNoiseToDerivatives(
                DiffusionTemperature, LearningRate,
                ctx->LearnProgress->Rand.GenRand(),
                &bodyTail.WeightedDerivatives, ctx->LocalExecutor);
        }
    }

Three facts follow from the placement. It is **after** the bootstrap, so the
noise is added to the already row-weighted derivatives and is *not* itself
scaled by the row's sample weight -- a row the bootstrap zeroed still gets a
full-size noise term written into its derivative slot. It is on
`WeightedDerivatives` alone: the hessians, the second derivatives, and the
weights are untouched. And `DoBootstrap` has three call sites
(`GreedyTensorSearch` once per tree when `IsSamplingPerTree`, and the
oblivious and leafwise level loops once per level otherwise), so under
`sampling_frequency=PerTreeLevel` the noise is **re-added at every level and
accumulates within one tree**. Under the default `PerTree` it is added once
per tree.

**The draw itself.** `langevin_utils.cpp`, per row:

    dersData[idx] += coef * StdNormalDistribution<double>(blockRng);

`StdNormalDistribution` (`util/random/normal.h`) is the *polar* (Marsaglia)
form of Box-Muller: it rejects until `x*x + y*y` lands in `(0, 1]`, so it
consumes an unbounded, draw-dependent number of uniforms.

**What it is seeded by, and CatBoost's own determinism.** The seed is one
draw from the sequential learn-progress generator,
`ctx->LearnProgress->Rand.GenRand()`, taken once per `DoBootstrap` call.
Inside, rows are cut into blocks by
`TSimpleIndexRangesGenerator<size_t>(TIndexRange(objectCount), CB_THREAD_LIMIT)`
and block `b` uses `TFastRng64(randomSeed + b)`. The block size is
`CB_THREAD_LIMIT`, and `restrictions.h` makes that `constexpr int
CB_THREAD_LIMIT = 128` -- a **constant, not a thread count**. So CatBoost's
per-row Langevin noise does not move with `thread_count`: the block count is
`ceil(rows / 128)` whatever the executor does. It does move with the row
count, and within a block it is a sequential stream, so row `i`'s noise
depends on how many draws rows `i-1`, `i-2`, ... consumed -- which, because
the polar transform rejects, is itself random. **Ours diverges here on
purpose; see the determinism paragraph below.**

**The second, differently scaled noise: leaf derivative sums.** There are two
call sites, both in `approx_calcer.cpp`, and they do not agree with each
other.

`CalcApproxDeltaSimple` (per body-tail approx deltas) branches on the leaf
estimation method and is **not** guarded by `Langevin`:

    if (estimationMethod == Gradient)  AddLangevinNoiseToLeafDerivativesSum(...)
    else if (estimationMethod == Newton) AddLangevinNoiseToLeafNewtonSum(...)

`CalcLeafValuesSimple` (the final leaf values) is guarded by `Langevin` and
calls the **Gradient** variant unconditionally, under Newton estimation too.
The two paths therefore scale the same noise differently for the same fit.
The guards also mean that `langevin=False` with `diffusion_temperature > 0`
set explicitly leaves the per-row noise off and the leaf-sum noise **on**,
because every `AddLangevinNoise*` function's first statement is
`if (diffusionTemperature == 0.0f) return;` and the temperature is the real
switch. That configuration is reachable only by setting `langevin` to False
by hand, since otherwise the coupling below turns it on.

The scales, from `langevin_utils.cpp`, with `scaledL2Regularizer =
l2_leaf_reg * (sumAllWeights / allDocCount)` (`online_predictor.h::ScaleL2Reg`):

    Gradient: sum.SumDer += coef * sqrt(sum.SumWeights + scaledL2Reg) * N(0,1)
    Newton:   sum.SumDer += coef * sqrt(|sum.SumDer2| + scaledL2Reg) * N(0,1)

Both skip a leaf with `sum.SumWeights < 1e-9`. Both perturb the numerator of
the leaf value and leave the denominator alone. The `sqrt(weight)` factor is
the point: a leaf's gradient sum is a sum of `n` per-row gradients, so the
noise that would arrive from `n` independent per-row draws grows like
`sqrt(n)`, and this reproduces that at the leaf without touching the rows.

### A14. What model shrinkage actually does

**The option surface.** `boosting_options.cpp`:
`ModelShrinkRate("model_shrink_rate", 0.0f, taskType)` and
`ModelShrinkMode("model_shrink_mode", EModelShrinkMode::Constant, taskType)`,
both `TCpuOnlyOption`. `EModelShrinkMode` (`enums.h`) has exactly two members,
`Constant` and `Decreasing`. `TBoostingOptions::Validate` gives each mode its
own admissible range, and they are different ranges:

    Constant:   0 <= model_shrink_rate * learning_rate < 1
    Decreasing: 0 <= model_shrink_rate < 1

**What it multiplies, and when.** `train.cpp::TrainOneIteration`, at the very
top, before the derivatives of this iteration are computed:

    if (modelShrinkRate > 0) {
        if (iterationIndex > 0) {
            modelShrinkage = Constant ? (1 - rate * learning_rate)
                                      : (1 - rate / iterationIndex);
            ScaleAllApproxes(modelShrinkage, ...);
            for (approx : *StartingApprox) approx *= modelShrinkage;
            ModelShrinkHistory.push_back(modelShrinkage);
        } else {
            ModelShrinkHistory.push_back(1.0);
        }
    }

Four things are load-bearing. The first iteration is exempt and records
`1.0`. `Decreasing` divides by the iteration **index**, not by the iteration
count, so the factor starts at `1 - rate` on iteration 1 and rises toward 1;
the shrink is strongest early and fades, which is the opposite of what
"decreasing" suggests about the multiplier and is the reason the two modes
have different admissible ranges. `ScaleAllApproxes` (`train.cpp`) touches the
**accumulated raw scores** -- every fold's body-tail approx, the averaging
fold, `AvrgApprox`, and every test approx -- and it does **not** touch any
leaf value; for an exp-approx loss it exponentiates instead, via
`ApplyLearningRate<true>`. And the base offset is scaled too, separately, as
`StartingApprox`.

**How the model ends up agreeing with the approxes.** It does not, until the
end of the fit. `train_model.cpp`, after the loop:

    accumulatedTreeShrinkage = 1.0;
    for (treeIndex = treeCount - 1; treeIndex >= 0; --treeIndex) {
        LeafValues[treeIndex] = ScaleElementwise(accumulatedTreeShrinkage, LeafValues[treeIndex]);
        accumulatedTreeShrinkage *= ModelShrinkHistory[treeIndex];
    }

So tree `t` is multiplied by the product of every shrink factor recorded
*after* it was grown, `prod_{j>t} History[j]`, and the last tree is not
scaled at all. The whole fit is one deferred fold rather than a rescale of
the model at every round. `ShrinkModel` resizes `ModelShrinkHistory` when
`use_best_model` truncates the ensemble, and the overfitting-detector path
pops the last entry, so the history stays exactly as long as the tree list.

**Two refusals, both CatBoost's.** `train_model.cpp` guards
`model_shrink_rate != 0` with `CB_ENSURE(!initModel, "Usage of
model_shrink_rate option in combination with learning continuation is
unimplemented yet.")` and the same for a baseline on the learn set or on any
test set. Continued training and an external `init_score` are both out of
scope for the feature in CatBoost itself.

### The coupling, verified from source

`catboost_options.cpp::SetNotSpecifiedOptionsToDefaults`, inside
`if (TaskType == ETaskType::CPU)`, in this order:

1. `if (PosteriorSampling) Langevin.SetDefault(true);`
2. `if (DiffusionTemperature > 0.0f && Langevin.NotSet()) Langevin.SetDefault(true);`
   -- setting a temperature turns Langevin on by itself.
3. `if (Langevin) { if (DiffusionTemperature.NotSet()) DiffusionTemperature.SetDefault(1e4); }`
4. `if (Langevin) { if (model_shrink_rate.NotSet()) shrinkRate = (mode == Constant) ? 0.001 : 0.01; }`
   -- **this is the coupling the two features are queued together for.**
   Enabling Langevin silently enables model shrinkage, and *which* rate it
   installs depends on `model_shrink_mode`.
5. `if (Langevin) { LeavesEstimationBacktrackingType.SetDefault(No); }` --
   Langevin also turns the leaf-estimation line search off, because a noisy
   step is not expected to improve the loss and a backtracking rule would
   reject it.
6. Separately, monotone constraints install their own shrink-rate default by
   the same shape: `0.01` for Constant, `0.2` for Decreasing. The Langevin
   block runs first, so on a fit with both, Langevin's smaller rate wins.

`Validate` adds three posterior-sampling refusals: it requires Langevin, it
refuses an explicitly set `diffusion_temperature`, and it requires
`model_shrink_mode == Constant`.

**The direction of the coupling matters for us.** CatBoost's default fit has
`model_shrink_rate = 0` and no shrink at all; the shrink only appears because
Langevin or a monotone constraint asked for it. So the honest statement of
A12 is not "add noise" but "add noise **and** start decaying the model", and
a mojotrees comparison against CatBoost with Langevin on that leaves the
shrink at zero is not comparing the same estimator.

### What mojotrees built, and where it diverges

`src/mojotrees/langevin.mojo`, both features **off by default**, no existing
default changed, nothing wired into a trainer by this lane (the call-site
glue is handed to the orchestrator as diffs). `LangevinParams.disabled()` and
`ModelShrinkParams.disabled()` are the constructed-by-default states and take
no draw and move no bit.

**Divergence 1, the normal transform.** Ours is the trigonometric Box-Muller,
exactly two uniforms per draw:
`z = sqrt(-2 ln(1 - u0)) * cos(2 pi u1)`. CatBoost's polar form rejects and so
consumes an unbounded number of uniforms per draw, which cannot be given a
fixed per-row stride in a counter-based stream without either capping the
retries (a distributional lie) or letting one row's substream run into the
next (a correlation bug). The distribution is identical; the numbers are not.
`1 - u0` rather than `u0` is deliberate: `rng.uniform` is half-open `[0, 1)`
and can return exactly 0, and `log(0)` is `-inf`.

**Divergence 2, determinism, and it is the point of the lane.** CatBoost's
per-row noise is a sequential stream inside blocks of 128 rows. Ours is a
counter-based splitmix64 keyed on `(seed, tree index, row, output)` with its
own domain constant `_LANGEVIN_DOMAIN`, following
`sampling.refresh_mvs_bootstrap`. Row `r`, output `k` reads exactly the two
counters `stream + 2*(r * n_outputs + k)` and `+ 1` and nothing advances, so
no row's noise depends on the row count, on the worker count, on which block
it landed in, on how many draws any earlier row took, or on whether an
earlier row was skipped. It is reproducible across `MOJOTREES_NUM_WORKERS`
and across machines, which CatBoost's is not with respect to the row count.

**Divergence 3, model shrinkage is a deferred fold, and it is strictly less
work.** CatBoost rescales the accumulated approxes every round, which it must
do because the derivatives of the next round are taken at the shrunk scores,
and separately folds the history into the leaf values once at the end. Ours
does the same two things, and records the fold as `(factor, trees grown so
far)` events rather than one entry per round -- because the mojotrees round
loop can finish a round *without* appending a tree (the degenerate
single-leaf guard `continue`s under any row sampler), and a history indexed
by round would then be misaligned with the tree list. The reverse scan over
events reproduces `prod_{j>t} History[j]` exactly, in one pass, and never
rescales an already-grown tree more than once. Classification: **(1) strictly
less work and exact** for the fold itself; the feature as a whole is **(3)
bit-moving when on, and off by default**.

**Divergence 4, the leaf-sum noise is built and not wired.** Both scalings,
Gradient and Newton, are in `langevin.mojo` as
`langevin_leaf_gradient_noise` and `langevin_leaf_newton_noise`, keyed by
`(seed, tree, leaf)` on a second domain constant so the leaf stream cannot
alias the row stream. Nothing calls them: the leaf sums live in `tree.mojo`'s
grower, which this lane does not own. A fit with `langevin` on today gets the
per-row half of CatBoost's mechanism and not the leaf-sum half, and
`LangevinParams.leaf_noise_wired()` returns False so a caller can see that
rather than infer it.

**Divergence 5, the coupling is offered and not applied silently.**
`langevin_default_model_shrink_rate` returns CatBoost's `0.001` / `0.01`, and
`couple_langevin_defaults` performs step 4 above, but a mojotrees caller has
to invoke it. Turning on a regularizer must not turn on a second, different
regularizer without the caller saying so; CatBoost does that and it is the
single most surprising thing in this pair.

**No hessian exclusion, and that is a checked claim.**
`langevin_varies_hessian` and `model_shrink_varies_hessian` both return
False, deliberately, unlike `sampling.mvs_varies_hessian`. Langevin writes
into the derivative buffer alone -- `AddLangevinNoiseToDerivatives` takes
`WeightedDerivatives` and nothing else, and both leaf-sum variants write
`sum.SumDer` while reading `SumDer2` and `SumWeights` -- so `hess[r]` is
exactly what the objective wrote and `boosting.round_has_constant_hessian`
stays correct. Model shrinkage moves the raw scores, which moves the
gradients and, for objectives whose hessian depends on the score, the
hessians too -- but it moves them to the values the objective would have
produced at those scores, and does not make a constant hessian non-constant.
`check_langevin_hessian_declaration` and
`check_model_shrink_hessian_declaration` exist anyway, raising only if the
predicates ever start returning True, so the pair cannot become a silent
exclusion by a later edit.

**Numeric range, flagged not measured.** The noise term is added to raw
gradients, so it widens the gradient range a fit sees, and the Int16
gradient-staging arm (`quantized_gradient.mojo`) derives its scale from that
range. No claim is made here about what that costs; this lane measured
nothing. A device lane wiring Langevin should re-derive the staging bound
before assuming it holds.

### A15 note: border selection, verified from source

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

#### What mojotrees built for A15

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

### A16 note: `one_hot_max_size`, verified from source, and the `max_cat_to_onehot` boundary

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

#### What mojotrees built for A16

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

### A18/A19/A20 note: the three CatBoost ranking objectives, verified from source

Files read, CatBoost `master`, 2026-08-16:

- `catboost/private/libs/algo_helpers/error_functions.h`
  (`TQueryRmseError::CalcDersForQueries`, `TQueryRmseError::CalcQueryAvrg`,
  `TPairLogitError::CalcDersForQueries`)
- `catboost/private/libs/algo_helpers/approx_updater_helpers.h`
  (`IsStoreExpApprox`, `CalcPairwiseWeights`)
- `catboost/private/libs/algo/yetirank_helpers.{h,cpp}`
  (`UpdatePairsForYetiRank`, `YetiRankRecalculation`,
  `GenerateYetiRankPairsForQuery`, `TYetiRankPairWeightsCalcer`)
- `catboost/private/libs/pairs/util.cpp` (`GeneratePairLogitPairs`,
  `GenerateBruteForce`, `GenerateRandomly`, `TryGeneratePair`)
- `catboost/private/libs/target/data_providers.cpp` (`GeneratePairs`)
- `catboost/private/libs/options/loss_description.cpp`
  (`GetYetiRankPermutations`, `GetYetiRankDecay`, `GetMaxPairCount`)

#### Sign convention, which everything below depends on

CatBoost's `TDers` carries `Der1` and `Der2` as derivatives of the
*log-likelihood*, so `Der1` is the NEGATIVE gradient and `Der2` is the
NEGATIVE hessian. mojotrees carries LightGBM's convention: `grad = dL/df`,
`hess = d2L/df2`. Every formula below is transcribed and then negated once,
and the negation is stated at each site rather than assumed. A lane that
copies `Der1` into `grad` gets a model that boosts away from the loss.

#### A18. QueryRMSE

`TQueryRmseError::CalcDersForQueries`, per document `i` of query `q`:

    avg      = sum_j w_j (t_j - a_j) / sum_j w_j          (0 if sum_j w_j <= 0)
    Der1[i]  = w_i (t_i - a_i - avg)
    Der2[i]  = -w_i

so in mojotrees terms

    grad[i]  = w_i (a_i - t_i + avg)
    hess[i]  = w_i

`CalcQueryAvrg` takes `w = 1` when the weight vector is empty, and computes
`querySum` and `queryCount` in document order over the query; the division
guard is `queryCount > 0`.

Three consequences worth naming:

1. **The hessian is the weight.** Under unit weights it is the literal 1.0,
   exactly as for squared error, so `QueryRMSE` is *admissible* on the
   two-plane constant-hessian path. It is the only one of the three that is.
   `catboost_ranking.query_rmse_varies_hessian` returns True exactly when a
   non-empty `sample_weight` was supplied, which is the same rule squared
   error already lives under.
2. **The gradients of a query sum to zero** when the weights are uniform, so
   the objective learns only within-query level and boosts from 0.
   `objective_init_kind` is `INIT_ZERO`, the same answer `LAMBDARANK` gets.
3. `IsStoreExpApprox` does **not** list `QueryRMSE`, so its approxes are raw.

#### A19. PairLogit, and where its pairs come from

`TPairLogitError::CalcDersForQueries`, for winner `i` and each competitor
`(j, c)` in `queriesInfo[q].Competitors[i - begin]`:

    p            = expApprox[j] / (expApprox[j] + expApprox[i])
    Der1[i]     += c p
    Der1[j]     -= c p
    Der2[i]     += c p (p - 1)
    Der2[j]     += c p (p - 1)

`IsStoreExpApprox` DOES list `PairLogit`, so `expApprox` is `exp(a)` and
therefore

    p        = sigmoid(a_j - a_i)
    grad[i] -= c p        grad[j] += c p
    hess[i] += c p (1-p)  hess[j] += c p (1-p)

**The hessian is per row and per round by construction.** `c p (1-p)` depends
on the current scores of both members of every pair the row appears in. There
is no configuration of `PairLogit` under which it is constant, which is why
`catboost_ranking.pairwise_varies_hessian` is unconditional rather than a
predicate over parameters, and why
`check_catboost_ranking_hessian_declaration` refuses a constant-hessian
declaration beside it with no escape hatch. This is the same guard shape as
`sampling.check_mvs_hessian_declaration`; the difference is that MVS's is a
consequence of a *knob* and this one is a consequence of the loss.

**Where the pairs come from when the user supplies none.**
`data_providers.cpp::GeneratePairs` calls `GeneratePairLogitPairs`, which for
each group counts the pairs of distinct target values and then takes one of
two branches. With `max_pairs` unset, `GetMaxPairCount` returns the sentinel
`MAX_AUTOGENERATED_PAIRS_COUNT` and the branch is `GenerateBruteForce` with
the truncation disabled: **every** ordered pair `(first, second)` with
`first < second` inside the group and `target[first] != target[second]`, the
higher target as winner, weight = the group weight, emitted in nested-loop
order. **No RNG is consulted on this path.** That is the path mojotrees
implements, and it is the path a default `loss_function=PairLogit` run takes.

`max_pairs` set is NOT implemented here and is refused rather than
approximated, for two reasons. It shuffles with `TRestorableFastRng64`, whose
stream is drawn from the unnamed global learn-progress generator; and its
branch condition is `pairCount / 2 < maxPairCount` *after* `pairCount` has
already been halved on the line above, so the brute-force branch is taken
whenever the true pair count is under `4 * max_pairs` and the truncation
inside `GenerateBruteForce` then fires anyway. Reproducing a bug we cannot
seed is worth nothing.

**Numerics, a deliberate divergence.** CatBoost forms
`e^{a_j} / (e^{a_j} + e^{a_i})` from stored exponentials, which overflows to
`inf/inf = NaN` once an approx passes about 709. mojotrees computes
`p = sigmoid(a_j - a_i)` through a branch on the sign of the difference, which
is finite for every finite pair of scores. Same value wherever CatBoost's is
finite, defined where CatBoost's is not. Classification: **(1) strictly less
work and exact** (one `exp` instead of two plus a division).

#### A20. YetiRank: what it samples, how many times, and what seeds it

This is the question that decides whether the objective is reproducible, so
it is answered in CatBoost's own terms first.

**What it samples.** `GenerateYetiRankPairsForQuery` does, per query, for
`permutationIndex` in `[0, permutationCount)`:

1. Copy the query's `expApproxes`.
2. `AddNoise`. In the default `Gumbel` arm this is
   `expApproxes[d] *= u / (1.000001f - u)` with `u = rand.GenRandReal1()`.
   Since `IsStoreExpApprox` lists `YetiRank`, the values are `exp(a)`, so the
   multiplication is `a_d + log(u) - log(1.000001 - u)` on the score scale:
   **logistic** noise, not Gumbel. The name in CatBoost's enum is wrong; the
   arithmetic is what it is. `u` is stored into a `float`, and `1.000001f` is
   a `float` literal, so CatBoost's noise is computed in single precision.
3. `StableSort` the query's indices by noisy value descending.
4. `CalcWeightsClassic` walks the sorted list once and charges each ADJACENT
   pair `0.15 * decayCoefficient * |rel[first] - rel[second]|`, with
   `decayCoefficient` starting at 1 and multiplied by `decay` after each
   position. The charge is added to `competitorsWeights[winner][loser]` where
   the winner is the member with the larger relevance; an equal-relevance pair
   is charged nothing (`AddWeight` has no `==` branch).

After all permutations, every nonzero cell becomes a competitor with weight
`queryWeight * cell / permutationCount`, scanned winner-major then
loser-major.

**How many times.** `GetYetiRankPermutations` defaults to **10**.

**Decay.** `GetYetiRankDecay` defaults to **0.85**. The `Decay = 0.99` field
initializer inside `TYetiRankPairWeightsCalcer::TConfig` is DEAD: the
constructor overwrites it unconditionally on the next line. Reading 0.99 off
the header and calling it CatBoost's default is wrong by a factor that
compounds over the whole ranking.

**The 0.15.** A literal in the source, commented `// Like in GPU`. It scales
every pair weight uniformly inside a query, so it is absorbed by the learning
rate up to the interaction with `reg_lambda`; it is carried across anyway
because that interaction is real.

**What seeds it, and why we cannot copy it.** `UpdatePairsForYetiRank` splits
the query range into `CB_THREAD_LIMIT` blocks, draws one `ui64` per block from
`GenRandUI64Vector(blockCount, randomSeed)`, and inside a block runs a single
`TFastRng64` from which each query in turn takes `rand.GenRand()` as its own
seed. Query `q`'s stream therefore depends on how many queries precede it
*inside its block*, and the per-document uniforms inside a query are one
sequential run shared by all 10 permutations. This is deterministic in
CatBoost only because `CB_THREAD_LIMIT` is a compile-time constant rather than
the thread count -- the same accident that makes MVS reproducible there (A11).
It is not a stream that can be named from outside, and it is exactly the
shape this repository forbids.

mojotrees keys instead on `(seed, iteration, query, permutation)` through
`_yetirank_stream`, with document `d` reading `stream + d`. Nothing advances,
nothing is shared between permutations, and no draw depends on how many
queries or documents were processed first. The draws differ from CatBoost's;
the distribution does not. Bit-identity was never on offer, because their
stream has no name.

**Storage, and a real saving.** CatBoost allocates a dense `querySize x
querySize` float matrix per query and then scans all of it, which is O(n^2)
time and memory in the group size for a mode that can only ever touch
`permutations * (querySize - 1)` cells. mojotrees collects the charged pairs
into a flat list, sorts it by `(winner, loser)` and merge-sums runs. The sort
key reproduces CatBoost's winner-major/loser-major emission order exactly, so
the pair sequence handed to the gradient is the same sequence, and the
summation order inside a cell is fixed by permutation index. Classification:
**(1) strictly less work and exact** in the pair set and its order;
the per-cell sum is a reassociation of the same 10-or-fewer addends and is
therefore **(3) bit-moving** at the last ulp.

**Modes not implemented.** `mode` in {`DCG`, `NDCG`, `MRR`, `ERR`, `MAP`},
`noise=Gauss`, `num_neighbors != 1`, and `top`. `Classic` with
`noise=Gumbel` and `num_neighbors=1` is the constructed default and is the
whole of what a default `loss_function=YetiRank` run does; `top` is not read
by `CalcWeightsClassic` at all. The others are refused by name rather than
silently ignored.

**`YetiRankPairwise` is a different loss** and is not implemented. It shares
the pair generator but is trained through CatBoost's pairwise scoring path
(`pairwise_scoring.cpp`), not through leafwise `PairLogit` derivatives.

#### Three further YetiRank facts from `catboost_options.cpp`, verified

`SetNotSpecifiedOptionsToDefaults` and `GetEstimationMethodDefaults` say
things about a `YetiRank` fit that a comparison harness has to honor or the
comparison is against a differently-configured CatBoost:

1. **The default eval metric is `PFound`**, not NDCG:
   `lossDescription.Load(LossDescriptionToJson("PFound")); MetricOptions->
   ObjectiveMetric.Set(lossDescription);` in the `YetiRank` /
   `YetiRankPairwise` case. `PairLogit`'s objective metric is `PairLogit`
   itself; mojotrees implements neither `QueryRMSE` nor `PairLogit` as a
   *metric*, so the registry diff points those two at `ndcg_catboost` and
   records the divergence rather than inventing a metric.
2. **`l2_leaf_reg` defaults to 0 under `YetiRank`**, not to CatBoost's usual
   3 (`defaultL2Reg = 0`). A harness that leaves `l2_leaf_reg` at 3 while
   CatBoost silently used 0 is comparing two different regularizations.
3. **Leaf estimation is Newton with exactly one iteration**
   (`defaultEstimationMethod = ELeavesEstimation::Newton`,
   `defaultNewtonIterations = 1`), and CatBoost *refuses* to let you change
   it: "At the moment, in the YetiRank mode, changing the
   leaf_estimation_method parameter is prohibited." Backtracking is likewise
   forced off. So catalog A6 (`leaf_estimation_iterations`) has no bearing on
   a YetiRank comparison.

CatBoost also records that `YetiRank` "cannot be used as a metric"
(`IsSkipInMetricsParamsExport`), which is why there is no YetiRank eval
metric to implement.

#### Why `YetiRank` is not aliased to `lambdarank`

Recorded by the naming lane and restated here because it is the mistake a
reader of this file is most likely to make. LambdaRank weights a pair by the
NDCG change that swapping it would cause, computed on the *deterministic*
current ranking. YetiRank weights a pair by how often it lands adjacent in a
*noisy* ranking, with a positional decay and a relevance-difference factor,
and never computes an NDCG at all. They agree only in being pairwise. An alias
would make a comparison table say two libraries were run on the same loss when
they were not.

### A21 note: `group_id` and the two eval metrics, verified from source

#### The `group_id` contract

`catboost/libs/data/objects.cpp::CheckGroupIds` is the whole of it:

- Rows of one group must be **CONTIGUOUS**. The function walks the column once
  collecting the id of each maximal run, then sorts the run ids and calls
  `std::adjacent_find`; a repeat means some group's rows were interleaved with
  another's, and it raises **`"group Ids are not consecutive"`**.
- Groups need **NOT be sorted** by id, and the ids need not be dense, ordered,
  or numeric (CatBoost hashes string ids through `CalcGroupIdFor`). Only the
  runs matter.
- The failure is **loud**. CatBoost does not regroup, does not sort, and does
  not drop. This is the one thing that has to be true, because a ranking
  objective reading a mis-grouped column produces a plausible model and a
  meaningless one, and nothing downstream can detect it.
- `catboost/libs/data/target.cpp::CheckGroupWeights` additionally requires the
  weight to be `FuzzyEquals`-constant inside a group: a group weight is a
  property of the group, and a per-row weight that varies inside one is
  refused rather than averaged.

mojotrees already had exactly the first three: `ranking.groups_from_query_ids`
walks the runs, sorts the run ids, and raises
`"query ids must be contiguous: rows of query N are not consecutive"`. It was
built against LightGBM and it happens to match CatBoost's rule line for line.
It is reused unchanged and re-exported under CatBoost's name. The fourth,
constant weight inside a group, did not exist and is added
(`check_group_weights_constant`).

#### `NDCG`, and how it differs from the `ndcg` mojotrees already has

`metric.cpp::TDcgMetric` + `dcg.cpp::{CalcNdcg,CalcDcg,CalcIDcg,CalcDcgSorted,
FillDcgDecay,GetTopSortedTargets}` + `doc_comparator.h::CompareDocs`.

Defaults, from `TDcgMetric`: `DefaultTopSize = -1` (passed as `ui32`, so it
becomes `Max<ui32>`, i.e. no truncation), `DefaultMetricType =
ENdcgMetricType::Base`, `DefaultDenominatorType =
ENdcgDenominatorType::LogPosition`, `UseWeights` default true.

    numerator(rel)  = rel                    (Base, the DEFAULT)
                    = 2^rel - 1              (Exp)
    decay[0]        = 1
    decay[i]        = 1 / log2(i + 2)        (LogPosition, the DEFAULT)
                    = 1 / (i + 1)            (Position)
    DCG             = sum_{i < top} numerator(target[order[i]]) * decay[i]
    IDCG            = same, over targets sorted DESCENDING by target alone
    NDCG            = IDCG > 0 ? DCG / IDCG : 1
    reported        = sum_q qw_q NDCG_q / sum_q qw_q      (0 if the sum is 0)

Three differences from `ranking.ndcg`, which is LightGBM's, and every one of
them changes the number:

1. **The gain.** CatBoost's default is `Base`: the numerator is the raw
   relevance. LightGBM's is `2^l - 1`, which is CatBoost's `Exp`. A graded
   relevance set scored under the wrong one is not a rescaling; it reorders
   which of two models is better.
2. **The truncation.** CatBoost's default is none. LightGBM's `eval_at`
   defaults to 5 and mojotrees follows it (`DEFAULT_NDCG_EVAL_AT`).
3. **The tie rule.** `CompareDocs` is
   `approxLeft != approxRight ? approxLeft > approxRight : targetLeft <
   targetRight`: on an exact score tie the LOWER relevance is ranked FIRST.
   That is adversarial on purpose and is what makes an untrained model score
   near its floor instead of at whatever the input order happened to give.
   LightGBM stable-sorts and keeps input order, which flatters the model on
   any dataset that arrives sorted by relevance -- and ranking datasets
   frequently do.

`GetTopSortedTargets` uses `PartialSort` with an index tiebreak when
`top < size` and `StableSort` otherwise. Both are the same total order
(`CompareDocs`, then input index), so mojotrees does one stable sort and
truncates. Classification: **(1) strictly less work and exact** -- one sort
instead of a branch over two.

`CalcIDcg` sorts by `left.Target > right.Target` alone, stably, so the ideal
ordering keeps input order among equal relevances. That matters only for
`top < size`.

#### `PFound`

`pfound.h::TPFoundCalcer::AddQuery` + `metric.cpp::TPFoundMetric`. Defaults
`DefaultTopSize = -1` (`ui32`, so no truncation), `DefaultDecay = 0.85`,
`UseWeights` default true.

    order    = StableSort by CompareDocs           (the SAME tie rule)
    pLook    = 1, pFound = 0
    for position in [0, min(size, top)):
        pRel   = relevance[order[position]]
        pFound += pRel * pLook
        pLook  *= (1 - pRel) * decay
    reported = sum_q qw_q pFound_q / sum_q qw_q    (0 if the sum is 0)

`GetBestValue` is `Max` with `bestValue = 0`, which is a CatBoost quirk (the
attainable maximum is 1, not 0) and is not carried across; mojotrees records
only "higher is better".

**Relevances are probabilities here**, not grades: the recursion is only a
probability if `pRel` is in `[0, 1]`. CatBoost does not check. mojotrees does
not check either, because refusing would break the CatBoost-compatible reading
of a graded column, but `pfound` documents the range and the value is
meaningless outside it.

**`subgroup_id` is not implemented.** CatBoost skips a document whose
`subgroupId` was already seen at a higher position, which deduplicates
near-identical results. There is no subgroup column in mojotrees and inventing
one is a data-model change this lane does not own. A `PFound` computed here on
data that HAS subgroups in CatBoost will read high. Named, not hidden.

#### The metric-name collision, which needs a ruling

`NDCG` is not a CatBoost-only concept, so `docs/PARAMETER_NAMING.md`'s rule
does not settle it, and the two vendors' `NDCG` are different functions (see
above). mojotrees already spells LightGBM's as `ndcg`. This lane registers the
CatBoost one as `ndcg_catboost` (alias `catboost_ndcg`), which is a mojotrees
coinage and not anybody's existing name; it is the only truthful option short
of carrying CatBoost's `NDCG:type=Base;denominator=LogPosition` parameter
syntax through a registry that has one scalar slot per metric. `PFound` has no
collision and is registered under CatBoost's own name, `pfound`. The naming
lane owns the final spelling of the first.
