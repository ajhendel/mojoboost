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
| A17 | `text_processing`: the tokenizer and the dictionaries (`library/cpp/text_processing/tokenizer/tokenizer.cpp` + `options.h`, `library/cpp/text_processing/dictionary/dictionary_builder.cpp` + `options.h`, `catboost/private/libs/options/text_processing_options.cpp`/`.h`, `catboost/libs/data/quantization.cpp::CreateDictionaries`, `catboost/private/libs/text_processing/dictionary.h::CreateDictionary`, `catboost/private/libs/data_types/text.h`) -- **verified from source**, see the A17 note | Turns a raw text column into a bag of dictionary token ids with counts. Default tokenizer splits on the literal string `" "` and does nothing else; default dictionaries are a unigram and a bigram over words, `occurrence_lower_bound=3`, `max_dictionary_size=50000` | No: pure host-side feature GENERATION feeding the existing binning path. Nothing in the trainer changes | Yes. It is the only way a text column becomes comparable at all; without it there is a whole class of dataset on which no CatBoost comparison exists | CPU (`text_processing.mojo`, NEW, imports nothing from the package) | (1) strictly-less-work does not apply -- it is new work that did not exist. Default OFF: no existing entry point calls it. A17 note |
| A18 | `text_features`: the `BoW` / `NaiveBayes` / `BM25` estimators (`catboost/private/libs/text_features/bow.cpp`, `naive_bayesian.cpp`, `bm25.cpp`, `feature_calcer.h`, `helpers.h`; `catboost/private/libs/feature_estimator/base_text_feature_estimator.h`, `text_feature_estimators.cpp`; `catboost/private/libs/algo/estimated_features.cpp`, `fold.cpp`, `data.cpp`, `full_model_saver.cpp`) -- **verified from source**, see the A18 note | Three numeric features off a token bag. `BoW` is target-free. **`NaiveBayes` and `BM25` are target-aware and ARE computed on the ordered permutation prefix, the same machinery the CTRs use** | Feature values only; they are handed to `binning.fit_bins` like any other numeric column | Yes, opt-in. `BoW` and the calcers are ours today; the ordered pass CONSUMES `lane/ordered-ts-2`'s permutation and must never build its own | CPU (`text_features.mojo`, NEW, imports only `text_processing`) | (3) bit-moving when on, inert when off. A18 note, and read the leakage section of it before writing any caller |
| A19 | Ordered target statistics, the simple (single-cat-feature) projection (`catboost/private/libs/algo/online_ctr.{h,cpp}`, `.../ctr_helper.{h,cpp}`, `.../fold.cpp`, `.../split.cpp`, `catboost/libs/model/online_ctr.h`) -- **verified from source**, see the A19 note | The mechanism A5 named. A categorical value becomes a *numeric* feature whose value at row `i` is a target statistic over the rows that precede `i` in a random permutation, quantized to 16 buckets. Four CTR types, three priors, an unshuffled fold 0, and a separate non-online table for inference | Yes -- it adds features that did not exist | Yes. This is the one mechanism that makes a categorical CatBoost comparison possible at all, and `bench/real_data` runs no such scenario today because we had no counterpart | CPU (`ctr.mojo`, new file; `categorical.mojo` untouched) | A19 note. **BUILT, off by default, unreached.** (3) when on. Deterministic under the seed, across `MOJOTREES_NUM_WORKERS`, and across machines, by a keyed sort rather than a shuffle |

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

### A17 note: the tokenizer and the dictionaries, verified from source

Read 2026-08-16 from `github.com/catboost/catboost` at `master`. Numbering:
A16 was the highest row in the catalog at `bfd6187` when this was written, so
this lane took A17 and A18. Files:

- `library/cpp/text_processing/tokenizer/options.h` -- `TTokenizerOptions`
  and every field default.
- `library/cpp/text_processing/tokenizer/tokenizer.cpp` --
  `SplitByDelimiter` (both overloads), `FilterNumbers`, `ProcessWordToken`,
  `TTokenizer::Tokenize`, `TokenizeWithoutCopy`.
- `library/cpp/text_processing/dictionary/options.h` --
  `TDictionaryOptions`, `TDictionaryBuilderOptions`.
- `library/cpp/text_processing/dictionary/dictionary_builder.cpp` --
  `TUnigramDictionaryBuilderImpl::AddImpl` / `FinishBuilding`,
  `TMultigramDictionaryBuilderImpl::AddImpl` / `Filter` / `CompareNGram`.
- `library/cpp/text_processing/dictionary/util.h` -- `GetMaxDictionarySize`.
- `library/cpp/text_processing/dictionary/frequency_based_dictionary_impl.cpp`
  -- `TUnigramDictionaryImpl::GetTopTokens`, `GetUnknownTokenId`.
- `library/cpp/text_processing/dictionary/types.h` -- `EUnknownTokenPolicy`.
- `catboost/private/libs/options/text_processing_options.h` --
  `DEFAULT_DICTIONARY_OPTIONS`, `DEFAULT_DICTIONARY_BUILDER_OPTIONS`.
- `catboost/private/libs/options/text_processing_options.cpp` --
  `TTextProcessingOptions::SetDefault`, `SetNotSpecifiedOptionsToDefaults`.
- `catboost/libs/data/quantization.cpp` -- `CreateDictionaries`.
- `catboost/private/libs/text_processing/dictionary.h` -- `CreateDictionary`.
- `catboost/private/libs/text_processing/dictionary.cpp` --
  `TDictionaryProxy::Apply`, `GetTopTokens`.
- `catboost/private/libs/text_processing/text_column_builder.cpp` --
  `TTextColumnBuilder::AddText`.
- `catboost/private/libs/data_types/text.h` -- `TText`.

**The default tokenizer is almost nothing, and that is the finding.**
`TTextProcessingOptions::SetDefault` installs one tokenizer named `"Space"`
constructed from a default `TTokenizerOptions()`, which is
`Lowercasing=false`, `Lemmatizing=false`,
`NumberProcessPolicy=LeaveAsIs`, `SeparatorType=ByDelimiter`,
`Delimiter=" "`, `SplitBySet=false`, `SkipEmpty=true`. `SplitByDelimiter`
then reduces to `StringSplitter(s).SplitByString(" ").SkipEmpty()`. So the
default tokenizer splits the raw UTF-8 bytes on the literal one-character
string `" "`, drops empty pieces, and **does not** lowercase, strip
punctuation, strip numbers, lemmatize, or normalize Unicode. `"Cat,"` and
`"cat"` are two different tokens at CatBoost's defaults. The `BySense`
separator, the lemmatizer and the token-type filter exist but are not
reachable from any default and the open-source lemmer is a `Y_ENSURE(false)`
stub (`TTokenizer::Initialize`).

**The default dictionaries are two, not one.** `SetDefault` installs
`"BiGram"` (`{ETokenLevelType::Word, GramOrder=2}`) and `"Word"`
(`{Word, GramOrder=1}`), in that order, and the default `BoW` processing
consumes both. So a text column at CatBoost's defaults produces `BoW` over a
bigram dictionary AND `BoW` over a unigram dictionary. The
`SetNotSpecifiedOptionsToDefaults` path, taken when the user supplied a
partial `text_processing` block, falls back to `"Word"` alone -- a different
default from the one an untouched fit gets, which is a trap for anyone
comparing configurations.

**The bounds are CatBoost's, not the library's.** The dictionary library
defaults `OccurrenceLowerBound=50` and `MaxDictionarySize=-1` (unbounded);
CatBoost overrides both in `DEFAULT_DICTIONARY_BUILDER_OPTIONS` to **3** and
**50000**. Quoting the library's 50 would be wrong for CatBoost.
`GetMaxDictionarySize(-1)` is `Max<ui32>()`, so -1 means unbounded and any
other non-positive value is refused.

**How the two bounds compose, which is the part that matters.**
`FinishBuilding` first DROPS every token whose corpus occurrence count is
`< OccurrenceLowerBound`, then sorts the survivors by **count descending,
token ascending**, then keeps the first `min(survivors, MaxDictionarySize)`
and assigns ids `StartTokenId, StartTokenId+1, ...` in that order. Three
consequences: the occurrence bound is a filter and the size bound is a
truncation, applied in that order and never interchangeable; **token id
order IS frequency order**, which is what makes `GetTopTokens(n)` on the
CatBoost side a bare `xrange(n)` (`TDictionaryProxy::GetTopTokens`); and the
count is total occurrences over the learn corpus, not document frequency
(`TokenToCount[token] += weight` per occurrence, `weight=1` per document).

**The dictionary is fitted on the learn pool only.** `CreateDictionaries`
runs inside learn quantization; test pools are quantized against an already
built `TQuantizedFeaturesInfo` and reuse the same digitizer. Unknown tokens
at apply time are dropped, not mapped to a sentinel:
`EUnknownTokenPolicy` defaults to `Skip` at every `Apply` overload.

**A document is a bag, and the bag is sorted.** `TText(TVector<ui32>&&)`
sorts the applied token ids ascending and run-length-encodes them into
`(tokenId, count)` pairs. Every estimator below iterates a document in
ascending token id. That is not incidental; it is what makes the Float64
accumulations in A18 order-fixed without anyone arranging for it.

**Verified but not implemented here**, each refused by name rather than
silently ignored: `ETokenLevelType::Letter`, `SkipStep > 0` (skip-grams),
`EEndOfWordTokenPolicy`, a non-`Skip` `EEndOfSentenceTokenPolicy`,
`SubTokensPolicy`, `BySense` separation, lemmatization, BPE dictionaries,
and `StartTokenId != 0`.

### A18 note: the three estimators, and the leakage answer

Read 2026-08-16 from `master`. Files:

- `catboost/private/libs/text_features/bow.cpp` / `.h`,
  `naive_bayesian.cpp` / `.h`, `bm25.cpp` / `.h`.
- `catboost/private/libs/text_features/feature_calcer.h` / `.cpp` --
  `ActiveFeatureIndices`, `ForEachActiveFeature`, `TrimFeatures`,
  `FeatureCount`.
- `catboost/private/libs/text_features/helpers.h` -- `Softmax`.
- `catboost/private/libs/feature_estimator/base_text_feature_estimator.h` --
  `TTextBaseEstimator::ComputeFeatures`, **`ComputeOnlineFeatures`**,
  `EstimateFeatureCalcer`, `MakeFinalFeatureCalcer`, `Calc`.
- `catboost/private/libs/feature_estimator/feature_estimator.h` --
  `IFeatureEstimator` vs `IOnlineFeatureEstimator`.
- `catboost/private/libs/feature_estimator/text_feature_estimators.cpp` --
  `TNaiveBayesEstimator`, `TBM25Estimator`, `TBagOfWordsEstimator`, and the
  two `CreateTextEstimators` overloads.
- `catboost/private/libs/algo/estimated_features.cpp` --
  `CreateEstimatedFeaturesData`, `const bool isOnline = learnPermutation.Defined()`.
- `catboost/private/libs/algo/fold.cpp` -- `TFold::InitOnlineEstimatedFeatures`
  and its two call sites in the dynamic and plain fold builders.
- `catboost/private/libs/algo/data.cpp` -- `CreateEstimators`, and the
  offline `CreateEstimatedFeaturesData` call with `/*learnPermutation*/ Nothing()`.
- `catboost/private/libs/algo/full_model_saver.cpp` -- `MakeFinalFeatureCalcer`.

#### The leakage answer

**`NaiveBayes` and `BM25` are target-aware, and CatBoost prevents the leak
exactly the way it prevents the CTR leak: an ordered prefix over the fold's
learn permutation. They do not merely resemble the CTR machinery; they are
driven by the same permutation array.**

The split is structural, not conditional. `IFeatureEstimator` has one
`ComputeFeatures`; `IOnlineFeatureEstimator` adds `ComputeOnlineFeatures`.
`TFeatureEstimators` holds two disjoint vectors and
`CreateEstimatedFeaturesData` picks one by
`const bool isOnline = learnPermutation.Defined()`. `CreateTextEstimators`
has two overloads: the one that takes a `TClassificationTargetPtr` returns
`NaiveBayes` and `BM25` as online estimators, the one that does not returns
`BoW` as an offline estimator. `BoW` never sees a target and is never online.

The ordered pass, verbatim in shape:

```
for (ui64 line : learnPermutation) {
    const TText& text = ds.GetText(line);
    Compute(featureCalcer, text, line, samplesCount, learnFeatures);   // read
    calcerVisitor.Update(target.Classes[line], text, &featureCalcer);  // then write
}
```

Read strictly before write. Row `i`'s feature is computed from a calcer
holding the statistics of exactly the rows that precede `i` in that
permutation, and row `i`'s own target enters the calcer only afterwards. The
first row of the permutation is scored against an empty calcer. That is the
ordered-target-statistic contract, spelled with a different accumulator.

The permutation is the FOLD's: `TFold::InitOnlineEstimatedFeatures` passes
`GetLearnPermutationArray()`, and it is called from both fold builders
immediately before `InitOnlineCtrs`. So with `k` CTR permutations there are
`k` independently ordered copies of every online text feature, one per fold,
and they are the same permutations the CTRs use. **A second permutation
implementation for text would be a bug, not a duplication.**

#### The train-versus-predict asymmetry

Three different calcer states exist, and confusing any two of them is where
implementations of this go wrong.

1. **Learn rows during training.** Prefix state, per fold, as above.
   Different in every fold; different for the same row in different folds.
2. **Test/eval rows during training.** `ComputeOnlineFeatures` finishes the
   permutation loop and then calls `Calc(featureCalcer, GetTestDataSets(), ...)`
   with the calcer in its FINAL state -- every learn row folded in. Not a
   prefix. A test row is not part of the learn ordering and has no position
   in it.
3. **Predict time, from a saved model.** `full_model_saver.cpp` calls
   `estimator->MakeFinalFeatureCalcer(...)`, which calls
   `EstimateFeatureCalcer()` -- a fresh calcer walked over the learn set in
   **natural row order**, no permutation -- then `TrimFeatures` to the
   features the model actually used. That frozen calcer is serialized into
   the model.

**So the answer to "what is a text feature's value at predict time when
there is no target" is: the target is not needed, because it was consumed at
fit time into the frozen frequency tables.** At predict time
`TMultinomialNaiveBayes::Compute` / `TBM25::Compute` are pure functions of
(the document's token bag, the frozen per-class tables). No target is read,
and none exists. The asymmetry is not "features are unavailable at predict"
-- it is that the training-time value is a prefix statistic and the
predict-time value is a full-corpus statistic, and they are systematically
different numbers for the same document. That is deliberate and is the same
asymmetry ordered CTRs have.

One further consequence worth stating because it is easy to get backwards:
**the ordered pass is a training-input construction, not a model.** The
model ships state 3 only. Nothing in a saved model can reproduce state 1.

#### The formulas, exactly

**`BoW`** (`TBagOfWordsEstimator`, `TBagOfWordsCalcer`). Target-free,
offline. `top_tokens_count` defaults to **2000**, clamped to the dictionary
size, and refused at zero. The features are the dictionary's top
`top_tokens_count` tokens -- which, because ids are assigned in descending
count order (A17), is `GetTopTokens(n) == xrange(n)`. Feature `t` of
document `d` is **1 if token id `t` occurs in `d`, else 0**: a binary
indicator, not a count and not a TF-IDF. `UniqueValuesUpperBoundHint = 2`
per feature, and CatBoost packs them 32 to a `ui32`.

**`NaiveBayes`** (`TMultinomialNaiveBayes`). `ClassPrior = TokenPrior =
DEFAULT_PRIOR = 0.5`; `SEEN_TOKENS_PRIOR = 1`. Per class `c`, with `text`
iterated in ascending token id:

```
value  = log(ClassDocs[c] + ClassPrior)
denom  = ClassTotalTokens[c] + TokenPrior * (NumSeenTokens + 1)
textLen = 0
for (token, count) in text:
    textLen += count
    num = TokenPrior + (Frequencies[c][token] if present else 0)
    if token not present in Frequencies[c]:  denom += TokenPrior
    value += count * log(num)
value -= textLen * log(denom)
```

then `Softmax` (max-shifted) over the `NumClasses` values, and the emitted
features are the active indices, which default to `[0, BaseFeatureCount)`
with `BaseFeatureCount = NumClasses > 2 ? NumClasses : 1`. **So binary
classification emits ONE feature**, the softmax probability of class 0, not
two.

Two things in that block are easy to mis-transcribe and are not typos in
CatBoost. `classTokensCount` is a **by-value** parameter, so the
`denom += TokenPrior` for an unseen token is local to the class currently
being scored: the denominator grows by the number of the document's tokens
this class has never seen, and therefore differs per class for the same
document. And `NumSeenTokens` is the count of DISTINCT token ids seen across
ALL classes so far (`TNaiveBayesVisitor::SeenTokens`, `Insert` then
`NumSeenTokens = SeenTokens.Size()`), so in the ordered pass it grows as the
prefix grows.

**`BM25`** (`TBM25`). `k = 1.5`, `b = 0.75`, `truncateBorder = 1e-3`, and
`TBM25Estimator` constructs with all three defaulted. `BaseFeatureCount =
NumClasses`, so binary emits two. `TotalTokens` is initialized to **1**, not
0. The class, not the document, plays the role of the BM25 document:

```
meanClassLength = TotalTokens / NumClasses
for (token, count) in text:                    # `count` is NOT used
    nz = number of classes whose table contains `token`
    for c in classes:
        tf = Frequencies[c][token] or 0
        s  = 0 if tf == 0 else tf*(k+1) / (tf + k*(1 - b + b*meanClassLength/ClassTotalTokens[c]))
        scores[c] += TruncatedInvClassFreq[nz] * s
TruncatedInvClassFreq[j] = max(log((NumClasses - j + 0.5) / (j + 0.5)), truncateBorder)
```

Three divergences from textbook BM25, all deliberate on CatBoost's side and
all reproduced rather than corrected. The length normalization is
**inverted**: textbook BM25 has `b * docLen / avgDocLen`, which penalizes
long documents; CatBoost has `b * meanClassLength / classLength`, which
REWARDS long classes. The in-document term frequency is discarded -- the
loop reads `tokenToCount.Token()` and never `Count()`, so a token occurring
five times in a document contributes exactly as much as one occurring once.
And the IDF is over CLASSES, a number in `[0, NumClasses]`, floored at
`truncateBorder` rather than at zero, so a term present in every class still
contributes `1e-3` times its score instead of nothing. The `tf == 0` early
return is also the only thing standing between the formula and a divide by
`ClassTotalTokens[c] == 0` on the first rows of an ordered pass, so it must
stay ahead of the division.

**Which estimators are on by default.** `SetDefault` installs `BoW` for
every objective and `NaiveBayes` additionally when the objective is a
classification one. **`BM25` is NOT a CatBoost default** -- it exists, it is
documented, and nothing turns it on unless the user names it.
`CreateEstimators` says so in a comment as well: "There're no online text
estimators for regression for now", and `TTextProcessingOptions::Validate`
refuses a classification-only calcer on a regression objective by name.

#### What mojotrees built

`src/mojotrees/text_processing.mojo` (tokenizer, dictionary, digitizer) and
`src/mojotrees/text_features.mojo` (the three estimators). Both are new
files, both are OFF by default in the only sense available to them: no
existing entry point calls either, no existing default changed, and no
existing module imports them. `text_processing` imports nothing from the
package and `text_features` imports only `text_processing`, so the pair is a
leaf of the import graph and cannot participate in the
`efb -> binning -> tree_parameters_extra` cycle.

**They generate numeric columns and stop.** The output of every feature
function is a column-major `List[Float64]` laid out `out[f * n_rows + r]`,
which is exactly what `binning.fit_bins` and `RawData.dense` already read.
There is no new trainer, no new binner, and no new split rule. A caller
concatenates these columns onto its numeric matrix and the existing pipeline
takes it from there.

**The ordered pass consumes a permutation; it does not make one.**
`naive_bayes_online_features` and `bm25_online_features` take
`permutation: List[Int]` and validate that it is a permutation of
`0 .. n_rows - 1`. Building that permutation is `lane/ordered-ts-2`'s job and
must stay there; see the lane report for the two functions this needs from
it. Until that lane lands, the only callers are tests passing the identity
permutation, which is the honest statement that the ordered path is built
and unreached.

**Determinism.** No random draw anywhere in either module, so there is no
domain constant and no `sampling.mojo` pattern to follow. The one real
hazard is dictionary construction, and it is handled the way CatBoost
handles it: counts accumulate in a hash map, but the map is never iterated
into the result. `_dictionary_order` sorts by **(count descending, key
ascending)**, a strict total order on distinct keys, so the id assignment is
independent of hash order, insertion order, allocation addresses, and
`MOJOTREES_NUM_WORKERS`. Everything else is sequential over ascending row
index or over the caller's permutation. Nothing in either module reads a
worker count or spawns a task.

**Bounds, derived.** For a dictionary of `D` entries whose keys have mean
byte length `L`: the fitted dictionary holds the key strings twice, once in
the `id -> key` list and once in the `key -> id` map, so
`2 * D * (L + 16)` bytes plus `8 * D` for the counts. At CatBoost's default
`max_dictionary_size = 50000` and an English unigram mean of about 8 bytes,
that is a derived bound of roughly 2.5 MB per dictionary per text column --
small, and the reason the size bound exists at all is the *pre-filter*
count, which is unbounded and is what `occurrence_lower_bound = 3` cuts
first. `BoW` at the default `top_tokens_count = 2000` costs `2000 * n_rows *
8` bytes of Float64 columns before binning, which is **16 KB per row** and
is the actual memory statement anyone enabling this needs: at a million rows
that is 16 GB, which is not a bound, it is a refusal. `BoW` is therefore
built but is the one estimator whose default this lane would not ship
without a packed representation; `NaiveBayes` costs `max(1, num_classes)`
columns and `BM25` costs `num_classes`, both of which are free.

**What is deliberately not built**, each refused by name: `SkipStep > 0`,
`ETokenLevelType::Letter`, BPE, lemmatization, `BySense` separation, the
end-of-word / end-of-sentence token policies, `StartTokenId != 0`,
per-feature `text_processing` blocks, the embedding estimators, and the
packed-binary `BoW` representation.

### A19 note: ordered target statistics, verified from source

Read 2026-08-16 from `github.com/catboost/catboost` at `master`. This note
discharges A5, which was written from the paper, was marked *verify*, and is
wrong in three places that are named below. A5's row is left standing so the
diff is readable; **A19 supersedes it.**

Files read, and every claim below cites one of them.

- `catboost/private/libs/algo/online_ctr.h` -- `CalcCTR`, `CalcNormalization`,
  `ComputeOnlineCTRs`, `CalcFinalCtrsAndSaveToModel`, `SIMPLE_CLASSES_COUNT`.
- `catboost/private/libs/algo/online_ctr.cpp` -- `CalcOnlineCTRSimple`,
  `CalcOnlineCTRClasses`, `CalcOnlineCTRMean`, `CalcOnlineCTRCounter`,
  `CalcStatsForEachBlock`, `SumCtrsFromBlocks`, `CalcQuantizedCtrs`,
  `CountOnlineCTRTotal`, `CalcFinalCtrsImpl`, `CalcFinalCtrs`.
- `catboost/private/libs/algo/ctr_helper.h` -- `TCtrInfo`,
  `GetTargetBorderCount`, `TCtrHelper::GetCtrInfo`.
- `catboost/private/libs/algo/ctr_helper.cpp` -- `MakeCtrInfo`,
  `TCtrHelper::InitCtrHelper`.
- `catboost/private/libs/algo/fold.cpp` -- `InitPermutationData`,
  `TFold::BuildPlainFold`, `TFold::BuildDynamicFold`, `TFold::AssignTarget`.
- `catboost/private/libs/algo/learn_context.cpp` -- `IsPermutationNeeded`,
  `CountLearningFolds`, `TFoldsCreationParams`, the fold-construction loop,
  the `AveragingFold` construction.
- `catboost/private/libs/algo/train.cpp` -- the per-tree fold draw.
- `catboost/private/libs/algo/split.cpp` and `split.h` --
  `TSplit::GetModelSplit`'s CTR branch, `EmulateUi8Rounding`, `GetBucketCount`.
- `catboost/libs/model/online_ctr.h` -- `TModelCtr::Calc`, `TModelCtr`'s
  fields.
- `catboost/libs/model/hash.h` -- `CalcHash`.
- `catboost/libs/model/target_classifier.h` -- `GetTargetClass`,
  `GetClassesCount`.
- `catboost/libs/model/model_export/resources/ctr_calcer.py` -- the reference
  inference calculator, which is the readable statement of the predict path.
- `catboost/private/libs/ctr_description/ctr_type.h` and `.cpp` -- `ECtrType`,
  `NeedTarget`, `NeedTargetClassifier`, `IsPermutationDependentCtrType`.
- `catboost/private/libs/options/cat_feature_options.h` and `.cpp` --
  `GetDefaultPriors`, `TCatFeatureParams`'s constructor (the defaults).
- `catboost/private/libs/options/catboost_options.cpp` -- `SetCtrDefaults`,
  `CreateDefaultCounter`.
- `catboost/private/libs/options/boosting_options.cpp` -- `permutation_count`.
- `catboost/private/libs/options/defaults_helper.h` --
  `DefaultFoldPermutationBlockSize`.
- `catboost/libs/data/objects_grouping.cpp` -- `NCB::Shuffle`.

#### The arithmetic, which is smaller than the folklore

One inline function is the whole of it
(`online_ctr.h:128`, and it is eight lines including the signature):

```cpp
inline ui8 CalcCTR(float countInClass, int totalCount, float prior, float shift, float norm, int borderCount) {
    float ctr = (countInClass + prior) / (totalCount + 1);
    return (ctr + shift) / norm * borderCount;
}
```

Three things fall straight out of it.

**The training denominator is `totalCount + 1`, not `totalCount + priorDenom`.**
There is no `prior_denom` in the training path at all. It exists in the option
surface (`GetDefaultPriors` returns pairs, `{0, 1} {0.5, 1} {1, 1}`) and it is
refused at load time on the CPU -- `MakeCtrInfo` does
`CB_ENSURE(denom == 1.0, "Error: CPU could use only 1 as denom for ctrs currently")`
and then stores only `prior[0]` into `TCtrInfo::Priors`. So on the CPU the
second element of every prior pair is a checked constant. It survives into the
model as a real field (`TModelCtr::PriorDenom`) because the GPU path can set
it, and `split.cpp:79` writes `PriorDenom = 1.0f` unconditionally on the CPU.

**The return value is a `ui8` bin index, not a ratio.** The training-time CTR
feature is already quantized when it is produced. There is no separate
binarization pass over CTR values, and this is why the CPU refuses anything but
uniform CTR binarization
(`ctr_helper.cpp:29`, `"Error: CPU supports only uniform binarization for CTRS"`).
`(ctr + shift) / norm` lands in `[0, 1]`, the multiply by `borderCount` lands
in `[0, borderCount]`, and the implicit `float -> ui8` conversion truncates.
That is `borderCount + 1` buckets, which is exactly what
`GetBucketCount` returns for a CTR candidate (`split.cpp`,
`return splitCandidate.Ctr.BorderCount + 1;`).

**`shift` and `norm` exist only to map a prior outside `[0, 1]` back into it**
(`CalcNormalization`, `online_ctr.cpp:102`):

```cpp
float left = Min(0.0f, prior);  float right = Max(1.0f, prior);
(*shift)[i] = -left;            (*norm)[i] = (right - left);
```

At every default prior (`0`, `0.5`, `1`) that is `shift = 0, norm = 1`, so the
normalization is the identity for a stock fit and only bites for a
user-supplied negative or greater-than-one prior.

#### Defaults, all from source

| thing | value | citation |
|---|---|---|
| `simple_ctr` for every loss except PairLogit | `Borders` with priors `{0, 0.5, 1}` **and** `Counter` with prior `{0}` | `catboost_options.cpp::SetCtrDefaults`, `default:` arm |
| `simple_ctr` for `PairLogit`/`PairLogitPairwise` | `Counter` only | same, the PairLogit arm |
| `combinations_ctr` | the same two, built by the same arms | same |
| CPU default `Counter` construction | `TCtrDescription(ECtrType::Counter, GetDefaultPriors(Counter))` | `CreateDefaultCounter`, CPU branch |
| GPU default counter is a **different type** | `FeatureFreq`, not `Counter` | `CreateDefaultCounter`, GPU branch |
| `ctr_border_count` | **15** | `TCtrDescription(type, priors)` delegates to `TBinarizationOptions(EBorderSelectionType::Uniform, 15)` |
| `ctr_target_border_count` | **1** (so two target classes) | `TargetBinarization("target_binarization", TBinarizationOptions(EBorderSelectionType::MinEntropy, 1))` |
| `counter_calc_method` | **`SkipTest`** | `CounterCalcMethod("counter_calc_method", ECounterCalc::SkipTest)` |
| `max_ctr_complexity` | **4** | `MaxTensorComplexity("max_ctr_complexity", 4)` |
| `one_hot_max_size` | 2 | already A16 |
| `permutation_count` | **4** | `boosting_options.cpp`, `PermutationCount("permutation_count", 4)` |
| `fold_permutation_block` | `min(256, docCount / 1000 + 1)` when unset | `defaults_helper.h::DefaultFoldPermutationBlockSize` |
| `ctr_leaf_count_limit` | `Max<ui64>()`, i.e. no limit | `cat_feature_options.cpp` |
| `store_all_simple_ctr` | false | same |

**A5 was wrong that there is one CTR.** At the defaults a single categorical
column produces `Borders x 3 priors x 1 target border = 3` features **plus**
`Counter x 1 prior = 1`, so **four** numeric features per categorical column,
each quantized to 16 buckets. That number, not the mechanism, is the reason
CTRs are expensive.

#### The four CTR types, and which loop each one runs

Dispatch is `ComputeOnlineCTRs`' final `ExecRange`
(`online_ctr.cpp:732-800`). The branch is on the type **and on the target
class count**, which is the part that is easy to miss.

- **`Borders` with exactly 2 target classes** -> `CalcOnlineCTRSimple`.
  `goodCount` is the running count of class-1 rows for this category among
  earlier rows, `totalCount` is the running count of all of them. This is the
  default path for a binary target and the one that matters.
- **`Buckets`, and `Borders` with more than 2 classes** -> `CalcOnlineCTRClasses`.
  Per category it keeps a running count per class. `UpdateGoodCount` is the
  only difference between the two types
  (`online_ctr.cpp:115`) -- `Buckets` sets `goodCount = curCount` (the count of
  *this* class), `Borders` does `goodCount -= curCount` walking down from the
  total (the count of classes *above* this border). One emits
  `targetClassesCount` features, the other `targetClassesCount - 1`
  (`GetTargetBorderCount`).
- **`BinarizedTargetMeanValue`** -> `CalcOnlineCTRMean`. The accumulator is a
  `(Sum, Count)` pair and the increment is
  `elem.Add(float(permutedTargetClass[docId]) / targetBorderCount)`, i.e. the
  running mean of the *class index normalized to [0, 1]*, not of the raw
  target. `GetTargetBorderCount` returns 1 for it, so it is one feature.
- **`Counter`** -> `CalcOnlineCTRCounter`, and **it is not online at all.**
  `counterCTRTotal` is filled once by `CountOnlineCTRTotal` over the whole
  array before any row is emitted, and the denominator is
  `*MaxElement(counterCTRTotal...)` -- the largest category's count, not the
  row count. Every row of a category therefore gets the *same* value, and that
  value is `(fullCount + prior) / (maxCount + 1)`. `IsPermutationDependentCtrType`
  says so directly (`Counter` and `FeatureFreq` return false). A5's "counters"
  clause was right that they exist and wrong to file them under "ordered".

`counter_calc_method` decides only how much data feeds that one count.
`SkipTest` (the default) counts `learnSampleCount` rows; `Full` counts
`hashArr.size()`, learn plus every test set
(`online_ctr.cpp:716-728`). It is a **train-time transduction switch**, and at
its default CatBoost does not look at the test features. On the final table the
same switch appears again in `CalcFinalCtrs` (`totalSampleCount +=
GetTestSampleCount()` under `Full`).

#### The permutation machinery, which is the actual heart

**How many.** `CountLearningFolds` is
`return isPermutationNeededForLearning ? Max<ui32>(1, permutationCount - 1) : 1;`
so at the default `permutation_count = 4` there are **3** learning folds, not
4. The fourth is spent on the separate `AveragingFold`, built by its own
`BuildPlainFold` call in `learn_context.cpp:575`.

**When there is a permutation at all.** `IsPermutationNeeded` is a three-line
function and its first line is the one that matters:

```cpp
if (hasTime) { return false; }
if (hasCtrs) { return true; }
return isOrderedBoosting && !isAveragingFold;
```

So `has_time` **disables the permutation entirely** and the CTR degenerates to
a prefix statistic in dataset order, which is the correct behavior for a
genuinely time-ordered pool. And `hasCtrs` is not "the user asked for CTRs", it
is
`CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn() > OneHotMaxSize` --
**the presence of a categorical column too wide to one-hot is what turns the
permutation on**, for plain boosting as much as ordered. This is the coupling
between A16 and A19 and it runs in the direction people do not expect.

**Fold 0 is the identity.** The fold loop passes `shuffle = (foldIdx != 0)`
(`learn_context.cpp:502` for ordered boosting, `:529` for plain). The
`shuffle = false` branch of `InitPermutationData` builds
`std::iota(learnPermutation.begin(), learnPermutation.end(), 0)` and comments
that it exists only because "implementation requires permutation vectors to
exist even if they are not shuffled". One of the three learning folds is
therefore dataset order.

**The permutation is a BLOCK permutation, not a shuffle of rows.** `NCB::Shuffle`
with `permuteBlockSize > 1` shuffles `blocksCount` block indices and then
copies each block's rows out **in their original relative order**
(`objects_grouping.cpp:205-217`). With the default block size
`min(256, n/1000 + 1)`, a 1M-row pool is permuted in blocks of 256 and a
10k-row pool in blocks of 11; only below 1000 rows is it a true row shuffle.
The consequence is that **a row's ordered prefix is not a uniform random
subset**: rows in the same block always see each other in the same relative
order. That is a deliberate cache concession and it weakens the ordering
guarantee the paper argues for.

**One permutation per tree, drawn per tree.**
`train.cpp:208`,
`TFold* takenFold = &ctx->LearnProgress->Folds[ctx->LearnProgress->Rand.GenRand() % foldCount];`
-- the tree *structure* is searched against one randomly chosen fold's CTR
values and derivatives. The leaf *values* come from `AveragingFold`, a fourth
fold that is permuted only if `IsAverageFoldPermuted`. So a tree is grown
against one permutation and valued against another.

**What a row is allowed to see.** In `CalcOnlineCTRSimple`'s inner loop
(`online_ctr.cpp:300-307`) the read happens **before** the write, in
permutation order:

```cpp
auto& elem = ctrArrSimple[enumeratedCatFeatures[docOffset + docIdx]].N;
goodCount[...] = elem[1];
totalCount[...] = elem[0] + elem[1];
++elem[permutedTargetClass[docOffset + docIdx]];
```

Row `i` sees the rows strictly before it in the permutation, and never its own
target. The first occurrence of a category sees `countInClass = totalCount = 0`
and gets the pure prior. That read-then-write ordering is the entire leakage
argument.

#### Train versus predict, which is where implementations go wrong

They are **different formulas with different types over different data**, and
the two are reconciled by a third piece of arithmetic.

|  | training | inference |
|---|---|---|
| function | `CalcCTR` (`online_ctr.h:128`) | `TModelCtr::Calc` (`model/online_ctr.h:289`) |
| formula | `((c + prior) / (t + 1) + shift) / norm * borderCount` | `((c + PriorNum) / (t + PriorDenom) + Shift) * Scale` |
| result type | `ui8`, a bucket index | `float`, an unquantized value |
| denominator | `t + 1`, hard-coded | `t + PriorDenom`, serialized |
| counts come from | the **online prefix** in one permutation | a **static table** over the whole learn set |
| unseen category | cannot occur; the first row of a category is the prefix's own zero state | occurs, and yields `calc(0, 0)`, the pure prior (`ctr_calcer.py:35`) |

The reconciliation is two lines in `split.cpp:78-82`:

```cpp
split.OnlineCtr.Ctr.PriorNum  = priors[Ctr.PriorIdx];
split.OnlineCtr.Ctr.PriorDenom = 1.0f;
split.OnlineCtr.Ctr.Shift     = shift[Ctr.PriorIdx];
split.OnlineCtr.Ctr.Scale     = ctrInfo.BorderCount / norm[Ctr.PriorIdx];
split.OnlineCtr.Border        = EmulateUi8Rounding(BinBorder);
```

`Scale = borderCount / norm` folds the training multiply into the inference
scale, so the two produce the same real number and the training one then
truncates. The truncation is put back at the *threshold* instead of the value
by `EmulateUi8Rounding(value) { return value + 0.999999f; }` (`split.h:512`):
a training comparison `bin > BinBorder` on integers becomes an inference
comparison `x > BinBorder + 0.999999f` on the unquantized float. The name says
what it is. **This is the asymmetry to get right, and it is not the one people
warn about**; the folklore warning is "use full counts at predict time", which
is true but is the easy half.

The static table is `CalcFinalCtrsImpl` (`online_ctr.cpp:875`). It is a plain
loop over all `totalSampleCount` rows with no permutation and no prefix, and
for `Counter` it ends with
`result->CounterDenominator = *MaxElement(...)`. The inference-side lookup that
consumes it is `ctr_calcer.py:22-67`, which is worth reading once because it
states in twenty lines what the C++ spreads over three files -- including that
`Borders` at two classes reads `ctr_history[bucket*2+1]` against
`ctr_history[bucket*2] + ctr_history[bucket*2+1]`, while `Borders` above two
classes sums classes `> target_border_idx` into `good_count` and *all* classes
into `total_count`, and `Buckets` takes one class as `good_count` against the
same total.

#### The target classifier

`Borders`, `Buckets` and `BinarizedTargetMeanValue` all need one
(`NeedTargetClassifier`); `Counter` and `FeatureFreq` do not, and CatBoost
still keeps a fake classifier at index 0 for them, with a comment calling it a
dirty hack (`ctr_helper.h:22`). The classifier itself is four lines
(`target_classifier.h:24`):

```cpp
int resClass = 0;
while (resClass < Borders.ysize() && target > Borders[resClass]) { ++resClass; }
return resClass;
```

Strict `>`, ascending borders, `classesCount = borders + 1`. At the default
`ctr_target_border_count = 1` there is one border, chosen by **MinEntropy** over
the target, and two classes. Border *selection* is A15's mechanism, not this
one, so A19 takes borders as an input and does not re-derive them.

#### What mojotrees built

`src/mojotrees/ctr.mojo`, a new file, **off by default and reached by
nothing**, with `tests/test_ctr.mojo` beside it (48 tests, all analytical, all
run). It imports `.rng` and `std.math` and nothing else from the package, so it
cannot participate in the `efb -> binning -> tree_parameters_extra` cycle;
`categorical.mojo` is unchanged and no existing default moved. Classification:
**(3) bit-moving when on** -- it manufactures features that did not exist --
and a no-op when off.

Built, and matching the source above.

- `CtrParams`, defaulting `enabled = False` and otherwise carrying CatBoost's
  verified numbers (`ctr_border_count = 15`, `target_border_count = 1`,
  `permutation_count = 4`, `counter_calc_method = SkipTest`,
  `max_ctr_complexity = 4`), so a port of a default CatBoost configuration
  reads the same even though ours refuses to run.
- `calc_normalization`, `ctr_train_bin`, `ctr_predict_value`,
  `ctr_predict_scale`, `ctr_predict_border` -- the five pieces of arithmetic
  above, each a transcription with the file and line in its docstring.
- `ordered_ctr_borders_binary`, `ordered_ctr_classes`, `ordered_ctr_mean`,
  `counter_ctr` -- the four loops, read-before-write, in permutation order.
- `final_ctr_class_table`, `final_ctr_mean_table`, `final_ctr_counter_table`
  and the three `predict_ctr_*` readers, which are the inference half and are
  tested against the training half on the same data.
- `target_class`, `ctr_feature_count` (the "four features per column" number
  above, computed rather than asserted).
- `default_permutation_block_size` and `ctr_permutation`.
- `ctr_combination_hash`, `ctr_projection_hash` -- `CalcHash` from
  `model/hash.h`, left in place for the combinations lane and used by nothing
  here.
- Guards that refuse rather than ignore -- `CtrParams.validate`,
  `check_ctr_prior_denom` (the CPU `denom == 1` rule, by name),
  `check_ctr_border_type` (uniform only, by name), `check_ctr_complexity`, and
  `check_ctr_trainer_support`, which raises on an *enabled* bundle because
  nothing appends CTR columns to a design matrix yet. That last one is the
  honest statement of "unreached"; deleting it is the first step of the wiring
  lane rather than a side effect of it.

**Divergences, deliberate and recorded.**

1. **The permutation is a keyed sort, not a Fisher-Yates shuffle.**
   `ctr_permutation` gives block `b` the key `splitmix64(stream + b)` and sorts
   blocks ascending by `(key, b)`. CatBoost's `CreateShuffledIndices` is a
   sequential draw whose result depends on the order the draws are consumed in.
   Ours does not, so it is order-independent, parallelizable without changing
   its answer, and identical across `MOJOTREES_NUM_WORKERS` and across
   machines by construction rather than by discipline. It uses no
   floating-point at any point, so it is also immune to the FMA and
   x87-versus-SSE differences that would otherwise make "across machines"
   an unearned claim. It is a *different* permutation from CatBoost's for the
   same seed, and that is fine -- nothing here claims bit-identity with
   CatBoost.
2. **Fold 0 is the identity here too**, matching `foldIdx != 0`, so the
   `permutation_index = 0` caller gets dataset order.
3. **The online accumulation runs in row order, sequentially.** CatBoost
   parallelizes it (`CalcStatsForEachBlock` + `SumCtrsFromBlocks` +
   `CalcQuantizedCtrs`, with each block seeded from the exclusive prefix of the
   blocks before it) and gets away with it *because the accumulators are
   integers* -- the count of earlier rows in a category is exact under any
   blocking, so CatBoost's `Borders`/`Buckets`/`Counter` values do not move
   with `thread_count`. That escape does **not** extend to
   `BinarizedTargetMeanValue`, whose accumulator is a `float` sum, and
   accordingly CatBoost leaves `CalcOnlineCTRMean` serial. Ours is serial for
   all four, which is deterministic for the same reason and stricter than it
   needs to be for three of them. The integer-block-prefix parallelization is
   available later, exactly, for the three; it is not available for the mean.
4. **The accumulators are `Int` and `Float64`, and only the final
   `ctr_train_bin` narrows to Float32** to reproduce CatBoost's `float`
   truncation. Accumulating in Float64 and narrowing once is strictly more
   accurate than accumulating in Float32; the bucket index is the same except
   where the exact value sits within one ULP of a bucket edge.

**Not built, deliberately.** Combinations (`max_ctr_complexity` > 1 -- the next
lane, see below), `FloatTargetMeanValue`, the GPU-only `FeatureFreq`,
`ctr_leaf_count_limit`'s top-K reindexing, `PriorEstimation`, per-feature CTR
descriptions, `ctr_history_unit = Group`, and the `Dynamic` (ordered boosting)
fold shape, which is A7 and a different mechanism.

**Left in place for the combinations lane, explicitly.** `ctr_combination_hash`
is `CalcHash` verbatim and is what a projection over several categorical
columns folds with; `ctr_projection_hash` folds a list of hashed values with it
in order, which is `calc_hashes`' cat-feature loop
(`ctr_calcer.py:9-19`). `CtrParams.max_ctr_complexity` exists, defaults to
CatBoost's 4, and `check_ctr_complexity` refuses anything above 1 **by name**
so the next lane deletes a guard rather than discovering an assumption. Every
CTR entry point takes an already-hashed `List[Int]` category code per row
rather than a column of raw categories, precisely so a combination -- which is
a hash of several columns -- is the same input shape as a single column. What
is **not** there: the binarized-feature half of `calc_hashes` (a projection may
include *float* splits, `TProjection::BinFeatures`, hashed as `0`/`1` against a
border), the candidate enumeration in `greedy_tensor_search.cpp` that decides
which combinations to try, and the `ctr_leaf_count_limit` top-K reindexing that
keeps a wide combination's bucket count bounded. That last one is not optional
for combinations the way it is for a single column.
