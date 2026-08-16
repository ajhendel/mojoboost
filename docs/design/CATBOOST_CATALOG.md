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
| A6 | `leaf_estimation_iterations` (`gradient_walker.h::FastGradientWalker`; defaults in `catboost_options.cpp::GetEstimationMethodDefaults`) -- **verified from source** | Re-estimate leaf values 1..k times, derivatives recomputed at the current leaf value each pass | Only when > 1 | Built on the CPU, opt-in, default 1. **DEVICE: not cheap, structural** (GPU orchestrator, 2026-08-16): each extra iteration needs per-leaf grad/hess sums after the previous raw-score update, so a second reduction per leaf per iteration inside the tree (more launches, against the oblivious command-buffer budget) or leaf estimation moved out of the device round (more host trips). **Correction:** an earlier version of this row said CatBoost defaults to 10 for logloss AND multiclass. Logloss is 10; **multiclass is 1**. The 10 in that block is the unreachable Gradient slot while the default method is Newton, and CatBoost's own docs agree. | CPU built; device design decision before any device lane | (3) when > 1; see A6 notes |
| A7 | Ordered boosting, `boosting_type=Ordered` (`fold.h::TFold::TBodyTail`, `fold.cpp::BuildDynamicFold`, `train.cpp::TrainOneIteration`) -- **verified from source**, see the A7 note | Derivatives for row i from a model that never saw i: a geometric ladder of prefixes of a permuted row order, each rung fitted on its own body and scored over its own tail | Yes | **BUILT**, `ordered_boosting.mojo` + the `_boost_rounds` round loop, opt-in `BoosterParams.ordered`, default off, single-output dense CPU trainer only. **Two corrections to what this row used to say.** (1) "Matters most on small data" was inferred from CatBoost's default and the default is not what it is usually said to be -- **on the CPU CatBoost's default is Plain at every row count**; the 50000 threshold is a GPU rule and it turns Ordered OFF. (2) The memory is much smaller than "K planes": the rungs are nested prefixes, so the planes total **under 3n** and are **2.64n at 1e6**, not 14n | CPU (`ordered_boosting.mojo`, `boosting.mojo`) | (2) trade behind a switch; deterministic under the seed, across `MOJOTREES_NUM_WORKERS` and across machines. **Carries no constant-hessian exclusion**, and that is a checked claim: see `ordered_varies_hessian` |
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
| A10 | `score_function` (`Cosine` is the CPU default for **every** grow policy, `L2`, and the GPU-only `SolarL2`/`NewtonL2`/`NewtonCosine`/`LOOL2`/`SatL2`) -- **verified from source**, see the A10 note | Split scoring alternatives. Cosine is a RATIO of two cross-leaf accumulators, not a sum | Only when set to Cosine | Built on the CPU, opt-in, default `SCORE_L2` (= today's LightGBM gain). **WIRED AND REACHABLE** (2026-08-16): `ExtraTreeParams.score_function` carries the choice, `tree._search` and `tree._grow_oblivious_levels` pass it in, and the parameter string, the Python estimator and the bindings all set it. **Headline finding: at `lambda_l2 = 0` Cosine is provably a no-op on the argmax *within one node*** -- it degenerates to `sqrt` of the L2 sum. That statement is narrower than it used to read here: section 5 of the A10 note already showed the leaf-wise queue compares gains from different parents, where `sqrt` does not preserve the ordering, so a lossguide tree can move at `lambda_l2 = 0` too | CPU (`split.mojo`) | (3) when set to Cosine; see the A10 note |
| A11 | MVS, `bootstrap_type=MVS` (**verified from source**, see the A11 note) | Minimal Variance Sampling: gradient-magnitude-weighted Poisson sampling with an inverse-probability weight, so large-gradient rows are kept certainly at weight 1 and small-gradient rows are kept with probability `g/mu` at weight `mu/g`. **This, not Bayesian, is what CatBoost actually runs on the CPU** for every objective we benchmark. Redrawn once per tree at the `PerTree` default. | Only when on | Yes. Extends A4's module, opt-in, our default does not move. **It is a row *dropper*, not only a reweighter**: a zero weight removes the row from CatBoost's score fold outright (`SetControlNoZeroWeighted`), so it touches fewer rows than a full pass. That is a speed side effect this lane does NOT claim and did NOT measure. **Carries A4's constant-hessian exclusion** for the same reason A4 does. | CPU (`sampling.mojo`, beside A4) | (3) when on |
| A12 | Automatic `learning_rate` (`catboost/libs/train_lib/options_helper.cpp`: `UpdateLearningRate`, `TAutoLRParamsGuesser`) -- **verified from source**, see the A12 note | The rate is fitted from the train row count and the iteration count off a 20-row coefficient table, not taken from the 0.03 constant, whenever the user set none of `learning_rate`, `leaf_estimation_method`, `leaf_estimation_iterations`, `l2_leaf_reg` | Yes when on | Yes, and it is bucket C as much as B: our harness pins the rate on the CatBoost arm *because* of this, so "defaults vs defaults" is not currently a real comparison | CPU (`auto_learning_rate.mojo`, new file) | (3) when on; **BUILT, off by default**, no existing default changed |
| A13 | Stochastic Gradient Langevin Boosting: `langevin`, `diffusion_temperature` (**verified from source**, see the A13/A14 note) | Seeded Gaussian noise added to every row's derivative once per bootstrap, at scale `sqrt(2 / (learning_rate * diffusion_temperature))`; a second, differently scaled draw is added to each leaf's derivative sum at leaf estimation | Only when on | Yes, `langevin.mojo`, off by default. Per-row half built and wired by glue; the leaf-sum half is built and **not** wired (it needs a leaf-sum call site in `tree.mojo`) | CPU (`langevin.mojo`) | (3) when on; deterministic under the seed, across `MOJOTREES_NUM_WORKERS`, and across machines. **Carries no constant-hessian exclusion**, and that is a checked claim, not an omission: see `langevin_varies_hessian` |
| A14 | Model shrinkage: `model_shrink_rate`, `model_shrink_mode` (**verified from source**, see the A13/A14 note) | At the top of every iteration after the first, every accumulated raw score is multiplied by `1 - rate * learning_rate` (Constant) or `1 - rate / iteration` (Decreasing); the products are folded back into the leaf values of the already-grown trees at the end of the fit | Only when on | Yes, `langevin.mojo`, off by default. Built as a deferred fold (strictly less work than CatBoost's per-round rescale of the model, and exact) | CPU (`langevin.mojo`) | (1) as built, off; (3) when on. Refused beside continued training and beside `init_score`, both of which CatBoost also refuses |
| A15 | `feature_border_type` / `border_count` (`library/cpp/grid_creator/binarization.cpp`, `catboost/libs/data/quantization.cpp`) | CatBoost's float quantization: seven border-selection algorithms, `GreedyLogSum` by default, bounded by `border_count` thresholds | New mode; bit-moving on the arm that selects it, exact on the arm that does not | Yes, opt-in `border_type`, default stays the LightGBM/mojotrees quantile fit | CPU (`binning.mojo`) | A15 note. **BUILT**, five of seven types, `binning.fit_bins(border_type=...)` |
| A16 | `one_hot_max_size` (`catboost/private/libs/options/cat_feature_options.cpp`, `catboost/private/libs/algo/greedy_tensor_search.cpp`) | CatBoost's categorical one-hot threshold, default 2, `<=` on the count of real categories seen on learn | New mode; bit-moving only when set | Yes, opt-in `CategoricalParams.one_hot_max_size`, default off | CPU (`categorical.mojo`) | A16 note. **BUILT**. It also settles the owed `max_cat_to_onehot` off-by-one, see A12 |
| A17 | `text_processing`: the tokenizer and the dictionaries (`library/cpp/text_processing/tokenizer/tokenizer.cpp` + `options.h`, `library/cpp/text_processing/dictionary/dictionary_builder.cpp` + `options.h`, `catboost/private/libs/options/text_processing_options.cpp`/`.h`, `catboost/libs/data/quantization.cpp::CreateDictionaries`, `catboost/private/libs/text_processing/dictionary.h::CreateDictionary`, `catboost/private/libs/data_types/text.h`) -- **verified from source**, see the A17 note | Turns a raw text column into a bag of dictionary token ids with counts. Default tokenizer splits on the literal string `" "` and does nothing else; default dictionaries are a unigram and a bigram over words, `occurrence_lower_bound=3`, `max_dictionary_size=50000` | No: pure host-side feature GENERATION feeding the existing binning path. Nothing in the trainer changes | Yes. It is the only way a text column becomes comparable at all; without it there is a whole class of dataset on which no CatBoost comparison exists | CPU (`text_processing.mojo`, NEW, imports nothing from the package) | (1) strictly-less-work does not apply -- it is new work that did not exist. Default OFF: no existing entry point calls it. A17 note |
| A18 | `text_features`: the `BoW` / `NaiveBayes` / `BM25` estimators (`catboost/private/libs/text_features/bow.cpp`, `naive_bayesian.cpp`, `bm25.cpp`, `feature_calcer.h`, `helpers.h`; `catboost/private/libs/feature_estimator/base_text_feature_estimator.h`, `text_feature_estimators.cpp`; `catboost/private/libs/algo/estimated_features.cpp`, `fold.cpp`, `data.cpp`, `full_model_saver.cpp`) -- **verified from source**, see the A18 note | Three numeric features off a token bag. `BoW` is target-free. **`NaiveBayes` and `BM25` are target-aware and ARE computed on the ordered permutation prefix, the same machinery the CTRs use** | Feature values only; they are handed to `binning.fit_bins` like any other numeric column | Yes, opt-in. `BoW` and the calcers are ours today; the ordered pass CONSUMES `lane/ordered-ts-2`'s permutation and must never build its own | CPU (`text_features.mojo`, NEW, imports only `text_processing`) | (3) bit-moving when on, inert when off. A18 note, and read the leakage section of it before writing any caller |
| A19 | Ordered target statistics, the simple (single-cat-feature) projection (`catboost/private/libs/algo/online_ctr.{h,cpp}`, `.../ctr_helper.{h,cpp}`, `.../fold.cpp`, `.../split.cpp`, `catboost/libs/model/online_ctr.h`) -- **verified from source**, see the A19 note | The mechanism A5 named. A categorical value becomes a *numeric* feature whose value at row `i` is a target statistic over the rows that precede `i` in a random permutation, quantized to 16 buckets. Four CTR types, three priors, an unshuffled fold 0, and a separate non-online table for inference | Yes -- it adds features that did not exist | Yes. This is the one mechanism that makes a categorical CatBoost comparison possible at all, and `bench/real_data` runs no such scenario today because we had no counterpart | CPU (`ctr.mojo`, new file; `categorical.mojo` untouched) | A19 note. **BUILT, off by default, unreached.** (3) when on. Deterministic under the seed, across `MOJOTREES_NUM_WORKERS`, and across machines, by a keyed sort rather than a shuffle |
| A20 | `embedding_features`: the `LDA` and `KNN` embedding estimators (`catboost/private/libs/embedding_features/lda.{h,cpp}`, `knn.{h,cpp}`, `catboost/private/libs/feature_estimator/{base_,}embedding_feature_estimators.{h,cpp}`, `catboost/private/libs/options/embedding_processing_options.{h,cpp}`, `catboost/private/libs/algo/{fold,estimated_features}.cpp`) -- **verified from source**, see the A20 note | Host-side generation of numeric columns from a raw embedding column, which are then quantized by the ordinary float binning path. LDA projects the embedding onto the leading generalized eigenvectors of (between-class scatter, within-class scatter); KNN emits per-class neighbour counts (classification) or the neighbour target mean (regression). **Both read the target**, and both are computed **online against the fold's permutation** -- the same `TFold::GetLearnPermutationArray()` the ordered CTRs use | Yes: new columns, so every downstream bit moves | Yes, and it is bucket C before it is bucket B: CatBoost takes a raw embedding column and mojotrees cannot, so an embedding benchmark today is not a comparison at all | CPU (`embedding.mojo`, new file). **Consumes** `ordered-ts-2`'s permutation; does not own one | A20 note. **BUILT, off by default**, no existing default changed, nothing in the package imports it |
| A21 | The learn-permutation layer: `permutation_count`, `fold_permutation_block`, `has_time` (`learn_context.cpp::IsPermutationNeeded` / `CountLearningFolds` / `TFoldsCreationParams`, `objects_grouping.cpp::NCB::Shuffle`, `defaults_helper.h::DefaultFoldPermutationBlockSize`) -- **verified from source**, see the A17 note | The permutations themselves, separately from what consumes them. CatBoost builds `max(1, permutation_count - 1)` **learning** permutations plus one **averaging** fold, each a permutation of *blocks* of `min(256, n/1000 + 1)` consecutive rows. **Both A5 (CTRs) and A7 (ordered boosting) read this same layer**, and `IsPermutationNeeded` is what decides whether either gets a shuffle at all | Yes when either consumer is on | Built as part of A7 (`ordered_boosting.ordered_permutation`, `default_permutation_block_size`, `permutation_choice`). **A5's lane should consume it rather than draw its own**: two independent permutation layers keyed by two seeds is two answers to one question | CPU (`ordered_boosting.mojo`); A5's lane is the second consumer | A17 note. Deterministic under the seed and worker-independent, which CatBoost's own `CreateShuffledIndices` is not |
| A22 | `QueryRMSE` (`catboost/private/libs/algo_helpers/error_functions.h::TQueryRmseError::CalcDersForQueries` and its private `CalcQueryAvrg`) -- **verified from source**, see the A22/A23/A24 note | Squared error with the query's weighted mean residual removed, so only *within-query* level is learned. `Der1 = w_i (t_i - a_i - avg)`, `Der2 = -w_i`, `avg = sum_j w_j (t_j - a_j) / sum_j w_j`. `IsStoreExpApprox` false | Yes -- it is a new objective | Yes. It is the cheapest of the three CatBoost ranking losses and the only one whose hessian stays constant under unit weights | CPU (`catboost_ranking.mojo`, new file) | A22/A23/A24 note. **BUILT, off by default, unreached.** (3) when on. Deterministic: no RNG at all |
| A23 | `PairLogit` + its automatic pair generation (`error_functions.h::TPairLogitError::CalcDersForQueries`; `catboost/private/libs/pairs/util.cpp::GeneratePairLogitPairs` / `GenerateBruteForce`; `catboost/private/libs/target/data_providers.cpp::GeneratePairs`) -- **verified from source** | Pairwise logistic loss over (winner, loser) pairs inside a group. `p = e^{a_lose} / (e^{a_lose} + e^{a_win})`, `Der1[win] += c p`, `Der1[lose] -= c p`, `Der2[both] += c p (p-1)`. With `max_pairs` unset, pairs are *every* label-distinct pair in the group, weight = group weight, no RNG | Yes -- a new objective, and it is the first mojotrees objective whose hessian is pairwise by construction | Yes. It is also the gradient kernel `YetiRank` reuses | CPU (`catboost_ranking.mojo`) | A22/A23/A24 note. **BUILT, off by default, unreached.** (3) when on. **Carries a constant-hessian exclusion.** Deterministic: default pair generation draws nothing |
| A24 | `YetiRank` (`catboost/private/libs/algo/yetirank_helpers.cpp`: `UpdatePairsForYetiRank`, `GenerateYetiRankPairsForQuery`, `TYetiRankPairWeightsCalcer::{AddNoise,CalcWeightsClassic}`; defaults in `catboost/private/libs/options/loss_description.cpp::GetYetiRankPermutations` / `GetYetiRankDecay`) -- **verified from source** | A *sampled* pairwise scheme, not a closed form: each round, per group, draw `permutations`=10 noisy rankings of the current scores, and in each ranking charge each adjacent pair `0.15 * decay^k * |rel_i - rel_j|`. Average the pair weights over the 10 draws and feed them to the `PairLogit` gradient | Yes, and it is the only objective in the catalog whose gradients depend on an RNG stream | Yes. It is CatBoost's flagship ranking loss and the one a CatBoost ranking comparison will be run under | CPU (`catboost_ranking.mojo`) | A22/A23/A24 note. **BUILT, off by default, unreached.** (3) when on. **Carries a constant-hessian exclusion.** Deterministic under `(seed, iteration, query, permutation, doc)` keying, across `MOJOTREES_NUM_WORKERS` and machines. CatBoost's own stream is NOT nameable |
| A25 | `group_id` and the two ranking eval metrics `NDCG`, `PFound` (`catboost/libs/data/objects.cpp::CheckGroupIds`; `catboost/libs/data/target.cpp::CheckGroupWeights`; `catboost/libs/metrics/dcg.cpp` + `catboost/libs/metrics/metric.cpp::TDcgMetric`; `catboost/libs/metrics/pfound.h::TPFoundCalcer` + `metric.cpp::TPFoundMetric`; tie rule in `catboost/libs/metrics/doc_comparator.h`) -- **verified from source**, see the A25 note | The grouping contract (contiguous, *not* sorted; loud failure when violated; weights constant inside a group) and the two metrics a CatBoost ranking run is actually judged by. CatBoost's default `NDCG` is **not** LightGBM's: `type=Base` (raw relevance, not `2^l - 1`), no truncation, and an adversarial tie rule | No (metrics move no bits) | Yes. Without them we cannot score a CatBoost ranking run at all, which is bucket C | CPU (`catboost_ranking.mojo`); `ranking.RankGroups` reused unchanged | A25 note. **BUILT, off by default, unreached.** (3) as a metric selection. Deterministic: no RNG |
| A26 | `Cox` (`catboost/private/libs/algo_helpers/error_functions.h` `TCoxError`, `.cpp:110-207` `ArgSort`/`CalcCoxApproxSum`/`CalcDersRange`; metric `catboost/libs/metrics/metric.cpp:880-947`) -- **verified from source**, see the A26 note | Cox proportional-hazards partial likelihood. One signed label column: `abs(y)` is the time, `y > 0` is an event and `y <= 0` is right-censored. The risk set is a **suffix of a stable sort by `abs(y)`**, so the derivative of every row depends on every other row and there is **no tie correction at all** | New objective | Yes, and it is the only one of the three whose whole shape our machinery already carries: approx dimension 1, one label column, one grad/hess plane | CPU (`survival.mojo`, new file) | (3) when selected, and it is a **coupled** objective: the hessian is per-row and non-constant, and must be declared. Cox is `IsPlainOnlyModeLoss` in CatBoost, which is CatBoost saying the same thing about ordered boosting |
| A27 | `SurvivalAft` (`error_functions.h` `TSurvivalAftError`, `.cpp:360-450`; `catboost/private/libs/algo_helpers/survival_aft_utils.{h,cpp}`; `catboost/libs/helpers/distribution_helpers.{h,cpp}`; construction in `BuildError`; metric `metric.cpp:408-444`) -- **verified from source**, see the A27 note | Accelerated failure time. **TWO label columns per row**, a lower and an upper bound, with `-1` the unbounded sentinel; `lower == upper` is an exact event, `upper == -1` right-censored, `lower == -1` left-censored, otherwise interval-censored. `dist` in {`Normal` (default), `Logistic`, `Extreme`}, `scale` default 1 and required positive. Approx dimension is **1** (`approx_dimension.cpp`) | New objective | Yes for the derivatives and the metric; **the input contract is the blocker, not the gradient** | CPU (`survival.mojo`), input path unowned | (3) when selected. Per-row non-constant hessian (clipped into `[1e-16, 15]`), declared. **Needs a two-column target that `List[Float64]` cannot spell** |
| A28 | `MultiRMSE` (`error_functions.h:191-218` `TMultiRMSEError`; `approx_dimension.cpp`; `online_predictor.cpp` `CalcDeltaNewtonMulti`; `hessian.cpp` `TDiagonalHessian::SolveNewtonEquation`; multi-dim score accumulation `catboost/private/libs/algo/scoring.cpp:741-767`; metric `metric.cpp:474-515`) -- **verified from source**, see the A28 note | Multi-target regression. **T label columns per row**, approx dimension T, diagonal hessian, `der[i] = w*(y[i] - p[i])`, `der2[i] = -w`. **One tree per iteration with a vector leaf value** (`model.h:118`, `LeafValues[leafId * ApproxDimension + dim]`), and the split score is the **sum over dimensions into one accumulator** | New objective **and a new tree shape** | The derivatives and the metric yes. The tree shape is `multiclass_tree_count` again, and this time it is not a cosmetic difference: see the A28 note | CPU (`multi_target.mojo`), tree shape unowned (`tree.mojo`, `histogram.mojo`) | (3) when selected. **The shared tree structure is the entire modeling content of MultiRMSE**; without it the objective degenerates into T independent RMSE fits |
| A29 | Model save/load and interchange export (`catboost/libs/model/model_export/`: `cbm`, `json`, `onnx`, `coreml`, `python`, `cpp`) | CatBoost saves every fitted artifact the model needs to score, CTR tables and text dictionaries included, and can additionally emit an ONNX `ai.onnx.ml` tree ensemble | No. Nothing here changes a trained model; the whole rule is that a reloaded model scores **bit-identically** | Yes, and it is bucket C: an arm we cannot save is an arm nobody can ship, and an export that silently drops fitted state is a wrong comparison that looks fine | `serialize.mojo` (mine), `onnx_export.mojo` (new, mine) | A29 note. **Enumeration is the deliverable.** Format: NO version bump needed for A8/A11-A16; ONNX exporter BUILT for the numerical/expressible subset and REFUSES the rest by name |
| A30 | CTR feature combinations: `max_ctr_complexity`, `ctr_leaf_count_limit`, `TreeCtrs` (`greedy_tensor_search.cpp::AddTreeCtrs`, `index_hash_calcer.cpp::CalcHashes` and `ComputeReindexHash`, `ctr_helper.h::GetCtrInfo`) -- **verified from source**, see the A30 note | A projection of several categorical columns plus float and one-hot splits, hashed to one bucket per row and fed to A19's ordered machinery. Candidates are **grown from the splits already in the tree**, one feature at a time -- NOT exhaustive to depth 4 -- and `max_ctr_complexity` silently resolves to **1**, not 4, whenever the iteration count is under 200 | Yes when on | Built, off by default, reached by nothing. **CatBoost's default of 4 is not affordable at 1M rows** (about 1.6e9 element ops and 1.1 GB of CTR columns per tree, against 3e8 for a 50-feature histogram build), so record the 4 and wire 1 | CPU (`ctr_combinations.mojo`, beside `ctr.mojo`) | (3) when on |

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

## A9, MultiRMSE (multi-target regression), device derivative plane

Status, 2026-08-16 (`lane/multitarget-device`): the **device derivative plane
and its per-output fixed-point scales are built and tested**; nothing above
them exists on either backend, and the path is **unreachable from any
user-facing API by construction**. Read the reachability paragraph at the
bottom of this section before quoting the first sentence.

**Tree shape, and why it is not CatBoost's.** CatBoost's MultiRMSE grows one
tree whose leaves hold a vector of `ApproxDimension` values and whose split is
scored against a gain summed over dimensions. mojotrees grows **K trees per
round, one scalar-leaf tree per output**, and this lane implements that shape.
It is the only shape the rest of the codebase can express: `tree.Tree.value`
is a `List[Float64]` indexed by node id and `Tree.predict_row` returns one
`Float64`, so a leaf cannot hold a vector without a new field on `Tree` and a
second dimension through every serializer, dumper and predictor that reads it.
It is also the shape mojotrees already uses for its one existing multi-output
trainer: `objective_is_multi_output` *defines* multi-output as "whether one
boosting iteration grows more than one tree", and
`boosting._boost_rounds_multiclass` grows `trees[round * n_classes + k]`. A
device path emitting vector-leaf trees would have no host model to put them
in. **This is a divergence from CatBoost and is recorded as one**, not a
parity claim.

**What the other shape would cost, since it was not built.** Two things, in
two files this lane does not own. (1) A gain summed over outputs in
`gpu_split_search.mojo`: a candidate would score `sum_j gain(G_j, H_j)` across
`n_outputs` histogram planes at one split point instead of one plane, which
changes the reduction the split kernel performs and the argmax it feeds. The
histogram *accumulation* needs nothing -- it already builds one plane per
slot. (2) A vector leaf value in `tree.mojo` and `model.mojo`. Neither is
refused as a runtime error, because neither is reachable: no type here can
hold a vector leaf, so no caller can ask for one.

**Derivatives.** Uncoupled, and that is the whole reason this was cheap.
Softmax couples its outputs -- class `k`'s gradient reads a denominator summed
over every class, so a round needs a probability pass before any class's
gradient exists. MultiRMSE does not: output `j`'s gradient is
`w * (raw[r, j] - y[r, j])` and reads nothing of output `j'`. So `prob_dev`,
`refresh_softmax` and the probability pass all disappear, and
`_multi_grad_hess_kernel` is a pure per-(row, output) map that writes the same
class-major plane `grad[slot * n_rows + r]` the batched softmax kernel writes.
Everything downstream is the multiclass machinery **unchanged**:
`GpuClassBatch.magnitude_sums`, `set_scales`, `scatter_slot` and every batched
histogram kernel take a multi-target round without one line changed, and
`update_raw(k=j)` advances output `j`'s slot exactly as it advances class
`k`'s. The one new method in `gpu_multiclass_batch.mojo` is
`fill_multi_output_gradients`, which is the launch and nothing else.

**One scale per output, never one shared.** The fixed-point scale is per round
and derived from gradient magnitudes, and with K outputs there are K magnitude
profiles. `GpuClassBatch.set_scales` already answers this per slot and the
multi-target path inherits the answer unchanged. This hazard is *sharper* here
than under softmax: softmax classes share a probability simplex and so cannot
differ by orders of magnitude, while a vector target routinely puts one output
in units of 1 beside another in units of 1000. A shared scale would size the
small output's lattice by the large output's magnitudes and quantize its
gradients toward zero, silently -- the fit would converge on the large output
and barely move on the small one.
`tests/test_gpu_multitarget.mojo` fixes three outputs three orders of
magnitude apart for exactly this, and asserts each slot's scale **equals**
what the scalar device path derives for that output alone. Equal-scale outputs
would pass a shared-scale implementation and would also hide an output-index
error entirely, since swapping two identically-scaled planes changes nothing
observable.

**Weights.** Per-row, not per (row, output): a sample weight weights the
observation and every output of it, which is what lets one weight plane serve
every output slot of a launch. A weighted multi-target round therefore stages
both derivative planes per output rather than the gradient alone, 9 bytes per
(row, feature) visit at the default feature group where an unweighted one is
on 7. That is the A4 arithmetic applied per output, by construction, and not a
regression.

**Bits do not move for a single-output fit.** The `multi_output` constructor
parameter defaults False and every scalar statement computes what it computed
before it existed. `git diff -U0` on `gpu_objectives_native.mojo` deletes
exactly twelve lines: one import, four docstring lines, one error *message*,
and six constructor statements, each of the six replaced by an expression
equal to it at `multi_output == False` --
`_check_weight_vector(..., n_rows)` for `(..., len(target))`,
`self.n_rows = n_rows` for `= len(target)`, `target_dev` sized `len(target)`
for `self.n_rows`, the upload loop over `range(len(target))` for
`range(self.n_rows)`, and `and not multi_output` appended to the two
`n_classes > 1` tests. **No kernel body is touched at all** -- not
`_grad_hess_kernel`, not `_abs_sum_kernel`, not the softmax or raw-update
kernels. Empirically, `tests/test_gpu_objectives_native.mojo` passes 16/16
unchanged.

**REACHABILITY: none, and this is the entry that says so.** Walked end to end
rather than checked at its entry point. `python/mojotrees/sklearn.py` has no
multi-target objective name and no estimator that accepts a 2-D `y` -- not an
allow-list gap, a missing estimator class. `params.mojo` has no `num_targets`
key and `_validate` refuses `num_class` for every objective but `multiclass`,
so a multi-target fit cannot be spelled in a parameter string.
`trainset.Dataset` holds one `List[Float64]` label column, so a target matrix
cannot be expressed at the dataset level. No multi-target objective code
exists in `objective_registry`. Nothing calls `fill_multi_grad_hess` but the
test. Making it reachable is four further pieces of work, none of them in
these files: a label matrix on `Dataset`, a `num_targets` parameter, a
`train_multi_target` round loop shaped like `_boost_rounds_multiclass`, and a
multi-output estimator on the Python side. Until then a user asking for
MultiRMSE gets a name error from sklearn.py, which is a refusal; the failure
mode to avoid is the opposite one, an accepted parameter with no reader.

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

### A20 note: the `LDA` and `KNN` embedding estimators, verified from source

Status: **verified from CatBoost source**, `master`, read 2026-08-16 by
`lane/embedding-features` before any code was written. Every claim below cites
the file and the function it came from. Paragraphs headed **OURS** are our
decisions and cite nothing, because there is nothing to cite. Nothing in this
section is measured; this lane took no timings and ran no benchmark.

Numbering: this lane deliberately skipped A17 and A18 and left them for
`ordered-ts-2` and `text-features`, the two live lanes working the same
target-aware problem shape, rather than taking the next free number. Three
lanes collided on numbering last round. Checked after the fact:
`lane/ordered-ts-2` took A17 and `lane/text-features` took A17 **and** A18, so
those two collide with each other and neither collides with this row. This
diff adds A20 and touches no other row of the table.

Files read:

- `catboost/private/libs/embedding_features/lda.h`, `lda.cpp` --
  `TLinearDACalcer`, `IncrementalCloud::AddVector`/`Update`,
  `TLinearDACalcer::TotalScatterCalculation`, `CalculateProjection`,
  `CalculateGaussianLikehood`, `InverseMatrix`,
  `TLinearDACalcerVisitor::Update`/`Flush`.
- `catboost/private/libs/embedding_features/knn.h`, `knn.cpp` --
  `TKNNCalcer`, `TKNNCalcer::Compute`, `TKNNCalcerVisitor::Update`,
  `TKNNUpdatableCloud`, `TKNNCloud`, `TL2Distance`.
- `catboost/private/libs/embedding_features/embedding_feature_calcer.h` --
  `TEmbeddingFeatureCalcer`, `IEmbeddingCalcerVisitor`.
- `catboost/private/libs/feature_estimator/base_embedding_feature_estimator.h`
  -- `TEmbeddingBaseEstimator::ComputeOnlineFeatures`, `::ComputeFeatures`,
  `::EstimateFeatureCalcer`, `::MakeFinalFeatureCalcer`, `::Calc`.
- `catboost/private/libs/feature_estimator/embedding_feature_estimators.cpp` --
  `TLDAEstimator`, `TKNNEstimator`, `CreateEmbeddingEstimators`. **This is
  where the reachable defaults live**, not in the calcer constructors.
- `catboost/private/libs/options/embedding_processing_options.h`, `.cpp` --
  `TEmbeddingProcessingOptions`, `DefaultEmbeddingCalcers`,
  `ParseEmbeddingProcessingOptionsFromPlainJson`.
- `catboost/private/libs/algo/estimated_features.cpp` --
  `CreateEstimatedFeaturesData`, the `isOnline` branch at 448-464.
- `catboost/private/libs/algo/fold.cpp` -- `TFold::InitOnlineEstimatedFeatures`
  and its two call sites, one in each fold builder.
- `catboost/private/libs/algo/data.cpp` -- the offline call site (537-546),
  which passes `learnPermutation = Nothing()` and therefore does **not** run
  these two.

#### 1. The leakage answer, and the train-versus-predict asymmetry

This is the centre of the lane and it comes out cleanly.

**Both estimators are target-aware and both are `IOnlineFeatureEstimator`s.**
`TLDAEstimator` and `TKNNEstimator` derive from
`TEmbeddingBaseEstimator<...>`, which derives from `IOnlineFeatureEstimator`.
`CreateEstimatedFeaturesData` splits on `isOnline = learnPermutation.Defined()`
and calls `GetOnlineFeatureEstimators()` on that branch
(`estimated_features.cpp:369-375`), so these two never reach the offline path
that `data.cpp:537` drives with `Nothing()`.

**They reuse the ordered-permutation machinery, and they reuse it per fold.**
`TFold::InitOnlineEstimatedFeatures` (`fold.cpp:377-394`) calls
`CreateEstimatedFeaturesData` with `GetLearnPermutationArray()` -- the fold's
own permutation, the same object the ordered CTRs and ordered boosting are
built on. It is called from **both** fold builders (`fold.cpp:200` and `:298`),
so the online treatment is not conditional on ordered boosting: a Plain fold
gets it too.

**The train-time rule, exactly.** `ComputeOnlineFeatures`
(`base_embedding_feature_estimator.h:44-82`) is:

```
TFeatureCalcer featureCalcer = CreateFeatureCalcer();   // empty
for (ui64 line : learnPermutation) {
    Compute(featureCalcer, learnDataset.GetVector(line), line, ...);
    calcerVisitor.Update(target[line], vector, &featureCalcer);
}
```

**Compute strictly precedes Update, for every row.** So the calcer state that
produces row `i`'s feature has seen exactly the rows that precede `i` in the
permutation, and has never seen `i`'s own target or `i`'s own embedding. That
is the ordered target statistic discipline, expressed as a loop instead of a
prefix sum, and it is the whole leakage answer for the training side.

Two consequences that fall straight out and that an implementation gets wrong
if it does not read this loop:

- **The first row of the permutation gets an empty calcer.** For KNN the HNSW
  index is empty and `GetNearestNeighbors` returns nothing, so
  `TKNNCalcer::Compute` leaves `result` at its `TVector<float> result(n, 0)`
  initialization and the row's features are all zero -- and the regression
  branch is explicitly guarded, `if (neighbors.size())`, so the mean is 0 and
  not a division by zero. For LDA the projection matrix is still zero, so the
  projection is zero.
- **The KNN query point is excluded from its own neighbourhood by
  construction, not by a filter.** There is no "skip self" test anywhere in
  `knn.cpp`. The point is simply not in the cloud yet: `TKNNCalcerVisitor::
  Update` calls `cloudPtr->AddItem(embed.data())` *after* `Compute` has already
  queried. A design that fits the neighbour structure over the whole training
  set first and then queries it needs an explicit self-exclusion and does not
  get one from CatBoost; CatBoost never needed one.

**The predict-time rule, exactly, and it is a different object.** Two things
change at once:

1. **The calcer is fitted on the whole learn set, unpermuted.** Test features
   go through `Calc` (`base_embedding_feature_estimator.h:107-132`) against
   the calcer returned by `EstimateFeatureCalcer` (`:137-151`), which loops
   `for line = 0 .. samplesCount` in **dataset order** and calls only
   `Update`, never `Compute`. `MakeFinalFeatureCalcer` (`:94-104`) -- the
   calcer that is serialized into the model and used at inference -- is that
   same full-data calcer with `TrimFeatures` applied. So the mapping the model
   applies at predict time was never applied to a single training row.
2. **There is no target for the query row and none is needed.** Both `Compute`
   methods take only the embedding. LDA's is a matrix-vector product against a
   fitted projection; KNN's is a neighbour lookup whose *neighbours'* targets
   supply the value. The target-awareness is entirely inside the fitted state,
   which is why the asymmetry is a state asymmetry and not a formula
   asymmetry.

**The asymmetry, stated as the thing to get right:** at train time the feature
is a function of a *strict prefix* of a permutation; at predict time it is a
function of the *entire* learn set. Row `i`'s training feature and the feature
the deployed model would compute for that same row are different numbers, and
they are meant to be. The train-time value is deliberately noisier and
deliberately less informative, and that is what stops the tree from fitting a
leaked target. An implementation that "fixes" the mismatch by using the
full-data calcer at train time has reintroduced exactly the leak the loop
exists to prevent; an implementation that uses a prefix at predict time has
thrown away half its data for nothing.

**One CatBoost-specific quirk of the predict-time state, which is a real
finding and not a transcription.** `IEmbeddingCalcerVisitor`
(`embedding_feature_calcer.h:87-90`) declares exactly one virtual method,
`Update`. `TLinearDACalcerVisitor::Flush` is **not** virtual and is not on the
interface, so the generic `EstimateFeatureCalcer` has no way to call it and
does not. The only caller of `Flush` is `TLinearDACalcerVisitor::Update`
itself, under `if (2 * LastFlush <= lda->Size)` -- a doubling schedule that
fires at `Size` = 1, 2, 4, 8, ... So **the final LDA calcer's projection
matrix is the one fitted at the largest power of two `<= n`**, and the rows
after it contributed to the class clouds but never to a re-solve. On a
1,000-row learn set the deployed projection is the one computed from the first
512 rows. We do not reproduce this by default; see section 6.

#### 2. What the LDA estimator computes

**The two matrices, and the names are inverted in the source.** Read the
identifiers as what they hold, not as what they are called:

- `TLinearDACalcer::BetweenMatrix` holds `sum_c (n_c / N) * Cov_c`, the
  **within-class** scatter `S_W`. `TLinearDACalcerVisitor::Flush` builds it
  from each class cloud's `ScatterMatrix`, weighted by that class's share of
  the rows (`lda.cpp:181-190`). In regression mode (`IsClassification` false)
  it is the single cloud's covariance (`:191-193`).
- The local `totalScatter`, passed to `CalculateProjection` as `scatterTotal`,
  holds `sum_c (n_c / N) mu_c mu_c^T - mu mu^T`, the **between-class**
  scatter `S_B` (`TotalScatterCalculation`, `lda.cpp:140-161`).

So the problem solved is the classical Fisher one, `S_B v = lambda S_W v`, and
the misnaming is cosmetic. Each cloud's `ScatterMatrix` is a running
*covariance*, not a raw second moment: `IncrementalCloud::Update`
(`lda.cpp:82-107`) rescales by `BaseSize/TotalSize`, folds in
`Buffer^T Buffer / TotalSize` for the newly buffered rows, and then subtracts
the outer product of the mean shift.

**The regularization.** `Flush` adds `RegParam` to the diagonal of the
within-class matrix and to nothing else (`lda.cpp:194-196`, striding by
`dim + 1`). The reachable default is **`5e-5`**, set by `TLDAEstimator` when
the user gives no `reg` key (`embedding_feature_estimators.cpp:28-32`). The
`0.01` in `TLinearDACalcer`'s constructor signature is dead: the estimator
always passes a value. `CB_ENSURE(RegParam >= 0)` -- zero is legal, and at
zero a rank-deficient `S_W` has nothing holding it up.

**How many components.** `ProjectionDimension`, from the `components` key.
The reachable default is
`min(num_classes - 1, embedding_dim - 1)` for classification and `1` for
regression (`embedding_feature_estimators.cpp:20-27`), with two guards:
`> 0`, and **strictly** `< embedding_dim`. So a 2-class problem defaults to
one component and a 10-class problem to nine. `num_classes - 1` is the right
cap and CatBoost picks it: `S_B` is a sum of `C` rank-one terms constrained by
one linear relation, so `rank(S_B) <= C - 1` and any further eigenvector has
eigenvalue zero. **But the cap is only the default.** An explicit
`components=` larger than `C - 1` is accepted, and the extra components are
then eigenvectors of a zero eigenvalue -- arbitrary directions in a null
space, whose only content is whatever the solver's arithmetic left there.

**More than two classes** needs no special case: `ClassesDist` is
`numClasses` clouds, the target is used as an integer class index
(`lda.cpp:167`, `(size_t)target`), and the weighted sums run over all of them.

**Regression LDA is degenerate, provably.** With one cloud,
`TotalScatterCalculation` computes `1.0 * mu mu^T - mu mu^T`, which is the
zero matrix identically -- so `S_B = 0`, every eigenvalue is zero, and the
projection is an arbitrary direction. The feature is not merely weak, it
carries no target information at all. CatBoost does not refuse it. We do; see
section 6.

**The optional likelihood features.** With `likelihood=true` (default false,
`embedding_feature_estimators.cpp:33-37`) LDA emits `num_classes` further
features: `Flush` inverts the regularized within-class matrix in place
(`InverseMatrix`, `lda.cpp:202-204`), and `Compute` evaluates
`exp(-0.5 * (mu_c - x)^T S_W^-1 (mu_c - x))` per class and normalizes across
classes, falling back to a uniform `1/C` when the total is below `1e-6`
(`lda.cpp:119-131`). Note what is *not* there: no class prior, no
`det(S_W)` term, no `(2 pi)^{-d/2}`. The shared covariance makes the missing
determinant a constant that cancels in the normalization; the missing prior
does not cancel, so these are not posterior probabilities, they are
normalized Mahalanobis kernels.

**The two LAPACK findings, which are the reason we do not copy
`CalculateProjection` line for line.** `CalculateProjection`
(`lda.cpp:10-37`) does exactly three things: `ssygst_(itype=1, uplo='L')`,
`ssyev_(jobz='V', uplo='L')`, and a `std::copy` of the tail of the reduced
matrix into the projection. Two steps of the standard recipe are absent:

1. **`spotrf_` is never called.** LAPACK's `ssygst` documents that its `B`
   argument "must have been returned by `SPOTRF`" -- it expects the Cholesky
   factor, and it forms `inv(L) A inv(L^T)`. CatBoost hands it the raw
   regularized within-class matrix. The reduction is therefore by the
   *matrix* treated as if it were its own Cholesky factor, which is not the
   generalized problem and is not any problem with a name.
2. **The back-transform is never applied.** After `ssyev` the eigenvectors
   `y` belong to the reduced matrix; the generalized eigenvectors are
   `x = inv(L^T) y`, which LAPACK expects you to obtain with `strsm`. There
   is no `strsm` and no second `sgemm`; the `std::copy` at `lda.cpp:36` takes
   `y` straight into `ProjectionMatrix`.

The copy itself is right, and is the one subtle correct step: `ssyev` returns
eigenvalues **ascending**, and `jobz='V'` overwrites `A` with eigenvectors as
**columns** in **column-major** order, so the tail of the buffer is exactly
the top-`k` eigenvectors laid out contiguously; reading that tail as
row-major `k x d` recovers them as rows, which is what `Compute`'s
`cblas_sgemv(RowMajor, NoTrans, M=k, N=d, lda=d)` then wants.

**We do not reproduce either omission.** Both are bugs by any reading, and
the first is not even reproducible: what `ssygst` does with a non-factor `B`
is whatever the arithmetic does, and that differs between LAPACK builds. A
parity claim against it would be a parity claim against one machine's
reference BLAS. See section 6 for what we solve instead.

#### 3. What the KNN estimator computes

- **`k` is `CloseNum`, default 5**, from the `k` key
  (`embedding_feature_estimators.cpp:96-100`). The `5` in `TKNNCalcer`'s
  signature agrees with it, so unlike LDA's `reg` there is no dead default
  here.
- **The metric is squared L2.** `TOnlineHnswCloud` is parameterized on
  `NHnsw::TL2SqrDistance<float>` (`knn.h:17`) and the static-cloud path wraps
  the same distance in `TL2Distance` (`knn.h:48-61`). No square root anywhere,
  which changes no ordering, and no normalization, which does: cosine
  similarity is not what this computes, and an unnormalized embedding column
  gets a magnitude-sensitive neighbourhood.
- **The structure is approximate.** `TKNNUpdatableCloud` wraps
  `NOnlineHnsw::TOnlineHnswDenseVectorIndex`, an online HNSW graph built with
  `TOnlineHnswBuildOptions({CloseNum, 300})` (`knn.h:109-110`) -- max
  neighbours per node equal to `k`, search neighbourhood 300 -- and the static
  cloud searches with the same `300` (`knn.cpp:23`). **CatBoost's KNN feature
  is not the exact k nearest neighbours** and does not claim to be. It is
  whatever that graph returns, and what that graph returns depends on the
  insertion order, which is the fold permutation.
- **The output, classification:** `++result[TargetClasses.at(neighbor)]` --
  raw **counts** per class, `num_classes` features, not normalized, not
  distance-weighted (`knn.cpp:34-37`). With `k=5` these are integers in
  `[0, 5]` carried in a float column, so the binning path downstream sees a
  low-cardinality numeric feature. That matters: our binning already has a
  "one bin per distinct value under budget" rule (memory: d7da434), so these
  columns quantize exactly and cheaply.
- **The output, regression:** the plain unweighted mean of the neighbours'
  targets, one feature, guarded to 0 on an empty neighbourhood
  (`knn.cpp:38-45`).
- `FeatureCount` is `num_classes` for classification and 1 for regression
  (`embedding_feature_estimators.cpp:91-95`).

**Cost, as a derived bound and not a measurement.** An *exact* KNN over a
growing prefix is quadratic: the online pass over `n` rows performs
`sum_{i<n} i = n(n-1)/2` distance evaluations of `d` multiply-adds each, i.e.
`Theta(n^2 d / 2)`. At `n = 10^6` and `d = 64` that is `3.2 x 10^13`
multiply-adds for one column of one fold, which is not a number that appears
in a benchmark that finishes. **Exact KNN cannot appear in a 1M-row
comparison, and this lane does not pretend otherwise.** CatBoost's cost is not
quadratic because HNSW is not exact: an online HNSW insert-and-query is
`O(ef * M * log n)` per row under the usual assumptions, so `O(n log n)`
overall with a large constant -- an approximation bound, not a guarantee.
Closing that gap means implementing an approximate index and inheriting its
nondeterminism, and this lane did not do it. What it did instead is refuse
loudly above a row bound; see section 6.

#### 4. How it is reached, and the default that is not off

`ParseEmbeddingProcessingOptionsFromPlainJson`
(`embedding_processing_options.cpp:104-133`) accepts either
`embedding_processing` or `embedding_calcers`, and refuses both together.
The part worth flagging for bucket C:

```
static TVector<TFeatureCalcerDescription> DefaultEmbeddingCalcers() {
    return {{ TFeatureCalcerDescription{EFeatureCalcerType::LDA},
              TFeatureCalcerDescription{EFeatureCalcerType::KNN} }};
}
```

(`embedding_processing_options.h:37-42`, applied by
`TEmbeddingProcessingOptions`'s constructor and by
`SetNotSpecifiedOptionsToDefaults`.) **Both estimators are on by default for
every embedding column.** A user who declares an embedding feature and sets
nothing else gets LDA *and* KNN, per column, per fold. So "CatBoost at its
defaults on a dataset with an embedding column" means CatBoost with two
target-aware online estimators running, and any mojotrees arm that flattens
the embedding into `d` raw float columns is not running a comparable model.
That is the bucket C statement of this row.

The resulting columns are ordinary floats from there on: `estimated_features.
cpp` hands them to `CreateSingleFeatureWriter` with the run's
`TBinarizationOptions` and the same border-selection machinery A15 describes.
Nothing about them is special downstream, which is the design we copy.

#### 5. What mojotrees built

`src/mojotrees/embedding.mojo`, a **new module that nothing in the package
imports**, so no existing default moves and no import edge is created in
either direction. (The `efb -> binning -> tree_parameters_extra` cycle this
repo has paid for is why that was checked rather than assumed: this module
imports `parallel` and `std` only, and `parallel` imports
`apple_cpu_policy`, so the one edge added points away from the data layer and
nothing points back.)

The shape is deliberately the shape CatBoost has, because it is the shape that
composes with what we already own: **generate float columns, hand them to the
existing binning path.** There is no new trainer, no new split rule, and no
new histogram.

Entry points, and they are two on purpose because the asymmetry in section 1
is two:

- `compute_online_features(embeddings, targets, permutation, params)` --
  the **train** side. Walks `permutation`, computes row `i` from the strict
  prefix before it, then folds row `i` in. Returns column-major
  `List[List[Float64]]` ready for `binning.fit_bins`.
- `fit_lda(...)` / `fit_knn(...)` then `apply_lda(...)` / `apply_knn(...)` --
  the **predict** side. Fitted once on the whole learn set, applied to rows
  that are not in it.

`EmbeddingEstimatorParams` carries `enabled = False`. Every function refuses
to run against a disabled parameter block by name rather than returning empty
columns, which is this repo's refuse-rather-than-ignore rule.

**The permutation is an argument, not something this module generates.** This
lane owns no RNG, no shuffle and no fold. `identity_permutation(n)` exists for
tests and says in its docstring that it is not a training permutation.

#### 6. Divergences from CatBoost, each deliberate and each recorded

1. **We solve the generalized eigenproblem properly.** Cholesky of the
   regularized `S_W`, reduce, symmetric eigensolve, back-substitute. That is
   the two missing LAPACK calls of section 2 put back. Consequence: our
   projection is *not* CatBoost's projection, and a numeric parity test
   against CatBoost's LDA column would fail by design. Reproducing an
   unreproducible bug is not parity.
2. **Eigensolver is cyclic Jacobi, not LAPACK.** Reason is determinism and
   dependency: a fixed sweep order over `(p, q)` pairs with a fixed
   convergence threshold gives the same rotations on every machine, where a
   blocked LAPACK driver may not. Cost is `O(d^3)` per sweep with a small
   number of sweeps -- worse than a tuned `ssyev` by a constant, on a `d x d`
   problem where `d` is an embedding width, run `O(log n)` times per fold.
   The eigensolve is not where the time is; the scatter accumulation
   (`O(n d^2)`) and the KNN scan are.
3. **LDA is refused for regression by name.** Section 2 shows `S_B` is
   identically zero there, so the feature is an arbitrary direction. CatBoost
   emits it. We raise.
4. **Exact KNN, with a row bound and a refusal.** No HNSW. `KnnParams.
   max_rows` defaults to 50,000 and `compute_online_features` raises above it
   rather than silently entering an `n^2` loop. This is the honest position:
   exact and deterministic up to a size we state, and absent above it, rather
   than approximate and irreproducible everywhere. **It also means A20 has
   nothing to say about a 1M-row benchmark**, and the catalog should not be
   read as claiming otherwise.
5. **Scatter accumulation is by raw second moments, not CatBoost's shifted
   incremental update.** We keep per class `n_c`, `sum_c`, and
   `M_c = sum x x^T`, and form `Cov_c = M_c/n_c - mu_c mu_c^T` at flush.
   Simpler, one pass, and exact in exact arithmetic; in float it is the less
   stable of the two formulations for data far from the origin, which is the
   trade and it is stated rather than hidden. Embeddings are conventionally
   near-centered, which is why the trade is acceptable and not why it is
   invisible.
6. **The final-calcer flush quirk is a switch, defaulting to correct.**
   `LdaParams.catboost_final_flush_only` defaults `False`, which re-solves on
   all `n` rows. Setting it `True` reproduces CatBoost's
   largest-power-of-two-prefix behavior for anyone who wants that parity.
   The *online* doubling schedule is reproduced unconditionally, because
   there it is not a quirk: it is what makes the online pass
   `O(log n)` eigensolves instead of `n`, and skipping it would be both
   slower and a different feature.
7. **Float64 throughout.** CatBoost's embeddings, calcers and estimated
   features are `float`. Ours are `Float64`, like the rest of the mojotrees
   data path. Another reason a numeric parity test is not the right test for
   this row.
8. **`likelihood` is not implemented.** It is off by CatBoost's own default
   and it needs a matrix inverse we would otherwise not need.

#### 7. Determinism, which for this row is two specific hazards

Bit-identity against CatBoost is not required and is not claimed.
Determinism across `MOJOTREES_NUM_WORKERS` and across machines **is**
required, and two mechanisms here threaten it in ways the rest of the package
does not.

**Hazard one: an eigenvector's sign is arbitrary.** If `v` is an eigenvector
so is `-v`, with the same eigenvalue, and a solver is free to return either.
A projection feature whose sign flips run to run is not wrong and is not
reproducible either. **Pinned by a canonical sign rule, applied to every
eigenvector after the solve and before the back-transform:** find the
component of largest absolute value; ties in absolute value go to the
**lowest index**; if that component is negative, negate the whole vector. An
all-zero vector is left alone. This is a total rule with no free choice in it.

**Hazard two: eigenvalue ties leave the ordering free.** Sorting descending
by eigenvalue is ambiguous when two eigenvalues are equal -- and they will be
equal, because with `C` classes and `k > C - 1` requested components the null
space of `S_B` supplies a whole block of exact zeros. **Pinned by a total
order on eigenpairs:** descending eigenvalue first, and on an exact tie
ascending lexicographic order of the sign-fixed eigenvector, and on a full
tie of both the vectors are identical and the order does not matter. Sorting
is a deterministic insertion sort over `d` items, not a library sort whose
tie behavior we do not control.

**Hazard three, the one the brief names: KNN neighbour ties.** Two candidate
rows at the same distance must not be ordered by whichever thread saw them
first. **Pinned by a total order on candidates:** `(distance, row index)`
compared lexicographically ascending, with a non-finite distance mapped to
`+inf` so that NaN coordinates cannot poison a comparison into
non-transitivity. Row indices are unique, so the order is total and the
selected `k` is a *set-valued function of the candidate set alone*, entirely
independent of the order they were examined in.

**Where parallelism is and is not.** The KNN pass is parallel over **query
rows**: query `i` scans its own prefix and writes its own output slots, so
the queries are independent and each one internally is a serial ascending
scan. Worker count therefore cannot change any value. The LDA pass is
**serial by construction** -- each row's feature depends on the accumulated
state after the previous row -- and the class scatter sums are accumulated in
ascending permutation order, so the float addend order is fixed by the loop
and not by a scheduler. No reduction in this module is split across workers.

#### 8. Classification, in the campaign's three buckets

- **(3) bit-moving**, and unusually so: this row does not change a rule
  applied to existing features, it *adds columns*. Every histogram, every
  split and every leaf downstream is different the moment it is switched on.
  It is off by default and nothing in the package calls it.
- The proper eigensolve of divergence 1 is **not** a speed trade in either
  direction; it is a correctness repair.
- Divergence 4's row bound is the only place a switch changes reachability
  rather than behavior, and it defaults to refusing.

#### 9. What this row needs from the other two live lanes

**From `ordered-ts-2`:** the permutation, as data. **Already satisfied**, and
checked against their branch rather than assumed:
`ctr.ctr_permutation(n_rows, block_size, seed, permutation_index)` returns
`List[Int]` with `perm[position] = row`, which is exactly the convention
`compute_online_features` takes. The consumption is two lines and needs no
adapter:

```mojo
var perm = ctr_permutation(n_rows, block_size, seed, fold_index)
var cols = compute_online_features(embed, targets, n_classes, perm, params)
```

The reason to insist on *their* permutation rather than a second one is
CatBoost's own structure: `CreateEstimatedFeaturesData` and `InitOnlineCtrs`
are called from the same `TFold` a few lines apart, off the same array. Two
permutations where CatBoost has one would be both slower and a different
model. **This lane deliberately implemented no permutation, no RNG and no
fold.**

One wrinkle worth carrying: their permutation index 0 is the identity, which
is CatBoost's `shuffle = (foldIdx != 0)`. For an embedding estimator that
means one learning fold computes its features in dataset order, so a dataset
whose rows arrive sorted by target gets a maximally-correlated prefix on that
fold. That is CatBoost's behavior and not a bug, but it is the reason
`identity_permutation` here carries a docstring saying it is not a training
permutation.

**From `text-features`:** nothing structural, and that is the point --
`TEmbeddingBaseEstimator` and `TTextBaseEstimator` are siblings under
`IOnlineFeatureEstimator` with the same `Compute`-then-`Update` loop, so if
that loop is factored out it should be factored out once. If `text-features`
lands a shared "online estimator driver" abstraction, this module's
`compute_online_features` should be rewritten onto it rather than kept
beside it. Until then the loop is nine lines and duplicating nine lines is
cheaper than coordinating on an abstraction neither lane has seen.

**Owed to glue, not taken:** an `embedding_features=` parameter key, a
`RawData` embedding column, and a call site that appends the generated
columns before `fit_bins`. All three are outside this lane's territory
(`params.mojo`, `raw_data.mojo`, `binning.mojo`, `bindings/`) and are handed
over as a diff in the lane report rather than applied here.

### A7 note: ordered boosting, verified from source

Status: **verified from CatBoost source**, `master` (the checkout at
`58c7bb8b`), read 2026-08-16. Built on the CPU, opt-in, default off. Nothing
in this note is measured; there is no timing in this lane at all.

Files read, and what each settles:

- `catboost/private/libs/algo/fold.h:33-66` -- `TFold::TBodyTail`: the
  `BodyQueryFinish` / `TailQueryFinish` / `BodyFinish` / `TailFinish` /
  `BodySumWeight` quintuple, and the three per-rung planes `Approx`,
  `WeightedDerivatives`, `SampleWeightedDerivatives`, each `[dim][]`.
  `:215` -- `TVector<TBodyTail> BodyTailArr` is the ladder.
- `catboost/private/libs/algo/fold.cpp:35-41` -- `SelectMinBatchSize`
  (`n > 500 ? min(100, n/50) : 1`) and `SelectTailSize`
  (`ceil(oldSize * multiplier)`).
- `catboost/private/libs/algo/fold.cpp:150-198` -- `BuildDynamicFold`'s
  ladder loop, and the allocation of all three planes at `bt.TailFinish`.
- `catboost/private/libs/algo/fold.cpp:43-96` -- `InitPermutationData`.
- `catboost/private/libs/algo/learn_context.cpp:38-50, :53-103, :492-592` --
  `IsPermutationNeeded`, `CountLearningFolds`, `TFoldsCreationParams`, and the
  loop that builds `Folds` **plus a separate `AveragingFold`**.
- `catboost/private/libs/algo/train.cpp:206-243` -- `TrainOneIteration`: one
  fold taken at random per tree, one `CalcWeightedDerivatives` per rung.
- `catboost/private/libs/algo/train.cpp:302-382` -- `CalcLeafValues` for the
  model, then `UpdateLearningFold` for every fold.
- `catboost/private/libs/algo/tensor_search_helpers.cpp:568-660` --
  `CalcWeightedDerivatives`: derivatives over `[0, bt.TailFinish)` computed
  from `bt.Approx`, that rung's own scores.
- `catboost/private/libs/algo/approx_calcer.cpp:706-830` --
  `CalcApproxDeltaSimple`: `CalcLeafDersSimple(..., bt.BodyFinish, ...)` fits,
  `UpdateApproxDeltas(..., bt.TailFinish, ...)` applies.
- `catboost/private/libs/algo/approx_calcer.cpp:1101-1159` --
  `CalcApproxForLeafStruct`, one delta plane per rung.
- `catboost/private/libs/algo/approx_calcer.cpp` `CalcLeafValues` -- **the
  model's** leaf values, read off `ctx->LearnProgress->AveragingFold`.
- `catboost/private/libs/algo/scoring.cpp:735-770` -- the split score is
  SUMMED over every rung, each rung scaling `l2` by its own
  `BodySumWeight / BodyFinish`.
- `catboost/private/libs/algo/calc_score_cache.cpp:274-295` -- the scoring
  copy holds `bodyFinish` derivatives per rung and `tailFinish` sampled ones.
- `catboost/private/libs/options/boosting_options.cpp:9-27, :64-78` -- the
  option surface, its defaults, and `Validate`.
- `catboost/private/libs/options/catboost_options.cpp:778-816, :1040-1044` --
  where Ordered is installed as a default and where `has_time` forces
  `PermutationCount = 1`.
- `catboost/private/libs/options/defaults_helper.h:7-42` --
  `DefaultFoldPermutationBlockSize` and `UpdateBoostingTypeOption`.

#### 1. What a body is, what a tail is, and how the ladder grows

`BuildDynamicFold` grows a geometric ladder of prefixes of the permuted row
order. Writing `b_0 < b_1 < ... < b_K = n`:

    b_0     = SelectMinBatchSize(n) = (n > 500 ? min(100, n / 50) : 1)
    b_{f+1} = min(ceil(b_f * fold_len_multiplier), n)

Rung `f` has **body** `[0, b_f)` and **tail** `[0, b_{f+1})`. There are `K`
rungs and CatBoost's loop condition (`while (BodyTailArr.empty() ||
leftPartLen < learnSampleCount)`) guarantees at least one.

- The **body** is what rung `f`'s leaf values are fitted on
  (`CalcLeafDersSimple(..., bt.BodyFinish, ...)`).
- The **tail** is what rung `f`'s raw-score plane covers and what its delta is
  applied over (`UpdateApproxDeltas(..., bt.TailFinish, ...)`).
- Therefore a row at permuted position `q` in `[b_f, b_{f+1})` is inside rung
  `f`'s tail and outside rung `f`'s body: **rung `f`'s model has never been
  fitted on it**, and `CalcWeightedDerivatives` evaluating that row against
  `bt.Approx` is the derivative-from-a-model-that-has-not-seen-it that the
  mechanism is named for.
- The first `b_0` positions are a **leak that CatBoost accepts**: they are
  inside every rung's body. That is the price of keeping `K` models rather
  than `n`.
- `UpdateSize` additionally rounds a body up to a query boundary when the data
  has group info. mojotrees has no query grouping on this path and does not
  carry that clause.

#### 2. The permutation count rule, with the citations the device needs

    LearningFoldCount = isPermutationNeeded ? max(1, permutation_count - 1) : 1

`CountLearningFolds`, `learn_context.cpp:48-50`, called at `:81-83`.
`permutation_count` defaults to **4** (`boosting_options.cpp:14`), so **three
learning permutations at CatBoost's defaults**, plus one more permutation for
the `AveragingFold` built at `learn_context.cpp:575-589` -- four permuted
orders in total, three of which carry ladders.

`isPermutationNeeded` is `IsPermutationNeeded(hasTime, hasCtrs,
isOrderedBoosting, isAveragingFold=false)`, `learn_context.cpp:38-46`:

    if (hasTime)  return false;
    if (hasCtrs)  return true;
    return isOrderedBoosting && !isAveragingFold;

and `hasTime` is `HasTimeFlag || objects order == Ordered`
(`learn_context.cpp:68-69`). Separately, `has_time` also forces
`BoostingOptions->PermutationCount = 1` outright
(`catboost_options.cpp:1042-1044`). So under `has_time` there is exactly one
learning fold and its permutation is the identity
(`InitPermutationData`'s `else` branch fills `iota`).

`fold_permutation_block_size` does **not** change the count; it changes what a
permutation *is*. `NCB::Shuffle` (`objects_grouping.cpp`) permutes **blocks**
of `permuteBlockSize` consecutive rows and keeps rows inside a block in their
original relative order. The default is
`DefaultFoldPermutationBlockSize(n) = min(256, n / 1000 + 1)`
(`defaults_helper.h:9-11`), and `!isLearnFoldPermuted` sets it to
`learnSampleCount` -- one block, identity (`learn_context.cpp:93-95`).

**The fold (rung) count is not constant and depends on `n`:**

    K = the number of steps from b_0 to n at ratio m = fold_len_multiplier
      = ceil(log_m(n / b_0)) + 1,   b_0 = SelectMinBatchSize(n)

At `n = 1e6` with the defaults (`b_0 = 100`, `m = 2`) that is **14 rungs**.
It grows logarithmically, and it is computable from `n` alone before the first
tree, so a device can size buffers once at fit time.

#### 3. Memory: a derived bound, and why it is not `K * n`

The rungs are **nested prefixes**, not disjoint blocks. Plane `f` has length
`b_{f+1}`, so with `b_{K-1} < n`:

    sum_{f=0}^{K-1} b_{f+1}  =  n + sum_{f=1}^{K-1} b_f
                             <  n + n * m / (m - 1)
                             =  n * (2m - 1) / (m - 1)     = 3n at m = 2

A strict `3n`, independent of `K`. The exact count at `n = 1e6` with the
defaults is **2 638 200 entries, or 2.64n** -- not 14n.
`ordered_boosting.ordered_plane_entries` computes it. Derived, not measured.

In bytes, per permutation, at `n = 1e6`: 21.1 MB of `Float64` on the host;
10.6 MB on a device staging derivatives at Int16 and 21.1 MB at Int32. At
CatBoost's three learning permutations that is 31.7 MB / 63.3 MB.

CatBoost is worse than this by a factor of three, because `BuildDynamicFold`
allocates `Approx`, `WeightedDerivatives` **and** `SampleWeightedDerivatives`
at `TailFinish` per rung per dimension. Ours keeps one plane per rung (the raw
score) and recomputes derivatives into a shared scratch buffer, because a
derivative is cheaper to recompute than to store `2.64n` of.

#### 4. What we built, and the three places it diverges

`src/mojotrees/ordered_boosting.mojo` (new) and the round loop in
`src/mojotrees/boosting.mojo`. `BoosterParams.ordered`, default disabled,
appended last so every positional caller is unaffected. Honored by the dense
single-output CPU `train` only; `train_with_valid`, the multiclass trainers
and continued training refuse it by name rather than ignore it.

Per round: pick a permutation from `(seed, round)`; read each row's raw score
off the tightest rung that never saw it; fill gradients from those scores;
grow the tree; **refit the leaf values on the plain scores over every row**;
update `raw`; then advance every rung of every permutation.

Three divergences, all deliberate:

1. **One gradient per row, not a sum over rungs.** CatBoost sums the split
   score over every rung's body (`scoring.cpp:746`), so a row in an early
   position contributes to `K - f` of the `K` summands and an early row is
   weighted far more heavily than a late one. We give each row exactly one
   gradient, from the tightest model that has not seen it, which is the
   CatBoost paper's Algorithm 1 with the same ladder as its approximation.
   Strictly less work -- one histogram pass instead of `K` -- and it removes a
   position-dependent row weight that nothing in the paper asks for. It is a
   **(2) trade behind a switch**, and it is the largest divergence here.
2. **The permutation is counter-keyed, not Fisher-Yates.** See the A17 note.
3. **We keep one permutation by default, not three.** CatBoost's rule is
   `max(1, permutation_count - 1) = 3`; `permutation_count=3` reproduces it
   and costs exactly three times the planes. One permutation is the mechanism;
   three is variance reduction on top of it.

And one thing that is **not** a divergence and is worth stating because it is
the half of the mechanism most summaries drop: **ordered boosting decides the
tree's structure, not the values the model carries.**
`train.cpp::TrainOneIteration` searches the structure on a ladder fold and
then calls `CalcLeafValues`, which reads the `AveragingFold` -- a plain,
single-rung fold over every row. `UpdateLearningFold` advances the ladder
folds' own approxes and those never reach the model. We do the same.

#### 5. Hessian declaration

`ordered_varies_hessian` is **False**, always, and it is a checked claim.
Ordered boosting changes the *point* at which derivatives are evaluated; it
installs no per-row weight and no multiplier. For the objectives whose
unweighted hessian is the literal 1.0 the value written is still exactly 1.0
at any raw score, so `boosting.round_has_constant_hessian` stays correct
without knowing the bundle exists and the two-plane histogram path stays
admissible. This is the opposite answer from A4 and A11, and the distinction
is exactly theirs: a *weight* that multiplies the derivative is a hessian, a
*different evaluation point* is not.
`check_ordered_hessian_declaration` is installed anyway and is called by the
round loop, on the same reasoning as `check_langevin_hessian_declaration`.

#### 6. What CatBoost's auto rule actually is

**The widely repeated "Ordered below 50 000 rows on CPU, Plain above" is not
in the source.** `ordered_boosting.catboost_auto_is_ordered` is the corrected
rule and `tests/test_ordered_boosting.mojo` pins it:

- `boosting_options.cpp:16` constructs `BoostingType("boosting_type",
  EBoostingType::Plain)`. **On the CPU that is the value a fit keeps, at every
  row count.**
- `catboost_options.cpp:802-806` is the only place Ordered is installed as a
  default and its condition includes `TaskType == ETaskType::GPU`.
- The same block excludes the multiclass-only metrics and
  `RMSEWithUncertainty` / `MultiLogloss` / `MultiCrossEntropy`.
  `docs/en/references/training-parameters/common.md` agrees:
  "Any number of objects, MultiClass or MultiClassOneVsAll mode: Plain."
- `defaults_helper.h:33-42::UpdateBoostingTypeOption` then hard-sets Plain
  when `learnSampleCount >= 50000 || IterationCount < 500`. **The iteration
  clause is real and is usually omitted.** `TOption::SetDefault` does not
  raise `IsSetFlag` (`option.h:28-31, 80-85`), so this later test still sees
  `NotSet()` and wins over the GPU default above.

So 50000 is a real number, but it is a threshold for turning Ordered **off**
on the GPU, not for turning it on at all on the CPU. Any comparison that
claims "CatBoost defaults to ordered boosting on small data" on a CPU run is
comparing against a Plain fit.

### A17 note: the learn-permutation layer, verified from source

Status: **verified from CatBoost source**, `master`, read 2026-08-16. Built as
part of A7. Files: `learn_context.cpp:38-50, :53-103`,
`objects_grouping.cpp::NCB::Shuffle`, `fold.cpp::InitPermutationData`,
`defaults_helper.h:7-11`, `boosting_options.cpp:9-27`.

The permutations are a layer that two mechanisms consume, not a private detail
of either. A7 (ordered boosting) reads them for the fold ladder; **A5 (ordered
target statistics / CTRs) reads the same ones** -- `IsPermutationNeeded`
returns true on `hasCtrs` alone, with no reference to the boosting type, and
`TFold::InitOnlineCtrs` computes the CTRs *inside* the fold that owns the
permutation. A5's lane should consume `ordered_boosting.ordered_permutation`
rather than draw its own: two permutation layers keyed by two seeds is two
answers to one question, and the CTR values and the ordered gradients would
then disagree about what "earlier" means for the same row.

#### What a permutation is

`NCB::Shuffle`'s trivial-grouping arm, `permuteBlockSize != 1`:

    blocksCount = ceil(n / permuteBlockSize)
    blockedPermute = CreateShuffledIndices(blocksCount, rand)
    emit each block's rows in ORIGINAL order, blocks in shuffled order

So it is a permutation of **blocks**, and rows inside a block keep their
original relative order. The short final block is placed wherever its draw
sends it and `currentIdx` advances by its actual width, so permuted positions
are **not** aligned to the block size in general.

`DefaultFoldPermutationBlockSize(n) = min(256, n / 1000 + 1)`: block size 1
below 1000 rows (a plain row permutation), the 256 cap from 255 000 rows up.

#### Determinism, and where we deliberately diverge

CatBoost's `CreateShuffledIndices` is a sequential Fisher-Yates over one
running generator, so a block's destination depends on every draw taken before
it, and the permutation drawn at a given point in a run depends on how many
draws the run has already made.

Ours is counter-based, in the same family as `sampling._mvs_stream` and
`langevin._langevin_row_stream` but a different **shape**, and the difference
is worth being explicit about:

- MVS and Langevin give each row an *independent* draw, so "row `r` reads
  `stream + r`, nothing advances" is the whole scheme.
- A permutation cannot be built that way: the draws have to be turned into a
  global total order. So each **block** gets an independent 64-bit key
  (`block_key(stream, b) = splitmix64(stream ^ (b * GOLDEN))`, a pure function
  of `(seed, permutation index, block index)`) and the blocks are sorted by
  `(key, block index)` with a bottom-up merge sort. A stable merge over a
  total order has exactly one answer, at any worker count, on any machine, and
  no comparison's result depends on how the range was cut.
- Two domain constants, for the reason `langevin.mojo` gives for having two:
  the permutation stream and the per-round permutation-choice stream are both
  keyed by a small integer against the same seed, so without separation
  permutation 3's keys and round 3's choice would come off one counter run.
- Key collisions: 64-bit keys over at most a few tens of thousands of blocks,
  so under `m^2 / 2^65`, about `5e-11` at `m = 40000`. The tie-break makes the
  *result* exact regardless, at a negligible bias toward low block indices
  among colliding pairs.

One property this buys that CatBoost's does not have, and which
`test_permutation_does_not_depend_on_how_the_sort_is_cut` pins: a block's key
does not depend on how many blocks there are, so changing `n` by less than a
block leaves every full block's relative order unchanged.

The per-round *choice* of permutation is the same shape:
`permutation_choice(seed, round, count)` is a function of `(seed, round)`
alone, where CatBoost's is `Folds[Rand.GenRand() % foldCount]`
(`train.cpp:208`) off the sequential learn-progress generator. Ours is what
makes a continued run draw what an uninterrupted run would have drawn, the
same reason every other seeded decision in `_boost_rounds` reads the absolute
round index.

### A22/A23/A24 note: the three CatBoost ranking objectives, verified from source

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

#### A22. QueryRMSE

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

#### A23. PairLogit, and where its pairs come from

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

#### A24. YetiRank: what it samples, how many times, and what seeds it

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

### A25 note: `group_id` and the two eval metrics, verified from source

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

### A26/A27/A28 note: the input contract comes first

Status: **verified from CatBoost source**, `master`, read 2026-08-16 by the
`survival-multitarget` lane. Files read, all under `catboost/`:

- `private/libs/algo_helpers/error_functions.h` (`IDerCalcer`,
  `TMultiDerCalcer`, `TMultiRMSEError`, `TSurvivalAftError`, `TCoxError`)
- `private/libs/algo_helpers/error_functions.cpp` (`ArgSort`,
  `CalcCoxApproxSum`, `TCoxError::CalcDersRange`,
  `TCoxError::CalcFirstDerRange`, `TSurvivalAftError::CalcDers`)
- `private/libs/algo_helpers/survival_aft_utils.h` / `.cpp`
  (`TDerivativeConstants`, `ECensoredType`, `InverseMonotoneTransform`,
  `ClipDerivatives`, `GetDerivativeLimits`, `DispatchDerivativeLimits`)
- `libs/helpers/distribution_helpers.h` / `.cpp` (`TNormalDistribution`,
  `TLogisticDistribution`, `TExtremeDistribution`)
- `private/libs/algo/approx_dimension.cpp` (`GetApproxDimension`)
- `private/libs/algo/tensor_search_helpers.cpp` (`BuildError`: the `dist` and
  `scale` parsing and their defaults)
- `private/libs/algo/scoring.cpp` (`CalcScoresForSubCandidate`, the
  `for (int dim ...) UpdateScores(..., scoreCalcer)` loop)
- `private/libs/algo_helpers/online_predictor.cpp` (`CalcDeltaNewtonMulti`),
  `private/libs/algo_helpers/hessian.cpp`
  (`TDiagonalHessian::SolveNewtonEquation`)
- `private/libs/options/enum_helpers.cpp` (`MultiRegressionObjectives`,
  `SurvivalRegressionObjectives`, `MultiTargetObjectives`,
  `IsPlainOnlyModeLoss`), `private/libs/options/enums.h`
- `private/libs/target/data_providers.cpp` ("SurvivalAft is compatible only
  with a single-dimensional model")
- `libs/metrics/metric.cpp` (`TCoxMetric`, `TSurvivalAftMetric`,
  `TMultiRMSEMetric`)
- `libs/model/model.h:118`

**The headline is not a gradient.** `boosting.fill_grad_hess`,
`boosting.train`, `objective.GradHessFn`, `Booster`, and every Python and C
entry point in this repo take the label as `List[Float64] target`, one number
per row. That is a **one-column contract**, and two of these three objectives
cannot be spelled in it at all:

| objective | label columns | approx dimension | fits `List[Float64]`? |
|---|---|---|---|
| `Cox` | 1 (signed) | 1 | **yes** |
| `SurvivalAft` | **2** (lower, upper) | 1 | no |
| `MultiRMSE` | **T** | **T** | no |

So the deliverable that matters here is `target_matrix.TargetMatrix`, a flat
row-major `List[Float64]` of `n_rows * n_targets` with `n_targets` beside it,
and the honest statement that **nothing reaches it yet**: the Python layer,
the C API, `bindings/_mojotrees.mojo` and the sklearn wrapper all pass one
column and none of them is this lane's to widen. A two-bound or vector target
is a change to the *ingestion* path, and it is a bigger and more valuable
piece of work than any of the three derivative functions.

CatBoost's own layout is the same shape: `TConstArrayRef<TConstArrayRef<float>>
target` indexed `[dim][row]`, column-major where ours is row-major. Row-major
is the deliberate choice: the gradient loop for `MultiRMSE` touches
`target[r*T + t]` and `raw[r*T + t]` for all `t` at one `r`, which is a
contiguous run of `T` in both, and it is the same layout
`train_multiclass` already uses for its `raw[r * n_classes + k]` scores.

#### A26. Cox, and the two things everyone gets wrong about it

**The target encoding.** One column. `ArgSort` (error_functions.cpp:110-127)
sorts by `std::abs(targets[i])`, and the event test throughout is `y > 0`.
So `abs(y)` is the time and the *sign* is the event indicator. `y == 0` is
censored, which means **an event at time 0 cannot be expressed**; our
`cox_signed_target` refuses `time <= 0` for an event rather than silently
recoding it as censored.

**How the risk set is formed, and it is not Breslow.** `CalcDersRange` walks
the rows in sorted order carrying a one-position-lagged `accumulatedSum`, and
subtracts it from `expPSum` at each event:

```cpp
accumulatedSum += lastExpP;
if (y > 0) { expPSum -= accumulatedSum; accumulatedSum = 0;
             rk += 1.0 / expPSum; sk += 1.0 / (expPSum * expPSum); }
const double grad = static_cast<double>(y > 0) - expP * rk;
const double hess = expP * rk - expP * expP * sk;
ders[ind].Der1 = grad;  ders[ind].Der2 = -hess;
```

so at sorted position `k` the risk set is exactly `{k, k+1, ..., n-1}` --
**the suffix of the sort order**, not the set of rows with time `>= t_k`.
The two differ precisely on ties. Two events at the same time: the first sees
both, the second does **not** see the first. That is neither Breslow (which
gives every tied event the full risk set) nor Efron (which averages); it is
**no tie correction**, with ties broken by `StableSort`, therefore by original
row index. mojotrees reproduces exactly this, and reproduces it with its own
stable index merge sort keyed on `(abs(y), row index)` so that the tiebreak is
a stated rule rather than a property of whatever sort is linked in.

**Consequences that are not details.**

- The hessian `h_k = e^{p_k} r_k - e^{2 p_k} s_k` is **per-row, non-constant,
  and coupled**: it depends on the approxes of every row that shares a risk
  set with `k`. `cox_varies_hessian()` is unconditionally `True` and
  `check_cox_hessian_declaration` refuses a `CONSTANT_HESSIAN` declaration
  beside it, in the shape of `sampling.check_mvs_hessian_declaration`. It is
  non-negative -- `h_k = e^{p_k} sum_i (S_i - e^{p_k}) / S_i^2` and
  `e^{p_k} <= S_i` whenever `k` is in `R_i` -- so the Newton step is safe, but
  it reaches exactly 0 for the last row and that is a real leaf-denominator
  case, not a numerical accident.
- **CatBoost ignores `weights` for Cox.** Both `CalcDersRange` and
  `CalcFirstDerRange` name the parameter `const float* /*weights*/`. A
  weighted Cox partial likelihood needs the weights inside `S_i`, which
  CatBoost never does. mojotrees **refuses** a non-empty `sample_weight` for
  Cox rather than accepting and dropping it.
- **`labelOrder` is indexed out of its own bounds when `start != 0`.**
  `ArgSort` returns a vector of length `count` filled `iota` from `start`, and
  the caller reads `labelOrder[i]` for `i` in `[start, start + count)`.
  `CalcFirstDerRange` also reads `getApprox(0)` where its sibling reads
  `getApprox(start)`. Both are only harmless because Cox is always called with
  `start == 0` over the whole learn set. It is worth recording as the
  strongest available evidence that CatBoost itself treats Cox as a
  whole-dataset objective.
- Adding a constant to every raw score leaves the partial likelihood
  unchanged, so Cox has **no meaningful base score**; ours starts at 0, which
  is CatBoost's `INIT_ZERO` equivalent.
- The metric (`TCoxMetric::Eval`) is the partial log-likelihood itself,
  higher-is-better, and it is **not** additive over rows -- CatBoost derives it
  from `TNonAdditiveSingleTargetMetric` for exactly that reason. It also does
  **not** subtract the maximum approx before exponentiating where the
  derivative code does, so it overflows above `|approx| ~ 709`. Ours subtracts
  the maximum in both places. That is analytically the identical quantity and
  bit-moving only, class (3).

#### A27. SurvivalAft, and how an interval target is spelled

**The input.** `TSurvivalAftError::CalcDers` reads `target[0]` and
`target[1]`. The four cases, in CatBoost's own branch order:

| test | `ECensoredType` | meaning |
|---|---|---|
| `target[0] == target[1]` | `Uncensored` | exact event at that time |
| `target[1] == -1` | `RightCensored` | still alive after `target[0]` |
| `target[0] == -1` | `LeftCensored` | event before `target[1]` |
| otherwise | `IntervalCensored` | event inside `(lower, upper)` |

`-1` is the sentinel, in **both** columns, and the metric agrees
(`realTarget` maps `-1` to `+infinity`). Note that the uncensored test comes
first, so `(-1, -1)` is read as an exact event at time `-1` and
`log(-1)` follows; `TargetMatrix.check_survival_aft` refuses it.

**The transform.** `InverseMonotoneTransform(approx, target, scale)` is
`(log(target) - approx) / scale`, so the model predicts `log` of the survival
time and `exp(approx)` is a time. Every target must be strictly positive; we
check it, CatBoost does not.

**The derivatives.** With `z = (log t - a) / scale`, `f` the distribution's
pdf and `F` its cdf, the loss is `-log(f(z) / (scale t))` uncensored and
`-log(F(z_U) - F(z_L))` otherwise, and CatBoost's two numerator/denominator
pairs are exactly the first and second derivatives of those:

```
uncensored:  g = f'(z) / (scale * f(z))
             h = -(f(z) f''(z) - f'(z)^2) / (scale * f(z))^2
censored:    P = f(z_U) - f(z_L),  D = F(z_U) - F(z_L),  Q = f'(z_U) - f'(z_L)
             g = P / (scale * D)
             h = (P^2 - D Q) / (scale * D)^2
```

with `f = F = f' = 0` substituted at an unbounded lower end and
`f = f' = 0, F = 1` at an unbounded upper end. CatBoost stores `Der1 = -g` and
`Der2 = -h` throughout (its `RMSE` stores `target - approx` against a loss
whose gradient is `approx - target`), which is why the two minus signs appear
in `CalcDers` and why the diagonal Newton solve reads
`negativeDer[d] / (Data[d] - l2)`.

**Then it clips, and the clip is load-bearing.** `TDerivativeConstants`:
`g` into `[-15, 15]`, `h` into `[1e-16, 15]`. The `1e-16` floor is what makes
the hessian strictly positive for every row and every distribution, so a leaf
denominator can never be zero -- and it is also why the hessian is **non-
constant per row**, declared through `survival_aft_varies_hessian`. The
`DispatchDerivativeLimits` table is the *second* fallback, reached only when
the denominator is below `1e-12` **and** the quotient came out NaN or
infinite; it substitutes a per-(distribution, order, censoring, sign) limit.
That table is transcribed verbatim in `survival.aft_derivative_limit`,
including the two entries that look like typos and are reproduced as written:
`Normal`/`Second`/`RightCensored` returns `(1/scale^2, 1e-16)` with the larger
value in the `min` slot, and `Extreme`/`Second` returns `(15, 1e-16)` the same
way. They are used as `targetSign ? min : max`, so the order of the pair is
the whole meaning and guessing at it would change answers.

**`scale` and `dist`.** `BuildError` accepts exactly two loss params, `dist`
and `scale`; `dist` defaults to `Normal` and `scale` to `1`, and
`TSurvivalAftError`'s constructor enforces `scale > 0`. The three
distributions are transcribed from `distribution_helpers.cpp` -- Normal,
Logistic, and Extreme (Gumbel-minimum, `F(x) = 1 - exp(-e^x)`).

**Divergences, both deliberate.** CatBoost calls `fast_exp` and `FastLogf`;
we call `std.math.exp` and `std.math.log`. **Both sides are approximate** and
this lane makes no accuracy claim in either direction, because Mojo's
transcendentals are not libm: measured on this toolchain, `exp(log(5.0))` is
`4.999999998698298` (2.6e-10 relative) and the Normal cdf at 1 is
`0.841344750494095` against libm's `0.8413447460685429` (5.3e-9 relative).
Class (3), bit-moving, and it is why the AFT tolerances in
`tests/test_survival_multitarget.mojo` are 1e-7 rather than 1e-12. And `TSurvivalAftError::CalcDers` takes
`float /*weight*/` and **drops it**, exactly as Cox does; we multiply both
derivatives by the row weight, because a weight is a clean multiplier here
(unlike Cox, where it would have to enter the risk-set sums) and silently
ignoring an argument the caller passed is the worse failure.

**The eval metric is not the likelihood.** `TSurvivalAftMetric` is the mean
weighted **distance from the interval in time units**: it exponentiates the
approx, and adds `min(|e^a - lower|, |e^a - upper|)` whenever `e^a <= lower`
or `e^a >= upper`, with `-1` read as `+infinity` in both columns so that the
right- and left-censored cases fall out of the same expression. Lower is
better, best 0. Anyone reporting "SurvivalAft" as a number is reporting that,
not a log-likelihood.

#### A28. MultiRMSE, where the tree shape *is* the objective

**The derivatives are trivial and are not the point.** For each of `T`
dimensions, `der[i] = w * (y[i] - p[i])` and `der2[i] = -w`; the hessian type
is `Diagonal`, so there is no cross-target term anywhere in the derivative.
`TMultiRMSEErrorWithMissingValues` is the same function with `NaN` targets
zeroing both, which is a second registered loss
(`MultiRMSEWithMissingValues`) and is built here beside it because it costs
one branch.

**The tree shape is the point, and it is verified.**
`approx_dimension.cpp` returns `targetDimension` for a multi-target
objective; `model.h:118` says leaf values are laid out
`[treeIndex][leafId * ApproxDimension + dimension]`; and `scoring.cpp:751-766`
loops `for (int dim = 0; dim < approxDimension; ++dim) UpdateScores(stats +
dim*splitStatsCount, ..., scoreCalcer)` into **one** `scoreCalcer`, whose
`AddLeaf` does `Scores[splitIdx] += ...`. So:

> CatBoost builds **one tree per iteration** whose structure is chosen by the
> **sum over targets** of the per-target split score, and gives it a **vector
> leaf value** solved per target from the diagonal hessian:
> `value[d] = -sum(der[d]) / (sum(der2[d]) - l2_scaled)`.

This is the same distinction `bench/real_data/scenarios.py`
`CATBOOST_UNMATCHABLE["multiclass_tree_count"]` already records for
`MultiClass` -- but for `MultiRMSE` it is not a bookkeeping difference, it is
**the entire model**. Because the `MultiRMSE` derivative has no cross-target
coupling, the shared structure is the *only* thing that couples the targets:
grow one tree per target per round, as our multiclass machinery does, and the
result is **bit-identical to `T` independent `SQUARED_ERROR` boosters**.
`train_multi_rmse` in `multi_target.mojo` is that shape, is labelled that way
in its docstring, and is useful as a multi-output regression API and as the
consumer that gives `TargetMatrix` a reason to exist -- but it is **not
CatBoost's `MultiRMSE`** and no comparison should say it is.

Two things are needed from the tree lanes and they are stated as an owed
interface rather than built here:

1. `histogram.mojo`: `T` gradient planes and `T` hessian planes per node, or
   one plane set of stride `T`. This is the same request `multiclass-device`
   makes; the only new part is that the planes must be summable into one score.
2. `tree.mojo`: a split gain that is `sum over d of gain_d`, not the gain of
   the summed planes. **These are different numbers** and collapsing them is
   the easy wrong answer here. `multi_target.multi_target_split_gain` and
   `multi_target.multi_target_leaf_values` are written and tested against
   CatBoost's formulas so that whoever owns the grower calls a checked
   function instead of rederiving it.

**The metric.** `sqrt( sum over d of sum over i of w_i (p - y)^2 / sum over i
of w_i )`. The denominator is the row-weight sum and is **not** multiplied by
`T`, so `MultiRMSE` is the root-mean-square *Euclidean norm* of the residual
vector, not the mean of the per-target RMSEs. At `T = 1` it is exactly RMSE.

### A30 note: CTR feature combinations, verified from source

Numbering: A20 was the last entry on `perf-round-2` at `56b08c1`; `A21` was
already taken by the `ranking-objectives` lane when this was written, so this
lane takes **A30**. `ordered-boosting` and `survival-multitarget` had written
nothing at that moment and may land on A21/A30 as well; renumber on merge.

Status: **verified from CatBoost source**, `master`, read 2026-08-16 by the
`ctr-combinations` lane. This entry discharges the four items A19 named as
deliberately not built and left to this lane. A19 stands; A30 extends it from
the complexity-1 projection to the general one.

Files read, and every claim below cites one of them.

- `catboost/private/libs/algo/projection.h` -- `TProjection`, `TBinFeature`,
  `IsRedundant`, `IsSingleCatFeature`, `GetFullProjectionLength`, `operator<`.
- `catboost/private/libs/algo/index_hash_calcer.h` and `.cpp` -- `CalcHashes`,
  `ComputeReindexHash`, `UpdateReindexHash`.
- `catboost/private/libs/algo/greedy_tensor_search.cpp` -- `AddSimpleCtrs`,
  `AddTreeCtrs`, `AddCtrsToCandList`, `DropStatsForProjection`,
  `SelectDatasetFeaturesForScoring`, `SelectFeaturesForScoring`,
  `GreedyTensorSearchOblivious`, `GreedyTensorSearch`, `TrimOnlineCTRcache`,
  `SelectCtrsToDropAfterCalc`, `MAX_ONLINE_CTR_FEATURES`.
- `catboost/private/libs/algo/ctr_helper.h` and `.cpp` -- `TCtrHelper::GetCtrInfo`,
  `TCtrHelper::InitCtrHelper`, `MakeCtrInfo`.
- `catboost/private/libs/algo/online_ctr.cpp` -- `ComputeOnlineCTRs`'
  single-cat shortcut and its general branch, `approxBucketsCount`, the
  `topSize` resolution, `CalcFinalCtrs`.
- `catboost/private/libs/options/cat_feature_options.h` and `.cpp` --
  `SimpleCtrs`, `CombinationCtrs`, `MaxTensorComplexity`, `CtrLeafCountLimit`,
  `StoreAllSimpleCtrs`, `TCatFeatureParams::Validate`.
- `catboost/private/libs/options/catboost_options.cpp` -- `SetCtrDefaults`,
  `CreateDefaultCounter`, `SetDefaultBinarizationsIfNeeded`, and the
  `IsSmallIterationCount` override.
- `catboost/private/libs/options/catboost_options.h` -- `IsSmallIterationCount`.
- `catboost/private/libs/options/restrictions.h` -- `GetMaxTreeDepth`.
- `catboost/libs/model/split.h` -- `IsTrueHistogram`, `IsTrueOneHotFeature`.
- `catboost/libs/model/model_export/resources/ctr_calcer.py` -- `calc_hashes`.
- `library/cpp/grid_creator/binarization.cpp` -- `MakeBinarizer`, `BestSplit`
  (the exact DP), `TWeightedFeatureBin::UpdateBestSplitProperties` (the
  greedy), `Penalty<MinEntropy>`. This last file is for the appendix at the
  bottom, which is A15's territory and not this lane's.

#### 1. A projection is not a tuple of categorical columns

`TProjection` (`projection.h:61`) has **three** vectors, not one:

```cpp
struct TProjection {
    TVector<int> CatFeatures;
    TVector<TBinFeature> BinFeatures;      // {FloatFeature, SplitIdx}
    TVector<TOneHotSplit> OneHotFeatures;  // {CatFeatureIdx, Value}
};
```

and `CalcHashes` (`index_hash_calcer.cpp:70`) folds all three into one `ui64`
per row, in that order, with `CalcHash` -- the same `MAGIC_MULT` fold A19
already built as `ctr_combination_hash`. What is folded in differs per kind.

| kind | folded value | citation |
|---|---|---|
| categorical, **training** | `(ui64)quantizedBin + 1` | `index_hash_calcer.cpp:104` |
| categorical, **final CTR table** | `(int)originalHashedValue` | `index_hash_calcer.cpp:117` |
| float split | `IsTrueHistogram(bin, SplitIdx)` = `bin > SplitIdx` | `:138`, `model/split.h:12` |
| one-hot split | `IsTrueOneHotFeature(bin, Value)` = `bin == Value` | `:159`, `model/split.h:16` |

Three things fall out and each is a way to get this wrong.

**The float and one-hot members contribute a 0 or a 1, not a value.** A
projection that names `f7 > bin 12` folds in the bit, so a projection over one
categorical column and two float splits has a bucket space of `cardinality x 4`
and not `cardinality x bins x bins`. `ctr_calcer.py:13-18` is the readable
statement of the same loop and it merges `BinFeatures` and `OneHotFeatures`
into one `binarized_indexes` list distinguished by a `check_value_equal` flag,
which is the export format's spelling of the `>` versus `==` difference above.

**The categorical fold is `+ 1` in training and is not `+ 1` in the final
table.** Training folds the perfect-hash bin index plus one; `CalcFinalCtrs`
passes a non-null `perfectHashedToHashedCatValuesMap` and folds the *original*
hashed value instead. The two hash spaces are therefore different, and it does
not matter because `ComputeReindexHash` renames every hash to a dense bucket id
on each side separately and the model file carries the final side's map. It
does matter to anyone who tries to compare a training bucket id against a model
bucket id, which is a thing implementations do. Note also that
`(int)origValsView[...]` narrows a `ui32` hash to a signed `int` before the
implicit widening to `CalcHash`'s `ui64` parameter, so any original hash with
the top bit set is sign-extended to `0xFFFFFFFF........`. That is CatBoost's
arithmetic as written and it is stable, not a bug that changes answers, but it
is not what a reimplementer would write.

**The single-categorical projection does not go through `CalcHashes` at all.**
`ComputeOnlineCTRs` (`online_ctr.cpp:626`) branches on `proj.IsSingleCatFeature()`
and copies the quantized column straight into the hash array
(`CopyCatColumnToHash`), no fold. So A19's remark that "a one-element
projection is `calc_hash(0, v)` and NOT `v`" is right about the export path and
is bypassed on the training path. Since `ComputeReindexHash` renames afterwards
either way, the induced *partition* is identical and only the labels differ.
The reason the shortcut exists is that it also gets to size the rehash table
from the known cardinality (`:653`) instead of guessing it (`:680-687`).

#### 2. Candidate enumeration is grown from the tree, not exhaustive

This is the question that changes the cost by orders of magnitude, and the
answer is unambiguous: **grown from the splits already in the current tree.**
`AddTreeCtrs` (`greedy_tensor_search.cpp:491`) is the whole of it.

```cpp
TProjection binAndOneHotFeaturesTree;
binAndOneHotFeaturesTree.BinFeatures    = currentTree.GetBinFeatures();
binAndOneHotFeaturesTree.OneHotFeatures = currentTree.GetOneHotFeatures();
seenProj.insert(binAndOneHotFeaturesTree);
for (const auto& ctr : currentTree.GetUsedCtrs()) { seenProj.insert(ctr.Projection); }

for (const auto& baseProj : seenProj) {
    if (baseProj.IsEmpty()) continue;
    for each available categorical feature f {
        if (isOneHot(f) || rand() > Rsm) continue;
        TProjection proj = baseProj;  proj.AddCatFeature(f);
        if (proj.IsRedundant() || proj.GetFullProjectionLength() > MaxTensorComplexity) continue;
        if (addedProjHash.contains(proj)) continue;
        addedProjHash.insert(proj);
        AddCtrsToCandList(*fold, *ctx, proj, candList);
    }
}
```

Five facts, each load-bearing.

**The base set is `{ the tree's bin+one-hot splits as ONE projection } union
{ the projection of every CTR already used in the tree }`.** So the bases at
depth `d` number at most `d + 1`. There is no enumeration over subsets of the
tree's splits; the whole set of float and one-hot splits in the tree is a
single base.

**One categorical feature is added per step.** A projection therefore only
grows by walking down the tree, and a combination of four categorical columns
can only be reached if a combination of three was itself chosen as a split
earlier in the same tree. That is the greedy in "greedy tensor search". It is
NOT exhaustive to depth 4, and the folklore reading -- "CatBoost tries all
4-way combinations" -- is wrong.

**`GetFullProjectionLength` counts the bin/one-hot blob as ONE**
(`projection.h:138`): `CatFeatures.size() + (BinFeatures.size() + OneHotFeatures.size() > 0 ? 1 : 0)`.
So `max_ctr_complexity = 4` permits four categorical columns, or three
categorical columns plus arbitrarily many float and one-hot splits.

**`AddSimpleCtrs` and `AddTreeCtrs` run at EVERY depth**, not once per tree:
`GreedyTensorSearchOblivious` (`:1189`) calls `SelectFeaturesForScoring` inside
the depth loop, and `SelectDatasetFeaturesForScoring` (`:997-1008`) calls both.
`AddTreeCtrs` is skipped only at depth 0, where `currentSplitTree` is empty and
every base is empty.

**Each surviving projection expands into several split candidates.**
`AddCtrsToCandList` (`:400`) emits one candidate per
`(ctrIdx, targetBorder, prior)`, i.e.
`sum over descriptions of GetTargetBorderCount x priors.size()`. At the CPU
defaults with a binary target that is `Borders: 1 x 3` plus `Counter: 1 x 1`,
so **four candidates per projection**, exactly as A19's `ctr_feature_count`
already computes for the simple case.

Two smaller mechanisms are attached to the same loop and are worth recording
because they are the reason the above is survivable at all.
`TrimOnlineCTRcache` (`:65`) clears a fold's entire CTR cache once it holds
more than `MAX_ONLINE_CTR_FEATURES = 50` projections, and it runs at the top of
every `GreedyTensorSearch` (`:1907`). `SelectCtrsToDropAfterCalc` (`:580`)
reads `NMemInfo::GetMemInfo().RSS` against `cpu_used_ram_limit` and marks
candidate sublists to be dropped immediately after scoring. **Both make
CatBoost's memory behavior depend on the machine's live RSS**, which is a thing
mojotrees must not copy if determinism across machines is to mean anything.

There is also a default override nobody documents. `catboost_options.cpp:1046`:

```cpp
if (CatFeatureParams->MaxTensorComplexity.NotSet() && IsSmallIterationCount(BoostingOptions->IterationCount)) {
    CatFeatureParams->MaxTensorComplexity = 1;
}
```

and `IsSmallIterationCount(n)` is `n < 200` (`catboost_options.h:88`). So **a
default CatBoost fit with fewer than 200 iterations does not build combinations
at all.** Any comparison harness that fits 100 trees and believes it is
measuring `max_ctr_complexity = 4` is measuring 1.

#### 3. `ctr_leaf_count_limit` and the top-K reindex

`ComputeReindexHash(topSize, reindexHash, begin, end)`
(`index_hash_calcer.cpp:171`) has three branches.

1. `topSize > learnSize`: assign ids in first-seen scan order. This is the
   default path, because `ctr_leaf_count_limit` defaults to `Max<ui64>()`
   (`cat_feature_options.cpp:236`).
2. otherwise, count frequencies first; if the distinct count is `<= topSize`,
   assign ids by **iterating the hash table** (`for (auto& it : reindexHash) it.second = counter++`).
   Same partition as branch 1, different labels.
3. otherwise, `std::nth_element` the distinct hashes by descending frequency,
   keep the first `topSize`, number them `0 .. topSize-1`, and map every other
   hash to `reindexHash.Size() - 1`.

Branch 3 has two properties that matter more than the mechanism.

**The overflow bucket is the last KEPT bucket, not a fresh one.** The code
writes `*hash = reindexHash.Size() - 1` (`:222`) where `reindexHash.Size()` is
`topSize`. The header comment two files up says "map other hash values to value
`reindexHash.Size()`" (`index_hash_calcer.h:42`), which would be a fresh
`topSize + 1`-th bucket. **The comment and the code disagree**, the code wins,
and the consequence is that every rare combination is merged into the least
frequent of the *kept* buckets rather than into a bucket of its own. Read that
twice before reproducing it.

**`std::nth_element` is not stable and the comparator only looks at the
count.** Two hashes with equal frequency at the `topSize` boundary are kept or
dropped according to the standard library's partition, which differs between
libstdc++ and libc++ and between versions of each. That is a portability hazard
for us and not for CatBoost, whose contract does not include reproducing across
machines.

`topSize` is resolved at `online_ctr.cpp:689`:

```cpp
ui64 topSize = catFeatureParams.CtrLeafCountLimit;
if (proj.IsSingleCatFeature() && catFeatureParams.StoreAllSimpleCtrs) { topSize = Max<ui64>(); }
```

so `store_all_simple_ctr` (default false) can exempt a single column and can
**never** exempt a combination. This is the source-level confirmation of A19's
claim that the top-K reindex is optional for a single column and not optional
for a wide one.

The bucket-space estimate CatBoost itself computes is at `:680-687`:

```cpp
size_t approxBucketsCount = 1;
for (auto cf : proj.CatFeatures) {
    approxBucketsCount *= uniqueValuesCount(cf);
    if (approxBucketsCount > learnSampleCount) break;
}
rehashHashVal->MakeEmpty(Min(learnSampleCount, approxBucketsCount));
```

-- the product of the cardinalities, capped at the row count, and note that it
ignores the bin and one-hot members, which can each multiply it by 2.

#### 4. A combination is routed to `TreeCtrs` and inherits nothing

`TCtrHelper::GetCtrInfo` (`ctr_helper.h:54`) is five lines:

```cpp
if (projection.IsSingleCatFeature()) {
    const int featureId = projection.CatFeatures[0];
    if (PerFeatureCtrs.contains(featureId)) { return PerFeatureCtrs.at(featureId); }
    else { return SimpleCtrs; }
}
return TreeCtrs;
```

and `IsSingleCatFeature()` (`projection.h:102`) requires `BinFeatures.empty() &&
OneHotFeatures.empty() && CatFeatures.size() == 1`. So **one categorical column
plus one float split is already a `TreeCtrs` projection**, not a simple one.

`InitCtrHelper` (`ctr_helper.cpp:59-114`) fills `SimpleCtrs` from
`catFeatureParams.SimpleCtrs` and `TreeCtrs` from
`catFeatureParams.CombinationCtrs`, two separate option lists
(`cat_feature_options.h:85-86`), each carrying its own priors, its own
`ctr_binarization` and its own `target_binarization`, each passed through the
same `MakeCtrInfo` independently. CatBoost warns about exactly this, twice, in
`SetCtrDefaults` (`catboost_options.cpp:455-460`):

```
"Change of simpleCtr will not affect combinations ctrs."
"Change of combinations ctrs will not affect simple ctrs"
```

On the CPU the two default lists happen to be built with identical content --
`{Borders with priors {0,0.5,1}, Counter with prior {0}}` for both -- because
`CreateDefaultCounter` ignores its `EProjectionType` argument on the CPU
(`:394-395`). On the GPU they genuinely differ: `FeatureFreq` with `MinEntropy`
binarization for simple, `FeatureFreq` with `Median` binarization for
combinations (`:398-414`), and `SetDefaultBinarizationsIfNeeded` (`:418`)
re-applies that split to any user list. So the *routing* is real everywhere and
the *values* differ only off the CPU defaults. An implementation that shares one
description list is correct at the CPU defaults and silently wrong the moment a
user sets `simple_ctr`.

The complexity bound itself is validated at `cat_feature_options.cpp:266-275`:
`CB_ENSURE(MaxTensorComplexity.Get() < GetMaxTreeDepth())` with
`GetMaxTreeDepth() == 16` (`restrictions.h:14`), and
`CB_ENSURE(CtrLeafCountLimit.Get() > 0)`.

#### 5. Derived bounds, and the honest answer about 1M rows

Every number in this section is a **derived bound** from the loop structure
above. This lane has no clock and measured nothing.

Let `C` be the number of categorical columns wide enough to escape one-hot
(`uniqueValues > one_hot_max_size`, `greedy_tensor_search.cpp:469`), `D` the
tree depth, `n` the learn row count, and `P` the number of split candidates one
projection expands into (4 at the CPU defaults with a binary target).

**Projections considered per tree.** At depth `d` there are at most `d + 1`
non-empty bases, so `AddTreeCtrs` considers at most `(d + 1) * C` projections
and `AddSimpleCtrs` a further `C`. Summing over `d = 0 .. D-1`:

    projections per tree  <=  C * D * (D + 3) / 2

which at `D = 6` is `27C` and at `D = 8` is `44C`. Redundancy rejection and the
complexity cap only reduce it. Split candidates are `P` times that: `108C` at
`D = 6` and the CPU defaults.

**Work per projection per fold.** Hashing is `(|cat| + |bin| + |onehot|)`
multiply-add folds over `n` rows; `ComputeReindexHash` is `n` hash probes plus,
in the top-K branch, a sort of the distinct set; and each of the `P`
`(border, prior)` pairs is one `O(n)` online pass writing `n` bucket indices.
So a projection costs **at least `6n` element operations** at the defaults, and
produces `4n` bytes of CTR column.

**Per tree, at `n = 10^6`, `C = 10`, `D = 6`, complexity 4.**
`27 * 10 = 270` projections, `270 * 6 * 10^6 = 1.6 * 10^9` element operations,
and `270 * 4 * 10^6 = 1.1 GB` of CTR columns if none were dropped -- which is
precisely why `MAX_ONLINE_CTR_FEATURES = 50` exists and why
`SelectCtrsToDropAfterCalc` reads live RSS. For scale, the histogram build of a
50-feature 1M-row depth-6 tree in this repo is on the order of
`n * F * D = 3 * 10^8` element operations. **The combination CTRs are a derived
~5x the entire rest of the tree, from ten categorical columns.** Over a
1000-tree fit that is `~1.6 * 10^12` element operations spent on candidate
features that mostly lose.

**So, plainly: at 1M rows the default `max_ctr_complexity = 4` is not
affordable in this repo and should not be the default here.** The derived
bounds say so and no measurement is needed to see the order of magnitude.
Complexity 1 costs `C` projections per tree -- `6 * 10^7` element operations at
the numbers above, comfortably under the histogram budget -- and is the setting
CatBoost itself silently selects for any fit under 200 iterations. Complexity 2
costs `C * (D + 1)` per tree, roughly `4 * 10^8`, which is the same order as the
histogram build and is the first setting that has to be paid for rather than
absorbed.

**The bucket-space bound.** A projection of `k` categorical columns with
cardinalities `m_1 .. m_k` and `b` binarized members has a bucket space of
`2^b * prod(m_i)`. Four columns of 1000 levels is `10^12`. The *realized*
bucket count is bounded by `min(n, distinct observed)`, and that bound is the
problem rather than the reassurance: as the realized count approaches `n`,
every bucket's ordered prefix holds zero or one row, `CalcCTR` returns the pure
prior almost everywhere, and the feature is noise with a constant. `ctr_leaf_count_limit`
is the only mechanism that bounds it, and **it is off by default**
(`Max<ui64>()`). CatBoost's actual defences against a degenerate wide
projection are the complexity cap, the one-hot cutoff, and the fact that the
CTR value is quantized into 16 buckets regardless. This is a real hole in the
default configuration and it is worth saying out loud rather than porting
silently.

#### 6. What mojotrees built for A30

`src/mojotrees/ctr_combinations.mojo`, a new file, **off by default and reached
by nothing**, with `tests/test_ctr_combinations.mojo` beside it. It imports
`.ctr` and `std` and nothing else, and `.ctr` imports only `.rng`, so the module
stays outside the `efb -> binning -> tree_parameters_extra` cycle for the same
reason A19's does. No existing default moved. Classification: **(3) bit-moving
when on** -- it manufactures projections and features that did not exist -- and
a no-op when off.

Built, and matching the source above.

- `BinSplit` and `OneHotSplit`, with `is_true` reproducing `IsTrueHistogram`
  (`bin > split_idx`) and `IsTrueOneHotFeature` (`bin == value`).
- `Projection`, with `TProjection`'s sorted inserts, `is_redundant`,
  `is_empty`, `is_single_cat_feature`, `has_single_feature` and
  `full_projection_length` -- including the rule that the whole bin/one-hot set
  counts as one.
- `cat_value_train` (`bin + 1`) and `cat_value_final` (the original hash, with
  the `(int)` sign extension reproduced and documented), the two halves of the
  categorical fold.
- `projection_row_hash`, the whole of `CalcHashes` for one row, in CatBoost's
  order: categorical, then float splits as a bit, then one-hot splits as a bit.
- `projection_hashes`, the bulk column-driven form.
- `compute_reindex_hash` and `update_reindex_hash`, the top-K machinery,
  including the overflow-into-the-last-kept-bucket rule.
- `projection_bucket_space_bound`, `approxBucketsCount` including the `2^b`
  factor CatBoost's version drops.
- `simple_ctr_projections`, `tree_base_projections`, `grow_tree_ctr_projections`
  -- `AddSimpleCtrs` and `AddTreeCtrs` without the Rsm coin flip and without
  the RSS-reading drop policy.
- `ctr_candidates_per_projection` and `ctr_candidate_count_bound`, which
  compute the `P` and the `C * D * (D + 3) / 2` of section 5 rather than
  asserting them.
- `CtrRouting` and `ctr_info_for_projection`, `GetCtrInfo`'s five lines, with
  `CtrRouting.catboost_cpu_defaults()` building the two lists **independently**
  so that changing one cannot change the other, and `ctr_routing_warning`
  carrying CatBoost's two warning strings verbatim.
- `resolve_max_ctr_complexity`, the `iterations < 200` override, and
  `check_max_ctr_complexity`, the `1 <= v < 16` bound.
- `check_ctr_combination_trainer_support`, the same honest "unreached" refusal
  A19's `check_ctr_trainer_support` is.

`src/mojotrees/ctr.mojo` changed in exactly one place: `check_ctr_complexity`
no longer refuses above 1. It now enforces CatBoost's own `1 <= v < 16` and
points at this module. That was the guard A19 left for this lane to delete and
it is the only edit to that file's behavior.

**Divergences, deliberate and recorded.**

1. **Enumeration order is canonical, not hash-set order.** CatBoost iterates
   `seenProj`, a `THashSet<TProjection>`, so the order in which bases are
   expanded is the table's bucket order. Ours sorts bases and candidates by
   `TProjection::operator<`'s own key (cat features, then bin splits, then
   one-hot splits, lexicographically) and emits in that order. The *set* of
   candidates is identical; only the order is, and the order is what a
   tie-breaking `SelectBestCandidate` would see. Ours does not depend on a
   hash table's layout, so it is identical across `MOJOTREES_NUM_WORKERS` and
   across machines. This is `ctr_permutation`'s keyed-sort discipline applied
   to a different object.
2. **The top-K tie-break is a total order, not `nth_element`.** Distinct hashes
   are ordered by `(count descending, hash ascending)` with a stable merge
   sort. The hashes are distinct by construction, so the key is a strict total
   order and the kept set is the same on every machine and every standard
   library. CatBoost's `nth_element` is not, and this is the one place where
   matching CatBoost would cost us the determinism claim.
3. **Bucket ids in the under-limit branch are first-seen order.** CatBoost's
   branch 2 numbers them in hash-table iteration order. The partition is
   identical; only the labels differ, and first-seen order makes branch 1 and
   branch 2 agree with each other, which CatBoost's do not.
4. **The overflow bucket rule IS reproduced**, `top_size - 1`, the last kept
   bucket, because it changes which rows share a statistic and is therefore
   behavior rather than labelling. It is reproduced with the discrepancy
   against `index_hash_calcer.h:42` written into the docstring so nobody
   "fixes" it by accident.
5. **No Rsm coin flip and no RSS-driven dropping.** `AddSimpleCtrs` and
   `AddTreeCtrs` each consult `ctx->LearnProgress->Rand.GenRandReal1() > Rsm`
   per feature per level, off the run's single advancing RNG, so CatBoost's
   candidate set for a given tree depends on every draw any earlier tree made.
   Reproducing that would forfeit the determinism this repo requires, and
   feature sampling in mojotrees already has its own keyed-stream mechanism.
   The enumeration here returns the full candidate set and a sampler is the
   caller's business. `SelectCtrsToDropAfterCalc` reads
   `NMemInfo::GetMemInfo().RSS`, which makes CatBoost's answer depend on what
   else is running on the machine; it is not ported and should never be.
6. **`max_ctr_complexity` still defaults to CatBoost's 4 in `CtrParams`, and
   the whole module is off.** Section 5's bound says 4 is unaffordable at 1M
   rows. Changing the recorded CatBoost default would make a ported
   configuration read differently from the configuration it was ported from,
   which is worse; the place to state the position is the wiring lane's
   default, and this note is the argument it should cite.

**Not built, deliberately.** The Rsm coin flip and the RSS drop policy (item 5
above); `PerFeatureCtrs`, which `GetCtrInfo` consults ahead of `SimpleCtrs` and
which is a per-feature option surface rather than a mechanism; the actual
`ComputeOnlineCTRs` call over a combination, which is A19's four loops taking
this module's bucket ids as their `categories` argument and needs no new code
here, only a caller; and the tree-level CTR cache with its 50-projection trim,
which is a caching policy for a trainer that does not exist yet.

#### Appendix: `ctr_target_border_count` and MinEntropy, for scoping

Not this lane's territory (A15's), asked for as a scoping question, and the
answer is: **the orchestrator's reading is correct, and the fix is much smaller
than 500 lines at the default.**

The two are genuinely different at one border. The greedy's
`TWeightedFeatureBin::UpdateBestSplitProperties` (`binarization.cpp:1473-1493`)
computes the weighted-median position `lb` by `LowerBound` on the cumulative
weights and then scores **exactly two** candidate cuts, `lb` and `lb + 1`:

```cpp
const double scoreLeft  = CalcSplitScore(lb);
const double scoreRight = CalcSplitScore(ub);   // ub = lb + 1
BestSplit = scoreLeft >= scoreRight ? lb : ub;
```

The exact binarizer runs the banded DP at `binarization.cpp:193`. But at
`maxBordersCount = 1` that DP does not run its main loop at all: `bins = 2`, the
loop is `for (l = 0; l < bins - 2; ++l)`, and the whole computation collapses to
the "Last match" block at `:630-667`. Written out, with `W` the cumulative
weights of the `k` distinct target values and `P(w) = w * log(w + 1e-8)`
(`:175`), it is:

    threshold = argmin over i in [0, k-2] of  P(W[i]) + P(W[k-1] - W[i])
    ties keep the smallest i        (the comparison at :649 is strict `<`)
    border    = (values[t] + values[t+1]) / 2                    (:691)

That is a **linear scan over the distinct target values, about ten lines**, not
a dynamic program. The weights are the counts of each distinct target value
(`SelectBorders` -> `BestSplit` -> `TExactBinarizer` ->
`SplitWithGuaranteedOptimum` -> `BestSplit<float, type>(weight, 1, thresholds,
E_RLM2)`), and `MinEntropy`'s argmin is invariant under scaling the weights, so
the normalization question does not arise.

**What it would take**, as a scoping estimate: implementing exact `MinEntropy`
*restricted to one border* is ten lines plus a test, and it is the only case a
default CTR comparison needs, since `ctr_target_border_count` defaults to 1.
Implementing it for `borderCount >= 2` is the real DP -- `E_RLM2`'s
divide-and-conquer band at `:560-627`, plus `E_Base`/`E_Base2`/`E_Linear_2L`
/`E_DaC` which are alternative modes of the same recurrence -- and only the one
mode CatBoost actually calls (`E_RLM2`) needs porting, which is roughly seventy
lines. So the honest scope is: **one border, ten lines, unblocks the default
comparison; all borders, one DP mode, ~seventy lines.** Neither is 500. It
belongs to whichever lane owns `binning.mojo`'s `check_border_type`, which is
not this one, and the `BORDER_MIN_ENTROPY` refusal is left standing here.
