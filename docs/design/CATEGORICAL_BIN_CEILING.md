# The categorical bin ceiling, and what it is actually made of

Status. **Scoping only. Nothing here was built, compiled, trained, or timed by
this lane.** Every number is labelled either QUOTED (measured by another lane
and repeated here), DERIVED (arithmetic over bytes, slots, or comparison
counts, shown in full), SIMULATED (a pure Python model of a rule, run here, no
mojotrees involved), or ASSUMED. The source claims are reads of this
repository at head on 2026-08-17, each citing the file and the symbol, and of
CatBoost `master` where CatBoost is named.

This note has two subjects. The second one is the important one.

1. An appendix that CLOSES the `GreedyLogSum` border placement item as NOT
   RECOMMENDED, with the analysis that supports the closure.
2. The scoping of the categorical bin ceiling, which is a standing accuracy
   gate failure and is the reason the lane pivoted.

---

# Part 1. The failure, and the size of it

QUOTED from the lane that measured it, on `high_cardinality_categorical`.

| quantity | value |
|---|---|
| our average precision | 0.421925 |
| LightGBM's average precision | 0.479591 |
| relative gap | 12.02 percent, against a 10 percent limit |
| per-feature bin counts, ours | 9, 65, 255, 255, 255 |
| per-feature bin counts, LightGBM's | 9, 65, 989, 19433, 15952 |
| row coverage of our three truncated columns | 26.5, 1.8, 10.8 percent |
| LightGBM's coverage of the same three | 98.9, 98.1, 44.5 percent |
| oracle at LightGBM's bin counts | 0.552770 |
| oracle at our bin counts | 0.442548 |
| so the ceiling costs | 19.9 percent, against a measured gap of 12.02 |
| CatBoost, which runs CTRs here | 0.440563 |

Two consequences of that table are worth stating before any design.

**The ceiling more than fully accounts for the failure**, so this is the right
target. **And a target statistic does not close it**, because CatBoost runs
CTRs on this scenario and lands at 0.440563, which is 8.1 percent behind
LightGBM and within noise of our own 254-bin oracle. That kills the standing
plan, which was to reach for a CTR default.

One qualification on the CTR half, because it changes what the evidence
supports. What was measured is CATBOOST's CTR, not `ctr_columns.mojo`. Ours is
a different estimator with a different denominator and a different
permutation, and it has never been run on this scenario. The honest reading is
that an ordered target statistic of CatBoost's shape does not close this gap,
which is strong indirect evidence about ours and not a measurement of it.

## Why the bin counts are what they are, on both sides

DERIVED, from `bench/real_data/scenarios.HIGH_CARDINALITY_LEVELS` and
`generators.high_cardinality_categorical`, and it explains all five numbers.

The five categorical columns at the standard tier carry nominal
`(8, 64, 1_000, 20_000, 200_000)` levels over 800,000 training rows, so 8 and
64 fit under any ceiling and the other three do not. Ours are `min(nominal,
254) + 1`, which is where 255, 255, 255 comes from
(`categorical.fit_categorical_spec` keeps `max_bins - 1` codes and
`_keep_most_frequent` chooses which). LightGBM's 989, 19433 and 15952 are the
levels present in its 200,000-row binning sample holding at least
`min_data_in_bin` of 3 rows, which is what its categorical `FindBin` admits.
That reading matters for one reason and one only, and it is the reason the
`UInt16` objection is retired. **LightGBM's own widest table on this scenario
is 19,433 entries, so parity needs 16 bits and never more.**

It also says something the row-coverage numbers do not. Our coverage of the
1,000-level column is 26.5 percent because the column is drawn UNIFORMLY, so
`_keep_most_frequent` keeps 254 arbitrary levels of 1,000 and 254/1000 is
25.4 percent. On the power-law column the same rule keeps the head and
reaches 10.8 percent. So on a uniform column the selection rule cannot be
blamed and cannot be fixed without the label; the width is the whole story
there. That is not true of the power-law column.

## Where the recoverable signal is, which decides the width required

QUOTED from `HIGH_CARDINALITY_LEVELS`, a leakage-free per-column target
statistic scored on a held-out split, taken at one fifth scale.

    column        rows/level   AUC from that column alone
    8 levels         19,956    0.7041
    64 levels         2,494    0.7031
    1,000 levels        798    0.6607
    20,000 levels        40    0.5753
    200,000 levels        4.5  0.5234

DERIVED, and this is the single most useful scoping fact in the note. A
category enters the split search only when it holds at least `cat_smooth` rows
AT THE NODE (`tree_parameters_extra.cat_enters_search`, `count >= cat_smooth`,
LightGBM's default 10). At the root, the 1,000-level column has 800 rows per
level and every level enters; the 20,000-level column has 40 and every level
enters; the 200,000-level column has 4.5 on average and most levels never
enter at all, at any depth. So the ceiling is costing us the most on the
column that needs the SMALLEST widening.

**The width the accuracy actually asks for is about 1,024 for the column that
carries the most recoverable signal, about 20,000 for the next, and nothing
at all for the widest.** Any design that treats this as a 200,000-level
problem is solving a problem the split search cannot use.

---

# Part 2. What the ceiling is made of

The 256 shows up in SIX independent representations, not four. Each is listed
with what it is, whether it binds for a CATEGORICAL feature specifically, and
what it would take.

### R1. The bin plane. `binning.BinnedMatrix.bins: List[UInt8]`

`binning.MAX_BINS = 256`, and `fit_bins` refuses `max_bins > MAX_BINS`.
`BinMapper.transform` writes a categorical cell as
`UInt8(cats.bin_of(f, v))`, a silent narrowing.

**BINDS, and it is the only one of the six that binds unavoidably.** Routing a
row at a categorical node needs that row's category identity, and an identity
drawn from more than 256 values does not fit in a byte, whatever the split
representation is. Note what it does NOT require. It does not require the
NUMERIC path to widen, and it does not require one plane; a second plane
holding only the wide categorical columns is enough.

### R2. The split set. `categorical.CatBitset = SIMD[DType.uint64, 4]`

`CAT_BITSET_WORDS = 4`, `CAT_MAX_BINS = 256`, with `cat_contains`, `cat_add`,
`cat_pool_contains` over it, `Tree.cat_bitset` as a flat pool at stride 4,
`predict.lvl_cat_words` at the same stride, and `serialize` plus
`lgbm_model_io` plus `model_editing` all validating against the stride.

**BINDS TODAY AND NEED NOT, AND THIS IS THE ANSWER TO THE QUESTION THAT WAS
ASKED FIRST.** A categorical split is a set membership test and it does not
need the same representation as a numeric bin index. It does not need a dense
bitset either, because the set is already capped at a small size independent
of the category count. `categorical.find_best_categorical_split` walks at most
`cat_side_cap(used, max_cat_threshold)` categories onto one side, which is
`min(max_cat_threshold, ceil(used / 2))`, and `max_cat_threshold` is LightGBM's
default 32. The one-hot branch selects exactly one. **So a split set never
holds more than 32 members at stock, at any bin count.**

A sorted list of at most `max_cat_threshold` ids is therefore a complete
replacement, and DERIVED it is SMALLER than the current dense set for every
width above 256. At 32 ids of `UInt16` it is 64 bytes against 32 bytes today,
and against 2,432 bytes for a dense 19,433-bit set. Membership becomes a
bounded binary search over at most 32 sorted entries, 5 compares, against one
shift and one mask today.

### R3. The distributed candidate wire.
`distributed_strategies.CANDIDATE_WORDS = 8 + 2 * CAT_BITSET_WORDS`

A fixed-width all-gather, 16 non-negative `Int` per rank per node, reduced by
`allreduce_max_int` over slots, with the 256-bit set carried as
`2 * CAT_BITSET_WORDS` 32-bit halves so no word can reach a sign bit.

**Binds only for a DENSE widening, and not for a sparse one.** DERIVED, a
dense 65,536-bit set makes `CANDIDATE_WORDS` 2,056 and multiplies the exchange
by 128. A sparse set makes it `8 + 1 + max_cat_threshold` at 32-bit ids, which
is 41 at stock, a fixed 2.6x that does not move with the bin count. The
sign-bit rule is satisfied by a bin id, which is non-negative by construction,
so the halving trick is not even needed for the id list.

### R4. The device kernel ABI, four scalar `UInt64`

`gpu_active_rows._row_goes_left(..., cat0, cat1, cat2, cat3)` and three
callers, plus `gpu_sparse` at :785; the device split record
`gpu_split_search.IREC_CAT0` with `CAT_WORDS = MAX_SPLIT_BINS // 16`; the node
table `gpu_tree_tables.TN_CAT0 + CAT_WORDS`; `gpu_predict`; and
`gpu_bin_packing.BIN_WIDTH_MAX = 8` with its explicit refusal to renumber a
bin.

**Does NOT bind, because the device already refuses this shape.**
`gpu_split_search` refuses `n_bins > MAX_SPLIT_BINS` twice, once as a
supported-check returning `SEARCH_TOO_MANY_BINS` (:5711, whose docstring names
`train_gpu._check_device_search_supported` as the caller that should consume
it) and once as a raise inside the search itself, "split search supports at
most 256 bins" (:6796). `MAX_SPLIT_BINS` is 256. `bench/real_data`'s categorical row is declared CPU
only (`SCENARIO_ROWS`, `"devices": ("cpu",)`), and Base A does not run there at
all. So a CPU-only wide path needs no new refusal and no ABI change; the
existing guard becomes the boundary, and the failure mode is a named raise
rather than a wrong answer. This is the same posture
`sampling.check_bootstrap_honored` and `oblivious_device_supported` already
take.

### R5. `categorical.CategoricalSpec` itself

`codes: List[Int]`, `offsets: List[Int]`, `bin_of` a binary search over the
ascending codes of one feature.

**Does not bind at all.** The table is 64-bit and unbounded. The ceiling
enters in exactly one expression, `_keep_most_frequent(raw_codes, raw_counts,
max_bins - 1)`. `distinct_category_codes` already exists beside it as the
untruncated view, built for the CTR path, and its docstring states the
principle this note is arguing about, that the cardinality has to move out of
the bin id because a byte caps it there.

### R6. The one that was not on the list. `histogram.Histogram` is RECTANGULAR

`Histogram` holds `n_features * n_bins` cells with ONE global `n_bins`
(`histogram.mojo`, `Histogram.zeroed(n_features, n_bins)`, cell index
`f * n_bins + b`), and `n_bins` comes from `BinnedMatrix.n_bins`, which is
`max_bins`. Note that `BinnedMatrix` ALREADY carries `feature_bins` and
`bin_offset`, a per-feature width table with LightGBM's
`group_bin_boundaries_` semantics, and that its docstring says only the
row-major blocked kernel's private partials use it while "the output histogram
keeps its `f * n_bins + b` shape, which every other file indexes with".

**This is the constraint that actually prices a global widening out**, and it
is not in `binning.mojo`. DERIVED, on this scenario at the standard tier,
800,000 training rows by 15 features, with three planes at 24 bytes per cell
(two `Float64` plus one count).

    today                 15 x   255 x 24 =  91.8 KB per node histogram
    global n_bins = 19433 15 x 19433 x 24 =  6.99 MB per node histogram   76x
    wide per feature      12 x 255 x 24 + (989 + 19433 + 15952) x 24
                                        = 73.4 KB + 873 KB = 946 KB       10x

The 76x is fatal and the arithmetic is why. A leaf-wise fit at 64 leaves would
hold a pool of order 64 histograms, so 5.9 MB becomes 448 MB, and every
histogram is ZEROED before it is built, so the clear pass alone moves
64 x 7 MB x 72 trees, about 32 GB of write traffic per fit, to produce
information the split search cannot use on the widest column. **A global
widening of `n_bins` is DECLINED at that price.** The 10x figure is the real
cost of the accuracy, and both surviving designs below pay it, because it is
the cost of the bins themselves rather than of any representation.

---

# Part 3. Three designs, priced

## Design A. Widen everything to `UInt16`

Widen R1, R2, R3, R4 and R6 together. Rejected. It pays the 76x of R6, breaks
the device ABI, multiplies the distributed exchange by 128, invalidates every
serialized model and every LightGBM model we can import, and doubles the bin
plane for the numeric columns that asked for nothing. Priced here so the
decline carries a number rather than a judgement.

## Design B. A wide categorical side plane, CPU only

- `BinnedMatrix` gains `wide_bins: List[UInt16]` and a per-feature index into
  it; a feature is either narrow (in `bins`) or wide (in `wide_bins`). Both in
  `binning.mojo`.
- `fit_categorical_spec` gains a per-feature `keep` cap so a designated wide
  column keeps up to 65,535 codes instead of `max_bins - 1`. One argument, one
  expression, in `categorical.mojo`.
- The histogram for a wide feature is built into its OWN array rather than
  into a rectangular slice. This is the piece that makes the design possible,
  and it is already supported by the search rather than needing to be
  invented. `find_best_categorical_split(gh, count, base, n_categories, ...)`
  takes a flat array and a base offset, so pointing it at a per-feature array
  with `base = 0` and `n_categories = 19432` is a call-site change and not an
  algorithm change.
- The split set becomes the sparse id list of R2. Sibling subtraction works
  per wide feature over its own array, unchanged in form.
- The device refuses a wide feature by name, which it already does.

What it costs, DERIVED. 10x histogram slots on this scenario (Part 2, R6),
plus 2 bytes per row per wide column for the side plane, which is
3 x 800,000 x 2 = 4.8 MB against the 12 MB main plane. Plus one argsort per
node per wide feature over the categories that pass the population filter.
That last one deserves its arithmetic because it looks alarming and is not.
`find_best_categorical_split` filters BEFORE it sorts, so the root sorts about
19,433 keys on the widest usable column and a node at depth d sorts only the
levels still holding `cat_smooth` rows, which falls off geometrically. Summing
the ladder gives roughly twice the root, so about 40,000 keys per tree per
wide column, or 40,000 x log2(40,000) which is about 600,000 comparisons, and
at three wide columns over 72 trees about 1.3e8 comparisons for the fit.
That is a small term next to the histogram build and it is not the reason to
prefer or reject this design.

Files it touches outside `binning.mojo`. `categorical.mojo` (the keep cap, the
sparse set, `cat_add`'s missing bound check), `histogram.mojo` (a per-feature
histogram entry point), `split.mojo` (the call site and the `n_cat >=
hist.n_bins` guard), `tree.mojo` and `serialize.mojo` and `predict.mojo` and
`model_editing.mojo` and `lgbm_model_io.mojo` (the set representation),
`distributed_strategies.mojo` (the wire). That is a large blast radius for one
scenario, and it is why Design C exists.

## Design C. Shard a wide column into several narrow categorical columns

**No representation changes anywhere. R1 through R6 are all untouched, both
backends keep working, and every serialized model stays valid.**

A 1,000-level column becomes four categorical columns. Column j declares the
levels `[254j, 254(j + 1))` of the ascending code list and sends every other
code to its own bin 0, which is where missing, unseen and dropped codes
already route and which is never a member of a split set
(`categorical.mojo`, "Default routing"). Each shard is an ordinary categorical
feature with at most 254 categories, so every ceiling in Part 2 is satisfied
by construction.

Why this is not a hack. The machinery already exists in `binning.mojo` and was
built for CTRs. `append_ctr_columns` appends synthetic columns to a
`BinnedMatrix` in place and its docstring states the property this design
needs, that "after this call `matrix.n_features` counts them, so every
histogram builder, every split search and every grower sees ordinary binned
features and needs no edit". `ctr_extend_cats`, `ctr_extend_missing` and
`ctr_extend_usable` are the three table extensions beside it, `BinMapper` has
`n_total_features` and `has_ctr` for the train and predict asymmetry, and
`transform` already synthesizes appended columns at predict time. A shard
append is the same shape as a CTR append with one difference, that the
appended columns are CATEGORICAL rather than numeric, so `ctr_extend_cats`
gains a sibling that appends code tables instead of `False` flags.

What it costs, DERIVED and stated as a range because two of the terms are
policy rather than arithmetic.

- Histogram and search work, the same 10x as Design B on this scenario, for
  the same reason. Sharding does not avoid the cost of the bins. It avoids
  the cost of the WIDTH. At the standard tier the three columns become
  4 + 77 + 63 = 144 shards, so `n_features` goes from 15 to 156 and the
  rectangular histogram goes from 15 x 255 to 156 x 255 cells, 10.4x, with
  `n_bins` still 255 and R6 never engaged.
- Bin plane, one `UInt8` column per shard, so 144 x 800,000 = 115 MB against
  the 12 MB today. **This is the one place Design C is materially worse than
  Design B**, which pays 4.8 MB for the same information, and the reason is
  that a shard spends a whole byte per row to say "not in this shard" for
  every row it does not own. At the standard tier that is a real 100 MB and it
  scales with rows times shards.
- Search quality is lower than Design B and the difference is bounded. A
  single split can only draw its set from one shard's levels rather than from
  the global gradient order, so a partition that would have spanned shards now
  takes stacked splits, each of which must clear `min_gain_to_split` on its
  own. What bounds the loss is that `max_cat_threshold` already caps a set at
  32 members, so no single split was ever going to draw 989 levels.
- Two semantic changes that are NOT free and must be decided rather than
  discovered. `feature_fraction` would sample shards independently, so a
  column's chance of being offered to a node changes with how many shards it
  has, which changes what the regularizer means. And feature importance for
  one user-visible column is spread over its shards, so the importance vector
  and every consumer of it reports something new.

## Which one, and the honest answer

Design C is the cheaper thing to TRY and Design B is the better thing to
SHIP. C proves or disproves the accuracy claim at a fraction of the
engineering, on both backends, with no model-format change and no wire change,
and its 115 MB and its `feature_fraction` distortion are exactly the reasons
it is a probe rather than a product. If C recovers most of the oracle's 19.9
percent, B is worth its blast radius and the measurement justifies it. If C
recovers little, B should not be built either, because the two share the
mechanism and differ only in how efficiently they pay for it.

---

# Part 4. The smallest honest change, and the cheapest experiment

The smallest change that could close the GATE touches no representation at
all, and the evidence for it is already in the numbers above.

Our measured 0.421925 sits below the 254-bin oracle's 0.442548. Closing that
distance alone would put the gap at
`(0.479591 - 0.442548) / 0.479591 = 7.72 percent`, INSIDE the 10 percent
limit, with the ceiling exactly where it is. The whole of that headroom lives
in `_keep_most_frequent`, which chooses WHICH 254 levels get bins.

Two things have to be said about that, and the second one is why this is
listed as an experiment rather than as a plan.

The oracle uses the generator's true per-level effects, so 0.442548 is an
upper bound on every possible selection rule at our width and not a target a
fit-time rule can be expected to reach. And on a UNIFORMLY drawn column, which
is what the 1,000-level and 20,000-level columns are, no label-free rule can
beat an arbitrary choice, because the levels are exchangeable in every respect
a fitter can see. A rule that uses the label to choose bins is a target
statistic wearing a different hat, with the leakage exposure that implies.
So the headroom is real, bounded, and probably mostly unreachable, and the
power-law column is the one place a better rule plausibly helps.

**The cheapest experiment, and it needs no compiler, no benchmark and no
training run.** Extend the oracle that produced 0.442548 and 0.552770 to
report, per column and at several widths, the AP it reaches. Three widths per
column, 254, 1024, and the LightGBM count, over the three truncated columns.
It is a Python model over the generator's per-level effects, the same
instrument already used, and it answers the only question that decides between
"do nothing", C, and B, which is HOW MUCH of the 19.9 percent comes from the
1,000-level column that needs 1,024 bins as against the 20,000-level column
that needs 19,433. If the answer is concentrated in the first, the required
width collapses and so does the cost of every design here.

---

# Part 5. Exact changes needed outside this lane's ownership

Written as requests, not applied.

**CORRECTION 1, `bench/real_data/engines.py` at the CatBoost binning refusal
(the `EngineError` that begins "the CatBoost arm takes no per-arm binning
overrides").** The text says "CatBoost's bin budget is border_count, whose
default is 65535 against LightGBM's 255". That is FALSE and it confuses the
limit with the default. VERIFIED from source at
`catboost/private/libs/options/data_processing_options.cpp:14-19`, which
constructs `TBinarizationOptions(GreedyLogSum, type == ETaskType::GPU ? 128 :
254, ENanMode::Min, 200000)`, and at
`catboost/private/libs/options/restrictions.h:10`, where
`GetMaxBinCount()` returns 65535 as the validation ceiling. So CatBoost's CPU
default is 254 BORDERS, which is 255 bins, the same granularity as our
`max_bin = 255`. Suggested replacement for the two clauses, keeping the
refusal itself, which is still right for the reason that follows it.

    "CatBoost's bin budget is border_count, which counts THRESHOLDS where
    max_bin counts BINS. Its CPU default is 254 borders, so 255 bins, the
    same granularity as this harness's max_bin of 255; 65535 is
    GetMaxBinCount(), the validation ceiling, not the default. The refusal
    stands anyway: moving max_bin on one arm while the other stays at its
    own stock number is two sweeps rather than one axis."

This is a CORRECTION and not a suggestion. It matters beyond tidiness, because
a reader who believes CatBoost quantizes at 65,535 borders has a ready
explanation for any accuracy gap, and the true figure is within one bin of
ours.

**CORRECTION 2, `bench/real_data/frontier.py`, the `max_bin` axis note.** It
says Base A's base value of 254 "is CatBoost's border_count, which is the same
granularity as 255 bins". Under the rule the catalog states, and it states it
correctly, our `max_bin` counts bins and CatBoost's `border_count` counts
thresholds, so CatBoost's 254 equals our 255 and Base A at 254 is one bin
COARSER than CatBoost, not equal to it. The number may well be worth keeping;
the sentence justifying it is off by one and should say so.

**REQUEST 3, `categorical.mojo`, a bound check.** `cat_add` writes
`bitset[bin >> 6]` with no range test while `cat_contains` and
`cat_pool_contains` both test `bin <= 0 or bin >= CAT_MAX_BINS`. It is
unreachable today because `split.find_best_split` raises on
`n_cat >= hist.n_bins` before the search runs, so this is hardening and not a
live defect. Any widening makes it live, and it is one line.

**REQUEST 4, if Design B or C proceeds.** `split.mojo`'s categorical branch
needs its guard and its call site widened or redirected, and `histogram.mojo`
needs a per-feature histogram entry point. Both are named here rather than
sketched, because the interface should be designed by whoever owns those
files.

---

# Part 6. What could not be verified without a compiler

Ranked, most consequential first.

1. **That the 254-bin oracle is achievable at all by any fit-time rule.** Part
   4 rests on a bound, not on a realizable rule, and the uniform columns argue
   against it.
2. **That Design C's sharded search recovers a useful fraction of the
   width.** The stacked-split argument is a plausibility argument. Only the
   oracle extension in Part 4, or a build, settles it.
3. **That `append_ctr_columns`' claim generalizes from numeric to categorical
   appended columns.** The docstring's "needs no edit" is stated for numeric
   CTR columns. A categorical appended column additionally needs the grower to
   read `matrix.cats` for the new ids, which `ctr_extend_cats` does not
   populate today.
4. **The 10x histogram figure's effect on wall clock.** It is a slot count and
   a byte count, not a time. The build is bandwidth-bound and the search is
   not, so the two terms scale differently and neither was measured.
5. **Whether the device refusal is reachable in practice on this scenario.**
   The scenario declares CPU only today. A later device categorical lane would
   meet the `MAX_SPLIT_BINS` raise, and I did not check every device entry
   point for a path that reaches a histogram before that guard.
6. **Whether any other consumer narrows a categorical bin silently.**
   `transform`'s `UInt8(cats.bin_of(...))` is the one I found, and it is
   currently protected by the fitter and by both import validators
   (`serialize` raises on "more categories than bins",  `lgbm_model_io` raises
   by name above `CAT_MAX_BINS - 1`). I did not audit `efb.mojo`,
   `sparse.mojo`, `external_memory.mojo` or `tree_sparse.mojo`, each of which
   constructs a `CategoricalSpec` from its own fitter.

---

# Appendix. `GreedyLogSum` border placement, CLOSED as NOT RECOMMENDED

The lane opened to implement CatBoost's `GreedyLogSum` border placement in
`binning.mojo` behind a switch. It closed without one. Three findings, then the
verdict.

**Finding 1. It was already built, and the brief did not know.** Five of
CatBoost's seven border selection algorithms are in `binning.mojo` at head,
verified against CatBoost source on 2026-08-16, catalogued as A15, and tested
in `tests/test_catboost_quantization.mojo`. `fit_bins` takes `border_type` and
defaults it to `BORDER_QUANTILE`. What was missing was REACHABILITY. No
trainer entry point passes the argument, and no Python or C surface spells it,
so the mode is unreachable from any fit. The lane's only real deliverable
would have been that edge.

**Finding 2. The implementation is faithful, and I checked it against source
rather than against the earlier lane's report.** A partial CatBoost checkout is
present on this machine, and `library/cpp/grid_creator/binarization.cpp` was
read directly. `TWeightedFeatureBin::CalcSplitScore` is
`Penalty(L + R) - Penalty(L) - Penalty(R)` with `-inf` at either end,
`UpdateBestSplitProperties` considers exactly two cuts at the ends of the
level holding the bin's median observation with ties going left, `GreedySplit`
loops `while (splits.size() <= maxBordersCount && splits.top().CanSplit())`,
and a non-first bin emits `0.5f * v[s - 1] + 0.5f * v[s]`. The Mojo port
matches all four. Two divergences are documented in the source and are real:
Float64 instead of `float`, and a total heap order (score, then ascending bin
start) where `std::priority_queue` has none, which this project needs for
reproducibility across worker counts. One further reading, VERIFIED here and
worth recording because the port chose the other form. CatBoost's DENSE path
runs `TFeatureBin` over the raw sorted array (`TGreedyBinarizer::BestSplit`,
the `else` branch), not `TWeightedFeatureBin` over grouped levels. The two
agree, and DERIVED rather than simulated this time. Both consider the two
level boundaries bracketing the median observation, and where they name
different pairs, which happens only when the median falls exactly on a
boundary, that boundary is in both pairs and wins both comparisons because
`log(L * R)` is maximized at balance.

**Finding 3. The rule cannot do what it was hypothesized to do, and this is
the analysis worth keeping.** SIMULATED, a pure Python port of both rules,
160,000 values, `max_bin` 255.

| column | greedy borders | quantile borders | identical | max rank shift |
|---|---|---|---|---|
| uniform continuous | 254 | 254 | no | 622 (0.389 percent of rows) |
| normal continuous | 254 | 254 | no | 622 (0.389 percent) |
| lognormal | 254 | 254 | no | 622 (0.389 percent) |
| Pareto a=1.2 | 254 | 254 | no | 622 (0.389 percent) |
| exponential | 254 | 254 | no | 622 (0.389 percent) |
| uniform, 1,001 levels | 254 | 254 | no | 344 (0.215 percent) |
| uniform, 301 levels | 254 | 254 | no | 555 (0.347 percent) |
| uniform, 61 levels | 60 | 60 | **yes** | 0 |
| Zipf integer codes | 254 | **232** | no | 311 |
| 40 percent atom at 0 | 254 | **154** | no | 193 |
| uniform continuous, `max_bin` 32 | 31 | 31 | **yes** | 0 |
| lognormal, `max_bin` 32 | 31 | 31 | **yes** | 0 |

Three readings follow, and each is a correction of something that was
believed going in.

- **Skew cannot separate the two rules.** The identical 622 across uniform,
  normal, lognormal, Pareto and exponential is not a coincidence, it is the
  mechanism. Both rules read only RANKS (`_cb_update_best` touches nothing but
  the cumulative counts, `quantile_boundary_indices` is arithmetic on
  `n_valid`) and both place a border at the midpoint of two adjacent levels,
  so both are invariant to any monotone transform of the column. The brief's
  instruction to look for a skewed generator was looking in a direction where
  no difference can exist.
- **Our rule is the OPTIMUM of CatBoost's objective on a tie-free column, not
  an approximation of it.** `Penalty<MaxSumLog>` sums to
  `sum(log(population))`, which is maximized at equal populations. The greedy
  reaches that only when the budget is a power of two, which is why `max_bin`
  32 comes out identical, and at 255 bins it leaves populations spanning 625
  to 1250 where equal frequency spans 627 to 628.
- **The border COUNTS differ, and the catalog says they do not.** Both engines
  produce `min(budget, distinct - 1)` only for CatBoost. Our rule derives
  boundaries from equally spaced ranks and collapses every boundary landing
  inside one run of equal values, so a tie-heavy column comes out UNDER
  budget, 232 against 254 and 154 against 254 above. The corrected statement
  is now in `binning.mojo`'s section header, recording what it used to say.
  `docs/design/CATBOOST_CATALOG.md` A15 and
  `scenarios.CATBOOST_UNMATCHABLE["binning_budget"]` carry the same
  overstatement and are owned elsewhere.

**And the data has no ties to exploit.** MEASURED here by reading the datasets
already on disk, which needed no training run. Every feature of
`generators.dense_regression` comes from `_stream`, a uniform, and 800,000
draws from a 53-bit uniform are distinct with probability about
`1 - 3.5e-05`. UCI YearPredictionMSD, the real dense row, is tie-free as well;
over its first 50,000 rows every one of its 90 features has between 49,098 and
49,982 distinct values and the largest single-value share of any feature is
0.01 percent. So on both dense rows the two rules differ by at most a fraction
of one bin, and the regime where they diverge materially, a column with more
distinct values than bins AND heavy tie mass, does not occur in either.

**Verdict, agreeing with the orchestrator's independent conclusion and adding
the mechanism.** NOT RECOMMENDED. The orchestrator's three reasons, that the
features are uniform rather than normal, that the effect reverses sign between
tiers, and that LightGBM is the worst arm on the term although we copy its
rule, are each sufficient on their own. The analysis here says why the sign
reversal is the expected behavior of a term with no mechanism behind it. Two
rank-only rules on tie-free columns differ by a sub-bin perturbation of the
same grid, so which one lands nearer the single discontinuity at `x4 = 0.7` is
decided by arithmetic coincidence at each size, and a coincidence changes sign
with the sample. Nothing was wired; the five modes stay reachable by argument
and unreached by any caller, and `binning.mojo` at head is bit-identical to
what it was before this lane, comments aside.
