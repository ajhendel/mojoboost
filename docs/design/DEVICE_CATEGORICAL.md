# Categorical splits on the device

**Status: design, nothing here is built. Written 2026-08-17 against head
`c775959`.** Every quantitative claim is cited to a file or to a recorded
record. Where a number does not exist this document says so instead of
producing one.

## The problem, and what it actually is

mojotrees performs no categorical splits on the GPU. Today's categorical
benchmark produced no GPU row at all, because
`bench/real_data/scenarios.py` declares `devices=["cpu"]` for
`high_cardinality_categorical`, and that declaration is honest rather than
conservative: the device genuinely cannot search a categorical column.

Andrew's directive of 2026-08-17 is that the GPU is the point of the product
and that every comparison shows us with and without it. So a whole scenario
class has no GPU story.

**The first thing to establish is that this is a SEARCH gap and not a device
categorical gap, because the two invite completely different projects.**
Device-side categorical *inference* already exists.
`gpu_categorical.mojo:173` `CatSetPool` is a device-resident pool of 256-bit
category sets, and its own docstring records that `gpu_predict.mojo` "already
concatenates a model's category sets into one device pool for inference" and
that `CatSetPool` is "the training-time twin". The routing side, the bitset
layout, the membership test (`cat_pool_contains`, `categorical.mojo:144`) and
the kernel-side offset convention are all built and in use. What is missing
is the part that *chooses* a set.

## What a categorical split is here today

Two modes, both in `categorical.mojo`, both mirroring LightGBM's
`FindBestThresholdCategoricalInner`, documented at `categorical.mojo:34-56`:

1. **One-hot, at or below `max_cat_to_onehot` categories.** Every category is
   tried one-vs-rest with the L2 term `lambda_l2`. The comparison is
   `n_categories + 1 <= max_cat_to_onehot`, corrected 2026-08-16 to match
   LightGBM's `num_bin <= max_cat_to_onehot` where `num_bin` counts the bin 0
   dummy (`categorical.mojo:65-73`).
2. **Sorted-prefix partition, above it.** Categories are ordered by
   `sum_grad / (sum_hess + cat_smooth)`, and prefixes of that order are
   accumulated from both ends, up to `max_cat_threshold` per side, with the L2
   term `lambda_l2 + cat_l2`. This is Fisher (1958): for a second-order
   objective the optimal many-vs-many partition is a prefix of that order.

Both produce a `CatBitset`, a fixed 4-word 256-bit set
(`categorical.mojo:116-118`). Bin 0 means missing, unseen or dropped, is never
a set member, and always routes right, matching LightGBM's
`CategoricalDecision` (`categorical.mojo:28-32`).

Only the winning partition leaves `find_best_categorical_split`. That single
fact is what every one of the three blocks below turns on.

## The three blocks, and which are inherent

### Cosine beside a categorical column: INHERENT, not unimplemented

`device_policy.mojo` BLOCK_SCORE_FUNCTION, narrowed to
`score_function == SCORE_COSINE and request.categorical` at `820c06b`.

This one is genuinely ill-defined and no amount of kernel work fixes it. The
partition search scores candidates with the L2 gain, and only its winner is
returned. Scoring that winner under Cosine and comparing it against numerical
candidates also scored under Cosine puts two functionals inside one argmax:
the *selection* of the category set happened under L2, so the number being
compared was produced by a different objective than the thing it is compared
to. Ordering under one functional and comparing under another is not an
implementation shortcut, it is a different algorithm with no defined answer.

**The CPU has exactly the same restriction, which is the proof that this is
inherent rather than a device limitation.** `split.mojo:843` raises on the
pair with the same reasoning. Anything claiming `device='cpu'` as an escape
is wrong, and the refusal message said so until it was corrected today.

The exit is not a kernel. It is to stop offering the column: a CTR-replaced
column is dropped from `BinnedMatrix.usable` and never reaches the scan
(`split.mojo:808-814`).

### `random_strength` beside a categorical column: INHERENT

BLOCK_RANDOM_STRENGTH, narrowed at `c775959`. Same shape of argument. A
numerical feature has every candidate noised; a categorical feature would
have only the partition search's single winner noised, because only the winner
exists by the time noise could be applied. That is a different regularizer
under the same parameter name, and it changes with category count rather than
being a property of the parameter. `split.mojo:816` refuses it on the CPU too.

LightGBM does randomize its categorical set search, but under a *separate
rule* over partitions rather than by noising a returned gain
(`split.mojo:766-774`). Matching that is a distinct piece of work from making
the device noise a category set, and it belongs to the search, not the device.

### `grow_policy=oblivious` beside a categorical column: UNIMPLEMENTED

BLOCK_GROW_POLICY, mirroring `gpu_resident_round:1647` `OBLIVIOUS_CATEGORICAL`.
The message is accurate: a symmetric level commits one `(feature, bin)` split
for the entire level and the device level search evaluates ordinal thresholds
only.

This is the only one of the three that is merely unbuilt. It is also the least
valuable to build, because a symmetric level sharing one category set across
every node at that depth is a strictly weaker model than the per-node sets the
CPU grower produces, and CatBoost, the library whose shape this is, does not
solve it this way either. It uses CTRs.

## Hard constraints any design must respect

**One byte per bin.** `BinnedMatrix.bins` is `List[UInt8]`
(`binning.mojo:2239`). 255 is a hard category ceiling today.

**The split set is sized to that byte and is on the wire.**
`CAT_BITSET_WORDS = 4` gives `CAT_MAX_BINS = 256`
(`categorical.mojo:116-117`), and the constant is consumed by
`distributed_strategies.mojo:606` as `CANDIDATE_WORDS = 8 + 2 *
CAT_BITSET_WORDS`, which is a wire record, by `gpu_fused_round.mojo:425` for
pool capacity, and by `gpu_categorical.mojo:205` and its kernel ABI, which
passes a set as four scalar `UInt64` arguments. Widening it is a multi-file
lane crossing serialization and the wire format, and it is *blocked behind the
byte anyway*: a bin id above 255 does not fit the cell, so widening the set
alone buys nothing.

**No cross-lane primitives.** `docs/GPU_BACKEND_SPECIALIZATIONS.md:104-108`
records that "no kernel here performs a shuffle, a ballot, or a warp-level
reduction", and that the gate a future cross-lane specialization would pass,
`require_subgroup_width_known`, is called by nothing. A cross-thread reduction
must use `block` operations.

**Queue depth 64, unraisable, and it is a PRICE and not a constraint.**
`bench/results/PHASE2_PREREGISTRATION.md:597` records the depth and that enqueue
cost lands "exactly where a 64-deep queue predicts", which is 6 to 7 microseconds
a launch under the depth and 14 to 17 over it. A design that adds launches per
node pays that rate, per launch, without limit. It is listed under hard
constraints because the rate is real and unavoidable, not because 64 is a number
a design may not exceed. A full queue blocks the enqueueing thread rather than
dropping work, and the fastest arm this package ships runs 2,303 buffers a tree
(`docs/GPU_PORTABILITY.md` 6.2, `docs/design/SWITCH_GRID.md` section 6 item 8).
This bullet ended "A design that adds launches per node pays here" with no such
qualification, and a reader took the depth for a ceiling in the ranking below.

## The options, ranked

### (c) CTR replacement so the column becomes numerical. RECOMMENDED.

The column is replaced by target-statistic columns, becomes numerical, is
dropped from `BinnedMatrix.usable`, and the existing device path runs
unmodified. This is what CatBoost does, and it is the reason CatBoost never
had to solve device category partitioning.

What makes this the recommendation rather than a dodge:

- **It dissolves all three blocks at once instead of one.** Both inherent
  blocks test whether the scan is OFFERED a categorical column
  (`split.mojo:808-814`, and the same declaration-versus-searchability
  distinction in `train_gpu.mojo:494-507`). A replaced column is not offered.
  Cosine and `random_strength` then work on the device beside what used to be
  a categorical column, which no device category search would ever achieve,
  because those two blocks are inherent to partition search itself.
- **It sidesteps the byte ceiling entirely.** A target statistic is a real
  number read from the raw category code, so it is never truncated by a
  255-bin cap. This matters because the 255-bin cap is the confirmed cause of
  today's accuracy failure, and no device work addresses that.
- **The machinery exists and was repaired today.** `ctr` was fixed at the
  source on 2026-08-17: the mode default now declines on entry points that
  cannot carry a bundle instead of refusing the fit, and an explicitly named
  rule on a bundle-less dataset is still refused.

**Kernel and memory shape: none.** That is the point. The replaced column is a
float column and the existing numerical device histogram, level scan and
routing handle it. Launches per tree are unchanged, so this option adds no
enqueue cost at all. That sentence read "so the 64-deep queue is unaffected",
which credited (c) with clearing a bar there is no bar at; the real credit is that
it adds no launches, and a launch is priced per launch (see the queue-depth
constraint above). Engineering price is *zero new kernels* and one lane of
integration and verification, mostly proving the replaced fit agrees with the
CPU.

**Payoff: NOT ESTABLISHED, and this is the honest weak point.** The mechanism
would newly enable a GPU cell where none exists, so the comparison would be
against our own CPU number. Today's categorical run at 799,110 rows by 15
features gives `mojotrees` CPU train median **2.343 s [2.197, 2.383]**,
against `catboost` **3.992 s** and `lightgbm` **5.677 s**, five repeats
(`bench/real_data/results/20260817T102725Z-cat1m/`). What a GPU cell would do
to 2.343 s **does not exist as a number**. The only adjacent measurement is
the dense scenario, where `mojotrees` cpu 7.431 s becomes gpu 3.616 s, about
2.1x (`20260817T101835Z-dense1m/`), but that is 100 features of dense
numerical data and this scenario is 15 columns of which 5 are categorical, so
transferring the ratio would be inventing a number. Note also that CTRs add
columns, so the shape changes, which is a further reason not to transfer it.

There is one recorded caution: `docs/LIGHTGBM_PARITY.md` asserted a measured
CTR accuracy remedy and that claim was **withdrawn today** as having no
provenance in any record. So CTR is the best-supported hypothesis for both the
accuracy gap and the device gap, and it is a hypothesis in both.

### (a) One-hot only on the device, refusing set splits. SECOND, and cheap.

Handle `n_categories + 1 <= max_cat_to_onehot` on the device, refuse above it.
One-vs-rest needs no ordering, no prefix accumulation and no sort: each
category is an independent candidate whose two children are "this bin" and
"everything else", which is a per-bin reduction the existing histogram already
materializes. This fits the current kernel shape almost unchanged.

The reason it is second rather than first is coverage, and coverage here is
bad. `max_cat_to_onehot` is small, and the scenario that motivates this work
is *high cardinality* by construction, at up to 200,000 levels. So this option
would light up a device path that the workload in question never takes. It is
worth doing as a by-product if (c) is done, and it is not worth a lane alone.

It also does not lift the two inherent blocks. A one-hot candidate is still
one candidate produced by a categorical search, so the two-functionals
argument still applies under Cosine, and the shipped default carries Cosine.

### (b) A full device partition search. NOT RECOMMENDED.

This is the option that looks like the real answer and is not. It requires a
per-node sort by `sum_grad / (sum_hess + cat_smooth)` on the device with no
cross-lane primitives available, then a two-ended prefix accumulation, then a
set write, per categorical feature per node. It adds launches per node, at 14 to
17 microseconds each once the stream is past the queue's depth, on a path whose
symmetric variant is *already* 2.4x slower than our own CPU. And after all of it,
the two inherent blocks still stand, so the shipped default configuration still
could not use it.

**One of the four reasons above was retired on 2026-08-18 and the ranking does
not move.** The launch sentence read "It adds launches per node against a 64-deep
queue", which read as a hazard cleared or not cleared. It is a price, so it is
stated as one here, and it is a price on a path already losing to our own CPU.
The three reasons that carried this refusal never touched the queue: there are no
cross-lane primitives to sort with, the two inherent blocks mean the shipped
default could not use the result, and the byte ceiling below caps what a partition
search could even see. **NOT RECOMMENDED** stands on those, unchanged.

It is also the option most exposed to the byte ceiling: a partition search
over a column truncated to 254 categories is a search over data we already
know is the cause of the accuracy failure.

## Recommendation

**Do not build a device categorical search. Do (c), and sequence it behind the
accuracy question it shares a mechanism with.**

The single experiment that informs both is the one already chosen for the
accuracy gap: one `high_cardinality_categorical` cell with CTRs on, beside the
existing `ctr='off'` cell. If CTRs close the accuracy gap, they also hand us
the GPU cell for free, because a replaced column is numerical and the device
path already runs. One measurement decides two questions.

If CTRs do not close the accuracy gap, then the categorical story is bounded
by the one-byte bin cell, and the honest next lane is widening that cell
rather than anything on this page. In that case device categorical search
should still not be built, because widening the cell changes the bitset, the
wire record and the kernel ABI that any device search would have been written
against.

**What to do instead, stated plainly, because "not yet" is only useful with an
alternative.** The GPU work with the best-established payoff right now is not
categorical at all. It is the symmetric path, where measured numbers exist:
`mojotrees_catboost_mode` gpu 17.118 s against its own cpu 9.009 s
(`20260817T101835Z-dense1m/`), an inversion on the shipped default's tree
shape, with a `block_dim=1` level scan identified as a cause. That is a
confirmed defect on the default configuration, and it outranks opening a new
scenario class.
