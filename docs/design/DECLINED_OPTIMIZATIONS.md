# Declined optimizations, priced

An audit lane, 2026-08-17. Read-only. Nothing here was built, run, or changed.

The question this file answers is the one asked on 2026-08-17 after two
confidently written declines were falsified in one day. Where else in this
codebase was a faster path considered and refused, built and defaulted off, or
reverted, and does the stated reason survive being priced?

## How to read a row

Every row carries five fields and they are kept apart on purpose.

- **Where.** File plus enough quoted text that the citation survives the file
  being edited. Line numbers are a hint, not the citation. Several
  implementation lanes were live in this checkout while this was written.
- **What was declined.** One sentence.
- **Stated reason.** Quoted from the source.
- **Evidence class.** One of three, and the distinction is the whole point.
  **MEASURED** means a number exists in this repository, in source or in a
  results file, that supports the reason. **ASSERTED** means the reason reads
  as settled fact and no number backs it. **STALE** means a number backed it
  once and the thing it measured has since changed.
- **Price.** My independent arithmetic, with the arithmetic shown. **Every
  price below is an estimate.** None of it was measured by this lane. This lane
  ran nothing.
- **Survives.** Whether the stated reason still stands after the pricing.

Verified and inferred are separated in every row. "Verified" means I read the
code or the results file and it says what I claim. "Inferred" means I reasoned
from what I read.

## The two rulers everything is measured against

Both come from the brief and both are measured on this machine.

**Ruler A, the symmetric plane's whole launch budget.** The oblivious schedule
is 56 command buffers per tree, confirmed launch by launch. (Since 2026-08-17
the shipped default is **55**, because `MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD`
became the default and drops the last level's two batch launches while paying
one back for its partition's copy-back. One launch does not move this ruler, and
the ruler is quoted at 56 below so the arithmetic reads as it was taken.) A
launch is 10 to 20 microseconds. So

    56 launches x 100 trees x 15 us = 84 ms

against a symmetric GPU fit of **17.07 s**. That is **0.49 percent**. *Every*
decline on the symmetric plane whose only justification is launch count or
command buffer count is bounded above, in total, by half a percent of the fit.
Not each. All of them together. This is the single most useful fact in this
document and it falsifies a whole category of reasoning on sight.

**Ruler B, the leaf-wise plane, where launches do matter.** The resident
leaf-wise step is nine command buffers, roughly 30 steps a tree, so about 270
launches a tree and 27,000 a fit. At 15 us that is **0.405 s** against a
leaf-wise GPU fit of **3.659 s**, or **11 percent**. A launch-count argument is
worth taking seriously here and is worth almost nothing on the symmetric plane.
Several declines in this codebase quote a launch count without saying which
plane they are on.

Supporting anchors used below, all from the brief or from the repository.
Symmetric GPU 17.07 s, symmetric CPU 9.09 s, leaf-wise GPU 3.659 s, depth-wise
GPU 3.245 s, CatBoost 3.269 s, at 799,110 x 100, 100 trees, depth 6.

**THE SYMMETRIC AND LEAF-WISE DENOMINATORS ARE PRE-FLIP, AND EVERY PERCENTAGE IN
THIS DOCUMENT IS AGAINST THEM. Added 2026-08-17.** Later that day four GPU
performance switches were measured and became defaults, so at the same shape the
symmetric arm is around **10.36 s** rather than 17.07 s (2.20x from
`OBLIVIOUS_SUBTRACT` plus `OBLIVIOUS_SKIP_LAST_BUILD` plus `OBLIVIOUS_WIDE`
together) and the leaf-wise arm is around **3.2 s** rather than 3.659 s (1.21x
from `SPLIT_WIDE`). The anchors above are deliberately left as they were taken,
because the prices below were computed against them and a half-renumbered
register would be worse than a uniformly stale one. What a reader has to do is
mechanical: a decline priced at X percent of 17.07 s is roughly 1.65 X percent of
the fit that ships today, and one priced against 3.659 s is roughly 1.14 X. The
ORDERING of the register is unaffected, since every item scales by the same
factor within a plane. Nothing about which declines survive changes. Histogram
construction is 86 percent of a symmetric fit. GPU idle 76.5 percent, compute
22.9 percent. A Metal `enqueue_copy` is a synchronous full-queue drain. Host
wait about 126 us, enqueue about 6 to 7 us through queue depth 64 and 14 to 17
beyond it. Device copy bandwidth 75 to 85 GB/s.

---

# Part 1. Two claims from other lanes, confirmed from source

Both were raised elsewhere on 2026-08-17 and neither was confirmed. Both are
now confirmed. The first is a correctness defect, not a performance decline,
and it is the most serious finding in this audit.

## Claim 1. `GpuLeafBatcher.feat_dev` on the oblivious device path. CONFIRMED, and it is a live wrong answer under `feature_fraction < 1`

**Verified, by reading five call sites.**

1. `GpuLeafBatcher.feat_dev` has exactly three writers, all in
   `src/mojotrees/gpu_leaf_batching.mojo`. The constructor, which fills it with
   the identity under the comment *"Every item's feature table starts as the
   identity, so a caller that never narrows the feature set can launch without
   staging one"*; `set_shared_features`; and `set_item_features`.
2. `set_shared_features` has exactly one call site in the package,
   `src/mojotrees/histogram_gpu.mojo`, inside `_build_leaves_batched`, guarded
   by `if self.batch_feat_stamp != stamp:`. That is the **host-driven** batched
   leaf build.
3. `set_item_features` has no call site at all. Its own docstring says so.
   *"Per-node feature sampling narrows the search today and not the
   accumulation, so nothing in the trainer needs this yet."*
4. The oblivious device level build is
   `GpuHistogramBuilder.enqueue_desc_level_children`, which calls
   `GpuLeafBatcher.enqueue_device_plan_batch_fused` (or
   `..._fused_subtracting`). Neither it, nor
   `GpuHistogramBuilder.stage_desc_level_plan`, nor
   `GpuLeafBatcher.stage_device_plan`, touches `stage_feat` or `feat_dev`.
   `stage_device_plan` writes only `items_dev` and `scales_dev`.
5. The accumulation kernel resolves both the source column and the destination
   slice through that table. In `_batch_hist_atomic_kernel`,

       var f = Int(feat_ids[unsafe_offset = k * Int(feat_stride) + slot][0])
       ...
       var col = f * nr
       ...
       var slice_base = out_slot * N_PLANES * hs + f * nb

   so with the identity table, `f == slot`, and slot *s* accumulates global
   feature *s* into slice *s*.

**And the reader disagrees with the writer.** The level search is
`_scan_slot_oblivious_wide_kernel` and its narrow sibling in
`src/mojotrees/gpu_split_search.mojo`, both of which do

       var f = Int(feat_ids[unsafe_offset = table + slot][0])

against `GpuSplitSearcher.feat_dev`, which **is** set, from
`train_gpu._grow_tree_gpu_device_search` and
`GpuSplitSearcherCache.reset_for_tree` calling `searcher.set_features(
tree_features)`. It then reads the slice at `NODE_HIST_BASE + f * nb`. So the
search reads slice `tree_features[slot]` from a pool where the batch wrote
slice `slot`.

**The root disagrees with its own children too.** In
`gpu_resident_round.grow_tree_device_oblivious` the root is built by
`builder.enqueue_leaf(0, resident_slot=0)`, which runs
`gpu_active_rows._range_hist_atomic_kernel` against
`GpuHistogramBuilder.feat_dev`. That table **is** set, by
`GpuHistogramBuilder.set_features(tree_features)`. That kernel indexes
`var base = fid[unsafe_offset=k] * nb`, a *global* feature id. So the root's
slices are keyed globally and every later level's slices are keyed by
compacted slot.

**The precondition is reachable. Verified.**
`gpu_resident_round.oblivious_device_supported` refuses
`params.feature_fraction_bynode != 1.0` and
`params.feature_fraction_bylevel != 1.0`. It does **not** test
`params.feature_fraction`. And `sampling.select_tree_features` returns the
identity list only when `fraction >= 1.0`; below that it returns an ascending
proper subset. So a fit with `grow_policy = oblivious`, device growth, and
`feature_fraction < 1` reaches this.

**What it costs, inferred.** Two independent wrong answers compound. The batch
reads the wrong columns, features 0 through k-1 rather than the sampled ones,
so the tree can never split on a sampled feature above index k-1. And the
search reads the histogram at the wrong offset, so the bins it scans belong to
a different feature than the one it names. Every invariant in the package
passes. Nothing raises. The plane produces a fitted model.

**Why no refusal makes it unreachable.** I looked for one specifically, as the
brief asked. `oblivious_device_supported` is the gate and it lists five
refusals plus a records test, and per-tree `feature_fraction` is not among
them. `_check_device_search_supported` refuses `params.extra` settings and does
not touch it either. The identity default in the batcher constructor is
*correct* at `feature_fraction = 1.0`, which is why this has never fired, and
`feature_fraction = 1.0` is the default in the shipped CatBoost-mode set. That
is the trap. **A default is never verified by the case that motivated it.**

**Cheapest fix, and it is a one-liner in the same file.** `stage_desc_level_plan`
already receives `len(self.active)` and already has `self.active` in scope. It
can call `self.batcher[0].set_shared_features(self.active)` once per tree,
exactly where `_build_leaves_batched` already does, before the first level
commit. That is one `enqueue_copy` per tree, which on Metal is one drain, and
`stage_desc_level_plan`'s own comment already says the drains are deliberately
placed there *"before anything is in flight rather than in the middle of the
tree"*. It costs nothing at the point it would go. Not implemented by this
lane. This is an audit lane.

**Secondary note.** `enqueue_device_plan_batch_fused_subtracting`, the sibling
subtraction arm, reads the same table and has the same defect. Fixing the
feature table fixes both arms.

## Claim 2. `set_features` ends in a blocking `feat_dev.map_to_host()`. CONFIRMED, with the inertness qualification intact

**Verified.** `src/mojotrees/histogram_gpu.mojo`, `GpuHistogramBuilder.set_features`,
whose last statement is

    with self.feat_dev.map_to_host() as host:
        var dst = host.unsafe_ptr()
        for i in range(len(features)):
            dst.unsafe_store(i, Int32(features[i]))

**Verified, the inertness.** The method returns early before reaching it.

    var changed = len(features) != len(self.active)
    for i in range(len(features)):
        ...
        if not changed and self.active[i] != features[i]: changed = True
    if not changed:
        return

`select_tree_features` returns the identity list at `fraction >= 1.0`, and the
builder's `self.active` is seeded to the identity in the constructor. So at the
shipped `feature_fraction = 1.0` the first call and every call after it take
the early return, and the map never runs. Confirmed.

**Verified, the cost when it does run.** The module docstring of
`histogram_gpu.mojo` records that `map_to_host` *"copies in both directions on
every use"*, and `docs/GPU_PORTABILITY.md` section 6.1 is cited across this
package for a Metal `enqueue_copy` being a synchronous full-queue drain.
`set_features` is called once per tree from
`train_gpu._grow_tree_gpu_device_search` and once more from the round loop.

**Price, estimate.** Under `feature_fraction < 1` this is one full device wait
per tree, and it lands at the top of the tree where the queue holds the
previous tree's tail. At a host wait of about 126 us that is 100 x 126 us =
**12.6 ms** of a 17.07 s fit, **0.07 percent**, if the queue is empty. If the
queue holds a whole tree it is bounded by the tree's own device time, and the
loss is the overlap, not the wait. I cannot bound the second case from reading
alone. **Claim confirmed, cost small in the empty-queue case and unpriceable in
the full-queue case.** Note that this is also the point at which claim 1 fires,
so both are conditioned on exactly the same parameter.

---

# Part 2. The register

Twenty-nine declines. **MEASURED 9. ASSERTED 15. STALE 5.**

Five of these are on the brief's "known and being worked" list and are recorded
for completeness only. They are marked KNOWN and are excluded from the ranking.

## Group A. The symmetric device plane

### A1. Sibling subtraction in the oblivious level build. KNOWN

**Where.** `src/mojotrees/histogram_gpu.mojo`,
`enqueue_desc_level_children`, *"TWO ARMS, AND LOW LAUNCH COUNT IS NOT WHAT
DECIDES BETWEEN THEM"*.

**Declined.** Sibling subtraction, to keep the level build at two launches.

**Stated reason, as it now reads after the correction.** *"This docstring used
to say ... and to present that as the price of two launches per level. The two
are not exclusive and the trade as stated was badly priced."*

**Class.** Originally ASSERTED. The correction and the arm both landed;
`MOJOTREES_GPU_OBLIVIOUS_SUBTRACT=1` selects it and the docstring now says
*"Default off, because no benchmark has priced it."* (That quoted fragment is
still in `histogram_gpu.enqueue_desc_level_children`'s docstring, but it is no
longer where the default is stated: since 2026-08-17 the default lives in the
code comment beneath `gpu_leaf_batching.oblivious_subtract_requested`, which
reads DEFAULT ON and carries the measurement, and the predicate there is
`!= "0"`. So the quoted docstring line is stale at head and the two are worth
reading together. The conclusion of this row is unchanged, and strengthened: the
arm is not merely switchable now, it ships.)

**Price, estimate.** Histogram is 86 percent of 17.07 s, so 14.7 s. Subtraction
halves the rows read per level, so the ceiling is about **7 s** less the
subtraction arithmetic, which is exact Int32 and folded into the same two
launches. The launches it bought back are 0.49 percent of the fit at most
(Ruler A). Off by roughly two orders of magnitude.

**Survives.** No. Already corrected in source.

### A2. The last level's child histograms. KNOWN

**Where.** `src/mojotrees/gpu_resident_round.mojo`,
`OBLIVIOUS_SKIP_LAST_BUILD_VAR`, *"WHAT IT REMOVES, AND WHY THE EXISTING NOTE
UNDER-PRICED IT"*.

**Declined.** Skipping a build whose output nothing reads, on the grounds that
*"skipping it would move the cost rather than remove it"*.

**Class.** Originally ASSERTED. Now counted in source, switched, **measured, and
the default since 2026-08-17.** This line read "still default off" and quoted
*"Counted, not measured ... Nothing has timed it."*, which is superseded: the
flip comment at `gpu_resident_round.oblivious_skip_last_build_requested` records
22.76 s to 18.06 s at 799,110 x 100 x 100 trees, **1.26x**, three interleaved
round-robin cycles, rmse 2.439382420 unchanged, and the predicate is now
`!= "0"`. The launch count moves 56 to 55 at depth 6, which is the least
interesting part of it.

**Price, estimate.** `train_gpu.mojo`'s four-combination table is the arithmetic
and it is correct. Row-read child builds at depth 6 go 126 to 62. One of six
levels of accumulation removed, so 14.7 s / 6 = **2.45 s**, which agrees with
the brief's 2 to 5 s. The copy-back it has to pay instead is one grid-strided
pass over `n_active` Int32, 3.2 MB per tree, 320 MB per fit at 80 GB/s = **4
ms**. The trade is 2,450 ms against 4 ms.

**Survives.** No.

### A3. The per-level `random_strength` noise copy. KNOWN

**Where.** `src/mojotrees/gpu_resident_round.mojo`,
`OBLIVIOUS_NOISE_HOIST_VAR`, *"The count is read from the source; the time is
unmeasured, which is why this is off."*

**Class.** ASSERTED as to time when this row was written; **MEASURED as of
2026-08-17, and the result is a null.** The arm was run that day on a fit that
sets `random_strength > 0` and came in **indistinguishable** in the registered
M0 sense (2026-08-17 lane brief), which is why it is the one of the four
symmetric arms whose default did not flip. The quoted docstring at
`OBLIVIOUS_NOISE_HOIST_VAR` still says the time is unmeasured and is stale on
that point. The count, six drains per depth-6 tree, is verified from source and
is untouched by the null.

**Price, estimate.** Floor is 6 x 100 x 126 us = **76 ms**, 0.44 percent. The
real cost is the lost run-ahead, since each drain sits between a level's build
and its search, and I cannot bound that from reading. Note it fires under the
shipped CatBoost-mode default, since `params.CATBOOST_RANDOM_STRENGTH` is 1.0.

**Survives.** Partly. The decline is honest about being unmeasured. The floor is
small; the ceiling is unknown.

### A4. The wide oblivious level scan. STALE, and this is the cleanest STALE in the codebase

**Where.** `src/mojotrees/gpu_split_search.mojo`,
`oblivious_wide_scan_requested`, *"Off unless asked for, which is this package's
rule for a path no benchmark has priced, and the rule is being followed here
even though the expected win is large."*

**Declined.** A 64-thread rewrite of the `block_dim=1` oblivious level scan.

**Class.** **STALE**, and the contradiction is inside one file. Roughly 1,760
lines earlier, `wide_scan_requested` in the same file says *"`oblivious_wide_scan_requested`
is the same widening of the same `block_dim=1` scan on the oblivious plane, and
it measured **4.5 percent, resolved and bit-identical**."* One docstring records
the measurement; the other still says no benchmark has priced it, and the other
is the one the code calls.

**Price, estimate.** 4.5 percent of 17.07 s = **0.77 s**, resolved and
bit-identical, sitting behind an environment variable.

**Survives.** No. The reason is false as written.

**RESOLVED 2026-08-17, later the same day.** The arm is now the shipped default:
the predicate at `oblivious_wide_scan_requested` is `!= "0"` and the flip comment
beneath it carries both readings, 20.40 s narrow against 19.49 s wide with
disjoint ranges on a quiet box, and 22.76 s against 21.28 s in a loaded
three-cycle round robin, with rmse 2.439382420 throughout. So it is no longer
"sitting behind an environment variable"; the variable restores the narrow scan.
The STALE finding stands as a finding, because the quoted docstring paragraph
above the flip comment still reads "Off unless asked for", which is a correction
owed in that file.

### A5. `set_shared_features` never called on the oblivious plane

See Part 1, claim 1. Not a performance decline. Recorded here because it lives
in the same call chain and because it shares its precondition with claim 2.

## Group B. The leaf-wise device plane, where launches are worth something

### B1. Folding the cross-slot reduction into the record filing. ASSERTED, with a count

**Where.** `src/mojotrees/gpu_split_search.mojo`, `REDUCE_SLOT_THREADS`,
*"It is **not built here**, because the consuming kernel is in another file.
Counted, it is 30 command buffers per tree and roughly 3,000 per hundred-tree
fit."*

**Declined.** Merging `_reduce_slots_block_kernel` into `gpu_tree_tables`'s
record filing, which runs in the very next command buffer.

**Stated reason.** File ownership. *"the consuming kernel is in another file."*

**Class.** ASSERTED as to time. The launch count is MEASURED by census. The
docstring itself already names the diagnosis, *"one launch in nine for something
under a percent of the step's arithmetic"*, and names the precedent, the
partition copy-back fold that took a tree from 308 to 278 command buffers.

**Price, estimate.** This is Ruler B, not Ruler A. 3,000 launches x 15 us =
**45 ms** of a 3.659 s leaf-wise fit, **1.2 percent**. Plus whatever the extra
ordering point costs in run-ahead, which I cannot bound.

**Survives.** The *reason* does not. File ownership is not an engineering
reason and the docstring is candid that it is not. The *decision* is defensible
at 1.2 percent, and this is exactly the shape the brief warned about, a decline
justified by launch count, except here the launch count argument runs the other
way and is being declined rather than asserted.

### B2. Removing the partition's copy-back launch by per-range ping-pong. ASSERTED, with a full audit

**Where.** `src/mojotrees/gpu_active_rows.mojo`, `_copy_back_kernel`,
*"Why this launch is still here ... Removing it is a real saving and was scoped
for this lane."*

**Declined.** Making `scratch` a co-equal second row buffer with a per-range
ownership bit, removing one of three partition launches and 8 bytes per row.

**Stated reason.** *"It is not done here because the bit has to be consulted by
every reader of the permutation, and the readers are not all in this file."* Six
readers are then enumerated, and one of them, `snapshot_rows`, is correctly
identified as not a pointer swap but a gather across both buffers feeding the
host replica path whose bit-for-bit agreement is load-bearing.

**Class.** ASSERTED as to time, MEASURED as to scope. The audit is real and
the `snapshot_rows` objection is a genuine structural obstacle, not a
convenience.

**Price, estimate.** On the leaf-wise plane, about 30 splits a tree, one launch
each removed, 3,000 a fit at 15 us = **45 ms**, plus 8 bytes per row of each
window. Summing window lengths over a leaf-wise tree at 799,110 rows gives
roughly one full pass per level, call it 6 x 6.4 MB = 38 MB a tree, 3.8 GB a
fit at 80 GB/s = **48 ms**. Total roughly **90 ms** of 3.659 s, **2.5 percent**.
On the symmetric plane it is worth almost nothing, because the copy-back is
already fused into the level batch's zeroing pass for free.

**Survives.** Yes. 2.5 percent against a change that touches the host replica
equivalence path is a defensible refusal, and the note prices its own scope
honestly.

### B3. The wide leaf-wise scan. ASSERTED

**Where.** `src/mojotrees/gpu_split_search.mojo`, `wide_scan_requested`,
*"the default flips when a run says so and not before."*

**Class.** ASSERTED, and the docstring is unusually well behaved about it. It
records that the oblivious sibling measured 4.5 percent, states that the two
shapes are different workloads so the number does not transfer, and adds
*"What has changed is that 'unmeasured' is no longer the same thing as
'unpromising'."*

**Price, estimate.** The module's own figure is that a split's fixed overhead is
about 280 us and the scan is a small share of it. If the leaf-wise wide scan
were worth what the oblivious one is, 4.5 percent of 3.659 s = **0.16 s**. The
honest expectation from the docstring's own reasoning is less.

**Survives.** Yes, as a decline, and **it was then resolved on 2026-08-17 by
exactly the one interleaved pair this row asked for.** The arm is the shipped
default now: three round-robin cycles at 799,110 x 100 continuous features,
leaf-wise at the shared defaults, 100 trees, M4, narrow 3.922 / 3.874 / 3.932 s
against wide 3.220 / 3.196 / 3.231 s, disjoint ranges so M0 resolved, rmse
6.116601511 in all six runs. **1.21x**, which is four times the oblivious arm
and well above this row's "the honest expectation is less" and above the 0.16 s
priced above. The predicate at `wide_scan_requested` is `!= "0"` and the flip
comment beneath it holds the numbers. The decline was defensible and the estimate
was low, which is the useful thing to record.

### B4. K=1 speculative prebuild. ASSERTED, and probably correct

**Where.** `src/mojotrees/gpu_resident_round.mojo`, `SPECULATION_BUILD_VAR`,
*"the honest cost is 964 wasted builds per fit at the same shape ... whether
that wins is a measurement nobody has taken."*

**Class.** ASSERTED as to time. The 66.8 percent hit rate is MEASURED and
recorded in `bench/results/session3_2026-08-16/RESULTS.md`. The K=1 sufficiency
is a proof in the module docstring, not a measurement, and the proof is sound.

**Price, estimate, and it comes out negative.** 964 wasted builds against about
3,000 growth steps a fit is roughly a third more child histogram builds. If
histogram is 60 percent of the 3.659 s leaf-wise fit, that is 2.2 s, and a
third more is **+0.73 s of added work**. Against that it buys occupancy only,
because the round trips are already gone from this plane. For it to break even
the occupancy gain has to exceed 20 percent of the whole fit.

**Survives.** Yes, and strongly. This is the one ASSERTED decline in this
document whose expected value I price as clearly negative. The docstring's own
registered condition, *"the launch-shape gain has to beat the wasted work in a
whole fit, not in a phase"*, is the right test and it is likely to fail.

### B5. Batching independent trees or classes into one launch. STALE

**Where.** `src/mojotrees/gpu_resident_round.mojo`, *"Leaving the door open for
batching"*, *"A measurement this round found that batching seven multiclass
classes into one launch was indistinguishable from one at a time at 465,000
rows."*

**Class.** **STALE**, and the file says so itself. *"The precondition that
gated it, that the one-round-trip loop be measured first, is now satisfied: it
is worth 0.75 seconds and the control plane is spent."* The indistinguishable
result was taken before the per-split wait was removed, so it measured a
workload that no longer exists.

**Price.** Cannot price. The mechanism is occupancy on small leaves and there is
no occupancy instrument in this repository whose output I could read. The file
is explicit that the question *"has to be argued on occupancy, which is a kernel
question and is measured with a kernel instrument."*

**Survives.** No, as a reason. Yes, as a decision, because nothing has replaced
the withdrawn measurement.

## Group C. Data layout and traffic

### C1. Row compaction. ASSERTED, and it is the largest unpriced item in the codebase

**Where.** `src/mojotrees/gpu_active_rows.mojo`, module docstring, *"Row
compaction, and what it is a trade against"*, and `set_row_compaction`.
*"**It is off by default and it must be argued into the default by a window,
not by this paragraph.**"*

**Declined.** CatBoost's physical reorder of the binned matrix and the
quantized gradient pair at every split, so a leaf's rows are contiguous in
memory rather than only in the index.

**Stated reason.** *"It removes gather traffic and pays a physical reorder at
every split to do it, which is exactly the shape of change this repository has
been burned by: the row-tile floor raised occupancy as designed and measured 22
percent slower at 50 features. What is claimed here is arithmetic, not a
measurement."*

**Class.** ASSERTED as to time. The 56.7 percent gather fraction of the
histogram phase at 1M x 50 is MEASURED and is quoted in the same docstring.

**Price, estimate, both sides.** The reorder cost is stated in source as
`4 * L * (nf + 8)` bytes for a window of length `L`. At `nf = 100` that is
432 bytes a row. Summing `L` over a level is `n_active`, so a level moves
432 x 799,110 = **345 MB**, a depth-6 tree moves **2.07 GB**, a 100-tree fit
moves **207 GB**, and at 80 GB/s that is **2.6 s**. Against that, the histogram
phase of the symmetric fit is 86 percent of 17.07 s = 14.7 s, of which the
measured gather share is 56.7 percent = **8.3 s**. Compaction does not remove
all of that, because the bytes still have to be read, but it converts a
strided read into a contiguous one. So the plausible net is somewhere between
zero and **+5.7 s**, and the sign is more likely positive than not.

**A composition gap this lane found and the docstring does not mention.**
Verified. `gpu_leaf_batching.mojo` contains no compaction parameter at all;
the only two occurrences of the word "compact" in that file are in prose. The
oblivious level build runs entirely through `GpuLeafBatcher`. Inferred, and I
am confident but did not trace every argument. **Row compaction is maintained
on the descriptor partition path (`enqueue_partition_desc` calls
`_maintain_compaction`) but cannot be consumed by the oblivious level
histogram**, so on the symmetric plane it would pay the whole 2.6 s reorder and
collect none of the 8.3 s. It reaches `enqueue_desc_histogram` and the
range histograms, which is the leaf-wise plane, where the fit is 3.659 s and
there is less absolute time to win.

**Survives.** The caution survives. The *framing* does not. Presenting this as
one switch obscures that the arm and its beneficiary are on two different
planes, which is the same "two options presented as mutually exclusive" shape
the brief warned about, in a different dress. The third option here is to teach
`GpuLeafBatcher` the compacted read, which is the same parameter
`_range_hist_atomic_kernel` already takes.

### C2. `set_compact_flag_read`, the partition's own flag pass. ASSERTED

**Where.** `src/mojotrees/gpu_active_rows.mojo`, *"There is one further arm
inside this one ... and it is off by default like everything else here."*

**Stated reason.** Attribution, and it is a good one. *"an arm that changed
both at the same time could not say which half paid."*

**Class.** ASSERTED.

**Price, estimate.** It removes one random-access bin read and one permutation
load per row per partition. Per level that is 799,110 rows x 5 bytes = 4 MB of
random access converted to sequential, 24 MB a tree, 2.4 GB a fit. At the
strided-versus-sequential gap this is worth **tens of milliseconds**, well under
1 percent. Also inert unless C1 is on.

**Survives.** Yes. Small, and correctly sequenced behind C1.

### C3. Int16 packed gradient staging. ASSERTED, with a correctness gate

**Where.** `src/mojotrees/gpu_active_rows.mojo`, the `packed_gradients` field.
*"Off by default, and the polarity is not timidity ... this one is exact only
while every row's quantized value fits sixteen bits, a bound that fails on
small fits ... an arm that can refuse a fit does not get to refuse it by
default."*

**Class.** ASSERTED as to time, MEASURED as to the failure mode by construction.

**Price, estimate.** It halves the gradient-pair gather from 8 bytes to 4 per
row per feature visit. Per tree the accumulation is 6 levels x 799,110 rows x
100 features = 479 M visits, so 4 bytes saved per visit is 1.9 GB a tree and
192 GB a fit, **2.4 s at 80 GB/s** if the pair load were the whole cost, which
it is not, since the same pair is reused across the feature group. Divide by the
feature group width, which is 2 on Metal at 255 bins, so call it **1.2 s**
upper bound.

**Survives.** Yes as stated, and the reason given is a correctness reason rather
than a performance one, which is the right kind. But 1.2 s of a 17.07 s fit
deserves a shape gate rather than a blanket off. The overflow bound is a
property of the fit's magnitudes, which
`_quantize_grad_hess_i16_kernel` already counts, so "on where it is exact" is
available and is not taken.

### C4. `narrow_index`, Int32 index arithmetic in the histogram row loop. ASSERTED

**Where.** `src/mojotrees/gpu_active_rows.mojo`, the `narrow_index` field.
*"An arm that needs a measurement to justify it waits for the measurement."*

**Class.** ASSERTED, and the docstring is honest that it is *"not provable in
either direction, because the wide form's expensive term is loop-invariant."*

**Price.** Cannot price from reading. It is an instruction-count change inside
a loop whose bound is memory, and the brief's own settled finding is that the
scan shape was never the bottleneck. Probably sub-1 percent.

**Survives.** Yes.

### C5. The blocked bin layout. ASSERTED, and it cannot reach the shipping shape

**Where.** `src/mojotrees/gpu_active_rows.mojo`, `set_blocked_layout`.
*"Since `free_feature_group` returns 1 at `bin_cap = 256`, **the shipping
default at every headline shape this project benchmarks cannot use this layout
at all**."*

**Class.** ASSERTED, and the reachability finding is stated in source.

**Price, estimate.** Zero at the headline shape by construction. Nonzero only at
`bin_cap <= 64`, which no published shape uses. Cost if taken is 52 MB of extra
residency at 1M x 50 with G = 4.

**Survives.** Yes.

### C6. The bit-packed bin layout, and the cross product with C5. ASSERTED

**Where.** `src/mojotrees/gpu_active_rows.mojo`, `set_packed_bins`.
*"Neither arm has been measured yet; building the cross product before either
is would be three unmeasured things instead of two."*

**Class.** ASSERTED.

**Price, estimate.** *"the reference shape this project benchmarks, 1,000,000 x
50 at 255 bins, gains nothing here"*, verified from source and correct, since a
255-bin column needs eight bits and packing it is the identity. Zero at every
published shape.

**Survives.** Yes. The refusal to build the cross product before either factor
is measured is exactly rule 6 of `LANE_RULES.md` applied correctly.

### C7. Feature groups wider than the free rung. ASSERTED

**Where.** `src/mojotrees/gpu_tiling.mojo`, `free_feature_group`, *"Anything
WIDER than this returns is not free: it trades resident blocks for row-side
traffic, and that trade is UNMEASURED on every device this project runs on."*
Restated at `src/mojotrees/histogram_gpu.mojo` in the constructor.

**Class.** ASSERTED.

**Price, estimate.** At 255 bins the rule returns the Metal baseline of 2 and
the next rung costs double the threadgroup memory. The one measured point in
the repository, in `_range_hist_partial_kernel`, is group 2 against group 4 at
1.034x. 3.4 percent of 14.7 s of histogram is **0.5 s**, but see C8 for why
that number no longer describes the current kernels.

**Survives.** Yes at 255 bins.

### C8. Re-measuring group 2 against group 4 now that planes are sized to the real bin count. STALE

**Where.** `src/mojotrees/gpu_active_rows.mojo`, `_range_hist_partial_kernel`,
*"Both were taken with three `MAX_BINS`-wide planes allocated at every bin
count, which is what made group 4 nearly a wash: the traffic halving was given
back by the occupancy halving. Whether sizing the planes to the real bin count
changes that verdict is exactly the thing this parameterization makes
measurable and it has NOT been measured."*

**Class.** **STALE**, self-declared. The measurement stands but the kernel it
measured no longer exists.

**Price, estimate.** Only at `bin_cap <= 64`. If the confound removal recovers
the traffic halving that the occupancy halving was giving back, the ceiling is
the 1.39x that group 1 to group 2 delivered, but the reach is datasets binned
well below 255. **Zero at the published shapes.**

**Survives.** As a decline, yes, because of reach. As a description of what is
known, no.

### C9. The interleaved gradient plane layout. ASSERTED

**Where.** `src/mojotrees/gpu_gradient_stream.mojo`, *"The default is SPLIT:
the interleaved path is unmeasured, and this module's contract is that nothing
changes until a benchmark says it should."*

**Class.** ASSERTED.

**Price, estimate.** Low reach. The device histogram already reads a
pre-quantized *interleaved* Int32 pair by default through
`MOJOTREES_GPU_QUANTIZED_GRADS`, which is on. This switch governs a different
consumer. **Under 1 percent, and quite possibly zero on the paths that matter.**

**Survives.** Yes.

## Group D. Transfers and round trips

### D1. Making the fixed-point scale device-resident. MEASURED, and the decline is right for a structural reason

**Where.** `src/mojotrees/histogram_gpu.mojo`, `set_scale_refresh`,
*"At `R = 100` that is 100 of the fit's 200 round trips, **estimated** at 0.046
seconds against a fit **measured** at about 2.58 seconds, which is below what
this machine can resolve in one window."*

**Class.** MEASURED, with the estimate labeled as an estimate in source, which
is the standard this whole document is asking for.

**Price, estimate.** 46 ms of 2,580 ms is 1.8 percent, below the resolution
floor. And the structural argument is decisive and correct. *"the gain and the
leaf value are not homogeneous in the scale (the regularizer `lambda` is not
scaled), so a kernel handed pre-scaled gradients and told the scale is 1.0 does
not compute the same gain. The host has to know the number."*

**Survives.** Yes, completely. This is the model row.

### D2. Six table copies collapsed to one. MEASURED, as a non-saving

**Where.** `src/mojotrees/gpu_tree_tables.mojo`, the gather kernel.
*"**Six to one is not a time saving and was never measured as one.** ... Five of
these six drains found a queue that the sixth was going to drain anyway, and
draining a queue that holds nothing costs nothing."*

**Class.** MEASURED, and it is a *withdrawal* of a previously claimed saving,
which is rarer and more valuable than the saving would have been.

**Price.** Zero, correctly. The round-trip count did not move.

**Survives.** Yes.

### D3. Removing the intermediate noise list. ASSERTED, and self-limiting

**Where.** `src/mojotrees/gpu_split_search.mojo`,
`noise_stage_parallel_requested`. *"Writing the draw straight into
`noise_stage` would remove the list too, and is not done here because the
stager holds `mut self` and handing a closure a pointer into a field of it is a
bigger change than this switch is asking for. The remaining copy is a Float32
move per entry with no libm call in it, so it is not the cost the profile
pointed at."*

**Class.** ASSERTED, but with the work counted. 15.3 M draws moved to workers,
15.3 M Float32 staging copies left behind.

**Price, estimate.** 15.3 M Float32 moves per fit, sequential, one thread. At a
few nanoseconds each that is **tens of milliseconds**, against a symmetric fit
of 17.07 s. Under 0.5 percent.

**Survives.** Yes, and the arithmetic in the docstring already justifies it.

### D4. The staging ring. Not a decline

Recorded because it surfaced in the sweep. `DEFAULT_STAGING_SLOTS = 2`, so the
ring is already on. `gpu_runtime.mojo` says *"What it is worth is still
unmeasured, and a number would need the interleaved harness."* That is an
unmeasured *default*, not a decline. It also carries an important correction,
withdrawn 2026-08-16, that `enqueue_copy` from a pinned `HostBuffer` on Metal
is **asynchronous**, which is narrower than the general "a copy is a drain"
rule quoted elsewhere in the package. Anyone pricing a copy should read that
section first.

## Group E. Measured declines that stand

These are recorded so the ranking below is not mistaken for the whole picture.
The codebase gets this right more often than it gets it wrong.

### E1. The row-tile floor. MEASURED, reverted, and stay away

**Where.** `src/mojotrees/gpu_resident_round.mojo`, *"A phase-level win the fit
does not show is the row-tile floor again, which measured 22 to 36 percent
slower end to end while doing exactly what its author intended."* Also
`bench/results/PROFILE_PROTOCOL.md` line 90.

**Class.** MEASURED. Survives.

### E2. The single-block in-place partition kernel. MEASURED

**Where.** `src/mojotrees/gpu_active_rows.mojo`, module docstring. *"measured
1.00x on 256 partitions of 600 rows, 78 us each either way: the per-partition
cost there is enqueue overhead, not launch count, so it was not kept. Do not
retry either without a new reason."*

**Class.** MEASURED. Survives. Note that this is a *launch-count* decline that
came out null, which is Ruler A in miniature and predates it.

### E3. The 80-tile reduction experiment. MEASURED

**Where.** `src/mojotrees/gpu_active_rows.mojo`, *"an earlier 80-tile
experiment measured 22% slower at 50 features and 36% at 100, with 12.3 MB of
partials per node histogram linear in tile count as the registered
explanation."*

**Class.** MEASURED. Survives.

### E4. SIMD-group feature packing and moving the categorical sort scratch. MEASURED

**Where.** `src/mojotrees/gpu_split_search.mojo` module docstring. *"both were
measured on an M4 at 50000 x 100 and both came back inside noise."*

**Class.** MEASURED. Survives.

### E5. EFB on by default. MEASURED by counting

**Where.** `src/mojotrees/efb.mojo`, *"turning it on is not free on a dense
matrix ... That is 3 * n_features * n_rows cell visits, none of them dispatched
to a worker, 150,000,000 visits over 150 MB ... And on dense continuous data
the verdict is always no, so those passes buy nothing."*

**Class.** MEASURED by counting, with the eligibility argument derived from the
binning rule rather than assumed. Survives, and it even names the fix that
would let the default flip.

### E6. The CTR slot-column source rule. MEASURED, reverted

**Where.** `src/mojotrees/ctr_columns.mojo`, *"It was the `Dataset` default for
part of 2026-08-16 and was reverted the same day ... Two runs on exactly the
shape this rule selects measured nulls in both directions."*

**Class.** MEASURED. Survives.

### E7. The CPU scatter unroll. MEASURED against the reference, not the clock

**Where.** `src/mojotrees/apple_cpu_policy.mojo`, *"There is deliberately no
unroll constant here ... `DenseBin::ConstructHistogramInner` does not: its
prefetching loop and its tail loop are both scalar ... The unroll was withdrawn
rather than kept as an unattributed invention."*

**Class.** MEASURED as a provenance fact, ASSERTED as a speed fact. Withdrawing
an invention nobody attributed is correct, but "LightGBM does not do it" is not
a measurement of our loop. Low value either way. Survives on grounds of
provenance discipline rather than speed.

### E8. `MOJOTREES_PARALLEL_MIN_OPS`. MEASURED null

**Where.** `bench/results/session3_2026-08-16/RESULTS.md` via
`bench/results/INSTRUCTION_AUDIT.md` line 290. Three alternating pairs, null.

**Class.** MEASURED. Survives, and the audit calls it *"the model of what a RUN
verdict looks like"*.

### E9. Publishing a labeled CPU-versus-LightGBM number. MEASURED policy

**Where.** `bench/results/PROFILE_PROTOCOL.md`, *"The alternative, a
'provisional, CPU float vs LightGBM quantized' label, was considered and
declined, because a labelled number still gets quoted without its label."*

**Class.** A reporting decline, not a speed one. Survives.

## Group F. CPU

### F1. `MOJOTREES_CPU_CORE_POOL=performance`. KNOWN

**Where.** `src/mojotrees/apple_cpu_policy.mojo`, *"the second throws away real
cores on the strength of an assumption and no measurement here justifies it."*

**Class.** ASSERTED. On the brief's known list, since it is the four-cores-beat-ten
anomaly. Recorded for completeness. Note that the module's own text already
anticipates the heterogeneity diagnosis, *"a task landing on an efficiency core
sets the pace"*, so the pool switch was built for exactly this and left off.

### F2. `TASK_BALANCE_FACTOR = 2`. ASSERTED

**Where.** `src/mojotrees/apple_cpu_policy.mojo`, *"It is unmeasured in both
directions."*

**Class.** ASSERTED, with an unusually careful derivation showing 2 is *"a
boundary, not an optimum"*, and with the honest note that it stopped mattering
on any node large enough to row-block.

**Price.** Cannot price. It changes utilization floors on small nodes, and the
brief's own finding is that ten cores buy only 1.7x on the CPU symmetric path,
so the binding constraint is elsewhere.

**Survives.** Yes.

### F3. `MOJOTREES_CPU_FLOAT64_GATHER`. ASSERTED, but the reach is tiny

**Where.** `src/mojotrees/histogram.mojo`, `float64_gather_arm`, *"the default
is off only so that nothing published moves before it is measured."*

**Class.** ASSERTED as to time, MEASURED by counting as to work. *"At 100 active
features and group width 8 that is 13 random-access passes over both derivative
arrays per node ... Gathering once makes the other 12 passes sequential."*

**Price, estimate.** Twelve of thirteen random passes converted to sequential is
a large multiple on that term. But it fires **only** under
`derivative_precision = "float64"`, which the same module calls *"a correctness
and accuracy instrument, not a performance configuration"* and says explicitly
that *"a timing taken under it is not a timing of this package's CPU path."* So
**zero on every shipped and published shape.**

**Survives.** Yes. The right call for the wrong-sounding reason. It is not
"unmeasured", it is "unreachable from anything we publish".

### F4. The row-major view budget of 1 GiB. ASSERTED

**Where.** `src/mojotrees/binning.mojo`, `ROW_MAJOR_DEFAULT_BUDGET_MB`,
*"It is chosen, not measured, and it is chosen to admit every shape this
optimization was built for and refuse the ones that are dangerous."*

**Class.** ASSERTED, and self-labeled. Admits 1M x 50, 1M x 200 and 10M x 100;
refuses 10M x 200.

**Price, estimate.** Zero on every published shape, all of which are admitted.

**Survives.** Yes, and the asymmetry argument, that an OOM is unrecoverable and
a slow fit is not, is the right one.

### F5. `ASSUMED_L1D_BYTES`. ASSERTED

**Where.** `src/mojotrees/binning.mojo`, *"Untuned, and no measurement here
justifies a different number."*

**Class.** ASSERTED. **Cannot price** without a sweep. Feeds the CPU feature
group width, which the symmetric diagnosis already found is not where the CPU
loses.

## Group G. Policy declines

Recorded briefly. These decline the *device*, not an optimization, and each one
is a routing rule rather than a kernel.

### G1. The 250,000-row GPU floor. MEASURED, provisional

`src/mojotrees/device_policy.mojo`, *"the row floor is a deliberate provisional
setting that is 250,000 rows below the smallest shape at which the GPU has been
measured to win, taken because the loss it risks is smaller than the loss it
prevents."* MEASURED, with the full three-row table quoted in source. Survives,
and it says when it expires.

### G2. Objective scope. MEASURED absence

`src/mojotrees/device_policy.mojo`, *"Unmeasured, so declined."* Survives.

### G3. The sparse GPU crossover. ASSERTED

`src/mojotrees/gpu_sparse.mojo`, *"**The crossover is unmeasured on every
device**"*. Survives, and it declines conservatively toward the CPU.

---

# Part 3. Ranked by expected value

ASSERTED and STALE only. MEASURED declines and the five KNOWN items are
excluded. Every figure is an estimate produced by this lane and none of it was
measured here. The falsifier column is the single sentence that, if true, kills
the row.

| # | Decline | Class | Estimated value | One sentence that falsifies it |
|---|---|---|---|---|
| 1 | **A4.** Wide oblivious level scan stays off. **CLOSED 2026-08-17, it is the default now** | STALE | **0.77 s of 17.07 s (4.5 percent), resolved and bit-identical** | The 4.5 percent quoted in `wide_scan_requested` was measured on a different kernel than the one `oblivious_wide_scan_requested` selects. |
| 2 | **C1.** Row compaction | ASSERTED | **0 to +5.7 s on the symmetric plane, and probably 0 as currently plumbed** | `GpuLeafBatcher` can already read a compacted plane, so the oblivious level build does collect the gather saving. |
| 3 | **C3.** Int16 packed gradient staging | ASSERTED | **up to 1.2 s of 17.07 s** | The pair load is already fully hidden behind the bin gather, so halving it changes no wall time. |
| 4 | **B2.** Removing the partition copy-back launch | ASSERTED | **about 90 ms of 3.659 s (2.5 percent)** | `snapshot_rows` and `download_rows` cannot be made to gather across two buffers without breaking host replica bit-equivalence, which the audit already suspects. |
| 5 | **B1.** Folding the cross-slot reduction into record filing | ASSERTED | **about 45 ms of 3.659 s (1.2 percent)** | The enqueue cost at this queue depth is 6 us rather than 15, which puts it under half a percent. |
| 6 | **B3.** Wide leaf-wise scan. **CLOSED 2026-08-17, measured and now the default** | ASSERTED | **up to 0.16 s of 3.659 s**, and this estimate was **LOW by about four times**: the measured win is 1.21x, roughly 0.68 s of 3.659 s | The falsifier as written ("there is no serialization for the width to remove") was tested and refuted; the width removed more on this plane than on the oblivious one, not less. |
| 7 | **A3.** Per-level noise copy drain. **CLOSED 2026-08-17 on a MEASURED NULL, and the default stays off** | ASSERTED (KNOWN) | **floor 76 ms, ceiling unknown**; the hoist was run and came in M0 indistinguishable, so the realized value is 0 at the shape measured | The falsifier as written is now the reading that survives: the collapse bought nothing measurable, so whatever those six drains cost, removing them did not show up in wall time. |
| 8 | **C8.** Group 2 against group 4 at real bin capacity | STALE | **0 at 255 bins, up to 0.5 s at 64 bins** | No published or planned shape bins below 255. |
| 9 | **C2.** `compact_flag_read` | ASSERTED | **tens of ms, and inert without C1** | It is inert without C1, so it has no standalone value at all. |
| 10 | **B5.** Batching independent trees or classes | STALE | **cannot price** | The 465,000-row null result still holds after the round-trip removal, because dispatch was never the cost. |
| 11 | **C9.** Interleaved gradient layout | ASSERTED | **under 1 percent, probably 0** | The split layout is already what the quantized interleaved pair replaced, so this switch has no live consumer. |
| 12 | **C4.** `narrow_index` | ASSERTED | **cannot price, probably under 1 percent** | The index arithmetic is not loop-invariant on the compacted arm, which would make it a real instruction saving. |
| 13 | **C5.** Blocked bin layout | ASSERTED | **0 at every published shape** | A published shape adopts `bin_cap <= 64`. |
| 14 | **C6.** Bit-packed bin layout | ASSERTED | **0 at every published shape** | A published shape has low-cardinality columns. |
| 15 | **C7.** Feature groups wider than the free rung | ASSERTED | **0 at 255 bins** | The threadgroup budget on this device admits the next rung at 255 bins. |
| 16 | **F2.** `TASK_BALANCE_FACTOR = 2` | ASSERTED | **cannot price** | Most nodes in a deep tree are too small to row-block, so the factor still chooses the width where it matters. |
| 17 | **F3.** `MOJOTREES_CPU_FLOAT64_GATHER` | ASSERTED | **0 on every published shape** | A published number is ever taken under `derivative_precision = "float64"`. |
| 18 | **D3.** Removing the intermediate noise list | ASSERTED | **tens of ms** | The Float32 staging copy is on the critical path between a level's build and its search, not overlapped. |
| 19 | **F4.** Row-major 1 GiB budget | ASSERTED | **0 at every published shape** | A target shape exceeds 1 GiB of row-major view. |
| 20 | **F5.** `ASSUMED_L1D_BYTES` | ASSERTED | **cannot price** | The CPU feature group width is on the critical path of the symmetric CPU fit. |
| 21 | **G3.** Sparse GPU crossover | ASSERTED | **cannot price** | A sparse workload is in scope for a published number. |
| 22 | **B4.** K=1 speculative prebuild | ASSERTED | **negative, about -0.73 s** | The occupancy gain on an underfilled leaf exceeds 20 percent of the whole leaf-wise fit. |

**The top of this list is short and it should be read that way.** One STALE row
worth 0.77 s that needs no measurement to act on, and one ASSERTED row worth up
to several seconds that needs a plumbing change before it can even be measured.
Everything below rank 5 is under 1.5 percent of the arm it lives on.

---

# Part 4. What I could not price, and why

- **B5, batching independent trees or classes.** The mechanism is device
  occupancy on an underfilled leaf. There is no occupancy instrument in this
  repository whose output I could read, `create_event()` raises on this
  backend, and the source itself says the question *"has to be argued on
  occupancy, which is a kernel question and is measured with a kernel
  instrument."* I will not invent a number for it.

- **The ceiling on A3, the noise copy drains.** I can price the floor, six
  host waits per tree at about 126 us. I cannot price the lost run-ahead,
  because that depends on how much of the previous level was still in flight
  when the host reached the copy, and nothing in the repository records queue
  occupancy at that point.

- **The full-queue case of claim 2, `set_features` blocking.** Same reason. The
  wait is bounded below by 126 us and above by the device time still in the
  queue, and I have no reading of the second.

- **The descriptor partition's over-provisioned grid.**
  `gpu_active_rows.enqueue_partition_desc` launches at a grid sized for the
  whole active prefix because the window length lives on the device. The source
  says *"What the surplus costs is UNMEASURED ... no benchmark has been run."*
  Blocks that own nothing exit at their first bounds check, and I have no
  figure for what an early-exit threadgroup costs on Metal. This is not a
  decline, it is a live unpriced cost, and it is the same shape as the
  descriptor histogram's grid.

- **C4, `narrow_index`.** Instruction count inside a memory-bound loop. The
  docstring is right that it is not provable in either direction and I could not
  do better by reading.

- **F2 and F5**, the CPU scheduling constants. Both feed a width choice on the
  CPU path, and the brief's own measured finding is that ten cores buy only 1.7x
  there, so the constants are not where that path loses. Pricing them would need
  a sweep.

---

# Part 5. Patterns, for the next person who writes a decline

Five things this sweep found more than once.

**1. A launch-count decline must name its plane.** Ruler A caps the *entire*
launch budget of a symmetric tree at half a percent of the fit. Ruler B puts
the leaf-wise budget at 11 percent. Three declines in this codebase quote a
command buffer count without saying which. Two of them are on the symmetric
plane and are therefore arguing about half a percent.

**2. "Not free" is not the same as "not worth it".** A5, C1 and C3 all decline
on the grounds that the change costs something. All three then omit the other
side of the subtraction. `LANE_RULES.md` rule 6 already says this and it was
written on the same day as this audit, so the rule is younger than most of the
text it governs.

**3. Two options presented as exclusive usually are not, and the third option
is in the same file.** The fused-subtraction case is the archetype. C1 is the
same shape wearing different clothes, a switch whose beneficiary is on a
different plane from the switch, where teaching `GpuLeafBatcher` the compacted
read is the third option and is thirty lines of parameter passing.

**4. A default is verified only by the case that breaks it.** The batcher's
identity feature table is *correct* at `feature_fraction = 1.0`, which is the
shipped default, which is why a wrong answer has sat in the oblivious device
path unreported. Compare the CTR default trap of 2026-08-17, same shape.

**5. Two docstrings over one fact will drift.** A4 is one file disagreeing with
itself about whether a measurement exists, 1,760 lines apart. The example that
stood here was a live duplicate predicate over `MOJOTREES_GPU_TREE_RESIDENT`,
`gpu_tree_tables.tree_resident_requested` spelling `== "1"` where
`gpu_resident_round.resident_round_enabled` spells `!= "0"`, which
`resident_round_enabled` itself flagged as *"a live hazard"* and asked to be
deleted. **That one was CLOSED on 2026-08-17 and this paragraph said "it is
still there" until then.** `gpu_tree_tables` now owns both predicates over one
`comptime TREE_RESIDENT_VAR` and `tree_resident_requested` is a one-line alias
for the diagnostic one that reads no variable, so the two spellings cannot
disagree. Four reader bodies over that variable survive and all four agree; the
remaining cleanup is recorded in `docs/design/GROWTH_POLICY_REACH.md`. The
lesson the example was chosen for is unchanged, which is why it is corrected in
place rather than swapped for a different example.

---

## Provenance

Read-only audit, 2026-08-17. Nothing was built and nothing was run. Files read
under `src/mojotrees/`, `bindings/`, `bench/results/` and `docs/`. Every price
in this document is an estimate produced by arithmetic over figures published
elsewhere in this repository or supplied in the audit brief, and every one of
them is labeled as an estimate at the point it is made.
