"""The device-owned growth loop: one host round trip per tree.

What this module is
-------------------
A second control plane for growing one tree on the GPU, built beside the one
in `train_gpu.mojo` rather than inside it. `train_gpu` gains a single call
site that routes here whenever the configuration is one this plane can
express; everything else about that file is untouched. The plan was new
plane beside old plane, measure one against the other, and only then retire
the old one.

**The measurement has been taken and this is now the default.** It ran
opt-in behind `MOJOTREES_GPU_TREE_RESIDENT=1` until 2026-08-16; it now runs
unless `MOJOTREES_GPU_TREE_RESIDENT=0` says otherwise, on the three
*measured* results recorded in `bench/results/session3_2026-08-16/RESULTS.md`
and argued at `resident_round_enabled` below. What has **not** happened is
the third step: nothing here deletes or restructures any part of the
shipping path, and `_device_search_resident` is still what every refusal and
every `=0` run falls back to. Retiring it is a separate and larger change
that has to land on a green default rather than tangled with the flip.

The measurement that justifies it
---------------------------------
A Metal System Trace on an Apple M4 (`docs/METAL_TIMELINE.md`,
`bench/results/metal_timeline_2026-08-15/`) found the GPU idle for 76.5% of
the training span at 200,000 rows and 87.5% at 50,000, with the device
sitting at its maximum performance state for 77.9% of the capture, so the
idleness is a stall and not a downclock. It counted the host blocking on
94.1% of blits: 3,208 serialization points, 32.1 per round, which
independently reproduces a stage profile's count of 31 host synchronizations
per tree from a different instrument. One blocking readback measured 606
microseconds, of which 3.7 microseconds was the GPU moving bytes. Thirty-two
of those is 20.16 ms of a 23.50 ms round.

So the round is one host round trip per split, and the whole of this module
is the removal of that round trip. Compute of every kind was 22.9% of a
round, which is the ceiling on what any kernel-level change could have been
worth.

The number that matters is round trips
--------------------------------------
The figure this module is built on, and the one to quote, is the count of
**round trips**, meaning points where host code reads a device answer and then
decides what to enqueue next. This plane makes one per tree. The shipping loop
makes one per split, which is `num_leaves` and is 31 at the default budget.

That count is now backed by a clock. **Measured**
(`bench/results/session3_2026-08-16/RESULTS.md`, 1,000,000 x 50, median of
five in-process repeats, alternating processes): this plane against the
shipping loop is worth **0.75 seconds in the fast regime**, 24% of the fit,
resolved by a wide margin against arm spreads under 0.02. Removing about thirty
round trips per tree is what bought it. In a slow window both arms rose about
65% and the delta fell to 0.35, or 8%, so the effect size itself moves with
machine state: the direction is regime-independent and a single figure for the
size is not.

The other number is copies, and it is not a time
------------------------------------------------
On Metal every `enqueue_copy` is itself a full-queue drain in both directions,
**measured** by disassembly (`docs/GPU_PORTABILITY.md` section 6.1). During
the 2026-08-15 round that mechanism was turned into a **cost** model here:
every copy is a wait, so the honest per-tree figure was called sixteen host
waits rather than one, and removing copies was argued as removing waits.

**That cost model was refuted on 2026-08-16 and section 6.1.1 records the
withdrawal.** Draining a queue that holds nothing costs nothing. The copy
count went from sixteen to **three** on the default arms, thirteen fewer per
tree and roughly 1,300 fewer per fit, and the **measured** effect of all
thirteen was **0.016 seconds** against a registered prediction of 0.64, which
is not resolved under M0 and sits inside the arm's own spread. It is a null.

So the copy count is kept, and it is kept as what it honestly is: a
**portability and hazard count**, not a time estimate. It says how many places
a stale byte could hide, how many buffers must stay alive, and how much would
have to be rethought on a backend where a copy is genuinely asynchronous. It
does not predict seconds. Count round trips for that.

The thirteen, and why none of it is reverted
--------------------------------------------
The tables reset is a kernel that takes three scalars rather than five staged
uploads (`gpu_tree_tables.DeviceTreeTables.set_reset_on_device`); the download
is one copy of one device-side concatenation rather than six
(`set_packed_download`); and the searcher's four staged tables are now
`create_sub_buffer` windows onto one parent allocation, so one copy writes all
four (`gpu_split_search.GpuSplitSearcher.set_table_upload_hoisting`). What is
left is one upload, one download, and the `synchronize`, which waits on
nothing because the copy before it already drained.

**CORRECTED 2026-08-16: a copy into a pinned `HostBuffer` on Metal is ASYNCHRONOUS** (measured by execution; only a copy into an arbitrary host pointer drains). Any `synchronize` after a pinned copy is load-bearing, not redundant. See `docs/GPU_PORTABILITY.md` 6.5.
The sentence above was wrong: these destinations are pinned, so the copy did
not drain and the `synchronize` is what makes the readback correct.
`grow_tree_device_resident` itemizes all three. Every replacement has a second
arm reachable at run time, because this machine's timings drift several-fold
across time windows and only interleaved arms compare.

**The collapse from sixteen waits per tree to three is correct, it is tested,
and it is worth an unmeasurable amount of time. It is not reverted and should
not be.** It makes staleness structurally impossible in the searcher's tables
rather than merely unlikely, it removes real work, and it is the right shape.
What it is not is a speed result, and every place it was argued as one has
been relabelled rather than deleted.

Two lanes produced that thirteen and neither could see the other, so each
reported a correct intermediate figure against its own baseline: sixteen to
six, and separately four to one giving thirteen. Composed it is three. Written
out because that is the arithmetic a multi-lane round gets wrong.

The control plane is finished. Do not spend another lane on it.
--------------------------------------------------------------
The three that remain are one upload, one download and one `synchronize`,
which together are a single round trip per tree, so roughly 100 per fit at a
hundred-tree default. **Estimated** at the derived ~458 microseconds per round
trip, that whole remaining budget is at most about **0.05 seconds**. M0 cannot
resolve 0.05 seconds on this machine: the arm spreads in the session above run
from 0.02 in a quiet window to several tenths in a slow one, and the machine
drifts two- to threefold between windows.

So there is nothing left here to win, and the next reader should not re-derive
an ambition from the fact that a count went from sixteen to three. **No
further lane should be spent on this control plane.** Whatever is next is in
the kernels, in the data layout, or in what the host does between trees, and
it has to be argued against a round-trip budget that is already spent.

The shape of the loop
---------------------
Everything the host used to decide per split is decided by a kernel, and
every kernel that used to be handed the decision as a launch argument now
reads it out of a flat Int32 row in device memory (the "step descriptor",
whose layout is in `gpu_active_rows.mojo`). A whole tree's launches are
therefore enqueued before the first split has happened:

    per tree, once     seed rows, root histogram, root search, reset tables,
                       seed the root's value
    per step, x30      pick and commit          (gpu_tree_tables)
                       stage the two searches   (gpu_tree_tables)
                       partition                (gpu_active_rows, 3 launches)
                       build the smaller child  (gpu_active_rows, 2 launches)
                       search both children     (gpu_split_search, 2 launches)
                       file the two records     (gpu_tree_tables)
    per tree, once     one further pick, which ends growth rather than
                       performing it            (gpu_tree_tables)
    per tree, once     download the tables      <- the one round trip

Eleven launches a step rather than ten when the K=1 speculative prebuild is
armed; the armed schedule is written out under "K=1 speculative prebuild"
below, and it adds no round trip.

The step count is `num_leaves - 1`, because a tree of L leaves is L-1 splits.
The step *after* those is not a split and is not optional: a terminal status
is a word that some execution of the commit kernel stores, and a step that
commits stores `running`, so a tree that spends its whole budget has no step
left to notice that it has. `_growth_finished_normally` tells that story in
full; it cost this plane every tree it grew until it was found.

A step past the end of growth still runs, which is the price of not knowing
on the host how many splits there will be, and is deliberately preferred to
the alternative, which is asking. Most of what such a step enqueues is free:
the commit, the partition, the child histogram, the search staging and the
record filing all read `STEP_LIVE` first and return when it is zero. One part
of it is not free, and calling it free would be wrong. The pair of search
launches goes through `gpu_split_search._launch_search`, which knows nothing
about the descriptor and scans the histogram slots the last live step left in
the searcher's node table. So a tree that stops at seventeen leaves pays
fourteen full two-record searches over slots it has already searched. They
write only the two scratch records, which the record filing then declines to
copy anywhere, so nothing is corrupted and no answer changes; what is spent
is real work. Making it free means a descriptor-aware search launch, which
lives in a module this lane does not own.

The queue this stream goes into is 64 deep
------------------------------------------
Ten launches a step over thirty steps, plus about six per tree, is on
the order of 306 command buffers between waits, because on Metal every
`enqueue_function` becomes its own single-encoder command buffer. The queue
holds 64 of them: MAX creates it with a bare `newCommandQueue` and never
sets `maxCommandBufferCount`, which was measured by disassembling the
shipped runtime and is recorded with its consequences in
`docs/GPU_PORTABILITY.md` section 6.2. So this loop does not fly; it fills
the queue and then runs at one in and one out, which is a derived bound from
that depth and this launch count, and is not a measurement.

That is not fatal and nothing is dropped. What it means for anyone measuring
this plane is that **enqueue time has to be timed separately from wall
time**. A queue-full stall blocks the host inside `objc_msgSend`, which
every instrument this repository has counts as enqueue with no attribution,
so it will read as the device getting slower when it is the host being held.
The Metal timeline that motivated this module is the instrument that can see
it, from outside the process.

Seeing what it did
------------------
    MOJOTREES_GPU_TREE_RESIDENT_TRACE=/path/to/file   one record per tree
    MOJOTREES_GPU_TREE_RESIDENT_TRACE=1               the same, to stdout
    MOJOTREES_GPU_TREE_RESIDENT_TRACE_STEPS=1         and one per step

Off by default and off in anything anyone measures. A record holds the status
word, the counters, the whole live frontier in slot order, the slot pool and
the commit log, all of it as integers a device kernel wrote. The step form
downloads after every step, which puts the per-split wait back and is a
debugging instrument only.

This exists because there is no other way to look. The shipping loop can be
read one split at a time from the host, since it downloads one split at a
time; this plane brings home one snapshot per tree and a wrong tree in it
offers no way to ask which split went wrong. Both faults this plane shipped
with were found by reading rather than by looking, which does not scale.

What day one supports, and what falls back
------------------------------------------
The scope for the first version is dense numeric features, a single output,
and any objective. Everything else either falls back to
`_device_search_resident` by name or is called out below as untested.

Supported, with the reason it needs nothing new:

- **Every built-in objective, and custom objectives.** Gradients are computed
  and uploaded before the tree starts and are read by the histogram kernel as
  two Float32 planes; nothing in this loop knows or asks which objective
  produced them. A custom objective's host callback is one wait per *round*,
  which is outside the tree and is unaffected.
- **Missing values.** The routing rule is the feature's missing bin and the
  node's default direction, both of which the commit kernel copies into the
  descriptor and the partition reads from it.
- **`max_depth`, `min_data_in_leaf`, `num_leaves`, the positive-gain floor.**
  All four are applied by the commit kernel, which is where
  `gpu_tree_tables` put them and where its equivalence harness tests them.

Refused by name, so the caller falls back rather than getting a different
tree:

- monotone constraints, `feature_fraction_bynode`, interaction constraints,
  depth-wise growth and `TreeParams.extra`, all of which
  `gpu_tree_tables.tree_resident_supported` refuses and for the reasons it
  states there. `TreeParams.extra` covers `min_gain_to_split`,
  `max_delta_step`, `path_smooth`, extra trees, monotone penalties, feature
  contributions and the CEGB costs, none of which the device split kernel
  scores anyway.
- **categorical features.** The plumbing is complete: the commit kernel
  copies the record's category set into the descriptor and the partition
  routes by it. What is missing is any evidence that the sixteen-Int32 to
  four-UInt64 repacking in `gpu_active_rows._step_cat_word` is right, and a
  sign-extension bug there would put low-numbered categories into every set
  and would not be visible on a sparse fixture. Refused until it is tested.
- **forced splits and linear trees**, which do not reach the device split
  search at all and so cannot reach this plane.
- a searcher whose record capacity is smaller than one record per live leaf
  plus the two scratch records this loop searches into.

Reaches this plane, and how far each one has been checked:

- **bagging.** A sampled tree differs only in which rows the root owns, and
  `GpuHistogramBuilder.begin_tree` already stages the bag; every window below
  is arithmetic on `[0, n_active)` and does not care how that prefix was
  chosen. It adds one drain per tree, inside `GpuActiveRows.begin_tree`, which
  empties the queue before it refills the row staging buffer. That is an
  ordering point and one more copy on the portability count; it is not a round
  trip, since no host decision reads a device answer there, so nothing here
  predicts that it costs time. **Run and checked** against
  `_device_search_resident` in
  `tests/test_gpu_tree_resident.mojo`, tree for tree with no tolerance.
- **GOSS.** The same argument as bagging and the same code path, but nothing
  has run it. GOSS also rescales the sampled rows' gradients before upload,
  which is outside this loop entirely. Believed correct, not run.
- **multiclass.** A multiclass round grows one tree per class through
  `grow_tree_gpu`, with that class's gradients in the builder, so each class's
  tree reaches this plane through the same call site and nothing in the loop
  is class-aware. Believed correct, not run. The batched multiclass path in
  `gpu_multiclass_batch.mojo` is a different route and this lane made no claim
  about it either way.

Leaving the door open for batching
----------------------------------
A measurement this round found that batching seven multiclass classes into
one launch was indistinguishable from one at a time at 465,000 rows, which is
what the timeline predicts: dispatch is not where the round goes. But once
the per-split wait is gone, what is left on a small leaf is a kernel that
cannot fill the device, and batching independent work is what fills it.

Nothing here closes that door, and two choices were made to keep it open.
First, every per-step launch takes its varying inputs from a descriptor in
device memory rather than from launch arguments, so widening a launch to
several independent trees or classes is a matter of indexing several
descriptors rather than of restructuring the loop. Second, the loop body
below is a straight sequence of enqueues with no host state carried between
iterations at all: there is nothing in it that would have to be replicated
per batch member, because there is nothing in it. What is *not* built is any
of the batching itself. The precondition that gated it, that the one-round-trip
loop be measured first, is now satisfied: it is worth 0.75 seconds and the
control plane is spent. So batching is no longer blocked on this module, and
it is also no longer a control-plane argument. It has to be argued on
occupancy, which is a kernel question and is measured with a kernel
instrument.

K=1 speculative prebuild, and the census that gates it
------------------------------------------------------
Leaf-wise growth commits one split at a time, so each step's launches are
sized by one node and the device is underfilled. Depth-wise growth batches a
level and fills it better, and half of that advantage is not about host round
trips -- those are already gone from this plane. The proposal registered as
K2 in `bench/results/PHASE2_PREREGISTRATION.md` is to recover the other half
without changing the tree: **while step k commits, speculatively build the
child histograms of the leaf step k+1 is most likely to pick.**

**Both ship: the census, and the speculation the census gates.** The census
landed first and alone, because the registration says in as many words that
*the measured hit rate is what makes K=1 sufficient or not*, and a hit rate
of 66.8 percent at 1,000,000 x 50 is what cleared the bar it had registered
in advance. The build followed and is behind `MOJOTREES_GPU_SPECULATION=1`,
off by default; `speculative_build_enabled` says why the default reads that
way and what has to be measured before it moves. "What the speculation
actually is" below is the schedule, and it is eleven launches a step where
the unarmed loop is ten.

**The speculatable set, which is a theorem and not a census.** At step k the
only leaves whose splits are known are the ones live *before* step k. The two
children step k creates have no records until step k's own search writes them,
which is the last device work of the step, so they cannot be speculated on by
that step at all: their candidacy is established exactly when speculating on
them would stop being speculation. So the candidate set for step k's
speculation is precisely the pre-existing leaves other than the one step k is
splitting.

Over that set the ranking does not move. A commit touches four things a pick
reads -- the split leaf's frontier row, the appended row at `n_live`, the
record at `STEP_LEFT_REC` (which is the split leaf's own) and the record at
`STEP_RIGHT_REC` (which is `n_live`) -- and every one of them belongs to the
leaf being split or to a child of it. Every other live leaf keeps its slot
index, its record index, its `FRONT_DEPTH`, its `FRONT_ROW_COUNT` and
therefore its admissibility and its gain, unchanged, bit for bit. The pick is
`block.max` over gains and then `block.min` over the slots that tie, so
ordering *within* an unchanged set is unchanged including its tie resolution.

Two consequences follow, and the second is the one that matters:

1. **The best pre-existing leaf at step k+1 is exactly the top runner-up at
   step k.** Not approximately and not usually: the same gains in the same
   slot order under the same comparison.
2. **Therefore K >= 2 is provably worthless.** A second speculative candidate
   is the third-best pre-existing leaf, and the third-best can only be picked
   at step k+1 if the second-best is not, which the paragraph above says
   cannot happen. K=1 covers the whole of the speculatable set. The registered
   proposal's K is right, and it is right for a reason stronger than the census
   that was offered for it.

**What the 100-percent census actually established.** A census reporting that
the greedy pick was the top runner-up in 100 percent of 4,030 decisions is
what consequence 1 predicts *for the decisions where the pick is a
pre-existing leaf*, and it is a tautology over those. It says nothing about
how often the pick is a pre-existing leaf, which is the hit rate. The
registration already suspected the figure could not be trusted as a hit rate;
the reason it cannot is that it is measuring a different quantity, and that
quantity is provable a priori.

**The hit rate is a different number, and it is free.** The pick at step k+1
is a hit exactly when it is not one of the two nodes step k created. Node ids
are assigned by `_pick_and_commit_kernel` from a counter that starts at 1 and
advances by two per commit, so commit `j` creates nodes `2j+1` and `2j+2` and
nothing else can. The commit log `TreeTablesSnapshot.commit_order` already
comes home in the one download this plane makes, and

    hit(j) == commit_order[j] not in (2*j - 1, 2*j)     for j >= 1

is the whole census. It is host arithmetic over a list of about thirty
integers, it enqueues nothing, and it cannot perturb what it measures.
`speculation_census` is that function and `SPECULATION_CENSUS_VAR` is where
its answer goes. Because it is derived from the tree rather than from a clock,
it is reproducible: the same fit reports the same census in a fast window and
a slow one.

Two structural facts fall out of the arithmetic and both are worth knowing
before reading a census:

- **Step 0 speculates nothing.** Before the root split the only live leaf is
  the root, which is also the pick, so the candidate set is empty. The
  decision at `j == 1` therefore has no build to consume and is excluded from
  the accounting rather than counted as a wrong guess.
- **`consumed + wasted == builds` by construction**, and all three are counts
  rather than fractions, deliberately. A fit's consumed fraction is the sum of
  the counts divided by the sum of the counts; averaging per-tree fractions
  would weight a three-leaf tree like a thirty-leaf one. The trace emits
  counts and the harness does the division.

**Exactness, by construction, and the two assumptions it rests on.** A
speculatively built child histogram is bit-identical to the one the
non-speculative path would build, and the argument is that neither the inputs
nor the arithmetic can differ:

- *Same rows.* The speculative partition permutes rows inside leaf B's window
  and inside nothing else, because a partition's grid is derived from the
  descriptor's window and the live windows are disjoint. If B is committed
  later, the split it is committed on is the one already in its record -- the
  record cannot have changed, by the invariance paragraph above -- so the
  routing flag of every row is the same flag, and re-running the partition
  over an already-partitioned window moves the same rows to the same side.
  The *multiset* of rows on each side is therefore identical whether the
  partition ran once or twice.
- *Order does not reach the answer.* Accumulation is fixed-point Int32 and
  integer addition is associative and commutative, so a histogram is a
  function of the multiset of rows and not of their order. This is the same
  property `enqueue_desc_histogram` already relies on to let the atomic and
  tiled strategies agree, and the same one `enqueue_resident_subtract` relies
  on for the sibling subtraction.
- *Same scales.* `PF_G_INV` and `PF_H_INV` are recomputed per round, and
  `builder.g_scale`/`h_scale` are per round, not per step. A speculative build
  and its consuming step are inside one tree and therefore inside one round.
- *Same feature set.* Under `resident_round_supported`'s refusals a node's
  feature set is the tree's feature set (`feature_fraction_bynode` and
  interaction constraints are refused by name), so there is no per-node
  narrowing that a speculative build could get wrong.

The two named assumptions, and both were checked against the code rather than
inherited when the build landed:

**(a) The search records of leaves other than the one being split are never
written.** Checked. A step writes exactly two records, and it writes them
through one kernel: `_copy_records_kernel`, whose two destinations are
`STEP_LEFT_REC` and `STEP_RIGHT_REC`, which `_pick_and_commit_kernel` sets to
the split leaf's own record index and to `n_live`. `n_live` is the index the
appended right child takes, so it belongs to a leaf that did not exist a
launch earlier. No third record is written anywhere in the step, and the
searcher's two scratch records are outside the frontier's range entirely.
This would break the moment a step wrote a third, which is why it is stated
as a standing property rather than as a one-time reading. It is also what the
speculation's own correctness rests on twice over: it is why the runner-up's
gains cannot move, and it is why the runner-up kernel must exclude the two
slots this step's commit just wrote -- their records have *not* been written
yet at the launch that reads them, and reducing over them would read an
earlier step's memory.

**(b) Rows within a leaf's window are order-insensitive to every consumer.**
Checked, and this one is load-bearing in a way the plan did not anticipate.
A speculative partition permutes the window of a leaf that may never be
split, so the armed plane leaves the active-row buffer in a **different
permutation** from the unarmed one -- same rows in the same windows, in a
different order inside some of them. Every consumer was checked against
that: the histogram is a function of the multiset (fixed-point Int32, above);
`update_raw_device` broadcasts one leaf value over a window and never looks
at which row is where; a later partition is stable and set-preserving, and
re-running one over an already-partitioned window is the identity, which is
also why a consumed step may skip its partition entirely. Nothing in the
package indexes a row by its position within a window. A test that compared
row buffers between the two arms would fail and would be asserting an
invariant this package does not hold; `tests/test_gpu_speculation_build.mojo`
says so at its head and compares trees instead, node for node, `value` and
`split_gain` as bit patterns, over eight rounds so that a divergence visible
only in the raw scores would show.

**Where the speculation must not fold the subtraction in.** `enqueue_desc_child`
builds the smaller child into a fresh slot and derives the larger by
subtracting *in place from the parent's slot*, which destroys the parent's
histogram. A speculative step must not do that: leaf B is still live and its
histogram must survive a miss. So the speculative build runs with `do_sub`
off (`GpuActiveRows.enqueue_desc_histogram` reads that off the descriptor
target, because it is the same decision and not a second one), leaves a
whole unsubtracted child histogram behind, and the subtraction a *consumed*
step owes is done afterwards by `_spec_subtract_kernel`.

**And it needs no extra pool slot to do it**, which is a correction to the
plan this section used to carry. The plan was to build into a spare slot,
making `open_resident(num_leaves)` into `num_leaves + 1` and adding two to
`RESIDENT_SCRATCH_RECORDS`. Neither is necessary and neither shipped.
`_pick_runner_up_kernel` builds into the slot the *next commit will acquire*,
which is not a guess: the commit kernel takes the lowest free slot by
scanning `slot_owner` upward, the runner-up kernel scans the same vector the
same way and writes nothing to it, and between the two nothing acquires and
nothing releases. On a miss that slot holds a histogram nobody wants and the
next real build zeroes it before accumulating, so a miss costs work and never
correctness. The records need no addition either, because the speculation
runs no search: it produces a histogram, and the consuming step's own search
pair reads it.

**Dead steps are nearly free, which is a measured correction to what this
section used to claim.** The claim was that a speculative build adds a
partition, a histogram and a search pair to *every* step including the dead
ones, so a tree that stops at seventeen leaves of a thirty-one budget pays
fourteen builds it cannot consume. That is not what shipped.
`_pick_runner_up_kernel` returns on `STEP_LIVE` before it reads the frontier,
so a dead step publishes nothing, and the speculative partition and
histogram launched behind it read one word and return exactly as the real
ones do. A dead step therefore costs three near-empty launches and no work
at all, and the speculation runs no search pair on any step.

**Measured**, on the early-stopping fixture in
`tests/test_gpu_speculation_build.mojo` (31-leaf budget, `min_data_in_leaf`
400, four commits and twenty-six dead steps per tree): the commit-log census
reports `builds=29 wasted=27`, and the device counters report
`device_builds=2 device_consumed=2`. **Twenty-seven wasted builds predicted,
zero issued.** The prediction was the census's, and the gap is not a census
bug so much as the limit of what a commit log can see; `SpeculationCensus.
builds` now states the four ways it overcounts and names the counter that
does not.

**What the speculation actually is.** Every descriptor-aware launch used to
read one fixed buffer, `GpuActiveRows.step_dev`. There are now three, chosen
by `GpuActiveRows.set_descriptor_target`, and every kernel below the launch
is unchanged: `step_dev` is the commit descriptor, `spec_dev` is the
runner-up's publication, and `build_dev` is what the real partition and the
real child histogram read. The consuming step is the one whose `build_dev`
says `STEP_LIVE = 0`, so a hit is expressed as a step the existing kernels
decline to do -- no kernel in this package knows what a speculation is.

Per step, armed, in order, with the six that are new marked:

    pick and commit                   (gpu_tree_tables)
    consume, publish build_dev        (gpu_active_rows)          NEW
    stage the two searches            (gpu_tree_tables)
    partition          x3   against build_dev
    build the child    x2   against build_dev
    subtract a consumed prebuild      (gpu_active_rows)          NEW
    search both children x2           (gpu_split_search)
    file the two records              (gpu_tree_tables)
    publish the runner-up             (gpu_tree_tables)          NEW
    speculative partition x3 against spec_dev                    NEW
    speculative child     x2 against spec_dev                    NEW

The ordering constraint that is not implied by data flow, and the only one:
the consume kernel compares against the *previous* step's publication and
there is one buffer holding it, so it must run before this step's runner-up
kernel overwrites it.

Numerics
--------
`docs/NUMERICS.md` records that this codebase's floats contract into fused
multiply-adds and that a multiply moving relative to an add changes a result
by one ulp and then changes the next tree. Two lanes hit that independently
this round. This module contains no floating-point arithmetic whatsoever: it
enqueues launches and decodes an Int32 table. The kernels it enqueues are the
kernels the shipping path enqueues, with the same arguments arriving from a
different place, and the two changes that were made to them (an early return
for an empty tile, and reading four scalars from memory instead of from a
register) are both outside every arithmetic expression.
"""

from std.os import getenv
from max.gpu.host import DeviceBuffer

from .gpu_active_rows import (
    DESC_BUILD,
    DESC_SPEC,
    DESC_STEP,
    SPEC_STAT_BUILDS,
    SPEC_STAT_CONSUMED,
    STEP_WORDS,
    LeafRange,
)
from .gpu_split_search import (
    GpuSplitParams,
    GpuSplitSearcher,
    SplitNodeRequest,
    _launch_search,
)
from .gpu_tree_tables import (
    TREE_BUDGET_SPENT,
    TREE_NO_CANDIDATE,
    TREE_RESIDENT_OK,
    TreeTablesSnapshot,
    tree_from_snapshot,
    tree_resident_reason_name,
    tree_resident_supported,
    tree_status_name,
)
from .histogram_gpu import GpuHistogramBuilder
from .monotone import OutputBounds
from .tree import Tree, TreeParams


# --- The gate -------------------------------------------------------------


def resident_round_enabled() -> Bool:
    """On, unless `MOJOTREES_GPU_TREE_RESIDENT=0`.

    **This is the default GPU growth plane.** It was not: it landed opt-in
    behind `=1`, to be measured before it was believed. It has now been
    measured, and the polarity of this predicate is the whole content of
    that decision.

    Why the default moved
    ---------------------
    Rule S1 in `bench/results/PROFILE_PROTOCOL.md` put three conditions on
    this plane and all three are answered in
    `bench/results/session3_2026-08-16/RESULTS.md`:

    - **Trees node-identical to the host plane.** Satisfied, and by an
      assertion rather than by an argument:
      `tests/test_gpu_tree_resident.mojo` compares this plane against
      `_device_search_resident` node for node with no tolerance anywhere,
      `value` as bit patterns, over six configurations, and asserts from the
      plane's own trace that the gate opened rather than trusting that it
      did.
    - **Faster at 250,000 and at 1,000,000 rows.** *Measured*: 44 percent
      and 24 percent, five alternating pairs per shape, both resolved under
      M0 with every pair agreeing in sign.
    - **No regression at 50,000 rows.** *Measured*: 2.2x faster (median
      0.789 s against 1.724 s, arm spreads under 0.02 s). The condition
      asked only for no regression and got the largest relative win of the
      three shapes.

    Those three figures are measurements, not fits, bounds, or estimates,
    and they are the only reason this predicate reads the way it does. They
    were taken on one machine in one thermal window with the arms
    interleaved, which is the only comparison this machine supports: it
    drifts several-fold between windows, so an absolute second from that
    file means nothing and a within-window ratio means everything.

    Why the opt-out stays
    ---------------------
    `=0` forces the shipping loop (`_device_search_resident`) in the same
    binary. That is a standing requirement on this repository rather than a
    courtesy: since only interleaved arms compare here, an A/B has to be
    reachable without a rebuild, and a rebuild between arms would put a
    different compile and a different thermal state on either side of the
    comparison. It is also where a run goes that hits a fault in this plane,
    and the answer to that must not be "use the host scan".

    Spelling
    --------
    An inequality against "0", which is how `MOJOTREES_GPU_SPLIT_RESIDENT`
    is spelled for the same polarity, so that an unset variable and a
    variable set to something unrecognized both land on the **default**
    rather than on whatever a permissive parser makes of them. The name
    changed with the polarity: this used to be `resident_round_requested`,
    and a predicate that means "was not opted out of" must not keep a name
    that means "was asked for".

    One gate for the tables and for the loop that drives them, because a
    caller has no reason to want one without the other and two variables
    would only make it possible to set them inconsistently.

    The stale duplicate, which is a live hazard
    -------------------------------------------
    `gpu_tree_tables.tree_resident_requested` reads the same variable and
    still spells it `== "1"`. It is **not** consulted by anything: nothing in
    `train_gpu.mojo` calls it and only `tests/test_gpu_tree_tables.mojo`
    imports it, which is why the default could move here without moving
    there and without a test failing anywhere. That is exactly what makes it
    dangerous. Two predicates over one variable now disagree about that
    variable's default, and the next caller to reach for the one in
    `gpu_tree_tables` gets the pre-flip answer with no warning.

    It should be deleted and its callers pointed here. It was left standing
    only because the lane that flipped this default did not own that file at
    the time. Anyone who does own it: delete it, do not "fix" it, because a
    second predicate that agrees is still a second predicate that can drift.

    Enabled is not the same as taken. `resident_round_supported` still
    refuses by name every configuration this plane cannot express, and
    `train_gpu` still falls back to the shipping loop for those; the default
    flip changes which plane runs where the plane is *admissible*, and
    changes nothing about what is admissible.
    """
    return getenv("MOJOTREES_GPU_TREE_RESIDENT") != "0"


# --- The trace ------------------------------------------------------------
#
# Why this exists at all. The shipping loop can be debugged by printing,
# because it downloads the frontier after every split and a reader can watch
# a tree grow one line at a time. This plane deletes exactly that: the host
# sees one snapshot per tree and nothing in between, so a tree that comes
# home wrong offers a reader thirty-one splits' worth of state collapsed into
# one answer, with no way to ask which split went wrong short of editing the
# module. That is not a hypothetical inconvenience. The first fault this
# plane produced -- a status of `running` on every tree it grew -- was
# invisible from the outside beyond the single word in the error message, and
# was found by reading rather than by looking, which does not scale to the
# next one.
#
# So: a trace, off by default, outside the hot path, and producing text a
# test can assert on rather than only text a human can read.


comptime RESIDENT_TRACE_VAR = "MOJOTREES_GPU_TREE_RESIDENT_TRACE"
"""Where a trace goes. Unset or empty is off, which is the default and the
only state any measurement may be taken in. `1`, `stdout` or `-` print to
standard output. Anything else is taken as a file path and is **appended**
to, so a fit's trees accumulate in order and a caller that wants a fresh file
truncates it before the fit."""

comptime RESIDENT_TRACE_STEPS_VAR = "MOJOTREES_GPU_TREE_RESIDENT_TRACE_STEPS"
"""`1` adds a download and a trace record after every enqueued step, which is
the per-split host view this plane otherwise does not have.

**This changes what is being measured and it is not subtle.** A download is a
full-queue drain, so step tracing reinstates exactly the per-split
synchronization the plane exists to remove, and a timing taken with it on is
a timing of the shipping loop's wait pattern with extra kernels. It is a
debugging instrument and nothing else. It is also the only way to see a
frontier mid-tree, which is why it is here.

Ignored unless `MOJOTREES_GPU_TREE_RESIDENT_TRACE` names a sink, so one
variable turns everything off."""


def resident_trace_sink() -> String:
    """The trace destination, or the empty string when tracing is off.

    Read once per tree rather than per step. `getenv` on a miss is cheap, but
    reading a variable inside a loop that is supposed to contain no host work
    at all is the kind of thing that quietly becomes the reason a loop is
    slow, and there is nothing to gain from it: a variable does not change
    inside a fit.
    """
    return getenv(RESIDENT_TRACE_VAR)


def resident_trace_steps_requested() -> Bool:
    """Whether per-step tracing was asked for. See the variable's note about
    what it does to the wait count before turning it on."""
    return getenv(RESIDENT_TRACE_STEPS_VAR) == "1"


def _resident_trace_emit(sink: String, text: String) raises:
    """Write one trace record to the sink, or nothing when tracing is off.

    Appending rather than truncating, and one open per record rather than a
    handle held across the fit. Both are deliberate. Appending is what makes
    a whole fit's trees readable in the order they were grown. Opening per
    record is what makes the trace survive the process dying mid-fit, which
    is the case a trace is most wanted in: a plane that raises, or a kernel
    that faults, leaves the records of every tree before it on disk. The cost
    is one open per tree, which is not on any path anybody measures because
    the sink is empty on every path anybody measures.
    """
    if sink == "":
        return
    if sink == "1" or sink == "stdout" or sink == "-":
        print(text, end="")
        return
    with open(sink, "a") as handle:
        handle.write(text)


# --- The speculation census -----------------------------------------------
#
# See "K=1 speculative prebuild, and the census that gates it" in the module
# docstring for the whole argument. What lives here is the arithmetic and the
# sink; the argument for why this arithmetic is the hit rate, and why K=1 is
# provably the right K, is there and is not repeated.


comptime SPECULATION_K = 1
"""How many pre-existing leaves a speculative step would prebuild.

One, and not as a tuning choice. The module docstring proves that the best
pre-existing leaf at step k+1 is exactly the top runner-up at step k, so a
second speculative candidate would be a leaf that provably cannot be picked
next. This constant exists to be *read* -- by the census line, so that a
result file records which K produced it -- rather than to be varied. A K of
two would need a different theorem, not a different number here."""

comptime SPECULATION_CENSUS_VAR = "MOJOTREES_GPU_SPECULATION_CENSUS"
"""Where the per-tree speculation census goes. Unset or empty is off. `1`,
`stdout` or `-` print to standard output; anything else is a file path, which
is **appended** to, exactly as `RESIDENT_TRACE_VAR` is and for the same
reasons.

Its own variable rather than a mode of the trace, because the two instruments
have opposite costs. `MOJOTREES_GPU_TREE_RESIDENT_TRACE_STEPS` reinstates a
download per split and cannot be on while anything is being measured; the
census adds one host loop over about thirty integers and one file append per
tree, and perturbs nothing it measures. A caller who wants both in one file
points this at the trace's path.

When this is unset but the trace sink is set, the census line goes to the
trace instead, so that a debugging trace is never missing it. When both are
set the census line goes here only, so that a census run's output file holds
one line per tree and nothing else."""


comptime SPECULATION_BUILD_VAR = "MOJOTREES_GPU_SPECULATION"
"""`1` arms the K=1 speculative prebuild. Anything else, including unset,
leaves it off.

**Off by default, and the polarity is the whole content of that decision.**
`MOJOTREES_GPU_TREE_RESIDENT` reads the other way round -- an inequality
against "0" -- because that plane has three measured results behind it. This
one has none. What is registered before the first run, in
`bench/results/session3_2026-08-16/RESULTS.md`, is that the census measured a
**66.8 percent** hit rate at 1,000,000 rows and that the honest cost is **964
wasted builds per fit** at the same shape, each a partition and a histogram
over a child window spent on a child that is discarded. So the trade is
roughly a third more child histogram builds for a better launch shape, and
whether that wins is a measurement nobody has taken.

The condition that decides it is registered here rather than left to whoever
reads the first number: **the launch-shape gain has to beat the wasted work
in a whole fit, not in a phase.** A phase-level win the fit does not show is
the row-tile floor again, which measured 22 to 36 percent slower end to end
while doing exactly what its author intended. Both shapes are expected to
answer differently and both must be taken: 50,000 rows, where launch shape
dominates and the effect should be largest, and 1,000,000, where compute is
roughly 60 percent of the run and the wasted builds are paid in full."""


def speculative_build_enabled() -> Bool:
    """Whether the K=1 speculative prebuild is armed.

    An equality against "1", which is how an *unproven* arm is spelled in
    this repository, so that an unset variable and a variable set to
    something unrecognized both land on off. The measured arms
    (`MOJOTREES_GPU_TREE_RESIDENT`, `MOJOTREES_GPU_SPLIT_RESIDENT`) are
    spelled as inequalities against "0" instead, and the difference between
    the two spellings is exactly the difference between a default that has
    been measured and one that has not.

    Reachable at run time in one binary, in the style of
    `GpuActiveRows.set_row_unroll`, and that is a standing requirement rather
    than a courtesy: this machine drifts two- to threefold between time
    windows, so only interleaved arms compare, and a rebuild between arms
    would put a different compile and a different thermal state on either
    side of the comparison.

    Read once per tree, next to the trace and census sinks, for the reason
    stated at `resident_trace_sink`."""
    return getenv(SPECULATION_BUILD_VAR) == "1"


def speculation_census_sink() -> String:
    """The census destination, or the empty string when the census is off.

    Read once per tree, next to `resident_trace_sink`, and for the same
    reason: a variable does not change inside a fit, and reading one inside a
    loop that is supposed to contain no host work at all is how such a loop
    quietly becomes slow."""
    return getenv(SPECULATION_CENSUS_VAR)


struct SpeculationCensus(Copyable, Movable):
    """What a K=1 speculative prebuild would have built, and how much of it
    would have been consumed, for one tree.

    Counts and not fractions. A fit's consumed fraction is the sum of
    `consumed` over its trees divided by the sum of `builds`; averaging the
    per-tree fractions would weight a three-leaf tree the same as a
    thirty-leaf one, which is the wrong answer and is the mistake counts make
    impossible.

    Every field is derived from the commit log of a tree that has already been
    grown. Nothing here is a timing, an estimate or a bound: given the log,
    each number is exact, and the log is what a device kernel wrote.
    """

    var steps: Int
    """Growth steps enqueued for this tree, `num_leaves - 1`. Not the
    terminal pick step, which performs no growth and would speculate on
    nothing."""

    var commits: Int
    """Splits the device actually took. Equal to `steps` on a tree that spends
    its budget, smaller on one that ran out of admissible leaves first."""

    var dead: Int
    """`steps - commits`: steps enqueued past the end of growth. Every one of
    them issues a speculative build under an enqueue-blind design and every
    one of those builds is wasted, which is why this is reported beside
    `builds` rather than folded into it."""

    var builds: Int
    """Speculative builds a K=1 design would issue for this tree: `steps - 1`,
    every step but the first, dead steps included.

    Step 0 is excluded because before the root split the only live leaf is the
    root, which is also the pick, so the candidate set is empty and there is
    nothing to speculate on.

    An **upper bound, and on some tree shapes a very loose one.** This was
    written as "an upper bound in one narrow case"; the shipped speculation
    has a device counter beside it now and the two disagree by more than the
    narrow case allows. `gpu_tree_tables._pick_runner_up_kernel` declines to
    publish in four situations, and a commit log can see none of them:

    - the step committed nothing, which is every dead step;
    - the leaf budget is spent after this commit, so the next step cannot
      commit -- this alone costs the *last* growth step of every full-budget
      tree, making the bound loose by one on every such tree;
    - every pre-existing leaf is inadmissible;
    - the slot pool has nothing free.

    **Measured** on `tests/test_gpu_speculation_build.mojo`'s early-stopping
    fixture: this field reports 29 where the device issued **2**. On its
    full-budget 31-leaf fixture it reports 29 where the device issued 28. So
    `wasted`, which is this minus `consumed`, is an upper bound on wasted
    work and not an estimate of it, and a fit that wants the real figure
    should read `device_builds` off the census line, which is present
    whenever the speculation is armed."""

    var consumed: Int
    """Builds whose leaf was picked by the very next commit. This is the hit
    count, and `consumed / builds` is the hit rate the registration calls
    decisive."""

    var wasted: Int
    """`builds - consumed`. Real device work -- a partition and a histogram
    accumulation over a whole leaf's rows -- spent on a child that is
    discarded.

    An upper bound, and inheriting `builds`'s looseness in full: read
    `device_builds - device_consumed` off the census line for what the device
    actually threw away."""

    def __init__(out self, steps: Int, commits: Int, consumed: Int):
        self.steps = steps
        self.commits = commits
        self.dead = steps - commits
        var b = steps - 1
        if b < 0:
            b = 0
        self.builds = b
        self.consumed = consumed
        self.wasted = b - consumed

    def trace_line(self) -> String:
        """The census as one line of `key=value` pairs.

        Flat, single-line and stable, the same contract `TreeTablesSnapshot.
        trace_line` states: a harness sums these with `grep` and `awk` rather
        than by understanding them, and anything added later goes on the end
        so an existing reader keeps working. Deliberately carrying no
        `plane=device-resident` token, since that substring is what
        `tests/test_gpu_tree_resident.mojo` counts to prove the plane ran and
        a second line carrying it would double every one of those counts.
        """
        return String(
            "k=",
            SPECULATION_K,
            " steps=",
            self.steps,
            " commits=",
            self.commits,
            " dead=",
            self.dead,
            " builds=",
            self.builds,
            " consumed=",
            self.consumed,
            " wasted=",
            self.wasted,
        )


def speculation_census(
    commit_order: List[Int], steps: Int
) raises -> SpeculationCensus:
    """How much of a K=1 speculative prebuild this tree's commit log would
    have consumed.

    A pure function of the log, so it is testable without a device and
    reproducible without a clock, which is most of why the census is worth
    having at all: it is a property of the tree rather than of the machine,
    and the same fit answers the same way in a fast window and a slow one.

    The rule, whose derivation is in the module docstring. `_pick_and_commit_
    kernel` assigns node ids from a counter that `_reset_tables_kernel` starts
    at 1 and every commit advances by two, so commit `j` creates nodes
    `2j + 1` and `2j + 2` and no other commit can create them. The build
    issued during step `j - 1` targets the best leaf that was live before that
    step, so the pick at commit `j` consumes it exactly when that pick is
    **not** one of the two nodes commit `j - 1` created:

        consumed(j) == commit_order[j] not in (2j - 1, 2j)

    for `j` in `[2, commits)`. `j == 1` is excluded and is not a miss: step 0
    issues no build, so there was nothing at commit 1 to consume.

    Two checks rather than none, because a log that does not describe a legal
    leaf-wise growth would otherwise produce a plausible-looking census. A
    commit's parent must be a node that already exists, which bounds it below
    `2j + 1`; and the log cannot be longer than the steps that could have
    written it. Both raise, since a census computed from a malformed log is
    worse than no census.
    """
    if steps < 0:
        raise Error("a tree cannot enqueue a negative number of growth steps")
    var commits = len(commit_order)
    if commits > steps:
        raise Error(
            String(
                "the commit log holds ",
                commits,
                " commits but only ",
                steps,
                " growth steps were enqueued; a step commits at most once",
            )
        )
    var consumed = 0
    for j in range(commits):
        var parent = commit_order[j]
        if parent < 0 or parent >= 2 * j + 1:
            raise Error(
                String(
                    "commit ",
                    j,
                    " names node ",
                    parent,
                    ", which did not exist yet; a leaf-wise commit log must"
                    " split a node an earlier commit created",
                )
            )
        if j < 2:
            # j == 0 is the root split, which no earlier step preceded, and
            # j == 1 is preceded by step 0, which speculates on an empty
            # candidate set. Neither is a decision a build could have served.
            continue
        if parent != 2 * j - 1 and parent != 2 * j:
            consumed += 1
    return SpeculationCensus(steps, commits, consumed)


# --- Record capacity ------------------------------------------------------
#
# The searcher this plane borrows has to hold one record per live leaf, plus
# two scratch records the child searches write into. The per-leaf records are
# not an implementation detail that could be optimized away: the commit
# kernel reduces over the frontier by reading each live leaf's record, so a
# leaf without one cannot be considered for splitting.

comptime RESIDENT_SCRATCH_RECORDS = 2
"""The two records a step's search writes, which are then copied into the
frontier slots that own them. Two rather than one because a split searches
both of its children in one launch pair, and a launch's record range is
consecutive while the two children's frontier slots are not; see
`gpu_tree_tables._copy_records_kernel`."""

comptime RESIDENT_MAX_RECORDS = 512
"""Ceiling on the searcher's record count, matching the one the depth-wise
level batcher uses in `train_gpu.MAX_LEVEL_RECORDS`. It bounds `grid.y` of
the search launch and the bytes `download_words` would move."""

comptime RESIDENT_MAX_TABLE_CELLS = 1 << 20
"""Ceiling on one per-record table, matching `train_gpu.MAX_LEVEL_TABLE_CELLS`.
The searcher strides its feature and allow tables by `n_features` rather than
by a batch's slot count, so a wide dataset buys its record capacity in cells.
2^20 cells is 4 MiB per table."""


def resident_round_record_slots(num_leaves: Int, n_features: Int) -> Int:
    """Record slots a searcher must hold for this plane to run.

    One per leaf the budget allows, plus the two scratch records, clamped by
    the same two ceilings the depth-wise batcher uses. A clamp here does not
    silently shrink the frontier: `resident_round_supported` compares the
    searcher's actual capacity against the unclamped requirement and refuses
    when it falls short, so a budget too large for the ceilings falls back to
    the shipping loop instead of growing a truncated tree.
    """
    var want = num_leaves + RESIDENT_SCRATCH_RECORDS
    if want > RESIDENT_MAX_RECORDS:
        want = RESIDENT_MAX_RECORDS
    var width = n_features if n_features > 0 else 1
    var by_cells = RESIDENT_MAX_TABLE_CELLS // width
    if want > by_cells:
        want = by_cells
    if want < 2:
        want = 2
    return want


# --- Refusals -------------------------------------------------------------
#
# A predicate rather than a raise, because the caller is choosing between two
# paths that both produce a correct tree. Every refusal names itself, so a
# configuration that falls back says why rather than being quietly slow.

comptime RESIDENT_OK = 0
comptime RESIDENT_TABLES = 1
"""`tree_resident_supported` refused: monotone constraints,
`feature_fraction_bynode`, interaction constraints, depth-wise growth, or
`TreeParams.extra`. The specific reason comes back separately."""

comptime RESIDENT_CATEGORICAL = 2
"""The dataset has a categorical feature. The device path for those is built
and untested; see the module docstring."""

comptime RESIDENT_MIN_DATA = 3
"""`min_data_in_leaf < 1`. The resident frontier needs the built child to be
nonempty, which is the same precondition `_grow_tree_gpu_device_search` puts
on the shipping resident loop and for the same reason: a subtraction is only
worth taking when the child that is actually accumulated has rows."""

comptime RESIDENT_BUDGET = 4
"""`num_leaves < 2`. A one-leaf tree has no split to commit, and the commit
kernel's own precondition is a budget of at least two."""

comptime RESIDENT_RECORDS = 5
"""The searcher holds fewer records than one per live leaf plus the two
scratch records. Happens when the leaf budget or the feature count exceeds
the table ceilings above."""

comptime RESIDENT_NO_POOL = 6
"""The resident histogram pool or the device tree tables could not be
opened, which on a wide dataset is a budget answer rather than an error."""


def resident_round_reason_name(reason: Int) -> String:
    if reason == RESIDENT_OK:
        return String("ok")
    if reason == RESIDENT_TABLES:
        return String("tree tables refuse this configuration")
    if reason == RESIDENT_CATEGORICAL:
        return String("categorical features")
    if reason == RESIDENT_MIN_DATA:
        return String("min_data_in_leaf below one")
    if reason == RESIDENT_BUDGET:
        return String("leaf budget below two")
    if reason == RESIDENT_RECORDS:
        return String("searcher record capacity too small")
    if reason == RESIDENT_NO_POOL:
        return String("no resident pool or tree tables")
    return String("unknown")


def resident_round_supported(
    params: TreeParams,
    builder: GpuHistogramBuilder,
    searcher_records: Int,
) raises -> Int:
    """Whether this plane can grow this fit's trees, and why not when it
    cannot.

    The order of the tests is the order a reader wants the reason reported
    in: the tables' own refusals first, because those are the ones with
    stated causes in `gpu_tree_tables`, then the data-shape refusal, then the
    two budget refusals, then capacity.
    """
    if tree_resident_supported(params) != TREE_RESIDENT_OK:
        return RESIDENT_TABLES
    if builder.cats.any_categorical():
        return RESIDENT_CATEGORICAL
    if params.min_data_in_leaf < 1:
        return RESIDENT_MIN_DATA
    if params.num_leaves < 2:
        return RESIDENT_BUDGET
    if searcher_records < params.num_leaves + RESIDENT_SCRATCH_RECORDS:
        return RESIDENT_RECORDS
    return RESIDENT_OK


def resident_round_refusal_detail(params: TreeParams) raises -> String:
    """The tables' own reason, for a refusal this module reports as
    `RESIDENT_TABLES`. Split out so the caller can print one line that names
    both the layer that refused and what it refused."""
    return tree_resident_reason_name(tree_resident_supported(params))


def resident_round_explicitly_requested() -> Bool:
    """`MOJOTREES_GPU_TREE_RESIDENT=1` was set by hand.

    Not the gate. `resident_round_enabled` is the gate and it is on by
    default; this distinguishes the caller who *asked* from the caller who
    simply did not opt out, and the only thing that distinction is used for
    is whether a refusal says so on standard output.

    It exists because the default flip would otherwise have turned a
    diagnostic into noise. While the plane was opt-in, "you asked for the
    resident plane and did not get it" was worth one line per fit, because
    the only way to see it was to have asked. On by default, the same line
    would print on every GPU fit with monotone constraints, depth-wise
    growth, a categorical column, or any other refused shape, none of which
    asked for anything and all of which are behaving correctly. A fallback
    that is correct and expected must be silent.
    """
    return getenv("MOJOTREES_GPU_TREE_RESIDENT") == "1"


def resident_round_report_refusal(detail: String) raises:
    """Say once per fit that the plane refused, where saying it is wanted.

    Two sinks and neither is on by default:

    - standard output, only when the caller set `=1` by hand, which is the
      pre-default-flip behavior preserved exactly for the caller who still
      asks by hand;
    - the trace sink, whenever `MOJOTREES_GPU_TREE_RESIDENT_TRACE` names
      one, which is how a default-path refusal is made visible without
      printing at anybody.

    The trace record deliberately does **not** carry the
    `plane=device-resident` token that `grow_tree_device_resident` writes
    once per tree it grows. That token is what
    `tests/test_gpu_tree_resident.mojo` counts to prove the plane ran, and a
    refusal writing it would turn the negative control into a false
    positive: a refused configuration would look like a plane that executed.
    """
    var line = String("mojotrees.resident refused: ") + detail + "\n"
    if resident_round_explicitly_requested():
        print(
            "resident_round unavailable, using the device-search resident"
            " loop:",
            detail,
        )
    _resident_trace_emit(resident_trace_sink(), line)


# --- The loop -------------------------------------------------------------


def _growth_finished_normally(status: Int) -> Bool:
    """Whether the commit kernel's last status is an ordinary end of growth.

    Two of its five are: the leaf budget was spent, or no live leaf offered
    an admissible split. Those are exactly the two ways the shipping loop's
    `while n_leaves < num_leaves` and its `if len(picks) == 0: break` end a
    tree, so a tree that ends either way is finished and not faulted.

    The other three are wiring faults. `pool_full` and `overflow` mean the
    tables were sized smaller than the budget the commit kernel was given.
    `running` means the last kernel to write a status committed a split and
    nothing ran after it, which is to say the schedule ended in the middle of
    growth rather than at the end of it.

    That third one is not hypothetical and the history is the reason this
    docstring is longer than the function. The loop originally enqueued
    `num_leaves - 1` steps, which is the right number of *splits*, and no
    step after them; a tree that spent its budget therefore ended on a
    committing step, whose status is `running` because from inside that
    kernel growth has not stopped. Every tree that reached its leaf budget
    came home saying `running` and this predicate rejected all of them. The
    loop now enqueues one further step whose only possible act is to observe
    that growth is over and say so, which is what makes `running` unreachable
    rather than routine. All three stay raises, because each would otherwise
    return a tree quietly missing leaves.
    """
    return status == TREE_BUDGET_SPENT or status == TREE_NO_CANDIDATE


def _launch_child_search(
    mut searcher: GpuSplitSearcher,
    mut hist: DeviceBuffer[DType.int32],
    split_params: GpuSplitParams,
    record_base: Int,
    n_records: Int,
    widest_slots: Int,
) raises:
    """One search launch pair over `n_records` consecutive record slots,
    without restaging or recopying any per-record table.

    `GpuSplitSearcher.enqueue_frontier` is the ordinary way in and it cannot
    be used here, for a reason that is easy to miss and would have been
    silent: it ends by copying the pinned staging tables over the device
    ones, and one of those tables is the per-record histogram base that a
    device kernel wrote a few launches earlier. Copying the host's stale copy
    over it would point both child searches at whatever slots the previous
    step used. So this reaches the free function underneath instead, which
    launches and copies nothing.

    Everything it passes is either a per-fit constant or a per-tree constant
    staged once before the first split: the bin count, the feature stride,
    the widest active feature set, the monotone vector, the categorical
    parameters, and the row floor. The only per-step input is the histogram
    base, and that is already in device memory.

    This is also the reason the whole module can leave the searcher's own
    tables alone: under the refusals in `resident_round_supported`, a node's
    feature set is the tree's feature set, its allow mask is "everything",
    and its output interval is unbounded, so all three are tree-level and are
    staged once. That is holdout three from `gpu_tree_tables`, and it is what
    makes this loop reachable at all.
    """
    _launch_search(
        searcher.ctx,
        hist,
        searcher.node_dev,
        searcher.feat_dev,
        searcher.allow_dev,
        searcher.missing_dev,
        searcher.catn_dev,
        searcher.mono_dev,
        searcher.fparam_dev,
        searcher.slot_i_dev,
        searcher.slot_f_dev,
        searcher.rec_i_dev,
        searcher.rec_f_dev,
        searcher.n_bins,
        searcher.n_features * searcher.n_bins,
        searcher.n_features,
        widest_slots,
        record_base,
        n_records,
        split_params.min_data_in_leaf,
        searcher.constrained,
        split_params.cat.max_cat_to_onehot,
        split_params.cat.max_cat_threshold,
        split_params.cat.min_data_per_group,
        searcher.wide_scan,
        searcher.use_primitives,
    )


def _publish_row_ranges(
    mut builder: GpuHistogramBuilder, snap: TreeTablesSnapshot
) raises:
    """Replay the device's commits onto the host row-range table.

    Why this exists, which is the part worth reading. `GpuActiveRows` keeps a
    host-side `LeafRangeTable` mapping node id to the window of the active-row
    buffer that node owns. The shipping loop maintains it as a side effect of
    splitting: `GpuActiveRows.partition` ends with
    `self.ranges.split(parent, left, right, n_left)` on every split, so when a
    tree is finished the table holds one window per leaf and the windows tile
    the active prefix. This plane's partition is `enqueue_partition_desc`,
    which routes from the step descriptor and updates nothing on the host, so
    a tree grown here used to leave the table saying node 0 owns everything
    and no other node owns anything.

    That was documented as harmless, on the reasoning that the windows live in
    the device tree tables now and the host has no further use for them. The
    reasoning was wrong about one caller, and it is not a marginal one.
    `train_gpu`'s device-gradient round -- which is the default for every
    built-in objective without row sampling, so the common case -- advances
    the raw scores with `GpuHistogramBuilder.update_raw_device`, and that
    reads exactly this table to find which rows belong to which leaf. Handed a
    table that says node 0 owns every row, it added the *root's* value to
    every row's score instead of each row's own leaf value. The tree itself
    was correct and identical; the scores it produced were not, so round one
    was right, round two saw wrong residuals, and every tree after the first
    diverged. A comparison that stopped at one round would have seen nothing.

    So the plane has to leave the same host state the shipping loop leaves,
    and it can, exactly and without a wait, because the snapshot it already
    downloaded holds everything needed: `commit_order` is the node ids in the
    order they were split, and each parent's node row names its two children,
    whose `count` fields are the exact integer row counts the histogram
    produced. Replaying `split` in commit order is the same sequence of calls,
    with the same arguments, that the shipping loop makes one at a time.
    `LeafRangeTable.split` raises when a child already owns a window or a
    count is outside the parent's, so a replay that did not correspond to a
    real growth would be rejected rather than written.

    This is host bookkeeping only. It enqueues nothing, transfers nothing and
    waits for nothing, and it costs one list write per node.

    **It is also the only thing that lifts the table's staleness refusal, and
    it lifts it only after proving the replayed table is the device's tree.**
    `GpuActiveRows.enqueue_partition_desc` poisons the table on every step, so
    between the first split of a resident tree and this function every window
    accessor raises rather than returning a stale window. The
    `end_descriptor_partition` call below is what ends that, and what it is
    handed is the device's **own frontier**: the node id and the row window of
    every live leaf, straight out of the snapshot that was already downloaded.
    Comparing the replayed windows against those is the proof, and it costs
    nothing -- `snap.leaves` is in host memory by the time this runs, and
    until now nothing read its `row_begin`/`row_count` at all. See
    `LeafRangeTable.end_descriptor_partition` for what each of its four checks
    catches and for the one thing this comparison cannot prove.

    Two checks are made here rather than there, and both are about the *log*
    rather than about the table:

    - `2 * len(commit_order) + 1 == next_node`, first. Every commit creates
      exactly two nodes, so a log that accounts for fewer nodes than the tree
      holds is a log the kernel stopped appending to once it was full. This is
      checked before a single window is written, so a truncated log is refused
      rather than half replayed.
    - That each logged parent names two real children, in the loop. A commit
      whose children are outside the node table cannot be replayed at all.

    Why the frontier crosses as two plain lists rather than as the snapshot:
    `gpu_tree_tables` imports `gpu_active_rows` (for `LeafRange`), so
    `gpu_active_rows` cannot import it back. Node ids and `LeafRange`s are
    both already in that direction's vocabulary. The lists are `n_live` long,
    a few dozen entries at the default budget, built once per tree.
    """
    if 2 * len(snap.commit_order) + 1 != snap.next_node:
        raise Error(
            "the device commit log holds ",
            len(snap.commit_order),
            " commits, which account for ",
            2 * len(snap.commit_order) + 1,
            " nodes, but the tree ended with ",
            snap.next_node,
            "; replaying a log that does not account for the whole tree"
            " would leave the host row-range table describing a different"
            " tree from the one the device grew",
        )
    for k in range(len(snap.commit_order)):
        var parent = snap.commit_order[k]
        if parent < 0 or parent >= len(snap.nodes):
            raise Error("the device commit log names a node outside the tree")
        var left = snap.nodes[parent].left
        var right = snap.nodes[parent].right
        if left < 0 or right < 0 or left >= len(snap.nodes):
            raise Error("a device commit log entry names no children")
        _ = builder.rows.ranges.split(
            parent, left, right, snap.nodes[left].count
        )

    # The device's own frontier, as the two lists the table can be checked
    # against. `snap.leaves` is `[0, n_live)` in slot order and every row of
    # it was written by the commit kernel, so this is the device's answer to
    # "which node owns which rows" and not a second derivation of the host's.
    var leaf_nodes = List[Int](capacity=len(snap.leaves))
    var leaf_windows = List[LeafRange](capacity=len(snap.leaves))
    for i in range(len(snap.leaves)):
        leaf_nodes.append(snap.leaves[i].node)
        leaf_windows.append(
            LeafRange(snap.leaves[i].row_begin, snap.leaves[i].row_end())
        )
    builder.rows.ranges.end_descriptor_partition(
        snap.next_node, leaf_nodes, leaf_windows
    )


def grow_tree_device_resident(
    mut builder: GpuHistogramBuilder,
    mut searcher: GpuSplitSearcher,
    split_params: GpuSplitParams,
    params: TreeParams,
    tree_features: List[Int],
    n_root: Int,
) raises -> Tree:
    """Grow one tree with the device owning the frontier, the tree, and the
    slot pool, and the host waiting once.

    Preconditions the caller has already met, in the order it met them:
    gradients uploaded, `builder.set_features(tree_features)` called,
    `builder.begin_tree(bag)` called so the root owns `[0, n_root)`,
    `builder.open_resident(params.num_leaves)` answered True,
    `builder.open_resident_tables(params.num_leaves)` answered True, the
    searcher reset for this tree so that every record slot carries the tree's
    feature set and an empty allow mask, and `resident_round_supported`
    answered `RESIDENT_OK`.

    **Round trips, which is the count that predicts time. One per tree.** A
    round trip is host code that reads a device answer and then decides what
    to enqueue next, and there is exactly one such point in this function: the
    download at the end. That is the property this plane was built for, and it
    is the figure to quote. The shipping loop makes one per split, which is
    `num_leaves` and is 31 at the default budget.

    **Measured**: this plane against the shipping loop is worth **0.75
    seconds** at 1,000,000 x 50, median of five in-process repeats,
    alternating processes, resolved by a wide margin
    (`bench/results/session3_2026-08-16/RESULTS.md`). That is the fast regime,
    where it is 24% of the fit. In a slow window both arms rose about 65% and
    the delta fell to 0.35, or 8%, so the effect size itself moves with machine
    state. The direction holds in every pair taken; the size does not, and no
    single figure may be quoted for it.

    **Copies, which is a portability and hazard count and is not a time.**
    Counted statically off the source. On Metal `enqueue_copy` is a
    synchronous full-queue drain in both directions, **measured** by
    disassembly (`docs/GPU_PORTABILITY.md` section 6.1), so each copy below is
    a real ordering point and a real place a stale byte could hide. An earlier
    version of this list read that mechanism as a cost, called each copy a
    host wait, and totalled sixteen waits per tree. **That reading was refuted
    on 2026-08-16** and the withdrawal is recorded in section 6.1.1: removing
    thirteen of these per tree, roughly 1,300 per fit, **measured** 0.016
    seconds against a registered prediction of 0.64, which is a null under M0.
    Draining a queue that holds nothing costs nothing. The list below is
    therefore kept in full and read as a hazard inventory, not a wait budget.

    Per tree, inside this function, on Metal, on the default arms:

    - **no upload at all** in `enqueue_desc_begin_tree`, which is
      `DeviceTreeTables.begin_tree(wait=False)`. It used to be five copies,
      one each for the frontier, the two node planes, the counters and the
      slot pool. Every byte of those five is a constant or a function of
      three scalars, so they are now one kernel launch that takes the three
      scalars as launch arguments and writes the tables where they live. A
      launch does not drain. `set_reset_on_device(False)`, or
      `MOJOTREES_GPU_TABLE_RESET=0` for a caller that cannot reach the
      setter, puts the five copies back; that is the arm a benchmark holds
      against this one.
    - **one upload** in `searcher.enqueue_frontier`, which is
      `GpuSplitSearcher._copy_tables`. It used to be four, one each for the
      node table, the feature table, the allow mask and the float parameter
      block. The four now live as `create_sub_buffer` windows onto a single
      parent allocation, so one copy of the parent writes all four regions
      and every kernel signature and call site is unchanged. That also makes
      staleness structurally impossible rather than merely unlikely: no
      region can hold a previous call's bytes while another is fresh, which
      is a stronger property than the per-table dirty flags originally asked
      for, and is why those flags were dropped rather than added.
      `set_table_upload_hoisting(False)`, or
      `MOJOTREES_GPU_SPLIT_TABLE_PACK=0`, restores the four copies.

      Two findings from that work are worth keeping here because both
      contradict what was assumed when this list was written. The float
      parameter block is **not** fit-constant: `PF_G_INV` and `PF_H_INV` are
      reciprocals of the builder's `g_scale`/`h_scale`, which
      `upload_gradients` recomputes every round from the round's actual
      gradient magnitudes. And the node table has a **device** writer,
      `GpuTreeTables.enqueue_stage_child_search`, reached through the public
      `searcher.node_dev` field, so the host staging buffer is not a mirror
      of the device's contents and no host-side flag could have been evidence
      about them.
    - **one download plus one `synchronize`** in `download_desc_tables`. It
      used to be six downloads: a kernel now concatenates the six tables into
      one device buffer and one copy brings the concatenation home. It was one
      round trip before and it is one round trip now; what changed is that the
      round trip is made of one copy instead of six.
      `set_packed_download(False)`, or `MOJOTREES_GPU_PACKED_DOWNLOAD=0`, is
      the six-copy arm. **CORRECTED 2026-08-16: a copy into a pinned `HostBuffer` on Metal is ASYNCHRONOUS** (measured by execution; only a copy into an arbitrary host pointer drains). Any `synchronize` after a pinned copy is load-bearing, not redundant. See `docs/GPU_PORTABILITY.md` 6.5. The `synchronize` does NOT wait on
      nothing -- these six destinations are pinned buffers, so it is the only
      thing making the arm correct, and it stays because it is what keeps this
      correct on a backend where a copy really is asynchronous.

    **Three copies**, then, of which one is an upload, one a download and one
    the trailing `synchronize`, where it was sixteen of which nine were
    uploads. All thirteen removed were copies of bytes that either the device
    could have written itself or that were already sitting next to each other
    in device memory; not one of them was removed by moving less data, and the
    data moved is very nearly unchanged.

    **The collapse from sixteen to three is correct, it is tested, and it is
    worth an unmeasurable amount of time.** It is not reverted and should not
    be. It makes staleness structurally impossible in the searcher's tables
    rather than merely unlikely, it removes real work, and it is the right
    shape. It is simply not a speed result. What this round actually
    established is narrower and more useful than what it set out to
    establish: on Metal the cost is neither the bytes nor the drain, it is the
    round trip, and thirteen of these thirteen were draining a queue that held
    nothing.

    **Nothing is left here to win, and no further lane should be spent on this
    control plane.** The three that remain are a single round trip per tree,
    so roughly 100 per fit at a hundred-tree default. **Estimated** at the
    derived ~458 microseconds per round trip, the entire remaining budget is
    at most about 0.05 seconds, which M0 cannot resolve on this machine: arm
    spreads run from 0.02 in a quiet window to several tenths in a slow one,
    and the machine drifts two- to threefold between windows. A reader who
    sees sixteen go to three should not read an ambition into the next
    reduction. There is not one.

    Two lanes produced that thirteen and neither could see the other, so the
    two intermediate figures each lane reported are both correct against the
    baseline it branched from and neither is the total. The tables lane took
    sixteen to six; the search lane, branching from the same sixteen, took
    its own four to one and reported thirteen. Composed, it is three. This is
    the arithmetic that gets misreported when two lanes land in the same
    round, so it is written out rather than left to be recomputed.

    Outside this function and not in the three: the monotone vector's
    `map_to_host` went from once per tree to once per fit, and the raw-score
    update's two copies became one. **None of that is a measurement**: it is a
    static count read off the source, and under 6.1.1 a copy count is not a
    time. That a copy drains at all is **measured** in
    `docs/GPU_PORTABILITY.md` section 6.1 by disassembly. What one *round
    trip* costs is **measured** in `docs/METAL_TIMELINE.md:550` at 606
    microseconds for one blocking readback, of which 3.7 microseconds is bytes
    moving; that readback was a round trip, and the number does not transfer
    to a copy that nothing was waiting behind. What these particular copy
    removals are worth in this loop is **unmeasured** and is expected to be
    nothing, by the same argument the 0.016 second null establishes.

    Outside this function and unchanged by it, on the copy count: **two**
    inside `GpuActiveRows.begin_tree` when the tree is bagged, its explicit
    `synchronize` and the bag upload; and whatever the caller's round does,
    which is the gradient upload (two drains, one per plane, every round), a
    GOSS row and scale upload where that is on, and a custom objective's host
    callback. Of those, only the custom objective's callback is a round trip,
    because it is the only one where the host reads a device answer before it
    can decide what to enqueue next. The rest are ordering points.

    The shipping loop's count for comparison, at the same leaf budget: one
    `synchronize` per split from `download_frontier` plus one for the root,
    which is `num_leaves` in total and 31 at the default budget, and on top of
    that its own copies, four staged tables per search among them. Each of
    those 31 is a **round trip**, not merely a copy: the host downloads the
    frontier and then decides the next split from what it read. That is why
    this comparison is the one that moved a clock. Both counts are still
    static counts read off the source, but the difference between them has
    since been **measured** end to end at 0.75 seconds, and the count and the
    measurement agree about which direction is faster and about roughly thirty
    round trips per tree being what changed.

    Equivalence to the shipping loop, which outranks the count entirely. The
    claim is that this grows the same tree `_device_search_resident` grows,
    and it rests on four things, each of which is argued where it lives
    rather than here:

    1. The pick is the same pick. `gpu_tree_tables._pick_and_commit_kernel`
       reproduces `GrowthSchedule.next_leaf`'s rule exactly, including its
       strict comparison and its ties-to-the-lower-slot resolution under a
       parallel reduction, and it was tested against the host decision after
       every step across seven tree shapes with no tolerance.
    2. The partition is the same permutation. The destination of a row is a
       function of the routing flags and of the row's position in the range,
       so a grid sized for a longer range than the one it finds produces the
       same permutation; `gpu_active_rows._scatter_kernel` writes that out.
    3. The histogram is the same histogram. Accumulation is fixed-point Int32
       and integer addition is associative, so the atomic strategy this path
       always takes and the tiled strategy the shipping path may choose
       produce identical bins; the shipping code already relies on that
       between its own forty instantiations.
    4. The records are the same records. The search reads the same histogram
       cells with the same per-tree parameters, and the two scratch records
       are copied word for word into the frontier's slots with no
       arithmetic.
    5. The host state left behind is the same host state. A tree is not only
       the object returned: the trainer reads the row-range table out of the
       builder afterwards to advance its raw scores, and this plane's
       partition does not maintain it. `_publish_row_ranges` replays the
       device's commit log onto it so that it holds what the shipping loop
       would have left. That claim was missing from this list originally and
       its absence cost every tree after the first.

    Claims one through four are now checked end to end rather than argued:
    `tests/test_gpu_tree_resident.mojo` compares this function's trees against
    `_device_search_resident`'s node for node with no tolerance, over six
    configurations, and asserts on the plane's own trace that the plane is
    what produced them.

    6. **The K=1 speculative prebuild moves no bit**, when it is armed.
       Nothing in claims one through five changes: the same rows are
       partitioned by the same rule into the same windows, and the same
       multiset of rows is accumulated into the same fixed-point Int32 bins
       under the same per-round scales. What moves is *when* the work
       happens, and on a consumed step the work does not happen twice --
       `build_dev` tells the partition and the accumulation that the step is
       dead, and `gpu_active_rows._spec_subtract_kernel` does the one thing
       the prebuild could not do ahead of time. Checked, not argued:
       `tests/test_gpu_speculation_build.mojo` runs the same fit with the
       speculation off and on and compares the forests node for node, `value`
       and `split_gain` as bit patterns, over eight rounds.

       The one thing that is deliberately *not* identical is the order of
       rows inside a window whose leaf was speculated on and never split.
       Assumption (b) in the module docstring is the check that nothing
       consumes it.

    **Still one round trip per tree when armed.** The speculation adds five
    launches to a step and no download: its two counters come home only when
    a caller has named a census sink, and then only after
    `download_desc_tables` has already drained the queue.

    **Not instrumented.** `PhaseProfile` is not threaded through this loop and
    a profiled run of a fit that takes this plane will report an empty table
    rather than a wrong one. That is the same honest failure
    `_device_search_incremental` has, and it is a deliberate omission rather
    than an oversight: the profile's three phases are separated by inserting
    fences, and a loop whose entire claim is that it makes one round trip
    cannot be measured by an instrument that adds two synchronizations per
    split without measuring something other than itself. What this plane
    needs instead is the Metal timeline that motivated it, which counts
    serialization points from outside the process.

    **Traceable, though.** `MOJOTREES_GPU_TREE_RESIDENT_TRACE` writes one
    record per tree, and with `MOJOTREES_GPU_TREE_RESIDENT_TRACE_STEPS` one
    per step as well, holding the status, the counters and the whole frontier.
    That is a debugging instrument rather than a profiling one and it is off
    by default; see `RESIDENT_TRACE_VAR`. It is also what a test asserts on to
    prove this function ran at all, which matters more than it sounds: the
    caller falls back to the shipping loop for any configuration this plane
    refuses, so a comparison of two fits can agree perfectly while the plane
    under test never executed, and the first test written against this module
    did exactly that.
    """
    if n_root < 0:
        raise Error("the root row count must be nonnegative")
    if len(tree_features) < 1:
        raise Error("a tree needs at least one active feature")
    if not builder.desc_tables_open():
        raise Error("open_resident_tables has not run on this builder")
    var scratch_l = searcher.max_records - RESIDENT_SCRATCH_RECORDS
    var scratch_r = searcher.max_records - 1
    if scratch_l < params.num_leaves:
        raise Error(
            "the searcher's record capacity leaves no room for one record"
            " per leaf plus the two scratch records"
        )
    var slot_cells = 3 * builder.n_features * builder.n_bins
    var widest = len(searcher.active)
    if widest < 1:
        widest = len(tree_features)
    # All four read once, before the first launch. See `resident_trace_sink`.
    var trace = resident_trace_sink()
    var trace_steps = trace != "" and resident_trace_steps_requested()
    var census_sink = speculation_census_sink()
    var spec = speculative_build_enabled()
    # Four handles onto buffers the builder already owns, taken once.
    #
    # Handles rather than pointers because of a language constraint that has
    # shaped every seam in this plane: Mojo will not let a pointer derived
    # from one field of an object cross a call that mutably borrows a sibling
    # field, and every speculative launch needs exactly that pairing --
    # `builder.resident_tables` with `builder.rows.spec_dev`, `builder.rows`
    # with `builder.batcher[0].out_dev`. A `DeviceBuffer` is a copyable
    # handle, so copying it here ends the borrow at this line and the calls
    # below take ordinary arguments. It moves no bytes and enqueues nothing.
    #
    # Taken unconditionally rather than inside `if spec`, because a
    # conditionally initialized local is a shape this file has no reason to
    # carry for four refcount bumps per tree.
    var pool_handle = builder.batcher[0].out_dev.copy()
    var step_handle = builder.rows.step_dev.copy()
    var spec_handle = builder.rows.spec_dev.copy()
    var stats_handle = builder.rows.spec_stats_dev.copy()
    # The row bound every descriptor-driven launch is sized by, real and
    # speculative alike. An upper bound and not a length: no live window can
    # exceed the active prefix, and the geometry never reaches the answer.
    var row_bound = n_root if n_root > 0 else 1

    # --- Per-tree staging, all of it, before any split ---------------------
    #
    # The float parameter block is per record and carries this round's fixed
    # point scales, so every slot the pick kernel might read has to hold this
    # tree's block and not the previous tree's. Under the refusals the only
    # per-node value in it, the monotone output interval, is unbounded
    # everywhere, so one block is broadcast to every slot. Staging only, no
    # copy: the root's `enqueue_frontier` below carries all of it across in
    # the one table copy it makes.
    for r in range(searcher.max_records):
        searcher._stage_params(
            split_params,
            builder.g_scale,
            builder.h_scale,
            OutputBounds.unbounded(),
            r,
        )

    # The root's histogram, into pool slot 0. Slot 0 and not "whatever the
    # pool hands out", because the device pool's own acquire scans upward for
    # the lowest free slot and would have chosen 0; a root anywhere else
    # would be a frontier the commit kernel could not have produced.
    #
    # `enqueue_leaf` and not `enqueue_resident_leaf`: the latter asks the
    # *host* slot pool whether the slot is live, and on this path the host
    # pool never acquires anything, because the device's `slot_owner` vector
    # is the authority. That is the second holdout closed, and
    # `GpuHistogramBuilder.enqueue_desc_child` says what is lost with it.
    builder.enqueue_leaf(0, resident_slot=0)

    # The tables, reset to a one-leaf frontier owning every active row. This
    # has to be enqueued before the root value is seeded, since the reset
    # zeroes the node table the seed writes into.
    builder.enqueue_desc_begin_tree(n_root)

    # The root's search, into record 0, which is the record frontier slot 0
    # reads. This is the one call that copies the searcher's per-record
    # tables to the device, and it is why the staging above had to come
    # first. An empty feature list leaves each record on the tree-level set
    # the caller broadcast, which is what every node uses under these
    # refusals.
    var root_batch = List[SplitNodeRequest]()
    root_batch.append(
        SplitNodeRequest(
            0, List[Int](), List[Bool](), OutputBounds.unbounded()
        )
    )
    searcher.enqueue_frontier(
        builder.batcher[0].out_dev,
        root_batch,
        split_params,
        builder.g_scale,
        builder.h_scale,
    )
    # The root's own Newton value, which the host path writes as
    # `tree.value[root] = root_rec.parent_value` once the record is home. Here
    # the record never comes home, so a kernel copies it where it is.
    builder.enqueue_desc_seed_root(searcher.rec_f_dev, 0)

    # --- Growth ------------------------------------------------------------
    #
    # `num_leaves - 1` steps, because a tree of L leaves is L-1 splits, and
    # every step is enqueued whether or not growth has already stopped. A
    # stopped step's kernels mostly read one word and return, with the search
    # pair the exception the module docstring names; see there for why that
    # is preferred to asking the host how many splits there will be, which is
    # the question that costs 606 microseconds. The step that *ends* growth is
    # below the loop rather than in it.
    #
    # There is no host state in this loop. Nothing is read, nothing is
    # accumulated, and the body would be identical for step 1 and step 30.
    # That is what would make widening it to several independent trees a
    # matter of indexing rather than of restructuring, and it is the door the
    # coordinator asked be left open. The step trace is the one thing in the
    # body that reads the device back, and it is off unless a variable turns
    # it on, precisely so that this stays true of every run that is not being
    # debugged.
    #
    # When the K=1 speculative prebuild is armed the body gains five launches
    # and one descriptor, and the whole of the addition is written out below
    # in place rather than split into a helper, because the *order* of the
    # eleven launches is the only thing keeping it correct and an order is
    # read, not called.
    if spec:
        # `spec_dev` has to start dead: step 0's consume kernel reads it
        # before any runner-up kernel has written it. The counters are zeroed
        # with it because a fit's hit rate is a sum over trees of counts, and
        # a counter carried across trees would be the wrong denominator.
        builder.rows.enqueue_spec_reset()
    for step in range(params.num_leaves - 1):
        # Pick the leaf, commit the split, write the tree, move the slot
        # pool, and publish the step descriptor. One block, one launch.
        builder.enqueue_desc_step(
            searcher.rec_i_dev,
            searcher.rec_f_dev,
            params.num_leaves,
            params.max_depth,
            params.min_data_in_leaf,
        )
        if spec:
            # Was the split this step just committed the split the previous
            # step prebuilt? The answer goes into `build_dev`, which the
            # partition and the child histogram below read instead of
            # `step_dev`: on a hit it says the step is dead and both of them
            # return at their first word. This is where `SPEC_STAT_CONSUMED`
            # moves, and it is the only place it moves.
            #
            # It has to run here: after the commit that wrote `step_dev`, and
            # before this step's own runner-up kernel overwrites `spec_dev`
            # with a publication for the step after this one.
            builder.rows.enqueue_spec_consume()
            builder.rows.set_descriptor_target(DESC_BUILD)
        # Point the two scratch search records at the children's pool slots.
        # The only per-record word this loop writes on the device. Reads
        # `step_dev` whatever the speculation decided, because a consumed step
        # still searches both of its children and still files both records.
        builder.enqueue_desc_stage_search(
            searcher.node_dev, slot_cells, scratch_l, scratch_r
        )
        # Reassign the parent's rows to its two children. Three launches at a
        # grid sized for the whole active prefix. Skipped on a consumed step,
        # where the previous step's speculative partition already produced
        # this exact permutation.
        builder.enqueue_desc_partition(row_bound)
        # Accumulate the smaller child from its own rows and derive the larger
        # by subtracting inside the same kernel. Two launches: the slot
        # zeroing the atomic strategy needs, and the accumulation. Both
        # skipped on a consumed step.
        builder.enqueue_desc_child(row_bound)
        if spec:
            builder.rows.set_descriptor_target(DESC_STEP)
            # The one thing a prebuild could not do ahead of time: the
            # sibling subtraction. It is deliberately not folded into the
            # speculative accumulation, because the leaf being speculated on
            # is still live and deriving its larger child in place would
            # destroy the histogram the next pick reads on a miss. On a miss
            # this launch reads one word and returns.
            builder.rows.enqueue_spec_subtract(pool_handle, slot_cells)
        # Search both children in one launch pair, into the scratch records.
        _launch_child_search(
            searcher,
            builder.batcher[0].out_dev,
            split_params,
            scratch_l,
            RESIDENT_SCRATCH_RECORDS,
            widest,
        )
        # File the two answers in the frontier slots that own them.
        builder.enqueue_desc_copy_records(
            searcher.rec_i_dev, searcher.rec_f_dev, scratch_l, scratch_r
        )
        if spec:
            # Name the leaf the *next* step is most likely to pick, and build
            # its smaller child now. The runner-up kernel excludes the two
            # slots this step's commit just wrote, which is a correctness
            # requirement and not a refinement: their records have not been
            # written yet at the launch that reads them, so including them
            # would reduce over an earlier step's memory.
            #
            # The two launches after it are the ordinary partition and the
            # ordinary child build, aimed at a different descriptor. Nothing
            # about either kernel is speculation-aware, which is the whole
            # reason the descriptor is a table in device memory.
            builder.resident_tables[0].enqueue_runner_up(
                searcher.rec_i_dev,
                searcher.rec_f_dev,
                step_handle,
                spec_handle,
                stats_handle,
                params.num_leaves,
                params.max_depth,
                params.min_data_in_leaf,
            )
            builder.rows.set_descriptor_target(DESC_SPEC)
            builder.enqueue_desc_partition(row_bound)
            builder.enqueue_desc_child(row_bound)
            builder.rows.set_descriptor_target(DESC_STEP)
        if trace_steps:
            # A download inside the loop, which is the wait this plane exists
            # to remove, taken deliberately and only when asked for. See
            # `RESIDENT_TRACE_STEPS_VAR`.
            var mid = builder.download_desc_tables()
            _resident_trace_emit(
                trace,
                String(
                    "mojotrees.resident step=",
                    step,
                    " ",
                    mid.trace_line(),
                    "\n",
                    mid.describe(),
                ),
            )

    # --- The step that ends growth rather than performing it ---------------
    #
    # One more pick-and-commit launch than there are splits, and the whole
    # reason for it is that a terminal status is *written by a step*, not by
    # the loop ending.
    #
    # The commit kernel's five statuses are `running`, `budget_spent`,
    # `no_candidate`, `pool_full` and `overflow`, and every one of them is a
    # word some execution of that kernel stored. A committing step stores
    # `running`, because from inside the kernel that is what has just
    # happened: a split was taken and the tree may take another. So a tree
    # that spends its whole budget has `num_leaves - 1` committing steps, the
    # last of which stores `running`, and if nothing runs after it the tables
    # come home saying growth is still in progress. That is not a wiring
    # mistake that produced a wrong tree; it is a correct tree with no full
    # stop on the end of it, and the download's check could not tell the two
    # apart. It is what made this plane fail on every tree that reached its
    # leaf budget, which at any useful data size is every tree.
    #
    # The fix is the step that would have run next. It cannot commit
    # anything, and that is worth arguing rather than asserting, because a
    # step that could commit would grow a tree of `num_leaves + 1` leaves:
    #
    # - If every earlier step committed, `n_live == num_leaves`, and the
    #   kernel's first phase returns on the budget before it reads a record.
    # - Otherwise some earlier step found no admissible leaf and stored
    #   `no_candidate`, which also wrote `STEP_LIVE = 0`. Every kernel after
    #   it in that step reads that word first and returns, so the frontier
    #   and the records it was decided from are exactly what they were, and
    #   this step reduces over the same numbers and reaches the same answer.
    #   The reduction is over device memory nothing has written since, and it
    #   is integer comparison of Float32 words, so "the same answer" is bit
    #   equality and not a probable outcome.
    #
    # The cost is one launch of one threadgroup per tree, and no wait: this
    # goes into the same queue as everything above it.
    builder.enqueue_desc_step(
        searcher.rec_i_dev,
        searcher.rec_f_dev,
        params.num_leaves,
        params.max_depth,
        params.min_data_in_leaf,
    )

    # --- The one round trip ------------------------------------------------
    var snap = builder.download_desc_tables()
    _resident_trace_emit(
        trace,
        String(
            "mojotrees.resident plane=device-resident ",
            snap.trace_line(),
            " budget=",
            params.num_leaves,
            " root_rows=",
            n_root,
            " steps=",
            params.num_leaves - 1,
            "\n",
            snap.describe(),
        ),
    )
    # Every check the shipping path made per split, made once here instead:
    # no two live leaves share a node id, a histogram slot, or a record slot;
    # the slot pool and the frontier agree about every owner; the live
    # windows do not overlap; and the node counter and the leaf count moved
    # together. That is `LeafFrontier.check_invariants` and
    # `HistogramSlotPool.check_live` combined, at one thirty-first of the
    # frequency and at the only point where checking is free.
    snap.check_invariants()
    if not _growth_finished_normally(snap.status):
        raise Error(
            String(
                "the device-owned tree stopped abnormally: ",
                tree_status_name(snap.status),
                "; ",
                snap.trace_line(),
                ". Set ",
                RESIDENT_TRACE_VAR,
                " to a path and ",
                RESIDENT_TRACE_STEPS_VAR,
                "=1 for the frontier at every step.",
            )
        )
    # Three numbers the loop above knows and the tables cannot check for
    # themselves, because `check_invariants` sees a snapshot and not the
    # schedule that produced it. Each one is a way the step count and the
    # growth could disagree without any table being malformed, which is
    # precisely the class of fault that produced a tree missing its last
    # leaves and reported nothing.
    if snap.commits != snap.n_live - 1:
        raise Error(
            String(
                "the device-owned tree committed ",
                snap.commits,
                " splits but holds ",
                snap.n_live,
                " leaves; a tree of L leaves is L-1 splits",
            )
        )
    if snap.n_live > params.num_leaves:
        raise Error(
            String(
                "the device-owned tree grew ",
                snap.n_live,
                " leaves against a budget of ",
                params.num_leaves,
            )
        )
    if snap.status == TREE_BUDGET_SPENT and snap.n_live != params.num_leaves:
        raise Error(
            String(
                "the device-owned tree reports a spent budget at ",
                snap.n_live,
                " leaves of ",
                params.num_leaves,
            )
        )
    # The host state the rest of the trainer reads back out of the builder.
    # Not optional and not cosmetic; see `_publish_row_ranges`.
    _publish_row_ranges(builder, snap)
    # What a K=1 speculative prebuild would have consumed on this tree, from
    # the commit log that is already in host memory. One loop over about
    # thirty integers, no launch, no transfer, and no effect on anything above
    # it; see "K=1 speculative prebuild, and the census that gates it".
    #
    # Computed only when somewhere wants it. The arithmetic is negligible but
    # the emission is a file open, and this plane's rule is that an instrument
    # nobody asked for costs nothing at all.
    if census_sink != "" or trace != "":
        var census = speculation_census(
            snap.commit_order, params.num_leaves - 1
        )
        var line = census.trace_line()
        if spec:
            # What the device actually did, beside what the commit log says it
            # would have done. Two independent instruments over one tree, and
            # the point of carrying both is that they can disagree: the host
            # census counts a build on every step but the first, because a
            # commit log cannot see a step whose pre-existing leaves were all
            # inadmissible or whose pool had nothing free, while
            # `_pick_runner_up_kernel` declines on exactly those and says so.
            # So `device_builds <= builds` always, and the gap is the census's
            # own overcount made visible rather than argued.
            #
            # `device_consumed` and `consumed` are expected to be **equal**,
            # and that is a theorem rather than a hope. A commit consumes
            # under the host rule exactly when the leaf it splits predates the
            # previous commit; it consumes under the device rule exactly when
            # that leaf is the one the runner-up named. Those coincide because
            # admissibility and gain are invariant for a leaf that was not
            # touched, so a pre-existing leaf that wins the pick at step k+1
            # was also the best pre-existing leaf at step k. An inequality
            # between them is therefore a fault report, which is what makes
            # printing both worth the transfer.
            #
            # The transfer: one copy of two Int32 and one `synchronize`, after
            # the queue has already drained at `download_desc_tables`. It
            # happens only because an instrument was named, so no measured run
            # pays it.
            var stats = builder.rows.download_spec_stats()
            line += String(
                " spec=on device_builds=",
                stats[SPEC_STAT_BUILDS],
                " device_consumed=",
                stats[SPEC_STAT_CONSUMED],
            )
        else:
            line += String(" spec=off")
        # The census's own sink when it has one, so that a census run's file
        # holds one line per tree and nothing else; the trace otherwise, so
        # that a debugging trace is never missing it. See
        # `SPECULATION_CENSUS_VAR`.
        var sink = census_sink if census_sink != "" else trace
        _resident_trace_emit(
            sink,
            String(
                "mojotrees.speculation ",
                line,
                " root_rows=",
                n_root,
                "\n",
            ),
        )
    return tree_from_snapshot(snap)
