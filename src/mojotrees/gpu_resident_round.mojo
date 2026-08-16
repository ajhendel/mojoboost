"""The device-owned growth loop: one host round trip per tree.

What this module is
-------------------
A second control plane for growing one tree on the GPU, built beside the one
in `train_gpu.mojo` rather than inside it. `train_gpu` gains a single call
site that routes here when the gate is on and the configuration is one this
plane can express; everything else about that file is untouched. The plan is
new plane beside old plane, measure one against the other, and only then
retire the old one. Nothing here deletes or restructures any part of the
shipping path.

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

**No speedup is claimed and none was measured.** What is claimed is a count,
and the count has since been corrected downward in ambition. The loop below
contains exactly one `synchronize` per tree against the shipping loop's
`num_leaves`, which is what was originally reported. But on Metal every
`enqueue_copy` is itself a full-queue drain in both directions, so the honest
per-tree figure for this function is sixteen host waits and not one: nine
uploads before the first split, six downloads and the `synchronize` at the
end. `grow_tree_device_resident` itemizes them. What survives intact is the
count of *round trips*, meaning points where host code reads a device answer
before it can decide what to enqueue next: this plane makes one per tree and
the shipping loop makes one per split. Whether that is worth anything is a
measurement nobody has taken.

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
  chosen. It costs one extra host synchronization per tree, inside
  `GpuActiveRows.begin_tree`, which drains the queue before it refills the row
  staging buffer. **Run and checked** against `_device_search_resident` in
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
of the batching itself, and it should not be built before the one-wait loop
has been measured.

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

from .gpu_active_rows import STEP_WORDS
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


def resident_round_requested() -> Bool:
    """`MOJOTREES_GPU_TREE_RESIDENT=1`, and off for every other value.

    One gate for the tables and for the loop that drives them, because a
    caller has no reason to want one without the other and two variables
    would only make it possible to set them inconsistently. Spelled as an
    equality against "1" so that an unset variable and a variable set to
    something unrecognized both land on the default, which is off.

    This lands as an opt-in path to be measured, not as a new default.
    """
    return getenv("MOJOTREES_GPU_TREE_RESIDENT") == "1"


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
    it lifts it only after proving the replay was complete.**
    `GpuActiveRows.enqueue_partition_desc` poisons the table on every step, so
    between the first split of a resident tree and this function every window
    accessor raises rather than returning a stale window. The
    `end_descriptor_partition` call below is what ends that, and it checks two
    things before it does: that the replay wrote a window for every node the
    tree ended with (`snap.next_node` of them), and that those windows tile
    the active prefix. See `LeafRangeTable.end_descriptor_partition` for why
    both are needed and what each one catches.

    Does the replay in fact cover every node? Yes, and the argument is short
    enough to state. `_pick_and_commit_kernel` advances `CTR_NEXT_NODE` by
    exactly two per commit and hands the two new ids to the parent it logged,
    so the finished tree's node ids are `0` (the root, whose window
    `reset_root` wrote) together with the two children of each logged commit,
    and there are `1 + 2 * commits` of them. The loop below visits every
    logged commit and `LeafRangeTable.split` writes both children of each, so
    the replay covers the tree exactly when the log holds every commit.

    That last clause is the check immediately below, and it is the one that
    has to be made here rather than in the table. The kernel stops appending
    to the log once it is full, and a *missing* commit is invisible to the
    table: `_grow_to` back-fills the gap with empty ranges, and the parent
    whose commit went missing stays a live leaf owning the rows its children
    should have, so the node count and the tiling both come out right on a
    table that is wrong. The identity `2 * len(commit_order) + 1 ==
    next_node` is what sees it, because every commit creates exactly two
    nodes; nothing downstream can reconstruct it once the log has been
    replayed and thrown away.
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
    builder.rows.ranges.end_descriptor_partition(snap.next_node)


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

    Host synchronizations, counted statically rather than measured, which is
    all this lane is allowed to do. **Every copy counts, in both directions**,
    because on Metal `enqueue_copy` is a synchronous full-queue drain in both
    directions, measured by disassembly and recorded in
    `docs/GPU_PORTABILITY.md` section 6.1. An earlier version of this list
    counted only the `synchronize` at the end and said "one"; that was a count
    of one kind of wait, not a count of the waits, and section 6.1 rule 1 says
    so in as many words.

    Per tree, inside this function, on Metal:

    - **five uploads** in `enqueue_desc_begin_tree`, which is
      `DeviceTreeTables.begin_tree(wait=False)`: the frontier, the two node
      planes, the counters and the slot pool, one copy each. The `wait=False`
      removes the explicit `synchronize` and none of the drains.
    - **four uploads** in `searcher.enqueue_frontier`, which is
      `GpuSplitSearcher._copy_tables`: the node table, the feature table, the
      allow mask and the float parameter block. One per table for the whole
      record range, not one per record, which is what that entry point buys.
    - **six downloads plus one `synchronize`** in `download_desc_tables`. The
      `synchronize` is free on Metal, since the copy before it already
      drained, and it stays because it is what keeps this correct on a
      backend where a copy really is asynchronous.

    Sixteen, then, of which nine are uploads. Not one. The one figure that is
    genuinely one is the number of *round trips*, in the sense of host code
    that reads a device answer and then decides what to enqueue next: there
    is one such point per tree and it is the download at the end. That is the
    property the plane was built for and it survives the correction; what
    does not survive is the wait count that was reported for it.

    Outside this function and unchanged by it: **two** inside
    `GpuActiveRows.begin_tree` when the tree is bagged, its explicit
    `synchronize` and the bag upload; and whatever the caller's round does,
    which is the gradient upload (two drains, one per plane, every round), a
    GOSS row and scale upload where that is on, and a custom objective's host
    callback.

    The shipping loop's count for comparison, at the same leaf budget: one
    `synchronize` per split from `download_frontier` plus one for the root,
    which is `num_leaves` in total and 31 at the default budget, and on top of
    that its own copies, four staged tables per search among them. Comparing
    31 against 16 is comparing two static counts read off the source, and
    neither is a measurement of anything.

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
    # Both read once, before the first launch. See `resident_trace_sink`.
    var trace = resident_trace_sink()
    var trace_steps = trace != "" and resident_trace_steps_requested()

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
        # Point the two scratch search records at the children's pool slots.
        # The only per-record word this loop writes on the device.
        builder.enqueue_desc_stage_search(
            searcher.node_dev, slot_cells, scratch_l, scratch_r
        )
        # Reassign the parent's rows to its two children. Three launches at a
        # grid sized for the whole active prefix.
        builder.enqueue_desc_partition(n_root if n_root > 0 else 1)
        # Accumulate the smaller child from its own rows and derive the larger
        # by subtracting inside the same kernel. Two launches: the slot
        # zeroing the atomic strategy needs, and the accumulation.
        builder.enqueue_desc_child(n_root if n_root > 0 else 1)
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
    return tree_from_snapshot(snap)
