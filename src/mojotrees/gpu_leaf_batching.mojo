"""Several leaves' histograms in one GPU launch.

`GpuActiveRows.enqueue_range_histogram` builds one node's histogram per
launch, over a grid of `(n_active_features, n_tiles)` threadgroups sized from
that node's own row count. That is right for the root and for the top of a
tree and wrong for the bottom of one: a leaf holding 400 rows out of a
million produces one row tile, so its launch creates `n_slots` threadgroups
against a device that wants hundreds, and a whole boosting round's tail is
made of such leaves. This module builds several leaves in one launch instead,
so the grid is filled by the batch rather than by any single leaf in it.

The packed grid
---------------
A batch is a list of `LeafWorkItem`s (see gpu_frontier.mojo). Each item
carries its own row window, its own rows-per-tile, and therefore its own tile
count, and the tiles of all the items are concatenated into one flat tile
axis. The launch is two dimensional:

    grid.x = n_slots            the active feature slot, as today
    grid.y = total_tiles        sum over the batch of each item's tiles

and a threadgroup finds which item it belongs to by binary searching the
items' tile offsets, which the host has already prefix summed. Two properties
follow, and both are the point:

- **No wasted threadgroups.** A grid that carried the leaf on a third axis
  would have to give every leaf the largest leaf's tile count, and the
  threadgroups past a small leaf's end would launch only to exit. A packed
  tile axis gives each leaf exactly the tiles it asked for, so a frontier of
  one huge leaf and thirty tiny ones costs exactly the tiles those thirty one
  leaves need. Highly unbalanced frontiers are the normal case in leaf-wise
  growth, so this is not a corner.
- **Two grid dimensions, not three.** The row-tile axis is already `grid.y`
  in the shipping kernels and the portable ceiling on it is known
  (`MAX_GRID_DIM_Y`). Nothing here needs a `grid.z` whose portable limit
  across Metal, CUDA, and HIP this project has not established.

The cost is a binary search per threadgroup over at most `max_items` entries,
run once by thread 0 into threadgroup memory, behind the barrier the kernel
already pays to zero its shared histogram. At 32 items that is five Int32
loads per threadgroup against a row loop of hundreds.

What batching does not change
-----------------------------
Every item writes to its own output slice, and no accumulation ever crosses
an item. Within an item, accumulation is the same fixed-point Int32 the
single-leaf kernels use, the tiled reduction sums an item's own tiles in
ascending order, and the atomic strategy folds shared partials into the
item's own slice with integer atomics. So a leaf's histogram is bit-identical
whether it was built alone or in a batch of thirty, and the batch size is a
launch decision that no result can observe. That is the invariant the whole
lane rests on.

The same invariant is what makes every knob on `_plan_hist_kernel` free of
numeric consequence. That kernel is the device-plan path's third accumulation
arm, default off behind `plan_lean_requested`, and it carries the two ideas
this project took from CatBoost's histogram kernel -- private threadgroup
accumulators instead of one contended shared histogram, and a feature group
so a row's gradient pair is read once and spent several times -- together with
a per-item row split that follows the item's own count rather than the tree's
row bound. All of them move where an integer is added and none of them changes
which integers are added, so they select traffic, contention and parallelism
and never a number.

Where the leaves come from
--------------------------
The primitives here are worth exactly as much as the grower above them can
feed them. `gpu_frontier.leaves_per_launch` computes the number for each of the
four growers, and `handoffs/algorithm_22_leaf_batching.md` states the
consequence plainly rather than burying it: batching is a change to the
*grower*, and these kernels are the half of it that can be built and reasoned
about first. The three growers that can offer more than one leaf are the
device-search path (two children per commit), a speculative frontier (up to two
per speculated commit, and speculation is semantically free, see
gpu_frontier.mojo), and level-wise growth (a whole level).

**A LEVEL-WISE GROWER ARRIVED AND THIS MODULE IS NO LONGER A NO-OP ON A DEFAULT
FIT. Corrected 2026-08-17.** This section read "on the trainer's default path
today that is one leaf per commit: `grow_tree_gpu` builds the smaller child and
derives the sibling by subtraction, so a batch of one is all there is and this
module is a no-op", and it called level-wise growth "a separate lane" that had
not landed. Both halves are false at head. The oblivious level build reaches
`GpuLeafBatcher.enqueue_device_plan_batch_fused` and
`enqueue_device_plan_batch_fused_subtracting` from
`histogram_gpu.GpuHistogramBuilder.enqueue_desc_level_children`, with `2L` items
in one plan, and the subtracting arm of that pair is what a default fit takes
(`oblivious_subtract_requested`, `!= "0"` since 2026-08-17). The leaf-wise
device-resident plane still commits one built child per split through
`enqueue_desc_child`, so the old sentence describes THAT path and only that
path.

Histogram subtraction, on the device
------------------------------------
`enqueue_subtract` derives a sibling's histogram from the parent's and the
built child's, in the fixed-point buffer where both already live. Integer
subtraction is exact, so the derived histogram is exactly the one the built
histogram would have been, and the parent no longer has to be downloaded for
a host-side `subtract_histogram`. Two conditions make it valid, and both are
checked rather than assumed: the two histograms must have been accumulated
under the same fixed-point scales (the scales are fixed per round and per
class, so this holds within a tree and not across rounds), and under the same
active feature set (a feature absent from one and present in the other would
leave a nonzero slice being subtracted from a zero one). `HistogramSlotPool`
carries the stamp that encodes both.

Memory
------
The three buffers this module sizes, symbolically, with `F = n_features`,
`B = n_bins`, `S = n_slots <= F`, `K` items in a batch, `P` pool slots:

    output      P * 3 * F * B * 4 bytes      one full-width histogram per slot
    partials    T * 3 * S * B * 4 bytes      T = total tiles in the batch
    item tables K * (ITEM_WORDS + 2) * 4 + K * F * 4 bytes

The output pool is what grows with the frontier, and it is why the pool is
bounded and evictable: a leaf whose slot was taken back can always be rebuilt
from its row range, so eviction costs a rebuild and never a wrong answer. The
partial buffer is what grows with the batch, and it is the term that can make
batching worse rather than better: a batch wide enough to fill the device can
ask for more tiles than the partial budget holds, at which point the planner
either gives the batch fewer tiles than it wanted or resolves it to the
atomic strategy, which needs no partial buffer at all. `plan_batch` reports
which of those happened; it does not silently do either.

The frontier seam
-----------------
A grower does not assemble a batch by hand. Four calls do it, and each one may
decline rather than force:

    admit_frontier_batch(frontier, pool, storage, features)
    assign_batch_slots(frontier, pool, admission, stamp)
    plan_frontier_batch(caps, frontier, admission, g_scale, h_scale, ...)
    batcher.enqueue_frontier_batch(plan, bins, rows, grad, hess)

`admit_frontier_batch` is the conservative switch and it declines for five
distinct reasons (`BATCH_*`), every one of which leaves the caller on the
established one-leaf-per-launch path that
`GpuActiveRows.enqueue_range_histogram` already implements. Two of them are
worth naming here: a build whose batched kernels are not compiled in and
validated (`KernelFeatures.batched_leaf_kernel`), and a bin matrix that is not
the one-`UInt8`-per-cell layout these kernels index
(`BinStorageDescriptor.is_dense_feature_major_u8`). Neither is a performance
judgement; both are facts about what can be run at all.

`enqueue_frontier_batch` takes the `GpuActiveRows` itself rather than a row
pointer, so a batch reads the permutation the tree is actually growing on and
every item's window is checked against that tree's live prefix before a kernel
starts. Under bagging the prefix is the bag, and a window outside it would be
inside the buffer and outside the dataset the tree is being fit to, which
nothing downstream could detect.

Nothing in the sequence allocates or synchronizes per leaf: slots come from a
pool sized once, the item table is two staged copies per batch, and the launch
is at most three kernels whatever the batch's width. `download_slots` closes
the loop on the way home, mapping the output pool once for a whole batch
rather than once per leaf.

Scope and coupling
------------------
This module owns kernels, buffers, and the launch planning for batched
histogram construction. It does not own the frontier (gpu_frontier.mojo), the
row permutation (`GpuActiveRows`), the gradients, or the split search: the
binned matrix and the gradient planes arrive as pointers and the row
permutation arrives as the struct that owns it, exactly as
`enqueue_range_histogram` takes them, so nothing is uploaded or copied twice.
`DeviceCaps`, `derive_block_threads`, the `STRATEGY_*` constants, the grid and
tile bounds, and `BYTES_PER_PARTIAL_CELL` are imported from `gpu_tiling.mojo`
rather than restated, and `MAX_BINS`/`PLANES_PER_HISTOGRAM` from
`gpu_histogram_specializations.mojo`, so every one of those numbers means one
thing across the backend. The per-item tile distribution is local, because a
shared tile budget across several leaves is a shape `derive_tiling` has no way
to express.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, global_idx, grid_dim, thread_idx
from std.math import round
from std.memory import stack_allocation
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .gpu_active_rows import (
    MAX_ROWS,
    STEP_LIVE,
    STEP_ROW_BEGIN,
    STEP_ROW_COUNT,
    GpuActiveRows,
    LeafRange,
)
from .gpu_frontier import NO_SLOT, LeafFrontier, LeafStats, LeafWorkItem
from .gpu_histogram_specializations import (
    MAX_BINS,
    PLANES_PER_HISTOGRAM,
    BinStorageDescriptor,
    KernelFeatures,
)
from .gpu_tiling import (
    BYTES_PER_PARTIAL_CELL,
    MAX_GRID_DIM_Y,
    MIN_ROWS_PER_TILE_BIN_FACTOR,
    MIN_ROWS_PER_TILE_THREAD_FACTOR,
    STRATEGY_ATOMIC,
    STRATEGY_AUTO,
    STRATEGY_TILED,
    TARGET_BLOCKS_PER_SM,
    DeviceCaps,
    derive_block_threads,
    derive_tiling,
)

# Gradient, hessian, and count planes, in that order, in every histogram.
# Imported rather than restated: it is the same three planes
# `gpu_histogram_specializations` sizes a shared histogram with and
# `histogram_gpu` lays its output buffer out as.
comptime N_PLANES = PLANES_PER_HISTOGRAM

# Items one launch may cover. The binary search is logarithmic in this and
# the item table is staged in full on every launch, so the bound keeps both
# costs flat; it is not a claim about where batching stops paying.
comptime DEFAULT_MAX_ITEMS = 32

comptime OBLIVIOUS_MAX_ITEMS = 64
"""Items a batcher serving `grow_policy = oblivious` must hold, and it is a
measured number rather than a sizing preference.

A depth-6 level's last generation offers `2 ** 6 = 64` children and one batch
covers at most `max_items` of them. At `DEFAULT_MAX_ITEMS` that level needs two
batches, which costs two extra command buffers, and
`gpu_resident_round.oblivious_launch_census(6, batch_max_items=32)` lands the
tree at **exactly 64** -- on the queue-depth knee rather than under it
(`docs/GPU_PORTABILITY.md` 6.2; the enqueue-cost ladder in
`bench/results/session3_2026-08-16/RESULTS.md` is flat at 6-7 microseconds
through 64 and 14-17 beyond). At 64 the same census is **62**, under by two.

**Depth 5 tops out at 32 children**, so a lane that validated only at depth 5
would see the two arms agree and would learn nothing; that is why this constant
exists rather than a comment asking the caller to think about it.

The cost is small and is worth stating so nobody trades it away: the item table
is `max_items * (ITEM_WORDS + SCALE_WORDS)` Int32 plus `max_items * n_features`
for the per-item feature table, which at 64 items and 50 features is under 14
kilobytes.

It is also exactly `gpu_split_search.OBLIVIOUS_MAX_LEAVES`, and the agreement is
not a coincidence: both are `2 ** 6`, both are CatBoost's default depth, and
both stop where the queue does."""

# --- Item table layout ---------------------------------------------------
#
# One row of Int32 per item, staged as a single copy per launch. The kernels
# read it and nothing else about the batch's shape, so adding an item is a
# host-side table write and never a new argument.

comptime ITEM_BEGIN = 0
"""First slot of this item's window in the active-row buffer. Absolute, so a
caller holding several row permutations at once (one per class, say) puts the
permutation's base into this offset and needs no extra word."""

comptime ITEM_COUNT = 1
"""Rows in this item's window. Zero is legal and means a live leaf that holds
no rows, whose histogram is all zeros; `ITEM_DEAD` is the one negative value
and means something else entirely, see below."""

comptime ITEM_DEAD = -1
"""`ITEM_COUNT` on an item that must not be touched at all.

The one word a device-written plan needs that a host-written one does not.
`_pick_and_commit_kernel` writes the plan on every execution including the
three that commit nothing, exactly as it writes `STEP_LIVE` on every exit, so
that a batch enqueued past the end of growth is already in the queue and reads
its way to doing nothing. Zeroing is what makes that a real distinction rather
than a cosmetic one: `_batch_zero_kernel` clears the slice `ITEM_OUT` names, so
a dead item that left a stale `ITEM_OUT` behind would erase a live leaf's
histogram. A dead item is skipped by the zeroing outright, and the
accumulation's `tile_end = min(tile_begin + rows_per_tile, count)` is already
negative for it, so its threadgroups exit at their first comparison.

`_stage_plan` refuses a negative count, so a host-staged plan can never carry
this value and the guard below is unreachable from every path that exists
today. That is deliberate: it is what makes the sentinel a new arm rather than
a change to the shipping one."""

comptime ITEM_ROWS_PER_TILE = 2
comptime ITEM_TILE_BEGIN = 3
"""This item's first tile on the packed tile axis: the exclusive prefix sum
of the batch's tile counts, and the key the kernels binary search."""

comptime ITEM_TILES = 4
comptime ITEM_OUT = 5
"""Histogram slot this item's result is written to."""

comptime ITEM_PLANE = 6
"""Gradient plane this item reads, as a multiple of `n_rows`."""

comptime ITEM_WORDS = 8
"""Padded to eight so an item row is a round number of words."""

# Per-item fixed-point scales, Float32, in the kernels' own precision. Per
# item rather than per launch because a multiclass round holds one scale pair
# per class, and a batch may span classes.
comptime SCALE_G = 0
comptime SCALE_H = 1
comptime SCALE_WORDS = 2

# --- Planner verdicts ----------------------------------------------------

comptime VERDICT_UNKNOWN = 0
"""The plan is legal and neither structural fact below applies. Whether it is
faster is a measurement, and this module does not guess at one."""

comptime VERDICT_SINGLE_FILLS = 1
"""Some item on its own already reaches the block target, so the device was
not underfilled and batching it with others buys no occupancy."""

comptime VERDICT_OCCUPANCY_GAIN = 2
"""Every item alone leaves the device underfilled and the batch does not.
This is the case the module exists for."""

comptime VERDICT_PARTIAL_BOUND = 3
"""The tiled partial buffer could not hold the tiles the batch wanted. The
plan either kept fewer tiles or fell back to the atomic strategy, and either
way the batch is paying for its width."""


def verdict_name(verdict: Int) -> String:
    if verdict == VERDICT_SINGLE_FILLS:
        return String("single_fills")
    if verdict == VERDICT_OCCUPANCY_GAIN:
        return String("occupancy_gain")
    if verdict == VERDICT_PARTIAL_BOUND:
        return String("partial_bound")
    return String("unknown")


def _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


# --- Kernels --------------------------------------------------------------
#
# A TERMINOLOGY CORRECTION, 2026-08-17, READ IT BEFORE THE KERNELS BELOW. It
# covers every use of the phrase in this FILE, the batcher's methods included,
# not only the kernels under this header.
#
# Several docstrings here say "the shipping arm" where they mean the
# NON-SUBTRACTING level build, the one that accumulates every child of a level
# from its own rows. That phrase was written while the subtracting arm was
# opt-in behind `MOJOTREES_GPU_OBLIVIOUS_SUBTRACT=1`. The predicate at
# `oblivious_subtract_requested` is now `!= "0"`, so **the SUBTRACTING arm is
# what a default GPU fit takes** and the arm those docstrings call "shipping" is
# the one reached only by setting the variable to "0".
#
# The comparisons themselves are still correct and are deliberately not
# reworded, because each one is an argument about two kernels rather than about
# a default: read "the shipping arm" as "the non-subtracting arm" throughout.
# This note exists because `bench/results/LANE_RULES.md` rule 5a records that
# labelling the arm we no longer take as "the shipped arm" is the exact defect
# that made a peer session withdraw a lane ranking.


def _batch_zero_kernel(
    out_hist: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    n_items: Int32,
    hist_size: Int32,
):
    """Zero every output slice this batch writes.

    Needed for the same reasons the single-leaf path needs it: the atomic
    strategy accumulates into whatever is there, a narrowed feature set
    leaves slices the reduction never writes, and an empty leaf's histogram
    is all zeros with no kernel to produce them. Indexed through the item
    table so slices belonging to other leaves are untouched.

    An item whose count is `ITEM_DEAD` is skipped rather than zeroed, which is
    the one thing that lets a device-written plan be enqueued before the
    commit that fills it has run. A live item holding no rows is *not* dead
    and is still zeroed: its histogram is legitimately all zeros and something
    has to write them. See `ITEM_DEAD`; no host-staged plan can reach this
    branch, because `_stage_plan` refuses a negative count.
    """
    var cells = N_PLANES * Int(hist_size)
    var i = global_idx.x
    if i < Int(n_items) * cells:
        var k = i // cells
        var r = i - k * cells
        var base = k * ITEM_WORDS
        if items[unsafe_offset = base + ITEM_COUNT][0] < Int32(0):
            return
        var slot = Int(items[unsafe_offset = base + ITEM_OUT][0])
        out_hist[unsafe_offset = slot * cells + r] = Int32(0)


def _batch_copy_back_zero_kernel(
    out_hist: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    n_items: Int32,
    hist_size: Int32,
    rows: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    desc: MutPointer[Int32, MutAnyOrigin],
):
    """`_batch_zero_kernel`, carrying the descriptor partition's deferred
    copy-back.

    **Why this exists, because without it the census fails by six.**
    `GpuActiveRows.enqueue_partition_desc` under `set_partition_fusion(True)`
    -- which is the default -- does not launch its copy-back. It leaves the
    compacted permutation in the scratch buffer and a debt, and the next
    `enqueue_desc_histogram` pays it inside the slot zeroing it has to launch
    anyway (`_copy_back_zero_slot_kernel`). A batched build does not go through
    that entry point, so a level whose children were built by a batch would
    either read a permutation that is one partition out of date, or pay the
    copy-back in a third partition launch. The third launch is 1 per level, 6
    per depth-6 tree, and it is exactly the margin: the tree goes from 62
    command buffers to 68 and back over the queue's knee. So the batch pays it
    in its own zeroing pass, in the launch it has to make anyway, and the
    partition stays two.

    Every store here is a store one of the two kernels it replaces made, of the
    same value, to the same address, under the same guard -- the same argument
    `_copy_back_zero_slot_kernel` makes. There is no arithmetic beyond index
    formation and no floating point anywhere, so nothing here can move a bit.

    **The two halves are unordered with respect to each other and that is what
    makes the fusion legal.** The copy-back writes `rows`, the zeroing writes
    `out_hist`, and neither reads what the other writes. The launch boundaries
    that matter are the ones this does not touch: the scatter before it writes
    `scratch` across blocks, and the accumulation after it reads `rows` and
    adds onto `out_hist` across blocks.

    **The two guards are different and neither may be borrowed for the other.**
    The copy-back is owed only by a live step, so it reads `STEP_LIVE`, exactly
    as `_copy_back_kernel`'s descriptor arm does. The zeroing is owed by every
    item the plan calls live, which is a per-item question (`ITEM_DEAD`) and
    not a property of the step: a step may be dead while the plan still names
    slots that must be cleared, and a step may be live while some item of the
    plan is not. An earlier draft guarded both on `STEP_LIVE` and would have
    skipped zeroing on exactly the steps a device-written plan is enqueued
    ahead of.
    """
    var stride = Int(block_dim.x) * Int(grid_dim.x)

    # Half one: the partition's deferred copy-back, `_copy_back_kernel`'s
    # descriptor arm verbatim.
    if desc[unsafe_offset=STEP_LIVE][0] != Int32(0):
        var b = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        var n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])
        var j = Int(global_idx.x)
        while j < n:
            var i = b + j
            rows[unsafe_offset=i] = scratch[unsafe_offset=i][0]
            j += stride

    # Half two: `_batch_zero_kernel` verbatim, guarded store and all.
    var cells = N_PLANES * Int(hist_size)
    var i2 = global_idx.x
    if i2 < Int(n_items) * cells:
        var k = i2 // cells
        var r = i2 - k * cells
        var base = k * ITEM_WORDS
        if items[unsafe_offset = base + ITEM_COUNT][0] < Int32(0):
            return
        var slot = Int(items[unsafe_offset = base + ITEM_OUT][0])
        out_hist[unsafe_offset = slot * cells + r] = Int32(0)


@always_inline
def _item_for_tile(
    items: MutPointer[Int32, MutAnyOrigin], n_items: Int, tile: Int
) -> Int:
    """The item owning global tile `tile`: the largest `k` whose
    `ITEM_TILE_BEGIN` is at most `tile`.

    The offsets are ascending by construction (they are an exclusive prefix
    sum of positive tile counts), so this is an ordinary upper-bound search
    and it terminates in `ceil(log2(n_items))` steps. Every thread of a
    threadgroup shares `block_idx.y`, so every thread that runs it computes
    the same answer; the kernels below run it on one thread anyway and
    publish the result through threadgroup memory.
    """
    var lo = 0
    var hi = n_items - 1
    while lo < hi:
        var mid = (lo + hi + 1) >> 1
        var begin = Int(
            items[unsafe_offset = mid * ITEM_WORDS + ITEM_TILE_BEGIN][0]
        )
        if begin <= tile:
            lo = mid
        else:
            hi = mid - 1
    return lo


def oblivious_subtract_requested() -> Bool:
    """`MOJOTREES_GPU_OBLIVIOUS_SUBTRACT=0`, the escape hatch that turns sibling
    subtraction OFF inside the oblivious level build.

    The summary line used to read `=1, the switch for sibling subtraction`,
    which was the spelling before the 2026-08-17 flip; the predicate is
    `!= "0"`.

    **ON unless turned off, since 2026-08-17.** This paragraph read "Off unless
    asked for, which is this package's rule for a path no benchmark has priced,
    and the rule holds here even though the expected win is the largest one on
    this plane's board" -- which was true of the predicate `== "1"` this
    function used to carry and has been false since the flip comment below it
    was written the same day. It has been priced twice since and the numbers are
    in that comment. Corrected 2026-08-17 by the GPU histogram lane, which found
    the docstring and the code disagreeing about the default.

    **Quote this arm's speedup as a RANGE of 1.58x to 1.78x, not as a single
    number, until another lane reconciles the two readings.** The first reading
    is recorded as 1.78x, 21.97 s to 12.34 s; the second is 22.76 s to 14.39 s
    interleaved, which is 1.58x. Both are the same shape, 799,110 x 100 x 100
    trees, symmetric depth 6, M4, and neither has been withdrawn, so picking one
    here would be a choice dressed as a fact. Reconciliation is another lane's
    item and this lane took no clock of its own.

    The shipped level build used to accumulate
    every child of the level from its own rows, so it read every active row
    once per level; this arm accumulates only the SMALLER child of
    each pair and derives the sibling from the parent, so it reads at most half
    of them and usually fewer. Both arms are two launches per level and
    `gpu_resident_round.oblivious_launch_census(6)` is 62 either way, which is
    a precondition of this arm rather than a happy result -- see
    `GpuLeafBatcher.enqueue_device_plan_batch_fused_subtracting`.

    **Nothing about the answer moves.** Every accumulator is fixed-point Int32
    under one pair of scales for the whole tree, the two children of a split
    partition the parent's rows exactly, and the quantization of a row is a
    function of the row alone, so the derived sibling is bit-for-bit the
    histogram a direct accumulation would have produced. The argument is
    written out leg by leg at `_batch_hist_atomic_subtract_kernel`. So this
    switch selects an amount of memory traffic and never a number.

    This paragraph read "The default flips when a run says so and not before,
    which is the same sentence `gpu_split_search.oblivious_wide_scan_requested`
    carries." **A run said so on 2026-08-17 and both defaults flipped**, so the
    sentence describes a state neither switch has been in since. Corrected here
    and left as a record that the belief moved.

    Read here rather than taken as a parameter for the same reason that switch
    is: the decision belongs beside the kernels it selects between. The one
    dispatch site is `GpuHistogramBuilder.enqueue_desc_level_children` over in
    `histogram_gpu`, which is the only oblivious-only entry point either arm
    has, and the branch is written there rather than inside
    `enqueue_device_plan_batch_fused` so that a two-item leaf-wise plan cannot
    reach the subtracting arm by having the environment set. That is not a
    style preference: a leaf-wise commit reassigns the PARENT's slot to one of
    the children and hands the other the lowest free slot, so the slot
    relationship this arm rests on does not hold there at all.
    """
    # DEFAULT ON since 2026-08-17, and the variable is now an escape hatch that
    # turns the subtraction OFF rather than a switch that turns it on. Measured
    # the day it was written, 799,110 x 100 x 100 trees, symmetric depth 6, M4,
    # three round-robin cycles against an interleaved baseline: 22.76 s to
    # 14.39 s alone, and 10.36 s combined with the skip-last-build and wide-scan
    # arms. Every arm of every cycle returned rmse 2.439382420, identical to nine
    # decimals, so the exact-integer identity argument in this function's
    # docstring holds in measurement as well as on paper.
    #
    # QUOTE THE COMBINED ARM AS A RANGE, 2.08x TO 2.20x, UNTIL A LANE
    # RECONCILES IT. This comment said "which is 2.20x", which is 22.76 / 10.36;
    # `gpu_resident_round.OBLIVIOUS_SKIP_LAST_BUILD_VAR` says 2.08x for the same
    # three arms in combination. Neither reading has been withdrawn and this lane
    # took no clock, so the range stands and the reconciliation is another lane's
    # item. The ALONE figure has the same problem and the range for it is 1.58x
    # to 1.78x; see this function's docstring.
    #
    # Flipped under LANE_RULES rule 5, added the same day. A bit-identical
    # change cannot alter any user's output, so flipping its default changes
    # the clock and nothing else, and a proven win left switched off is not a
    # conservative position, it is an unshipped one. Per that rule this variable
    # survives ONE round as an off switch and is then deleted.
    return getenv("MOJOTREES_GPU_OBLIVIOUS_SUBTRACT") != "0"


@always_inline
def _live_item_pairs(
    items: MutPointer[Int32, MutAnyOrigin], n_items: Int
) -> Int:
    """Half the number of live items at the head of the plan, which for a
    committed oblivious level is the level's parent count `L`.

    A level commit fills items `[0, 2L)` and kills everything from `2L` to the
    staged width (`gpu_tree_tables._commit_level_kernel`, and
    `_kill_level_plan` for why the kill covers the whole width and not just the
    level's own prefix). So `ITEM_COUNT >= 0` holds on a prefix and
    `ITEM_COUNT == ITEM_DEAD` on the rest, and the boundary is an ordinary
    binary search, six loads at the 64-item width this mode is staged at.

    Derived on the device rather than passed from the host, and the reason is
    not only that the host would need a synchronization to learn it. A level
    whose commit stopped growth kills the WHOLE plan, so the live prefix is
    zero and every arm below does nothing; a host that passed `1 << level`
    instead would zero and copy slots for a level the tree never grew. The
    table is the authority in both cases.

    Returns zero on a dead plan, and the callers below are written so that
    zero means "this launch does nothing", exactly as `ITEM_DEAD` already does
    for the shipping arm."""
    var lo = 0
    var hi = n_items
    while lo < hi:
        var mid = (lo + hi) >> 1
        if items[unsafe_offset = mid * ITEM_WORDS + ITEM_COUNT][0] >= Int32(0):
            lo = mid + 1
        else:
            hi = mid
    return lo >> 1


def _batch_copy_back_zero_subtract_kernel(
    out_hist: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    n_items: Int32,
    hist_size: Int32,
    rows: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    desc: MutPointer[Int32, MutAnyOrigin],
):
    """`_batch_copy_back_zero_kernel` for the subtracting level build: the same
    deferred copy-back, and a second half that prepares each PAIR of children
    rather than zeroing each child.

    **The slot relationship this rests on, stated first because everything else
    depends on it.** `gpu_tree_tables._commit_level_kernel` pins slot index to
    leaf index, gives the left child of parent `j` leaf index `j` and the right
    child leaf index `j + L`, and the parent itself was leaf index `j` of the
    previous level and therefore already owns slot `j`. So **the parent's
    histogram is sitting in the left child's destination slot** when this
    kernel runs, and the level's `2L` plan items are the `L` left children at
    `[0, L)` followed by the `L` right children at `[L, 2L)`, item `k` writing
    slot `ITEM_OUT[k]`. That is read out of the item table where it can be
    (`ITEM_OUT`, `ITEM_COUNT`) and assumed where it cannot: no table records
    which slot holds the parent, so if that pinning is ever relaxed this arm
    computes wrong histograms silently. It is checked by nothing here and is
    the first precondition to re-read if this arm ever disagrees with the
    shipping one.

    **What the pair needs before the build.** The build accumulates the
    smaller child into its own slot and subtracts it out of the sibling's slot,
    so the sibling's slot has to hold the parent and the built child's slot has
    to hold zeros. Two cases, and the built child is the smaller one:

    - *The right child is built* (`n_right <= n_left`). The parent is already
      in the left child's slot, which is where the derived left child belongs.
      Zero the right child's slot and touch nothing else. This is the cheap
      case and it is why ties build the right child.
    - *The left child is built* (`n_left < n_right`). The derived right child
      belongs in the right child's slot, so the parent is copied there first
      and the left child's slot is then zeroed. One extra slot-sized read and
      write for the pair, in the pass that was already writing both slots.

    Both cases are one thread per `(pair, cell)`, and in the copying case the
    same thread reads the parent cell and then zeroes it, so no thread reads a
    cell another thread writes and the copy needs no launch of its own. That
    is the whole reason a per-pair choice is affordable: the alternative to the
    copy is building a FIXED child of each pair, which is accuracy-neutral and
    reads more rows whenever a split is lopsided the other way.

    The copy moves `N_PLANES * n_features * n_bins` cells; building the
    sibling instead would visit `count * n_features` bins and read a gradient
    and a hessian at each, so the copy is the cheaper of the two once a child
    holds more than roughly `n_bins` rows. That is a comparison of counts read
    off the two loops and not a measurement; no timing of either arm has been
    taken by this lane.

    The copy-back half is `_batch_copy_back_zero_kernel`'s verbatim, guard and
    stride and all, and carries the same debt for the same census reason. The
    `barrier()` between the halves is reached by every thread of the block --
    there is no return above it -- and exists only to publish the pair count
    one thread resolved."""
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var s_pairs = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        s_pairs[unsafe_offset=0] = Int32(_live_item_pairs(items, Int(n_items)))

    # Half one: the partition's deferred copy-back, `_copy_back_kernel`'s
    # descriptor arm verbatim.
    if desc[unsafe_offset=STEP_LIVE][0] != Int32(0):
        var b = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        var n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])
        var j = Int(global_idx.x)
        while j < n:
            var i = b + j
            rows[unsafe_offset=i] = scratch[unsafe_offset=i][0]
            j += stride

    barrier()

    # Half two: one thread per (pair, cell). A dead plan has no pairs and this
    # half does nothing, which is what `ITEM_DEAD` buys the shipping arm.
    var n_pairs = Int(s_pairs[unsafe_offset=0][0])
    var cells = N_PLANES * Int(hist_size)
    var i2 = Int(global_idx.x)
    if i2 >= n_pairs * cells:
        return
    var p = i2 // cells
    var r = i2 - p * cells
    var lbase = p * ITEM_WORDS
    var rbase = (p + n_pairs) * ITEM_WORDS
    var n_left = items[unsafe_offset = lbase + ITEM_COUNT][0]
    var n_right = items[unsafe_offset = rbase + ITEM_COUNT][0]
    var left_slot = Int(items[unsafe_offset = lbase + ITEM_OUT][0])
    var right_slot = Int(items[unsafe_offset = rbase + ITEM_OUT][0])
    if n_right <= n_left:
        out_hist[unsafe_offset = right_slot * cells + r] = Int32(0)
    else:
        out_hist[unsafe_offset = right_slot * cells + r] = out_hist[
            unsafe_offset = left_slot * cells + r
        ][0]
        out_hist[unsafe_offset = left_slot * cells + r] = Int32(0)


# --- The pre-quantized gradient source, on the batched family -------------
#
# WHAT THE OBLIVIOUS LEVEL BUILD WAS DOING, AND WHY IT IS THE SLOW ARM
# --------------------------------------------------------------------
# The three kernels below take `grad` and `hess` as Float32 planes and form
# `Int32(round(g * g_scale))` and its hessian twin INSIDE the row loop, which
# is once per (row, feature) visit. `grid.x` is the feature axis, so every one
# of the `n_slots` threadgroups covering a row's tile repeats the same two
# gathers, the same two Float32 multiplies and the same two rounds on the same
# row. At the 90-feature real benchmark that is ninety redundant evaluations of
# an expression whose operands do not depend on the feature.
#
# `gpu_active_rows` fixed this on the single-leaf range family and the fix
# never reached here: `_quantize_grad_hess_kernel` writes one interleaved
# `Int32` pair per row, once per round, and the range kernels gather the pair
# instead of the two Float32 words. This section is that source, on the
# batched family.
#
# WHY IT CANNOT MOVE A BIT
# ------------------------
# 1. **The scale is global per round on the path that matters, and it is
#    checked rather than assumed.** `stage_device_plan` takes ONE `g_scale`
#    and ONE `h_scale` and writes that pair into every item's row of
#    `scales_dev`; its only caller, `histogram_gpu.stage_desc_level_plan`,
#    passes the builder's own `g_scale`/`h_scale`, which are the round's. So
#    within a staged plan the pair is uniform over items, uniform over
#    features (the kernels never index scales by feature), and constant over
#    the tree's levels, because `stage_device_plan` runs once per tree. The
#    quantized source is offered only on the device-plan arms for exactly
#    that reason; `enqueue_batch` stages a per-item scale list that a
#    multiclass batch may vary, so it keeps the Float32 arm and passes
#    `use_quant = 0`.
# 2. **The value is the same expression.** `_batch_quantize_kernel` evaluates
#    `Int32(round(grad[r] * g_scale))` -- the same Float32 multiply, the same
#    `round`, the same device compiler -- and stores it. Hoisting an
#    expression out of a loop whose other operands it does not depend on
#    cannot change its value.
# 3. **The accumulation is unchanged.** The same Int32 lands in the same
#    threadgroup bin through the same atomic, and Int32 addition is
#    associative and commutative, so nothing about ORDER was ever load
#    bearing and nothing about it moves here.
#
# WHAT IT COSTS, STATED SO IT IS NOT DISCOVERED
# ---------------------------------------------
# One streaming launch over `n_rows` per TREE, not per level: `stage_device
# _plan` invalidates the buffer and the first accumulation of the tree
# rebuilds it. `gpu_resident_round.oblivious_launch_census(6)` is 62 command
# buffers and the Metal queue on the measured machine is 64 deep, so this
# takes a depth-6 tree to **63**, still under the knee -- but it is one of the
# two, and it is the reason this arm is default off rather than default on
# like its range-family twin. Folding the pass into
# `_batch_copy_back_zero_kernel`, which already launches per level and could
# run it on the level-zero pass alone, would take the census back to 62; that
# is a follow-up and not this edit.
#
# The buffer is `2 * n_planes * n_rows` Int32, which at 463,715 rows and one
# plane is 3.7 MB.

comptime BATCH_QUANT_VAR = "MOJOTREES_GPU_BATCH_QUANT"
"""Whether the batched family gathers the pre-quantized Int32 pairs.

`1` selects it, anything else leaves the Float32 planes in force, and `0`
additionally declines the buffer's allocation so a fit that will never use it
does not carry 3.7 MB. Default off because it moves the launch census by one
and no benchmark has priced either half yet; the identity argument above is
complete, so under `bench/results/LANE_RULES.md` rule 5 a measured win flips
this in the session that measures it."""

comptime BATCH_CONST_HESSIAN_VAR = "MOJOTREES_CONST_HESSIAN"
"""The same withdrawal switch `GpuActiveRows` reads, and it is deliberately
the same name. Constant hessian is a DECLARATION ABOUT THE OBJECTIVE that the
trainer owns; an environment variable may only take the permission away, never
grant it, so `0` here refuses `set_constant_hessian(True)` and nothing else
turns it on."""


comptime BATCH_CONST_HESSIAN_FORWARD_VAR = "MOJOTREES_GPU_BATCH_CONST_HESS"
"""`1` forwards the trainer's constant-hessian DECLARATION to the batcher, so
the batched kernels actually elide the hessian plane.

A separate name from `BATCH_CONST_HESSIAN_VAR` above and it is not a
duplicate. That one is the withdrawal switch, which only ever takes the
permission away. This one exists because the declaration never ARRIVED:
`GpuHistogramBuilder.set_constant_hessian` forwarded it to `self.rows` and
stopped, so `GpuLeafBatcher.constant_hessian` was False on every fit that ever
ran and the elision inside `_batch_hist_atomic_kernel` and its two siblings has
never executed on the oblivious level build. Verified by reading both setters
and every caller, 2026-08-17.

It is a switch rather than a plain fix so that the elision can be measured
against the arm that has been shipping. When true the declaration cannot move
a bit -- the plane it stops accumulating is rebuilt as `hq_const * vc`, the
exact Int32 the accumulation would have left there, argued at
`_batch_hist_atomic_kernel` -- so under `bench/results/LANE_RULES.md` rule 5 a
measured win flips this in the session that measures it. When the declaration
is FALSE this switch does nothing whatever, because it forwards a false."""


def _batch_quant_allowed() -> Bool:
    """Whether this batcher allocates the quantized buffer at all."""
    return getenv(BATCH_QUANT_VAR) != "0"


def batch_const_hessian_forward_requested() -> Bool:
    """`MOJOTREES_GPU_BATCH_CONST_HESS=1`. See the constant above."""
    return getenv(BATCH_CONST_HESSIAN_FORWARD_VAR) == "1"


def batch_quantized_grads_requested() -> Bool:
    """`MOJOTREES_GPU_BATCH_QUANT=1`. The arm's initial state; a benchmark
    holding both arms in one process moves it with
    `GpuLeafBatcher.set_quantized_gradients` instead of re-execing, for the
    reason `GpuActiveRows.set_feature_group` gives at length."""
    return getenv(BATCH_QUANT_VAR) == "1"


def _batch_quantize_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    plane_base: Int32,
    n_rows: Int32,
    g_scale: Float32,
    h_scale: Float32,
):
    """One plane's gradients and hessians, quantized once, interleaved.

    `gpu_active_rows._quantize_grad_hess_kernel` plus the plane offset a
    batched item carries in `ITEM_PLANE`. A separate symbol rather than an
    import for the reason the header gives about `_plan_env_int`: this module
    owns its own launches, and the scales are launch arguments rather than
    table reads because the pair is global to the staged plan. The section
    above carries the check that establishes that.

    One thread per row of one plane, two Float32 multiplies, two rounds, two
    stores, and no read of the item table at all, so it does not have to be
    ordered after anything the plan writes.

    `gq[2 * (plane_base + r)]` and its odd twin are exactly the two integers
    the histogram row loop would otherwise have formed for row `r`, so a
    histogram accumulated from this buffer is the same word for word. The
    argument is in the section above rather than here, in one place, because
    three kernels rest on it.
    """
    var r = Int(global_idx.x)
    if r < Int(n_rows):
        var i = Int(plane_base) + r
        gq[unsafe_offset = 2 * i] = Int32(
            round(grad[unsafe_offset=i][0] * g_scale)
        )
        gq[unsafe_offset = 2 * i + 1] = Int32(
            round(hess[unsafe_offset=i][0] * h_scale)
        )


def _batch_hist_partial_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    scales: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_items: Int32,
    n_slots: Int32,
    n_bins: Int32,
    feat_stride: Int32,
):
    """One (feature slot, packed tile) partial histogram, written to its own
    slot of the global partial buffer.

    The row loop is `_range_hist_partial_kernel`'s, unchanged: gather the
    row through the active-row permutation, read its bin, quantize gradient
    and hessian into Int32, accumulate in threadgroup memory, flush once.
    What is new is that the tile index selects the leaf as well as the rows,
    and that the feature and the scales are read per item, so a batch may
    span leaves with different feature sets and different fixed-point scales
    without the kernel branching on either.

    No atomics on global memory and one write per cell, so the partial buffer
    is written exactly once per (tile, slot, bin) and the reduction below is
    the only reader.
    """
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var ns = Int(n_slots)
    var slot = Int(block_idx.x)
    var g_tile = Int(block_idx.y)

    var meta = stack_allocation[
        8, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sg = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    # One thread resolves the tile to its item, behind the barrier the shared
    # histogram already needs.
    if tid == 0:
        var k = _item_for_tile(items, Int(n_items), g_tile)
        var base = k * ITEM_WORDS
        meta[unsafe_offset=0] = Int32(k)
        meta[unsafe_offset=1] = items[unsafe_offset = base + ITEM_BEGIN][0]
        meta[unsafe_offset=2] = items[unsafe_offset = base + ITEM_COUNT][0]
        meta[unsafe_offset=3] = items[
            unsafe_offset = base + ITEM_ROWS_PER_TILE
        ][0]
        meta[unsafe_offset=4] = Int32(
            g_tile - Int(items[unsafe_offset = base + ITEM_TILE_BEGIN][0])
        )
        meta[unsafe_offset=5] = items[unsafe_offset = base + ITEM_PLANE][0]

    var b = tid
    while b < nb:
        sg[unsafe_offset=b] = 0
        sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var k = Int(meta[unsafe_offset=0][0])
    var begin = Int(meta[unsafe_offset=1][0])
    var count = Int(meta[unsafe_offset=2][0])
    var rows_per_tile = Int(meta[unsafe_offset=3][0])
    var t = Int(meta[unsafe_offset=4][0])
    var plane_base = Int(meta[unsafe_offset=5][0]) * nr

    var f = Int(feat_ids[unsafe_offset = k * Int(feat_stride) + slot][0])
    var g_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_G][0]
    var h_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_H][0]

    var tile_begin = t * rows_per_tile
    var tile_end = tile_begin + rows_per_tile
    if tile_end > count:
        tile_end = count

    var col = f * nr
    var j = tile_begin + tid
    while j < tile_end:
        var r = Int(rows[unsafe_offset = begin + j][0])
        var bin = Int(bins[unsafe_offset = col + r])
        var gq = Int32(round(grad[unsafe_offset = plane_base + r][0] * g_scale))
        var hq = Int32(round(hess[unsafe_offset = plane_base + r][0] * h_scale))
        _ = Atomic.fetch_add(sg.unsafe_offset(bin), gq)
        _ = Atomic.fetch_add(sh.unsafe_offset(bin), hq)
        _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        j += block_dim.x
    barrier()

    var plane = ns * nb
    var out_base = g_tile * N_PLANES * plane + slot * nb
    b = tid
    while b < nb:
        partials[unsafe_offset = out_base + b] = sg[unsafe_offset=b][0]
        partials[unsafe_offset = out_base + plane + b] = sh[unsafe_offset=b][0]
        partials[unsafe_offset = out_base + 2 * plane + b] = sc[
            unsafe_offset=b
        ][0]
        b += block_dim.x


def _batch_hist_atomic_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    scales: MutPointer[Float32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_items: Int32,
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
    feat_stride: Int32,
    use_quant: Int32,
    const_hess: Int32,
):
    """The same batched accumulation, folding each partial straight into its
    item's output slice with global integer atomics.

    The preserved fallback, for the same reason `STRATEGY_ATOMIC` is
    preserved in the single-leaf path: it needs no partial buffer, so it is
    the strategy a batch resolves to when the partial budget cannot hold the
    tiles the batch wants. Contention is no worse than the single-leaf
    kernel's, because tiles of different items fold into different slices;
    within an item it is exactly the same. Integer atomics make the result
    order-independent, so this path is bit-identical to the tiled one.

    **`use_quant` selects the pre-quantized source.** With it set the row loop
    reads the interleaved Int32 pair `_batch_quantize_kernel` wrote for the
    round instead of the two Float32 planes, which moves two multiplies and
    two rounds off the inner loop and halves the derivative gather. The
    integers are the same integers; the argument is at that kernel and in the
    section above it. `gq` is never dereferenced while `use_quant` is clear,
    so a caller with no such buffer passes any pointer it likes.

    **`const_hess` elides the hessian plane.** With it set the caller has
    declared that every row's hessian this round is exactly
    `histogram.CONSTANT_HESSIAN`, which is 1.0. The row loop then performs two
    shared atomics per visit instead of three, the shared zeroing covers two
    planes instead of three, and the flush writes the global hessian plane as
    `hq_const * vc`. That reconstruction is exact: `1.0f * h_scale` is
    `h_scale` because multiplying by one cannot round, so every row's hessian
    contribution is the single value `Int32(round(h_scale))`, computed below
    in this kernel by this device compiler from this launch argument; a bin
    that received `vc` rows holds `vc` copies of it added together, and Int32
    addition and multiplication agree modulo 2^32. `sh` is a comptime
    `stack_allocation` and is still allocated, so no occupancy follows -- what
    follows is a third fewer shared atomics and a third less shared zeroing.
    """
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var hs = Int(hist_size)
    var slot = Int(block_idx.x)
    var g_tile = Int(block_idx.y)

    var meta = stack_allocation[
        8, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sg = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    if tid == 0:
        var k = _item_for_tile(items, Int(n_items), g_tile)
        var base = k * ITEM_WORDS
        meta[unsafe_offset=0] = Int32(k)
        meta[unsafe_offset=1] = items[unsafe_offset = base + ITEM_BEGIN][0]
        meta[unsafe_offset=2] = items[unsafe_offset = base + ITEM_COUNT][0]
        meta[unsafe_offset=3] = items[
            unsafe_offset = base + ITEM_ROWS_PER_TILE
        ][0]
        meta[unsafe_offset=4] = Int32(
            g_tile - Int(items[unsafe_offset = base + ITEM_TILE_BEGIN][0])
        )
        meta[unsafe_offset=5] = items[unsafe_offset = base + ITEM_PLANE][0]
        meta[unsafe_offset=6] = items[unsafe_offset = base + ITEM_OUT][0]

    # Both flags are block-uniform launch scalars, so every branch below on
    # either of them is taken by the whole threadgroup together and no
    # barrier is reached by a subset of it.
    var quant = Int(use_quant) != 0
    var celide = Int(const_hess) != 0

    var b = tid
    while b < nb:
        sg[unsafe_offset=b] = 0
        if not celide:
            sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var k = Int(meta[unsafe_offset=0][0])
    var begin = Int(meta[unsafe_offset=1][0])
    var count = Int(meta[unsafe_offset=2][0])
    var rows_per_tile = Int(meta[unsafe_offset=3][0])
    var t = Int(meta[unsafe_offset=4][0])
    var plane_base = Int(meta[unsafe_offset=5][0]) * nr
    var out_slot = Int(meta[unsafe_offset=6][0])

    var f = Int(feat_ids[unsafe_offset = k * Int(feat_stride) + slot][0])
    var g_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_G][0]
    var h_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_H][0]
    # The elided plane's one quantized value. `Float32(1.0) * h_scale` is
    # exactly `h_scale`, so this is `Int32(round(h_scale))`, formed here by
    # the same device compiler that would have formed the per-row value it
    # stands in for. Never read while `celide` is clear.
    var hq_const = Int32(round(Float32(1.0) * h_scale))

    var tile_begin = t * rows_per_tile
    var tile_end = tile_begin + rows_per_tile
    if tile_end > count:
        tile_end = count

    var col = f * nr
    var j = tile_begin + tid
    while j < tile_end:
        var r = Int(rows[unsafe_offset = begin + j][0])
        var bin = Int(bins[unsafe_offset = col + r])
        var gqv = Int32(0)
        var hqv = Int32(0)
        if quant:
            gqv = gq[unsafe_offset = 2 * (plane_base + r)][0]
            if not celide:
                hqv = gq[unsafe_offset = 2 * (plane_base + r) + 1][0]
        else:
            gqv = Int32(
                round(grad[unsafe_offset = plane_base + r][0] * g_scale)
            )
            if not celide:
                hqv = Int32(
                    round(hess[unsafe_offset = plane_base + r][0] * h_scale)
                )
        _ = Atomic.fetch_add(sg.unsafe_offset(bin), gqv)
        if not celide:
            _ = Atomic.fetch_add(sh.unsafe_offset(bin), hqv)
        _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        j += block_dim.x
    barrier()

    var slice_base = out_slot * N_PLANES * hs + f * nb
    b = tid
    while b < nb:
        var vc = sc[unsafe_offset=b][0]
        if vc != 0:
            # The refill: `hq_const * vc` is the sum of `vc` copies of
            # `hq_const`, which is what the three-plane accumulation left in
            # `sh`.
            var vh = hq_const * vc if celide else sh[unsafe_offset=b][0]
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(slice_base + b), sg[unsafe_offset=b][0]
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(slice_base + hs + b), vh
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(slice_base + 2 * hs + b), vc
            )
        b += block_dim.x


def _batch_hist_atomic_subtract_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    scales: MutPointer[Float32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_items: Int32,
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
    feat_stride: Int32,
    use_quant: Int32,
    const_hess: Int32,
):
    """`_batch_hist_atomic_kernel` over half a level's items, folding the
    sibling subtraction into the same flush.

    The identical argument list, the identical row loop, the identical
    threadgroup accumulation, and the identical global fold. Two things are
    added and nothing is taken away.

    `use_quant` and `const_hess` mean exactly what they mean on
    `_batch_hist_atomic_kernel` and are argued there. The subtraction is
    unaffected by either: it folds the NEGATION of the cells this block just
    added, so whichever integer reached the built child's slot reaches the
    derived sibling's with its sign flipped, and `hq_const * vc` is one such
    integer like any other.

    **One: an item builds only if it is the smaller child of its pair.** Thread
    zero already resolves this block's tile to its item; it now also resolves
    the item to its pair (`_live_item_pairs` for the level's width `L`, then
    `k` and `k + L` or `k - L`), compares the two `ITEM_COUNT`s, and publishes
    `-1` in `meta[0]` when this block's item is the one that will be DERIVED
    rather than built. The whole block then returns after the barrier it was
    already paying. Ties go to the right child, which is the pair the zeroing
    pass can prepare without copying the parent; see
    `_batch_copy_back_zero_subtract_kernel`.

    An item outside the live prefix returns the same way, exactly as
    `ITEM_DEAD` already makes it accumulate nothing on the shipping arm. The
    return sits AFTER the barrier and after the shared histogram is zeroed, so
    a derived item's block costs precisely what a dead item's block already
    costs today and no more. Moving it earlier would need every thread to redo
    the two binary searches, which is the trade this deliberately does not
    make.

    **Two: the flush subtracts what it adds from the sibling's slot.** The same
    thread that folds a cell into `slice_base` folds its negation into
    `sub_base`, under the same `sc[b] != 0` guard, with the same integer
    atomics. `gpu_active_rows._range_hist_atomic_kernel`'s `do_sub` arm is this
    statement for statement, and this is the batched form of the fusion
    `histogram_gpu.enqueue_resident_leaf_subtracting` already ships: a
    standalone subtraction would be a launch that reads and writes every cell
    of two whole slots, and the census has no room for it.

    WHY THE DERIVED HISTOGRAM IS BIT-IDENTICAL TO A BUILT ONE
    ---------------------------------------------------------
    Leg by leg, and every leg is exact integer arithmetic:

    1. Every cell of every histogram in this pool is fixed-point Int32. A
       row's contribution is `Int32(round(grad[r] * g_scale))`, `Int32(round(
       hess[r] * h_scale))` and `1`, all three functions of the row and of the
       tree's scales alone, and neither the node a row lands in nor the block
       that visits it can change them.
    2. Integer addition is associative and commutative, so a histogram cell is
       the exact sum of its rows' contributions whatever order the tiles and
       the atomics fold them in. That is the same property that lets the
       shipping arm claim a batched build equals a single-leaf one.
    3. The level's partition is total and disjoint over the parent's window:
       `_commit_level_kernel` derives `n_left` and `n_right` from the parent's
       own histogram with `_row_goes_left` written over bins, and the scatter
       routes by the same rule over rows, so the two children's row sets
       partition the parent's exactly and share no row.
    4. Therefore, cell by cell, `parent = left + right` exactly, and the
       derived child `parent - built` is exactly the sum over the derived
       child's own rows: bit for bit the histogram a direct accumulation would
       have written. No rounding, no reassociation of floating point, nothing
       to tolerance.
    5. The cells the fusion never touches are cells where the built child
       contributed nothing, and there the parent's word IS the sibling's word.
       That covers the bins a small child put no rows in and, when the feature
       set is narrowed, the whole inactive slices -- which are zero in the
       parent because the parent was itself zeroed and accumulated under the
       same active set. A tree that narrowed its features BETWEEN levels would
       break that last clause; this plane refuses `feature_fraction_bylevel`,
       and that refusal is now load-bearing here as well as where it is
       written.
    6. Overflow is no nearer than it already is. The minuend is a parent cell
       the shipping arm already accumulates, and the difference is a sum over a
       subset of the same rows, so every intermediate here is a value the
       shipping arm also holds.

    So the arm moves no bits, and the only thing it changes is how many rows
    the level reads."""
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var hs = Int(hist_size)
    var slot = Int(block_idx.x)
    var g_tile = Int(block_idx.y)

    var meta = stack_allocation[
        8, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sg = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    if tid == 0:
        var n_pairs = _live_item_pairs(items, Int(n_items))
        var k = _item_for_tile(items, Int(n_items), g_tile)
        var base = k * ITEM_WORDS
        # Which of the pair this item is, and whether it is the one built.
        # `meta[0] = -1` is "this block has nothing to do", and it covers both
        # the derived sibling and any item past the live prefix.
        var build = False
        var sub_slot = 0
        if k < 2 * n_pairs:
            var partner = k + n_pairs if k < n_pairs else k - n_pairs
            var n_self = items[unsafe_offset = base + ITEM_COUNT][0]
            var n_other = items[
                unsafe_offset = partner * ITEM_WORDS + ITEM_COUNT
            ][0]
            # The right child of the pair takes ties, for the reason the
            # zeroing pass gives.
            if k < n_pairs:
                build = n_self < n_other
            else:
                build = n_self <= n_other
            sub_slot = Int(
                items[unsafe_offset = partner * ITEM_WORDS + ITEM_OUT][0]
            )
        if not build:
            meta[unsafe_offset=0] = Int32(-1)
        else:
            meta[unsafe_offset=0] = Int32(k)
            meta[unsafe_offset=1] = items[unsafe_offset = base + ITEM_BEGIN][0]
            meta[unsafe_offset=2] = items[unsafe_offset = base + ITEM_COUNT][0]
            meta[unsafe_offset=3] = items[
                unsafe_offset = base + ITEM_ROWS_PER_TILE
            ][0]
            meta[unsafe_offset=4] = Int32(
                g_tile - Int(items[unsafe_offset = base + ITEM_TILE_BEGIN][0])
            )
            meta[unsafe_offset=5] = items[unsafe_offset = base + ITEM_PLANE][0]
            meta[unsafe_offset=6] = items[unsafe_offset = base + ITEM_OUT][0]
            meta[unsafe_offset=7] = Int32(sub_slot)

    # Block-uniform launch scalars, exactly as on the non-subtracting twin.
    var quant = Int(use_quant) != 0
    var celide = Int(const_hess) != 0

    var b = tid
    while b < nb:
        sg[unsafe_offset=b] = 0
        if not celide:
            sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var k = Int(meta[unsafe_offset=0][0])
    # Uniform across the block: `meta[0]` is one word one thread wrote and
    # every thread reads, so the whole threadgroup leaves together and no
    # barrier below is reached by a subset of it.
    if k < 0:
        return
    var begin = Int(meta[unsafe_offset=1][0])
    var count = Int(meta[unsafe_offset=2][0])
    var rows_per_tile = Int(meta[unsafe_offset=3][0])
    var t = Int(meta[unsafe_offset=4][0])
    var plane_base = Int(meta[unsafe_offset=5][0]) * nr
    var out_slot = Int(meta[unsafe_offset=6][0])
    var sub_slot = Int(meta[unsafe_offset=7][0])

    var f = Int(feat_ids[unsafe_offset = k * Int(feat_stride) + slot][0])
    var g_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_G][0]
    var h_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_H][0]
    # `Float32(1.0) * h_scale` is exactly `h_scale`; see the twin above.
    var hq_const = Int32(round(Float32(1.0) * h_scale))

    var tile_begin = t * rows_per_tile
    var tile_end = tile_begin + rows_per_tile
    if tile_end > count:
        tile_end = count

    var col = f * nr
    var j = tile_begin + tid
    while j < tile_end:
        var r = Int(rows[unsafe_offset = begin + j][0])
        var bin = Int(bins[unsafe_offset = col + r])
        var gqv = Int32(0)
        var hqv = Int32(0)
        if quant:
            gqv = gq[unsafe_offset = 2 * (plane_base + r)][0]
            if not celide:
                hqv = gq[unsafe_offset = 2 * (plane_base + r) + 1][0]
        else:
            gqv = Int32(
                round(grad[unsafe_offset = plane_base + r][0] * g_scale)
            )
            if not celide:
                hqv = Int32(
                    round(hess[unsafe_offset = plane_base + r][0] * h_scale)
                )
        _ = Atomic.fetch_add(sg.unsafe_offset(bin), gqv)
        if not celide:
            _ = Atomic.fetch_add(sh.unsafe_offset(bin), hqv)
        _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        j += block_dim.x
    barrier()

    var slice_base = out_slot * N_PLANES * hs + f * nb
    var sub_base = sub_slot * N_PLANES * hs + f * nb
    b = tid
    while b < nb:
        if sc[unsafe_offset=b][0] != 0:
            var vg = sg[unsafe_offset=b][0]
            var vc = sc[unsafe_offset=b][0]
            var vh = hq_const * vc if celide else sh[unsafe_offset=b][0]
            _ = Atomic.fetch_add(out_hist.unsafe_offset(slice_base + b), vg)
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(slice_base + hs + b), vh
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(slice_base + 2 * hs + b), vc
            )
            _ = Atomic.fetch_add(out_hist.unsafe_offset(sub_base + b), -vg)
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(sub_base + hs + b), -vh
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(sub_base + 2 * hs + b), -vc
            )
        b += block_dim.x


# --- The pair-indexed grid ------------------------------------------------
#
# `_batch_hist_atomic_subtract_kernel` above dispatches ONE THREADGROUP PER
# STAGED ITEM PER TILE PER FEATURE SLOT and then elects, inside the kernel,
# which of a pair is built. Two multipliers of waste follow and neither is
# data dependent, so both can be removed on the host:
#
# - Half the LIVE items are the derived sibling and return without
#   accumulating. That is the election, and it is the half the docstring
#   above describes.
# - Every item past the live prefix is dead and also returns. That half is
#   not described anywhere and it is the larger one: the plan is staged at
#   `1 << max_depth` items for the whole tree (`stage_device_plan`, called
#   once from `histogram_gpu.stage_desc_level_plan` with `budget`), while a
#   level at depth `l` fills only `2 << l` of them. At depth 6 that is 64
#   staged rows against 2 filled ones at level zero.
#
# Multiply them out over a depth-6 tree that skips the last level's build:
# levels 0 to 4 dispatch `5 * 64 = 320` item-slots' worth of threadgroups and
# build `1 + 2 + 4 + 8 + 16 = 31` of them, which is **9.3 threadgroups that
# do nothing for every one that does**. That ratio was measured independently
# by an earlier lane before this arithmetic was written down, which is the
# reason it is quoted here rather than derived and left at that.
#
# The fix is to index `grid.y` BY PAIR. At a level of width `2L` there are
# exactly `L` pairs and exactly one built child per pair, always; only WHICH
# child depends on the data, and that is a lookup and not a count. So the
# host launches `L * tiles` rows of grid instead of `plan_items * tiles`, the
# kernel resolves its pair by division, and the election that decides which
# member of the pair to build is performed EXACTLY where and how it was
# before. Nothing about the answer moves; see the kernel's docstring.


comptime PAIR_GRID_VAR = "MOJOTREES_GPU_HIST_PAIR_GRID"
"""`1` launches the subtracting level build over a pair-indexed grid.

Default off, and off it changes nothing at all: the same kernel symbol, the
same grid, the same launch count. On it selects
`_batch_hist_pair_subtract_kernel` at `grid.y = pairs * tiles`, which is
bit-identical output over between 2x and 64x fewer threadgroups depending on
the level. Still ONE launch, so
`gpu_resident_round.oblivious_launch_census(6)` is 62 either way, which is a
precondition of this plane and not a happy result."""

comptime PAIR_TILES_VAR = "MOJOTREES_GPU_HIST_PAIR_TILES"
"""Row tiles per item on the pair-indexed arm, zero meaning the staged
geometry. `MOJOTREES_GPU_HIST_ROW_SPLIT` for that kernel, named separately so
the two arms can be swept in one process without either one's sweep moving
the other's control.

The occupancy argument is `plan_row_split_requested`'s, word for word, and
the reason it is worth re-offering here is that the pair grid is what makes
it affordable: raising the tile count on the item-indexed grid multiplies the
dead threadgroups by the same factor it multiplies the live ones."""


def _batch_hist_pair_subtract_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    scales: MutPointer[Float32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_items: Int32,
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
    feat_stride: Int32,
    tiles_per_item: Int32,
    split_rows: Int32,
    use_quant: Int32,
    const_hess: Int32,
):
    """`_batch_hist_atomic_subtract_kernel` over a grid indexed by PAIR.

    The identical row loop, the identical threadgroup accumulation, the
    identical fused flush, and the identical election. What changes is the
    meaning of `block_idx.y` and therefore how many threadgroups the level
    dispatches; what does not change is which item is built, from which rows,
    into which slot, by which arithmetic.

    WHY THE OUTPUT IS BIT-IDENTICAL, AND THE ELECTION IS THE WHOLE OF IT
    --------------------------------------------------------------------
    1. **The election is unmoved.** The shipping kernel resolves a block's
       item `k`, finds its partner (`k + L` when `k < L`, else `k - L`), and
       builds when `n_self < n_other` for a left child and when
       `n_self <= n_other` for a right child. Exactly one member of a pair
       satisfies its own test: the left child builds iff `n_left < n_right`,
       and the right child builds iff `not (n_left < n_right)`, ties
       included, because `n_right <= n_left` is the negation of
       `n_left < n_right` over Int32. This kernel evaluates that single
       predicate once, `build_left = n_left < n_right`, and takes the same
       branch. **Ties still go to the right child**, which is the case
       `_batch_copy_back_zero_subtract_kernel` prepared the slots for and the
       one place a disagreement would be silent rather than loud.
    2. **The work is unmoved.** The elected item's `ITEM_BEGIN`,
       `ITEM_COUNT`, `ITEM_PLANE`, `ITEM_OUT`, its feature id, its scales and
       its partner's `ITEM_OUT` are the same words read from the same table.
    3. **The tiles are unmoved.** `stage_device_plan` gives every item the
       same `tiles` and writes `ITEM_TILE_BEGIN = k * tiles`, so the shipping
       kernel's `t = g_tile - ITEM_TILE_BEGIN[k]` ranges over `[0, tiles)`
       for the elected item and this kernel's `t = g_tile - p * tiles` ranges
       over the same interval once. The set of `(item, tile)` pairs that run
       is identical, so the multiset of row contributions is identical, and
       every accumulator on both arms is exact integer.
    4. **The blocks that stop stopping earlier changes nothing.** A block
       whose pair is past the live prefix, or whose elected child holds no
       rows, would on the shipping arm have zeroed a threadgroup histogram,
       reached the flush, found every `sc[b]` zero and stored nothing. Here
       it returns before the allocation instead. `_plan_hist_kernel` already
       makes the same removal for the same reason.

    THE ROOT, AND WHY IT NEEDS NO CASE
    ----------------------------------
    The root's histogram never reaches this kernel. It is built once per tree
    by `GpuHistogramBuilder.enqueue_leaf(0, resident_slot=0)` on the
    single-leaf path, before the level loop opens, and every batch this
    kernel serves is a batch of a level's CHILDREN. The shallowest batch is
    the root's own split, which is `L = 1` pair and two children, and one
    pair is a legal grid. There is no level with zero pairs: a level that
    committed nothing kills the whole plan, `_live_item_pairs` answers zero,
    and every block returns at the first comparison, which is the same thing
    `ITEM_DEAD` already buys the shipping arm.

    AN ODD WIDTH CANNOT ARISE
    -------------------------
    `_commit_level_kernel` writes `2L` items as `L` left children followed by
    `L` right children, so the live prefix is even by construction;
    `enqueue_device_plan_batch_fused_subtracting` refuses an odd staged width
    outright; and `_live_item_pairs` floors, so an odd prefix would drop its
    last item here exactly as it already does on the shipping arm. No
    behavior is invented for a state that cannot occur.

    `tiles_per_item` and `split_rows` are `_plan_hist_kernel`'s two, with the
    same meanings and the same neutral values: `split_rows = 0` keeps the
    staged tile LENGTH, which reproduces the shipping geometry when the
    staged tile count is one. `use_quant` and `const_hess` mean what they
    mean on `_batch_hist_atomic_kernel` and are argued there."""
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var hs = Int(hist_size)
    var slot = Int(block_idx.x)
    var tpi = Int(tiles_per_item)
    if tpi < 1:
        tpi = 1

    # The PAIR and the local tile, by division rather than by search. Both
    # are functions of `block_idx.y` alone, so every thread of the block gets
    # the same answer and none of what follows needs threadgroup memory or a
    # barrier to publish it.
    var g_tile = Int(block_idx.y)
    var p = g_tile // tpi
    var t = g_tile - p * tpi

    # Run redundantly by every thread rather than by thread zero: six loads
    # from one cache line, every thread reading the same address, against a
    # barrier and eight words of threadgroup memory on the shipping arm. The
    # same trade `_plan_hist_kernel` already makes.
    var n_pairs = _live_item_pairs(items, Int(n_items))
    if p >= n_pairs:
        return

    var lbase = p * ITEM_WORDS
    var rbase = (p + n_pairs) * ITEM_WORDS
    var n_left = Int(items[unsafe_offset = lbase + ITEM_COUNT][0])
    var n_right = Int(items[unsafe_offset = rbase + ITEM_COUNT][0])
    # The one predicate. Ties go to the right child; see the docstring.
    var build_left = n_left < n_right
    var k = p if build_left else p + n_pairs
    var partner = (p + n_pairs) if build_left else p
    var base = k * ITEM_WORDS
    var count = n_left if build_left else n_right
    if count <= 0:
        return

    var out_slot = Int(items[unsafe_offset = base + ITEM_OUT][0])
    var sub_slot = Int(
        items[unsafe_offset = partner * ITEM_WORDS + ITEM_OUT][0]
    )

    var rpt = Int(items[unsafe_offset = base + ITEM_ROWS_PER_TILE][0])
    if Int(split_rows) != 0:
        rpt = (count + tpi - 1) // tpi
    if rpt < 1:
        rpt = 1
    var tile_begin = t * rpt
    if tile_begin >= count:
        return
    var tile_end = tile_begin + rpt
    if tile_end > count:
        tile_end = count

    var begin = Int(items[unsafe_offset = base + ITEM_BEGIN][0])
    var plane_base = Int(items[unsafe_offset = base + ITEM_PLANE][0]) * nr
    var f = Int(feat_ids[unsafe_offset = k * Int(feat_stride) + slot][0])
    var g_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_G][0]
    var h_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_H][0]
    # `Float32(1.0) * h_scale` is exactly `h_scale`; see the shipping twin.
    var hq_const = Int32(round(Float32(1.0) * h_scale))
    var quant = Int(use_quant) != 0
    var celide = Int(const_hess) != 0

    var sg = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    var b = tid
    while b < nb:
        sg[unsafe_offset=b] = 0
        if not celide:
            sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var col = f * nr
    var j = tile_begin + tid
    while j < tile_end:
        var r = Int(rows[unsafe_offset = begin + j][0])
        var bin = Int(bins[unsafe_offset = col + r])
        var gqv = Int32(0)
        var hqv = Int32(0)
        if quant:
            gqv = gq[unsafe_offset = 2 * (plane_base + r)][0]
            if not celide:
                hqv = gq[unsafe_offset = 2 * (plane_base + r) + 1][0]
        else:
            gqv = Int32(
                round(grad[unsafe_offset = plane_base + r][0] * g_scale)
            )
            if not celide:
                hqv = Int32(
                    round(hess[unsafe_offset = plane_base + r][0] * h_scale)
                )
        _ = Atomic.fetch_add(sg.unsafe_offset(bin), gqv)
        if not celide:
            _ = Atomic.fetch_add(sh.unsafe_offset(bin), hqv)
        _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        j += block_dim.x
    barrier()

    var slice_base = out_slot * N_PLANES * hs + f * nb
    var sub_base = sub_slot * N_PLANES * hs + f * nb
    b = tid
    while b < nb:
        if sc[unsafe_offset=b][0] != 0:
            var vg = sg[unsafe_offset=b][0]
            var vc = sc[unsafe_offset=b][0]
            var vh = hq_const * vc if celide else sh[unsafe_offset=b][0]
            _ = Atomic.fetch_add(out_hist.unsafe_offset(slice_base + b), vg)
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(slice_base + hs + b), vh
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(slice_base + 2 * hs + b), vc
            )
            _ = Atomic.fetch_add(out_hist.unsafe_offset(sub_base + b), -vg)
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(sub_base + hs + b), -vh
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(sub_base + 2 * hs + b), -vc
            )
        b += block_dim.x


# --- The lean device-plan kernel, and its three knobs ---------------------
#
# `_batch_hist_atomic_kernel` and its subtracting twin above are the SHIPPING
# accumulation for the oblivious level build. This is one kernel that
# subsumes both and adds three orthogonal knobs, each behind its own
# environment switch and each default off, so the orchestrator can measure
# them separately and together against the pair above.
#
# Everything here is exact integer arithmetic over the same fixed-point
# planes, so no knob and no combination of knobs can move a bit. The argument
# is written out leg by leg in the kernel's docstring rather than asserted.

comptime PLAN_LEAN_VAR = "MOJOTREES_GPU_HIST_LEAN"
"""Selects this kernel with every knob neutral, which isolates the three
STRICTLY-LESS-WORK removals it makes against the shipping pair: no
threadgroup `meta` block and no barrier to publish it, no binary search for
the item, and an early return that happens BEFORE the shared histogram is
zeroed rather than after it. See `_plan_hist_kernel`."""

comptime PLAN_PRIVATE_VAR = "MOJOTREES_GPU_HIST_PRIVATE"
"""Private (replicated) threadgroup accumulators. `1` asks for as many copies
as the shipping kernel's own threadgroup footprint already pays for, which is
`(MAX_BINS + PLAN_COPY_SLACK) / (n_bins + 1)` floored and is therefore FREE;
an explicit integer above one asks for that many copies and spends threadgroup
memory to get them."""

comptime PLAN_ROW_SPLIT_VAR = "MOJOTREES_GPU_HIST_ROW_SPLIT"
"""Row tiles per plan item, derived from the ITEM'S OWN row count on the
device instead of from the tree's row bound on the host. The shallow-depth
occupancy multiplier; see `plan_row_split_requested` for why the host cannot
compute this number and why the shipping geometry leaves most of its tiles
empty at depth."""

comptime PLAN_GROUP_VAR = "MOJOTREES_GPU_HIST_GROUP"
"""Feature slots one threadgroup owns. `1` is the shipping shape. Above one,
the row's gradient and hessian are read once and spent for every feature in
the group, which is the same `GROUP` parameter
`gpu_active_rows._range_hist_atomic_kernel` already carries on the SINGLE-LEAF
path and that the batched path has never had."""

comptime PLAN_MAX_GROUP = 4
"""Feature slots one threadgroup may own here.

Four rather than the single-leaf family's sixteen, and the bound is
threadgroup memory rather than taste: a group of `G` features holding `C`
private copies needs `G * C * 3 * (n_bins + 1)` Int32, which at 256 bins and
one copy is about 3.1 KB per feature. Four is 12.4 KB, still under the 16 KB
`gpu_tiling.FALLBACK_SHARED_MEMORY_PER_BLOCK` a device that reports nothing
is assumed to have. Eight would be 24.8 KB, which the measured M4 has and a
conservative device may not, so it is a one-line extension behind a
capability check rather than a default."""

comptime PLAN_COPY_SLACK = 8
"""Spare bins per plane in a unit, so that padding the copy stride does not
cost a copy.

**The pad word on the copy stride is load-bearing and is the difference
between replication working and replication buying nothing.** Threadgroup
memory is banked in 32 four-byte banks on every backend this project targets.
With an unpadded stride the copies of bin `b` sit `n_bins` words apart, and
`n_bins` is a multiple of 32 at every bin count this package uses, so every
copy of bin `b` would land in the SAME bank: the replication would convert an
atomic conflict into a bank conflict and serialize just as hard. At stride
`n_bins + 1` the copies of a bin walk the banks one at a time.

The slack is what stops that pad from eating a whole copy at the wide bin
counts, and eight is the smallest value that does it for every ladder entry
this package bins at. Without it a unit is `MAX_BINS + 1 = 257` bins wide and
two 128-bin copies want 258, so 128 bins would get ONE copy and the switch
would silently do nothing there. With it a unit is 264 bins wide and the free
copy counts are 1 at 256 bins, 2 at 128, 4 at 64, 8 at 32 and 15 at 16, which
is `264 / (n_bins + 1)` floored."""

comptime PLAN_UNIT_CELLS = N_PLANES * (MAX_BINS + PLAN_COPY_SLACK)
"""Int32 cells one accumulator unit occupies: three planes, each as wide as
the largest bin count plus `PLAN_COPY_SLACK`.

A *unit* is one three-plane histogram for one (feature-in-group, copy) pair.
792 cells, 3,168 bytes."""

comptime PLAN_UNITS_NARROW = 1
comptime PLAN_UNITS_WIDE = 4
"""The two threadgroup budgets this kernel is instantiated at, in units.

Two rather than a ladder. `PLAN_UNITS_NARROW` is 3,168 bytes against the
3,104 the shipping kernel already allocates (three `MAX_BINS` Int32 planes at
3,072, plus its eight-word `meta` block at 32), so the narrow instantiation is
footprint-neutral to within 64 bytes and every free replication lives inside
it. `PLAN_UNITS_WIDE` is 12,672 bytes, still under the 16,384
`gpu_tiling.FALLBACK_SHARED_MEMORY_PER_BLOCK` a device that reports nothing is
assumed to have, and it is the one that trades residency for either a wider
feature group or more copies than the bin count leaves room for."""


def _plan_env_int(name: String, default: Int) -> Int:
    """`parallel._env_int`, restated here rather than imported: `parallel`
    is a CPU module and this file is on the device side of the backend."""
    var s = getenv(name)
    if s.byte_length() == 0:
        return default
    try:
        var n = Int(s)
        if n < 0:
            return default
        return n
    except:
        return default


def plan_group_requested() -> Int:
    """`MOJOTREES_GPU_HIST_GROUP` as a feature-group width, one meaning the
    shipping one-feature-per-threadgroup shape.

    WHY THIS IS THE LARGEST OF THE THREE, BY COUNTING BYTES
    ------------------------------------------------------
    A threadgroup owning one feature slot reads, per row of its tile, four
    bytes of row index, four of gradient, four of hessian and one bin byte, so
    a `(row, feature)` visit costs 13 bytes of load traffic. Twelve of those
    thirteen bytes are re-read by every other feature's threadgroup, because
    `grid.x` is the feature axis and the gradient plane does not depend on the
    feature. A group of `G` slots reads them once and spends them `G` times,
    so a visit costs `1 + 12 / G` bytes: 7 at `G = 2`, 4 at `G = 4`. That is
    a factor of 3.25 off the traffic at `G = 4` and it is arithmetic over the
    loads in the row loop, not a prediction.

    What it costs is residency: `G * 3 * (n_bins + 1)` Int32 of threadgroup
    memory instead of `3 * (n_bins + 1)`, so 12.3 KB instead of 3.1 KB at 256
    bins, which is roughly four times fewer resident threadgroups per core.
    Whether the traffic or the residency wins is exactly what an interleaved
    A/B settles and is why this is a switch rather than a default.

    The same parameter on the SINGLE-LEAF kernel
    (`gpu_active_rows._range_hist_atomic_kernel`, comptime `GROUP`) was
    measured at **1.17x for group 2 over group 1** on an Apple M4, and that
    kernel's docstring records that groups 8 and 16 have never been launched.
    The batched kernels this file ships have never had the parameter at all,
    which is why the oblivious level build reads its gradients `n_slots` times
    per level and the leaf-wise path does not."""
    var n = _plan_env_int(PLAN_GROUP_VAR, 1)
    if n < 1:
        return 1
    return n


def plan_row_split_requested() -> Int:
    """`MOJOTREES_GPU_HIST_ROW_SPLIT` as row tiles per item, zero meaning the
    shipping geometry.

    THE SHALLOW-DEPTH OCCUPANCY MULTIPLIER, AND WHAT IT IS REALLY FIXING
    -------------------------------------------------------------------
    `stage_device_plan` fixes one tile geometry for the whole tree, because it
    runs once per tree and the host never learns a level's row counts without
    a synchronization. It derives it from `derive_tiling` against the ROW BOUND
    with `n_slots` features already occupying `grid.x`, and
    `gpu_tiling.row_tile_floor` then asks for `ceil(target_blocks / n_slots)`
    tiles. On the ten-core M4 `target_blocks` is 80, so **at any feature count
    at or above 80 that term is 1**: every item gets exactly one tile whose
    length is the whole row bound, and one threadgroup per feature scans a
    whole leaf however many rows it holds.

    Two consequences, and the second is the one worth the switch.

    - At depth the item is small and the tile is the size of the ROOT, so the
      single tile is mostly empty and the level's parallelism is `built_items *
      n_slots` threadgroups. That grows with depth, so the SHALLOWEST level is
      the thin one: at level zero with sibling subtraction there is exactly one
      built child, and the level is `n_slots` threadgroups wide.
    - The level's latency is set by the LARGEST built child, not by the average
      one, because each child is one threadgroup per feature. Oblivious splits
      are lopsided by construction, so a level whose total work is spread over
      thirty-two children can still wait on one of them.

    This switch splits every item into `T` tiles of `ceil(count / T)` rows
    each, with `count` read from the plan on the device, so the split follows
    the item's own size at every depth instead of the root's. `T = 1` is the
    control arm: it reproduces the shipping geometry exactly whenever the
    staged tile count is one, which on this machine at 50 or more features it
    is.

    **The honest expectation, stated before the measurement rather than after
    it.** At 100 features `n_slots` alone already puts 100 threadgroups of 256
    threads on a ten-core device, which is 25,600 threads, and the shipping
    threadgroup footprint of 3.0 KB admits about ten blocks per core, so the
    grid is already resident in roughly one wave. The multiplier therefore has
    NO occupancy to win on that shape and what remains is wave quantization
    and the largest-child term above. It is expected to pay where `n_slots` is
    small -- a narrow real-data frame, or `feature_fraction` well below one --
    and to be a null or a small loss at 100 features, where the extra tiles
    are extra threadgroups over the same rows. That is the arithmetic; the
    clock is the orchestrator's."""
    return _plan_env_int(PLAN_ROW_SPLIT_VAR, 0)


def pair_grid_requested() -> Bool:
    """Whether the subtracting level build takes the pair-indexed grid.

    Read here rather than taken as a parameter, for the reason
    `oblivious_subtract_requested` gives: the decision belongs beside the
    kernels it selects between. Default off under this package's rule for a
    path no benchmark has priced, and it is bit-identical, so LANE_RULES rule
    5 flips it the moment a measurement says so. See `PAIR_GRID_VAR` and
    `_batch_hist_pair_subtract_kernel`."""
    return getenv(PAIR_GRID_VAR) == "1"


def pair_grid_tiles() -> Int:
    """`MOJOTREES_GPU_HIST_PAIR_TILES` as row tiles per item, zero meaning
    the staged tile count and the staged tile LENGTH.

    `plan_row_split_requested` for the pair-indexed arm, and the occupancy
    argument is that function's word for word. It has its own name so a sweep
    of one arm cannot move the other arm's control."""
    return _plan_env_int(PAIR_TILES_VAR, 0)


def plan_private_copies_requested(n_bins: Int) -> Int:
    """`MOJOTREES_GPU_HIST_PRIVATE` as a count of private accumulator copies,
    one meaning the shipping single shared histogram.

    `1` asks for the free answer: as many copies as fit in the threadgroup
    footprint the shipping kernel ALREADY pays, which allocates three
    `MAX_BINS`-wide planes whatever the dataset's bin count is. At 64 bins that
    is four copies for nothing; at 256 bins it is one, and the switch is a
    no-op. An explicit integer above one asks for that many and spends
    threadgroup memory, which the caller resolves against `PLAN_UNITS_WIDE`.

    WHAT THIS REMOVES, AND WHAT IT HONESTLY DOES NOT
    ------------------------------------------------
    CatBoost accumulates per-warp private histograms with no atomics at all
    and reduces them afterwards. **We cannot reach zero atomics and the reason
    is a language fact rather than a design choice.** Zero atomics requires one
    private copy per thread that can collide, Mojo 1.0 has no warp primitives
    at all (only `block`), so the smallest unit whose threads this file can
    reason about is the whole threadgroup, and a copy per thread at 256 threads
    and 256 bins is 3.1 KB times 256, which is 786 KB of threadgroup memory
    against a 32 KB budget. So this knob REDUCES contention by the replication
    factor; it does not eliminate the atomic.

    The contention it reduces is real and is worst exactly where the
    replication is free. With a 32-lane group accumulating into `n_bins` bins,
    the expected number of lanes sharing a bin scales as `32 / n_bins`, so a
    16-bin categorical feature has an average multiplicity of two and a
    worst-case run far above that, while a 256-bin continuous feature is mostly
    conflict-free. A 16-bin dataset gets fifteen free copies out of the
    footprint already allocated.

    WHY IT CANNOT MOVE A BIT, LEG BY LEG
    ------------------------------------
    1. A row's contribution to a bin is `Int32(round(grad[r] * g_scale))`,
       `Int32(round(hess[r] * h_scale))` and `1`. All three are functions of
       the row and of the tree's two fixed-point scales alone. Which copy a
       thread accumulates into is not an argument to any of them.
    2. Every visit is accumulated exactly once, into exactly one copy, by
       exactly the thread that gathered it. The copies partition the block's
       threads (`tid % copies`), so no visit is duplicated and none is dropped.
    3. The flush sums the copies of a bin with Int32 addition. Integer addition
       is associative and commutative, so the sum over copies of the sums
       within copies is the sum over visits, whatever order either takes. This
       is the same property the shipping kernels already rely on to claim that
       a batched build equals a single-leaf one and that an atomic fold equals
       a tiled reduction.
    4. The `vc != 0` flush guard has the same meaning it had: counts are
       non-negative, so the summed count is zero exactly when no visit landed
       in that bin, which is exactly when the shipping kernel's `sc[b]` is
       zero.
    5. Overflow is no nearer. Every partial sum here is a sub-sum of a value
       the shipping kernel already forms in one accumulator, and Int32 addition
       agrees modulo 2^32 however it is grouped, so even a hypothetical
       overflow would land on the same bits.

    So this switch selects an amount of threadgroup contention and never a
    number. Under LANE_RULES rule 5 that means a measured win flips it in the
    session that measures it."""
    var s = getenv(PLAN_PRIVATE_VAR)
    if s.byte_length() == 0 or s == "0":
        return 1
    var stride = n_bins + 1
    var free_copies = PLAN_UNIT_CELLS // (N_PLANES * stride)
    if free_copies < 1:
        free_copies = 1
    if s == "1":
        return free_copies
    var n = _plan_env_int(PLAN_PRIVATE_VAR, 1)
    if n < 1:
        return 1
    return n


def plan_lean_requested() -> Bool:
    """Whether the device-plan build runs `_plan_hist_kernel` rather than the
    shipping `_batch_hist_atomic_kernel` pair.

    True when `MOJOTREES_GPU_HIST_LEAN=1`, and also true whenever any of the
    three knobs is asked for, so that a caller setting one knob does not also
    have to know that the knob lives on a different kernel. All four are
    default off, which is this package's rule for work no benchmark has priced
    yet; the flip is rule 5's and belongs to the session that measures it."""
    if getenv(PLAN_LEAN_VAR) == "1":
        return True
    if plan_row_split_requested() > 0:
        return True
    if plan_group_requested() > 1:
        return True
    var p = getenv(PLAN_PRIVATE_VAR)
    return p.byte_length() != 0 and p != "0"


def _plan_hist_kernel[
    UNITS: Int
](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    scales: MutPointer[Float32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_items: Int32,
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
    feat_stride: Int32,
    tiles_per_item: Int32,
    split_rows: Int32,
    n_copies: Int32,
    group_width: Int32,
    subtract: Int32,
    use_quant: Int32,
    const_hess: Int32,
):
    """One accumulation kernel for the device-written plan, with the two
    CatBoost ideas and the sibling subtraction as knobs rather than as arms.

    It replaces both `_batch_hist_atomic_kernel` and
    `_batch_hist_atomic_subtract_kernel` on the device-plan path and produces
    bit-for-bit what either of them produces, at every setting of every knob.

    WHAT IT REMOVES UNCONDITIONALLY, WHICH IS WHY `MOJOTREES_GPU_HIST_LEAN=1`
    IS A MEASURABLE ARM ON ITS OWN
    ------------------------------------------------------------------------
    Three removals, all of them strictly less work for the identical result,
    all of them available because a device-written plan gives every item the
    SAME number of tiles (`stage_device_plan` writes `ITEM_TILE_BEGIN = k *
    tiles` and `ITEM_TILES = tiles` for every `k`).

    1. **No binary search for the item.** With uniform tiles the owning item
       is `block_idx.y / tiles_per_item` and the local tile is the remainder.
       `_item_for_tile` costs six dependent Int32 loads at the 64-item width
       this mode stages; a division costs no loads at all. The shipping kernels
       need the search because `_stage_plan` may write non-uniform tile counts
       from the host, and that path still uses them.
    2. **No threadgroup `meta` block and no barrier to publish it.** The
       shipping kernels have thread zero resolve the item into eight words of
       threadgroup memory because the search was expensive enough to be worth
       doing once. A division is not, so every thread resolves it redundantly
       from block-uniform values, which are broadcast loads.
    3. **The early return happens BEFORE the shared histogram is zeroed.**
       This is the largest of the three and the count is the argument. An
       oblivious level plan is staged at `1 << max_depth` items and a level
       fills only `2 * L` of them, so at depth 6 the levels leave 62, 60, 56,
       48, 32 and 0 items dead, which is **258 dead item-slots per tree**;
       with sibling subtraction on, half of the live items are derived rather
       than built, which is 63 more. Every one of those blocks currently zeroes
       three `n_bins`-wide planes and walks the whole flush loop before
       discovering it had nothing to do, because the shipping kernels place
       their return after the barrier "so a derived item's block costs
       precisely what a dead item's block already costs today". That sentence
       is true and prices the wrong thing: the dead block's cost was never
       free. At 100 feature slots that is 32,100 threadgroups per tree zeroing
       3.0 KB apiece to accumulate nothing. Here they return on their first
       comparison, before the `stack_allocation` is touched.

    The return is still block-uniform, which is the property that matters: it
    is taken on `block_idx.y`, `n_slots`, and words of the item table that
    every thread of the block reads identically, so the whole threadgroup
    leaves together and no barrier below is reached by a subset of it.

    THE THREE KNOBS
    ---------------
    `n_copies` replicates the threadgroup accumulator to cut atomic
    contention; see `plan_private_copies_requested`, which also states plainly
    that this reduces atomics rather than removing them and why Mojo 1.0
    cannot remove them.

    `tiles_per_item` with `split_rows` set derives each item's row tile from
    the ITEM'S own count rather than from the tree's row bound, which is the
    shallow-depth occupancy multiplier; see `plan_row_split_requested`. With
    `split_rows` clear the row tile is read from `ITEM_ROWS_PER_TILE` and the
    geometry is the shipping one.

    `group_width` gives one threadgroup several feature slots so a row's
    gradient pair is read once and spent for all of them; see
    `plan_group_requested` for the byte arithmetic.

    THE SHARED LAYOUT
    -----------------
    One flat allocation of `UNITS * PLAN_UNIT_CELLS` Int32. A *unit* is one
    three-plane histogram for one (feature-in-group, copy) pair, and unit
    `u = gi * copies + c` starts at `u * N_PLANES * stride` with
    `stride = n_bins + 1`. The pad word per plane is what keeps the copies of
    a bin out of one memory bank; `PLAN_UNIT_CELLS` argues it. The caller
    guarantees `group_width * n_copies * N_PLANES * stride <= UNITS *
    PLAN_UNIT_CELLS` and raises rather than clamping, because a silent clamp
    would drop a feature's accumulation and produce a wrong histogram with no
    symptom.

    WHY NO KNOB AND NO COMBINATION OF KNOBS CAN MOVE A BIT
    -----------------------------------------------------
    Leg by leg, and every leg is exact integer arithmetic.

    1. A visit's three contributions are `Int32(round(grad[r] * g_scale))`,
       `Int32(round(hess[r] * h_scale))` and `1`. Every one is a function of
       the row and of the tree's two scales, and no knob is an argument to any
       of them. The two floating-point products are the only floating-point
       operations in the kernel and they are written exactly as the shipping
       kernels write them, in the same order, with no multiply moved relative
       to an add, so there is no FMA contraction difference to argue about.
       **`use_quant` moves WHERE those two products are evaluated and not
       WHAT they evaluate to**: `_batch_quantize_kernel` forms the identical
       expression once per row per round instead of once per (row, feature)
       visit, and hoisting an expression out of a loop whose other operands
       it does not depend on cannot change its value. **`const_hess`
       reconstructs the second of the three as `hq_const * vc`**, which is
       exact because `1.0f * h_scale` is `h_scale` and Int32 addition and
       multiplication agree modulo 2^32; the declaration that every hessian
       is 1.0 belongs to the caller and `set_constant_hessian` says so.
    2. **Every visit happens exactly once.** Over the feature axis: block
       `block_idx.x` owns slots `[bx * G, bx * G + G)` clamped to `n_slots`, so
       the blocks partition the slots and a tail block owns only what remains.
       Over the row axis: tiles `t = 0 .. T-1` cover `[t * rpt, min((t+1) *
       rpt, count))`, and `rpt = ceil(count / T)` gives `T * rpt >= count`, so
       the tiles partition `[0, count)` exactly -- no row twice, none dropped,
       at any `T`. Within a tile, thread `tid` takes `tile_begin + tid` and
       every `block_dim.x`-th row after it, which is the shipping walk. Over
       the copies: a visit goes into copy `tid % copies`, decided by the thread
       that gathered it.
    3. The reduction is Int32 addition, which is associative and commutative,
       so summing copies then folding into global gives the same word as
       folding every visit into global directly. This is the property the
       shipping kernels already stand on.
    4. **The subtraction is exact and its election is the same election.**
       With `subtract` set, a block builds only the smaller child of its pair
       and folds the negation of every cell it adds into the sibling's slot,
       under the same guard, with the same integer atomics. The election is
       `n_self < n_other` for a left child and `n_self <= n_other` for a right
       child, which is ties-to-the-right and is character for character the
       rule `_batch_copy_back_zero_subtract_kernel` prepared the pair under.
       The two children of an oblivious split partition the parent's rows
       exactly (`gpu_tree_tables._commit_level_kernel` routes rows by the same
       rule it derived the counts from), so `parent - built` is the exact
       integer sum over the derived child's own rows. The full argument is at
       `_batch_hist_atomic_subtract_kernel` and is not restated.
    5. An item with `count <= 0` returns before doing anything, and that is
       correct rather than merely cheap: `ITEM_DEAD` is negative and must not
       be touched at all, and a live leaf holding no rows has an all-zero
       histogram which the zeroing pass has already written. A derived
       sibling's block returns for the same reason it returns on the shipping
       arm.
    6. An empty tail tile (`tile_begin >= count`, which `split_rows` makes
       reachable whenever `count < T`) returns having added nothing and
       subtracted nothing. On the shipping arm such a block zeroes, gathers
       nothing, and flushes nothing because its counts are all zero, so the
       two are the same store-for-store.

    So every knob selects an amount of traffic, contention, or parallelism,
    and never a number.
    """
    var tid = Int(thread_idx.x)
    var nthreads = Int(block_dim.x)
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var hs = Int(hist_size)
    var stride = nb + 1
    var tpi = Int(tiles_per_item)
    if tpi < 1:
        tpi = 1

    # The item and the local tile, by division rather than by search. Both
    # are functions of `block_idx.y` alone, so every thread of the block gets
    # the same pair and no threadgroup memory is needed to share it.
    var g_tile = Int(block_idx.y)
    var k = g_tile // tpi
    var t = g_tile - k * tpi
    var base = k * ITEM_WORDS

    # Removal three: this comparison is reached before the `stack_allocation`
    # below is touched. `ITEM_DEAD` is negative and a live-but-empty leaf's
    # histogram was already written by the zeroing pass.
    var count = Int(items[unsafe_offset = base + ITEM_COUNT][0])
    if count <= 0:
        return

    var out_slot = Int(items[unsafe_offset = base + ITEM_OUT][0])
    var sub_slot = 0
    var subtracting = Int(subtract) != 0
    if subtracting:
        # The same pair resolution and the same ties-to-the-right election
        # `_batch_hist_atomic_subtract_kernel` performs, and the same one
        # `_batch_copy_back_zero_subtract_kernel` prepared the slots under.
        # Run redundantly by every thread: six loads from one cache line, all
        # threads reading the same address, against a barrier and eight words
        # of threadgroup memory on the shipping arm.
        var n_pairs = _live_item_pairs(items, Int(n_items))
        if k >= 2 * n_pairs:
            return
        var partner = k + n_pairs if k < n_pairs else k - n_pairs
        var n_other = Int(
            items[unsafe_offset = partner * ITEM_WORDS + ITEM_COUNT][0]
        )
        var build = (
            (count < n_other) if k < n_pairs else (count <= n_other)
        )
        if not build:
            return
        sub_slot = Int(
            items[unsafe_offset = partner * ITEM_WORDS + ITEM_OUT][0]
        )

    # The row tile. `split_rows` is the occupancy multiplier: the tile follows
    # the item's own count, so all `tpi` tiles of every item carry rows at
    # every depth instead of one tile carrying them and `tpi - 1` sitting
    # empty because the length was derived from the root's row bound.
    var rpt = Int(items[unsafe_offset = base + ITEM_ROWS_PER_TILE][0])
    if Int(split_rows) != 0:
        rpt = (count + tpi - 1) // tpi
    if rpt < 1:
        rpt = 1
    var tile_begin = t * rpt
    if tile_begin >= count:
        return
    var tile_end = tile_begin + rpt
    if tile_end > count:
        tile_end = count

    # The feature slots this block owns.
    var ns = Int(n_slots)
    var gw = Int(group_width)
    if gw < 1:
        gw = 1
    if gw > PLAN_MAX_GROUP:
        gw = PLAN_MAX_GROUP
    var slot0 = Int(block_idx.x) * gw
    if slot0 >= ns:
        return
    var owned = ns - slot0
    if owned > gw:
        owned = gw

    var copies = Int(n_copies)
    if copies < 1:
        copies = 1
    if copies > nthreads:
        copies = nthreads

    var begin = Int(items[unsafe_offset = base + ITEM_BEGIN][0])
    var plane_base = Int(items[unsafe_offset = base + ITEM_PLANE][0]) * nr
    var g_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_G][0]
    var h_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_H][0]

    # The two sources, both block-uniform launch scalars, and the elided
    # plane's one quantized value. `Float32(1.0) * h_scale` is exactly
    # `h_scale`, so `hq_const` is `Int32(round(h_scale))` formed by this
    # device compiler from this launch's own argument. Never read while
    # `celide` is clear.
    var quant = Int(use_quant) != 0
    var celide = Int(const_hess) != 0
    var hq_const = Int32(round(Float32(1.0) * h_scale))

    # One global feature id per owned slot, read once and spent for every row.
    # `PLAN_MAX_GROUP` and not `owned`, because a `stack_allocation` needs a
    # comptime extent; the unowned entries are never read.
    var fid = stack_allocation[PLAN_MAX_GROUP, Scalar[DType.int32]]()
    var fbase = k * Int(feat_stride) + slot0
    for gi in range(owned):
        fid[unsafe_offset=gi] = feat_ids[unsafe_offset = fbase + gi][0]

    var shm = stack_allocation[
        UNITS * PLAN_UNIT_CELLS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var unit_cells = N_PLANES * stride
    var live_cells = owned * copies * unit_cells
    var z = tid
    while z < live_cells:
        shm[unsafe_offset=z] = Int32(0)
        z += nthreads
    barrier()

    # This thread's copy, and the step from one feature's copy to the next
    # feature's same-numbered copy. Unit `gi * copies + c`.
    var my_unit = tid % copies
    var ubase = my_unit * unit_cells
    var ustep = copies * unit_cells

    var j = tile_begin + tid
    while j < tile_end:
        var r = Int(rows[unsafe_offset = begin + j][0])
        var gqv = Int32(0)
        var hqv = Int32(0)
        if quant:
            gqv = gq[unsafe_offset = 2 * (plane_base + r)][0]
            if not celide:
                hqv = gq[unsafe_offset = 2 * (plane_base + r) + 1][0]
        else:
            gqv = Int32(
                round(grad[unsafe_offset = plane_base + r][0] * g_scale)
            )
            if not celide:
                hqv = Int32(
                    round(hess[unsafe_offset = plane_base + r][0] * h_scale)
                )
        var u = ubase
        for gi in range(owned):
            var bin = Int(
                bins[unsafe_offset = Int(fid[unsafe_offset=gi][0]) * nr + r]
            )
            _ = Atomic.fetch_add(shm.unsafe_offset(u + bin), gqv)
            if not celide:
                _ = Atomic.fetch_add(shm.unsafe_offset(u + stride + bin), hqv)
            _ = Atomic.fetch_add(
                shm.unsafe_offset(u + 2 * stride + bin), Int32(1)
            )
            u += ustep
        j += nthreads
    barrier()

    # The flush: sum this feature's copies, then fold into the output slice
    # exactly as the shipping kernels do, with the subtraction fused when the
    # block built the smaller child of a pair.
    var slot_cells = N_PLANES * hs
    for gi in range(owned):
        var f = Int(fid[unsafe_offset=gi][0])
        var slice_base = out_slot * slot_cells + f * nb
        var sub_base = sub_slot * slot_cells + f * nb
        var ug = gi * ustep
        var b = tid
        while b < nb:
            var vg = Int32(0)
            var vh = Int32(0)
            var vc = Int32(0)
            var u = ug
            for _ in range(copies):
                vg += shm[unsafe_offset = u + b][0]
                if not celide:
                    vh += shm[unsafe_offset = u + stride + b][0]
                vc += shm[unsafe_offset = u + 2 * stride + b][0]
                u += unit_cells
            # The refill, after the copies are summed: `vc` copies of
            # `hq_const` added together is `hq_const * vc`. The hessian plane
            # of `shm` is still zeroed above and simply not read, because the
            # zeroing here is one flat loop over `live_cells` and splitting it
            # per plane would cost a division in the loop to save a store.
            if celide:
                vh = hq_const * vc
            if vc != 0:
                _ = Atomic.fetch_add(
                    out_hist.unsafe_offset(slice_base + b), vg
                )
                _ = Atomic.fetch_add(
                    out_hist.unsafe_offset(slice_base + hs + b), vh
                )
                _ = Atomic.fetch_add(
                    out_hist.unsafe_offset(slice_base + 2 * hs + b), vc
                )
                if subtracting:
                    _ = Atomic.fetch_add(
                        out_hist.unsafe_offset(sub_base + b), -vg
                    )
                    _ = Atomic.fetch_add(
                        out_hist.unsafe_offset(sub_base + hs + b), -vh
                    )
                    _ = Atomic.fetch_add(
                        out_hist.unsafe_offset(sub_base + 2 * hs + b), -vc
                    )
            b += nthreads


def _batch_reduce_kernel(
    partials: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_items: Int32,
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
    feat_stride: Int32,
):
    """Sum each item's own tiles into its own output slice.

    One thread per output cell of the batch, decomposed into
    `(item, plane, slot, bin)`. An item's tiles are contiguous on the packed
    axis, so its reduction is a walk of `ITEM_TILES` strided reads starting
    at `ITEM_TILE_BEGIN`, in ascending tile order. The order is fixed and the
    values are exact integers, so the sum is reproducible and identical to
    the single-leaf reduction's.
    """
    var ns = Int(n_slots)
    var nb = Int(n_bins)
    var hs = Int(hist_size)
    var plane = ns * nb
    var per_item = N_PLANES * plane
    var i = global_idx.x
    if i >= Int(n_items) * per_item:
        return

    var k = i // per_item
    var rem = i - k * per_item
    var p = rem // plane
    var rem2 = rem - p * plane
    var slot = rem2 // nb
    var b = rem2 - slot * nb

    var base = k * ITEM_WORDS
    var tile_begin = Int(items[unsafe_offset = base + ITEM_TILE_BEGIN][0])
    var n_tiles = Int(items[unsafe_offset = base + ITEM_TILES][0])
    var out_slot = Int(items[unsafe_offset = base + ITEM_OUT][0])
    var f = Int(feat_ids[unsafe_offset = k * Int(feat_stride) + slot][0])

    var acc = Int32(0)
    var off = tile_begin * per_item + p * plane + slot * nb + b
    for _ in range(n_tiles):
        acc += partials[unsafe_offset=off][0]
        off += per_item
    out_hist[
        unsafe_offset = out_slot * N_PLANES * hs + p * hs + f * nb + b
    ] = acc


def _subtract_slice_kernel(
    out_hist: MutPointer[Int32, MutAnyOrigin],
    parent_slot: Int32,
    child_slot: Int32,
    dst_slot: Int32,
    cells: Int32,
):
    """`dst = parent - child`, elementwise over one histogram slot.

    The sibling subtraction trick, done where the histograms already live.
    Both operands are fixed-point Int32 accumulated under the same scales, so
    the difference is exact rather than exact-to-Float32, and the parent
    never has to cross to the host to be subtracted there.

    Each thread reads one cell of each of the three slices at the same
    offset and writes one cell, so `dst_slot` may alias either operand: the
    read of an aliased slice happens in the same thread that overwrites it,
    and no thread reads a cell another thread writes. Deriving in place over
    the parent's slot is therefore legal, and is what lets a frontier hold
    one slot per live leaf rather than one per node ever created.
    """
    var i = global_idx.x
    if i < Int(cells):
        var n = Int(cells)
        var a = out_hist[unsafe_offset = Int(parent_slot) * n + i][0]
        var b = out_hist[unsafe_offset = Int(child_slot) * n + i][0]
        out_hist[unsafe_offset = Int(dst_slot) * n + i] = a - b


# --- Batch planning -------------------------------------------------------


@fieldwise_init
struct BatchItemPlan(Copyable, Movable):
    """One item's resolved launch geometry."""

    var row_begin: Int
    var row_count: Int
    var out_slot: Int
    var plane: Int
    var rows_per_tile: Int
    var n_tiles: Int
    var tile_begin: Int
    var g_scale: Float32
    var h_scale: Float32


struct BatchPlan(Movable):
    """A resolved batched launch.

    Holds everything a launch needs and nothing about the frontier it came
    from, so a plan can be costed, compared against the serial plan, and
    checked for its memory bounds without a device present.
    """

    var items: List[BatchItemPlan]
    var strategy: Int
    var block_threads: Int
    var n_slots: Int
    var n_bins: Int
    var total_tiles: Int
    var partial_cells: Int
    """`total_tiles * n_slots * n_bins`, one cell carrying all three planes,
    so the buffer holds `3 * partial_cells` Int32. Zero under the atomic
    strategy."""

    var verdict: Int
    var target_blocks: Int

    def __init__(
        out self,
        var items: List[BatchItemPlan],
        strategy: Int,
        block_threads: Int,
        n_slots: Int,
        n_bins: Int,
        total_tiles: Int,
        partial_cells: Int,
        verdict: Int,
        target_blocks: Int,
    ):
        self.items = items^
        self.strategy = strategy
        self.block_threads = block_threads
        self.n_slots = n_slots
        self.n_bins = n_bins
        self.total_tiles = total_tiles
        self.partial_cells = partial_cells
        self.verdict = verdict
        self.target_blocks = target_blocks

    def n_items(self) -> Int:
        return len(self.items)

    def blocks(self) -> Int:
        """Threadgroups the histogram launch creates."""
        return self.n_slots * self.total_tiles

    def fills_device(self) -> Bool:
        return self.blocks() >= self.target_blocks


def _min_rows_per_tile(n_bins: Int, block_threads: Int) -> Int:
    var floor = MIN_ROWS_PER_TILE_BIN_FACTOR * n_bins
    var by_threads = MIN_ROWS_PER_TILE_THREAD_FACTOR * block_threads
    if by_threads > floor:
        floor = by_threads
    return floor


def plan_batch(
    caps: DeviceCaps,
    items: List[LeafWorkItem],
    scales: List[Float32],
    n_slots: Int,
    n_bins: Int,
    strategy: Int = STRATEGY_AUTO,
    max_partial_cells: Int = 0,
    max_items: Int = DEFAULT_MAX_ITEMS,
) raises -> BatchPlan:
    """Resolve one batched launch. Pure host arithmetic, no device needed.

    Tiles are handed out in three passes, all in exact integers so the plan
    is reproducible.

    1. **Want.** The batch as a whole aims at `sm_count * TARGET_BLOCKS_PER_SM`
       threadgroups, which is `ceil(target / n_slots)` tiles to share out.
       Each item's share is proportional to its row count, at least one tile,
       and never more than its rows can fill at the amortization floor, so a
       leaf of 300 rows does not get eight tiles of 40 rows each.
    2. **Fit.** The tiled strategy needs `total_tiles * n_slots * n_bins`
       partial cells. Where the budget is smaller, shares are scaled down
       proportionally, still with one tile per item as the floor. If the
       floors alone do not fit, the tiled strategy cannot serve this batch at
       all and the plan resolves to the atomic strategy, which needs no
       partial buffer. Both outcomes are reported through `verdict`.
    3. **Square up.** Rows per tile are re-derived from the final tile count
       and the tile count from that, so the last tile of every item is
       nonempty and `grid.y` covers exactly the rows the items hold.

    `max_partial_cells` is the partial buffer the caller already holds, in
    cells. Zero means unbounded, which is what a planner running without a
    device wants; `GpuLeafBatcher.enqueue_batch` refuses a plan that does not
    fit its own buffer, so an unbounded plan is checked before it can launch.
    An explicit `STRATEGY_TILED` that the buffer cannot serve is downgraded
    to the atomic strategy rather than refused, because the atomic strategy
    produces the identical histogram and refusing would leave the caller with
    no way to build the batch at all. `verdict` reports the downgrade.

    `scales` is `2 * len(items)` Float32, the fixed-point gradient and
    hessian scales the corresponding item's histogram must be accumulated
    with. They are per item because a batch may span classes, whose scales
    differ; a single-class batch passes the same pair repeatedly.
    """
    if len(items) < 1:
        raise Error("a batch needs at least one item")
    if len(items) > max_items:
        raise Error("batch holds more items than the launch bound allows")
    if len(scales) != SCALE_WORDS * len(items):
        raise Error("a batch needs one gradient and hessian scale per item")
    if n_slots < 1:
        raise Error("a batch needs at least one active feature")
    if n_bins < 1 or n_bins > MAX_BINS:
        raise Error("the GPU backend supports 1 to 256 bins")

    var block_threads = derive_block_threads(caps)
    var floor_rows = _min_rows_per_tile(n_bins, block_threads)
    var target_blocks = caps.sm_count * TARGET_BLOCKS_PER_SM
    if target_blocks < 1:
        target_blocks = 1

    var n = len(items)
    var counts = List[Int](capacity=n)
    var tile_caps = List[Int](capacity=n)
    var total_rows = 0
    for i in range(n):
        if items[i].row_begin < 0 or items[i].row_count < 0:
            raise Error("a batch item holds a negative row window")
        if items[i].out_slot < 0:
            raise Error("a batch item has no output slot")
        if items[i].plane < 0:
            raise Error("a batch item holds a negative gradient plane")
        for k in range(i):
            if items[k].out_slot == items[i].out_slot:
                raise Error("two batch items share an output slot")
        # An empty leaf still needs a launchable geometry, and one row is
        # what `GpuActiveRows.range_tiling` derives it at. Its row loop runs
        # zero iterations, so the histogram it produces is the zeros the
        # zeroing pass already wrote.
        var c = items[i].row_count
        if c < 1:
            c = 1
        counts.append(c)
        tile_caps.append(_ceil_div(c, floor_rows))
        total_rows += c

    # Pass 1: proportional shares of the tile target.
    var tiles_target = _ceil_div(target_blocks, n_slots)
    if tiles_target < n:
        tiles_target = n
    var tiles = List[Int](capacity=n)
    var total_tiles = 0
    for i in range(n):
        var want = _ceil_div(tiles_target * counts[i], total_rows)
        if want < 1:
            want = 1
        if want > tile_caps[i]:
            want = tile_caps[i]
        tiles.append(want)
        total_tiles += want

    # Pass 2: the partial-buffer bound.
    var resolved = strategy
    var verdict = VERDICT_UNKNOWN
    var hist_cells = n_slots * n_bins
    var tiles_budget: Int
    if max_partial_cells > 0:
        tiles_budget = max_partial_cells // hist_cells
    else:
        tiles_budget = MAX_GRID_DIM_Y
    if tiles_budget > MAX_GRID_DIM_Y:
        tiles_budget = MAX_GRID_DIM_Y

    if total_tiles > tiles_budget:
        if n > tiles_budget:
            # Not even one tile per item fits, so no tiled plan exists for
            # this batch. The atomic strategy allocates nothing and can.
            resolved = STRATEGY_ATOMIC
            verdict = VERDICT_PARTIAL_BOUND
            if total_tiles > MAX_GRID_DIM_Y:
                raise Error(
                    "batched histogram needs more row tiles than the portable"
                    " grid limit allows; batch fewer leaves"
                )
        else:
            var scaled = 0
            for i in range(n):
                var want = (tiles[i] * tiles_budget) // total_tiles
                if want < 1:
                    want = 1
                tiles[i] = want
                scaled += want
            # Integer flooring can leave the total above the budget by at
            # most one tile per item; give the surplus back in ascending item
            # order, never below the floor of one. Ascending rather than
            # largest-first because the order has to be a function of the
            # batch and not of a sort's tie-breaking, and every order that is
            # gives the same total.
            var over = scaled - tiles_budget
            var idx = 0
            while over > 0 and idx < n:
                if tiles[idx] > 1:
                    var give = tiles[idx] - 1
                    if give > over:
                        give = over
                    tiles[idx] -= give
                    over -= give
                idx += 1
            total_tiles = 0
            for k in range(n):
                total_tiles += tiles[k]
            verdict = VERDICT_PARTIAL_BOUND

    # Pass 3: square the geometry up so no tile is empty.
    var plans = List[BatchItemPlan](capacity=n)
    var running = 0
    for i in range(n):
        var rows_per_tile = _ceil_div(counts[i], tiles[i])
        if rows_per_tile < 1:
            rows_per_tile = 1
        var n_tiles = _ceil_div(counts[i], rows_per_tile)
        plans.append(
            BatchItemPlan(
                items[i].row_begin,
                items[i].row_count,
                items[i].out_slot,
                items[i].plane,
                rows_per_tile,
                n_tiles,
                running,
                scales[SCALE_WORDS * i + SCALE_G],
                scales[SCALE_WORDS * i + SCALE_H],
            )
        )
        running += n_tiles
    total_tiles = running
    if total_tiles > MAX_GRID_DIM_Y:
        raise Error(
            "batched histogram needs more row tiles than the portable grid"
            " limit allows; batch fewer leaves"
        )

    if resolved == STRATEGY_AUTO:
        # Same rule the single-leaf path uses. More than one tile per feature
        # is what the tiled path exists for; at one tile the partial buffer
        # is the same size as the output and the second launch buys nothing.
        if total_tiles > n:
            resolved = STRATEGY_TILED
        else:
            resolved = STRATEGY_ATOMIC

    var partial_cells = 0
    if resolved == STRATEGY_TILED:
        partial_cells = total_tiles * hist_cells
        if max_partial_cells > 0 and partial_cells > max_partial_cells:
            raise Error(
                "batch plan exceeds the partial buffer it was given"
            )

    if verdict == VERDICT_UNKNOWN:
        verdict = _occupancy_verdict(
            tile_caps, n_slots, target_blocks, total_tiles
        )

    return BatchPlan(
        plans^,
        resolved,
        block_threads,
        n_slots,
        n_bins,
        total_tiles,
        partial_cells,
        verdict,
        target_blocks,
    )


def _occupancy_verdict(
    tile_caps: List[Int],
    n_slots: Int,
    target_blocks: Int,
    total_tiles: Int,
) -> Int:
    """The structural occupancy fact about this plan, and only that.

    Three states are decidable without measuring anything. Some item would
    already reach the block target launched on its own, so batching cannot
    improve its occupancy. Or no item would and the batch does, which is the
    case batching exists for. Or neither, which is an honest unknown.

    The first test is against what an item would get *alone*, not against the
    tiles it got inside this batch, since inside a batch it shares the target
    with everyone else. Alone, an item gets the whole tile target, capped by
    the tiles its own rows can fill at the amortization floor, which is what
    `tile_caps` holds.

    No crossover threshold is implied by any of the three, and none is
    invented here. Which of them is worth batching on a given device is what
    `bench/apple/leaf_batching_plan.json` is for.
    """
    var alone_target = _ceil_div(target_blocks, n_slots)
    if alone_target < 1:
        alone_target = 1
    for i in range(len(tile_caps)):
        var alone = tile_caps[i]
        if alone > alone_target:
            alone = alone_target
        if alone < 1:
            alone = 1
        if alone * n_slots >= target_blocks:
            return VERDICT_SINGLE_FILLS
    if total_tiles * n_slots >= target_blocks:
        return VERDICT_OCCUPANCY_GAIN
    return VERDICT_UNKNOWN


# --- Symbolic cost --------------------------------------------------------


@fieldwise_init
struct BatchCost(Copyable, Movable):
    """What a plan costs, in countable units rather than in seconds.

    Every field is a number a profiler can be held to, which is the point:
    the comparison between batched and serial launches should be an
    arithmetic prediction that a measurement confirms or refutes, not an
    argument.
    """

    var launches: Int
    """Kernel launches, counting the zeroing pass and the reduction."""

    var blocks: Int
    """Threadgroups the histogram launch (or launches) creates."""

    var idle_blocks: Int
    """Threadgroups that exit without reading a row. Zero for a packed tile
    axis by construction, and carried anyway so a plan that stops being
    packed cannot hide it."""

    var bin_reads: Int
    """`sum over items of row_count * n_slots`: the gathered bin loads, the
    dominant term and the one batching does not change."""

    var partial_bytes: Int
    var output_bytes: Int
    var reduce_reads: Int
    """Int32 loads the reduction performs, `3 * n_slots * n_bins` per tile."""

    var atomic_folds: Int
    """Global integer atomics the atomic strategy issues, three per
    (tile, slot, nonempty bin) in the worst case."""


def batch_cost(plan: BatchPlan, n_features: Int) raises -> BatchCost:
    """The cost of running `plan` as one batched launch."""
    if n_features < 1:
        raise Error("cost model needs at least one feature")
    var bin_reads = 0
    var tiles = 0
    for i in range(plan.n_items()):
        bin_reads += plan.items[i].row_count * plan.n_slots
        tiles += plan.items[i].n_tiles
    var hist_cells = plan.n_slots * plan.n_bins
    var out_bytes = plan.n_items() * N_PLANES * n_features * plan.n_bins * 4
    var launches: Int
    var partial_bytes = 0
    var reduce_reads = 0
    var atomic_folds = 0
    if plan.strategy == STRATEGY_TILED:
        # Zero pass plus histogram plus reduction. The zero pass is still
        # needed whenever the batch does not write every feature slice.
        launches = 3
        partial_bytes = tiles * hist_cells * BYTES_PER_PARTIAL_CELL
        reduce_reads = tiles * N_PLANES * hist_cells
    else:
        launches = 2
        atomic_folds = tiles * N_PLANES * hist_cells
    return BatchCost(
        launches,
        plan.n_slots * tiles,
        0,
        bin_reads,
        partial_bytes,
        out_bytes,
        reduce_reads,
        atomic_folds,
    )


def serial_cost(plan: BatchPlan, n_features: Int) raises -> BatchCost:
    """The cost of the same items launched one leaf at a time, which is what
    the trainer does today.

    The row work is identical, because the same rows are read either way.
    What differs is the launch count, which scales with the item count
    instead of being constant, and the occupancy each launch reaches, which
    `BatchPlan.fills_device` reports and this model deliberately does not
    turn into a time.
    """
    var costed = batch_cost(plan, n_features)
    var per_item_launches = 3 if plan.strategy == STRATEGY_TILED else 2
    return BatchCost(
        per_item_launches * plan.n_items(),
        costed.blocks,
        costed.idle_blocks,
        costed.bin_reads,
        costed.partial_bytes,
        costed.output_bytes,
        costed.reduce_reads,
        costed.atomic_folds,
    )


def subtraction_saves_reads(n_left: Int, n_right: Int, n_slots: Int) -> Int:
    """Gathered bin loads the subtraction trick avoids on one commit.

    Building both children costs `(n_left + n_right) * n_slots`; building the
    smaller and subtracting costs `min(n_left, n_right) * n_slots` plus a
    launch over `3 * n_features * n_bins` cells that does not touch a row.
    The difference is the larger child's rows, which is what this returns,
    and it is why a batch of two children is not automatically better than a
    batch of one child plus a subtraction.
    """
    var larger = n_left if n_left > n_right else n_right
    return larger * n_slots


# --- Histogram slots ------------------------------------------------------


struct HistogramSlotPool(Movable):
    """Which device histogram slot belongs to which leaf.

    A slot is one full-width `3 * n_features * n_bins` Int32 histogram, the
    same shape and layout `GpuHistogramBuilder.out_dev` holds, so a slot can
    be downloaded and decoded by the existing `histogram_from_host` and read
    by the existing split-search kernels without a translation step.

    The pool is bounded and every slot is reclaimable, because the output
    buffer is the term that grows with the frontier: `P * 3 * F * B * 4`
    bytes. A leaf whose slot is taken back can be rebuilt from its row range
    at the cost of one more histogram, so the bound is a performance policy,
    never a correctness one.

    `stamp` is what makes `enqueue_subtract` safe. Two histograms may only be
    subtracted if they were accumulated under the same fixed-point scales and
    the same active feature set, and the stamp is the caller's encoding of
    both (a round counter and a feature-set counter folded together is
    enough). Slots with different stamps are refused rather than silently
    producing a histogram whose zero slices are not zero.
    """

    var capacity: Int
    var owner: List[Int]
    var stamp: List[Int]

    def __init__(out self, capacity: Int) raises:
        if capacity < 1:
            raise Error("a histogram slot pool needs at least one slot")
        self.capacity = capacity
        self.owner = List[Int](capacity=capacity)
        self.stamp = List[Int](capacity=capacity)
        for _ in range(capacity):
            self.owner.append(-1)
            self.stamp.append(-1)

    def free_slots(self) -> Int:
        var n = 0
        for i in range(self.capacity):
            if self.owner[i] < 0:
                n += 1
        return n

    def acquire(mut self, owner: Int, stamp: Int) raises -> Int:
        """The lowest free slot, or -1 when the pool is full.

        Lowest rather than most recently freed, so a sequence of acquires and
        releases produces the same slot assignment every run, which keeps the
        device buffer's contents a function of the tree and not of the
        allocator's history.
        """
        if owner < 0:
            raise Error("a histogram slot needs a nonnegative owner")
        for i in range(self.capacity):
            if self.owner[i] < 0:
                self.owner[i] = owner
                self.stamp[i] = stamp
                return i
        return -1

    def release(mut self, slot: Int) raises:
        if slot < 0 or slot >= self.capacity:
            raise Error("histogram slot out of range")
        self.owner[slot] = -1
        self.stamp[slot] = -1

    def release_all(mut self):
        for i in range(self.capacity):
            self.owner[i] = -1
            self.stamp[i] = -1

    def owner_of(self, slot: Int) raises -> Int:
        if slot < 0 or slot >= self.capacity:
            raise Error("histogram slot out of range")
        return self.owner[slot]

    def stamp_of(self, slot: Int) raises -> Int:
        """The compatibility stamp a slot was accumulated under, or -1 for a
        free slot. Two slots may only be subtracted when these agree; see
        `check_subtractable` and `subtraction_stamp`."""
        if slot < 0 or slot >= self.capacity:
            raise Error("histogram slot out of range")
        return self.stamp[slot]

    def reassign(mut self, slot: Int, owner: Int) raises:
        """Hand a live slot to a different leaf without freeing it.

        The one move `acquire`/`release` cannot express, and the subtraction
        trick needs it: `enqueue_subtract(parent, child, dst=parent)` leaves
        the parent's slot holding the *larger child's* histogram, so the slot
        outlives its owner by one generation and the leaf that reads it next
        is not the leaf that filled it. Releasing and reacquiring would be
        wrong twice over, since it could hand the slot to another leaf in
        between and it would reset the stamp the derived histogram was
        actually accumulated under.

        The stamp is deliberately left alone for that reason: it describes
        the scales and feature set the words in the slot were accumulated
        with, which a change of owner does not move.
        """
        if slot < 0 or slot >= self.capacity:
            raise Error("histogram slot out of range")
        if owner < 0:
            raise Error("a histogram slot needs a nonnegative owner")
        if self.owner[slot] < 0:
            raise Error("a free histogram slot has no owner to reassign")
        self.owner[slot] = owner

    def slot_of_owner(self, owner: Int) -> Int:
        """The live slot `owner` holds, or -1. Owners are leaf node ids and a
        node holds at most one slot, so the answer is unique."""
        for i in range(self.capacity):
            if self.owner[i] == owner:
                return i
        return -1

    def check_live(self, slot: Int) raises:
        if slot < 0 or slot >= self.capacity:
            raise Error("histogram slot out of range")
        if self.owner[slot] < 0:
            raise Error("histogram slot holds no leaf")

    def check_subtractable(self, a: Int, b: Int) raises:
        """Both slots live, and both accumulated under the same conditions."""
        self.check_live(a)
        self.check_live(b)
        if self.stamp[a] != self.stamp[b]:
            raise Error(
                "histograms accumulated under different scales or feature"
                " sets cannot be subtracted"
            )


def slot_bytes(n_features: Int, n_bins: Int) -> Int:
    """Device bytes one histogram slot occupies."""
    return N_PLANES * n_features * n_bins * 4


def pool_bytes(capacity: Int, n_features: Int, n_bins: Int) -> Int:
    return capacity * slot_bytes(n_features, n_bins)


def slots_for_budget(
    budget_bytes: Int, n_features: Int, n_bins: Int
) raises -> Int:
    """How many slots a byte budget buys, at least one.

    The number a caller needs to see before deciding to hold a frontier's
    histograms on the device at all: at 1000 features and 256 bins a slot is
    3 MB, so a 255-leaf frontier would ask for 750 MB and has to be told no.
    """
    if n_features < 1 or n_bins < 1:
        raise Error("slot sizing needs positive features and bins")
    var one = slot_bytes(n_features, n_bins)
    var n = budget_bytes // one
    if n < 1:
        return 1
    return n


def subtraction_stamp(round_index: Int, feature_epoch: Int) raises -> Int:
    """The compatibility stamp two histograms must share to be subtracted.

    `HistogramSlotPool` refuses a subtraction across differing stamps, and
    this is the encoding that makes the refusal mean the right thing. The two
    conditions are exactly the ones `enqueue_subtract`'s docstring names:

    - **Same fixed-point scales.** The scales are re-derived once per round
      and per class from the gradient magnitude sums, so they are constant
      within a tree and generally differ between trees. `round_index` is the
      caller's counter over (boosting round, class) pairs, which is the
      granularity a scale actually changes at.
    - **Same active feature set.** A feature present in one histogram and
      absent from the other would leave a nonzero slice being subtracted from
      a zero one. `feature_epoch` is the caller's counter, bumped whenever
      `set_features`/`set_shared_features` narrows or widens the set, which
      the trainer does once per tree.

    Folded rather than paired because the pool stores one Int and only ever
    compares stamps for equality; folding two nonnegative counters into
    `round * 2^24 + epoch` keeps that comparison exact for any run inside the
    two bounds below, and refuses outside them rather than aliasing two
    incompatible histograms onto one stamp. The bounds are far above any real
    fit (16.7 million feature epochs, and rounds counted over (iteration,
    class) pairs into the billions) and are checked anyway, because an
    aliased stamp is a wrong histogram and not a lost slot.
    """
    if round_index < 0 or feature_epoch < 0:
        raise Error("stamp counters must be nonnegative")
    if feature_epoch >= (1 << 24):
        raise Error("feature epoch is too large for a subtraction stamp")
    if round_index >= (1 << 32):
        raise Error("round index is too large for a subtraction stamp")
    return (round_index << 24) + feature_epoch


# --- The frontier seam ----------------------------------------------------
#
# What a grower calls to turn `LeafFrontier` state into one launch, without
# knowing anything below about tiles, item tables, or the packed grid axis.
# Four steps, in order, and each one may be declined rather than forced:
#
#     admit_frontier_batch   is a batch legal and worth assembling at all
#     assign_batch_slots     give each covered leaf an output histogram slot
#     plan_frontier_batch    resolve the launch geometry
#     GpuLeafBatcher.enqueue_frontier_batch   launch it over the active rows
#
# The admission step is the conservative switch. Every reason it can decline
# leaves the caller on the established one-leaf-per-launch path, which is what
# `GpuActiveRows.enqueue_range_histogram` already does correctly, so a grower
# can adopt the batched path without a second correctness story.

comptime BATCH_OK = 0

comptime BATCH_NO_PENDING = 1
"""No leaf needs work, so there is nothing to launch. Not a failure."""

comptime BATCH_SINGLE_ITEM = 2
"""Only one leaf needs work. A batch of one is the single-leaf launch with an
item table in front of it, so the established path is preferred outright."""

comptime BATCH_NO_SLOTS = 3
"""The histogram slot pool cannot give every covered leaf its own output
slice. Slots are reclaimable and a leaf can always be rebuilt from its row
range, so this is a memory bound, never a correctness one."""

comptime BATCH_KERNEL_ABSENT = 4
"""`KernelFeatures.batched_leaf_kernel` is false: the batched kernels are not
compiled in and validated for this build."""

comptime BATCH_STORAGE_UNSUPPORTED = 5
"""The bin matrix is not the one-`UInt8`-per-cell, one-feature-per-block
matrix these kernels index. A packed or blocked layout needs a decoding
kernel that does not exist; see `BinStorageDescriptor.check_shipping`."""


def batch_admission_name(reason: Int) -> String:
    if reason == BATCH_OK:
        return String("ok")
    if reason == BATCH_NO_PENDING:
        return String("no_pending_leaves")
    if reason == BATCH_SINGLE_ITEM:
        return String("single_item")
    if reason == BATCH_NO_SLOTS:
        return String("no_free_histogram_slots")
    if reason == BATCH_KERNEL_ABSENT:
        return String("batched_kernel_absent")
    if reason == BATCH_STORAGE_UNSUPPORTED:
        return String("bin_storage_unsupported")
    return String("unknown")


struct BatchAdmission(Copyable, Movable):
    """Whether a batch may be assembled from this frontier, and over which
    slots.

    Carries the numbers the decision was made from, so a caller that declines
    can report why rather than only that. `slots` is empty unless
    `reason == BATCH_OK`, which keeps "declined" and "admitted over nothing"
    from being the same value.
    """

    var reason: Int
    var slots: List[Int]
    """Frontier slots the batch would cover, ascending."""

    var pending: Int
    """Leaves needing work, before the `max_items` cap."""

    var free_slots: Int

    def __init__(
        out self,
        reason: Int,
        var slots: List[Int],
        pending: Int,
        free_slots: Int,
    ):
        self.reason = reason
        self.slots = slots^
        self.pending = pending
        self.free_slots = free_slots

    def admitted(self) -> Bool:
        return self.reason == BATCH_OK

    def n_items(self) -> Int:
        return len(self.slots)

    def describe(self) -> String:
        return (
            String("batch ")
            + batch_admission_name(self.reason)
            + " items "
            + String(len(self.slots))
            + " pending "
            + String(self.pending)
            + " free_slots "
            + String(self.free_slots)
        )


def _declined(reason: Int, pending: Int, free_slots: Int) -> BatchAdmission:
    """A declined admission, spelled once so every reason below produces the
    same shape: no slots, and the two numbers the decision was made from."""
    return BatchAdmission(reason, List[Int](), pending, free_slots)


def admit_frontier_batch(
    frontier: LeafFrontier,
    pool: HistogramSlotPool,
    storage: BinStorageDescriptor,
    features: KernelFeatures,
    max_items: Int = DEFAULT_MAX_ITEMS,
    min_items: Int = 2,
) raises -> BatchAdmission:
    """Decide whether this frontier can be batched, and over which slots.

    Pure host arithmetic against state that already exists: no device, no
    allocation, and nothing mutated, so a caller may ask before committing to
    anything. The order of the tests is the order a caller would want them
    reported in, cheapest structural fact first.

    A leaf that already holds a live slot is counted as needing no new one, so
    re-batching a frontier whose slots survived does not spuriously decline
    for want of capacity.

    `min_items` is the width below which the batched path is not worth taking;
    two is the smallest batch that is not a single-leaf launch in disguise,
    and it is the default because that is a structural fact rather than a
    measured threshold. A benchmark that wants a higher bar passes one.
    """
    if max_items < 1:
        raise Error("a batch holds at least one item")
    if min_items < 1:
        raise Error("a batch holds at least one item")
    storage.check()

    var pending = len(frontier.pending())
    var free = pool.free_slots()
    if not features.batched_leaf_kernel:
        return _declined(BATCH_KERNEL_ABSENT, pending, free)
    if not storage.is_dense_feature_major_u8():
        return _declined(BATCH_STORAGE_UNSUPPORTED, pending, free)
    if pending < 1:
        return _declined(BATCH_NO_PENDING, pending, free)
    if pending < min_items:
        return _declined(BATCH_SINGLE_ITEM, pending, free)

    var slots = frontier.batch_slots(max_items)
    if len(slots) < min_items:
        return _declined(BATCH_SINGLE_ITEM, pending, free)
    var needed = 0
    for i in range(len(slots)):
        var leaf = frontier.leaf(slots[i])
        if leaf.hist_slot == NO_SLOT:
            needed += 1
        elif pool.owner_of(leaf.hist_slot) != leaf.node:
            # The slot the leaf remembers was taken back by the pool, so it
            # needs a fresh one. Eviction costs a rebuild, never an answer.
            needed += 1
    if needed > free:
        return _declined(BATCH_NO_SLOTS, pending, free)
    return BatchAdmission(BATCH_OK, slots^, pending, free)


def assign_batch_slots(
    mut frontier: LeafFrontier,
    mut pool: HistogramSlotPool,
    admission: BatchAdmission,
    stamp: Int,
) raises -> List[Int]:
    """Give every leaf the batch covers its own output histogram slot.

    Returns the pool slots, in the admission's order, and writes each one back
    into the frontier so `work_items` picks it up. A leaf already holding a
    live slot under the same stamp keeps it, which is what makes re-batching
    cheap; a slot under a *different* stamp is released and reacquired,
    because a histogram accumulated under other scales or another feature set
    is not a histogram this batch may subtract from or add to.

    All-or-nothing. If the pool runs dry part way through (which
    `admit_frontier_batch` is meant to prevent, but which a caller that
    acquired slots in between can still cause), every slot this call took is
    given back and every frontier assignment it made is undone, so the
    frontier is exactly as it was and the caller can fall back to the
    single-leaf path without a leaked slot.
    """
    if not admission.admitted():
        raise Error(
            "only an admitted batch may be given slots: "
            + batch_admission_name(admission.reason)
        )
    var taken = List[Int]()
    var taken_at = List[Int]()
    var out = List[Int](capacity=len(admission.slots))
    for i in range(len(admission.slots)):
        var s = admission.slots[i]
        var leaf = frontier.leaf(s)
        var held = leaf.hist_slot
        if held != NO_SLOT:
            if (
                pool.owner_of(held) == leaf.node
                and pool.stamp_of(held) == stamp
            ):
                out.append(held)
                continue
            if pool.owner_of(held) == leaf.node:
                pool.release(held)
            frontier.assign_slot(s, NO_SLOT)
        var got = pool.acquire(leaf.node, stamp)
        if got < 0:
            for k in range(len(taken)):
                pool.release(taken[k])
                frontier.assign_slot(taken_at[k], NO_SLOT)
            raise Error(
                "the histogram slot pool ran out mid-batch; grow the pool"
                " (slots_for_budget) or batch fewer leaves"
            )
        taken.append(got)
        taken_at.append(s)
        frontier.assign_slot(s, got)
        out.append(got)
    return out^


def release_batch_slots(
    mut frontier: LeafFrontier, mut pool: HistogramSlotPool, slots: List[Int]
) raises:
    """Give a batch's output slots back and clear the frontier's memory of
    them. For a caller abandoning a planned batch, and for the end of a tree,
    where every slot is dead at once and `HistogramSlotPool.release_all` is
    the cheaper call."""
    for i in range(len(slots)):
        var s = slots[i]
        if s < 0 or s >= frontier.size():
            raise Error("frontier slot out of range")
        var held = frontier.leaf(s).hist_slot
        if held != NO_SLOT:
            pool.release(held)
            frontier.assign_slot(s, NO_SLOT)


def uniform_scales(
    n_items: Int, g_scale: Float32, h_scale: Float32
) raises -> List[Float32]:
    """One gradient/hessian scale pair repeated for every item.

    The single-class case, which is every batch the trainer assembles today:
    a round has one pair of fixed-point scales and every leaf of every tree in
    it accumulates under them. A multiclass batch that spanned classes would
    build the list itself, pair by pair, which is why `plan_batch` takes a
    list rather than a pair.
    """
    if n_items < 1:
        raise Error("a batch needs at least one item")
    if g_scale <= 0.0 or h_scale <= 0.0:
        raise Error("fixed-point scales must be positive")
    var out = List[Float32](capacity=SCALE_WORDS * n_items)
    for _ in range(n_items):
        out.append(g_scale)
        out.append(h_scale)
    return out^


def plan_frontier_batch(
    caps: DeviceCaps,
    frontier: LeafFrontier,
    admission: BatchAdmission,
    g_scale: Float32,
    h_scale: Float32,
    n_slots: Int,
    n_bins: Int,
    strategy: Int = STRATEGY_AUTO,
    max_partial_cells: Int = 0,
    max_items: Int = DEFAULT_MAX_ITEMS,
) raises -> BatchPlan:
    """Resolve the launch for an admitted, slotted batch.

    Thin on purpose: it turns frontier slots into `LeafWorkItem`s (which is
    where the row windows, the output slots, and the gradient plane come
    from), repeats the round's scales once per item, and hands both to
    `plan_batch`, which is the only tile arithmetic in this lane. Call
    `assign_batch_slots` first; an item whose leaf still holds `NO_SLOT` is
    refused here with a message that says so, rather than reaching
    `plan_batch` as a negative index.
    """
    if not admission.admitted():
        raise Error(
            "only an admitted batch may be planned: "
            + batch_admission_name(admission.reason)
        )
    var items = frontier.work_items(admission.slots)
    for i in range(len(items)):
        if items[i].out_slot == NO_SLOT:
            raise Error(
                "a batched leaf holds no output histogram slot; call"
                " assign_batch_slots before plan_frontier_batch"
            )
    var scales = uniform_scales(len(items), g_scale, h_scale)
    return plan_batch(
        caps,
        items,
        scales,
        n_slots,
        n_bins,
        strategy,
        max_partial_cells,
        max_items,
    )


def check_batch_covers_ranges(
    plan: BatchPlan, frontier: LeafFrontier, admission: BatchAdmission
) raises:
    """Every item's window is the window its frontier leaf holds.

    Cheap, and worth running before a launch that reads through the active-row
    permutation: an item whose `row_begin`/`row_count` had drifted from its
    leaf would accumulate a histogram of some other leaf's rows entirely
    inside the buffer, so no bound would be violated and no later check would
    notice. `GpuActiveRows.check_frontier` is the other half, holding the
    frontier equal to the device's own range table.
    """
    if plan.n_items() != len(admission.slots):
        raise Error("plan and admission cover different item counts")
    for i in range(plan.n_items()):
        var leaf = frontier.leaf(admission.slots[i])
        if plan.items[i].row_begin != leaf.row_begin:
            raise Error("a batch item does not start at its leaf's rows")
        if plan.items[i].row_count != leaf.row_count:
            raise Error("a batch item does not cover its leaf's rows")


def batch_windows(plan: BatchPlan) raises -> List[LeafRange]:
    """Each item's row window, as the lane's one window type.

    `LeafRange` is `gpu_active_rows`' half-open `[begin, end)`, and a batch
    item carries the same window as a begin and a count. Returning the shared
    type rather than a second pair is what keeps a caller from inventing a
    third spelling of a leaf's rows; `LeafRange.overlaps` is then the ready
    answer to "do two items of this batch read the same rows", which they
    never should, because live leaves tile the active prefix.
    """
    var out = List[LeafRange](capacity=plan.n_items())
    for i in range(plan.n_items()):
        var begin = plan.items[i].row_begin
        out.append(LeafRange(begin, begin + plan.items[i].row_count))
    return out^


def batched_leaf_stats(
    raw: List[Int32],
    n_features: Int,
    n_bins: Int,
    feature: Int,
    g_scale: Float64,
    h_scale: Float64,
) raises -> LeafStats:
    """One downloaded slot's gradient sum, hessian sum, and row count.

    A histogram's count plane sums to the leaf's rows over any one feature,
    and its gradient and hessian planes to the leaf's sums, so a single
    feature's slice is enough and the whole slot does not have to be scanned.
    The scales are the ones the slot was accumulated under, which is the same
    pair `HistogramSlotPool`'s stamp is meant to keep constant, so a caller
    that respects the stamp cannot convert with the wrong divisor.

    This is the host-side bridge from a batched result back into
    `LeafFrontier.set_stats`, and it is deliberately the only one: nothing
    here re-derives a leaf's rows from anything but its own counts.
    """
    var hist_size = n_features * n_bins
    if len(raw) != N_PLANES * hist_size:
        raise Error("raw histogram is not one full-width slot")
    if feature < 0 or feature >= n_features:
        raise Error("feature index out of range")
    if g_scale == 0.0 or h_scale == 0.0:
        raise Error("fixed-point scales must not be zero")
    var base = feature * n_bins
    var g = 0.0
    var h = 0.0
    var c = 0
    for b in range(n_bins):
        g += Float64(raw[base + b])
        h += Float64(raw[hist_size + base + b])
        c += Int(raw[2 * hist_size + base + b])
    return LeafStats(g / g_scale, h / h_scale, c)


# --- The batcher ----------------------------------------------------------


struct GpuLeafBatcher(Movable):
    """Device buffers and launches for batched multi-leaf histograms.

    Construct once per training session on the histogram builder's own
    `DeviceContext`, so the batch's kernels queue behind that builder's
    gradient upload and partition kernels with no fence. The binned matrix,
    the active-row permutation, and the gradient planes stay where they are
    and arrive as pointers; this struct owns only the item tables, the
    partial buffer, and the histogram slot pool's backing store.
    """

    var ctx: DeviceContext
    var n_features: Int
    var n_bins: Int
    var n_rows: Int
    var max_items: Int
    var n_planes: Int
    # One Int32 row per item: see the ITEM_* layout above.
    var items_dev: DeviceBuffer[DType.int32]
    # Per-item fixed-point scales, gradient then hessian.
    var scales_dev: DeviceBuffer[DType.float32]
    # Global feature ids per (item, slot), strided by `n_features` rather
    # than by the batch's `n_slots`, so narrowing the feature set never moves
    # an item's row. The kernels are told the stride. Replicated across items
    # when the batch shares one feature set, which is the trainer's case.
    var feat_dev: DeviceBuffer[DType.int32]
    # `pool.capacity` full-width histograms, in the builder's layout.
    var out_dev: DeviceBuffer[DType.int32]
    # `3 * partial_capacity` Int32, or one element when the atomic strategy
    # is all this batcher will ever run.
    var part_dev: DeviceBuffer[DType.int32]
    var partial_capacity: Int
    var block_threads: Int
    var stage_items: HostBuffer[DType.int32]
    var stage_scales: HostBuffer[DType.float32]
    var stage_feat: HostBuffer[DType.int32]
    var pool: HistogramSlotPool
    # The device-written-plan arm's staged geometry: how many item rows the
    # commit kernel may fill, how many packed tiles each of them was given,
    # and how wide the feature axis was when they were sized. Zero until
    # `stage_device_plan` has run, which is what `enqueue_device_plan_batch`
    # refuses on. Kept as fields rather than passed per launch because they
    # are the half of the plan the host owns and they do not move within a
    # tree; see `stage_device_plan`.
    var plan_items: Int
    var plan_tiles_per_item: Int
    var plan_slots: Int
    # Pairs the NEXT subtracting level batch may hold, or zero for "unknown".
    # A hint and never a fact: it only ever narrows the grid the pair-indexed
    # arm launches, and the kernel reads the item table for the real count.
    # Written by `set_level_pairs`, which is called from the one level-commit
    # site that knows the depth; see that method for why an upper bound is
    # sufficient and why zero is always safe.
    var plan_level_pairs: Int
    # The interleaved pre-quantized gradient buffer, `2 * n_planes * n_rows`
    # Int32, or one word when `MOJOTREES_GPU_BATCH_QUANT=0` declined it. Only
    # the staged plan's own plane is ever written; see `_ensure_quantized`.
    var gq_dev: DeviceBuffer[DType.int32]
    var quant_capacity: Int
    # Whether the next accumulation gathers that buffer, and whether it
    # currently holds THIS tree's round. `stage_device_plan` clears the
    # second, which is the only invalidation there is: the gradients cannot
    # move inside a tree and that method runs once per tree.
    var quant_grads: Bool
    var quant_valid: Bool
    # The staged plan's scales and plane, kept host side so the quantization
    # pass needs no readback. Written by `stage_device_plan` and by nothing
    # else, which is also the proof that the pair is uniform over the plan's
    # items: that method writes one pair into every item's row.
    var plan_g_scale: Float32
    var plan_h_scale: Float32
    var plan_plane: Int
    # The caller's declaration that every row's hessian this round is exactly
    # `histogram.CONSTANT_HESSIAN`, and the environment's power to withdraw
    # the permission but never to grant it. See `set_constant_hessian`.
    var constant_hessian: Bool
    var const_hessian_allowed: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        caps: DeviceCaps,
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        pool_capacity: Int,
        partial_capacity: Int,
        max_items: Int = DEFAULT_MAX_ITEMS,
        n_planes: Int = 1,
    ) raises:
        """Allocate everything a session will use. Nothing below is sized by
        a leaf, so a batch allocates nothing.

        `partial_capacity` is in cells, `3 * partial_capacity` Int32 words,
        and passing zero means this batcher will only ever run the atomic
        strategy. `pool_capacity` slots at `3 * F * B` Int32 each is the term
        to watch; `slots_for_budget` sizes it against a byte budget.
        """
        if n_rows < 1:
            raise Error("the GPU backend requires at least one row")
        if n_rows > MAX_ROWS:
            raise Error("the GPU backend supports at most 2^31 - 1 rows")
        if n_features < 1:
            raise Error("the GPU backend requires at least one feature")
        if n_bins < 1 or n_bins > MAX_BINS:
            raise Error("the GPU backend supports 1 to 256 bins")
        if max_items < 1:
            raise Error("a batch holds at least one item")
        if n_planes < 1:
            raise Error("at least one gradient plane must be resident")
        if partial_capacity < 0:
            raise Error("partial capacity must be nonnegative")

        self.ctx = ctx
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins
        self.max_items = max_items
        self.n_planes = n_planes
        self.partial_capacity = partial_capacity
        self.block_threads = derive_block_threads(caps)
        self.pool = HistogramSlotPool(pool_capacity)
        self.plan_items = 0
        self.plan_tiles_per_item = 0
        self.plan_slots = 0
        self.plan_level_pairs = 0
        self.plan_g_scale = Float32(0.0)
        self.plan_h_scale = Float32(0.0)
        self.plan_plane = 0
        self.quant_grads = batch_quantized_grads_requested()
        self.quant_valid = False
        # `MOJOTREES_CONST_HESSIAN=0` withdraws the permission, so a later
        # `set_constant_hessian(True)` is refused rather than silently
        # ignored. The same reading `GpuActiveRows` makes of the same name.
        self.const_hessian_allowed = getenv(BATCH_CONST_HESSIAN_VAR) != "0"
        self.constant_hessian = False

        var hist_size = n_features * n_bins
        self.items_dev = self.ctx.enqueue_create_buffer[DType.int32](
            max_items * ITEM_WORDS
        )
        self.scales_dev = self.ctx.enqueue_create_buffer[DType.float32](
            max_items * SCALE_WORDS
        )
        self.feat_dev = self.ctx.enqueue_create_buffer[DType.int32](
            max_items * n_features
        )
        self.out_dev = self.ctx.enqueue_create_buffer[DType.int32](
            pool_capacity * N_PLANES * hist_size
        )
        var part_size = N_PLANES * partial_capacity
        if part_size < 1:
            part_size = 1
        self.part_dev = self.ctx.enqueue_create_buffer[DType.int32](part_size)
        # One interleaved Int32 pair per row per gradient plane. Allocated
        # unless the environment declined it outright, so a benchmark can move
        # `set_quantized_gradients` between repeats in one process rather than
        # re-execing; at 463,715 rows and one plane it is 3.7 MB.
        self.quant_capacity = 1
        if _batch_quant_allowed():
            self.quant_capacity = 2 * n_planes * n_rows
        self.gq_dev = self.ctx.enqueue_create_buffer[DType.int32](
            self.quant_capacity
        )
        self.stage_items = self.ctx.enqueue_create_host_buffer[DType.int32](
            max_items * ITEM_WORDS
        )
        self.stage_scales = self.ctx.enqueue_create_host_buffer[
            DType.float32
        ](max_items * SCALE_WORDS)
        self.stage_feat = self.ctx.enqueue_create_host_buffer[DType.int32](
            max_items * n_features
        )

        # Every item's feature table starts as the identity, so a caller that
        # never narrows the feature set can launch without staging one.
        var dst = self.stage_feat.unsafe_ptr()
        for k in range(max_items):
            for f in range(n_features):
                dst.unsafe_store(k * n_features + f, Int32(f))
        self.ctx.enqueue_copy(dst_buf=self.feat_dev, src_ptr=dst)
        self.ctx.synchronize()

    def synchronize(self) raises:
        self.ctx.synchronize()

    def set_quantized_gradients(mut self, on: Bool) raises:
        """Whether the device-plan accumulation gathers the pre-quantized
        Int32 pairs instead of quantizing per (row, feature) visit.

        An argument as well as an environment variable, for the reason
        `GpuActiveRows.set_feature_group` gives: an A/B that reads its arm
        from the environment can run one arm under the other's label, which
        has happened in this repository once, so a benchmark holding both
        arms in one process passes the arm in rather than re-execing.

        Raises rather than silently leaving the Float32 arm selected when
        `MOJOTREES_GPU_BATCH_QUANT=0` declined the buffer's allocation, for
        the same reason `set_scan_primitives` raises at an unsupported width:
        a benchmark told to run this arm must not quietly measure the other
        one.

        Takes effect on the next device-plan accumulation, and it cannot
        change a histogram. The buffer is rebuilt on the next launch either
        way, because turning the arm on mid-tree leaves `quant_valid` clear.
        """
        if on and self.quant_capacity < 2 * self.n_planes * self.n_rows:
            raise Error(
                "the quantized gradient buffer was not allocated;"
                " MOJOTREES_GPU_BATCH_QUANT=0 declined it for this session"
            )
        self.quant_grads = on
        self.quant_valid = False

    def quantized_gradients_on(self) -> Bool:
        """Whether the next device-plan accumulation gathers Int32 pairs."""
        return self.quant_grads

    def set_constant_hessian(mut self, on: Bool):
        """Declare that every row's hessian this round is exactly
        `histogram.CONSTANT_HESSIAN`, so the batched kernels may stop
        accumulating the hessian plane and reconstruct it from the count.

        **This is a declaration about the objective, and the caller owns
        it.** `histogram.objective_has_constant_hessian` is the predicate that
        answers it correctly, and it is false for every weighted fit and for
        every GOSS round regardless of the objective code, because both put
        more than one value into `hess`. Declaring it on a round where it is
        not true produces a wrong hessian plane, silently. This is the same
        contract `GpuActiveRows.set_constant_hessian` carries word for word,
        and it is a second setter rather than a read of that one because this
        module is handed pointers and never the struct that owns them.

        `MOJOTREES_CONST_HESSIAN=0` wins over an argument in the one direction
        that is always safe, off, and this reports what it actually did
        through `constant_hessian_on`.

        Takes effect on the next accumulation. When the declaration is true it
        cannot change a histogram: the plane it stops accumulating is
        reconstructed as the identical Int32, argued at
        `_batch_hist_atomic_kernel`.

        **No caller sets this yet, and that is a cross-file item rather than
        an oversight.** `GpuHistogramBuilder.set_constant_hessian` in
        `histogram_gpu` forwards the trainer's declaration to `self.rows` and
        stops there; one line forwarding it to `self.batcher[0]` as well is
        what reaches this, and that file belongs to another lane.
        """
        self.constant_hessian = on and self.const_hessian_allowed

    def constant_hessian_on(self) -> Bool:
        """Whether the next accumulation elides the hessian plane."""
        return self.constant_hessian

    def staged_gradient_bytes_per_visit(self) -> Int:
        """Bytes of staged derivative one (row, feature) VISIT costs the
        batched family, at the arms currently set.

        `GpuActiveRows.staged_gradient_bytes_per_row` is the range family's
        figure and the two were not comparable, because the two families do
        not spend the derivative the same way. The range family gathers a
        row's derivative once per threadgroup and spends it over a feature
        GROUP, so its per-visit share is its per-row width divided by the
        group. The batched family has no group at all on the shipping arms
        (`grid.x` is the feature axis, one slot per threadgroup), so a visit
        pays the whole per-row width and this number IS the per-row width,
        divided by the lean kernel's group when that arm is selected.

        Four on the Float32 arm under a constant hessian and eight otherwise;
        the same four or eight on the quantized arm, the pair being one load
        either way. What the quantized arm buys at equal width is the two
        multiplies and two rounds, not the bytes -- so a trace comparing the
        two families must read this beside
        `_plan_hist_kernel`'s group width and not on its own.
        """
        var wide = 4 if self.constant_hessian else 8
        var gw = plan_group_requested() if plan_lean_requested() else 1
        if gw < 1:
            gw = 1
        return wide // gw if wide >= gw else 1

    def staged_gradient_bytes_per_row(self) -> Int:
        """The per-row width, spelled as `GpuActiveRows` spells it, so the two
        families can be read off a trace side by side without either reader
        recomputing the other's convention. Four under a constant hessian and
        eight otherwise, on both the Float32 and the quantized arm; this
        module has no Int16 staging arm."""
        return 4 if self.constant_hessian else 8

    def hist_size(self) -> Int:
        return self.n_features * self.n_bins

    def slot_cells(self) -> Int:
        return N_PLANES * self.n_features * self.n_bins

    def max_tiles(self, n_slots: Int) raises -> Int:
        """Row tiles the partial buffer holds at `n_slots` active features.

        The capacity is in cells, and one tile costs `n_slots * n_bins` of
        them, so the tile budget moves with the feature set. This is the
        number to hand `plan_batch` as `max_partial_cells / hist_cells`
        reasoning, and the one that decides whether a wide batch has to fall
        back to the atomic strategy.
        """
        if n_slots < 1:
            raise Error("tile budget needs at least one active feature")
        return self.partial_capacity // (n_slots * self.n_bins)

    def set_shared_features(mut self, features: List[Int]) raises:
        """One feature set for every item in the next batch.

        The trainer's case: `GpuHistogramBuilder.set_features` narrows the
        grid to the tree's sampled features once per tree, and every node of
        that tree accumulates the same set. Replicating it per item costs
        `max_items * n_slots` Int32 of staging and lets the kernels index one
        table without branching on whether the batch shares a set.

        Shares the staging contract on `_stage_plan`: this writes the pinned
        feature table, so it runs once per tree, before that tree's first
        batch, and never between a batch's launch and its completion.
        """
        if len(features) == 0:
            raise Error("an active feature set must not be empty")
        if len(features) > self.n_features:
            raise Error("active feature set is larger than n_features")
        for i in range(len(features)):
            if features[i] < 0 or features[i] >= self.n_features:
                raise Error("feature index out of range")
        var dst = self.stage_feat.unsafe_ptr()
        for k in range(self.max_items):
            for i in range(len(features)):
                dst.unsafe_store(k * self.n_features + i, Int32(features[i]))
        self.ctx.enqueue_copy(dst_buf=self.feat_dev, src_ptr=dst)

    def set_item_features(
        mut self, item: Int, features: List[Int]
    ) raises:
        """One item's own feature set, for a caller that narrows per node.

        Per-node feature sampling narrows the *search* today and not the
        accumulation, so nothing in the trainer needs this yet. It exists
        because the batched kernels read the feature table per item anyway,
        so supporting a per-leaf set costs one index and no branch, and a
        grower that wanted to accumulate only a node's sampled features could
        take it without a kernel change. Every item in a batch must still
        list the same *number* of features, which per-node sampling already
        guarantees since the count comes from the tree's set and the
        fraction.
        """
        if item < 0 or item >= self.max_items:
            raise Error("batch item index out of range")
        if len(features) == 0:
            raise Error("an active feature set must not be empty")
        if len(features) > self.n_features:
            raise Error("active feature set is larger than n_features")
        var dst = self.stage_feat.unsafe_ptr()
        for i in range(len(features)):
            if features[i] < 0 or features[i] >= self.n_features:
                raise Error("feature index out of range")
            dst.unsafe_store(item * self.n_features + i, Int32(features[i]))
        self.ctx.enqueue_copy(dst_buf=self.feat_dev, src_ptr=dst)

    def _stage_plan(mut self, plan: BatchPlan) raises:
        """Write the item table and the scales into pinned memory and upload
        both.

        **Ordering contract.** The three staging buffers are pinned host
        memory reused by every batch, and the host write below is not ordered
        against a copy the device has already been handed. So a batch's copies
        must have completed before the next batch stages over them: the
        caller synchronizes, or downloads a result, between two batches. This
        is the same contract `GpuSplitSearcher.enqueue` states for its allow
        mask and parameter block, and it is stated rather than enforced for
        the same reason, that enforcing it with a synchronization here would
        serialize exactly the pipelining a batch exists to buy. A caller that
        wants two batches in flight needs a staging ring, not a change to the
        kernels.
        """
        if plan.n_items() > self.max_items:
            raise Error("batch holds more items than this batcher allows")
        var dst = self.stage_items.unsafe_ptr()
        var sdst = self.stage_scales.unsafe_ptr()
        for i in range(plan.n_items()):
            var it = plan.items[i].copy()
            # Keeps `ITEM_DEAD` out of every host-staged plan, so the dead
            # branch in `_batch_zero_kernel` is reachable only from a
            # device-written plan and nothing on the shipping path can take
            # it. Stated as its own refusal rather than folded into the bound
            # below, because the bound would pass a negative count.
            if it.row_count < 0:
                raise Error("a batch item's row count must be nonnegative")
            if it.row_begin < 0 or it.row_begin + it.row_count > self.n_rows:
                raise Error("a batch item's rows escape the row buffer")
            if it.out_slot < 0 or it.out_slot >= self.pool.capacity:
                raise Error("a batch item's output slot is out of range")
            if it.plane < 0 or it.plane >= self.n_planes:
                raise Error("a batch item's gradient plane is out of range")
            if it.g_scale <= 0.0 or it.h_scale <= 0.0:
                raise Error("fixed-point scales must be positive")
            var base = i * ITEM_WORDS
            dst.unsafe_store(base + ITEM_BEGIN, Int32(it.row_begin))
            dst.unsafe_store(base + ITEM_COUNT, Int32(it.row_count))
            dst.unsafe_store(
                base + ITEM_ROWS_PER_TILE, Int32(it.rows_per_tile)
            )
            dst.unsafe_store(base + ITEM_TILE_BEGIN, Int32(it.tile_begin))
            dst.unsafe_store(base + ITEM_TILES, Int32(it.n_tiles))
            dst.unsafe_store(base + ITEM_OUT, Int32(it.out_slot))
            dst.unsafe_store(base + ITEM_PLANE, Int32(it.plane))
            dst.unsafe_store(base + 7, Int32(0))
            sdst.unsafe_store(SCALE_WORDS * i + SCALE_G, it.g_scale)
            sdst.unsafe_store(SCALE_WORDS * i + SCALE_H, it.h_scale)
        self.ctx.enqueue_copy(dst_buf=self.items_dev, src_ptr=dst)
        self.ctx.enqueue_copy(dst_buf=self.scales_dev, src_ptr=sdst)

    def enqueue_batch[
        bins_origin: MutOrigin,
        rows_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin, //
    ](
        mut self,
        plan: BatchPlan,
        bins: MutPointer[UInt8, bins_origin],
        rows: MutPointer[Int32, rows_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
    ) raises:
        """Enqueue one batched histogram build. Does not transfer or
        synchronize.

        `bins` is the device-resident binned matrix, `rows` the active-row
        permutation `GpuActiveRows` maintains, and `grad`/`hess` the round's
        gradient planes; every item's window indexes into `rows` absolutely,
        so nothing about which permutation it came from reaches the kernels.

        The zeroing pass runs whenever the batch does not write every feature
        slice of every output, which is whenever the feature set is narrowed,
        whenever the atomic strategy is in use, or whenever an item holds no
        rows. It is cheaper to state that as "always but the widest tiled
        case" than to make the caller reason about it, so that is what this
        does.
        """
        if plan.n_items() < 1:
            raise Error("a batch needs at least one item")
        if plan.n_bins != self.n_bins:
            raise Error("plan and batcher disagree on the bin count")
        if plan.n_slots > self.n_features:
            raise Error("plan holds more feature slots than features")
        if plan.strategy == STRATEGY_TILED:
            if plan.partial_cells > self.partial_capacity:
                raise Error(
                    "batch plan needs more partial cells than this batcher"
                    " allocated"
                )
        self._stage_plan(plan)

        var threads = plan.block_threads
        var hs = self.hist_size()
        var n_items = plan.n_items()

        var needs_zero = (
            plan.strategy != STRATEGY_TILED or plan.n_slots < self.n_features
        )
        if not needs_zero:
            for i in range(n_items):
                if plan.items[i].row_count <= 0:
                    needs_zero = True
                    break
        if needs_zero:
            var cells = n_items * N_PLANES * hs
            self.ctx.enqueue_function[_batch_zero_kernel](
                self.out_dev.unsafe_ptr(),
                self.items_dev.unsafe_ptr(),
                Int32(n_items),
                Int32(hs),
                grid_dim=_ceil_div(cells, threads),
                block_dim=threads,
            )

        if plan.strategy == STRATEGY_TILED:
            self.ctx.enqueue_function[_batch_hist_partial_kernel](
                bins,
                rows,
                grad,
                hess,
                self.feat_dev.unsafe_ptr(),
                self.items_dev.unsafe_ptr(),
                self.scales_dev.unsafe_ptr(),
                self.part_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Int32(n_items),
                Int32(plan.n_slots),
                Int32(self.n_bins),
                Int32(self.n_features),
                grid_dim=(plan.n_slots, plan.total_tiles),
                block_dim=threads,
            )
            var cells = n_items * N_PLANES * plan.n_slots * self.n_bins
            self.ctx.enqueue_function[_batch_reduce_kernel](
                self.part_dev.unsafe_ptr(),
                self.feat_dev.unsafe_ptr(),
                self.items_dev.unsafe_ptr(),
                self.out_dev.unsafe_ptr(),
                Int32(n_items),
                Int32(plan.n_slots),
                Int32(self.n_bins),
                Int32(hs),
                Int32(self.n_features),
                grid_dim=_ceil_div(cells, threads),
                block_dim=threads,
            )
        else:
            # Both new arms are OFF on the host-staged path, and neither is an
            # oversight. The quantized source rests on the scale pair being
            # global to the plan, which `_stage_plan` does not guarantee -- it
            # takes a per-item list precisely so a multiclass batch can vary
            # it -- and the hessian elision would have to be carried through
            # `_batch_hist_partial_kernel` and `_batch_reduce_kernel` as well
            # to cover the tiled strategy this path can also resolve to. So
            # this launch is the shipping one, statement for statement.
            self.ctx.enqueue_function[_batch_hist_atomic_kernel](
                bins,
                rows,
                grad,
                hess,
                self.gq_dev.unsafe_ptr(),
                self.feat_dev.unsafe_ptr(),
                self.items_dev.unsafe_ptr(),
                self.scales_dev.unsafe_ptr(),
                self.out_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Int32(n_items),
                Int32(plan.n_slots),
                Int32(self.n_bins),
                Int32(hs),
                Int32(self.n_features),
                Int32(0),
                Int32(0),
                grid_dim=(plan.n_slots, plan.total_tiles),
                block_dim=threads,
            )

    def enqueue_frontier_batch[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin, //
    ](
        mut self,
        plan: BatchPlan,
        bins: MutPointer[UInt8, bins_origin],
        mut rows: GpuActiveRows,
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
    ) raises:
        """`enqueue_batch` against the active-row permutation itself.

        The connection that makes a batch read *only* the rows its leaves own:
        the row pointer is `GpuActiveRows`' own buffer rather than something
        the caller found, and every item's window is checked against that
        permutation's live prefix before anything launches. A window outside
        `[0, n_active)` is not a bounds violation on the buffer (the buffer is
        `n_rows` long and the prefix is at most that), so nothing downstream
        would catch it: it would silently accumulate over rows this tree does
        not grow on, which under bagging is a different dataset.

        Enqueues only. No transfer, no synchronization, and no allocation, so
        a batch of thirty leaves costs the same host round trips as a batch of
        one: `_stage_plan`'s two copies and at most three launches.
        """
        if rows.n_rows != self.n_rows:
            raise Error(
                "the active-row permutation and this batcher were built for"
                " different datasets"
            )
        var live = rows.n_active()
        for i in range(plan.n_items()):
            var begin = plan.items[i].row_begin
            var count = plan.items[i].row_count
            if begin < 0 or count < 0 or begin + count > live:
                raise Error(
                    "a batch item's rows escape this tree's active row"
                    " prefix"
                )
        self.enqueue_batch(
            plan, bins, rows.rows_dev.unsafe_ptr(), grad, hess
        )

    # --- The device-written plan ------------------------------------------
    #
    # The gap this closes, in one sentence: `enqueue_batch` already builds a
    # whole frontier in two launches whatever its width, and the only thing
    # that keeps it out of a device-owned tree is that `_stage_plan` writes
    # `items_dev` from the host, so a plan cannot exist until the host has
    # read back the commit that implies it -- the round trip
    # `gpu_resident_round` exists to remove.
    #
    # The plan splits cleanly in two, and that split is the whole design:
    #
    #   geometry   ROWS_PER_TILE, TILE_BEGIN, TILES, PLANE, and the scales
    #   contents   BEGIN, COUNT, OUT
    #
    # The geometry is what the *grid* is derived from and a grid is a host
    # argument to `enqueue_function`; no kernel can change it. So the host
    # fixes it once per tree from a row bound, exactly as
    # `GpuActiveRows.enqueue_desc_histogram` sizes its atomic launch from
    # `max_rows` rather than from the row count it does not know. The
    # contents are what the commit decides, and those are three Int32 per
    # item that `gpu_tree_tables._pick_and_commit_kernel` writes in the same
    # launch that writes the step descriptor -- so the plan costs **zero
    # extra command buffers**, which is the property the census turns on.
    #
    # A geometry sized from a bound gives some threadgroups nothing to do,
    # and that is not a wrong answer: a tile past its item's real count has
    # `tile_begin >= count`, its row loop runs zero times, and its shared
    # histogram flushes nothing because every count bin is zero. The same
    # argument the descriptor histogram makes for its own bound.

    def stage_device_plan(
        mut self,
        n_items: Int,
        max_rows: Int,
        n_slots: Int,
        caps: DeviceCaps,
        g_scale: Float32,
        h_scale: Float32,
        plane: Int = 0,
    ) raises -> Int:
        """Fix the host half of a device-written plan and upload it. Returns
        the packed tile count the launch grid will use.

        Called once per tree, before the first commit that writes into the
        plan, and not again: the geometry is a function of the row bound, the
        feature width and the item count, and none of the three moves inside a
        tree. That is also what keeps `_stage_plan`'s pinned-memory ordering
        contract satisfiable here -- the staging buffers are written once and
        the device reads `items_dev` from then on, so there is no host write
        racing a launch.

        **Every item is staged dead.** `ITEM_COUNT` is `ITEM_DEAD` and
        `ITEM_OUT` is zero, so a batch enqueued before any commit kernel has
        filled the plan, or after growth has ended, zeroes nothing and
        accumulates nothing. This is the same discipline
        `_pick_and_commit_kernel` already applies to `STEP_LIVE`, and it is
        what allows a whole tree's launches to sit in the queue before the
        tree has decided how many splits it will take.

        The strategy is the atomic one and there is no choice about it, for
        the reason `enqueue_desc_histogram` states: the tiled strategy's
        partial buffer and its reduction are both sized by the tile count, the
        tile count would have to come from row counts the host does not have,
        and deriving it from the row bound instead would give every shallow
        leaf a reduction over thousands of empty tiles. The two strategies
        produce the identical histogram, so this is a launch decision and not
        a numeric one.
        """
        if n_items < 1 or n_items > self.max_items:
            raise Error(
                "a device-written plan holds between one item and this"
                " batcher's max_items"
            )
        if n_slots < 1 or n_slots > self.n_features:
            raise Error("active feature count out of range")
        if max_rows < 1:
            raise Error("a device-written plan needs a positive row bound")
        if max_rows > self.n_rows:
            raise Error("the row bound exceeds the row buffer")
        if plane < 0 or plane >= self.n_planes:
            raise Error("a plan's gradient plane is out of range")
        if g_scale <= 0.0 or h_scale <= 0.0:
            raise Error("fixed-point scales must be positive")

        var tiling = derive_tiling(
            caps,
            max_rows,
            n_slots,
            self.n_bins,
            STRATEGY_ATOMIC,
            0,
            0,
            0,
            0,
        )
        var tiles = tiling.n_tiles
        if tiles < 1:
            tiles = 1
        var total_tiles = n_items * tiles
        if total_tiles > MAX_GRID_DIM_Y:
            raise Error(
                "a device-written plan's packed tile axis exceeds the"
                " portable grid.y bound; give the batcher fewer items or a"
                " smaller row bound"
            )

        var dst = self.stage_items.unsafe_ptr()
        var sdst = self.stage_scales.unsafe_ptr()
        for k in range(n_items):
            var base = k * ITEM_WORDS
            dst.unsafe_store(base + ITEM_BEGIN, Int32(0))
            dst.unsafe_store(base + ITEM_COUNT, Int32(ITEM_DEAD))
            dst.unsafe_store(
                base + ITEM_ROWS_PER_TILE, Int32(tiling.rows_per_tile)
            )
            dst.unsafe_store(base + ITEM_TILE_BEGIN, Int32(k * tiles))
            dst.unsafe_store(base + ITEM_TILES, Int32(tiles))
            dst.unsafe_store(base + ITEM_OUT, Int32(0))
            dst.unsafe_store(base + ITEM_PLANE, Int32(plane))
            dst.unsafe_store(base + 7, Int32(0))
            sdst.unsafe_store(SCALE_WORDS * k + SCALE_G, g_scale)
            sdst.unsafe_store(SCALE_WORDS * k + SCALE_H, h_scale)
        self.ctx.enqueue_copy(dst_buf=self.items_dev, src_ptr=dst)
        self.ctx.enqueue_copy(dst_buf=self.scales_dev, src_ptr=sdst)

        self.plan_items = n_items
        self.plan_tiles_per_item = tiles
        self.plan_slots = n_slots
        # Cleared per tree with the rest of the plan's host half: a hint left
        # over from the previous tree's last level would narrow this tree's
        # first grid. Zero is the safe value and means "use the staged width".
        self.plan_level_pairs = 0
        # The round's scales and plane, kept host side for the quantization
        # pass, and the invalidation that makes that pass once per tree. This
        # is the only place the pair is written, which is what makes "the
        # scale is global to the plan" a property of the code rather than an
        # assumption: the loop above stores the SAME `g_scale` and `h_scale`
        # into every item's row of `scales_dev`.
        self.plan_g_scale = g_scale
        self.plan_h_scale = h_scale
        self.plan_plane = plane
        self.quant_valid = False
        return total_tiles

    def set_level_pairs(mut self, pairs: Int):
        """Tell the next subtracting level batch how many pairs its level can
        hold at most. Zero, the initial value, means "assume the staged
        width".

        **An upper bound is all this has to be, and that is what makes it a
        host-side number rather than a readback.** An oblivious level at depth
        `l` commits `1 << l` parents and `2 << l` children, so the caller that
        knows `l` knows the pair count exactly whenever the level committed at
        all; and when growth stopped, `_kill_level_plan` kills the WHOLE staged
        width, `_live_item_pairs` answers zero on the device, and every block
        of whatever grid was launched returns at its first comparison. So the
        bound is exact when it matters and harmlessly large when it does not.
        Too LARGE only costs threadgroups that return; too SMALL would drop a
        child's histogram, which is why the launch clamps this into
        `[1, plan_items / 2]` rather than trusting it.

        Read by the pair-indexed arm alone. Every other arm launches the
        staged geometry and cannot see this value at all."""
        self.plan_level_pairs = pairs if pairs > 0 else 0

    def device_plan_staged(self) -> Bool:
        """Whether `stage_device_plan` has run on this batcher.

        Exposed so a caller can branch between the two arms rather than catch,
        and so a test can assert that the device-plan arm was actually entered
        instead of asserting that the host arm produced the right answer and
        calling that a pass."""
        return self.plan_items > 0

    def _quant_flag(self) -> Int32:
        """`use_quant` for a device-plan launch: on only when the arm is
        selected AND the buffer holds this tree's round."""
        var on = self.quant_grads and self.quant_valid
        return Int32(1) if on else Int32(0)

    def _const_hess_flag(self) -> Int32:
        return Int32(1) if self.constant_hessian else Int32(0)

    def _ensure_quantized[
        grad_origin: MutOrigin,
        hess_origin: MutOrigin, //
    ](
        mut self,
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
    ) raises:
        """Rebuild the interleaved quantized buffer unless it already holds
        this tree's round. One streaming launch over the staged plane's rows,
        ordered before the accumulation that reads it by the queue.

        **Once per tree and not once per level.** `stage_device_plan` is the
        only invalidation and it runs once per tree, and the gradients cannot
        move inside a tree because a round computes them once. So a depth-6
        tree pays this on the level-zero enqueue and on no other:
        `gpu_resident_round.oblivious_launch_census(6)` goes from 62 command
        buffers to 63, under the measured 64-deep queue by one instead of by
        two. That is the cost, it is the reason the arm is default off, and
        folding the pass into `_batch_copy_back_zero_kernel` on the level-zero
        pass would take it back to 62.

        Enqueues only. Nothing here waits, because unlike the Int16 staging
        arm in `gpu_active_rows` there is no bound to read back: Int32 holds
        every value the Float32 arm would have formed.

        Called at the top of every device-plan accumulation, so a caller that
        never turns the arm on pays one Bool test per level.
        """
        if not self.quant_grads or self.quant_valid:
            return
        if self.plan_items < 1:
            raise Error(
                "the quantized gradient source needs a staged plan: its"
                " scales and plane come from stage_device_plan"
            )
        var threads = self.block_threads
        self.ctx.enqueue_function[_batch_quantize_kernel](
            grad,
            hess,
            self.gq_dev.unsafe_ptr(),
            Int32(self.plan_plane * self.n_rows),
            Int32(self.n_rows),
            self.plan_g_scale,
            self.plan_h_scale,
            grid_dim=_ceil_div(self.n_rows, threads),
            block_dim=threads,
        )
        self.quant_valid = True

    def enqueue_device_plan_batch[
        bins_origin: MutOrigin,
        rows_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        rows: MutPointer[Int32, rows_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
    ) raises:
        """Build every item of the device-written plan. **Two launches,
        whatever the plan holds**, plus the one quantization pass a tree pays
        on its first level when the quantized source is on. Does not stage,
        transfer, or synchronize.

        This is `enqueue_batch` with the staging removed and the strategy
        fixed, and it reads `items_dev` exactly as that path's kernels do, so
        the two arms run the same two kernels over the same table and differ
        only in who wrote the table. That is what makes the batched shape
        bit-identical to the host-staged one by inspection rather than by
        argument: both arms launch `_batch_hist_atomic_kernel` and
        `_batch_zero_kernel`.

        This paragraph used to end "there is one accumulation kernel and one
        zeroing kernel in this module and both arms launch those", which stopped
        being true when the subtracting pair and then `_plan_hist_kernel` were
        added. The inspection argument is unchanged and now rests on naming the
        two kernels rather than on there being only two; under
        `plan_lean_requested` this arm launches `_plan_hist_kernel` instead,
        whose own docstring carries the identity argument leg by leg. Corrected
        2026-08-17.

        The zeroing is unconditional here, where `enqueue_batch` reasons about
        whether it is needed. It always is: this arm is always the atomic
        strategy, which accumulates into whatever the slot already holds.
        """
        if self.plan_items < 1:
            raise Error(
                "no device-written plan is staged; call stage_device_plan"
                " before enqueueing a plan batch"
            )
        var n_items = self.plan_items
        var total_tiles = n_items * self.plan_tiles_per_item
        var threads = self.block_threads
        var hs = self.hist_size()

        # No-op unless the quantized source is on and the buffer is stale,
        # which is once per tree; see `_ensure_quantized`.
        self._ensure_quantized(grad, hess)
        var cells = n_items * N_PLANES * hs
        self.ctx.enqueue_function[_batch_zero_kernel](
            self.out_dev.unsafe_ptr(),
            self.items_dev.unsafe_ptr(),
            Int32(n_items),
            Int32(hs),
            grid_dim=_ceil_div(cells, threads),
            block_dim=threads,
        )
        if plan_lean_requested():
            self._launch_plan_hist(bins, rows, grad, hess, False)
            return
        self.ctx.enqueue_function[_batch_hist_atomic_kernel](
            bins,
            rows,
            grad,
            hess,
            self.gq_dev.unsafe_ptr(),
            self.feat_dev.unsafe_ptr(),
            self.items_dev.unsafe_ptr(),
            self.scales_dev.unsafe_ptr(),
            self.out_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(n_items),
            Int32(self.plan_slots),
            Int32(self.n_bins),
            Int32(hs),
            Int32(self.n_features),
            self._quant_flag(),
            self._const_hess_flag(),
            grid_dim=(self.plan_slots, total_tiles),
            block_dim=threads,
        )

    def plan_lean_geometry(self, subtract: Bool) raises -> List[Int]:
        """The lean kernel's resolved launch geometry, as
        `[grid_x, total_tiles, tiles_per_item, split_rows, n_copies,
        group_width, units, subtract]`.

        Split out from the launch so a test can assert what the switches
        resolved to without launching anything, which is the only way to prove
        a knob's gate opened rather than assume it. Raises rather than clamping
        when the requested group and copy counts do not fit threadgroup memory,
        because a clamp would silently drop a feature's accumulation.
        """
        if self.plan_items < 1 or self.plan_slots < 1:
            raise Error(
                "no device-written plan is staged; call stage_device_plan"
                " before asking for a lean plan geometry"
            )
        var nb = self.n_bins
        var stride = nb + 1
        var threads = self.block_threads

        var split = plan_row_split_requested()
        var tpi = self.plan_tiles_per_item
        var split_rows = 0
        if split > 0:
            tpi = split
            split_rows = 1
        if tpi < 1:
            tpi = 1

        var gw = plan_group_requested()
        if gw > PLAN_MAX_GROUP:
            gw = PLAN_MAX_GROUP
        if gw > self.plan_slots:
            gw = self.plan_slots
        if gw < 1:
            gw = 1

        var copies = plan_private_copies_requested(nb)
        if copies > threads:
            copies = threads
        if copies < 1:
            copies = 1

        # The two budgets, in cells. The group is resolved first and the
        # copies give way to it, because the group's win is a traffic argument
        # over bytes and the copies' is a contention argument this file cannot
        # price. Neither is clamped silently past that point.
        var wide = PLAN_UNITS_WIDE * PLAN_UNIT_CELLS
        var narrow = PLAN_UNITS_NARROW * PLAN_UNIT_CELLS
        var need = gw * copies * N_PLANES * stride
        if need > wide:
            copies = wide // (gw * N_PLANES * stride)
            if copies < 1:
                copies = 1
            need = gw * copies * N_PLANES * stride
        if need > wide:
            raise Error(
                "a lean plan batch cannot fit its feature group in"
                " threadgroup memory: lower MOJOTREES_GPU_HIST_GROUP"
            )
        var units = PLAN_UNITS_WIDE
        if need <= narrow:
            units = PLAN_UNITS_NARROW

        var total_tiles = self.plan_items * tpi
        if total_tiles > MAX_GRID_DIM_Y:
            raise Error(
                "a lean plan batch's packed tile axis exceeds the portable"
                " grid.y bound: lower MOJOTREES_GPU_HIST_ROW_SPLIT"
            )
        var out = List[Int]()
        out.append(_ceil_div(self.plan_slots, gw))
        out.append(total_tiles)
        out.append(tpi)
        out.append(split_rows)
        out.append(copies)
        out.append(gw)
        out.append(units)
        out.append(1 if subtract else 0)
        return out^

    def _launch_plan_hist[
        bins_origin: MutOrigin,
        rows_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        rows: MutPointer[Int32, rows_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        subtract: Bool,
    ) raises:
        """`_plan_hist_kernel` at the geometry the switches resolved.

        The zeroing pass is the caller's, unchanged: this replaces only the
        accumulation launch, so both arms are still two launches per level and
        `gpu_resident_round.oblivious_launch_census(6)` is still 62. That is a
        precondition of the whole plane and not a happy result; see
        `enqueue_device_plan_batch_fused_subtracting`.
        """
        var g = self.plan_lean_geometry(subtract)
        var hs = self.hist_size()
        var threads = self.block_threads
        if g[6] == PLAN_UNITS_NARROW:
            self.ctx.enqueue_function[_plan_hist_kernel[PLAN_UNITS_NARROW]](
                bins,
                rows,
                grad,
                hess,
                self.gq_dev.unsafe_ptr(),
                self.feat_dev.unsafe_ptr(),
                self.items_dev.unsafe_ptr(),
                self.scales_dev.unsafe_ptr(),
                self.out_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Int32(self.plan_items),
                Int32(self.plan_slots),
                Int32(self.n_bins),
                Int32(hs),
                Int32(self.n_features),
                Int32(g[2]),
                Int32(g[3]),
                Int32(g[4]),
                Int32(g[5]),
                Int32(g[7]),
                self._quant_flag(),
                self._const_hess_flag(),
                grid_dim=(g[0], g[1]),
                block_dim=threads,
            )
        else:
            self.ctx.enqueue_function[_plan_hist_kernel[PLAN_UNITS_WIDE]](
                bins,
                rows,
                grad,
                hess,
                self.gq_dev.unsafe_ptr(),
                self.feat_dev.unsafe_ptr(),
                self.items_dev.unsafe_ptr(),
                self.scales_dev.unsafe_ptr(),
                self.out_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Int32(self.plan_items),
                Int32(self.plan_slots),
                Int32(self.n_bins),
                Int32(hs),
                Int32(self.n_features),
                Int32(g[2]),
                Int32(g[3]),
                Int32(g[4]),
                Int32(g[5]),
                Int32(g[7]),
                self._quant_flag(),
                self._const_hess_flag(),
                grid_dim=(g[0], g[1]),
                block_dim=threads,
            )

    def enqueue_device_plan_batch_fused[
        bins_origin: MutOrigin,
        rows_origin: MutOrigin,
        scratch_origin: MutOrigin,
        desc_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        rows: MutPointer[Int32, rows_origin],
        scratch: MutPointer[Int32, scratch_origin],
        desc: MutPointer[Int32, desc_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        copy_back_blocks: Int = 0,
    ) raises:
        """`enqueue_device_plan_batch`, also paying the descriptor
        partition's deferred copy-back, and still in two launches.

        This is the entry point a level schedule wants, and the reason is
        arithmetic rather than taste. `GpuActiveRows.enqueue_partition_desc`
        defers its copy-back under `set_partition_fusion(True)`, which is the
        default, and the debt is paid by the next `enqueue_desc_histogram`. A
        batched build does not go through that entry point, so a schedule using
        the plain `enqueue_device_plan_batch` would have to run the partition
        with its fusion off and spend a third launch on the copy-back: one per
        level, six per depth-6 tree, which is the whole margin between 62
        command buffers and 68. `_batch_copy_back_zero_kernel` carries it in
        the zeroing pass instead, which this arm has to launch anyway.

        `desc` is the step descriptor the partition was launched against,
        `scratch` the active-row scratch buffer it compacted into, and
        `copy_back_blocks` the block count that partition used, so the fused
        grid is at least as wide as the copy-back's own would have been. Both
        halves are grid-strided or singly guarded, so a wider grid changes no
        store; see the kernel.

        **The caller still owes `GpuActiveRows` the bookkeeping half of the
        debt.** Nothing here can clear `copy_back_debt`, because that flag
        lives in a struct this module does not own and is checked by four
        refusals there. A schedule wiring this up marks the debt paid on the
        rows object in the same place it enqueues this.
        """
        if self.plan_items < 1:
            raise Error(
                "no device-written plan is staged; call stage_device_plan"
                " before enqueueing a plan batch"
            )
        var n_items = self.plan_items
        var total_tiles = n_items * self.plan_tiles_per_item
        var threads = self.block_threads
        var hs = self.hist_size()

        # No-op unless the quantized source is on and the buffer is stale,
        # which is once per tree; see `_ensure_quantized`. It is ahead of the
        # zeroing pass so that the level-zero enqueue is quantize, zero,
        # accumulate, and the two later launches are unmoved.
        self._ensure_quantized(grad, hess)
        var cells = n_items * N_PLANES * hs
        var blocks = _ceil_div(cells, threads)
        if copy_back_blocks > blocks:
            blocks = copy_back_blocks
        self.ctx.enqueue_function[_batch_copy_back_zero_kernel](
            self.out_dev.unsafe_ptr(),
            self.items_dev.unsafe_ptr(),
            Int32(n_items),
            Int32(hs),
            rows,
            scratch,
            desc,
            grid_dim=blocks,
            block_dim=threads,
        )
        if plan_lean_requested():
            self._launch_plan_hist(bins, rows, grad, hess, False)
            return
        self.ctx.enqueue_function[_batch_hist_atomic_kernel](
            bins,
            rows,
            grad,
            hess,
            self.gq_dev.unsafe_ptr(),
            self.feat_dev.unsafe_ptr(),
            self.items_dev.unsafe_ptr(),
            self.scales_dev.unsafe_ptr(),
            self.out_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(n_items),
            Int32(self.plan_slots),
            Int32(self.n_bins),
            Int32(hs),
            Int32(self.n_features),
            self._quant_flag(),
            self._const_hess_flag(),
            grid_dim=(self.plan_slots, total_tiles),
            block_dim=threads,
        )

    def enqueue_device_plan_batch_fused_subtracting[
        bins_origin: MutOrigin,
        rows_origin: MutOrigin,
        scratch_origin: MutOrigin,
        desc_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        rows: MutPointer[Int32, rows_origin],
        scratch: MutPointer[Int32, scratch_origin],
        desc: MutPointer[Int32, desc_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        copy_back_blocks: Int = 0,
    ) raises:
        """`enqueue_device_plan_batch_fused` with sibling subtraction, for an
        oblivious level's plan. **Still two launches, and the same two grids.**

        The identical argument list and the identical launch geometry as the
        arm it replaces, so the only difference between the two is which two
        kernel symbols the queue receives. That is a precondition rather than a
        happy result: `gpu_resident_round.oblivious_launch_census(6)` is 62
        command buffers with two per level here, the Metal queue on the
        measured machine is 64 deep and does not raise when overrun, and a
        third launch per level would put a depth-6 tree at 68. A standalone
        subtraction pass is exactly that third launch, which is why the
        subtraction is folded into the accumulation instead
        (`_batch_hist_atomic_subtract_kernel`) and the parent copy the lopsided
        case needs is folded into the zeroing
        (`_batch_copy_back_zero_subtract_kernel`). The deferred copy-back is
        carried in the same pass and by the same statements as before, so the
        partition still costs two.

        **This arm is for an oblivious LEVEL plan and for nothing else**, and
        the reason is a slot relationship rather than a shape:
        `gpu_tree_tables._commit_level_kernel` pins slot index to leaf index,
        which puts the parent's histogram in the left child's destination slot
        and lays the level's items out as `L` left children then `L` right
        children. A leaf-wise two-item plan satisfies neither, so the caller
        selects this arm at the one oblivious call site rather than the
        environment selecting it inside the shipping one. See
        `oblivious_subtract_requested`.

        What reaches the pool afterwards is bit-for-bit what the shipping arm
        would have left there; the argument is at the accumulation kernel.
        Enqueues only: no transfer and no synchronization.
        """
        if self.plan_items < 1:
            raise Error(
                "no device-written plan is staged; call stage_device_plan"
                " before enqueueing a plan batch"
            )
        if self.plan_items % 2 != 0:
            raise Error(
                "a subtracting level batch needs an even item width: the"
                " level's children come in pairs and the plan is staged at"
                " 1 << max_depth"
            )
        var n_items = self.plan_items
        var total_tiles = n_items * self.plan_tiles_per_item
        var threads = self.block_threads
        var hs = self.hist_size()

        # No-op unless the quantized source is on and the buffer is stale,
        # which is once per tree; see `_ensure_quantized`.
        self._ensure_quantized(grad, hess)
        var cells = n_items * N_PLANES * hs
        var blocks = _ceil_div(cells, threads)
        if copy_back_blocks > blocks:
            blocks = copy_back_blocks
        self.ctx.enqueue_function[_batch_copy_back_zero_subtract_kernel](
            self.out_dev.unsafe_ptr(),
            self.items_dev.unsafe_ptr(),
            Int32(n_items),
            Int32(hs),
            rows,
            scratch,
            desc,
            grid_dim=blocks,
            block_dim=threads,
        )
        if plan_lean_requested():
            self._launch_plan_hist(bins, rows, grad, hess, True)
            return
        if pair_grid_requested():
            self._launch_pair_hist(bins, rows, grad, hess)
            return
        self.ctx.enqueue_function[_batch_hist_atomic_subtract_kernel](
            bins,
            rows,
            grad,
            hess,
            self.gq_dev.unsafe_ptr(),
            self.feat_dev.unsafe_ptr(),
            self.items_dev.unsafe_ptr(),
            self.scales_dev.unsafe_ptr(),
            self.out_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(n_items),
            Int32(self.plan_slots),
            Int32(self.n_bins),
            Int32(hs),
            Int32(self.n_features),
            self._quant_flag(),
            self._const_hess_flag(),
            grid_dim=(self.plan_slots, total_tiles),
            block_dim=threads,
        )

    def _launch_pair_hist[
        bins_origin: MutOrigin,
        rows_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        rows: MutPointer[Int32, rows_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
    ) raises:
        """`_batch_hist_pair_subtract_kernel` at the pair-indexed geometry.

        The zeroing pass is the caller's, unchanged, so this replaces only the
        accumulation launch and both arms are still two launches per level:
        `gpu_resident_round.oblivious_launch_census(6)` is still 62. That is a
        precondition of this plane and not a happy result; see
        `enqueue_device_plan_batch_fused_subtracting`.

        THE GRID, AND THE ONE DIRECTION THE CLAMP PROTECTS
        --------------------------------------------------
        `grid.y` is `pairs * tiles` where `pairs` is the level's own pair
        count when a caller supplied it (`set_level_pairs`) and half the
        staged width otherwise. Half the staged width is the ALWAYS-CORRECT
        answer and is what this falls back to, because a level can never hold
        more pairs than the plan was staged for; the hint only narrows it. A
        hint that was too large would cost threadgroups that return at their
        first comparison, and a hint that was too small would silently drop a
        child's histogram, so the clamp is written in that direction and the
        hint is never trusted upward."""
        var threads = self.block_threads
        var hs = self.hist_size()

        var tpi = self.plan_tiles_per_item
        var split_rows = 0
        var req = pair_grid_tiles()
        if req > 0:
            tpi = req
            split_rows = 1
        if tpi < 1:
            tpi = 1

        var pairs = self.plan_items // 2
        if self.plan_level_pairs > 0 and self.plan_level_pairs < pairs:
            pairs = self.plan_level_pairs
        if pairs < 1:
            pairs = 1

        var total_tiles = pairs * tpi
        if total_tiles > MAX_GRID_DIM_Y:
            raise Error(
                "a pair-indexed batch's packed tile axis exceeds the portable"
                " grid.y bound: lower MOJOTREES_GPU_HIST_PAIR_TILES"
            )
        self.ctx.enqueue_function[_batch_hist_pair_subtract_kernel](
            bins,
            rows,
            grad,
            hess,
            self.gq_dev.unsafe_ptr(),
            self.feat_dev.unsafe_ptr(),
            self.items_dev.unsafe_ptr(),
            self.scales_dev.unsafe_ptr(),
            self.out_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.plan_items),
            Int32(self.plan_slots),
            Int32(self.n_bins),
            Int32(hs),
            Int32(self.n_features),
            Int32(tpi),
            Int32(split_rows),
            self._quant_flag(),
            self._const_hess_flag(),
            grid_dim=(self.plan_slots, total_tiles),
            block_dim=threads,
        )

    def enqueue_subtract(
        mut self, parent_slot: Int, child_slot: Int, dst_slot: Int
    ) raises:
        """`dst = parent - child`, on the device, over one histogram slot.

        Refuses operands whose stamps differ, which is the check that the two
        histograms were accumulated under the same fixed-point scales and the
        same active feature set. `dst_slot` may be `parent_slot`, which
        derives the sibling in place and is how a frontier keeps one slot per
        live leaf.
        """
        self.pool.check_subtractable(parent_slot, child_slot)
        if dst_slot < 0 or dst_slot >= self.pool.capacity:
            raise Error("histogram slot out of range")
        var cells = self.slot_cells()
        var threads = self.block_threads
        self.ctx.enqueue_function[_subtract_slice_kernel](
            self.out_dev.unsafe_ptr(),
            Int32(parent_slot),
            Int32(child_slot),
            Int32(dst_slot),
            Int32(cells),
            grid_dim=_ceil_div(cells, threads),
            block_dim=threads,
        )

    def download_slot(mut self, slot: Int) raises -> List[Int32]:
        """One slot's fixed-point histogram, host side. Synchronizes.

        The words come back in `GpuHistogramBuilder`'s own
        `[grad | hess | count]` layout, so a caller converts them with the
        same arithmetic `histogram_from_host` uses and needs no second
        decoder.

        This is the path the batching exists to avoid, so it is deliberately
        the unoptimized one: a mapping of the whole output pool rather than a
        pinned one-way copy of one slice. It is here so a batched result can
        be read at all, by a caller mixing batched construction with a
        host-side split search and by anything checking a batched histogram
        against a single-leaf one. A caller doing that per node should move
        the search to the device instead of making this fast.
        """
        self.pool.check_live(slot)
        var cells = self.slot_cells()
        var out = List[Int32](capacity=cells)
        with self.out_dev.map_to_host() as host:
            var p = host.unsafe_ptr()
            for i in range(cells):
                out.append(p.unsafe_load(slot * cells + i))
        return out^

    def download_slots(mut self, slots: List[Int]) raises -> List[Int32]:
        """Several slots' fixed-point histograms in one mapping.

        The whole point of batching is that a launch is not paid per leaf, and
        `download_slot` gives that back on the way home: one `map_to_host` per
        leaf is one synchronization per leaf. This maps once and copies every
        requested slot out of that mapping, so a batch of thirty costs one.

        The result is the slots concatenated in the order given, `slot_cells()`
        words each, so slot `k` of the request starts at
        `k * slot_cells()`. Each one is in `GpuHistogramBuilder`'s
        `[grad | hess | count]` layout, exactly as `download_slot` returns it.
        """
        if len(slots) < 1:
            raise Error("a download needs at least one slot")
        var cells = self.slot_cells()
        for i in range(len(slots)):
            self.pool.check_live(slots[i])
            for k in range(i):
                if slots[k] == slots[i]:
                    raise Error("a download may not list a slot twice")
        var out = List[Int32](capacity=len(slots) * cells)
        with self.out_dev.map_to_host() as host:
            var p = host.unsafe_ptr()
            for i in range(len(slots)):
                var base = slots[i] * cells
                for c in range(cells):
                    out.append(p.unsafe_load(base + c))
        return out^
