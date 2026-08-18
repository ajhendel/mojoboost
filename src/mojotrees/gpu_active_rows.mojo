"""Device-side active-row compaction.

The GPU trainer currently finds a node's rows by scanning all `n_rows` and
filtering on a per-row leaf id (see histogram_gpu.mojo): every histogram
costs a full pass over the dataset whatever the node's size, so a tree costs
`num_leaves * n_rows * n_features` bin reads where the CPU builder's
row-index lists cost `n_rows * n_features` per level. This module holds the
replacement: one device-resident permutation of the row indices in which
every live leaf owns a *contiguous* half-open range, so a node's histogram
reads exactly its own rows and nothing else.

The layout
----------
`rows[0 : n_active)` is a permutation of the rows this tree grows on (the
bag, or every row when unbagged). Leaf `node` owns `rows[begin : end)`, its
`LeafRange`. Splitting a leaf stably partitions its own range in place: the
rows going left come first, in their original relative order, then the rows
going right, also in their original relative order. The children's ranges
are therefore `[begin, begin + n_left)` and `[begin + n_left, end)` and the
live leaves keep tiling `[0, n_active)` exactly, with no gaps and no
overlap, for the whole tree. Nothing outside the parent's range is ever read
or written by its split, so sibling leaves cannot be disturbed.

Why stability matters
---------------------
The CPU grower's `partition_rows_into` (tree.mojo) is a block-counted,
prefix-summed, stable partition, and its comment records the property this
module has to keep: both sides come out in ascending row order whatever the
block count, so the parallel result is identical to the serial one, index
for index. Compaction here reproduces that exactly, which is what lets the
two backends be compared row list against row range. Since the partition is
stable in *buffer* order rather than in row-id order, a root seeded from a
GOSS sample that is not ascending still tracks the CPU grower's list, which
is seeded from the same sample in the same order.

Determinism
-----------
The device partition is a flag pass, an exclusive prefix sum, and a scatter.
Every destination index is a pure function of the element's position and the
integer prefix sums, so no atomic decides where a row lands and no run-to-run
scheduling difference can reorder the output. The prefix sums are exact
Int32 sums over 0/1 flags, so they cannot round.

Say it once more precisely, because a later reader will want to change the
launch shape again and this is the invariant that constrains what may be
changed. Write `flag(i)` for whether element `i` of the range goes left,
`P(j)` for the number of `i < j` with `flag(i)`, and `T` for `P(count)`. The
partition writes element `j` to `P(j)` when it goes left and to `T + (j -
P(j))` when it goes right. `P` and `T` are properties of the flags and of the
positions and of nothing else, so *no* choice of how the range is cut into
threadgroups, how many threadgroups there are, how many tiles each one walks,
or which kernel arm computes the prefix can move a single row. That is why
this module treats the block cap, the tile count, and the scan arm as
launch-shape knobs and says so at every one of them.

Collective primitives, and a rule this module repeals
-----------------------------------------------------
The scan inside a threadgroup is `max.gpu.primitives.block.prefix_sum`, warp
shuffles plus a single shared round, in place of the hand-rolled
Hillis-Steele scan that cost two barriers per doubling step. That repeals the
standing project rule against warp-level primitives, on the grounds that the
block and warp modules are portable across NVIDIA, AMD, and Metal and are
Modular's to tune, so calling them inherits their improvements instead of
maintaining a scan here forever. The two rules that stay are untouched: there
is still one portable source with no per-backend branch, and the scan is
still integer-only, so the determinism above is unaffected, `prefix_sum` over
Int32 being exact in whatever order it sums. The hand-rolled kernels are kept
as a fallback arm, selected by `MOJOTREES_GPU_SCAN_PRIMITIVES=0` or
`set_scan_primitives(False)`, so a backend where the primitive misbehaves has
somewhere to go and a benchmark can hold both arms in one process. The two
arms produce the identical permutation, asserted element for element in
tests/test_gpu_scan_primitives.mojo. Neither arm has been timed against the
other on any device, so nothing here claims one is faster.

What the partition costs, and what does not help
------------------------------------------------
Measured on an M4 with an interleaved A/B against the same buffers: a
leaf-wise chain of seven partitions from 5M rows down to 78k takes about
4.6 ms of device time in total, so a 100-tree, 31-leaf fit spends roughly a
second partitioning against ten-plus seconds of histograms; the phase trace
in train_gpu.mojo charges the phase more than that only because its sync
after `apply_split` pays queue latency the untraced fit does not. Carrying
the flag in the offset word so the scatter skips its bin gather measured
1.05x on that chain, resolved but small, because the flag pass has just
pulled the same lines into cache. A single-block in-place kernel for ranges
under one threadgroup (flag, scan, scatter in one launch, no scratch, no
copy back) measured 1.00x on 256 partitions of 600 rows, 78 us each either
way: the per-partition cost there is enqueue overhead, not launch count, so
it was not kept. Do not retry either without a new reason.

The launch count, and an open tension
-------------------------------------
A partition is three launches: flag and scan, scatter, copy back. It was four
until the block-sums scan was folded into the head of the scatter; see
`_scatter_kernel` for how, `_partition_grid` for the block cap that keeps the
folded scan's redundant reads bounded, and `_copy_back_kernel` for the design
that would remove the third launch and for the readers that stopped it being
built here.

The reason to want fewer launches is a stage-level profile of a 1M by 50
round of 100 trees on an M4, which charged partition 17.7 percent against
histogram's 49.3 and transfer's 28.2, and a follow-up audit of it which
believed roughly half of that 17.7 was fence attribution and put partition's
real floor at a four-launch minimum near 237 microseconds on a machine where
an enqueue costs about 20 microseconds and a host wait about 126. On that
reading a launch is worth removing on its own.

That reading is in tension with the paragraph above, which records that
collapsing a small range's four launches into one measured 1.00x. Both cannot
be the whole story, and this module does not know which is right: neither
number was re-measured by the change that removed the launch, and nothing
here claims the removal made anything faster. What it claims is that three
launches compute what four computed, element for element, which is a
statement about the permutation and not about time.

Bagging
-------
Compaction subsumes the OUT_OF_BAG sentinel: the bag is written into the
first `len(bag)` slots and the root range covers only those, so an unbagged
row is simply not in the range any kernel iterates. Nothing has to mark it,
match it, or skip it, and a bagged tree reads `len(bag)` rows per node
instead of `n_rows`.

Missing values and categoricals
-------------------------------
`RowRouting.goes_left` is the same rule as `Tree.goes_left`,
`SplitInfo.goes_left`, and the existing `_partition_kernel`: a categorical
node routes by 256-bit set membership (bin 0, the missing/unseen/dropped
bin, is never a member, so those rows go right), a numerical node sends the
feature's missing bin whichever way `default_left` says, and every other row
goes by the inclusive threshold. The host reference model and the device
kernels call the same rule with the same arguments, so they cannot drift.

Bounds safety
-------------
Ranges are validated on the host before any launch: a range must sit inside
`[0, n_active)`, `n_left` must lie in `[0, count]`, and child ids must be
new and distinct from the parent. Every kernel guards its own element index
against the range length, so the tail block of a partial threadgroup writes
nothing. Row ids are Int32 and are checked against `n_rows` when the root is
seeded, so the indirect bin loads `bins[feature * n_rows + row]` stay in the
matrix.

Allocation
----------
Every device buffer is allocated once, at construction, sized by `n_rows`
(plus one Int32 per scan block and one for the total). Splitting a leaf
allocates nothing on the device and, on the host, at most appends to the
range table. The scatter writes into a second full-size buffer and a copy
kernel folds just the parent's range back, so the ranges other leaves hold
stay valid: ping-ponging whole buffers would invalidate them.

**Ownership and growth.** `GpuActiveRows` owns exactly the index machinery:
`rows`, `scratch`, `offsets` (one Int32 per row each), the per-scan-block sums,
the one-element total, and the pinned staging buffers. Nothing here ever
reallocates: the buffers are sized for the *dataset*, and every tree, node,
and split works inside that fixed footprint, so there is no capacity growth
policy because there is no capacity that grows. What does grow is the host
`LeafRangeTable`, by at most two `LeafRange` appends per split, and it is
cleared at every `reset_root`. A dataset larger than the buffers is not a
resize, it is a different `GpuActiveRows`.

Two partition planes, one host table
------------------------------------
`partition` and `enqueue_partition_desc` write the same device row buffer and
only the first writes the host `LeafRangeTable`. The second routes rows from a
step descriptor a device kernel filled, so the host does not know which leaf
was split or how the rows fell, and updating the table from here would mean
reading the device back -- which is the wait the resident plane exists to
delete.

The consequence is that a host table is *stale*, not wrong-looking, for the
whole of a device-owned tree: its windows are in bounds, they tile the active
prefix, and they describe the tree before the first split. That combination
shipped a bug. `train_gpu`'s device-gradient round reads this table to advance
the raw scores, read one that still said node 0 owned everything, and added the
root's value to every row; tree 0 was bit-identical and every tree after it
diverged, with nothing raised.

So the table carries a validity state. `enqueue_partition_desc` poisons it,
every accessor that returns a window refuses while poisoned, and
`gpu_resident_round._publish_row_ranges` -- which replays the device's commit
log onto it -- is the only thing that clears it, and only after checking that
the replay covered every node the tree ended with. `LeafRangeTable` carries
the argument, including the alternatives rejected.

Empty leaves
------------
An empty range is a first-class state, not an error. `partition` of an empty
window enqueues nothing and reports zero rows left, and its two children are
recorded as empty ranges at the parent's own offset, so the live ranges keep
tiling `[0, n_active)`. `enqueue_range_histogram` for an empty node zeroes the
output and returns without launching an accumulation, which is the histogram
that node has. A leaf that holds no rows therefore costs one zeroing pass and
no row work, and never a special case in the caller.

Multiclass, bagging, and GOSS
-----------------------------
One `GpuActiveRows` holds one permutation, and a K-class round grows K trees
over it, one per class, in sequence: `begin_tree` reseeds the root and clears
the range table, so class `k + 1` starts from a clean permutation. The class
index does not reach this module at all, because which rows exist does not
depend on the class; it reaches the gradient plane instead, through
`LeafFrontier.plane` and `gpu_leaf_batching.ITEM_PLANE`.

Bagging and GOSS reach it in exactly one way: as the `bag` list handed to
`begin_tree`, which becomes the live prefix. There are no per-row weights on
the device and none are wanted. GOSS's amplification of the small-gradient
sample is applied to the gradients before they are uploaded (see
`goss.apply_goss_scaling`), so a row's weight is already inside `grad[r]` and
`hess[r]` by the time any kernel here reads it. A second weight vector would
be a second place to scale, and two places to scale is one too many.

Row compaction, and what it is a trade against
----------------------------------------------
Everything above compacts the *index*: a leaf's rows are contiguous in
`rows[]`. What that does not fix is where those rows point. At the root
`rows[j] == j`, so the histogram's `bins[f * n_rows + rows[j]]` is a dense
run; after the first split every node reads a scattered subset, and a
measurement this round put the histogram phase at 1M by 50 at **56.7 percent
gather**. Feature-major layout makes a thread's features adjacent and does
nothing for a scattered row.

`set_row_compaction(True)` is CatBoost's answer to that, transplanted:
physically reorder the data at every split so a leaf's rows are contiguous in
*memory* (`catboost/cuda/methods/greedy_subsets_searcher/`
`split_properties_helper.cpp`, `MakeSplit`). Two extra planes hold the binned
matrix and the quantized gradient pair in permutation order, the same
destination map the row scatter computes is applied to them, and the
histogram launch is handed those planes and an identity index. The block
comment above `_compact_build_kernel` states the invariant and proves the
histogram cannot move; `set_row_compaction` states the memory cost.

**It is off by default and it must be argued into the default by a window,
not by this paragraph.** It removes gather traffic and pays a physical
reorder at every split to do it, which is exactly the shape of change this
repository has been burned by: the row-tile floor raised occupancy as
designed and measured 22 percent slower at 50 features. What is claimed here
is arithmetic, not a measurement. Writing `L` for a split's window length and
`nf` for the feature count, a split moves `4 * L * (nf + 8)` bytes -- the
scatter reads and writes the window, the copy-back reads and writes it again
-- and every one of those accesses is contiguous. Against that, an
un-compacted node at depth `d` reads a *sparse* column: its rows are spread
with stride `2^d` over the full `n_rows` extent, so while `2^d` is below a
cache line's worth of bins the node's read costs a full `n_rows * nf` however
few rows it owns, and a level of `2^d` nodes costs that many full passes.
Compaction replaces that with one full pass per level for the moves and one
for the reads. The crossover on paper is around depth two or three; where it
actually is, is what the interleaved A/B is for.

**How it sits beside the two other arms that touch the same bytes**, because
all three arrived in one round and only one pair composes.

- The **Int16 gradient staging** arm composes, and is supported. It narrows
  the staged derivative pair inside the same allocation, so the compacted
  copy simply moves four bytes per row instead of eight; the three compaction
  kernels take the width as `packed_grads` and `compact_packed` records which
  width the planes were written at.
- The **packed bin layout** does not compose and is refused from both
  setters. That arm decodes bin ids out of a bit stream through the *same*
  `bins` pointer the byte gather reads, so a live compaction would hand a
  bit-stream decoder a byte plane and every id it produced would be legal and
  wrong. Permuting a bit stream is a re-encode rather than a move, so there is
  no plane both readers could share.
- The **blocked layout** is refused for the reason it was already refused by
  the packed one: two rearrangements of one matrix, one pointer.

There is one further arm inside this one, `set_compact_flag_read` and
`MOJOTREES_GPU_COMPACT_FLAG_READ`, and it is off by default like everything
else here. It points the *partition's own* flag pass at the compacted plane:
position `j` of the range reads `cbins[f * n_rows + begin + j]` rather than
`bins[f * n_rows + rows[begin + j]]`, which under the invariant is the same
byte and is therefore bit-identical for the same reason the histogram's
compacted launch is. What it removes is the paragraph above's own admission
that the bin load is "the one random-access read of the partition", plus the
permutation load that feeds it. It is a second switch rather than part of the
first because the compaction A/B has to be able to price the physical reorder
against the gather it removes from the *histogram* before it prices it against
two stages at once; an arm that changed both at the same time could not say
which half paid. `compact_flag_read_live` is the predicate and it conjoins
`row_compaction_live`, so this arm is inert whenever the planes are not the
ones the histogram is already reading.

One warning for whoever edits these kernels next, because it cost this lane a
debugging session and it *compiles*. `_scatter_kernel` and its two copies have
carried a local named `packed` since the partition was written -- the packed
routing offset, `(prefix << 1) | flag`. A kernel parameter named `packed` in
that scope is silently shadowed by it, which on the Int32 arm moves the
gradient plane at the Int16 width and produces a wrong histogram over
legal-looking values. That is why the width parameter is spelled
`packed_grads` here and nowhere `packed`.

What is not here
----------------
This module owns no dataset, gradients, or histogram output. Those live in
`GpuHistogramBuilder`, and the range-histogram entry points below take them
as pointers so the builder can call in without a second copy of anything.
The frontier seam this module offers a grower is `begin_tree_with`,
`apply_commit` and `check_frontier`.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, global_idx, grid_dim, thread_idx
from std.math import round
from std.memory import stack_allocation
from std.os import getenv
from std.sys import has_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import broadcast as block_broadcast
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.sync import barrier

from .binning import BinnedMatrix
from .categorical import CatBitset, cat_empty
from .gpu_frontier import CommitPlan, LeafFrontier
from .gpu_binned_layout import BLOCK_ALIGN_BYTES
from .gpu_blocked_bins import (
    BLOCKED_STRIDE_NONE,
    blocked_bytes,
    blocked_is_identity,
    check_blocked_group_matches,
    enqueue_blocked_relayout,
)
from .gpu_packed_bins import (
    PACKED_TABLE_STRIDE,
    PACKED_WIDTH_OFF,
    check_packed_widths,
    enqueue_packed_pack,
    packed_bytes,
    packed_is_identity,
    packed_max_stream_bytes,
    packed_table,
)
from .gpu_histogram_specializations import MAX_BINS
from .gpu_tiling import (
    HIST_FEATURE_GROUP_LADDER,
    HIST_FEATURE_GROUP_MAX,
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    DeviceCaps,
    HistogramTiling,
    derive_block_threads,
    derive_tiling,
    free_feature_group,
    histogram_bin_capacity,
    histogram_shared_bytes,
    is_feature_group_width,
)
from .gpu_split_search import CAT_WORDS, CAT_WORD_BITS
from .parallel import _env_int
from .split import SplitInfo

# Row indices and leaf ids cross into the kernels as Int32.
comptime MAX_ROWS = Int(Int32.MAX)


# --- The step descriptor --------------------------------------------------
#
# One committed growth step, written by the device and read by the device.
#
# Why this table exists at all. Every kernel below takes the split it is
# routing by, and the window it is routing, as *launch arguments*: the host
# knows both because the host chose the split. A device-owned tree
# (`gpu_tree_tables.mojo`) moves that choice onto the device, and then the
# host does not know either one and cannot pass them without reading them
# back first. A blocking readback on an Apple M4 was measured at 606
# microseconds against 3.7 microseconds of actual byte movement
# (`docs/METAL_TIMELINE.md`), and there are 31 of them in a default 31-leaf
# tree, so the readback is the cost and removing it is the point.
#
# The descriptor is therefore the seam. The commit kernel in
# `gpu_tree_tables.mojo` writes it as the last act of a commit, and the
# partition and histogram
# kernels here read the same words instead of taking them as arguments. It
# is a flat Int32 row, in the same style as the split records in
# `gpu_split_search.mojo`, because a flat row is what a kernel can index
# without knowing anything about the layout of the tables it came from.
#
# The layout lives *here*, in the consumer, and not in the producer, purely
# to keep the import graph a tree: `gpu_tree_tables` already imports this
# module's neighbours, and nothing in this file may import
# `gpu_tree_tables` without creating a cycle through
# `histogram_gpu -> gpu_active_rows`.
#
# **Everything is written on every commit and nothing is left over from the
# step before.** A step that commits nothing writes `STEP_LIVE = 0` and
# leaves the rest at whatever it held, so `STEP_LIVE` is the only word a
# reader may consult before it has checked `STEP_LIVE`. Every kernel here
# that takes a descriptor returns immediately when it is zero, which is what
# makes a whole tree's worth of steps safe to enqueue up front: the steps
# past the end of growth run, read one word, and do nothing.

comptime STEP_LIVE = 0
"""1 when the step this descriptor describes committed a split, 0 otherwise.
The guard every descriptor-driven kernel checks first."""

comptime STEP_ROW_BEGIN = 1
"""Start of the split leaf's window into the active-row permutation. The
partition rewrites exactly `[STEP_ROW_BEGIN, STEP_ROW_BEGIN + STEP_ROW_COUNT)`
and nothing else, which is what leaves every other leaf's window intact."""

comptime STEP_ROW_COUNT = 2
"""Rows in that window."""

comptime STEP_FEATURE = 3
comptime STEP_THRESHOLD = 4
comptime STEP_MISSING_BIN = 5
comptime STEP_DEFAULT_LEFT = 6
comptime STEP_IS_CAT = 7
"""The five words of `RowRouting`, in the same meanings: the split feature,
the inclusive threshold bin (-1 on a categorical split), the feature's
missing bin (-1 when it reserves none), the direction missing rows take, and
whether the node routes by category set instead of by threshold."""

comptime STEP_BUILT_BEGIN = 8
comptime STEP_BUILT_COUNT = 9
"""The window of the child whose histogram is *accumulated* rather than
derived, which is the smaller of the two under
`gpu_frontier.subtraction_builds_left`. Written by the commit because it is
`begin`/`begin + n_left` arithmetic the commit has already done, and reading
it here saves the histogram kernel from having to know which child won that
comparison."""

comptime STEP_BUILT_SLOT = 10
"""The resident pool slot the accumulated child's histogram is written
into."""

comptime STEP_SUB_SLOT = 11
"""The pool slot the accumulation is also subtracted from, which is the
parent's slot and therefore the derived sibling's slot after the commit
reassigns it. The device mirror of `enqueue_resident_leaf_subtracting`'s
`parent_slot`."""

comptime STEP_LEFT_SLOT = 12
comptime STEP_RIGHT_SLOT = 13
"""The pool slots the two children's histograms end up in, whichever of them
was built and whichever was derived. What the next search reads."""

comptime STEP_LEFT_REC = 14
comptime STEP_RIGHT_REC = 15
"""The split-record slots the two children's records must end up in, which
are the frontier slots' own record indices. The search itself writes a fixed
pair of scratch records, because a launch's record range is a host-side
constant; a device kernel then copies those two into these two. See
`gpu_tree_tables._copy_records_kernel`."""

comptime STEP_CAT0 = 16
"""First of `CAT_WORDS` category-set words, in the split record's own 16-bit
packing. Copied out of the record without reinterpretation, and unpacked
into the four UInt64 the routing rule wants by `_step_cat_word` below."""

comptime STEP_SPEC_HIT = STEP_CAT0 + CAT_WORDS
"""1 when this descriptor describes a step whose child histogram was already
built speculatively during the previous step, 0 otherwise.

**Written by `_spec_consume_kernel` and by nothing else**, which means it is
meaningful on the *build* descriptor that kernel produces and is stale on
`step_dev`. `_pick_and_commit_kernel` writes every word up to and including
the category set and stops there, so the commit never touches this one; no
kernel reads it off `step_dev`, and the only reader is
`_spec_subtract_kernel`, which is launched against the build descriptor.

It is one word rather than a separate buffer so that the decision and the
slot numbers it applies to travel together: a subtraction that read its hit
flag from one place and its two slots from another could be handed a hit
from step k and slots from step k+1, and would silently corrupt one
histogram."""

comptime STEP_WORDS = STEP_SPEC_HIT + 1


# --- The descriptor a descriptor-aware launch reads -----------------------
#
# There are three of them once the K=1 speculative prebuild is armed, and
# every descriptor-aware launch reads exactly one, chosen by
# `GpuActiveRows.set_descriptor_target`. A field rather than a parameter
# because the choice has to reach `_enqueue_atomic_at`, four call sites deep
# through two dispatch layers whose signatures are already eighteen
# arguments wide, and because the alternative -- passing a pointer down from
# a caller that also holds `mut self` -- is an aliasing the compiler
# correctly refuses.
#
# The default is `DESC_STEP`, which is what every path that predates the
# speculation reads, so an unarmed fit sees byte-for-byte the launches it
# saw before.

comptime DESC_STEP = 0
"""`step_dev`: the commit descriptor `gpu_tree_tables._pick_and_commit_kernel`
writes. The default, and the only target any caller outside
`gpu_resident_round`'s speculative arm ever selects."""

comptime DESC_BUILD = 1
"""`build_dev`: what the *real* step's partition and child histogram read.

A copy of `step_dev` on a speculation miss, and a copy with `STEP_LIVE`
forced to zero on a hit, so that a hit skips both without any kernel having
to learn what a speculation is. That is the whole mechanism by which a
prebuild is consumed rather than merely computed: the work the consuming
step would have done reads a descriptor that says the step is dead, and a
slot-sized subtraction runs in its place."""

comptime DESC_SPEC = 2
"""`spec_dev`: the speculative descriptor `gpu_tree_tables._pick_runner_up_
kernel` publishes, naming the leaf the *next* step is most likely to pick.

A launch against this target also suppresses the sibling subtraction, which
is not a second knob but the same one: the speculation must not derive the
larger child in place from its parent's slot, because the parent is a leaf
that is still live and whose histogram has to survive a miss."""


@always_inline
def _step_cat_word(
    desc: MutPointer[Int32, MutAnyOrigin], w: Int
) -> UInt64:
    """UInt64 word `w` of the descriptor's category set.

    `_row_goes_left` wants the set as four UInt64; a split record carries it
    as sixteen Int32 holding sixteen bits each, and the tree table stores the
    record's words verbatim so that no bit is reinterpreted between the
    search that chose the set and the partition that routes by it. This is
    the one place the two spellings meet, and it is a reassembly rather than
    a reinterpretation: bit `b` of the set is bit `b % CAT_WORD_BITS` of Int32
    word `b // CAT_WORD_BITS`, and bit `b` of UInt64 word `w` is bit `b - 64 *
    w`, so four consecutive Int32 words make one UInt64 word in ascending
    order.

    The mask against 0xFFFF is not decoration. A 16-bit field held in an
    Int32 is sign-extended by the widening conversion whenever bit 15 is set,
    which would set every bit above it in the UInt64 and put categories into
    the set that the search never chose. That is a fault a small fixture
    would miss, because bit 15 of word 0 is category 15 and most category
    sets are sparse and low.
    """
    var out = UInt64(0)
    comptime WORDS_PER_U64 = 64 // CAT_WORD_BITS
    for k in range(WORDS_PER_U64):
        var idx = w * WORDS_PER_U64 + k
        if idx < CAT_WORDS:
            var v = UInt64(
                UInt32(Int32(desc[unsafe_offset = STEP_CAT0 + idx][0]))
            ) & UInt64(0xFFFF)
            out |= v << UInt64(k * CAT_WORD_BITS)
    return out

# The scan kernels keep one Int32 per thread in shared memory, so the block
# size they can be launched with is bounded by this allocation rather than by
# the device maximum. 1024 Int32 is 4 KB, well inside every backend's
# per-threadgroup budget, and 1024 is also the largest block any supported
# device accepts.
comptime SCAN_MAX_THREADS = 1024


def _scan_primitive_width_supported(threads: Int) -> Bool:
    """Whether the primitive scan arm has a kernel instantiated at this
    threadgroup width.

    `block.prefix_sum` takes its block size as a *compile-time* parameter, so
    the primitive kernels cannot read `block_dim.x` the way the hand-rolled
    ones do: each width is a separate instantiation and the host has to pick
    one. Two constraints bound the menu. The primitive itself carries
    `constrained` "Block size must be a multiple of warp size", which
    `gpu_tiling.clamp_block_threads` already satisfies unconditionally by
    rounding every width down to a multiple of `WARP_GRANULARITY` (64, which
    is a multiple of the 32-wide warp on NVIDIA and Metal and is AMD's
    wavefront exactly). And every instantiation is compile time spent, so the
    menu is the four powers of two from 128 up, which is every width
    `derive_block_threads` produces on a supported device: the target is 256
    and the clamp only lowers it on a device whose maximum threadgroup is
    smaller than that, which none of the three backends has.

    A width outside the menu is reachable only through an explicit
    `MOJOTREES_GPU_BLOCK_THREADS` override (192 and 320 are legal widths that
    are not on it). That is not an error and is not silently rounded, because
    rounding the width would change the block count and the block count is
    what `block_sums_dev` was sized for: such a width simply runs the
    hand-rolled arm, which reads its width from `block_dim.x` and serves any
    of them. Both arms produce the same permutation, so this changes nothing
    a caller can observe.
    """
    return threads == 128 or threads == 256 or threads == 512 or (
        threads == 1024
    )


def _partition_grid(n: Int, threads: Int, cap: Int) raises -> Tuple[Int, Int]:
    """The partition's launch geometry: how many threadgroups, and how many
    `threads`-wide tiles each of them walks.

    Before this lane the answer was always "one tile per threadgroup, as many
    threadgroups as it takes", and the per-block left counts that shape
    produced were turned into global offsets by a *separate* launch. The
    scatter now does that scan itself, at its own head, which is what removes
    the launch; the price is that every threadgroup of the scatter re-reads
    the whole block-sums array, so the block count has to be bounded or the
    re-read grows quadratically in the range length. Bounding it is what this
    function is for.

    The rule, given a range of `n` elements:

        tiles_total = ceil(n / threads)      tiles the range needs at all
        blocks      = min(tiles_total, cap)  bounded threadgroup count
        tiles       = ceil(tiles_total / blocks)   tiles each one walks
        blocks      = ceil(tiles_total / tiles)    trim the empty tail

    The second assignment to `blocks` only ever lowers it, and it exists so
    that a range like `cap + 1` tiles does not launch `cap` threadgroups of
    which half own nothing: with two tiles each, `ceil((cap + 1) / 2)` of them
    is enough to cover the range. It cannot raise `blocks` above `cap`,
    because `tiles >= tiles_total / cap` gives `blocks <= cap`.

    Two properties the kernels depend on, both of which hold by construction.
    First, `blocks * tiles * threads >= n`, so every element of the range is
    covered by exactly one (block, tile, lane) triple. Second, the blocks
    partition the range into *contiguous* chunks in ascending order, block `b`
    owning `[b * tiles * threads, (b + 1) * tiles * threads)` clipped to the
    range, which is what keeps the scatter's destination arithmetic a stable
    partition.

    Note what this does *not* change: while `tiles_total <= cap` the geometry
    is identical to what the module launched before this lane, one tile per
    threadgroup. With the default cap (`block_threads`, see
    `GpuActiveRows.partition_block_cap`) that covers every range shorter than
    `threads * threads` elements, which at the usual 256-wide threadgroup is
    every range under 65,536 rows. Only ranges larger than that see the
    multi-tile loop at all, so on a small dataset, and on every node below the
    first level or two of a large one, the flag pass runs exactly the code
    path it ran before.
    """
    if n < 1:
        raise Error("partition grid needs at least one element")
    if threads < 1:
        raise Error("partition grid needs a positive threadgroup width")
    if cap < 1:
        raise Error("partition grid needs a positive block cap")
    var tiles_total = (n + threads - 1) // threads
    var blocks = tiles_total
    if blocks > cap:
        blocks = cap
    var tiles = (tiles_total + blocks - 1) // blocks
    blocks = (tiles_total + tiles - 1) // tiles
    return (blocks, tiles)


# Features one histogram threadgroup may accumulate at once, and the top of
# the ladder the kernel family is instantiated over. One threadgroup's shared
# histogram is three Int32 planes of `GROUP * BIN_CAP` cells, where `BIN_CAP`
# is the dataset's bin count rounded up the capacity ladder
# (`gpu_tiling.histogram_bin_capacity`) rather than fixed at `MAX_BINS`, so a
# group of G on a 64-bin dataset costs `3 * G * 64 * 4` bytes and not
# `3 * G * 256 * 4`. That is what makes widths past the pairing affordable at
# all: group 8 at 64 bins occupies the 6 KiB group 2 used to occupy at every
# bin count. `set_feature_group` still refuses a width that is not a rung,
# 3 among them, because a width with no instantiation cannot launch.
comptime FEATURE_GROUP_MAX = HIST_FEATURE_GROUP_MAX
comptime FEATURE_GROUP_LADDER = HIST_FEATURE_GROUP_LADDER


@always_inline
def _row_goes_left(
    bin: Int32,
    threshold_bin: Int32,
    missing_bin: Int32,
    default_left: Int32,
    is_categorical: Int32,
    cat0: UInt64,
    cat1: UInt64,
    cat2: UInt64,
    cat3: UInt64,
) -> Bool:
    """The routing rule, device side. Identical to `Tree.goes_left` and to
    the existing `_partition_kernel`: set membership for a categorical node,
    the default direction for the missing bin, the inclusive threshold
    otherwise. A feature with no missing bin passes -1, which no bin id can
    equal, so those rows fall through to the threshold."""
    if is_categorical != 0:
        var word: UInt64
        var w = Int(bin) >> 6
        if w == 0:
            word = cat0
        elif w == 1:
            word = cat1
        elif w == 2:
            word = cat2
        else:
            word = cat3
        return ((word >> UInt64(Int(bin) & 63)) & 1) != 0
    if bin == missing_bin:
        return default_left != 0
    return bin <= threshold_bin


def _iota_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
):
    """Seed the active-row buffer with the identity permutation. The unbagged
    root is every row in ascending order, which is what the CPU grower builds
    its root list as."""
    var r = global_idx.x
    if r < Int(n_rows):
        rows[unsafe_offset=r] = Int32(r)


def _zero_int32_kernel(
    buf: MutPointer[Int32, MutAnyOrigin],
    n: Int32,
):
    """Zero an Int32 device range. Used instead of `enqueue_memset` where the
    caller handed in a pointer rather than the buffer."""
    var i = global_idx.x
    if i < Int(n):
        buf[unsafe_offset=i] = Int32(0)


def _zero_slot_desc_kernel(
    pool: MutPointer[Int32, MutAnyOrigin],
    cells: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
):
    """Zero the resident pool slot the step descriptor names, and only that
    one.

    The atomic histogram strategy folds its threadgroup partials into the
    output with `Atomic.fetch_add`, so it accumulates onto whatever the
    destination already held and the destination has to start at zero. On the
    host-driven path `enqueue_range_histogram` zeroes it with
    `_zero_int32_kernel` over a pointer the host has already offset to the
    right slot. Here the slot is a device value, so the offset is computed
    inside the kernel from `STEP_BUILT_SLOT` instead.

    Guarded on `STEP_LIVE` for a reason that is worth spelling out, because
    getting it wrong would be silent and destructive rather than merely
    wrong: on a step that committed nothing the descriptor's slot word is
    whatever it was left at, and zeroing an arbitrary slot would erase a live
    leaf's histogram, which would then be searched and would produce a
    plausible split with no rows behind it. A dead step must touch nothing at
    all, and this is the kernel where that mattered most.

    Grid-strided, so the caller sizes the grid to the pool's slot width once
    and never to a per-step quantity.
    """
    if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
        return
    var base = Int(desc[unsafe_offset=STEP_BUILT_SLOT][0]) * Int(cells)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var i = Int(global_idx.x)
    while i < Int(cells):
        pool[unsafe_offset = base + i] = Int32(0)
        i += stride


def _copy_back_zero_slot_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    pool: MutPointer[Int32, MutAnyOrigin],
    cells: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
):
    """`_copy_back_kernel`'s descriptor arm and `_zero_slot_desc_kernel`, in
    one command buffer.

    Why this exists at all
    ----------------------
    On Metal every `enqueue_function` becomes its own single-encoder command
    buffer and the queue holds 64 of them (`docs/GPU_PORTABILITY.md` 6.2). The
    per-launch enqueue cost is flat at 6 to 7 microseconds through a stream of
    64 and rises to 14 to 17 beyond it, which is a knee exactly where that
    depth predicts. The device-resident growth loop emits on the order of 306
    command buffers between waits, so it sits past the knee for most of every
    tree and pays roughly double per launch. A launch removed from the step
    body is removed thirty times a tree.

    What makes these two the pair that can be folded
    ------------------------------------------------
    They are adjacent in the stream -- the copy-back is the last launch of
    `enqueue_partition_desc` and the slot zeroing is the first of
    `enqueue_desc_histogram`, with nothing between them and no host decision
    anywhere near them -- and they are independent of each other in every way
    that a launch boundary could matter:

    - **Disjoint memory.** The copy-back writes `rows` and reads `scratch`;
      the zeroing writes `pool`. No address is touched by both halves, so
      there is no ordering between them to preserve and no barrier standing
      between them to remove. A fused kernel could run the two loops in either
      order, or interleaved, and produce the same bytes.
    - **The same guard, on the same word of the same descriptor.** Both halves
      return on `STEP_LIVE == 0` and neither writes the descriptor, so this
      folds two reads of one word into one read of it. It does **not** fold
      across a descriptor *write*, which is the thing that would break: a
      write is what makes the next kernel's read meaningful, and there is no
      write anywhere between these two.
    - **Both were already grid-strided**, so neither has a launch geometry
      that reaches its answer, and one grid can serve both. `_copy_back_kernel`
      walks `[begin, begin + count)` of the window the descriptor names;
      `_zero_slot_desc_kernel` walks `cells` of the slot the descriptor names.
      Each thread visits distinct positions of its own range in both arms, and
      an Int32 store is not an accumulation, so nothing about the number of
      blocks or the number of threads changes what is stored.

    What the launch boundary that *stays* is for. The barrier this fusion does
    not touch is the one before it: the scatter's writes to `scratch` land at
    positions derived from a global prefix, so a block of the copy-back may
    read a word some other block of the scatter wrote, and that is a
    device-wide dependency a threadgroup barrier cannot express. Same for the
    barrier after: the atomic accumulation reads `rows` and adds onto `pool`,
    both of which this kernel writes across blocks. So the partition's three
    launches stay three-minus-one and the histogram's two stay two-minus-one,
    and this is the only place in the step where two adjacent launches are
    genuinely unordered with respect to each other.

    **Exactness.** Fusing changes the schedule and not the arithmetic. Every
    store this kernel makes is a store one of the two kernels it replaces made,
    of the same value, to the same address, under the same guard. There is no
    arithmetic here at all beyond index formation, and no floating point
    anywhere, so `docs/NUMERICS.md`'s contraction rule has nothing to act on.

    What moves in the stream, exactly: the copy-back was the launch before the
    zeroing and is now the same launch. Two optional passes may be enqueued
    between this kernel and the accumulation that consumes it -- the gradient
    quantization and the bin re-layout -- and neither reads the active-row
    buffer (`_quantize_grad_hess_kernel` is indexed by row id and not by the
    permutation, which its own docstring records; the re-layout reads the
    binned matrix). So the copy-back is still after every write to `scratch`
    and still before every read of `rows`, which is the whole of what its
    position had to satisfy.

    The two loops are written out rather than sharing one bounds test because
    their extents are different quantities -- a row window and a slot's cell
    count -- and a merged loop would have to carry both bounds anyway.
    """
    if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
        return
    var stride = Int(block_dim.x) * Int(grid_dim.x)

    # Half one: the copy-back. `_copy_back_kernel`'s descriptor arm verbatim,
    # window and all, with `use_desc` folded to the constant it always is here.
    var b = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
    var n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])
    var j = Int(global_idx.x)
    while j < n:
        var i = b + j
        rows[unsafe_offset=i] = scratch[unsafe_offset=i][0]
        j += stride

    # Half two: the slot zeroing. `_zero_slot_desc_kernel` verbatim.
    var base = Int(desc[unsafe_offset=STEP_BUILT_SLOT][0]) * Int(cells)
    var k = Int(global_idx.x)
    while k < Int(cells):
        pool[unsafe_offset = base + k] = Int32(0)
        k += stride


# --- The speculation's two kernels ----------------------------------------
#
# Both are mine to state plainly because the whole correctness of the K=1
# prebuild sits in them: one decides whether the work already done is the
# work this step needs, and the other is what the step does instead of that
# work when it is. The prebuild itself uses no kernel of its own -- it is
# `enqueue_partition_desc` and `enqueue_desc_histogram` launched against a
# different descriptor, which is the point of the descriptor being a table
# in device memory rather than a launch argument.

comptime SPEC_STAT_BUILDS = 0
"""Speculative builds this tree actually issued.

Not `steps - 1`. `_pick_runner_up_kernel` declines to publish on a step that
committed nothing, on a step after which the leaf budget is spent, on a step
whose pre-existing leaves are all inadmissible, and on a step where the slot
pool has nothing free. Each of those is a step the host census counts as a
build because the commit log cannot see the difference; this counter can, so
the two together bound the census's own overcount."""

comptime SPEC_STAT_CONSUMED = 1
"""Speculative builds a later commit actually used.

**Incremented on the consuming branch of `_spec_consume_kernel` and nowhere
else.** That is the whole reason this counter exists rather than a count of
launches: a speculation that never once hit would launch exactly as many
kernels as one that always hits, so a test that counted launches would pass
on a mechanism that does nothing. This increments only where the step's own
descriptor and the previous step's speculative descriptor agree field for
field, which is the same branch that suppresses the real build."""

comptime SPEC_STAT_WORDS = 2


def _spec_consume_kernel(
    step: MutPointer[Int32, MutAnyOrigin],
    spec: MutPointer[Int32, MutAnyOrigin],
    build: MutPointer[Int32, MutAnyOrigin],
    stats: MutPointer[Int32, MutAnyOrigin],
):
    """Decide whether this step's committed split is the one the previous step
    speculatively built, and publish the descriptor the real build will read.

    One thread. It reads two flat rows of `STEP_WORDS` Int32 and writes one,
    which is why it is not worth a grid.

    **The decision is by identity of the work, not by identity of the leaf.**
    The obvious test is "is the node this step split the node the speculation
    guessed", and the theorem in `gpu_resident_round.mojo` says that test is
    equivalent to this one. This kernel does not rely on the theorem. It
    compares every descriptor field the speculative partition and the
    speculative histogram actually consumed --- the window, the routing rule,
    the built child's own window, and the slot it was built into --- and
    consumes only when all of them are equal. So a hit means, literally, that
    the launches already run were handed the same arguments this step's
    launches would be handed.

    That direction matters and it is deliberate. A comparison that was too
    strict costs a hit and nothing else: the real build runs, as it always
    did. A comparison that was too loose would hand a step a histogram of
    some other leaf's rows, which is a wrong tree that nothing downstream
    could detect. Every extra field in the conjunction below moves the error
    into the direction that is merely slower.

    `build` is a full copy of `step` with one word changed on a hit. Copying
    rather than aliasing is what lets `STEP_LIVE` be forced to zero for the
    partition and the histogram while `gpu_tree_tables._stage_child_search_
    kernel` and `_copy_records_kernel` -- which must still run, because a
    consumed step still searches its two children and still files their
    records -- keep reading the live commit descriptor.

    A dead step (`step[STEP_LIVE] == 0`) copies through as dead and consumes
    nothing, which is right twice over: there is no commit to serve, and the
    speculative descriptor from the step before it is the one thing on this
    path that could be stale.
    """
    if Int(global_idx.x) != 0:
        return
    for w in range(STEP_WORDS):
        build[unsafe_offset=w] = step[unsafe_offset=w][0]
    build[unsafe_offset=STEP_SPEC_HIT] = Int32(0)
    if step[unsafe_offset=STEP_LIVE][0] == Int32(0):
        return
    if spec[unsafe_offset=STEP_LIVE][0] == Int32(0):
        return
    # The window this step is about to partition, the rule it will route by,
    # the child window it will accumulate, and the slot it will accumulate
    # into. Every one of them is an input the speculative launches already
    # consumed, so equality here is equality of the work and not a guess
    # about it.
    var same = True
    if step[unsafe_offset=STEP_ROW_BEGIN][0] != spec[
        unsafe_offset=STEP_ROW_BEGIN
    ][0]:
        same = False
    if step[unsafe_offset=STEP_ROW_COUNT][0] != spec[
        unsafe_offset=STEP_ROW_COUNT
    ][0]:
        same = False
    if step[unsafe_offset=STEP_FEATURE][0] != spec[unsafe_offset=STEP_FEATURE][
        0
    ]:
        same = False
    if step[unsafe_offset=STEP_THRESHOLD][0] != spec[
        unsafe_offset=STEP_THRESHOLD
    ][0]:
        same = False
    if step[unsafe_offset=STEP_MISSING_BIN][0] != spec[
        unsafe_offset=STEP_MISSING_BIN
    ][0]:
        same = False
    if step[unsafe_offset=STEP_DEFAULT_LEFT][0] != spec[
        unsafe_offset=STEP_DEFAULT_LEFT
    ][0]:
        same = False
    if step[unsafe_offset=STEP_IS_CAT][0] != spec[unsafe_offset=STEP_IS_CAT][
        0
    ]:
        same = False
    if step[unsafe_offset=STEP_BUILT_BEGIN][0] != spec[
        unsafe_offset=STEP_BUILT_BEGIN
    ][0]:
        same = False
    if step[unsafe_offset=STEP_BUILT_COUNT][0] != spec[
        unsafe_offset=STEP_BUILT_COUNT
    ][0]:
        same = False
    if step[unsafe_offset=STEP_BUILT_SLOT][0] != spec[
        unsafe_offset=STEP_BUILT_SLOT
    ][0]:
        same = False
    if step[unsafe_offset=STEP_IS_CAT][0] != Int32(0):
        for w in range(CAT_WORDS):
            if step[unsafe_offset = STEP_CAT0 + w][0] != spec[
                unsafe_offset = STEP_CAT0 + w
            ][0]:
                same = False
    if not same:
        return
    # The consuming branch, and the only place `SPEC_STAT_CONSUMED` moves.
    build[unsafe_offset=STEP_LIVE] = Int32(0)
    build[unsafe_offset=STEP_SPEC_HIT] = Int32(1)
    stats[unsafe_offset=SPEC_STAT_CONSUMED] = (
        stats[unsafe_offset=SPEC_STAT_CONSUMED][0] + Int32(1)
    )


def _spec_subtract_kernel(
    pool: MutPointer[Int32, MutAnyOrigin],
    cells: Int32,
    build: MutPointer[Int32, MutAnyOrigin],
):
    """On a consumed step, derive the sibling the skipped build would have
    derived: `pool[sub_slot] -= pool[built_slot]`, cell for cell.

    The one piece of a consuming step's work that the prebuild could not do
    ahead of time, and the reason it could not is the design correction this
    whole lane turns on. `_range_hist_atomic_kernel` folds the sibling
    subtraction into the accumulation, subtracting the built child out of the
    parent's slot as it goes. A speculative build must not: the parent is a
    leaf that is still live, and on a miss its histogram is the one the next
    pick reads. So the prebuild runs with `do_sub` off and leaves a whole,
    unsubtracted child histogram in the slot, and this kernel does the
    subtraction later, once the commit has proved the parent is a parent.

    **Exact, and for the same reason the fused form is.** Both operands are
    fixed-point Int32 under one round's scales, so the difference is integer
    arithmetic with no rounding anywhere. The fused form touches only cells
    the accumulation wrote and this one touches every cell of the slot; the
    difference is cells where the built child's value is zero, and
    subtracting zero is the identity. `_zero_slot_desc_kernel` is what makes
    that true of the inactive feature slices as well: the speculative build
    zeroes the whole slot before it accumulates, exactly as the real one
    does.

    Grid-strided over a slot's cells, which is a per-fit constant, so the
    caller sizes the grid once.

    Guarded on `STEP_SPEC_HIT` rather than on `STEP_LIVE`, and the two are
    opposite here: the build descriptor says *dead* on precisely the steps
    this kernel must act on. Reading the wrong one of those two words would
    subtract on every miss, which would corrupt the parent's histogram on
    two thirds of all steps.
    """
    if build[unsafe_offset=STEP_SPEC_HIT][0] != Int32(1):
        return
    var n = Int(cells)
    var built = Int(build[unsafe_offset=STEP_BUILT_SLOT][0]) * n
    var sub = Int(build[unsafe_offset=STEP_SUB_SLOT][0]) * n
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var i = Int(global_idx.x)
    while i < n:
        pool[unsafe_offset = sub + i] = (
            pool[unsafe_offset = sub + i][0]
            - pool[unsafe_offset = built + i][0]
        )
        i += stride


def _flag_scan_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    begin: Int32,
    count: Int32,
    feature: Int32,
    threshold_bin: Int32,
    missing_bin: Int32,
    default_left: Int32,
    is_categorical: Int32,
    cat0: UInt64,
    cat1: UInt64,
    cat2: UInt64,
    cat3: UInt64,
    tiles: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
    compacted: Int32,
):
    """Stage one of the stable partition: flag each row of the range and scan
    the flags within each threadgroup.

    Block `b` owns the contiguous chunk of `tiles * block_dim.x` elements
    starting at `b * tiles * block_dim.x`, clipped to the range, and walks it
    one `block_dim.x`-wide tile at a time. `offsets[j]` becomes the number of
    left-going rows before `j` *inside j's own chunk*, shifted left one bit,
    with j's own flag in the low bit; `block_sums[b]` is the number of
    left-going rows in the whole chunk. The running count across the chunk's
    tiles is carried in a register that every thread of the block computes
    identically, exactly the way the block-sums scan this lane deleted carried
    its own running total.

    Before this lane `tiles` was always 1 and the chunk was always one tile,
    so this loop had no iterations to run and the code below reduces to what
    was here before whenever `_partition_grid` returns `tiles == 1`, which it
    does for every range shorter than `block_threads * block_threads`
    elements. What changed is only that a longer range is now covered by a
    bounded number of threadgroups walking several tiles each rather than by
    an unbounded number walking one, which is what lets the scatter scan the
    block sums itself instead of a separate launch doing it. The permutation
    cannot notice: a row's destination is a function of the routing flags and
    the row's position in the range, and neither depends on how the range was
    cut into blocks. See `_scatter_kernel` for that argument written out.

    Carrying the flag in the offset word is what lets the scatter route the
    row without gathering its bin a second time: the bin loads are the one
    random-access read of the partition (`bins[feature * n_rows + row]`, one
    byte from a fresh cache line per row once the leaf's rows are sparse in
    the dataset), so reading each once instead of twice halves the
    partition's gathered traffic, and the packed word costs nothing the plain
    prefix did not. The prefix is bounded by the chunk length, which is
    bounded by the range length, which is bounded by `n_rows` and therefore
    by `Int32.MAX / 2` for any dataset this module accepts, so the shift
    cannot overflow. Threads past the end of the range contribute a zero
    flag, so a partial tail tile still sums correctly.

    The scan is the plain Hillis-Steele shared-memory scan: two barriers per
    doubling step, with the read separated from the write so no thread reads a
    slot another has already advanced. This is the *fallback* arm.
    `_flag_scan_prim_kernel` is the default and scans with `block.prefix_sum`
    instead; see the module docstring for which rule that repeals and why.
    This kernel is kept rather than deleted because it reads its width from
    `block_dim.x` and so serves any threadgroup width, including the ones no
    primitive instantiation exists for, and because a benchmark wanting both
    arms in one process needs both to be present. It produces the identical
    offsets and block sums.

    The descriptor arm
    ------------------
    With `use_desc`, the window and the routing rule are read out of `desc`
    (see "The step descriptor" at the head of this module) instead of out of
    the launch arguments, and the launch arguments for those eight values are
    ignored. That is the whole of what a device-owned tree needs from this
    kernel: the split it routes by was chosen by a kernel rather than by the
    host, so it lives in device memory rather than in a register the host
    filled.

    Nothing else changes, and in particular the *geometry* does not. A
    descriptor-driven launch cannot size its grid to a range length it does
    not know, so it is launched at a grid that covers the whole active
    prefix, which is an upper bound on any window. That is safe because the
    permutation this partition computes is a function of the routing flags
    and of each row's position in the range and of nothing else; the argument
    is written out in `_scatter_kernel`, which is where the destination
    arithmetic lives. Cutting a short range into more chunks than it needs
    therefore produces the same permutation as cutting it into few. What an
    over-provisioned grid costs is threadgroups that own nothing, and the
    early return below is what keeps that cost to a launch slot.

    Whether the extra threadgroups are cheap enough to pay for the readback
    they remove is UNMEASURED: no benchmark of this arm has been run.

    The compacted read
    ------------------
    With `compacted`, `bins` is the compacted plane rather than the dataset's
    and the bin of range position `j` is at `feature * n_rows + row_begin + j`
    instead of at `feature * n_rows + rows[row_begin + j]`. The two are the
    same byte, by the invariant `_compact_build_kernel` establishes and
    `_compact_scatter_kernel` maintains, so every flag, every prefix, every
    packed offset and every block sum below is unchanged value for value.
    What changes is that consecutive threads read consecutive bytes and the
    permutation is not read at all. The host decides it in
    `GpuActiveRows.compact_flag_read_live`, which will not set it unless the
    compacted planes are live, and this kernel never checks: a launch that
    passed `compacted` against the dataset's own `bins` would route by the
    bins of the wrong rows and nothing here could tell.
    """
    var tid = thread_idx.x
    var nthreads = block_dim.x
    var n = Int(count)
    var row_begin = Int(begin)
    var feat = feature
    var thr = threshold_bin
    var miss = missing_bin
    var dleft = default_left
    var iscat = is_categorical
    var c0 = cat0
    var c1 = cat1
    var c2 = cat2
    var c3 = cat3
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        row_begin = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])
        feat = desc[unsafe_offset=STEP_FEATURE][0]
        thr = desc[unsafe_offset=STEP_THRESHOLD][0]
        miss = desc[unsafe_offset=STEP_MISSING_BIN][0]
        dleft = desc[unsafe_offset=STEP_DEFAULT_LEFT][0]
        iscat = desc[unsafe_offset=STEP_IS_CAT][0]
        c0 = _step_cat_word(desc, 0)
        c1 = _step_cat_word(desc, 1)
        c2 = _step_cat_word(desc, 2)
        c3 = _step_cat_word(desc, 3)
    var col = Int(feat) * Int(n_rows)
    var chunk = Int(tiles) * nthreads
    var base = Int(block_idx.x) * chunk

    # A block whose whole chunk lies past the end of the range. `base`, `n`
    # and `tiles` are block-uniform, so the entire threadgroup returns
    # together and no thread is left waiting at a barrier the others skipped.
    # The zero it writes is exactly what the tile loop below would have
    # accumulated, so this is an early exit and not a behavior change; the
    # host-argument arm reaches it only on a grid that was already
    # over-provisioned, which `_partition_grid` does not produce.
    if base >= n:
        if tid == 0:
            block_sums[unsafe_offset = Int(block_idx.x)] = Int32(0)
        return

    var s = stack_allocation[
        SCAN_MAX_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    # Block-uniform: one test here rather than one per row.
    var dense = Int(compacted) != 0

    var carry = Int32(0)
    for t in range(Int(tiles)):
        var j = base + t * nthreads + tid

        var flag = Int32(0)
        if j < n:
            var at = col + row_begin + Int(j)
            if not dense:
                at = col + Int(rows[unsafe_offset = row_begin + j][0])
            var bin = Int32(bins[unsafe_offset=at])
            if _row_goes_left(
                bin, thr, miss, dleft, iscat, c0, c1, c2, c3
            ):
                flag = Int32(1)
        s[unsafe_offset=tid] = flag
        barrier()

        # Every thread runs the same number of steps (the bound is the
        # uniform block size), so the barriers are reached by the whole
        # threadgroup. `tiles` is uniform too, so the tile loop itself
        # cannot leave a thread behind at a barrier.
        var offset = 1
        while offset < nthreads:
            var carried = Int32(0)
            if tid >= offset:
                carried = s[unsafe_offset = tid - offset][0]
            barrier()
            if tid >= offset:
                s[unsafe_offset=tid] = s[unsafe_offset=tid][0] + carried
            barrier()
            offset += offset

        if j < n:
            # Inclusive minus own flag is the exclusive prefix within the
            # tile; plus the chunk's running count it is the exclusive
            # prefix within the chunk. The flag rides in the low bit for
            # the scatter.
            offsets[unsafe_offset=j] = (
                (carry + s[unsafe_offset=tid][0] - flag) * Int32(2)
            ) + flag
        carry += s[unsafe_offset = nthreads - 1][0]
        # The tile total has been read by every thread; the next iteration
        # may now overwrite the shared buffer.
        barrier()

    if tid == 0:
        block_sums[unsafe_offset = Int(block_idx.x)] = carry


def _flag_scan_prim_kernel[block_size: Int](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    begin: Int32,
    count: Int32,
    feature: Int32,
    threshold_bin: Int32,
    missing_bin: Int32,
    default_left: Int32,
    is_categorical: Int32,
    cat0: UInt64,
    cat1: UInt64,
    cat2: UInt64,
    cat3: UInt64,
    tiles: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
    compacted: Int32,
):
    """Stage one, scanned with `block.prefix_sum` instead of by hand.

    Everything a later stage reads is byte for byte what `_flag_scan_kernel`
    writes: `offsets[j]` is still `(exclusive_prefix_within_j's_chunk << 1) |
    flag`, and `block_sums[b]` is still block b's left count over the same
    contiguous chunk, so the two arms remain interchangeable stage by stage.
    The difference is only how the intra-tile scan is computed.
    `block.prefix_sum` is warp shuffles plus one shared round rather than a
    doubling loop with two barriers per step, and it sums exact Int32 flags,
    so the permutation cannot differ; the test file asserts that against the
    hand-rolled arm rather than asserting it here in a comment.

    Signature, checked against the installed toolchain rather than
    remembered: `block.prefix_sum[dtype: DType, //, *, block_size: Int,
    exclusive: Bool = False](val: Scalar[dtype]) -> Scalar[dtype]`. The
    inclusive form is asked for because both things this kernel needs come
    out of it: the exclusive prefix is the inclusive one minus the thread's
    own flag, exactly as the hand-rolled arm derives it, and the last
    thread's inclusive value is the block total. Asking for the exclusive
    form instead would need a second collective to recover the total.

    `block_size` is a compile-time parameter of the primitive, not a
    `block_dim.x` read, which is why this kernel is parametric and why the
    launch has to pass the matching `block_dim`; see
    `_scan_primitive_width_supported` for the menu of widths and why a width
    outside it falls back rather than being rounded. The primitive is a
    collective, so every thread of the threadgroup reaches it: the flag is
    computed as zero for threads past the end of the range and the call sits
    outside the `j < n` guard, which is also what keeps a partial tail tile
    summing correctly.

    The tile loop and the register-carried running count are the same
    construction `_flag_scan_kernel` uses and the same one the deleted
    block-sums kernel used to carry its total across chunks: the chunk total
    reaches every thread through `block.broadcast[block_size=block_size](
    inclusive, src_thread=block_size - 1)` rather than through a shared slot
    every thread re-reads. Signature, checked against the toolchain:
    `block.broadcast[dtype: DType, width: SIMDLength, //, *, block_size:
    Int](val: SIMD[dtype, width], src_thread: Int = 0) -> SIMD[dtype,
    width]`. Broadcasting from the last thread is what keeps `carry`
    bit-identical on every thread of the block, which is what makes the
    tile-to-tile carry deterministic; summing exact Int32 in a fixed tile
    order, it cannot depend on scheduling.

    The trailing `barrier()` is deliberate and is not the primitive's job to
    supply: the next iteration calls the same collectives again, and this
    kernel does not get to assume anything about whether a primitive fences
    its own scratch on exit. `tiles` is a uniform argument, so every thread
    of the block runs the same number of iterations and reaches every
    collective.

    `desc` and `use_desc` are the descriptor arm, identical in meaning to
    `_flag_scan_kernel`'s: read that kernel's docstring for what the arm is
    for and why an over-provisioned grid computes the same permutation. The
    early return matters more here than there, because this arm's collectives
    are called once per tile whether or not the tile owns any element, so an
    empty block that ran the loop would pay `tiles` prefix sums and `tiles`
    broadcasts to sum a column of zeros.

    `compacted` is the compacted read and it means here exactly what it means
    in `_flag_scan_kernel`: read that kernel's docstring for the invariant it
    rests on and for why the flags it produces are the same flags. The two
    arms have to agree byte for byte on `offsets` and `block_sums` whichever
    way the bin was addressed, and they do, because the addressing changes
    which pointer holds the byte and not which byte it is.
    """
    var tid = thread_idx.x
    var n = Int(count)
    var row_begin = Int(begin)
    var feat = feature
    var thr = threshold_bin
    var miss = missing_bin
    var dleft = default_left
    var iscat = is_categorical
    var c0 = cat0
    var c1 = cat1
    var c2 = cat2
    var c3 = cat3
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        row_begin = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])
        feat = desc[unsafe_offset=STEP_FEATURE][0]
        thr = desc[unsafe_offset=STEP_THRESHOLD][0]
        miss = desc[unsafe_offset=STEP_MISSING_BIN][0]
        dleft = desc[unsafe_offset=STEP_DEFAULT_LEFT][0]
        iscat = desc[unsafe_offset=STEP_IS_CAT][0]
        c0 = _step_cat_word(desc, 0)
        c1 = _step_cat_word(desc, 1)
        c2 = _step_cat_word(desc, 2)
        c3 = _step_cat_word(desc, 3)
    var col = Int(feat) * Int(n_rows)
    var chunk = Int(tiles) * block_size
    var base = Int(block_idx.x) * chunk

    # Block-uniform, so the whole threadgroup leaves together and no
    # collective is reached by part of a block. See `_flag_scan_kernel`.
    if base >= n:
        if tid == 0:
            block_sums[unsafe_offset = Int(block_idx.x)] = Int32(0)
        return

    # Block-uniform: one test here rather than one per row.
    var dense = Int(compacted) != 0

    var carry = Int32(0)
    for t in range(Int(tiles)):
        var j = base + t * block_size + tid

        var flag = Int32(0)
        if j < n:
            var at = col + row_begin + Int(j)
            if not dense:
                at = col + Int(rows[unsafe_offset = row_begin + j][0])
            var bin = Int32(bins[unsafe_offset=at])
            if _row_goes_left(
                bin, thr, miss, dleft, iscat, c0, c1, c2, c3
            ):
                flag = Int32(1)

        var inclusive = block_prefix_sum[
            block_size=block_size, exclusive=False
        ](flag)
        var tile_total = block_broadcast[block_size=block_size](
            inclusive, src_thread = block_size - 1
        )

        if j < n:
            offsets[unsafe_offset=j] = (
                (carry + inclusive - flag) * Int32(2)
            ) + flag
        carry += tile_total
        barrier()

    if tid == 0:
        block_sums[unsafe_offset = Int(block_idx.x)] = carry


def _scatter_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    total: MutPointer[Int32, MutAnyOrigin],
    begin: Int32,
    count: Int32,
    tiles: Int32,
    n_blocks: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
):
    """Stage two: scan the block sums at the head of the block, then write
    each row of this block's chunk to its compacted slot.

    Before this lane the block-sums scan was a launch of its own, one
    threadgroup that turned the per-block left counts into per-block starting
    offsets in place and wrote the range's total. It is now the first thing
    every scatter block does, over the same block sums, with the same chunked
    single-threadgroup scan the deleted kernel used; the block sums are read
    and not modified, so every block computing the scan redundantly is safe
    as well as necessary. That is the launch this lane removes: four launches
    per partition become three.

    What the scan has to yield here is two numbers, and only two. `mine` is
    the number of left-going rows in every chunk before this block's, which is
    the exclusive prefix of `block_sums` at `block_idx.x`; `grand` is the
    range's total left count. Both are block-uniform, so the loop that
    produces them broadcasts rather than leaving them in one lane. The
    exclusive prefix at the block's own index is recovered as the inclusive
    prefix there minus that index's own block sum, which any thread can read
    straight out of `block_sums` because nothing writes it.

    The cost of doing it here rather than in a launch of its own is that each
    of the `n_blocks` blocks re-reads all `n_blocks` block sums, so the extra
    traffic is quadratic in the block count. That is why the block count is
    capped (`_partition_grid`, `GpuActiveRows.partition_block_cap`): with the
    default cap of one threadgroup width, `n_blocks <= threads` and the worst
    case, a range of exactly `threads * threads` elements, re-reads
    `threads * threads` Int32 against `3 * threads * threads` Int32 of useful
    streamed traffic, so it is bounded by a third of the row work and is far
    below that for any longer range. The block sums are a few kilobytes and
    are read by every block at once, so they are cache-resident; none of this
    has been measured, and the bound is an argument about counts, not a
    timing.

    Only the blocks that own an element pay it. A block whose chunk lies
    wholly past the end of the range returns before the scan, which is what
    keeps `enqueue_partition_desc`'s deliberately over-provisioned grid from
    charging a full prefix sum per idle threadgroup; the exemption and the
    reason block 0 is excluded from it are stated at the return itself.

    The scatter itself. A left-going row at local index `j` lands at its
    global left rank `p`; a right-going one lands after every left-going row,
    at `grand + (j - p)`, where `j - p` is its rank among the right-going
    rows. Both ranks are monotone in `j`, so both sides keep their relative
    order and the partition is stable. `p` is `mine` plus the exclusive prefix
    within the chunk that stage one packed into `offsets[j]`, which is exactly
    the number of left-going rows at positions before `j` in the range. Note
    that `p` is therefore a function of the routing flags and of `j` alone: it
    does not depend on how the range was cut into chunks, on how many chunks
    there were, or on how many tiles each walked. That is the whole argument
    that this lane leaves the permutation element for element unchanged.

    The direction comes out of the low bit of the packed offset stage one
    wrote, so this stage touches no bin and needs no routing rule: it is a
    pure permutation of the range from three streamed Int32 reads per row.

    Launched with the same width and the same `tiles` stage one used, because
    `block_idx.x` and `tiles` are what select this element's chunk, and the
    packed offsets are chunk-relative.

    This is the fallback arm; `_scatter_prim_kernel` is the default and
    differs only in scanning with `block.prefix_sum` and `block.broadcast`
    instead of in shared memory by hand.

    `desc` and `use_desc` are the descriptor arm: only the window moves,
    since this stage reads the routing decision out of the packed offsets and
    never touches a bin. A dead step leaves the permutation untouched, which
    is what makes a tree's worth of steps safe to enqueue before growth has
    decided how many of them there will be.
    """
    var tid = thread_idx.x
    var nthreads = block_dim.x
    var nb = Int(n_blocks)
    var me = Int(block_idx.x)
    var n = Int(count)
    var b = Int(begin)
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        b = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])

    # A block whose whole chunk lies past the end of the range writes nothing
    # and so does not need the head scan. Block-uniform, so the whole
    # threadgroup leaves together and no barrier below is reached by part of a
    # block, which is the same early exit `_flag_scan_kernel` already takes.
    # Block 0 is exempt because it stores `total` and `carry` is only correct
    # after the scan; while `n > 0` its chunk starts at zero and it never
    # qualifies anyway. Nothing written changes: the returning blocks stored
    # nothing, and the head scan only reads `block_sums`. See
    # `_scatter_prim_kernel` for why the descriptor grid makes this worth
    # having.
    if me != 0 and me * Int(tiles) * Int(nthreads) >= n:
        return

    var s = stack_allocation[
        SCAN_MAX_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var carry = Int32(0)
    var mine = Int32(0)
    var base = 0
    while base < nb:
        var idx = base + tid
        var own = Int32(0)
        if idx < nb:
            own = block_sums[unsafe_offset=idx][0]
        s[unsafe_offset=tid] = own
        barrier()

        var offset = 1
        while offset < nthreads:
            var carried = Int32(0)
            if tid >= offset:
                carried = s[unsafe_offset = tid - offset][0]
            barrier()
            if tid >= offset:
                s[unsafe_offset=tid] = s[unsafe_offset=tid][0] + carried
            barrier()
            offset += offset

        # `me` is block-uniform, so this branch is taken by the whole
        # threadgroup or by none of it and no barrier is skipped.
        if me >= base and me < base + nthreads:
            mine = (
                carry
                + s[unsafe_offset = me - base][0]
                - block_sums[unsafe_offset=me][0]
            )
        carry += s[unsafe_offset = nthreads - 1][0]
        # The chunk total has been read by every thread; the next iteration
        # may now overwrite the shared buffer.
        barrier()
        base += nthreads

    if me == 0 and tid == 0:
        total[unsafe_offset=0] = carry

    var chunk = Int(tiles) * nthreads
    var first = me * chunk
    for t in range(Int(tiles)):
        var j = first + t * nthreads + tid
        if j < n:
            var row = rows[unsafe_offset = b + j][0]
            var packed = offsets[unsafe_offset=j][0]
            var p = Int(mine) + (Int(packed) >> 1)
            var dst: Int
            if (packed & Int32(1)) != 0:
                dst = p
            else:
                dst = Int(carry) + (j - p)
            scratch[unsafe_offset = b + dst] = row


def _scatter_prim_kernel[block_size: Int](
    rows: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    total: MutPointer[Int32, MutAnyOrigin],
    begin: Int32,
    count: Int32,
    tiles: Int32,
    n_blocks: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
):
    """Stage two on the primitive arm: the same head scan and the same
    scatter, with `block.prefix_sum` and `block.broadcast` in place of the
    hand-rolled shared-memory scan.

    Byte for byte the same writes as `_scatter_kernel` makes, for the same
    reason the two stage-one arms agree: the scan sums exact Int32 over a
    fixed order, so the prefix it produces is a value and not an
    approximation, and every destination index is computed from that prefix
    by the same integer arithmetic. `tests/test_gpu_partition_launches.mojo`
    asserts that against the hand-rolled arm element for element rather than
    leaving it as a comment here.

    Two broadcasts per chunk rather than one. `block_size - 1` gives the
    chunk total, as on stage one; `me - base` gives the inclusive prefix at
    this block's own index, from which its own block sum is subtracted to
    make the exclusive prefix. `src_thread` is a runtime argument of
    `block.broadcast`, and `me - base` is block-uniform inside the guarded
    branch, so the collective is reached by the whole threadgroup with the
    same source lane.

    `desc` and `use_desc` are the descriptor arm, as in `_scatter_kernel`:
    only the window moves, and a dead step returns before any collective, so
    the whole threadgroup leaves together.
    """
    var tid = thread_idx.x
    var nb = Int(n_blocks)
    var me = Int(block_idx.x)
    var n = Int(count)
    var b = Int(begin)
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        b = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])

    # A block whose whole chunk lies past the end of the range writes nothing,
    # so it does not need the head scan either. `me`, `tiles` and `n` are all
    # block-uniform, so the entire threadgroup leaves together and no
    # collective below is reached by part of a block -- which matters more on
    # this arm than on the hand-rolled one, because these are collectives and
    # not bare barriers. Block 0 is exempt unconditionally: it is the block
    # that stores `total`, and `carry` is only correct after the scan. It is
    # never the block that skips anyway while `n > 0`, since its chunk starts
    # at zero.
    #
    # This cannot change a written value. The blocks it returns early store
    # nothing in either branch, and the head scan reads `block_sums` without
    # modifying it, so a block that does not compute the scan removes no input
    # from any block that does. It exists because
    # `enqueue_partition_desc` launches a grid sized to the whole active
    # prefix, so a short window leaves most of the grid owning nothing, and
    # before this every one of those blocks ran a full prefix sum over the
    # block sums to discover it.
    if me != 0 and me * Int(tiles) * block_size >= n:
        return

    var carry = Int32(0)
    var mine = Int32(0)
    var base = 0
    while base < nb:
        var idx = base + tid
        var own = Int32(0)
        if idx < nb:
            own = block_sums[unsafe_offset=idx][0]

        var inclusive = block_prefix_sum[
            block_size=block_size, exclusive=False
        ](own)
        var chunk_total = block_broadcast[block_size=block_size](
            inclusive, src_thread = block_size - 1
        )
        if me >= base and me < base + block_size:
            var at_me = block_broadcast[block_size=block_size](
                inclusive - own, src_thread = me - base
            )
            mine = carry + at_me
        carry += chunk_total
        barrier()
        base += block_size

    if me == 0 and tid == 0:
        total[unsafe_offset=0] = carry

    var chunk = Int(tiles) * block_size
    var first = me * chunk
    for t in range(Int(tiles)):
        var j = first + t * block_size + tid
        if j < n:
            var row = rows[unsafe_offset = b + j][0]
            var packed = offsets[unsafe_offset=j][0]
            var p = Int(mine) + (Int(packed) >> 1)
            var dst: Int
            if (packed & Int32(1)) != 0:
                dst = p
            else:
                dst = Int(carry) + (j - p)
            scratch[unsafe_offset = b + dst] = row


def _copy_back_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    begin: Int32,
    count: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
):
    """Stage three: fold the compacted range back over the parent's slots.

    Only the
    parent's own range is touched, so every other leaf's range survives the
    split untouched; that is why this is a copy and not a buffer swap.

    Launched over its own grid, `ceil(count / threads)` blocks of one tile
    each, and not over the capped grid the flag pass and the scatter share.
    It has no cross-block dependency of any kind, so there is nothing for a
    bounded block count to buy it, and giving it the uncapped grid keeps its
    shape exactly what it was before this lane.

    The loop is grid-strided rather than a single guarded store. On the
    host-argument arm that is the same thing: the grid is `ceil(count /
    threads)` blocks, so the stride is at least `count` and every thread runs
    exactly one iteration, which is the store this kernel has always made.
    What it buys is the descriptor arm, which cannot size a grid to a count
    it does not know and would otherwise have to launch
    `ceil(n_active / threads)` blocks to cover the largest window a step
    might pick. With a stride it can launch the capped grid the other two
    stages use and let each thread walk. No element is visited twice either
    way, because the destinations are distinct positions of one range.

    `desc` and `use_desc` are the descriptor arm; see `_flag_scan_kernel`.

    Why this launch is still here
    -----------------------------
    Removing it is a real saving and was scoped for this lane: it is one of
    three launches, and it moves 8 bytes per row of the range (a 4-byte read
    from `scratch` and a 4-byte write to `rows`) that exist only to undo the
    fact that the scatter cannot write in place. The design that removes it
    is per-range ping-pong: `scratch` becomes a co-equal second row buffer,
    each `LeafRange` carries a bit saying which of the two currently holds
    its rows, the scatter reads from the parent's live buffer and writes to
    the other one, and the two children are recorded as living in the buffer
    the scatter just wrote. No copy, and the ranges other leaves hold stay
    valid because nothing outside the parent's window is touched either way.

    It is not done here because the bit has to be consulted by *every* reader
    of the permutation, and the readers are not all in this file. The audit,
    at the commit that added this note:

    - `enqueue_range_histogram` in this module, which hands the row pointer
      to `_range_hist_atomic_kernel` and `_range_hist_partial_kernel`. It
      knows the node, so it could pick the pointer on the host with no kernel
      change at all, which is the easy half.
    - `GpuHistogramBuilder.readback_range` (histogram_gpu.mojo), which builds
      a sub-buffer view of `rows_dev[begin : begin + count]` and hands it to a
      host histogram build. Same fix, different file.
    - `GpuHistogramBuilder.snapshot_rows` (histogram_gpu.mojo), which copies
      the *whole* permutation in one transfer. This one is not a pointer
      swap: under per-range ping-pong there is no single buffer that holds
      the whole permutation, so a snapshot becomes a gather across both
      buffers driven by the range table, or a merge pass, and the host
      replica path that consumes it is exactly the path whose bit-for-bit
      agreement with the device is load-bearing.
    - `download_rows` and `download_range` in this module, which have the
      same problem as `snapshot_rows` for the same reason.
    - the two launches in `gpu_objectives_native.mojo` that take
      `rows.rows_dev.unsafe_ptr()` over the whole active prefix, which have
      it again.
    - `_quantize_grad_hess_kernel` here, which reads no permutation at all
      (it is indexed by row id) and is therefore *not* affected; recorded so
      that a later reader does not have to re-derive that.

    Three of those live in files this lane may not edit and one of them,
    `snapshot_rows`, is not a one-line change anywhere. A half-applied
    ping-pong leaves one reader looking at the stale buffer for one range,
    which produces a wrong histogram for one node on some tree shapes and
    not others, which is precisely the kind of fault no fixture catches. So
    the copy stays, and the design is written down here rather than
    half-built.
    """
    var n = Int(count)
    var b = Int(begin)
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        b = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var j = Int(global_idx.x)
    while j < n:
        var i = b + j
        rows[unsafe_offset=i] = scratch[unsafe_offset=i][0]
        j += stride


# --- Row compaction ---------------------------------------------------------
#
# CatBoost's GPU grower physically reorders the *data* at every split so that
# a leaf's rows are contiguous in memory rather than named by a scattered
# index (`catboost/cuda/methods/greedy_subsets_searcher/`
# `split_properties_helper.cpp`, `MakeSplit`: a segmented reorder plus part
# offsets). These three kernels are that mechanism, and nothing else in this
# module changes shape because of them.
#
# **The invariant, and it is the whole of the correctness argument.** Two
# extra planes are held, `cbins` and `cgq`, and for every position `j` of the
# row buffer
#
#     cbins[f * n_rows + j] == bins[f * n_rows + rows[j]]     for every f
#     cgq[2 * j]            == gq[2 * rows[j]]
#     cgq[2 * j + 1]        == gq[2 * rows[j] + 1]
#
# A histogram launched against `(cbins, cgq, identity)` therefore reads, for
# every thread, **the identical bytes** the launch against `(bins, gq, rows)`
# would have read -- not the same multiset in a different order, the same byte
# at the same step of the same thread. So the histogram is bit-identical by
# construction and not by an associativity argument, and the accumulation loop
# needs no change at all: only which three pointers the launch is handed.
#
# **The permutation is untouched.** `rows_dev` is neither read nor written by
# the incremental path and is read (never written) by the rebuild. Every
# consumer of the permutation -- `_publish_row_ranges`, `update_raw_device`'s
# segment table, `download_rows`, `snapshot_rows`, the speculative prebuild's
# set-preservation argument -- sees exactly the buffer it saw before. That is
# what makes this lane a data movement and not a reordering.
#
# **What it costs.** One extra copy of the binned matrix and of the quantized
# gradients, doubled for the scatter's destination: `2 * n_rows * n_features`
# bytes plus `16 * n_rows`. At the reference shape (1,000,000 by 50) that is
# 100 MB plus 16 MB of device memory that the arm did not need before. This is
# why it is off by default and allocates nothing until asked for.


@always_inline
def _any_origin_u8[
    o: MutOrigin, //
](p: MutPointer[UInt8, o]) -> MutPointer[UInt8, MutAnyOrigin]:
    """Widen a bin pointer's origin to the one every kernel here declares.

    Needed for one reason and it is worth writing down, because the spelling
    looks gratuitous. The row-compaction arm has to hold **either** the
    caller's `bins` **or** `self.cbins_dev.unsafe_ptr()` in one variable and
    hand it to the launch. Those two have different origins -- `bins_origin`
    and `origin_of(self.cbins_dev)` -- and `Pointer` has no origin-widening
    conversion: the compiler refuses both, and it refuses
    `self.rows_dev` against `self.ident_dev` for the same reason.

    The two spellings that would avoid the widening both fail on something
    else. Threading the three pointers in as arguments cannot be done, because
    `_enqueue_atomic_family` takes `mut self` and a pointer into a field
    cannot be handed down from a caller that holds it -- which is the reason
    `blocked_ptr` and `desc` are already read inside the launch site rather
    than passed to it. Duplicating each of the eight launch blocks under an
    `if` would be sixteen copies of a twenty-eight-argument launch, which is
    exactly the shape of edit that lets two arms drift apart.

    What is actually unsafe about it is the lifetime, and the lifetime is not
    in doubt at either call site: every pointer widened here is either a field
    of `self` or an argument of the enclosing call, and both outlive the
    enqueue. The kernels this feeds already declare `MutAnyOrigin`, so nothing
    downstream sees a type it did not see before.
    """
    return MutPointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p))


@always_inline
def _any_origin_i32[
    o: MutOrigin, //
](p: MutPointer[Int32, o]) -> MutPointer[Int32, MutAnyOrigin]:
    """The Int32 twin of `_any_origin_u8`, for the permutation and the
    quantized gradient pair. Same argument, same lifetimes."""
    return MutPointer[Int32, MutAnyOrigin](unsafe_from_address=Int(p))


comptime COMPACTION_TRACE_VAR = "MOJOTREES_GPU_COMPACTION_TRACE"
"""Where the per-tree compaction record goes, or empty for no record.

A path is appended to; `1`, `stdout` or `-` goes to standard output. The same
contract `RESIDENT_TRACE_VAR` has in `gpu_resident_round.mojo`, deliberately,
because a reader who has learned one should not have to learn the other.

It exists because the arm this module ships is reachable end to end only
through an environment variable, and this repository has already once run one
arm under the other's label that way. The record says, per tree, whether the
arm was on and how many launches it had actually issued by then, which is a
wire rather than a switch: an off arm reports zero for the whole fit and a
requested-but-never-engaged arm is distinguishable from a working one.
"""


def _compact_trace_sink() -> String:
    return getenv(COMPACTION_TRACE_VAR)


def _compact_trace_emit(sink: String, text: String) raises:
    """One record to the sink, or nothing when the sink is empty.

    Appending, one open per record, for the reasons
    `gpu_resident_round._resident_trace_emit` gives: a whole fit reads in the
    order it was grown, and a fit that dies mid-way leaves every tree before
    it on disk. One open per tree, on a path nobody measures because the sink
    is empty on every path anybody measures.
    """
    if sink == "":
        return
    if sink == "1" or sink == "stdout" or sink == "-":
        print(text, end="")
        return
    with open(sink, "a") as handle:
        handle.write(text)


def _compact_build_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    cbins: MutPointer[UInt8, MutAnyOrigin],
    cgq: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_features: Int32,
    packed_grads: Int32,
):
    """Establish the invariant from scratch, over the whole row buffer.

    One thread per position, each walking every feature. This is the only
    scattered read in the mechanism -- `bins[f * n_rows + r]` with `r` out of
    the permutation -- and it is paid once per tree, not once per split. Every
    later split maintains the invariant incrementally out of the already
    compacted plane, which is what makes the total telescope; a design that
    re-gathered from `bins` at each split would pay a full scattered pass per
    split and would be strictly worse than not compacting at all.

    **Why it walks the whole buffer and not the active prefix.** `begin_tree`
    fills every one of `n_rows` entries -- iota unbagged, the staged bag then
    zeros when bagged -- so every entry is a valid row id and the extra work
    past `n_active` is bounded by the buffer rather than unbounded. Reading
    `n_active` here would mean reading `self.ranges`, and that table is
    deliberately poisoned for the width of a descriptor partition; a rebuild
    that raised because a window was under device control would be a rebuild
    that failed exactly when it was needed.

    **`packed_grads` is the staged gradient width, and it is `quant_packed`
    and not `packed_gradients`.** The gradient-staging lane made `gq_dev` hold
    either interleaved Int32 pairs or interleaved Int16 pairs in the first
    `4 * n_rows` bytes of the same allocation, selected per launch. The
    compacted copy has to hold whichever the source holds, or the histogram
    would gather one width out of a plane written at the other, which is a
    wrong histogram that decodes to legal values and that nothing downstream
    would catch. The caller reads the *staging* state rather than the request
    for exactly the reason the histogram launch does, and
    `GpuActiveRows.compact_packed` then records which width these planes were
    written at so a later flip cannot be missed.

    **Why it is not just called `packed`.** Because `_compact_scatter_kernel`
    already has a local of that name -- the *packed routing offset*,
    `(prefix << 1) | flag`, which this file has spelled `packed` since the
    partition was written -- and a parameter named `packed` there is silently
    shadowed by it. That shadowing compiles, and what it produces is a
    gradient plane moved at the Int16 width on the Int32 arm, which is a
    wrong histogram over legal-looking values. It was caught by the
    byte-level invariant check and by nothing else, so the three kernels
    carry the longer name and this paragraph.

    Under `packed_grads` the two words are moved through an Int16 view of both
    pointers. That is a bitcast of the same allocation and not a second
    buffer: `cgq` is `2 * n_rows` Int32 exactly as `gq` is, so the Int16 pair
    for position `j` lives at Int16 index `2j`, at byte `4j`, inside the first
    `4 * n_rows` bytes, which is the identical arithmetic the source uses.
    """
    var n = Int(n_rows)
    var nf = Int(n_features)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var j = Int(global_idx.x)
    # Dereferenced only under `packed_grads`, which is block-uniform.
    var gq16 = gq.unsafe_bitcast[Int16]()
    var cgq16 = cgq.unsafe_bitcast[Int16]()
    while j < n:
        var r = Int(rows[unsafe_offset=j][0])
        for f in range(nf):
            cbins[unsafe_offset=f * n + j] = bins[unsafe_offset=f * n + r][0]
        if Int(packed_grads) != 0:
            cgq16[unsafe_offset=2 * j] = gq16[unsafe_offset=2 * r][0]
            cgq16[unsafe_offset=2 * j + 1] = gq16[unsafe_offset=2 * r + 1][0]
        else:
            cgq[unsafe_offset=2 * j] = gq[unsafe_offset=2 * r][0]
            cgq[unsafe_offset=2 * j + 1] = gq[unsafe_offset=2 * r + 1][0]
        j += stride


def _compact_scatter_kernel(
    cbins: MutPointer[UInt8, MutAnyOrigin],
    cbins_alt: MutPointer[UInt8, MutAnyOrigin],
    cgq: MutPointer[Int32, MutAnyOrigin],
    cgq_alt: MutPointer[Int32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    begin: Int32,
    count: Int32,
    tiles: Int32,
    n_blocks: Int32,
    n_rows: Int32,
    n_features: Int32,
    packed_grads: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
):
    """Apply the row scatter's permutation to the compacted planes.

    The destination arithmetic below is `_scatter_kernel`'s, reproduced
    expression for expression: the same head scan over `block_sums`, the same
    `mine` and `carry`, the same packed offset, the same
    `dst = p` / `dst = carry + (j - p)`. That is not a coincidence to be
    tidied away into a shared helper -- the two kernels must agree on every
    element or the invariant breaks silently, and the way this file already
    argues its exactness is that the destination is a pure function of the
    routing flags and the element's position in the range. This kernel
    consumes the *same* `offsets` and `block_sums` the row scatter consumes,
    written by the *same* flag pass, so the two cannot disagree even in
    principle: they are evaluating one function of one input twice.

    It writes no total. `_scatter_kernel` owns `total_dev` and this one is
    launched beside it, so a second write would be two kernels storing the
    same value to the same address for no reason.

    One thread per element of the range, walking every feature, exactly as the
    rebuild does. The alternative -- a second grid dimension over features --
    would make every one of `n_blocks * n_features` blocks redo the head scan
    over the block sums, and would read the packed offset once per feature
    instead of once per row.

    **Reads are contiguous, which is the entire point.** Position `j` of the
    range reads `cbins[f * n_rows + begin + j]`, and consecutive threads take
    consecutive `j`, so each feature's read is a dense run. The writes are two
    monotone runs, the left rows ascending from `begin` and the right rows
    ascending from `begin + carry`, because the partition is stable. So both
    sides of the move coalesce, where the gather this exists to remove does
    not.

    **A dead step writes nothing.** The kernel returns at the descriptor's
    `STEP_LIVE` word, which is right, because the copy-back beside it also
    returns and `cbins` is left holding the plane it already held. A `swap`
    parameter used to make a dead step copy the window identically instead, so
    that the whole-buffer ping-pong `_enqueue_compact_partition` describes
    stayed a pure renaming in both cases; that ping-pong was removed with the
    compacted level read on 2026-08-18 and this parameter went with it. See
    `docs/design/DECLINED_OPTIMIZATIONS.md` row C1.
    """
    var tid = thread_idx.x
    var nthreads = block_dim.x
    var nb = Int(n_blocks)
    var me = Int(block_idx.x)
    var n = Int(count)
    var b = Int(begin)
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        b = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])

    # A block whose whole chunk lies past the end of the range writes nothing
    # and so does not need the head scan. Block-uniform, so the whole
    # threadgroup leaves together. Unlike `_scatter_kernel` there is no block
    # 0 exemption, because this kernel writes no total: the docstring above
    # says so and it is the reason the condition here is the simpler one.
    if me * Int(tiles) * Int(nthreads) >= n:
        return

    var s = stack_allocation[
        SCAN_MAX_THREADS,
        Scalar[DType.int32],
        address_space=AddressSpace.SHARED,
    ]()

    var carry = Int32(0)
    var mine = Int32(0)
    var base = 0
    while base < nb:
        var idx = base + tid
        var own = Int32(0)
        if idx < nb:
            own = block_sums[unsafe_offset=idx][0]
        s[unsafe_offset=tid] = own
        barrier()

        var offset = 1
        while offset < nthreads:
            var carried = Int32(0)
            if tid >= offset:
                carried = s[unsafe_offset=tid - offset][0]
            barrier()
            if tid >= offset:
                s[unsafe_offset=tid] = s[unsafe_offset=tid][0] + carried
            barrier()
            offset += offset

        if me >= base and me < base + nthreads:
            mine = (
                carry
                + s[unsafe_offset=me - base][0]
                - block_sums[unsafe_offset=me][0]
            )
        carry += s[unsafe_offset=nthreads - 1][0]
        barrier()
        base += nthreads

    var nr = Int(n_rows)
    var nf = Int(n_features)
    # Dereferenced only under `packed_grads`, which is block-uniform. The
    # name is deliberately not `packed`: that is the packed routing offset
    # read four lines below, and a parameter of that name is shadowed by it
    # silently. See `_compact_build_kernel`.
    var cgq16 = cgq.unsafe_bitcast[Int16]()
    var cgq16_alt = cgq_alt.unsafe_bitcast[Int16]()
    var chunk = Int(tiles) * nthreads
    var first = me * chunk
    for t in range(Int(tiles)):
        var j = first + t * nthreads + tid
        if j < n:
            var dst: Int
            var packed = offsets[unsafe_offset=j][0]
            var p = Int(mine) + (Int(packed) >> 1)
            if (packed & Int32(1)) != 0:
                dst = p
            else:
                dst = Int(carry) + (j - p)
            var src = b + j
            var out = b + dst
            for f in range(nf):
                cbins_alt[unsafe_offset=f * nr + out] = cbins[
                    unsafe_offset=f * nr + src
                ][0]
            if Int(packed_grads) != 0:
                cgq16_alt[unsafe_offset=2 * out] = cgq16[
                    unsafe_offset=2 * src
                ][0]
                cgq16_alt[unsafe_offset=2 * out + 1] = cgq16[
                    unsafe_offset=2 * src + 1
                ][0]
            else:
                cgq_alt[unsafe_offset=2 * out] = cgq[unsafe_offset=2 * src][0]
                cgq_alt[unsafe_offset=2 * out + 1] = cgq[
                    unsafe_offset=2 * src + 1
                ][0]


def _compact_copy_back_kernel(
    cbins: MutPointer[UInt8, MutAnyOrigin],
    cbins_alt: MutPointer[UInt8, MutAnyOrigin],
    cgq: MutPointer[Int32, MutAnyOrigin],
    cgq_alt: MutPointer[Int32, MutAnyOrigin],
    begin: Int32,
    count: Int32,
    n_rows: Int32,
    n_features: Int32,
    packed_grads: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
):
    """Fold the scattered window back over the planes the next split reads.

    The data twin of `_copy_back_kernel`, and it is a copy for exactly the
    reason that one is: only the parent's own window moved, so a buffer swap
    would strand every other live leaf's compacted rows in the other plane.
    The row path writes up per-range ping-pong as a design it did not build
    because four readers outside this module hold `rows_dev`; here there are
    no outside readers at all, but there is still no single window whose swap
    would leave both planes whole, so the copy stays.

    It is the honest cost of the mechanism and it is reported as such: the
    scatter moves the window once and this moves it again, so a split moves
    `4 * count * (n_features + 8)` bytes in total rather than `2 *`.
    """
    var n = Int(count)
    var b = Int(begin)
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        b = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])
    var nr = Int(n_rows)
    var nf = Int(n_features)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var j = Int(global_idx.x)
    # Dereferenced only under `packed_grads`; see `_compact_build_kernel`.
    var cgq16 = cgq.unsafe_bitcast[Int16]()
    var cgq16_alt = cgq_alt.unsafe_bitcast[Int16]()
    while j < n:
        var i = b + j
        for f in range(nf):
            cbins[unsafe_offset=f * nr + i] = cbins_alt[
                unsafe_offset=f * nr + i
            ][0]
        if Int(packed_grads) != 0:
            cgq16[unsafe_offset=2 * i] = cgq16_alt[unsafe_offset=2 * i][0]
            cgq16[unsafe_offset=2 * i + 1] = cgq16_alt[
                unsafe_offset=2 * i + 1
            ][0]
        else:
            cgq[unsafe_offset=2 * i] = cgq_alt[unsafe_offset=2 * i][0]
            cgq[unsafe_offset=2 * i + 1] = cgq_alt[unsafe_offset=2 * i + 1][0]
        j += stride


def _quantize_grad_hess_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    g_scale: Float32,
    h_scale: Float32,
):
    """Quantize this round's gradients once, into one interleaved buffer.

    `gq[2r]` and `gq[2r + 1]` are exactly the `Int32(round(grad[r] *
    g_scale))` and `Int32(round(hess[r] * h_scale))` a histogram kernel would
    otherwise compute per (row, feature) visit: same Float32 product, same
    rounding, so a histogram accumulated from this buffer is bit-identical to
    one accumulated from the Float32 planes. That is an argument from the
    expression, not a measurement: the two sites evaluate the same Float32
    multiply and the same `round`, and hoisting an expression out of a loop
    whose other operands it does not depend on cannot change its value.

    What changes is the traffic. The histogram kernels gather `grad[r]` and
    `hess[r]` by row id, so at a leaf whose rows are sparse in the dataset
    each is a separate cache line per row per threadgroup, and a 50-feature
    histogram at the pairing gathers both 25 times per row. Interleaving the
    two quantized words puts them in one line, halving those gathers, and
    moves the two multiplies and rounds off the inner loop entirely. One
    streaming pass per tree over `n_rows` (16 bytes each), against
    `n_features` gathered passes per node.

    Because the result is provably identical and the work is strictly less,
    this is the default source (`MOJOTREES_GPU_QUANTIZED_GRADS`, on unless set
    to 0). The Float32 arms of the kernels below are kept so a benchmark can
    hold both, not because either is more correct than the other.
    """
    var r = global_idx.x
    if r < Int(n_rows):
        gq[unsafe_offset = 2 * r] = Int32(
            round(grad[unsafe_offset=r][0] * g_scale)
        )
        gq[unsafe_offset = 2 * r + 1] = Int32(
            round(hess[unsafe_offset=r][0] * h_scale)
        )


# --- The packed Int16 staging arm ----------------------------------------
#
# What the histogram gather actually costs, which is the number this arm is
# aimed at. Per (row, feature) visit at `GROUP` owned slots, the loop reads
# the row index once per row (4 bytes), the staged derivative once per row,
# and one bin byte per visit -- so the derivative's share of the per-visit
# traffic is its width divided by `GROUP`. At the default `GROUP = 1` that
# share is the whole of it, and halving the staged width halves it.
#
# Nothing about the accumulation changes. `Int32(Int16(q)) == q` for every
# `q` in `[-32768, 32767]`, sign extension being exact, so the shared plane
# receives the identical integer and the histogram is bit-identical to the
# Int32 arm's. `quantized_gradient.check_int16_staging` is the bound written
# out, its scope argued, and the reason it is not section 5 of
# `docs/design/ACCURACY_BUDGET.md`.

comptime QUANT_SOURCE_FLOAT = 0
"""`use_quant`: the two Float32 planes, rounded in the row loop."""
comptime QUANT_SOURCE_INT32 = 1
"""`use_quant`: the interleaved Int32 pairs, the shipped default."""
comptime QUANT_SOURCE_PACKED16 = 2
"""`use_quant`: the same allocation reinterpreted as interleaved Int16 pairs.
Selected by `GpuActiveRows.set_packed_gradients`; refused at staging time,
never at the launch, because the bound is a property of the round's values
and not of its shape."""

comptime STAGE16_MAX = Int32(32767)
comptime STAGE16_MIN = Int32(-32768)
"""The signed 16-bit range, as the device compares against it. Named rather
than spelled inline so the kernel's condition and
`quantized_gradient.INT16_LIMIT` are visibly the same bound."""

comptime STAGE16_FLAG_WORDS = 2
comptime STAGE16_FLAG_GRAD = 0
comptime STAGE16_FLAG_HESS = 1
"""One counter per plane. Two rather than one because the hessian word is not
read at all on a constant-hessian round, so a hessian that did not fit is not
a reason to refuse that round; `_ensure_quantized` reads the gradient counter
always and the hessian counter only when the plane will be gathered."""


def _quantize_grad_hess_i16_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq16: MutPointer[Int16, MutAnyOrigin],
    flag: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    g_scale: Float32,
    h_scale: Float32,
):
    """`_quantize_grad_hess_kernel` at half the width, plus the overflow
    counters that make the narrowing checkable rather than assumed.

    **The value is formed identically.** `Int32(round(x * scale))` is
    reproduced here character for character from the Int32 kernel, for the
    same contraction reason `docs/NUMERICS.md` section 5.6 records and the
    same reason the Float32 arm of `_hist_rows_step` reproduces it: the
    multiply is consumed by `round`, so there is no add for it to fuse into,
    and that argument survives only while the expression does. The narrowing
    is applied *after* that integer exists, so the two kernels agree on the
    integer and disagree only on how many bytes it is stored in.

    **The check is on the integer, not on a magnitude.** A host check would
    need `max_r |g_r|`, which nothing on the device path reduces today --
    `_abs_sum_kernel` reduces the sum, and adding a second reduction would
    have to reach `GpuHistogramBuilder._refresh_scales` to be read, in a file
    this lane may not edit. Comparing the value each thread has already
    computed costs two compares per row on a pass that runs once per tree and
    needs no reduction at all. `quantized_gradient.check_int16_staging` is the
    same inequality in the form a host test can call.

    **The counter is a count and not a flag** so the message can say how many
    rows failed, which is the difference between "your scale is one row's
    outlier away" and "this shape does not fit". The atomic executes only on
    a row that fails, so a passing round pays the compare and nothing else.

    **Both words are always written.** The hessian word is dead on a
    constant-hessian round, and writing it anyway is what keeps one layout,
    one kernel and one cache key; a layout that depended on `constant_hessian`
    would be stale the moment a caller flipped it between trees without a
    refill, which is a wrong histogram no fixture catches. Its overflow is
    counted separately so that the dead word cannot refuse a round that never
    reads it.
    """
    var r = global_idx.x
    if r < Int(n_rows):
        var qg = Int32(round(grad[unsafe_offset=r][0] * g_scale))
        var qh = Int32(round(hess[unsafe_offset=r][0] * h_scale))
        if qg > STAGE16_MAX or qg < STAGE16_MIN:
            _ = Atomic.fetch_add(
                flag.unsafe_offset(STAGE16_FLAG_GRAD), Int32(1)
            )
        if qh > STAGE16_MAX or qh < STAGE16_MIN:
            _ = Atomic.fetch_add(
                flag.unsafe_offset(STAGE16_FLAG_HESS), Int32(1)
            )
        gq16[unsafe_offset = 2 * r] = qg.cast[DType.int16]()
        gq16[unsafe_offset = 2 * r + 1] = qh.cast[DType.int16]()


def _raise_stage16(
    plane: String, n_over: Int, n_rows: Int, scale: Float64
) raises:
    """The refusal `GpuActiveRows._check_stage16_bound` issues, with the
    numbers a caller needs to decide what to do rather than only the fact.

    Three of them. The **count** separates "one outlier row" from "this shape
    does not fit", which are different problems with different answers. The
    **row total** is what the bound moves with -- under the shipped
    `2^30 / sum|g|` scale a row carries about `1/n` of the magnitude, so the
    bound tightens as `n` falls and a fit that failed at fifty thousand rows
    will not pass by retrying. The **scale** is there because it is the other
    factor in the product and because a caller who has moved
    `GpuHistogramBuilder.set_scale_refresh`'s headroom has moved exactly it.

    The remedy named is the Int32 buffer and not a smaller scale, deliberately.
    Shrinking the scale until every row fits sixteen bits is the
    magnitude-sum-at-`2^14` move that `docs/design/ACCURACY_BUDGET.md`
    section 5 measures at 4.31 percent worse held-out logloss with the same
    sign on all six seeds, outside this project's own three percent tolerance.
    It is the obvious-looking fix and it is the worst option on the table, so
    the message does not offer it.
    """
    raise Error(
        String(
            "the packed Int16 gradient staging arm lost ",
            String(n_over),
            " of ",
            String(n_rows),
            " rows on the ",
            plane,
            " plane: their quantized values do not fit a signed 16-bit word"
            " at the scale in force (",
            String(scale),
            "). This bound is per row over the whole round and it tightens as"
            " the row count falls, so the remedy is the Int32 staged buffer"
            " (GpuActiveRows.set_packed_gradients(False), which is the"
            " default), not a smaller scale -- narrowing the scale to fit is"
            " the arm docs/design/ACCURACY_BUDGET.md section 5 measures"
            " outside the project's own accuracy tolerance.",
        )
    )


# --- The row walk, shared by both accumulation strategies ----------------
#
# Rows one thread keeps in flight at once inside the histogram row loop.
#
# What this constant is for. The row loop's per-row work is a chain of
# dependent loads: read the row index out of the permutation, use it to
# gather the quantized gradient pair, use it again to gather this slot's bin
# byte, and only then issue the shared atomics. Written one row at a time,
# each of those loads has to complete before the next address can be formed,
# so a thread has at most one memory request outstanding at any moment and
# the loop runs at the latency of the gather rather than at the bandwidth of
# it. Walking `HIST_ROW_UNROLL` rows per iteration, with every load of a
# stage issued before any load of the next stage is consumed, puts that many
# requests in flight per thread instead of one.
#
# Why 4 and not 2 or 8. This is a guess and it should be read as one. Four
# rows costs four Int registers for the row indices plus three Int32 arrays
# of four, which is small against what a threadgroup already holds, and four
# is the smallest depth that covers a three-stage chain with one to spare.
# Nothing on this machine measured the right value, and nothing could
# without a device counter this backend does not expose. Changing it is a
# one-line edit and a rebuild, and the result is bit-identical at every
# value (see `_hist_rows_step`), so an A/B costs two builds and no
# correctness argument.
#
# `GpuActiveRows.set_row_unroll(False)` puts the loop back to one row per
# iteration at run time without a second instantiation, which is the arm a
# benchmark holds against this one; see `_hist_accumulate_rows`.
comptime HIST_ROW_UNROLL = 4


# --- The blocked row-major stripe, and why it is not here ----------------
#
# `docs/design/CLEANSHEET_GPU.md` section 3.3 proposes replacing the
# feature-major bin matrix with stripes of `W` features stored row-major,
# `W = floor(threadgroup_memory / (n_bins * cell_bytes))`, so that one
# gathered read of a padded `W`-byte record serves `W` features instead of
# one. Its traffic model puts a whole tree at 752 MB against today's 2650 MB
# at a 64-byte cache line, and 1136 MB at 128. That is the largest single
# lever anyone has costed on this backend and it is not built here. The
# reasons are worth writing down, because the next lane to reach for it
# should reach for the measurement first and not for the layout.
#
# **It is not a change to this file.** The row loop above reads `bins` as
# `feature * n_rows + row`. Every other reader of that buffer would have to
# change with it or a second copy would have to be maintained, and the
# readers are not in one place: `_partition_kernel` and `_flag_scan_kernel`
# in this module, the whole of `gpu_binned_layout.mojo`, the leaf-id scan in
# `histogram_gpu.mojo`, `gpu_split_search.mojo`, `gpu_sparse.mojo`,
# `gpu_categorical.mojo`, `gpu_leaf_batching.mojo`, `gpu_multiclass_batch.mojo`,
# and the CPU builder in `histogram.mojo` that the host replica has to stay
# bit-identical to. A half-applied stripe layout, where some kernels read the
# striped copy and others read the column-major one, is a much worse state
# than none: the two agree until a lane forgets to convert one of them, and
# then one node's histogram is wrong on some shapes and not others.
#
# **The arithmetic that motivates it turns on a number nobody here has
# measured.** The whole 3.5x-versus-2.3x spread in that section is the cache
# line size, and section 8 of the same document lists it as unknown. There is
# no counter on an Apple M4 that reports it and no documented figure this
# project trusts. So the honest order of work is: measure the line first,
# then decide whether the layout is worth its blast radius.
#
# **The measurement, which is small and is not this lane's file.** A
# standalone kernel that gathers one 4-byte word per thread from a buffer far
# larger than any cache, at a stride `S` swept over 4, 8, 16, 32, 64, 128,
# and 256 bytes, and reports achieved bytes per second. Useful traffic is
# constant across the sweep and real traffic is `max(S, L)` per gather, so
# the achieved rate falls in proportion to `S` above the line size and is
# flat at or below it. The knee is `L`. Two further points are worth taking
# in the same harness, because they are the assumptions the stripe rests on
# and not merely the line size: whether a 16-byte gathered read costs the
# same as a 4-byte one at the same stride, which is the specific claim
# section 3.3 flags as its least certain, and whether padding a record to a
# power of two changes the answer. That harness belongs in `bench/`, which
# this lane does not own, and it should be written before a single byte of a
# second bin layout is.
#
# Nothing above is an argument that the stripe is wrong. It is an argument
# that it is a project and not a patch, and that a cheap experiment stands
# between here and knowing which of the two numbers in the model applies.


# The three shared accumulation planes, as `stack_allocation` hands them
# back. Named here because the row walk takes them as arguments rather than
# allocating them, and a threadgroup pointer's type has to be written out to
# cross a function boundary.
comptime _SharedI32 = Pointer[
    Int32, MutUntrackedOrigin, address_space = AddressSpace.SHARED
]

# Per-thread scratch: the column bases, the rows in flight, and the values
# gathered for them. Every index into these arrays is a compile-time
# constant, so they exist to be promoted into registers and never to be
# addressed.
comptime _LocalInt = Pointer[
    Int, MutUntrackedOrigin, address_space = AddressSpace.GENERIC
]
comptime _LocalI32 = Pointer[
    Int32, MutUntrackedOrigin, address_space = AddressSpace.GENERIC
]


# --- The histogram decomposition probes ----------------------------------
#
# Four arms of one kernel at one launch geometry, so that the histogram
# phase can be split into its parts rather than guessed at. Every arm above
# `OFF` builds a histogram that is WRONG BY CONSTRUCTION and exists only to
# produce a wall-clock ratio; `GpuActiveRows.set_histogram_probe_mode` is
# where the refusals and the licensing live.
#
# Why the set is exactly this. `NO_ATOMICS` was built by the atomic-fraction
# lane and measured the three shared atomics at 9.8% of the phase at
# 1,000,000 x 50 x 256 bins. That left roughly ninety percent of the phase
# unattributed, and every kernel proposal on the table aimed at either the
# atomics or the addresses -- the addresses having separately been closed as
# a no-op at the shipping configuration. `NO_GATHER` and `EMPTY` are the two
# cuts that partition the remainder: what the loop's memory traffic costs,
# and what the launch itself costs before any thread does anything.
#
# The four are subtractive, not additive. Nothing here claims the four
# numbers sum to one; each arm removes a part and what is left is a residual
# that includes the shared zeroing, the flush, the barriers, and the parts
# every arm keeps. Reading them as a partition is the one mistake this
# instrument makes easy, which is why each arm's docstring says which way
# its number errs.
comptime HIST_PROBE_OFF = 0
# Arm two, the atomic-fraction lane's. Keeps the row load, the gradient-pair
# load, the bin gather and the shared-cell index; drops the three
# `Atomic.fetch_add`; folds into a per-thread XOR sink.
comptime HIST_PROBE_NO_ATOMICS = 1
# Arm three. Drops all three loads -- the row index, the quantized gradient
# pair, and the bin byte -- and keeps the three shared atomics, the loop trip
# count, the shared zeroing, the barrier and the whole flush. The bin walks a
# per-thread rolling counter instead of being gathered. See
# `_hist_rows_step`.
comptime HIST_PROBE_NO_GATHER = 2
# Arm four. The kernel returns immediately after its threadgroup planes are
# declared and touched, so what remains is the launch, the grid, the
# threadgroup footprint that bounds occupancy, and nothing else. See
# `_hist_probe_empty_mark`.
comptime HIST_PROBE_EMPTY = 3


@always_inline
def _hist_probe_empty_mark[
    CELLS: Int
](
    sg: _SharedI32,
    sh: _SharedI32,
    sc: _SharedI32,
    out_ptr: MutPointer[Int32, MutAnyOrigin],
    mark: Int32,
    tid: Int,
):
    """Arm four's whole body: touch each threadgroup plane and leave.

    **This is not an optimization and a kernel that takes this path produces
    no histogram at all.** It is the launch-and-occupancy floor, which is the
    one term in the histogram phase nobody has ever measured, and at ten-odd
    launches per resident step it is not obviously small.

    **Why an "empty" arm is the hardest of the four to build.** A kernel body
    that does nothing is exactly what a compiler is best at deleting, and the
    deletion is invisible in the output: an arm optimized into nothing
    reports the launch floor as zero and sends the next lane at fusion for no
    reason. Three separate deletions have to be prevented, and the first
    version of this function fell to two of them, which is why the shape
    below is as elaborate as it is.

    1. **Store-to-load forwarding.** Version one wrote six cells and read
       them back at the same indices. The compiler forwarded each store to
       its own load, folded `mark ^ mark ^ mark` down to `mark`, and the
       whole function compiled to a single `ret`. The fix is that the three
       loads are at `r`, an index the compiler cannot prove equal to either
       written index and cannot prove unequal either, so it may neither
       forward nor drop.
    2. **Dead-store elimination.** Threadgroup memory is not observable after
       a kernel ends, so six stores nobody reads are removable in principle.
       They survive because the loads at `r` may alias them.
    3. **Dead-allocation elimination.** The three planes are what bounds
       resident blocks per core, so an arm that let them be dropped would run
       at an occupancy the real kernel never gets and report a floor that is
       too cheap. They survive because the stores do; and their *size*
       survives because the only bound a compiler can put on `p`, `q` and `r`
       is the mask, which is `[0, CELLS)`, so no range analysis can narrow
       the array. That is a claim about the *static* range and not the
       runtime one: with 256 threads and 4096 cells the block only ever
       touches a sixteenth of each plane, and it is the mask rather than the
       traffic that keeps the declaration its full width.

    The chain terminates in a **global** store, which is the only kind of
    write in a kernel that is observably not removable. It is guarded by
    `v == Int32.MIN`, a condition no analysis here can discharge and that no
    run reaches: `mark` is an XOR of a thread index and two block indices, so
    it and every XOR of copies of it are far below 2^31, and the store never
    executes. If it somehow did, it would write one garbage word into the
    first cell of a histogram this arm has already made entirely garbage.

    `CELLS` is `GROUP * BIN_CAP`, a power of two at every one of the twenty
    instantiations, so the two index masks are `and` instructions and not
    divisions.

    **What was read, and the negative control.** A reduced replica -- three
    `CELLS`-wide arrays, these six stores, these three loads, this guarded
    global store -- was compiled to optimized target assembly and read. The
    positive reading: a 48 KB frame for three 4096-cell Int32 arrays, six
    `str` at two masked indices, three `ldr` at the third, the two `eor`, and
    the compare-and-branch around `str w8, [x0]`. Thirty instructions, all of
    them the ones written. **The negative control:** with the guarded global
    store deleted and nothing else changed, the function does not shrink --
    it disappears. No symbol, no call from `main`, and no 48 KB frame
    anywhere in the object. That is what gives the positive reading teeth,
    and it is the same standard the atomics arm's replica was held to.

    The replica is host arm64, because this repository has no dump path for a
    Metal kernel compiled at launch; store-to-load forwarding, dead-store
    elimination and dead-allocation elimination are LLVM middle-end
    transforms common to both, and the claim is scoped to that. What the host
    replica cannot show is Metal's threadgroup-memory allocation decision,
    which is made by a backend this project cannot read. That is the arm's
    one unverified assumption and it errs in a known direction: if the
    threadgroup planes were narrowed anyway, the arm would run at a higher
    occupancy than the real kernel and report a floor that is **too cheap**.

    **Which way arm four's number errs.** In two directions, and they do not
    cancel, so both are stated:

    - It is an **over**estimate of a bare launch, by six threadgroup stores,
      three threadgroup loads, two masks and one branch per thread. Against
      the thousands of shared atomics a real thread issues that is a fraction
      of a percent, but it is not zero.
    - It is an **under**estimate of the fixed per-launch cost of the real
      histogram kernel, and by much more. It skips the shared zeroing, both
      barriers, the feature-id reads and the entire flush, every one of which
      a real launch pays whether or not it has rows. Those land in the
      residual, which is where they belong: they are work, not launch.

    So the number bounds *launch overhead proper* from above and *fixed
    per-launch cost* from below, and the second gap is by far the larger.
    """
    var p = tid & (CELLS - 1)
    var q = CELLS - 1 - p
    var r = (p * 3 + 1) & (CELLS - 1)
    sg[unsafe_offset=p] = mark
    sh[unsafe_offset=p] = mark
    sc[unsafe_offset=p] = mark
    sg[unsafe_offset=q] = mark
    sh[unsafe_offset=q] = mark
    sc[unsafe_offset=q] = mark
    var v = (
        sg[unsafe_offset=r][0]
        ^ sh[unsafe_offset=r][0]
        ^ sc[unsafe_offset=r][0]
    )
    if v == Int32.MIN:
        out_ptr[unsafe_offset=0] = v


def narrow_index_fits(n_rows: Int, n_features: Int) -> Bool:
    """Whether a dataset of this shape admits the histogram kernels' Int32
    index arm (`GpuActiveRows.set_narrow_index`).

    Host-side arithmetic, deliberately a free function rather than a method,
    so the bound can be checked at its own boundary without allocating a
    device buffer for two billion rows.

    The two quantities the narrow arm forms in 32-bit arithmetic, and the
    only two, are the bin column offset `feature * n_rows + row` and the
    quantized pair offset `2 * row`. Both must fit a signed 32-bit integer
    for the narrow expression to have the same value as the wide one:

        n_features * n_rows <= Int32.MAX      (2,147,483,647)
        2 * n_rows          <= Int32.MAX

    The first bounds the column offset because a feature id is below
    `n_features` and a row id is below `n_rows`, so the largest offset the
    gather can form is `(n_features - 1) * n_rows + (n_rows - 1)`, which is
    `n_features * n_rows - 1`. Requiring the product itself to fit is
    therefore one short of tight, by exactly one cell, and it is written that
    way because the product is the quantity a reader can check against the
    matrix they allocated.

    The second binds only at `n_features == 1`, where the first is the weaker
    of the two. It is checked rather than inferred for that reason: a
    single-feature fit of 1.5 billion rows passes the first test and would
    wrap the second.

    Both products are evaluated in `Int`, which is 64 bits on every platform
    this backend builds for, so the check cannot itself overflow.
    `GpuActiveRows.__init__` already refuses `n_rows > MAX_ROWS`, so
    `n_features * n_rows` is at most `Int32.MAX * n_features` and cannot
    approach the 64-bit range for any feature count a machine can hold.

    This is a **derived bound**, not a measurement, and it is nowhere near
    tight against anything this library can run: at 1,000,000 rows it admits
    2,147 features, and at 50 features it admits 42,949,672 rows. A shape
    that fails it needs a binned matrix above 2.1 GB, which is past the
    memory of the device this backend is developed on. Nonpositive shapes
    answer False, because a shape that is not a dataset admits nothing;
    `__init__` refuses them outright and this is the answer for a caller that
    asks anyway.
    """
    if n_rows < 1 or n_features < 1:
        return False
    return n_features * n_rows <= MAX_ROWS and 2 * n_rows <= MAX_ROWS


@always_inline
def _hist_rows_step[
    GROUP: Int, U: Int, CELIDE: Bool, QUANT: Bool
](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    sg: _SharedI32,
    sh: _SharedI32,
    sc: _SharedI32,
    col: _LocalInt,
    bw: _LocalInt,
    rr: _LocalI32,
    gv: _LocalI32,
    hv: _LocalI32,
    bv: _LocalI32,
    sink: _LocalI32,
    pbin: _LocalI32,
    owned: Int,
    nb: Int,
    row_begin: Int,
    j: Int,
    stride: Int,
    g_scale: Float32,
    h_scale: Float32,
    narrow: Bool,
    aligned: Bool,
    packed: Bool,
    rstride: Int,
    probe_mode: Int,
):
    """Accumulate `U` rows of one thread's stride into the shared planes.

    The rows are `j, j + stride, ..., j + (U - 1) * stride` of the node's
    range, and the caller guarantees every one of them is inside the tile.
    Written in four stages, each of which issues all `U` of its loads before
    the next stage consumes any of them:

    1. `U` row indices out of the permutation.
    2. `U` quantized gradient pairs, gathered by those indices.
    3. per owned slot, `U` bin bytes out of that slot's column.
    4. per owned slot, `U` shared atomic triples.

    Stages 2 and 3 both depend on stage 1 and stage 4 depends on stage 3, so
    the chain is three deep however large `U` is; what `U` buys is `U`
    independent requests at each level of it rather than one.

    **Why this cannot change a histogram.** Every instantiation performs the
    same set of adds. For each (row, feature) in the tile it adds the same
    three quantized values -- the row's gradient, its hessian, and one -- to
    the same three shared cells, chosen by the same bin byte read from the
    same address. What `U` changes is only the order in which those adds are
    issued and the order in which the loads that feed them are issued.
    Accumulation is Int32 throughout and integer addition is associative and
    commutative, so a reordering of the adds cannot move a bit. It is the
    identical argument `_range_hist_atomic_kernel` already makes for `GROUP`
    and `BIN_CAP`, applied to a third launch-shape parameter, and it is
    structural rather than a tolerance.

    The one thing that would break it is arithmetic that is not integer, and
    there is exactly one such site: the Float32 arm's
    `Int32(round(x * scale))`. That expression is reproduced here character
    for character from the loop it replaces, on purpose. `docs/NUMERICS.md`
    section 5.6 records why it is structurally immune to contraction (the
    multiply is consumed by `round`, so there is no add for it to fuse
    into), and that argument survives unrolling only because the expression
    survives unrolling. A future edit that hoists the multiply, or that
    lets an add reach it, breaks a property this file depends on. The
    quantized arm, which is the default, has no floating-point arithmetic in
    it at all.

    **The gradient pair is one load.** `_quantize_grad_hess_kernel` writes
    the two quantized words interleaved, so `gq[2r]` and `gq[2r + 1]` are
    adjacent and, since the pair starts at byte `8r` of a device allocation,
    8-byte aligned. A width-2 load therefore reads both in one instruction
    where the loop it replaces issued two. It reads the identical eight bytes
    and assigns them to the identical two variables, so it cannot change a
    value; it is one fewer load and one fewer address computation per row.
    The elided-hessian arm reads only the gradient word and keeps the scalar
    load, because the second word is not wanted there.

    `aligned` is that load's alignment arm and it is not cosmetic.
    `unsafe_load[width=2]` without an explicit alignment emits
    `load <2 x i32>, align 4`, which is the element alignment and not the
    vector's: **observed** by compiling the two spellings side by side and
    reading the emitted LLVM IR, which is a compiler fact rather than a
    device measurement. An under-aligned vector load is one a backend is
    free to split back into two scalar loads, which would leave the width-2
    spelling costing exactly what the two loads it replaced cost.
    `alignment=8` states the alignment the address actually has and is the
    default here; `set_pair_alignment(False)` restores the weaker
    annotation so the two can be held as arms in one process. Both spellings
    read the same eight bytes and assign them to the same two variables, so
    the arm cannot change a histogram.

    **What the alignment claim rests on.** `gq` is
    `ctx.enqueue_create_buffer[DType.int32](2 * n_rows)`, and every device
    allocator this project targets returns memory aligned far past 8 bytes
    (256 on CUDA and HIP by their own documentation, page-aligned on Metal
    for a buffer created without a host pointer). The pair for row `r`
    begins at byte `8r` of that base, so the address is 8-byte aligned for
    every row. That is a **derived bound** from the allocator contract, not
    a measurement, which is why the weaker annotation stays reachable.

    **`packed` halves the bytes that load fetches, and is the one arm here
    whose exactness rests on a bound rather than on associativity.** It reads
    the same allocation reinterpreted as interleaved *Int16* pairs, which
    `_quantize_grad_hess_i16_kernel` wrote: two bytes per plane per row
    instead of four, so the elided arm's scalar load is 2 bytes rather than 4
    and the pairing arm's `width=2` load is 4 bytes rather than 8. Nothing
    else moves. The value assigned to `gv` is `Int32(gq16[2r])`, and sign
    extension of a 16-bit two's-complement word to 32 bits is exact, so
    whenever the stored word is the same integer the Int32 buffer would have
    held, this arm adds the identical integer to the identical shared cell.
    The accumulators stay Int32 in threadgroup memory, Int32 in the global
    planes, and Int32 through sibling subtraction: **this arm narrows a
    value, never an accumulator**, which is why it is not the candidate
    `docs/design/ACCURACY_BUDGET.md` section 5 prices and does not pay that
    candidate's two percent of effective sample size.

    **The bound, its scope, and where it is checked.** The stored word is the
    same integer only while every row of the round satisfies
    `-32768 <= Int32(round(x * scale)) <= 32767`. That is a **per-row bound
    over every row of the round**, not a per-node bound: the staged buffer is
    built once per round and read by every node of the tree, so a bound
    established on one node's rows would license nothing for the others. It
    is checked, not assumed, and not here -- a kernel cannot raise. Each row's
    own integer is compared against the range by
    `_quantize_grad_hess_i16_kernel` on the pass that stores it, failures are
    counted into a device word, and `GpuActiveRows._ensure_quantized` reads
    that word and **raises** before the histogram that would have read the
    truncated buffer is enqueued. So a violation ends the fit with a message
    naming the plane and the count, and no wrong histogram is ever built --
    which is a stronger position than `histogram_gpu._check_window_bound`, a
    check of the same family that can only report after the fact.
    `quantized_gradient.check_int16_staging` is the same inequality in the
    form a host test can call.

    **Why `packed` carries no `aligned` twin.** `aligned` exists because a
    `width=2` load of Int32 is 8 bytes at an address whose 8-byte alignment
    had to be argued from the allocator contract, and the weaker annotation
    stays reachable so the two spellings can be held as arms. The Int16 pair
    is 4 bytes at byte `4r` of the same base, so `alignment=4` is the pair's
    own natural alignment and is not a claim about anything: there is no
    second spelling to hold it against. `narrow` does apply and both of its
    spellings are written, because the index `2r` it narrows is formed the
    same way at either width.

    **The index arms.** `narrow` selects the width of the two index
    computations in this loop that a bound can narrow. The shared-plane
    index `s` is Int32 on both arms, because it is bounded by
    `GROUP * BIN_CAP`, at most 16 * 256 = 4096, by construction rather than
    by data. The two that depend on the dataset are:

    - `col[k] + r`, which is `feature * n_rows + row` and reaches
      `n_features * n_rows`. Wide it is an Int add of an Int product; narrow
      it is an Int32 add whose result is widened once to form the address.
    - `2 * r`, which reaches `2 * n_rows`.

    `row_begin + j + u * stride` stays Int on both arms. It is bounded by
    `n_rows` and would fit, but it is formed once per row against `GROUP`
    formed per row in the gather, and it is the induction variable of the
    enclosing loop, so narrowing it trades a 64-bit add for a 32-bit add
    plus a widening at every use.

    **What is claimed for `narrow`, and what is not.** Claimed: the narrow
    arm forms those two indices in 32-bit arithmetic where the wide arm
    forms them in 64-bit, on a backend whose ALU is 32 bits wide. Not
    claimed: that this removes any instruction. The wide gather is
    `bins + (base + sext(r))` with `base` loop-invariant, and a compiler
    that reassociates the address and hoists `bins + base` out of the row
    loop has already spent the 64-bit add once per slot rather than once per
    visit, in which case the narrow arm saves nothing and may cost a
    widening. Whether it does is exactly what the arm exists to measure, and
    the honest prior is that it is a null: this loop issues three shared
    atomics and one scattered global gather per (row, feature), against
    which one index add either way is noise.

    **Why `narrow` is off by default.** Every other launch-shape arm in this
    file that defaults on does so on a proof that it is strictly less work
    for the identical result. `narrow` has no such proof -- see the
    paragraph above -- and it additionally carries a precondition on the
    dataset shape that the wide arm does not. A default that can only be
    justified by a measurement waits for the measurement.

    **The precondition, and where it is checked.** The narrow arm is only
    correct while both quantities fit in a signed 32-bit integer:

        n_features * n_rows <= Int32.MAX      (2,147,483,647)
        2 * n_rows          <= Int32.MAX

    `GpuActiveRows.narrow_index_supported` evaluates both on the host in
    64-bit arithmetic, so the check itself cannot overflow, and
    `set_narrow_index` refuses a request the shape does not admit. Under
    that precondition the Int32 expression and the Int expression have the
    same mathematical value -- no wraparound occurs -- so the two arms
    address the same bytes and accumulate the same integers. This is
    therefore an exactness claim of a different kind from `U` or `GROUP`:
    those are reorderings of integer adds and are exact whatever the data,
    while this one is exact *given a checked bound on the shape*. That is
    why the bound is enforced rather than asserted.

    The bound is not tight against anything this library can run. At
    1,000,000 rows it admits 2,147 features; at 50 features it admits
    42,949,672 rows. A shape that fails it needs a `bins` matrix larger than
    2.1 GB, which is past the memory of the device this backend is developed
    on. `2 * n_rows` binds only when `n_features` is 1, and is the reason it
    is checked separately rather than inferred.

    **The layout arm, `rstride`.** How many bytes one row advances inside a
    bin column: 1 for the feature-major matrix this backend uploads, `G` for
    the `[block][row][G]` layout of `gpu_blocked_bins.mojo`. One value
    selects the whole reader, because the two layouts differ in exactly two
    places and both are captured by it -- the per-slot base, which
    `_hist_accumulate_rows` computes, and the per-row term, which is
    `r * rstride` here.

    It is a runtime value rather than a comptime parameter for the reason
    `narrow` and `aligned` are: this family is forty instantiations, each
    inlining this body twice, and the arm is one block-uniform test against
    `U * GROUP * 3` shared atomics. It is also what makes the layout
    selectable **in one binary**, which is what the interleaved-arms protocol
    on this machine requires and what a comptime variant could not have
    given: `GpuActiveRows.set_blocked_layout` moves it between repeats in one
    process.

    **Why this cannot change a histogram, and it is the strongest of the
    exactness arguments in this file.** `U` and `GROUP` reorder integer adds
    and lean on addition being associative and commutative. This arm does not
    even reorder them. The relayout writes
    `blocked[offset(f, r)] = bins[f * n_rows + r]` for every cell, so the
    byte this loop reads at `base + r * rstride` is the same byte it read at
    `feature * n_rows + row`; the same bin then selects the same shared cell,
    the same three quantized values are added to it, and the set of
    `(row, feature)` visits, the tiling, the slot assignment and the flush
    are all untouched. Nothing about the arithmetic moves, only the address
    the identical byte is fetched from. `gpu_blocked_bins.blocked_roundtrips`
    is that statement as an executable check, and
    `gpu_blocked_bins.check_blocked_matches_plan` proves the address this
    loop forms is the one the priced `BinLayoutPlan` publishes.

    The blocked arm carries no `narrow` twin. Its base already folds a
    16-byte-aligned block stride and a lane into one Int, and `narrow` is
    registered by its own author as an expected null against three shared
    atomics and a scattered gather. A fourth address arm to measure a null
    inside an arm is not a trade worth the code.

    **The packed arm, `bw`.** One entry per owned slot: the feature's storage
    width in bits, or `PACKED_WIDTH_OFF` when no packed matrix is on the
    device. `gpu_packed_bins.mojo` stores feature `f` as a stream of `n_rows`
    values at `ceil(log2(bins_used(f)))` bits beginning at `base[f]`, so a
    64-bin feature moves six bits per visit where it moved eight, a 16-bin one
    four, and a boolean one. A 255-bin feature needs all eight and gains
    nothing, which is why `set_packed_bins` refuses an all-width-8 table
    outright rather than uploading a second copy of the matrix.

    Widths 1 through 7 take the window decode; width 8 falls through to the
    byte gather two branches below, because at width 8 every shift is zero,
    the mask is the identity and the second window byte contributes nothing --
    `gpu_bin_packing`'s module docstring proves it -- so a width-8 stream is
    the feature's own column at an aligned base and reads exactly as it reads
    today. That is what makes a mixed-cardinality matrix pay the decode only
    on the features that are actually narrow.

    The width, the mask and the base are read once per owned slot, on exactly
    the footing `feature * n_rows` is on. What the row walk gains per visit is
    `(8 - w)/8` of the bin column's bytes; what it pays is one multiply, one
    shift, one mask, and a second byte load *adjacent to the first*. The two
    window bytes are one memory sector except where the pair straddles a
    sector boundary, so the load count doubles while the sector count does
    not; `gpu_packed_bins.packed_window_bytes` and
    `packed_bin_bytes_per_visit` report the two sides separately so a
    measurement has both.

    The packed arm has no `narrow` twin either, and for a sharper reason than
    the blocked arm's. `narrow`'s precondition is `n_features * n_rows <=
    Int32.MAX`, which bounds `feature * n_rows + row`; it says nothing about
    `row * width`, which this arm forms instead. The address is formed in Int
    throughout and is bounded by `gpu_packed_bins.packed_bytes`, which the
    host checked against `Int32.MAX` before the buffer was allocated.

    **Why this cannot change a histogram.** By construction, and the argument
    is that the bin id is not touched. `gpu_bin_packing` never renumbers a
    value: packing at width `w` and unpacking at width `w` returns the integer
    that went in, and the reader's expression here is
    `unpack_value(base, row, w)` written out, so the id this loop decodes for
    `(row, feature)` is the id `bins[feature * n_rows + row]` held. The same id
    then selects the same shared cell, the same three quantized values are
    added to it, and the tiling, the slot assignment, the visit set and the
    flush are all untouched -- nothing reorders, so associativity is not even
    needed, though accumulation is fixed-point Int32 and would supply it.

    The encoding is lossless only while every width covers its feature's
    largest bin id, its missing sentinel, and its categorical split-set bins.
    All three fail *silently* -- a truncated id is a legal id -- so all three
    are refused on the host before the buffer exists:
    `gpu_packed_bins.packed_widths_from_matrix` derives widths from the
    observed extents including the reserved missing bin,
    `check_packed_widths_cover` re-checks a table against a matrix cell by
    cell, and `packed_roundtrips` is the whole invariant as one function.

    **`probe_mode == HIST_PROBE_NO_GATHER` removes the loads and keeps the
    atomics, which is the reverse cut.** It is the third arm of the
    decomposition and it builds a histogram that is wrong on purpose, like
    the second. What it drops is all three global reads: the row index out of
    `rows`, the quantized gradient pair out of `gq` (or the two Float32 plane
    gathers), and the bin byte out of `bins`. What it keeps is everything the
    atomic arm keeps and the atomic arm's own atomics: the same loop trip
    count over the same rows of the same tile, the same `lift + bin`
    shared-cell index, the same two or three `Atomic.fetch_add` per (row,
    feature), the same shared zeroing, the same barrier, and the entire
    flush or partial write. So this arm's ratio against the full kernel
    isolates the loop's **memory traffic** from the arithmetic and the
    contention that consume it, which is precisely the term the atomic arm
    could not see.

    **The synthetic bin, and why it is not a constant.** The bin comes from a
    per-thread rolling counter in `pbin`, advanced by one per (row, feature)
    and wrapped by a compare and a reset to zero -- no division, so it is
    correct at any `n_bins` and touches no divide unit. It is seeded at
    `tid % nb`, so at any point in the walk the threads of a warp hold
    distinct consecutive bins wherever the warp is narrower than `nb`, which
    at the reference shape's 256 bins is every warp. That is as close to the
    conflict profile of uniform random bins as an address stream that reads
    nothing can be, and uniform is what the reference dataset has. A single
    constant bin was the obvious spelling and is the wrong one: every thread
    would hammer one cell, the shared atomic unit would serialize the entire
    threadgroup, and the arm would come out *slower* than the kernel it is
    meant to be a floor for. A rolling counter reproduces the scatter without
    reading a byte.

    The gradient is a genuine per-thread constant, `pbin[1]`, set once
    outside the walk. Nothing folds it away: the three atomics write three
    distinct threadgroup planes and an atomic read-modify-write is a side
    effect, so no value analysis can remove one.

    **Why this arm needs no sink.** The atomic arm needed one because it had
    removed every side effect from the loop body. This arm has removed none:
    two or three shared atomics per visit are unremovable, the trip count is
    driven by `tile_begin`, `tile_end` and `block_dim.x` which the compiler
    cannot bound, and the rolling counter is a loop-carried dependence of the
    atomics' own addresses. The loop survives because the atomics survive.

    **How the non-elimination was established for this arm.** The same method
    as the atomic arm's, and the same standard. A reduced replica -- the
    rolling counter, the `lift + b` index, the fetch-adds on three arrays,
    one caller whose bounds all come from `argv` so none of them folds -- was
    compiled to optimized target assembly and read. At `GROUP = 4` and
    `U = 4` the body holds exactly **48 `ldaddal`**, which is four slots by
    four rows by three atomics and not one more; **16 `csinc`**, one wrap per
    visit, each preceded by its `add` and `cmp`; and 16 `sbfiz`, one index
    widening per visit. Every instruction the arm is supposed to issue is
    there and the ones it dropped are not: the body contains no gather.
    **The negative control:** with the three fetch-adds replaced by nothing
    and no other change, the counter, the wrap, the index arithmetic and all
    three pointer arguments vanish -- the symbol is even renamed
    `_REMOVED_ARG` because the pointers were dropped from the signature --
    leaving the bare `while j < tile_end: j += U * stride` skeleton that a
    `@no_inline` on the replica forces to remain, and `ret`. In this file
    there is no such attribute and nothing would remain at all. That is what
    shows the check can fail, and it shows the specific thing this arm relies
    on: the loop's survival is bought by the atomics and by nothing else.

    **Which way this arm's number errs, and it is not one-signed.** Two
    terms, in opposite directions:

    1. **Upward, on the traffic side.** The rolling counter's `add`, `cmp`
       and `csinc` replace one byte gather; the arm therefore pays three
       ALU operations per visit that the full kernel does not, which lands in
       the arm's own time and makes the traffic's measured share come out
       *smaller*. This term makes the number a lower bound.
    2. **Downward, on the contention side.** Consecutive visits from one
       thread hit consecutive bins rather than random ones, which is a
       friendlier bank pattern than uniform random bins in the tail and is
       identical to it in the head. If the device's shared atomic unit is
       sensitive to that, the arm is faster than a traffic-free kernel with
       the real address stream, and the traffic's measured share comes out
       *larger*. This term makes the number an upper bound.

    Term one is bounded and small -- three ALU operations against three
    shared atomics and a global gather. Term two is unbounded by anything
    written down here. So the honest reading is: **the number brackets the
    memory-traffic share, and if it is large the confound to rule out first
    is term two**, which is done by re-running the arm at a lower `n_bins`,
    where consecutive-bin and random-bin address streams converge because
    both fit in fewer banks.

    **`probe_mode == HIST_PROBE_NO_ATOMICS` builds a histogram that is wrong
    on purpose.** It is
    not an optimization, it is not selectable by any shipping path, and it
    must never be read as one. What it does is skip the three
    `Atomic.fetch_add` calls at the bottom of this function and nothing else,
    accumulating instead into a per-thread `sink` that
    `_hist_accumulate_rows` writes to threadgroup memory once at the end of
    the walk. The only thing it exists to produce is a ratio: what share of
    this loop's cost the three shared atomics per (row, feature) are, against
    everything that feeds them. `GpuActiveRows.set_histogram_atomic_probe`
    is the only way to reach it and says what the number does and does not
    license.

    What the probe keeps, and this is the whole difficulty of building one:

    - stage one, the row index load out of `rows`;
    - stage two, the quantized gradient pair load out of `gq` (or the two
      Float32 plane gathers and their `Int32(round(x * scale))`);
    - stage three, the bin byte gather out of `bins`, at whichever of the
      three address forms `narrow` and `rstride` select;
    - the shared-cell index `lift + bv[u]`, formed exactly as the atomic arm
      forms it.

    What it drops: the three shared atomics, and with them all threadgroup
    contention and the atomic unit's own latency. It also drops the third
    atomic's constant `+1`, which is not a load and not an address, so
    nothing data-dependent leaves with it.

    What it adds, and it is not nothing: one XOR and one or two integer adds
    per (row, feature) to fold `s`, `gv` and `hv` into the sink. Those are
    ALU operations against the atomics they replace, so the probe arm is not
    a floor of "the loop without atomics" but a floor plus a few cheap
    instructions. That is the direction the error runs and it is the safe
    direction: it makes the atomics' measured share an **underestimate**.

    **Why the sink is shaped this way.** A compiler deletes work whose result
    is unused, so a probe that skipped the atomics and wrote nothing would
    measure an empty loop and report a flattering number. The sink is
    `acc = (acc ^ s) + gv (+ hv)` rather than a plain sum for one specific
    reason: `lift` is loop-invariant, so a sum of `lift + bv[u]` over `u`
    would let the compiler hoist `U * lift` out and drop the per-visit add
    that forms the shared-cell index. XOR does not distribute over the add,
    so the address arithmetic has to happen where the atomic arm makes it
    happen.

    **How the non-elimination was established, and it was not by assertion.**
    A reduced replica of this loop -- the same three loads, the same
    `lift + bin` index, the same XOR-and-add sink, one terminal store -- was
    compiled to optimized target assembly with `mojo build --emit asm` and
    read. With the terminal store present the loop survives whole, and every
    load in it is visible: `ldrsw` for the row index, `ldrb` for the bin
    byte, `ldp w, w` for the 8-byte gradient pair as the one paired load the
    alignment argument above claims, `add w, w, w` for `lift + bin`, then the
    `eor`/`add` sink chain and `str` at the exit. With the terminal store
    deleted and nothing else changed, the entire function compiles to a
    single `ret`: no loop, no loads, no arithmetic. That negative control is
    what gives the positive one teeth -- it shows the check can fail -- and
    it is the same standard by which the `align 4` fact above was
    established. The replica is host arm64 rather than Metal, because this
    repository has no dump path for a Metal kernel that is compiled at
    launch; dead-code elimination of a value with no reachable use is an
    LLVM middle-end transform common to both, and the claim is scoped to
    that.
    """
    # Arm three, and it leaves before stage one because stage one is a load.
    # Everything below this block is skipped; everything the flush does after
    # the walk is untouched. `probe_mode` is a launch argument and therefore
    # block-uniform, so this is one scalar test per step and never a
    # divergence. See the docstring for what is kept, what the rolling
    # counter is for, and which way the number errs.
    if probe_mode == HIST_PROBE_NO_GATHER:
        var b = pbin[unsafe_offset=0][0]
        var gsyn = pbin[unsafe_offset=1][0]
        var nb32 = Int32(nb)
        comptime for k in range(GROUP):
            if k < owned:
                var lift = Int32(k * nb)
                comptime for u in range(U):
                    # Wrap by compare-and-subtract rather than by modulo: a
                    # divide per visit would be a bigger charge than the
                    # gather this arm exists to remove. `b` is in `[0, nb)`
                    # on entry, so one add of one leaves it in `[0, nb]` and
                    # a single conditional reset restores the range.
                    b += Int32(1)
                    if b >= nb32:
                        b = Int32(0)
                    var s = Int(lift + b)
                    _ = Atomic.fetch_add(sg.unsafe_offset(s), gsyn)
                    comptime if not CELIDE:
                        _ = Atomic.fetch_add(sh.unsafe_offset(s), gsyn)
                    _ = Atomic.fetch_add(sc.unsafe_offset(s), Int32(1))
        pbin[unsafe_offset=0] = b
        return

    # Stage one: the row indices, in the width the permutation holds them.
    # Int32 on both arms: `rows` is an Int32 buffer and `MAX_ROWS` is
    # `Int32.MAX`, so the value fits by the same check that admits the fit at
    # all, and the wide arm widens at the point of use exactly as the load
    # itself used to.
    comptime for u in range(U):
        rr[unsafe_offset=u] = rows[
            unsafe_offset = row_begin + j + u * stride
        ][0]

    # Stage two: the quantized gradient (and hessian) for each of them.
    #
    # `narrow` and `aligned` are block-uniform launch arguments, so each of
    # the branches below is one scalar test per unrolled step and never a
    # divergence. They are runtime branches rather than comptime parameters
    # on purpose: this family is already forty instantiations, each of which
    # inlines this body twice, and two more comptime flags would be a
    # four-fold compile-time cost on every build on every backend. Branching
    # once per `U` rows against `U * GROUP * 3` shared atomics is not
    # measurable; four times the instantiations is.
    comptime if QUANT:
        # The packed arm's view of the very same allocation. A pointer
        # reinterpretation costs nothing and is hoisted out of the loop by
        # any optimizer; it is spelled once here rather than inside each of
        # the four `packed` branches so there is one place the width changes.
        # `gq16` is dereferenced only under `packed`, which is set only when
        # `_ensure_quantized` filled the buffer through
        # `_quantize_grad_hess_i16_kernel` and its bound check passed.
        var gq16 = gq.unsafe_bitcast[Int16]()
        comptime if CELIDE:
            if packed:
                if narrow:
                    comptime for u in range(U):
                        gv[unsafe_offset=u] = gq16[
                            unsafe_offset = Int(
                                Int32(2) * rr[unsafe_offset=u][0]
                            )
                        ][0].cast[DType.int32]()
                else:
                    comptime for u in range(U):
                        gv[unsafe_offset=u] = gq16[
                            unsafe_offset = 2 * Int(rr[unsafe_offset=u][0])
                        ][0].cast[DType.int32]()
            elif narrow:
                comptime for u in range(U):
                    gv[unsafe_offset=u] = gq[
                        unsafe_offset = Int(Int32(2) * rr[unsafe_offset=u][0])
                    ][0]
            else:
                comptime for u in range(U):
                    gv[unsafe_offset=u] = gq[
                        unsafe_offset = 2 * Int(rr[unsafe_offset=u][0])
                    ][0]
        else:
            if packed:
                # Four bytes for the pair where the Int32 arm reads eight,
                # at the pair's own alignment. `alignment=4` and not the
                # default for the reason the docstring's alignment paragraph
                # gives: the unannotated `width=2` load is emitted at the
                # element alignment, which a backend is free to split back
                # into two scalar loads and which would give the packing back.
                if narrow:
                    comptime for u in range(U):
                        var pair16 = gq16.unsafe_load[width=2, alignment=4](
                            Int(Int32(2) * rr[unsafe_offset=u][0])
                        )
                        gv[unsafe_offset=u] = pair16[0].cast[DType.int32]()
                        hv[unsafe_offset=u] = pair16[1].cast[DType.int32]()
                else:
                    comptime for u in range(U):
                        var pair16 = gq16.unsafe_load[width=2, alignment=4](
                            2 * Int(rr[unsafe_offset=u][0])
                        )
                        gv[unsafe_offset=u] = pair16[0].cast[DType.int32]()
                        hv[unsafe_offset=u] = pair16[1].cast[DType.int32]()
            elif narrow:
                if aligned:
                    comptime for u in range(U):
                        var pair = gq.unsafe_load[width=2, alignment=8](
                            Int(Int32(2) * rr[unsafe_offset=u][0])
                        )
                        gv[unsafe_offset=u] = pair[0]
                        hv[unsafe_offset=u] = pair[1]
                else:
                    comptime for u in range(U):
                        var pair = gq.unsafe_load[width=2](
                            Int(Int32(2) * rr[unsafe_offset=u][0])
                        )
                        gv[unsafe_offset=u] = pair[0]
                        hv[unsafe_offset=u] = pair[1]
            elif aligned:
                comptime for u in range(U):
                    var pair = gq.unsafe_load[width=2, alignment=8](
                        2 * Int(rr[unsafe_offset=u][0])
                    )
                    gv[unsafe_offset=u] = pair[0]
                    hv[unsafe_offset=u] = pair[1]
            else:
                comptime for u in range(U):
                    var pair = gq.unsafe_load[width=2](
                        2 * Int(rr[unsafe_offset=u][0])
                    )
                    gv[unsafe_offset=u] = pair[0]
                    hv[unsafe_offset=u] = pair[1]
    else:
        # The Float32 arm gathers two separate planes by row id. There is no
        # pair to widen and no product to narrow: the index is the row, which
        # fits Int32 on every fit this module admits, so neither arm applies
        # and the expression is written once. It is reproduced character for
        # character from the loop it replaced, for the contraction reason
        # `docs/NUMERICS.md` section 5.6 records.
        comptime for u in range(U):
            gv[unsafe_offset=u] = Int32(
                round(
                    grad[unsafe_offset = Int(rr[unsafe_offset=u][0])][0]
                    * g_scale
                )
            )
            comptime if not CELIDE:
                hv[unsafe_offset=u] = Int32(
                    round(
                        hess[unsafe_offset = Int(rr[unsafe_offset=u][0])][0]
                        * h_scale
                    )
                )

    # Stages three and four, one owned slot at a time. The narrow branch is
    # taken once per owned slot per step, and only the `U` address
    # computations are written twice; the atomics that follow are one copy,
    # because they do not depend on how the bin byte's address was formed.
    comptime for k in range(GROUP):
        if k < owned:
            var base = col[unsafe_offset=k]
            var lift = Int32(k * nb)
            # The packed arm. `bw[k]` is this slot's storage width in bits and
            # is `PACKED_WIDTH_OFF` (zero) unless `set_packed_bins` put a
            # bit-packed matrix on the device; a width of 8 is packed *and*
            # takes the byte gather below, because a width-8 stream is the
            # feature's own column displaced to an aligned base and the second
            # window byte is provably never needed there. So this branch is
            # entered only at widths 1 through 7, which are exactly the
            # features that are narrower than a byte.
            #
            # `w`, `mask` and `base` are all per owned slot -- at most sixteen
            # per threadgroup -- and never per row. What the row walk pays for
            # a narrow feature is one multiply, one shift, one mask and a
            # second adjacent byte load; what it saves is `(8 - w)/8` of the
            # column's bytes.
            var w = bw[unsafe_offset=k]
            if w != PACKED_WIDTH_OFF and w != 8:
                var mask = Int32((1 << w) - 1)
                comptime for u in range(U):
                    # `gpu_bin_packing.element_byte_offset` and
                    # `element_bit_shift` written out, because a kernel cannot
                    # call a raising host function.
                    # `gpu_packed_bins.check_packed_matches_plan` is what
                    # proves these two lines are the plan's own addresses.
                    #
                    # Formed in Int throughout: `r * w` reaches `8 * n_rows`
                    # and the `narrow` arm's Int32 bound was derived for
                    # `f * n_rows`, not for this product. The packed arm
                    # therefore has no narrow twin, and the address it forms
                    # is bounded by `packed_bytes`, which the host already
                    # checked against `Int32.MAX`.
                    var bit = Int(rr[unsafe_offset=u][0]) * w
                    var at = base + (bit >> 3)
                    var window = Int32(Int(bins[unsafe_offset=at][0])) | (
                        Int32(Int(bins[unsafe_offset = at + 1][0])) << 8
                    )
                    bv[unsafe_offset=u] = (
                        window >> Int32(bit & 7)
                    ) & mask
            elif rstride != BLOCKED_STRIDE_NONE:
                # The blocked layout. `base` is the feature's block base plus
                # its lane, computed once per slot in `_hist_accumulate_rows`;
                # a row advances `rstride` bytes because the block is
                # row-major. The multiply replaces nothing -- the
                # feature-major arm's implicit stride is one -- so it is one
                # extra integer multiply per (row, feature) against the
                # sectors the arrangement is meant to save. If the layout ever
                # loses, that multiply is the first thing to suspect and
                # `rstride` being a power of two is what makes it a shift.
                comptime for u in range(U):
                    bv[unsafe_offset=u] = Int32(
                        Int(
                            bins[
                                unsafe_offset = base
                                + Int(rr[unsafe_offset=u][0]) * rstride
                            ]
                        )
                    )
            elif narrow:
                # Exact under the precondition: `base` is
                # `feature * n_rows`, which the host proved below
                # `Int32.MAX`, so this truncation loses nothing.
                var base32 = Int32(base)
                comptime for u in range(U):
                    bv[unsafe_offset=u] = Int32(
                        Int(
                            bins[
                                unsafe_offset = Int(
                                    base32 + rr[unsafe_offset=u][0]
                                )
                            ]
                        )
                    )
            else:
                comptime for u in range(U):
                    bv[unsafe_offset=u] = Int32(
                        Int(
                            bins[
                                unsafe_offset = base
                                + Int(rr[unsafe_offset=u][0])
                            ]
                        )
                    )
            if probe_mode == HIST_PROBE_NO_ATOMICS:
                # The probe arm. Everything above this line has already run;
                # what is skipped is exactly the three atomics below it. The
                # index `s` is formed here in the same expression the atomic
                # arm forms it in, and the XOR is what stops the compiler
                # hoisting `lift` out of it. See the docstring.
                comptime for u in range(U):
                    var s = lift + bv[unsafe_offset=u][0]
                    var acc = sink[unsafe_offset=0][0]
                    acc = (acc ^ s) + gv[unsafe_offset=u][0]
                    comptime if not CELIDE:
                        acc += hv[unsafe_offset=u][0]
                    sink[unsafe_offset=0] = acc
            else:
                comptime for u in range(U):
                    var s = Int(lift + bv[unsafe_offset=u][0])
                    _ = Atomic.fetch_add(
                        sg.unsafe_offset(s), gv[unsafe_offset=u][0]
                    )
                    comptime if not CELIDE:
                        _ = Atomic.fetch_add(
                            sh.unsafe_offset(s), hv[unsafe_offset=u][0]
                        )
                    _ = Atomic.fetch_add(sc.unsafe_offset(s), Int32(1))


@always_inline
def _hist_accumulate_rows[
    GROUP: Int, UNROLL: Int, CELIDE: Bool, QUANT: Bool
](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    sg: _SharedI32,
    sh: _SharedI32,
    sc: _SharedI32,
    slot0: Int,
    owned: Int,
    nb: Int,
    nr: Int,
    row_begin: Int,
    tile_begin: Int,
    tile_end: Int,
    g_scale: Float32,
    h_scale: Float32,
    unrolled: Bool,
    narrow: Bool,
    aligned: Bool,
    packed: Bool,
    rstride: Int,
    probe_mode: Int,
    ptab: MutPointer[Int32, MutAnyOrigin],
    packed_bins: Bool,
):
    """One thread's whole walk over its share of one tile.

    This is the row loop both `_range_hist_atomic_kernel` and
    `_range_hist_partial_kernel` run, factored out so there is one copy of it
    rather than eight. The two kernels differ in what they do with the shared
    planes afterwards, not in how they fill them, and the four arms each of
    them used to write out longhand (`const_hess` by `use_quant`) are the two
    comptime flags here. Selecting them at compile time rather than
    branching per row is what the two longhand loops bought and it is kept:
    the caller picks one of four instantiations from two block-uniform
    runtime flags, once, outside the loop.

    **The walk.** Thread `tid` owns rows `tile_begin + tid`,
    `+ block_dim.x`, `+ 2 * block_dim.x`, ... of the range, up to
    `tile_end`. That set is unchanged from the one-row-per-iteration loop
    this replaces. The main loop takes `UNROLL` of them per iteration while
    a full group remains, and the tail loop takes the rest one at a time, so
    the union of the two is exactly the arithmetic progression above with no
    row visited twice and none dropped. The tail is the same
    `_hist_rows_step` at `U = 1`, so the two cannot drift apart under a later
    edit.

    **`unrolled`.** Setting it false collapses the main loop's bound to the
    thread's first row, so the loop body never runs and the tail loop does
    the whole walk one row at a time. That is precisely the shape this
    module shipped before this lane, reachable at run time and therefore
    interleavable inside one process, which is the only protocol that
    compares anything on a machine whose device timings drift several-fold
    between time windows. It costs one block-uniform branch outside the loop
    and no second kernel instantiation, which matters: this family is already
    forty instantiations and the const-hessian flag was left a runtime flag
    for exactly that reason.

    Both arms produce the identical histogram, for the reason `_hist_rows_step`
    argues in full, so this is a launch-shape knob and never a numeric one.

    **`narrow` and `aligned`.** Two further block-uniform arms, passed
    straight through to `_hist_rows_step`, which is where both are argued.
    They are runtime flags for the same reason `unrolled` and `const_hess`
    are: this family is forty kernel instantiations and each additional
    comptime flag doubles the row-loop code every build on every backend
    compiles. Neither changes a histogram; `narrow` carries a precondition
    on the dataset shape that `GpuActiveRows` checks on the host before it
    can ever be set.

    **What is claimed, and what is not.** Claimed, by counting: the main loop
    executes one loop test and one induction add per `UNROLL` rows instead of
    per row, and the quantized non-elided arm issues one 8-byte load per row
    where the loop it replaces issued two 4-byte loads. Not claimed: any
    speedup. The unroll raises the number of live registers in the row loop
    by roughly `UNROLL` times the values a single row needs, and threadgroup
    residency on this backend is bounded by things this project cannot query.
    If the unroll loses, that is how it loses, and `set_row_unroll(False)` is
    the arm that shows it.

    **`probe_mode`, and every value of it above zero produces a wrong
    histogram on purpose.** Passed
    straight through to `_hist_rows_step`, which is where it is argued. The
    one part of it that lives here is the sink itself: one Int32 per thread,
    zeroed before the walk and written to threadgroup memory once after it,
    which is what keeps the gather and the address arithmetic from being
    deleted as work whose result nobody reads.

    The gather-free arm's two-word scratch lives here for the same reason and
    is not a sink: it is the rolling bin counter and the synthetic gradient,
    both seeded once per thread outside the walk so that no part of their
    setup is charged per row. The seed is `tid % nb`, which spreads a warp's
    threads over `nb` consecutive bins and is the one division either arm
    pays, once. It is computed unconditionally on all four arms, because a
    `stack_allocation` under a runtime `if` is not a thing this loop can
    have and two Int32 of stack the compiler promotes to registers is not a
    cost worth branching over. `HIST_PROBE_EMPTY` never reaches this
    function at all: its kernel returns before the walk.

    The store is unconditional and every thread makes it, to the same two
    cells. That races, and the value left there is whichever thread wrote
    last; both facts are deliberate. A store guarded by `tid == 0` would let
    a compiler sink the whole loop into the guard and leave the probe timing
    one thread's work instead of the block's, and an atomic store would put
    back a shared atomic this arm exists to remove. A racy garbage cell is
    the right answer in a kernel whose entire output is already garbage, and
    it is two plain threadgroup stores per thread per tile against the
    millions of atomics they stand in for. Why two, and why one of them is
    `sc`, is at the store itself.
    """
    var tid = Int(thread_idx.x)
    var stride = Int(block_dim.x)

    # One column base per owned slot, read once and spent for every row.
    # The unowned tail entries are zeroed rather than left undefined so that
    # a comptime-unrolled body which computes an address it never uses --
    # which is what `if k < owned` leaves the compiler free to do -- cannot
    # form one out of stack garbage.
    #
    # Under the blocked layout the base is the feature's block base plus its
    # lane, `(f / G) * block_stride + (f mod G)`, which is
    # `gpu_blocked_bins.blocked_column_base` written out. The block stride is
    # derived here rather than passed in so the launch keeps one new argument
    # instead of two; `BLOCK_ALIGN_BYTES` is imported rather than spelled 16,
    # so the alignment has one definition and it is `BinLayoutPlan`'s.
    #
    # The division and the remainder are per owned slot -- at most `GROUP`,
    # so at most sixteen per threadgroup -- and never per row. That is the
    # same footing the feature-major multiply is on and is why the blocked
    # arm adds nothing to the row walk except the `r * rstride` term.
    # Under the packed layout the base and the width both come out of the
    # device table `gpu_packed_bins.packed_table` built: `ptab[2f]` is the
    # feature's stream base and `ptab[2f + 1]` its width in bits. Two reads per
    # owned slot, on the same footing as the feature-major multiply, and
    # nothing per row. `bw` stays `PACKED_WIDTH_OFF` on every other arm, which
    # is what makes the row loop's packed branch a per-slot test against a
    # value that is a compile-time-invisible zero for every existing caller.
    #
    # The packed arm and the blocked arm are mutually exclusive and
    # `set_packed_bins` refuses to have both on; the branch order here states
    # it rather than assuming it.
    var col = stack_allocation[GROUP, Int]()
    var bw = stack_allocation[GROUP, Int]()
    var bstride = 0
    if rstride != BLOCKED_STRIDE_NONE:
        bstride = (
            (nr * rstride + BLOCK_ALIGN_BYTES - 1) // BLOCK_ALIGN_BYTES
        ) * BLOCK_ALIGN_BYTES
    comptime for k in range(GROUP):
        col[unsafe_offset=k] = 0
        bw[unsafe_offset=k] = PACKED_WIDTH_OFF
        if k < owned:
            var fid_k = Int(feat_ids[unsafe_offset = slot0 + k][0])
            if packed_bins:
                col[unsafe_offset=k] = Int(
                    ptab[unsafe_offset = PACKED_TABLE_STRIDE * fid_k][0]
                )
                bw[unsafe_offset=k] = Int(
                    ptab[unsafe_offset = PACKED_TABLE_STRIDE * fid_k + 1][0]
                )
            elif rstride != BLOCKED_STRIDE_NONE:
                col[unsafe_offset=k] = (fid_k // rstride) * bstride + (
                    fid_k % rstride
                )
            else:
                col[unsafe_offset=k] = fid_k * nr

    # The rows in flight and what was gathered for them. Allocated once,
    # outside both loops, so there is one alloca per thread and not one per
    # iteration.
    #
    # The rows in flight are held Int32, which is the width `rows` stores
    # them in and the width `MAX_ROWS` bounds them to. Holding them Int cost
    # `UNROLL` 64-bit registers to carry values that had just been widened
    # from 32 bits, and the wide index arm widens at the point of use
    # instead, which is where the widening happened before the load was
    # hoisted into this array.
    var rr = stack_allocation[UNROLL, Scalar[DType.int32]]()
    var gv = stack_allocation[UNROLL, Scalar[DType.int32]]()
    var hv = stack_allocation[UNROLL, Scalar[DType.int32]]()
    var bv = stack_allocation[UNROLL, Scalar[DType.int32]]()

    # The probe arm's per-thread sink. Allocated on both arms because a
    # `stack_allocation` under a runtime `if` is not a thing this loop can
    # have, and it costs one Int32 of stack the compiler promotes to a
    # register; the atomic arm never reads or writes it.
    var sink = stack_allocation[1, Scalar[DType.int32]]()
    sink[unsafe_offset=0] = 0

    # The gather-free arm's scratch: [0] the rolling bin, [1] the synthetic
    # quantized gradient. Both are per-thread and both are set once here
    # rather than per row, so the arm's per-visit cost is the counter's three
    # ALU operations and nothing else. `nb` is at least one on every launch
    # this module makes -- `__init__` refuses a bin count below one -- and the
    # guard is here anyway because a modulo by zero on a device is not an
    # error anybody gets to see.
    var pbin = stack_allocation[2, Scalar[DType.int32]]()
    pbin[unsafe_offset=0] = Int32(tid % nb) if nb > 0 else Int32(0)
    pbin[unsafe_offset=1] = Int32(tid & 7) - Int32(3)

    var j = tile_begin + tid
    # `limit` is where the last full group of `UNROLL` rows can start. Left
    # at `j` when the unroll is off or unavailable, which skips the main
    # loop entirely without a second code path.
    var limit = j
    comptime if UNROLL > 1:
        if unrolled:
            limit = tile_end - (UNROLL - 1) * stride
    while j < limit:
        _hist_rows_step[GROUP, UNROLL, CELIDE, QUANT](
            bins,
            rows,
            grad,
            hess,
            gq,
            sg,
            sh,
            sc,
            col,
            bw,
            rr,
            gv,
            hv,
            bv,
            sink,
            pbin,
            owned,
            nb,
            row_begin,
            j,
            stride,
            g_scale,
            h_scale,
            narrow,
            aligned,
            packed,
            rstride,
            probe_mode,
        )
        j += UNROLL * stride
    while j < tile_end:
        _hist_rows_step[GROUP, 1, CELIDE, QUANT](
            bins,
            rows,
            grad,
            hess,
            gq,
            sg,
            sh,
            sc,
            col,
            bw,
            rr,
            gv,
            hv,
            bv,
            sink,
            pbin,
            owned,
            nb,
            row_begin,
            j,
            stride,
            g_scale,
            h_scale,
            narrow,
            aligned,
            packed,
            rstride,
            probe_mode,
        )
        j += stride

    # The sink's one visible effect, and the only reason the probe arm's
    # loads survive optimization. Unconditional, racing, and garbage; see the
    # docstring for why each of those three is the right choice here.
    #
    # Two cells rather than one, and the second is the load-bearing one. The
    # atomic kernel's flush reads `sg[s]` only under `if sc[s] != 0`, so a
    # sink written to `sg` alone is a store whose value a sufficiently clever
    # compiler could try to prove unread. Writing `sc[0]` as well puts the
    # sink into the condition that guards three global atomics, which is a
    # use no analysis can discharge, and it makes the probe's wrongness
    # visible in the output rather than silent: bin 0 of the first owned slot
    # comes out as whatever the race left there.
    if probe_mode == HIST_PROBE_NO_ATOMICS:
        sg[unsafe_offset=0] = sink[unsafe_offset=0][0]
        sc[unsafe_offset=0] = sink[unsafe_offset=0][0]


@always_inline
def _hist_accumulate_dispatch[
    GROUP: Int, UNROLL: Int
](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    sg: _SharedI32,
    sh: _SharedI32,
    sc: _SharedI32,
    slot0: Int,
    owned: Int,
    nb: Int,
    nr: Int,
    row_begin: Int,
    tile_begin: Int,
    tile_end: Int,
    g_scale: Float32,
    h_scale: Float32,
    celide: Bool,
    quant: Bool,
    unrolled: Bool,
    narrow: Bool,
    aligned: Bool,
    packed: Bool,
    rstride: Int,
    probe_mode: Int,
    ptab: MutPointer[Int32, MutAnyOrigin],
    packed_bins: Bool,
):
    """Resolve the two comptime-worthy runtime flags into one of four arms,
    once, above the row loop.

    Every flag here is a launch argument and therefore block-uniform, so this
    is a scalar branch taken once per threadgroup and not a divergence. It
    exists so that the four arms are written once here instead of once in
    each of the two kernels.

    Only `celide` and `quant` become comptime parameters. `unrolled`,
    `narrow`, `aligned`, and `probe_mode` stay runtime values and are passed
    through, because each one that became a parameter would double a row-loop
    body that is already inlined twice into forty kernel instantiations.
    Where they are consumed, and why each is cheap as a branch, is argued at
    `_hist_accumulate_rows` and `_hist_rows_step`.

    `probe_mode` is the odd one and is not a launch shape: it is the only
    flag here that changes the answer, and every nonzero value of it changes
    the answer to a wrong one on purpose. Zero is the shipping arm; the two
    values that reach this function are `HIST_PROBE_NO_ATOMICS` and
    `HIST_PROBE_NO_GATHER`, and `HIST_PROBE_EMPTY` never gets here because
    its kernel returns before the walk. `GpuActiveRows`'s two acknowledged
    setters are the only way any of them can be turned on, all are off by
    default, and two of that struct's histogram entry points refuse to launch
    while any is set."""
    if celide:
        if quant:
            _hist_accumulate_rows[GROUP, UNROLL, True, True](
                bins,
                rows,
                grad,
                hess,
                gq,
                feat_ids,
                sg,
                sh,
                sc,
                slot0,
                owned,
                nb,
                nr,
                row_begin,
                tile_begin,
                tile_end,
                g_scale,
                h_scale,
                unrolled,
                narrow,
                aligned,
                packed,
                rstride,
                probe_mode,
                ptab,
                packed_bins,
            )
        else:
            _hist_accumulate_rows[GROUP, UNROLL, True, False](
                bins,
                rows,
                grad,
                hess,
                gq,
                feat_ids,
                sg,
                sh,
                sc,
                slot0,
                owned,
                nb,
                nr,
                row_begin,
                tile_begin,
                tile_end,
                g_scale,
                h_scale,
                unrolled,
                narrow,
                aligned,
                packed,
                rstride,
                probe_mode,
                ptab,
                packed_bins,
            )
    elif quant:
        _hist_accumulate_rows[GROUP, UNROLL, False, True](
            bins,
            rows,
            grad,
            hess,
            gq,
            feat_ids,
            sg,
            sh,
            sc,
            slot0,
            owned,
            nb,
            nr,
            row_begin,
            tile_begin,
            tile_end,
            g_scale,
            h_scale,
            unrolled,
            narrow,
            aligned,
            packed,
            rstride,
            probe_mode,
            ptab,
            packed_bins,
        )
    else:
        _hist_accumulate_rows[GROUP, UNROLL, False, False](
            bins,
            rows,
            grad,
            hess,
            gq,
            feat_ids,
            sg,
            sh,
            sc,
            slot0,
            owned,
            nb,
            nr,
            row_begin,
            tile_begin,
            tile_end,
            g_scale,
            h_scale,
            unrolled,
            narrow,
            aligned,
            packed,
            rstride,
            probe_mode,
            ptab,
            packed_bins,
        )


def _range_hist_atomic_kernel[
    GROUP: Int, BIN_CAP: Int
](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
    rows_per_tile: Int32,
    begin: Int32,
    count: Int32,
    g_scale: Float32,
    h_scale: Float32,
    sub_offset: Int32,
    do_sub: Int32,
    use_quant: Int32,
    const_hess: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
    row_unroll: Int32,
    narrow_index: Int32,
    pair_align: Int32,
    bins_blocked: MutPointer[UInt8, MutAnyOrigin],
    bin_row_stride: Int32,
    hist_probe: Int32,
    bins_packed: MutPointer[UInt8, MutAnyOrigin],
    bin_pack_tab: MutPointer[Int32, MutAnyOrigin],
    bin_packed: Int32,
):
    """One node's histogram over a compacted row range, accumulated in
    threadgroup memory and folded into the output with global atomics.

    One kernel for the whole family. The hand-written one-feature and
    two-feature atomic kernels this replaces differed in exactly two things,
    how many feature slots a block owned and how wide its shared planes were,
    and both are now comptime parameters: `GROUP` walks the ladder 1, 2, 4, 8,
    16 and `BIN_CAP` walks 32, 64, 128, 256. `enqueue_range_histogram` picks
    the pair, taking `BIN_CAP` from `gpu_tiling.histogram_bin_capacity` and
    the group from the free-footprint rule in
    `gpu_tiling.free_feature_group`.

    **What the capacity parameter fixes.** The old kernels allocated three
    `MAX_BINS`-wide Int32 planes whatever the dataset's bin count was, so a
    64-bin dataset paid four times the threadgroup memory it needed and a
    32-bin one eight times. The allocation here is `GROUP * BIN_CAP` per
    plane, so it is the memory the histogram occupies and nothing more.
    Threadgroup memory is what bounds resident blocks per core, so this is a
    residency change and not a bandwidth one, which is why the effect it has
    on wall clock is a per-device measurement rather than an argument. It is
    UNMEASURED at every capacity below 256 on every device this project runs
    on. What would measure it is an interleaved A/B in one process of one
    shape binned at 64 against the same shape binned at 256 with the same row
    count, which is not the comparison `bench_histogram.mojo` makes today.

    **What the group parameter buys.** With one feature per block,
    `rows[begin + j]` and the row's quantized gradient pair are read once per
    feature. A block owning `GROUP` slots reads them once and spends them
    `GROUP` times, which is the `pair_features` walk `histogram.mojo` already
    does on the CPU, lifted to a threadgroup and widened. Measured for the
    tiled twin below on an Apple M4 at 1.17x for group 2 over group 1 on the
    atomic path (`pixi run bench-hist 100000 100 20`, arms interleaved in one
    process because this machine's device timings drift several-fold between
    time windows). Groups 8 and 16 have never been launched on any device, so
    nothing is claimed for them.

    **Exactness.** Every instantiation produces the identical integer
    histogram, and this is structural rather than a tolerance. Accumulation is
    fixed-point Int32 throughout, integer addition is associative and
    commutative, and every atomic adds the same per-(row, feature) quantized
    value into the same bin whatever the block shape, so only the order of the
    adds differs between instantiations and the order of integer adds cannot
    change a sum. `GROUP` and `BIN_CAP` change where a partial lives, never
    what goes into it.

    **Tail blocks.** A block whose `slot0 + GROUP` overruns `n_slots` owns
    only the slots that remain. It zeroes only its own span, accumulates only
    into its own slices, and flushes only those, so a feature is never counted
    twice and never dropped. The owned slices are stacked at `k * n_bins`
    inside the allocation, so the stacking is bounded by
    `GROUP * n_bins <= GROUP * BIN_CAP` and the zeroing walk covers exactly
    the cells the flush will read.

    **The row loop.** It lives in `_hist_accumulate_rows`, shared with
    `_range_hist_partial_kernel`, because the two kernels differ in what they
    do with the shared planes after the barrier and not at all in how they
    fill them. `use_quant` selects between the pre-quantized interleaved
    buffer `_quantize_grad_hess_kernel` writes and the two Float32 planes,
    and it does so by picking one of four comptime arms above the loop rather
    than by branching per row, so the default path spends no floating-point
    arithmetic in the row loop at all. It is a runtime flag rather than a
    third comptime parameter because doubling forty instantiations to eighty
    is compile time every build on every backend pays. The two produce
    bit-identical histograms by construction; see
    `_quantize_grad_hess_kernel`.

    `row_unroll` is a second such flag, selecting how many rows one thread
    keeps in flight; both settings visit the same rows and add the same
    integers in a different order, which cannot change a sum. See
    `_hist_accumulate_rows` and `_hist_rows_step` for the argument in full.

    `narrow_index` and `pair_align` are two more, and they are launch
    arguments for the same compile-time reason. `narrow_index` selects the
    width of the two data-dependent index computations in the row loop and is
    exact under a bound on the dataset shape that
    `GpuActiveRows.narrow_index_supported` checks on the host before the flag
    can be set. `pair_align` selects whether the width-2 load of the
    quantized gradient pair states the alignment its address has; both
    spellings read the same eight bytes. Neither reaches the flush, the
    subtraction, or any floating-point expression. See `_hist_rows_step`.

    **Fused subtraction.** With `do_sub`, the flush also subtracts what it
    added from the slot `sub_offset` words away, which is the sibling
    subtraction folded into the build. The destination is a signed offset
    rather than a second pointer because both slots live in one pool buffer,
    and two pointers into one buffer is an aliasing the compiler is right to
    refuse. It is the same arithmetic the standalone
    `gpu_leaf_batching._subtract_slice_kernel` performs and exact for the same
    reason, both slots being fixed-point Int32
    under one scale, so a parent's bins are the exact integer sum of its
    children's. Bins this node never touched are left alone, which is what
    subtracting zero from them would have done anyway.

    **The elided hessian plane.** With `const_hess` the caller has declared
    that every row's hessian is exactly `histogram.CONSTANT_HESSIAN`, which
    four of the built-in objectives guarantee when the fit carries no sample
    weights. The row loop then performs two shared-memory atomics per (row,
    feature) instead of three, the shared zeroing covers two planes instead
    of three, and the flush writes the global hessian plane as
    `hq_const * vc` instead of reading a third shared plane.

    That reconstruction is the exact integer the three-plane path would have
    accumulated, and the argument is worth spelling out because everything
    rests on it. Every row contributes
    `Int32(round(hess[r] * h_scale))` to its bin. With `hess[r]` equal to
    1.0f for every row, and `1.0f * x` exactly `x` for every finite Float32
    because multiplying by one cannot round, every one of those contributions
    is the single value `Int32(round(h_scale))`, which is what `hq_const`
    below computes -- in this kernel, on this device, from the same launch
    argument, so no host-versus-device rounding claim is made or needed. A
    bin that received `vc` rows therefore holds `vc` copies of `hq_const`
    added together, and `hq_const * vc` is that sum. Int32 addition and
    multiplication agree modulo 2^32, so the two are the same bits even in
    the overflow regime neither of them ever reaches: under
    `SCALE_MAGNITUDE_SUM` the hessian scale is `2^30 / sum|h|`, so the whole
    dataset's hessian plane sums to about 2^30 and one bin is a subset of
    that.

    `const_hess` is a runtime flag and not a third comptime parameter, for
    the reason the two row loops give below: forty instantiations are already
    what every build on every backend pays for, and eighty is a compile-time
    cost this specialization has not earned. The consequence is honest and
    worth stating rather than eliding: `sh` below is a comptime
    `stack_allocation` and is still allocated on the elided path, so the
    threadgroup footprint does not shrink and **no occupancy improvement
    follows from this flag**. What follows is one third fewer shared atomics
    in the row loop and one third less shared zeroing. Making the plane count
    comptime, and getting the residency with it, is a separate change behind
    a measurement of the compile-time trade.

    The global flush still writes three planes, because the device-resident
    split search in `gpu_split_search.mojo` scans this buffer in place and
    reads all three. Eliding the plane from the device *layout* would be a
    change to that module's contract and is not this lane's; the download in
    `histogram_gpu.histogram_from_host` therefore still moves three planes as
    well.

    The descriptor arm
    ------------------
    With `use_desc`, four values move from launch arguments into device
    memory: the row window this build reads (`STEP_BUILT_BEGIN`,
    `STEP_BUILT_COUNT`), the resident pool slot it writes
    (`STEP_BUILT_SLOT`), and the slot it subtracts itself from
    (`STEP_SUB_SLOT`). `out_hist` is then the *base* of the pool rather than a
    pointer the host has already offset, and `sub_offset` is derived here as
    `(sub_slot - built_slot) * 3 * hist_size` instead of being passed in.
    Which of the two children was chosen to be built and which derived was
    decided by the commit kernel and is already baked into those four words.

    Nothing about the accumulation moves. The same rows are gathered, the
    same fixed-point products are formed, the same shared atomics run and the
    same global atomics flush. The only floating-point operations in this
    kernel are the two per-row quantizing products, which are untouched, so a
    histogram built on this arm is the one the host-argument arm would have
    built, bin for bin. That is a structural statement and not a tolerance.

    The empty-tile early return below is what makes an over-provisioned grid
    affordable. A descriptor-driven launch does not know the row count, so it
    is launched with `grid.y` sized for the whole active prefix; a block whose
    tile begins past the end of the range would otherwise zero its shared
    planes, gather nothing, and then walk every bin of the flush finding
    nothing to add. `block_idx.y` and the count are block-uniform, so the
    whole threadgroup leaves together, and the result is unchanged on both
    arms because such a block contributes nothing either way.
    """
    var slot0 = GROUP * Int(block_idx.x)
    var owned = Int(n_slots) - slot0
    if owned > GROUP:
        owned = GROUP
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var hs = Int(hist_size)
    var n = Int(count)
    var row_begin = Int(begin)
    # Where this build's own slot starts inside `out_hist`, in Int32 words.
    # Zero on the host-argument arm, because there the host offset the
    # pointer before the launch.
    var out_base = 0
    var sub_off = Int(sub_offset)
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        row_begin = Int(desc[unsafe_offset=STEP_BUILT_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_BUILT_COUNT][0])
        var cells = 3 * hs
        var built_slot = Int(desc[unsafe_offset=STEP_BUILT_SLOT][0])
        var from_slot = Int(desc[unsafe_offset=STEP_SUB_SLOT][0])
        out_base = built_slot * cells
        sub_off = (from_slot - built_slot) * cells

    # Empty tiles cost a launch slot and nothing else. See the docstring.
    var tile_begin = Int(block_idx.y) * Int(rows_per_tile)
    if tile_begin >= n:
        return
    var tile_end = tile_begin + Int(rows_per_tile)
    if tile_end > n:
        tile_end = n

    var sg = stack_allocation[
        GROUP * BIN_CAP,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh = stack_allocation[
        GROUP * BIN_CAP,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sc = stack_allocation[
        GROUP * BIN_CAP,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    # Arm four of the decomposition, and it is the whole kernel. Placed after
    # the three threadgroup declarations and before the zeroing, so that what
    # it measures is the launch, the grid and the threadgroup footprint that
    # bounds occupancy -- and not the zeroing, the barriers or the flush,
    # which are work and belong in the residual. `_hist_probe_empty_mark` is
    # where the six stores are argued and where the negative control that
    # proves they survive is recorded.
    #
    # The empty-tile early return above already ran, so an over-provisioned
    # grid leaves the same blocks early on this arm as on the shipping one:
    # the arm changes the body, never the geometry.
    var probe_mode = Int(hist_probe)
    if probe_mode == HIST_PROBE_EMPTY:
        _hist_probe_empty_mark[GROUP * BIN_CAP](
            sg,
            sh,
            sc,
            out_hist,
            Int32(thread_idx.x) ^ Int32(block_idx.x) ^ Int32(block_idx.y),
            Int(thread_idx.x),
        )
        return

    # The elided plane's one quantized value, and the flag that selects it.
    # `Float32(1.0) * h_scale` is exactly `h_scale`, so this is
    # `Int32(round(h_scale))`, computed here rather than passed in so that it
    # is the same expression evaluated by the same device compiler as the
    # per-row quantization it stands in for.
    var celide = Int(const_hess) != 0
    var hq_const = Int32(round(Float32(1.0) * h_scale))

    var span = owned * nb
    var b = tid
    while b < span:
        sg[unsafe_offset=b] = 0
        if not celide:
            sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    # One feature id per owned slot, read once and used by the flush. The
    # loop is unrolled so the array is indexed by a compile-time constant and
    # stays in registers rather than spilling to local memory. The column
    # bases the row loop needs are read the same way inside
    # `_hist_accumulate_rows`, which is the only place they are wanted.
    var fid = stack_allocation[GROUP, Int]()
    comptime for k in range(GROUP):
        fid[unsafe_offset=k] = 0
        if k < owned:
            fid[unsafe_offset=k] = Int(feat_ids[unsafe_offset = slot0 + k][0])

    # `celide` takes two atomics per (row, feature) instead of three. The
    # hessian is neither read nor accumulated: it is the same Int32 for every
    # row, so the count plane already carries everything the hessian plane
    # would have.
    # The bin matrix this launch reads, and how many bytes one row advances
    # inside a column. `bin_row_stride` of one is the feature-major buffer the
    # backend uploads, which every other reader of `bins` in this package
    # still indexes; anything else is the `[block][row][G]` relayout of
    # `gpu_blocked_bins.mojo`, which `GpuActiveRows` owns and built once per
    # fit. Two pointers rather than one because a launch may not be handed the
    # same buffer twice, and the unselected one is simply never dereferenced.
    #
    # Resolved here, once per threadgroup, so the row loop below sees one
    # pointer and one stride and never asks which layout it is on.
    var bsrc = bins
    var rstride = Int(bin_row_stride)
    if rstride != BLOCKED_STRIDE_NONE:
        bsrc = bins_blocked
    # The packed arm, resolved the same way and for the same reason: a third
    # pointer, never dereferenced unless selected, because a launch may not be
    # handed the same buffer twice. `bin_packed` and `bin_row_stride` are never
    # both set -- `set_packed_bins` and `set_blocked_layout` refuse each other
    # -- and the order here is the order that statement is enforced in.
    var packed = Int(bin_packed) != 0
    if packed:
        bsrc = bins_packed

    _hist_accumulate_dispatch[GROUP, HIST_ROW_UNROLL](
        bsrc,
        rows,
        grad,
        hess,
        gq,
        feat_ids,
        sg,
        sh,
        sc,
        slot0,
        owned,
        nb,
        nr,
        row_begin,
        tile_begin,
        tile_end,
        g_scale,
        h_scale,
        celide,
        Int(use_quant) != 0,
        Int(row_unroll) != 0,
        Int(narrow_index) != 0,
        Int(pair_align) != 0,
        Int(use_quant) == QUANT_SOURCE_PACKED16,
        rstride,
        probe_mode,
        bin_pack_tab,
        packed,
    )
    barrier()

    var sub = Int(do_sub) != 0
    comptime for k in range(GROUP):
        if k < owned:
            var base = fid[unsafe_offset=k] * nb
            var lift = k * nb
            var c = tid
            while c < nb:
                var s = lift + c
                if sc[unsafe_offset=s][0] != 0:
                    var vg = sg[unsafe_offset=s][0]
                    var vc = sc[unsafe_offset=s][0]
                    # The refill. `hq_const * vc` is the sum of `vc` copies of
                    # `hq_const`, which is precisely what the three-plane
                    # accumulation would have left in `sh`.
                    var vh = (
                        hq_const * vc if celide else sh[unsafe_offset=s][0]
                    )
                    var go = out_base + base + c
                    _ = Atomic.fetch_add(out_hist.unsafe_offset(go), vg)
                    _ = Atomic.fetch_add(
                        out_hist.unsafe_offset(hs + go), vh
                    )
                    _ = Atomic.fetch_add(
                        out_hist.unsafe_offset(2 * hs + go), vc
                    )
                    if sub:
                        var so = out_base + sub_off + base + c
                        _ = Atomic.fetch_add(out_hist.unsafe_offset(so), -vg)
                        _ = Atomic.fetch_add(
                            out_hist.unsafe_offset(hs + so), -vh
                        )
                        _ = Atomic.fetch_add(
                            out_hist.unsafe_offset(2 * hs + so), -vc
                        )
                c += block_dim.x


def _range_hist_partial_kernel[
    GROUP: Int, BIN_CAP: Int
](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    partials: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_slots: Int32,
    n_bins: Int32,
    rows_per_tile: Int32,
    begin: Int32,
    count: Int32,
    g_scale: Float32,
    h_scale: Float32,
    use_quant: Int32,
    const_hess: Int32,
    row_unroll: Int32,
    narrow_index: Int32,
    pair_align: Int32,
    bins_blocked: MutPointer[UInt8, MutAnyOrigin],
    bin_row_stride: Int32,
    hist_probe: Int32,
    bins_packed: MutPointer[UInt8, MutAnyOrigin],
    bin_pack_tab: MutPointer[Int32, MutAnyOrigin],
    bin_packed: Int32,
):
    """The tiled twin of `_range_hist_atomic_kernel`: the same threadgroup
    accumulation, written to a per-(tile, slot) partial slot instead of folded
    into the output with global atomics.

    Same two comptime parameters and the same rules for them, so everything
    the atomic kernel's docstring argues about capacity, group width,
    exactness, tail blocks, and the row loop holds here unchanged; the row
    loop is not merely equivalent but literally the same code, since both
    kernels call `_hist_accumulate_dispatch`. `row_unroll`, `narrow_index`,
    and `pair_align` reach it from here for the same reason and with the same
    guarantees. What
    is particular to this kernel is the partial layout, and it is untouched:
    a block owning several slots writes the same per-slot
    `[grad | hess | count]` slices that as many one-slot blocks would have
    written, each slice still written by exactly one block and by no atomic at
    all, so `_range_reduce_kernel` and the fused sibling subtraction it
    carries combine these partials unchanged whatever `GROUP` and `BIN_CAP`
    are.

    This is the kernel a large fit actually runs: `resolve_tiling` picks the
    tiled strategy whenever occupancy wants more than one tile per feature,
    and at 5M x 50 the histogram build is measured at about 79% of GPU
    training wall clock. Two measurements from the hand-written variants this
    replaces carry over, both on an Apple M4 with three interleaved repeats
    per arm in one process, same binned data, predictions byte-identical: a
    full 100-round `train_gpu` at 5M x 50 ran 15.96s at group 1 against 11.49s
    at group 2 (1.39x, the group-2 arm's own spread 0.03%), and 12.09s at
    group 2 against 11.70s at group 4 (1.034x, both spreads under 1%). Both
    were taken with three `MAX_BINS`-wide planes allocated at every bin count,
    which is what made group 4 nearly a wash: the traffic halving was given
    back by the occupancy halving. Whether sizing the planes to the real bin
    count changes that verdict is exactly the thing this parameterization
    makes measurable and it has NOT been measured. The measurement is an
    interleaved A/B of group 2 against group 4 at a bin count of 64 or below,
    in one process, the protocol `bench_histogram.mojo` already implements.

    **The elided hessian plane.** `const_hess` carries the same declaration
    the atomic kernel's docstring argues in full, and has one extra
    consequence here: the partial buffer holds two planes per tile rather
    than three, laid out as `[grad | count]`, so the write this kernel does
    and the read `_range_reduce_kernel` does are both a third smaller. That
    layout change is contained: the partial buffer is written by this kernel
    and read by that one and by nothing else in the package, and the
    allocation is unchanged, so a two-plane build simply uses two thirds of a
    buffer sized for three. The two kernels are handed the same flag from the
    same launch site, which is what keeps them agreeing about which layout
    they are looking at.

    The reconstruction happens in the reduction rather than here, because a
    partial tile's count is not the bin's count. Multiplying each tile's
    count by `hq_const` and summing would give the same integer -- integer
    multiplication distributes over addition exactly, at every width, and
    modulo 2^32 as well -- but it would spend one multiply per tile per cell
    instead of one per cell, so the reduction is where it belongs.
    """
    var slot0 = GROUP * Int(block_idx.x)
    var owned = Int(n_slots) - slot0
    if owned > GROUP:
        owned = GROUP
    var t = block_idx.y
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var n = Int(count)
    var plane = Int(n_slots) * nb
    var celide = Int(const_hess) != 0
    # Planes per tile in the partial buffer: `[grad | hess | count]`, or
    # `[grad | count]` when the hessian plane is elided.
    var n_planes = 2 if celide else 3

    var sg = stack_allocation[
        GROUP * BIN_CAP,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh = stack_allocation[
        GROUP * BIN_CAP,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sc = stack_allocation[
        GROUP * BIN_CAP,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    # Arm four, in the kernel a large fit actually runs. Same placement and
    # the same argument as the atomic twin's: after the threadgroup planes
    # are declared, so the footprint that bounds occupancy is the real one,
    # and before the zeroing, so no work is charged to the launch floor. The
    # tiled path is the one that matters most here, because it is the path
    # whose launch count grows with the tile count.
    var probe_mode = Int(hist_probe)
    if probe_mode == HIST_PROBE_EMPTY:
        _hist_probe_empty_mark[GROUP * BIN_CAP](
            sg,
            sh,
            sc,
            partials,
            Int32(thread_idx.x) ^ Int32(block_idx.x) ^ Int32(block_idx.y),
            Int(thread_idx.x),
        )
        return

    var span = owned * nb
    var b = tid
    while b < span:
        sg[unsafe_offset=b] = 0
        if not celide:
            sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var tile_begin = t * Int(rows_per_tile)
    var tile_end = tile_begin + Int(rows_per_tile)
    if tile_end > n:
        tile_end = n

    # The bin matrix this launch reads, and how many bytes one row advances
    # inside a column. `bin_row_stride` of one is the feature-major buffer the
    # backend uploads, which every other reader of `bins` in this package
    # still indexes; anything else is the `[block][row][G]` relayout of
    # `gpu_blocked_bins.mojo`, which `GpuActiveRows` owns and built once per
    # fit. Two pointers rather than one because a launch may not be handed the
    # same buffer twice, and the unselected one is simply never dereferenced.
    #
    # Resolved here, once per threadgroup, so the row loop below sees one
    # pointer and one stride and never asks which layout it is on.
    var bsrc = bins
    var rstride = Int(bin_row_stride)
    if rstride != BLOCKED_STRIDE_NONE:
        bsrc = bins_blocked
    # The packed arm, resolved the same way and for the same reason: a third
    # pointer, never dereferenced unless selected, because a launch may not be
    # handed the same buffer twice. `bin_packed` and `bin_row_stride` are never
    # both set -- `set_packed_bins` and `set_blocked_layout` refuse each other
    # -- and the order here is the order that statement is enforced in.
    var packed = Int(bin_packed) != 0
    if packed:
        bsrc = bins_packed

    _hist_accumulate_dispatch[GROUP, HIST_ROW_UNROLL](
        bsrc,
        rows,
        grad,
        hess,
        gq,
        feat_ids,
        sg,
        sh,
        sc,
        slot0,
        owned,
        nb,
        nr,
        Int(begin),
        tile_begin,
        tile_end,
        g_scale,
        h_scale,
        celide,
        Int(use_quant) != 0,
        Int(row_unroll) != 0,
        Int(narrow_index) != 0,
        Int(pair_align) != 0,
        Int(use_quant) == QUANT_SOURCE_PACKED16,
        rstride,
        probe_mode,
        bin_pack_tab,
        packed,
    )
    barrier()

    # The count plane is last in both layouts, so its index is
    # `n_planes - 1`: plane 2 of three, plane 1 of two.
    var count_plane = (n_planes - 1) * plane
    comptime for k in range(GROUP):
        if k < owned:
            var base = t * n_planes * plane + (slot0 + k) * nb
            var lift = k * nb
            var c = tid
            while c < nb:
                var s = lift + c
                partials[unsafe_offset = base + c] = sg[unsafe_offset=s][0]
                if not celide:
                    partials[unsafe_offset = base + plane + c] = sh[
                        unsafe_offset=s
                    ][0]
                partials[unsafe_offset = base + count_plane + c] = sc[
                    unsafe_offset=s
                ][0]
                c += block_dim.x


def _range_reduce_kernel(
    partials: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
    n_tiles: Int32,
    sub_offset: Int32,
    do_sub: Int32,
    h_scale: Float32,
    const_hess: Int32,
    hist_probe: Int32,
):
    """Element-wise reduction of the tiled partials, in ascending tile order.

    Byte for byte what `histogram_gpu._hist_reduce_kernel` does: the
    reduction depends on the partial layout only, not on which rows produced
    it, so integration should delete this copy and call the existing kernel.
    It lives here so the range path can be exercised end to end from this
    module alone.

    With `do_sub`, the same thread that writes a cell also subtracts it from
    the slot `sub_offset` words away, which is the sibling subtraction folded
    into the build. The destination is a signed offset rather than a second
    pointer because both slots live in one pool buffer. Each
    thread owns one cell of each slot and no thread reads a cell another
    writes, so the in-place update is race free even though the destination
    is a live histogram. The cells this reduction does not cover are the
    inactive features', which are zero in every live slot -- the build zeroes
    a whole slot whenever the feature set is narrowed -- so subtracting them
    would have been a no-op.

    **The elided hessian plane.** With `const_hess` the partials hold two
    planes per tile, `[grad | count]`, so the grid covers two thirds of the
    cells and each thread reads two thirds of the words. The thread that owns
    a count cell writes two output cells instead of one: the count itself,
    and the hessian plane as `hq_const * acc`, where `hq_const` is
    `Int32(round(Float32(1.0) * h_scale))` and `acc` is the bin's total row
    count. That is the exact integer the three-plane path would have reduced
    to, because on the three-plane path every row contributed the same
    `hq_const` to the hessian plane and `acc` rows contributed in total; the
    argument is written out in full in `_range_hist_atomic_kernel`.

    The fused subtraction stays exact for the same reason and needs no
    special case beyond doing it twice: the sibling's hessian cell is reduced
    by `hq_const * acc`, which is what the three-plane path would have
    subtracted from it.

    No thread races another over the extra write. The hessian output cell is
    reached only from the count plane's thread for the same (feature, bin),
    and on the elided path no thread is assigned the hessian plane at all,
    since the grid is sized by the partial layout rather than by the output
    layout.
    """
    # Arm four of the decomposition reaches here too, and it has to.
    #
    # On the tiled strategy -- which is the strategy a large fit runs -- one
    # `enqueue_range_histogram` is TWO launches, this one and the partial
    # accumulation before it. An empty arm that emptied only the accumulation
    # would have measured "an empty accumulation plus a full reduction" and
    # reported it as the launch floor, which is a mixture and not a floor.
    # Emptying both is what makes the arm's number what it says it is: the
    # cost of issuing the launches a histogram build issues, and the
    # occupancy they are issued at, with no work inside either.
    #
    # The consequence is stated rather than elided: **the reduction's own
    # cost therefore lands in the residual**, together with the shared
    # zeroing, the barriers and the flush. That is exactly where the
    # reduce-bound verdict reads it from -- three small fractions and a large
    # residual is the signature that sends the next lane at the tile count
    # and the reduction shape.
    #
    # Nothing keeps this body alive against a compiler and nothing needs to:
    # the launch is issued by the host and a kernel that returns immediately
    # is still dispatched, scheduled and retired. Unlike the accumulation
    # arm, there is no threadgroup allocation here whose size an occupancy
    # claim depends on, so there is nothing to anchor.
    if Int(hist_probe) == HIST_PROBE_EMPTY:
        return

    var i = global_idx.x
    var nb = Int(n_bins)
    var plane = Int(n_slots) * nb
    var celide = Int(const_hess) != 0
    var n_planes = 2 if celide else 3
    var n = n_planes * plane
    if i < n:
        var acc = Int32(0)
        var off = i
        for _ in range(Int(n_tiles)):
            acc += partials[unsafe_offset=off][0]
            off += n
        var p = i // plane
        var rem = i - p * plane
        var slot = rem // nb
        var b = rem - slot * nb
        var f = Int(feat_ids[unsafe_offset=slot][0])
        # Plane `p` of the partial layout maps to plane `p` of the output,
        # except that the elided layout's second plane is the count, which is
        # the output's third.
        var out_plane = 2 if (celide and p == 1) else p
        var cell = out_plane * Int(hist_size) + f * nb + b
        out_hist[unsafe_offset=cell] = acc
        if Int(do_sub) != 0:
            var sc = Int(sub_offset) + cell
            out_hist[unsafe_offset=sc] = out_hist[unsafe_offset=sc][0] - acc
        if celide and p == 1:
            var hq_const = Int32(round(Float32(1.0) * h_scale))
            var hv = hq_const * acc
            var hcell = Int(hist_size) + f * nb + b
            out_hist[unsafe_offset=hcell] = hv
            if Int(do_sub) != 0:
                var hs = Int(sub_offset) + hcell
                out_hist[unsafe_offset=hs] = (
                    out_hist[unsafe_offset=hs][0] - hv
                )


@fieldwise_init
struct LeafRange(Copyable, Movable):
    """The half-open window `[begin, end)` of the active-row buffer owned by
    one leaf. An empty range (`begin == end`) means the leaf holds no rows,
    which is what a split leaf becomes: its rows now belong to its
    children."""

    var begin: Int
    var end: Int

    @staticmethod
    def empty() -> LeafRange:
        return LeafRange(0, 0)

    def count(self) -> Int:
        return self.end - self.begin

    def is_empty(self) -> Bool:
        return self.end <= self.begin

    def contains(self, index: Int) -> Bool:
        return index >= self.begin and index < self.end

    def overlaps(self, other: LeafRange) -> Bool:
        if self.is_empty() or other.is_empty():
            return False
        return self.begin < other.end and other.begin < self.end


struct LeafRangeTable(Copyable, Movable):
    """Node id -> `LeafRange`, for one tree.

    Node ids double as leaf ids in the GPU grower and are handed out in
    ascending order, so the table is a `List` indexed by node id that grows
    by appending. Splitting a leaf costs at most two appends and no device
    work: the children's ranges are the two halves of the parent's, and the
    parent's becomes empty so the live ranges keep tiling `[0, n_active)`.

    **Validity, and why this table has a validity state at all.** Two
    partition planes write the device row buffer and only one of them writes
    this table. `GpuActiveRows.partition` ends every split with `split()`
    here, so the shipping plane leaves the table describing the tree it grew.
    `GpuActiveRows.enqueue_partition_desc` routes rows from a step descriptor
    a device kernel wrote, moves the windows in the device tree tables, and
    touches nothing on the host. While that second plane owns the
    partitioning, every window in this table is the window it held before the
    tree grew -- which for a fresh tree means node 0 owning the entire active
    prefix and every other node owning nothing.

    That is not a hypothetical. It shipped. `train_gpu`'s device-gradient
    round reads this table to advance the raw scores, was handed exactly that
    stale table, and added the root's value to every row instead of each
    row's own leaf value. Tree 0 was bit-identical, every tree after it
    diverged, and nothing raised. The read was silently wrong because a
    stale window is a perfectly well-formed window: it is in bounds, it tiles
    the prefix, and it satisfies every invariant this struct can check.

    So the invalidity is recorded explicitly rather than checked for.
    `begin_descriptor_partition` poisons the table, every accessor that
    returns a window refuses while poisoned, and
    `end_descriptor_partition` -- which
    `gpu_resident_round._publish_row_ranges` calls after replaying the
    device's commit log -- is the only thing that clears it, and only after
    verifying that the replay covered every node.

    Rejected alternative: arming the poison from the caller, at the resident
    round's first descriptor partition. The bug being guarded against is
    precisely a caller that did not know it had a host table to maintain, so
    a rule the callers have to remember is the rule that already failed. The
    poison is armed by the code whose behavior creates the invalidity, and
    `begin_descriptor_partition` is made load-bearing (it returns the row
    bound the launch geometry is derived from) so that a future edit cannot
    drop the arming and still compile.

    Rejected alternative: keeping a second host copy of the windows updated
    from the descriptor path. That is the bookkeeping the resident plane
    exists to delete, and it would have to be updated from device state the
    host has deliberately not read back.
    """

    var ranges: List[LeafRange]
    var n_rows: Int
    var n_active: Int
    # True while a plane that does not maintain this table owns the
    # partitioning. Named for the situation rather than for the mechanism
    # ("invalid", "dirty") because the sentence a reader needs is *who* owns
    # the windows, not that something is wrong.
    var resident_owned: Bool
    # Descriptor partitions armed since the last `reset_root`. Not used by
    # any decision; it exists so a test can assert that the poisoned path was
    # entered rather than assert that a read raised and hope the raise came
    # from the poison. A test that only checks "reads raise" passes just as
    # well when the arming silently did nothing and the read raised for some
    # unrelated reason, which is the failure mode this repository has already
    # shipped once (six resident-plane fixtures that all ran below the gate
    # they were meant to exercise and compared the fallback against itself).
    var desc_partitions: Int

    def __init__(out self, n_rows: Int):
        self.ranges = List[LeafRange]()
        self.n_rows = n_rows
        self.n_active = 0
        self.resident_owned = False
        self.desc_partitions = 0

    def reset_root(mut self, n_active: Int) raises:
        """Start a new tree with `n_active` rows at node 0.

        Clears the poison, and that is a decision worth defending rather than
        an oversight. The claim the poison makes is "the windows in this table
        describe an earlier state of the tree"; `reset_root` re-establishes
        the one state the host can assert without reading the device, because
        `GpuActiveRows.begin_tree` re-seeds the row buffer in the same call
        and node 0 then genuinely owns `[0, n_active)` with no other node in
        existence. Refusing to clear here would make the poison a lie in the
        other direction: it would refuse the shipping plane, which is
        entirely correct after a `begin_tree`, for the rest of the session.

        It is also not a hole. The window in which a stale read does damage
        runs from the last descriptor partition to the next `begin_tree`, and
        the read that motivated all of this -- `update_raw_device` -- lives
        inside it by contract: its docstring says "call after `grow_tree_gpu`
        returns and before the next `begin_tree`".
        """
        if n_active < 0 or n_active > self.n_rows:
            raise Error("active row count out of range")
        self.ranges.clear()
        self.ranges.append(LeafRange(0, n_active))
        self.n_active = n_active
        self.resident_owned = False
        self.desc_partitions = 0

    def is_resident_owned(self) -> Bool:
        """Whether a descriptor-driven partition currently owns the windows,
        so every window accessor below refuses. For tests and for a caller
        that wants to branch rather than catch."""
        return self.resident_owned

    def _refuse_stale(self) raises:
        """The one place the refusal is worded.

        The text names the situation and the two files involved, because the
        person who next reads it will be debugging a model that diverged after
        the first boosting round and needs the sentence to point at the seam,
        not to tell them a state is invalid.
        """
        raise Error(
            "the device-resident partition owns the active-row windows and"
            " this host table is stale: GpuActiveRows.enqueue_partition_desc"
            " routed rows from the device step descriptor, which moves the"
            " windows in the device tree tables and writes nothing here, so"
            " every window below is the one it held before this tree grew"
            " (node 0 owning the whole active prefix, every other node"
            " owning nothing). Reading one attributes rows to the wrong leaf"
            " without raising, which is how a device-gradient round once"
            " added the root's value to every row and diverged every tree"
            " after the first. gpu_resident_round._publish_row_ranges"
            " replays the device commit log onto this table and is the only"
            " thing that makes it readable again; run it first, or read the"
            " windows out of the device tree tables instead."
        )

    def begin_descriptor_partition(mut self, max_count: Int) raises -> Int:
        """Arm the poison for one descriptor-driven partition and return the
        row bound the launch geometry is derived from.

        Called by `GpuActiveRows.enqueue_partition_desc` and by nothing else.
        It validates and returns `max_count` rather than returning nothing,
        so that the arming is load-bearing: the partition cannot size a grid
        without it, and an edit that deletes the call fails to compile
        instead of quietly restoring the silent-divergence bug. Validation
        lives here for the same reason -- there is then no version of the
        entry point that checks its bound and skips the arming.

        Idempotent by design. Every step of a resident tree calls this, dead
        steps included, and re-arming an already-poisoned table is a no-op
        apart from the counter.
        """
        if max_count < 1:
            raise Error("a descriptor partition needs a positive row bound")
        if max_count > self.n_rows:
            raise Error("the row bound exceeds the row buffer")
        self.resident_owned = True
        self.desc_partitions += 1
        return max_count

    def end_descriptor_partition(
        mut self,
        n_nodes: Int,
        leaf_nodes: List[Int],
        leaf_windows: List[LeafRange],
    ) raises:
        """Clear the poison, after proving the replayed table is window for
        window the frontier the device came home with.

        `n_nodes` is the finished tree's node count and the two lists are the
        device's own live frontier: `leaf_nodes[i]` owns `leaf_windows[i]`.
        The caller reads both straight out of the downloaded snapshot
        (`gpu_resident_round._publish_row_ranges`) and passes them as plain
        integers and `LeafRange`s, so that this module never learns the
        snapshot type -- `gpu_tree_tables` imports this file and must not be
        imported back.

        Four checks, in order of what each one can catch:

        1. `len(self.ranges) == n_nodes`. The replay writes windows only for
           the nodes its commit log names, so a log that came home short --
           `_pick_and_commit_kernel` stops appending once the log is full --
           leaves the tail of the tree with no window at all.

        2. Every device leaf's window is byte for byte the window the replay
           gave that node. Not just the count: `begin` too, because the
           device computed it as `begin, begin + n_left` inside the commit
           kernel while the replay recomputed it by walking `split` in commit
           order, and those two arithmetics agreeing is the thing worth
           checking. This is what makes un-poisoning a proof rather than an
           argument, and it is what catches a commit missing from the
           *middle* of the log. Such a log leaves a table that is structurally
           impeccable: `_grow_to` back-fills the gap with empty ranges so the
           node count is right, the parent whose commit went missing stays a
           live leaf owning the rows its children should have, and the live
           windows still tile the prefix. Nothing about it is malformed. What
           is wrong is that the two back-filled children are empty here and
           are real windows on the device, which this check sees at the first
           of them. An earlier draft of this docstring claimed check 4 caught
           that case. It does not.

        3. Every node the device did *not* name as a live leaf owns nothing.
           A backstop rather than a first line, and worth being honest about:
           given check 2 and a device frontier that tiles, a non-live node
           holding rows would have to overlap a leaf that already matched, so
           check 4 would also refuse it. What this buys is the error message.
           A snapshot whose frontier does not tile fails here naming the node
           and the commit that should have split it, rather than downstream as
           "ranges overlap".

           **This item said such a snapshot is "which
           `TreeTablesSnapshot.check_invariants` is supposed to have refused
           upstream". IT IS NOT REFUSED UPSTREAM AND NEVER WAS.** Corrected
           2026-08-18, by reading that function. It tests nonnegativity,
           pairwise disjointness, distinct node, slot and record ids, the live
           leaf count, and `next_node == 2 * n_live - 1`. It tests no total, no
           bound against the active prefix, and no coverage of any kind, and it
           could not: `TreeTablesSnapshot` has no `n_active` field, so the
           snapshot never carries the length a tiling claim would be about. A
           frontier whose windows account for a fraction of the prefix passes
           every one of those tests. So a non-tiling frontier arrives here
           unrefused, and checks 2, 3 and 4 are the first thing it meets rather
           than a second line behind an upstream gate. Whether check 3 or check
           4 speaks first is a question about error messages; neither is
           optional, because there is nothing in front of them.

        4. `_check_invariants()`. The check that owns the relationship to
           `n_active`, which nothing above mentions: a publish onto a table
           whose root was never reset for this tree (`n_active` moves per tree
           under bagging and GOSS) fails here, with the error that names the
           actual problem rather than as two hundred individually wrong
           leaves.

        **What this does not prove, and it is the one gap worth naming.** The
        device's `row_begin`/`row_count` and the replay's are both ultimately
        derived from the same integer, the search record's left count off the
        parent histogram's count plane. So this check catches a replay that
        went wrong and cannot catch a left count that disagrees with the rows
        `_scatter_kernel` actually routed left. The shipping plane has an
        observation of that -- `MOJOTREES_GPU_VERIFY_ROWS` downloads
        `total_dev` and compares -- and the descriptor path has no equivalent,
        because reading `total_dev` per step is exactly the host wait the
        resident plane exists to remove. Recorded rather than closed.

        Clearing is unconditional once all four pass: there is then no
        untouched node to scope around, and no window that is not the
        device's.
        """
        if n_nodes < 1:
            raise Error("a finished tree holds at least the root")
        if len(leaf_nodes) != len(leaf_windows):
            raise Error("each device leaf must come with exactly one window")
        if len(self.ranges) != n_nodes:
            raise Error(
                "the device commit log was replayed onto ",
                len(self.ranges),
                " nodes but the tree ended with ",
                n_nodes,
                "; the untouched nodes have no active-row window, so"
                " clearing the stale-table refusal here would be a lie",
            )
        # Marks the nodes the device calls live, so check 3 can require an
        # empty window of everything else without a second search per node.
        var is_live = List[Bool](capacity=n_nodes)
        for _ in range(n_nodes):
            is_live.append(False)
        for i in range(len(leaf_nodes)):
            var node = leaf_nodes[i]
            if node < 0 or node >= n_nodes:
                raise Error(
                    "the device frontier names leaf ",
                    node,
                    " which is outside a tree of ",
                    n_nodes,
                    " nodes",
                )
            if is_live[node]:
                raise Error(
                    "the device frontier lists node ", node, " twice"
                )
            is_live[node] = True
            var got = self.ranges[node].copy()
            var want = leaf_windows[i].copy()
            if got.begin != want.begin or got.end != want.end:
                raise Error(
                    "leaf ",
                    node,
                    " owns rows [",
                    got.begin,
                    ", ",
                    got.end,
                    ") in the replayed host table and [",
                    want.begin,
                    ", ",
                    want.end,
                    ") on the device; the replay does not describe the tree"
                    " that was grown",
                )
        for n in range(n_nodes):
            if not is_live[n] and not self.ranges[n].is_empty():
                raise Error(
                    "node ",
                    n,
                    " is not a live leaf on the device but owns ",
                    self.ranges[n].count(),
                    " rows in the replayed host table; the commit that split"
                    " it is missing from the device commit log, so the two"
                    " describe different trees",
                )
        self._check_invariants()
        self.resident_owned = False

    def n_nodes(self) raises -> Int:
        """How many node ids have a window. Refuses while the resident
        partition owns them: the host's count is as stale as the windows it
        counts, and it is read as a loop bound by the raw-score update, which
        would otherwise walk one node and call it the whole tree."""
        if self.resident_owned:
            self._refuse_stale()
        return len(self.ranges)

    def get(self, node: Int) raises -> LeafRange:
        if self.resident_owned:
            self._refuse_stale()
        return self._get_raw(node)

    def _get_raw(self, node: Int) raises -> LeafRange:
        """`get` without the staleness refusal, for the replay that clears it.

        `split` is the writer the poison exists to wait for, so it cannot be
        subject to the poison; it reaches the parent's window through here.
        Private, and the only two callers are in this struct.
        """
        if node < 0 or node >= len(self.ranges):
            raise Error("leaf id has no active-row range")
        return self.ranges[node].copy()

    def _grow_to(mut self, node: Int):
        while len(self.ranges) <= node:
            self.ranges.append(LeafRange.empty())

    def split(
        mut self, parent: Int, left: Int, right: Int, n_left: Int
    ) raises -> LeafRange:
        """Hand the parent's range to its two children and return the left
        child's. `n_left` is the number of rows the partition sent left; the
        left child takes the parent's first `n_left` slots and the right
        child the rest, which is exactly what the stable partition wrote.

        Permitted while the table is poisoned, and it has to be: this is the
        call `gpu_resident_round._publish_row_ranges` replays to bring the
        table back, so subjecting the writer to the reader's refusal would
        make the poison unclearable. It reads the parent through `_get_raw`
        for that reason. The refusals below are unchanged and still hold on
        the replay, which is what makes a replay of a log that does not
        describe a real growth an error rather than a rewrite.
        """
        var r = self._get_raw(parent)
        if n_left < 0 or n_left > r.count():
            raise Error("left row count is outside the parent's range")
        if left < 0 or right < 0:
            raise Error("leaf ids must be nonnegative")
        if left > MAX_ROWS or right > MAX_ROWS or parent > MAX_ROWS:
            raise Error("leaf ids must fit in Int32")
        if left == parent or right == parent or left == right:
            raise Error(
                "child leaf ids must differ from the parent and each other"
            )
        var mid = r.begin + n_left
        self._grow_to(left)
        self._grow_to(right)
        # A leaf that already owns rows would be overwritten, which would
        # silently orphan them; ids come from the tree's node counter, so
        # this can only fire on a wiring mistake.
        if not self.ranges[left].is_empty():
            raise Error("left child already owns an active-row range")
        if not self.ranges[right].is_empty():
            raise Error("right child already owns an active-row range")
        self.ranges[left] = LeafRange(r.begin, mid)
        self.ranges[right] = LeafRange(mid, r.end)
        self.ranges[parent] = LeafRange.empty()
        return LeafRange(r.begin, mid)

    def set_window(mut self, node: Int, begin: Int, end: Int) raises:
        """Give `node` the window `[begin, end)` directly, without requiring
        it to be inside anything.

        **Why containment cannot be the only way in.** `split` is written for
        a partition that keeps a parent's rows in one block, which is what
        leaf-wise growth produces and what the shipping partition writes. An
        oblivious level does not produce that. One stable partition of the
        whole active prefix by the level's single routing rule sends all of
        the level's left children to the front and all of its right children
        to the back, so the level's new leaves are numbered `j` and
        `j + 2^l` -- `gpu_resident_round.OBLIVIOUS_LEAF_INDEX_RULE`, which is
        CatBoost's numbering and the one the cross-leaf scan sums in -- and a
        parent's rows land in **two blocks that are not adjacent**. There is
        no `n_left` that expresses that, so `split` cannot replay such a tree
        at all, and `gpu_resident_round._publish_row_ranges` is not optional:
        `GpuHistogramBuilder.update_raw_device` reads this table to advance
        the raw scores, and the last time it was handed a table describing a
        different partition the result was correct trees, wrong scores, and
        every round after the first diverging. See this struct's docstring;
        that is not a hypothetical, it shipped.

        The alternative was a per-leaf-contiguous partition, which keeps
        containment and numbers the level `2j` and `2j+1`, but needs a
        segmented prefix scan per level rather than one scan over the prefix.
        `gpu_resident_round.OBLIVIOUS_ROW_RANGES` costs that out at 3 launches
        per leaf, 189 at depth 6, against the 20 host lines below.

        **How this stays inside the poisoning discipline, which is the part
        that matters.** The poison guards *readers*: `get`, `n_nodes`,
        `total_active` and `check_invariants` all refuse while
        `resident_owned`, because a stale window is well formed and nothing
        downstream can tell it from a fresh one. This is a *writer*, and it is
        permitted while poisoned for exactly the reason `split` is -- it is
        how the replay that clears the poison happens -- and it is not a way
        around the poison for three reasons, all of them structural rather
        than conventional:

        1. It returns nothing and reads no window out, not even through
           `_get_raw`. A caller cannot learn a window from it, so it cannot
           be used to observe a table it is not allowed to observe.
        2. It never touches `resident_owned`. `end_descriptor_partition` is
           still the only thing that clears the poison, and it still clears it
           only after checking every replayed window against the device's own
           frontier byte for byte and running `_check_invariants`. A table
           built entirely out of this setter is therefore un-poisoned by the
           same proof a table built out of `split` is, and by no weaker one.
        3. It keeps `split`'s orphan guard, which is the one refusal that
           does not depend on containment: a node that already owns rows
           cannot be given a second nonempty window. Emptying a node is always
           allowed, because that is what a level replay does to the parents it
           has just replaced, and an empty window orphans nothing.

        What it deliberately does not check is that the windows tile. That is
        a property of the whole table and not of one write, it is false
        halfway through any replay, and `_check_invariants` inside
        `end_descriptor_partition` is where it is established -- once, over
        the finished table, before the poison lifts.
        """
        if node < 0 or node > MAX_ROWS:
            raise Error("leaf ids must be nonnegative and fit in Int32")
        if begin < 0 or end < begin:
            raise Error("an active-row window must be a nonnegative range")
        if end > self.n_rows:
            raise Error("an active-row window escapes the row buffer")
        self._grow_to(node)
        if begin != end and not self.ranges[node].is_empty():
            raise Error(
                "node ",
                node,
                " already owns an active-row window and a second nonempty"
                " window would orphan the rows it holds",
            )
        self.ranges[node] = LeafRange(begin, end)

    def total_active(self) raises -> Int:
        """Rows the live windows account for. Refuses while the resident
        partition owns the windows, because it is a sum over exactly the
        numbers that are stale."""
        if self.resident_owned:
            self._refuse_stale()
        return self._total_active()

    def _total_active(self) -> Int:
        var total = 0
        for i in range(len(self.ranges)):
            total += self.ranges[i].count()
        return total

    def check_invariants(self) raises:
        """The live ranges must tile `[0, n_active)`. Refuses while the
        resident partition owns them: a stale table passes this check, which
        is the whole reason the poison had to be recorded rather than
        detected."""
        if self.resident_owned:
            self._refuse_stale()
        self._check_invariants()

    def _check_invariants(self) raises:
        """The live ranges must tile `[0, n_active)`: inside the buffer,
        pairwise disjoint, and covering every active slot exactly once.

        Disjointness plus a total of `n_active` is coverage, since the
        ranges all sit inside a buffer of that length. Tree growth here is a
        few hundred leaves at most, so the quadratic check is cheaper than
        sorting and is what a test wants to read.

        Private so that `end_descriptor_partition` can run it on a table that
        is still poisoned, which is the one moment the check has to be made
        before the refusal is lifted rather than after.
        """
        for i in range(len(self.ranges)):
            var a = self.ranges[i].copy()
            if a.is_empty():
                continue
            if a.begin < 0 or a.end > self.n_active:
                raise Error("active-row range escapes the active prefix")
            for k in range(i + 1, len(self.ranges)):
                if a.overlaps(self.ranges[k]):
                    raise Error("active-row ranges overlap")
        if self._total_active() != self.n_active:
            raise Error("active-row ranges do not cover the active prefix")


@fieldwise_init
struct RowRouting(Copyable, Movable):
    """Everything a partition needs to route one leaf's rows.

    The same three fields the host grower, `Tree.goes_left`, and the device
    kernels already agree on, gathered so the host reference model and the
    kernels take their arguments from one place and cannot drift apart.
    """

    var feature: Int
    var threshold_bin: Int
    var missing_bin: Int
    var default_left: Bool
    var is_categorical: Bool
    var cat_bitset: CatBitset

    @staticmethod
    def numerical(
        feature: Int,
        threshold_bin: Int,
        missing_bin: Int = -1,
        default_left: Bool = False,
    ) -> RowRouting:
        return RowRouting(
            feature,
            threshold_bin,
            missing_bin,
            default_left,
            False,
            cat_empty(),
        )

    @staticmethod
    def categorical(feature: Int, bitset: CatBitset) -> RowRouting:
        """Bin 0 is never in the set, so missing, unseen, and dropped
        categories route right, as they do everywhere else."""
        return RowRouting(feature, -1, -1, False, True, bitset)

    @staticmethod
    def from_split(split: SplitInfo, missing_bin: Int) -> RowRouting:
        """The routing a chosen split implies. `missing_bin` is the split
        feature's missing bin, or -1; a categorical split ignores it, exactly
        as the grower does when it passes -1 for one."""
        if split.is_categorical:
            return RowRouting.categorical(split.feature, split.cat_bitset)
        return RowRouting.numerical(
            split.feature, split.bin, missing_bin, split.default_left
        )

    def goes_left(self, bin: Int) -> Bool:
        """Host mirror of `_row_goes_left`."""
        return _row_goes_left(
            Int32(bin),
            Int32(self.threshold_bin),
            Int32(self.missing_bin),
            Int32(1) if self.default_left else Int32(0),
            Int32(1) if self.is_categorical else Int32(0),
            self.cat_bitset[0],
            self.cat_bitset[1],
            self.cat_bitset[2],
            self.cat_bitset[3],
        )

    def check(self, n_features: Int, n_bins: Int) raises:
        if self.feature < 0 or self.feature >= n_features:
            raise Error("split feature out of range")
        if not self.is_categorical and (
            self.threshold_bin < 0 or self.threshold_bin >= n_bins
        ):
            raise Error("split threshold bin out of range")
        if self.missing_bin >= n_bins:
            raise Error("split missing bin out of range")


def partition_range_host(
    mut rows: List[Int32],
    mut scratch: List[Int32],
    data: BinnedMatrix,
    window: LeafRange,
    routing: RowRouting,
) raises -> Int:
    """Stable-partition `rows[window]` in place and return the left count.

    The reference model for the device kernels: same routing rule, same
    stable order, same in-place contiguous result, computed serially on the
    host. Tests hold the device output to it, and it documents the contract
    in code that can be read without a GPU. `scratch` is caller-owned and
    only its `window` window is used, so repeated calls allocate nothing.
    """
    if len(rows) != len(scratch):
        raise Error("scratch must be the same length as the row buffer")
    if window.begin < 0 or window.end > len(rows) or window.count() < 0:
        raise Error("range escapes the row buffer")
    routing.check(data.n_features, data.n_bins)

    var n_left = 0
    for j in range(window.begin, window.end):
        var row = Int(rows[j])
        if row < 0 or row >= data.n_rows:
            raise Error("active row index out of range")
        if routing.goes_left(data.bin_at(row, routing.feature)):
            n_left += 1

    var li = window.begin
    var ri = window.begin + n_left
    for j in range(window.begin, window.end):
        var row = rows[j]
        if routing.goes_left(data.bin_at(Int(row), routing.feature)):
            scratch[li] = row
            li += 1
        else:
            scratch[ri] = row
            ri += 1
    for j in range(window.begin, window.end):
        rows[j] = scratch[j]
    return n_left


struct GpuActiveRows(Movable):
    """The device-resident active-row permutation and its leaf ranges.

    Construct once per training session against the histogram builder's
    `DeviceContext`, call `begin_tree` per tree, `partition` per split, and
    `enqueue_range_histogram` per node. It owns only the index machinery: the
    binned matrix, the gradients, and the histogram output stay with
    `GpuHistogramBuilder` and arrive as pointers.
    """

    var ctx: DeviceContext
    # The active-row permutation; `rows[0 : n_active)` is live.
    var rows_dev: DeviceBuffer[DType.int32]
    # Scatter destination. Only the partitioned range is ever read back out
    # of it, so the two buffers never have to agree outside that window.
    var scratch_dev: DeviceBuffer[DType.int32]
    # Chunk-local exclusive left-prefix, one per element of the range, with
    # the element's own routing flag packed into the low bit. A chunk is one
    # threadgroup's share of the range (`_partition_grid`).
    var offsets_dev: DeviceBuffer[DType.int32]
    # Per-chunk left counts. Written by the flag pass and read, never
    # rewritten, by every block of the scatter, which scans them at its own
    # head; they were turned into block offsets in place by a launch of their
    # own until that fold removed it.
    var block_sums_dev: DeviceBuffer[DType.int32]
    # One Int32: the range's total left count, written by the scatter.
    var total_dev: DeviceBuffer[DType.int32]
    # The step descriptor (see "The step descriptor"), `STEP_WORDS` Int32.
    #
    # It lives here rather than with the tree tables that write it for two
    # reasons. The kernels that read it are all in this file, so the buffer
    # sits with its readers; and every host-driven launch has to pass
    # *something* for the descriptor argument, which must be a real pointer
    # and must not be any other buffer that launch already passes, because a
    # kernel launch may not be handed the same buffer twice mutably. Reusing
    # one of the partition's own buffers as the inert stand-in is what the
    # first draft did and the compiler correctly refused it.
    #
    # Nothing here writes it. `gpu_tree_tables.DeviceTreeTables.enqueue_step`
    # takes this pointer and its commit kernel fills it, which is the whole
    # of the coupling between that module and this one.
    var step_dev: DeviceBuffer[DType.int32]
    # The K=1 speculation's two extra descriptors and its two counters. All
    # three are allocated unconditionally and all three are inert unless
    # `set_descriptor_target` is ever moved off `DESC_STEP`: `STEP_WORDS` and
    # `SPEC_STAT_WORDS` Int32 are together under 200 bytes, which is not worth
    # a conditional allocation and a nullable field to save.
    #
    # `spec_dev` is written by `gpu_tree_tables._pick_runner_up_kernel` and
    # read by the speculative partition and the speculative histogram.
    # `build_dev` is written by `_spec_consume_kernel` and read by the real
    # ones. `spec_stats_dev` is written by both of those kernels and read only
    # by a caller that asks for it, so that a measured run moves no bytes it
    # did not before.
    var spec_dev: DeviceBuffer[DType.int32]
    var build_dev: DeviceBuffer[DType.int32]
    var spec_stats_dev: DeviceBuffer[DType.int32]
    # Which of the three descriptors a descriptor-aware launch reads. See the
    # `DESC_*` constants; `DESC_STEP` is what everything that predates the
    # speculation selects and is what this is reset to after every
    # speculative launch.
    var desc_target: Int
    # launch-fusion lane: whether the descriptor partition's copy-back is
    # folded into the descriptor histogram's slot zeroing, which is one fewer
    # command buffer per growth step. See `set_partition_fusion` for the arm
    # and `_copy_back_zero_slot_kernel` for why these two and no others.
    var fuse_partition_tail: Bool
    # Whether a `enqueue_partition_desc` has deferred its copy-back to the
    # `enqueue_desc_histogram` that must follow it. True for the width of that
    # one pairing and never across a call the pairing does not contain; every
    # entry point that could observe the row buffer in between refuses while it
    # is set, so a caller that breaks the pairing gets an error rather than a
    # window of rows that are half in `scratch`.
    var copy_back_debt: Bool
    # The block count the deferred copy-back would have launched at, carried
    # from the partition to the fused launch so the fused grid is at least as
    # wide as either half's was. Correctness does not depend on it -- both
    # halves are grid-strided -- but a grid narrower than the copy-back's would
    # make each thread walk further for no reason.
    var copy_back_blocks: Int
    # How many fused copy-back-and-zero launches this instance has issued, ever.
    # A count of launches actually enqueued and not a prediction of how many
    # there should be, which is the whole reason it is worth carrying: the arm
    # can be asserted to have engaged rather than assumed to have. Monotone
    # across the fit; a caller that wants a per-tree figure takes the
    # difference, which is what `gpu_resident_round` puts in its trace. One Int
    # increment per growth step on the host, no launch and no transfer.
    var copy_back_folds: Int
    var host_total: HostBuffer[DType.int32]
    var host_rows: HostBuffer[DType.int32]
    var stage_rows: HostBuffer[DType.int32]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var block_threads: Int
    var ranges: LeafRangeTable
    var verify_counts: Bool
    # Feature slots one histogram threadgroup accumulates: a rung of the
    # ladder 1, 2, 4, 8, 16, each of which has a kernel instantiation. Never
    # read by anything but `enqueue_range_histogram`, and it changes no
    # histogram it produces.
    var feature_group: Int
    # The interleaved pre-quantized gradient buffer (`_quantize_grad_hess_kernel`),
    # 2 * n_rows Int32, and the state that says whether it describes the
    # gradients the next histogram will be asked to read. It is invalidated by
    # `begin_tree`, which every gradient refill in train_gpu.mojo is followed
    # by, and by a change of scale, which any refill that changed a value
    # changes with overwhelming likelihood; the two together are what make it
    # safe to key on rather than on a round counter this module cannot see.
    var gq_dev: DeviceBuffer[DType.int32]
    var quantized_gradients: Bool
    var quant_valid: Bool
    var quant_g_scale: Float32
    var quant_h_scale: Float32
    # --- gradient-staging lane ---
    # Whether that buffer is filled and read as interleaved *Int16* pairs
    # rather than Int32 ones. One allocation serves both: the Int16 layout
    # uses the first `4 * n_rows` bytes of the same `2 * n_rows` Int32
    # allocation, so no pointer anywhere else in this file changes and the
    # eight `enqueue_function` argument lists are untouched. The allocation is
    # not shrunk; what halves is the traffic the histogram gather issues, and
    # that is the whole of the claim.
    #
    # `packed_gradients` is what the caller asked for and `quant_packed` is
    # which layout the buffer currently holds, kept apart so `_ensure_quantized`
    # rebuilds when an A/B moves the arm between repeats in one process rather
    # than handing the next histogram a buffer of the other width.
    #
    # Off by default, and the polarity is not timidity. Every other arm in this
    # file that defaults on is exact whatever the data; this one is exact only
    # while every row's quantized value fits sixteen bits, a bound that fails
    # on small fits (see `_quantize_grad_hess_i16_kernel`), and an arm that can
    # refuse a fit does not get to refuse it by default.
    var packed_gradients: Bool
    var quant_packed: Bool
    # The two per-plane overflow counters `_quantize_grad_hess_i16_kernel`
    # raises, their host mirror, and a zeroed source to reset them from. Eight
    # bytes each; allocated unconditionally because a `DeviceBuffer` field
    # cannot be conditionally absent, and never touched on the Int32 arm.
    var stage16_flag_dev: DeviceBuffer[DType.int32]
    var stage16_flag_host: HostBuffer[DType.int32]
    var stage16_zero: HostBuffer[DType.int32]
    # scan-primitives lane: whether the two scan stages run on
    # `block.prefix_sum` (the default) or on the hand-rolled Hillis-Steele
    # kernels. A launch-shape knob and nothing else, because the two arms
    # produce the identical permutation and the identical left count.
    var scan_primitives: Bool
    # The most threadgroups one partition launches, whatever the range
    # length; a longer range is covered by giving each of them more tiles to
    # walk rather than by launching more of them (`_partition_grid`). It is a
    # cap and not a count: a range short enough to fit in fewer blocks
    # launches fewer. Bounded because the scatter scans the block sums at its
    # own head, once per block, so the redundant reads grow as the square of
    # the block count; see `_scatter_kernel` for the bound the default
    # (`block_threads`) buys. A launch-shape knob and nothing else, because
    # the permutation is a function of the routing flags and each row's
    # position in the range and so cannot see the tiling at all.
    var partition_block_cap: Int
    # --- hist-kernel-family lane ---
    # The shared-plane width every histogram launch from here instantiates its
    # kernel at: `histogram_bin_capacity(n_bins)`, one of 32, 64, 128, 256.
    # Held rather than recomputed per node because it is a property of the
    # dataset, and read by `set_feature_group` so a width whose footprint the
    # device cannot hold is refused where it is asked for rather than at the
    # launch that would fail.
    var bin_cap: Int
    # `caps.max_shared_memory_per_block` as reported when this was
    # constructed. The only device fact this struct keeps, and it keeps it for
    # one reason: `3 * group * bin_cap * 4` has to be checked against
    # something before a group is accepted.
    var max_shared_bytes: Int
    # --- const-hessian lane ---
    # Whether the caller has declared that this round's objective guarantees a
    # per-row hessian of exactly `histogram.CONSTANT_HESSIAN`, and whether the
    # environment permits acting on such a declaration at all. The effective
    # answer is the conjunction, kept in `constant_hessian` so the launch site
    # reads one field. Declared, never inferred: see
    # `histogram.objective_has_constant_hessian` for why a round whose
    # hessians happen to be equal is not the same thing.
    var constant_hessian: Bool
    var const_hessian_allowed: Bool
    # --- hist-kernel-margin lane ---
    # Whether the histogram row loop walks `HIST_ROW_UNROLL` rows per
    # iteration with the loads of each stage issued together, or one row per
    # iteration as it did before that lane. A launch-shape knob and nothing
    # else: both arms visit the same rows and add the same integers, only in
    # a different order, and integer addition does not care
    # (`_hist_rows_step`).
    #
    # On by default, on the same footing the primitive scan arm is on by
    # default: it is strictly fewer instructions and, on the quantized arm,
    # strictly fewer loads for the identical result, so it does not need a
    # measurement to justify being the default. What it is not is free of
    # risk -- it raises the live register count in the row loop, and
    # threadgroup residency on this backend is bounded by things this project
    # cannot query -- which is why `set_row_unroll` exists.
    #
    # No environment variable, for the reason `partition_block_cap` gives:
    # the arm belongs in the call, because an A/B that reads its arm from the
    # environment has already once in this repository run one arm under the
    # other's label.
    var row_unroll: Bool
    # --- hist-latency lane ---
    # Whether the histogram row loop forms its two data-dependent indices --
    # the bin column offset `feature * n_rows + row` and the quantized pair
    # offset `2 * row` -- in Int32 rather than in Int.
    #
    # OFF by default, and the reasoning is the reverse of `row_unroll`'s.
    # That arm defaults on because it is provably strictly fewer
    # instructions; this one is not provable in either direction, because the
    # wide form's expensive term is loop-invariant and a compiler that
    # hoists it has already paid it once per slot rather than once per visit
    # (see `_hist_rows_step`). An arm that needs a measurement to justify it
    # waits for the measurement.
    #
    # It also carries a precondition the wide arm does not:
    # `narrow_index_supported` must hold. `set_narrow_index` refuses a
    # request the shape does not admit rather than honoring it quietly,
    # because the failure mode is a silently corrupted histogram and this
    # project's whole accuracy argument rests on the accumulation being
    # exact.
    var narrow_index: Bool
    # Whether the width-2 load of the quantized gradient pair states the
    # 8-byte alignment the address actually has.
    #
    # ON by default, on the same footing `row_unroll` is: the unannotated
    # spelling emits `align 4` for a `<2 x i32>` load, which is **observed**
    # from the emitted LLVM IR rather than measured on a device, and a
    # backend is free to split an under-aligned vector load back into the
    # two scalar loads the width-2 spelling exists to replace. Stating the
    # true alignment can only remove work, never add it. Off is reachable so
    # the two can be interleaved in one process.
    var pair_alignment: Bool
    # --- atomic-fraction lane ---
    # Whether the histogram row loop SKIPS its three shared atomics and
    # writes a per-thread sink instead. **A histogram built with this set is
    # wrong.** It is not an optimization, it is not an arm anything may ship
    # on, and the only thing it produces is a ratio.
    #
    # Off by default and reachable only through
    # `set_histogram_atomic_probe`, which takes a second, non-defaulted
    # acknowledgment argument and refuses without it. `enqueue_desc_histogram`
    # and any subtracting `enqueue_range_histogram` refuse to launch while it
    # is set, which between them is every histogram the device-resident
    # growth plane builds and every non-root node the host plane builds.
    var hist_atomic_probe: Bool
    # --- hist-decomposition lane ---
    # The other two arms of the histogram decomposition, as one of
    # `HIST_PROBE_OFF`, `HIST_PROBE_NO_GATHER`, `HIST_PROBE_EMPTY`. **A
    # histogram built with either set is wrong**, on the same footing as
    # `hist_atomic_probe` and with the same refusals.
    #
    # Why a second field rather than one widened enum. `hist_atomic_probe` is
    # a Bool a shipped test asserts on by name, and the arm it selects was
    # built, argued and measured by another lane; widening it would have
    # rewritten that lane's contract to add two arms beside it. The two
    # fields are held mutually exclusive by both setters instead, so there is
    # never a state in which one arm is on and another is also on, and the
    # launch resolves them to a single Int32 at the one place they reach a
    # kernel.
    #
    # Off by default and reachable only through `set_histogram_probe_mode`,
    # which takes the same non-defaulted acknowledgment argument
    # `set_histogram_atomic_probe` takes and refuses without it.
    var hist_probe_mode: Int
    # Row-tile requests, overriding what `gpu_tiling` would otherwise take
    # from the environment. Zero means "no request", which is what
    # `MOJOTREES_GPU_MIN_TILES` and `MOJOTREES_GPU_ROW_TILE` unset already
    # mean, so a caller that sets neither gets exactly the geometry this
    # module chose before the fields existed.
    #
    # They exist because the row-tile floor has to be re-measured and the
    # only protocol that compares anything on this machine is interleaved
    # arms in one process. An environment variable cannot be an arm: it is
    # read once per launch out of a global the harness cannot vary between
    # repeats, and this repository has already run one arm under the other's
    # label that way. See `gpu_tiling.row_tile_floor`.
    var min_tiles_request: Int
    var rows_per_tile_request: Int
    # --- blocked bin layout lane ---
    # The `[block][row][G]` re-layout of the binned matrix
    # (`gpu_blocked_bins.mojo`), and the `G` it was built at. `blocked_group`
    # is zero when no layout has been requested, in which case `blocked_dev`
    # is a one-byte placeholder that exists only so that every histogram
    # launch has a real pointer to pass.
    #
    # It does not replace `bins`. `_flag_scan_kernel` and `_scatter_kernel` in
    # this file, `gpu_predict`, `gpu_sparse`, `gpu_categorical` and the CPU
    # builder all still index `bins[f * n_rows + r]`, and a half-converted
    # matrix -- some kernels reading the blocked copy and some the
    # feature-major one -- is a much worse state than either layout, because
    # the two agree until a lane forgets one of them and then a histogram is
    # wrong on some shapes and not others. So both are resident and the
    # device pays for both; see `set_blocked_layout` for what that costs.
    #
    # `blocked_valid` says whether the buffer holds this fit's matrix. It is
    # set by `_ensure_blocked` on the first histogram launch after a layout is
    # requested and never cleared, because the binned matrix is uploaded once
    # per fit and no path in this backend mutates it: unlike `quant_valid`,
    # which tracks a per-tree quantity, there is nothing for this one to go
    # stale against.
    var blocked_dev: DeviceBuffer[DType.uint8]
    var blocked_group: Int
    var blocked_valid: Bool
    # --- packed bin layout lane ---
    # The bit-packed re-encoding of the binned matrix
    # (`gpu_packed_bins.mojo`): feature `f` as a stream of `n_rows` values at
    # `packed_widths[f]` bits. `packed_widths` is empty when no layout has been
    # requested, in which case both buffers below are one-byte placeholders
    # that exist only so every histogram launch has a real pointer to pass.
    #
    # `packed_tab_dev` is the `[base, width]` table the row loop reads, two
    # Int32 per feature. It is a device buffer rather than four launch
    # arguments because the width is per *feature* and a threadgroup's slots
    # are an arbitrary subset of them under `colsample`.
    #
    # It does not replace `bins`, for exactly the reason the blocked buffer
    # does not: `_flag_scan_kernel` and `_scatter_kernel` here, `gpu_predict`,
    # `gpu_sparse`, `gpu_categorical` and the CPU builder all still index
    # `bins[f * n_rows + r]`. So both are resident. Unlike the blocked copy,
    # this one is *smaller* than the matrix it shadows -- that is the whole
    # point of it -- so the residency charge is `packed_bin_bytes_ratio` of
    # the binned matrix and not another whole copy of it.
    #
    # `packed_valid` says whether the buffer holds this fit's matrix. Set by
    # `_ensure_packed` on the first histogram launch after a layout is
    # requested and never cleared, because the binned matrix is uploaded once
    # per fit and nothing in this backend mutates it.
    var packed_dev: DeviceBuffer[DType.uint8]
    var packed_tab_dev: DeviceBuffer[DType.int32]
    var packed_widths: List[Int]
    var packed_valid: Bool
    # --- row-compaction lane ---
    # CatBoost's physical reorder, behind a switch. See the block comment
    # above `_compact_build_kernel` for the invariant and the cost.
    #
    # `row_compaction` is what the caller asked for; `compact_valid` is
    # whether the planes currently satisfy the invariant. A histogram reads
    # the compacted planes only when **both** hold, so an invalidation can
    # never produce a wrong histogram, only a slower one: the next histogram
    # rebuilds and the launch before it went to `bins` and `rows` exactly as
    # it always did.
    var row_compaction: Bool
    var compact_valid: Bool
    # Which staged gradient width `cgq_dev` was written at, mirroring
    # `quant_packed` for the compacted copy. It exists because the two can
    # come apart: `set_packed_gradients` invalidates `quant_valid` but a
    # partition between that call and the next histogram would otherwise move
    # the gradient pair at the newly requested width through a plane written
    # at the old one, which is a wrong histogram that decodes to legal values.
    # `row_compaction_live` requires the two to agree, so a mismatch falls back
    # to the un-compacted launch until `_ensure_compacted` rebuilds.
    var compact_packed: Bool
    # Whether the four planes have been allocated at full size. False until
    # the first `set_row_compaction(True)`, at which point the arm's memory
    # cost is paid; before that all four are one-element placeholders, on the
    # footing `blocked_dev` is a one-byte placeholder on.
    var compact_allocated: Bool
    var cbins_dev: DeviceBuffer[DType.uint8]
    var cbins_alt_dev: DeviceBuffer[DType.uint8]
    var cgq_dev: DeviceBuffer[DType.int32]
    var cgq_alt_dev: DeviceBuffer[DType.int32]
    # The identity permutation, `ident[j] == j`. This is what the histogram
    # launch passes in place of `rows_dev` when the compacted planes are live,
    # and it is a buffer rather than a kernel arm because the kernel that
    # reads it is the accumulation loop and this lane does not touch it.
    var ident_dev: DeviceBuffer[DType.int32]
    # Launches actually issued, ever, not a prediction of how many there
    # should be -- the same discipline `copy_back_folds` follows and for the
    # same reason: the arm can be asserted to have engaged rather than assumed
    # to have. `compact_builds` counts full rebuilds (one per tree in a clean
    # run), `compact_scatters` counts incremental maintenance (one per split).
    var compact_builds: Int
    var compact_scatters: Int
    # Whether the partition's flag pass reads the compacted plane instead of
    # gathering. Requested here, but only *live* when the compacted planes are
    # live as well; see `compact_flag_read_live`. Off by default, and it is a
    # second arm inside a default-off arm on purpose: the compaction A/B has to
    # be able to price the physical reorder on its own before it prices the
    # reorder plus the gather it removes from this stage.
    var compact_flag_read: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        caps: DeviceCaps,
    ) raises:
        """Allocate every buffer the whole session will use.

        Sized by `n_rows`, never by a node: a split allocates nothing. The
        scan block size is the tiling policy's block size, clamped to what
        the shared-memory scan buffer holds.
        """
        if n_rows < 1:
            raise Error("GPU backend requires at least one row")
        if n_features < 1:
            raise Error("GPU backend requires at least one feature")
        if n_bins < 1:
            raise Error("GPU backend requires at least one bin")
        if n_bins > MAX_BINS:
            raise Error("GPU backend supports at most 256 bins")
        if n_rows > MAX_ROWS:
            raise Error("GPU backend supports at most 2^31 - 1 rows")

        self.ctx = ctx
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins

        var threads = derive_block_threads(caps)
        if threads > SCAN_MAX_THREADS:
            threads = SCAN_MAX_THREADS
        self.block_threads = threads
        var max_blocks = (n_rows + threads - 1) // threads

        self.rows_dev = self.ctx.enqueue_create_buffer[DType.int32](n_rows)
        self.scratch_dev = self.ctx.enqueue_create_buffer[DType.int32](n_rows)
        self.offsets_dev = self.ctx.enqueue_create_buffer[DType.int32](n_rows)
        self.block_sums_dev = self.ctx.enqueue_create_buffer[DType.int32](
            max_blocks
        )
        self.total_dev = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.step_dev = self.ctx.enqueue_create_buffer[DType.int32](STEP_WORDS)
        self.spec_dev = self.ctx.enqueue_create_buffer[DType.int32](STEP_WORDS)
        self.build_dev = self.ctx.enqueue_create_buffer[DType.int32](STEP_WORDS)
        self.spec_stats_dev = self.ctx.enqueue_create_buffer[DType.int32](
            SPEC_STAT_WORDS
        )
        self.desc_target = DESC_STEP
        # The partition-tail fusion is **on**, which is the polarity this
        # repository reserves for an arm that has been measured -- and this one
        # does not need measuring, which is a stronger claim than a measurement.
        # It is strictly one fewer command buffer per growth step, storing the
        # same bytes to the same addresses under the same guard; there is no
        # trade-off here for a benchmark to resolve, only a launch that is not
        # issued. `set_partition_fusion(False)` is the arm a window holds it
        # against.
        self.fuse_partition_tail = True
        self.copy_back_debt = False
        self.copy_back_blocks = 1
        self.copy_back_folds = 0
        self.host_total = self.ctx.enqueue_create_host_buffer[DType.int32](1)
        self.host_rows = self.ctx.enqueue_create_host_buffer[DType.int32](
            n_rows
        )
        self.stage_rows = self.ctx.enqueue_create_host_buffer[DType.int32](
            n_rows
        )
        self.ranges = LeafRangeTable(n_rows)
        # Off by default: the grower already knows the exact left count from
        # the parent's integer histogram counts, so the download is a check,
        # not a dependency. `MOJOTREES_GPU_VERIFY_ROWS=1` turns it on, which
        # is what tests and a first integration want.
        self.verify_counts = _env_int("MOJOTREES_GPU_VERIFY_ROWS", 0) != 0
        # The shared-plane width every kernel launched from here is
        # instantiated at, fixed by the dataset's bin count.
        self.bin_cap = histogram_bin_capacity(n_bins)
        self.max_shared_bytes = caps.max_shared_memory_per_block
        # One feature slot per threadgroup unless asked otherwise. Every rung
        # of the ladder produces the identical histogram, so this is a
        # launch-shape knob and nothing else. This constructor cannot see
        # which API the context speaks, so the per-device default lives in the
        # caller: GpuHistogramBuilder widens it by the free-footprint rule in
        # `gpu_tiling.free_feature_group` when the environment does not
        # choose. A request that is not a rung, or one whose footprint this
        # device cannot hold, is rounded down here rather than raised, because
        # a constructor is not where an environment variable should fail a
        # fit; `set_feature_group` raises, because there the width came from a
        # caller who can be told.
        var group = _env_int("MOJOTREES_GPU_FEATURE_GROUP", 1)
        if group < 1:
            group = 1
        if group > FEATURE_GROUP_MAX:
            group = FEATURE_GROUP_MAX
        var rung = 1
        var chosen = 1
        for _ in range(FEATURE_GROUP_LADDER):
            if rung <= group and (
                histogram_shared_bytes(self.bin_cap, rung)
                <= self.max_shared_bytes
            ):
                chosen = rung
            rung *= 2
        self.feature_group = chosen
        # Pre-quantized interleaved gradients, the default source for every
        # histogram kernel. `MOJOTREES_GPU_QUANTIZED_GRADS=0` or
        # `set_quantized_gradients(False)` forces the Float32 planes back, so
        # a benchmark can hold both arms. Cannot change a histogram, only what
        # the kernel gathers per row; see `_quantize_grad_hess_kernel` for why
        # that is an argument from the expression rather than a measurement.
        self.gq_dev = self.ctx.enqueue_create_buffer[DType.int32](2 * n_rows)
        self.quantized_gradients = (
            _env_int("MOJOTREES_GPU_QUANTIZED_GRADS", 1) != 0
        )
        self.quant_valid = False
        self.quant_g_scale = 0.0
        self.quant_h_scale = 0.0
        # The packed Int16 staging arm, off. `MOJOTREES_GPU_PACKED_GRADS=1` or
        # `set_packed_gradients(True)` turns it on; see the field for why this
        # is the one staging arm that does not default on.
        self.packed_gradients = (
            _env_int("MOJOTREES_GPU_PACKED_GRADS", 0) != 0
        )
        self.quant_packed = False
        self.stage16_flag_dev = self.ctx.enqueue_create_buffer[DType.int32](
            STAGE16_FLAG_WORDS
        )
        self.stage16_flag_host = self.ctx.enqueue_create_host_buffer[
            DType.int32
        ](STAGE16_FLAG_WORDS)
        self.stage16_zero = self.ctx.enqueue_create_host_buffer[DType.int32](
            STAGE16_FLAG_WORDS
        )
        var z16 = self.stage16_zero.unsafe_ptr()
        for i in range(STAGE16_FLAG_WORDS):
            z16.unsafe_store(i, Int32(0))
        # The primitive scan arm is the default: it computes the same
        # permutation with strictly less work, so it does not need a
        # measurement to justify being on. `MOJOTREES_GPU_SCAN_PRIMITIVES=0`
        # takes it off, and a width the primitive has no instantiation for
        # takes it off here rather than at every launch.
        self.scan_primitives = _env_int(
            "MOJOTREES_GPU_SCAN_PRIMITIVES", 1
        ) != 0 and _scan_primitive_width_supported(threads)
        # One threadgroup width's worth of scan blocks. Chosen by the
        # counting argument in `_scatter_kernel` and not by a measurement:
        # at this cap the scatter's redundant re-read of the block sums is
        # bounded by a third of the range's own streamed traffic in the worst
        # case, and every range shorter than `threads * threads` gets exactly
        # the one-tile-per-block geometry the module launched before the head
        # scan was folded in. `set_partition_block_cap` moves it, which is
        # how a benchmark holds two caps in one process and how the test file
        # forces the multi-tile path on a range small enough to check by
        # hand. No environment variable: the arm belongs in the call, because
        # an A/B that reads its arm from the environment has already once in
        # this repository run one arm under the other's label.
        self.partition_block_cap = threads
        # The constant-hessian specialization is available but not declared.
        # A builder that says nothing gets the three-plane path that shipped,
        # which is the only default a specialization keyed on the objective
        # can have: this constructor cannot see an objective.
        # `MOJOTREES_CONST_HESSIAN=0` withdraws the permission, so a later
        # `set_constant_hessian(True)` is refused rather than silently
        # honored, which is the off switch a bisection wants.
        self.const_hessian_allowed = _env_int("MOJOTREES_CONST_HESSIAN", 1) != 0
        self.constant_hessian = False
        # The unrolled row walk is the default; see the field.
        self.row_unroll = True
        # The narrow index arm is off until something measures it, and the
        # aligned pair load is on because a truer alignment annotation cannot
        # cost anything. Neither has an environment variable, for the reason
        # `partition_block_cap` gives. See both fields.
        self.narrow_index = False
        self.pair_alignment = True
        # The atomics probe is off, and "off" here is not a default anyone
        # should read as adjustable: on, this instance builds histograms that
        # are wrong. See `set_histogram_atomic_probe`.
        self.hist_atomic_probe = False
        # The other two decomposition arms, off for the same reason and with
        # the same warning. See `set_histogram_probe_mode`.
        self.hist_probe_mode = HIST_PROBE_OFF
        self.min_tiles_request = 0
        self.rows_per_tile_request = 0
        # No bin re-layout until one is asked for. The placeholder is one
        # byte, not `n_rows * n_features`: a layout that is never requested
        # must cost nothing, and the reference shape's blocked buffer is 52 MB
        # of device memory to allocate on the chance somebody wants it.
        # `set_blocked_layout` is where the real allocation happens, and it is
        # also where the residency cost is stated to the caller.
        self.blocked_dev = self.ctx.enqueue_create_buffer[DType.uint8](1)
        self.blocked_group = 0
        self.blocked_valid = False
        # No bit-packed matrix until one is asked for, and the same one-byte
        # placeholders for the same reason. `set_packed_bins` is where the real
        # allocation happens and where its size is stated to the caller.
        self.packed_dev = self.ctx.enqueue_create_buffer[DType.uint8](1)
        self.packed_tab_dev = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.packed_widths = List[Int]()
        self.packed_valid = False

        # Row compaction, off, allocating nothing. The four placeholders are
        # one element each for the reason `blocked_dev`'s is: an arm nobody
        # asked for must not cost `2 * n_rows * n_features` bytes of device
        # memory on the chance somebody does.
        #
        # `MOJOTREES_GPU_ROW_COMPACTION=1` turns it on here. Every other
        # launch-shape arm in this file that has no environment variable says
        # in its own comment why the arm belongs in the call: because an A/B
        # that reads its arm from the environment has already once in this
        # repository run one arm under the other's label. That reasoning is
        # sound and it is overridden here for one specific reason. This arm
        # has to be A/B'd **end to end**, through `train_gpu` and the
        # device-resident plane, and neither of those files may be edited by
        # the lane that built it, so there is no call site for the arm to
        # belong in. `set_row_compaction` is the arm a caller that *can* reach
        # this struct should use, it is what the test file uses, and it is
        # what makes both arms reachable at run time in one binary; the
        # variable exists so the shipping trainer can reach the arm at all.
        # `MOJOTREES_GPU_COMPACTION_TRACE` is what stops the two being
        # confused: it says per tree whether the arm engaged.
        self.row_compaction = False
        self.compact_valid = False
        self.compact_packed = False
        self.compact_allocated = False
        self.cbins_dev = self.ctx.enqueue_create_buffer[DType.uint8](1)
        self.cbins_alt_dev = self.ctx.enqueue_create_buffer[DType.uint8](1)
        self.cgq_dev = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.cgq_alt_dev = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.ident_dev = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.compact_builds = 0
        self.compact_scatters = 0
        # `MOJOTREES_GPU_COMPACT_FLAG_READ=1`, off by default, and inert on its
        # own: it changes nothing at all unless the compaction arm above is
        # also on. Read here once rather than per launch, the way
        # `scan_primitives` is, and overridable in process through
        # `set_compact_flag_read` so one benchmark can hold both arms.
        self.compact_flag_read = (
            _env_int("MOJOTREES_GPU_COMPACT_FLAG_READ", 0) != 0
        )
        # A bagged tree stages only its bag's slots, and the copy that
        # follows takes the whole buffer, so the tail is zeroed once here
        # rather than left as whatever the allocation held. No kernel reads
        # past the root range, so this is hygiene, not correctness.
        var stage = self.stage_rows.unsafe_ptr()
        for r in range(n_rows):
            stage.unsafe_store(r, Int32(0))

        # Last, because it allocates and because it refuses configurations the
        # fields above decide. A request the shape does not admit raises here
        # rather than being honored quietly, which is the polarity
        # `set_narrow_index` established for a precondition whose failure mode
        # is a silently wrong histogram.
        if _env_int("MOJOTREES_GPU_ROW_COMPACTION", 0) != 0:
            self.set_row_compaction(True)

    def synchronize(mut self) raises:
        self.ctx.synchronize()

    def n_active(self) -> Int:
        """Rows this tree grows on: the bag, or every row when unbagged.

        Deliberately *not* refused while the resident partition owns the
        windows, unlike everything that returns a window. A partition is a
        permutation of the active prefix: it moves rows between leaves and
        cannot change how many rows the tree grows on. `n_active` is set by
        `begin_tree` and is exactly as true after thirty descriptor
        partitions as it was before the first, so refusing it would be a
        false alarm -- and it is the number the resident round passes back in
        as the descriptor partition's own row bound.
        """
        return self.ranges.n_active

    def range_of(self, node: Int) raises -> LeafRange:
        """One node's window. Raises while a descriptor-driven partition owns
        the windows; see `LeafRangeTable`."""
        return self.ranges.get(node)

    def rows_stale(self) -> Bool:
        """Whether the host row-range table is currently disowned by a
        descriptor-driven partition, so `range_of`, `range_tiling`,
        `check_frontier`, `download_range`, `partition` and
        `enqueue_range_histogram` all refuse. Exposed so a caller can branch
        instead of catching, and so a test can assert on the state rather
        than only on the raise."""
        return self.ranges.is_resident_owned()

    def set_feature_group(mut self, group: Int) raises:
        """How many feature slots one histogram threadgroup accumulates.

        An argument rather than only an environment variable, because an A/B
        that reads its arm from the environment can silently run one arm
        under the other's label; that happened once in this repository with
        the split-search strategy and the benchmark harness now passes the
        arm in. Takes effect on the next `enqueue_range_histogram`, and
        cannot change a histogram, only the launch that builds it.

        Two refusals, and they are different refusals. A width that is not a
        rung of the ladder (3, or anything above 16) has no kernel
        instantiation and could not launch. A width that is a rung but whose
        three `group * bin_cap` Int32 planes exceed what this device reported
        for one threadgroup would compile and then fail at the launch, which
        is a worse place to find out; at 256 bins that is every group past 2
        on a 32 KiB budget, and at 32 bins it is nothing on the ladder.
        """
        if not is_feature_group_width(group):
            raise Error(
                "feature group must be 1, 2, 4, 8, or 16: those are the"
                " widths a kernel is instantiated at"
            )
        var need = histogram_shared_bytes(self.bin_cap, group)
        if need > self.max_shared_bytes:
            raise Error(
                "a feature group of ",
                group,
                " at ",
                self.bin_cap,
                " bins needs ",
                need,
                " bytes of threadgroup memory and this device reported ",
                self.max_shared_bytes,
            )
        self.feature_group = group

    def set_quantized_gradients(mut self, on: Bool):
        """Whether every histogram kernel reads the pre-quantized interleaved
        gradient buffer instead of the two Float32 planes.

        On by default, unlike when this was wired into one kernel only. The
        reason it can be a default is that it is exact by construction rather
        than by measurement: `gq[2r]` is the same Float32 multiply and the
        same `round` the row loop would have evaluated, hoisted out of a loop
        it does not depend on, so the histogram is bit-identical and the row
        loop does strictly less work. See `_quantize_grad_hess_kernel`.

        It stays an argument, and `MOJOTREES_GPU_QUANTIZED_GRADS=0` stays a
        way to force the Float32 planes, for the same reason
        `set_feature_group` is an argument: an A/B holds both arms in one
        process rather than reading its arm from the environment. Takes
        effect on the next `enqueue_range_histogram` and cannot change a
        histogram.
        """
        self.quantized_gradients = on
        self.quant_valid = False
        # The compacted planes hold the quantized pair, so a change of arm
        # invalidates them along with `gq` itself. `row_compaction_live` also
        # tests `quantized_gradients`, so a caller that turns the quantized
        # arm off gets the un-compacted launch rather than an identity index
        # against a plane that is not there.
        self.compact_valid = False

    def set_packed_gradients(mut self, on: Bool):
        """Whether the pre-quantized buffer is staged and gathered as
        interleaved **Int16** pairs instead of Int32 ones.

        Halves the bytes the histogram's per-row derivative fetch issues: two
        bytes per plane per row instead of four, which at the default one
        feature slot per threadgroup is half of the loop's per-visit traffic
        that is not the bin byte. It does not touch the accumulators, which
        stay Int32 in threadgroup memory, Int32 in the global planes and Int32
        through sibling subtraction, so it is not the packed-accumulator
        candidate `docs/design/ACCURACY_BUDGET.md` section 5 prices and it
        costs none of that candidate's accuracy.

        **Exact under a bound, and the bound is checked.** Sign extension of a
        16-bit word is exact, so the shared plane receives the identical
        integer whenever the stored word is the integer the Int32 arm would
        have stored -- that is, whenever every row of the round satisfies
        `-32768 <= Int32(round(x * scale)) <= 32767`. That is a per-row bound
        over the whole round, because the buffer is staged once per round and
        every node of the tree reads it. `_quantize_grad_hess_i16_kernel`
        compares each row's own integer against the range as it stores it and
        counts the failures; `_ensure_quantized` reads the counters and raises
        before the histogram that would have read a truncated buffer is
        enqueued. Nothing is clamped and nothing is silently widened.

        **Which is why this is off by default.** Under the shipped
        `2^30 / sum|g|` scale the bound is `max|g| / sum|g| <= 3.05e-5`, so it
        holds at large row counts and fails at small ones. That is the useful
        direction -- the gather it cuts is the largest component of the
        histogram phase only at the large shapes -- but an arm that can refuse
        a fit is an arm a caller opts into. `MOJOTREES_GPU_PACKED_GRADS=1` is
        the other way in, and both exist so an interleaved A/B can hold the
        two widths in one process, which is the only comparison this machine's
        drifting timings admit.

        Requires the quantized source: it is a narrower spelling of that
        buffer, not a third one, so it is ignored while
        `set_quantized_gradients(False)` is in force. Takes effect on the next
        `enqueue_range_histogram`, which restages the buffer at the new width.
        """
        self.packed_gradients = on
        self.quant_valid = False

    def packed_gradients_on(self) -> Bool:
        """Whether the next histogram will gather Int16 pairs. The conjunction
        of both staging arms, because the packed arm is a width of the
        quantized buffer and not an alternative to it."""
        return self.packed_gradients and self.quantized_gradients

    def staged_gradient_bytes_per_row(self) -> Int:
        """Bytes of staged derivative one row occupies in the buffer the
        histogram gathers, at the arms currently set.

        The number this lane is measured on, published rather than recomputed
        by each reader, and checkable without a clock. Per (row, feature)
        visit the row loop fetches this many bytes divided by the feature
        group, plus four for the row index divided by the feature group, plus
        one bin byte. Four on the Float32 arm because a constant-hessian round
        gathers only the gradient plane and eight because a general one
        gathers both; four or eight likewise on the Int32 quantized arm, the
        pair being one load either way; two or four packed.
        """
        var wide = 4 if self.packed_gradients_on() else 8
        return wide // 2 if self.constant_hessian else wide

    def set_row_unroll(mut self, on: Bool):
        """Whether the histogram row loop keeps `HIST_ROW_UNROLL` rows in
        flight or walks one row per iteration.

        Off is the shape this module shipped before the hist-kernel-margin
        lane, and it is reachable at run time rather than through a second
        kernel instantiation, so a benchmark can hold both arms in one
        process. That matters more here than it usually would: this machine's
        device timings drift several-fold between time windows, so only
        interleaved arms compare, and a comptime-only knob would have forced
        a two-build comparison instead.

        It cannot change a histogram. Both arms visit the same rows of the
        same range and add the same fixed-point integers into the same bins;
        only the order of the adds and of the loads that feed them differs,
        and integer addition is associative and commutative. See
        `_hist_rows_step` for the argument written out, and note that the one
        floating-point expression involved, the Float32 arm's
        `Int32(round(x * scale))`, is reproduced unchanged in both arms
        because `docs/NUMERICS.md` section 5.6 depends on its exact form.

        Takes effect on the next `enqueue_range_histogram`.
        """
        self.row_unroll = on

    def narrow_index_supported(self) -> Bool:
        """Whether this dataset's shape admits the Int32 index arm.

        `narrow_index_fits` over this instance's shape; the bound and its
        derivation are written out there, where a test can reach them without
        a device.
        """
        return narrow_index_fits(self.n_rows, self.n_features)

    def set_narrow_index(mut self, on: Bool) raises:
        """Whether the histogram row loop forms its two data-dependent
        indices in Int32 rather than in Int.

        Off is what this module shipped and is the default; see the
        `narrow_index` field for why this arm does not get the
        on-by-default treatment `row_unroll` gets.

        **Refuses rather than ignores.** A request the shape does not admit
        raises, because the two failure modes are not symmetric: honoring it
        quietly on an oversized dataset wraps an index and silently
        accumulates into the wrong bin, and a wrong histogram is the one
        error this project has no tolerance for. `narrow_index_supported`
        is the predicate and states the bound.

        Under that bound both arms address the same bytes and add the same
        integers, so the histogram is identical bit for bit. Unlike
        `set_row_unroll`, which is exact whatever the data because it only
        reorders integer adds, this one is exact *given a checked bound on
        the shape*. That difference is why there is a check at all.

        Takes effect on the next `enqueue_range_histogram`.
        """
        if on and not self.narrow_index_supported():
            raise Error(
                "the Int32 index arm needs n_features * n_rows and 2 *"
                " n_rows to fit a signed 32-bit integer; this shape exceeds"
                " it and the wide arm is the only correct one for it"
            )
        self.narrow_index = on

    def set_pair_alignment(mut self, on: Bool):
        """Whether the width-2 load of the quantized gradient pair states
        the 8-byte alignment its address actually has.

        On is the default. Off is the spelling that shipped with the
        width-2 load, which emits `align 4` for a `<2 x i32>` load and
        therefore permits a backend to split it back into the two scalar
        loads the width-2 spelling replaced. Both read the same eight bytes
        into the same two variables, so this cannot change a histogram; it
        is an assertion to the code generator and nothing else.

        Reachable at run time for the same reason `set_row_unroll` is: only
        interleaved arms compare on this machine. The alignment itself is a
        comptime parameter of the load, so the arm is a block-uniform branch
        over two spellings of it rather than a second kernel instantiation.

        Has no effect on the Float32 gradient arm, which gathers two
        separate planes and issues no pair load, nor on the elided-hessian
        arm, which wants only the first word and keeps a scalar load.

        Takes effect on the next `enqueue_range_histogram`.
        """
        self.pair_alignment = on

    def set_histogram_atomic_probe(
        mut self, on: Bool, acknowledge_wrong_histogram: Bool
    ) raises:
        """**The histogram this produces is wrong on purpose.** Turn the row
        loop's three shared atomics off and accumulate a per-thread sink
        instead.

        This is an instrument, not an arm. Nothing may ship on it, nothing
        may train on it, and no result taken from a fit that had it set means
        anything. Its entire output is one ratio: what share of the histogram
        phase the three `Atomic.fetch_add` calls per (row, feature) are,
        against the row-side loads, the quantized gradient load, the bin
        gather and the shared-cell address arithmetic that feed them --
        every one of which the probe still performs. `_hist_rows_step` lists
        what is kept and what is dropped, item by item, and how the kept work
        was shown not to have been optimized away.

        **Why the question is worth an instrument.** Two lanes in this
        campaign independently revised memory-side estimates downward and
        named the same reason: the three shared atomics are untouched by any
        layout or address change, so they bound every such estimate from
        above, and nobody has measured their share. A layout lane's 1.10x is
        an upper bound on wall clock only while the atomics' share is
        unknown. This method makes it known.

        **Which way the number errs.** It is a **lower bound** on what
        removing atomic contention could buy, and an upper bound on nothing.
        Three reasons, all in the same direction:

        1. The probe keeps the shared-cell index `lift + bin`, which a real
           atomics change (per-node rescaling to pack two counters into one
           word, or threadgroup privatization) would also keep. So the probe
           does not credit itself with work such a change would not remove
           either.
        2. The probe pays for its own sink: one XOR and one or two integer
           adds per (row, feature). Those land in the residual, so the
           residual is larger than a true atomic-free loop's and the atomics'
           share comes out smaller.
        3. Removing the atomics also removes the shared-plane zeroing and the
           flush's dependence on them, and the probe removes neither.

        What it therefore licenses: if the fraction is large, the next kernel
        lane goes at the atomics and the layout estimates should be read as
        the small numbers they are. If the fraction is small, no atomics
        change can be worth its risk and the next lane goes at addresses.
        What it does not license: quoting `1 / (1 - fraction)` as a speedup.
        Nothing here builds an atomic-free histogram that is correct, so
        nothing here has measured what one would cost.

        **The refusals, and why they are shaped like this.** The
        acknowledgment argument has no default, so no existing call site can
        acquire this flag by recompiling and no new one can acquire it
        without writing the words down. Beyond that, two entry points refuse
        outright while it is set: `enqueue_desc_histogram`, which is every
        histogram the device-resident growth plane builds, and any
        `enqueue_range_histogram` that folds a sibling subtraction, which is
        every non-root node the host plane builds. Between them, a fit cannot
        get past its root with this on. That is the closest thing to "refuses
        to run under any configuration that would produce a model" this
        struct can enforce from inside itself, and it is stated as a bound
        rather than a guarantee: a caller determined to build one root
        histogram and call it a model is not something a flag can stop.

        Takes effect on the next histogram launch.
        """
        if on and not acknowledge_wrong_histogram:
            raise Error(
                "the histogram atomics probe builds a WRONG histogram by"
                " design; pass acknowledge_wrong_histogram=True to say so in"
                " the call, and never on a fit whose model anyone will read"
            )
        if on and self.hist_probe_mode != HIST_PROBE_OFF:
            raise Error(
                "another histogram probe arm is already selected; the arms"
                " are exclusive because two of them at once is a kernel"
                " nobody described -- clear it with"
                " set_histogram_probe_mode(HIST_PROBE_OFF, True) first"
            )
        self.hist_atomic_probe = on

    def histogram_probe_active(self) -> Bool:
        """Whether any arm of the histogram decomposition is selected.

        One predicate over both fields, so that the entry points that refuse
        a wrong histogram refuse all of them and cannot be extended with a
        third arm and left behind. True means **this instance builds
        histograms that are wrong**.
        """
        return self.hist_atomic_probe or self.hist_probe_mode != HIST_PROBE_OFF

    def set_histogram_probe_mode(
        mut self, mode: Int, acknowledge_wrong_histogram: Bool
    ) raises:
        """**Every mode above `HIST_PROBE_OFF` produces a wrong histogram on
        purpose.** Select one of the two decomposition arms this lane added.

        This is an instrument, not an arm. Nothing may ship on it, nothing
        may train on it, and no result taken from a fit that had it set means
        anything. Its entire output is a wall-clock ratio against the full
        kernel at the identical launch geometry.

        **Why these two arms exist.** The atomic-fraction lane measured the
        three shared atomics at 9.8% of the histogram phase at
        1,000,000 x 50 x 256 bins, and a separate lane closed the address
        question by showing the feature-blocked layout is a no-op at the
        shipping configuration. Between them that accounts for about a tenth
        of the phase, and every kernel proposal on the table aimed at one of
        those two small parts. These two arms cut the remaining nine tenths:

        - `HIST_PROBE_NO_GATHER` drops the row load, the gradient-pair load
          and the bin gather and keeps the atomics, the trip count, the
          zeroing, the barrier and the flush. Its ratio is the loop's
          **memory traffic** share.
        - `HIST_PROBE_EMPTY` returns from the kernel as soon as its
          threadgroup planes exist. Its ratio is the **launch and occupancy**
          share, times however many launches a round issues.

        **What the fractions decide, and this is the point of the lane.**
        Gather-bound sends the next lane at bit-packed bins -- an
        `ellpack`-style `ceil(log2(n_bins))` bits per feature -- and at
        Float16 or packed gradient staging. Reduce-bound sends it at the tile
        count and the reduction shape, where the arms already exist and where
        an earlier 80-tile experiment measured 22% slower at 50 features and
        36% at 100, with 12.3 MB of partials per node histogram linear in
        tile count as the registered explanation. Launch-bound sends it at
        fusion, at the speculative prebuild that is already built and
        unmeasured, and at issuing fewer launches per round.

        **Which way each number errs** is written at the arm: at
        `_hist_rows_step` for the gather arm, which is bracketed rather than
        one-signed, and at `_hist_probe_empty_mark` for the empty arm, which
        bounds launch overhead from above and fixed per-launch cost from
        below.

        **What none of them licenses.** Quoting `1 / (1 - fraction)` as a
        speedup, or reading the four arms as a partition that sums to one.
        Each arm removes a part; what is left over is a residual that
        contains everything every arm keeps, and no arm here builds a
        histogram that is correct, so no arm here has measured what any real
        change would cost.

        **The refusals.** The acknowledgment has no default, so no existing
        call site can acquire an arm by recompiling. The arms are mutually
        exclusive with `set_histogram_atomic_probe`, because two arms at once
        is a kernel nobody described. And `enqueue_desc_histogram` and any
        subtracting `enqueue_range_histogram` refuse outright while any arm
        is set, which between them is every histogram the device-resident
        growth plane builds and every non-root node the host plane builds --
        so a fit cannot get past its root with one of these on.

        Takes effect on the next histogram launch.
        """
        if mode != HIST_PROBE_OFF and not acknowledge_wrong_histogram:
            raise Error(
                "the histogram decomposition probes build a WRONG histogram"
                " by design; pass acknowledge_wrong_histogram=True to say so"
                " in the call, and never on a fit whose model anyone will"
                " read"
            )
        if (
            mode != HIST_PROBE_OFF
            and mode != HIST_PROBE_NO_GATHER
            and mode != HIST_PROBE_EMPTY
        ):
            # `HIST_PROBE_NO_ATOMICS` is deliberately not accepted here: it
            # has its own setter, its own field and its own lane's argument,
            # and letting it in through two doors would leave the two fields
            # able to disagree about which arm is live.
            raise Error(
                "unknown histogram probe mode; the modes this setter accepts"
                " are HIST_PROBE_OFF, HIST_PROBE_NO_GATHER and"
                " HIST_PROBE_EMPTY, and HIST_PROBE_NO_ATOMICS is reached"
                " through set_histogram_atomic_probe instead"
            )
        if mode != HIST_PROBE_OFF and self.hist_atomic_probe:
            raise Error(
                "the histogram atomics probe is already selected; the arms"
                " are exclusive because two of them at once is a kernel"
                " nobody described -- clear it with"
                " set_histogram_atomic_probe(False, True) first"
            )
        self.hist_probe_mode = mode

    def set_row_tiling(
        mut self, min_tiles: Int = 0, rows_per_tile: Int = 0
    ) raises:
        """Request a row-tile floor, a rows-per-tile length, or neither.

        Zero, the default for both, means "no request" and reproduces
        exactly the geometry this module chose before the fields existed:
        `gpu_tiling.resolve_tiling` then falls back to
        `MOJOTREES_GPU_MIN_TILES` and `MOJOTREES_GPU_ROW_TILE`, which are
        themselves unset by default.

        `min_tiles` is a floor and is still clamped by row amortization, by
        the partial-buffer budget, and by `MAX_GRID_DIM_Y`, and the
        occupancy term stays a floor underneath it, so it can raise the tile
        count and never lower it. `rows_per_tile` sets the tile length
        directly and is the only way to ask for *fewer* tiles than the
        occupancy term gives, which is the arm the re-test needs: the
        earlier tile experiment only ever moved in one direction and
        therefore could not tell a bad floor from a bad gradient. Passing
        the node's whole row count is how one tile is spelled.

        Neither can change a histogram. Tiling picks a launch geometry;
        accumulation is fixed-point Int32 and integer addition is
        associative and commutative, so two geometries over the same rows
        sum the same bins in a different order to the same value. That is
        the property `gpu_tiling`'s module docstring states and the strategy
        tests assert bit-exactly.

        Takes effect on the next `range_tiling`, which is where a node's
        geometry is derived.
        """
        if min_tiles < 0:
            raise Error("a row-tile floor is zero (no request) or positive")
        if rows_per_tile < 0:
            raise Error("a row tile is zero (no request) or positive")
        self.min_tiles_request = min_tiles
        self.rows_per_tile_request = rows_per_tile

    def blocked_row_stride(self) -> Int:
        """`G` when a blocked bin layout is in force and built, else 1.

        The one value that selects the histogram reader's layout, and the
        only thing about the layout the launch sites read. It stays 1 until
        `_ensure_blocked` has actually filled the buffer, so a requested but
        unbuilt layout reads the feature-major matrix rather than an
        uninitialized one -- which is the failure this returns a number
        rather than a flag to prevent: an unbuilt blocked buffer would decode
        to legal bin ids belonging to no row and would be caught by nothing.
        """
        if self.blocked_group > 1 and self.blocked_valid:
            return self.blocked_group
        return BLOCKED_STRIDE_NONE

    def set_blocked_layout(mut self, group: Int) raises:
        """Ask the histogram kernels to read a `[block][row][G]` bin matrix.

        `group` of zero turns the layout off and frees the buffer. Otherwise
        it must equal `feature_group`, and the refusal is the point of the
        method rather than a guard on it: the whole mechanism is that a
        threadgroup consumes every one of the `G` adjacent bytes it pulls for
        a row, and a threadgroup consumes `feature_group` slots. A mismatch
        produces a **correct** histogram at a different cost, so it is not
        something a measurement would find; see
        `gpu_blocked_bins.check_blocked_group_matches` for the two directions
        it can be wrong in and what each would look like.

        `G = 1` is refused for a different reason: a one-feature block is a
        feature's column, so the arrangement is the one already on the device
        and a second copy of it can only cost residency. Since
        `free_feature_group` returns 1 at `bin_cap = 256`, **the shipping
        default at every headline shape this project benchmarks cannot use
        this layout at all** without `set_feature_group(2)` or wider first,
        which is a residency trade `gpu_tiling.free_feature_group` says in as
        many words is unmeasured. That is a finding about the layout and not
        an obstacle to it, and it belongs where a caller reads it.

        **What it costs.** One `n_rows * padded_features` byte allocation, on
        top of the feature-major matrix, which stays because every other
        reader of `bins` in this package still indexes it. At 1,000,000 x 50
        with `G = 4` that is 52 MB on top of 50 MB. The transform itself is
        one streamed kernel launch per fit, deferred to the first histogram
        (`_ensure_blocked`) so that requesting a layout and never building a
        tree costs an allocation and no device work.

        No environment variable, for the reason `partition_block_cap` gives:
        the arm belongs in the call, because an A/B that reads its arm from
        the environment has already once in this repository run one arm under
        the other's label.
        """
        if group == 0:
            self.blocked_dev = self.ctx.enqueue_create_buffer[DType.uint8](1)
            self.blocked_group = 0
            self.blocked_valid = False
            return
        if len(self.packed_widths) > 0:
            raise Error(
                "the packed and blocked bin layouts cannot both be on; turn"
                " one off with set_packed_bins([]) or set_blocked_layout(0)"
            )
        check_blocked_group_matches(group, self.feature_group)
        if blocked_is_identity(group):
            raise Error(
                "a blocked bin layout at G = 1 is the feature-major layout"
                " already on the device; widen the feature group first"
            )
        if group == self.blocked_group:
            return
        # Held mutually exclusive with row compaction, from both sides. Both
        # arms re-arrange the same matrix and a histogram reads one pointer,
        # so a state in which both are on is a state in which one of them is
        # silently ignored. See `set_row_compaction`.
        if group >= 2 and self.row_compaction:
            raise Error(
                "the blocked bin layout and row compaction both re-arrange"
                " the binned matrix and a histogram cannot read both; turn"
                " one off"
            )
        var total = blocked_bytes(self.n_rows, self.n_features, group)
        self.blocked_dev = self.ctx.enqueue_create_buffer[DType.uint8](total)
        self.blocked_group = group
        self.blocked_valid = False

    def _ensure_blocked[
        bins_origin: MutOrigin, //
    ](mut self, bins: MutPointer[UInt8, bins_origin]) raises:
        """Build the blocked buffer from the feature-major one, once.

        Deferred to the first histogram launch rather than done in
        `set_blocked_layout` because that is where the `bins` pointer is: this
        struct owns the index machinery and the binned matrix arrives as an
        argument (see the struct docstring). Enqueued, not synchronized, so it
        is ordered before the histogram that reads it by the queue, exactly as
        `_ensure_quantized`'s pass is.

        Never rebuilt. The binned matrix is uploaded once per fit and nothing
        in this backend mutates it, so unlike the quantized gradients there is
        no per-tree quantity for this to go stale against. A caller that
        changes the matrix under a live `GpuActiveRows` is already outside
        what every other buffer here assumes.
        """
        if self.blocked_group < 2 or self.blocked_valid:
            return
        enqueue_blocked_relayout(
            self.ctx,
            bins,
            self.blocked_dev.unsafe_ptr(),
            self.n_rows,
            self.n_features,
            self.blocked_group,
            self.block_threads,
        )
        self.blocked_valid = True

    def packed_bins_active(self) -> Bool:
        """Whether the histogram reader is decoding bit-packed bins.

        The one value the launch sites read about this layout, and it is
        false until `_ensure_packed` has actually filled the buffer. A
        requested-but-unbuilt packed buffer would decode to legal bin ids
        belonging to no row, which is the failure mode this layout family
        shares and which nothing downstream would catch, so the flag tracks
        the buffer rather than the request.
        """
        return len(self.packed_widths) > 0 and self.packed_valid

    def packed_bin_widths(self) -> List[Int]:
        """The per-feature storage widths in force, or an empty list when the
        packed layout is off. A copy, so a caller cannot narrow a width under
        a live buffer."""
        return self.packed_widths.copy()

    def set_packed_bins(mut self, var widths: List[Int]) raises:
        """Ask the histogram kernels to read a bit-packed bin matrix.

        An empty `widths` turns the layout off and frees both buffers.
        Otherwise it is one width per feature, in bits, and
        `gpu_packed_bins.packed_widths_from_matrix` is what derives a correct
        one: a width has to cover its feature's largest bin id, its reserved
        missing bin, and, for a categorical feature, its highest category bin.
        All three fail *silently* if the width is short -- a truncated bin id
        is a legal bin id -- so the derivation is not a convenience and a
        hand-written table should be checked with
        `gpu_packed_bins.check_packed_widths_cover` against the matrix it will
        be used on.

        **All widths 8 is refused**, because that plan's buffer is
        `BinnedMatrix.bins` itself: a 255-bin feature needs eight bits and
        packing it is the identity. So a matrix whose every column is
        high-cardinality cannot use this layout at all, which is a finding
        about the layout rather than an obstacle to it, and it is why the
        reference shape this project benchmarks -- 1,000,000 x 50 at 255 bins
        -- gains nothing here. What gains is a matrix with low-cardinality
        columns in it, at `width / 8` of the bin bytes per column.

        **The blocked layout is refused while this one is on**, and the other
        way round. The two are orthogonal in principle -- one narrows the
        cell, the other rearranges it -- and composing them would need a block
        base, a lane, a row stride and a bit width in one address. Neither arm
        has been measured yet; building the cross product before either is
        would be three unmeasured things instead of two.

        **What it costs.** One `gpu_packed_bins.packed_bytes` allocation plus
        an `8 * n_features` byte table, on top of the feature-major matrix,
        which stays because every other reader of `bins` in this package still
        indexes it. Unlike the blocked copy that allocation is *smaller* than
        the matrix it shadows, by exactly the ratio the layout exists to
        deliver. The transform is one streamed kernel launch per fit, deferred
        to the first histogram (`_ensure_packed`) so that requesting a layout
        and never growing a tree costs an allocation and no device work.

        No environment variable, for the reason `set_blocked_layout` gives:
        an A/B that reads its arm from the environment has already once in
        this repository run one arm under the other's label.
        """
        if len(widths) == 0:
            self.packed_dev = self.ctx.enqueue_create_buffer[DType.uint8](1)
            self.packed_tab_dev = self.ctx.enqueue_create_buffer[DType.int32](
                1
            )
            self.packed_widths = List[Int]()
            self.packed_valid = False
            return
        if len(widths) != self.n_features:
            raise Error(
                "a packed bin layout needs one width per feature of the"
                " dataset this instance was built for"
            )
        check_packed_widths(widths)
        if packed_is_identity(widths):
            raise Error(
                "a packed bin layout at eight bits throughout is the"
                " feature-major matrix already on the device; a matrix with"
                " no low-cardinality column has nothing to pack"
            )
        if self.blocked_group > 0:
            raise Error(
                "the packed and blocked bin layouts cannot both be on; turn"
                " one off with set_blocked_layout(0) or set_packed_bins([])"
            )
        # And row compaction, from this side, so the two orders of the same
        # mistake reach the same refusal. Not orthogonal-in-principle the way
        # the blocked pair is: this one narrows a cell to fewer bits and that
        # one moves whole cells into permutation order, and a compacted bit
        # stream would mean re-encoding packed runs under a permutation.
        # Composing them is a project, and neither has been measured yet.
        if self.row_compaction:
            raise Error(
                "the packed bin layout and row compaction cannot both be on:"
                " the compacted plane is one byte per cell and this one is a"
                " bit stream; turn one off with set_row_compaction(False) or"
                " set_packed_bins([])"
            )
        var total = packed_bytes(self.n_rows, widths)
        var table = packed_table(self.n_rows, widths)
        self.packed_dev = self.ctx.enqueue_create_buffer[DType.uint8](total)
        self.packed_tab_dev = self.ctx.enqueue_create_buffer[DType.int32](
            len(table)
        )
        # The table is small (two Int32 per feature) and is uploaded here
        # rather than in `_ensure_packed`, because it is pure host arithmetic
        # that needs no `bins` pointer. Synchronized before the staging buffer
        # leaves scope, which is once per fit against a copy of a few hundred
        # bytes.
        var host = self.ctx.enqueue_create_host_buffer[DType.int32](
            len(table)
        )
        var dst = host.unsafe_ptr()
        for i in range(len(table)):
            dst.unsafe_store(i, table[i])
        self.ctx.enqueue_copy(dst_buf=self.packed_tab_dev, src_ptr=dst)
        self.ctx.synchronize()
        self.packed_widths = widths^
        self.packed_valid = False

    def _ensure_packed[
        bins_origin: MutOrigin, //
    ](mut self, bins: MutPointer[UInt8, bins_origin]) raises:
        """Build the packed buffer from the feature-major one, once.

        Deferred to the first histogram launch rather than done in
        `set_packed_bins` because that is where the `bins` pointer is: this
        struct owns the index machinery and the binned matrix arrives as an
        argument. Enqueued, not synchronized, so it is ordered before the
        histogram that reads it by the queue, exactly as `_ensure_blocked`'s
        pass is.

        Never rebuilt, for the same reason: the binned matrix is uploaded once
        per fit and nothing in this backend mutates it.
        """
        if len(self.packed_widths) == 0 or self.packed_valid:
            return
        enqueue_packed_pack(
            self.ctx,
            bins,
            self.packed_dev.unsafe_ptr(),
            self.packed_tab_dev.unsafe_ptr(),
            self.n_rows,
            self.n_features,
            packed_max_stream_bytes(self.n_rows, self.packed_widths),
            self.block_threads,
        )
        self.packed_valid = True
    # --- Row compaction ----------------------------------------------------

    def set_row_compaction(mut self, on: Bool) raises:
        """Turn CatBoost-style physical row compaction on or off.

        **Off by default and it stays off until a window resolves it.** This
        is a trade and not a saving: it removes the scattered gather from the
        histogram and pays a physical reorder of the binned matrix at every
        split to do it. The repository has been burned by exactly this shape
        of change before -- the row-tile floor raised occupancy as designed
        and measured 22 percent slower at 50 features -- so the arm ships
        reachable, default off, and is not argued into the default here.

        **What it costs to say yes**, and it is stated where it is asked for,
        on the footing `set_blocked_layout` states its residency cost:
        `2 * n_rows * n_features` bytes of device memory for the binned matrix
        and its scatter destination, plus `16 * n_rows` for the quantized
        gradient pair and its destination, plus `4 * n_rows` for the identity
        index. At 1,000,000 rows by 50 features that is 100 MB plus 16 MB plus
        4 MB, allocated here and held for the life of the instance. Turning
        the arm back off does not free them, deliberately: a benchmark that
        interleaves arms would otherwise pay an allocation inside the window
        it is timing.

        **Two preconditions, both refused rather than worked around.**

        - The quantized gradient arm must be on. The compacted planes hold
          `cgq`, not the Float32 `grad` and `hess` planes, because the row
          index feeds *both* gathers in the accumulation loop and an identity
          index against un-compacted gradient planes would read the wrong
          row's gradient. Compacting the Float32 planes as well is possible
          and is not built: it is two more buffers and 8 more bytes per row
          for the arm this backend does not default to.
        - The blocked bin layout must be off. Both arms re-arrange the same
          matrix and the blocked buffer is built from `bins` by row id, so a
          histogram cannot read both. They are held mutually exclusive here
          and in `set_blocked_layout`, which is the same discipline the two
          histogram probe modes are held under.
        - The packed bin layout must be off, and this one is not a
          composability question that a later lane can just wire up. `cbins`
          is one byte per cell; the packed plane is a bit stream at a
          per-feature width. Permuting a bit stream is a re-encode, not a
          move, so there is no plane both readers could share. Refused here
          and in `set_packed_bins`.

        **The gradient-staging lane's second width is NOT refused**, and the
        difference is worth stating because it is the whole reason these two
        neighbours are treated differently. `set_packed_gradients` narrows the
        staged pair to Int16 inside the same allocation, which is a width and
        not an encoding: the compacted copy simply moves four bytes per row
        instead of eight. So the three compaction kernels take a `packed`
        argument, `compact_packed` records which width the planes hold, and
        `row_compaction_live` refuses to read them while that disagrees with
        what the next staging would produce.

        Enabling always invalidates the planes, so the first histogram after
        this call rebuilds them. Disabling invalidates them too, so that a
        re-enable cannot pick up planes that went stale while the arm was off.
        """
        if on:
            if not self.quantized_gradients:
                raise Error(
                    "row compaction needs the quantized gradient arm: the"
                    " compacted planes hold the interleaved quantized pair,"
                    " and the row index feeds the gradient gather as well as"
                    " the bin gather"
                )
            if self.blocked_group >= 2:
                raise Error(
                    "row compaction and the blocked bin layout both"
                    " re-arrange the binned matrix and a histogram cannot"
                    " read both; turn one off"
                )
            if len(self.packed_widths) > 0:
                raise Error(
                    "row compaction and the packed bin layout cannot both be"
                    " on: the compacted plane is one byte per cell and the"
                    " packed plane is a bit stream, so a histogram cannot read"
                    " both; turn one off with set_packed_bins([]) or"
                    " set_row_compaction(False)"
                )
            if not self.compact_allocated:
                self.cbins_dev = self.ctx.enqueue_create_buffer[DType.uint8](
                    self.n_rows * self.n_features
                )
                self.cbins_alt_dev = self.ctx.enqueue_create_buffer[
                    DType.uint8
                ](self.n_rows * self.n_features)
                self.cgq_dev = self.ctx.enqueue_create_buffer[DType.int32](
                    2 * self.n_rows
                )
                self.cgq_alt_dev = self.ctx.enqueue_create_buffer[DType.int32](
                    2 * self.n_rows
                )
                self.ident_dev = self.ctx.enqueue_create_buffer[DType.int32](
                    self.n_rows
                )
                # Guarded 2026-08-17 for the same reason as `begin_tree`: on a
                # CPU-only build any reachable `enqueue_function` elaborates a
                # GPU kernel and fails the compile, whatever the kernel does.
                # `__init__` reaches this whenever
                # MOJOTREES_GPU_ROW_COMPACTION is set, so it is reachable from
                # construction and not only from a caller who asked for the
                # arm by name.
                #
                # Narrower than the whole-body wrap the four entry points use,
                # and deliberately so. A `comptime if` with an `else` prunes
                # the untaken branch at compile time wherever it sits, so
                # wrapping the launch is sufficient to stop elaboration; the
                # whole-body form exists because an EARLY RETURN does not
                # prune, not because the wrap has to be maximal. Keeping it
                # tight here leaves the allocation and bookkeeping above
                # readable as one block.
                comptime if not has_accelerator():
                    raise Error(
                        "row compaction needs an accelerator; this binary was"
                        " built without one, so leave"
                        " MOJOTREES_GPU_ROW_COMPACTION unset"
                    )
                else:
                    var threads = self.block_threads
                    var blocks = (self.n_rows + threads - 1) // threads
                    self.ctx.enqueue_function[_iota_kernel](
                        self.ident_dev.unsafe_ptr(),
                        Int32(self.n_rows),
                        grid_dim=blocks,
                        block_dim=threads,
                    )
                self.compact_allocated = True
        self.row_compaction = on
        self.compact_valid = False

    def row_compaction_requested(self) -> Bool:
        """Whether the arm is on. Not the same question as whether a histogram
        will read the compacted planes; see `row_compaction_live`."""
        return self.row_compaction

    def row_compaction_live(self) -> Bool:
        """Whether the next histogram launch will read the compacted planes.

        The arm has to be on, the invariant has to currently hold, and the
        quantized gradient arm has to be on. This is the one predicate the
        launch sites test, so that "the planes are stale", "the arm is off"
        and "the gradient plane the compaction holds is not the one the
        kernel would read" all reach the same, always-correct, fallback to
        `bins` and `rows_dev`.

        The third conjunct is why `set_quantized_gradients` did not have to
        grow a refusal and a `raises`: withdrawing the quantized arm withdraws
        compaction with it at the launch, rather than leaving a state in which
        an identity index is read against un-compacted gradient planes.

        The fourth is the same discipline against the gradient-staging lane's
        second width. `compact_packed` is the width `cgq_dev` was written at
        and `packed_gradients` is the width the next staging would use; while
        they disagree the compacted plane is a valid encoding of the wrong
        thing, and the only safe reading of it is not to read it. They agree
        again on the next `_ensure_compacted`, which the histogram entry points
        call after `_ensure_quantized` has settled the width.

        The packed *bin* layout is not a conjunct here and is refused at the
        setter instead, because unlike these two it cannot be made to agree:
        `cbins` is a byte-per-cell plane and the packed layout is a bit stream,
        so there is no state in which both are readable. See
        `set_row_compaction`.
        """
        return (
            self.row_compaction
            and self.compact_valid
            and self.quantized_gradients
            and self.compact_packed == self.packed_gradients
        )

    def set_compact_flag_read(mut self, on: Bool):
        """Turn the compacted flag read on or off in process.

        The in-process handle for `MOJOTREES_GPU_COMPACT_FLAG_READ`, on the
        footing `set_scan_primitives` is the handle for
        `MOJOTREES_GPU_SCAN_PRIMITIVES`: the variable decides the default and
        this overrides it, so one benchmark process can alternate the arms
        instead of re-execing and hoping the label matched the arm.

        No refusal and no `raises`, because there is no state this can put the
        instance into that a launch would then have to cope with.
        `compact_flag_read_live` conjoins this with `row_compaction_live`, so
        asking for it while the compaction arm is off, or while its planes are
        stale, simply reaches the gathering flag pass that every fit before
        this arm existed reached.

        Takes effect on the next partition. Nothing is invalidated, because
        this changes which plane a *read* comes from and writes nothing.
        """
        self.compact_flag_read = on

    def compact_flag_read_requested(self) -> Bool:
        """Whether the arm was asked for. Not whether a partition will take it;
        see `compact_flag_read_live`."""
        return self.compact_flag_read

    def compact_flag_read_live(self) -> Bool:
        """Whether the next partition's flag pass reads the compacted plane.

        The arm has to be on and the compacted planes have to be live, and the
        second conjunct is the load-bearing one. `row_compaction_live` is the
        predicate the histogram launch already tests, so this arm engages on
        exactly the launches the histogram's identity-index arm engages on and
        on no others.

        **Why the read is bit-identical, which is the whole argument.** The
        compaction invariant is `cbins[f * n_rows + j] == bins[f * n_rows +
        rows[j]]` for every position `j` of the row buffer, established by
        `_compact_build_kernel` and maintained across every split by
        `_compact_scatter_kernel` applying the row scatter's own permutation.
        The flag pass computes, for position `row_begin + j` of the range, the
        routing of `bins[feature * n_rows + rows[row_begin + j]]`. Under the
        invariant that byte *is* `cbins[feature * n_rows + row_begin + j]`, so
        the arm substitutes one spelling of one byte for another. The flag it
        derives, the prefix that flag feeds, the packed offset, the block sums,
        the permutation and therefore every histogram downstream are unchanged
        value for value. This is the same form of argument the histogram's own
        compacted launch rests on, and it is an argument about addresses rather
        than about the order of a sum.

        **What it is worth, and it is a count and not a timing.** The module
        docstring calls the bin load "the one random-access read of the
        partition". Under this arm the flag pass reads `cbins` at
        `row_begin + j` for consecutive `j`, which is a dense run, and it does
        not read `rows` at all: two loads per row, one of them a scattered
        byte, become one contiguous byte. Nothing here has been measured.
        """
        return self.compact_flag_read and self.row_compaction_live()

    def compaction_packed_gradients(self) -> Bool:
        """Which staged gradient width the compacted planes currently hold.

        Not the same question as `packed_gradients_on()`, which is what the
        next staging would produce; while the two disagree the planes are not
        read at all. Exposed so a test can decode `download_compacted_grads`
        at the width it was written at rather than guessing.
        """
        return self.compact_packed

    def compaction_counts(self) -> Tuple[Int, Int]:
        """Full rebuilds and incremental scatters issued, ever.

        Launches enqueued, not launches predicted. A test asserts the arm
        engaged on this rather than on the arm having been requested, which
        is the difference between checking a switch and checking a wire.
        """
        return (self.compact_builds, self.compact_scatters)

    def _ensure_compacted[
        bins_origin: MutOrigin, //
    ](mut self, bins: MutPointer[UInt8, bins_origin]) raises:
        """Re-establish the compaction invariant if it does not hold.

        Called by both histogram entry points, after `_ensure_quantized` and
        after any deferred copy-back has been enqueued: it reads `gq` and it
        reads `rows_dev`, so both have to be current in the queue before it.
        Ordering is by the queue, not by a synchronize; nothing here waits.

        A no-op on every launch but the first of a tree, because every
        partition maintains the invariant incrementally. When it is *not* a
        no-op mid-tree -- a gradient rescale invalidated `gq`, say -- it is a
        full pass over the buffer, which is slow and correct rather than fast
        and conditional.

        **It builds from whichever `bins` the first histogram of the tree was
        handed**, and every later launch of that tree reads the plane built
        from it. That rests on the same statement `_ensure_blocked` rests on:
        the binned matrix is uploaded once per fit and nothing in this backend
        mutates it, so there is one matrix and the pointer identifies it. A
        caller that handed two different matrices to one `GpuActiveRows`
        inside one tree is already outside what the blocked layout assumes as
        well, and would be outside what `n_rows` and `n_features` assume.
        """
        if not self.row_compaction or self.compact_valid:
            return
        var threads = self.block_threads
        var blocks = (self.n_rows + threads - 1) // threads
        self.ctx.enqueue_function[_compact_build_kernel](
            bins,
            self.gq_dev.unsafe_ptr(),
            self.rows_dev.unsafe_ptr(),
            self.cbins_dev.unsafe_ptr(),
            self.cgq_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.n_features),
            # The width the buffer actually holds, not the width that was
            # asked for. `_ensure_quantized` ran immediately before this and
            # settled `quant_packed`; reading `packed_gradients` here instead
            # would be the one spelling that could disagree with the plane
            # this kernel is copying from.
            Int32(1) if self.quant_packed else Int32(0),
            grid_dim=blocks,
            block_dim=threads,
        )
        self.compact_valid = True
        self.compact_packed = self.quant_packed
        self.compact_builds += 1

    def _maintain_compaction(
        mut self,
        begin: Int,
        count: Int,
        tiles: Int,
        blocks: Int,
        back_blocks: Int,
        use_desc: Int32,
    ) raises:
        """Keep the compacted planes in step with a partition, or admit that
        they no longer are.

        The **else branch is the load-bearing half** and it is why this is a
        method rather than an `if` at each partition. A partition that runs
        while the planes are not live moves the permutation and leaves them
        describing the tree before the split. If the reason they were not live
        was transient -- `set_packed_gradients` flipped the staged width, say,
        and nothing has restaged yet -- then a later flip back would make
        `row_compaction_live` true again over planes that had silently missed
        a split, and the histogram after it would be a correct sum over the
        wrong rows. Invalidating here means the only way back to live is
        through `_ensure_compacted`, which rebuilds from `rows_dev` and cannot
        inherit a missed split.

        Costs nothing when the arm was never requested, which is the default:
        one field test and no launch.
        """
        if not self.row_compaction:
            return
        if self.row_compaction_live():
            self._enqueue_compact_partition(
                begin, count, tiles, blocks, back_blocks, use_desc
            )
            return
        self.compact_valid = False

    def _enqueue_compact_partition(
        mut self,
        begin: Int,
        count: Int,
        tiles: Int,
        blocks: Int,
        back_blocks: Int,
        use_desc: Int32,
    ) raises:
        """Maintain the invariant across one partition, in two launches.

        Enqueued from inside both partition entry points, after the flag pass
        that wrote `offsets` and `block_sums` and beside the row scatter that
        reads them. It must be given the **same** `blocks` and `tiles` the flag
        pass and the row scatter were given, because the packed offsets are
        chunk-relative and a different chunking would read a packed offset
        against the wrong chunk's block sum. That is the same coupling
        `_enqueue_scan_primitives` records for the two kernels it launches.

        Not folded into anything. The partition tail fusion pairs the row
        copy-back with the next histogram's slot zeroing; this pair is not a
        candidate for it, because the second launch here reads what the first
        writes and a fold would have to preserve that order across a command
        buffer boundary the fusion exists to remove.

        **A WHOLE-BUFFER PING-PONG REPLACED THE SECOND LAUNCH AND WAS REMOVED
        WITH THE ARM THAT NEEDED IT.** `set_compact_swap(True)` used to
        exchange the two allocations here instead of enqueuing the copy-back,
        which is legal only for a caller whose every partition covers the whole
        live prefix, and the one such caller was the oblivious level schedule
        reading the compacted plane. That read measured 0.757x on real data and
        was removed on 2026-08-18, so the swap had no caller left and went with
        it; see `docs/design/DECLINED_OPTIMIZATIONS.md` row C1. A leaf-wise
        partition moves ONE leaf's window, and swapping the buffers under it
        would strand every other live leaf's compacted rows in the plane that
        was swapped out, which is why the copy-back is the only arm here.
        """
        var desc = self._desc_buffer()
        self.ctx.enqueue_function[_compact_scatter_kernel](
            self.cbins_dev.unsafe_ptr(),
            self.cbins_alt_dev.unsafe_ptr(),
            self.cgq_dev.unsafe_ptr(),
            self.cgq_alt_dev.unsafe_ptr(),
            self.offsets_dev.unsafe_ptr(),
            self.block_sums_dev.unsafe_ptr(),
            Int32(begin),
            Int32(count),
            Int32(tiles),
            Int32(blocks),
            Int32(self.n_rows),
            Int32(self.n_features),
            # The width these planes were written at, held on the instance
            # rather than re-derived, because a partition can be enqueued
            # between a `set_packed_gradients` and the histogram that would
            # restage at the new width. `row_compaction_live` refuses to reach
            # here at all while the two disagree; passing the recorded width is
            # the second half of the same guard.
            Int32(1) if self.compact_packed else Int32(0),
            desc.unsafe_ptr(),
            use_desc,
            grid_dim=blocks,
            block_dim=self.block_threads,
        )
        self.ctx.enqueue_function[_compact_copy_back_kernel](
            self.cbins_dev.unsafe_ptr(),
            self.cbins_alt_dev.unsafe_ptr(),
            self.cgq_dev.unsafe_ptr(),
            self.cgq_alt_dev.unsafe_ptr(),
            Int32(begin),
            Int32(count),
            Int32(self.n_rows),
            Int32(self.n_features),
            Int32(1) if self.compact_packed else Int32(0),
            desc.unsafe_ptr(),
            use_desc,
            grid_dim=back_blocks,
            block_dim=self.block_threads,
        )
        self.compact_scatters += 1

    def download_compacted_bins(mut self) raises -> List[UInt8]:
        """The whole compacted bin plane, host side. Synchronizes.

        For the test that checks the invariant against `bins` and the
        permutation, and for nothing else: training never needs it on the
        host, and at the reference shape this is a 50 MB transfer.
        """
        if not self.compact_allocated:
            raise Error("row compaction was never enabled on this instance")
        var n = self.n_rows * self.n_features
        var host = self.ctx.enqueue_create_host_buffer[DType.uint8](n)
        self.ctx.enqueue_copy(
            dst_ptr=host.unsafe_ptr(), src_buf=self.cbins_dev
        )
        self.ctx.synchronize()
        var out = List[UInt8](capacity=n)
        var src = host.unsafe_ptr()
        for i in range(n):
            out.append(src.unsafe_load(i))
        return out^

    def download_compacted_grads(mut self) raises -> List[Int32]:
        """The whole compacted quantized-gradient plane. Synchronizes. Same
        audience as `download_compacted_bins`."""
        if not self.compact_allocated:
            raise Error("row compaction was never enabled on this instance")
        var n = 2 * self.n_rows
        var host = self.ctx.enqueue_create_host_buffer[DType.int32](n)
        self.ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.cgq_dev)
        self.ctx.synchronize()
        var out = List[Int32](capacity=n)
        var src = host.unsafe_ptr()
        for i in range(n):
            out.append(src.unsafe_load(i))
        return out^

    def download_quantized_grads(mut self) raises -> List[Int32]:
        """The un-compacted quantized-gradient plane, indexed by row id.

        The other half of the invariant check: `cgq[2j]` has to equal
        `gq[2 * rows[j]]`, so a test needs both planes.
        """
        var n = 2 * self.n_rows
        var host = self.ctx.enqueue_create_host_buffer[DType.int32](n)
        self.ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.gq_dev)
        self.ctx.synchronize()
        var out = List[Int32](capacity=n)
        var src = host.unsafe_ptr()
        for i in range(n):
            out.append(src.unsafe_load(i))
        return out^

    def set_constant_hessian(mut self, on: Bool):
        """Declare that every row's hessian this round is exactly
        `histogram.CONSTANT_HESSIAN`, so the histogram kernels may stop
        accumulating the hessian plane and reconstruct it from the count.

        **This is a declaration about the objective, and the caller owns
        it.** `histogram.objective_has_constant_hessian` is the predicate
        that answers it correctly, and it is false for every weighted fit and
        for every GOSS round regardless of the objective code, because both
        put more than one value into `hess`. Declaring it on a round where it
        is not true produces a wrong hessian plane, silently, so it belongs
        next to where the trainer decides whether to pass weights and not
        anywhere further down.

        An argument as well as an environment variable, for the reason
        `set_feature_group` gives: an A/B that reads its arm from the
        environment can run one arm under the other's label, which happened
        once in this repository, so a benchmark holding both arms in one
        process passes the arm in rather than re-execing.
        `MOJOTREES_CONST_HESSIAN=0` wins over an argument in the one
        direction that is always safe, off, and this method reports what it
        actually did through `constant_hessian`.

        Takes effect on the next `enqueue_range_histogram`. When the
        declaration is true it cannot change a histogram: the plane it stops
        accumulating is reconstructed as the identical Int32, argued in
        `_range_hist_atomic_kernel`.
        """
        self.constant_hessian = on and self.const_hessian_allowed

    def set_scan_primitives(mut self, on: Bool) raises:
        """Whether the partition's two scan stages run on `block.prefix_sum`
        or on the hand-rolled Hillis-Steele kernels.

        An argument as well as an environment variable for the reason
        `set_feature_group` gives: an A/B that reads its arm from the
        environment can run one arm under the other's label, which has
        happened in this repository once, so a benchmark holding both arms in
        one process passes the arm in instead of re-execing. Takes effect on
        the next `enqueue_partition`. It cannot change the permutation or the
        left count, only which kernel computes the prefix sums, and the two
        being identical is what tests/test_gpu_scan_primitives.mojo asserts.

        Turning it on at a threadgroup width no primitive instantiation
        exists for raises rather than silently leaving the fallback arm
        selected, because a benchmark told to run the primitive arm must not
        quietly measure the other one.
        """
        if on and not _scan_primitive_width_supported(self.block_threads):
            raise Error(
                "the primitive scan arm has no kernel at this threadgroup"
                " width: 128, 256, 512, and 1024 are the widths instantiated"
            )
        self.scan_primitives = on

    def set_partition_block_cap(mut self, cap: Int) raises:
        """The most threadgroups one partition may launch.

        A launch-shape knob in exactly the sense `set_feature_group` and
        `set_scan_primitives` are: it changes how the range is cut into
        chunks and therefore how much redundant block-sums scanning the
        scatter does, and it cannot change one element of the permutation or
        one unit of the left count, because a row's destination is a function
        of the routing flags and of the row's position in the range and of
        nothing else. `tests/test_gpu_partition_launches.mojo` asserts that
        across caps rather than leaving it as a claim here.

        Two reasons it is settable. A test needs to reach the multi-tile path
        without allocating a range of `block_threads * block_threads` rows, so
        it lowers the cap instead and gets the same code path on a range small
        enough to check against the serial reference model. And a benchmark
        that wants to know whether the default (`block_threads`, chosen by a
        counting argument in `_scatter_kernel` and by no measurement at all)
        is the right cap on a given device has to be able to hold two caps in
        one process.

        Takes effect on the next `enqueue_partition`. There is no upper bound
        beyond the one the block-sums buffer already implies: the scatter's
        head scan walks the sums in chunks of a threadgroup width, so it
        serves any block count, and `_partition_grid` never returns more
        blocks than the range has tiles.
        """
        if cap < 1:
            raise Error("partition block cap must be at least one block")
        self.partition_block_cap = cap

    def begin_tree(mut self, bag: List[Int] = []) raises:
        """Seed the root's rows and make node 0 own all of them.

        Unbagged, the root is the identity permutation, written by a kernel
        so a tree costs no host-to-device row transfer at all. Bagged, the
        bag's rows are staged and copied in the caller's order, which is the
        order the CPU grower's root list has, and the rows left out are
        simply not inside the root range: no sentinel leaf id, no filtering,
        and no cost per node for a row this tree ignores.
        """
        # GUARDED 2026-08-17, and this one was PREDICTED and left unguarded on
        # purpose, which is worth recording because the prediction was right.
        # The lane that guarded the four histogram and partition entry points
        # observed that `_iota_kernel` elaborated fine on the failing CPU-only
        # build, unlike the kernels that allocate shared memory, and reasoned
        # that `begin_tree` therefore did not need a wrap. CI disagreed: the
        # compiler emits ONE error stack and stops, so the first round of
        # guards only revealed the next site rather than proving there were
        # none. The chain CI named is
        # `histogram_gpu.mojo:1929` -> `:1940` -> here -> the launch below.
        #
        # The general lesson, which costs a CI cycle every time it is
        # relearned: on a CPU-only build ANY `enqueue_function` in a reachable
        # body is a compile hazard, not only the ones whose kernels look
        # expensive. An Apple machine can never reproduce it, so the rule has
        # to be applied by construction rather than by testing.
        comptime if not has_accelerator():
            raise Error(
                "seeding a tree's device rows needs an accelerator; this"
                " binary was built without one, so grow the tree through the"
                " CPU backend instead"
            )
        else:
            if len(bag) > self.n_rows:
                raise Error("bag is larger than the dataset")
            # A tree boundary with a copy-back still owed means the previous
            # tree's last partition was never paired with a histogram, which
            # is a broken schedule and not something to reseed over.
            self._refuse_copy_back_debt("starting a tree")
            # A new tree may carry a new round's gradients; the quantized copy
            # is rebuilt on the first histogram that asks for it.
            self.quant_valid = False
            # And so are the compacted planes, for two reasons at once: the
            # permutation is about to be reseeded, and the gradients they hold
            # are about to be stale. Rebuilt by `_ensure_compacted` on the
            # first histogram of this tree, which is one full pass per tree and
            # is the only scattered read the mechanism performs.
            self.compact_valid = False
            # The record, and the sink is looked up before the line is built so
            # that the default path pays one `getenv` and no String. See
            # `COMPACTION_TRACE_VAR` for why the record exists at all.
            var sink = _compact_trace_sink()
            if sink != "":
                _compact_trace_emit(
                    sink,
                    String(
                        "mojotrees.compaction tree arm=",
                        "on" if self.row_compaction else "off",
                        " builds=",
                        self.compact_builds,
                        " scatters=",
                        self.compact_scatters,
                        "\n",
                    ),
                )
            if len(bag) == 0:
                var blocks = (
                    self.n_rows + self.block_threads - 1
                ) // self.block_threads
                self.ctx.enqueue_function[_iota_kernel](
                    self.rows_dev.unsafe_ptr(),
                    Int32(self.n_rows),
                    grid_dim=blocks,
                    block_dim=self.block_threads,
                )
                self.ranges.reset_root(self.n_rows)
                return

            for i in range(len(bag)):
                if bag[i] < 0 or bag[i] >= self.n_rows:
                    raise Error("bag row index out of range")
            # Any copy still reading the staging buffer has to finish before it
            # is overwritten.
            self.ctx.synchronize()
            var dst = self.stage_rows.unsafe_ptr()
            for i in range(len(bag)):
                dst.unsafe_store(i, Int32(bag[i]))
            self.ctx.enqueue_copy(dst_buf=self.rows_dev, src_ptr=dst)
            self.ranges.reset_root(len(bag))

    def enqueue_partition[
        bins_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        window: LeafRange,
        routing: RowRouting,
    ) raises:
        """Enqueue the stable partition of `window`. Does not transfer or
        synchronize; the left count lands in `total_dev`.

        Three launches, whatever the range length: the flag-and-scan pass,
        the scatter (which scans the block sums at its own head), and the
        copy back. It was four until this lane folded the block-sums scan
        into the scatter. The saving is a launch, not bytes: nothing about
        what is read or written per row changed, and the copy back still
        moves 8 bytes per row of the range that per-range ping-pong would
        remove, which `_copy_back_kernel` writes up as a design and explains
        why this lane did not build.

        No timing is claimed. The stage profile that motivated the lane
        attributes 17.7 percent of a 1M by 50 GPU round to partition on an
        M4, of which an audit believes about half is fence attribution, and
        puts partition's floor at a four-launch minimum of roughly 237
        microseconds against a per-enqueue cost of roughly 20; that arithmetic
        says a launch is worth removing, and it is also in tension with the
        earlier finding recorded in the module docstring, that collapsing a
        small range's four launches into one measured 1.00x. Both of those
        are measurements this lane could not repeat and did not repeat. What
        is asserted here is only that three launches now do what four did,
        element for element.
        """
        comptime if not has_accelerator():
            raise Error(
                "the device row partition needs an accelerator; this binary"
                " was built without one, so partition rows through the CPU"
                " backend instead"
            )
        else:
            routing.check(self.n_features, self.n_bins)
            if window.begin < 0 or window.end > self.n_rows:
                raise Error("range escapes the row buffer")
            var n = window.count()
            if n <= 0:
                return

            var threads = self.block_threads
            var grid = _partition_grid(n, threads, self.partition_block_cap)
            var blocks = grid[0]
            var tiles = grid[1]
            # The copy back has no cross-block dependency, so it keeps the
            # plain one-tile-per-block grid rather than the capped one the
            # scan and the scatter share.
            var copy_blocks = (n + threads - 1) // threads
            var cat = routing.cat_bitset
            var default_left = Int32(1) if routing.default_left else Int32(0)
            var is_cat = Int32(1) if routing.is_categorical else Int32(0)

            # The flag pass and the scatter, on whichever arm is selected.
            # The primitive kernels are instantiated per width, so the
            # dispatch is an if-chain over the menu rather than a runtime
            # block size; the whole chain sits inside a `comptime if
            # has_accelerator()` because a build with no accelerator target
            # has no warp size for `block.prefix_sum` to constrain against,
            # and pruning the branch is what keeps those instantiations out
            # of a CPU-only extension build entirely.
            var scanned = False
            comptime if has_accelerator():
                if self.scan_primitives:
                    scanned = True
                    if threads == 128:
                        self._enqueue_scan_primitives[128](
                            bins, window, routing, n, blocks, tiles
                        )
                    elif threads == 256:
                        self._enqueue_scan_primitives[256](
                            bins, window, routing, n, blocks, tiles
                        )
                    elif threads == 512:
                        self._enqueue_scan_primitives[512](
                            bins, window, routing, n, blocks, tiles
                        )
                    elif threads == 1024:
                        self._enqueue_scan_primitives[1024](
                            bins, window, routing, n, blocks, tiles
                        )
                    else:
                        # `__init__` and `set_scan_primitives` both refuse to
                        # select the arm at an uninstantiated width, so this
                        # is unreachable; falling back rather than raising
                        # keeps a width that slipped through producing
                        # correct rows.
                        scanned = False

            # The compacted flag read, on the fallback arm. Resolved here for
            # the same reason `_enqueue_scan_primitives` resolves it at its
            # own launch; computed unconditionally so the two arms cannot
            # disagree about which plane they read.
            var flag_bins = _any_origin_u8(bins)
            var dense = self.compact_flag_read_live()
            if dense:
                flag_bins = _any_origin_u8(self.cbins_dev.unsafe_ptr())

            if not scanned:
                self.ctx.enqueue_function[_flag_scan_kernel](
                    flag_bins,
                    self.rows_dev.unsafe_ptr(),
                    self.offsets_dev.unsafe_ptr(),
                    self.block_sums_dev.unsafe_ptr(),
                    Int32(self.n_rows),
                    Int32(window.begin),
                    Int32(n),
                    Int32(routing.feature),
                    Int32(routing.threshold_bin),
                    Int32(routing.missing_bin),
                    default_left,
                    is_cat,
                    cat[0],
                    cat[1],
                    cat[2],
                    cat[3],
                    Int32(tiles),
                    # Inert descriptor. The host chose this split and passed
                    # it as arguments, so `use_desc` is zero and the kernel
                    # never dereferences this pointer; a real buffer rather
                    # than a null keeps the argument well typed everywhere,
                    # and it must be a buffer this launch does not already
                    # pass, which is why it is `step_dev` and not one of the
                    # partition's own.
                    self.step_dev.unsafe_ptr(),
                    Int32(0),
                    Int32(1) if dense else Int32(0),
                    grid_dim=blocks,
                    block_dim=threads,
                )
                # Same width and the same tiling as the flag pass: the
                # scatter looks its chunk up by `block_idx.x` and `tiles`,
                # and the packed offsets it reads are chunk-relative.
                self.ctx.enqueue_function[_scatter_kernel](
                    self.rows_dev.unsafe_ptr(),
                    self.scratch_dev.unsafe_ptr(),
                    self.offsets_dev.unsafe_ptr(),
                    self.block_sums_dev.unsafe_ptr(),
                    self.total_dev.unsafe_ptr(),
                    Int32(window.begin),
                    Int32(n),
                    Int32(tiles),
                    Int32(blocks),
                    # Inert descriptor. The host chose this split and passed
                    # it as arguments, so `use_desc` is zero and the kernel
                    # never dereferences this pointer; a real buffer rather
                    # than a null keeps the argument well typed everywhere,
                    # and it must be a buffer this launch does not already
                    # pass, which is why it is `step_dev` and not one of the
                    # partition's own.
                    self.step_dev.unsafe_ptr(),
                    Int32(0),
                    grid_dim=blocks,
                    block_dim=threads,
                )
            self.ctx.enqueue_function[_copy_back_kernel](
                self.rows_dev.unsafe_ptr(),
                self.scratch_dev.unsafe_ptr(),
                Int32(window.begin),
                Int32(n),
                # Inert descriptor; see the flag pass above.
                self.step_dev.unsafe_ptr(),
                Int32(0),
                grid_dim=copy_blocks,
                block_dim=threads,
            )
            # The data twin of everything above, when the arm is on. It reads
            # the same `offsets` and `block_sums` at the same `blocks` and
            # `tiles`, so it applies the identical permutation to the
            # compacted planes that the scatter just applied to the rows.
            # Nothing here is conditional on which arm the scan took: both
            # arms write the same offsets.
            self._maintain_compaction(
                window.begin, n, tiles, blocks, copy_blocks, Int32(0)
            )

    def _enqueue_scan_primitives[
        width: Int, bins_origin: MutOrigin
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        window: LeafRange,
        routing: RowRouting,
        n: Int,
        blocks: Int,
        tiles: Int,
        use_desc: Int32 = Int32(0),
    ) raises:
        """The flag pass and the scatter, on the primitive arm, at one
        compile-time threadgroup width.

        Split out only so `enqueue_partition` can name the width once per
        menu entry instead of repeating seventeen kernel arguments per entry.
        `block_dim` is `width` and not `self.block_threads`: the two are equal
        wherever this is reached, and writing the compile-time one is what
        makes a mismatch between the parameter and the launch impossible to
        introduce later. `grid_dim` and `tiles` are the caller's, computed
        once by `_partition_grid` from `self.block_threads`, so the geometry
        and therefore the `block_sums_dev` footprint are exactly what the
        fallback arm uses. Passing the same pair to both kernels rather than
        recomputing it in each is deliberate: the scatter's chunk arithmetic
        has to be the flag pass's chunk arithmetic exactly, or a packed offset
        would be read against the wrong chunk's block sum.
        """
        var cat = routing.cat_bitset
        var default_left = Int32(1) if routing.default_left else Int32(0)
        var is_cat = Int32(1) if routing.is_categorical else Int32(0)
        # Inert when `use_desc` is zero, which is the host arm; the selected
        # descriptor otherwise. `desc_target` is `DESC_STEP` unless the K=1
        # speculation armed it, so the host arm's pointer is the one it has
        # always been passed.
        var desc = self._desc_buffer()
        # --- the compacted flag read ---
        # Resolved here rather than passed in, for the reason `_any_origin_u8`
        # is spelled the way it is and the reason the histogram resolves its
        # own three pointers at the launch: this method holds `mut self`, so a
        # pointer into a field cannot be handed down from the caller, and the
        # caller's `bins` and `self.cbins_dev` carry different origins that
        # nothing widens implicitly. `compact_flag_read_live` is the one
        # predicate, so the pointer and the flag cannot come apart.
        var flag_bins = _any_origin_u8(bins)
        var dense = self.compact_flag_read_live()
        if dense:
            flag_bins = _any_origin_u8(self.cbins_dev.unsafe_ptr())
        self.ctx.enqueue_function[_flag_scan_prim_kernel[width]](
            flag_bins,
            self.rows_dev.unsafe_ptr(),
            self.offsets_dev.unsafe_ptr(),
            self.block_sums_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(window.begin),
            Int32(n),
            Int32(routing.feature),
            Int32(routing.threshold_bin),
            Int32(routing.missing_bin),
            default_left,
            is_cat,
            cat[0],
            cat[1],
            cat[2],
            cat[3],
            Int32(tiles),
            desc.unsafe_ptr(),
            use_desc,
            Int32(1) if dense else Int32(0),
            grid_dim=blocks,
            block_dim=width,
        )
        self.ctx.enqueue_function[_scatter_prim_kernel[width]](
            self.rows_dev.unsafe_ptr(),
            self.scratch_dev.unsafe_ptr(),
            self.offsets_dev.unsafe_ptr(),
            self.block_sums_dev.unsafe_ptr(),
            self.total_dev.unsafe_ptr(),
            Int32(window.begin),
            Int32(n),
            Int32(tiles),
            Int32(blocks),
            desc.unsafe_ptr(),
            use_desc,
            grid_dim=blocks,
            block_dim=width,
        )

    # --- The K=1 speculative prebuild -------------------------------------
    #
    # Five small members, and between them they are the whole of what this
    # file contributes to the speculation. Nothing about a partition or a
    # histogram changes; what changes is which flat Int32 row the launch
    # reads its window and its split out of, which is exactly the freedom the
    # step descriptor was introduced to create.
    #
    # The argument for why a speculatively built histogram is bit-identical
    # to the one the consuming step would have built is in
    # `gpu_resident_round.mojo` under "Exactness, by construction". The two
    # legs of it that live here are that a partition's permutation is a
    # function of the routing flags and the row's position in its window and
    # of nothing else (`_scatter_kernel`), and that accumulation is
    # fixed-point Int32 so a histogram is a function of the multiset of rows
    # and not of their order (`_range_hist_atomic_kernel`).

    def set_descriptor_target(mut self, target: Int) raises:
        """Choose which step descriptor the next descriptor-aware launch
        reads.

        `DESC_STEP`, `DESC_BUILD` or `DESC_SPEC`. A run-time arm in the same
        style as `set_row_unroll`, and for the same reason this repository
        insists on that style: the two arms have to be reachable in one
        binary, because this machine's timings drift several-fold between
        time windows and a rebuild between arms would put a different compile
        and a different thermal state on either side of the comparison.

        Not sticky by convention: `gpu_resident_round` sets it immediately
        before each launch and puts it back to `DESC_STEP` immediately after,
        so that any path that has not heard of the speculation -- the host
        partition, the host histogram, the whole non-resident plane -- reads
        the buffer it has always read. A target left set would not corrupt
        anything on those paths, since they pass `use_desc = 0` and the
        pointer is inert, but "would not corrupt anything" is a property that
        stops being true the first time someone gives one of them a
        descriptor.
        """
        if target != DESC_STEP and target != DESC_BUILD and target != DESC_SPEC:
            raise Error("unknown step descriptor target")
        # A deferred copy-back is owed against the descriptor the partition
        # that deferred it read. Moving the target under it would pay it
        # against a different window, which is the one way this fusion could
        # produce a wrong permutation rather than an error.
        self._refuse_copy_back_debt("changing the step descriptor target")
        self.desc_target = target

    def set_partition_fusion(mut self, on: Bool) raises:
        """Whether the descriptor partition's copy-back is folded into the
        descriptor histogram's slot zeroing.

        On is the default and off is the arm a window holds it against. A
        run-time arm in the same style as `set_row_unroll` and
        `set_descriptor_target`, and for the same standing reason: this
        machine's device timings drift two- to threefold between time windows,
        so only interleaved arms compare, and a comptime knob would put a
        different compile and a different thermal state on either side of the
        comparison.

        **It cannot change a histogram, a tree or a score.** Both arms store
        the same values to the same addresses under the same `STEP_LIVE`
        guard; what differs is how many command buffers carry those stores.
        `_copy_back_zero_slot_kernel` writes the whole argument out, including
        why these two launches and no other adjacent pair in the growth step
        may be folded.

        The pairing contract, which is the only thing a caller has to hold up
        --------------------------------------------------------------------
        With this on, `enqueue_partition_desc` does not launch its copy-back;
        it records a debt that the very next `enqueue_desc_histogram` pays,
        against the same descriptor. That is the shape
        `gpu_resident_round.grow_tree_device_resident` already has -- its
        partition and its child build are adjacent in both the unarmed and the
        armed schedules, against `DESC_STEP`/`DESC_BUILD` and against
        `DESC_SPEC` respectively -- and it is the only shape in the package
        that pairs them at all.

        A caller that breaks the pairing does not get a wrong answer quietly:
        `begin_tree`, `download_rows`, `download_range` and a second
        `enqueue_partition_desc` all refuse while a debt is outstanding, and
        the refusal names this method. That is deliberate rather than
        defensive. The bug this plane already shipped once was a caller that
        did not know it had host state to maintain (`_publish_row_ranges`), so
        a rule the callers have to remember is a rule that has already failed
        here.

        Turning it off while a debt is outstanding is refused for the same
        reason: the debt was incurred under the other arm and the launch that
        would have paid it is the one being turned off.
        """
        if self.copy_back_debt:
            raise Error(
                "the descriptor partition's copy-back is still owed to the"
                " next descriptor histogram; set_partition_fusion cannot"
                " change arms in the middle of that pairing"
            )
        self.fuse_partition_tail = on

    def partition_fusion_pending(self) -> Bool:
        """Whether a deferred copy-back is outstanding.

        Exposed so a test can assert that the fused arm actually deferred
        something rather than quietly taking the unfused path, which is the
        vacuous pass this whole lane's assertions are written against.
        """
        return self.copy_back_debt

    def copy_back_debt_blocks(self) -> Int:
        """Blocks the outstanding copy-back was enqueued against, or zero when
        nothing is owed.

        The number a batched build has to launch at least as wide as, so that
        the fused grid covers what `_copy_back_kernel`'s own would have covered.
        Zero when there is no debt, which is the value
        `GpuLeafBatcher.enqueue_device_plan_batch_fused` reads as "size the grid
        from the zeroing alone"; the copy-back half of that kernel is guarded on
        `STEP_LIVE` and stores nothing when the step is dead, so a zero here and
        a dead step agree.
        """
        return self.copy_back_blocks if self.copy_back_debt else 0

    def mark_copy_back_fused(mut self) raises:
        """Record that a batched build has paid the deferred copy-back.

        The bookkeeping half of `GpuLeafBatcher.enqueue_device_plan_batch_fused`,
        which carries the partition's deferred row copy-back inside the zeroing
        pass it launches anyway. That kernel lives in a module that cannot reach
        this flag -- `gpu_leaf_batching` does not own a `GpuActiveRows` -- so the
        schedule that enqueues it marks the debt paid here, in the same place.

        `enqueue_desc_histogram` clears the same flag for the single-child path,
        and it clears it *after* enqueueing the fused kernel. This does the same,
        and it is the caller's obligation to call it in that order: four refusals
        on this struct read `copy_back_debt`, and clearing it before the launch
        that pays it would open a window in which they all pass wrongly.

        Refuses when nothing is owed, because a schedule that thinks it paid a
        debt it never incurred is a schedule whose partition ran with the fusion
        off, and the batched build it just enqueued was then sized against a
        stale `copy_back_blocks`.
        """
        if not self.copy_back_debt:
            raise Error(
                "no descriptor partition copy-back is outstanding, so nothing"
                " was fused; a batched level build must follow the"
                " enqueue_partition_desc that deferred one"
            )
        self.copy_back_debt = False
        self.copy_back_folds += 1

    def _refuse_copy_back_debt(self, what: String) raises:
        """Refuse an operation that would observe the row buffer, or start a
        new tree, while the partition's copy-back has not been paid.

        Named rather than inlined so every refusal reads the same and points at
        the same method, and so the list of places that make it is greppable.
        """
        if self.copy_back_debt:
            raise Error(
                what
                + " while the descriptor partition's copy-back is still owed"
                " to the next descriptor histogram: the active rows of the"
                " window being split are in the scratch buffer, not in the"
                " row buffer. Every enqueue_partition_desc under"
                " set_partition_fusion(True) must be followed immediately by"
                " the enqueue_desc_histogram that pays it"
            )

    def _desc_buffer(self) -> DeviceBuffer[DType.int32]:
        """The descriptor buffer `desc_target` names, as a handle a launch can
        take a pointer out of.

        A returned handle rather than a returned pointer, and that is a
        language constraint rather than a preference. `DeviceBuffer.
        unsafe_ptr` carries the origin of the buffer it came from, so the
        three fields' pointers have three incompatible types and cannot be
        selected between by an `if`; a `DeviceBuffer` is a copyable handle, so
        the three *buffers* can be, and the pointer is then taken once from
        the local. The copy is a handle copy on the host and enqueues
        nothing.
        """
        if self.desc_target == DESC_BUILD:
            return self.build_dev.copy()
        if self.desc_target == DESC_SPEC:
            return self.spec_dev.copy()
        return self.step_dev.copy()

    def enqueue_spec_reset(mut self) raises:
        """Zero the speculative descriptor and the two counters, once per
        tree.

        `spec_dev` has to start dead rather than uninitialized, because the
        first step's consume kernel reads it before any runner-up kernel has
        written it. An uninitialized `STEP_LIVE` that happened to be 1 would
        make step 0 compare its commit against uninitialized memory, and the
        conjunction in `_spec_consume_kernel` makes a false hit unlikely
        rather than impossible.

        The counters are per tree for the same reason the census is per tree:
        a fit's hit rate is the sum of the counts over the sum of the counts,
        and a counter that accumulated across trees would be an answer to a
        question nobody asked and would silently be the wrong denominator.

        One launch, no transfer, no synchronization.
        """
        comptime if not has_accelerator():
            raise Error(
                "the speculative prebuild needs an accelerator; this binary"
                " was built without one"
            )
        else:
            self.ctx.enqueue_function[_zero_int32_kernel](
                self.spec_dev.unsafe_ptr(),
                Int32(STEP_WORDS),
                grid_dim=1,
                block_dim=STEP_WORDS,
            )
            self.ctx.enqueue_function[_zero_int32_kernel](
                self.spec_stats_dev.unsafe_ptr(),
                Int32(SPEC_STAT_WORDS),
                grid_dim=1,
                block_dim=SPEC_STAT_WORDS,
            )

    def enqueue_spec_consume(mut self) raises:
        """Publish the build descriptor for this step, consuming the previous
        step's prebuild when it is the work this step needs.

        Must be enqueued after the commit that wrote `step_dev` and before
        anything reads `build_dev`, which is to say between
        `enqueue_desc_step` and the real partition. It must also be enqueued
        *before* this step's own runner-up kernel overwrites `spec_dev`,
        which is the one ordering constraint in the whole schedule that is
        not implied by data flow: the comparison is against the previous
        step's publication, and there is exactly one buffer holding it.

        One launch of one thread. See `_spec_consume_kernel`.
        """
        comptime if not has_accelerator():
            raise Error(
                "the speculative prebuild needs an accelerator; this binary"
                " was built without one"
            )
        else:
            self.ctx.enqueue_function[_spec_consume_kernel](
                self.step_dev.unsafe_ptr(),
                self.spec_dev.unsafe_ptr(),
                self.build_dev.unsafe_ptr(),
                self.spec_stats_dev.unsafe_ptr(),
                grid_dim=1,
                block_dim=1,
            )

    def enqueue_spec_subtract(
        mut self, mut pool: DeviceBuffer[DType.int32], cells: Int
    ) raises:
        """The sibling subtraction a consumed step owes, and a no-op on every
        step that consumed nothing.

        `pool` is the resident histogram pool's base buffer and `cells` is one
        slot's width, `3 * n_features * n_bins`. Both come from the builder
        rather than from here, exactly as they do for the ordinary child
        build, because this struct owns rows and descriptors and not
        histograms.

        Enqueued unconditionally on every step of an armed fit. That is a
        launch on a step that will not act, which is the same bargain every
        other descriptor-aware kernel on this plane already takes: the host
        does not know what the device decided and asking costs the round trip
        the plane exists to remove.
        """
        if cells < 1:
            raise Error("a histogram slot cannot have zero cells")
        comptime if not has_accelerator():
            raise Error(
                "the speculative prebuild needs an accelerator; this binary"
                " was built without one"
            )
        else:
            var threads = self.block_threads
            var blocks = (cells + threads - 1) // threads
            if blocks > 256:
                blocks = 256
            if blocks < 1:
                blocks = 1
            self.ctx.enqueue_function[_spec_subtract_kernel](
                pool.unsafe_ptr(),
                Int32(cells),
                self.build_dev.unsafe_ptr(),
                grid_dim=blocks,
                block_dim=threads,
            )

    def download_spec_stats(mut self) raises -> List[Int]:
        """`[builds, consumed]` for the tree just grown.

        **A transfer, and therefore not on any measured path.** The caller
        asks for this only when an instrument wants it; the speculation
        itself never reads it, and neither does any kernel. It is here so
        that a test can assert on an observable only a *consuming* step can
        produce, which a count of launches is not: a speculation that never
        once hit would enqueue exactly the same kernels as one that always
        does.

        Synchronizes, so a caller inside a growth loop would be reinstating
        the per-split wait the plane removes. `gpu_resident_round` calls it
        after `download_desc_tables`, where the queue has already drained.
        """
        var host = self.ctx.enqueue_create_host_buffer[DType.int32](
            SPEC_STAT_WORDS
        )
        self.ctx.enqueue_copy(host, self.spec_stats_dev)
        self.ctx.synchronize()
        var out = List[Int](capacity=SPEC_STAT_WORDS)
        for i in range(SPEC_STAT_WORDS):
            out.append(Int(host[i]))
        return out^

    def enqueue_partition_desc[
        bins_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        max_count: Int,
    ) raises:
        """Enqueue the partition of whatever window the step descriptor names.

        The device-owned counterpart of `enqueue_partition`. It launches the
        same three kernels in the same order on the same arm; what it does not
        do is know which leaf is being split, which split is being applied, or
        how many rows are involved, because on this path a kernel chose all
        three and the host has not read them back. `max_count` is the only
        number the host still supplies and it is an upper bound rather than a
        length: the active row count, which no window can exceed because the
        live windows tile `[0, n_active)`.

        Why an upper bound is enough. The geometry a partition is launched at
        never reaches the permutation. `_scatter_kernel` derives each row's
        destination from the routing flags and from the row's position in the
        range, and from nothing else; the chunking appears only in how the
        prefix is accumulated, and an integer prefix is the same number
        however it is summed. So a grid sized for a longer range than the one
        it finds produces exactly the permutation the exactly-sized grid would
        have produced, and the surplus threadgroups return at their first
        bounds check. That argument is written out at greater length in
        `_flag_scan_kernel` and `_scatter_kernel`.

        What the surplus costs is UNMEASURED. It is bounded above by the
        capped block count times the tile count that covers `max_count`, and
        every one of those blocks that owns nothing exits before it reads a
        row; whether that is cheaper than the host round trip it replaces is
        exactly the thing a benchmark would have to answer and no benchmark
        has been run.

        Nothing is downloaded and nothing synchronizes, which is the point.
        `total_dev` is still written by the scatter and is still the range's
        left count, but the device-owned path does not read it: the commit
        kernel already took both children's counts off the parent histogram's
        integer count plane, exactly as the host path takes them off the
        record. It stays written so that `MOJOTREES_GPU_VERIFY_ROWS` remains
        meaningful for anyone who wants to check the two against each other.

        **What this does to the host row-range table, and why the first line
        of the body is the arming call.** This entry point moves rows between
        leaves and updates no window in `self.ranges`. Every window that table
        holds is therefore, from here on, a description of the tree before
        this split -- well formed, in bounds, satisfying every invariant, and
        wrong. `LeafRangeTable.begin_descriptor_partition` records that, and
        every accessor that returns a window refuses until
        `gpu_resident_round._publish_row_ranges` replays the device's commit
        log and clears it.

        The arming is done here, in the code whose behavior creates the
        invalidity, rather than by the caller that knows it is starting a
        resident tree. The bug this replaces was a caller that did not know
        it had a host table to maintain, so a rule the callers have to
        remember is the rule that already failed once. It is also load
        bearing rather than advisory: `bound` is what the launch geometry is
        derived from, so deleting the arming does not compile.

        `bound` is `max_count`, validated. The validation moved into
        `begin_descriptor_partition` with it so that there is no version of
        this entry point that checks its bound and forgets to arm.

        **Three launches, or two under `set_partition_fusion(True)`, which is
        the default.** The third, `_copy_back_kernel`, is then not issued here
        at all: the debt is recorded and the next `enqueue_desc_histogram`
        discharges it in the same command buffer as its slot zeroing, which is
        one fewer command buffer per growth step. This clause read "on a queue
        that is 64 deep and past its knee" and the second half is retired as of
        2026-08-18: the depth is a price per launch and not a hazard to be past,
        since `MTLCommandQueue.commandBuffer` blocks rather than dropping work
        (`docs/GPU_PORTABILITY.md` 6.2, `docs/design/SWITCH_GRID.md` section 6
        item 8). The price is what carries this. The leaf-wise grower this entry
        point serves enqueues 8 + 9 * (num_leaves - 1) buffers a tree, 2,303 at
        256 leaves, and was measured backpressured with `device_wait` at exactly
        0 calls, so every buffer it drops is bought at the over-depth enqueue
        cost of 14 to 17 microseconds rather than the 6 to 7 under it. One fewer
        per growth step is a real saving at that rate and it is not a rescue from
        anything. See `set_partition_fusion` for the pairing contract and
        `_copy_back_zero_slot_kernel` for why the fold is exact and why the
        other two launches here cannot be folded into anything.
        """
        comptime if not has_accelerator():
            raise Error(
                "the descriptor-driven device row partition needs an"
                " accelerator; this binary was built without one, so"
                " partition rows through the CPU backend instead"
            )
        else:
            self._refuse_copy_back_debt("a second descriptor partition")
            var bound = self.ranges.begin_descriptor_partition(max_count)
            var threads = self.block_threads
            var grid = _partition_grid(
                bound, threads, self.partition_block_cap
            )
            var blocks = grid[0]
            var tiles = grid[1]
            # Which descriptor this partition routes by. `DESC_STEP` for
            # every caller that predates the K=1 speculation, which is every
            # caller today apart from `gpu_resident_round`'s armed loop; see
            # `set_descriptor_target`.
            var desc = self._desc_buffer()

            # The routing arguments are placeholders: `use_desc` is 1, so
            # every kernel below reads the split out of `step_dev` and
            # ignores them. A window of `[0, max_count)` is passed for the
            # same reason, and is deliberately the widest legal one rather
            # than an empty one, so that a wiring mistake that left
            # `use_desc` at zero would partition a real range and be caught
            # by a row check rather than silently do nothing.
            var window = LeafRange(0, bound)
            var routing = RowRouting.numerical(0, 0, -1, False)

            var scanned = False
            comptime if has_accelerator():
                if self.scan_primitives:
                    scanned = True
                    if threads == 128:
                        self._enqueue_scan_primitives[128](
                            bins, window, routing, bound, blocks, tiles,
                            Int32(1),
                        )
                    elif threads == 256:
                        self._enqueue_scan_primitives[256](
                            bins, window, routing, bound, blocks, tiles,
                            Int32(1),
                        )
                    elif threads == 512:
                        self._enqueue_scan_primitives[512](
                            bins, window, routing, bound, blocks, tiles,
                            Int32(1),
                        )
                    elif threads == 1024:
                        self._enqueue_scan_primitives[1024](
                            bins, window, routing, bound, blocks, tiles,
                            Int32(1),
                        )
                    else:
                        scanned = False

            # The compacted flag read; see `enqueue_partition` for why the
            # pointer is resolved at the launch and not handed down.
            var flag_bins = _any_origin_u8(bins)
            var dense = self.compact_flag_read_live()
            if dense:
                flag_bins = _any_origin_u8(self.cbins_dev.unsafe_ptr())

            if not scanned:
                self.ctx.enqueue_function[_flag_scan_kernel](
                    flag_bins,
                    self.rows_dev.unsafe_ptr(),
                    self.offsets_dev.unsafe_ptr(),
                    self.block_sums_dev.unsafe_ptr(),
                    Int32(self.n_rows),
                    Int32(0),
                    Int32(bound),
                    Int32(0),
                    Int32(0),
                    Int32(-1),
                    Int32(0),
                    Int32(0),
                    UInt64(0),
                    UInt64(0),
                    UInt64(0),
                    UInt64(0),
                    Int32(tiles),
                    desc.unsafe_ptr(),
                    Int32(1),
                    Int32(1) if dense else Int32(0),
                    grid_dim=blocks,
                    block_dim=threads,
                )
                self.ctx.enqueue_function[_scatter_kernel](
                    self.rows_dev.unsafe_ptr(),
                    self.scratch_dev.unsafe_ptr(),
                    self.offsets_dev.unsafe_ptr(),
                    self.block_sums_dev.unsafe_ptr(),
                    self.total_dev.unsafe_ptr(),
                    Int32(0),
                    Int32(bound),
                    Int32(tiles),
                    Int32(blocks),
                    desc.unsafe_ptr(),
                    Int32(1),
                    grid_dim=blocks,
                    block_dim=threads,
                )
            # The data twin, when the arm is on, and it is placed **above**
            # the fusion's early return on purpose: the fusion defers the row
            # copy-back and this pair is not part of that deferral, so a
            # return taken before this would skip the compaction on every
            # step of every device-resident tree and leave the planes
            # describing the tree before the split. It reads neither
            # `rows_dev` nor `scratch_dev`, so it is indifferent to which of
            # the two currently holds the permutation and the deferral cannot
            # reach it.
            self._maintain_compaction(
                0, bound, tiles, blocks, blocks, Int32(1)
            )

            # Under the fusion the copy-back is not a launch of its own: the
            # next descriptor histogram folds it into its slot zeroing, which
            # is a launch that has to happen anyway and is unordered with
            # respect to this one. Nothing else is deferred and nothing is
            # skipped -- the same stores are made, in the next command buffer
            # instead of this one, and still before the accumulation that
            # reads them.
            if self.fuse_partition_tail:
                self.copy_back_debt = True
                self.copy_back_blocks = blocks
                return

            # The copy back runs on the capped grid here, not on the uncapped
            # `ceil(count / threads)` one the host arm gives it, because a
            # count this launch does not know cannot size a grid. It is
            # grid-strided, so the same threads walk the range instead of one
            # thread owning one element; see `_copy_back_kernel`.
            self.ctx.enqueue_function[_copy_back_kernel](
                self.rows_dev.unsafe_ptr(),
                self.scratch_dev.unsafe_ptr(),
                Int32(0),
                Int32(bound),
                desc.unsafe_ptr(),
                Int32(1),
                grid_dim=blocks,
                block_dim=threads,
            )

    def enqueue_desc_histogram[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin,
        feat_origin: MutOrigin,
        pool_origin: MutOrigin, //
    ](
        mut self,
        pool_slots: Int,
        max_rows: Int,
        bins: MutPointer[UInt8, bins_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        feat_ids: MutPointer[Int32, feat_origin],
        pool: MutPointer[Int32, pool_origin],
        n_slots: Int,
        g_scale: Float32,
        h_scale: Float32,
        caps: DeviceCaps,
    ) raises:
        """Accumulate the built child's histogram into the pool slot the step
        descriptor names, subtracting it from the sibling's slot as it goes.

        The device-owned counterpart of `enqueue_range_histogram`, narrowed to
        exactly the one shape a device-owned tree needs: one node, into a
        resident pool slot, with the sibling subtraction folded in.

        **The atomic strategy, always, and this is a decision rather than an
        oversight.** The tiled strategy's partial buffer and its reduction are
        both sized by `n_tiles`, and `n_tiles` is derived from the node's row
        count, which on this path the host does not know. Deriving it from the
        active row count instead would give a shallow node a reduction over
        thousands of empty tiles, which is a cost that grows exactly where the
        tree spends most of its splits. The atomic strategy has no such term:
        its only row-derived quantity is the grid's second dimension, and a
        block whose tile is empty returns at its first comparison
        (`_range_hist_atomic_kernel`).

        The two strategies produce the same histogram, which is what makes
        this a launch decision and not a numeric one. Accumulation is
        fixed-point Int32 in both, integer addition is associative and
        commutative, and each strategy adds the same per-(row, feature)
        quantized value into the same bin; only the order of the additions
        differs, and the order of integer additions cannot change a sum. That
        is the same argument `_range_hist_atomic_kernel` makes for its own
        forty instantiations agreeing with each other.

        **What is not claimed.** That the atomic strategy is as fast here as
        the strategy the host arm would have chosen. It may well not be on a
        large first split, where the tiled path exists precisely because
        atomics contend. No benchmark of this arm has been run and none is
        reported.

        `max_rows` is an upper bound on the built child's row count, and the
        active row count is the bound a caller has. The built child is the
        smaller of the two by `subtraction_builds_left`, so `n_active / 2`
        would also bound it; the looser bound is used because a bound that
        depends on which child won a comparison is a bound that has to be
        re-derived if that comparison ever changes.
        """
        comptime if not has_accelerator():
            raise Error(
                "the descriptor-driven device histogram needs an accelerator;"
                " this binary was built without one, so build histograms"
                " through the CPU backend instead"
            )
        else:
            if n_slots < 1 or n_slots > self.n_features:
                raise Error("active feature count out of range")
            if pool_slots < 1:
                raise Error("the resident pool must hold at least one slot")
            if max_rows < 1:
                raise Error(
                    "a descriptor histogram needs a positive row bound"
                )
            # This entry point is the device-resident growth plane: every
            # histogram of every non-root node of every tree comes through
            # here. A probe histogram reaching it would grow a whole tree out
            # of garbage, and a subtraction chain would then carry the
            # garbage into siblings that never ran the probe at all. See
            # `set_histogram_atomic_probe`.
            if self.histogram_probe_active():
                raise Error(
                    "a histogram probe arm builds a wrong histogram and"
                    " must never reach the device-resident growth plane; turn"
                    " it off with set_histogram_atomic_probe(False, True) or"
                    " set_histogram_probe_mode(HIST_PROBE_OFF, True)"
                )

            # Which descriptor this build reads, and -- the same decision,
            # not a second one -- whether it folds the sibling subtraction in.
            # A speculative build must not: `DESC_SPEC` names a leaf that is
            # still live, and deriving the larger child in place from that
            # leaf's slot would destroy the histogram of the very leaf being
            # speculated on, which on a miss is the histogram the next pick
            # reads. The subtraction a consumed step owes is done later by
            # `_spec_subtract_kernel`, once the commit has proved the leaf is
            # a parent.
            var desc = self._desc_buffer()
            var do_sub = Int32(
                0
            ) if self.desc_target == DESC_SPEC else Int32(1)

            var hist_size = self.n_features * self.n_bins
            var cells = 3 * hist_size
            # The two tiling requests are passed here for the same reason
            # `range_tiling` passes them: this is how the device-owned growth
            # plane builds every non-root histogram, so omitting them made the
            # row-tile arms reach the root and nothing else. Under the default
            # resident plane that is 1 node of 61, which would have made the
            # tile question look answered while being almost entirely unasked
            # -- the same shape as a test that runs below the gate it is
            # testing.
            #
            # Zero on both is byte-for-byte the previous behavior, so the
            # default path is unchanged and only an explicitly requested arm
            # moves.
            var tiling = derive_tiling(
                caps,
                max_rows,
                n_slots,
                self.n_bins,
                STRATEGY_ATOMIC,
                self.part_capacity_unused(),
                0,
                self.min_tiles_request,
                self.rows_per_tile_request,
            )
            var threads = tiling.block_threads

            # The atomic flush is `Atomic.fetch_add` onto whatever the slot
            # already holds, so the slot has to start at zero. Only the one
            # slot the descriptor names is cleared, and only when the step is
            # live; clearing a slot on a dead step would erase a live leaf's
            # histogram.
            var zero_blocks = (cells + threads - 1) // threads
            if zero_blocks > pool_slots * 4:
                zero_blocks = pool_slots * 4
            if zero_blocks < 1:
                zero_blocks = 1
            if self.copy_back_debt:
                # The partition immediately before this one deferred its
                # copy-back, so this launch carries both halves. One command
                # buffer where the step used to spend two, with every store
                # unchanged; `_copy_back_zero_slot_kernel` argues that in
                # full.
                #
                # The grid is the wider of the two the halves would have had.
                # Neither half's answer depends on it -- both are grid-strided
                # over distinct positions of their own range -- so this is a
                # work distribution and not a correctness input. `threads` is
                # the histogram's block width rather than the partition's; the
                # same argument covers it.
                var fused_blocks = zero_blocks
                if self.copy_back_blocks > fused_blocks:
                    fused_blocks = self.copy_back_blocks
                self.ctx.enqueue_function[_copy_back_zero_slot_kernel](
                    self.rows_dev.unsafe_ptr(),
                    self.scratch_dev.unsafe_ptr(),
                    pool,
                    Int32(cells),
                    desc.unsafe_ptr(),
                    grid_dim=fused_blocks,
                    block_dim=threads,
                )
                self.copy_back_debt = False
                self.copy_back_folds += 1
            else:
                self.ctx.enqueue_function[_zero_slot_desc_kernel](
                    pool,
                    Int32(cells),
                    desc.unsafe_ptr(),
                    grid_dim=zero_blocks,
                    block_dim=threads,
                )

            var use_quant = Int32(QUANT_SOURCE_FLOAT)
            if self.quantized_gradients:
                self._ensure_quantized(grad, hess, g_scale, h_scale)
                # Which *width* of the quantized buffer, read from the
                # staging state rather than from the request, so the kernel
                # is told the layout the buffer actually holds.
                # `_ensure_quantized` has just made those agree, and reading
                # the request here instead would be the one spelling that
                # could disagree with it.
                use_quant = Int32(
                    QUANT_SOURCE_PACKED16
                ) if self.quant_packed else Int32(QUANT_SOURCE_INT32)
            # The device-owned growth plane builds every non-root histogram
            # through this entry point, so the re-layout has to be reachable
            # from here as well as from the host-driven one; omitting it would
            # leave the layout arm reaching the root and nothing else, which
            # is the exact shape the row-tile arms were found in.
            self._ensure_blocked(bins)
            self._ensure_packed(bins)
            # The compacted planes, if the arm is on. Placed after the zeroing
            # branch above rather than before it, because under
            # `set_partition_fusion(True)` that branch is what discharges the
            # previous partition's deferred row copy-back, and a rebuild here
            # reads `rows_dev`. Launches on one stream are ordered, so
            # enqueuing after it is what makes the rebuild see the permutation
            # the partition produced rather than the one before it.
            self._ensure_compacted(bins)
            var const_hess = Int32(1) if self.constant_hessian else Int32(0)

            self._enqueue_atomic_family(
                bins,
                grad,
                hess,
                feat_ids,
                pool,
                n_slots,
                hist_size,
                tiling.rows_per_tile,
                # Window, destination slot and subtraction offset all come out
                # of the descriptor, so these three are placeholders the
                # kernel overwrites. The subtraction is on for a real split,
                # which always derives one child from the parent's slot, and
                # off for a speculative one; see `do_sub` above.
                0,
                max_rows,
                g_scale,
                h_scale,
                0,
                do_sub,
                use_quant,
                const_hess,
                tiling.n_tiles,
                threads,
                Int32(1),
            )

    def part_capacity_unused(self) -> Int:
        """Zero, the "no preallocated partial buffer" value `derive_tiling`
        takes.

        The atomic strategy allocates no partials, so the capacity a caller
        would pass to cap the tiled strategy's buffer has nothing to cap. It
        is a named method rather than a bare literal so that the reason is
        attached to the argument at the one call site that passes it.
        """
        return 0

    def download_left_count(mut self) raises -> Int:
        """The left count of the last enqueued partition. Synchronizes."""
        self.ctx.enqueue_copy(
            dst_ptr=self.host_total.unsafe_ptr(), src_buf=self.total_dev
        )
        self.ctx.synchronize()
        return Int(self.host_total.unsafe_ptr().unsafe_load(0))

    def partition[
        bins_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        parent: Int,
        left: Int,
        right: Int,
        routing: RowRouting,
        expected_left: Int = -1,
    ) raises -> Int:
        """Split `parent`'s range into its children's and return the left
        count.

        With `expected_left` given (the grower has it exactly, from the
        parent histogram's integer counts) nothing is downloaded and nothing
        synchronizes, so a split stays fully enqueued. `verify_counts`
        downloads the device count anyway and raises if the two disagree,
        which is the check that the routing rule the host counted with is
        the routing rule the device applied.
        """
        var window = self.ranges.get(parent)
        if expected_left > window.count():
            raise Error("left row count is outside the parent's range")
        self.enqueue_partition(bins, window, routing)

        var n_left: Int
        if window.count() == 0:
            n_left = 0
        elif expected_left < 0:
            n_left = self.download_left_count()
        elif self.verify_counts:
            var device_left = self.download_left_count()
            if device_left != expected_left:
                raise Error(
                    "device left count disagrees with the histogram count"
                )
            n_left = expected_left
        else:
            n_left = expected_left

        _ = self.ranges.split(parent, left, right, n_left)
        return n_left

    # --- The frontier seam -------------------------------------------------
    #
    # Three calls a grower makes instead of maintaining the row ranges and the
    # host-side leaf list separately. `LeafFrontier` holds the same windows
    # this module's `LeafRangeTable` holds, and holding them twice is only
    # safe if something keeps them equal; `check_frontier` is that something,
    # and `apply_commit` is the one state transition that moves both.

    def begin_tree_with(
        mut self,
        mut frontier: LeafFrontier,
        bag: List[Int] = [],
        max_leaves: Int = 0,
        plane: Int = 0,
    ) raises -> Int:
        """Start a tree on the device and on the host frontier at once, and
        return the number of rows it grows on.

        One call rather than two, because the two agree on exactly one number
        and letting a caller pass it twice is letting a caller pass it wrong:
        `n_active` is `len(bag)` when bagging or GOSS selected rows and
        `n_rows` when they did not, and both the root `LeafRange` and the root
        `FrontierLeaf` have to be that window.

        `max_leaves` is the `num_leaves` budget the frontier reports
        completion against and `plane` the class index of a multiclass round;
        neither reaches the device, because neither changes which rows exist.
        A multiclass round grows one tree per class over this same
        permutation, sequentially, so calling this once per class is correct
        and costs one root seed each.
        """
        self.begin_tree(bag)
        var n_active = self.ranges.n_active
        frontier.begin_tree(n_active, max_leaves, plane)
        return n_active

    def check_frontier(self, frontier: LeafFrontier) raises:
        """Every live frontier leaf's window is the range its node owns here.

        The check that makes a disagreement between the two tables loud
        instead of silent. A frontier leaf whose `row_begin`/`row_count` had
        drifted from its `LeafRange` would build a histogram of the wrong
        rows, and nothing about the launch would be out of bounds, so nothing
        else would notice. Linear in the frontier, which is a few hundred
        entries, and meant to be called after a commit or before a batch
        rather than inside a row loop.
        """
        if frontier.n_active != self.ranges.n_active:
            raise Error(
                "frontier and active-row table disagree on the active row"
                " count"
            )
        for i in range(frontier.size()):
            var leaf = frontier.leaf(i)
            var window = self.ranges.get(leaf.node)
            if window.begin != leaf.row_begin:
                raise Error(
                    "frontier leaf's row window does not start where its"
                    " active-row range does"
                )
            if window.count() != leaf.row_count:
                raise Error(
                    "frontier leaf's row count does not match its active-row"
                    " range"
                )

    def partition_commit[
        bins_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        plan: CommitPlan,
    ) raises -> Int:
        """Split the committed leaf's range into its children's, from the
        plan alone.

        The routing rule is built here, from `plan.split` and
        `plan.missing_bin`, through the same `RowRouting.from_split` the host
        reference model uses, so a caller never restates the
        missing/categorical rule and cannot state it differently from the
        kernels. The left count is the plan's `left_count`, which the search
        already counted exactly off the parent histogram's integer count
        plane, so nothing is downloaded and nothing synchronizes unless
        `MOJOTREES_GPU_VERIFY_ROWS=1` asks for the cross-check.

        An empty parent partitions to two empty children: the enqueue returns
        without a launch and the range table records `[b, b)` twice, which is
        what keeps the live ranges tiling `[0, n_active)` whatever the tree
        does.
        """
        var routing = RowRouting.from_split(plan.split, plan.missing_bin)
        return self.partition(
            bins,
            plan.parent_node,
            plan.left_node,
            plan.right_node,
            routing,
            plan.left_count,
        )

    def apply_commit[
        bins_origin: MutOrigin, //
    ](
        mut self,
        mut frontier: LeafFrontier,
        bins: MutPointer[UInt8, bins_origin],
        plan: CommitPlan,
    ) raises -> Int:
        """The whole of one commit: partition on the device, move the
        frontier on the host, and return the left row count.

        This is the single state transition a grower needs per split. It runs
        the device half first so a failure to route leaves the frontier
        untouched rather than half advanced, and it re-checks the two tables
        against each other afterwards, which is cheap next to a partition and
        is what turns a wiring mistake into an error at the split that caused
        it instead of at some later histogram.
        """
        var n_left = self.partition_commit(bins, plan)
        frontier.apply_commit(plan)
        self.check_frontier(frontier)
        return n_left

    def range_tiling(
        self,
        caps: DeviceCaps,
        node: Int,
        n_slots: Int,
        strategy: Int,
        max_partial_cells: Int,
    ) raises -> HistogramTiling:
        """Launch geometry for one node's histogram, derived from the rows
        that node actually owns rather than from the dataset.

        This is the second half of the win: an empty or tiny node gets a
        grid sized for its own rows instead of for `n_rows`. A node with no
        rows still needs a launchable geometry, so it is derived at one row.

        The two tile requests are the ones `set_row_tiling` holds, and they
        are passed here rather than read from the environment inside
        `gpu_tiling` so that a benchmark can vary them between interleaved
        repeats in one process. Zero on both, the default, is the same
        answer as an unset environment variable, so a caller that asks for
        nothing gets the geometry this method returned before they existed.
        """
        var n = self.ranges.get(node).count()
        if n < 1:
            n = 1
        return derive_tiling(
            caps,
            n,
            n_slots,
            self.n_bins,
            strategy,
            max_partial_cells,
            0,
            self.min_tiles_request,
            self.rows_per_tile_request,
        )

    def enqueue_range_histogram[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin,
        feat_origin: MutOrigin,
        out_origin: MutOrigin,
        part_origin: MutOrigin, //
    ](
        mut self,
        tiling: HistogramTiling,
        node: Int,
        bins: MutPointer[UInt8, bins_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        feat_ids: MutPointer[Int32, feat_origin],
        out_hist: MutPointer[Int32, out_origin],
        partials: MutPointer[Int32, part_origin],
        n_slots: Int,
        g_scale: Float32,
        h_scale: Float32,
        sub_offset: Int = 0,
        subtract: Bool = False,
    ) raises:
        """Enqueue the histogram of `node`'s rows, reading only those rows.

        Everything but the row loop matches `GpuHistogramBuilder.enqueue_leaf`
        (same output layout, same fixed-point accumulation, same two
        strategies), so the histogram is the one that path would have built
        by scanning the whole dataset. The output is zeroed here with a
        kernel rather than `enqueue_memset` because the caller handed in a
        pointer; the zeroing rule is unchanged, only the atomic path and a
        narrowed feature set need it.

        Under `subtract`, the histogram `sub_offset` Int32 words from
        `out_hist` is a second one this build is taken out of as it goes:
        `sub -= out`, folded into whichever kernel writes the result. That is
        the sibling subtraction of a resident frontier done without the
        launch and without the pass over two whole slots that a separate
        `_subtract_slice_kernel` costs, and it is the same exact fixed-point
        arithmetic. A nonzero offset is required, since the two histograms
        are different slots of one pool; `sub_offset` is ignored without
        `subtract`. An empty node subtracts nothing, which is right: a child
        with no rows leaves its sibling holding the parent's histogram
        unchanged.

        Which kernel instantiation runs is resolved here and nowhere else:
        `bin_cap` from the dataset's bin count, the group from
        `feature_group` through `_launch_group`, and the pair through the two
        static trees below. Every instantiation produces the same integers, so
        this resolution is a launch shape and never a numeric decision. The
        pre-quantized gradient buffer is rebuilt here too, once per tree per
        scale, above the strategy branch because every strategy and every
        width now reads it.

        The constant-hessian flag is resolved here for the same reason, from
        `constant_hessian`, and handed to whichever kernels this launch uses.
        On the tiled path it also changes the reduction's grid, because the
        partial layout it selects has two planes per tile rather than three;
        the two kernels get the same flag from this one place, which is what
        keeps them agreeing about the layout. It does not change the zeroing
        rule: the reduction still writes every active feature's slice of all
        three output planes, so the same three conditions that force a
        zeroing pass force it here.
        """
        comptime if not has_accelerator():
            raise Error(
                "the device histogram of a node's rows needs an accelerator;"
                " this binary was built without one, so build the histogram"
                " through the CPU backend instead"
            )
        else:
            if n_slots < 1 or n_slots > self.n_features:
                raise Error("active feature count out of range")
            if subtract and sub_offset == 0:
                raise Error(
                    "a fused subtraction must name a slot other than the"
                    " build's"
                )
            # A fused sibling subtraction means a tree is being grown: this
            # node is a child and its sibling is about to be derived from it.
            # A probe histogram there would corrupt a sibling that never ran
            # the probe, which is worse than a wrong node -- it is a wrong
            # node that looks like a correct one. See
            # `set_histogram_atomic_probe`.
            if subtract and self.histogram_probe_active():
                raise Error(
                    "a histogram probe arm builds a wrong histogram and"
                    " must never feed a sibling subtraction; it is reachable"
                    " only on a from-scratch build of one node's rows"
                )
            var window = self.ranges.get(node)
            var hist_size = self.n_features * self.n_bins
            var threads = tiling.block_threads

            # The reduction of the tiled path writes every active feature's
            # slice, so that path only needs zeroing when some feature is
            # inactive; the atomic path always does, and so does a node with
            # no rows, whose histogram is entirely zeros.
            if (
                tiling.strategy != STRATEGY_TILED
                or n_slots < self.n_features
                or window.count() <= 0
            ):
                var cells = 3 * hist_size
                var zero_blocks = (cells + threads - 1) // threads
                self.ctx.enqueue_function[_zero_int32_kernel](
                    out_hist,
                    Int32(cells),
                    grid_dim=zero_blocks,
                    block_dim=threads,
                )
            if window.count() <= 0:
                return

            var do_sub = Int32(1) if subtract else Int32(0)

            # One quantizing pass per tree per scale, ordered before the
            # histogram that reads it by the queue. Every strategy and every
            # group width reads the same buffer now, so this sits above the
            # dispatch rather than inside one arm of it.
            var use_quant = Int32(QUANT_SOURCE_FLOAT)
            if self.quantized_gradients:
                self._ensure_quantized(grad, hess, g_scale, h_scale)
                # Which *width* of the quantized buffer, read from the
                # staging state rather than from the request, so the kernel
                # is told the layout the buffer actually holds.
                # `_ensure_quantized` has just made those agree, and reading
                # the request here instead would be the one spelling that
                # could disagree with it.
                use_quant = Int32(
                    QUANT_SOURCE_PACKED16
                ) if self.quant_packed else Int32(QUANT_SOURCE_INT32)

            # The bin re-layout, if one was asked for, on the same footing
            # and in the same place: one enqueued pass ordered before the
            # histogram that reads it, above the strategy branch because both
            # strategies read the same buffer. It runs once per fit rather
            # than once per tree, which is the whole reason it can be
            # afforded at all.
            self._ensure_blocked(bins)
            self._ensure_packed(bins)

            # And the compacted planes, on the same footing and in the same
            # place, and after `_ensure_quantized` because it reads what that
            # writes. A no-op unless the arm is on and the invariant has
            # lapsed, which after the first histogram of a tree it has not.
            self._ensure_compacted(bins)

            var const_hess = Int32(1) if self.constant_hessian else Int32(0)
            var hist_planes = 2 if self.constant_hessian else 3

            if tiling.strategy == STRATEGY_TILED:
                self._enqueue_partial_family(
                    bins,
                    grad,
                    hess,
                    feat_ids,
                    partials,
                    n_slots,
                    tiling.rows_per_tile,
                    window.begin,
                    window.count(),
                    g_scale,
                    h_scale,
                    use_quant,
                    const_hess,
                    tiling.n_tiles,
                    threads,
                )
                var n_cells = hist_planes * n_slots * self.n_bins
                var blocks = (n_cells + threads - 1) // threads
                self.ctx.enqueue_function[_range_reduce_kernel](
                    partials,
                    feat_ids,
                    out_hist,
                    Int32(n_slots),
                    Int32(self.n_bins),
                    Int32(hist_size),
                    Int32(tiling.n_tiles),
                    Int32(sub_offset),
                    do_sub,
                    h_scale,
                    const_hess,
                    # The decomposition probes, resolved the same way the
                    # accumulation launches resolve them. Only the empty arm
                    # has any effect here; see the kernel for why it has to.
                    (
                        Int32(HIST_PROBE_NO_ATOMICS)
                        if self.hist_atomic_probe
                        else Int32(self.hist_probe_mode)
                    ),
                    grid_dim=blocks,
                    block_dim=threads,
                )
            else:
                self._enqueue_atomic_family(
                    bins,
                    grad,
                    hess,
                    feat_ids,
                    out_hist,
                    n_slots,
                    hist_size,
                    tiling.rows_per_tile,
                    window.begin,
                    window.count(),
                    g_scale,
                    h_scale,
                    sub_offset,
                    do_sub,
                    use_quant,
                    const_hess,
                    tiling.n_tiles,
                    threads,
                    # No descriptor: this arm's window and destination came
                    # from the host, so the kernel is told to ignore the
                    # descriptor buffer it is handed. See `step_dev`.
                    Int32(0),
                )

    def _launch_group(self, n_slots: Int) -> Int:
        """The group width this launch runs at.

        `feature_group`, except that a single active feature slot runs at 1
        whatever was asked for. That is the `n_slots >= 2` guard the
        hand-written variants carried, kept verbatim so the grids this
        dispatch produces at widths 1, 2, and 4 are the grids the launch site
        produced before the family was parameterized. Nothing else is
        narrowed: a request stays the request.
        """
        if n_slots < 2:
            return 1
        return self.feature_group

    def _enqueue_partial_family[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin,
        feat_origin: MutOrigin,
        part_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        feat_ids: MutPointer[Int32, feat_origin],
        partials: MutPointer[Int32, part_origin],
        n_slots: Int,
        rows_per_tile: Int,
        begin: Int,
        count: Int,
        g_scale: Float32,
        h_scale: Float32,
        use_quant: Int32,
        const_hess: Int32,
        n_tiles: Int,
        threads: Int,
    ) raises:
        """Pick the `GROUP` rung, then hand off to the rung's own dispatch.

        The (GROUP, BIN_CAP) matrix is resolved as a static tree in two
        halves, five ways here and four ways in `_enqueue_partial_at`, rather
        than as twenty branches written out. Both halves are exhaustive over
        their ladder, so the resolution cannot fall through to a width the
        launch was not sized for.
        """
        var group = self._launch_group(n_slots)
        if group >= 16:
            self._enqueue_partial_at[16](
                bins,
                grad,
                hess,
                feat_ids,
                partials,
                n_slots,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                n_tiles,
                threads,
            )
        elif group >= 8:
            self._enqueue_partial_at[8](
                bins,
                grad,
                hess,
                feat_ids,
                partials,
                n_slots,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                n_tiles,
                threads,
            )
        elif group >= 4:
            self._enqueue_partial_at[4](
                bins,
                grad,
                hess,
                feat_ids,
                partials,
                n_slots,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                n_tiles,
                threads,
            )
        elif group >= 2:
            self._enqueue_partial_at[2](
                bins,
                grad,
                hess,
                feat_ids,
                partials,
                n_slots,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                n_tiles,
                threads,
            )
        else:
            self._enqueue_partial_at[1](
                bins,
                grad,
                hess,
                feat_ids,
                partials,
                n_slots,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                n_tiles,
                threads,
            )

    def _enqueue_partial_at[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin,
        feat_origin: MutOrigin,
        part_origin: MutOrigin, //,
        GROUP: Int,
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        feat_ids: MutPointer[Int32, feat_origin],
        partials: MutPointer[Int32, part_origin],
        n_slots: Int,
        rows_per_tile: Int,
        begin: Int,
        count: Int,
        g_scale: Float32,
        h_scale: Float32,
        use_quant: Int32,
        const_hess: Int32,
        n_tiles: Int,
        threads: Int,
    ) raises:
        """The tiled launch at one group width, over the four bin capacities.

        `blocks` is `ceil(n_slots / GROUP)` at every rung, which reproduces
        the grids the hand-written variants launched exactly: `n_slots` at
        one, `(n_slots + 1) // 2` at the pairing, `(n_slots + 3) // 4` at the
        quad. A tail block owns the slots that remain and the kernel handles
        it, so nothing here rounds the slot count up.

        `unroll` is the row-walk arm (`set_row_unroll`), read from the field
        here rather than threaded through the family dispatch above, because
        it changes no grid, no threadgroup footprint, and no histogram.
        """
        var blocks = (n_slots + GROUP - 1) // GROUP
        var unroll = Int32(1) if self.row_unroll else Int32(0)
        # `narrow_index` is already false whenever the shape does not admit
        # it (`set_narrow_index` refuses such a request), so the conjunction
        # is belt and braces: a field that could only have been set through a
        # checked setter is re-checked at the one place it reaches a kernel.
        var narrow = (
            Int32(1)
            if (self.narrow_index and self.narrow_index_supported())
            else Int32(0)
        )
        var palign = Int32(1) if self.pair_alignment else Int32(0)
        # The decomposition probes, resolved from the two fields into the
        # one Int32 the kernel reads. Off unless one of the two acknowledged
        # setters was called, which no shipping path does; read from the
        # fields here on the same footing as the arms above, and the only
        # ones of them that change what the launch computes.
        #
        # The two fields cannot both be live -- each setter refuses while the
        # other is -- so the order of this resolution is not a precedence
        # rule, it is a spelling. `HIST_PROBE_OFF` is zero, so a launch with
        # neither set passes exactly the zero it passed before this lane.
        var probe = (
            Int32(HIST_PROBE_NO_ATOMICS)
            if self.hist_atomic_probe
            else Int32(self.hist_probe_mode)
        )
        # The bin layout arm. `blocked_row_stride` is 1 unless
        # `set_blocked_layout` put the `[block][row][G]` buffer on the device
        # and `_ensure_blocked` filled it, and the second pointer is the
        # buffer itself. Both are read from the fields here rather than
        # threaded through the family dispatch above, for the same reason
        # `unroll`, `narrow` and `palign` are: the arm changes no grid, no
        # threadgroup footprint, and no histogram.
        #
        # The blocked buffer is passed at every launch whether or not it is
        # selected, because a kernel argument must be a real pointer and must
        # not be a buffer this launch already passes. It is a one-byte
        # placeholder until a layout is requested, and the kernel never
        # dereferences it at stride one.
        var rstride = Int32(self.blocked_row_stride())
        var blocked_ptr = self.blocked_dev.unsafe_ptr()
        # The packed bin arm, resolved the same way and passed at every launch
        # for the same reason the blocked buffer is: a kernel argument must be
        # a real pointer, and both are one-byte placeholders until a layout is
        # asked for. `packed_bins_active` is 0 until `_ensure_packed` has
        # actually filled the buffer, so a requested but unbuilt layout reads
        # the feature-major matrix rather than an uninitialized one -- which
        # matters more here than anywhere else in this file, because an
        # unbuilt packed buffer decodes to legal bin ids belonging to no row
        # and nothing downstream would catch it.
        var packed_on = Int32(1) if self.packed_bins_active() else Int32(0)
        var packed_ptr = self.packed_dev.unsafe_ptr()
        var pack_tab_ptr = self.packed_tab_dev.unsafe_ptr()
        # --- row-compaction lane ---
        # The three pointers the mechanism swaps, and it swaps nothing else.
        # When the compacted planes are live, the launch is handed `cbins`,
        # the identity index, and `cgq` in place of the caller's `bins`, the
        # permutation, and `gq`. Every thread then reads, at the identical
        # step of the identical loop, the byte the un-compacted launch would
        # have read at `bins[f * n_rows + rows[j]]` -- because that is exactly
        # what `cbins[f * n_rows + j]` is defined to hold. So the histogram is
        # bit-identical by construction, not by an argument about the order of
        # integer adds, and the accumulation loop is untouched: this is an
        # argument selection at the launch and nothing below it changes.
        #
        # Resolved here rather than threaded through the family dispatch for
        # the reason `unroll`, `narrow` and `blocked_ptr` are: the dispatch is
        # already eighteen arguments wide and a pointer cannot be handed down
        # from a caller that also holds `mut self`.
        var hist_bins = _any_origin_u8(bins)
        var hist_rows = _any_origin_i32(self.rows_dev.unsafe_ptr())
        var hist_gq = _any_origin_i32(self.gq_dev.unsafe_ptr())
        if self.row_compaction_live():
            hist_bins = _any_origin_u8(self.cbins_dev.unsafe_ptr())
            hist_rows = _any_origin_i32(self.ident_dev.unsafe_ptr())
            hist_gq = _any_origin_i32(self.cgq_dev.unsafe_ptr())
        if self.bin_cap <= 32:
            self.ctx.enqueue_function[_range_hist_partial_kernel[GROUP, 32]](
                hist_bins,
                hist_rows,
                grad,
                hess,
                hist_gq,
                feat_ids,
                partials,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                unroll,
                narrow,
                palign,
                blocked_ptr,
                rstride,
                probe,
                packed_ptr,
                pack_tab_ptr,
                packed_on,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )
        elif self.bin_cap <= 64:
            self.ctx.enqueue_function[_range_hist_partial_kernel[GROUP, 64]](
                hist_bins,
                hist_rows,
                grad,
                hess,
                hist_gq,
                feat_ids,
                partials,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                unroll,
                narrow,
                palign,
                blocked_ptr,
                rstride,
                probe,
                packed_ptr,
                pack_tab_ptr,
                packed_on,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )
        elif self.bin_cap <= 128:
            self.ctx.enqueue_function[_range_hist_partial_kernel[GROUP, 128]](
                hist_bins,
                hist_rows,
                grad,
                hess,
                hist_gq,
                feat_ids,
                partials,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                unroll,
                narrow,
                palign,
                blocked_ptr,
                rstride,
                probe,
                packed_ptr,
                pack_tab_ptr,
                packed_on,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )
        else:
            self.ctx.enqueue_function[_range_hist_partial_kernel[GROUP, 256]](
                hist_bins,
                hist_rows,
                grad,
                hess,
                hist_gq,
                feat_ids,
                partials,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                unroll,
                narrow,
                palign,
                blocked_ptr,
                rstride,
                probe,
                packed_ptr,
                pack_tab_ptr,
                packed_on,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )

    def _enqueue_atomic_family[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin,
        feat_origin: MutOrigin,
        out_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        feat_ids: MutPointer[Int32, feat_origin],
        out_hist: MutPointer[Int32, out_origin],
        n_slots: Int,
        hist_size: Int,
        rows_per_tile: Int,
        begin: Int,
        count: Int,
        g_scale: Float32,
        h_scale: Float32,
        sub_offset: Int,
        do_sub: Int32,
        use_quant: Int32,
        const_hess: Int32,
        n_tiles: Int,
        threads: Int,
        use_desc: Int32,
    ) raises:
        """The atomic twin of `_enqueue_partial_family`, same static tree."""
        var group = self._launch_group(n_slots)
        if group >= 16:
            self._enqueue_atomic_at[16](
                bins,
                grad,
                hess,
                feat_ids,
                out_hist,
                n_slots,
                hist_size,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                sub_offset,
                do_sub,
                use_quant,
                const_hess,
                n_tiles,
                threads,
                use_desc,
            )
        elif group >= 8:
            self._enqueue_atomic_at[8](
                bins,
                grad,
                hess,
                feat_ids,
                out_hist,
                n_slots,
                hist_size,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                sub_offset,
                do_sub,
                use_quant,
                const_hess,
                n_tiles,
                threads,
                use_desc,
            )
        elif group >= 4:
            self._enqueue_atomic_at[4](
                bins,
                grad,
                hess,
                feat_ids,
                out_hist,
                n_slots,
                hist_size,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                sub_offset,
                do_sub,
                use_quant,
                const_hess,
                n_tiles,
                threads,
                use_desc,
            )
        elif group >= 2:
            self._enqueue_atomic_at[2](
                bins,
                grad,
                hess,
                feat_ids,
                out_hist,
                n_slots,
                hist_size,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                sub_offset,
                do_sub,
                use_quant,
                const_hess,
                n_tiles,
                threads,
                use_desc,
            )
        else:
            self._enqueue_atomic_at[1](
                bins,
                grad,
                hess,
                feat_ids,
                out_hist,
                n_slots,
                hist_size,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                sub_offset,
                do_sub,
                use_quant,
                const_hess,
                n_tiles,
                threads,
                use_desc,
            )

    def _enqueue_atomic_at[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin,
        feat_origin: MutOrigin,
        out_origin: MutOrigin, //,
        GROUP: Int,
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        feat_ids: MutPointer[Int32, feat_origin],
        out_hist: MutPointer[Int32, out_origin],
        n_slots: Int,
        hist_size: Int,
        rows_per_tile: Int,
        begin: Int,
        count: Int,
        g_scale: Float32,
        h_scale: Float32,
        sub_offset: Int,
        do_sub: Int32,
        use_quant: Int32,
        const_hess: Int32,
        n_tiles: Int,
        threads: Int,
        use_desc: Int32,
    ) raises:
        """The atomic launch at one group width, over the four bin
        capacities.

        `unroll` is the row-walk arm (`set_row_unroll`), read from the field
        here rather than threaded through the family dispatch above, because
        it changes no grid, no threadgroup footprint, and no histogram."""
        var blocks = (n_slots + GROUP - 1) // GROUP
        var unroll = Int32(1) if self.row_unroll else Int32(0)
        # `narrow_index` is already false whenever the shape does not admit
        # it (`set_narrow_index` refuses such a request), so the conjunction
        # is belt and braces: a field that could only have been set through a
        # checked setter is re-checked at the one place it reaches a kernel.
        var narrow = (
            Int32(1)
            if (self.narrow_index and self.narrow_index_supported())
            else Int32(0)
        )
        var palign = Int32(1) if self.pair_alignment else Int32(0)
        # The decomposition probes, resolved from the two fields into the
        # one Int32 the kernel reads. Off unless one of the two acknowledged
        # setters was called, which no shipping path does; read from the
        # fields here on the same footing as the arms above, and the only
        # ones of them that change what the launch computes.
        #
        # The two fields cannot both be live -- each setter refuses while the
        # other is -- so the order of this resolution is not a precedence
        # rule, it is a spelling. `HIST_PROBE_OFF` is zero, so a launch with
        # neither set passes exactly the zero it passed before this lane.
        var probe = (
            Int32(HIST_PROBE_NO_ATOMICS)
            if self.hist_atomic_probe
            else Int32(self.hist_probe_mode)
        )
        # The bin layout arm. `blocked_row_stride` is 1 unless
        # `set_blocked_layout` put the `[block][row][G]` buffer on the device
        # and `_ensure_blocked` filled it, and the second pointer is the
        # buffer itself. Both are read from the fields here rather than
        # threaded through the family dispatch above, for the same reason
        # `unroll`, `narrow` and `palign` are: the arm changes no grid, no
        # threadgroup footprint, and no histogram.
        #
        # The blocked buffer is passed at every launch whether or not it is
        # selected, because a kernel argument must be a real pointer and must
        # not be a buffer this launch already passes. It is a one-byte
        # placeholder until a layout is requested, and the kernel never
        # dereferences it at stride one.
        var rstride = Int32(self.blocked_row_stride())
        var blocked_ptr = self.blocked_dev.unsafe_ptr()
        # The packed bin arm, resolved the same way and passed at every launch
        # for the same reason the blocked buffer is: a kernel argument must be
        # a real pointer, and both are one-byte placeholders until a layout is
        # asked for. `packed_bins_active` is 0 until `_ensure_packed` has
        # actually filled the buffer, so a requested but unbuilt layout reads
        # the feature-major matrix rather than an uninitialized one -- which
        # matters more here than anywhere else in this file, because an
        # unbuilt packed buffer decodes to legal bin ids belonging to no row
        # and nothing downstream would catch it.
        var packed_on = Int32(1) if self.packed_bins_active() else Int32(0)
        var packed_ptr = self.packed_dev.unsafe_ptr()
        var pack_tab_ptr = self.packed_tab_dev.unsafe_ptr()
        # --- row-compaction lane ---
        # The three pointers the mechanism swaps, and it swaps nothing else.
        # When the compacted planes are live, the launch is handed `cbins`,
        # the identity index, and `cgq` in place of the caller's `bins`, the
        # permutation, and `gq`. Every thread then reads, at the identical
        # step of the identical loop, the byte the un-compacted launch would
        # have read at `bins[f * n_rows + rows[j]]` -- because that is exactly
        # what `cbins[f * n_rows + j]` is defined to hold. So the histogram is
        # bit-identical by construction, not by an argument about the order of
        # integer adds, and the accumulation loop is untouched: this is an
        # argument selection at the launch and nothing below it changes.
        #
        # Resolved here rather than threaded through the family dispatch for
        # the reason `unroll`, `narrow` and `blocked_ptr` are: the dispatch is
        # already eighteen arguments wide and a pointer cannot be handed down
        # from a caller that also holds `mut self`.
        var hist_bins = _any_origin_u8(bins)
        var hist_rows = _any_origin_i32(self.rows_dev.unsafe_ptr())
        var hist_gq = _any_origin_i32(self.gq_dev.unsafe_ptr())
        if self.row_compaction_live():
            hist_bins = _any_origin_u8(self.cbins_dev.unsafe_ptr())
            hist_rows = _any_origin_i32(self.ident_dev.unsafe_ptr())
            hist_gq = _any_origin_i32(self.cgq_dev.unsafe_ptr())
        # Inert when `use_desc` is zero, which is every host-argument caller;
        # the descriptor `set_descriptor_target` selected otherwise. Read here
        # rather than threaded through `_enqueue_atomic_family` for the same
        # reason `unroll` is: the dispatch above is already eighteen arguments
        # wide, and a pointer cannot be handed down from a caller that also
        # holds `mut self`.
        var desc = self._desc_buffer()
        if self.bin_cap <= 32:
            self.ctx.enqueue_function[_range_hist_atomic_kernel[GROUP, 32]](
                hist_bins,
                hist_rows,
                grad,
                hess,
                hist_gq,
                feat_ids,
                out_hist,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(hist_size),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                Int32(sub_offset),
                do_sub,
                use_quant,
                const_hess,
                desc.unsafe_ptr(),
                use_desc,
                unroll,
                narrow,
                palign,
                blocked_ptr,
                rstride,
                probe,
                packed_ptr,
                pack_tab_ptr,
                packed_on,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )
        elif self.bin_cap <= 64:
            self.ctx.enqueue_function[_range_hist_atomic_kernel[GROUP, 64]](
                hist_bins,
                hist_rows,
                grad,
                hess,
                hist_gq,
                feat_ids,
                out_hist,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(hist_size),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                Int32(sub_offset),
                do_sub,
                use_quant,
                const_hess,
                desc.unsafe_ptr(),
                use_desc,
                unroll,
                narrow,
                palign,
                blocked_ptr,
                rstride,
                probe,
                packed_ptr,
                pack_tab_ptr,
                packed_on,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )
        elif self.bin_cap <= 128:
            self.ctx.enqueue_function[_range_hist_atomic_kernel[GROUP, 128]](
                hist_bins,
                hist_rows,
                grad,
                hess,
                hist_gq,
                feat_ids,
                out_hist,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(hist_size),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                Int32(sub_offset),
                do_sub,
                use_quant,
                const_hess,
                desc.unsafe_ptr(),
                use_desc,
                unroll,
                narrow,
                palign,
                blocked_ptr,
                rstride,
                probe,
                packed_ptr,
                pack_tab_ptr,
                packed_on,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )
        else:
            self.ctx.enqueue_function[_range_hist_atomic_kernel[GROUP, 256]](
                hist_bins,
                hist_rows,
                grad,
                hess,
                hist_gq,
                feat_ids,
                out_hist,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(hist_size),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                Int32(sub_offset),
                do_sub,
                use_quant,
                const_hess,
                desc.unsafe_ptr(),
                use_desc,
                unroll,
                narrow,
                palign,
                blocked_ptr,
                rstride,
                probe,
                packed_ptr,
                pack_tab_ptr,
                packed_on,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )

    def _ensure_quantized[
        grad_origin: MutOrigin,
        hess_origin: MutOrigin, //
    ](
        mut self,
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        g_scale: Float32,
        h_scale: Float32,
    ) raises:
        """Rebuild the interleaved quantized gradient buffer unless it was
        built for this tree at these scales already. One streaming launch
        over `n_rows`, ordered before the histogram that reads it by the
        queue.

        **The width is part of the cache key.** `quant_packed` records which
        layout the buffer currently holds, so an A/B that moves
        `set_packed_gradients` between repeats in one process restages rather
        than handing the next histogram a buffer of the other width. The
        setter also drops `quant_valid`, which makes this belt and braces; it
        is here because the two states are not the same thing and a later
        edit that changed one would otherwise not have to think about it.

        **What the packed arm costs, stated so it is not discovered.** One
        eight-byte upload to zero the counters, and one `synchronize` to read
        them back. That is one host wait per *rebuild*, and a rebuild happens
        when `begin_tree` invalidates the buffer or the round's scales move,
        so it is one wait per tree and not one per node. It buys the right to
        raise **before** the histogram is enqueued rather than after it has
        run, which is the difference between refusing a fit and reporting on
        a corrupt one; `histogram_gpu._check_window_bound`, the check of the
        same family that cannot get ahead of its own data, has to settle for
        the latter and says so.

        Nothing on the Int32 arm waits, and nothing on it was moved.
        """
        if (
            self.quant_valid
            and self.quant_packed == self.packed_gradients
            and self.quant_g_scale == g_scale
            and self.quant_h_scale == h_scale
        ):
            return
        var threads = self.block_threads
        var blocks = (self.n_rows + threads - 1) // threads
        if self.packed_gradients:
            # Zero the counters from a host source rather than from a kernel:
            # eight bytes on the same queue, ordered before the launch that
            # increments them, and one fewer launch than a zeroing kernel.
            self.ctx.enqueue_copy(
                dst_buf=self.stage16_flag_dev,
                src_ptr=self.stage16_zero.unsafe_ptr(),
            )
            self.ctx.enqueue_function[_quantize_grad_hess_i16_kernel](
                grad,
                hess,
                self.gq_dev.unsafe_ptr().unsafe_bitcast[Int16](),
                self.stage16_flag_dev.unsafe_ptr(),
                Int32(self.n_rows),
                g_scale,
                h_scale,
                grid_dim=blocks,
                block_dim=threads,
            )
            self._check_stage16_bound(g_scale, h_scale)
        else:
            self.ctx.enqueue_function[_quantize_grad_hess_kernel](
                grad,
                hess,
                self.gq_dev.unsafe_ptr(),
                Int32(self.n_rows),
                g_scale,
                h_scale,
                grid_dim=blocks,
                block_dim=threads,
            )
        self.quant_valid = True
        self.quant_packed = self.packed_gradients
        self.quant_g_scale = g_scale
        self.quant_h_scale = h_scale
        # `gq` just moved under the compacted copy of it, so the invariant no
        # longer holds and `_ensure_compacted` -- which the two histogram
        # entry points call immediately after this one -- rebuilds both planes
        # together. Only `cgq` is actually stale, and rebuilding `cbins` with
        # it is deliberate: a partial rebuild would need a second kernel and a
        # second validity flag to save a pass that, on every path this backend
        # takes, happens once per tree at the same moment the full rebuild
        # would have happened anyway.
        self.compact_valid = False

    def _check_stage16_bound(
        mut self, g_scale: Float32, h_scale: Float32
    ) raises:
        """Read the two overflow counters the Int16 staging pass wrote and
        refuse the fit if either plane that will be gathered lost a row.

        The bound this enforces is
        `-32768 <= Int32(round(x * scale)) <= 32767`, **per row, over every
        row of the round**, which is the scope the representation needs: one
        buffer is staged per round and every node of the tree gathers it, so
        a bound holding for one node's rows would license nothing for the
        others. `quantized_gradient.check_int16_staging` is the same
        inequality written against a magnitude, for a host test; the device
        evaluates it against the integer it has just formed, which needs no
        reduction and cannot disagree with the value it stored.

        **The hessian counter is consulted only when the hessian is read.** On
        a constant-hessian round `_hist_rows_step` never gathers the second
        word, so a hessian that did not fit is a dead byte and not a wrong
        histogram. Refusing on it would reject rounds this arm serves exactly.
        The gradient counter is consulted always, because there is no arm on
        which the gradient word is unread.

        Raising and not clamping, for the reason
        `histogram_gpu._check_window_bound` gives at greater length: a clamped
        row is a gradient the fit never had, on one plane, for one round, and
        no fixture tells it from a data change. The whole value of this arm is
        that it is bit-identical or it is nothing.
        """
        self.ctx.enqueue_copy(
            dst_ptr=self.stage16_flag_host.unsafe_ptr(),
            src_buf=self.stage16_flag_dev,
        )
        self.ctx.synchronize()
        var flags = self.stage16_flag_host.unsafe_ptr()
        var n_g = Int(flags.unsafe_load(STAGE16_FLAG_GRAD))
        var n_h = Int(flags.unsafe_load(STAGE16_FLAG_HESS))
        if n_g > 0:
            _raise_stage16("gradient", n_g, self.n_rows, Float64(g_scale))
        if n_h > 0 and not self.constant_hessian:
            _raise_stage16("hessian", n_h, self.n_rows, Float64(h_scale))

    def download_rows(mut self) raises -> List[Int32]:
        """The whole active-row buffer, host side. Synchronizes, and is for
        tests and debugging: training never needs the permutation on the
        host."""
        # A deferred copy-back means the split window's rows are in `scratch`
        # and not in `rows_dev` yet, so this would return a permutation correct
        # everywhere except the one window a reader is most likely looking at.
        # See `set_partition_fusion`.
        self._refuse_copy_back_debt("downloading the active rows")
        self.ctx.enqueue_copy(
            dst_ptr=self.host_rows.unsafe_ptr(), src_buf=self.rows_dev
        )
        self.ctx.synchronize()
        var out = List[Int32](capacity=self.n_rows)
        var src = self.host_rows.unsafe_ptr()
        for r in range(self.n_rows):
            out.append(src.unsafe_load(r))
        return out^

    def download_range(mut self, node: Int) raises -> List[Int]:
        """One node's rows, in compacted order. Tests compare this against
        the CPU grower's row list for the same node."""
        var window = self.ranges.get(node)
        var all_rows = self.download_rows()
        var out = List[Int](capacity=window.count())
        for j in range(window.begin, window.end):
            out.append(Int(all_rows[j]))
        return out^
