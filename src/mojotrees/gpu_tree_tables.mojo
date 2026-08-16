"""Device-resident frontier and tree tables, and the kernel that picks a leaf.

Why this module exists, in numbers
----------------------------------
A stage-level profile taken on an Apple M4 at 1,000,000 rows by 50 features
over 100 trees attributed 49.3% of the run to histogram accumulation, 28.2%
to transfer, 17.7% to the row partition, and 2.4% to the split search. The
same profile counted 24,400 device dispatches and 3,100 host
synchronizations, which is 31 synchronizations per tree against a 31-leaf
budget: exactly one host round trip per committed split.

The host compute inside one of those round trips is small. Measured
separately, the host spends roughly 15 to 20 microseconds per split deciding
which leaf to split and writing the node, against a measured wait cost of
about 126 microseconds for the synchronization itself. The cost is therefore
not the deciding, it is the draining: every decision empties the device
queue, blocks, and then refills a pipeline that was already full.

The end state that removes it is a device that picks the leaf and commits the
split itself, so the host waits once per tree instead of once per split. This
module builds two thirds of that: the tables the device needs in order to
hold a frontier and a tree, and the single kernel that reduces the frontier
to a winner and commits it. It wires none of it. Nothing in
`train_gpu.mojo` calls anything here, `MOJOTREES_GPU_TREE_RESIDENT` is off by
default, and no shipping fit changes in any way.

**No speedup is claimed and none was measured.** This lane ran no benchmark
and no training run. What it can claim is an equivalence, and only that: the
kernel below reaches the same decision, and writes the same split, as the
host path in `train_gpu._device_search_resident` reaches and writes, over the
configurations named under "What the device can and cannot decide".

What a device mirror is, and what it is not
--------------------------------------------
Almost everything here already exists on the host and the job is a
translation rather than a new model. The correspondence is exact and worth
reading before the layouts:

    host                                    device mirror here
    ----                                    ------------------
    `gpu_frontier.FrontierLeaf`             one `FRONT_WORDS` row of `front`
    `gpu_active_rows.LeafRange`             `FRONT_ROW_BEGIN`/`FRONT_ROW_COUNT`
    `gpu_leaf_batching.HistogramSlotPool`   the `slot_owner` vector
    `tree.Tree`'s flat arrays               `node_i` and `node_f` rows
    `Tree`'s node counter                   `CTR_NEXT_NODE`
    `_device_search_resident`'s `n_leaves`  `CTR_N_LIVE`

The one structure that is genuinely new is the counter block, and it is new
only because a host loop keeps those numbers in local variables where a
device has to keep them somewhere addressable.

Row windows rather than row lists. A frontier leaf owns a half-open window
into the active-row permutation, and a split rewrites only that window, which
is what `gpu_active_rows.mojo` already makes structural. So a committed split
splits a window into two adjacent windows at `begin + n_left` and needs no
row data at all. The commit below is therefore integer bookkeeping over six
words per leaf, which is why it fits in one thread.

Layout conventions
------------------
Every table is a flat device buffer of Int32 or Float32, addressed by a row
stride and a word offset, in exactly the style `gpu_split_search.mojo` uses
for its split records: integers and floats live in separate buffers so no
value is ever bit-cast between the two, and a "row" is a slice of both. The
constants below are the whole of the layout and nothing else in the package
depends on it.

**Record slot equals frontier slot.** A frontier leaf carries the index of
the split record that describes its best split. `FRONT_RECORD` holds that
index explicitly, so the pick kernel never assumes the identity; but
`begin_tree` establishes it (root leaf at slot 0 reads record 0) and the
commit preserves it (the left child keeps the parent's frontier slot and
therefore its record, and the right child is appended at slot `n_live` and
takes record `n_live`). A wiring lane staging the next search should stage
record `i` for frontier slot `i`, which is the same convention
`GpuSplitSearcher.enqueue_frontier` already uses for a batch.

**Slot order is the tie-breaking rule.** The frontier is the trainer's list,
in the trainer's order: a commit overwrites the parent's slot with the left
child and appends the right child. That is what `LeafFrontier.apply_commit`
does, what `_device_search_resident` does with
`frontier[index] = ...; frontier.append(...)`, and it is not an
implementation detail, because the leaf-wise pick breaks ties toward the
lower slot. Changing the order silently grows a different tree.

The tie-break, found rather than assumed
-----------------------------------------
The host rule was read out of the source, not guessed at. It lives in
`growth_policy.GrowthSchedule.next_leaf` under `GROW_LEAFWISE`:

    var best = -1
    var best_gain = 0.0
    for i in range(len(candidates)):
        if candidates[i].eligible and candidates[i].gain > best_gain:
            best_gain = candidates[i].gain
            best = i

Three properties matter and all three are preserved below.

1. The comparison is **strict**. A candidate equal to the running best does
   not displace it.
2. The scan runs in **ascending slot order**. Combined with the strict
   comparison, the winner among equal gains is the lowest-indexed one.
3. `best_gain` starts at **0.0**, so a nonpositive gain never wins. That is
   redundant with `eligible`, which already requires `gain > 0.0`, and both
   are reproduced anyway so a future change to either stays visible.

The same rule is written twice more in the package, in
`gpu_frontier.LeafFrontier.select_best` and in
`gpu_frontier.speculative_order`, both with the same strict comparison over
the same ascending order, and it is stated in the docstring of
`gpu_split_search._pick_best_record_kernel` as "ties going to the lower
record index". Four sites, one rule.

Preserving it under a parallel reduction is the only delicate part of the
kernel, because `block.max` over gains returns a value and not a position.
The resolution is the one `_reduce_slots_block_kernel` already uses for its
cross-feature reduction, and this module copies that pattern rather than
editing the file it lives in (see "Relationship to `_pick_best_record_kernel`"
below):

- Each thread walks its strided share of the frontier **in ascending slot
  order** with a strict `>`, so a tie inside one thread keeps the lower slot.
- `block.max` over the per-thread gains gives the winning gain.
- Every thread whose own gain equals that winner contributes its slot to a
  `block.min`; every other thread contributes `NO_PICK`, which is `Int32.MAX`.
  The minimum is therefore the lowest slot among the tied winners.

Carrying the index alongside the gain and resolving toward the lower index is
what makes the parallel answer identical to the serial one rather than merely
equal in gain. Getting it wrong would not produce an error; it would produce a
different tree, which is why it has its own assertions in the test file.

Why the collectives are allowed here at all. `block.max` and `block.min`
reassociate, and reassociation is exact on values with no NaN and no signed
zero, which gains and slot indices are. No floating-point **sum** crosses a
thread boundary anywhere in this module. That distinction is the same one
`gpu_split_search.mojo` states at length for its own collectives, and it is
what licenses a block argmax where a block float sum would be forbidden.

Arithmetic, and the contraction rule
-------------------------------------
`docs/NUMERICS.md` records that this codebase's floats contract into fused
multiply-adds under the default `--fp-mode contract=fast`, that binding a
product to a local does not stop it, and that the only stable fix is to move
a multiply out of a kernel entirely. Two incidents in this optimization round
came from a multiply moving relative to an add.

This kernel is therefore built so that the question does not arise. **It
introduces no floating-point arithmetic of any kind.** Gains are compared,
never combined. Leaf values, the split gain, and the child statistics are
copied word for word out of the split record that
`gpu_split_search._reduce_slots_block_kernel` already wrote, so every float
this module stores was produced by an expression this module does not
contain. There is no multiply and no add on the gain path here, so there is
nothing for the optimizer to fuse.

All of the kernel's own arithmetic is integer and exact: `begin + n_left` for
the right child's window, `depth + 1` for the children's depth,
`next_node + 2` for the node counter, and `2 * min_data_in_leaf` for the
shape rule. Integer addition is associative, so none of it is order sensitive
either.

What the device can and cannot decide
--------------------------------------
This is the list a wiring lane needs, and it is deliberately conservative.

**Expressible here, and implemented:**

- The leaf-wise pick itself: best gain anywhere in the frontier, ties to the
  lower slot.
- `max_depth`, as `depth >= max_depth` on the leaf being considered.
- `min_data_in_leaf` as the parent precondition
  `n_rows < 2 * min_data_in_leaf or n_rows < 2`. Note carefully what this is
  and is not: the **per-child** row floor is enforced inside the split scan
  kernel, which already takes `min_data_in_leaf` as a launch argument, so a
  candidate whose left or right child is too small never reaches a record.
  What the host applies on top, in `train_gpu._apply_shape_rules`, is the
  parent-level test above, and that is what is reproduced here. Both halves
  of "min_data_in_leaf on both children" are therefore enforced, in the two
  places the shipping path already enforces them.
- The positive-gain floor.
- The `num_leaves` budget.
- Node id assignment, left child before right, matching two consecutive
  `Tree._add_node` calls.
- The histogram slot pool: lowest free slot to the built child, the parent's
  slot reassigned to the derived child. `subtraction_builds_left` is
  `n_left <= n_right`, an integer comparison, so it is exact here.
- Writing the split, both child values, both child row counts, and the two
  child windows.

**Not expressible here, and refused rather than approximated:**

- **Monotone constraints.** The host clamps each child value into the
  parent's interval and, if rounding inverted the pair, collapses both to
  `(a + b) / 2.0`, all in Float64 (`train_gpu._commit_device_split`, over
  `monotone.midpoint` and `OutputBounds.clamp`). The clamp is a comparison
  and a select and would survive the move to Float32 exactly; the midpoint
  would not, and `child_bounds` computes a second midpoint to hand the
  children their intervals. A Float32 midpoint is a different number, and by
  `docs/NUMERICS.md` a different number is a different tree. So
  `tree_resident_supported` refuses a constrained fit outright.
- **Per-node feature subsampling** (`feature_fraction_bynode < 1.0`). The
  node's feature set is drawn from its node id, and the device does not have
  the RNG. `gpu_frontier.search_is_order_free` is the predicate for exactly
  this and it is the reason a commit order can matter at all.
- **Interaction constraints.** The allow mask a child's search reads is
  `params.constraints.allowed_features(branch)`, and `branch` is the ancestor
  feature chain, which nothing on the device tracks.
- **Depth-wise growth.** `GrowthSchedule` under `GROW_DEPTHWISE` ranks a
  whole level and admits a gain-ordered prefix under the budget. That is a
  different and larger reduction and this kernel does not attempt it.
- **`TreeParams.extra`** and `feature_fraction_bylevel`, which the device
  split search already refuses for its own reasons
  (`train_gpu._check_device_search_supported`).

So what a wiring lane gets from this module is the default configuration and
nothing more: leaf-wise growth, no monotone constraints, no interaction
constraints, no per-node feature draw. That covers the shipped defaults, and
`tree_resident_supported` names the reason for every configuration it does
not cover.

Relationship to `_pick_best_record_kernel`
-------------------------------------------
`gpu_split_search._pick_best_record_kernel` already exists, is already
tested, and is unused. It reduces a set of finished records to the best-gain
one, ties to the lower record index, on a single thread, and copies the
winner into a destination record slot. It was read first and it is most of
the selection, but it is not enough for this lane for three reasons: it knows
nothing about a frontier, so it cannot apply `max_depth`, the row floor, or
the leaf budget, which are properties of the leaf rather than of the record;
it produces a record rather than a commit, so nothing is written into a tree;
and it is serial, so the tie-break question the parallel reduction raises
never comes up in it.

`gpu_split_search.mojo` belongs to another lane this round, so its kernel is
left exactly as it is and the pattern is copied here instead. Concretely,
`_pick_and_commit_kernel` below takes its record-scanning and its
`block.max` / `block.min` argmax structure from that file's
`_reduce_slots_block_kernel`, and its "no candidate" handling from
`_pick_best_record_kernel`. If the two ever have to be reconciled, this is
the duplicate and that is the original.

Scope
-----
The host half of this module (the layout constants, the snapshot structs, the
gate) touches no device and is readable on a machine with no accelerator. The
device half opens a `DeviceContext` and is guarded the way every other GPU
entry point in this package is guarded, with the whole body behind
`comptime if not has_accelerator()`.
"""

from std.gpu import thread_idx
from std.os import getenv
from std.sys import has_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.primitives import block

from .categorical import (
    CAT_BITSET_WORDS,
    CatBitset,
    cat_add,
    cat_empty,
)
from .gpu_active_rows import (
    STEP_BUILT_BEGIN,
    STEP_BUILT_COUNT,
    STEP_BUILT_SLOT,
    STEP_CAT0,
    STEP_DEFAULT_LEFT,
    STEP_FEATURE,
    STEP_IS_CAT,
    STEP_LEFT_REC,
    STEP_LEFT_SLOT,
    STEP_LIVE,
    STEP_MISSING_BIN,
    STEP_RIGHT_REC,
    STEP_RIGHT_SLOT,
    STEP_ROW_BEGIN,
    STEP_ROW_COUNT,
    STEP_SUB_SLOT,
    STEP_THRESHOLD,
    STEP_WORDS,
)
from .gpu_split_search import NODE_HIST_BASE, NODE_WORDS
from .growth_policy import GROW_LEAFWISE
from .gpu_split_search import (
    CAT_WORD_BITS,
    CAT_WORDS,
    FLAG_CATEGORICAL,
    FLAG_DEFAULT_LEFT,
    FLAG_FOUND,
    FREC_GAIN,
    FREC_LEFT_VALUE,
    FREC_PARENT_VALUE,
    FREC_RIGHT_VALUE,
    IREC_BIN,
    IREC_CAT0,
    IREC_FEATURE,
    IREC_FLAGS,
    IREC_LEFT_COUNT,
    IREC_RIGHT_COUNT,
    MAX_SPLIT_BINS,
    SPLIT_FWORDS,
    SPLIT_IWORDS,
)
from .split import SplitInfo
from .tree import Tree, TreeParams


# --- The gate -------------------------------------------------------------


def tree_resident_requested() -> Bool:
    """`MOJOTREES_GPU_TREE_RESIDENT=1`, and off for every other value.

    Default off, and off means off: no shipping code path consults this, and
    nothing in `train_gpu.mojo` imports this module at all. It exists so that
    the wiring lane that eventually does has a switch already agreed on, and
    so that the test file can name the contract it is testing.

    Spelled as an equality against "1" rather than as a truthiness test, for
    the same reason `MOJOTREES_GPU_SPLIT_RESIDENT` is spelled as an equality
    against "0": an unset variable and a variable set to something
    unrecognized must land on the default, not on whatever a permissive
    parser makes of them.
    """
    return getenv("MOJOTREES_GPU_TREE_RESIDENT") == "1"


comptime TREE_RESIDENT_OK = 0
"""Every decision this configuration needs is one the device can make."""

comptime TREE_RESIDENT_MONOTONE = 1
"""A monotone constraint is active. The child-value clamp is a comparison and
would move to the device exactly; the midpoint collapse and `child_bounds`
are Float64 divisions by two on the host and a Float32 midpoint is a
different number. Refused rather than approximated."""

comptime TREE_RESIDENT_BYNODE = 2
"""`feature_fraction_bynode < 1.0`: a node's feature set is drawn from its
node id and the device has no RNG. This is also the predicate
`gpu_frontier.search_is_order_free` exists for."""

comptime TREE_RESIDENT_INTERACTION = 3
"""Interaction constraints are active. A child's allow mask comes from its
ancestor feature chain, which nothing on the device tracks."""

comptime TREE_RESIDENT_DEPTHWISE = 4
"""`grow_policy = depthwise`. A level is ranked and admitted as a
gain-ordered prefix under the budget, which is a different reduction than the
single argmax below."""

comptime TREE_RESIDENT_EXTRA = 5
"""`TreeParams.extra` is active, or `feature_fraction_bylevel != 1.0`. The
device split search already refuses both for its own reasons; see
`train_gpu._check_device_search_supported`."""


def tree_resident_reason_name(reason: Int) -> String:
    if reason == TREE_RESIDENT_OK:
        return String("ok")
    if reason == TREE_RESIDENT_MONOTONE:
        return String("monotone constraints")
    if reason == TREE_RESIDENT_BYNODE:
        return String("feature_fraction_bynode")
    if reason == TREE_RESIDENT_INTERACTION:
        return String("interaction constraints")
    if reason == TREE_RESIDENT_DEPTHWISE:
        return String("depthwise growth")
    if reason == TREE_RESIDENT_EXTRA:
        return String("extra tree parameters")
    return String("unknown")


def tree_resident_supported(params: TreeParams) raises -> Int:
    """Whether the device could own this fit's tree, and why not when it
    could not.

    A predicate rather than a raise, because the caller that will eventually
    ask this is choosing between two paths that both produce a correct tree,
    not validating user input. The order of the tests is the order a reader
    would want the reason reported in: the numeric refusal first, because it
    is the one with a stated cause in `docs/NUMERICS.md`, then the three
    order-and-provenance refusals, then the catch-all for what the split
    search already refuses.
    """
    if params.monotone.is_active():
        return TREE_RESIDENT_MONOTONE
    if params.feature_fraction_bynode != 1.0:
        return TREE_RESIDENT_BYNODE
    if params.constraints.n_groups() > 0:
        return TREE_RESIDENT_INTERACTION
    if params.grow_policy != GROW_LEAFWISE:
        return TREE_RESIDENT_DEPTHWISE
    if params.extra.is_active() or params.feature_fraction_bylevel != 1.0:
        return TREE_RESIDENT_EXTRA
    return TREE_RESIDENT_OK


# --- Frontier table layout ------------------------------------------------
#
# One row per frontier slot, six Int32 words. Everything a live leaf needs in
# order to be picked and committed, and nothing else: the branch feature
# chain, the interaction allow mask, and the monotone interval that
# `gpu_frontier.FrontierLeaf` also carries are all host bookkeeping that the
# refusals above put out of reach, so none of them crosses into a device
# table where it would only be dead weight.

comptime FRONT_NODE = 0
"""The tree node id this leaf is, which is also the id its split will be
written under."""

comptime FRONT_ROW_BEGIN = 1
"""Start of this leaf's window into the device-resident active-row
permutation. The device mirror of `gpu_active_rows.LeafRange.begin` and of
`gpu_frontier.FrontierLeaf.row_begin`, which are already held equal by
`GpuActiveRows.check_frontier`."""

comptime FRONT_ROW_COUNT = 2
"""Rows in the window. `LeafRange` stores an end rather than a count; a count
is stored here because every use in the kernel is a count (the shape rule,
the child windows, the subtraction choice) and the end is one addition
away."""

comptime FRONT_DEPTH = 3
"""Depth in edges from the root. Descends from the ancestor chain, so it is
invariant to commit order, which is what lets the depth rule be applied at
pick time rather than at leaf-creation time."""

comptime FRONT_HIST_SLOT = 4
"""The histogram pool slot holding this leaf's histogram. `slot_owner` is the
inverse map and the two are maintained together."""

comptime FRONT_RECORD = 5
"""Which split record describes this leaf's best split. Carried explicitly so
the pick never assumes the record-slot-equals-frontier-slot identity, even
though this module's own commit preserves it."""

comptime FRONT_WORDS = 6


# --- Tree table layout ----------------------------------------------------
#
# The flat arrays of `tree.Tree`, one row per node, split across an Int32
# table and a Float32 table. The field set is `Tree`'s exactly, minus
# `cat_offset`: the host tree stores category sets in a side pool addressed
# by an offset, which needs an allocator, so a device node carries its own
# 256-bit set inline instead and a host reader appends them into the pool in
# node order. At 16 words per node and at most `2 * num_leaves - 1` nodes,
# that is under 4 KB for a default tree.

comptime TN_FEATURE = 0
"""Split feature, or -1 for a leaf. `Tree` uses the same sentinel and the
same test (`feature[i] >= 0` means internal)."""

comptime TN_THRESHOLD = 1
"""Threshold bin, or -1 for a leaf and for a categorical split, matching
`Tree._set_split`."""

comptime TN_LEFT = 2
comptime TN_RIGHT = 3

comptime TN_DEFAULT_LEFT = 4
"""1 when missing rows route left. Held as a word rather than folded into a
flags field so that a host reader decodes it without a mask, and because the
tree table is small enough that the packing would buy nothing."""

comptime TN_MISSING_BIN = 5
"""The split feature's missing bin, or -1. Read from the per-feature table
this module uploads once, and forced to -1 on a categorical split exactly as
`Tree._set_split` does, since bin 0 already collects the missing rows there
and is never a set member."""

comptime TN_IS_CATEGORICAL = 6
comptime TN_COUNT = 7
"""Training rows the node covers, as an exact integer. `Tree.count` is a
Float64 and the host grower fills it with `Float64(n_left)`, so storing the
integer and widening on the host reproduces that value exactly rather than
approximately."""

comptime TN_CAT0 = 8
"""First of `CAT_WORDS` category-set words, in the split record's own 16-bit
packing, copied straight across so no bit is reinterpreted on the way."""

comptime TN_IWORDS = TN_CAT0 + CAT_WORDS

comptime TN_VALUE = 0
"""The node's output. For a child this is the raw Newton value the split
record carries, which under the refusals above is exactly the value
`_commit_device_split` would write, because an unconstrained clamp is the
identity."""

comptime TN_SPLIT_GAIN = 1
"""The gain of the split recorded on this node, or 0.0 for a leaf, matching
`Tree._set_split` and `Tree._add_node`."""

comptime TN_FWORDS = 2


# --- Counters -------------------------------------------------------------
#
# The scalars a host loop would keep in local variables. They live in one
# small Int32 buffer so a single download brings the whole state of a step
# home, which is what a caller checking a step needs and what a wiring lane
# would read once per tree rather than once per split.

comptime CTR_N_LIVE = 0
"""Live frontier slots, which is also the leaf count: a split removes one
leaf and adds two, so `n_leaves` and `len(frontier)` move together, exactly
as they do in `_device_search_resident`."""

comptime CTR_NEXT_NODE = 1
"""The id `Tree._add_node` would hand out next."""

comptime CTR_PICK = 2
"""Frontier slot committed by the last step, or -1 when none was."""

comptime CTR_PICK_NODE = 3
"""Node id of the leaf that was split, recorded because the commit overwrites
that frontier slot with the left child and the parent id would otherwise have
to be recovered from the tree."""

comptime CTR_STATUS = 4
comptime CTR_COMMITS = 5
comptime CTR_WORDS = 6


comptime TREE_RUNNING = 0
"""The step committed a split and growth may continue."""

comptime TREE_BUDGET_SPENT = 1
"""`num_leaves` is reached. The host loop's `while n_leaves <
params.num_leaves` is this test, and it is applied before the reduction for
the same reason the host applies it before `plan_level`."""

comptime TREE_NO_CANDIDATE = 2
"""No live leaf offered an admissible split: every one failed the found flag,
the positive-gain floor, the depth limit, or the row floor."""

comptime TREE_POOL_FULL = 3
"""The histogram slot pool had no free slot. The host sizes the pool for
`num_leaves` and the budget test above runs first, so this can only fire on a
wiring mistake; it is a status rather than a silent stop so that the mistake
is visible."""

comptime TREE_OVERFLOW = 4
"""The step would have written past the frontier or node table. Same
character as `TREE_POOL_FULL`: unreachable when the tables are sized from
`num_leaves`, reported rather than trusted."""


def tree_status_name(status: Int) -> String:
    if status == TREE_RUNNING:
        return String("running")
    if status == TREE_BUDGET_SPENT:
        return String("budget_spent")
    if status == TREE_NO_CANDIDATE:
        return String("no_candidate")
    if status == TREE_POOL_FULL:
        return String("pool_full")
    if status == TREE_OVERFLOW:
        return String("overflow")
    return String("unknown")


comptime NO_PICK = Int32(2147483647)
"""The identity a thread holding no candidate contributes to the `block.min`
over frontier slots. `Int32.MAX`, spelled out for the same reason
`gpu_split_search.NO_CANDIDATE` is: every real slot is a frontier index, far
below it, so the minimum can never mistake the identity for an answer."""

comptime PICK_THREADS = 64
"""Threads in the one threadgroup the pick kernel launches.

A warp multiple on every supported backend, which is what `block.max` and
`block.min` want, and the same width `gpu_split_search.REDUCE_SLOT_THREADS`
uses for the cross-feature reduction. It is a `comptime` constant rather than
a runtime value because the collectives take `block_size` as a compile-time
parameter in this release, and fixing it at one value is what keeps this
module from needing the host-side instantiation menu that a variable block
size would force.

Sixty-four threads over a frontier that is 31 leaves by default is a thread
per leaf and then some, which is deliberate: the reduction is not the cost
here, the launch is, and a narrower block would not make the launch cheaper.
"""


# --- Host-side snapshot ---------------------------------------------------
#
# What a download decodes into. Plain host structs with no device types, so a
# test can build an expected state, compare it field by field, and print it,
# and so the layout above has exactly one reader.


@fieldwise_init
struct DeviceLeafRow(Copyable, Movable):
    """One frontier row, decoded. The device mirror of
    `gpu_frontier.FrontierLeaf` narrowed to what a device commit reads."""

    var node: Int
    var row_begin: Int
    var row_count: Int
    var depth: Int
    var hist_slot: Int
    var record: Int

    def row_end(self) -> Int:
        return self.row_begin + self.row_count

    def same_as(self, other: DeviceLeafRow) -> Bool:
        """Field-for-field equality, no tolerance. Every field is an exact
        integer, so this is the whole comparison."""
        return (
            self.node == other.node
            and self.row_begin == other.row_begin
            and self.row_count == other.row_count
            and self.depth == other.depth
            and self.hist_slot == other.hist_slot
            and self.record == other.record
        )


struct DeviceNodeRow(Copyable, Movable):
    """One tree node, decoded.

    `value` and `split_gain` are kept as Float32 rather than widened, because
    the comparison a test wants is on bits and widening would invite a
    tolerance. The host tree stores them as Float64 and fills them from the
    same Float32 record words, so `Float64(row.value)` is the host's number
    exactly.
    """

    var feature: Int
    var threshold_bin: Int
    var left: Int
    var right: Int
    var default_left: Bool
    var missing_bin: Int
    var is_categorical: Bool
    var count: Int
    var cat_words: List[Int32]
    var value: Float32
    var split_gain: Float32

    def __init__(out self):
        """A leaf, as `Tree._add_node` leaves one: no split, no routing, no
        children, zero gain."""
        self.feature = -1
        self.threshold_bin = -1
        self.left = -1
        self.right = -1
        self.default_left = False
        self.missing_bin = -1
        self.is_categorical = False
        self.count = 0
        self.cat_words = List[Int32](capacity=CAT_WORDS)
        self.cat_words.resize(CAT_WORDS, Int32(0))
        self.value = Float32(0.0)
        self.split_gain = Float32(0.0)

    def is_leaf(self) -> Bool:
        return self.feature < 0

    def cat_bitset(self) -> CatBitset:
        """Reassemble the 256-bit category set from the 16-bit words.

        The same reconstruction `gpu_split_search._bitset_from_words`
        performs on a downloaded record, repeated here rather than imported
        because that function is private to its module. Bit 0 is skipped for
        the same reason it is skipped there: bin 0 collects the missing rows
        and is never a set member.
        """
        var bits = cat_empty()
        for b in range(1, MAX_SPLIT_BINS):
            var w = self.cat_words[b // CAT_WORD_BITS]
            if (w & Int32(1 << (b % CAT_WORD_BITS))) != Int32(0):
                cat_add(bits, b)
        return bits

    def to_split_info(self) -> SplitInfo:
        """The `SplitInfo` this node's split is, so a comparison against the
        host path can be made against the type the host path produces rather
        than against a field list. Mirrors
        `GpuSplitRecord.to_split_info`."""
        if self.feature < 0:
            return SplitInfo(-1, -1, 0.0, False)
        if self.is_categorical:
            return SplitInfo.categorical(
                self.feature, Float64(self.split_gain), self.cat_bitset()
            )
        return SplitInfo(
            self.feature,
            self.threshold_bin,
            Float64(self.split_gain),
            True,
            self.default_left,
        )

    def same_as(self, other: DeviceNodeRow) -> Bool:
        """Field-for-field equality with the two floats compared as bits.

        No tolerance anywhere, deliberately. A leaf value that differs in its
        last bit is a different model by the time it has been through a
        boosting round, which is the whole finding `docs/NUMERICS.md`
        records, so a comparison that would pass on a one-ulp difference
        would not be testing what this module has to guarantee.
        """
        if (
            self.feature != other.feature
            or self.threshold_bin != other.threshold_bin
            or self.left != other.left
            or self.right != other.right
            or self.default_left != other.default_left
            or self.missing_bin != other.missing_bin
            or self.is_categorical != other.is_categorical
            or self.count != other.count
        ):
            return False
        if self.value.to_bits() != other.value.to_bits():
            return False
        if self.split_gain.to_bits() != other.split_gain.to_bits():
            return False
        for w in range(CAT_WORDS):
            if self.cat_words[w] != other.cat_words[w]:
                return False
        return True


struct TreeTablesSnapshot(Copyable, Movable):
    """The whole device state of one step, brought home in one wait.

    A snapshot rather than a live view, because the point of the eventual
    wiring is that the host does not look at this per split. A caller that
    downloads is either finishing a tree or running a test.
    """

    var leaves: List[DeviceLeafRow]
    """Live frontier rows only, `[0, n_live)`, in slot order. Slot order is
    the tie-breaking rule, so the list order is load-bearing and not a
    presentation choice."""

    var nodes: List[DeviceNodeRow]
    """Nodes `[0, next_node)`, in id order, which is the order
    `Tree._add_node` created them in."""

    var slot_owner: List[Int]
    """Histogram pool: slot index to owning node id, or -1 for free. The
    device mirror of `gpu_leaf_batching.HistogramSlotPool.owner`."""

    var commit_order: List[Int]
    """Node ids in the order they were split, one entry per commit. Only
    `tree_from_snapshot` reads it, and only to append categorical sets into
    the flat pool in the order the host grower would have appended them; see
    `_pick_and_commit_kernel`. Empty on a tree with no categorical split, in
    the sense that nothing consults it, not in the sense that it is unfilled.
    """

    var n_live: Int
    var next_node: Int
    var pick: Int
    var pick_node: Int
    var status: Int
    var commits: Int

    def __init__(out self):
        self.leaves = List[DeviceLeafRow]()
        self.nodes = List[DeviceNodeRow]()
        self.slot_owner = List[Int]()
        self.commit_order = List[Int]()
        self.n_live = 0
        self.next_node = 0
        self.pick = -1
        self.pick_node = -1
        self.status = TREE_RUNNING
        self.commits = 0

    def slot_of_owner(self, owner: Int) -> Int:
        """The pool slot `owner` holds, or -1. Node ids are unique and a node
        holds at most one slot, so the answer is unique; the same statement
        `HistogramSlotPool.slot_of_owner` makes."""
        for i in range(len(self.slot_owner)):
            if self.slot_owner[i] == owner:
                return i
        return -1

    def check_invariants(self) raises:
        """The live windows must tile `[0, n_active)` and the leaves must
        hold distinct node ids, distinct histogram slots, and distinct record
        slots.

        The host mirror of this check is `LeafFrontier.check_invariants`, and
        holding both is what makes a disagreement between the device tables
        and the host frontier detectable instead of silent. Growth here is a
        few hundred leaves at most, so the quadratic form is cheaper than
        sorting and is what a test wants to read.
        """
        for i in range(len(self.leaves)):
            var a = self.leaves[i].copy()
            if a.row_count < 0:
                raise Error("a device leaf holds a negative row count")
            if a.row_begin < 0:
                raise Error("a device leaf begins before the active prefix")
            if a.hist_slot < 0 or a.hist_slot >= len(self.slot_owner):
                raise Error("a device leaf holds no histogram slot")
            if self.slot_owner[a.hist_slot] != a.node:
                raise Error(
                    "a device leaf and the slot pool disagree about who owns"
                    " a histogram slot"
                )
            for k in range(i + 1, len(self.leaves)):
                var b = self.leaves[k].copy()
                if a.node == b.node:
                    raise Error("two device leaves share a node id")
                if a.hist_slot == b.hist_slot:
                    raise Error("two device leaves share a histogram slot")
                if a.record == b.record:
                    raise Error("two device leaves share a record slot")
                if a.row_count > 0 and b.row_count > 0:
                    if (
                        a.row_begin < b.row_end()
                        and b.row_begin < a.row_end()
                    ):
                        raise Error("two device leaves overlap")
        if len(self.leaves) != self.n_live:
            raise Error("live leaf count disagrees with the frontier length")
        # A tree of `n_live` leaves has `2 * n_live - 1` nodes, because every
        # split adds exactly two. This is the cheapest available check that
        # the node counter and the frontier moved together.
        if self.next_node != 2 * self.n_live - 1:
            raise Error(
                "the node counter and the live leaf count disagree; a tree of"
                " L leaves holds 2L-1 nodes"
            )


# --- The kernel -----------------------------------------------------------


def _pick_and_commit_kernel(
    front: MutPointer[Int32, MutAnyOrigin],
    node_i: MutPointer[Int32, MutAnyOrigin],
    node_f: MutPointer[Float32, MutAnyOrigin],
    ctr: MutPointer[Int32, MutAnyOrigin],
    slot_owner: MutPointer[Int32, MutAnyOrigin],
    missing: MutPointer[Int32, MutAnyOrigin],
    rec_i: MutPointer[Int32, MutAnyOrigin],
    rec_f: MutPointer[Float32, MutAnyOrigin],
    step: MutPointer[Int32, MutAnyOrigin],
    order: MutPointer[Int32, MutAnyOrigin],
    num_leaves: Int32,
    max_depth: Int32,
    min_data_in_leaf: Int32,
    pool_capacity: Int32,
    leaf_capacity: Int32,
    node_capacity: Int32,
):
    """One leaf-wise growth step, entirely on the device.

    Launched as a single threadgroup of `PICK_THREADS` threads over a grid of
    one block. One block and not more, for two reasons that are both about
    the size of the problem rather than about the size of the machine: the
    reduction is over at most `num_leaves` entries, which is 31 by default
    and a few hundred at the top of the useful range, so a second block would
    have nothing to do; and the commit is inherently serial, so a
    multi-block form would need a device-wide barrier between the reduction
    and the commit, which this release does not offer, and would therefore
    cost a second launch to buy a reduction that is already free.

    The three phases, in order:

    **Budget.** `n_live >= num_leaves` ends the step before anything is read.
    This is the host's `while n_leaves < params.num_leaves`, and it is first
    for the same reason it is first there: a spent budget stops growth
    whatever the gains say. Both operands are uniform across the block, so
    the early return is uniform and no thread reaches a collective that the
    others skip.

    **Reduction.** Every thread walks its strided share of the frontier in
    ascending slot order, applies the shape rules to each leaf, and keeps its
    own best under a strict comparison. `block.max` then finds the winning
    gain and `block.min` over the slots of the threads that tied it finds the
    lowest such slot. See the module docstring for why that is exactly the
    host's rule and not merely an equivalent-looking one.

    **Commit.** Thread 0 alone assigns the two node ids, writes the split
    onto the parent, writes both children as leaves, moves the histogram slot
    pool, replaces the parent's frontier row with the left child, appends the
    right child, and advances the counters. Every write is to a location no
    other thread reads in this launch, so no barrier is needed after the two
    collectives.

    No floating-point arithmetic occurs anywhere in this kernel. Gains are
    compared; leaf values and the split gain are copied out of the record.
    See the contraction note in the module docstring.

    Two outputs beyond the tables, both added by the wiring lane:

    `step` is the step descriptor (`gpu_active_rows`'s `STEP_*` layout), the
    flat row the row partition and the child histogram read instead of taking
    their arguments from the host. It is the whole of what makes a
    device-owned tree possible: without it the host would have to read the
    commit back before it could enqueue the work the commit implies, which is
    the round trip this lane exists to remove. **Every exit writes
    `STEP_LIVE`**, and the three that commit nothing write it as zero, so a
    step enqueued past the end of growth reads one word and does nothing.
    That is what lets a whole tree's launches be enqueued before growth has
    decided how many splits there will be.

    `order` is the commit log: `order[k]` is the node id split by the k-th
    commit. It exists for one narrow reason and it is worth stating, because
    the reason is not obvious. The host tree stores a categorical node's
    category set in a flat side pool addressed by an offset, and
    `Tree._set_split` appends to that pool in *commit* order, while a reader
    decoding the device node table would naturally walk it in *node id*
    order. Under leaf-wise growth those two orders differ (the root splits
    into 1 and 2, then 2 may split before 1), so a decode that ignored the
    log would build a tree whose `cat_offset` values differed from the host's
    even though every routing decision was identical. The log costs one Int32
    per commit and removes that discrepancy entirely.
    """
    var tid = Int(thread_idx.x)
    var n_live = Int(ctr[unsafe_offset=CTR_N_LIVE][0])

    # Phase 1: the leaf budget, before anything is read.
    if n_live >= Int(num_leaves):
        if tid == 0:
            ctr[unsafe_offset=CTR_PICK] = Int32(-1)
            ctr[unsafe_offset=CTR_PICK_NODE] = Int32(-1)
            ctr[unsafe_offset=CTR_STATUS] = Int32(TREE_BUDGET_SPENT)
            step[unsafe_offset=STEP_LIVE] = Int32(0)
        return

    # Phase 2: this thread's best over its strided share of the frontier.
    #
    # Ascending slot order with a strict `>` is what keeps a tie inside one
    # thread on the lower slot, which is the host's rule reproduced at the
    # per-thread level before the collectives reproduce it across threads.
    # An admissible gain is always strictly positive, so 0.0 is a safe
    # identity for a thread that owns no admissible leaf, which includes
    # every thread past `n_live`.
    var my_gain = Float32(0.0)
    var my_slot = NO_PICK
    var s = tid
    while s < n_live:
        var fo = s * FRONT_WORDS
        var rec = Int(front[unsafe_offset = fo + FRONT_RECORD][0])
        var flags = rec_i[unsafe_offset = rec * SPLIT_IWORDS + IREC_FLAGS][0]
        var admissible = (flags & Int32(FLAG_FOUND)) != Int32(0)
        # The shape rules, which are `train_gpu._apply_shape_rules` written
        # against the frontier row instead of against a freshly downloaded
        # record. The host applies them once, when the leaf is created, using
        # that leaf's own depth and row count; both are invariant for the
        # life of the leaf, so applying them here at pick time reaches the
        # same answer for the same leaf on every step.
        var depth = front[unsafe_offset = fo + FRONT_DEPTH][0]
        var rows = front[unsafe_offset = fo + FRONT_ROW_COUNT][0]
        if max_depth > Int32(0) and depth >= max_depth:
            admissible = False
        if rows < Int32(2) * min_data_in_leaf or rows < Int32(2):
            admissible = False
        if admissible:
            var gain = rec_f[unsafe_offset = rec * SPLIT_FWORDS + FREC_GAIN][
                0
            ]
            if gain > my_gain:
                my_gain = gain
                my_slot = Int32(s)
        s += PICK_THREADS

    var top = block.max[block_size=PICK_THREADS](my_gain)
    # Only a thread whose own best equals the winning gain may name a slot.
    # A thread that found nothing holds `NO_PICK` and contributes it whatever
    # the comparison says, so the minimum below is taken over genuine
    # winners only.
    var mine = NO_PICK
    if my_gain == top:
        mine = my_slot
    var best = block.min[block_size=PICK_THREADS](mine)

    if tid != 0:
        return

    # Phase 3: the commit, on one thread.
    if top <= Float32(0.0):
        ctr[unsafe_offset=CTR_PICK] = Int32(-1)
        ctr[unsafe_offset=CTR_PICK_NODE] = Int32(-1)
        ctr[unsafe_offset=CTR_STATUS] = Int32(TREE_NO_CANDIDATE)
        step[unsafe_offset=STEP_LIVE] = Int32(0)
        return

    var slot = Int(best)
    var fo = slot * FRONT_WORDS
    var parent = Int(front[unsafe_offset = fo + FRONT_NODE][0])
    var begin = Int(front[unsafe_offset = fo + FRONT_ROW_BEGIN][0])
    var depth = Int(front[unsafe_offset = fo + FRONT_DEPTH][0])
    var parent_slot = Int(front[unsafe_offset = fo + FRONT_HIST_SLOT][0])
    var rec = Int(front[unsafe_offset = fo + FRONT_RECORD][0])
    var ri = rec * SPLIT_IWORDS
    var rf = rec * SPLIT_FWORDS

    var flags = rec_i[unsafe_offset = ri + IREC_FLAGS][0]
    var is_cat = (flags & Int32(FLAG_CATEGORICAL)) != Int32(0)
    var feature = rec_i[unsafe_offset = ri + IREC_FEATURE][0]
    # Exact integers off the record, counted from the same histogram count
    # plane the host `_count_left` would sum.
    var n_left = rec_i[unsafe_offset = ri + IREC_LEFT_COUNT][0]
    var n_right = rec_i[unsafe_offset = ri + IREC_RIGHT_COUNT][0]

    var next_node = Int(ctr[unsafe_offset=CTR_NEXT_NODE][0])
    var left_node = next_node
    var right_node = next_node + 1

    # Capacity, checked before the first write so that an undersized table
    # reports rather than corrupts. Unreachable when the tables are sized
    # from `num_leaves`, since the budget test above already bounds both.
    if (
        n_live + 1 > Int(leaf_capacity)
        or right_node >= Int(node_capacity)
    ):
        ctr[unsafe_offset=CTR_PICK] = Int32(-1)
        ctr[unsafe_offset=CTR_PICK_NODE] = Int32(-1)
        ctr[unsafe_offset=CTR_STATUS] = Int32(TREE_OVERFLOW)
        step[unsafe_offset=STEP_LIVE] = Int32(0)
        return

    # The histogram slot pool, device side. `acquire` takes the lowest free
    # slot rather than the most recently freed one, so a sequence of
    # acquires and releases produces the same assignment every run and the
    # buffer's contents stay a function of the tree; that is
    # `HistogramSlotPool.acquire`'s stated reason and it is reproduced here
    # by scanning upward. Taken before any write, so a full pool leaves the
    # tables untouched.
    var built_slot = -1
    for i in range(Int(pool_capacity)):
        if slot_owner[unsafe_offset=i][0] < Int32(0):
            built_slot = i
            break
    if built_slot < 0:
        ctr[unsafe_offset=CTR_PICK] = Int32(-1)
        ctr[unsafe_offset=CTR_PICK_NODE] = Int32(-1)
        ctr[unsafe_offset=CTR_STATUS] = Int32(TREE_POOL_FULL)
        step[unsafe_offset=STEP_LIVE] = Int32(0)
        return

    # `Tree._set_split` on the parent, field for field.
    var po = parent * TN_IWORDS
    node_i[unsafe_offset = po + TN_FEATURE] = feature
    node_i[unsafe_offset = po + TN_LEFT] = Int32(left_node)
    node_i[unsafe_offset = po + TN_RIGHT] = Int32(right_node)
    node_f[unsafe_offset = parent * TN_FWORDS + TN_SPLIT_GAIN] = rec_f[
        unsafe_offset = rf + FREC_GAIN
    ][0]
    if is_cat:
        # A categorical node routes only by its set: no threshold, no
        # default direction, and no missing bin, since bin 0 already
        # collects the missing rows and is never a set member.
        node_i[unsafe_offset = po + TN_THRESHOLD] = Int32(-1)
        node_i[unsafe_offset = po + TN_DEFAULT_LEFT] = Int32(0)
        node_i[unsafe_offset = po + TN_MISSING_BIN] = Int32(-1)
        node_i[unsafe_offset = po + TN_IS_CATEGORICAL] = Int32(1)
        for w in range(CAT_WORDS):
            node_i[unsafe_offset = po + TN_CAT0 + w] = rec_i[
                unsafe_offset = ri + IREC_CAT0 + w
            ][0]
    else:
        node_i[unsafe_offset = po + TN_THRESHOLD] = rec_i[
            unsafe_offset = ri + IREC_BIN
        ][0]
        var dl = Int32(0)
        if (flags & Int32(FLAG_DEFAULT_LEFT)) != Int32(0):
            dl = Int32(1)
        node_i[unsafe_offset = po + TN_DEFAULT_LEFT] = dl
        node_i[unsafe_offset = po + TN_MISSING_BIN] = missing[
            unsafe_offset = Int(feature)
        ][0]
        node_i[unsafe_offset = po + TN_IS_CATEGORICAL] = Int32(0)

    # The two `Tree._add_node` calls, left then right, each a leaf holding
    # its child value and its exact row count. The values are copied out of
    # the record with no arithmetic: under the refusals in
    # `tree_resident_supported` the host's clamp is the identity, so the
    # number the host would write is the number the record already holds.
    var lo = left_node * TN_IWORDS
    var ro = right_node * TN_IWORDS
    for w in range(TN_IWORDS):
        node_i[unsafe_offset = lo + w] = Int32(0)
        node_i[unsafe_offset = ro + w] = Int32(0)
    node_i[unsafe_offset = lo + TN_FEATURE] = Int32(-1)
    node_i[unsafe_offset = lo + TN_THRESHOLD] = Int32(-1)
    node_i[unsafe_offset = lo + TN_LEFT] = Int32(-1)
    node_i[unsafe_offset = lo + TN_RIGHT] = Int32(-1)
    node_i[unsafe_offset = lo + TN_MISSING_BIN] = Int32(-1)
    node_i[unsafe_offset = lo + TN_COUNT] = n_left
    node_i[unsafe_offset = ro + TN_FEATURE] = Int32(-1)
    node_i[unsafe_offset = ro + TN_THRESHOLD] = Int32(-1)
    node_i[unsafe_offset = ro + TN_LEFT] = Int32(-1)
    node_i[unsafe_offset = ro + TN_RIGHT] = Int32(-1)
    node_i[unsafe_offset = ro + TN_MISSING_BIN] = Int32(-1)
    node_i[unsafe_offset = ro + TN_COUNT] = n_right
    node_f[unsafe_offset = left_node * TN_FWORDS + TN_VALUE] = rec_f[
        unsafe_offset = rf + FREC_LEFT_VALUE
    ][0]
    node_f[unsafe_offset = left_node * TN_FWORDS + TN_SPLIT_GAIN] = Float32(
        0.0
    )
    node_f[unsafe_offset = right_node * TN_FWORDS + TN_VALUE] = rec_f[
        unsafe_offset = rf + FREC_RIGHT_VALUE
    ][0]
    node_f[unsafe_offset = right_node * TN_FWORDS + TN_SPLIT_GAIN] = Float32(
        0.0
    )

    # The subtraction choice, `gpu_frontier.subtraction_builds_left`: the
    # smaller child is accumulated from its own rows into the fresh slot and
    # the larger is derived by subtracting it from the parent's slot in
    # place, so the parent's slot outlives its owner by one generation and is
    # reassigned rather than freed. Ties go to the left child, which is the
    # `n_left <= n_right` test `grow_tree`, `grow_tree_gpu`, and
    # `_enqueue_resident_split` all use. An integer comparison, so it is
    # exact here and cannot drift from the host.
    var build_left = n_left <= n_right
    var built_node = right_node
    var derived_node = left_node
    if build_left:
        built_node = left_node
        derived_node = right_node
    slot_owner[unsafe_offset=built_slot] = Int32(built_node)
    slot_owner[unsafe_offset=parent_slot] = Int32(derived_node)
    var left_slot = parent_slot
    var right_slot = built_slot
    if build_left:
        left_slot = built_slot
        right_slot = parent_slot

    # The frontier move. The left child takes the parent's slot and keeps its
    # record; the right child is appended at `n_live` and takes record
    # `n_live`. That is the trainer's convention and therefore the
    # tie-breaking order every later pick depends on.
    front[unsafe_offset = fo + FRONT_NODE] = Int32(left_node)
    front[unsafe_offset = fo + FRONT_ROW_BEGIN] = Int32(begin)
    front[unsafe_offset = fo + FRONT_ROW_COUNT] = n_left
    front[unsafe_offset = fo + FRONT_DEPTH] = Int32(depth + 1)
    front[unsafe_offset = fo + FRONT_HIST_SLOT] = Int32(left_slot)
    front[unsafe_offset = fo + FRONT_RECORD] = Int32(rec)

    var no = n_live * FRONT_WORDS
    front[unsafe_offset = no + FRONT_NODE] = Int32(right_node)
    front[unsafe_offset = no + FRONT_ROW_BEGIN] = Int32(begin) + n_left
    front[unsafe_offset = no + FRONT_ROW_COUNT] = n_right
    front[unsafe_offset = no + FRONT_DEPTH] = Int32(depth + 1)
    front[unsafe_offset = no + FRONT_HIST_SLOT] = Int32(right_slot)
    front[unsafe_offset = no + FRONT_RECORD] = Int32(n_live)

    # The step descriptor. Written last, after every table this commit
    # touches, so that a reader of `STEP_LIVE` on the same queue is reading a
    # descriptor whose tables are already consistent with it. The queue is in
    # order and this is one thread, so that ordering is the program order
    # above and needs no fence.
    #
    # Every word is written on every commit. Nothing is inherited from the
    # step before, which matters because a step that commits nothing leaves
    # the row untouched apart from `STEP_LIVE`, and a reader that skipped the
    # liveness check would then act on a stale split.
    step[unsafe_offset=STEP_LIVE] = Int32(1)
    step[unsafe_offset=STEP_ROW_BEGIN] = Int32(begin)
    step[unsafe_offset=STEP_ROW_COUNT] = n_left + n_right
    step[unsafe_offset=STEP_FEATURE] = feature
    step[unsafe_offset=STEP_IS_CAT] = Int32(1) if is_cat else Int32(0)
    if is_cat:
        # A categorical node routes only by its set, so the three numerical
        # routing words are the neutral values `_row_goes_left` ignores: no
        # threshold can match, no bin equals -1, and no row takes a default
        # direction.
        step[unsafe_offset=STEP_THRESHOLD] = Int32(-1)
        step[unsafe_offset=STEP_MISSING_BIN] = Int32(-1)
        step[unsafe_offset=STEP_DEFAULT_LEFT] = Int32(0)
    else:
        step[unsafe_offset=STEP_THRESHOLD] = rec_i[
            unsafe_offset = ri + IREC_BIN
        ][0]
        step[unsafe_offset=STEP_MISSING_BIN] = missing[
            unsafe_offset = Int(feature)
        ][0]
        var sdl = Int32(0)
        if (flags & Int32(FLAG_DEFAULT_LEFT)) != Int32(0):
            sdl = Int32(1)
        step[unsafe_offset=STEP_DEFAULT_LEFT] = sdl
    for w in range(CAT_WORDS):
        # Copied whether or not the split is categorical, because a stale
        # category set behind a `STEP_IS_CAT` of zero is exactly the kind of
        # thing that becomes a fault the day a reader stops checking the flag.
        step[unsafe_offset = STEP_CAT0 + w] = rec_i[
            unsafe_offset = ri + IREC_CAT0 + w
        ][0]
    # The built child is the one accumulated from its own rows; the derived
    # one is subtracted out of the parent's slot. `build_left` was decided
    # above by the same `n_left <= n_right` integer comparison every other
    # grower in this package uses.
    if build_left:
        step[unsafe_offset=STEP_BUILT_BEGIN] = Int32(begin)
        step[unsafe_offset=STEP_BUILT_COUNT] = n_left
    else:
        step[unsafe_offset=STEP_BUILT_BEGIN] = Int32(begin) + n_left
        step[unsafe_offset=STEP_BUILT_COUNT] = n_right
    step[unsafe_offset=STEP_BUILT_SLOT] = Int32(built_slot)
    step[unsafe_offset=STEP_SUB_SLOT] = Int32(parent_slot)
    step[unsafe_offset=STEP_LEFT_SLOT] = Int32(left_slot)
    step[unsafe_offset=STEP_RIGHT_SLOT] = Int32(right_slot)
    # The record slots the children's searches must end up in, which are the
    # frontier's own record indices: the left child kept the parent's and the
    # right child took `n_live`.
    step[unsafe_offset=STEP_LEFT_REC] = Int32(rec)
    step[unsafe_offset=STEP_RIGHT_REC] = Int32(n_live)

    var commits = Int(ctr[unsafe_offset=CTR_COMMITS][0])
    if commits < Int(leaf_capacity):
        order[unsafe_offset=commits] = Int32(parent)
    ctr[unsafe_offset=CTR_NEXT_NODE] = Int32(next_node + 2)
    ctr[unsafe_offset=CTR_N_LIVE] = Int32(n_live + 1)
    ctr[unsafe_offset=CTR_PICK] = Int32(slot)
    ctr[unsafe_offset=CTR_PICK_NODE] = Int32(parent)
    ctr[unsafe_offset=CTR_STATUS] = Int32(TREE_RUNNING)
    ctr[unsafe_offset=CTR_COMMITS] = Int32(commits + 1)


def _seed_root_value_kernel(
    node_f: MutPointer[Float32, MutAnyOrigin],
    rec_f: MutPointer[Float32, MutAnyOrigin],
    record: Int32,
):
    """Copy the root's Newton value out of its search record into node 0.

    One thread, one Float32, and it exists only because of *when* the root's
    value is known. `Tree._add_node` creates the root with a value of 0.0 and
    the grower overwrites it with `root_rec.parent_value` after the root's
    search comes home; on the host path that overwrite is free, because the
    record is already on the host. On a device-owned tree the record never
    comes to the host, so the copy has to happen where the record is.

    `begin_tree` still takes a `root_value` argument for callers that have
    one (the test file builds trees with no search behind them at all). This
    kernel is what a caller uses when it does not, and it is a copy with no
    arithmetic, so the value node 0 ends up holding is bit for bit the Float32
    the search wrote.
    """
    node_f[unsafe_offset = TN_VALUE] = rec_f[
        unsafe_offset = Int(record) * SPLIT_FWORDS + FREC_PARENT_VALUE
    ][0]


def _stage_child_search_kernel(
    node_tbl: MutPointer[Int32, MutAnyOrigin],
    step: MutPointer[Int32, MutAnyOrigin],
    slot_cells: Int32,
    left_record: Int32,
    right_record: Int32,
):
    """Point the two scratch search records at the children's histogram slots.

    The split searcher reads a per-record table whose only per-node word that
    varies within one tree, once monotone intervals and per-node feature
    draws are refused, is `NODE_HIST_BASE`: where in the resident pool that
    record's histogram starts. Everything else in that table
    (`NODE_SLOTS`, the feature list, the allow mask, the float parameter
    block) is a property of the tree and is staged once by the host before
    the first split.

    So this kernel is the whole of holdout three's device half: two Int32
    writes that say "search these two slots", derived from the commit that
    just happened. Without it the host would have to know the slots, and the
    slots are chosen by the commit kernel from a device-resident pool.

    `left_record` and `right_record` are the two *scratch* record slots the
    search launch writes, which are host constants because a launch's record
    range is a host-side argument. They are not the frontier's record slots;
    `_copy_records_kernel` moves the answers into those afterwards.

    A dead step writes nothing, so the searcher's table keeps whatever it
    last held and the search that follows re-searches the previous step's
    slots into scratch records nothing will read. That is wasted work on a
    tree that stopped early and it is deliberately preferred to a branchier
    alternative: the copy that would have propagated those records is guarded
    by the same liveness word, so nothing stale can reach the frontier.
    """
    if step[unsafe_offset=STEP_LIVE][0] == Int32(0):
        return
    node_tbl[
        unsafe_offset = Int(left_record) * NODE_WORDS + NODE_HIST_BASE
    ] = step[unsafe_offset=STEP_LEFT_SLOT][0] * slot_cells
    node_tbl[
        unsafe_offset = Int(right_record) * NODE_WORDS + NODE_HIST_BASE
    ] = step[unsafe_offset=STEP_RIGHT_SLOT][0] * slot_cells


def _copy_records_kernel(
    rec_i: MutPointer[Int32, MutAnyOrigin],
    rec_f: MutPointer[Float32, MutAnyOrigin],
    step: MutPointer[Int32, MutAnyOrigin],
    left_record: Int32,
    right_record: Int32,
):
    """Move the two scratch search records into the frontier slots that own
    them.

    A search launch writes a *consecutive* range of record slots, because
    `record_base` and `n_records` are host arguments. The two children of a
    committed split do not occupy consecutive frontier slots: the left child
    keeps the parent's slot, which can be anywhere, and the right child is
    appended at the end. Reconciling those two facts without a host round
    trip is what this kernel is for. The search always writes one fixed pair
    of scratch records, and this copies each into the record slot its
    frontier leaf reads, which the commit recorded as `STEP_LEFT_REC` and
    `STEP_RIGHT_REC`.

    A record is `SPLIT_IWORDS` Int32 and `SPLIT_FWORDS` Float32, thirty-four
    words in all, so the copy is small enough for one narrow threadgroup and
    is written as a strided walk rather than as two serial loops on thread
    zero.

    Every word is copied, and none is interpreted. In particular no float is
    read as a float: they are moved by assignment, so nothing here can round,
    contract, or reorder anything. That is the same discipline the commit
    kernel keeps and for the same reason (`docs/NUMERICS.md`).

    A dead step copies nothing, which is what leaves the frontier holding the
    records it already had when growth has stopped.
    """
    if step[unsafe_offset=STEP_LIVE][0] == Int32(0):
        return
    var dst_l = Int(step[unsafe_offset=STEP_LEFT_REC][0])
    var dst_r = Int(step[unsafe_offset=STEP_RIGHT_REC][0])
    var src_l = Int(left_record)
    var src_r = Int(right_record)
    var tid = Int(thread_idx.x)
    var w = tid
    while w < SPLIT_IWORDS:
        rec_i[unsafe_offset = dst_l * SPLIT_IWORDS + w] = rec_i[
            unsafe_offset = src_l * SPLIT_IWORDS + w
        ][0]
        rec_i[unsafe_offset = dst_r * SPLIT_IWORDS + w] = rec_i[
            unsafe_offset = src_r * SPLIT_IWORDS + w
        ][0]
        w += PICK_THREADS
    w = tid
    while w < SPLIT_FWORDS:
        rec_f[unsafe_offset = dst_l * SPLIT_FWORDS + w] = rec_f[
            unsafe_offset = src_l * SPLIT_FWORDS + w
        ][0]
        rec_f[unsafe_offset = dst_r * SPLIT_FWORDS + w] = rec_f[
            unsafe_offset = src_r * SPLIT_FWORDS + w
        ][0]
        w += PICK_THREADS


# --- Host-side owner ------------------------------------------------------


struct DeviceTreeTables(Movable):
    """The device buffers of one tree under construction, and the launch.

    Sized once from the leaf budget and reused across trees, because none of
    the shapes move between the trees of one fit: a tree holds at most
    `num_leaves` live leaves, `2 * num_leaves - 1` nodes, and `num_leaves`
    histogram slots, whatever the data. `begin_tree` resets the contents; it
    allocates nothing.

    The context is supplied by the caller in the ordinary case, because these
    kernels have to queue behind the histogram build and the split search
    that produced the records they read, and one in-order queue is what makes
    that safe without a fence. The private-context constructor exists so the
    tables are exercisable on their own, which is what the test file uses.

    Nothing here reads an environment variable. `tree_resident_requested` is
    the gate and it is the caller's to consult; a struct that silently did
    nothing when a variable was unset would be much harder to test than one
    that always works and is simply never constructed.
    """

    var ctx: DeviceContext
    var leaf_capacity: Int
    var node_capacity: Int
    var pool_capacity: Int
    var n_features: Int

    var front_dev: DeviceBuffer[DType.int32]
    var node_i_dev: DeviceBuffer[DType.int32]
    var node_f_dev: DeviceBuffer[DType.float32]
    var ctr_dev: DeviceBuffer[DType.int32]
    var slot_dev: DeviceBuffer[DType.int32]
    var missing_dev: DeviceBuffer[DType.int32]
    # `order[k]` is the node id split by the k-th commit. Read once per tree
    # by `download`, and used by `tree_from_snapshot` to append categorical
    # sets in the order the host tree would have appended them. See
    # `_pick_and_commit_kernel` for why that order is not the node id order.
    var order_dev: DeviceBuffer[DType.int32]
    # A step descriptor of this struct's own, used only by the `enqueue_step`
    # overload that is handed none. The wiring path passes the descriptor
    # buffer that `GpuActiveRows` owns, because that is the buffer the
    # partition and histogram kernels are launched against; this one exists so
    # that a caller exercising the commit kernel on its own -- which is what
    # the test file does -- does not have to construct a `GpuActiveRows` to
    # get a pointer. Eight words. It is never read by any other kernel.
    var step_scratch: DeviceBuffer[DType.int32]

    # Pinned staging, so a reset and a download are ordinary asynchronous
    # copies rather than `map_to_host` mappings, each of which blocks until
    # the device is idle. The same choice `GpuSplitSearcher` makes for its
    # per-node tables and for the same measured reason.
    #
    # One staging buffer per destination rather than one shared buffer, which
    # matters more than it looks. An enqueued copy reads its source
    # asynchronously, so a shared buffer forces a `synchronize` between every
    # pair of copies before the host may refill it, and a `begin_tree` that
    # writes five tables then costs five drains instead of one. That is the
    # opposite of what this whole module is for, and it is not a theoretical
    # cost: on an Apple M4 the five-drain form of `begin_tree` and a
    # five-drain `stage_frontier` were together the dominant cost of the
    # test file, well above the kernel they existed to set up.
    var stage_front: HostBuffer[DType.int32]
    var stage_node_i: HostBuffer[DType.int32]
    var stage_node_f: HostBuffer[DType.float32]
    var stage_ctr: HostBuffer[DType.int32]
    var stage_slot: HostBuffer[DType.int32]
    var host_front: HostBuffer[DType.int32]
    var host_node_i: HostBuffer[DType.int32]
    var host_node_f: HostBuffer[DType.float32]
    var host_ctr: HostBuffer[DType.int32]
    var host_slot: HostBuffer[DType.int32]
    var host_order: HostBuffer[DType.int32]

    def __init__(
        out self,
        num_leaves: Int,
        n_features: Int,
        missing_bins: List[Int] = [],
    ) raises:
        """Open a private context and size everything from the leaf budget."""
        var ctx = DeviceContext()
        self = Self(ctx, num_leaves, n_features, missing_bins)

    def __init__(
        out self,
        ctx: DeviceContext,
        num_leaves: Int,
        n_features: Int,
        missing_bins: List[Int] = [],
    ) raises:
        """Build on a caller-supplied context. Sharing the histogram
        builder's and the searcher's context is what lets the pick kernel be
        enqueued behind the search with no fence, reading the record buffer
        the search just wrote rather than a copy of it."""
        if num_leaves < 2:
            raise Error("a tree table needs a budget of at least two leaves")
        if n_features < 1:
            raise Error("a tree table needs at least one feature")
        if len(missing_bins) > 0 and len(missing_bins) != n_features:
            raise Error("missing_bins length must equal n_features")

        self.ctx = ctx
        self.leaf_capacity = num_leaves
        # A tree of L leaves holds exactly 2L-1 nodes, because the root is
        # one node and every split adds two.
        self.node_capacity = 2 * num_leaves - 1
        # One resident slot per live leaf, which is what
        # `GpuHistogramBuilder.open_resident` already sizes its pool for.
        self.pool_capacity = num_leaves
        self.n_features = n_features

        self.front_dev = self.ctx.enqueue_create_buffer[DType.int32](
            self.leaf_capacity * FRONT_WORDS
        )
        self.node_i_dev = self.ctx.enqueue_create_buffer[DType.int32](
            self.node_capacity * TN_IWORDS
        )
        self.node_f_dev = self.ctx.enqueue_create_buffer[DType.float32](
            self.node_capacity * TN_FWORDS
        )
        self.ctr_dev = self.ctx.enqueue_create_buffer[DType.int32](CTR_WORDS)
        self.slot_dev = self.ctx.enqueue_create_buffer[DType.int32](
            self.pool_capacity
        )
        self.missing_dev = self.ctx.enqueue_create_buffer[DType.int32](
            n_features
        )
        self.order_dev = self.ctx.enqueue_create_buffer[DType.int32](
            self.leaf_capacity
        )
        self.step_scratch = self.ctx.enqueue_create_buffer[DType.int32](
            STEP_WORDS
        )

        self.stage_front = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.leaf_capacity * FRONT_WORDS
        )
        self.stage_node_i = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.node_capacity * TN_IWORDS
        )
        self.stage_node_f = self.ctx.enqueue_create_host_buffer[
            DType.float32
        ](self.node_capacity * TN_FWORDS)
        self.stage_ctr = self.ctx.enqueue_create_host_buffer[DType.int32](
            CTR_WORDS
        )
        self.stage_slot = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.pool_capacity
        )
        self.host_front = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.leaf_capacity * FRONT_WORDS
        )
        self.host_node_i = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.node_capacity * TN_IWORDS
        )
        self.host_node_f = self.ctx.enqueue_create_host_buffer[DType.float32](
            self.node_capacity * TN_FWORDS
        )
        self.host_ctr = self.ctx.enqueue_create_host_buffer[DType.int32](
            CTR_WORDS
        )
        self.host_slot = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.pool_capacity
        )
        self.host_order = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.leaf_capacity
        )

        # The per-feature missing-bin table, uploaded once. -1 is the "this
        # feature has no missing bin" value every other routing site in the
        # package already uses.
        var stage_missing = self.ctx.enqueue_create_host_buffer[DType.int32](
            n_features
        )
        var dst = stage_missing.unsafe_ptr()
        for f in range(n_features):
            var m = missing_bins[f] if len(missing_bins) > 0 else -1
            dst.unsafe_store(f, Int32(m))
        self.ctx.enqueue_copy(
            dst_buf=self.missing_dev, src_ptr=stage_missing.unsafe_ptr()
        )
        self.ctx.synchronize()

    def begin_tree(
        mut self,
        n_active: Int,
        root_slot: Int = 0,
        root_value: Float32 = 0.0,
        wait: Bool = True,
    ) raises:
        """Reset to a one-leaf frontier whose root owns `[0, n_active)`.

        `n_active` is the bag when bagging is on and every row when it is
        not, which is the same thing `LeafFrontier.begin_tree` takes.
        `root_slot` is the pool slot the root's histogram was accumulated
        into, which a real caller gets from `acquire_resident` and which is 0
        for the first tree of a fit because `acquire` takes the lowest free
        slot.

        `root_value` is the root's own Newton value, which on the host comes
        from the root record's `parent_value` and is written by
        `_device_search_resident` as `tree.value[root] = root_rec.
        parent_value`. It is a parameter rather than something read out of a
        record here, because the root's record is searched before the first
        pick and the host already holds the number; taking it as an argument
        keeps this module from having to know which record slot the root
        used.

        Everything is staged into pinned memory and copied rather than
        written through a `map_to_host` mapping, because a mapping blocks
        until the device is idle and a copy does not. Each of the five tables
        has its own staging buffer, so all five copies are enqueued before
        anything waits and the whole reset costs one synchronization. The
        earlier form shared one staging buffer and had to drain between
        copies, which is both five times the waits and, when a drain was
        omitted, a frontier row silently overwritten by node-table words.

        `wait` is the one synchronization this call makes, and a device-owned
        growth loop passes False to remove it. That is safe under exactly one
        condition, which is worth stating rather than assuming: the staging
        buffers must not be refilled before the copies reading them have
        finished, and the only thing that refills them is the *next*
        `begin_tree`. A caller that downloads the tree at the end of every
        tree therefore already has a synchronization between any two
        `begin_tree` calls, and the wait here is redundant for it. A caller
        that does not must leave `wait` alone. It defaults to True so every
        existing caller, the test file included, keeps the behavior it was
        written against.
        """
        if n_active < 0:
            raise Error("active row count must be nonnegative")
        if root_slot < 0 or root_slot >= self.pool_capacity:
            raise Error("root histogram slot out of range")

        var fi = self.stage_front.unsafe_ptr()
        for i in range(self.leaf_capacity * FRONT_WORDS):
            fi.unsafe_store(i, Int32(0))
        fi.unsafe_store(FRONT_NODE, Int32(0))
        fi.unsafe_store(FRONT_ROW_BEGIN, Int32(0))
        fi.unsafe_store(FRONT_ROW_COUNT, Int32(n_active))
        fi.unsafe_store(FRONT_DEPTH, Int32(0))
        fi.unsafe_store(FRONT_HIST_SLOT, Int32(root_slot))
        # The record-slot-equals-frontier-slot identity starts here; the
        # commit preserves it.
        fi.unsafe_store(FRONT_RECORD, Int32(0))

        # Every node starts as `Tree._add_node` leaves one, so a node the
        # tree never reaches reads as a zero-value leaf rather than as
        # whatever the previous tree wrote there.
        var ni = self.stage_node_i.unsafe_ptr()
        for n in range(self.node_capacity):
            var o = n * TN_IWORDS
            for w in range(TN_IWORDS):
                ni.unsafe_store(o + w, Int32(0))
            ni.unsafe_store(o + TN_FEATURE, Int32(-1))
            ni.unsafe_store(o + TN_THRESHOLD, Int32(-1))
            ni.unsafe_store(o + TN_LEFT, Int32(-1))
            ni.unsafe_store(o + TN_RIGHT, Int32(-1))
            ni.unsafe_store(o + TN_MISSING_BIN, Int32(-1))
        ni.unsafe_store(TN_COUNT, Int32(n_active))

        var nf = self.stage_node_f.unsafe_ptr()
        for i in range(self.node_capacity * TN_FWORDS):
            nf.unsafe_store(i, Float32(0.0))
        nf.unsafe_store(TN_VALUE, root_value)

        var ci = self.stage_ctr.unsafe_ptr()
        for w in range(CTR_WORDS):
            ci.unsafe_store(w, Int32(0))
        ci.unsafe_store(CTR_N_LIVE, Int32(1))
        ci.unsafe_store(CTR_NEXT_NODE, Int32(1))
        ci.unsafe_store(CTR_PICK, Int32(-1))
        ci.unsafe_store(CTR_PICK_NODE, Int32(-1))
        ci.unsafe_store(CTR_STATUS, Int32(TREE_RUNNING))

        var si = self.stage_slot.unsafe_ptr()
        for i in range(self.pool_capacity):
            si.unsafe_store(i, Int32(-1))
        si.unsafe_store(root_slot, Int32(0))

        self.ctx.enqueue_copy(
            dst_buf=self.front_dev, src_ptr=self.stage_front.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.node_i_dev, src_ptr=self.stage_node_i.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.node_f_dev, src_ptr=self.stage_node_f.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.ctr_dev, src_ptr=self.stage_ctr.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.slot_dev, src_ptr=self.stage_slot.unsafe_ptr()
        )
        if wait:
            self.ctx.synchronize()

    def stage_frontier(
        mut self, leaves: List[DeviceLeafRow], next_node: Int
    ) raises:
        """Replace the live frontier and the slot pool with `leaves`.

        `begin_tree` makes the one frontier a real tree starts from. This
        makes an arbitrary one, which is what a test needs in order to place
        a tie at a chosen pair of slots, and what a caller resuming a tree
        from a host-side frontier would need. It writes the frontier rows,
        rebuilds the slot pool as the inverse of their `hist_slot` fields,
        and resets the step counters; it does not touch the node table, so a
        caller that wants a clean tree calls `begin_tree` first.

        The preconditions are the frontier's own invariants, checked here
        rather than trusted, because a frontier with two leaves in one
        histogram slot or two leaves reading one record is precisely the
        shape that would make a device result look plausible and be wrong.
        Row-window disjointness is not checked here and is left to
        `TreeTablesSnapshot.check_invariants`, since a test that stages a
        deliberately degenerate frontier to probe the pick has no use for
        windows at all.
        """
        if len(leaves) < 1:
            raise Error("a frontier holds at least one leaf")
        if len(leaves) > self.leaf_capacity:
            raise Error("frontier is wider than these tables were sized for")
        for i in range(len(leaves)):
            if leaves[i].node < 0 or leaves[i].node >= next_node:
                raise Error("a staged leaf's node id is outside the tree")
            if leaves[i].node >= self.node_capacity:
                raise Error("a staged leaf's node id exceeds the node table")
            if (
                leaves[i].hist_slot < 0
                or leaves[i].hist_slot >= self.pool_capacity
            ):
                raise Error("a staged leaf's histogram slot is out of range")
            if leaves[i].record < 0 or leaves[i].record >= self.leaf_capacity:
                raise Error("a staged leaf's record slot is out of range")
            for k in range(i):
                if leaves[k].node == leaves[i].node:
                    raise Error("two staged leaves share a node id")
                if leaves[k].hist_slot == leaves[i].hist_slot:
                    raise Error("two staged leaves share a histogram slot")
                if leaves[k].record == leaves[i].record:
                    raise Error("two staged leaves share a record slot")

        var fi = self.stage_front.unsafe_ptr()
        for i in range(self.leaf_capacity * FRONT_WORDS):
            fi.unsafe_store(i, Int32(0))
        for i in range(len(leaves)):
            var o = i * FRONT_WORDS
            fi.unsafe_store(o + FRONT_NODE, Int32(leaves[i].node))
            fi.unsafe_store(o + FRONT_ROW_BEGIN, Int32(leaves[i].row_begin))
            fi.unsafe_store(o + FRONT_ROW_COUNT, Int32(leaves[i].row_count))
            fi.unsafe_store(o + FRONT_DEPTH, Int32(leaves[i].depth))
            fi.unsafe_store(o + FRONT_HIST_SLOT, Int32(leaves[i].hist_slot))
            fi.unsafe_store(o + FRONT_RECORD, Int32(leaves[i].record))

        var si = self.stage_slot.unsafe_ptr()
        for i in range(self.pool_capacity):
            si.unsafe_store(i, Int32(-1))
        for i in range(len(leaves)):
            si.unsafe_store(leaves[i].hist_slot, Int32(leaves[i].node))

        var ci = self.stage_ctr.unsafe_ptr()
        for w in range(CTR_WORDS):
            ci.unsafe_store(w, Int32(0))
        ci.unsafe_store(CTR_N_LIVE, Int32(len(leaves)))
        ci.unsafe_store(CTR_NEXT_NODE, Int32(next_node))
        ci.unsafe_store(CTR_PICK, Int32(-1))
        ci.unsafe_store(CTR_PICK_NODE, Int32(-1))
        ci.unsafe_store(CTR_STATUS, Int32(TREE_RUNNING))

        self.ctx.enqueue_copy(
            dst_buf=self.front_dev, src_ptr=self.stage_front.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.slot_dev, src_ptr=self.stage_slot.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.ctr_dev, src_ptr=self.stage_ctr.unsafe_ptr()
        )
        self.ctx.synchronize()

    def enqueue_step(
        mut self,
        mut rec_i: DeviceBuffer[DType.int32],
        mut rec_f: DeviceBuffer[DType.float32],
        num_leaves: Int,
        max_depth: Int,
        min_data_in_leaf: Int,
    ) raises:
        """One pick-and-commit step, writing the descriptor into this
        struct's own scratch buffer.

        The overload for a caller that only wants the commit: it exercises
        the same kernel and leaves the same tables, and the descriptor it
        writes goes somewhere nothing else reads. A device-owned growth loop
        uses the other overload and hands in the descriptor buffer the row
        partition and the child histogram are launched against, since a
        descriptor no kernel reads would leave those two with nothing to
        route by.

        The launch is written out rather than delegated to the other
        overload, and that is a language constraint rather than a taste:
        passing `self.step_scratch.unsafe_ptr()` to a method that also takes
        `self` mutably is an aliasing the compiler correctly refuses. Inside
        one method body the two are disjoint borrows and the launch is fine,
        which is why every other launch in this file reads the same way.
        """
        self._check_step_args(num_leaves, min_data_in_leaf)
        comptime if not has_accelerator():
            raise Error(
                "the device tree tables need an accelerator; this binary was"
                " built without one"
            )
        else:
            self.ctx.enqueue_function[_pick_and_commit_kernel](
                self.front_dev.unsafe_ptr(),
                self.node_i_dev.unsafe_ptr(),
                self.node_f_dev.unsafe_ptr(),
                self.ctr_dev.unsafe_ptr(),
                self.slot_dev.unsafe_ptr(),
                self.missing_dev.unsafe_ptr(),
                rec_i.unsafe_ptr(),
                rec_f.unsafe_ptr(),
                self.step_scratch.unsafe_ptr(),
                self.order_dev.unsafe_ptr(),
                Int32(num_leaves),
                Int32(max_depth),
                Int32(min_data_in_leaf),
                Int32(self.pool_capacity),
                Int32(self.leaf_capacity),
                Int32(self.node_capacity),
                grid_dim=1,
                block_dim=PICK_THREADS,
            )

    def _check_step_args(
        self, num_leaves: Int, min_data_in_leaf: Int
    ) raises:
        """The preconditions both `enqueue_step` overloads share."""
        if num_leaves < 2:
            raise Error("a leaf budget of at least two is required")
        if num_leaves > self.leaf_capacity:
            raise Error(
                "the leaf budget exceeds the capacity these tables were"
                " sized for"
            )
        if min_data_in_leaf < 0:
            raise Error("min_data_in_leaf must be nonnegative")

    def enqueue_step[
        step_origin: MutOrigin, //
    ](
        mut self,
        mut rec_i: DeviceBuffer[DType.int32],
        mut rec_f: DeviceBuffer[DType.float32],
        step: MutPointer[Int32, step_origin],
        num_leaves: Int,
        max_depth: Int,
        min_data_in_leaf: Int,
    ) raises:
        """Enqueue one pick-and-commit step. Does not transfer and does not
        synchronize.

        `rec_i` and `rec_f` are the split-record buffers a search wrote,
        which in a wired caller are `GpuSplitSearcher.rec_i_dev` and
        `rec_f_dev` and in a test are buffers the test filled itself. They
        are ordinary arguments rather than fields for the same reason
        `gpu_split_search._launch_search` takes its histogram as an argument:
        the records belong to whoever produced them, and copying them into a
        buffer this struct owned would be a transfer with no purpose.

        `num_leaves`, `max_depth`, and `min_data_in_leaf` come straight from
        `TreeParams` and are launch arguments rather than table entries
        because they do not vary across the frontier.

        One launch, one block, `PICK_THREADS` threads. That count is the
        whole point of the module: a tree of `num_leaves` leaves costs
        `num_leaves - 1` of these, all of which can sit in the queue behind
        the histogram and search work they depend on, where today each of
        them is a host round trip.

        `step` is the descriptor buffer the row partition and the child
        histogram will be launched against, which on the wired path is the one
        `GpuActiveRows` owns. Handing it in rather than owning it here is what
        keeps the coupling between the two modules to a single flat row of
        Int32 with a layout stated in one place.
        """
        self._check_step_args(num_leaves, min_data_in_leaf)
        comptime if not has_accelerator():
            raise Error(
                "the device tree tables need an accelerator; this binary was"
                " built without one"
            )
        else:
            self.ctx.enqueue_function[_pick_and_commit_kernel](
                self.front_dev.unsafe_ptr(),
                self.node_i_dev.unsafe_ptr(),
                self.node_f_dev.unsafe_ptr(),
                self.ctr_dev.unsafe_ptr(),
                self.slot_dev.unsafe_ptr(),
                self.missing_dev.unsafe_ptr(),
                rec_i.unsafe_ptr(),
                rec_f.unsafe_ptr(),
                step,
                self.order_dev.unsafe_ptr(),
                Int32(num_leaves),
                Int32(max_depth),
                Int32(min_data_in_leaf),
                Int32(self.pool_capacity),
                Int32(self.leaf_capacity),
                Int32(self.node_capacity),
                grid_dim=1,
                block_dim=PICK_THREADS,
            )

    def enqueue_seed_root_value(
        mut self, mut rec_f: DeviceBuffer[DType.float32], record: Int = 0
    ) raises:
        """Copy the root's own Newton value out of record `record`. One
        launch, one thread, no transfer and no synchronization.

        The device-owned equivalent of `_device_search_resident`'s
        `tree.value[root] = root_rec.parent_value`, which is the one thing the
        host does with the root's record besides deciding whether the root can
        split at all. See `_seed_root_value_kernel`.
        """
        if record < 0:
            raise Error("record index must be nonnegative")
        comptime if not has_accelerator():
            raise Error(
                "the device tree tables need an accelerator; this binary was"
                " built without one"
            )
        else:
            self.ctx.enqueue_function[_seed_root_value_kernel](
                self.node_f_dev.unsafe_ptr(),
                rec_f.unsafe_ptr(),
                Int32(record),
                grid_dim=1,
                block_dim=1,
            )

    def enqueue_stage_child_search[
        step_origin: MutOrigin, //
    ](
        mut self,
        mut node_tbl: DeviceBuffer[DType.int32],
        step: MutPointer[Int32, step_origin],
        slot_cells: Int,
        left_record: Int,
        right_record: Int,
    ) raises:
        """Point the searcher's two scratch records at the children's slots.

        `node_tbl` is `GpuSplitSearcher.node_dev` and `slot_cells` is
        `3 * n_features * n_bins`, the resident pool's slot stride, which is
        the same unit `enqueue_frontier` multiplies a `hist_slot` by. This is
        the only write to that table inside a tree on the device-owned path:
        everything else in it is staged once by the host before the first
        split, which is what the module docstring's third holdout is about.
        """
        if slot_cells < 1:
            raise Error("the pool slot stride must be positive")
        if left_record < 0 or right_record < 0:
            raise Error("record indices must be nonnegative")
        if left_record == right_record:
            raise Error("the two scratch records must differ")
        comptime if not has_accelerator():
            raise Error(
                "the device tree tables need an accelerator; this binary was"
                " built without one"
            )
        else:
            self.ctx.enqueue_function[_stage_child_search_kernel](
                node_tbl.unsafe_ptr(),
                step,
                Int32(slot_cells),
                Int32(left_record),
                Int32(right_record),
                grid_dim=1,
                block_dim=1,
            )

    def enqueue_copy_records[
        step_origin: MutOrigin, //
    ](
        mut self,
        mut rec_i: DeviceBuffer[DType.int32],
        mut rec_f: DeviceBuffer[DType.float32],
        step: MutPointer[Int32, step_origin],
        left_record: Int,
        right_record: Int,
    ) raises:
        """Move the two scratch records into the frontier slots that own them.
        See `_copy_records_kernel` for why the indirection exists at all."""
        if left_record < 0 or right_record < 0:
            raise Error("record indices must be nonnegative")
        if left_record == right_record:
            raise Error("the two scratch records must differ")
        comptime if not has_accelerator():
            raise Error(
                "the device tree tables need an accelerator; this binary was"
                " built without one"
            )
        else:
            self.ctx.enqueue_function[_copy_records_kernel](
                rec_i.unsafe_ptr(),
                rec_f.unsafe_ptr(),
                step,
                Int32(left_record),
                Int32(right_record),
                grid_dim=1,
                block_dim=PICK_THREADS,
            )

    def download(mut self) raises -> TreeTablesSnapshot:
        """Copy every table home and decode it. One host synchronization.

        A wired caller would do this once per tree. A test does it once per
        step, which is the opposite of the point and is exactly why the
        measurement in the module docstring cannot be taken from a test run.
        """
        self.ctx.enqueue_copy(
            dst_ptr=self.host_front.unsafe_ptr(), src_buf=self.front_dev
        )
        self.ctx.enqueue_copy(
            dst_ptr=self.host_node_i.unsafe_ptr(), src_buf=self.node_i_dev
        )
        self.ctx.enqueue_copy(
            dst_ptr=self.host_node_f.unsafe_ptr(), src_buf=self.node_f_dev
        )
        self.ctx.enqueue_copy(
            dst_ptr=self.host_ctr.unsafe_ptr(), src_buf=self.ctr_dev
        )
        self.ctx.enqueue_copy(
            dst_ptr=self.host_slot.unsafe_ptr(), src_buf=self.slot_dev
        )
        self.ctx.enqueue_copy(
            dst_ptr=self.host_order.unsafe_ptr(), src_buf=self.order_dev
        )
        self.ctx.synchronize()

        var out = TreeTablesSnapshot()
        var c = self.host_ctr.unsafe_ptr()
        out.n_live = Int(c.unsafe_load(CTR_N_LIVE))
        out.next_node = Int(c.unsafe_load(CTR_NEXT_NODE))
        out.pick = Int(c.unsafe_load(CTR_PICK))
        out.pick_node = Int(c.unsafe_load(CTR_PICK_NODE))
        out.status = Int(c.unsafe_load(CTR_STATUS))
        out.commits = Int(c.unsafe_load(CTR_COMMITS))

        var f = self.host_front.unsafe_ptr()
        for i in range(out.n_live):
            var o = i * FRONT_WORDS
            out.leaves.append(
                DeviceLeafRow(
                    Int(f.unsafe_load(o + FRONT_NODE)),
                    Int(f.unsafe_load(o + FRONT_ROW_BEGIN)),
                    Int(f.unsafe_load(o + FRONT_ROW_COUNT)),
                    Int(f.unsafe_load(o + FRONT_DEPTH)),
                    Int(f.unsafe_load(o + FRONT_HIST_SLOT)),
                    Int(f.unsafe_load(o + FRONT_RECORD)),
                )
            )

        var ni = self.host_node_i.unsafe_ptr()
        var nf = self.host_node_f.unsafe_ptr()
        for n in range(out.next_node):
            var o = n * TN_IWORDS
            var row = DeviceNodeRow()
            row.feature = Int(ni.unsafe_load(o + TN_FEATURE))
            row.threshold_bin = Int(ni.unsafe_load(o + TN_THRESHOLD))
            row.left = Int(ni.unsafe_load(o + TN_LEFT))
            row.right = Int(ni.unsafe_load(o + TN_RIGHT))
            row.default_left = ni.unsafe_load(o + TN_DEFAULT_LEFT) != Int32(0)
            row.missing_bin = Int(ni.unsafe_load(o + TN_MISSING_BIN))
            row.is_categorical = ni.unsafe_load(
                o + TN_IS_CATEGORICAL
            ) != Int32(0)
            row.count = Int(ni.unsafe_load(o + TN_COUNT))
            for w in range(CAT_WORDS):
                row.cat_words[w] = ni.unsafe_load(o + TN_CAT0 + w)
            row.value = nf.unsafe_load(n * TN_FWORDS + TN_VALUE)
            row.split_gain = nf.unsafe_load(n * TN_FWORDS + TN_SPLIT_GAIN)
            out.nodes.append(row^)

        var s = self.host_slot.unsafe_ptr()
        for i in range(self.pool_capacity):
            out.slot_owner.append(Int(s.unsafe_load(i)))

        var o = self.host_order.unsafe_ptr()
        var n_commits = out.commits
        if n_commits > self.leaf_capacity:
            n_commits = self.leaf_capacity
        for k in range(n_commits):
            out.commit_order.append(Int(o.unsafe_load(k)))
        return out^


# --- Decoding a snapshot into the tree the host would have grown -----------


def tree_from_snapshot(snap: TreeTablesSnapshot) raises -> Tree:
    """The `Tree` the host grower would have built from the same commits.

    This is the one place a device-owned tree becomes an ordinary model, and
    it is a transcription rather than a construction: every field below is
    read out of a node row that a device kernel wrote, and nothing is
    recomputed. Written as an argument, field by field, because it is the
    only part of the device-owned path that a reader can check against
    `tree.Tree` directly.

        Tree field        device source                    why it is equal
        ----------        -------------                    ---------------
        feature           TN_FEATURE                       `_set_split` writes
                                                           `split.feature`;
                                                           the commit kernel
                                                           writes the record's
                                                           feature, which is
                                                           what `to_split_info`
                                                           puts in `split`.
        threshold_bin     TN_THRESHOLD                     both -1 on a leaf
                                                           and on a categorical
                                                           node, both the
                                                           record's bin
                                                           otherwise.
        left / right      TN_LEFT / TN_RIGHT               both assign the two
                                                           ids consecutively,
                                                           left first.
        value             TN_VALUE widened                 the host stores the
                                                           record's Float32
                                                           child value in a
                                                           Float64 field, which
                                                           is a widening and so
                                                           is exact; the same
                                                           Float32 widened here
                                                           is the same Float64.
        split_gain        TN_SPLIT_GAIN widened            same argument.
        default_left      TN_DEFAULT_LEFT                  same flag bit.
        missing_bin       TN_MISSING_BIN                   both read the same
                                                           per-feature table
                                                           at the split
                                                           feature, and both
                                                           force -1 on a
                                                           categorical node.
        count             TN_COUNT widened                 the host writes
                                                           `Float64(n_left)`
                                                           from the record's
                                                           integer count; this
                                                           widens the same
                                                           integer.
        cat_offset /      TN_CAT0.. in commit order        see below.
        cat_bitset

    **Why the commit order matters.** `Tree._set_split` appends a categorical
    node's 256-bit set to a flat pool and records the offset, so the offsets
    a tree carries depend on the order its nodes were split, which under
    leaf-wise growth is not the order of their ids. Walking the node table in
    id order would therefore produce a tree that routed identically and whose
    `cat_offset` array differed, which is a difference a structural comparison
    would report and a prediction comparison would not. `snap.commit_order`
    is the device's log of that order and this walks it.

    **What is not checked here.** That the snapshot is internally consistent;
    `TreeTablesSnapshot.check_invariants` is that check and it is the
    caller's to run. This function assumes a well-formed snapshot and will
    build a nonsense tree from a malformed one rather than raising, which is
    the same contract the host grower has with its own frontier.
    """
    var n_nodes = snap.next_node
    if n_nodes < 1:
        raise Error("a tree snapshot holds at least the root")
    if n_nodes > len(snap.nodes):
        raise Error("the snapshot's node table is shorter than its counter")

    var feature = List[Int](capacity=n_nodes)
    var threshold_bin = List[Int](capacity=n_nodes)
    var left = List[Int](capacity=n_nodes)
    var right = List[Int](capacity=n_nodes)
    var value = List[Float64](capacity=n_nodes)
    var split_gain = List[Float64](capacity=n_nodes)
    var default_left = List[Bool](capacity=n_nodes)
    var missing_bin = List[Int](capacity=n_nodes)
    var cat_offset = List[Int](capacity=n_nodes)
    var count = List[Float64](capacity=n_nodes)
    var cat_bitset = List[UInt64]()

    for n in range(n_nodes):
        var row = snap.nodes[n].copy()
        feature.append(row.feature)
        threshold_bin.append(row.threshold_bin)
        left.append(row.left)
        right.append(row.right)
        value.append(Float64(row.value))
        split_gain.append(Float64(row.split_gain))
        default_left.append(row.default_left)
        missing_bin.append(row.missing_bin)
        count.append(Float64(row.count))
        cat_offset.append(-1)

    # The category pool, in commit order, exactly as `Tree._set_split` fills
    # it. A node that is not categorical contributes nothing and keeps the
    # -1 written above, which is what the whole array holds for a model with
    # no categorical features.
    for k in range(len(snap.commit_order)):
        var node = snap.commit_order[k]
        if node < 0 or node >= n_nodes:
            raise Error("the commit log names a node outside the tree")
        var row = snap.nodes[node].copy()
        if not row.is_categorical:
            continue
        cat_offset[node] = len(cat_bitset)
        var bits = row.cat_bitset()
        for w in range(CAT_BITSET_WORDS):
            cat_bitset.append(bits[w])

    return Tree(
        feature^,
        threshold_bin^,
        left^,
        right^,
        value^,
        split_gain^,
        snap.n_live,
        default_left^,
        missing_bin^,
        cat_offset^,
        cat_bitset^,
        count^,
    )
