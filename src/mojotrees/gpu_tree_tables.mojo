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
not the deciding, it is the round trip: every decision empties the device
queue, blocks on an answer it needs before it can enqueue anything else, and
then refills a pipeline that was already full.

**That is the half of the transfer model that survived 2026-08-16.** Removing
about thirty round trips per tree **measured** 0.75 seconds at 1,000,000 x 50,
resolved by a wide margin. Removing thirteen *copies* per tree, on the same
plane in the same session, measured 0.016 and did not resolve
(`bench/results/session3_2026-08-16/RESULTS.md`, and
`docs/GPU_PORTABILITY.md` section 6.1.1 for the withdrawal). A copy that
drains a queue holding nothing costs nothing. Read every copy count in this
module as a portability and hazard number; read the round-trip counts as the
ones that predict time.

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

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from std.os import getenv
from std.sys import has_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.primitives import block
from max.gpu.sync import barrier

from .categorical import (
    CAT_BITSET_WORDS,
    CatBitset,
    cat_add,
    cat_empty,
)
from .gpu_active_rows import (
    SPEC_STAT_BUILDS,
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
from .gpu_leaf_batching import (
    ITEM_BEGIN,
    ITEM_COUNT,
    ITEM_DEAD,
    ITEM_OUT,
    ITEM_WORDS,
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
    OBLIVIOUS_MAX_LEAVES,
    PF_G_INV,
    PF_H_INV,
    PF_LAMBDA_L1,
    PF_LAMBDA_L2,
    PF_WORDS,
    SPLIT_FWORDS,
    SPLIT_IWORDS,
    gpu_leaf_value,
    gpu_resolve_gain_form,
    gpu_right_sum,
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
    # `device_unsupported_reason` rather than `is_active()`. The two agreed
    # while every member of the bundle was unimplemented on the device; they
    # stopped agreeing when the Cosine and random_strength kernels landed, and
    # this gate went on refusing two settings the device can now score.
    #
    # `has_categorical=False` because this function takes no data, and that is
    # sound rather than optimistic: the only behavioral caller,
    # `gpu_resident_round.resident_round_supported`, tests
    # `builder.cats.any_categorical()` on the very next line and returns
    # `RESIDENT_CATEGORICAL`. The data question stays with the caller that
    # holds the data, which is the same division `oblivious_device_supported`
    # keeps.
    if (
        params.extra.device_unsupported_reason(False).byte_length() > 0
        or params.feature_fraction_bylevel != 1.0
    ):
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

    def trace_line(self) -> String:
        """The counters as one line of `key=value` pairs.

        This exists because a device-owned growth loop has no per-split host
        view at all. On the shipping loop a wrong tree can be bisected by
        printing the frontier after each download, since there is a download
        after each split. Here there is one download per tree, so the only
        thing a reader can inspect without changing what runs is what that
        one download brought home, and this is it.

        The format is deliberately flat, single-line, and stable: a test
        parses it by counting substrings rather than by understanding it, so
        a passing assertion about `status=budget_spent` is an assertion about
        a word a device counter produced. Anything added later goes on the
        end, so an existing reader keeps working.
        """
        return String(
            "status=",
            tree_status_name(self.status),
            " commits=",
            self.commits,
            " leaves=",
            self.n_live,
            " nodes=",
            self.next_node,
            " pick=",
            self.pick,
            " pick_node=",
            self.pick_node,
        )

    def describe(self) -> String:
        """The frontier, the slot pool and the commit log, as indented text.

        Several lines rather than one, and indented, so that it reads under a
        `trace_line` above it. Everything printed is an exact integer that a
        device kernel wrote; nothing here recomputes or interprets, because a
        trace that computed anything would be a second implementation of the
        thing being debugged.

        Deliberately *not* including `trace_line`, even though a caller almost
        always wants both. A trace is counted as well as read: a test asserts
        on how many records carry `status=budget_spent`, and a status printed
        twice per record would be counted twice. The caller writes the summary
        line and then this.

        The frontier is printed in slot order, which is load bearing rather
        than cosmetic: slot order is the pick kernel's tie-breaking rule, so
        a reader comparing two traces is comparing the order decisions were
        made in and not merely the set of leaves.
        """
        var out = String("")
        for i in range(len(self.leaves)):
            var leaf = self.leaves[i].copy()
            out += String(
                "  leaf slot=",
                i,
                " node=",
                leaf.node,
                " rows=[",
                leaf.row_begin,
                ",",
                leaf.row_end(),
                ") count=",
                leaf.row_count,
                " depth=",
                leaf.depth,
                " hist_slot=",
                leaf.hist_slot,
                " record=",
                leaf.record,
                "\n",
            )
        out += String("  slot_owner=")
        for i in range(len(self.slot_owner)):
            out += String(" ", self.slot_owner[i])
        out += String("\n  commit_order=")
        for k in range(len(self.commit_order)):
            out += String(" ", self.commit_order[k])
        out += String("\n")
        return out^


# --- The kernel -----------------------------------------------------------


comptime PLAN_ITEMS = 2
"""Item rows one leaf-wise commit fills in a batched-histogram plan: the two
children, left then right.

Two rather than one because the batched shape builds *both* children from
their own rows instead of building the smaller and subtracting. That is the
same two launches whatever the batch holds, so the second child is free where
the descriptor path would have paid a second pair for it -- and it is what
makes the same writer generalize to a level, where the count is `2^(l+1)` and
the launch count is still two."""


@always_inline
def _write_plan_item(
    plan: MutPointer[Int32, MutAnyOrigin],
    item: Int,
    begin: Int32,
    count: Int32,
    out_slot: Int32,
):
    """One row of `gpu_leaf_batching`'s item table, the three words a commit
    decides.

    The other four -- `ITEM_ROWS_PER_TILE`, `ITEM_TILE_BEGIN`, `ITEM_TILES`,
    `ITEM_PLANE` -- are launch geometry and are staged once per tree by
    `GpuLeafBatcher.stage_device_plan`, because a grid is a host argument to
    `enqueue_function` and no kernel can change one. This is the whole of the
    split between the two halves of a device-written plan, and it is why the
    plan costs no launch of its own: everything the device gets to decide is
    three Int32 written by a kernel that was already running."""
    var base = item * ITEM_WORDS
    plan[unsafe_offset = base + ITEM_BEGIN] = begin
    plan[unsafe_offset = base + ITEM_COUNT] = count
    plan[unsafe_offset = base + ITEM_OUT] = out_slot


@always_inline
def _kill_plan(plan: MutPointer[Int32, MutAnyOrigin], write_plan: Int32):
    """Mark both plan items dead, so a batch already in the queue does
    nothing.

    Called from every exit of the commit kernel that commits nothing, for the
    same reason those exits all write `STEP_LIVE = 0`: a whole tree's launches
    are enqueued before the tree has decided how many splits it will take, so
    "do nothing" has to be a state the tables can express rather than a launch
    the host declines to make. `ITEM_DEAD` is stronger than a zero count --
    zero means a live leaf with no rows, whose histogram slot must still be
    cleared -- and a dead item's stale `ITEM_OUT` would otherwise have
    `_batch_zero_kernel` erase a live leaf's histogram."""
    if write_plan != Int32(0):
        for i in range(PLAN_ITEMS):
            _write_plan_item(plan, i, Int32(0), Int32(ITEM_DEAD), Int32(0))


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
    plan: MutPointer[Int32, MutAnyOrigin],
    write_plan: Int32,
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

    `plan` is the third, and it is the one that changes what the growth loop
    above can be shaped like. It is `gpu_leaf_batching`'s item table -- the
    array `GpuLeafBatcher.enqueue_batch` reads its per-leaf windows and
    destination slots out of -- and this kernel fills the three words of it
    that a commit decides, for each of the commit's two children, under
    `write_plan`. Until now that table was staged from the host, so a batched
    build could not be driven from inside a device-owned tree without the
    round trip the resident plane exists to remove. The geometry half of the
    table is still host-staged, because a grid is a host argument to
    `enqueue_function` and no kernel can change one, but the geometry does not
    move inside a tree; see `GpuLeafBatcher.stage_device_plan`.

    `write_plan = 0` writes nothing there and is what both `enqueue_step`
    overloads pass, so every path that exists today is byte for byte
    unchanged. `enqueue_step_with_plan` is the arm.
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
            _kill_plan(plan, write_plan)
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
        _kill_plan(plan, write_plan)
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
        _kill_plan(plan, write_plan)
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
        _kill_plan(plan, write_plan)
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

    # The batched-histogram plan, when the caller asked for one. Two items,
    # left child then right child, each naming its own rows and its own pool
    # slot -- which are exactly the numbers this kernel has just decided and
    # the numbers a batched build needs.
    #
    # **This describes a different build from `STEP_BUILT_*`, and the caller
    # picks one of the two rather than running both.** The descriptor's build
    # is the subtraction shape: accumulate the smaller child from its rows and
    # derive the larger by subtracting it from the parent's slot, which is two
    # launches per parent. The plan's build is the batched shape: accumulate
    # *both* children from their own rows, which is two launches per batch
    # however many parents the batch covers, and needs no subtraction at all.
    # The parent's histogram is destroyed either way -- the pool reassigns its
    # slot to a child in both -- so nothing downstream can tell which was used
    # except by counting command buffers.
    #
    # **The two produce bit-identical histograms.** Both accumulate the same
    # per-`(row, feature)` quantized value into the same bin of the same slot,
    # in fixed-point Int32; a child's rows are the same rows either way,
    # because both read the window this same commit wrote. Integer addition is
    # associative and commutative, so a different launch shape over the same
    # visits is the same sum, and the sibling the subtraction derives from two
    # exact integer histograms is exactly the one the batch accumulates
    # directly. There is no floating point in either path past the
    # quantization, which both do identically.
    #
    # The one arm this must not be combined with is the K=1 speculation.
    # `_pick_runner_up_kernel` publishes a leaf that is still live and the
    # speculative build deliberately does not fold its subtraction in, for
    # exactly the reason a batched build of both children would break it: the
    # speculated leaf's own histogram is what the next pick reads on a miss,
    # and this plan overwrites it.
    if write_plan != Int32(0):
        _write_plan_item(
            plan, 0, Int32(begin), n_left, Int32(left_slot)
        )
        _write_plan_item(
            plan, 1, Int32(begin) + n_left, n_right, Int32(right_slot)
        )

    var commits = Int(ctr[unsafe_offset=CTR_COMMITS][0])
    if commits < Int(leaf_capacity):
        order[unsafe_offset=commits] = Int32(parent)
    ctr[unsafe_offset=CTR_NEXT_NODE] = Int32(next_node + 2)
    ctr[unsafe_offset=CTR_N_LIVE] = Int32(n_live + 1)
    ctr[unsafe_offset=CTR_PICK] = Int32(slot)
    ctr[unsafe_offset=CTR_PICK_NODE] = Int32(parent)
    ctr[unsafe_offset=CTR_STATUS] = Int32(TREE_RUNNING)
    ctr[unsafe_offset=CTR_COMMITS] = Int32(commits + 1)


# --- grow_policy = oblivious: the level commit ----------------------------
#
# The kernel below is `_pick_and_commit_kernel`'s rule with the *pick* taken
# out of it and the *commit* widened from one parent to a whole level. Both
# halves of that sentence are load bearing:
#
# - There is no pick. An oblivious level's split is decided by one search over
#   the whole level (`gpu_split_search._scan_slot_oblivious_kernel` folded by
#   `_reduce_slots_kernel` into one record), so by the time this runs the
#   decision is a single record and the reduction the leaf-wise kernel spends
#   its first two phases on has already happened, one launch earlier, fused
#   into the scan. What is left is the commit.
#
# - The commit applies that one split to every leaf of the level, in ascending
#   leaf index, left child before right, which is
#   `gpu_resident_round.OBLIVIOUS_LEAF_INDEX_RULE` and is the half of the
#   cross-backend contract that decides node *ids*.
#
# It is a separate kernel rather than an arm of `_pick_and_commit_kernel`, and
# that is a deliberate reading of "bits move only in the new mode": every path
# that exists today keeps the kernel it had, byte for byte, and no leaf-wise
# launch grows a branch it can never take.

comptime HIST_PLANES = 3
"""Planes in one resident histogram slot: gradient, hessian, count, in that
order and each `n_features * n_bins` Int32 long.

The same number `gpu_leaf_batching.N_PLANES` is and the same stride
`3 * n_features * n_bins` that every `hist_slot` in this package is multiplied
by. Named here because the level commit is the only kernel in this file that
indexes a histogram, and a bare 3 next to a bare 2 is exactly the kind of
constant that gets read as an off-by-one."""

comptime OBLIVIOUS_LEVEL_LEAVES = OBLIVIOUS_MAX_LEAVES
"""Leaves in one level this commit will apply a split to, which is `2 ** 6`.

The same bound `gpu_split_search.OBLIVIOUS_MAX_LEAVES` puts on the cross-leaf
scan, imported rather than restated so the two cannot drift: a level this
kernel would commit and that kernel would not search is a level whose split
came from a truncated sum."""

comptime OBLIVIOUS_PLAN_ITEMS = 2 * OBLIVIOUS_MAX_LEAVES
"""Item rows a level commit may fill in a batched-histogram plan.

A level of `L` parents makes `2L` children and every one of them is built from
its own rows, so the plan is twice as wide as the level. The widest level a
depth-6 tree commits is `L = 32`, giving 64 items -- which is exactly
`gpu_leaf_batching.OBLIVIOUS_MAX_ITEMS`, and that agreement is the sizing
precondition of the whole census rather than a coincidence. The constant here
is `2 * 64` because it bounds the *scratch* plan a caller may hand this struct
without a batcher, and a caller that hands in a real batcher's `items_dev` is
bounded by that batcher's own `max_items`."""


@always_inline
def _kill_level_plan(
    plan: MutPointer[Int32, MutAnyOrigin], write_plan: Int32, n_items: Int
):
    """Mark every item of a level's plan dead.

    `_kill_plan`'s reasoning at a level's width, and it kills the whole staged
    width rather than the level's own `2L`: a batch is launched over the item
    count `stage_device_plan` fixed for the tree, so an item this level does not
    use is an item some *earlier, narrower* level filled and whose `ITEM_OUT`
    still names a histogram slot that is now some other leaf's. Killing only the
    prefix would leave `_batch_zero_kernel` erasing a live leaf's histogram --
    the exact failure `ITEM_DEAD` exists to prevent, arriving from the other
    direction."""
    if write_plan != Int32(0):
        for i in range(n_items):
            _write_plan_item(plan, i, Int32(0), Int32(ITEM_DEAD), Int32(0))


def _commit_level_kernel(
    front: MutPointer[Int32, MutAnyOrigin],
    node_i: MutPointer[Int32, MutAnyOrigin],
    node_f: MutPointer[Float32, MutAnyOrigin],
    ctr: MutPointer[Int32, MutAnyOrigin],
    slot_owner: MutPointer[Int32, MutAnyOrigin],
    missing: MutPointer[Int32, MutAnyOrigin],
    rec_i: MutPointer[Int32, MutAnyOrigin],
    rec_f: MutPointer[Float32, MutAnyOrigin],
    hist: MutPointer[Int32, MutAnyOrigin],
    fparams: MutPointer[Float32, MutAnyOrigin],
    step: MutPointer[Int32, MutAnyOrigin],
    order: MutPointer[Int32, MutAnyOrigin],
    plan: MutPointer[Int32, MutAnyOrigin],
    write_plan: Int32,
    level_record: Int32,
    n_bins: Int32,
    hist_size: Int32,
    level_depth: Int32,
    max_depth: Int32,
    plan_items: Int32,
    pool_capacity: Int32,
    leaf_capacity: Int32,
    node_capacity: Int32,
    gain_form: Int32,
):
    """One oblivious level, committed entirely on the device.

    Launched as a single threadgroup of `PICK_THREADS` threads over a grid of
    one block, exactly as `_pick_and_commit_kernel` is, and for a reason that is
    now even more direct: `PICK_THREADS` is 64 and so is
    `OBLIVIOUS_LEVEL_LEAVES`, so the level's per-leaf work is one thread per
    leaf and the block is neither too small nor too large by construction.

    The three phases, in order:

    **Stop.** Growth ends when the depth budget is spent, when the level search
    found nothing, or when a previous level already stopped. All three are
    uniform across the block -- every thread reads the same words -- so the
    early return is uniform and no thread reaches the barrier that the others
    skip. `level_depth >= max_depth` is what makes the schedule's tail launch a
    *terminal status* rather than a step that quietly does nothing, which is the
    same job `_growth_finished_normally` needs done and the same shape the
    leaf-wise loop's extra `enqueue_desc_step` has.

    A stopped tree stays stopped. The commit reads `CTR_STATUS` first, so once
    a level writes `TREE_NO_CANDIDATE` every later level's commit returns
    without reading a record. That matters here in a way it does not in the
    leaf-wise loop: after a stop the frontier stops growing while the host
    schedule keeps enqueueing searches sized for the level the tree *would*
    have reached, so those searches read leaf records the frontier no longer
    fills. They are in bounds -- the schedule reserves records for the widest
    level of the tree and the tables are zeroed at `begin_tree` -- and their
    answers are read by nothing, because this test is the first thing every
    later commit makes.

    **Per-leaf statistics, one thread per leaf.** Thread `l` walks leaf `l`'s
    own histogram slice for the winning feature and produces that leaf's own
    left and right counts and gradient/hessian sums. This is the arithmetic a
    leaf-wise commit gets for free off the record and an oblivious one cannot:
    the level record's `IREC_LEFT_COUNT` is a sum over the leaves that found the
    candidate *legal*, which is the right number for scoring the level and the
    wrong number for any one leaf's window. Each leaf's rows are its own.

    The left/right test is `gpu_active_rows._row_goes_left` written over bins
    instead of over rows -- the missing bin takes the level's default direction
    and every other bin takes the inclusive threshold -- so the count this
    derives is by construction the count `_scatter_kernel` will route, and not
    a second derivation that has to be argued equal to it.

    **The commit, on one thread.** Serial and deliberately so. It assigns
    `2L` node ids in ascending leaf index with the left child before the right,
    writes the same split onto all `L` parents, writes both children of each as
    leaves with their own counts and their own Newton values, lays the level's
    `2L` windows out over the prefix, moves the slot pool, rewrites the
    frontier, fills the plan, and publishes one step descriptor. None of that
    reduces over anything, so a parallel form would buy nothing and would have
    to agree with this one about an order that *is* the answer.

    **Why one partition of the whole prefix produces the whole level, and why
    the windows below are the ones it writes.** At every level the live leaves
    tile `[0, n_active)` in ascending leaf index -- the invariant this kernel
    both assumes and re-establishes. One stable partition of that prefix by the
    level's single routing rule therefore leaves every leaf's left-going rows in
    the front block, in ascending leaf order, and every leaf's right-going rows
    in the back block, in ascending leaf order. The new leaf indices are `j` for
    the left child of `j` and `j + L` for its right child
    (`OBLIVIOUS_LEAF_INDEX_RULE`, CatBoost's numbering), so the physical order
    after the partition *is* ascending new leaf index and the invariant holds
    again. The two cursors below are the exclusive prefix sums of the left and
    right counts, which is the whole of that argument written as arithmetic.

    That is also why `gpu_active_rows.LeafRangeTable.split` cannot replay this
    tree and `set_window` exists: a parent's rows end up in two blocks that are
    not adjacent.

    **Slot index equals leaf index**, so the pool assignment is a function of
    the tree and not of an allocation order. A leaf-wise commit scans `slot_owner`
    upward for the lowest free slot because its frontier slots and its pool slots
    have no relationship; here they do, and pinning them removes an ordering
    from the answer rather than merely simplifying the code. The pool is sized at
    `1 << max_depth`, so the widest level's `2L = 1 << max_depth` children fit
    exactly.

    **The parent's own histogram is read here and destroyed afterwards**, and
    the order is what makes that safe: this commit runs before the partition and
    before the batched build, so every leaf's slice is still the slice its own
    search read. The batch that follows zeroes each destination slot before it
    accumulates, so nothing inherits.

    **Every parent of the level carries the level's summed gain** in
    `TN_SPLIT_GAIN`, which is not an approximation of a per-node gain but the
    quantity that was actually maximized; `tree._grow_oblivious_levels` writes
    the same number onto the same nodes through `Tree._set_split`, so this is one
    of the cross-backend agreements a node-for-node comparison checks.

    No floating-point comparison decides anything here. The gain is tested
    against zero, the counts are exact Int32, and the leaf values are the same
    `gpu_leaf_value` a leaf-wise commit copies out of the record -- computed
    here instead of copied, because there is no per-leaf record to copy from.
    """
    var tid = Int(thread_idx.x)
    var status = Int(ctr[unsafe_offset=CTR_STATUS][0])
    var n_live = Int(ctr[unsafe_offset=CTR_N_LIVE][0])
    var ri = Int(level_record) * SPLIT_IWORDS
    var rf = Int(level_record) * SPLIT_FWORDS
    var flags = rec_i[unsafe_offset = ri + IREC_FLAGS][0]
    var gain = rec_f[unsafe_offset = rf + FREC_GAIN][0]

    # Phase 1: has growth ended, or is this the launch that ends it?
    var depth_spent = Int(level_depth) >= Int(max_depth)
    var no_candidate = (flags & Int32(FLAG_FOUND)) == Int32(0) or gain <= (
        Float32(0.0)
    )
    if status != TREE_RUNNING or depth_spent or no_candidate:
        if tid == 0:
            ctr[unsafe_offset=CTR_PICK] = Int32(-1)
            ctr[unsafe_offset=CTR_PICK_NODE] = Int32(-1)
            step[unsafe_offset=STEP_LIVE] = Int32(0)
            _kill_level_plan(plan, write_plan, Int(plan_items))
            if status == TREE_RUNNING:
                if depth_spent:
                    ctr[unsafe_offset=CTR_STATUS] = Int32(TREE_BUDGET_SPENT)
                else:
                    ctr[unsafe_offset=CTR_STATUS] = Int32(TREE_NO_CANDIDATE)
        return

    # The capacities, all of them uniform and all of them checked before a
    # single table is written, so an undersized table reports rather than
    # corrupts. Unreachable when the caller sized every table from
    # `1 << max_depth`, which `gpu_resident_round.oblivious_leaf_budget` is.
    var next_node = Int(ctr[unsafe_offset=CTR_NEXT_NODE][0])
    var commits = Int(ctr[unsafe_offset=CTR_COMMITS][0])
    var pool_short = 2 * n_live > Int(pool_capacity)
    var over = (
        n_live > OBLIVIOUS_LEVEL_LEAVES
        or 2 * n_live > Int(leaf_capacity)
        or next_node + 2 * n_live > Int(node_capacity)
        or commits + n_live > Int(leaf_capacity)
        or (write_plan != Int32(0) and 2 * n_live > Int(plan_items))
    )
    if pool_short or over:
        if tid == 0:
            ctr[unsafe_offset=CTR_PICK] = Int32(-1)
            ctr[unsafe_offset=CTR_PICK_NODE] = Int32(-1)
            if pool_short:
                ctr[unsafe_offset=CTR_STATUS] = Int32(TREE_POOL_FULL)
            else:
                ctr[unsafe_offset=CTR_STATUS] = Int32(TREE_OVERFLOW)
            step[unsafe_offset=STEP_LIVE] = Int32(0)
            _kill_level_plan(plan, write_plan, Int(plan_items))
        return

    # Phase 2: each leaf's own split statistics, one thread per leaf.
    var feature = rec_i[unsafe_offset = ri + IREC_FEATURE][0]
    var threshold = rec_i[unsafe_offset = ri + IREC_BIN][0]
    var default_left = (flags & Int32(FLAG_DEFAULT_LEFT)) != Int32(0)
    var missing_bin = missing[unsafe_offset = Int(feature)][0]
    var nb = Int(n_bins)
    var hs = Int(hist_size)
    var pf = Int(level_record) * PF_WORDS
    var g_inv = fparams[unsafe_offset = pf + PF_G_INV][0]
    var h_inv = fparams[unsafe_offset = pf + PF_H_INV][0]
    var lambda_l1 = fparams[unsafe_offset = pf + PF_LAMBDA_L1][0]
    var lambda_l2 = fparams[unsafe_offset = pf + PF_LAMBDA_L2][0]
    var form = gpu_resolve_gain_form(gain_form, lambda_l1)

    var s_parent = stack_allocation[
        OBLIVIOUS_LEVEL_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_begin = stack_allocation[
        OBLIVIOUS_LEVEL_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_nleft = stack_allocation[
        OBLIVIOUS_LEVEL_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_nright = stack_allocation[
        OBLIVIOUS_LEVEL_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_lval = stack_allocation[
        OBLIVIOUS_LEVEL_LEAVES,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_rval = stack_allocation[
        OBLIVIOUS_LEVEL_LEAVES,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    if tid < n_live:
        var fo = tid * FRONT_WORDS
        s_parent[unsafe_offset=tid] = front[unsafe_offset = fo + FRONT_NODE][0]
        s_begin[unsafe_offset=tid] = front[
            unsafe_offset = fo + FRONT_ROW_BEGIN
        ][0]
        var slot = Int(front[unsafe_offset = fo + FRONT_HIST_SLOT][0])
        var base = slot * HIST_PLANES * hs + Int(feature) * nb
        var lg = Int32(0)
        var lh = Int32(0)
        var lc = Int32(0)
        var tg = Int32(0)
        var th = Int32(0)
        var tc = Int32(0)
        for b in range(nb):
            var g = hist[unsafe_offset = base + b][0]
            var h = hist[unsafe_offset = hs + base + b][0]
            var c = hist[unsafe_offset = 2 * hs + base + b][0]
            tg += g
            th += h
            tc += c
            # `_row_goes_left` over bins: the missing bin takes the level's
            # default direction, every other bin takes the inclusive
            # threshold. A feature with no missing bin has -1 here, which no
            # bin id equals.
            var goes_left: Bool
            if Int32(b) == missing_bin:
                goes_left = default_left
            else:
                goes_left = Int32(b) <= threshold
            if goes_left:
                lg += g
                lh += h
                lc += c
        s_nleft[unsafe_offset=tid] = lc
        s_nright[unsafe_offset=tid] = tc - lc
        var lgf = lg.cast[DType.float32]() * g_inv
        var lhf = lh.cast[DType.float32]() * h_inv
        var tgf = tg.cast[DType.float32]() * g_inv
        var thf = th.cast[DType.float32]() * h_inv
        var rgf = gpu_right_sum(tgf, lgf, tg, lg, g_inv, form)
        var rhf = gpu_right_sum(thf, lhf, th, lh, h_inv, form)
        s_lval[unsafe_offset=tid] = gpu_leaf_value(
            lgf, lhf, lambda_l1, lambda_l2
        )
        s_rval[unsafe_offset=tid] = gpu_leaf_value(
            rgf, rhf, lambda_l1, lambda_l2
        )
    barrier()

    if tid != 0:
        return

    # Phase 3: the commit, on one thread.
    var origin = Int(s_begin[unsafe_offset=0][0])
    var total_left = 0
    var total_rows = 0
    for j in range(n_live):
        total_left += Int(s_nleft[unsafe_offset=j][0])
        total_rows += Int(s_nleft[unsafe_offset=j][0]) + Int(
            s_nright[unsafe_offset=j][0]
        )
    # The two block cursors: all of the level's left children first, in
    # ascending leaf index, then all of its right children. See the docstring.
    var left_cursor = origin
    var right_cursor = origin + total_left

    for j in range(n_live):
        var parent = Int(s_parent[unsafe_offset=j][0])
        var left_node = next_node + 2 * j
        var right_node = left_node + 1
        var n_left = s_nleft[unsafe_offset=j][0]
        var n_right = s_nright[unsafe_offset=j][0]

        # `Tree._set_split` on the parent, field for field. Numerical only:
        # the level search refuses a dataset with a categorical feature, so
        # nothing here can be a category set.
        var po = parent * TN_IWORDS
        node_i[unsafe_offset = po + TN_FEATURE] = feature
        node_i[unsafe_offset = po + TN_THRESHOLD] = threshold
        node_i[unsafe_offset = po + TN_LEFT] = Int32(left_node)
        node_i[unsafe_offset = po + TN_RIGHT] = Int32(right_node)
        node_i[unsafe_offset = po + TN_DEFAULT_LEFT] = (
            Int32(1) if default_left else Int32(0)
        )
        node_i[unsafe_offset = po + TN_MISSING_BIN] = missing_bin
        node_i[unsafe_offset = po + TN_IS_CATEGORICAL] = Int32(0)
        node_f[unsafe_offset = parent * TN_FWORDS + TN_SPLIT_GAIN] = gain

        # The two `Tree._add_node` calls, left then right.
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
        node_f[unsafe_offset = left_node * TN_FWORDS + TN_VALUE] = s_lval[
            unsafe_offset=j
        ][0]
        node_f[
            unsafe_offset = left_node * TN_FWORDS + TN_SPLIT_GAIN
        ] = Float32(0.0)
        node_f[unsafe_offset = right_node * TN_FWORDS + TN_VALUE] = s_rval[
            unsafe_offset=j
        ][0]
        node_f[
            unsafe_offset = right_node * TN_FWORDS + TN_SPLIT_GAIN
        ] = Float32(0.0)

        # The windows, and with them the frontier. The left child takes leaf
        # index `j` and the right child `j + L`, which is where each of them
        # lands in the partitioned prefix and is also which pool slot each of
        # them owns.
        var left_begin = left_cursor
        left_cursor += Int(n_left)
        var right_begin = right_cursor
        right_cursor += Int(n_right)

        var lfo = j * FRONT_WORDS
        front[unsafe_offset = lfo + FRONT_NODE] = Int32(left_node)
        front[unsafe_offset = lfo + FRONT_ROW_BEGIN] = Int32(left_begin)
        front[unsafe_offset = lfo + FRONT_ROW_COUNT] = n_left
        front[unsafe_offset = lfo + FRONT_DEPTH] = level_depth + Int32(1)
        front[unsafe_offset = lfo + FRONT_HIST_SLOT] = Int32(j)
        front[unsafe_offset = lfo + FRONT_RECORD] = Int32(j)

        var rfo = (j + n_live) * FRONT_WORDS
        front[unsafe_offset = rfo + FRONT_NODE] = Int32(right_node)
        front[unsafe_offset = rfo + FRONT_ROW_BEGIN] = Int32(right_begin)
        front[unsafe_offset = rfo + FRONT_ROW_COUNT] = n_right
        front[unsafe_offset = rfo + FRONT_DEPTH] = level_depth + Int32(1)
        front[unsafe_offset = rfo + FRONT_HIST_SLOT] = Int32(j + n_live)
        front[unsafe_offset = rfo + FRONT_RECORD] = Int32(j + n_live)

        slot_owner[unsafe_offset=j] = Int32(left_node)
        slot_owner[unsafe_offset = j + n_live] = Int32(right_node)

        # The commit log, in the order `Tree._set_split` is called on the host:
        # over the level's leaves in ascending leaf index.
        order[unsafe_offset = commits + j] = Int32(parent)

        if write_plan != Int32(0):
            _write_plan_item(
                plan, j, Int32(left_begin), n_left, Int32(j)
            )
            _write_plan_item(
                plan,
                j + n_live,
                Int32(right_begin),
                n_right,
                Int32(j + n_live),
            )

    # Slots past this level's children belong to nobody. Written every level
    # rather than only when the level narrows, because `check_invariants`
    # compares the pool against the frontier and a stale owner is exactly the
    # disagreement it looks for.
    for s in range(2 * n_live, Int(pool_capacity)):
        slot_owner[unsafe_offset=s] = Int32(-1)
    # Items past this level's children build nothing. See `_kill_level_plan`.
    if write_plan != Int32(0):
        for i in range(2 * n_live, Int(plan_items)):
            _write_plan_item(plan, i, Int32(0), Int32(ITEM_DEAD), Int32(0))

    # The step descriptor. Written last, after every table this commit touched,
    # for the reason `_pick_and_commit_kernel` gives: a reader of `STEP_LIVE` on
    # the same queue then sees tables already consistent with it.
    #
    # The window is the **whole level**, which is the one thing about this
    # descriptor that differs in kind from a leaf-wise one, and it is what makes
    # `GpuActiveRows.enqueue_partition_desc` produce an entire level in one
    # stable partition with no change at all.
    step[unsafe_offset=STEP_LIVE] = Int32(1)
    step[unsafe_offset=STEP_ROW_BEGIN] = Int32(origin)
    step[unsafe_offset=STEP_ROW_COUNT] = Int32(total_rows)
    step[unsafe_offset=STEP_FEATURE] = feature
    step[unsafe_offset=STEP_THRESHOLD] = threshold
    step[unsafe_offset=STEP_MISSING_BIN] = missing_bin
    step[unsafe_offset=STEP_DEFAULT_LEFT] = (
        Int32(1) if default_left else Int32(0)
    )
    step[unsafe_offset=STEP_IS_CAT] = Int32(0)
    for w in range(CAT_WORDS):
        step[unsafe_offset = STEP_CAT0 + w] = Int32(0)
    # The single-child build's words. This schedule does not launch
    # `enqueue_desc_child` -- the level's children come from one batch -- and a
    # zero count is what makes that unambiguous: a caller who wired the
    # single-child path in here by mistake would accumulate nothing and be
    # caught by an empty histogram, rather than silently rebuild one leaf of the
    # level over the whole prefix.
    step[unsafe_offset=STEP_BUILT_BEGIN] = Int32(origin)
    step[unsafe_offset=STEP_BUILT_COUNT] = Int32(0)
    step[unsafe_offset=STEP_BUILT_SLOT] = Int32(0)
    step[unsafe_offset=STEP_SUB_SLOT] = Int32(0)
    step[unsafe_offset=STEP_LEFT_SLOT] = Int32(0)
    step[unsafe_offset=STEP_RIGHT_SLOT] = Int32(0)
    step[unsafe_offset=STEP_LEFT_REC] = Int32(0)
    step[unsafe_offset=STEP_RIGHT_REC] = Int32(0)

    ctr[unsafe_offset=CTR_NEXT_NODE] = Int32(next_node + 2 * n_live)
    ctr[unsafe_offset=CTR_N_LIVE] = Int32(2 * n_live)
    ctr[unsafe_offset=CTR_COMMITS] = Int32(commits + n_live)
    # There is no pick. `CTR_PICK_NODE` carries the level's lowest node id,
    # which is leaf index 0's parent and is the same key
    # `tree._grow_oblivious_levels` uses for the level; `CTR_PICK` is -1
    # because no frontier slot was selected over any other.
    ctr[unsafe_offset=CTR_PICK] = Int32(-1)
    ctr[unsafe_offset=CTR_PICK_NODE] = s_parent[unsafe_offset=0][0]
    ctr[unsafe_offset=CTR_STATUS] = Int32(TREE_RUNNING)


def _stage_level_search_kernel(
    node_tbl: MutPointer[Int32, MutAnyOrigin],
    front: MutPointer[Int32, MutAnyOrigin],
    ctr: MutPointer[Int32, MutAnyOrigin],
    slot_cells: Int32,
    leaf_base: Int32,
    max_leaves: Int32,
):
    """Point the level's leaf search records at the level's histogram slots.

    `_stage_child_search_kernel` for a whole level: one Int32 write per live
    leaf instead of two per split. `gpu_split_search._scan_slot_oblivious_kernel`
    reads records `[leaf_base, leaf_base + n_leaves)` for exactly one word each,
    `NODE_HIST_BASE`, and reads everything else it needs -- the feature list, the
    allow mask, the monotone vector, the float parameter block -- from the level
    record; under this plane's refusals all four are tree-level and are staged
    once by the host.

    **Record index equals frontier slot equals leaf index equals pool slot.**
    The level commit pins all four together, so this kernel is a copy rather
    than a mapping, and the order the scan sums in is the order the frontier is
    in without anything having to sort it. That is
    `gpu_resident_round.OBLIVIOUS_LEAF_INDEX_RULE` reduced to an identity.

    Records past `n_live` are written too, up to `max_leaves`, and pointed at
    slot 0. They are the records a level search reads after growth has stopped:
    the host schedule keeps enqueueing searches sized for the level the tree
    would have reached, and a record left holding a previous tree's base would
    aim that search at a slot outside the pool. Pointing them at slot 0 makes
    those reads in-bounds and meaningless, which is what they should be -- the
    commit that follows returns on `CTR_STATUS` before it reads any record.

    One thread. The work is at most 64 stores and a launch of one block of one
    thread is what the leaf-wise counterpart already costs."""
    if thread_idx.x != 0:
        return
    var n_live = Int(ctr[unsafe_offset=CTR_N_LIVE][0])
    for i in range(Int(max_leaves)):
        var slot = Int32(0)
        if i < n_live:
            slot = front[unsafe_offset = i * FRONT_WORDS + FRONT_HIST_SLOT][0]
        node_tbl[
            unsafe_offset = (Int(leaf_base) + i) * NODE_WORDS + NODE_HIST_BASE
        ] = slot * slot_cells


def _pick_runner_up_kernel(
    front: MutPointer[Int32, MutAnyOrigin],
    ctr: MutPointer[Int32, MutAnyOrigin],
    slot_owner: MutPointer[Int32, MutAnyOrigin],
    missing: MutPointer[Int32, MutAnyOrigin],
    rec_i: MutPointer[Int32, MutAnyOrigin],
    rec_f: MutPointer[Float32, MutAnyOrigin],
    step: MutPointer[Int32, MutAnyOrigin],
    spec: MutPointer[Int32, MutAnyOrigin],
    stats: MutPointer[Int32, MutAnyOrigin],
    num_leaves: Int32,
    max_depth: Int32,
    min_data_in_leaf: Int32,
    pool_capacity: Int32,
):
    """Publish the split the *next* step is most likely to commit, so that its
    child histogram can be built now instead of then. Writes a descriptor and
    no table.

    Phase two of `_pick_and_commit_kernel` with its committing phase three
    replaced by a descriptor write: the same strided walk in ascending slot
    order, the same shape rules, the same strict `>`, the same
    `block.max` then `block.min`, and therefore the same tie resolution. It
    has to be the same rule and not an equivalent-looking one, because the
    whole value of the prebuild is that the leaf it names is the leaf the
    next commit names.

    **Why one candidate is the whole of the candidate set**, which is a
    theorem rather than a tuning choice and is argued at length in
    `gpu_resident_round.mojo`: at this point in a step the two children the
    commit just created have no search records yet -- the child search has
    not run -- so they cannot be speculated on at all, and their candidacy is
    established at exactly the moment speculating on them stops being
    speculation. The candidate set is therefore the leaves that were live
    *before* this step, and over that set a commit moves no ranking: it
    writes only the split leaf's frontier row, the appended row at `n_live`,
    and the two records those children own. Every other leaf keeps its slot,
    its record, its depth and its row count bit for bit, so the best
    pre-existing leaf at the next step is exactly the best one here. A second
    speculative candidate would be a leaf that provably cannot be picked
    next.

    **The two slots this excludes, and why the exclusion is a correctness
    requirement rather than an optimization.** Slot `CTR_PICK` now holds the
    left child and slot `n_live - 1` holds the right child, and both of them
    carry record indices whose contents this step has not written yet: the
    search that will fill them is enqueued after this launch. Including
    either would reduce over a record left by an earlier step or an earlier
    tree, which can carry `FLAG_FOUND` and an arbitrarily large gain. That is
    not a wrong guess, it is a read of stale memory, and it is the one way
    this kernel could publish a descriptor that partitions a window by a
    split nothing chose.

    **Four ways it declines, and all four write `STEP_LIVE = 0`.** A step
    that committed nothing has nothing to speculate behind; a step after
    which the leaf budget is spent is followed by a step that returns on the
    budget before it reads anything; a frontier whose pre-existing leaves are
    all inadmissible offers no candidate; and a slot pool with nothing free
    has nowhere to build. The last of those cannot happen while a next commit
    can, since a commit needs a free slot too, and it is checked anyway
    because a speculative build writing into slot -1 would be a device fault
    rather than a wrong answer.

    **The slot it builds into is the slot the next commit will acquire, and
    that is a prediction with a proof rather than a guess.** The commit
    kernel takes the lowest free slot by scanning `slot_owner` upward; this
    kernel scans the same vector the same way and writes nothing to it.
    Between here and the next commit nothing acquires and nothing releases,
    so the lowest free slot is the same slot. Building there rather than into
    a reserved spare is what keeps the pool at `num_leaves` slots and keeps
    `open_resident`'s size argument, which lives in a file this lane does not
    own, exactly where it is. On a miss the slot holds a histogram nobody
    wants, and the next real build zeroes it before accumulating
    (`_zero_slot_desc_kernel`), so a miss costs work and never correctness.

    **`STEP_SUB_SLOT` is written and must not be used by the prebuild.** The
    speculative histogram launch passes `do_sub = 0`, so the word is inert
    there; it is written because the descriptor's contract is that every word
    is written on every publication, and because `_spec_subtract_kernel`
    reads the same word off the *build* descriptor when the prebuild is
    consumed.

    No floating-point arithmetic: gains are compared and record words are
    copied. Same as the commit kernel and for the same reason.
    """
    var tid = Int(thread_idx.x)

    # Nothing committed, so there is no "next step" to serve and, worse, the
    # frontier this would reduce over is the one the *previous* commit left.
    # Written by thread 0 alone and returned from uniformly.
    if step[unsafe_offset=STEP_LIVE][0] == Int32(0):
        if tid == 0:
            spec[unsafe_offset=STEP_LIVE] = Int32(0)
        return

    var n_live = Int(ctr[unsafe_offset=CTR_N_LIVE][0])
    # The next step's phase one, evaluated one step early. A step that would
    # return on the budget commits nothing, so a prebuild for it is pure
    # waste; declining here is what keeps the schedule's tail free.
    if n_live >= Int(num_leaves):
        if tid == 0:
            spec[unsafe_offset=STEP_LIVE] = Int32(0)
        return

    var taken = Int(ctr[unsafe_offset=CTR_PICK][0])
    var appended = n_live - 1

    var my_gain = Float32(0.0)
    var my_slot = NO_PICK
    var s = tid
    while s < n_live:
        if s != taken and s != appended:
            var fo = s * FRONT_WORDS
            var rec = Int(front[unsafe_offset = fo + FRONT_RECORD][0])
            var flags = rec_i[unsafe_offset = rec * SPLIT_IWORDS + IREC_FLAGS][
                0
            ]
            var admissible = (flags & Int32(FLAG_FOUND)) != Int32(0)
            var depth = front[unsafe_offset = fo + FRONT_DEPTH][0]
            var rows = front[unsafe_offset = fo + FRONT_ROW_COUNT][0]
            if max_depth > Int32(0) and depth >= max_depth:
                admissible = False
            if rows < Int32(2) * min_data_in_leaf or rows < Int32(2):
                admissible = False
            if admissible:
                var gain = rec_f[
                    unsafe_offset = rec * SPLIT_FWORDS + FREC_GAIN
                ][0]
                if gain > my_gain:
                    my_gain = gain
                    my_slot = Int32(s)
        s += PICK_THREADS

    var top = block.max[block_size=PICK_THREADS](my_gain)
    var mine = NO_PICK
    if my_gain == top:
        mine = my_slot
    var best = block.min[block_size=PICK_THREADS](mine)

    if tid != 0:
        return

    if top <= Float32(0.0):
        spec[unsafe_offset=STEP_LIVE] = Int32(0)
        return

    var slot = Int(best)
    var fo = slot * FRONT_WORDS
    var begin = Int(front[unsafe_offset = fo + FRONT_ROW_BEGIN][0])
    var parent_slot = Int(front[unsafe_offset = fo + FRONT_HIST_SLOT][0])
    var rec = Int(front[unsafe_offset = fo + FRONT_RECORD][0])
    var ri = rec * SPLIT_IWORDS

    var flags = rec_i[unsafe_offset = ri + IREC_FLAGS][0]
    var is_cat = (flags & Int32(FLAG_CATEGORICAL)) != Int32(0)
    var feature = rec_i[unsafe_offset = ri + IREC_FEATURE][0]
    var n_left = rec_i[unsafe_offset = ri + IREC_LEFT_COUNT][0]
    var n_right = rec_i[unsafe_offset = ri + IREC_RIGHT_COUNT][0]

    # The slot the next commit will acquire, by the commit's own rule.
    var built_slot = -1
    for i in range(Int(pool_capacity)):
        if slot_owner[unsafe_offset=i][0] < Int32(0):
            built_slot = i
            break
    if built_slot < 0:
        spec[unsafe_offset=STEP_LIVE] = Int32(0)
        return

    spec[unsafe_offset=STEP_LIVE] = Int32(1)
    spec[unsafe_offset=STEP_ROW_BEGIN] = Int32(begin)
    spec[unsafe_offset=STEP_ROW_COUNT] = n_left + n_right
    spec[unsafe_offset=STEP_FEATURE] = feature
    spec[unsafe_offset=STEP_IS_CAT] = Int32(1) if is_cat else Int32(0)
    if is_cat:
        spec[unsafe_offset=STEP_THRESHOLD] = Int32(-1)
        spec[unsafe_offset=STEP_MISSING_BIN] = Int32(-1)
        spec[unsafe_offset=STEP_DEFAULT_LEFT] = Int32(0)
    else:
        spec[unsafe_offset=STEP_THRESHOLD] = rec_i[
            unsafe_offset = ri + IREC_BIN
        ][0]
        spec[unsafe_offset=STEP_MISSING_BIN] = missing[
            unsafe_offset = Int(feature)
        ][0]
        var sdl = Int32(0)
        if (flags & Int32(FLAG_DEFAULT_LEFT)) != Int32(0):
            sdl = Int32(1)
        spec[unsafe_offset=STEP_DEFAULT_LEFT] = sdl
    for w in range(CAT_WORDS):
        spec[unsafe_offset = STEP_CAT0 + w] = rec_i[
            unsafe_offset = ri + IREC_CAT0 + w
        ][0]
    # `subtraction_builds_left`, the same integer comparison the commit makes,
    # so the child this accumulates is the child that commit would accumulate.
    var build_left = n_left <= n_right
    if build_left:
        spec[unsafe_offset=STEP_BUILT_BEGIN] = Int32(begin)
        spec[unsafe_offset=STEP_BUILT_COUNT] = n_left
    else:
        spec[unsafe_offset=STEP_BUILT_BEGIN] = Int32(begin) + n_left
        spec[unsafe_offset=STEP_BUILT_COUNT] = n_right
    spec[unsafe_offset=STEP_BUILT_SLOT] = Int32(built_slot)
    spec[unsafe_offset=STEP_SUB_SLOT] = Int32(parent_slot)
    var left_slot = parent_slot
    var right_slot = built_slot
    if build_left:
        left_slot = built_slot
        right_slot = parent_slot
    spec[unsafe_offset=STEP_LEFT_SLOT] = Int32(left_slot)
    spec[unsafe_offset=STEP_RIGHT_SLOT] = Int32(right_slot)
    spec[unsafe_offset=STEP_LEFT_REC] = Int32(rec)
    spec[unsafe_offset=STEP_RIGHT_REC] = Int32(n_live)
    stats[unsafe_offset=SPEC_STAT_BUILDS] = (
        stats[unsafe_offset=SPEC_STAT_BUILDS][0] + Int32(1)
    )


def _reset_tables_kernel(
    front: MutPointer[Int32, MutAnyOrigin],
    node_i: MutPointer[Int32, MutAnyOrigin],
    node_f: MutPointer[Float32, MutAnyOrigin],
    ctr: MutPointer[Int32, MutAnyOrigin],
    slot_owner: MutPointer[Int32, MutAnyOrigin],
    n_active: Int32,
    root_slot: Int32,
    root_value: Float32,
    leaf_capacity: Int32,
    node_capacity: Int32,
    pool_capacity: Int32,
):
    """Write the one-leaf frontier of a fresh tree into all five tables.

    The device half of `begin_tree`, and the reason it exists is a transfer
    count rather than a compute one. Every byte the host used to stage for
    this reset is a constant or a function of three scalars, and on Metal an
    `enqueue_copy` is a synchronous full-queue drain in both directions
    whatever it carries (**measured** by disassembly of the shipped runtime,
    `docs/GPU_PORTABILITY.md` section 6.1), so five staged tables were five
    drains to move a few kilobytes that the device could have written from
    three numbers. This kernel is those three numbers arriving as launch
    arguments instead. It is the same move `_seed_root_value_kernel` and
    `_stage_child_search_kernel` already make for `TN_VALUE` and for
    `NODE_HIST_BASE`, applied to the whole reset.

    **What that is worth is a hazard count and not a time.** Section 6.1.1
    withdrew, on 2026-08-16, the cost model that turned the drain into a
    price: removing thirteen copies per tree from this plane, of which these
    five were a part, **measured** 0.016 seconds at 1,000,000 x 50 against a
    registered prediction of 0.64, which is a null under M0
    (`bench/results/session3_2026-08-16/RESULTS.md`). Draining a queue that
    holds nothing costs nothing, and nothing was queued behind these five.
    What five copies fewer per tree does buy is real and is not a stopwatch
    number: five fewer staging buffers whose lifetime has to be argued, five
    fewer places a stale byte can hide, and five fewer copies to rethink on a
    backend where a copy is genuinely asynchronous. The kernel is correct and
    is the right shape; it is not a speed result and should not be cited as
    one.

    **Every word has exactly one writer and no barrier is used.** That is the
    property that lets this run on a grid of any width, and it is worth
    stating because the obvious form of this kernel does not have it: zero
    the tables with every thread, barrier, then let thread 0 write the
    handful of nonzero words. A barrier synchronizes one threadgroup and not
    a grid, so that form is only correct on a single block, and a single
    block is a needless constraint on a kernel whose whole job is stores.
    Instead each strided loop below decides the *final* value of the word it
    owns, including the nonzero ones, so a thread never overwrites another
    thread's answer and the grid needs no ordering at all. The counter block
    is written whole by the one thread that owns it, since no loop covers it.

    The fields, in the order `begin_tree`'s host path writes them, because
    the two must agree word for word:

    - `front`: zeros everywhere, then slot 0's row is the root, which is node
      0 at depth 0 owning `[0, n_active)` out of pool slot `root_slot` and
      reading record 0. Only two of its six words are nonzero, and the other
      four are the zero the loop already writes, so they are named in
      comments rather than stored twice.
    - `node_i`: every node as `Tree._add_node` leaves one, which is all words
      zero except the five `-1` sentinels at `TN_FEATURE`, `TN_THRESHOLD`,
      `TN_LEFT`, `TN_RIGHT` and `TN_MISSING_BIN`. Node 0 additionally carries
      `TN_COUNT = n_active`.
    - `node_f`: zeroed, with node 0's `TN_VALUE` set to `root_value`.
    - `ctr`: one live leaf, one node, no pick, running, no commits.
    - `slot_owner`: `-1` everywhere except `root_slot`, which node 0 holds.

    `root_value` is carried through rather than dropped, and the honest note
    is that **every caller in this repository passes zero**: the resident
    plane leaves it at its default and then overwrites node 0's value with
    `_seed_root_value_kernel` once the root's search record exists, and the
    test file passes an explicit `Float32(0.0)`. So the argument is redundant
    today in the sense that removing it would change no in-tree behavior. It
    stays because `begin_tree` takes it, and the host arm and the device arm
    have to be word-for-word identical for arguments a caller *may* pass, not
    merely for the ones callers happen to pass now. It costs one Float32 in
    the launch record and no store beyond the one the zero fill would have
    made anyway.

    No floating-point arithmetic occurs here: `root_value` is stored, and
    every other float written is a literal zero. See the contraction note in
    the module docstring for why that is worth saying out loud.
    """
    var gid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)

    # The frontier. `FRONT_NODE`, `FRONT_ROW_BEGIN`, `FRONT_DEPTH` and
    # `FRONT_RECORD` of slot 0 are all zero for a root, which is what the
    # fill already writes; the record-slot-equals-frontier-slot identity
    # starts at record 0 here and the commit preserves it.
    var i = gid
    var n_front = Int(leaf_capacity) * FRONT_WORDS
    while i < n_front:
        var v = Int32(0)
        if i == FRONT_ROW_COUNT:
            v = n_active
        elif i == FRONT_HIST_SLOT:
            v = root_slot
        front[unsafe_offset=i] = v
        i += stride

    # The node table, one thread per node so that the sixteen words of a row
    # are written by the thread that owns the row.
    var n = gid
    while n < Int(node_capacity):
        var o = n * TN_IWORDS
        for w in range(TN_IWORDS):
            node_i[unsafe_offset = o + w] = Int32(0)
        node_i[unsafe_offset = o + TN_FEATURE] = Int32(-1)
        node_i[unsafe_offset = o + TN_THRESHOLD] = Int32(-1)
        node_i[unsafe_offset = o + TN_LEFT] = Int32(-1)
        node_i[unsafe_offset = o + TN_RIGHT] = Int32(-1)
        node_i[unsafe_offset = o + TN_MISSING_BIN] = Int32(-1)
        if n == 0:
            node_i[unsafe_offset = o + TN_COUNT] = n_active
        n += stride

    # The float plane. `TN_VALUE` is word 0 of node 0's row and therefore
    # index 0 of the flat plane, which is the one word that is not zero.
    var k = gid
    var n_nodef = Int(node_capacity) * TN_FWORDS
    while k < n_nodef:
        var fv = Float32(0.0)
        if k == TN_VALUE:
            fv = root_value
        node_f[unsafe_offset=k] = fv
        k += stride

    # The slot pool: free is -1, and the root's slot is owned by node 0.
    var s = gid
    while s < Int(pool_capacity):
        var owner = Int32(-1)
        if s == Int(root_slot):
            owner = Int32(0)
        slot_owner[unsafe_offset=s] = owner
        s += stride

    # The counters, whole, on the one thread that owns them. No loop above
    # covers this buffer, so there is no other writer to race with.
    if gid == 0:
        for w in range(CTR_WORDS):
            ctr[unsafe_offset=w] = Int32(0)
        ctr[unsafe_offset=CTR_N_LIVE] = Int32(1)
        ctr[unsafe_offset=CTR_NEXT_NODE] = Int32(1)
        ctr[unsafe_offset=CTR_PICK] = Int32(-1)
        ctr[unsafe_offset=CTR_PICK_NODE] = Int32(-1)
        ctr[unsafe_offset=CTR_STATUS] = Int32(TREE_RUNNING)
        # `CTR_COMMITS` is zero, which the fill above already wrote.


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


def _pack_tables_kernel(
    pack: MutPointer[Int32, MutAnyOrigin],
    front: MutPointer[Int32, MutAnyOrigin],
    node_i: MutPointer[Int32, MutAnyOrigin],
    node_f: MutPointer[Float32, MutAnyOrigin],
    ctr: MutPointer[Int32, MutAnyOrigin],
    slot_owner: MutPointer[Int32, MutAnyOrigin],
    order: MutPointer[Int32, MutAnyOrigin],
    off_node_i: Int32,
    off_node_f: Int32,
    off_ctr: Int32,
    off_slot: Int32,
    off_order: Int32,
    n_words: Int32,
):
    """Gather all six downloadable tables into one contiguous Int32 buffer.

    The download is the one genuine round trip a device-owned tree makes, and
    that count is unchanged by this kernel: it was one round trip when it was
    six copies and it is one round trip now. What changed is the copy count.
    `download` issued one `enqueue_copy` per table, and on Metal each of those
    is a synchronous full-queue drain in both directions whatever it carries
    (**measured** by disassembly, `docs/GPU_PORTABILITY.md` section 6.1). Six
    drains to bring home a few kilobytes that are already in device memory
    next to each other is a layout problem, and this kernel is the layout: the
    device concatenates, the host copies once and takes the buffer apart.

    **Six to one is not a time saving and was never measured as one.** Under
    section 6.1.1, withdrawn 2026-08-16, a copy count predicts portability
    risk and ordering hazards; a round-trip count predicts time, and the
    round-trip count here did not move. Five of these six drains found a queue
    that the sixth was going to drain anyway, and draining a queue that holds
    nothing costs nothing. What this kernel earns is five fewer ordering
    points, five fewer staging buffers to keep alive, and one arrival instead
    of six for the decode to be wrong about. Those are the arguments it should
    be defended on.

    **The float plane travels as its bits.** `node_f` is Float32 and the
    packed buffer is Int32, so its words are moved with `std.memory.bitcast`,
    which is a reinterpretation and not a conversion: no value is rounded,
    widened, or normalized, and the host reverses it with the inverse
    `bitcast` before the ordinary decode reads it. That is the one place in
    this module where a float and an integer share a buffer, and it is
    deliberately confined to a transport: nothing between the two bitcasts
    reads the words as either type. The alternative that keeps the planes
    apart is two copies rather than one, which is one more ordering point and
    one more buffer to keep alive for no gain, since a bitcast pair cannot
    change a bit. That is a hazard argument, not a timing one; nothing here
    claims a second copy would cost measurable time.

    Every offset is a launch argument rather than a formula repeated here,
    because the host has to take the buffer apart at exactly the boundaries
    the device wrote it at, and two prefix sums in two languages is the kind
    of duplicate that agrees until someone adds a table. `DeviceTreeTables`
    computes them once in its constructor and both sides read those numbers.

    The region lengths are the differences between consecutive offsets, so
    `n_words` is the total and the last region runs to it.
    """
    var gid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)

    var i = gid
    while i < Int(off_node_i):
        pack[unsafe_offset=i] = front[unsafe_offset=i][0]
        i += stride

    var n_node_i = Int(off_node_f) - Int(off_node_i)
    var j = gid
    while j < n_node_i:
        pack[unsafe_offset = Int(off_node_i) + j] = node_i[unsafe_offset=j][0]
        j += stride

    var n_node_f = Int(off_ctr) - Int(off_node_f)
    var k = gid
    while k < n_node_f:
        pack[unsafe_offset = Int(off_node_f) + k] = bitcast[DType.int32, 1](
            node_f[unsafe_offset=k][0]
        )
        k += stride

    var n_ctr = Int(off_slot) - Int(off_ctr)
    var c = gid
    while c < n_ctr:
        pack[unsafe_offset = Int(off_ctr) + c] = ctr[unsafe_offset=c][0]
        c += stride

    var n_slot = Int(off_order) - Int(off_slot)
    var s = gid
    while s < n_slot:
        pack[unsafe_offset = Int(off_slot) + s] = slot_owner[
            unsafe_offset=s
        ][0]
        s += stride

    var n_order = Int(n_words) - Int(off_order)
    var o = gid
    while o < n_order:
        pack[unsafe_offset = Int(off_order) + o] = order[unsafe_offset=o][0]
        o += stride


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

    Nothing here reads an environment variable to decide **whether to work**.
    `tree_resident_requested` is the gate and it is the caller's to consult; a
    struct that silently did nothing when a variable was unset would be much
    harder to test than one that always works and is simply never
    constructed. Two variables do choose between arms that produce identical
    tables, `MOJOTREES_GPU_TABLE_RESET` and `MOJOTREES_GPU_PACKED_DOWNLOAD`,
    and they are read once in the constructor to seed a field that
    `set_reset_on_device` and `set_packed_download` then override. That is
    the shape `GpuActiveRows.set_constant_hessian` uses and it is here for the
    same reason it is there: the in-process setter is what a benchmark needs,
    because only interleaved arms compare on this machine, and the variable is
    what a caller who cannot reach the setter needs. Nothing between this
    struct and a benchmark exposes it today: it is constructed inside
    `GpuHistogramBuilder.open_resident_tables`, which is reached from the
    trainer, so without the variable the other arm would require an edit to a
    module this lane does not own.
    """

    var ctx: DeviceContext
    var leaf_capacity: Int
    var node_capacity: Int
    var pool_capacity: Int
    var n_features: Int

    var reset_on_device: Bool
    """Which arm `begin_tree` takes. See `set_reset_on_device`."""

    var packed_download: Bool
    """Which arm `download` takes. See `set_packed_download`."""

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

    # The same idea for the batched-histogram plan: two item rows of
    # `gpu_leaf_batching`'s layout, so that the two `enqueue_step` overloads
    # -- neither of which writes a plan -- still have a legal pointer to hand
    # the kernel. `enqueue_step_with_plan` is the overload that aims the
    # kernel at a real one, which on a wired path is `GpuLeafBatcher.items_dev`
    # and not this. Never read by any other kernel.
    var plan_scratch: DeviceBuffer[DType.int32]

    # Pinned staging, so a reset and a download are ordinary one-way copies
    # rather than `map_to_host` mappings, each of which moves the buffer in
    # both directions. The same choice `GpuSplitSearcher` makes for its
    # per-node tables and for the same measured reason.
    #
    # One staging buffer per destination rather than one shared buffer. The
    # argument for that used to be that an enqueued copy reads its source
    # asynchronously, so a shared buffer would force a `synchronize` between
    # every pair of copies before the host could refill it, and a
    # `begin_tree` writing five tables would force five drains instead of
    # one. On Metal that argument does not hold: `enqueue_copy` is itself a
    # synchronous full-queue drain in both directions (**measured** by
    # disassembly of the shipped runtime, `docs/GPU_PORTABILITY.md` section
    # 6.1), so five copies are five drains whatever memory they read, and a
    # shared buffer would have added no ordering point that was not already
    # there.
    #
    # The separate buffers stay, for two reasons that survive. They are what
    # makes this correct on a backend where the copy really is asynchronous,
    # and they cost nothing. What does not survive is the expected saving on
    # Metal, and this note deliberately stops short of reinterpreting the
    # M4 measurement it replaces: that measurement was taken, it was real,
    # and nobody has re-taken it knowing that both forms drain five times.
    # Until someone does, treat the separate buffers as free and portable
    # rather than as a wait that was removed. A design that wants fewer
    # drains has to stage fewer times, not stage into more places.
    #
    # And under section 6.1.1, withdrawn 2026-08-16, "fewer drains" is not a
    # request for speed. A drain of a queue holding nothing costs nothing;
    # removing thirteen such copies per tree from this plane **measured**
    # 0.016 seconds at 1,000,000 x 50 against a registered prediction of 0.64
    # (`bench/results/session3_2026-08-16/RESULTS.md`), a null under M0.
    # Staging fewer times is worth doing because it removes ordering points,
    # buffer lifetimes and places a stale byte can hide, which is exactly what
    # the paragraph above already argues these separate buffers on. It is not
    # worth doing for the clock. Rule 5 of 6.1.1 says the same thing from the
    # other side: never argue a new pinned staging buffer on Metal grounds.
    #
    # The tables-reset lane took that last sentence literally, and these five
    # buffers are consequently **no longer on the default path**. `begin_tree`
    # now writes the reset with a kernel from three scalars and stages
    # nothing; `set_reset_on_device(False)` is what puts these back, and it is
    # the arm a benchmark holds against the kernel. `stage_frontier` still
    # uses `stage_front`, `stage_slot` and `stage_ctr` unconditionally, since
    # it uploads an arbitrary frontier a test invented rather than one a
    # scalar describes, so none of these allocations became dead.
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

    # The packed download: one device buffer holding all six tables end to
    # end, and one pinned host buffer to receive it. The six host buffers
    # above stay, and stay the only thing the decode reads: the packed arm
    # scatters this buffer back into them on the host and then runs exactly
    # the decode the six-copy arm runs, so a snapshot cannot depend on which
    # arm produced it. See `_fetch_packed`.
    var pack_dev: DeviceBuffer[DType.int32]
    var host_pack: HostBuffer[DType.int32]

    # Where each table starts in the packed buffer. Computed once here and
    # handed to the kernel as launch arguments, so the concatenation and the
    # host's disassembly of it read the same numbers rather than two copies
    # of the same prefix sum. `pack_words` is the total and the last region
    # runs to it.
    var pack_off_node_i: Int
    var pack_off_node_f: Int
    var pack_off_ctr: Int
    var pack_off_slot: Int
    var pack_off_order: Int
    var pack_words: Int

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
        # Both arms default to the device form, and each variable can only
        # move it back to the pre-lane form, which is the one direction that
        # is always safe: an unset variable, and a variable set to anything
        # unrecognized, land on the default. Spelled as an equality against
        # "0" for the reason `MOJOTREES_GPU_SPLIT_RESIDENT` is. See
        # `set_reset_on_device` and `set_packed_download`.
        self.reset_on_device = getenv("MOJOTREES_GPU_TABLE_RESET") != "0"
        self.packed_download = getenv("MOJOTREES_GPU_PACKED_DOWNLOAD") != "0"

        # The packed download's layout. One prefix sum, here, over the same
        # six regions in the same order `_pack_tables_kernel` writes them and
        # `_fetch_packed` reads them.
        self.pack_off_node_i = self.leaf_capacity * FRONT_WORDS
        self.pack_off_node_f = (
            self.pack_off_node_i + self.node_capacity * TN_IWORDS
        )
        self.pack_off_ctr = (
            self.pack_off_node_f + self.node_capacity * TN_FWORDS
        )
        self.pack_off_slot = self.pack_off_ctr + CTR_WORDS
        self.pack_off_order = self.pack_off_slot + self.pool_capacity
        self.pack_words = self.pack_off_order + self.leaf_capacity

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
        # Wide enough for a level commit as well as a leaf-wise one. A level of
        # `L` parents fills `2L` item rows and the widest level a depth-6 tree
        # commits is `L = 32`, so this is `OBLIVIOUS_PLAN_ITEMS` and not
        # `PLAN_ITEMS`. Two hundred and fifty-six Int32; the buffer is never
        # read by any other kernel and exists so that the overloads which write
        # no plan still have a legal pointer to hand one.
        self.plan_scratch = self.ctx.enqueue_create_buffer[DType.int32](
            OBLIVIOUS_PLAN_ITEMS * ITEM_WORDS
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
        self.pack_dev = self.ctx.enqueue_create_buffer[DType.int32](
            self.pack_words
        )
        self.host_pack = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.pack_words
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

    def set_reset_on_device(mut self, on: Bool):
        """Whether `begin_tree` resets the five tables with a kernel or with
        five staged uploads.

        On is the device arm added by the tables-reset lane: one launch that
        writes the frontier, both node planes, the counters and the slot pool
        from three scalars. Off is the shape this module shipped before it,
        which stages every word on the host and issues five `enqueue_copy`
        calls. On Metal those five copies are five synchronous full-queue
        drains (**measured** by disassembling the shipped runtime,
        `docs/GPU_PORTABILITY.md` section 6.1), so the arms differ by five
        drains per tree and by nothing else.

        **Five drains is not five waits, and this arm is not a speed knob.**
        An earlier version of this docstring priced each of those drains at
        the 606 microsecond median blocking readback of
        `docs/METAL_TIMELINE.md:550`. That figure is **measured**, it is
        correct, and it is the price of a *round trip*: host code blocking on
        a device answer it needs before it can decide what to enqueue next.
        None of these five is one. Section 6.1.1 records the withdrawal, on
        2026-08-16, of the inference that carried the number across:
        collapsing thirteen copies per tree on this plane, these five among
        them, **measured** 0.016 seconds at 1,000,000 x 50 against a
        registered prediction of 0.64, a null under M0
        (`bench/results/session3_2026-08-16/RESULTS.md`). Draining a queue
        that holds nothing costs nothing.

        So flip this expecting a different hazard surface, not a different
        clock. On removes five staging buffers from the per-tree path, five
        ordering points, and five places a stale byte could hide; it also
        removes real host work that is too small for M0 to see. Off is kept
        because it is the arm an interleaved A/B holds this against, not
        because anyone expects it to lose.

        Reachable at run time rather than through a second build, and that is
        the point rather than a convenience: this machine's device timings
        drift several-fold between time windows, so only interleaved arms
        compare, and an environment-only knob would have forced a two-build
        comparison. It is the same argument `GpuActiveRows.set_row_unroll`
        makes for the same reason.

        **It cannot change a table.** Both arms write the same words to the
        same offsets; `_reset_tables_kernel` enumerates them field by field
        against the staging loops below. What the device arm does not do is
        touch the pinned staging buffers, so a caller that flips this on has
        no staging lifetime to protect and `wait` becomes a courtesy rather
        than a requirement; see `begin_tree`.

        `MOJOTREES_GPU_TABLE_RESET=0` seeds the field to the staging arm at
        construction, for a caller that cannot reach this setter; the setter
        wins over it, because it runs later. The variable exists only because
        nothing between a benchmark and these tables exposes them today, and
        an A/B that reads its arm from the environment can run one arm under
        the other's label, which has happened once in this repository. Prefer
        the setter.

        Takes effect on the next `begin_tree`.
        """
        self.reset_on_device = on

    def set_packed_download(mut self, on: Bool):
        """Whether `download` brings the six tables home in one copy or in
        six.

        On is the device arm added by the tables-reset lane: a kernel
        concatenates the six tables into one Int32 buffer, one
        `enqueue_copy` carries it, and the host scatters it back into the
        same six pinned buffers the other arm copies into. Off is the shape
        this module shipped before it. On Metal the arms differ by five drains
        per tree, for the reason `set_reset_on_device` states, and by one
        device launch and one host scatter of a few thousand words in the
        other direction. **Both arms make the same one round trip**, which is
        the count that predicts time under section 6.1.1, so neither arm is
        expected to measure faster and the collapse was priced at 0.016
        seconds for all thirteen copies it was part of
        (`bench/results/session3_2026-08-16/RESULTS.md`, **measured**, a null
        under M0). What On buys is five fewer ordering points and one arrival
        for the decode to be wrong about instead of six.

        **It cannot change a snapshot.** Both arms end with the same six host
        buffers holding the same words and then fall into the same decode,
        which is the body of `download` below the fetch and is the only
        reader of the table layout on this side. The float plane
        crosses the packed buffer as its bits and is bitcast back before the
        decode sees it, which is a reinterpretation in each direction and so
        is exact; see `_pack_tables_kernel`.

        `MOJOTREES_GPU_PACKED_DOWNLOAD=0` seeds the field to the six-copy arm
        at construction, with the same standing and the same caveat
        `set_reset_on_device` gives its variable.

        Takes effect on the next `download`.
        """
        self.packed_download = on

    def _enqueue_reset(
        mut self, n_active: Int, root_slot: Int, root_value: Float32
    ) raises:
        """Launch the device reset. Enqueues only: no transfer, no wait.

        The grid is sized from the widest of the four strided regions, since
        every loop in the kernel is grid-strided and a thread that finds its
        region already covered simply does not enter it. One block would also
        be correct and is what the reduction kernels in this file use; a
        proportional grid is used instead because this kernel has no
        collective in it, so nothing constrains it to one threadgroup, and
        the tables at a large leaf budget are tens of thousands of words.

        `PICK_THREADS` as the block width for no reason beyond consistency
        with the rest of the file: this kernel has no `block.max` and no
        `block.min`, so the width is not load bearing here.
        """
        var widest = self.leaf_capacity * FRONT_WORDS
        if self.node_capacity > widest:
            widest = self.node_capacity
        if self.node_capacity * TN_FWORDS > widest:
            widest = self.node_capacity * TN_FWORDS
        if self.pool_capacity > widest:
            widest = self.pool_capacity
        var blocks = (widest + PICK_THREADS - 1) // PICK_THREADS
        if blocks < 1:
            blocks = 1
        comptime if not has_accelerator():
            raise Error(
                "the device tree tables need an accelerator; this binary was"
                " built without one"
            )
        else:
            self.ctx.enqueue_function[_reset_tables_kernel](
                self.front_dev.unsafe_ptr(),
                self.node_i_dev.unsafe_ptr(),
                self.node_f_dev.unsafe_ptr(),
                self.ctr_dev.unsafe_ptr(),
                self.slot_dev.unsafe_ptr(),
                Int32(n_active),
                Int32(root_slot),
                root_value,
                Int32(self.leaf_capacity),
                Int32(self.node_capacity),
                Int32(self.pool_capacity),
                grid_dim=blocks,
                block_dim=PICK_THREADS,
            )

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

        **Two arms, chosen by `set_reset_on_device`, writing the same words.**
        The default arm is one kernel launch: every byte this reset writes is
        a constant or a function of `n_active`, `root_slot` and `root_value`,
        so the three scalars go across as launch arguments and the device
        fills the tables. It transfers nothing and therefore waits for
        nothing. `_reset_tables_kernel` enumerates the fields against the
        staging loops below, which are the other arm.

        The other arm, `set_reset_on_device(False)`, is what this module
        shipped before: everything is staged into pinned memory and copied
        rather than written through a `map_to_host` mapping, because a
        mapping blocks until the device is idle and a copy does not. Each of
        the five tables has its own staging buffer, so all five copies are
        enqueued before anything waits. On Metal that is five drains and not
        one, because `enqueue_copy` there is a synchronous full-queue drain in
        both directions (**measured**, `docs/GPU_PORTABILITY.md` section 6.1);
        removing those five is what the device arm is for. **Five ordering
        points, not five waits**: none of the five blocks on a device answer
        the host needs, and under section 6.1.1 that is the difference between
        a copy count and a time. See `set_reset_on_device` for the measurement
        that forced the distinction.

        The two argument checks above stay on the host in both arms. A kernel
        cannot raise, so a negative row count or an out-of-range root slot has
        to be refused before the launch or not at all, and "not at all" would
        mean a frontier pointing at a pool slot that does not exist.

        `wait` means the same thing in both arms -- the reset has landed on
        the device when this returns -- but it protects something only in the
        staging arm, and a device-owned growth loop passes False to remove it.
        In the staging arm that is safe under exactly one condition, which is
        worth stating rather than assuming: the staging buffers must not be
        refilled before the copies reading them have finished, and the only
        thing that refills them is the *next* `begin_tree`. A caller that
        downloads the tree at the end of every tree therefore already has a
        synchronization between any two `begin_tree` calls, and the wait here
        is redundant for it. A caller that does not must leave `wait` alone.
        In the device arm there is no staging buffer and therefore no
        lifetime to protect: the launch is ordered against everything else on
        the same in-order queue, so `wait=False` is unconditionally safe
        there. It defaults to True so every existing caller, the test file
        included, keeps the behavior it was written against.
        """
        if n_active < 0:
            raise Error("active row count must be nonnegative")
        if root_slot < 0 or root_slot >= self.pool_capacity:
            raise Error("root histogram slot out of range")

        if self.reset_on_device:
            self._enqueue_reset(n_active, root_slot, root_value)
            if wait:
                self.ctx.synchronize()
            return

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
                self.plan_scratch.unsafe_ptr(),
                Int32(0),
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
                self.plan_scratch.unsafe_ptr(),
                Int32(0),
                Int32(num_leaves),
                Int32(max_depth),
                Int32(min_data_in_leaf),
                Int32(self.pool_capacity),
                Int32(self.leaf_capacity),
                Int32(self.node_capacity),
                grid_dim=1,
                block_dim=PICK_THREADS,
            )

    def enqueue_step_with_plan[
        step_origin: MutOrigin,
        plan_origin: MutOrigin, //
    ](
        mut self,
        mut rec_i: DeviceBuffer[DType.int32],
        mut rec_f: DeviceBuffer[DType.float32],
        step: MutPointer[Int32, step_origin],
        plan: MutPointer[Int32, plan_origin],
        num_leaves: Int,
        max_depth: Int,
        min_data_in_leaf: Int,
    ) raises:
        """`enqueue_step`, additionally writing a batched-histogram plan into
        `plan`.

        **One launch, exactly as before: the plan is three Int32 per child
        written by a kernel that was already running.**

        `plan` is `GpuLeafBatcher.items_dev`, whose geometry words
        `stage_device_plan` has already fixed for this tree. This kernel writes
        only `ITEM_BEGIN`, `ITEM_COUNT` and `ITEM_OUT`, and it writes them on
        every execution -- `ITEM_DEAD` on the three exits that commit nothing,
        for the same reason those exits write `STEP_LIVE = 0`.

        This is the whole of what `gpu_resident_round.oblivious_launch_census`
        needs to be true rather than assumed. The census's per-level figure of
        two launches for the whole level's children is `enqueue_batch`'s
        figure, and the one thing that kept `enqueue_batch` out of a
        device-owned tree was that its item table came from the host, which
        means from a read-back of this commit. It no longer does.

        **A caller uses this or `enqueue_desc_child`, not both**, and both
        produce the same histograms. See the plan-writing block in
        `_pick_and_commit_kernel` for why the two shapes are bit-identical and
        for the one arm -- the K=1 speculation -- that must not use this one.
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
                plan,
                Int32(1),
                Int32(num_leaves),
                Int32(max_depth),
                Int32(min_data_in_leaf),
                Int32(self.pool_capacity),
                Int32(self.leaf_capacity),
                Int32(self.node_capacity),
                grid_dim=1,
                block_dim=PICK_THREADS,
            )

    def enqueue_level[
        step_origin: MutOrigin,
        plan_origin: MutOrigin,
        hist_origin: MutOrigin, //
    ](
        mut self,
        mut rec_i: DeviceBuffer[DType.int32],
        mut rec_f: DeviceBuffer[DType.float32],
        mut fparams: DeviceBuffer[DType.float32],
        hist: MutPointer[Int32, hist_origin],
        step: MutPointer[Int32, step_origin],
        plan: MutPointer[Int32, plan_origin],
        write_plan: Bool,
        level_record: Int,
        n_bins: Int,
        hist_size: Int,
        level_depth: Int,
        max_depth: Int,
        plan_items: Int,
        gain_form: Int,
    ) raises:
        """Commit one oblivious level: one launch, one block.

        The level's split is already decided -- it is the single record at
        `level_record` that `gpu_split_search`'s cross-leaf scan and the ordinary
        cross-feature reduction wrote one launch earlier -- so this applies it to
        every leaf of the level and publishes everything the rest of the level's
        schedule reads: the step descriptor its partition routes by, the plan its
        batched build accumulates from, and the frontier its next search covers.

        `hist` is the resident histogram pool (`GpuLeafBatcher.out_dev`), read
        here and only here inside this file. A level commit cannot get its
        per-leaf row counts and child values off the record the way a leaf-wise
        commit does, because the record's are the level's aggregates; see
        `_commit_level_kernel`.

        `plan_items` is the item count the batcher was staged for, which is the
        width every level of this tree kills or fills. Handing in a narrower
        number than the batcher's would leave items live that the batch still
        launches over.

        **A caller uses this or `enqueue_step`/`enqueue_step_with_plan`, never
        both on one tree.** They are two growth policies, not two shapes of one.
        """
        if level_record < 0:
            raise Error("the level record index must be nonnegative")
        if n_bins < 1 or hist_size < n_bins:
            raise Error("a level commit needs a positive histogram geometry")
        if max_depth < 1:
            raise Error("an oblivious tree needs a positive max_depth")
        if level_depth < 0:
            raise Error("a level depth must be nonnegative")
        if write_plan and (
            plan_items < 2 or plan_items > OBLIVIOUS_PLAN_ITEMS
        ):
            raise Error(
                "a level plan holds between two items and"
                " OBLIVIOUS_PLAN_ITEMS"
            )
        comptime if not has_accelerator():
            raise Error(
                "the device tree tables need an accelerator; this binary was"
                " built without one"
            )
        else:
            self.ctx.enqueue_function[_commit_level_kernel](
                self.front_dev.unsafe_ptr(),
                self.node_i_dev.unsafe_ptr(),
                self.node_f_dev.unsafe_ptr(),
                self.ctr_dev.unsafe_ptr(),
                self.slot_dev.unsafe_ptr(),
                self.missing_dev.unsafe_ptr(),
                rec_i.unsafe_ptr(),
                rec_f.unsafe_ptr(),
                hist,
                fparams.unsafe_ptr(),
                step,
                self.order_dev.unsafe_ptr(),
                plan,
                Int32(1) if write_plan else Int32(0),
                Int32(level_record),
                Int32(n_bins),
                Int32(hist_size),
                Int32(level_depth),
                Int32(max_depth),
                Int32(plan_items),
                Int32(self.pool_capacity),
                Int32(self.leaf_capacity),
                Int32(self.node_capacity),
                Int32(gain_form),
                grid_dim=1,
                block_dim=PICK_THREADS,
            )

    def enqueue_stage_level_search(
        mut self,
        mut node_tbl: DeviceBuffer[DType.int32],
        slot_cells: Int,
        leaf_base: Int,
        max_leaves: Int,
    ) raises:
        """Point the level's leaf search records at the level's slots.

        `enqueue_stage_child_search` for a whole level. `node_tbl` is
        `GpuSplitSearcher.node_dev` and `slot_cells` is
        `3 * n_features * n_bins`; `leaf_base` is the first record the level's
        leaves occupy and `max_leaves` is the widest level this tree will reach,
        so the records a stopped tree's search still covers are written too. See
        `_stage_level_search_kernel`.
        """
        if slot_cells < 1:
            raise Error("the pool slot stride must be positive")
        if leaf_base < 0:
            raise Error("the leaf record base must be nonnegative")
        if max_leaves < 1 or max_leaves > OBLIVIOUS_LEVEL_LEAVES:
            raise Error(
                "a level holds between one leaf and OBLIVIOUS_LEVEL_LEAVES"
            )
        comptime if not has_accelerator():
            raise Error(
                "the device tree tables need an accelerator; this binary was"
                " built without one"
            )
        else:
            self.ctx.enqueue_function[_stage_level_search_kernel](
                node_tbl.unsafe_ptr(),
                self.front_dev.unsafe_ptr(),
                self.ctr_dev.unsafe_ptr(),
                Int32(slot_cells),
                Int32(leaf_base),
                Int32(max_leaves),
                grid_dim=1,
                block_dim=1,
            )

    def enqueue_runner_up(
        mut self,
        mut rec_i: DeviceBuffer[DType.int32],
        mut rec_f: DeviceBuffer[DType.float32],
        mut step: DeviceBuffer[DType.int32],
        mut spec: DeviceBuffer[DType.int32],
        mut stats: DeviceBuffer[DType.int32],
        num_leaves: Int,
        max_depth: Int,
        min_data_in_leaf: Int,
    ) raises:
        """Publish the K=1 speculative descriptor for the step after this one.
        Writes no table, transfers nothing, synchronizes nothing.

        One launch, one block, `PICK_THREADS` threads -- the same shape as
        `enqueue_step`, because it is the same reduction over the same
        frontier. See `_pick_runner_up_kernel`.

        The three descriptors cross as `DeviceBuffer`s rather than as
        pointers, which is not the convention `enqueue_step` uses and is
        deliberate. Its caller can take a pointer because the buffer belongs
        to a different object; this one's caller holds `step` and `spec` on
        one field of the builder and reaches these tables through another,
        and Mojo will not let a pointer derived from one field cross a call
        that mutably borrows a sibling. A `DeviceBuffer` is a copyable handle,
        so a caller hands in copies and no borrow spans the call. It moves no
        bytes.

        `stats` is the speculation's two-word counter; this kernel increments
        `SPEC_STAT_BUILDS` on every publication and touches nothing else in
        it. The counter that matters -- consumption -- is incremented
        somewhere else entirely, by `gpu_active_rows._spec_consume_kernel`, on
        the branch that suppresses the real build. Splitting the two is what
        makes the pair evidence: a mechanism that published perfectly and
        consumed nothing would show it.
        """
        self._check_step_args(num_leaves, min_data_in_leaf)
        comptime if not has_accelerator():
            raise Error(
                "the device tree tables need an accelerator; this binary was"
                " built without one"
            )
        else:
            self.ctx.enqueue_function[_pick_runner_up_kernel](
                self.front_dev.unsafe_ptr(),
                self.ctr_dev.unsafe_ptr(),
                self.slot_dev.unsafe_ptr(),
                self.missing_dev.unsafe_ptr(),
                rec_i.unsafe_ptr(),
                rec_f.unsafe_ptr(),
                step.unsafe_ptr(),
                spec.unsafe_ptr(),
                stats.unsafe_ptr(),
                Int32(num_leaves),
                Int32(max_depth),
                Int32(min_data_in_leaf),
                Int32(self.pool_capacity),
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

    def _fetch_six(mut self) raises:
        """Bring the six tables home in six copies, the shape this module
        shipped with. One `enqueue_copy` per table into its own pinned
        buffer, which on Metal is six full-queue drains and not one. Six
        drains, one round trip: the round-trip count is the same as the packed
        arm's and it is the count that predicts time, so this arm is slower
        only in ordering points and buffer lifetimes. See
        `set_packed_download` and `docs/GPU_PORTABILITY.md` section 6.1.1."""
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

    def _fetch_packed(mut self) raises:
        """Concatenate the six tables on the device, copy once, and scatter
        the result back into the same six pinned buffers `_fetch_six` fills.

        The scatter is what keeps the two arms honest. It would be faster to
        decode straight out of the packed buffer, and it would also be a
        second reader of the table layout: two decoders that agree until
        someone adds a word to a row. Instead the packed arm's only extra
        work on the host is a word-for-word copy into the buffers the one
        decoder already reads, which is a few thousand stores against a drain
        whose order is a **measured** hundreds of microseconds
        (`docs/METAL_TIMELINE.md:550`). Neither side of that comparison was
        measured here. The float plane is bitcast back in this loop, which is
        the exact inverse of the bitcast the kernel made and so restores the
        same bits.

        The launch is enqueued before the copy and on the same in-order
        queue, so the copy reads a buffer the kernel has finished writing
        with no fence of its own.
        """
        var blocks = (self.pack_words + PICK_THREADS - 1) // PICK_THREADS
        if blocks < 1:
            blocks = 1
        comptime if not has_accelerator():
            raise Error(
                "the device tree tables need an accelerator; this binary was"
                " built without one"
            )
        else:
            self.ctx.enqueue_function[_pack_tables_kernel](
                self.pack_dev.unsafe_ptr(),
                self.front_dev.unsafe_ptr(),
                self.node_i_dev.unsafe_ptr(),
                self.node_f_dev.unsafe_ptr(),
                self.ctr_dev.unsafe_ptr(),
                self.slot_dev.unsafe_ptr(),
                self.order_dev.unsafe_ptr(),
                Int32(self.pack_off_node_i),
                Int32(self.pack_off_node_f),
                Int32(self.pack_off_ctr),
                Int32(self.pack_off_slot),
                Int32(self.pack_off_order),
                Int32(self.pack_words),
                grid_dim=blocks,
                block_dim=PICK_THREADS,
            )
            self.ctx.enqueue_copy(
                dst_ptr=self.host_pack.unsafe_ptr(), src_buf=self.pack_dev
            )
            self.ctx.synchronize()

            var p = self.host_pack.unsafe_ptr()
            var f = self.host_front.unsafe_ptr()
            for i in range(self.pack_off_node_i):
                f.unsafe_store(i, p.unsafe_load(i))
            var ni = self.host_node_i.unsafe_ptr()
            for i in range(self.node_capacity * TN_IWORDS):
                ni.unsafe_store(i, p.unsafe_load(self.pack_off_node_i + i))
            var nf = self.host_node_f.unsafe_ptr()
            for i in range(self.node_capacity * TN_FWORDS):
                nf.unsafe_store(
                    i,
                    bitcast[DType.float32, 1](
                        p.unsafe_load(self.pack_off_node_f + i)
                    ),
                )
            var c = self.host_ctr.unsafe_ptr()
            for i in range(CTR_WORDS):
                c.unsafe_store(i, p.unsafe_load(self.pack_off_ctr + i))
            var s = self.host_slot.unsafe_ptr()
            for i in range(self.pool_capacity):
                s.unsafe_store(i, p.unsafe_load(self.pack_off_slot + i))
            var o = self.host_order.unsafe_ptr()
            for i in range(self.leaf_capacity):
                o.unsafe_store(i, p.unsafe_load(self.pack_off_order + i))

    def download(mut self) raises -> TreeTablesSnapshot:
        """Copy every table home and decode it. One host round trip.

        A wired caller would do this once per tree. A test does it once per
        step, which is the opposite of the point and is exactly why the
        measurement in the module docstring cannot be taken from a test run.

        The round trip is one in both arms and is not what
        `set_packed_download` moves; what it moves is the number of *copies*
        that round trip is made of, which on Metal is the number of drains,
        since a copy there drains the queue. Six became one.

        **Round trips are the count that predicts time and it did not move
        here.** Section 6.1.1, withdrawn 2026-08-16, took back the reading
        that made each of those six copies a host wait worth the
        per-synchronization constant: five of the six drained a queue that the
        sixth would have drained anyway, and draining a queue that holds
        nothing costs nothing. The collapse this arm is part of **measured**
        0.016 seconds at 1,000,000 x 50 against a registered prediction of
        0.64, a null under M0. It is kept for the copy count, which predicts
        portability risk and where a stale byte can hide, and not for the
        clock.

        **The `synchronize` afterwards is LOAD-BEARING, and the reason given
        here until 2026-08-16 was false.** It said the wait "waits on nothing
        on Metal" and stayed only for a hypothetical backend where a copy is
        really asynchronous. On Metal a copy into a **pinned `HostBuffer` --
        which is what all six of these destinations are -- IS asynchronous**,
        measured by execution: a pinned copy read without an intervening
        `synchronize` returned 64 of 64 stale words on 4 of 4 attempts behind
        a slow kernel. Only a copy into an arbitrary host pointer drains.

        So this wait is the only thing making the six-copy arm correct.
        Deleting it on the old reasoning would produce a readback that is
        right under a fast kernel and wrong under a slow one -- green on a
        small fixture, silently wrong on a real fit, which is the same shape
        as the stale-row-range bug that once cost this plane every tree after
        the first.

        Everything below the fetch is common to both arms deliberately: the
        packed arm ends by scattering into the same six pinned buffers the
        six-copy arm copies into, so the decode has one implementation and a
        snapshot cannot depend on which arm produced it.
        """
        if self.packed_download:
            self._fetch_packed()
        else:
            self._fetch_six()
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
