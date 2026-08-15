"""The device pick-and-commit against the host decision it has to reproduce.

What this file asserts, and what it does not
---------------------------------------------
`gpu_tree_tables._pick_and_commit_kernel` exists so that a device can choose
the next leaf and commit its split without a host round trip. It is not wired
into anything, so there is no fit to compare and no timing to take. The only
claim available is an equivalence, and this file is the whole of it.

The reference is not a re-implementation. Every decision the reference makes
is made by calling the shipping code that makes it today:

    `gpu_split_search.decode_record`        reads the record words
    `train_gpu._apply_shape_rules`          clears `found` under max_depth
                                            and the parent row floor
    `growth_policy.LeafCandidate`           describes a frontier leaf
    `growth_policy.GrowthSchedule`          picks the leaf, leaf-wise
    `GpuSplitRecord.to_split_info`          turns the winner into a split
    `gpu_frontier.subtraction_builds_left`  chooses the built child
    `gpu_leaf_batching.HistogramSlotPool`   assigns the histogram slot

The loop scaffolding around those calls is the body of
`train_gpu._device_search_resident` and `_enqueue_resident_split`, written out
here over the device tables' own row types. If the reference and the kernel
ever disagree, one of those seven is where the disagreement is.

Comparison is field by field with no tolerance. Integers compare as integers
and the two Float32 fields compare as bit patterns, through
`DeviceLeafRow.same_as` and `DeviceNodeRow.same_as`, because a leaf value one
unit in the last place away is a different model once a boosting round has
compounded it; `docs/NUMERICS.md` records two occasions on which exactly that
happened and passed every tolerance-based test in the suite.

The tie-break gets its own tests, separately from the randomized ones, because
it is the one property a randomized comparison could pass by luck. The host
rule is a strict comparison over an ascending scan, so the winner among equal
gains is the lowest-indexed leaf; the kernel reproduces it under a parallel
reduction by carrying the slot alongside the gain and taking a `block.min`
over the tied slots. Ties are placed deliberately at slot pairs that fall on
one thread and at pairs that fall on different threads, because those are two
different code paths and only the second one involves a collective at all.

Determinism is asserted directly: the same tables, the same records, the same
step, run twice, must produce identical snapshots. A reduction that depended
on thread arrival order would fail this and pass everything else.

This file opens a device and is therefore GPU-only, which `tools/run_tests.sh`
derives from the `test_gpu_` prefix. Nothing here runs a fit, builds a
histogram, or measures anything.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from mojotrees.gpu_frontier import subtraction_builds_left
from mojotrees.gpu_leaf_batching import HistogramSlotPool
from mojotrees.gpu_split_search import (
    CAT_WORDS,
    FLAG_CATEGORICAL,
    FLAG_DEFAULT_LEFT,
    FLAG_FOUND,
    FREC_GAIN,
    FREC_LEFT_GRAD,
    FREC_LEFT_HESS,
    FREC_LEFT_VALUE,
    FREC_PARENT_VALUE,
    FREC_RIGHT_GRAD,
    FREC_RIGHT_HESS,
    FREC_RIGHT_VALUE,
    FREC_TOTAL_GRAD,
    FREC_TOTAL_HESS,
    IREC_BIN,
    IREC_CAT0,
    IREC_FEATURE,
    IREC_FLAGS,
    IREC_LEFT_COUNT,
    IREC_ORDINAL,
    IREC_RIGHT_COUNT,
    IREC_TOTAL_COUNT,
    SPLIT_FWORDS,
    SPLIT_IWORDS,
    decode_record,
)
from mojotrees.gpu_tree_tables import (
    CTR_WORDS,
    FRONT_WORDS,
    PICK_THREADS,
    TN_FWORDS,
    TN_IWORDS,
    TREE_BUDGET_SPENT,
    TREE_NO_CANDIDATE,
    TREE_RESIDENT_BYNODE,
    TREE_RESIDENT_DEPTHWISE,
    TREE_RESIDENT_EXTRA,
    TREE_RESIDENT_INTERACTION,
    TREE_RESIDENT_MONOTONE,
    TREE_RESIDENT_OK,
    TREE_RUNNING,
    DeviceLeafRow,
    DeviceNodeRow,
    DeviceTreeTables,
    TreeTablesSnapshot,
    tree_resident_reason_name,
    tree_resident_requested,
    tree_resident_supported,
    tree_status_name,
)
from mojotrees.growth_policy import (
    GROW_DEPTHWISE,
    GrowthSchedule,
    LeafCandidate,
)
from mojotrees.interaction import InteractionConstraints
from mojotrees.monotone import MonotoneConstraints
from mojotrees.train_gpu import _apply_shape_rules
from mojotrees.tree import TreeParams
from mojotrees.tree_parameters_extra import ExtraTreeParams


comptime _N_FEATURES = 8
comptime _N_BINS = 32


# --- A deterministic generator for synthetic split records ----------------
#
# The records the kernel reads are ordinarily written by the split search.
# This file writes them itself, for two reasons. A search would have to be
# fed a histogram, which would make the test about the histogram; and a
# search will not on demand produce an exact tie in Float32 between two named
# leaves, which is the case that matters most here.
#
# Every number below is a small integer or an exact multiple of a power of
# two, so nothing in the generator rounds and every "tie" is a genuine
# bit-for-bit tie rather than two values that happen to print the same.


def _mix(x: UInt64) -> UInt64:
    """splitmix64's finalizer. A bijection with full avalanche, so a record
    derived from a node id is uncorrelated with its neighbors and is still a
    pure function of that id, which is what makes a failing case
    reproducible from the seed alone."""
    var z = x + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _empty_words_i(capacity: Int) -> List[Int32]:
    var out = List[Int32](capacity=capacity * SPLIT_IWORDS)
    out.resize(capacity * SPLIT_IWORDS, Int32(0))
    return out^


def _empty_words_f(capacity: Int) -> List[Float32]:
    var out = List[Float32](capacity=capacity * SPLIT_FWORDS)
    out.resize(capacity * SPLIT_FWORDS, Float32(0.0))
    return out^


def _write_record(
    mut words_i: List[Int32],
    mut words_f: List[Float32],
    record: Int,
    found: Bool,
    feature: Int,
    bin: Int,
    default_left: Bool,
    categorical: Bool,
    gain: Float32,
    n_left: Int,
    n_right: Int,
    left_value: Float32,
    right_value: Float32,
) raises:
    """One record slot, in the layout `decode_record` reads and the kernel
    reads. The statistics fields are filled with recognizable values rather
    than zeros so that a kernel copying the wrong word is visible."""
    var io = record * SPLIT_IWORDS
    var fo = record * SPLIT_FWORDS
    for w in range(SPLIT_IWORDS):
        words_i[io + w] = Int32(0)
    for w in range(SPLIT_FWORDS):
        words_f[fo + w] = Float32(0.0)
    if not found:
        words_i[io + IREC_FEATURE] = Int32(-1)
        words_i[io + IREC_BIN] = Int32(-1)
        words_i[io + IREC_ORDINAL] = Int32(-1)
        return
    var flags = Int32(FLAG_FOUND)
    if default_left:
        flags |= Int32(FLAG_DEFAULT_LEFT)
    if categorical:
        flags |= Int32(FLAG_CATEGORICAL)
    words_i[io + IREC_FEATURE] = Int32(feature)
    words_i[io + IREC_BIN] = Int32(-1) if categorical else Int32(bin)
    words_i[io + IREC_FLAGS] = flags
    words_i[io + IREC_ORDINAL] = Int32(-1) if categorical else Int32(2 * bin)
    words_i[io + IREC_LEFT_COUNT] = Int32(n_left)
    words_i[io + IREC_RIGHT_COUNT] = Int32(n_right)
    words_i[io + IREC_TOTAL_COUNT] = Int32(n_left + n_right)
    if categorical:
        # A set holding two categories, packed sixteen bits to a word exactly
        # as the scan kernel packs it.
        words_i[io + IREC_CAT0] = Int32(1 << 3) | Int32(1 << 5)
        words_i[io + IREC_CAT0 + 1] = Int32(1 << 2)
    words_f[fo + FREC_GAIN] = gain
    words_f[fo + FREC_LEFT_GRAD] = Float32(n_left) * Float32(0.25)
    words_f[fo + FREC_LEFT_HESS] = Float32(n_left)
    words_f[fo + FREC_RIGHT_GRAD] = Float32(n_right) * Float32(0.25)
    words_f[fo + FREC_RIGHT_HESS] = Float32(n_right)
    words_f[fo + FREC_TOTAL_GRAD] = Float32(n_left + n_right) * Float32(0.25)
    words_f[fo + FREC_TOTAL_HESS] = Float32(n_left + n_right)
    words_f[fo + FREC_LEFT_VALUE] = left_value
    words_f[fo + FREC_RIGHT_VALUE] = right_value
    words_f[fo + FREC_PARENT_VALUE] = Float32(-0.5)


def _records_for(
    leaves: List[DeviceLeafRow],
    capacity: Int,
    seed: UInt64,
    tie_classes: Int,
    balanced: Bool,
) raises -> Tuple[List[Int32], List[Float32]]:
    """A record for every live leaf, a pure function of the leaf's node id.

    Deriving from the node id rather than from the step number means a leaf
    that survives several steps keeps the same candidate, which is what
    really happens: a leaf's histogram and its best split do not change until
    the leaf is split.

    `tie_classes` sets how coarse the gains are. Small values force exact
    Float32 ties across the frontier, which is the case the tie-break rule
    exists for; large values make ties rare, which is the case that checks
    the ordinary path is not accidentally relying on the tie rule.

    `balanced` splits a leaf near the middle so a frontier can actually grow
    to its budget; the skewed alternative produces small leaves quickly,
    which is what exercises the row floor and the "no candidate left" exit.
    """
    var words_i = _empty_words_i(capacity)
    var words_f = _empty_words_f(capacity)
    for i in range(len(leaves)):
        var leaf = leaves[i].copy()
        var h = _mix(seed ^ (UInt64(leaf.node) * 0x2545F4914F6CDD1D))
        var found = (h % 16) != 0
        if leaf.row_count < 2:
            found = False
        var n_left = 0
        var n_right = 0
        if found:
            if balanced:
                var half = leaf.row_count // 2
                var jitter = Int((h >> 8) % 3) - 1
                n_left = half + jitter
                if n_left < 1:
                    n_left = 1
                if n_left > leaf.row_count - 1:
                    n_left = leaf.row_count - 1
            else:
                n_left = 1 + Int((h >> 8) % UInt64(leaf.row_count - 1))
            n_right = leaf.row_count - n_left
        var klass = Int((h >> 20) % UInt64(tie_classes))
        var gain = Float32(klass + 1)
        var feature = Int((h >> 32) % UInt64(_N_FEATURES))
        var bin = Int((h >> 40) % UInt64(_N_BINS - 1))
        var default_left = ((h >> 48) & 1) == 1
        # One split in eight is categorical, so both branches of the commit's
        # routing write are reached.
        var categorical = ((h >> 52) % 8) == 0
        var lv = Float32(Int((h >> 12) % 2001) - 1000) * Float32(0.03125)
        var rv = Float32(Int((h >> 24) % 2001) - 1000) * Float32(0.03125)
        _write_record(
            words_i,
            words_f,
            leaf.record,
            found,
            feature,
            bin,
            default_left,
            categorical,
            gain,
            n_left,
            n_right,
            lv,
            rv,
        )
    return (words_i^, words_f^)


# --- Record upload --------------------------------------------------------


struct _RecordBuffers(Movable):
    """Device-side split records the test owns.

    In a wired caller these are `GpuSplitSearcher.rec_i_dev` and `rec_f_dev`;
    `DeviceTreeTables.enqueue_step` takes them as arguments precisely so a
    test can supply its own without standing up a searcher.
    """

    var ctx: DeviceContext
    var capacity: Int
    var rec_i: DeviceBuffer[DType.int32]
    var rec_f: DeviceBuffer[DType.float32]
    var stage_i: HostBuffer[DType.int32]
    var stage_f: HostBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        self.ctx = ctx
        self.capacity = capacity
        self.rec_i = self.ctx.enqueue_create_buffer[DType.int32](
            capacity * SPLIT_IWORDS
        )
        self.rec_f = self.ctx.enqueue_create_buffer[DType.float32](
            capacity * SPLIT_FWORDS
        )
        self.stage_i = self.ctx.enqueue_create_host_buffer[DType.int32](
            capacity * SPLIT_IWORDS
        )
        self.stage_f = self.ctx.enqueue_create_host_buffer[DType.float32](
            capacity * SPLIT_FWORDS
        )

    def upload(
        mut self, words_i: List[Int32], words_f: List[Float32]
    ) raises:
        """Enqueue both record copies. Does not synchronize.

        The caller's next wait covers them, which in every use here is the
        `download` that follows the step. That matters for the runtime of
        this file rather than for its correctness: a `DeviceContext.
        synchronize` measured about a quarter of a second in this harness, so
        a step that waits twice costs twice what a step that waits once
        costs, and there are a few hundred steps below.

        The staging buffers are only refilled on the next call, which is
        always after that wait, so no copy is ever reading a buffer the host
        is rewriting.
        """
        if len(words_i) != self.capacity * SPLIT_IWORDS:
            raise Error("integer record buffer is the wrong length")
        if len(words_f) != self.capacity * SPLIT_FWORDS:
            raise Error("float record buffer is the wrong length")
        var di = self.stage_i.unsafe_ptr()
        for i in range(len(words_i)):
            di.unsafe_store(i, words_i[i])
        var df = self.stage_f.unsafe_ptr()
        for i in range(len(words_f)):
            df.unsafe_store(i, words_f[i])
        self.ctx.enqueue_copy(
            dst_buf=self.rec_i, src_ptr=self.stage_i.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.rec_f, src_ptr=self.stage_f.unsafe_ptr()
        )


# --- The host reference ---------------------------------------------------


struct _RefTree(Movable):
    """The host's answer, built by calling the host's own decision code.

    Holds exactly what `TreeTablesSnapshot` holds, in the same types, so the
    comparison is `same_as` on every row and an integer equality on every
    counter, with nothing translated in between.
    """

    var leaves: List[DeviceLeafRow]
    var nodes: List[DeviceNodeRow]
    var pool: HistogramSlotPool
    var next_node: Int
    var pick: Int
    var pick_node: Int
    var status: Int
    var commits: Int
    var schedule: GrowthSchedule

    def __init__(
        out self, n_active: Int, num_leaves: Int, root_slot: Int = 0
    ) raises:
        """The one-leaf frontier `DeviceTreeTables.begin_tree` makes."""
        self.leaves = List[DeviceLeafRow]()
        self.leaves.append(DeviceLeafRow(0, 0, n_active, 0, root_slot, 0))
        self.nodes = List[DeviceNodeRow]()
        var root = DeviceNodeRow()
        root.count = n_active
        self.nodes.append(root^)
        self.pool = HistogramSlotPool(num_leaves)
        _ = self.pool.acquire(0, 0)
        self.next_node = 1
        self.pick = -1
        self.pick_node = -1
        self.status = TREE_RUNNING
        self.commits = 0
        self.schedule = GrowthSchedule()

    def step(
        mut self,
        words_i: List[Int32],
        words_f: List[Float32],
        params: TreeParams,
        missing_bins: List[Int],
    ) raises:
        """One growth step, as `_device_search_resident` takes one.

        The order of the three tests is that loop's order: the leaf budget is
        the `while` condition and runs before anything else, then the
        candidates are described exactly as the loop describes them, then
        `plan_level` answers.
        """
        self.pick = -1
        self.pick_node = -1
        if len(self.leaves) >= params.num_leaves:
            self.status = TREE_BUDGET_SPENT
            return

        var cands = List[LeafCandidate](capacity=len(self.leaves))
        for i in range(len(self.leaves)):
            var rec = decode_record(words_i, words_f, self.leaves[i].record)
            _apply_shape_rules(
                rec, self.leaves[i].row_count, self.leaves[i].depth, params
            )
            cands.append(
                LeafCandidate(
                    self.leaves[i].node,
                    self.leaves[i].depth,
                    rec.gain,
                    rec.found and rec.gain > 0.0,
                )
            )
        var picks = self.schedule.plan_level(
            cands, len(self.leaves), params.num_leaves, params.max_depth
        )
        if len(picks) == 0:
            self.status = TREE_NO_CANDIDATE
            return

        var idx = picks[0]
        var leaf = self.leaves[idx].copy()
        var rec = decode_record(words_i, words_f, leaf.record)
        var split = rec.to_split_info()
        var missing = -1 if split.is_categorical else missing_bins[
            split.feature
        ]
        var n_left = rec.left.count
        var n_right = rec.right.count
        var left_node = self.next_node
        var right_node = self.next_node + 1

        # `Tree._add_node(0.0, Float64(n))` twice, left then right, each then
        # given its child value by `_commit_device_split`. Under the
        # configurations `tree_resident_supported` admits, the clamp in that
        # function is the identity, so the value written is the record's.
        var left = DeviceNodeRow()
        left.count = n_left
        left.value = Float32(rec.left_value)
        var right = DeviceNodeRow()
        right.count = n_right
        right.value = Float32(rec.right_value)
        self.nodes.append(left^)
        self.nodes.append(right^)

        # `Tree._set_split` on the parent.
        var parent = self.nodes[leaf.node].copy()
        parent.feature = split.feature
        parent.split_gain = Float32(split.gain)
        parent.left = left_node
        parent.right = right_node
        if split.is_categorical:
            parent.threshold_bin = -1
            parent.default_left = False
            parent.missing_bin = -1
            parent.is_categorical = True
            var io = leaf.record * SPLIT_IWORDS
            for w in range(CAT_WORDS):
                parent.cat_words[w] = words_i[io + IREC_CAT0 + w]
        else:
            parent.threshold_bin = split.bin
            parent.default_left = split.default_left
            parent.missing_bin = missing
            parent.is_categorical = False
        self.nodes[leaf.node] = parent^

        # The subtraction trick's slot bookkeeping, in the order
        # `_enqueue_resident_split` performs it: acquire while the parent
        # still owns its slot, then hand the parent's slot to the derived
        # child.
        var build_left = subtraction_builds_left(n_left, n_right)
        var built_node = left_node if build_left else right_node
        var derived_node = right_node if build_left else left_node
        var built_slot = self.pool.acquire(built_node, 0)
        if built_slot < 0:
            raise Error("the reference slot pool ran out")
        self.pool.reassign(leaf.hist_slot, derived_node)
        var left_slot = built_slot if build_left else leaf.hist_slot
        var right_slot = leaf.hist_slot if build_left else built_slot

        # The left child takes the parent's frontier slot and its record; the
        # right child is appended and takes the record with its own index.
        self.leaves[idx] = DeviceLeafRow(
            left_node,
            leaf.row_begin,
            n_left,
            leaf.depth + 1,
            left_slot,
            leaf.record,
        )
        self.leaves.append(
            DeviceLeafRow(
                right_node,
                leaf.row_begin + n_left,
                n_right,
                leaf.depth + 1,
                right_slot,
                len(self.leaves),
            )
        )
        self.next_node += 2
        self.pick = idx
        self.pick_node = leaf.node
        self.commits += 1
        self.status = TREE_RUNNING


def _assert_matches(snap: TreeTablesSnapshot, host_ref: _RefTree) raises:
    """Every field of the device state against every field of the host's,
    with no tolerance anywhere."""
    assert_equal(
        tree_status_name(snap.status), tree_status_name(host_ref.status)
    )
    assert_equal(snap.pick, host_ref.pick)
    assert_equal(snap.pick_node, host_ref.pick_node)
    assert_equal(snap.commits, host_ref.commits)
    assert_equal(snap.n_live, len(host_ref.leaves))
    assert_equal(snap.next_node, host_ref.next_node)
    assert_equal(len(snap.leaves), len(host_ref.leaves))
    for i in range(len(host_ref.leaves)):
        assert_true(
            snap.leaves[i].same_as(host_ref.leaves[i]),
            String("frontier slot ") + String(i) + " differs",
        )
    assert_equal(len(snap.nodes), len(host_ref.nodes))
    for n in range(len(host_ref.nodes)):
        assert_true(
            snap.nodes[n].same_as(host_ref.nodes[n]),
            String("tree node ") + String(n) + " differs",
        )
    assert_equal(len(snap.slot_owner), host_ref.pool.capacity)
    for s in range(host_ref.pool.capacity):
        assert_equal(snap.slot_owner[s], host_ref.pool.owner_of(s))


# --- Host-only tests ------------------------------------------------------


def test_the_gate_is_off_unless_it_is_set_to_one() raises:
    """Default off, and off for anything but the exact string "1". Nothing
    in the package consults this yet, which is the point: the switch is
    agreed before the wiring rather than after it."""
    assert_false(tree_resident_requested())


def test_every_refusal_has_a_reason_and_a_name() raises:
    """`tree_resident_supported` is the list of host decisions the device
    cannot make, so each one is checked against a parameter set that turns
    exactly it on."""
    assert_equal(tree_resident_supported(TreeParams.default()), TREE_RESIDENT_OK)

    var mono = TreeParams.default()
    mono.monotone = MonotoneConstraints.from_signs([1, 0, -1, 0], 4)
    assert_equal(tree_resident_supported(mono), TREE_RESIDENT_MONOTONE)

    var bynode = TreeParams.default()
    bynode.feature_fraction_bynode = 0.5
    assert_equal(tree_resident_supported(bynode), TREE_RESIDENT_BYNODE)

    var inter = TreeParams.default()
    inter.constraints = InteractionConstraints.from_groups([[0, 1]], 4)
    assert_equal(tree_resident_supported(inter), TREE_RESIDENT_INTERACTION)

    var deep = TreeParams.default()
    deep.grow_policy = GROW_DEPTHWISE
    assert_equal(tree_resident_supported(deep), TREE_RESIDENT_DEPTHWISE)

    var extra = TreeParams.default()
    var gain_floor = ExtraTreeParams()
    gain_floor.min_gain_to_split = 0.5
    extra.extra = gain_floor^
    assert_equal(tree_resident_supported(extra), TREE_RESIDENT_EXTRA)

    var bylevel = TreeParams.default()
    bylevel.feature_fraction_bylevel = 0.5
    assert_equal(tree_resident_supported(bylevel), TREE_RESIDENT_EXTRA)

    # Every code names itself, so a failure downstream reports a reason
    # rather than a number.
    assert_equal(tree_resident_reason_name(TREE_RESIDENT_OK), "ok")
    assert_equal(
        tree_resident_reason_name(TREE_RESIDENT_MONOTONE),
        "monotone constraints",
    )
    assert_equal(tree_resident_reason_name(99), "unknown")


def test_the_layouts_are_disjoint_and_the_strides_cover_them() raises:
    """A word offset that collided with another would corrupt one field with
    another silently, so the strides are asserted against the largest offset
    each layout defines."""
    assert_equal(FRONT_WORDS, 6)
    assert_equal(CTR_WORDS, 6)
    # Eight scalar node fields, then the category set.
    assert_equal(TN_IWORDS, 8 + CAT_WORDS)
    assert_equal(TN_FWORDS, 2)
    # The reduction's block width has to be a warp multiple on every
    # supported backend, and it is the same width the split search's
    # cross-feature reduction uses.
    assert_equal(PICK_THREADS, 64)


def test_a_fresh_node_row_reads_as_a_leaf() raises:
    """`DeviceNodeRow()` is what `Tree._add_node` leaves: no split, no
    routing, no children, zero gain."""
    var row = DeviceNodeRow()
    assert_true(row.is_leaf())
    assert_equal(row.feature, -1)
    assert_equal(row.threshold_bin, -1)
    assert_equal(row.left, -1)
    assert_equal(row.right, -1)
    assert_equal(row.missing_bin, -1)
    assert_false(row.is_categorical)
    assert_equal(row.value.to_bits(), Float32(0.0).to_bits())
    assert_true(row.to_split_info().feature < 0)
    assert_false(row.to_split_info().found)


# --- Device tests ---------------------------------------------------------


def _missing_bins() -> List[Int]:
    """One missing bin per feature, distinct so a commit reading the wrong
    feature's entry is visible."""
    var out = List[Int](capacity=_N_FEATURES)
    for f in range(_N_FEATURES):
        out.append(f % 3)
    return out^


def _run_tree(
    seed: UInt64,
    n_active: Int,
    num_leaves: Int,
    max_depth: Int,
    min_data_in_leaf: Int,
    tie_classes: Int,
    balanced: Bool,
) raises:
    """Grow one whole tree twice, once on the device and once by calling the
    host's decision code, comparing after every single step.

    Comparing after every step rather than at the end is deliberate. The two
    frontiers feed each other: the records for step k+1 are generated from
    the frontier step k produced, so a divergence that was allowed to
    accumulate would show up as an avalanche of differences with no way to
    tell which step caused it.
    """
    var params = TreeParams.default()
    params.num_leaves = num_leaves
    params.max_depth = max_depth
    params.min_data_in_leaf = min_data_in_leaf
    var missing = _missing_bins()

    var ctx = DeviceContext()
    var tables = DeviceTreeTables(ctx, num_leaves, _N_FEATURES, missing)
    var recs = _RecordBuffers(ctx, num_leaves)
    tables.begin_tree(n_active, root_slot=0, root_value=Float32(0.0))
    var host_ref = _RefTree(n_active, num_leaves, root_slot=0)

    # A tree of `num_leaves` leaves takes `num_leaves - 1` commits; one more
    # iteration reaches the budget-spent or no-candidate exit, and the bound
    # keeps a wiring mistake from looping forever.
    for _ in range(num_leaves + 1):
        var pair = _records_for(
            host_ref.leaves, num_leaves, seed, tie_classes, balanced
        )
        var words_i = pair[0].copy()
        var words_f = pair[1].copy()
        recs.upload(words_i, words_f)

        tables.enqueue_step(
            recs.rec_i, recs.rec_f, num_leaves, max_depth, min_data_in_leaf
        )
        var snap = tables.download()
        host_ref.step(words_i, words_f, params, missing)
        _assert_matches(snap, host_ref)
        if snap.status == TREE_RUNNING:
            snap.check_invariants()
        else:
            break


def test_the_device_pick_matches_the_host_over_many_trees() raises:
    """The lane's acceptance test.

    Seven shapes, chosen so that each of the pick's exits is reached and each
    of the shape rules binds somewhere:

      - a frontier narrower than one threadgroup and one wider than it, so
        both the one-slot-per-thread and the several-slots-per-thread paths
        run;
      - coarse gains, which force exact ties, and fine gains, which do not;
      - balanced splits, which grow to the budget, and skewed ones, which run
        out of admissible leaves first;
      - `max_depth` binding and not binding;
      - `min_data_in_leaf` large enough that the parent row floor stops
        growth before the budget does.
    """
    comptime if not has_accelerator():
        return
    else:
        # Narrow frontier, coarse gains: ties everywhere, one leaf per
        # thread, growth to the budget.
        _run_tree(0x51ED, 100000, 31, -1, 1, 2, True)
        # Narrow frontier, fine gains: the ordinary path with ties rare.
        _run_tree(0xA113, 100000, 31, -1, 1, 4096, True)
        # Wider than one threadgroup, coarse gains: a thread owns several
        # slots, so both the intra-thread tie rule and the collective one
        # decide the winner.
        _run_tree(0x7A2C, 200000, 72, -1, 1, 3, True)
        # A depth limit that binds well before the budget.
        _run_tree(0x0D33, 100000, 63, 4, 1, 3, True)
        # A row floor that binds: skewed splits make small leaves fast.
        _run_tree(0xF100, 4000, 63, -1, 50, 3, False)
        # Skewed splits with no floor, which is the "ran out of candidates"
        # exit rather than the budget one.
        _run_tree(0xC0DE, 2000, 63, -1, 1, 3, False)
        # A tiny active prefix, so the root itself may be inadmissible.
        _run_tree(0x00A1, 6, 31, -1, 2, 2, False)


def _tie_pick(
    mut tables: DeviceTreeTables,
    mut recs: _RecordBuffers,
    slots: List[Int],
    num_leaves: Int,
    n_live: Int,
) raises -> Int:
    """Stage `n_live` leaves whose only positive gains sit at `slots`, all
    exactly equal, and return the slot the device picked.

    The tables and the record buffers come from the caller and are reset
    here rather than rebuilt, because opening a `DeviceContext` and its
    seventeen buffers dominates everything else this test does; one context
    serves every case below.
    """
    tables.begin_tree(1000000, root_slot=0, root_value=Float32(0.0))

    var leaves = List[DeviceLeafRow]()
    for i in range(n_live):
        # Node ids are deliberately not in slot order, so a kernel that broke
        # the tie on node id instead of on slot would fail here.
        leaves.append(DeviceLeafRow(n_live - i, 1000 * i, 1000, 0, i, i))
    tables.stage_frontier(leaves, n_live + 1)

    var words_i = _empty_words_i(num_leaves)
    var words_f = _empty_words_f(num_leaves)
    for i in range(n_live):
        _write_record(
            words_i,
            words_f,
            i,
            False,
            -1,
            -1,
            False,
            False,
            Float32(0.0),
            0,
            0,
            Float32(0.0),
            Float32(0.0),
        )
    for k in range(len(slots)):
        _write_record(
            words_i,
            words_f,
            slots[k],
            True,
            1,
            7,
            False,
            False,
            Float32(7.0),
            400,
            600,
            Float32(0.25),
            Float32(-0.25),
        )
    recs.upload(words_i, words_f)
    tables.enqueue_step(recs.rec_i, recs.rec_f, num_leaves, -1, 1)
    var snap = tables.download()
    assert_equal(tree_status_name(snap.status), tree_status_name(TREE_RUNNING))
    return snap.pick


def test_ties_go_to_the_lowest_frontier_slot() raises:
    """The tie-break, placed by hand at slot pairs that fall on one thread
    and at pairs that fall on different ones.

    The host rule is `growth_policy.GrowthSchedule.next_leaf` under
    `GROW_LEAFWISE`: a strict `>` over an ascending scan, so the winner among
    equal gains is the lowest-indexed candidate. Under a parallel reduction
    that is not automatic, because `block.max` returns a gain and not a
    position, so the kernel carries the slot alongside the gain and takes a
    `block.min` over the slots of every thread that tied the winner.

    With `PICK_THREADS` at 64, thread `t` owns slots `t`, `t + 64`,
    `t + 128`, and so on. So `{5, 69}` is one thread deciding against itself
    and `{5, 6}` is two threads deciding against each other, and the two are
    genuinely different code.
    """
    comptime if not has_accelerator():
        return
    else:
        var missing = _missing_bins()
        var ctx = DeviceContext()
        var wide = DeviceTreeTables(ctx, 256, _N_FEATURES, missing)
        var wide_recs = _RecordBuffers(ctx, 256)

        # A single winner, no tie at all.
        assert_equal(_tie_pick(wide, wide_recs, [37], 256, 200), 37)
        # Two adjacent slots: different threads.
        assert_equal(_tie_pick(wide, wide_recs, [5, 6], 256, 200), 5)
        assert_equal(_tie_pick(wide, wide_recs, [6, 5], 256, 200), 5)
        # One thread's own slots: 5 and 69 and 133 all belong to thread 5.
        assert_equal(_tie_pick(wide, wide_recs, [5, 69], 256, 200), 5)
        assert_equal(_tie_pick(wide, wide_recs, [69, 5], 256, 200), 5)
        assert_equal(_tie_pick(wide, wide_recs, [133, 69, 5], 256, 200), 5)
        # A tie between two threads that are not thread 0, so the winner is
        # named by a thread that does not perform the commit.
        assert_equal(_tie_pick(wide, wide_recs, [69, 70], 256, 200), 69)
        assert_equal(_tie_pick(wide, wide_recs, [131, 67], 256, 200), 67)
        # The first and the last slot.
        assert_equal(_tie_pick(wide, wide_recs, [199, 0], 256, 200), 0)
        # Every slot tied.
        var all_slots = List[Int]()
        for i in range(200):
            all_slots.append(i)
        assert_equal(_tie_pick(wide, wide_recs, all_slots, 256, 200), 0)

        # A frontier narrower than the block, so most threads hold nothing.
        var narrow = DeviceTreeTables(ctx, 64, _N_FEATURES, missing)
        var narrow_recs = _RecordBuffers(ctx, 64)
        assert_equal(_tie_pick(narrow, narrow_recs, [3, 9], 64, 12), 3)
        assert_equal(_tie_pick(narrow, narrow_recs, [11, 3], 64, 12), 3)


def test_a_frontier_with_nothing_admissible_commits_nothing() raises:
    """The three ways a step can decline, each checked for the status it
    reports and for leaving the tables untouched."""
    comptime if not has_accelerator():
        return
    else:
        var missing = _missing_bins()
        var ctx = DeviceContext()
        var tables = DeviceTreeTables(ctx, 31, _N_FEATURES, missing)
        var recs = _RecordBuffers(ctx, 31)

        # No record carries a found flag.
        tables.begin_tree(1000, root_slot=0, root_value=Float32(0.0))
        var words_i = _empty_words_i(31)
        var words_f = _empty_words_f(31)
        recs.upload(words_i, words_f)
        tables.enqueue_step(recs.rec_i, recs.rec_f, 31, -1, 1)
        var snap = tables.download()
        assert_equal(
            tree_status_name(snap.status),
            tree_status_name(TREE_NO_CANDIDATE),
        )
        assert_equal(snap.pick, -1)
        assert_equal(snap.n_live, 1)
        assert_equal(snap.next_node, 1)
        assert_equal(snap.commits, 0)

        # A found split whose gain is zero. The host's `best_gain` starts at
        # 0.0 and the comparison is strict, so this never wins either.
        _write_record(
            words_i,
            words_f,
            0,
            True,
            2,
            5,
            False,
            False,
            Float32(0.0),
            400,
            600,
            Float32(0.5),
            Float32(-0.5),
        )
        recs.upload(words_i, words_f)
        tables.enqueue_step(recs.rec_i, recs.rec_f, 31, -1, 1)
        snap = tables.download()
        assert_equal(
            tree_status_name(snap.status),
            tree_status_name(TREE_NO_CANDIDATE),
        )
        assert_equal(snap.commits, 0)

        # A positive gain that the parent row floor refuses:
        # `n_rows < 2 * min_data_in_leaf`.
        _write_record(
            words_i,
            words_f,
            0,
            True,
            2,
            5,
            False,
            False,
            Float32(3.0),
            400,
            600,
            Float32(0.5),
            Float32(-0.5),
        )
        recs.upload(words_i, words_f)
        tables.enqueue_step(recs.rec_i, recs.rec_f, 31, -1, 501)
        snap = tables.download()
        assert_equal(
            tree_status_name(snap.status),
            tree_status_name(TREE_NO_CANDIDATE),
        )
        assert_equal(snap.commits, 0)

        # The same record with the floor lowered does commit, so the three
        # refusals above are refusals and not a broken record.
        tables.enqueue_step(recs.rec_i, recs.rec_f, 31, -1, 500)
        snap = tables.download()
        assert_equal(
            tree_status_name(snap.status), tree_status_name(TREE_RUNNING)
        )
        assert_equal(snap.commits, 1)
        assert_equal(snap.n_live, 2)
        assert_equal(snap.next_node, 3)


def test_the_leaf_budget_stops_a_step_before_it_reads_anything() raises:
    """`n_live >= num_leaves` is the host's `while n_leaves <
    params.num_leaves`, and it wins over a frontier full of good splits."""
    comptime if not has_accelerator():
        return
    else:
        var missing = _missing_bins()
        var ctx = DeviceContext()
        var tables = DeviceTreeTables(ctx, 64, _N_FEATURES, missing)
        var recs = _RecordBuffers(ctx, 64)
        tables.begin_tree(100000, root_slot=0, root_value=Float32(0.0))

        var leaves = List[DeviceLeafRow]()
        for i in range(8):
            leaves.append(DeviceLeafRow(i + 1, 1000 * i, 1000, 0, i, i))
        tables.stage_frontier(leaves, 9)

        var words_i = _empty_words_i(64)
        var words_f = _empty_words_f(64)
        for i in range(8):
            _write_record(
                words_i,
                words_f,
                i,
                True,
                1,
                4,
                True,
                False,
                Float32(i + 1),
                400,
                600,
                Float32(0.5),
                Float32(-0.5),
            )
        recs.upload(words_i, words_f)

        # Eight live leaves against a budget of eight: spent.
        tables.enqueue_step(recs.rec_i, recs.rec_f, 8, -1, 1)
        var snap = tables.download()
        assert_equal(
            tree_status_name(snap.status),
            tree_status_name(TREE_BUDGET_SPENT),
        )
        assert_equal(snap.pick, -1)
        assert_equal(snap.n_live, 8)
        assert_equal(snap.commits, 0)

        # Nine, and the best gain wins, which here is the last slot.
        tables.enqueue_step(recs.rec_i, recs.rec_f, 9, -1, 1)
        snap = tables.download()
        assert_equal(
            tree_status_name(snap.status), tree_status_name(TREE_RUNNING)
        )
        assert_equal(snap.pick, 7)
        assert_equal(snap.pick_node, 8)
        assert_equal(snap.n_live, 9)


def test_the_step_is_bit_deterministic_run_to_run() raises:
    """The same tables, the same records, the same step, twice.

    A reduction whose answer depended on the order threads arrived would
    pass every comparison against a single host run and fail this one, so it
    is asserted separately rather than assumed from the equivalence tests.
    """
    comptime if not has_accelerator():
        return
    else:
        var missing = _missing_bins()
        var ctx = DeviceContext()
        var first = TreeTablesSnapshot()
        for run in range(2):
            var tables = DeviceTreeTables(ctx, 96, _N_FEATURES, missing)
            var recs = _RecordBuffers(ctx, 96)
            tables.begin_tree(200000, root_slot=0, root_value=Float32(0.0))
            var host_ref = _RefTree(200000, 96, root_slot=0)
            for _ in range(20):
                var pair = _records_for(host_ref.leaves, 96, 0x9111, 3, True)
                var words_i = pair[0].copy()
                var words_f = pair[1].copy()
                recs.upload(words_i, words_f)
                tables.enqueue_step(recs.rec_i, recs.rec_f, 96, -1, 1)
                var snap = tables.download()
                var params = TreeParams.default()
                params.num_leaves = 96
                params.min_data_in_leaf = 1
                host_ref.step(words_i, words_f, params, missing)
                if snap.status != TREE_RUNNING:
                    break
            var final = tables.download()
            if run == 0:
                first = final^
            else:
                assert_equal(final.n_live, first.n_live)
                assert_equal(final.next_node, first.next_node)
                assert_equal(final.commits, first.commits)
                for i in range(len(first.leaves)):
                    assert_true(final.leaves[i].same_as(first.leaves[i]))
                for n in range(len(first.nodes)):
                    assert_true(final.nodes[n].same_as(first.nodes[n]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
