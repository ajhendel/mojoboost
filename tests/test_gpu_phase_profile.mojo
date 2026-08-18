"""The per-phase, per-node-size profiler (phase_profile.mojo).

Four claims, and none of them is a timing. Nothing here measures anything,
which is deliberate: a test that asserted a duration would be asserting the
speed of the machine it happens to run on, and this instrument exists
precisely because nobody should be reading speeds off anything but an
interleaved benchmark.

1. **The size classes partition the row counts.** Every `node_rows` in
   `[0, root_rows]` falls in exactly one class, the classes are ordered, and
   the boundaries sit exactly where the constants say. Checked exhaustively
   over every row count of a small root against an independently written
   Float64 rule, not by sampling, because an off-by-one at a boundary is
   exactly the bug a sample would miss.

2. **An off profile counts nothing.** Charged with large durations and large
   counts across every phase, every one of the fifty buckets stays zero. This
   is the counter half of the free-when-off claim; the clock half is
   structural and lives in `PhaseProfile.clock`, which returns 0 without
   reading `perf_counter_ns` when the mode is off.

3. **An on profile's counts are the ones the grower structurally must have
   produced.** A leaf-wise tree with `L` leaves commits `L - 1` splits, and
   from that every count below follows with no freedom: `L` histogram builds,
   `L - 1` subtractions, `L` row-list constructions, `2L - 1` split scans.
   Asserting a real invariant rather than a recorded number is what makes
   this test able to catch a charge that was added twice or dropped.

4. **The instrument does not move a model.** Predictions are compared bit for
   bit across off, async, and fenced, at the tree level and through a whole
   `train`. A charge is an integer add and a fence is a wait, so any
   difference at all would be a bug in the wiring rather than a tolerance
   question, and the assertions are exact equality.

No accelerator is opened. Every path exercised here is the CPU grower and the
pure classification arithmetic, which is why this file stays out of the
runner's GPU_ONLY list: it guards the instrument on a machine with no device,
and the device call sites in train_gpu.mojo are covered by the fits in
tests/test_gpu_training.mojo continuing to produce the same trees.
"""

# run_tests: cpu-safe -- host arithmetic only, opens no device.

from std.os import setenv
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from support import _make_features, _uniform

from mojotrees.binning import BinnedMatrix, fit_bins
from mojotrees.boosting import SQUARED_ERROR, BoosterParams, train
from mojotrees.cegb import CegbLedger
from mojotrees.phase_profile import (
    CLASS_LARGE,
    CLASS_MEDIUM,
    CLASS_ROOT,
    CLASS_SMALL,
    CLASS_TINY,
    HOST_HIST_DISPATCHES,
    HOST_PARTITION_DISPATCHES,
    HOST_SPLIT_SEARCH_DISPATCHES,
    HOST_SUBTRACT_DISPATCHES,
    LARGE_MIN_INVERSE,
    MEDIUM_MIN_INVERSE,
    N_NODE_CLASSES,
    N_PROFILE_PHASES,
    PROFILE_ASYNC,
    PROFILE_FENCED,
    PROFILE_OFF,
    PROF_CONVERT,
    PROF_GRAD_FILL,
    PROF_HISTOGRAM,
    PROF_HIST_ALLOC,
    PROF_HOST_SYNC,
    PROF_PARTITION,
    PROF_SCORE_UPDATE,
    PROF_SPLIT_SEARCH,
    PROF_SUBTRACT,
    PROF_TRANSFER,
    SCOPE_FIT,
    SCOPE_TREE,
    SMALL_MIN_INVERSE,
    HOST_ALLOC,
    HOST_ENCODE,
    HOST_PLAN,
    HOST_READBACK,
    HOST_SLOT_EPILOGUE,
    HOST_SLOT_OVERFLOW,
    HOST_SLOT_PROLOGUE,
    HOST_STEP_SLOTS,
    HOST_UPLOAD,
    HOST_WAIT,
    N_HOST_SPANS,
    N_HOST_STEP_SLOTS,
    PhaseProfile,
    classify_node,
    env_profile_mode,
    host_slot_name,
    host_span_name,
    host_step_slot,
    node_class_name,
    profile_phase_name,
)
from mojotrees.tree import TreeParams, grow_tree, grow_tree_profiled


def _data(n_rows: Int, n_features: Int) raises -> BinnedMatrix:
    """A small deterministic binned matrix, from the shared generator."""
    var features = _make_features(n_rows, n_features)
    var mapper = fit_bins(features, n_rows, n_features, 32)
    return mapper.transform(features, n_rows)


def _target(n_rows: Int, n_features: Int) -> List[Float64]:
    """A target with real structure in the first two features, so a tree has
    something to split on and actually reaches its leaf budget."""
    var features = _make_features(n_rows, n_features)
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        out.append(3.0 * x0 + 2.0 * x1 * x1 + 0.1 * _uniform(UInt64(r)))
    return out^


def _grad_hess(
    mut grad: List[Float64],
    mut hess: List[Float64],
    n_rows: Int,
    target: List[Float64],
):
    """Squared-error gradients and hessians against a zero raw score."""
    grad = List[Float64](capacity=n_rows)
    hess = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        grad.append(-target[r])
        hess.append(1.0)


def _expected_class(node_rows: Int, root_rows: Int) -> Int:
    """The class rule written a second time, in floating point, so the
    integer comparisons in `classify_node` are checked against something that
    is not a copy of themselves."""
    if root_rows <= 0:
        return CLASS_ROOT
    if node_rows >= root_rows:
        return CLASS_ROOT
    var f = Float64(node_rows) / Float64(root_rows)
    if f > 1.0 / Float64(LARGE_MIN_INVERSE):
        return CLASS_LARGE
    if f > 1.0 / Float64(MEDIUM_MIN_INVERSE):
        return CLASS_MEDIUM
    if f > 1.0 / Float64(SMALL_MIN_INVERSE):
        return CLASS_SMALL
    return CLASS_TINY


def test_classes_partition_every_row_count() raises:
    """Exactly one class per row count, over every row count of a root.

    `root_rows` is 4096 so all three boundaries land on integers (512, 64, 8)
    and the walk crosses each of them with a row count on either side. A
    function cannot return two classes at once, so "no overlap" is free; what
    is asserted is that there is no *gap* -- every value in range gets a class
    in `[0, N_NODE_CLASSES)` -- and that the class the integer rule picks is
    the one the fraction rule picks.
    """
    var root = 4096
    var seen = List[Int](capacity=N_NODE_CLASSES)
    for _ in range(N_NODE_CLASSES):
        seen.append(0)
    var previous = CLASS_ROOT
    for n in range(root, -1, -1):
        var cls = classify_node(n, root)
        assert_true(cls >= 0 and cls < N_NODE_CLASSES)
        assert_equal(cls, _expected_class(n, root))
        # Monotone as the row count falls: a child is never in a larger class
        # than its parent, which is what makes a class ordering meaningful.
        assert_true(cls >= previous)
        previous = cls
        seen[cls] += 1
    # Every class is reachable, so none of the five is a name with no
    # territory behind it.
    for c in range(N_NODE_CLASSES):
        assert_true(seen[c] > 0)
    # And the counts add back up to the whole range, which is the partition
    # stated as arithmetic.
    var total = 0
    for c in range(N_NODE_CLASSES):
        total += seen[c]
    assert_equal(total, root + 1)


def test_class_boundaries_are_where_the_constants_say() raises:
    """The three boundaries, checked on both sides.

    A node holding exactly `root / 8` is *not* large: the rule opens `large`
    above an eighth, not at it. Pinning that here is what stops a later
    refactor from turning a strict comparison into a loose one and shifting
    every profile by one class without any test noticing.
    """
    var root = 4096
    assert_equal(classify_node(root, root), CLASS_ROOT)
    assert_equal(classify_node(root - 1, root), CLASS_LARGE)

    var large_edge = root // LARGE_MIN_INVERSE
    assert_equal(classify_node(large_edge + 1, root), CLASS_LARGE)
    assert_equal(classify_node(large_edge, root), CLASS_MEDIUM)

    var medium_edge = root // MEDIUM_MIN_INVERSE
    assert_equal(classify_node(medium_edge + 1, root), CLASS_MEDIUM)
    assert_equal(classify_node(medium_edge, root), CLASS_SMALL)

    var small_edge = root // SMALL_MIN_INVERSE
    assert_equal(classify_node(small_edge + 1, root), CLASS_SMALL)
    assert_equal(classify_node(small_edge, root), CLASS_TINY)
    assert_equal(classify_node(0, root), CLASS_TINY)

    # Degenerate roots do not raise and do not produce a class outside the
    # range; an instrument that can abort a fit is worse than one that
    # mis-files a node it should never have seen.
    assert_equal(classify_node(5, 0), CLASS_ROOT)
    assert_equal(classify_node(0, 0), CLASS_ROOT)
    assert_equal(classify_node(9999, 100), CLASS_ROOT)


def test_an_off_profile_counts_nothing() raises:
    """The counter half of free-when-off.

    Charged with a nonzero start timestamp and large counts on every phase at
    every size, an off profile must leave all fifty buckets, both totals, and
    the wall clock at zero. `charge` returns on the mode test before it
    classifies, so this is the whole of what an off profile does.
    """
    var off = PhaseProfile(PROFILE_OFF, SCOPE_FIT, String("off"))
    assert_true(not off.enabled())
    assert_true(not off.fenced())
    assert_equal(off.clock(), 0)

    var sizes: List[Int] = [1000, 500, 60, 8, 1, 0]
    off.begin_tree(1000, 1000)
    off.note_node()
    off.note_wall(1)
    off.end_tree(1)
    for p in range(N_PROFILE_PHASES):
        for i in range(len(sizes)):
            off.charge(
                p, sizes[i], 1, dispatches=7, syncs=3, slots_per_row=4,
                cells=99,
            )
    for p in range(N_PROFILE_PHASES):
        for c in range(N_NODE_CLASSES):
            assert_equal(off.nanos_of(p, c), 0)
            assert_equal(off.calls_of(p, c), 0)
            assert_equal(off.dispatches_of(p, c), 0)
            assert_equal(off.syncs_of(p, c), 0)
            assert_equal(off.rows_of(p, c), 0)
            assert_equal(off.slots_of(p, c), 0)
            assert_equal(off.cells_of(p, c), 0)
    assert_equal(off.total_calls(), 0)
    assert_equal(off.total_dispatches(), 0)
    assert_equal(off.total_syncs(), 0)
    assert_equal(off.attributed_nanos(), 0)
    assert_equal(off.wall_nanos, 0)
    assert_equal(off.trees, 0)
    assert_equal(off.nodes, 0)

    # A merge into an off profile is a no-op too, so a disabled accumulator
    # cannot pick up counts from an enabled part.
    var on = PhaseProfile(PROFILE_ASYNC, SCOPE_TREE, String("on"))
    on.begin_tree(1000, 1000)
    on.charge(PROF_HISTOGRAM, 1000, 0, dispatches=5)
    off.merge(on)
    assert_equal(off.total_dispatches(), 0)


def test_an_on_profile_counts_what_it_was_charged() raises:
    """The mirror of the previous test: the same charges on an enabled
    profile land in the class the row count names, and nowhere else."""
    var on = PhaseProfile(PROFILE_ASYNC, SCOPE_TREE, String("on"))
    assert_true(on.enabled())
    on.begin_tree(1024, 4096)

    # One charge per class, at a row count squarely inside it.
    on.charge(PROF_HISTOGRAM, 1024, 0, dispatches=2, slots_per_row=3, cells=10)
    on.charge(PROF_HISTOGRAM, 512, 0, dispatches=2, slots_per_row=3, cells=10)
    on.charge(PROF_HISTOGRAM, 64, 0, dispatches=2, slots_per_row=3, cells=10)
    on.charge(PROF_HISTOGRAM, 8, 0, dispatches=2, slots_per_row=3, cells=10)
    on.charge(PROF_HISTOGRAM, 1, 0, dispatches=2, slots_per_row=3, cells=10)

    assert_equal(on.calls_of(PROF_HISTOGRAM, CLASS_ROOT), 1)
    assert_equal(on.calls_of(PROF_HISTOGRAM, CLASS_LARGE), 1)
    assert_equal(on.calls_of(PROF_HISTOGRAM, CLASS_MEDIUM), 1)
    assert_equal(on.calls_of(PROF_HISTOGRAM, CLASS_SMALL), 1)
    assert_equal(on.calls_of(PROF_HISTOGRAM, CLASS_TINY), 1)
    assert_equal(on.rows_of(PROF_HISTOGRAM, CLASS_ROOT), 1024)
    assert_equal(on.slots_of(PROF_HISTOGRAM, CLASS_ROOT), 3 * 1024)
    assert_equal(on.cells_of(PROF_HISTOGRAM, CLASS_TINY), 10)
    assert_equal(on.phase_dispatches(PROF_HISTOGRAM), 10)
    assert_equal(on.phase_calls(PROF_HISTOGRAM), 5)
    # A start timestamp of 0 charges the counts and no time, which is what
    # lets a call site record a launch it did not stopwatch.
    assert_equal(on.phase_nanos(PROF_HISTOGRAM), 0)

    # Nothing leaked into a phase nobody charged.
    for p in range(N_PROFILE_PHASES):
        if p != PROF_HISTOGRAM:
            assert_equal(on.phase_calls(p), 0)

    # An unknown phase is a programming error at a call site and is named as
    # one rather than silently filed somewhere.
    with assert_raises():
        on.charge(N_PROFILE_PHASES, 10, 0)


def test_env_mode_words() raises:
    """The environment contract: three accepted spellings, and a refusal."""
    _ = setenv("MOJOTREES_PHASE_PROFILE", "")
    assert_equal(env_profile_mode(), PROFILE_OFF)
    _ = setenv("MOJOTREES_PHASE_PROFILE", "off")
    assert_equal(env_profile_mode(), PROFILE_OFF)
    _ = setenv("MOJOTREES_PHASE_PROFILE", "0")
    assert_equal(env_profile_mode(), PROFILE_OFF)
    _ = setenv("MOJOTREES_PHASE_PROFILE", "1")
    assert_equal(env_profile_mode(), PROFILE_ASYNC)
    _ = setenv("MOJOTREES_PHASE_PROFILE", "async")
    assert_equal(env_profile_mode(), PROFILE_ASYNC)
    _ = setenv("MOJOTREES_PHASE_PROFILE", "fenced")
    assert_equal(env_profile_mode(), PROFILE_FENCED)
    # A typo raises rather than defaulting to off, because a caller who set
    # the variable meant to profile and a silently unprofiled run wastes the
    # run they set it for.
    _ = setenv("MOJOTREES_PHASE_PROFILE", "asynch")
    with assert_raises():
        _ = env_profile_mode()
    _ = setenv("MOJOTREES_PHASE_PROFILE", "")
    assert_equal(env_profile_mode(), PROFILE_OFF)


def test_host_grower_counts_are_the_structural_invariant() raises:
    """What a leaf-wise tree of `L` leaves must have charged.

    `L - 1` splits follow from `L` leaves, and from that every count is fixed:
    the root plus one built child per split is `L` histogram builds; each
    split derives its other child, so `L - 1` subtractions; the root's row
    list plus one partition per split is `L` charges; the root's scan plus two
    child scans per split is `2L - 1`. None of these is a number read off a
    run -- they are what the loop cannot avoid doing -- so a charge added
    twice or dropped moves one of them and fails here.
    """
    var n_rows = 2000
    var n_features = 6
    var data = _data(n_rows, n_features)
    var target = _target(n_rows, n_features)
    var grad = List[Float64]()
    var hess = List[Float64]()
    _grad_hess(grad, hess, n_rows, target)
    var params = TreeParams(8, 1, 1.0, 1e-3)

    var profile = PhaseProfile(PROFILE_ASYNC, SCOPE_TREE, String("test"))
    var ledger = CegbLedger.none()
    var tree = grow_tree_profiled(
        profile, data, grad, hess, params, ledger
    )
    var leaves = tree.n_leaves
    assert_true(leaves > 1)
    var splits = leaves - 1

    assert_equal(profile.trees, 1)
    assert_equal(profile.root_rows, n_rows)
    assert_equal(profile.dataset_rows, n_rows)

    assert_equal(profile.phase_calls(PROF_HISTOGRAM), leaves)
    assert_equal(profile.phase_calls(PROF_SUBTRACT), splits)
    assert_equal(profile.phase_calls(PROF_HIST_ALLOC), leaves)
    assert_equal(profile.phase_calls(PROF_PARTITION), leaves)
    assert_equal(profile.phase_calls(PROF_SPLIT_SEARCH), 2 * splits + 1)
    # One `note_node` per histogram and one per subtraction.
    assert_equal(profile.nodes, leaves + splits)

    # The dispatch counts are the per-operation constants times the operation
    # counts, which is what makes the column priceable at all.
    assert_equal(
        profile.phase_dispatches(PROF_HISTOGRAM),
        leaves * HOST_HIST_DISPATCHES,
    )
    assert_equal(
        profile.phase_dispatches(PROF_SUBTRACT),
        splits * HOST_SUBTRACT_DISPATCHES,
    )
    assert_equal(
        profile.phase_dispatches(PROF_PARTITION),
        leaves * HOST_PARTITION_DISPATCHES,
    )
    assert_equal(
        profile.phase_dispatches(PROF_SPLIT_SEARCH),
        (2 * splits + 1) * HOST_SPLIT_SEARCH_DISPATCHES,
    )

    # The host grower reaches no device, so the three device-only phases are
    # empty and the round-level phases belong to the boosting loop above it.
    assert_equal(profile.phase_calls(PROF_TRANSFER), 0)
    assert_equal(profile.phase_calls(PROF_CONVERT), 0)
    assert_equal(profile.phase_calls(PROF_HOST_SYNC), 0)
    assert_equal(profile.phase_calls(PROF_GRAD_FILL), 0)
    assert_equal(profile.phase_calls(PROF_SCORE_UPDATE), 0)

    # The root's own charges are in the root class, and the root's histogram
    # covered every row over every feature the tree drew.
    assert_equal(profile.calls_of(PROF_HISTOGRAM, CLASS_ROOT), 1)
    assert_equal(profile.rows_of(PROF_HISTOGRAM, CLASS_ROOT), n_rows)
    assert_equal(
        profile.slots_of(PROF_HISTOGRAM, CLASS_ROOT), n_rows * n_features
    )
    # Every split partitions its parent's whole window, so the rows charged
    # to the partition phase are the root's rows once for the root list plus
    # one parent window per split. Each is at most the root, so the total is
    # bounded by `leaves * n_rows` and is at least `n_rows` twice over (the
    # root list and the root's own split).
    assert_true(profile.phase_calls(PROF_PARTITION) == leaves)
    var partition_rows = 0
    for c in range(N_NODE_CLASSES):
        partition_rows += profile.rows_of(PROF_PARTITION, c)
    assert_true(partition_rows >= 2 * n_rows)

    # Wall time was recorded and no phase claimed more than the whole tree.
    assert_true(profile.wall_nanos > 0)
    assert_true(profile.attributed_nanos() <= profile.wall_nanos)


def test_report_is_parseable_and_complete() raises:
    """The report's shape, which is the half of it a diff depends on.

    Every line begins with `phase_profile`, the row block is emitted in full
    at all fifty cells whether or not they hold anything, and the header names
    the mode and the class boundaries. Emitting the zeros is what makes two
    reports of the same shape diff line for line rather than shifting when a
    class empties.
    """
    var profile = PhaseProfile(PROFILE_FENCED, SCOPE_FIT, String("arm"))
    profile.begin_tree(1000, 1000)
    profile.charge(PROF_HISTOGRAM, 1000, 0, dispatches=1, cells=10)
    var text = profile.report()

    assert_true(text.find("phase_profile begin label=arm scope=fit") >= 0)
    assert_true(text.find("mode=fenced") >= 0)
    assert_true(text.find("root_rows=1000") >= 0)
    assert_true(text.find("phase_profile classes root=all large>1/8") >= 0)
    assert_true(text.find("phase_profile columns kind phase class") >= 0)
    assert_true(text.find("phase_profile totals attributed_ns=") >= 0)
    assert_true(text.find("phase_profile end") >= 0)

    # All fifty cells, all ten phase rollups, all five class rollups.
    assert_equal(
        len(text.split("phase_profile row ")) - 1,
        N_PROFILE_PHASES * N_NODE_CLASSES,
    )
    assert_equal(
        len(text.split("phase_profile phase ")) - 1, N_PROFILE_PHASES
    )
    assert_equal(len(text.split("phase_profile class ")) - 1, N_NODE_CLASSES)

    # Every phase and class name appears, so a parser keyed on names finds
    # every bucket rather than silently missing one that was never printed.
    for p in range(N_PROFILE_PHASES):
        assert_true(text.find(profile_phase_name(p)) >= 0)
    for c in range(N_NODE_CLASSES):
        assert_true(text.find(node_class_name(c)) >= 0)

    # An off profile still renders a table, so a caller that reports
    # unconditionally gets a well-formed empty one rather than a crash; what
    # it must never do is print, which is `print_report`'s job and is why the
    # two are separate.
    var off = PhaseProfile(PROFILE_OFF, SCOPE_TREE, String(""))
    var empty = off.report()
    assert_true(empty.find("mode=off") >= 0)
    assert_true(empty.find("label=-") >= 0)
    off.print_report()


def test_the_tree_is_the_same_tree_whatever_the_profile_says() raises:
    """A charge is an integer add, so the grower must produce one tree.

    Compared node for node rather than by prediction alone, because two
    different trees can agree on a small sample and a structural comparison
    cannot.
    """
    var n_rows = 1500
    var n_features = 5
    var data = _data(n_rows, n_features)
    var target = _target(n_rows, n_features)
    var grad = List[Float64]()
    var hess = List[Float64]()
    _grad_hess(grad, hess, n_rows, target)
    var params = TreeParams(8, 1, 1.0, 1e-3)

    _ = setenv("MOJOTREES_PHASE_PROFILE", "")
    var plain = grow_tree(data, grad, hess, params)

    var ledger = CegbLedger.none()
    var traced_profile = PhaseProfile(
        PROFILE_ASYNC, SCOPE_TREE, String("async")
    )
    var traced = grow_tree_profiled(
        traced_profile, data, grad, hess, params, ledger
    )
    var fenced_profile = PhaseProfile(
        PROFILE_FENCED, SCOPE_TREE, String("fenced")
    )
    var ledger2 = CegbLedger.none()
    var fenced = grow_tree_profiled(
        fenced_profile, data, grad, hess, params, ledger2
    )

    assert_equal(traced.n_leaves, plain.n_leaves)
    assert_equal(fenced.n_leaves, plain.n_leaves)
    assert_equal(len(traced.value), len(plain.value))
    assert_equal(len(fenced.value), len(plain.value))
    for i in range(len(plain.value)):
        assert_equal(traced.value[i], plain.value[i])
        assert_equal(fenced.value[i], plain.value[i])
        assert_equal(traced.feature[i], plain.feature[i])
        assert_equal(fenced.feature[i], plain.feature[i])
        assert_equal(traced.threshold_bin[i], plain.threshold_bin[i])
        assert_equal(fenced.threshold_bin[i], plain.threshold_bin[i])
        assert_equal(traced.left[i], plain.left[i])
        assert_equal(fenced.left[i], plain.left[i])
        assert_equal(traced.right[i], plain.right[i])
        assert_equal(fenced.right[i], plain.right[i])


def test_predictions_are_byte_identical_across_the_modes() raises:
    """The end-to-end claim, through the trainer that reads the variable.

    `train` runs `_boost_rounds`, which is where the fit-scope profile lives
    and where the gradient fill and the score update are charged, so this is
    the path a profiled benchmark actually takes. The `async` and `fenced`
    fits print one report block each, which is also the only place in this
    file the report is exercised through a real run.
    """
    var n_rows = 1200
    var n_features = 5
    var data = _data(n_rows, n_features)
    var target = _target(n_rows, n_features)
    var params = BoosterParams.default()
    params.n_estimators = 3
    params.tree.num_leaves = 6
    params.tree.min_data_in_leaf = 1

    _ = setenv("MOJOTREES_PHASE_PROFILE", "")
    var plain = train(data, target, SQUARED_ERROR, params)

    _ = setenv("MOJOTREES_PHASE_PROFILE", "async")
    var traced = train(data, target, SQUARED_ERROR, params)

    _ = setenv("MOJOTREES_PHASE_PROFILE", "fenced")
    var fenced = train(data, target, SQUARED_ERROR, params)

    _ = setenv("MOJOTREES_PHASE_PROFILE", "")

    assert_equal(len(traced.trees), len(plain.trees))
    assert_equal(len(fenced.trees), len(plain.trees))
    for r in range(n_rows):
        var want = plain.predict_row(data, r)
        assert_equal(traced.predict_row(data, r), want)
        assert_equal(fenced.predict_row(data, r), want)


def test_host_step_slots_are_total_and_disjoint() raises:
    """Every step index lands in exactly one slot and no step lands in a
    bookend.

    Asserted exhaustively over the whole in-range domain rather than sampled,
    the same way `classify_node`'s partition is, because a slot map that
    quietly aliases step 0 onto the prologue would put per-tree setup and the
    first step's launches on one line and nobody reading the curve would see
    it.
    """
    assert_equal(host_step_slot(-1), HOST_SLOT_PROLOGUE)
    assert_equal(host_step_slot(HOST_STEP_SLOTS), HOST_SLOT_OVERFLOW)
    assert_equal(host_step_slot(HOST_STEP_SLOTS + 1000), HOST_SLOT_OVERFLOW)
    var seen = List[Int]()
    for _ in range(N_HOST_STEP_SLOTS):
        seen.append(0)
    for step in range(HOST_STEP_SLOTS):
        var slot = host_step_slot(step)
        assert_true(slot != HOST_SLOT_PROLOGUE)
        assert_true(slot != HOST_SLOT_EPILOGUE)
        assert_true(slot != HOST_SLOT_OVERFLOW)
        seen[slot] += 1
    for step in range(HOST_STEP_SLOTS):
        assert_equal(seen[host_step_slot(step)], 1)
    # Every slot names itself, so a parser keyed on names finds every line.
    for slot in range(N_HOST_STEP_SLOTS):
        assert_true(host_slot_name(slot) != String("unknown"))
    for span in range(N_HOST_SPANS):
        assert_true(host_span_name(span) != String("unknown"))


def test_an_off_profile_charges_no_host_span() raises:
    """The counter half of "free when off", on the host axis.

    The clock half is structural and is visible in `clock` in four lines: it
    tests the mode and returns 0 without reading `perf_counter_ns`. This is
    the half a test can hold, and it is held the same way the phase axis's is
    -- charge an off profile a large duration and a large count, and require
    every bucket to stay zero.
    """
    var off = PhaseProfile(PROFILE_OFF, SCOPE_FIT, String("off"))
    for span in range(N_HOST_SPANS):
        for slot in range(N_HOST_STEP_SLOTS):
            off.charge_host(span, slot, 1)
            off.charge_host_nanos(span, slot, 1_000_000)
    assert_equal(off.host_total_nanos(), 0)
    assert_equal(off.host_total_calls(), 0)
    for span in range(N_HOST_SPANS):
        assert_equal(off.span_nanos(span), 0)
        assert_equal(off.span_calls(span), 0)
        assert_equal(off.span_max(span), 0)


def test_host_spans_accumulate_and_keep_the_worst_bracket() raises:
    """A charge adds to its own cell, a maximum folds by comparison, and a
    merge does both.

    The maximum is what tells encoding from queue backpressure apart, so a
    merge that summed two worsts would report a bracket that never happened
    and would turn a null into a finding.
    """
    var a = PhaseProfile(PROFILE_ASYNC, SCOPE_FIT, String("a"))
    a.charge_host_nanos(HOST_ENCODE, HOST_SLOT_PROLOGUE, 100)
    a.charge_host_nanos(HOST_ENCODE, HOST_SLOT_PROLOGUE, 300)
    assert_equal(a.host_nanos_of(HOST_ENCODE, HOST_SLOT_PROLOGUE), 400)
    assert_equal(a.host_calls_of(HOST_ENCODE, HOST_SLOT_PROLOGUE), 2)
    assert_equal(a.host_max_of(HOST_ENCODE, HOST_SLOT_PROLOGUE), 300)

    var b = PhaseProfile(PROFILE_ASYNC, SCOPE_FIT, String("b"))
    b.charge_host_nanos(HOST_ENCODE, HOST_SLOT_PROLOGUE, 50)
    b.charge_host_nanos(HOST_READBACK, HOST_SLOT_EPILOGUE, 7000)
    a.merge(b)
    assert_equal(a.host_nanos_of(HOST_ENCODE, HOST_SLOT_PROLOGUE), 450)
    assert_equal(a.host_calls_of(HOST_ENCODE, HOST_SLOT_PROLOGUE), 3)
    # 300 and not 350: a worst folds by comparison.
    assert_equal(a.host_max_of(HOST_ENCODE, HOST_SLOT_PROLOGUE), 300)
    assert_equal(a.span_nanos(HOST_READBACK), 7000)
    assert_equal(a.host_total_nanos(), 7450)

    a.reset()
    assert_equal(a.host_total_nanos(), 0)
    assert_equal(a.host_total_calls(), 0)
    assert_equal(a.span_max(HOST_ENCODE), 0)

    # An unknown span or slot is a call-site error and is refused, on the same
    # principle `charge` refuses an unknown phase. Refused only on an ENABLED
    # profile, because the mode test comes first, which is the whole of the
    # free-when-off claim.
    with assert_raises():
        a.charge_host(N_HOST_SPANS, HOST_SLOT_PROLOGUE, 0)
    with assert_raises():
        a.charge_host(HOST_WAIT, N_HOST_STEP_SLOTS, 0)


def test_the_host_table_is_emitted_in_full() raises:
    """Six span lines and all thirty-five slot lines, zeros included, so a
    per-step curve read off two reports has the same x axis in both."""
    var profile = PhaseProfile(PROFILE_ASYNC, SCOPE_FIT, String("arm"))
    profile.begin_tree(1000, 1000)
    profile.charge_host_nanos(HOST_UPLOAD, HOST_SLOT_PROLOGUE, 25)
    profile.charge_host_nanos(HOST_ALLOC, HOST_SLOT_EPILOGUE, 25)
    profile.charge_host_nanos(HOST_PLAN, host_step_slot(3), 50)
    var text = profile.report()

    assert_equal(len(text.split("phase_profile host ")) - 1, N_HOST_SPANS)
    assert_equal(
        len(text.split("phase_profile hoststep ")) - 1, N_HOST_STEP_SLOTS
    )
    assert_true(text.find("phase_profile hostcolumns kind span calls") >= 0)
    assert_true(text.find("phase_profile hosttotals host_ns=100") >= 0)
    assert_true(text.find("phase_profile hoststep step03 ") >= 0)
    assert_true(text.find("phase_profile hoststep prologue ") >= 0)
    assert_true(text.find("phase_profile hoststep step32plus ") >= 0)

    # The host block must not disturb the phase block's own shape, which an
    # existing test counts by splitting on these three prefixes. Held here as
    # well so that a future line whose kind starts with one of them is caught
    # by the test that added it rather than by the one it broke.
    assert_equal(
        len(text.split("phase_profile row ")) - 1,
        N_PROFILE_PHASES * N_NODE_CLASSES,
    )
    assert_equal(
        len(text.split("phase_profile phase ")) - 1, N_PROFILE_PHASES
    )
    assert_equal(len(text.split("phase_profile class ")) - 1, N_NODE_CLASSES)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
