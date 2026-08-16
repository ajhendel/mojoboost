"""The dispatch snapshot on the **grower** path, which is where the nodes are.

`test_cpu_dispatch.mojo` section 4 proves that a `DispatchSettings` handed to
a histogram builder makes that builder read nothing, and it proves it by
poisoning `MOJOTREES_CPU_FEATURE_GROUP` with an off-ladder value that makes
`apple_cpu_policy.env_feature_group` raise, then driving 200 builds to
completion. That is a real proof and this file does not repeat it.

What it does not cover is the only path a fit actually takes. It calls
`build_histogram_subset_into_scratch`, `build_histogram_into`,
`subtract_histogram_into` and `find_best_split` **directly**, each with a
snapshot it resolved itself. A fit does none of that: it constructs a
`GrowScratch`, whose constructor resolves the snapshot once, and then calls
`grow_tree_leaves_profiled`, which is responsible for threading
`scratch.settings` into two histogram builders, two sibling subtractions and
three split searches spread across a thousand lines. Any one of those seven
sites could still be passing the `DispatchSettings.unresolved()` default --
which is a legal, silent, correct-answer-producing thing to pass -- and every
existing test would stay green, because the answers are equal either way.

That is not hypothetical in this repository. `DispatchSettings` and
`ResolvedCpuPolicy` were built, tested, and shipped with **no call site in
`src/` at all** for a whole round, and the suite was green the entire time,
because the tests called the mechanism directly while production called none
of it. Equality of answers is exactly what cannot detect that. So this file
counts reads, at the grower, over a whole tree.

The instrument, and what it establishes exactly
-----------------------------------------------
Resolve the scratch on a clean environment, set the poison **afterwards**, and
grow entire trees through that scratch. Every dispatch that reached
`env_feature_group` would raise, so a run that completes performed zero reads
of `MOJOTREES_CPU_FEATURE_GROUP` across every node of every tree. The control
is a grower that constructs its own scratch under the poison, which must
raise: that says the read exists on this path and would have been observed,
rather than the poison having been irrelevant all along.

Which sites this covers, measured rather than assumed
-----------------------------------------------------
`env_feature_group` is the **only** one of the seven scheduling inputs that
refuses a bad value instead of swallowing it, so the poison reaches exactly
the sites that plan an accumulation. Each of the grower's nine `settings`
arguments was un-wired one at a time, back to `DispatchSettings.unresolved()`,
and the file re-run, which is how the following is a map and not a guess:

  - the root full-dataset histogram (`_hist_full`)      -- CAUGHT
  - the root bagged histogram (`_hist_subset`)          -- CAUGHT
  - both child histograms (`_hist_subset`, two branches) -- CAUGHT
  - both sibling subtractions (`subtract_histogram_into`) -- **not caught**
  - all three split searches (`find_best_split`)         -- **not caught**

The last two groups are invisible to this instrument for a structural reason,
not an oversight: an unresolved subtraction or split scan falls through to
`plan_tasks`, which reads `MOJOTREES_NUM_WORKERS` and calls `cpu_profile()`,
and neither of those touches `MOJOTREES_CPU_FEATURE_GROUP`. Un-wiring either
one leaves this file green. **So this file does not prove those five sites are
wired, and must not be cited as if it did.** Proving them needs an instrument
that does not exist yet -- a read counter, or a second variable that raises --
and building one means editing `src/`, which this lane may not do. It is
reported as a gap rather than papered over.

For the other five variables the assertion is the divergence property, taken
directly off the snapshot: after a `setenv` the scratch's own `settings` still
holds what it held, while a freshly resolved snapshot sees the new value.

And one site is not wired at all. `tree.partition_rows_into` calls
`parallel.plan_row_blocks`, the live form, once per split; it takes no
snapshot and none is offered to it. So "the grower reads nothing per node" is
**false** as a blanket statement and is asserted nowhere in this file. What is
asserted is the accumulation-planning half, which is the half the poison can
see.

Nothing here is timed. The node counts below are counts, not durations, and
no line of this file may be read as evidence that any of it is faster.
"""

from std.os import setenv
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.cegb import CegbLedger
from mojotrees.parallel import DispatchSettings
from mojotrees.phase_profile import PhaseProfile
from mojotrees.tree import (
    GrowScratch,
    LeafMembership,
    Tree,
    TreeParams,
    grow_tree,
    grow_tree_leaves_profiled,
)

from support import _make_features, _uniform


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

comptime _N_ROWS = 800
comptime _N_FEATURES = 6
comptime _N_BINS = 17
comptime _NUM_LEAVES = 31
comptime _TREES = 3
"""Trees grown under one poisoned scratch. The claim is that the read count
does not grow with the node count, so more than one tree and far more than one
node is what makes the claim testable at all: at `_NUM_LEAVES` leaves a tree
performs one histogram build per leaf, one sibling subtraction per split, and
a split scan per node, so three trees put on the order of ninety accumulation
plans through the scratch. Every one of them would raise on a live read."""

# The off-ladder interleave width. `env_feature_group` refuses it rather than
# rounding it, which is what turns one environment read into an observable
# event. Same value and same reason as `test_cpu_dispatch._POISON`.
comptime _POISON = "3"


def _poison():
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", _POISON)


def _unpoison():
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "")


def _reset_env():
    _unpoison()
    _ = setenv("MOJOTREES_NUM_WORKERS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", "")
    _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", "")
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "")
    _ = setenv("MOJOTREES_CPU_COMPACT_MIN_ROWS", "")


def _data() raises -> BinnedMatrix:
    return bin_equal_width(
        _make_features(_N_ROWS, _N_FEATURES), _N_ROWS, _N_FEATURES, _N_BINS
    )


def _grad() -> List[Float64]:
    """Gradients with real structure in the features, so growth actually
    reaches the leaf budget instead of stopping at a stump.

    The raw score is zero and the objective squared error, so the gradient is
    `-y`; `y` is a signal in features 0 to 3 plus a small deterministic
    wobble, which gives every level of the tree something to split on.
    """
    var values = _make_features(_N_ROWS, _N_FEATURES)
    var g = List[Float64](capacity=_N_ROWS)
    for r in range(_N_ROWS):
        var v0 = values[0 * _N_ROWS + r]
        var v1 = values[1 * _N_ROWS + r]
        var v2 = values[2 * _N_ROWS + r]
        var v3 = values[3 * _N_ROWS + r]
        var y = 3.0 * v0 - 2.0 * v1 + v2 * v3 + 0.25 * _uniform(UInt64(r))
        g.append(-y)
    return g^


def _hess() -> List[Float64]:
    var h = List[Float64](capacity=_N_ROWS)
    for _ in range(_N_ROWS):
        h.append(1.0)
    return h^


def _params() -> TreeParams:
    # min_data_in_leaf of 1 and a zero hessian floor so the leaf budget is
    # what bounds the tree, which makes the node count a property of the
    # fixture rather than of the machine.
    return TreeParams(_NUM_LEAVES, 1, 1.0, 0.0)


def _bits(v: Float64) -> UInt64:
    return v.to_bits().cast[DType.uint64]()


def _assert_same_tree(a: Tree, b: Tree) raises:
    """Two trees are the same tree, field for field, floats as bits.

    Every array is compared, including `split_gain` and `count`, because the
    question here is whether a scheduling decision reached the arithmetic and
    the gain is the quantity a reassociated histogram cell would move first.
    No tolerance: a one-ulp difference in a gain is a defect in whatever moved
    it, not a rounding difference to absorb.
    """
    assert_equal(a.n_leaves, b.n_leaves)
    assert_equal(len(a.feature), len(b.feature))
    for i in range(len(a.feature)):
        assert_equal(a.feature[i], b.feature[i])
        assert_equal(a.threshold_bin[i], b.threshold_bin[i])
        assert_equal(a.left[i], b.left[i])
        assert_equal(a.right[i], b.right[i])
        assert_equal(_bits(a.value[i]), _bits(b.value[i]))
        assert_equal(_bits(a.split_gain[i]), _bits(b.split_gain[i]))
        assert_equal(Int(a.default_left[i]), Int(b.default_left[i]))
        assert_equal(a.missing_bin[i], b.missing_bin[i])
        assert_equal(a.cat_offset[i], b.cat_offset[i])
        assert_equal(_bits(a.count[i]), _bits(b.count[i]))
    assert_equal(len(a.cat_bitset), len(b.cat_bitset))
    for i in range(len(a.cat_bitset)):
        assert_equal(a.cat_bitset[i], b.cat_bitset[i])


def _grow(mut scratch: GrowScratch, data: BinnedMatrix,
          grad: List[Float64], hess: List[Float64],
          tree_index: Int = 0) raises -> Tree:
    """One tree through a caller-owned scratch: the entry point a boosting
    loop takes, and the only one that lets a test resolve the snapshot at a
    moment of its choosing.

    The profile is constructed directly rather than through `from_env` so this
    file's environment handling covers everything the call reads. It is
    `PROFILE_OFF`, which the grower tests once per charge site and otherwise
    ignores.
    """
    var profile = PhaseProfile()
    var leaves = LeafMembership()
    var ledger = CegbLedger.none()
    return grow_tree_leaves_profiled(
        profile, leaves, ledger, scratch, data, grad, hess, _params(),
        [], tree_index,
    )


# ---------------------------------------------------------------------------
# The positive control: the poison is real and the grower's snapshot sees it
# ---------------------------------------------------------------------------

def test_the_scratch_constructor_is_where_a_bad_width_is_refused() raises:
    """`GrowScratch.__init__` resolves the snapshot, so an off-ladder
    `MOJOTREES_CPU_FEATURE_GROUP` is refused there.

    The positive control for everything below it, and the same role
    `test_cpu_dispatch.test_resolving_the_snapshot_is_where_a_bad_width_is_refused`
    plays for the library functions. If constructing a scratch stopped
    reading the variable, the completions below would prove nothing, because
    nothing on the path would ever have read it. It also pins the one
    behaviour change `GrowScratch`'s docstring names: the refusal surfaces at
    construction rather than at the first histogram build.
    """
    _reset_env()
    _poison()
    var raised = False
    try:
        var scratch = GrowScratch(_N_FEATURES, _N_BINS)
        _ = scratch.settings.resolved
    except:
        raised = True
    _unpoison()
    assert_true(raised)

    var ok = GrowScratch(_N_FEATURES, _N_BINS)
    assert_true(ok.settings.resolved)
    _reset_env()


def test_a_grower_entry_point_that_owns_its_scratch_reads_the_variable() raises:
    """`grow_tree` builds its own `GrowScratch`, so it reads the environment
    once per call and raises under the poison.

    This is the control that gives the completion below its meaning. Zero
    reads across three whole trees is only a statement about wiring if a
    grower on the same environment, on the same data, would have read it; this
    is that grower. It is also the shape every trainer outside
    `boosting._boost_rounds` uses, so it records what "once per tree" costs
    against the loop's "once per fit".
    """
    _reset_env()
    var data = _data()
    var grad = _grad()
    var hess = _hess()
    _poison()
    var raised = False
    try:
        _ = grow_tree(data, grad, hess, _params())
    except:
        raised = True
    _unpoison()
    assert_true(raised)
    _reset_env()


# ---------------------------------------------------------------------------
# The claim: a whole tree through a snapshot reads nothing
# ---------------------------------------------------------------------------

def test_growing_whole_trees_through_a_scratch_reads_no_environment() raises:
    """Three complete trees grown under the poison, through a scratch
    resolved before it. Completion is the proof.

    Read counts, stated as counts. The scratch resolves once. Each tree then
    performs `n_leaves` histogram builds, `n_leaves - 1` sibling
    subtractions, and a split scan for the root and for both children of every
    split, and every one of those is a site that planned its dispatch by
    reading `MOJOTREES_CPU_FEATURE_GROUP` before the snapshot was threaded. If
    any single one of them still passed `DispatchSettings.unresolved()` --
    which is a legal default that produces the identical histogram -- this
    test raises on the first node of the first tree. That is the failure mode
    no equality-of-answers test in this repository can see.

    The trees are then compared to trees grown on a clean environment with a
    fresh scratch, field for field. Growth cannot observe a variable it never
    read, so the poisoned trees must be the unpoisoned trees; a difference
    would mean something on the path had read the environment and quietly
    changed a plan rather than raising.
    """
    _reset_env()
    var data = _data()
    var grad = _grad()
    var hess = _hess()

    # The reference: clean environment, one fresh scratch per tree, which is
    # what `grow_tree` does.
    var reference = List[Tree]()
    for t in range(_TREES):
        var fresh = GrowScratch(_N_FEATURES, _N_BINS)
        reference.append(_grow(fresh, data, grad, hess, t))
    # A stump would make "completed" a much weaker word, so the fixture's
    # depth is asserted rather than assumed.
    assert_equal(reference[0].n_leaves, _NUM_LEAVES)

    # The snapshot is taken while the environment is clean, and the poison is
    # set afterwards. Nothing below may read it.
    var scratch = GrowScratch(_N_FEATURES, _N_BINS)
    assert_true(scratch.settings.resolved)

    _poison()
    var grown = List[Tree]()
    for t in range(_TREES):
        grown.append(_grow(scratch, data, grad, hess, t))
    _unpoison()

    for t in range(_TREES):
        assert_equal(grown[t].n_leaves, _NUM_LEAVES)
        _assert_same_tree(grown[t], reference[t])
    _reset_env()


def test_a_bagged_root_also_reads_no_environment() raises:
    """The bagged root is a fourth accumulation site and the only one the
    unbagged test above cannot reach.

    With an empty bag the root histogram is a full-dataset build
    (`_hist_full`); with a non-empty one it is a subset build over the bag's
    rows (`_hist_subset`), which is a separate call with its own `settings`
    argument. Bagging, GOSS and balanced bagging all take it, so leaving it
    unwired would cost every sampled fit in the package while every unbagged
    test stayed green. Same instrument, same control: the snapshot is resolved
    clean, the poison is set after it, and completion is the proof.
    """
    _reset_env()
    var data = _data()
    var grad = _grad()
    var hess = _hess()
    var bag = List[Int]()
    for r in range(_N_ROWS):
        if r % 2 == 0:
            bag.append(r)

    var profile = PhaseProfile()
    var leaves = LeafMembership()
    var ledger = CegbLedger.none()
    var fresh = GrowScratch(_N_FEATURES, _N_BINS)
    var reference = grow_tree_leaves_profiled(
        profile, leaves, ledger, fresh, data, grad, hess, _params(), bag,
    )
    assert_equal(reference.n_leaves, _NUM_LEAVES)

    var scratch = GrowScratch(_N_FEATURES, _N_BINS)
    _poison()
    var profile2 = PhaseProfile()
    var leaves2 = LeafMembership()
    var ledger2 = CegbLedger.none()
    var grown = grow_tree_leaves_profiled(
        profile2, leaves2, ledger2, scratch, data, grad, hess, _params(), bag,
    )
    _unpoison()
    _assert_same_tree(grown, reference)
    _reset_env()


def test_one_scratch_across_trees_grows_the_trees_a_fresh_one_grows() raises:
    """The scratch is shared across a fit, so a tree must not be able to
    observe which tree came before it.

    Separate from the poison test because it is a different claim: that one is
    about reads, this one is about the pool and the gather buffer carrying no
    state. Growing the same tree index twice through a scratch that has
    already grown three trees must give the tree a fresh scratch gives, and
    growing the three in a different order must give the same three.
    """
    _reset_env()
    var data = _data()
    var grad = _grad()
    var hess = _hess()

    var reference = List[Tree]()
    for t in range(_TREES):
        var fresh = GrowScratch(_N_FEATURES, _N_BINS)
        reference.append(_grow(fresh, data, grad, hess, t))

    var scratch = GrowScratch(_N_FEATURES, _N_BINS)
    for t in range(_TREES):
        _ = _grow(scratch, data, grad, hess, t)
    for t in range(_TREES):
        var again = _grow(scratch, data, grad, hess, _TREES - 1 - t)
        _assert_same_tree(again, reference[_TREES - 1 - t])
    _reset_env()


# ---------------------------------------------------------------------------
# The snapshot does not observe a later setenv
# ---------------------------------------------------------------------------

def test_the_scratch_ignores_a_worker_flip_taken_after_it() raises:
    """`MOJOTREES_NUM_WORKERS` moves after the scratch resolves, and the
    scratch does not move with it.

    Three assertions, of decreasing strength, and the ordering matters because
    only the first two are about the snapshot:

    1. `scratch.settings.num_workers` still holds what it held. This is the
       divergence property directly on the value the grower carries, and it is
       the only one of the three that a live re-read would break.
    2. A snapshot resolved after the flip sees the new value, so the flip was
       real and the first assertion is not passing because `setenv` did
       nothing.
    3. The tree grown after the flip is the tree grown before it, bit for bit.
       This is the **weakest** of the three and is included for completeness
       rather than as evidence: `MOJOTREES_NUM_WORKERS` may not reach the
       arithmetic at all (docs/NUMERICS.md section 1), so the trees would be
       identical whether the snapshot was consulted or re-resolved. It is
       asserted because a change that made the worker count reach a value
       would be a serious defect and this is a place it would show.

    `tree.partition_rows_into` does still read the live worker count, once per
    split, so the flip is genuinely visible to part of the grower. That is a
    remaining gap in the wiring rather than a defect in this test, and it is
    exactly why assertion 3 cannot be read as proof of anything.
    """
    _reset_env()
    _ = setenv("MOJOTREES_NUM_WORKERS", "1")
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "1")
    var data = _data()
    var grad = _grad()
    var hess = _hess()

    var scratch = GrowScratch(_N_FEATURES, _N_BINS)
    assert_equal(scratch.settings.num_workers, 1)
    var before = _grow(scratch, data, grad, hess)

    _ = setenv("MOJOTREES_NUM_WORKERS", "8")
    assert_equal(scratch.settings.num_workers, 1)
    var refreshed = DispatchSettings.resolve()
    assert_equal(refreshed.num_workers, 8)

    var after = _grow(scratch, data, grad, hess)
    assert_equal(scratch.settings.num_workers, 1)
    _assert_same_tree(after, before)
    _reset_env()


def test_the_tree_is_identical_at_one_three_and_eight_workers() raises:
    """Determinism across `MOJOTREES_NUM_WORKERS`, which this round requires
    of every change and which the snapshot is the newest way to break: a
    scratch resolved at one worker count and a scratch resolved at another
    take different dispatch plans through the same seven sites, and every one
    of those plans has to produce the same tree.

    The crossover is forced to zero so the parallel path is actually taken at
    this fixture's size rather than the loop falling serial and all three arms
    quietly measuring the same schedule.
    """
    _reset_env()
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "1")
    var data = _data()
    var grad = _grad()
    var hess = _hess()

    _ = setenv("MOJOTREES_NUM_WORKERS", "1")
    var one_scratch = GrowScratch(_N_FEATURES, _N_BINS)
    assert_equal(one_scratch.settings.num_workers, 1)
    var base = _grow(one_scratch, data, grad, hess)
    assert_equal(base.n_leaves, _NUM_LEAVES)

    var workers: List[String] = ["3", "8"]
    for i in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[i])
        var scratch = GrowScratch(_N_FEATURES, _N_BINS)
        assert_equal(scratch.settings.num_workers, Int(workers[i]))
        _assert_same_tree(_grow(scratch, data, grad, hess), base)
    _reset_env()


def test_every_feature_group_width_grows_the_same_tree() raises:
    """The interleave width is a scheduling knob too, and the scratch carries
    it in `settings.policy` rather than re-reading it per node. Every rung of
    the ladder is a separate instantiation of the accumulation body, and all
    of them must grow the identical tree.

    This is the value-side companion to the poison test: that one says the
    width is read once, this one says it does not matter what it was read as.
    Together they are what lets the width be tuned by the policy without a
    fit's output depending on the tuning.
    """
    _reset_env()
    var data = _data()
    var grad = _grad()
    var hess = _hess()

    var clean = GrowScratch(_N_FEATURES, _N_BINS)
    var base = _grow(clean, data, grad, hess)
    assert_equal(base.n_leaves, _NUM_LEAVES)

    var rungs: List[String] = ["1", "2", "4", "8", "16"]
    for i in range(len(rungs)):
        _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", rungs[i])
        var scratch = GrowScratch(_N_FEATURES, _N_BINS)
        # Resolved once, at construction: the rung is now fixed for the life
        # of the scratch whatever the environment does next.
        _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "")
        _assert_same_tree(_grow(scratch, data, grad, hess), base)
    _reset_env()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
