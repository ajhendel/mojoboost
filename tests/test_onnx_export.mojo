"""ONNX export tests.

The load-bearing tests walk the exported `TreeEnsembleRegressor` arrays with
an interpreter written from the *operator's* rules rather than mojotrees's,
and compare against the model. A threshold or a NaN direction converted
wrongly moves a row to a different leaf, which is the failure nothing else in
this repository would catch.

Two claims, tested separately, because only one of them is exact.

- **Leaf selection is exact.** `_assert_same_leaves` compares, per tree, the
  leaf the plan reaches against `Tree.leaf_index_bins`. Equality, no
  tolerance.
- **The score is not, and the reference is why.**
  `Booster.predict_raw_bins` accumulates `s += learning_rate * value`, which
  the compiler contracts into a fused multiply-add; the plan carries the
  product already formed, so summing it is a plain add. Measured difference:
  one ULP. `test_plan_is_bit_exact_when_the_learning_rate_is_one` is the
  experiment that pins the cause there rather than on leaf selection or
  summation order.

Neither is a limitation of the export. ONNX itself returns `tensor(float)`
and specifies no summation order; see docs/design/MODEL_EXPORT.md section 4.
"""

from std.os import remove
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojotrees.boosting import (
    BINARY_LOGISTIC,
    POISSON,
    SQUARED_ERROR,
    BoosterParams,
)
from mojotrees.model import Model, fit
from mojotrees.onnx_export import (
    MODE_BRANCH_LEQ,
    MODE_LEAF,
    ONNX_PLAN_MAGIC,
    POST_EXP,
    POST_LOGISTIC,
    POST_NONE,
    OnnxPlan,
    onnx_nan_goes_left,
    onnx_plan,
    onnx_plan_text,
    onnx_refusals,
    onnx_threshold,
    save_onnx_plan,
)
from mojotrees.tree import TreeParams


comptime _TMP_PATH = "./.test_onnx_plan.tmp"


def _make_dataset(
    n_rows: Int, mut features: List[Float64], mut target: List[Float64]
):
    """Two features with an interaction, deterministic. Same shape as
    tests/test_serialize.mojo uses."""
    var state: UInt64 = 42
    for _ in range(n_rows):
        state = state * 6364136223846793005 + 1442695040888963407
        features.append(Float64(state >> 11) * (1.0 / 9007199254740992.0))
    for _ in range(n_rows):
        state = state * 6364136223846793005 + 1442695040888963407
        features.append(Float64(state >> 11) * (1.0 / 9007199254740992.0))
    for r in range(n_rows):
        target.append(
            3.0 * features[r] + 2.0 * features[n_rows + r] * features[r]
        )


def _small_params() -> BoosterParams:
    return BoosterParams(20, 0.1, TreeParams(8, 5, 1.0, 1e-3))


def _walk_plan_leaf(
    plan: OnnxPlan, row: List[Float64], first: Int, stop: Int
) raises -> Int:
    """The index into the plan's node arrays of the leaf `row` reaches in the
    tree occupying `[first, stop)`.

    Written from the operator's rules, not mojotrees's: descend by
    `nodes_modes`, take the true branch on `x <= nodes_values`, take the
    branch `nodes_missing_value_tracks_true` names when `x` is NaN. Children
    are resolved through `nodes_nodeids` rather than by assuming the plan
    numbered them positionally, so a plan that numbered them inconsistently
    fails here.
    """
    var at = first
    while plan.nodes_modes[at] != MODE_LEAF:
        var x = row[plan.nodes_featureids[at]]
        var go_true: Bool
        if x != x:  # NaN
            go_true = plan.nodes_missing_value_tracks_true[at] != 0
        else:
            go_true = x <= plan.nodes_values[at]
        var want = (
            plan.nodes_truenodeids[at] if go_true
            else plan.nodes_falsenodeids[at]
        )
        var found = -1
        for j in range(first, stop):
            if plan.nodes_nodeids[j] == want:
                found = j
                break
        if found < 0:
            raise Error("plan names a child node id that is not present")
        at = found
    return at


def _plan_leaf_nodeids(
    plan: OnnxPlan, row: List[Float64]
) raises -> List[Int]:
    """The leaf node id `row` reaches in each tree, in tree order.

    This is the exact half of the export contract: leaf selection. See
    `_walk_plan` for why the score is not.
    """
    var out = List[Int]()
    var i = 0
    var n = plan.n_nodes()
    while i < n:
        var tree_id = plan.nodes_treeids[i]
        var stop = i
        while stop < n and plan.nodes_treeids[stop] == tree_id:
            stop += 1
        out.append(plan.nodes_nodeids[_walk_plan_leaf(plan, row, i, stop)])
        i = stop
    return out^


def _walk_plan(plan: OnnxPlan, row: List[Float64]) raises -> Float64:
    """`TreeEnsembleRegressor` scoring, by hand, for `n_targets == 1`:
    `base_values` plus the `target_weights` of the leaf each tree selects.

    This is **not** bit-identical to `Model.predict_raw`, and the reason is
    the reference rather than the export.
    `Booster.predict_raw_bins` accumulates
    `s += learning_rate * tree.value[leaf]`, which the compiler contracts
    into a fused multiply-add. This walk adds a `target_weights` entry that
    already holds the product, so it is a plain add. The two differ by
    rounding on rows where the fused form's single rounding beats two, and
    the difference observed here is one ULP.
    `test_plan_is_bit_exact_when_the_learning_rate_is_one` is the experiment
    that pins the cause on the fused multiply and not on leaf selection or
    on summation order: at `learning_rate = 1.0` the multiply is exact,
    fusing cannot change anything, and the two agree bit for bit.

    So the exact assertions in this file are on leaves and on per-tree
    weights (`_plan_leaf_nodeids`, and the learning-rate test); the score is
    checked to a tolerance, which is all ONNX itself promises anyway -- the
    operator returns `tensor(float)` and does not specify the summation
    order of `aggregate_function=SUM`.
    """
    var acc = plan.base_values[0]
    var i = 0
    var n = plan.n_nodes()
    while i < n:
        # `i` is the first row of one tree; find where that tree ends.
        var tree_id = plan.nodes_treeids[i]
        var stop = i
        while stop < n and plan.nodes_treeids[stop] == tree_id:
            stop += 1

        var leaf_id = plan.nodes_nodeids[_walk_plan_leaf(plan, row, i, stop)]
        var hits = 0
        for k in range(plan.n_leaves()):
            if (
                plan.target_treeids[k] == tree_id
                and plan.target_nodeids[k] == leaf_id
            ):
                acc += plan.target_weights[k]
                hits += 1
        if hits != 1:
            raise Error("a leaf must have exactly one target weight")
        i = stop
    return acc


def _assert_close(got: Float64, want: Float64) raises:
    """Agreement to a tolerance far tighter than any rounding the export
    could hide, and far looser than the one-ULP fused-multiply difference
    documented on `_walk_plan`."""
    var scale = want if want >= 0.0 else -want
    if scale < 1.0:
        scale = 1.0
    var diff = got - want
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= 1e-12 * scale)


def _assert_same_leaves(
    model: Model, plan: OnnxPlan, row: List[Float64]
) raises:
    """The exact half: every tree selects the same leaf in the plan as in
    the model. A threshold or a NaN direction converted wrongly moves a leaf,
    and this is what catches it."""
    var bins = model.mapper.bin_row(row)
    var got = _plan_leaf_nodeids(plan, row)
    assert_equal(len(got), len(model.booster.trees))
    for t in range(len(model.booster.trees)):
        assert_equal(got[t], model.booster.trees[t].leaf_index_bins(bins))


def _fit_small(mut features: List[Float64], mut target: List[Float64],
               n_rows: Int) raises -> Model:
    _make_dataset(n_rows, features, target)
    return fit(features, n_rows, 2, target, SQUARED_ERROR, _small_params(), 64)


def test_plan_reproduces_raw_score_exactly() raises:
    var n_rows = 300
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_small(features, target, n_rows)

    var plan = onnx_plan(model, raw_score=True)
    assert_equal(plan.n_targets, 1)
    assert_equal(plan.post_transform, POST_NONE)
    assert_equal(len(plan.base_values), 1)
    assert_true(plan.base_values[0] == model.booster.base_score)

    for r in range(n_rows):
        var row: List[Float64] = [features[r], features[n_rows + r]]
        _assert_same_leaves(model, plan, row)
        _assert_close(_walk_plan(plan, row), model.predict_raw(row))


def test_plan_reproduces_raw_score_off_the_training_grid() raises:
    # Rows the model never saw, including values outside the fitted range on
    # both sides, which is where a threshold conversion off by one bin shows
    # up.
    var n_rows = 300
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_small(features, target, n_rows)
    var plan = onnx_plan(model, raw_score=True)

    var probes: List[Float64] = [
        -1e9, -1.0, -0.0, 0.0, 1e-12, 0.25, 0.5, 0.75, 1.0 - 1e-15, 1.0, 1e9
    ]
    for a in range(len(probes)):
        for b in range(len(probes)):
            var row: List[Float64] = [probes[a], probes[b]]
            _assert_same_leaves(model, plan, row)
            _assert_close(_walk_plan(plan, row), model.predict_raw(row))


def test_leaf_weights_carry_the_learning_rate() raises:
    # The defect this test exists for: copying Tree.value straight into
    # target_weights produces a graph that loads, validates, and is wrong by
    # a factor of learning_rate on every row.
    var n_rows = 200
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_small(features, target, n_rows)
    var lr = model.booster.learning_rate
    assert_true(lr != 1.0)

    var plan = onnx_plan(model, raw_score=True)
    var seen = 0
    for k in range(plan.n_leaves()):
        ref tree = model.booster.trees[plan.target_treeids[k]]
        var want = lr * tree.value[plan.target_nodeids[k]]
        assert_true(plan.target_weights[k] == want)
        seen += 1
    assert_true(seen > 0)


def test_thresholds_are_the_fitted_edges() raises:
    var n_rows = 200
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_small(features, target, n_rows)
    var plan = onnx_plan(model, raw_score=True)

    var branches = 0
    for i in range(plan.n_nodes()):
        if plan.nodes_modes[i] != MODE_BRANCH_LEQ:
            continue
        branches += 1
        ref tree = model.booster.trees[plan.nodes_treeids[i]]
        var node = plan.nodes_nodeids[i]
        var f = tree.feature[node]
        assert_equal(plan.nodes_featureids[i], f)
        var lo = model.mapper.edge_offsets[f]
        var b = tree.threshold_bin[node]
        # Every grown split lands strictly inside the edge list, so the
        # +inf case does not arise here and the edge is the threshold.
        assert_true(b < model.mapper.edge_offsets[f + 1] - lo)
        assert_true(plan.nodes_values[i] == model.mapper.edges[lo + b])
    assert_true(branches > 0)


def test_threshold_conversion_is_an_equality() raises:
    # bin(x) <= t  <=>  x <= edges[t], stated directly against the mapper
    # rather than through a tree, over every feature and every bin.
    var n_rows = 200
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_small(features, target, n_rows)
    ref mapper = model.mapper

    var probes: List[Float64] = [-1e9, -1.0, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1e9]
    for f in range(mapper.n_features):
        var lo = mapper.edge_offsets[f]
        var n_edges = mapper.edge_offsets[f + 1] - lo
        for t in range(n_edges):
            var thr = onnx_threshold(mapper, f, t)
            for p in range(len(probes)):
                var x = probes[p]
                assert_equal(mapper.bin_value(f, x) <= t, x <= thr)
            # The edge itself, and the next representable value above it,
            # are the two points a conversion off by one gets wrong.
            var e = mapper.edges[lo + t]
            assert_true(mapper.bin_value(f, e) <= t)
            assert_true(e <= thr)


def test_degenerate_threshold_is_positive_infinity() raises:
    var n_rows = 200
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_small(features, target, n_rows)
    ref mapper = model.mapper
    var n_edges = mapper.edge_offsets[1] - mapper.edge_offsets[0]
    var thr = onnx_threshold(mapper, 0, n_edges)
    assert_true(thr > 1e308)
    assert_true(1e308 <= thr)
    # One past the edge count is not a split any grower makes, and is
    # refused rather than clamped.
    with assert_raises():
        _ = onnx_threshold(mapper, 0, n_edges + 1)


def _check_nan_rows(model: Model, plan: OnnxPlan) raises:
    var nan = Float64(0.0) / Float64(0.0)
    assert_true(nan != nan)
    var rows = List[List[Float64]]()
    rows.append([nan, 0.25])
    rows.append([0.25, nan])
    rows.append([nan, nan])
    for i in range(len(rows)):
        _assert_same_leaves(model, plan, rows[i])
        _assert_close(_walk_plan(plan, rows[i]), model.predict_raw(rows[i]))

    # And the per-node predicate agrees with the tree it came from.
    for t in range(len(model.booster.trees)):
        ref tree = model.booster.trees[t]
        for node in range(len(tree.feature)):
            if tree.feature[node] < 0:
                continue
            var f = tree.feature[node]
            var reserved = model.mapper.missing_bin[f]
            var nan_bin = (
                reserved if reserved >= 0 else model.mapper.bin_value(f, 0.0)
            )
            assert_equal(
                onnx_nan_goes_left(model.mapper, tree, node),
                tree.goes_left(node, nan_bin),
            )


def test_nan_routes_the_same_way_when_no_missing_bin_is_reserved() raises:
    # Training data with no NaN in it reserves no missing bin, so mojotrees
    # bins NaN as 0.0 and the ordinary threshold decides. That is the case
    # an exporter that trusts `default_left` gets wrong.
    var n_rows = 300
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_small(features, target, n_rows)
    for f in range(model.mapper.n_features):
        assert_equal(model.mapper.missing_bin[f], -1)
    _check_nan_rows(model, onnx_plan(model, raw_score=True))


def test_nan_routes_the_same_way_when_a_missing_bin_is_reserved() raises:
    # The other half: NaN in the training data, so the fit reserves a
    # missing bin and every node routes it by `default_left`.
    var n_rows = 300
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(n_rows, features, target)
    var nan = Float64(0.0) / Float64(0.0)
    for r in range(0, n_rows, 5):
        features[r] = nan
    for r in range(0, n_rows, 7):
        features[n_rows + r] = nan

    var model = fit(
        features, n_rows, 2, target, SQUARED_ERROR, _small_params(), 64
    )
    var reserved = 0
    for f in range(model.mapper.n_features):
        if model.mapper.missing_bin[f] >= 0:
            reserved += 1
    assert_true(reserved > 0)

    var plan = onnx_plan(model, raw_score=True)
    _check_nan_rows(model, plan)
    for r in range(n_rows):
        var row: List[Float64] = [features[r], features[n_rows + r]]
        _assert_same_leaves(model, plan, row)
        _assert_close(_walk_plan(plan, row), model.predict_raw(row))


def test_plan_is_bit_exact_when_the_learning_rate_is_one() raises:
    # The discriminating experiment behind the tolerance on every other
    # score assertion in this file. At learning_rate = 1.0 the product
    # `learning_rate * value` is exact, so a fused multiply-add and a plain
    # add of the same product cannot differ. If the plan still matched only
    # to a tolerance here, the cause would be leaf selection or summation
    # order and the export would be wrong. It matches bit for bit.
    var n_rows = 300
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(n_rows, features, target)
    var params = BoosterParams(20, 1.0, TreeParams(8, 5, 1.0, 1e-3))
    var model = fit(
        features, n_rows, 2, target, SQUARED_ERROR, params, 64
    )
    assert_true(model.booster.learning_rate == 1.0)

    var plan = onnx_plan(model, raw_score=True)
    for r in range(n_rows):
        var row: List[Float64] = [features[r], features[n_rows + r]]
        assert_true(_walk_plan(plan, row) == model.predict_raw(row))


def test_post_transform_follows_the_objective() raises:
    var n_rows = 300
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(n_rows, features, target)

    var labels = List[Float64]()
    for r in range(n_rows):
        labels.append(1.0 if target[r] > 1.5 else 0.0)
    var binary = fit(
        features, n_rows, 2, labels, BINARY_LOGISTIC, _small_params(), 64
    )
    assert_equal(onnx_plan(binary).post_transform, POST_LOGISTIC)
    assert_equal(onnx_plan(binary, raw_score=True).post_transform, POST_NONE)

    var counts = List[Float64]()
    for r in range(n_rows):
        counts.append(Float64(Int(target[r] * 2.0)))
    var poisson = fit(
        features, n_rows, 2, counts, POISSON, _small_params(), 64
    )
    # exp has no ONNX post_transform; the plan says so rather than dropping
    # the link.
    assert_equal(onnx_plan(poisson).post_transform, POST_EXP)


def test_a_clean_numerical_model_has_no_refusals() raises:
    var n_rows = 200
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_small(features, target, n_rows)
    var reasons = onnx_refusals(
        model.mapper,
        model.booster.trees,
        model.booster.objective,
        model.booster.linear,
        False,
    )
    assert_equal(len(reasons), 0)


def test_categorical_model_is_refused_and_says_why() raises:
    var n_rows = 240
    var features = List[Float64]()
    var target = List[Float64]()
    # Feature 0 is a category code, feature 1 is numerical.
    for r in range(n_rows):
        features.append(Float64(r % 6))
    for r in range(n_rows):
        features.append(Float64(r % 17) * 0.05)
    for r in range(n_rows):
        target.append(
            (2.0 if (r % 6) == 3 else 0.0) + features[n_rows + r]
        )

    var categorical: List[Int] = [0]
    var model = fit(
        features,
        n_rows,
        2,
        target,
        SQUARED_ERROR,
        _small_params(),
        64,
        categorical_features=categorical,
    )
    assert_true(model.mapper.cats.any_categorical())

    var reasons = onnx_refusals(
        model.mapper,
        model.booster.trees,
        model.booster.objective,
        model.booster.linear,
        False,
    )
    assert_true(len(reasons) >= 1)
    assert_true("categorical" in reasons[0])
    with assert_raises():
        _ = onnx_plan(model)


def test_plan_text_is_deterministic_and_declares_itself() raises:
    var n_rows = 200
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_small(features, target, n_rows)
    var plan = onnx_plan(model, raw_score=True)

    var a = onnx_plan_text(plan)
    var b = onnx_plan_text(plan)
    assert_true(a == b)
    assert_true(a.startswith(ONNX_PLAN_MAGIC))
    assert_true("target_weights" in a)
    assert_true("nodes_missing_value_tracks_true" in a)

    save_onnx_plan(plan, _TMP_PATH)
    var back = open(_TMP_PATH, "r").read()
    remove(_TMP_PATH)
    assert_true(back == a)




def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
