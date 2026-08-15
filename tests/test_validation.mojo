"""`validation.mojo` is reached from the public entry points.

Each test hands an invalid input to a shipping entry (`Dataset`,
`train_dataset_multiclass`, `train_dataset_ranker`, `parse_params`) and
matches the message `validation.mojo` writes, which names the rule, the
offending index, and the value. If a trainer grew back a local copy of one
of these rules, the message here is what would change.
"""

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees import (
    SQUARED_ERROR,
    BoosterParams,
    Dataset,
    TreeParams,
    parse_params,
    train_dataset,
    train_dataset_multiclass,
    train_dataset_ranker,
)


def _features(n_rows: Int, n_features: Int) -> List[Float64]:
    var out = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        out.append(Float64((k * 7919) % 101) / 101.0)
    return out^


def _labels(n_rows: Int) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        y.append(Float64(r % 3))
    return y^


def _params() -> BoosterParams:
    var tree = TreeParams.default()
    tree.num_leaves = 4
    tree.min_data_in_leaf = 2
    return BoosterParams(2, 0.1, tree^)


def test_dataset_group_counts_are_validations() raises:
    var n_rows = 30
    var features = _features(n_rows, 3)
    with assert_raises(contains="group counts must sum to n_rows: they sum to 20 for 30 rows"):
        _ = Dataset(features, n_rows, 3, _labels(n_rows), group=[10, 10])
    with assert_raises(contains="group counts must be positive: query 1 has 0 rows"):
        _ = Dataset(features, n_rows, 3, _labels(n_rows), group=[30, 0])


def test_dataset_categorical_declaration_is_validations() raises:
    var n_rows = 30
    var features = _features(n_rows, 3)
    with assert_raises(contains="categorical feature index 7 at position 0 out of range for 3 features"):
        _ = Dataset(features, n_rows, 3, _labels(n_rows), categorical_features=[7])
    with assert_raises(contains="categorical feature index 1 listed twice, at positions 0 and 1"):
        _ = Dataset(features, n_rows, 3, _labels(n_rows), categorical_features=[1, 1])


def test_dataset_column_lengths_are_validations() raises:
    var n_rows = 30
    var features = _features(n_rows, 3)
    with assert_raises(contains="label must have one entry per row: got 29 for 30 rows"):
        var short = _labels(n_rows)
        _ = short.pop()
        _ = Dataset(features, n_rows, 3, short^)
    with assert_raises(contains="weight must have one entry per row: got 2 for 30 rows"):
        _ = Dataset(features, n_rows, 3, _labels(n_rows), weight=[1.0, 1.0])


def test_multiclass_codes_are_validations() raises:
    var n_rows = 30
    var features = _features(n_rows, 3)
    var frac = _labels(n_rows)
    frac[4] = 1.5
    var ds = Dataset(features, n_rows, 3, frac^)
    with assert_raises(contains="class labels must be whole numbers: row 4 is 1.5"):
        _ = train_dataset_multiclass(ds, 3, _params())
    var ds2 = Dataset(features, n_rows, 3, _labels(n_rows))
    with assert_raises(contains="class label out of range: row 2 is 2, which is outside [0, 2)"):
        _ = train_dataset_multiclass(ds2, 2, _params())


def test_ranking_relevance_is_validations() raises:
    var n_rows = 30
    var features = _features(n_rows, 3)
    var big = _labels(n_rows)
    big[7] = 31.0
    var ds = Dataset(features, n_rows, 3, big^, group=[10, 10, 10])
    with assert_raises(contains="relevance labels must be at most 30: row 7 is 31"):
        _ = train_dataset_ranker(ds, _params())


def test_sample_weight_domain_is_validations() raises:
    var n_rows = 30
    var features = _features(n_rows, 3)
    var w = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        w.append(1.0)
    w[3] = -2.0
    var ds = Dataset(features, n_rows, 3, _labels(n_rows), weight=w^)
    with assert_raises(contains="sample_weight must be nonnegative: row 3 is -2.0"):
        _ = train_dataset(ds, SQUARED_ERROR, _params())


def test_parameter_ranges_are_validations() raises:
    with assert_raises(contains="learning_rate must be positive, got 0.0"):
        _ = parse_params("learning_rate=0")
    with assert_raises(contains="num_leaves must be at least 2, got 1"):
        _ = parse_params("num_leaves=1")
    with assert_raises(contains="feature_fraction must be in (0, 1], got 1.5"):
        _ = parse_params("feature_fraction=1.5")
    with assert_raises(contains="max_bin must be at least 2, got 1"):
        _ = parse_params("max_bin=1")


def test_valid_inputs_still_train() raises:
    var n_rows = 30
    var features = _features(n_rows, 3)
    var ds = Dataset(features, n_rows, 3, _labels(n_rows))
    var model = train_dataset(ds, SQUARED_ERROR, _params())
    assert_true(len(model.booster.trees) >= 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
