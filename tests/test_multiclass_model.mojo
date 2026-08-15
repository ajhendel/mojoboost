"""End-to-end multiclass model tests: fit on raw data, predict on raw
data, and serialization round trips."""

from std.os import remove
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojotrees import (
    BINARY_LOGISTIC,
    BoosterParams,
    MulticlassModel,
    TreeParams,
    fit,
    fit_multiclass,
    load_model,
    load_multiclass_model,
    save_model,
    save_multiclass_model,
)

comptime _TMP_PATH = "./.test_multiclass_roundtrip.tmp"


def _cluster_features() -> List[Float64]:
    return [0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0]


def _cluster_labels() -> List[Int]:
    return [0, 0, 0, 1, 1, 1, 2, 2, 2]


def _small_params() -> BoosterParams:
    return BoosterParams(150, 0.2, TreeParams(4, 1, 1.0, 1e-3))


def test_fit_multiclass_predicts_raw_data() raises:
    var features = _cluster_features()
    var labels = _cluster_labels()
    var model = fit_multiclass(
        features, 9, 1, labels, 3, _small_params(), max_bins=16
    )
    # Training rows plus unseen values inside each cluster.
    var probes: List[Float64] = [0.5, 1.7, 11.5, 21.3]
    var expected: List[Int] = [0, 0, 1, 2]
    for r in range(9):
        var row: List[Float64] = [features[r]]
        assert_equal(model.predict_class(row), labels[r])
        var proba = model.predict_proba(row)
        var total = 0.0
        for k in range(3):
            total += proba[k]
        assert_true(abs(total - 1.0) < 1e-9)
        assert_true(proba[labels[r]] > 0.7)
    for i in range(len(probes)):
        var row: List[Float64] = [probes[i]]
        assert_equal(model.predict_class(row), expected[i])


def test_multiclass_roundtrip_predictions_exact() raises:
    var features = _cluster_features()
    var labels = _cluster_labels()
    var model = fit_multiclass(
        features, 9, 1, labels, 3, _small_params(), max_bins=16
    )
    save_multiclass_model(model, _TMP_PATH)
    var loaded = load_multiclass_model(_TMP_PATH)
    remove(_TMP_PATH)

    assert_equal(loaded.booster.n_classes, model.booster.n_classes)
    assert_equal(len(loaded.booster.trees), len(model.booster.trees))
    for k in range(3):
        assert_true(
            loaded.booster.base_scores[k] == model.booster.base_scores[k]
        )
    for r in range(9):
        var row: List[Float64] = [features[r]]
        var pa = model.predict_proba(row)
        var pb = loaded.predict_proba(row)
        for k in range(3):
            assert_true(pa[k] == pb[k])


def test_load_wrong_kind_raises() raises:
    var features = _cluster_features()
    var labels = _cluster_labels()
    var mc = fit_multiclass(
        features, 9, 1, labels, 3, _small_params(), max_bins=16
    )
    save_multiclass_model(mc, _TMP_PATH)
    with assert_raises():
        _ = load_model(_TMP_PATH)
    remove(_TMP_PATH)

    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var binary = fit(
        features, 9, 1, target, BINARY_LOGISTIC, _small_params(), 16
    )
    save_model(binary, _TMP_PATH)
    with assert_raises():
        _ = load_multiclass_model(_TMP_PATH)
    remove(_TMP_PATH)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
