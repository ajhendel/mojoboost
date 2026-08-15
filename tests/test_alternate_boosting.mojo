"""LightGBM's `boosting` parameter: gbdt, dart, and rf through one door.

`alternate_boosting.fit_boosting` bins and trains under a named mode and
returns the same `Model` every predictor reads. This suite checks the three
things the route owes: `gbdt` through the dispatcher is bit-for-bit
`model.fit`; `dart` and `rf` train, predict finite values, and are not the
gbdt model in disguise; and a dart or rf model survives a save/load round
trip as the plain ensemble it was folded into.
"""

from std.os import remove
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    BaggingParams,
    BoosterParams,
    TreeParams,
    fit,
    fit_multiclass,
    load_model,
    save_model,
)
from mojotrees.alternate_boosting import (
    BOOSTING_DART,
    BOOSTING_GBDT,
    BOOSTING_RF,
    AlternateBoostingParams,
    boosting_name,
    fit_boosting,
    fit_boosting_multiclass,
    parse_boosting,
)
from mojotrees.boosting_dart import DartParams

comptime _TMP_PATH = "./.test_alternate_boosting_roundtrip.tmp"
comptime _N_ROWS = 400
comptime _N_FEATURES = 4


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _features() -> List[Float64]:
    var features = List[Float64](capacity=_N_ROWS * _N_FEATURES)
    for k in range(_N_ROWS * _N_FEATURES):
        features.append(_uniform(UInt64(k)))
    return features^


def _target(features: List[Float64]) -> List[Float64]:
    var y = List[Float64](capacity=_N_ROWS)
    for r in range(_N_ROWS):
        var x0 = features[0 * _N_ROWS + r]
        var x1 = features[1 * _N_ROWS + r]
        var x2 = features[2 * _N_ROWS + r]
        y.append(3.0 * x0 - 2.0 * x1 + 4.0 * (x2 - 0.5) * (x2 - 0.5))
    return y^


def _row(features: List[Float64], r: Int) -> List[Float64]:
    var row = List[Float64](capacity=_N_FEATURES)
    for f in range(_N_FEATURES):
        row.append(features[f * _N_ROWS + r])
    return row^


def _params(learning_rate: Float64) -> BoosterParams:
    return BoosterParams(
        20, learning_rate, TreeParams(15, 10, 1.0, 1e-3, 0.0)
    )


def _is_finite(x: Float64) -> Bool:
    return x == x and x < 1e300 and x > -1e300


def test_names_round_trip() raises:
    assert_equal(parse_boosting("gbdt"), BOOSTING_GBDT)
    assert_equal(parse_boosting("dart"), BOOSTING_DART)
    assert_equal(parse_boosting("rf"), BOOSTING_RF)
    assert_equal(parse_boosting("random_forest"), BOOSTING_RF)
    assert_equal(boosting_name(BOOSTING_DART), "dart")
    var named = AlternateBoostingParams.named("dart")
    assert_equal(named.mode, BOOSTING_DART)
    assert_true(named.dart.enabled)


def test_gbdt_through_dispatcher_is_bit_exact() raises:
    var features = _features()
    var y = _target(features)
    var direct = fit(
        Span(features), _N_ROWS, _N_FEATURES, y, SQUARED_ERROR, _params(0.1)
    )
    var routed = fit_boosting(
        Span(features),
        _N_ROWS,
        _N_FEATURES,
        y,
        SQUARED_ERROR,
        _params(0.1),
        AlternateBoostingParams.named("gbdt"),
    )
    assert_equal(len(direct.booster.trees), len(routed.booster.trees))
    for r in range(0, _N_ROWS, 7):
        var row = _row(features, r)
        assert_equal(direct.predict(row), routed.predict(row))


def test_dart_trains_predicts_and_round_trips() raises:
    var features = _features()
    var y = _target(features)
    var gbdt = fit(
        Span(features), _N_ROWS, _N_FEATURES, y, SQUARED_ERROR, _params(0.1)
    )
    var dart = fit_boosting(
        Span(features),
        _N_ROWS,
        _N_FEATURES,
        y,
        SQUARED_ERROR,
        _params(0.1),
        AlternateBoostingParams.dart_with(
            DartParams.enable(drop_rate=0.5, skip_drop=0.0)
        ),
    )
    assert_equal(len(dart.booster.trees), 20)
    # The per-tree weights were folded into the leaves; the ensemble's own
    # factor is unity, which is what makes the saved model a plain one.
    assert_equal(dart.booster.learning_rate, 1.0)
    var differs = False
    for r in range(0, _N_ROWS, 5):
        var row = _row(features, r)
        var p = dart.predict(row)
        assert_true(_is_finite(p))
        if p != gbdt.predict(row):
            differs = True
    assert_true(differs)
    save_model(dart, _TMP_PATH)
    var back = load_model(_TMP_PATH)
    remove(_TMP_PATH)
    for r in range(0, _N_ROWS, 5):
        var row = _row(features, r)
        assert_equal(back.predict(row), dart.predict(row))


def test_rf_trains_predicts_and_round_trips() raises:
    var features = _features()
    var y = _target(features)
    var gbdt = fit(
        Span(features), _N_ROWS, _N_FEATURES, y, SQUARED_ERROR, _params(0.1)
    )
    var rf = fit_boosting(
        Span(features),
        _N_ROWS,
        _N_FEATURES,
        y,
        SQUARED_ERROR,
        _params(1.0),
        AlternateBoostingParams.named("rf"),
        bagging=BaggingParams(0.7, 1, 3),
    )
    assert_equal(len(rf.booster.trees), 20)
    var differs = False
    for r in range(0, _N_ROWS, 5):
        var row = _row(features, r)
        var p = rf.predict(row)
        assert_true(_is_finite(p))
        if p != gbdt.predict(row):
            differs = True
    assert_true(differs)
    save_model(rf, _TMP_PATH)
    var back = load_model(_TMP_PATH)
    remove(_TMP_PATH)
    for r in range(0, _N_ROWS, 5):
        var row = _row(features, r)
        assert_equal(back.predict(row), rf.predict(row))


def test_multiclass_dart_and_rf_train_through_the_raw_entry() raises:
    var features = _features()
    var y = _target(features)
    var labels = List[Int](capacity=_N_ROWS)
    for r in range(_N_ROWS):
        labels.append(0 if y[r] < 0.5 else (1 if y[r] < 1.5 else 2))
    var gbdt = fit_multiclass(
        Span(features), _N_ROWS, _N_FEATURES, labels, 3, _params(0.1)
    )
    var dart = fit_boosting_multiclass(
        Span(features),
        _N_ROWS,
        _N_FEATURES,
        labels,
        3,
        _params(0.1),
        AlternateBoostingParams.named("dart"),
    )
    var rf = fit_boosting_multiclass(
        Span(features),
        _N_ROWS,
        _N_FEATURES,
        labels,
        3,
        _params(1.0),
        AlternateBoostingParams.named("rf"),
        bagging=BaggingParams(0.7, 1, 3),
    )
    assert_equal(dart.booster.n_classes, 3)
    assert_equal(rf.booster.n_classes, 3)
    var dart_differs = False
    var rf_differs = False
    for r in range(0, _N_ROWS, 5):
        var row = _row(features, r)
        var base = gbdt.predict_proba(row)
        var pd = dart.predict_proba(row)
        var pr = rf.predict_proba(row)
        var sd = 0.0
        var sr = 0.0
        for k in range(3):
            assert_true(_is_finite(pd[k]) and _is_finite(pr[k]))
            sd += pd[k]
            sr += pr[k]
            if pd[k] != base[k]:
                dart_differs = True
            if pr[k] != base[k]:
                rf_differs = True
        assert_true(abs(sd - 1.0) < 1e-9 and abs(sr - 1.0) < 1e-9)
    assert_true(dart_differs and rf_differs)


def test_rf_binary_response_is_a_probability() raises:
    var features = _features()
    var y = _target(features)
    var labels = List[Float64](capacity=_N_ROWS)
    for r in range(_N_ROWS):
        labels.append(1.0 if y[r] > 0.5 else 0.0)
    var rf = fit_boosting(
        Span(features),
        _N_ROWS,
        _N_FEATURES,
        labels,
        BINARY_LOGISTIC,
        _params(1.0),
        AlternateBoostingParams.named("rf"),
        bagging=BaggingParams(0.7, 1, 3),
    )
    for r in range(0, _N_ROWS, 5):
        var p = rf.predict(_row(features, r))
        assert_true(p >= 0.0 and p <= 1.0)


def test_rf_refuses_unrandomized_and_shrunk_runs() raises:
    var features = _features()
    var y = _target(features)
    var refused = False
    try:
        _ = fit_boosting(
            Span(features),
            _N_ROWS,
            _N_FEATURES,
            y,
            SQUARED_ERROR,
            _params(1.0),
            AlternateBoostingParams.named("rf"),
        )
    except:
        refused = True
    assert_true(refused)
    refused = False
    try:
        _ = fit_boosting(
            Span(features),
            _N_ROWS,
            _N_FEATURES,
            y,
            SQUARED_ERROR,
            _params(0.1),
            AlternateBoostingParams.named("rf"),
            bagging=BaggingParams(0.7, 1, 3),
        )
    except:
        refused = True
    assert_true(refused)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
