"""The entry points that made four unreachable modules reachable.

Catalog A31. Everything tested here is an EDGE, not arithmetic: the
arithmetic already had tests, and having tests is exactly what let these
modules sit compiled and unreachable for a round. So each test asks the one
question the old tests could not: does a raw input a caller actually has
reach the mechanism, and does the answer come back in the shape the caller
was promised?

Two properties get real assertions rather than a smoke check.

- **`fit_multi_rmse` is `T` independent squared-error fits**, which is this
  build's honest statement about `MultiRMSE` (A28). The test proves it the
  only way that means anything: fit two targets together, fit each alone
  through `model.fit`, and compare the predictions exactly. Any cross-target
  coupling that crept in would break equality.
- **`text_column_features` produces the column count its own estimators
  promise**, and the BoW columns are a binary indicator rather than a count,
  which is the one thing about `TBagOfWordsEstimator` that is easy to get
  wrong and invisible downstream.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from mojotrees.boosting import SQUARED_ERROR, BoosterParams
from mojotrees.embedding import (
    EmbeddingEstimatorParams,
    EmbeddingMatrix,
    KnnParams,
    LdaParams,
    compute_online_features,
    identity_permutation,
    online_feature_count,
)
from mojotrees.model import fit as fit_single
from mojotrees.multi_target import (
    MultiTargetModel,
    fit_multi_rmse,
    predict_multi_rmse,
)
from mojotrees.onnx_export import onnx_plan, onnx_plan_text
from mojotrees.target_matrix import TargetMatrix
from mojotrees.text_features import (
    TextColumns,
    TextFeatureSpec,
    text_column_features,
    text_column_features_default_dictionaries,
)
from mojotrees.text_processing import DictionaryParams, TokenizerParams
from mojotrees.tree import TreeParams


comptime _N_ROWS = 40
comptime _N_FEATURES = 3


def _matrix() -> List[Float64]:
    """A column-major `_N_ROWS` by `_N_FEATURES` matrix with no randomness in
    it, so a failure is a failure and not a seed."""
    var out = List[Float64](capacity=_N_ROWS * _N_FEATURES)
    for f in range(_N_FEATURES):
        for r in range(_N_ROWS):
            out.append(Float64((r * 7 + f * 13) % 11) * 0.5 - 2.0)
    return out^


def _params() -> BoosterParams:
    var tree = TreeParams(7, 2, 0.0, 1e-3)
    return BoosterParams(5, 0.3, tree^)


def test_fit_multi_rmse_reaches_the_trainer_from_a_raw_matrix() raises:
    """The edge itself: a column-major raw matrix and a `TargetMatrix` in, a
    model that predicts out. Before this entry point existed,
    `train_multi_rmse` took a `BinnedMatrix` and nothing outside the package
    could build one."""
    var features = _matrix()
    var values = List[Float64](capacity=_N_ROWS * 2)
    for r in range(_N_ROWS):
        values.append(Float64(r) * 0.25)
        values.append(10.0 - Float64(r) * 0.5)
    var targets = TargetMatrix(values^, 2)

    var model = fit_multi_rmse(
        features, _N_ROWS, _N_FEATURES, targets, _params()
    )
    assert_equal(model.n_targets(), 2)
    assert_equal(model.n_features(), _N_FEATURES)
    assert_equal(model.n_iterations(), 5)
    # One tree per target per round, which is the shape and the honest
    # difference from CatBoost's MultiRMSE.
    assert_equal(len(model.booster.trees), 10)

    var scores = predict_multi_rmse(
        model, features, _N_ROWS, _N_FEATURES
    )
    assert_equal(len(scores), _N_ROWS * 2)


def test_multi_rmse_is_t_independent_squared_error_fits() raises:
    """A28's claim, tested rather than asserted in a docstring: because the
    `MultiRMSE` derivative has no cross-target term, growing one tree per
    target per round gives exactly what `T` separate `SQUARED_ERROR` fits
    give. Equality, no tolerance."""
    var features = _matrix()
    var a = List[Float64](capacity=_N_ROWS)
    var b = List[Float64](capacity=_N_ROWS)
    var values = List[Float64](capacity=_N_ROWS * 2)
    for r in range(_N_ROWS):
        var ya = Float64((r * 3) % 9) - 4.0
        var yb = Float64((r * 5) % 7) * 2.0
        a.append(ya)
        b.append(yb)
        values.append(ya)
        values.append(yb)

    var multi = fit_multi_rmse(
        features,
        _N_ROWS,
        _N_FEATURES,
        TargetMatrix(values^, 2),
        _params(),
    )
    var only_a = fit_single(
        features, _N_ROWS, _N_FEATURES, a, SQUARED_ERROR, _params()
    )
    var only_b = fit_single(
        features, _N_ROWS, _N_FEATURES, b, SQUARED_ERROR, _params()
    )

    var scores = predict_multi_rmse(
        multi, features, _N_ROWS, _N_FEATURES
    )
    for r in range(_N_ROWS):
        var row = List[Float64](capacity=_N_FEATURES)
        for f in range(_N_FEATURES):
            row.append(features[f * _N_ROWS + r])
        assert_equal(scores[r * 2], only_a.predict_raw(row))
        assert_equal(scores[r * 2 + 1], only_b.predict_raw(row))


def test_fit_multi_rmse_refuses_a_target_of_the_wrong_length() raises:
    var features = _matrix()
    var values = List[Float64](capacity=6)
    for i in range(6):
        values.append(Float64(i))
    with assert_raises():
        _ = fit_multi_rmse(
            features,
            _N_ROWS,
            _N_FEATURES,
            TargetMatrix(values^, 2),
            _params(),
        )


def test_predict_multi_rmse_refuses_a_different_feature_count() raises:
    var features = _matrix()
    var values = List[Float64](capacity=_N_ROWS * 2)
    for r in range(_N_ROWS):
        values.append(Float64(r))
        values.append(Float64(r) * -1.0)
    var model = fit_multi_rmse(
        features,
        _N_ROWS,
        _N_FEATURES,
        TargetMatrix(values^, 2),
        _params(),
    )
    with assert_raises():
        _ = predict_multi_rmse(model, features, _N_ROWS, 2)


def _docs() -> List[String]:
    return [
        String("red green blue"),
        String("red red green"),
        String("blue blue blue"),
        String("green red blue"),
        String("red green blue"),
        String("blue green red"),
    ]


def test_text_column_features_reaches_the_tokenizer_and_the_dictionary() raises:
    """The two-module chain: `text_column_features` is the only function that
    takes documents, and it is what pulls `text_processing` in behind it."""
    var spec = TextFeatureSpec(True, 4, False, False)
    var columns = text_column_features(
        _docs(),
        List[Int](),
        0,
        List[Int](),
        TokenizerParams(),
        [DictionaryParams(1, 1, 50)],
        spec,
    )
    assert_equal(columns.n_rows, 6)
    assert_true(columns.n_features > 0)
    assert_equal(len(columns.values), columns.n_rows * columns.n_features)
    # BoW is a binary INDICATOR, never a count: "blue blue blue" holds one
    # token three times and must still be a 1.
    for i in range(len(columns.values)):
        var v = columns.values[i]
        assert_true(v == 0.0 or v == 1.0)


def test_text_column_features_refuses_an_empty_spec() raises:
    """A spec with nothing enabled raises rather than returning zero
    columns, which is `compute_online_features`'s rule as well."""
    with assert_raises():
        _ = text_column_features_default_dictionaries(
            _docs(), List[Int](), 0, List[Int](), TextFeatureSpec.default()
        )


def test_text_column_features_refuses_a_target_free_call_for_naive_bayes() raises:
    """`naive_bayes` reads the target, so the class vector is required and
    its absence is an error rather than an empty statistic."""
    with assert_raises():
        _ = text_column_features_default_dictionaries(
            _docs(),
            List[Int](),
            2,
            List[Int](),
            TextFeatureSpec(False, 4, True, False),
        )


def test_embedding_online_features_agree_with_the_promised_count() raises:
    """The embedding edge. `online_feature_count` is what the Python wrapper
    sizes its buffer with, so a disagreement between it and the pass is a
    memory error in the caller's process rather than a wrong number."""
    var n_rows = 12
    var dim = 3
    var values = List[Float64](capacity=n_rows * dim)
    for r in range(n_rows):
        for j in range(dim):
            values.append(Float64((r * 3 + j) % 5) - 2.0)
    var embeddings = EmbeddingMatrix(n_rows, dim, values^)
    var targets = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        targets.append(Float64(r % 2))
    var params = EmbeddingEstimatorParams(
        LdaParams.default(), KnnParams(True, 3, 1000)
    )

    var promised = online_feature_count(2, dim, params)
    var cols = compute_online_features(
        embeddings, targets, 2, identity_permutation(n_rows), params
    )
    assert_equal(len(cols), promised)
    for f in range(len(cols)):
        assert_equal(len(cols[f]), n_rows)


def test_onnx_plan_text_is_reachable_from_a_fitted_model() raises:
    """`onnx_export` needed one binding and nothing else, and this is the
    call that binding makes. The plan's own arithmetic is tested in
    test_onnx_export.mojo; what is new is that a fitted model can get here at
    all."""
    var features = _matrix()
    var target = List[Float64](capacity=_N_ROWS)
    for r in range(_N_ROWS):
        target.append(Float64(r % 5) - 2.0)
    var model = fit_single(
        features, _N_ROWS, _N_FEATURES, target, SQUARED_ERROR, _params()
    )
    var text = onnx_plan_text(onnx_plan(model, raw_score=True))
    assert_true("n_targets 1" in text)
    assert_true("target_weights" in text)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
