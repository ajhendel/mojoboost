"""Pickling, deep-copying, and how they differ from save()/load()."""

import copy
import os
import pickle
import tempfile

import numpy as np
import pytest

from mojoboost import MojoBoostClassifier, MojoBoostRegressor, NotFittedError


def _round_trip(est):
    return pickle.loads(pickle.dumps(est))


def test_unfitted_estimator_pickles():
    est = MojoBoostRegressor(num_leaves=7)
    twin = _round_trip(est)
    assert twin.get_params() == est.get_params()
    with pytest.raises(NotFittedError):
        twin.predict(np.zeros((1, 1)))


def test_regressor_predictions_survive_pickling(fitted_regressor, regression):
    X, _ = regression
    twin = _round_trip(fitted_regressor)
    assert np.array_equal(twin.predict(X), fitted_regressor.predict(X))


@pytest.mark.parametrize("fixture", ["fitted_binary", "fitted_multiclass"])
def test_classifier_predictions_survive_pickling(request, fixture):
    est = request.getfixturevalue(fixture)
    X = request.getfixturevalue(
        "binary" if fixture == "fitted_binary" else "multiclass"
    )[0]
    twin = _round_trip(est)
    assert np.array_equal(twin.predict(X), est.predict(X))
    assert np.allclose(twin.predict_proba(X), est.predict_proba(X))


def test_pickling_keeps_the_fitted_state(fitted_multiclass):
    twin = _round_trip(fitted_multiclass)
    assert twin.n_features_in_ == fitted_multiclass.n_features_in_
    assert list(twin.classes_) == list(fitted_multiclass.classes_)
    assert twin.best_iteration_ == fitted_multiclass.best_iteration_
    assert twin.get_params() == fitted_multiclass.get_params()


def test_pickling_keeps_string_class_labels(binary):
    X, y = binary
    labels = np.where(y == 1, "yes", "no")
    est = MojoBoostClassifier(n_estimators=10).fit(X, labels)
    twin = _round_trip(est)
    assert list(twin.classes_) == ["no", "yes"]
    assert np.array_equal(twin.predict(X), est.predict(X))


@pytest.mark.parametrize("importance_type", ["split", "gain"])
def test_pickling_keeps_both_importances(regression, importance_type):
    """Split gains are not in the serialized model, so this is the thing
    pickle preserves that save()/load() cannot."""
    X, y = regression
    est = MojoBoostRegressor(
        n_estimators=10, importance_type=importance_type
    ).fit(X, y)
    twin = _round_trip(est)
    assert np.allclose(twin.feature_importances_, est.feature_importances_)
    assert np.asarray(twin.feature_importances_).sum() > 0


def test_deepcopy_of_a_fitted_estimator(fitted_regressor, regression):
    X, _ = regression
    twin = copy.deepcopy(fitted_regressor)
    assert np.array_equal(twin.predict(X), fitted_regressor.predict(X))


def test_pickled_estimator_can_be_refitted(fitted_regressor, regression):
    X, y = regression
    twin = _round_trip(fitted_regressor)
    twin.set_params(n_estimators=3).fit(X, y)
    assert twin.best_iteration_ == 3


# -- the documented alternative -------------------------------------------


def test_save_load_restores_predictions(fitted_regressor, regression):
    X, _ = regression
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.mbst")
        fitted_regressor.save(path)
        loaded = MojoBoostRegressor.load(path)
    assert np.array_equal(loaded.predict(X), fitted_regressor.predict(X))
    assert loaded.n_features_in_ == fitted_regressor.n_features_in_
    assert loaded.best_iteration_ == fitted_regressor.best_iteration_


def test_loaded_multiclass_model_knows_its_shape(fitted_multiclass, multiclass):
    X, _ = multiclass
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.mbst")
        fitted_multiclass.save(path)
        loaded = MojoBoostClassifier.load(path)
    assert loaded.n_classes_ == 3
    assert loaded.n_features_in_ == 4
    assert loaded.best_iteration_ == fitted_multiclass.best_iteration_
    assert np.array_equal(loaded.predict(X), fitted_multiclass.predict(X))


def test_loaded_model_reports_codes_not_original_labels(binary):
    """The label mapping is not part of the model file. This is the gap
    pickle exists to cover, and it is documented on load()."""
    X, y = binary
    est = MojoBoostClassifier(n_estimators=10).fit(X, np.where(y == 1, "b", "a"))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.mbst")
        est.save(path)
        loaded = MojoBoostClassifier.load(path)
    assert list(loaded.classes_) == [0, 1]
    assert np.array_equal(
        np.asarray(loaded.predict(X)),
        (np.asarray(est.predict(X)) == "b").astype(int),
    )


def test_loaded_model_preserves_gain_importance(fitted_regressor):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.mbst")
        fitted_regressor.save(path)
        loaded = MojoBoostRegressor.load(path)
    # Model format v4 serializes both split counts and split gains.
    assert np.allclose(
        loaded.feature_importances_, fitted_regressor.feature_importances_
    )
    loaded.importance_type = "gain"
    fitted_regressor.importance_type = "gain"
    assert np.allclose(
        loaded.feature_importances_, fitted_regressor.feature_importances_
    )
