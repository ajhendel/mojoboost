"""Basic compatibility with scikit-learn's meta-estimators.

This is deliberately not `check_estimator`: mojotrees does not claim to
pass the full estimator suite (see mojotrees._sklearn). What is claimed,
and tested here, is that the pieces people actually reach for work.
"""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
pytest.importorskip("sklearn.exceptions")

from sklearn.base import clone, is_classifier, is_regressor  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils import get_tags  # noqa: E402
from sklearn.utils.validation import check_is_fitted  # noqa: E402

from mojotrees import (  # noqa: E402
    MojoTreesClassifier,
    MojoTreesRegressor,
    NotFittedError,
)

ESTIMATORS = [MojoTreesRegressor, MojoTreesClassifier]


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_clone_reproduces_parameters(cls):
    est = cls(num_leaves=9, learning_rate=0.3, n_estimators=4)
    twin = clone(est)
    assert twin is not est
    assert twin.get_params() == est.get_params()


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_clone_of_a_fitted_estimator_is_unfitted(cls, regression, binary):
    X, y = regression if cls is MojoTreesRegressor else binary
    est = cls(n_estimators=3).fit(X, y)
    twin = clone(est)
    assert not twin.__sklearn_is_fitted__()


def test_estimator_type_dispatch():
    assert is_regressor(MojoTreesRegressor())
    assert not is_classifier(MojoTreesRegressor())
    assert is_classifier(MojoTreesClassifier())
    assert not is_regressor(MojoTreesClassifier())


def test_tags_report_what_the_estimators_actually_accept():
    tags = get_tags(MojoTreesRegressor())
    assert tags.estimator_type == "regressor"
    assert tags.input_tags.allow_nan is True
    # True because fit and predict take SciPy sparse and keep it sparse. A
    # scikit-learn utility that reads this tag as False densifies before
    # calling us, which is what the sparse path exists to avoid.
    assert tags.input_tags.sparse is True


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_sparse_input_survives_a_meta_estimator(cls, regression, binary):
    """The sparse tag is a promise about a code path, so exercise the path.

    `cross_val_score` splits, refits, and scores through scikit-learn's own
    validation and indexing, which is where a False `sparse` tag would have
    turned the matrix dense on the way in.
    """
    sparse = pytest.importorskip("scipy.sparse")
    X, y = regression if cls is MojoTreesRegressor else binary
    scores = cross_val_score(
        cls(n_estimators=3, num_leaves=7), sparse.csr_matrix(X), y, cv=3
    )
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_check_is_fitted(cls, regression, binary):
    est = cls()
    # check_is_fitted raises scikit-learn's own NotFittedError, which is a
    # base class of mojotrees's, so it is the one to expect here.
    with pytest.raises(sklearn.exceptions.NotFittedError):
        check_is_fitted(est)
    X, y = regression if cls is MojoTreesRegressor else binary
    check_is_fitted(est.fit(X, y))


def test_mojotrees_not_fitted_error_is_catchable_as_sklearns(regression):
    X, _ = regression
    with pytest.raises(sklearn.exceptions.NotFittedError):
        MojoTreesRegressor().predict(X)
    assert issubclass(NotFittedError, sklearn.exceptions.NotFittedError)


def test_pipeline_regression(regression):
    X, y = regression
    pipe = Pipeline(
        [("scale", StandardScaler()), ("gbdt", MojoTreesRegressor(
            n_estimators=10
        ))]
    )
    pipe.fit(X, y)
    assert np.asarray(pipe.predict(X)).shape == (len(X),)
    assert pipe.score(X, y) > 0.5


def test_pipeline_classification_and_proba(multiclass):
    X, y = multiclass
    pipe = Pipeline(
        [("scale", StandardScaler()), ("gbdt", MojoTreesClassifier(
            n_estimators=10
        ))]
    )
    pipe.fit(X, y)
    proba = np.asarray(pipe.predict_proba(X))
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert pipe.score(X, y) > 0.5


def test_pipeline_forwards_sample_weight(regression):
    X, y = regression
    pipe = Pipeline([("gbdt", MojoTreesRegressor(n_estimators=5))])
    weights = np.where(X[:, 0] > 0.5, 10.0, 0.1)
    pipe.fit(X, y, gbdt__sample_weight=weights)
    plain = MojoTreesRegressor(n_estimators=5).fit(X, y)
    assert not np.array_equal(pipe.predict(X), plain.predict(X))


def test_pipeline_set_params_reaches_the_estimator(regression):
    X, y = regression
    pipe = Pipeline([("gbdt", MojoTreesRegressor(n_estimators=5))])
    pipe.set_params(gbdt__num_leaves=3)
    assert pipe.named_steps["gbdt"].num_leaves == 3
    pipe.fit(X, y)


def test_grid_search_regression(regression):
    X, y = regression
    search = GridSearchCV(
        MojoTreesRegressor(n_estimators=8),
        {"num_leaves": [3, 15], "learning_rate": [0.1, 0.3]},
        cv=3,
    )
    search.fit(X, y)
    assert set(search.best_params_) == {"num_leaves", "learning_rate"}
    assert len(search.cv_results_["mean_test_score"]) == 4
    assert np.asarray(search.predict(X)).shape == (len(X),)


def test_grid_search_classification_uses_stratified_folds(multiclass):
    """A classifier gets StratifiedKFold from `is_classifier`, so every
    fold sees all three classes even though the target is sorted here."""
    X, y = multiclass
    search = GridSearchCV(
        MojoTreesClassifier(n_estimators=8),
        {"num_leaves": [3, 15]},
        cv=3,
    )
    search.fit(X, y)
    assert search.best_score_ > 0.5


def test_cross_val_score(regression):
    X, y = regression
    scores = cross_val_score(MojoTreesRegressor(n_estimators=8), X, y, cv=3)
    assert scores.shape == (3,)
    assert (scores > 0).all()


def test_cross_val_score_with_a_proba_metric(binary):
    X, y = binary
    scores = cross_val_score(
        MojoTreesClassifier(n_estimators=8),
        X,
        y,
        cv=3,
        scoring="neg_log_loss",
    )
    assert scores.shape == (3,)
    assert (scores < 0).all()


def test_dataframe_through_a_pipeline(regression):
    pd = pytest.importorskip("pandas")
    X, y = regression
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    pipe = Pipeline([("gbdt", MojoTreesRegressor(n_estimators=5))])
    pipe.fit(frame, y)
    assert list(pipe.named_steps["gbdt"].feature_names_in_) == [
        "a",
        "b",
        "c",
        "d",
    ]
    assert np.asarray(pipe.predict(frame)).shape == (len(X),)
