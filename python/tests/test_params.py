"""get_params/set_params and the estimator-construction contract."""

import inspect

import pytest

from mojoboost import MojoBoostClassifier, MojoBoostRegressor
from mojoboost import _sklearn

ESTIMATORS = [MojoBoostRegressor, MojoBoostClassifier]


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_get_params_covers_every_constructor_parameter(cls):
    """Every keyword any `__init__` in the MRO takes must show up in
    get_params. This is what keeps `clone` faithful, and it is the check
    that catches a hyperparameter added to the shared base and forgotten
    everywhere else."""
    params = cls().get_params()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for p in inspect.signature(init).parameters.values():
            if p.name == "self" or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            assert p.name in params, f"{klass.__name__}.{p.name} missing"


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_get_params_returns_the_values_passed(cls):
    est = cls(num_leaves=7, learning_rate=0.25, n_estimators=13)
    params = est.get_params()
    assert params["num_leaves"] == 7
    assert params["learning_rate"] == 0.25
    assert params["n_estimators"] == 13
    # Defaults come back too, not just what was passed.
    assert params["max_bin"] == 255


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_constructing_from_get_params_reproduces_the_estimator(cls):
    """What `clone` does, without needing scikit-learn installed."""
    est = cls(num_leaves=9, lambda_l2=0.5)
    twin = cls(**est.get_params(deep=False))
    assert twin.get_params() == est.get_params()


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_init_stores_parameters_unmodified(cls):
    """scikit-learn requires `__init__` to store arguments as they arrive:
    no coercion, no validation, no renaming."""
    sentinel = 12.5
    est = cls(learning_rate=sentinel, num_leaves="not-an-int")
    assert est.learning_rate is sentinel
    assert est.num_leaves == "not-an-int"


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_set_params_round_trip(cls):
    est = cls()
    assert est.set_params(num_leaves=5, n_estimators=3) is est
    assert est.num_leaves == 5
    assert est.n_estimators == 3
    assert est.get_params()["num_leaves"] == 5


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_set_params_rejects_unknown_names(cls):
    with pytest.raises(ValueError, match="invalid parameter"):
        cls().set_params(no_such_parameter=1)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_set_params_with_nothing_is_a_no_op(cls):
    est = cls()
    assert est.set_params() is est


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_repr_shows_only_non_defaults(cls):
    assert repr(cls()) == f"{cls.__name__}()"
    text = repr(cls(num_leaves=4))
    assert text.startswith(f"{cls.__name__}(")
    assert "num_leaves=4" in text
    assert "max_bin" not in text


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_set_params_after_fit_leaves_the_model_alone(cls, regression, binary):
    X, y = regression if cls is MojoBoostRegressor else binary
    est = cls(n_estimators=5).fit(X, y)
    before = list(est.predict(X[:10]))
    est.set_params(n_estimators=500)
    after = list(est.predict(X[:10]))
    assert before == after, "set_params must not touch the fitted model"


def test_var_positional_init_is_rejected():
    class Bad(_sklearn.ParamsMixin):
        def __init__(self, *args):
            pass

    with pytest.raises(RuntimeError, match=r"\*args"):
        Bad()._get_param_names()
