"""get_params/set_params and the estimator-construction contract."""

import inspect

import numpy as np
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


REG_ALIASES = [("lambda_l1", "reg_alpha"), ("lambda_l2", "reg_lambda")]


@pytest.mark.parametrize("native, alias", REG_ALIASES)
def test_regularization_aliases_train_the_same_model(native, alias, regression):
    """LightGBM's scikit-learn estimators spell lambda_l1 and lambda_l2
    reg_alpha and reg_lambda. Either spelling must reach the trainer."""
    X, y = regression
    named = MojoBoostRegressor(n_estimators=5, **{native: 3.0}).fit(X, y)
    aliased = MojoBoostRegressor(n_estimators=5, **{alias: 3.0}).fit(X, y)
    assert list(named.predict(X[:20])) == list(aliased.predict(X[:20]))
    # Regularizing that hard has to move the fit, or the test above would
    # pass for a parameter that goes nowhere.
    plain = MojoBoostRegressor(n_estimators=5).fit(X, y)
    assert list(plain.predict(X[:20])) != list(aliased.predict(X[:20]))


@pytest.mark.parametrize("native, alias", REG_ALIASES)
def test_agreeing_regularization_aliases_are_accepted(native, alias, regression):
    X, y = regression
    est = MojoBoostRegressor(n_estimators=5, **{native: 3.0, alias: 3.0})
    assert list(est.fit(X, y).predict(X[:20]))


@pytest.mark.parametrize("native, alias", REG_ALIASES)
def test_conflicting_regularization_aliases_raise(native, alias, regression):
    """LightGBM warns and keeps one value; mojoboost raises, so a typo
    cannot quietly train a different model."""
    X, y = regression
    est = MojoBoostRegressor(n_estimators=5, **{native: 3.0, alias: 1.0})
    with pytest.raises(ValueError, match="aliases"):
        est.fit(X, y)
    # Stored unmodified either way, so the clash surfaces at fit time and
    # `clone` still reproduces the estimator that raised.
    assert getattr(est, native) == 3.0
    assert getattr(est, alias) == 1.0


def test_boosting_alias_selects_goss(regression):
    """`boosting` is LightGBM's native name and `boosting_type` its
    scikit-learn spelling; both must reach the sampler."""
    X, y = regression
    named = MojoBoostRegressor(
        n_estimators=10, boosting="goss", goss_warmup_rounds=0
    ).fit(X, y)
    aliased = MojoBoostRegressor(
        n_estimators=10, boosting_type="goss", goss_warmup_rounds=0
    ).fit(X, y)
    assert list(named.predict(X[:20])) == list(aliased.predict(X[:20]))
    # Sampling has to move the fit, or the equality above would hold for a
    # parameter that goes nowhere.
    plain = MojoBoostRegressor(n_estimators=10).fit(X, y)
    assert list(plain.predict(X[:20])) != list(named.predict(X[:20]))


def test_conflicting_boosting_aliases_raise(regression):
    X, y = regression
    est = MojoBoostRegressor(
        n_estimators=5, boosting="goss", boosting_type="gbdt"
    )
    with pytest.raises(ValueError, match="aliases"):
        est.fit(X, y)
    assert est.boosting == "goss"
    assert est.boosting_type == "gbdt"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"boosting": "dart"},
        {"boosting": "goss", "top_rate": 1.5},
        {"boosting": "goss", "other_rate": -0.1},
        {"boosting": "goss", "top_rate": 0.7, "other_rate": 0.4},
        {"boosting": "goss", "top_rate": 0.0, "other_rate": 0.0},
        {"boosting": "goss", "goss_seed": -1},
        {"boosting": "goss", "goss_warmup_rounds": -2},
        # GOSS and row bagging both own the sampled row list.
        {"boosting": "goss", "bagging_fraction": 0.5, "bagging_freq": 1},
    ],
)
def test_goss_parameter_validation(kwargs, regression):
    X, y = regression
    with pytest.raises(ValueError):
        MojoBoostRegressor(n_estimators=5, **kwargs).fit(X, y)


def test_categorical_feature_alias_trains_the_same_model(regression):
    """`categorical_features` is an accepted alias for LightGBM's
    `categorical_feature`, and marking a column must actually change the
    model it trains."""
    X, y = regression
    named = MojoBoostRegressor(
        n_estimators=5, categorical_feature=[0]
    ).fit(X, y)
    aliased = MojoBoostRegressor(
        n_estimators=5, categorical_features=[0]
    ).fit(X, y)
    assert list(named.predict(X[:20])) == list(aliased.predict(X[:20]))

    plain = MojoBoostRegressor(n_estimators=5).fit(X, y)
    assert list(plain.predict(X[:20])) != list(aliased.predict(X[:20]))


def test_conflicting_categorical_aliases_raise(regression):
    X, y = regression
    est = MojoBoostRegressor(
        n_estimators=5, categorical_feature=[0], categorical_features=[1]
    )
    with pytest.raises(ValueError, match="aliases"):
        est.fit(X, y)


def test_categorical_columns_are_split_by_category_not_threshold(regression):
    """A code whose effect alternates with its value is invisible to an
    ordinal threshold; native categorical splits must fit it and an
    ordinal treatment of the same column must not."""
    gen = np.random.default_rng(11)
    codes = gen.integers(0, 12, size=800)
    y = np.where(codes % 2 == 0, 1.0, -1.0)
    X = codes.astype(float).reshape(-1, 1)

    common = dict(
        num_leaves=4,
        n_estimators=20,
        learning_rate=0.2,
        min_data_in_leaf=5,
        min_data_per_group=5,
        cat_smooth=2.0,
    )
    as_cat = MojoBoostRegressor(categorical_feature=[0], **common).fit(X, y)
    as_num = MojoBoostRegressor(**common).fit(X, y)

    cat_mse = float(np.mean((np.asarray(as_cat.predict(X)) - y) ** 2))
    num_mse = float(np.mean((np.asarray(as_num.predict(X)) - y) ** 2))
    assert cat_mse < 0.05
    assert num_mse > 10.0 * cat_mse


def test_unseen_and_missing_categories_route_together(regression):
    """Unseen codes, negative codes, and NaN all share the reserved bin, so
    all three follow the documented right-hand default."""
    gen = np.random.default_rng(12)
    codes = gen.integers(0, 6, size=600)
    y = np.where(codes % 2 == 0, 1.0, -1.0)
    X = codes.astype(float).reshape(-1, 1)

    model = MojoBoostRegressor(
        num_leaves=8,
        n_estimators=20,
        learning_rate=0.2,
        min_data_in_leaf=5,
        min_data_per_group=5,
        cat_smooth=2.0,
        categorical_feature=[0],
    ).fit(X, y)

    odd = np.array([[99.0], [-1.0], [np.nan]])
    preds = list(model.predict(odd))
    assert preds[0] == preds[1] == preds[2]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"categorical_feature": [4]},
        {"categorical_feature": [-1]},
        {"categorical_feature": [0, 0]},
        {"categorical_feature": [0.5]},
        {"categorical_feature": "0"},
        {"categorical_feature": [0], "max_cat_to_onehot": -1},
        {"categorical_feature": [0], "max_cat_threshold": 0},
        {"categorical_feature": [0], "cat_smooth": -1.0},
        {"categorical_feature": [0], "cat_l2": -1.0},
        {"categorical_feature": [0], "min_data_per_group": 0},
    ],
)
def test_categorical_parameter_validation(kwargs, regression):
    X, y = regression
    with pytest.raises(ValueError):
        MojoBoostRegressor(n_estimators=5, **kwargs).fit(X, y)


def test_var_positional_init_is_rejected():
    class Bad(_sklearn.ParamsMixin):
        def __init__(self, *args):
            pass

    with pytest.raises(RuntimeError, match=r"\*args"):
        Bad()._get_param_names()
