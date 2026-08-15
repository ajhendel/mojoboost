"""get_params/set_params and the estimator-construction contract."""

import inspect

import numpy as np
import pytest

from mojotrees import MojoTreesClassifier, MojoTreesRegressor
from mojotrees import _sklearn

ESTIMATORS = [MojoTreesRegressor, MojoTreesClassifier]


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
    X, y = regression if cls is MojoTreesRegressor else binary
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
    named = MojoTreesRegressor(n_estimators=5, **{native: 3.0}).fit(X, y)
    aliased = MojoTreesRegressor(n_estimators=5, **{alias: 3.0}).fit(X, y)
    assert list(named.predict(X[:20])) == list(aliased.predict(X[:20]))
    # Regularizing that hard has to move the fit, or the test above would
    # pass for a parameter that goes nowhere.
    plain = MojoTreesRegressor(n_estimators=5).fit(X, y)
    assert list(plain.predict(X[:20])) != list(aliased.predict(X[:20]))


@pytest.mark.parametrize("native, alias", REG_ALIASES)
def test_agreeing_regularization_aliases_are_accepted(native, alias, regression):
    X, y = regression
    est = MojoTreesRegressor(n_estimators=5, **{native: 3.0, alias: 3.0})
    assert list(est.fit(X, y).predict(X[:20]))


@pytest.mark.parametrize("native, alias", REG_ALIASES)
def test_conflicting_regularization_aliases_raise(native, alias, regression):
    """LightGBM warns and keeps one value; mojotrees raises, so a typo
    cannot quietly train a different model."""
    X, y = regression
    est = MojoTreesRegressor(n_estimators=5, **{native: 3.0, alias: 1.0})
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
    named = MojoTreesRegressor(
        n_estimators=10, boosting="goss", goss_warmup_rounds=0
    ).fit(X, y)
    aliased = MojoTreesRegressor(
        n_estimators=10, boosting_type="goss", goss_warmup_rounds=0
    ).fit(X, y)
    assert list(named.predict(X[:20])) == list(aliased.predict(X[:20]))
    # Sampling has to move the fit, or the equality above would hold for a
    # parameter that goes nowhere.
    plain = MojoTreesRegressor(n_estimators=10).fit(X, y)
    assert list(plain.predict(X[:20])) != list(named.predict(X[:20]))


def test_conflicting_boosting_aliases_raise(regression):
    X, y = regression
    est = MojoTreesRegressor(
        n_estimators=5, boosting="goss", boosting_type="gbdt"
    )
    with pytest.raises(ValueError, match="aliases"):
        est.fit(X, y)
    assert est.boosting == "goss"
    assert est.boosting_type == "gbdt"


@pytest.mark.parametrize(
    "kwargs",
    [
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
        MojoTreesRegressor(n_estimators=5, **kwargs).fit(X, y)


def test_dart_and_rf_train_through_the_estimator(regression):
    """`boosting="dart"` and `"rf"` reach `alternate_boosting.fit_boosting`
    from the plain fit path; both must move the fit off gbdt."""
    X, y = regression
    gbdt = MojoTreesRegressor(n_estimators=8, num_leaves=5).fit(X, y)
    dart = MojoTreesRegressor(
        n_estimators=8, num_leaves=5, boosting="dart", drop_rate=0.5,
        skip_drop=0.0, drop_seed=3,
    ).fit(X, y)
    rf = MojoTreesRegressor(
        n_estimators=8, num_leaves=5, boosting_type="rf",
        bagging_fraction=0.6, bagging_freq=1, bagging_seed=3,
    ).fit(X, y)
    base = gbdt.predict(X[:20])
    assert list(dart.predict(X[:20])) != list(base)
    assert list(rf.predict(X[:20])) != list(base)
    assert dart.booster_.num_trees() == 8
    assert rf.booster_.num_trees() == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"boosting": "dart", "drop_rate": 1.5},
        {"boosting": "dart", "skip_drop": -0.1},
        {"boosting": "dart", "drop_seed": -1},
    ],
)
def test_dart_parameter_validation(kwargs, regression):
    X, y = regression
    with pytest.raises(ValueError):
        MojoTreesRegressor(n_estimators=5, **kwargs).fit(X, y)


def test_categorical_feature_alias_trains_the_same_model(regression):
    """`categorical_features` is an accepted alias for LightGBM's
    `categorical_feature`, and marking a column must actually change the
    model it trains."""
    X, y = regression
    X = X.copy()
    # Categorical columns must hold whole-number codes, and the labels are
    # permuted so no ordinal threshold can reproduce the categorical splits.
    X[:, 0] = (np.floor(X[:, 0] * 8) * 5) % 8
    named = MojoTreesRegressor(
        n_estimators=5, categorical_feature=[0]
    ).fit(X, y)
    aliased = MojoTreesRegressor(
        n_estimators=5, categorical_features=[0]
    ).fit(X, y)
    assert list(named.predict(X[:20])) == list(aliased.predict(X[:20]))

    plain = MojoTreesRegressor(n_estimators=5).fit(X, y)
    assert list(plain.predict(X[:20])) != list(aliased.predict(X[:20]))


def test_conflicting_categorical_aliases_raise(regression):
    X, y = regression
    est = MojoTreesRegressor(
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
    as_cat = MojoTreesRegressor(categorical_feature=[0], **common).fit(X, y)
    as_num = MojoTreesRegressor(**common).fit(X, y)

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

    model = MojoTreesRegressor(
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
        MojoTreesRegressor(n_estimators=5, **kwargs).fit(X, y)


def test_var_positional_init_is_rejected():
    class Bad(_sklearn.ParamsMixin):
        def __init__(self, *args):
            pass

    with pytest.raises(RuntimeError, match=r"\*args"):
        Bad()._get_param_names()


# -- objectives and their scalar parameters -------------------------------


@pytest.mark.parametrize(
    "objective",
    [
        "regression",
        "huber",
        "quantile",
        "mae",
        "regression_l1",
        "poisson",
        "gamma",
        "tweedie",
        "mape",
        "fair",
        "cross_entropy",
        "xentropy",
    ],
)
def test_objective_names_resolve(objective):
    """Every documented name maps to a code without touching data."""
    assert isinstance(
        MojoTreesRegressor(objective=objective)._objective_code(), int
    )


@pytest.mark.parametrize(
    "objective,label",
    [
        ("cross_entropy_lambda", "cross_entropy"),
        ("multiclassova", "one-vs-rest"),
        ("rank_xendcg", "lambdarank"),
        ("multiclass", "MojoTreesClassifier"),
    ],
)
def test_unimplemented_objectives_are_named_not_merely_unknown(
    objective, label
):
    """A LightGBM objective mojotrees does not implement says so, and says
    what to use instead; it is not lumped in with a typo."""
    with pytest.raises(ValueError) as excinfo:
        MojoTreesRegressor(objective=objective)._objective_code()
    assert label in str(excinfo.value)


def test_unknown_objective_still_lists_the_known_ones():
    with pytest.raises(ValueError, match="unknown objective"):
        MojoTreesRegressor(objective="regresion")._objective_code()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"objective": "huber", "alpha": 0.0},
        {"objective": "quantile", "alpha": 1.0},
        {"objective": "quantile", "alpha": 0.0},
        {"objective": "fair", "fair_c": 0.0},
        {"objective": "tweedie", "tweedie_variance_power": 1.0},
        {"objective": "tweedie", "tweedie_variance_power": 2.0},
        {"objective": "tweedie", "tweedie_variance_power": 0.5},
    ],
)
def test_objective_parameter_ranges(kwargs):
    with pytest.raises(ValueError):
        MojoTreesRegressor(**kwargs)._objective_code()


def test_objective_parameter_defaults_follow_lightgbm():
    assert MojoTreesRegressor(objective="fair")._objective_param() == 1.0
    assert (
        MojoTreesRegressor(objective="tweedie")._objective_param() == 1.5
    )
    assert MojoTreesRegressor(objective="quantile")._objective_param() == 0.9
    # An objective with no scalar parameter still reports something for the
    # trainer's slot, and never fails on it.
    assert MojoTreesRegressor(objective="poisson")._objective_param() == 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"objective": "tweedie", "fair_c": 2.0},
        {"objective": "fair", "tweedie_variance_power": 1.2},
        {"objective": "regression", "fair_c": 2.0},
        {"objective": "quantile", "tweedie_variance_power": 1.2},
    ],
)
def test_a_scalar_parameter_from_another_objective_is_rejected(kwargs):
    """`fair_c` and `tweedie_variance_power` name one objective each, and
    they share the trainer's one slot, so setting the wrong one would
    quietly do nothing. LightGBM ignores it; this reports it."""
    with pytest.raises(ValueError, match="does not apply to objective"):
        MojoTreesRegressor(**kwargs)._objective_code()


def test_alpha_stays_lenient_for_the_objectives_that_ignore_it():
    """`alpha` is the shared default name several objectives ignore, and
    passing it alongside any objective is long-standing usage."""
    assert MojoTreesRegressor(
        objective="regression", alpha=0.3
    )._objective_code() == 0


@pytest.mark.parametrize(
    "objective,label,expected",
    [
        ("poisson", 2.0, 2.0),
        ("gamma", 2.0, 2.0),
        ("tweedie", 2.0, 2.0),
        ("cross_entropy", 0.75, 0.75),
    ],
)
def test_link_objectives_predict_on_the_response_scale(
    objective, label, expected
):
    """A constant target: the base score alone is the answer, and it comes
    back through the objective's inverse link rather than as a raw score."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.full(4, label)
    model = MojoTreesRegressor(
        objective=objective, n_estimators=5, num_leaves=2
    ).fit(X, y)
    assert model.predict(X) == pytest.approx(np.full(4, expected), rel=1e-6)


@pytest.mark.parametrize(
    "objective,y,message",
    [
        ("gamma", np.array([1.0, 0.0, 2.0, 3.0]), "gamma target"),
        ("tweedie", np.array([1.0, -1.0, 2.0, 3.0]), "tweedie target"),
        ("cross_entropy", np.array([0.0, 1.5, 0.5, 1.0]), "cross entropy"),
        ("poisson", np.array([1.0, -1.0, 2.0, 3.0]), "poisson target"),
    ],
)
def test_link_objectives_validate_their_labels(objective, y, message):
    """The label range each link needs is checked by the trainer, which is
    the one place that sees every entry point. It surfaces as the
    extension's own exception type rather than as a ValueError; that is how
    every fit-time Mojo error arrives today."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    with pytest.raises(Exception, match=message):
        MojoTreesRegressor(objective=objective, n_estimators=3).fit(X, y)


# -- class_weight ---------------------------------------------------------


def test_class_weight_is_a_constructor_parameter():
    est = MojoTreesClassifier(class_weight={0: 1.0, 1: 3.0})
    assert est.get_params()["class_weight"] == {0: 1.0, 1: 3.0}
    assert MojoTreesClassifier().get_params()["class_weight"] is None


def _unbalanced():
    X = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    y = np.zeros(40, dtype=np.int64)
    y[-4:] = 1
    return X, y


def test_balanced_class_weight_raises_the_minority_probability():
    X, y = _unbalanced()
    plain = MojoTreesClassifier(n_estimators=5, num_leaves=2).fit(X, y)
    balanced = MojoTreesClassifier(
        n_estimators=5, num_leaves=2, class_weight="balanced"
    ).fit(X, y)
    # Weighting moves the whole probability scale toward the rare class,
    # base score included; that is what the option is for.
    assert balanced.predict_proba(X)[:, 1].mean() > (
        plain.predict_proba(X)[:, 1].mean()
    )


def test_class_weight_dict_equals_the_same_sample_weight():
    """class_weight is expanded into row weights, so the two spellings must
    train the identical model."""
    X, y = _unbalanced()
    weights = np.where(y == 1, 5.0, 1.0)
    by_dict = MojoTreesClassifier(
        n_estimators=8, num_leaves=3, class_weight={0: 1.0, 1: 5.0}
    ).fit(X, y)
    by_hand = MojoTreesClassifier(n_estimators=8, num_leaves=3).fit(
        X, y, sample_weight=weights
    )
    assert by_dict.predict_proba(X) == pytest.approx(
        by_hand.predict_proba(X), abs=0.0
    )


def test_class_weight_multiplies_sample_weight():
    X, y = _unbalanced()
    given = np.full(len(y), 2.0)
    combined = np.where(y == 1, 6.0, 2.0)
    with_both = MojoTreesClassifier(
        n_estimators=6, num_leaves=3, class_weight={1: 3.0}
    ).fit(X, y, sample_weight=given)
    by_hand = MojoTreesClassifier(n_estimators=6, num_leaves=3).fit(
        X, y, sample_weight=combined
    )
    assert with_both.predict_proba(X) == pytest.approx(
        by_hand.predict_proba(X), abs=0.0
    )


@pytest.mark.parametrize(
    "class_weight,error",
    [
        ("unbalanced", ValueError),
        ({7: 2.0}, ValueError),
        ({0: -1.0}, ValueError),
        ({0: 0.0, 1: 0.0}, ValueError),
        (3.0, TypeError),
    ],
)
def test_class_weight_validation(class_weight, error):
    X, y = _unbalanced()
    with pytest.raises(error):
        MojoTreesClassifier(
            n_estimators=3, class_weight=class_weight
        ).fit(X, y)


# -- grow_policy ---------------------------------------------------------


def test_grow_policy_default_is_leafwise_and_names_resolve(regression):
    """The default trains the leaf-wise model exactly; XGBoost's alias
    spellings resolve to it; depth-wise is a different, balanced model."""
    from mojotrees.inspection import dump_model

    X, y = regression
    kw = dict(num_leaves=8, n_estimators=5, min_data_in_leaf=5)
    plain = MojoTreesRegressor(**kw).fit(X, y)
    assert plain.grow_policy == "leafwise"
    for alias in ("leafwise", "lossguide", "leaf_wise"):
        same = MojoTreesRegressor(grow_policy=alias, **kw).fit(X, y)
        np.testing.assert_array_equal(same.predict(X), plain.predict(X))

    deep = MojoTreesRegressor(grow_policy="depthwise", **kw).fit(X, y)
    assert not np.array_equal(deep.predict(X), plain.predict(X))
    # 8 leaves out of 300 rows at min_data_in_leaf=5: three complete levels,
    # so every leaf of every tree sits at depth 3.
    for tree in dump_model(deep)["tree_info"]:
        leaves = []
        stack = [tree["tree_structure"]]
        while stack:
            node = stack.pop()
            if "leaf_index" in node:
                leaves.append(node)
            else:
                stack.append(node["left_child"])
                stack.append(node["right_child"])
        assert len(leaves) == 8
        assert {leaf["depth"] for leaf in leaves} == {3}


def test_grow_policy_rejects_unknown_names(regression):
    X, y = regression
    with pytest.raises(ValueError, match="grow_policy"):
        MojoTreesRegressor(grow_policy="sideways", n_estimators=2).fit(X, y)
