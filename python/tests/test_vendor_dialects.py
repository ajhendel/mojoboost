"""A LightGBM, XGBoost or CatBoost script runs unchanged.

This is the success criterion of the canonical-naming layer
(docs/PARAMETER_NAMING.md), written as a test rather than as a claim: one
model, configured four times -- once in each vendor's spelling -- must
produce the *same model*, not a similar one.

Why it is worth a file of its own. The reason for a canonical name with
every vendor's name aliased onto it is that a configuration ported by hand
is a configuration that has been retyped, and retyping is where two
"identical" configurations quietly stop being identical: `eta` becomes
`learning_rate` but `0.05` becomes `0.5`, or `depth=6` is dropped because
this library spells it `max_depth`. A comparison run on two configurations
that differ in one number is not a comparison. So the assertion here is
exact equality of predictions, not closeness, and there is no tolerance
anywhere in this file.

Three of the four dialects also exercise a value vocabulary, not only a key
vocabulary: CatBoost writes `RMSE` and `Plain`, XGBoost writes
`reg:squarederror` and `lossguide`, and value strings are case insensitive
throughout. `grow_policy` has a file of its own,
`test_grow_policy_oblivious.py`, because reaching a grower that had no
Python spelling at all is a separate claim from resolving a name.

`_prediction_bits` compares `float64` bit patterns. Two fits of the same
configuration on the same data are bit-identical by contract (the
determinism rule in bench/results/LANE_RULES.md), so anything weaker than
`to_bits()` equality would pass while a real difference hid under it.
"""

import numpy as np
import pytest

from mojotrees import MojoTreesClassifier, MojoTreesRanker, MojoTreesRegressor


def _prediction_bits(estimator, X):
    """A fitted model's predictions as raw `float64` bit patterns.

    Exact comparison, deliberately: the point of the alias layer is that
    two spellings select the same fit, and a fit is either the same one or
    it is not.
    """
    return np.asarray(estimator.predict(X), dtype=np.float64).view(np.uint64)


def _fit(cls, X, y, **kwargs):
    return cls(**kwargs).fit(X, y)


# The one configuration, spelled four ways. Every number is identical
# across the four dictionaries; only the keys and the value words differ.
# Nothing here is at its default, because a parameter left at its default
# would pass this test whether or not its alias resolved.
_LIGHTGBM = dict(
    objective="regression",
    num_iterations=25,
    learning_rate=0.07,
    num_leaves=13,
    max_depth=4,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=0.02,
    lambda_l1=0.3,
    lambda_l2=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    bagging_seed=17,
    feature_fraction=0.9,
    feature_fraction_bynode=0.85,
    min_gain_to_split=0.01,
    max_bin=63,
    boosting="gbdt",
    grow_policy="leafwise",
    max_cat_to_onehot=5,
)

# Where a vendor has no word for a parameter at all, the canonical name is
# used: XGBoost has no minimum-rows-per-leaf and no per-component bagging
# seed, and the spec table records those cells as "-". That is what a ported
# script would have to write too, so it is what the dialect writes here.
_XGBOOST = dict(
    objective="reg:squarederror",
    n_estimators=25,
    eta=0.07,
    max_leaves=13,
    max_depth=4,
    min_child_samples=6,
    min_child_weight=0.02,
    reg_alpha=0.3,
    reg_lambda=0.7,
    subsample=0.8,
    subsample_freq=1,
    bagging_seed=17,
    colsample_bytree=0.9,
    colsample_bynode=0.85,
    gamma=0.01,
    max_bin=63,
    booster="gbtree",
    grow_policy="lossguide",
    max_cat_to_onehot=5,
)

_CATBOOST = dict(
    loss_function="RMSE",
    iterations=25,
    learning_rate=0.07,
    max_leaves=13,
    depth=4,
    min_data_in_leaf=6,
    min_child_weight=0.02,
    reg_alpha=0.3,
    l2_leaf_reg=0.7,
    subsample=0.8,
    subsample_freq=1,
    bagging_seed=17,
    rsm=0.9,
    colsample_bynode=0.85,
    min_split_gain=0.01,
    border_count=63,
    boosting_type="Plain",
    grow_policy="Lossguide",
    one_hot_max_size=5,
)

# scikit-learn's `HistGradientBoosting*` covers the fewest of these, so this
# dialect is the most mixed: only the eight cells its column of the spec
# table fills are its own spellings and the rest are canonical.
_SKLEARN_HGB = dict(
    loss="squared_error",
    max_iter=25,
    learning_rate=0.07,
    max_leaf_nodes=13,
    max_depth=4,
    min_samples_leaf=6,
    min_child_weight=0.02,
    reg_alpha=0.3,
    l2_regularization=0.7,
    subsample=0.8,
    subsample_freq=1,
    bagging_seed=17,
    colsample_bytree=0.9,
    max_features=0.85,
    min_split_gain=0.01,
    max_bins=63,
    boosting_type="gbdt",
    grow_policy="lossguide",
    max_cat_to_onehot=5,
)


@pytest.mark.parametrize(
    "dialect,spelling",
    [
        ("xgboost", _XGBOOST),
        ("catboost", _CATBOOST),
        ("sklearn_hgb", _SKLEARN_HGB),
    ],
)
def test_vendor_spelling_fits_the_lightgbm_configuration(
    regression, dialect, spelling
):
    """A configuration written in a vendor's spelling trains the model the
    LightGBM spelling of it trains, bit for bit."""
    X, y = regression
    baseline = _fit(MojoTreesRegressor, X, y, **_LIGHTGBM)
    ported = _fit(MojoTreesRegressor, X, y, **spelling)
    np.testing.assert_array_equal(
        _prediction_bits(baseline, X),
        _prediction_bits(ported, X),
        err_msg=f"the {dialect} spelling trained a different model",
    )


def test_the_configuration_is_not_the_default(regression):
    """The guard on the three tests above.

    Every one of them would pass trivially if the aliases resolved to
    nothing and all four fits ran at the defaults. This asserts that the
    configuration they share actually moves the model away from a default
    fit, so a resolution failure shows up as a difference rather than as
    four identical default models.
    """
    X, y = regression
    configured = _fit(MojoTreesRegressor, X, y, **_LIGHTGBM)
    default = _fit(MojoTreesRegressor, X, y)
    assert not np.array_equal(
        _prediction_bits(configured, X), _prediction_bits(default, X)
    )


def test_every_canonical_name_is_exercised_by_each_dialect():
    """The four dictionaries above describe one configuration.

    A parameter present in one spelling and missing from another would make
    the comparison weaker without making it fail, because the missing one
    would take a default that happened to match. This checks they are the
    same size and carry the same values, which is a property of the test
    data rather than of the library and so is asserted here rather than
    assumed.
    """
    dialects = (_LIGHTGBM, _XGBOOST, _CATBOOST, _SKLEARN_HGB)
    sizes = {len(d) for d in dialects}
    assert sizes == {len(_LIGHTGBM)}, "the dialects differ in size"
    numeric = [
        sorted(v for v in d.values() if isinstance(v, (int, float)))
        for d in dialects
    ]
    assert all(n == numeric[0] for n in numeric), (
        "the dialects carry different numbers, so they are not one "
        "configuration written four ways"
    )


def test_subsample_below_one_implies_a_frequency(regression):
    """`subsample < 1` alone bags; it is not a silent no-op.

    LightGBM leaves `bagging_freq` at 0 there, so `bagging_fraction=0.8` on
    its own changes nothing and says nothing. That defect is not copied:
    an unset frequency becomes 1. The proof is that the fit differs from an
    unbagged one and matches the fit that names the frequency outright.
    """
    X, y = regression
    unbagged = _fit(MojoTreesRegressor, X, y, n_estimators=15)
    implied = _fit(
        MojoTreesRegressor, X, y, n_estimators=15, subsample=0.7,
        bagging_seed=5,
    )
    explicit = _fit(
        MojoTreesRegressor, X, y, n_estimators=15, subsample=0.7,
        subsample_freq=1, bagging_seed=5,
    )
    assert not np.array_equal(
        _prediction_bits(unbagged, X), _prediction_bits(implied, X)
    ), "subsample=0.7 was a no-op"
    np.testing.assert_array_equal(
        _prediction_bits(implied, X), _prediction_bits(explicit, X)
    )


def test_explicit_subsample_freq_zero_still_disables_bagging(regression):
    """The fix reads an omission, not a statement. `subsample_freq=0` is a
    statement and keeps its meaning."""
    X, y = regression
    off = _fit(
        MojoTreesRegressor, X, y, n_estimators=15, subsample=0.7,
        subsample_freq=0,
    )
    unbagged = _fit(MojoTreesRegressor, X, y, n_estimators=15)
    np.testing.assert_array_equal(
        _prediction_bits(off, X), _prediction_bits(unbagged, X)
    )


def test_random_state_seeds_every_component(regression):
    """`random_state` reproduces a fit, and a seed named outright wins.

    The draws it feeds are the row bag and the feature subsets, so the
    configuration below turns both on; without them the seed would have
    nothing to change and the test would pass on a `random_state` that did
    nothing at all.
    """
    X, y = regression
    common = dict(
        n_estimators=15, subsample=0.7, colsample_bytree=0.6,
        colsample_bynode=0.8,
    )
    a = _fit(MojoTreesRegressor, X, y, random_state=3, **common)
    b = _fit(MojoTreesRegressor, X, y, random_state=3, **common)
    c = _fit(MojoTreesRegressor, X, y, random_state=8, **common)
    np.testing.assert_array_equal(
        _prediction_bits(a, X), _prediction_bits(b, X)
    )
    assert not np.array_equal(
        _prediction_bits(a, X), _prediction_bits(c, X)
    ), "random_state did not reach any draw"
    named = MojoTreesRegressor(
        random_state=3, feature_fraction_seed=99, **common
    )
    assert named._resolve_seeds()["feature_fraction_seed"] == 99
    assert named._resolve_seeds()["bagging_seed"] == 3


@pytest.mark.parametrize(
    "spelling", ["random_state", "seed", "random_seed"]
)
def test_every_seed_spelling_is_one_parameter(regression, spelling):
    X, y = regression
    common = dict(n_estimators=12, subsample=0.7, colsample_bytree=0.6)
    baseline = _fit(MojoTreesRegressor, X, y, random_state=4, **common)
    other = _fit(MojoTreesRegressor, X, y, **{spelling: 4}, **common)
    np.testing.assert_array_equal(
        _prediction_bits(baseline, X), _prediction_bits(other, X)
    )


@pytest.mark.parametrize(
    "objective", ["binary", "binary:logistic", "Logloss", "log_loss"]
)
def test_classifier_accepts_the_objective_it_trains(binary, objective):
    """Every vendor's word for two-class logistic names what a two-class
    fit here already is, so naming it is honored rather than refused.

    Before this the classifier raised for any `objective` at all, which
    failed a LightGBM, XGBoost or CatBoost classification script on a line
    that was telling the truth.
    """
    X, y = binary
    baseline = MojoTreesClassifier(n_estimators=12).fit(X, y)
    named = MojoTreesClassifier(n_estimators=12, objective=objective).fit(X, y)
    np.testing.assert_array_equal(
        np.asarray(baseline.predict_proba(X), dtype=np.float64).view(
            np.uint64
        ),
        np.asarray(named.predict_proba(X), dtype=np.float64).view(np.uint64),
    )


def test_classifier_refuses_an_objective_that_disagrees_with_y(multiclass):
    """A two-class objective on three-class labels is a mismatch and is
    named as one, not accepted and then softmaxed anyway."""
    X, y = multiclass
    with pytest.raises(ValueError, match="two-class objective"):
        MojoTreesClassifier(n_estimators=5, objective="binary").fit(X, y)


def test_classifier_still_refuses_a_foreign_objective(binary):
    X, y = binary
    with pytest.raises(ValueError, match="MojoTreesClassifier"):
        MojoTreesClassifier(n_estimators=5, objective="poisson").fit(X, y)


@pytest.mark.parametrize("objective", ["lambdarank", "rank:ndcg"])
def test_ranker_accepts_the_objective_it_trains(objective):
    """`MojoTreesRanker(objective="lambdarank")` is what an LGBMRanker
    script says, and it is true; it used to be a TypeError."""
    gen = np.random.default_rng(5)
    X = gen.random((60, 3))
    y = (X[:, 0] * 3).astype(np.int64)
    group = [20, 20, 20]
    baseline = MojoTreesRanker(n_estimators=8).fit(X, y, group=group)
    named = MojoTreesRanker(n_estimators=8, objective=objective).fit(
        X, y, group=group
    )
    np.testing.assert_array_equal(
        _prediction_bits(baseline, X), _prediction_bits(named, X)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_jobs": 4},
        {"num_threads": 4},
        {"thread_count": 4},
        {"max_ctr_complexity": 4},
        {"random_strength": 1.0},
        {"score_function": "Cosine"},
        {"bootstrap_type": "Bayesian"},
        {"bagging_temperature": 1.0},
        {"leaf_estimation_iterations": 5},
        {"boosting_type": "Ordered"},
        {"tree_method": "exact"},
    ],
)
def test_a_request_that_cannot_be_honored_is_refused_by_name(
    regression, kwargs
):
    """Refuse rather than ignore.

    Each of these names a real thing that this build does not do. Accepting
    the value and training something else is the one outcome that must not
    happen, so each raises and the message says what it would take.
    """
    X, y = regression
    with pytest.raises((ValueError, RuntimeError)):
        MojoTreesRegressor(n_estimators=3, **kwargs).fit(X, y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_jobs": -1},
        {"n_jobs": 0},
        {"thread_count": -1},
        {"random_strength": 0.0},
        {"score_function": "L2"},
        {"bootstrap_type": "No"},
        {"leaf_estimation_iterations": 1},
        {"boosting_type": "Plain"},
        {"tree_method": "hist"},
        {"verbose": 0},
        {"verbose": False},
        {"verbosity": -1},
    ],
)
def test_a_value_that_states_the_current_behavior_is_accepted(
    regression, kwargs
):
    """The other half of refuse-rather-than-ignore.

    A CatBoost script that spells out `bootstrap_type="No"` or a LightGBM
    script that writes `verbose=-1` has described what happens here anyway,
    so it runs, and the model is the one an unadorned fit trains.
    """
    X, y = regression
    baseline = MojoTreesRegressor(n_estimators=8).fit(X, y)
    ported = MojoTreesRegressor(n_estimators=8, **kwargs).fit(X, y)
    np.testing.assert_array_equal(
        _prediction_bits(baseline, X), _prediction_bits(ported, X)
    )


@pytest.mark.parametrize(
    "pair",
    [
        ("num_iterations", "iterations"),
        ("num_leaves", "max_leaves"),
        ("max_depth", "depth"),
        ("lambda_l2", "l2_leaf_reg"),
        ("feature_fraction", "rsm"),
        ("random_state", "seed"),
        ("min_gain_to_split", "gamma"),
    ],
)
def test_two_spellings_with_different_values_raise(regression, pair):
    """LightGBM warns and keeps one value when a parameter and its alias
    disagree. That is how a typo trains a model nobody asked for, so this
    raises instead."""
    X, y = regression
    first, second = pair
    with pytest.raises(ValueError, match="aliases with different values"):
        MojoTreesRegressor(**{first: 5, second: 9}).fit(X, y)


def test_get_params_round_trips_every_alias():
    """`clone` has to reconstruct an estimator from `get_params`, and an
    alias that `__init__` takes but does not store would silently drop out
    of the clone."""
    est = MojoTreesRegressor(
        iterations=17, depth=3, rsm=0.5, l2_leaf_reg=0.25,
        boosting_type="Plain", grow_policy="SymmetricTree", random_seed=9,
    )
    params = est.get_params()
    for name, value in (
        ("iterations", 17), ("depth", 3), ("rsm", 0.5),
        ("l2_leaf_reg", 0.25), ("boosting_type", "Plain"),
        ("grow_policy", "SymmetricTree"), ("random_seed", 9),
    ):
        assert params[name] == value
    clone = MojoTreesRegressor(**params)
    assert clone.get_params() == params
