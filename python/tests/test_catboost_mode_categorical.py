"""CatBoost mode and categorical columns: what works, and what does not.

`grow_policy="symmetrictree"` is our shipped default policy and it mirrors
CatBoost, where a categorical column wider than `one_hot_max_size` is not
searched at all: it is REPLACED by its target-statistic columns and the tree
is grown on numbers. Two refusals in the package read what the binned matrix
still OFFERS to the split search, and both fire on a raw categorical column:
`tree.mojo:1874` refuses the symmetric grower, because a level shares one
split while a category partition's order comes from one node's own
statistics, and `split.mojo:778` refuses `random_strength`, because only a
categorical search's winner reaches the loop that would add the noise.

`ctr="on"` clears both, and it is OPT-IN. This file pins that, and pins why
it is not the default.

**The mode does not mirror CatBoost here yet, and these tests say so rather
than asserting the mirror we want.** `_CATBOOST_CTR` was moved to `"on"` and
moved back within one afternoon, because `SimpleCtrConfig.catboost_defaults()`
is not inert on a dataset with no categorical columns: it needs target borders
that only exist for a two-valued target, so `MojoTreesRegressor(
grow_policy="symmetrictree")` on an ordinary continuous target raised. The
default policy on the commonest task. `ctr_target_border_count` is not an
estimator parameter, so nothing on this surface can supply them.

The tests below are therefore in two halves: what `ctr="on"` buys, and what
stops it being free. If the second half starts failing, the build that closes
the gap has landed and the default can move -- which is exactly when someone
should be reading this file.
"""

import numpy as np
import pytest

from mojotrees import MojoTreesClassifier, MojoTreesRegressor


def _categorical_frame(seed=0, n=400, levels=6):
    """One numeric column and one categorical column that predicts the label.

    Six levels, well under any category table, which is the case
    `ctr="auto"` declines to replace and therefore the case that fails.
    """
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, levels, size=n)
    num = rng.normal(size=n)
    X = np.column_stack([num, cat.astype(float)])
    y = ((cat % 2) ^ (num > 0)).astype(int)
    return X, y


def _fit(**kwargs):
    X, y = _categorical_frame()
    kwargs.setdefault("n_estimators", 8)
    model = MojoTreesClassifier(categorical_feature=[1], **kwargs)
    return model.fit(X, y), X, y


# -- what ctr="on" buys ---------------------------------------------------


def test_ctr_on_lets_the_symmetric_grower_take_a_categorical_column():
    """The mechanism works. It is the default that does not."""
    model, X, y = _fit(grow_policy="symmetrictree", ctr="on")
    assert model.score(X, y) > 0.5, (
        "the fit succeeded but learned nothing, which would mean the "
        "categorical information was dropped rather than replaced"
    )


def test_ctr_on_clears_random_strength_too():
    """`random_strength=1.0` is in the shipped CatBoost-mode set and its
    refusal reads the same `usable` pool the grower's does, so one
    replacement clears both. This fails separately if only one cleared."""
    model, X, y = _fit(
        grow_policy="symmetrictree", random_strength=1.0, ctr="on"
    )
    assert model.score(X, y) > 0.5


# -- why it is not the default -------------------------------------------


def test_the_default_still_refuses_a_categorical_column():
    """The defect, pinned so it fails when it is fixed.

    This is not the behavior anyone wants. It is the behavior, and a test
    that asserted the mirror instead would be asserting an intention.
    """
    with pytest.raises(Exception) as caught:
        _fit(grow_policy="symmetrictree")
    assert "OFFERS a categorical column" in str(caught.value)


def test_ctr_on_breaks_regression_which_is_why_it_is_not_the_default():
    """The reason the default did not move, pinned in the same file.

    No categorical column at all: `catboost_defaults()` is not inert without
    one, and a continuous target has no two-valued midpoint to put a CTR
    border on. When this stops raising, `ctr_target_border_count` has become
    reachable (or CatBoost mode derives it), and `_CATBOOST_CTR` can be
    `"on"`.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 8))
    y = X[:, 0] + 0.5 * rng.normal(size=300)
    with pytest.raises(Exception) as caught:
        MojoTreesRegressor(
            grow_policy="symmetrictree", n_estimators=6, ctr="on"
        ).fit(X, y)
    assert "ctr target borders" in str(caught.value)


def test_regression_fits_under_the_shipped_default():
    """The half that matters most and would have caught the mistake.

    `MojoTreesRegressor(grow_policy="symmetrictree")` and nothing else. The
    shipped default policy on the commonest task must fit.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 8))
    y = X[:, 0] + 0.5 * rng.normal(size=300)
    model = MojoTreesRegressor(grow_policy="symmetrictree", n_estimators=6)
    model.fit(X, y)
    assert model.predict(X).shape == (300,)


def test_lossguide_is_untouched():
    """`lossguide` mirrors LightGBM, which has no CTR, so its default is
    `"off"` and it searches the categorical column directly."""
    model, X, y = _fit(grow_policy="lossguide")
    assert model.score(X, y) > 0.5
