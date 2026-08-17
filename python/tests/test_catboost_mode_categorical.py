"""CatBoost mode and categorical columns: the mirror, and what it took.

`grow_policy="symmetrictree"` is our shipped default policy and it mirrors
CatBoost, where a categorical column wider than `one_hot_max_size` is not
searched at all: it is REPLACED by its target-statistic columns and the tree
is grown on numbers. Two refusals in the package read what the binned matrix
still OFFERS to the split search, and both fire on a raw categorical column:
`tree.mojo:1874` refuses the symmetric grower, because a level shares one
split while a category partition's order comes from one node's own
statistics, and `split.mojo:778` refuses `random_strength`, because only a
categorical search's winner reaches the loop that would add the noise.

`ctr="on"` clears both, and as of 2026-08-17 it is CatBoost mode's DEFAULT.

**This file used to pin the opposite, and the history is the point.**
`_CATBOOST_CTR` was moved to `"on"` on 2026-08-17 and moved back within one
afternoon, because `SimpleCtrConfig.catboost_defaults()` was not inert on a
dataset with no categorical column: it demanded target borders that only
existed for a two-valued target, so `MojoTreesRegressor(
grow_policy="symmetrictree")` on an ordinary continuous target raised. The
default policy on the commonest task. Two builds closed that (catalog A40):
the source columns are now decided before any target border is asked for, so
a bundle with nothing to replace reads the target not at all; and the target
quantization is CatBoost's own, `BuildTargetClassifier` ->  `SelectBorders`
-> MinEntropy at one border over the ACTUAL target, so a continuous target
needs no special case.

The tests below are therefore in two halves: what CatBoost mode does with a
categorical column, and the four shapes that must keep fitting for the
default to be allowed to stay where it is. The second half is the half that
would have caught the mistake, so it is the half that must never be deleted.
"""

import numpy as np
import pytest

from mojotrees import MojoTreesClassifier, MojoTreesRegressor


def _categorical_frame(seed=0, n=400, levels=6):
    """One numeric column and one categorical column that predicts the label.

    Six levels, well under any category table, which is the case
    `ctr="auto"` declines to replace and therefore the case that used to
    fail.
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


def _regression_frame(seed=1, n=300, n_features=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = X[:, 0] + 0.5 * rng.normal(size=n)
    return X, y


# -- what CatBoost mode does with a categorical column --------------------


def test_the_default_takes_a_categorical_column():
    """The mirror, asserted at the DEFAULT.

    This test asserted the refusal until 2026-08-17. `_fit` names no `ctr`,
    so it is the shipped CatBoost-mode default that has to replace the
    six-level column, and a fit that merely succeeded without learning would
    mean the column had been dropped rather than replaced.
    """
    model, X, y = _fit(grow_policy="symmetrictree")
    assert model.score(X, y) > 0.5


def test_ctr_on_lets_the_symmetric_grower_take_a_categorical_column():
    """The same thing named explicitly, which is a different assertion: it
    still holds if the default ever moves back."""
    model, X, y = _fit(grow_policy="symmetrictree", ctr="on")
    assert model.score(X, y) > 0.5


def test_the_default_clears_random_strength_too():
    """`random_strength=1.0` is in the shipped CatBoost-mode set and its
    refusal reads the same `usable` pool the grower's does, so one
    replacement clears both. This fails separately if only one cleared."""
    model, X, y = _fit(grow_policy="symmetrictree", random_strength=1.0)
    assert model.score(X, y) > 0.5


# -- the four shapes that hold the default in place -----------------------
#
# Every one of these is a fit that the 2026-08-17 morning's default broke or
# would have broken. They are cheap and they are the whole verification bar.


def test_regression_fits_under_the_shipped_default():
    """`MojoTreesRegressor(grow_policy='symmetrictree')` and nothing else.

    The shipped default policy on the commonest task. This is the test the
    reverted commit did not have.
    """
    X, y = _regression_frame()
    model = MojoTreesRegressor(grow_policy="symmetrictree", n_estimators=6)
    model.fit(X, y)
    assert model.predict(X).shape == (300,)


def test_regression_fits_with_ctr_on_and_no_categorical_column():
    """The exact shape that caused the revert: `ctr='on'` named explicitly,
    eight NUMERIC columns, no categorical column anywhere.

    It raised about the number of distinct values in the TARGET, on a matrix
    whose feature side already decided that no CTR column would be built. A
    configuration that refuses over data it never reads is a defect
    independent of what the default is, and this is the assertion that says
    so.
    """
    X, y = _regression_frame()
    model = MojoTreesRegressor(
        grow_policy="symmetrictree", n_estimators=6, ctr="on"
    )
    model.fit(X, y)
    assert model.predict(X).shape == (300,)


def test_regression_fits_with_ctr_on_beside_a_categorical_column():
    """A continuous target AND a categorical column, so the CTR machinery
    actually runs and has to quantize a continuous target to do it.

    This is the half `select_target_borders` bought: MinEntropy at one
    border over the actual target values, `target_classifier.cpp:18-37`.
    """
    rng = np.random.default_rng(7)
    cat = rng.integers(0, 6, size=300)
    num = rng.normal(size=300)
    X = np.column_stack([num, cat.astype(float)])
    y = num + 0.5 * cat + 0.25 * rng.normal(size=300)
    model = MojoTreesRegressor(
        grow_policy="symmetrictree",
        n_estimators=6,
        ctr="on",
        categorical_feature=[1],
    )
    model.fit(X, y)
    assert model.predict(X).shape == (300,)


def test_classification_fits_with_a_categorical_column_and_random_strength():
    """Both refusals cleared at the default, on the classifier."""
    model, X, y = _fit(grow_policy="symmetrictree", random_strength=1.0)
    assert model.score(X, y) > 0.5


def test_lossguide_is_untouched():
    """`lossguide` mirrors LightGBM, which has no CTR, so its default is
    `"off"` and it searches the categorical column directly."""
    model, X, y = _fit(grow_policy="lossguide")
    assert model.score(X, y) > 0.5


def test_lossguide_regression_is_untouched():
    """The other half of "untouched": `lossguide` on a continuous target
    never reaches a CTR bundle and its default is still `'off'`."""
    X, y = _regression_frame()
    model = MojoTreesRegressor(grow_policy="lossguide", n_estimators=6)
    model.fit(X, y)
    assert model.ctr_ == "off"


# -- the target binarization, reachable from the estimator ----------------


def test_ctr_target_border_count_is_an_estimator_parameter():
    """The absence of this parameter is why the earlier refusal could not be
    worked around from this surface, so its presence is worth pinning.

    Two borders over a continuous target needs the exact dynamic program that
    is not ported, and it refuses by name rather than approximating. The
    refusal names the DP; the one it replaced named the shape of the target.
    """
    X, y = _regression_frame()
    model = MojoTreesRegressor(
        grow_policy="symmetrictree",
        n_estimators=6,
        ctr="on",
        categorical_feature=[],
        ctr_target_border_count=2,
    )
    # No categorical column, so the bundle is inert and the count is never
    # read: an inert bundle must stay inert whatever else was asked for.
    model.fit(X, y)

    rng = np.random.default_rng(7)
    cat = rng.integers(0, 6, size=300)
    num = rng.normal(size=300)
    Xc = np.column_stack([num, cat.astype(float)])
    yc = num + 0.5 * cat
    with pytest.raises(Exception) as caught:
        MojoTreesRegressor(
            grow_policy="symmetrictree",
            n_estimators=6,
            ctr="on",
            categorical_feature=[1],
            ctr_target_border_count=2,
        ).fit(Xc, yc)
    assert "dynamic program" in str(caught.value)


def test_ctr_target_border_count_is_validated_before_the_fit():
    X, y = _regression_frame()
    with pytest.raises(ValueError):
        MojoTreesRegressor(
            grow_policy="symmetrictree", ctr_target_border_count=0
        ).fit(X, y)


def test_ctr_target_border_type_is_an_estimator_parameter():
    """`'multiclass'` is CatBoost's `GetMultiClassBorders` arm. It is
    reachable by name and is NOT chosen automatically from the objective: a
    `Dataset` is built before a loss is picked and cannot see one."""
    rng = np.random.default_rng(7)
    cat = rng.integers(0, 6, size=300)
    num = rng.normal(size=300)
    X = np.column_stack([num, cat.astype(float)])
    y = (cat % 2).astype(float)
    model = MojoTreesClassifier(
        grow_policy="symmetrictree",
        n_estimators=6,
        ctr="on",
        categorical_feature=[1],
        ctr_target_border_type="multiclass",
    )
    model.fit(X, y)
    assert model.score(X, y) > 0.5

    with pytest.raises(ValueError):
        MojoTreesClassifier(
            grow_policy="symmetrictree", ctr_target_border_type="nonsense"
        ).fit(X, y)
