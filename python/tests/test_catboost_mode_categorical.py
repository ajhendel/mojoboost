"""CatBoost mode takes a categorical column, because its CTRs replace one.

`grow_policy="symmetrictree"` is our shipped default policy and it mirrors
CatBoost, where a categorical column wider than `one_hot_max_size` is not
searched at all: it is REPLACED by its target-statistic columns and the tree
is grown on numbers. Two refusals in the package read what the binned matrix
still OFFERS to the split search, and both fire on a raw categorical column:
`tree.mojo:1874` refuses the symmetric grower, because a level shares one
split while a category partition's order comes from one node's own
statistics, and `split.mojo:778` refuses `random_strength`, because only a
categorical search's winner reaches the loop that would add the noise.

Both refusals are correct and neither should fire on the shipped default,
because the shipped default replaces the column before either can see it.

**This is a regression test for a default, not for a mechanism.** The
mechanism worked the whole time. `_CATBOOST_CTR` was `"auto"`, and `"auto"`
gives CTR columns only to the categorical columns that overflowed their
category table -- a different question, reading a different number. A column
narrower than the table kept its raw code, stayed inside `BinnedMatrix.usable`
and was still offered to the search, so
`MojoTreesClassifier(grow_policy="symmetrictree", categorical_feature=[...])`
raised on every fit. One word, and the difference between a mode that handles
categorical data the way CatBoost does and a mode that cannot take it at all.

The fixture is deliberately small and deliberately narrow: six levels, well
under any category table, which is exactly the case `"auto"` declined to
replace and therefore the case that failed.
"""

import numpy as np
import pytest

from mojotrees import MojoTreesClassifier


def _frame(seed=0, n=400, levels=6):
    """One numeric column and one categorical column that predicts the label.

    The label reads BOTH columns, so a fit that dropped the categorical
    information entirely would still fit and still be wrong; the accuracy
    assertion below is what notices.
    """
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, levels, size=n)
    num = rng.normal(size=n)
    X = np.column_stack([num, cat.astype(float)])
    y = ((cat % 2) ^ (num > 0)).astype(int)
    return X, y


def _fit(**kwargs):
    X, y = _frame()
    kwargs.setdefault("n_estimators", 8)
    model = MojoTreesClassifier(categorical_feature=[1], **kwargs)
    return model.fit(X, y), X, y


def test_symmetric_mode_fits_a_categorical_column_by_default():
    """The load-bearing one. No `ctr` named, nothing but the policy."""
    model, X, y = _fit(grow_policy="symmetrictree")
    assert model.score(X, y) > 0.5, (
        "the fit succeeded but learned nothing, which would mean the "
        "categorical information was dropped rather than replaced"
    )


def test_symmetric_mode_fits_beside_random_strength():
    """`random_strength=1.0` is in the shipped CatBoost-mode set, and its
    refusal reads the same `usable` pool the grower's does. One replacement
    clears both, so this fails separately if only one of them was cleared."""
    model, X, y = _fit(grow_policy="symmetrictree", random_strength=1.0)
    assert model.score(X, y) > 0.5


def test_lossguide_is_untouched():
    """`lossguide` mirrors LightGBM, which has no CTR, so its default stays
    `"off"` and it searches the categorical column directly. It has always
    been able to, and this asserts the change did not reach it."""
    model, X, y = _fit(grow_policy="lossguide")
    assert model.score(X, y) > 0.5


@pytest.mark.parametrize("rule", ["off", "auto"])
def test_a_named_rule_still_gets_that_rule(rule):
    """A caller who NAMES a rule gets the rule and its consequence.

    This is the half that makes the change a default rather than a special
    case: `ctr="off"` and `ctr="auto"` leave a six-level column searchable,
    the symmetric grower refuses, and it refuses with the sentence that says
    why. If this ever starts passing, the mode has begun overriding what the
    caller asked for.
    """
    with pytest.raises(Exception) as caught:
        _fit(grow_policy="symmetrictree", ctr=rule)
    assert "OFFERS a categorical column" in str(caught.value)
