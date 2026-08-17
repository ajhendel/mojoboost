"""`grow_policy="symmetrictree"` reaches the oblivious grower.

`growth_policy.GROW_OBLIVIOUS` shipped in the Mojo package and
`src/mojotrees/tree.mojo` grows it, but the estimator validated
`grow_policy` against a table holding the two frontier orders and nothing
else. Every `bench/real_data` arm and every scikit-learn user goes through
that validator, so a policy that had been built, tested and merged could not
be asked for through the API at all.

The assertions here are about reachability, not about tree shape: that the
value survives to a fit rather than being accepted and dropped, and that
every spelling of it selects the same fit. Accepted-and-ignored is worse
than rejected, so the load-bearing test is the one that would fail if the
string were swallowed -- a symmetric tree is a different tree from a
leaf-wise one, so the predictions must differ.

Exact comparison throughout: two fits of one configuration on one dataset
are bit-identical by contract, so anything weaker would pass while a real
difference hid under it.
"""

import numpy as np
import pytest

from mojotrees import MojoTreesRegressor


def _bits(estimator, X):
    """Predictions as raw `float64` bit patterns."""
    return np.asarray(estimator.predict(X), dtype=np.float64).view(np.uint64)


def _fit(X, y, **kwargs):
    return MojoTreesRegressor(**kwargs).fit(X, y)


def test_symmetrictree_reaches_the_oblivious_grower(regression):
    """The value changes the model, which is what an accepted-and-dropped
    value would fail to do."""
    X, y = regression
    common = dict(n_estimators=12, max_depth=4, learning_rate=0.1)
    lossguide = _fit(X, y, grow_policy="lossguide", **common)
    oblivious = _fit(X, y, grow_policy="symmetrictree", **common)
    assert not np.array_equal(_bits(lossguide, X), _bits(oblivious, X)), (
        "grow_policy='symmetrictree' was accepted and then ignored"
    )


def test_oblivious_trees_are_symmetric(regression):
    """The stronger claim: it reaches the oblivious grower specifically,
    not merely a different one.

    A symmetric tree of depth d has exactly 2**d leaves, all at depth d,
    whatever the data does -- that is the property the mode exists for, and
    neither of the two frontier orders produces it at these settings
    (`max_leaves` defaults to 31, so a leaf-wise or depth-wise tree of
    depth 4 is not forced to fill).
    """
    from mojotrees.inspection import dump_model

    X, y = regression
    model = _fit(X, y, n_estimators=4, max_depth=3, grow_policy="symmetrictree")
    for tree in dump_model(model)["tree_info"]:
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


@pytest.mark.parametrize(
    "spelling",
    ["symmetrictree", "SymmetricTree", "symmetric_tree", "oblivious",
     "symmetric", "SYMMETRIC"],
)
def test_every_symmetric_spelling_is_one_policy(regression, spelling):
    """Value strings are case insensitive, and CatBoost writes
    `SymmetricTree`."""
    X, y = regression
    common = dict(n_estimators=10, max_depth=4)
    baseline = _fit(X, y, grow_policy="symmetrictree", **common)
    other = _fit(X, y, grow_policy=spelling, **common)
    np.testing.assert_array_equal(_bits(baseline, X), _bits(other, X))


@pytest.mark.parametrize(
    "spelling", ["lossguide", "leafwise", "leaf_wise", "Lossguide"]
)
def test_every_lossguide_spelling_is_one_policy(regression, spelling):
    """`lossguide` is canonical and `leafwise` is the alias; both are the
    growth every fit before this parameter existed used, so this also
    guards the default."""
    X, y = regression
    common = dict(num_leaves=8, n_estimators=5, min_data_in_leaf=5)
    default = _fit(X, y, **common)
    named = _fit(X, y, grow_policy=spelling, **common)
    np.testing.assert_array_equal(_bits(default, X), _bits(named, X))


@pytest.mark.parametrize(
    "spelling", ["depthwise", "depth_wise", "Depthwise"]
)
def test_every_depthwise_spelling_is_one_policy(regression, spelling):
    X, y = regression
    common = dict(num_leaves=8, n_estimators=5, min_data_in_leaf=5)
    baseline = _fit(X, y, grow_policy="depthwise", **common)
    other = _fit(X, y, grow_policy=spelling, **common)
    np.testing.assert_array_equal(_bits(baseline, X), _bits(other, X))


def test_the_three_policies_are_three_models(regression):
    """The guard on the spelling tests above: they would all pass if every
    spelling resolved to one policy.

    **The leaf budget has to BIND, and that is the whole reason this
    configuration is not the one beside it.** It read
    `dict(n_estimators=10, max_depth=4)` and left `num_leaves` at 31. A
    complete tree of depth 4 has 16 leaves, so the budget never bound, and
    `lossguide` and `depthwise` differ only in the ORDER they spend a
    budget: with none to spend, both split every splittable node and grow
    the same tree, node for node. The two arms came back bit-identical and
    the guard failed on a true statement about the parameters rather than
    on anything about the policies. Corrected 2026-08-17.

    `num_leaves=8` binds for both frontier orders (best-first picks eight
    leaves by gain, a level at a time fills three complete levels) and is
    ignored by the symmetric grower, which a level splits entirely or not
    at all, so all three arms stay distinguishable for reasons the policy
    names.
    """
    X, y = regression
    common = dict(
        n_estimators=10, max_depth=4, num_leaves=8, min_data_in_leaf=5
    )
    fits = [
        _bits(_fit(X, y, grow_policy=p, **common), X)
        for p in ("lossguide", "depthwise", "symmetrictree")
    ]
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            assert not np.array_equal(fits[i], fits[j])


def test_the_default_is_lossguide():
    """The canonical spelling is what `get_params` reports, so a cloned
    estimator carries a value this validator accepts."""
    est = MojoTreesRegressor()
    assert est.grow_policy == "lossguide"
    assert MojoTreesRegressor(**est.get_params()).grow_policy == "lossguide"


def test_an_unknown_policy_names_all_three_canonical_values(regression):
    """The message a user gets has to list what they may say, and it listed
    two of the three."""
    X, y = regression
    with pytest.raises(ValueError) as excinfo:
        MojoTreesRegressor(grow_policy="sideways", n_estimators=2).fit(X, y)
    message = str(excinfo.value)
    for value in ("lossguide", "depthwise", "symmetrictree"):
        assert value in message


def test_oblivious_takes_catboosts_depth_when_none_is_named(regression):
    """`max_depth` is the only bound on a symmetric tree, and an unset one
    resolves to CatBoost's `depth` default of 6 rather than being refused.

    This test asserted the refusal until 2026-08-17. The refusal is real and
    still lives in `growth_policy` -- a symmetric grower will not run
    unbounded -- but it stopped being reachable from this estimator, on
    purpose: `grow_policy='symmetrictree'` is CatBoost mode, CatBoost mode
    supplies CatBoost's own defaults, and `_CATBOOST_DEPTH = 6`
    (`oblivious_tree_options.cpp:12`) is the one of them without which
    `MojoTreesRegressor(grow_policy='symmetrictree')` and nothing else
    raised on every fit. A shipped default that cannot fit is the defect
    that change closed.

    So the assertion is the substitution and not the absence of a refusal:
    the unset fit is the depth-6 fit **bit for bit**. That is the claim a
    silently different bound would fail, and it is a claim the old refusal
    test could not make. `max_depth` is not one of CatBoost's four
    learning-rate gate keys, so naming it changes nothing else in the
    resolution and the two arms are comparable exactly.
    """
    X, y = regression
    unset = _fit(X, y, grow_policy="symmetrictree", n_estimators=2)
    named = _fit(
        X, y, grow_policy="symmetrictree", n_estimators=2, max_depth=6
    )
    np.testing.assert_array_equal(_bits(unset, X), _bits(named, X))
    # -1 is the absence of a bound and not a depth, so it reads the same way
    # rather than reaching the grower and being refused there.
    explicit = _fit(
        X, y, grow_policy="symmetrictree", n_estimators=2, max_depth=-1
    )
    np.testing.assert_array_equal(_bits(unset, X), _bits(explicit, X))
