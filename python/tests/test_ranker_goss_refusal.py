"""`boosting_type="goss"` on a ranker is refused, from the top surface.

**Why this file exists rather than a unit test in `tests/`.** Until
2026-08-18 a typed `MojoTreesRanker(boosting_type="goss")` fit did all of
the following and trained a plain unsampled LambdaRank model anyway:

- `python/mojotrees/sklearn.py` validated `top_rate`, `other_rate`,
  `goss_seed` and `goss_warmup_rounds` (:3370-3392) and refused GOSS beside
  row bagging, so a malformed configuration was caught;
- it put `"goss": int(goss)` on the wire (:3492) for every fit;
- `_refuse_alternate_boosting("for a ranker")` (:2064, called at :7074 and
  :7172) covered `dart`, `rf` and `ordered` and not `goss`;
- and the three native ranker entry points -- `fit_ranker`,
  `fit_ranker_with_metrics`, `train_dataset_ranker` -- never called
  `_parse_goss`, which is called at eleven other sites in
  `bindings/_mojotrees.mojo`.

Every one of those steps passes a validation test. The model was not wrong
in the correctness sense, since an unsampled ranker is a valid model; the
LABEL was wrong, and a user reading "GOSS ranking, NDCG 0.83" was reading a
number produced by a different fit.

A predicate test on `goss.check_goss_honored` would not have caught it,
because the defect was that nothing on the ranking path ever called the
predicate. This repository has shipped blocks that were implemented,
documented, tested and unreachable for exactly that reason, so the assertion
here goes through `MojoTreesRanker` and nothing lower.

Both halves are load-bearing. The refusal must fire on a TYPED
`boosting_type`, and an UNSET one must still fit: `sampling.BootstrapRequest`
documents the rule the whole boundary keeps -- a typed parameter that cannot
be honored raises, a defaulted one is dropped silently -- and a library whose
out-of-the-box `MojoTreesRanker().fit(...)` raised would be unshippable.
"""

import numpy as np
import pytest

import mojotrees
from mojotrees import MojoTreesRanker

# Fragments the refusal message must carry, one per thing the message owes
# the reader: what was asked for, why the ranker cannot do it, what the
# ranker's sampler actually is, and what to pass instead.
_MESSAGE_PARTS = (
    "goss",
    "GossParams",
    "QUERIES",
    "bagging",
    "subsample",
)


def _ranking_data(n_queries=40, rows_per_query=5, n_features=4, seed=13):
    """A small graded-relevance ranking problem with equal-size queries.

    The label is a rank of a linear score inside each query, so the fit has
    real signal and a default `MojoTreesRanker()` converges; the refusal
    tests do not depend on that, but the "still fits" test does.
    """
    gen = np.random.default_rng(seed)
    n_rows = n_queries * rows_per_query
    X = gen.random((n_rows, n_features))
    score = 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
    y = np.zeros(n_rows, dtype=np.int32)
    for q in range(n_queries):
        lo = q * rows_per_query
        hi = lo + rows_per_query
        order = np.argsort(score[lo:hi])
        # Relevance grades 0..rows_per_query-1, capped at 3 so the labels
        # stay in the range `_check_relevance` and `label_gain` allow.
        y[lo:hi][order] = np.minimum(np.arange(rows_per_query), 3)
    group = [rows_per_query] * n_queries
    return X, y, group


def _assert_names_goss(excinfo):
    message = str(excinfo.value)
    missing = [part for part in _MESSAGE_PARTS if part not in message]
    assert not missing, f"refusal message omits {missing}: {message}"


def test_typed_goss_is_refused_on_the_plain_ranker_fit():
    """`fit_ranker`. The path a `MojoTreesRanker(...).fit(X, y, group)` with
    no eval_set takes."""
    X, y, group = _ranking_data()
    with pytest.raises(Exception) as excinfo:
        MojoTreesRanker(
            n_estimators=4,
            boosting_type="goss",
            top_rate=0.2,
            other_rate=0.1,
        ).fit(X, y, group=group)
    _assert_names_goss(excinfo)


def test_typed_goss_is_refused_on_the_eval_set_ranker_fit():
    """`fit_ranker_with_metrics`. A separate entry point with its own
    missing `_parse_goss` call, so it needs its own assertion; the one
    Python-side refusal that does exist (`_refuse_alternate_boosting`) is
    shared by both paths and covered neither."""
    X, y, group = _ranking_data()
    with pytest.raises(Exception) as excinfo:
        MojoTreesRanker(n_estimators=4, boosting_type="goss").fit(
            X,
            y,
            group=group,
            eval_set=[(X, y)],
            eval_group=[group],
        )
    _assert_names_goss(excinfo)


def test_typed_goss_is_refused_on_the_dataset_ranker_fit():
    """`train_dataset_ranker`. The third entry point, reached through
    `MojoTreesRanker.fit(Dataset)`, and the one `basic.Booster` also uses."""
    X, y, group = _ranking_data()
    dataset = mojotrees.Dataset(X, label=y, group=group)
    with pytest.raises(Exception) as excinfo:
        MojoTreesRanker(n_estimators=4, boosting_type="goss").fit(dataset)
    _assert_names_goss(excinfo)


def test_the_refusal_names_goss_and_not_dart():
    """The message must be GOSS's, not `_refuse_alternate_boosting`'s.

    `dart`, `rf` and `ordered` were already refused for a ranker with a
    sentence about dense single-output CPU fits, which is true of them and
    says nothing about query sampling. If GOSS were ever folded into that
    list the tests above would still pass while the user got the wrong
    explanation, so this pins the difference.
    """
    X, y, group = _ranking_data()
    with pytest.raises(Exception) as excinfo:
        MojoTreesRanker(n_estimators=4, boosting_type="goss").fit(
            X, y, group=group
        )
    message = str(excinfo.value)
    assert "dense, single-output models on the CPU" not in message


def test_unset_goss_still_fits_a_ranker():
    """The other half of the rule. A defaulted `boosting_type` sends
    `goss=0` on the wire, `GossParams.enabled` is False, and
    `check_goss_honored` returns quietly, so the shipped default trains.

    Asserting the fit ranks rather than merely returning: a refusal that
    fired here would be caught by the exception, but a fit that silently
    produced no trees would not.
    """
    X, y, group = _ranking_data()
    model = MojoTreesRanker(n_estimators=25, random_state=4).fit(
        X, y, group=group
    )
    scores = model.predict(X)
    assert scores.shape == (X.shape[0],)
    assert np.isfinite(scores).all()
    # The model learned something: within a query, the top-graded document
    # outscores the bottom-graded one more often than not.
    wins = 0
    for q in range(len(group)):
        lo = q * group[q]
        hi = lo + group[q]
        best = lo + int(np.argmax(y[lo:hi]))
        worst = lo + int(np.argmin(y[lo:hi]))
        wins += scores[best] > scores[worst]
    assert wins > 0.7 * len(group)


def test_explicit_gbdt_still_fits_a_ranker():
    """A TYPED `boosting_type="gbdt"` is not GOSS and must not be caught by
    the refusal. The predicate keys on `GossParams.enabled`, which is
    `int(boosting == "goss")`, so this is the case that would fail if the
    check were ever keyed on "the user named boosting_type" instead.
    """
    X, y, group = _ranking_data()
    model = MojoTreesRanker(
        n_estimators=6, boosting_type="gbdt", random_state=4
    ).fit(X, y, group=group)
    assert np.isfinite(model.predict(X)).all()


def test_goss_rates_alone_do_not_refuse_a_ranker():
    """`top_rate` and `other_rate` without `boosting_type="goss"` configure
    nothing: `sklearn.py` reads them only when the resolved boosting is
    goss. They must not refuse a ranking fit, because the user did not ask
    for GOSS.
    """
    X, y, group = _ranking_data()
    model = MojoTreesRanker(
        n_estimators=6, top_rate=0.3, other_rate=0.2, random_state=4
    ).fit(X, y, group=group)
    assert np.isfinite(model.predict(X)).all()
