"""Installation smoke test for a mojotrees wheel, numpy or no numpy.

Run against an installed package from a directory that is not the source
tree, so an accidental import of the working copy cannot make it pass:

    python packaging/smoke_test.py

It exercises the estimator surface the wheel has to ship working: fitting
on plain Python lists, predicting, the fitted attributes, get_params and
set_params, pickling, and the not-fitted error. Training behavior is
covered by python/test_python_api.py and python/tests instead.
"""

import pickle
import sys


def main():
    import mojotrees
    from mojotrees import (
        MojoTreesClassifier,
        MojoTreesRegressor,
        NotFittedError,
    )

    path = mojotrees.__file__
    if "site-packages" not in path:
        print(f"warning: importing mojotrees from {path}", file=sys.stderr)

    X = [[i / 40.0, (i % 5) / 5.0] for i in range(40)]
    y = [3.0 * r[0] + r[1] for r in X]
    labels = ["lo" if r[0] < 0.5 else "hi" for r in X]

    unfitted = MojoTreesRegressor()
    try:
        unfitted.predict(X)
        raise AssertionError("predict before fit should raise")
    except NotFittedError:
        pass

    params = unfitted.get_params()
    assert params["n_estimators"] == 100, params
    assert unfitted.set_params(n_estimators=5) is unfitted
    assert unfitted.get_params()["n_estimators"] == 5

    reg = MojoTreesRegressor(n_estimators=10, min_data_in_leaf=2).fit(X, y)
    assert reg.n_features_in_ == 2
    assert reg.best_iteration_ == 10
    assert len(reg.feature_importances_) == 2
    assert reg.score(X, y) > 0.5
    pred = list(reg.predict(X))
    assert len(pred) == len(X)

    twin = pickle.loads(pickle.dumps(reg))
    assert list(twin.predict(X)) == pred, "pickling changed the predictions"

    clf = MojoTreesClassifier(n_estimators=10, min_data_in_leaf=2).fit(
        X, labels
    )
    assert list(clf.classes_) == ["hi", "lo"]
    assert set(clf.predict(X)) <= {"hi", "lo"}
    proba = clf.predict_proba(X)
    assert all(abs(sum(row) - 1.0) < 1e-9 for row in proba)

    try:
        import numpy

        flavor = f"numpy {numpy.__version__}"
    except ImportError:
        flavor = "stdlib fallback (no numpy)"
    print(f"smoke test ok: mojotrees {mojotrees.__version__}, {flavor}")


if __name__ == "__main__":
    main()
