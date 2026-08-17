"""The Python readers of the native registry, dataset, model, startup, and
distributed status bindings.

Every check here compares a Python-side answer with the extension's own,
or exercises a binding Python did not call before the integration round.
The estimator-side objective literal (`MojoTreesRegressor._OBJECTIVES`) is
the frozen contract tools/api_snapshot.py reads; the registry is the
resolver. This file is what keeps the two from drifting.

Runs under pytest and as a script (`python test_registry_readers.py`).
"""

import io
import os
import tempfile

import numpy as np

import mojotrees
from mojotrees import MojoTreesRegressor, _eval, _fit_args, diagnostics
from mojotrees import _mojotrees as ext
from mojotrees.basic import Booster, Dataset


def _data(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n)
    return X, y


def test_metric_registry_is_native_and_complete():
    assert _eval.registry_source() == "native"
    names = _eval.metric_names()
    assert len(names) == int(ext.registry_vocabulary()["n_builtin_metrics"])
    # Every accepted spelling resolves through metric_code_of_name to the
    # code the snapshot holds for its canonical name.
    aliases = _eval.metric_aliases()
    for spelling, canonical in aliases.items():
        assert int(ext.metric_code_of_name(spelling)) == _eval._TABLE.metric_code(
            canonical
        ), spelling
    # The module constants are the registry's numbers.
    assert _eval.NDCG == int(ext.metric_code_of_name("ndcg"))
    assert _eval.MAP == int(ext.metric_code_of_name("map"))
    assert _eval.L2 == int(ext.metric_code_of_name("l2"))


def test_regressor_objective_literal_matches_registry():
    """Every spelling the estimator accepts resolves to the registry's code
    for it, through the registry or through the one documented bridge.

    **The registry is LightGBM's vocabulary and the literal is four
    vendors', and that is deliberate.** Until 2026-08-16 the literal held
    LightGBM's spellings only and every one of them resolved natively, so
    this test asked the registry directly. `objective="reg:squarederror"`,
    `"rmse"`, `"quantile_loss"` and the rest arrived that day; the registry
    was not taught them, because a name it does not know is a name the
    parameter string and the Mojo API do not accept either, and teaching it
    would widen three surfaces to serve one.

    `_Base._registry_objective_name` is the bridge that exists for this and
    names `reg:squarederror` in its own docstring: a vendor spelling is
    looked up in `_OBJECTIVES` and reported as the first LightGBM alias of
    the same code. It is what `_eval.default_metric` reads, so a fit with an
    `eval_set` and no `eval_metric` under a vendor spelling depends on it.
    Asserting through it is therefore stronger than the old direct lookup,
    not weaker: it pins the codes AND the resolution a real fit takes.
    """
    literal = MojoTreesRegressor._OBJECTIVES
    for spelling, code in literal.items():
        bridged = MojoTreesRegressor(
            objective=spelling
        )._registry_objective_name()
        assert _fit_args._objective_status(bridged) == "supported", spelling
        assert _fit_args._objective_code_of_name(bridged) == code, spelling
        # A spelling the registry knows is passed through unchanged, so the
        # bridge cannot quietly relabel a LightGBM name as another objective.
        if _fit_args._objective_status(spelling) == "supported":
            assert bridged == spelling, spelling
            assert _fit_args._objective_code_of_name(spelling) == code
    # Every registry spelling that resolves to a regression code is in the
    # literal, so the contract is complete in the direction that still holds:
    # the estimator accepts everything the registry does. The other direction
    # is the vendor aliases above and is checked through the bridge.
    regression_codes = set(literal.values())
    for spelling, code in ext.registry_objective_aliases():
        if int(code) in regression_codes:
            assert str(spelling) in literal, spelling


def test_unimplemented_objectives_come_from_registry():
    reasons = _fit_args._unimplemented_objectives()
    assert reasons, "the registry names at least one unimplemented objective"
    for spelling in reasons:
        assert _fit_args._objective_status(spelling) == "unimplemented"
        note = _fit_args._unimplemented_objective_note(spelling)
        assert reasons[spelling] in note
    assert _fit_args._objective_status("no_such_objective") == "unknown"
    assert _fit_args._unimplemented_objective_note("no_such_objective") == ""


def test_objective_param_check_is_the_trainers():
    for objective, kwargs, fragment in (
        ("huber", {"alpha": -1.0}, "huber requires alpha"),
        ("quantile", {"alpha": 2.0}, "quantile requires"),
        ("fair", {"fair_c": 0.0}, "fair requires"),
        ("tweedie", {"tweedie_variance_power": 3.0}, "tweedie requires"),
    ):
        try:
            MojoTreesRegressor(objective=objective, **kwargs)._objective_code()
        except ValueError as exc:
            assert fragment in str(exc)
        else:
            raise AssertionError(f"{objective} {kwargs} was accepted")
    # And the accepted values are accepted with no message at all.
    assert MojoTreesRegressor(objective="huber", alpha=0.5)._objective_code() >= 0


def test_dataset_field_readers():
    X, y = _data()
    w = np.linspace(1.0, 2.0, len(y))
    ds = Dataset(X, label=y, weight=w).construct()
    assert ds.num_data() == len(y) and ds.num_feature() == 3
    # get_field goes through the native buffer copy for a constructed set.
    np.testing.assert_allclose(np.asarray(ds.get_field("label")), y)
    np.testing.assert_allclose(np.asarray(ds.get_field("weight")), w)
    assert ds.get_field("init_score") is None
    for f in range(3):
        n_bin = ds.feature_num_bin(f)
        edges = ds.bin_upper_bounds(f)
        assert n_bin >= 1
        assert list(edges) == sorted(edges)
    missing = ds.missing_bins()
    assert len(missing) == 3
    # A dataset read back from disk answers the same through the handle.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ds.bin")
        ds.save_binned(path)
        loaded = Dataset.load_binned(path)
        assert loaded.num_data() == len(y)
        assert loaded.feature_num_bin(0) == ds.feature_num_bin(0)
        assert Booster.file_kind(path) == "dataset"


def test_model_kind_json_and_versions():
    X, y = _data()
    est = MojoTreesRegressor(n_estimators=3, num_leaves=4).fit(X, y)
    booster = est.booster_
    assert booster.num_trees() == 3
    text = booster.model_to_json()
    assert text.lstrip().startswith("{")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.mbst")
        booster.save_model(path)
        assert Booster.file_kind(path) == "objective"
        assert Booster.model_file_kind(path) == "objective"
        again = Booster(model_file=path)
        assert again.num_trees() == 3
    versions = mojotrees.build_info()["model_format_versions"]
    assert set(versions) == {"model_format_version", "dump_format_version"}
    assert versions["dump_format_version"] == booster.dump_model()[
        "dump_format_version"
    ]


def test_startup_contract_and_versions_report():
    assert diagnostics.check_phase_contract() == []
    info = mojotrees.build_info()
    startup = info["startup"]
    assert [p["name"] for p in startup["phases"]] == list(
        diagnostics.PHASE_NAMES
    )
    assert startup["clock_ns"] > 0
    assert set(startup["environment"]) == {"trace_enabled", "warmup_level"}
    out = io.StringIO()
    mojotrees.show_versions(file=out)
    assert "model format" in out.getvalue()


def test_distributed_readers():
    from mojotrees import _dask_runtime, dask

    record = dask.check_machine_list("10.0.0.1:5000\n10.0.0.2:5000\n", rank=1)
    assert record["world_size"] == 2 and record["is_root"] is False
    assert record["addresses"] == ["10.0.0.1:5000", "10.0.0.2:5000"]
    try:
        dask.check_machine_list(["h:1", "h:1"])
    except dask.PartitionError:
        pass
    else:
        raise AssertionError("a duplicate address was accepted")
    assert isinstance(_dask_runtime.status_message(0), str)
    assert isinstance(_dask_runtime.status_message(0, transport=True), str)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
