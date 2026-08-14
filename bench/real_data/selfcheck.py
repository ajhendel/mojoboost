"""Static checks on the harness itself. Trains nothing, downloads nothing.

    python bench/real_data/selfcheck.py

A harness that has never been run is exactly where a typo lives longest, so
this is the thing to run after editing any of it. It compiles every module,
parses every JSON file, and checks that the pieces refer to each other
consistently: every scenario has a generator that exists, a dataset that is
registered, a threshold entry, and a primary metric the quality module
knows the direction of.

It also exercises the metrics against fixtures whose answers are known by
hand, because `quality.py` is the one module in here whose output is
compared against a threshold. A metric that is quietly wrong turns the
whole suite into a machine for producing confident nonsense.

The engines are deliberately not imported. This runs with nothing but the
standard library and numpy, in any environment, in well under a second.
"""

import json
import os
import py_compile
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILURES = []


def check(condition, message):
    if condition:
        return True
    FAILURES.append(message)
    return False


def check_compiles():
    # Byte-compiled into a temporary directory rather than next to the
    # sources, so a self-check never leaves a __pycache__ behind.
    with tempfile.TemporaryDirectory() as scratch:
        for name in sorted(os.listdir(HERE)):
            if not name.endswith(".py"):
                continue
            try:
                py_compile.compile(
                    os.path.join(HERE, name),
                    cfile=os.path.join(scratch, name + "c"),
                    doraise=True,
                )
            except py_compile.PyCompileError as exc:
                FAILURES.append(f"{name} does not compile: {exc}")


def check_json():
    documents = {}
    for name in ("sources.json", "checksums.lock.json", "thresholds.json", "schema.json"):
        path = os.path.join(HERE, name)
        try:
            with open(path) as handle:
                documents[name] = json.load(handle)
        except (OSError, ValueError) as exc:
            FAILURES.append(f"{name} is not readable JSON: {exc}")
    return documents


def check_registry(documents):
    import generators
    import loaders
    import quality
    import scenarios

    sources = documents.get("sources.json", {}).get("datasets", {})
    limits = documents.get("thresholds.json", {}).get("scenarios", {})

    for name, spec in scenarios.SCENARIOS.items():
        check(
            spec.get("generator") in generators.GENERATORS,
            f"scenario {name} names generator {spec.get('generator')!r}, which does not exist",
        )
        dataset = spec.get("dataset")
        check(
            dataset is None or dataset in sources,
            f"scenario {name} names dataset {dataset!r}, which is not in sources.json",
        )
        check(name in limits, f"scenario {name} has no entry in thresholds.json")
        metric = spec.get("primary_metric")
        check(
            metric in quality.HIGHER_IS_BETTER,
            f"scenario {name} has primary metric {metric!r} with no known direction",
        )
        check(
            metric in quality.TASK_METRICS.get(spec["task"], ()),
            f"scenario {name} primary metric {metric!r} is not scored for task {spec['task']!r}",
        )
        for tier in scenarios.TIERS:
            check(
                tier in spec.get("generator_sizes", {}),
                f"scenario {name} has no {tier} size",
            )

    for name in limits:
        check(name in scenarios.SCENARIOS, f"thresholds.json has unknown scenario {name!r}")
        rules = limits[name]
        for block in ("differential", "baseline"):
            check(block in rules, f"thresholds.json {name} has no {block} rule")
        direction = rules.get("differential", {}).get("kind")
        check(
            direction in ("relative", "absolute"),
            f"thresholds.json {name} differential kind {direction!r} is not relative or absolute",
        )
        for metric in [rules.get("primary_metric")] + list(rules.get("secondary", {})):
            check(
                metric in quality.HIGHER_IS_BETTER,
                f"thresholds.json {name} gates on {metric!r}, which has no known direction",
            )

    for name in loaders.LOADERS:
        check(name in sources, f"loaders.py has a loader for unregistered {name!r}")
    for name, spec in sources.items():
        check(
            name in loaders.LOADERS,
            f"sources.json registers {name!r} with no loader",
        )
        check(
            spec["archive"]["format"] in ("zip", "bz2", "gzip", "directory"),
            f"sources.json {name} uses unsupported archive format {spec['archive']['format']!r}",
        )
        check(
            spec.get("sha256") is None,
            f"sources.json {name} carries a sha256. Digests belong in "
            "checksums.lock.json, written by a fetch that observed them",
        )

    lock = documents.get("checksums.lock.json", {})
    for name in lock.get("pins", {}):
        check(name in sources, f"checksums.lock.json pins unregistered {name!r}")


def check_params():
    import scenarios

    for name in scenarios.SCENARIOS:
        spec = scenarios.resolve(name, "standard")
        if spec["task"] == "multiclass":
            spec = dict(spec)
        extra = {"num_class": 3} if spec["task"] == "multiclass" else None
        lgb = scenarios.lightgbm_params(spec, 4, dict(extra or {}, bin_construct_sample_cnt=1000))
        mb = scenarios.mojoboost_params(spec, "cpu", extra)
        check(lgb["num_threads"] == 4, f"{name}: lightgbm thread count did not survive translation")
        check(
            "num_threads" not in mb and "n_jobs" not in mb,
            f"{name}: mojoboost params carry a thread setting, which belongs in the environment",
        )
        for key, alias in (("lambda_l2", "lambda_l2"), ("min_child_hess", "min_sum_hessian_in_leaf")):
            check(
                lgb[alias] == mb[key],
                f"{name}: {key} differs between the engines after translation",
            )
        if spec["task"] == "ranking":
            for key in ("lambdarank_truncation_level", "sigmoid", "lambdarank_norm"):
                check(
                    lgb[key] == mb[key],
                    f"{name}: ranking parameter {key} differs between the engines",
                )
            check(
                lgb["eval_at"] == [mb["ndcg_eval_at"]],
                f"{name}: the two engines were asked for NDCG at different cutoffs",
            )
        if spec["task"] == "multiclass":
            check(
                lgb["num_class"] == mb["num_class"],
                f"{name}: class count differs between the engines",
            )
        check(lgb["enable_bundle"] is False, f"{name}: lightgbm bundling was left on")
        check(
            lgb["feature_pre_filter"] is False,
            f"{name}: lightgbm feature pre-filter was left on, which deletes columns",
        )
        check(mb["device"] == "cpu", f"{name}: mojoboost device did not survive translation")


def check_metrics():
    import numpy as np

    import quality

    y = np.array([0.0, 0.0, 1.0, 1.0])
    check(quality.auc(y, np.array([0.1, 0.2, 0.3, 0.4])) == 1.0, "auc of a perfect ranking is not 1")
    check(quality.auc(y, np.array([0.4, 0.3, 0.2, 0.1])) == 0.0, "auc of a reversed ranking is not 0")
    check(
        abs(quality.auc(y, np.ones(4)) - 0.5) < 1e-12,
        "auc with every score tied is not 0.5, so ties are not averaged",
    )
    check(
        np.isnan(quality.auc(np.zeros(4), np.arange(4.0))),
        "auc with one class present is not nan",
    )
    check(
        abs(quality.average_precision(y, np.array([0.1, 0.2, 0.3, 0.4])) - 1.0) < 1e-12,
        "average precision of a perfect ranking is not 1",
    )
    check(
        abs(quality.logloss(np.array([1.0, 0.0]), np.array([0.5, 0.5])) - np.log(2)) < 1e-12,
        "log loss at p=0.5 is not log 2",
    )
    check(
        abs(quality.rmse(np.array([1.0, 3.0]), np.array([2.0, 2.0])) - 1.0) < 1e-12,
        "rmse of a unit error is not 1",
    )

    labels = np.array([3.0, 2.0, 1.0, 0.0])
    group = np.array([4])
    check(
        abs(quality.ndcg(labels, np.array([4.0, 3.0, 2.0, 1.0]), group, 4) - 1.0) < 1e-12,
        "ndcg of the ideal ordering is not 1",
    )
    reversed_score = quality.ndcg(labels, np.array([1.0, 2.0, 3.0, 4.0]), group, 4)
    check(0.0 < reversed_score < 1.0, "ndcg of the worst ordering is not strictly between 0 and 1")
    check(
        np.isnan(quality.ndcg(np.zeros(4), np.arange(4.0), group, 4)),
        "ndcg of a query with no positive label is not skipped",
    )
    check(
        quality.ndcg(np.zeros(4), np.arange(4.0), group, 4, empty_query="one") == 1.0,
        "empty_query='one' does not reproduce LightGBM's convention",
    )

    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    check(
        abs(quality.multi_logloss(np.array([0, 1]), probs) - np.log(2)) < 1e-12,
        "multiclass log loss at uniform probabilities is not log 2",
    )
    try:
        quality.ndcg(labels, np.arange(4.0), np.array([3]), 4)
    except ValueError:
        pass
    else:
        FAILURES.append("ndcg accepted a group vector that does not sum to the row count")


def check_generators_are_pure():
    """The generators must not be called here, but their signatures can be
    checked: every tier's keyword arguments have to be accepted."""
    import inspect

    import generators
    import scenarios

    for name, spec in scenarios.SCENARIOS.items():
        fn = generators.GENERATORS[spec["generator"]]
        accepted = set(inspect.signature(fn).parameters)
        for tier, kwargs in spec.get("generator_sizes", {}).items():
            unknown = set(kwargs) - accepted
            check(
                not unknown,
                f"scenario {name} tier {tier} passes {sorted(unknown)} to "
                f"{spec['generator']}, which does not take it",
            )


def check_outputs():
    """The CSV projection must not name a field the flattener cannot fill."""
    import run

    produced = set(run._flat({}))
    declared = set(run.CSV_COLUMNS)
    check(
        produced == declared,
        f"run.py CSV columns and the flattener disagree: "
        f"missing {sorted(declared - produced)}, extra {sorted(produced - declared)}",
    )


def main():
    check_compiles()
    documents = check_json()
    if not FAILURES:
        check_registry(documents)
        check_params()
        check_metrics()
        check_generators_are_pure()
        check_outputs()

    if FAILURES:
        for failure in FAILURES:
            print(f"FAIL {failure}")
        print(f"\n{len(FAILURES)} problems")
        return 1
    print("harness self-check passed. Nothing was trained and nothing was downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
