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

The engine LIBRARIES are deliberately not imported. `engines.py` itself is,
so that the peer-arm check can prove the CatBoost-mode arm goes through its
own translator rather than being the plain arm under another name, but every
`import lightgbm` and `import catboost` in that module is inside a method
and none of them runs here. This needs the standard library, **numpy and pandas** (the categorical
fixture builds a mixed float-and-integer frame, which numpy has no dtype
for), in well under a second. **pandas is why "any environment" is no longer
true**, and it is declared under `[feature.bench.dependencies]` in
`pixi.toml`: run this as `pixi run -e bench python bench/real_data/selfcheck.py`
rather than with whatever `python3` is on PATH.
"""

import json
import os
import py_compile
import struct
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


def _routed(name, canonical, side, routing, translated, dataset):
    """The value a shared parameter has on one side, wherever it lives.

    Returns (found, value). `found` is False for the "engine" destination,
    where the parameter is a call argument rather than a dict entry and
    there is nothing here to read; that case is checked differently, by
    proving there is no competing copy in the dicts.
    """
    where, key = routing
    if where == "train":
        if key not in translated:
            FAILURES.append(
                f"{name}: {canonical} routes to {side} {key!r} in the "
                f"training dict and is not there"
            )
            return False, None
        return True, translated[key]
    if where == "dataset":
        if key not in dataset:
            FAILURES.append(
                f"{name}: {canonical} routes to {side} {key!r} in the "
                f"dataset dict and is not there"
            )
            return False, None
        return True, dataset[key]
    # "engine": the adapter passes it at the call site from BASE_PARAMS, so
    # both sides get the same number by construction. What can go wrong is
    # a second copy appearing in a dict and quietly winning, which is what
    # this checks instead.
    check(
        key not in translated,
        f"{name}: {canonical} is a call argument on the {side} side, but "
        f"{key!r} is also in its parameter dict, so there are now two "
        "sources for one number",
    )
    return False, None


def check_params():
    """Every shared parameter, on both sides, under whichever name and in
    whichever dict it ends up.

    This used to check two of them, which is how LightGBM sat at
    min_data_in_bin 3 against a binner with no minimum population for as
    long as it did. A divergence in the alignment should fail here, in a
    check that runs in under a second and trains nothing, rather than in
    an audit of the results months later.
    """
    import scenarios

    for name in scenarios.SCENARIOS:
        spec = scenarios.resolve(name, "standard")
        if spec["task"] == "multiclass":
            spec = dict(spec)
        extra = {"num_class": 3} if spec["task"] == "multiclass" else None
        # No `bin_construct_sample_cnt` injected: this is the dict the
        # adapter now builds, and the point of the two checks below is that
        # nothing adds one.
        lgb = scenarios.lightgbm_params(spec, 4, dict(extra or {}))
        mb = scenarios.mojotrees_params(spec, "cpu", extra)
        ds = scenarios.dataset_params(spec)
        check(lgb["num_threads"] == 4, f"{name}: lightgbm thread count did not survive translation")
        check(
            "num_threads" not in mb and "n_jobs" not in mb,
            f"{name}: mojotrees params carry a thread setting, which belongs in the environment",
        )

        for canonical in scenarios.BASE_PARAMS:
            routing = scenarios.SHARED_PARAM_ROUTING.get(canonical)
            if routing is None:
                FAILURES.append(
                    f"{canonical} is in BASE_PARAMS with no row in "
                    "SHARED_PARAM_ROUTING, so nothing checks that both "
                    "engines were given it"
                )
                continue
            lgb_routing, mb_routing = routing
            # LightGBM's Dataset and train take the one dict, so its
            # training dict is also its dataset dict. mojotrees's are two.
            lgb_found, lgb_value = _routed(
                name, canonical, "lightgbm", lgb_routing, lgb, lgb
            )
            mb_found, mb_value = _routed(
                name, canonical, "mojotrees", mb_routing, mb, ds
            )
            if lgb_found and mb_found:
                check(
                    lgb_value == mb_value,
                    f"{name}: {canonical} differs between the engines after "
                    f"translation, {lgb_value!r} against {mb_value!r}",
                )

        # The categorical block reaches both sides under one set of names,
        # so it needs no routing table, only the same equality.
        for canonical in scenarios.CATEGORICAL_PARAMS:
            if canonical in lgb or canonical in mb:
                check(
                    lgb.get(canonical) == mb.get(canonical),
                    f"{name}: categorical parameter {canonical} differs "
                    f"between the engines, {lgb.get(canonical)!r} against "
                    f"{mb.get(canonical)!r}",
                )

        # `min_data_in_bin` used to be pinned to 1 on the LightGBM side,
        # because mojotrees's numerical binner had no minimum-population
        # rule and LightGBM's default of 3 would have merged levels
        # mojotrees kept. mojotrees's binner defaults to 3 now, so the pin
        # is gone from LIGHTGBM_ALIGNMENT and both engines run stock: this
        # check is not "is it 1" any more, it is "is neither side pinning
        # it", which is what the two assertions below say. Reading
        # `lgb["min_data_in_bin"]` here would now raise KeyError.
        check(
            "min_data_in_bin" not in lgb,
            f"{name}: lightgbm min_data_in_bin is pinned to "
            f"{lgb.get('min_data_in_bin')!r}; both binners default to 3 now, "
            "so a pin here makes the comparator bin differently from us for "
            "a reason this repository imposed",
        )
        check(
            "bin_construct_sample_cnt" not in lgb,
            f"{name}: lightgbm bin_construct_sample_cnt is pinned to "
            f"{lgb.get('bin_construct_sample_cnt')!r}; both engines sample "
            "200,000 rows by default, so a pin here is inverted rather than "
            "dropped and buys us binning time the stock comparison does not",
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
        check(mb["device"] == "cpu", f"{name}: mojotrees device did not survive translation")


def check_comparator():
    """The comparator is what it says it is, and cannot quietly grow.

    There is exactly one comparator, `stock+det`: LightGBM at its own
    defaults plus `deterministic=true`. Everything else it is passed has to
    be declared, either as a deviation from stock with a reason and an exit
    condition, or as a harness setting that changes neither the model nor
    the work.

    The failure this guards is not hypothetical and it is not slow to
    happen. The previous comparator carried six settings that had accreted
    one at a time, each defensible on its own, and together they described
    an engine no user runs. Every one of them made the comparison easier
    for us in at least one respect. A pin added by editing one dict now
    fails here, in a check that trains nothing and takes a second.
    """
    import json

    import scenarios

    declared = set(scenarios.LIGHTGBM_DEVIATIONS_FROM_STOCK) | set(
        scenarios.LIGHTGBM_HARNESS_SETTINGS
    )
    for key, value in scenarios.LIGHTGBM_ALIGNMENT.items():
        check(
            key in declared,
            f"the comparator passes {key}={value!r} and declares it nowhere. "
            "Add it to LIGHTGBM_DEVIATIONS_FROM_STOCK with a reason and a "
            "removed_when, or to LIGHTGBM_HARNESS_SETTINGS if it changes "
            "neither the model nor the work, or take it out",
        )
        check(
            key not in scenarios.LIGHTGBM_STOCK_DEFAULTS,
            f"{key} is both passed by the comparator and listed as left at "
            "LightGBM's default. It cannot be both",
        )

    for key, entry in scenarios.LIGHTGBM_DEVIATIONS_FROM_STOCK.items():
        check(
            key in scenarios.LIGHTGBM_ALIGNMENT,
            f"{key} is declared as a deviation from stock and is not passed",
        )
        check(
            scenarios.LIGHTGBM_ALIGNMENT.get(key) == entry["here"],
            f"{key} is declared as {entry['here']!r} and passed as "
            f"{scenarios.LIGHTGBM_ALIGNMENT.get(key)!r}",
        )
        check(
            entry["stock"] != entry["here"],
            f"{key} is declared as a deviation from stock and its declared "
            "stock value is what it passes, so it is not a deviation",
        )
        check(
            bool(entry.get("why")) and bool(entry.get("removed_when")),
            f"{key} deviates from stock without both a reason and the "
            "condition that removes it",
        )

    spec = scenarios.resolve("dense_regression", "standard")
    resolved = scenarios.lightgbm_params(spec, 4)
    check(
        resolved.get("deterministic") is True,
        "the comparator is stock+det and the resolved dict does not set "
        "deterministic",
    )
    check(
        resolved.get("feature_pre_filter") is False,
        "feature_pre_filter is load-bearing until mojotrees implements the "
        "filter, and the resolved dict does not pin it off",
    )
    for absent in (
        "bin_construct_sample_cnt",
        "min_data_in_bin",
        "force_row_wise",
        "force_col_wise",
        "use_quantized_grad",
    ):
        check(
            absent not in resolved,
            f"the resolved comparator sets {absent}={resolved.get(absent)!r}. "
            "It is stock in stock+det: see LIGHTGBM_STOCK_DEFAULTS",
        )

    block = scenarios.comparator_block()
    for field in (
        "id", "label", "registered", "one_line", "lightgbm_passed",
        "lightgbm_left_at_stock", "lightgbm_defaults_source",
        "reproducibility", "like_for_like",
    ):
        check(field in block, f"comparator_block has no {field!r}")
    try:
        json.dumps(block)
    except (TypeError, ValueError) as exc:
        FAILURES.append(
            f"comparator_block does not serialise, so no run can record it: {exc}"
        )
    check(
        block.get("lightgbm_passed") == scenarios.LIGHTGBM_ALIGNMENT,
        "comparator_block reports a different dict from the one the engines "
        "are passed",
    )


def check_no_row_count_injection():
    """No caller rebuilds the binning pin that was just removed.

    `scenarios.lightgbm_params` refuses `bin_construct_sample_cnt` and
    `min_data_in_bin` by name, which catches any call site at runtime. This
    catches it before a run: `check_params` builds its dict from
    `scenarios.lightgbm_params` and cannot see what an adapter adds to the
    dict afterwards, and the two adapters are exactly where the row-count
    injection lived. So the adapters' own source is read.

    The source is parsed rather than grepped. Both files explain at length
    why the parameter is not set, in comments and in docstrings, and that
    explanation is the thing most worth keeping; a text search finds the
    explanation and calls it the offence. What is looked for is the name
    used as a value: a dict key, a subscript, or a keyword argument.

    The CatBoost names below are the same defect in CatBoost's vocabulary
    and they are here for the same reason. `border_count` is its binning
    budget, `dev_max_subset_size_for_build_borders` is its direct
    counterpart of `bin_construct_sample_cnt`, `used_ram_limit` changes its
    blocking and its quantization strategy, and the two bundling knobs are
    the CatBoost shape of `enable_bundle`.

    This static list is deliberately SHORTER than
    `scenarios.CATBOOST_REFUSED_PARAMS`, and the difference is not an
    oversight. That list also refuses `max_bin`, `min_data_in_leaf`,
    `min_child_samples`, `subsample` and `thread_count`, every one of which
    is a legitimate string constant somewhere in these two files: `max_bin`
    and `min_data_in_leaf` are shared parameters both other engines are
    given by name. A static rule that flagged those would fire on correct
    code, and a check that cries wolf is a check that gets deleted. Those
    names are caught at their real call site instead, by
    `scenarios.catboost_params` raising.
    """
    import ast

    targets = {
        "bench/real_data/engines.py": os.path.join(HERE, "engines.py"),
        "bench/bench_lightgbm.py": os.path.join(
            os.path.dirname(HERE), "bench_lightgbm.py"
        ),
    }
    pinned = (
        "bin_construct_sample_cnt",
        "min_data_in_bin",
        "border_count",
        "dev_max_subset_size_for_build_borders",
        "dev_efb_max_buckets",
        "sparse_features_conflict_fraction",
        "used_ram_limit",
    )
    for name, path in targets.items():
        try:
            with open(path) as handle:
                tree = ast.parse(handle.read(), filename=path)
        except (OSError, SyntaxError) as exc:
            FAILURES.append(f"{name} could not be parsed for the pin check: {exc}")
            continue
        prose = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                prose.add(id(node.value))
        for node in ast.walk(tree):
            hit = None
            if isinstance(node, ast.keyword) and node.arg in pinned:
                hit = node.arg
            elif (
                isinstance(node, ast.Constant)
                and node.value in pinned
                and id(node) not in prose
            ):
                hit = node.value
            if hit is None:
                continue
            FAILURES.append(
                f"{name}:{getattr(node, 'lineno', 0)} sets {hit} in code. "
                "Every name on this list is a binning or bundling knob that "
                "is left stock on the engine that owns it, and every engine "
                "here runs the value that engine chose; pinning one makes a "
                "comparator or a peer arm bin differently from us for a "
                "reason this repository imposed. The row-count form of it "
                "made the LightGBM comparator do strictly more binning work "
                "than mojotrees does, in our favor, and it was caught only "
                "after a ratio had been published"
            )


def _as_float32(value):
    """`value` rounded to the nearest float32, as a Python float.

    XGBoost stores its tree parameters as C floats, so `save_config()` returns
    `eta` as 0.300000012 where the arm passes the Python double 0.3. Comparing
    the two exactly fails forever on a difference in the eighth significant
    figure; comparing them under a hand-picked tolerance is a number nobody can
    justify. Rounding OUR value the way XGBoost rounds it is the comparison
    that is actually being asked for, and it stays exact: a mirror value that
    is genuinely wrong still differs after the round.
    """
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _xgboost_config_from_defaults(scenarios):
    """A `save_config()`-shaped dict built from `XGBOOST_RESOLVED_DEFAULTS`.

    Only `check_xgboost_arm`'s round trip uses it: a drift check fed the table
    it checks against must report nothing, and if it reports something then the
    paths and the values in that table disagree with each other. That is a real
    failure mode, because the paths were transcribed from one read and a typo
    in one of them would make the check silently vacuous on that key forever.
    """
    root = {}
    for entry in scenarios.XGBOOST_RESOLVED_DEFAULTS.values():
        pieces = entry["path"].split(".")
        node = root
        for piece in pieces[:-1]:
            node = node.setdefault(piece, {})
        node[pieces[-1]] = entry["value"]
    return root


def check_catboost_arm():
    """The CatBoost peer arm is what it says it is, does not become the
    comparator, and cannot quietly grow.

    Three separate things are guarded here and the first is the one that
    matters most.

    **The headline row does not move.** A peer arm is only allowed to be
    additive: `COMPARATOR_ID`, `LIGHTGBM_ALIGNMENT` and every field
    `comparator_block()` had before are checked to be untouched by anything
    in this section, and `peers` is checked to be a key beside them rather
    than a replacement for any of them.

    **The tree count is matched and the learning rate deliberately is not.**
    The first is checked against the resolved dicts rather than asserted in a
    docstring. The second is checked as a NEGATIVE: `learning_rate` must not
    be passed to CatBoost and must not have a static value on our side,
    because `cb-shipped` runs each engine at its own resolved rate and a
    reinstated pin would silently turn the row back into `cb-default` under
    the new label.

    **The two resolved dicts agree key by key.** This is the check that
    changed on 2026-08-16 and the reason this function is worth having.
    Before that date it compared `MOJOTREES_CATBOOST_MODE` against a dict
    built from `MOJOTREES_CATBOOST_MODE`, which proves the translator does not
    DROP a key -- a real thing to prove, and it caught `grow_policy` -- and
    which cannot tell whether any value in the dict is RIGHT. It now walks
    `scenarios.CATBOOST_PARAM_MAP` and fails on any key claimed matchable
    whose two resolved values differ, on any key CatBoost resolves that
    nothing classifies, and on any unmatchable claim that does not name a live
    `CATBOOST_UNMATCHABLE` entry.

    **The arm cannot grow a setting.** Same rule as `check_comparator`:
    everything CatBoost is passed is either a declared deviation from stock
    with a reason and an exit condition, or a harness setting that changes
    neither the model nor the work.
    """
    import json

    import engines
    import scenarios
    import verify

    # 1. The comparator is untouched.
    check(
        scenarios.COMPARATOR_ID == "stock+det",
        f"the comparator id is {scenarios.COMPARATOR_ID!r}. Adding a peer "
        "arm is not allowed to move the headline row",
    )
    check(
        set(scenarios.LIGHTGBM_ALIGNMENT) == {
            "deterministic", "feature_pre_filter", "enable_bundle",
            "verbosity", "seed",
        },
        "LIGHTGBM_ALIGNMENT changed while a peer arm was added. The peer "
        "arm is additive: it may not touch what the comparator is passed",
    )
    block = scenarios.comparator_block()
    check("peers" in block, "comparator_block does not carry the peer arms")
    check(
        block.get("id") == scenarios.COMPARATOR_ID
        and block.get("lightgbm_passed") == scenarios.LIGHTGBM_ALIGNMENT,
        "comparator_block's own fields moved when the peer arm was added",
    )
    try:
        json.dumps(block)
    except (TypeError, ValueError) as exc:
        FAILURES.append(
            f"comparator_block stopped serializing once it carried the peer "
            f"arm, so no run can record either: {exc}"
        )

    # 2. The arm declares everything it passes.
    arm = scenarios.catboost_arm_block()
    check(
        arm.get("is_the_comparator") is False,
        "the CatBoost arm does not record itself as a non-comparator",
    )
    declared = set(scenarios.CATBOOST_DEVIATIONS_FROM_STOCK) | set(
        scenarios.CATBOOST_HARNESS_SETTINGS
    )
    for key, value in scenarios.CATBOOST_ALIGNMENT.items():
        check(
            key in declared,
            f"the CatBoost arm passes {key}={value!r} and declares it "
            "nowhere. Add it to CATBOOST_DEVIATIONS_FROM_STOCK with a reason "
            "and a removed_when, or to CATBOOST_HARNESS_SETTINGS if it "
            "changes neither the model nor the work, or take it out",
        )
        check(
            key not in scenarios.CATBOOST_LEFT_AT_STOCK,
            f"{key} is both passed to CatBoost and listed as left at its "
            "default. It cannot be both",
        )
    for key in scenarios.CATBOOST_MATCHED:
        check(
            key in scenarios.CATBOOST_DEVIATIONS_FROM_STOCK,
            f"{key} is matched onto the CatBoost arm and is not declared as "
            "a deviation from CatBoost's own default",
        )
    for key, entry in scenarios.CATBOOST_DEVIATIONS_FROM_STOCK.items():
        check(
            key in scenarios.CATBOOST_ALIGNMENT
            or key in scenarios.CATBOOST_MATCHED,
            f"{key} is declared as a CatBoost deviation and is not passed",
        )
        check(
            bool(entry.get("why")) and bool(entry.get("removed_when")),
            f"{key} deviates from CatBoost's default without both a reason "
            "and the condition that removes it",
        )
        check(
            entry.get("stock") != entry.get("here"),
            f"{key} is declared as a deviation and its declared stock value "
            "is what it passes, so it is not a deviation",
        )

    # 2b. The learning rate is deliberately NOT matched, and the shape of
    #     that decision is checked rather than trusted.
    #
    #     Every clause below is a way the pin could come back without anybody
    #     meaning it to: through CATBOOST_MATCHED, through CATBOOST_ALIGNMENT,
    #     through `extra` at a call site, or through a static value in
    #     MOJOTREES_CATBOOST_MODE. cb-shipped and cb-default compute different
    #     models, so a silent reversion is a published number under the wrong
    #     label.
    check(
        "learning_rate" not in scenarios.CATBOOST_MATCHED,
        "learning_rate is back in CATBOOST_MATCHED. cb-shipped runs each "
        "engine at its own resolved rate; see CATBOOST_DELIBERATE_DIVERGENCE",
    )
    check(
        "learning_rate" not in scenarios.CATBOOST_ALIGNMENT,
        "learning_rate is passed to CatBoost through CATBOOST_ALIGNMENT, "
        "which makes the arm cb-default under cb-shipped's label",
    )
    check(
        "learning_rate" in scenarios.CATBOOST_REFUSED_PARAMS,
        "learning_rate is not refused by name. It was PASSED until "
        "2026-08-16, so 'not currently passed' is not enough: a caller "
        "handing it through `extra` would restore the old model silently",
    )
    check(
        "learning_rate" in scenarios.CATBOOST_DELIBERATE_DIVERGENCE,
        "the learning rate is unmatched and CATBOOST_DELIBERATE_DIVERGENCE "
        "does not say so, so a reader of a CatBoost accuracy number has "
        "nothing telling them the two engines ran different rates",
    )
    check(
        "learning_rate" in scenarios.CATBOOST_RESOLVED_PER_FIT,
        "learning_rate has no CATBOOST_RESOLVED_PER_FIT entry, so nothing "
        "records that its value cannot be known without a read-back",
    )
    check(
        "learning_rate" not in scenarios.CATBOOST_LEFT_AT_STOCK,
        "CATBOOST_LEFT_AT_STOCK carries a learning_rate. That table holds "
        "constants and CatBoost derives this one from the dataset, so any "
        "value there is false on every shape but the one it was read on",
    )
    check(
        "learning_rate" not in scenarios.MOJOTREES_CATBOOST_MODE,
        "MOJOTREES_CATBOOST_MODE carries a static learning_rate. It has to "
        "come from CatBoost's own read-back for the same cell: a constant "
        "here is a hand-written belief about a value that moves with the "
        "dataset. See MOJOTREES_CATBOOST_MODE_FROM_READBACK",
    )
    check(
        "learning_rate" in scenarios.MOJOTREES_CATBOOST_MODE_FROM_READBACK,
        "learning_rate is not declared as coming from the read-back, so "
        "nothing makes the CatBoost-mode arm take CatBoost's resolved rate",
    )
    for _key in scenarios.MOJOTREES_CATBOOST_MODE_FROM_READBACK:
        check(
            _key in scenarios.MOJOTREES_CATBOOST_MODE_REASONS,
            f"{_key} comes from the read-back with no reason recorded beside "
            "it in MOJOTREES_CATBOOST_MODE_REASONS",
        )
        check(
            _key not in scenarios.MOJOTREES_CATBOOST_MODE,
            f"{_key} is both a static entry of MOJOTREES_CATBOOST_MODE and a "
            "read-back entry. It cannot be both, and the static value would "
            "win or lose depending on dict order",
        )
    check(
        scenarios.CATBOOST_ARM_ID != "cb-default",
        "the arm id is still cb-default while the arm no longer pins the "
        "learning rate. Two materially different models must not share an id",
    )
    check(
        bool(getattr(scenarios, "CATBOOST_ARM_SUPERSEDES", "")),
        "the arm does not record what it supersedes, so a reader holding a "
        "cb-default number has nothing telling them it is stale",
    )
    # The arm REFUSES rather than falling back. Checked by calling it, not by
    # reading the source, because the fallback this guards against is exactly
    # the kind that looks correct in a diff.
    _spec = scenarios.resolve(
        scenarios.MOJOTREES_CATBOOST_MODE_PARITY_SCENARIOS[0], "standard"
    )
    try:
        scenarios.mojotrees_catboost_mode_params(_spec, "cpu", None)
    except scenarios.CatBoostReadbackMissing:
        pass
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(
            "the CatBoost-mode arm built without a read-back failed with the "
            f"wrong error type: {type(exc).__name__}: {exc}"
        )
    else:
        FAILURES.append(
            "the CatBoost-mode arm BUILT without CatBoost's read-back. It "
            "must refuse by name: every key in "
            "MOJOTREES_CATBOOST_MODE_FROM_READBACK has no static value, so "
            "whatever it used for the learning rate is a guess, and a guess "
            "of 0.1 against CatBoost's resolved 0.43 is the largest single "
            "error this arm can make"
        )

    # 3. Matched tree count, on the resolved dicts, on every scenario the arm
    #    runs.
    for name in scenarios.SCENARIOS:
        spec = scenarios.resolve(name, "standard")
        runs, reason = scenarios.catboost_supports(spec)
        listed = "catboost" in spec["engines"]
        check(
            runs == listed,
            f"{name}: catboost_supports says {runs} and the scenario's "
            f"engine list says {listed}",
        )
        if not runs:
            check(
                bool(reason),
                f"{name} does not run the CatBoost arm and gives no reason",
            )
            continue
        # The CatBoost side first, and unconditionally. These used to sit
        # below the CatBoost-mode gate and stopped running the moment that arm
        # was parked, which would have taken the refusal list offline at the
        # same time as the arm it protects.
        extra = {"num_class": 3} if spec["task"] == "multiclass" else None
        cb = scenarios.catboost_params(spec, 4, dict(extra or {}))
        check(
            cb["iterations"] == scenarios.BASE_PARAMS["n_estimators"],
            f"{name}: the CatBoost arm is not at the matched tree count, "
            f"{cb['iterations']!r} against "
            f"{scenarios.BASE_PARAMS['n_estimators']!r}",
        )
        check(
            "learning_rate" not in cb,
            f"{name}: the resolved CatBoost dict passes learning_rate="
            f"{cb.get('learning_rate')!r}. cb-shipped lets CatBoost resolve "
            "its own; passing one is cb-default",
        )
        check(
            cb["thread_count"] == 4,
            f"{name}: catboost thread count did not survive translation",
        )
        # Nothing binning-shaped, and nothing on the refusal list, reached
        # CatBoost. `thread_count` is on that list because it must not arrive
        # from a caller, not because it must be absent: catboost_params sets
        # it itself from the runner's number, exactly as num_threads is set on
        # the LightGBM side.
        for absent in scenarios.CATBOOST_REFUSED_PARAMS:
            if absent == "thread_count":
                continue
            check(
                absent not in cb,
                f"{name}: the resolved CatBoost dict sets {absent}="
                f"{cb.get(absent)!r}. It is stock in "
                f"{scenarios.CATBOOST_ARM_LABEL}",
            )
        for refused in scenarios.CATBOOST_REFUSED_PARAMS:
            try:
                scenarios.catboost_params(spec, 4, {refused: 1})
            except ValueError:
                pass
            else:
                FAILURES.append(
                    f"{name}: catboost_params accepted {refused}, which is "
                    "the CatBoost shape of the row-count binning pin"
                )
        # The CatBoost-mode arm pairs with the CatBoost arm wherever it can,
        # and where it cannot it must say so by name. Until the arm carried
        # row sampling this was an unconditional check and it was right to be
        # one: every key in MOJOTREES_CATBOOST_MODE was tree shape or
        # regularization, and every trainer reads those. `bootstrap_type` is
        # the first key our own trainers disagree about, and the multiclass
        # and sparse trainers refuse it by name rather than dropping it.
        #
        # The teeth are kept: a missing arm still fails unless
        # MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT gives a reason, and a
        # scenario the table says should run but that is absent from the
        # engine list fails too. What is no longer asserted is that the two
        # arms cover the same scenarios, because they now do not.
        mode_reason = scenarios.MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT[name]
        mode_listed = "mojotrees_catboost_mode" in spec["engines"]
        check(
            mode_listed == (mode_reason is None),
            f"{name}: MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT says "
            f"{'runs' if mode_reason is None else 'skipped'} and the "
            f"scenario's engine list says "
            f"{'runs' if mode_listed else 'skipped'}",
        )
        check(
            mode_listed or bool(mode_reason),
            f"{name} runs the CatBoost arm with no mojotrees "
            "CatBoost-mode arm beside it and gives no reason, so 'us in "
            "CatBoost mode' has no row and nothing says why",
        )

    # 3b. THE PARITY GATE. Our arm's resolved parameters against CatBoost's,
    #     key by key, on every scenario in
    #     MOJOTREES_CATBOOST_MODE_PARITY_SCENARIOS.
    #
    #     Run over that tuple rather than over the currently-scheduled cells
    #     on purpose. The scheduling table can park the arm -- it is parked
    #     right now, waiting on the read-back wiring -- and a gate that goes
    #     quiet whenever the thing it guards is in flux is not a gate. This
    #     one keeps checking the parameters of an arm that is not running, so
    #     that unparking it is a one-line edit rather than a re-audit.
    for name in scenarios.MOJOTREES_CATBOOST_MODE_PARITY_SCENARIOS:
        check(
            name in scenarios.SCENARIOS,
            f"MOJOTREES_CATBOOST_MODE_PARITY_SCENARIOS names {name!r}, which "
            "is not a scenario",
        )
        if name not in scenarios.SCENARIOS:
            continue
        spec = scenarios.resolve(name, "standard")
        extra = {"num_class": 3} if spec["task"] == "multiclass" else None
        mb = scenarios.mojotrees_params(spec, "cpu", extra)
        stand_in = scenarios._READBACK_STANDIN(spec)
        cm = scenarios.mojotrees_catboost_mode_resolved(
            spec, "cpu", extra, stand_in
        )
        rows = scenarios.catboost_parity_rows(spec, "cpu", extra)
        for message in scenarios.catboost_parity_failures(rows):
            FAILURES.append(f"{name}: {message}")
        # Every row's verdict is well formed, which is what stops a key being
        # dropped by giving it a shape nothing enforces.
        for row in rows:
            entry = scenarios.CATBOOST_PARAM_MAP.get(row["catboost"], {})
            if row["status"] == "unmatchable":
                check(
                    row.get("unmatchable_key")
                    in scenarios.CATBOOST_UNMATCHABLE,
                    f"{name}: {row['catboost']} is declared unmatchable "
                    f"against CATBOOST_UNMATCHABLE"
                    f"[{row.get('unmatchable_key')!r}], which does not exist. "
                    "An unmatchable key has to point at the reason, or it is "
                    "a key that was dropped with a label on it",
                )
            if row["status"] in ("unmatchable", "not_reached"):
                check(
                    bool(row.get("detail")),
                    f"{name}: {row['catboost']} is {row['status']} with no "
                    "reason. Every key this arm does not match carries the "
                    "reason it does not",
                )
            # `required_when_scenarios` names the scenarios that would make a
            # key live, and what it demands depends on the verdict.
            #
            # not_reached: none of them may be scheduled for this arm, or the
            # key is reached and unmatched.
            #
            # matched: the key must be NAMED in MOJOTREES_CATBOOST_MODE rather
            # than agreeing through CATBOOST_PARAM_MAP's declared default,
            # because a match by coincidence is one default change away from a
            # silent divergence. This is what keeps `max_cat_to_onehot=2` set:
            # it is inert on all three scenarios this arm runs today, so
            # nothing else in this file would notice it being deleted.
            for reachable in entry.get("required_when_scenarios", ()):
                live = (
                    scenarios.MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT.get(
                        reachable, "skipped"
                    )
                    is None
                )
                if row["status"] == "not_reached":
                    check(
                        not live,
                        f"{row['catboost']} is declared not_reached because "
                        f"no scenario this arm runs reaches it, and "
                        f"{reachable} now runs it. Either set the matching "
                        "key in MOJOTREES_CATBOOST_MODE and move this row to "
                        "matched, or record why the two arms may differ on it",
                    )
                else:
                    check(
                        row["mojotrees"] in scenarios.MOJOTREES_CATBOOST_MODE,
                        f"{row['catboost']} -> {row['mojotrees']} is matched "
                        "and required_when_scenarios names "
                        f"{reachable}, and MOJOTREES_CATBOOST_MODE does not "
                        f"set {row['mojotrees']}. It agrees by default rather "
                        "than by being asked for, so one default change on "
                        "either side makes the two arms differ with nothing "
                        "recording it",
                    )
        # The old teeth, kept. A mode entry the translator drops produces no
        # diff against the plain arm and must still fail: the parity table
        # would catch it only for keys CatBoost also resolves, and
        # MOJOTREES_CATBOOST_MODE holds keys it does not (min_child_hess,
        # lambda_l1).
        moved = {k for k in set(mb) | set(cm) if mb.get(k) != cm.get(k)}
        declared_moves = set(scenarios.MOJOTREES_CATBOOST_MODE) | set(
            scenarios.MOJOTREES_CATBOOST_MODE_FROM_READBACK
        ) | {"n_estimators", "max_bin", "use_missing"}
        check(
            moved <= declared_moves,
            f"{name}: the CatBoost-mode arm differs from the plain mojotrees "
            f"arm in {sorted(moved - declared_moves)}, which "
            "MOJOTREES_CATBOOST_MODE does not name",
        )
        for key, value in scenarios.MOJOTREES_CATBOOST_MODE.items():
            check(
                key in cm and cm[key] == value,
                f"{name}: MOJOTREES_CATBOOST_MODE sets {key}={value!r} and "
                f"the resolved arm has {cm.get(key, '<absent>')!r}. A mode "
                "entry the translator drops produces an arm that is neither "
                "CatBoost's shape nor ours",
            )
        for key in scenarios.MOJOTREES_CATBOOST_MODE:
            check(
                key in scenarios.MOJOTREES_CATBOOST_MODE_REASONS,
                f"MOJOTREES_CATBOOST_MODE sets {key} with no reason recorded "
                "beside it",
            )
        # Every key CatBoost resolves is classified. The loop above only
        # produces rows for keys the map knows; this is the other direction.
        declared = scenarios.catboost_resolved_declared(spec, None, extra)
        for key, value in sorted(declared.items()):
            classified = (
                key in scenarios.CATBOOST_PARAM_MAP
                or key in scenarios.CATBOOST_PARAM_NOT_MAPPED
            )
            check(
                classified,
                f"{name}: CatBoost resolves {key}={value!r} and neither "
                "CATBOOST_PARAM_MAP nor CATBOOST_PARAM_NOT_MAPPED says "
                "whether this arm matches it. A key nobody classified is the "
                "hand-written belief this table exists to remove",
            )
            entry = scenarios.CATBOOST_PARAM_NOT_MAPPED.get(key)
            if entry is None:
                continue
            check(
                bool(entry.get("why")),
                f"{key} is excluded from the parity diff with no reason",
            )
            holds = entry.get("holds_while")
            if holds is not None:
                check(
                    scenarios._parity_equal(holds, value),
                    f"{name}: {key} is excluded from the parity diff on the "
                    f"grounds that it is inert at {holds!r}, and CatBoost now "
                    f"resolves {value!r}. The exclusion's premise is gone: "
                    "either the key is matchable and belongs in "
                    "CATBOOST_PARAM_MAP, or the reason in "
                    "CATBOOST_PARAM_NOT_MAPPED has to be rewritten for the "
                    "new value",
                )

    # 3c. The parity map itself is well formed, independent of any scenario.
    for key, entry in scenarios.CATBOOST_PARAM_MAP.items():
        check(
            entry.get("verdict") in ("matched", "unmatchable", "not_reached"),
            f"CATBOOST_PARAM_MAP[{key!r}] has verdict "
            f"{entry.get('verdict')!r}; expected matched, unmatchable or "
            "not_reached",
        )
        check(
            entry.get("translate", "identity")
            in scenarios._PARITY_TRANSLATORS,
            f"CATBOOST_PARAM_MAP[{key!r}] names translation "
            f"{entry.get('translate')!r}, which _PARITY_TRANSLATORS does not "
            "define",
        )
        check(
            key not in scenarios.CATBOOST_PARAM_NOT_MAPPED,
            f"{key} is both mapped and declared not mapped. It cannot be both",
        )
        if entry.get("verdict") == "matched":
            check(
                bool(entry.get("ours")),
                f"CATBOOST_PARAM_MAP[{key!r}] is matched and names no key on "
                "our side",
            )
    # The two mechanisms for the learning rate are mutually exclusive, and
    # half-wiring both is an arm that runs a rate neither engine chose.
    _readback_keys = set(scenarios.MOJOTREES_CATBOOST_MODE_FROM_READBACK)
    _unset_keys = set(scenarios.MOJOTREES_CATBOOST_MODE_UNSET)
    check(
        not (_readback_keys & _unset_keys),
        f"{sorted(_readback_keys & _unset_keys)} is both taken from CatBoost's "
        "read-back and removed from the parameters. Removals run before the "
        "read-back is applied, so the key would be set anyway and the removal "
        "would look effective while doing nothing. See "
        "CATBOOST_LEARNING_RATE_TRANSITION",
    )
    check(
        not (_unset_keys & set(scenarios.MOJOTREES_CATBOOST_MODE)),
        f"{sorted(_unset_keys & set(scenarios.MOJOTREES_CATBOOST_MODE))} is "
        "both set by MOJOTREES_CATBOOST_MODE and removed by "
        "MOJOTREES_CATBOOST_MODE_UNSET. The removal wins and the value is "
        "dead, which reads as a pin that is not one",
    )
    for _key in _unset_keys:
        check(
            _key in scenarios.MOJOTREES_CATBOOST_MODE_REASONS,
            f"{_key} is removed from the CatBoost-mode parameters with no "
            "reason recorded beside it in MOJOTREES_CATBOOST_MODE_REASONS",
        )
    check(
        bool(scenarios.CATBOOST_LEARNING_RATE_TRANSITION.get("today"))
        and bool(
            scenarios.CATBOOST_LEARNING_RATE_TRANSITION.get(
                "where_the_comparison_lives"
            )
        ),
        "CATBOOST_LEARNING_RATE_TRANSITION does not say how the arm gets the "
        "rate today or where the comparison against CatBoost's own value "
        "lives, and two mechanisms are in flight for that one value",
    )
    check(
        set(scenarios.MOJOTREES_CATBOOST_MODE_FROM_READBACK.values())
        <= set(scenarios.CATBOOST_PARAM_MAP),
        "a read-back key is not in the parity map, so nothing checks that the "
        "value the arm took is the value CatBoost ran",
    )
    for _ours, _theirs in scenarios.MOJOTREES_CATBOOST_MODE_FROM_READBACK.items():
        check(
            scenarios.CATBOOST_PARAM_MAP.get(_theirs, {}).get("static", True)
            is False,
            f"{_theirs} comes from the read-back and CATBOOST_PARAM_MAP does "
            "not mark it static: False, so the static diff would compare it "
            "against a value this file cannot honestly hold",
        )
    # The parity block a published table carries is the same table the gate
    # saw, and it serializes.
    parity = scenarios.resolved_parity_block()
    for name, entry in parity.items():
        check(
            "error" not in entry,
            f"the resolved parity table could not be built for {name}: "
            f"{entry.get('error')}",
        )
        # The same failures the gate above already reported, seen through the
        # field a published record carries. Both are checked because they can
        # come apart: the gate walks catboost_parity_rows directly and this
        # walks the block catboost_arm_block writes, so a bug that drops the
        # rows on their way into the record shows up here and only here.
        check(
            not entry.get("failures"),
            f"{name}: the parity table recorded in every manifest carries "
            f"{len(entry.get('failures') or ())} failing key(s): "
            f"{entry.get('failures')}",
        )
    try:
        json.dumps(parity)
    except (TypeError, ValueError) as exc:
        FAILURES.append(
            f"resolved_parity_block does not serialize, so no record can "
            f"carry it: {exc}"
        )

    # 4. The wiring the runner needs.
    for engine in scenarios.PEER_ENGINES:
        check(
            engine in engines.ENGINES,
            f"{engine} is named by a scenario and is not in engines.ENGINES",
        )
        check(
            engines.ENGINE_ARM.get(engine, "").startswith("peer"),
            f"{engine} is not recorded as a peer arm, so a reader of a "
            "record cannot tell it from the comparator row",
        )
    check(
        engines.ENGINES["mojotrees_catboost_mode"].params_fn
        is scenarios.mojotrees_catboost_mode_params,
        "the CatBoost-mode arm does not go through "
        "mojotrees_catboost_mode_params, so it is the plain arm under "
        "another name",
    )
    check(
        engines.ENGINES["mojotrees"].params_fn is scenarios.mojotrees_params,
        "the plain mojotrees arm's translator moved when the peer arm was "
        "added",
    )

    # The depth-wise subject variant. Three properties, and each one is a
    # defect this harness has already had once under another name.
    check(
        "mojotrees_depthwise" in engines.ENGINES,
        "the depthwise arm is not in engines.ENGINES, so worker.py cannot "
        "build it by name",
    )
    check(
        engines.ENGINES["mojotrees_depthwise"].params_fn
        is scenarios.mojotrees_depthwise_params,
        "the depthwise arm does not go through mojotrees_depthwise_params, "
        "so it is the plain arm under another name and its row would report "
        "leaf-wise growth as depth-wise",
    )
    check(
        engines.ENGINE_ARM.get("mojotrees_depthwise") == "subject_variant",
        "the depthwise arm is not recorded as subject_variant, so a reader "
        "of a record cannot tell it from the headline mojotrees row",
    )
    check(
        "mojotrees_depthwise" not in scenarios.PEER_ENGINES,
        "the depthwise arm is listed as a peer, and it is our own engine "
        "rather than a competitor",
    )
    # The depthwise arm is the XGBoost MIRROR as of 2026-08-17, and this check
    # replaces one that asserted the opposite. What stood here was
    # `MOJOTREES_DEPTHWISE == {"grow_policy": "depthwise"}`, guarding the arm
    # against ever growing a second key, which was the right guard for an
    # isolation arm and is the wrong one now: Andrew pointed the arm at
    # XGBoost's defaults ("make our depthwise params match xgboost").
    #
    # The guard that replaces it is stronger than the one it replaces, because
    # it checks the VALUES against something outside itself. Every entry is
    # compared to `XGBOOST_RESOLVED_DEFAULTS`, which was read out of a live
    # `Booster.save_config()`, and the arithmetic ones are checked as
    # arithmetic. A dict compared only against itself proves the translator
    # does not drop a key and cannot tell whether any value is right; that is
    # the lesson `check_catboost_arm`'s docstring records from the other side.
    check(
        scenarios.MOJOTREES_DEPTHWISE.get("grow_policy") == "depthwise",
        "the depthwise arm does not grow depth-wise. That is XGBoost's "
        "resolved grow_policy and the one key this arm has always had",
    )
    _xgb_defaults = scenarios.XGBOOST_RESOLVED_DEFAULTS
    for _ours, _theirs in (
        ("max_depth", "max_depth"),
        ("learning_rate", "eta"),
        ("lambda_l2", "lambda"),
        ("min_child_hess", "min_child_weight"),
    ):
        _want = _xgb_defaults[_theirs]["value"]
        _got = scenarios.MOJOTREES_DEPTHWISE.get(_ours)
        # Both sides rounded to float32, which is the precision XGBoost keeps
        # a tree parameter in. See `_as_float32`; without it `eta` fails
        # forever, because 0.3 reads back as 0.300000012.
        check(
            _got is not None
            and _as_float32(_got) == _as_float32(_want),
            f"the depthwise arm sets {_ours}={_got!r} where XGBoost resolves "
            f"{_theirs}={_want!r} ({_xgb_defaults[_theirs]['path']}). This arm "
            "mirrors XGBoost's defaults, so a value that is not XGBoost's is "
            "a row wearing a label it does not match. Compared at float32, "
            "the precision XGBoost keeps these in, so this is a real "
            "difference and not a rounding one. See MOJOTREES_DEPTHWISE",
        )
    check(
        _xgb_defaults["max_leaves"]["value"] == "0"
        and scenarios.MOJOTREES_DEPTHWISE.get("num_leaves")
        == 2 ** int(_xgb_defaults["max_depth"]["value"]),
        "the depthwise arm's num_leaves is not 2**max_depth. XGBoost's "
        "max_leaves resolves to 0, meaning unbounded, so a depthwise tree at "
        "its resolved max_depth may hold 2**max_depth leaves, while "
        "growth_policy.mojo keeps num_leaves a HARD bound and admits a "
        "gain-ranked prefix of the level that would overshoot. Any smaller "
        "value compares a truncated tree against XGBoost's and calls it a "
        "shape match. See XGBOOST_UNMATCHABLE['leaf_bound']",
    )
    check(
        scenarios.MOJOTREES_DEPTHWISE.get("min_data_in_leaf") == 1,
        "the depthwise arm imposes a row-count rule on a leaf. XGBoost has "
        "none at all: it bounds a leaf by hessian mass through "
        "min_child_weight and no resolved parameter counts rows. Leaving the "
        "shared 20 makes our trees smaller than XGBoost's for a reason "
        "invisible in the parameter table. See "
        "XGBOOST_UNMATCHABLE['leaf_row_count']",
    )
    check(
        "min_child_weight" not in scenarios.MOJOTREES_DEPTHWISE,
        "the depthwise arm passes min_child_weight, which is an ALIAS of "
        "min_child_hess in our own surface (python/mojotrees/sklearn.py::_Base). Setting both "
        "spellings raises, and setting the alias hides which quantity the arm "
        "meant. XGBoost's min_child_weight is a HESSIAN SUM and maps onto "
        "min_child_hess, never onto min_data_in_leaf",
    )
    check(
        "max_bin" not in scenarios.MOJOTREES_DEPTHWISE,
        "the depthwise arm carries max_bin. That is a BINNING parameter and "
        "this dict is a TRAINING override applied to what mojotrees_params "
        "returns, so it would be handed to mojotrees.train, which does not "
        "take one. The bin budget is a dataset parameter and the gap to "
        "XGBoost's resolved 256 is declared rather than closed: see "
        "XGBOOST_UNMATCHABLE['bin_budget']",
    )
    check(
        set(scenarios.MOJOTREES_DEPTHWISE_SCENARIO_SUPPORT)
        == set(scenarios.SCENARIOS),
        "MOJOTREES_DEPTHWISE_SCENARIO_SUPPORT does not have one entry per "
        "scenario, so a scenario was added without a support decision for "
        "the depthwise arm",
    )
    check(
        "mojotrees_depthwise" in verify.SUBJECT_ENGINES,
        "the depthwise arm is not in verify.SUBJECT_ENGINES, so its GPU rows "
        "would be published with no backend proof and no cpu-versus-gpu "
        "agreement check",
    )
    # The XGBoost peer arm, added 2026-08-17. The mojotrees side of this
    # pairing is `mojotrees_depthwise`, checked above; there is no separate
    # mirror engine and the checks that asserted one are gone with it.
    check(
        "xgboost" in engines.ENGINES,
        "the xgboost peer arm is not in engines.ENGINES, so worker.py cannot "
        "build it by name",
    )
    check(
        "xgboost" in scenarios.PEER_ENGINES
        and engines.ENGINE_ARM.get("xgboost") == "peer",
        "the xgboost arm is not recorded as a peer, so a reader of a record "
        "cannot tell it from the comparator row",
    )
    check(
        "xgboost" not in verify.SUBJECT_ENGINES,
        "the xgboost arm is in verify.SUBJECT_ENGINES. It is a competitor "
        "library, so the subject gates would demand a mojotrees backend proof "
        "and a cpu-versus-gpu agreement from a CPU-only competitor column",
    )
    check(
        "xgboost" not in verify.DEFAULT_ACCURACY_PEER["competitors"],
        "xgboost joined the peer scoreboard's competitor set. Whether the "
        "comparison is against the better of two engines or of three is "
        "Andrew's decision and not a wiring detail: adding an engine can only "
        "make us look further behind, and this one runs its own learning rate "
        "of 0.3 at a matched tree count. This mattered more when the number "
        "gated; since 2026-08-17 it is a column. See the comment on "
        "verify.DEFAULT_ACCURACY_PEER",
    )
    check(
        "xgboost" in scenarios.peer_arms_block(),
        "peer_arms_block does not carry the xgboost arm, so its whole "
        "configuration reaches no manifest, no records.json and no CSV "
        "column. Every constant existed for a day before this was noticed, "
        "because this function is the only route any of them takes",
    )
    check(
        scenarios.XGBOOST_DETERMINISM.get("status") == "seeded, not guaranteed",
        "the XGBoost arm claims a determinism status other than 'seeded, not "
        "guaranteed'. XGBoost has no deterministic flag, so a fixed thread "
        "count plus a fixed seed is not the guarantee the comparator carries",
    )
    # The read-back is wired both ways: the table the mirror was built from
    # must be reachable by the checker that re-reads it, and the checker must
    # actually fail on a value that moved. The second half is checked by
    # feeding it a configuration with one value changed, because a drift check
    # that silently passes everything is indistinguishable from one that works
    # and is exactly what this harness has been bitten by before.
    _paths_ok = all(
        isinstance(entry, dict) and "path" in entry and "value" in entry
        for entry in scenarios.XGBOOST_RESOLVED_DEFAULTS.values()
    )
    check(
        _paths_ok,
        "an entry of XGBOOST_RESOLVED_DEFAULTS is missing its path or its "
        "value, so check_xgboost_readback cannot re-read it and the mirror "
        "arm is asserted against nothing",
    )
    _fake = {"learner": {"gradient_booster": {"tree_train_param": {
        "grow_policy": "lossguide",
    }}}}
    check(
        len(scenarios.check_xgboost_readback(_fake))
        == len(scenarios.XGBOOST_RESOLVED_DEFAULTS),
        "check_xgboost_readback did not report every asserted default as "
        "drifted on a configuration that holds one wrong value and nothing "
        "else. A drift check that passes a configuration it has never seen is "
        "not a check",
    )
    check(
        not scenarios.check_xgboost_readback(
            _xgboost_config_from_defaults(scenarios)
        ),
        "check_xgboost_readback reports drift against a configuration built "
        "from XGBOOST_RESOLVED_DEFAULTS itself, so the paths and the values "
        "in that table disagree with each other",
    )
    # The rng_state trim is a REPORTING reduction and must not reach the
    # checker, which reads the raw configuration. Checked because the two run
    # over the same dict shape and swapping them would leave a check that
    # passes on a digest.
    _trimmed = scenarios.xgboost_readback_for_record(
        {"learner": {"generic_param": {"rng_state": "1 2 3", "seed": "190019"}}}
    )
    check(
        isinstance(
            _trimmed["learner"]["generic_param"]["rng_state"], dict
        )
        and _trimmed["learner"]["generic_param"]["seed"] == "190019",
        "xgboost_readback_for_record did not replace rng_state with its "
        "digest, or it touched a field beside it. It is a size reduction of "
        "the record and must never be a reduction of what the record says",
    )
    check(
        scenarios.XGBOOST_DETERMINISM.get("status") == "seeded, not guaranteed",
        "the XGBoost arm claims a determinism status other than 'seeded, not "
        "guaranteed'. XGBoost has no deterministic flag, so a fixed thread "
        "count plus a fixed seed is not the guarantee the comparator carries",
    )
    for engine in ("mojotrees", "lightgbm", "catboost", "xgboost"):
        check(
            engine in scenarios.PHASE_SHAPE,
            f"PHASE_SHAPE does not say what {engine}'s phases contain, so "
            "the e2e lines cannot be read against each other",
        )
    check(
        scenarios.CATBOOST_DETERMINISM.get("status") == "seeded, not guaranteed",
        "the CatBoost arm claims a determinism status other than 'seeded, "
        "not guaranteed'. CatBoost has no deterministic flag and a fixed "
        "thread count plus a fixed seed is not a guarantee",
    )
    for name in scenarios.CATBOOST_SCENARIO_COST:
        check(
            name in scenarios.SCENARIOS,
            f"CATBOOST_SCENARIO_COST warns about unknown scenario {name!r}",
        )
        check(
            scenarios.CATBOOST_SCENARIO_SUPPORT.get(name) is None,
            f"CATBOOST_SCENARIO_COST warns about {name!r}, which the "
            "CatBoost arm does not run, so the warning is about a cell that "
            "cannot be scheduled",
        )
    check(
        "cost_warnings" in arm,
        "the CatBoost arm block does not carry its cost warnings, so a "
        "matrix can be scheduled without them",
    )
    # 5. The categorical path, and the digest promise it rests on.
    #
    # The failure this guards against is the one the lane that built it was
    # warned about: a categorical CatBoost row that LOOKS rigorous because
    # the record carries a digest, on a re-encoding nobody proved was the
    # same data. Three things have to hold and none of them is checkable by
    # reading the record afterwards, so they are checked here.
    import numpy as np

    import measure
    import worker

    check(
        engines.CATBOOST_CATEGORICAL_FORM
        == scenarios.CATBOOST_CATEGORICAL_ENCODING["form"],
        "the encoder's form name and the scenario table's disagree, so a "
        "record says it was encoded one way and the argument for why that "
        "is still one dataset is written about another",
    )
    check(
        worker.CANONICAL_ENCODING != engines.CATBOOST_CATEGORICAL_FORM,
        "the re-encoded form is named the same as the canonical one, so a "
        "reader of a record cannot tell which container an arm got",
    )
    check(
        "categorical_encoding" in arm,
        "the CatBoost arm block does not carry its categorical encoding "
        "contract, so a published table can print a CatBoost categorical "
        "number without the one field that says why it is comparable",
    )
    # The encoder is a bijection on a column that looks like every
    # categorical column this harness produces, and REFUSES one that holds a
    # missing value. The second half is the load-bearing one: it is what
    # keeps `categorical_missing` out even if somebody flips its support
    # entry, and it is checked by calling the encoder rather than by reading
    # CATBOOST_SCENARIO_SUPPORT, because the support table is the thing that
    # could be wrong.
    _clean = {
        "X": np.ascontiguousarray(
            np.column_stack([
                np.array([0.5, -1.25, 3.0, 7.75]),
                np.array([0.0, 3.0, 1.0, 2.0]),
                np.array([250000.0, 1.0, 0.0, 199999.0]),
            ])
        ),
        "y": np.array([0.0, 1.0, 0.0, 1.0]),
    }
    try:
        frame, report = engines._catboost_categorical_frame(_clean, [1, 2])
    except Exception as exc:  # pragma: no cover - environment-dependent
        frame, report = None, None
        check(
            False,
            "the CatBoost categorical encoder refused a clean integer-coded "
            f"column: {type(exc).__name__}: {exc}",
        )
    if report is not None:
        check(
            report["canonical_digest_recomputed"]
            == measure.canonical_digest(_clean["X"], _clean["y"]),
            "the CatBoost encoder's reconstruction does not hash to the "
            "canonical matrix on a four-row case, so the per-run digest "
            "proof would pass nothing or fail everything",
        )
        check(
            [str(frame.dtypes.iloc[i]) for i in range(3)]
            == ["float64", "int64", "int64"],
            "the CatBoost encoder did not produce a mixed frame; CatBoost "
            "refuses cat_features on an all-float container",
        )
    _missing = {
        "X": np.ascontiguousarray(
            np.column_stack([
                np.array([0.5, -1.25, 3.0, 7.75]),
                np.array([0.0, np.nan, 1.0, 2.0]),
            ])
        ),
        "y": np.array([0.0, 1.0, 0.0, 1.0]),
    }
    try:
        engines._catboost_categorical_frame(_missing, [1])
    except engines.EngineError:
        pass
    except Exception as exc:  # pragma: no cover - environment-dependent
        FAILURES.append(
            "the CatBoost categorical encoder failed on a missing category "
            f"with the wrong error type: {type(exc).__name__}: {exc}"
        )
    else:
        FAILURES.append(
            "the CatBoost categorical encoder ACCEPTED a categorical column "
            "holding a missing value. CatBoost has no representation for a "
            "missing category, so whatever it built is a level the harness "
            "invented, and on categorical_missing that level is correlated "
            "with the target and unavailable to the other two arms. See "
            "CATBOOST_CATEGORICAL_ENCODING['missing_categories']"
        )
    check(
        scenarios.CATBOOST_SCENARIO_SUPPORT["categorical_missing"] is not None,
        "categorical_missing runs the CatBoost arm. It must not: its "
        "generator drops values inside categorical columns as a function of "
        "the target, and any level the harness invents for absence is a "
        "target-correlated feature only one of the three arms can use",
    )
    check(
        "encode" in scenarios.PHASE_SHAPE["catboost"],
        "PHASE_SHAPE does not describe the CatBoost encode phase, so an "
        "e2e line that contains it cannot be read against the other two",
    )
    for _knob in ("max_ctr_complexity", "one_hot_max_size", "has_time"):
        check(
            _knob in scenarios.CATBOOST_REFUSED_PARAMS,
            f"{_knob} is not refused. The arm now has a categorical path, "
            "and every one of these turns 'CatBoost at its own defaults' "
            "into something else while leaving the column heading alone",
        )

    check(
        set(scenarios.CATBOOST_SCENARIO_SUPPORT) == set(scenarios.SCENARIOS),
        "CATBOOST_SCENARIO_SUPPORT and SCENARIOS disagree: "
        f"{sorted(set(scenarios.CATBOOST_SCENARIO_SUPPORT) ^ set(scenarios.SCENARIOS))} "
        "has no support decision, so nobody decided whether the arm runs it",
    )


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


def check_categorical_sources():
    """`scenarios.CATEGORICAL_DATA_SOURCES` must match the sources themselves.

    That table decides whether a cell may be SCHEDULED on the accelerator
    (`run._engine_skip_reason` reads it through
    `scenarios.scenario_has_categorical`), and it is a hand-written list of
    names. A generator or loader that starts returning a `categorical_feature`
    key without being added to it would silently put a categorical column in
    front of a device that refuses one, and the failure would arrive as a
    raising cell inside a measurement window rather than as a table nobody
    updated.

    So it is re-derived here by introspecting the two modules for the key,
    which is the same thing the table's own comment says it was read from. The
    check is on the source text rather than on a call, because
    `check_generators_are_pure` establishes that nothing in this file may call
    a generator and a loader would download.
    """
    import inspect

    import generators
    import loaders
    import scenarios

    for module, kind, prefix in (
        (generators, "generators", ""),
        (loaders, "datasets", "load_"),
    ):
        found = set()
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if fn.__module__ != module.__name__ or not name.startswith(prefix):
                continue
            if "categorical_feature" in inspect.getsource(fn):
                found.add(name[len(prefix):])
        declared = set(scenarios.CATEGORICAL_DATA_SOURCES[kind])
        check(
            found == declared,
            f"CATEGORICAL_DATA_SOURCES[{kind!r}] declares {sorted(declared)} "
            f"and {module.__name__} returns a categorical_feature key from "
            f"{sorted(found)}. Undeclared: {sorted(found - declared)}; "
            f"declared but not found: {sorted(declared - found)}. This table "
            "decides which cells may be scheduled on the accelerator, so a "
            "source missing from it schedules a categorical column onto a "
            "device that refuses one",
        )


def check_pending_scenarios():
    """A scenario that is specified and not built must stay both.

    `scenarios.TEXT_SCENARIO_PENDING` is a written design for a shape whose
    input contract does not exist: the harness hands every engine one
    float64 matrix and a text column is not one, nobody has decided who
    tokenizes, and `worker.py`'s data digest cannot hash a numpy object
    array of strings. The design is in the tree so the lane that lands the
    contract has a target rather than a blank page.

    Two ways that goes wrong and both are checked here rather than left to
    review. It gets merged into `SCENARIOS` while the blockers are open, and
    then either fails `check_registry` on a missing generator, or -- if
    somebody adds a stub generator to make the self-check green -- runs and
    measures something nobody specified. Or it quietly loses the fields that
    say it is pending, and a reader six weeks from now cannot tell a design
    from a scenario. So: it must not be registered, it must name what it
    waits on, and it must not name a generator that exists, because the day
    one does exist is the day this dict is supposed to become a real entry
    through a commit that also writes its thresholds and its support
    decisions.
    """
    import generators
    import scenarios

    pending = scenarios.TEXT_SCENARIO_PENDING
    scenario_id = pending["spec"].get("id", pending["id"])
    check(
        pending["id"] not in scenarios.SCENARIOS
        and scenario_id not in scenarios.SCENARIOS,
        f"{pending['id']} is registered in SCENARIOS while it is still "
        f"marked {pending['status']!r}. Registering it is a commit that also "
        "writes its generator, its thresholds.json entry and its per-engine "
        "support decision; it is not a one-line move",
    )
    check(
        bool(pending.get("blocked_on")),
        f"{pending['id']} is pending and does not say what it waits on",
    )
    check(
        pending["spec"].get("generator") not in generators.GENERATORS,
        f"{pending['id']} names a generator that now EXISTS. That is good "
        "news and it is not a self-check failure to leave alone: the "
        "scenario is ready to be registered, with a thresholds.json entry "
        "and a support decision per engine, and this check is the reminder",
    )
    check(
        pending["id"] not in json.load(
            open(os.path.join(HERE, "thresholds.json"))
        )["scenarios"],
        f"{pending['id']} has a thresholds.json entry and is not a scenario. "
        "A gate written before anybody knows what the implementations "
        "differ by is a number chosen to pass",
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


#: The mojotrees arms whose parameters are a static override dict, and the
#: dict. Walked by the COVERAGE assertions in `check_correctness_arms`, which
#: are the ones that survive a rename.
#:
#: The plain arm is `{}` on purpose and is not a placeholder. It is what the
#: harness runs by default, and the whole finding of 2026-08-17 is that every
#: arm here resolved `feature_fraction` to 1.0 and `score_function` to L2, so
#: an empty dict is a real row of this table and the reason the coverage check
#: needed writing.
#:
#: `mojotrees_catboost_mode` carries a `learning_rate` that is not in its dict
#: -- it comes from CatBoost's read-back per cell -- and that is fine here,
#: because nothing below reads a rate. A key added to this table's readers
#: that CAN move with the cell must not be read off these dicts.
def _mojotrees_override_dicts():
    import scenarios

    return {
        "mojotrees": {},
        "mojotrees_depthwise": scenarios.MOJOTREES_DEPTHWISE,
        "mojotrees_catboost_mode": scenarios.MOJOTREES_CATBOOST_MODE,
        "mojotrees_symmetric_colsample": scenarios.MOJOTREES_SYMMETRIC_COLSAMPLE,
        "mojotrees_cosine_leafwise": scenarios.MOJOTREES_COSINE_LEAFWISE,
    }


def _resolved_grow_policy(override):
    """The growth order an arm carrying `override` actually fits under.

    An ABSENT key is `lossguide`, because that is the estimator's default
    (`python/mojotrees/sklearn.py::_Base.__init__`) and `scenarios.BASE_PARAMS` does not
    set one. Reading an absent key as "unknown" would let an arm count as
    leaf-wise coverage without saying so, which is the exact shape of the gap
    this whole function exists about: `mojotrees_catboost_mode` DOES set
    `score_function=Cosine` and covers nothing leaf-wise, because it also
    names a symmetric grower.
    """
    value = str(override.get("grow_policy", "lossguide")).strip().lower()
    if value in ("leafwise", "lossguide"):
        return "lossguide"
    if value in ("oblivious", "symmetric", "symmetrictree"):
        return "symmetrictree"
    return value


def check_correctness_arms():
    """The two CORRECTNESS arms, and the coverage they exist to hold.

    **WHAT THIS FUNCTION IS ABOUT, because it is not the usual wiring check.**
    On 2026-08-17 two live wrong answers were found in the trainer and neither
    was found here. The oblivious GPU path returned a wrong answer whenever
    `feature_fraction < 1`, and `score_function='Cosine'` was silently ignored
    on every leaf-wise GPU fit. Both were found by reading code. The suite was
    fully green throughout, and it was green for a reason that no threshold
    and no tolerance would have changed: no arm in this harness set either
    parameter to a non-default value, so no cell ever executed the broken
    code. A gate cannot fail on a configuration nobody runs.

    So the failure mode this guards is an arm DISAPPEARING, which restores
    that state exactly and would do it silently. The checks come in three
    kinds and the third is the one that matters most.

    WIRING. Each arm is registered, goes through its own translator, is a
    subject rather than a peer, and is inside `verify.SUBJECT_ENGINES`. That
    last one is not bookkeeping: `check_device_agreement` skips any engine not
    in that tuple, and it is the only thing either arm produces.

    SCHEDULABILITY. A matrix asking for these arms on both backends really
    does produce a cpu cell and a gpu cell that `check_device_agreement` will
    key together. Built through `run.build_matrix`, which runs no cells and
    touches no data. An arm that is registered and then skipped by a rule
    somewhere is worth nothing, and that is not visible from the registry.

    COVERAGE, and this is the kind that survives a rename. Rather than
    asserting that two named engines exist, it asserts that SOME registered
    mojotrees arm fits with `feature_fraction < 1` under a symmetric grower,
    and that SOME registered arm scores with Cosine under leaf-wise growth.
    Those two sentences are the coverage gap of 2026-08-17 stated as
    properties. An arm renamed still satisfies them; both arms deleted does
    not, whatever it is replaced with.
    """
    import engines
    import run
    import scenarios
    import verify

    overrides = _mojotrees_override_dicts()

    # --- 1. Wiring, per arm. ------------------------------------------------
    per_arm = {
        "mojotrees_symmetric_colsample": (
            scenarios.mojotrees_symmetric_colsample_params,
            scenarios.MOJOTREES_SYMMETRIC_COLSAMPLE_SCENARIO_SUPPORT,
            scenarios.MOJOTREES_SYMMETRIC_COLSAMPLE_CLAIMS,
        ),
        "mojotrees_cosine_leafwise": (
            scenarios.mojotrees_cosine_leafwise_params,
            scenarios.MOJOTREES_COSINE_LEAFWISE_SCENARIO_SUPPORT,
            scenarios.MOJOTREES_COSINE_LEAFWISE_CLAIMS,
        ),
    }
    check(
        set(scenarios.CORRECTNESS_ARMS) == set(per_arm),
        "scenarios.CORRECTNESS_ARMS and this check's table disagree about "
        "which arms are correctness arms, so one of them is walking a set the "
        "other does not know about",
    )
    for name in scenarios.CORRECTNESS_ARMS:
        if name not in per_arm:
            continue
        translator, support, claims = per_arm[name]
        if not check(
            name in engines.ENGINES,
            f"{name} is a correctness arm and is not in engines.ENGINES, so "
            "worker.py cannot build it by name and the configuration it "
            "covers is unreachable again. Both configurations these arms "
            "cover had a live wrong answer in them on 2026-08-17 and neither "
            "was catchable from this harness",
        ):
            continue
        check(
            engines.ENGINES[name].params_fn is translator,
            f"{name} does not go through its own translator, so it is the "
            "plain arm under another name and its record would claim to "
            "exercise a configuration it does not",
        )
        check(
            engines.ENGINE_ARM.get(name) == "subject_variant",
            f"{name} is not recorded as subject_variant. It is our own "
            "engine, so it is not a peer, and it is not the headline row "
            "either",
        )
        check(
            name not in scenarios.PEER_ENGINES,
            f"{name} is listed as a peer, and it is our own engine rather "
            "than a competitor",
        )
        check(
            name in verify.SUBJECT_ENGINES,
            f"{name} is not in verify.SUBJECT_ENGINES. For a correctness arm "
            "that is not a missing label, it is the whole arm: "
            "check_device_agreement skips any engine not in that tuple, and "
            "the cpu-versus-gpu comparison is the only thing this arm "
            "produces. Its cells would still run, still be timed and still be "
            "published, and nothing would compare them",
        )
        check(
            set(support) == set(scenarios.SCENARIOS),
            f"{name}'s scenario support table does not have one entry per "
            "scenario, so a scenario was added without a support decision "
            "for this arm",
        )
        runs = sorted(k for k, v in support.items() if v is None)
        check(
            runs,
            f"{name} runs no scenario at all, so it is registered and inert. "
            "An arm that schedules nothing is the coverage gap it was built "
            "to close, wearing a name",
        )
        for scenario_id in runs:
            check(
                "gpu" in scenarios.SCENARIOS[scenario_id]["devices"],
                f"{name} runs {scenario_id}, which declares no gpu support. "
                "A correctness arm's product is check_device_agreement, "
                "which needs a cpu row AND a gpu row of the same arm, so a "
                "cpu-only scenario gives this arm a number nobody asked for "
                "and no verdict",
            )
            check(
                name in scenarios.SCENARIOS[scenario_id]["engines"],
                f"{name}'s support table says it runs {scenario_id} and the "
                "scenario does not list it, so the registration loop did not "
                "run or was undone by a later edit",
            )
        check(
            claims and "correctness" in claims.lower(),
            f"{name}'s claims string does not say it is a correctness arm. "
            "The string travels into every record, and a reader who takes "
            "one of these rows for a comparison row will read its accuracy "
            "against a comparator it was never meant to be read against",
        )
        # A TRAINING override must not carry a BINNING key. `max_bin` reached
        # mojotrees.train once through MOJOTREES_CATBOOST_MODE and killed a
        # whole smoke pass with "'max_bin' describes the data, not the
        # training run", which under an infrastructure failure withholds the
        # quality verdict for every cell that did run.
        for key in scenarios.dataset_params(
            scenarios.resolve("dense_regression", "smoke", "synthetic")
        ):
            check(
                key not in overrides[name],
                f"{name} carries the binning parameter {key} in a TRAINING "
                "override. It would be handed to mojotrees.train, which "
                "refuses it by name",
            )

    # --- 2. The parameter values, checked against what makes them bite. -----
    symmetric = scenarios.MOJOTREES_SYMMETRIC_COLSAMPLE
    check(
        _resolved_grow_policy(symmetric) == "symmetrictree",
        "the symmetric-colsample arm does not grow symmetric trees. The bug "
        "it covers was in the OBLIVIOUS level build "
        "(histogram_gpu.stage_desc_level_plan), which no other growth order "
        "reaches",
    )
    check(
        0.0 < float(symmetric.get("feature_fraction", 1.0)) < 1.0,
        "the symmetric-colsample arm's feature_fraction is not below 1.0, so "
        "it covers nothing. sampling.select_tree_features returns the "
        "IDENTITY at fraction >= 1.0, which is exactly the case where the "
        "broken feature table read correctly and nothing showed",
    )
    check(
        float(symmetric.get("feature_fraction"))
        == float(scenarios.SYMMETRIC_COLSAMPLE_FRACTION_CHOICE),
        "the symmetric-colsample arm's feature_fraction and the constant "
        "that documents the choice have come apart, so the number in the "
        "record is not the number the argument is about",
    )
    check(
        int(symmetric.get("max_depth", -1)) > 0,
        "the symmetric-colsample arm does not set a positive max_depth. "
        "tree.mojo::_check_oblivious raises 'grow_policy=oblivious requires "
        "max_depth > 0' "
        "and BASE_PARAMS carries -1, so this arm would not run at all",
    )
    check(
        int(symmetric.get("num_leaves", 0))
        == 2 ** int(symmetric.get("max_depth", 0)),
        "the symmetric-colsample arm's num_leaves is not 2**max_depth, which "
        "is what a symmetric tree at that depth holds. A smaller bound would "
        "truncate a level and make the arm a different tree from the one its "
        "depth states",
    )
    for refused in (
        "feature_fraction_bynode", "feature_fraction_bylevel",
        "random_strength", "score_function",
    ):
        check(
            refused not in symmetric,
            f"the symmetric-colsample arm carries {refused}. The first two "
            "are refused by oblivious_device_supported and are NOT fields of "
            "a DeviceRequest, so no device_policy block can decline them and "
            "train_gpu raises instead, which is an infrastructure failure. "
            "The other two are the refusals that keep "
            "mojotrees_catboost_mode off four scenarios, and carrying one "
            "would mean a red cell could be several things",
        )

    cosine = scenarios.MOJOTREES_COSINE_LEAFWISE
    check(
        str(cosine.get("score_function", "l2")).strip().lower() == "cosine",
        "the Cosine arm does not set score_function=cosine, so it covers "
        "nothing",
    )
    check(
        _resolved_grow_policy(cosine) == "lossguide",
        "the Cosine arm does not grow leaf-wise. The bug it covers was "
        "specific to the LEAF-WISE device path: the one arm that already set "
        "Cosine, mojotrees_catboost_mode, grows symmetric trees and never "
        "reached it",
    )
    check(
        "grow_policy" in cosine,
        "the Cosine arm does not NAME its growth order and is relying on the "
        "estimator's default. The arm's whole value is which code path it "
        "reaches, so a moved default would silently retire it. The shipped "
        "defaults have already moved in this direction once",
    )
    check(
        float(cosine.get("lambda_l2", 0.0)) > 0.0,
        "the Cosine arm runs at lambda_l2=0, where the Cosine score "
        "degenerates to sqrt of the L2 score and cannot move the argmax "
        "within a node (sklearn.py, the score_function docstring). It would "
        "still detect the bug through frontier ORDER alone, which with "
        "num_leaves a hard bound may end at the same leaf set; that is a "
        "much weaker detector than the arm claims to be",
    )
    check(
        "random_strength" not in cosine,
        "the Cosine arm carries random_strength, which "
        "run.DEVICE_PARAMETER_DIVERGENCE declares a gpu skip for: the gpu "
        "resolves it to 0.0. The gpu cell would be skipped or would be a "
        "different regularizer under the cpu cell's name, and a "
        "device_agreement pair whose halves are not the same arm is worse "
        "than no pair",
    )

    # --- 3. Coverage, stated as properties rather than as names. ------------
    #
    # This is the pair of assertions that would have failed on 2026-08-16, the
    # day before the bugs were found, with every other check in this file
    # green. They are written over the override dicts rather than over the
    # engine names so that renaming an arm keeps them satisfied and deleting
    # the coverage does not.
    colsample_on_symmetric = sorted(
        name for name, override in overrides.items()
        if _resolved_grow_policy(override) == "symmetrictree"
        and 0.0 < float(override.get("feature_fraction", 1.0)) < 1.0
    )
    check(
        colsample_on_symmetric,
        "NO ARM in this harness fits with feature_fraction < 1 under a "
        "symmetric grower. That was the state on 2026-08-16, and on "
        "2026-08-17 the oblivious GPU path was found returning a wrong "
        "answer in exactly that configuration: GpuLeafBatcher.feat_dev was "
        "never set by the level build, so the histogram wrote feature slice "
        "`slot` while the split searcher read slice `active[slot]`. It was "
        "found by reading code, because no gate here could fail on a "
        "configuration no arm ran. Restore an arm that sets it, on a "
        "scenario that declares gpu support, and put it in "
        "verify.SUBJECT_ENGINES",
    )
    cosine_on_leafwise = sorted(
        name for name, override in overrides.items()
        if _resolved_grow_policy(override) == "lossguide"
        and str(override.get("score_function", "l2")).strip().lower() != "l2"
    )
    check(
        cosine_on_leafwise,
        "NO ARM in this harness scores splits with anything but L2 under "
        "leaf-wise growth. That was the state on 2026-08-16, and on "
        "2026-08-17 score_function='Cosine' was found to be silently ignored "
        "on every leaf-wise GPU fit, because "
        "GpuSplitSearcher.set_score_function had no production caller. Note "
        "that mojotrees_catboost_mode DOES set Cosine and does not satisfy "
        "this: it grows symmetric trees, so it never reaches the path the "
        "bug lived on. A parameter being covered by an arm is not the same "
        "as the code path being covered",
    )

    # --- 4. The record schema admits the names. -----------------------------
    #
    # A general check and not a two-name one. This enum was TWO NAMES SHORT
    # until 2026-08-17, with `mojotrees_depthwise` and `xgboost` both writing
    # records under it, and nothing here noticed; it is written as an equality
    # against the registry so that the next arm cannot repeat it.
    schema_path = os.path.join(HERE, "schema.json")
    with open(schema_path) as handle:
        schema = json.load(handle)
    declared = set(schema["properties"]["engine"]["enum"])
    check(
        declared == set(engines.ENGINES),
        "schema.json's engine enum and engines.ENGINES disagree: the schema "
        f"is missing {sorted(set(engines.ENGINES) - declared)} and carries "
        f"{sorted(declared - set(engines.ENGINES))} that no engine produces. "
        "A record written under a name the schema does not admit is a record "
        "that validates nowhere",
    )

    # --- 5. Every subject arm gets the subject gates. -----------------------
    #
    # The general form of the per-arm SUBJECT_ENGINES check above, and the
    # reason it is general: `verify.SUBJECT_ENGINES`'s own comment records
    # that three checks were written as a literal `!= "mojotrees"` and each
    # silently stopped covering a new arm the day one was added. A tuple
    # maintained by hand beside a mapping that already holds the answer is the
    # same defect one level up.
    for name, role in sorted(engines.ENGINE_ARM.items()):
        if not role.startswith("subject"):
            continue
        check(
            name in verify.SUBJECT_ENGINES,
            f"{name} is recorded as {role} in engines.ENGINE_ARM and is not "
            "in verify.SUBJECT_ENGINES, so its accelerator rows would be "
            "published with no backend proof and no cpu-versus-gpu agreement "
            "check. Both of those have caught a real defect in this campaign",
        )

    # --- 6. Schedulability, on a matrix rather than on the registry. --------
    parser = run.build_parser()
    argv = ["--tier", "smoke", "--scenario", "dense_regression",
            "--device", "cpu", "--device", "gpu", "--threads", "4",
            "--repeats", "3"]
    for name in scenarios.CORRECTNESS_ARMS:
        argv += ["--engine", name]
    args = parser.parse_args(argv)
    args.scenario = args.scenario or []
    args.engine = args.engine or []
    args.device = args.device or ["cpu"]
    args.threads = args.threads or [1]
    matrix = run.build_matrix(args)
    scheduled = [job for job in matrix if "skip" not in job]
    for name in scenarios.CORRECTNESS_ARMS:
        for device in ("cpu", "gpu"):
            cells = [
                job for job in scheduled
                if job["engine"] == name and job["device"] == device
            ]
            check(
                cells,
                f"{name} on the {device} is not SCHEDULED by a matrix that "
                "asks for it on both backends, so check_device_agreement "
                f"has no {device} half to compare. The skip reasons in this "
                "matrix are: "
                + "; ".join(
                    sorted(
                        {
                            job["skip"] for job in matrix
                            if "skip" in job and job["engine"] == name
                        }
                    )
                ),
            )
        # The oracle asymmetry, on these arms specifically. It is what makes
        # the pair affordable: the gpu cell keeps --repeats and the cpu twin
        # runs once, because device_agreement needs one cpu prediction per
        # cell and no more.
        cpu_cells = [
            job for job in scheduled
            if job["engine"] == name and job["device"] == "cpu"
        ]
        check(
            all(job.get("cell_role") == "oracle" for job in cpu_cells),
            f"{name}'s cpu cell is not marked as an oracle beside its "
            "accelerator cell, so it runs --repeats times for a purpose that "
            "needs one prediction",
        )

    # --- 7. The tier cap, which is the only thing bounding the cell count. --
    ok_standard, _ = scenarios.correctness_arm_tier_ok("standard")
    ok_large, reason_large = scenarios.correctness_arm_tier_ok("large")
    check(
        ok_standard,
        "a correctness arm is capped below the standard tier. The standard "
        "tier is the one the headline table reads and these arms have to be "
        "runnable there",
    )
    check(
        not ok_large and reason_large,
        "a correctness arm is not capped at the large tier, so a --tier large "
        "run that names one pays for a row-level prediction comparison at "
        "five times the rows, which is not more true for it. The cap is a "
        "declared COST bound and its reason has to travel with the skip",
    )


def check_oracle_cells():
    """The ORACLE CELL rule, checked on a matrix rather than on a run.

    An oracle cell is one of our own arms on the cpu in a run that also
    schedules that arm on an accelerator. `run.py` reduces its repeats and
    `report.py` keeps it out of the speed story, which means two correctness
    gates in `verify.py` now depend on a SCHEDULING decision:
    `check_device_agreement` compares every accelerator row against its own cpu
    twin and `check_backend_proof` reads that twin's prediction digest. Neither
    can run without the twin, and neither FAILS when the twin is missing, so a
    bug that dropped the cpu cell entirely would take both gates offline and
    turn the verdict greener rather than redder. That is the class of defect
    this function exists to catch, and it is why the first assertion below is
    about the cell EXISTING rather than about its label.

    Everything here is built through `run.build_matrix`, which runs no cells
    and touches no data.
    """
    import run

    parser = run.build_parser()

    # The default is read off the parser rather than restated, so a default
    # changed in one place cannot pass here.
    check(
        parser.get_default("oracle_repeats") == 1,
        "run.py --oracle-repeats must default to 1: one cpu prediction per "
        "cell is exactly what verify.py's device_agreement and backend_proof "
        "each need, and it is the whole saving at the large tier. Raising it "
        "is a decision with a cost; lowering it disarms two gates",
    )

    base = [
        "--tier", "smoke", "--scenario", "dense_regression",
        "--engine", "mojotrees", "--engine", "lightgbm",
        "--threads", "4", "--repeats", "3",
    ]
    args = parser.parse_args(base + ["--device", "cpu", "--device", "gpu"])
    args.scenario = args.scenario or []
    args.engine = args.engine or []
    args.device = args.device or ["cpu"]
    args.threads = args.threads or [1]
    jobs = [job for job in run.build_matrix(args) if "skip" not in job]

    def cells(matrix, engine, device):
        return [
            job for job in matrix
            if job["engine"] == engine and job["device"] == device
        ]

    oracle = cells(jobs, "mojotrees", "cpu")
    accelerator = cells(jobs, "mojotrees", "gpu")

    # FIRST, and it is first on purpose. The reduction must never become a
    # removal. The cell has to be scheduled or the two gates above have nothing.
    check(
        len(oracle) >= 1,
        "the cpu cell of a subject arm must still be SCHEDULED when the same "
        "arm runs on the accelerator. verify.py's device_agreement and "
        "backend_proof both compare an accelerator row against its own cpu "
        "twin, and both go silent rather than red without one",
    )
    check(
        len(oracle) == args.oracle_repeats,
        f"a cpu subject cell beside an accelerator cell should run "
        f"{args.oracle_repeats} repeat(s), got {len(oracle)}",
    )
    check(
        len(accelerator) == args.repeats,
        f"the accelerator cell is a MEASURED cell and keeps --repeats "
        f"({args.repeats}), got {len(accelerator)}",
    )
    check(
        all(job.get("cell_role") == "oracle" for job in oracle),
        "every cpu subject job beside an accelerator job must carry "
        "cell_role=oracle, which is what carries the decision into the record "
        "and out of whichever tool renders the table",
    )
    check(
        all(job.get("cell_role_note") for job in oracle),
        "an oracle job must carry cell_role_note, so that a records file read "
        "on its own says why one cpu row sits beside three gpu rows",
    )
    check(
        all(job.get("cell_role") == "measured" for job in accelerator),
        "an accelerator job is a measured cell, not an oracle",
    )
    check(
        all(job.get("cell_role") == "measured"
            for job in cells(jobs, "lightgbm", "cpu")),
        "the comparator on the cpu is NOT an oracle. The cpu is LightGBM's "
        "best available backend on this machine, so its row is a competitor "
        "row and stays in the ranking",
    )
    check(
        len(cells(jobs, "lightgbm", "cpu")) == args.repeats,
        "the comparator's repeats must not be touched by the oracle pass",
    )
    check(
        sorted(job["job_index"] for job in run.build_matrix(args))
        == list(range(len(run.build_matrix(args)))),
        "job_index must be a gapless range after the oracle pass. A gap reads "
        "as a cell that ran and failed to write a record",
    )

    # No accelerator in the run, so the cpu cell is the MEASUREMENT and
    # nothing about it moves. This is the case that must stay exactly as it
    # was, because reducing it would reduce the only number there is.
    plain = parser.parse_args(base + ["--device", "cpu"])
    plain.scenario = plain.scenario or []
    plain_jobs = [job for job in run.build_matrix(plain) if "skip" not in job]
    check(
        len(cells(plain_jobs, "mojotrees", "cpu")) == plain.repeats,
        "with no accelerator cell in the run, a cpu subject cell is the "
        "measurement rather than an oracle and keeps --repeats",
    )
    check(
        not any(job.get("cell_role") == "oracle" for job in plain_jobs),
        "nothing in a cpu-only run is an oracle",
    )

    # A scenario that declares no accelerator support, inside a run that asked
    # for both devices. Its cpu cell must be untouched even though sibling
    # scenarios in the same run have oracles.
    mixed = parser.parse_args(
        base + ["--scenario", "ranking", "--device", "cpu", "--device", "gpu"]
    )
    mixed_jobs = [job for job in run.build_matrix(mixed) if "skip" not in job]
    ranking_cpu = [
        job for job in mixed_jobs
        if job["scenario"] == "ranking" and job["engine"] == "mojotrees"
    ]
    check(
        len(ranking_cpu) == mixed.repeats
        and all(job.get("cell_role") == "measured" for job in ranking_cpu),
        "a scenario with no accelerator support keeps full repeats and is not "
        "an oracle, even in a run where another scenario has oracle cells",
    )

    # The runner and the gate must agree about what an oracle is. `run.py`
    # decides at matrix time off the job's variant and `verify.py` reads it
    # back off a record's resolved data kind, so this walks the job through the
    # shape a record has.
    import verify

    as_records = [
        {
            "status": "ok",
            "scenario": job["scenario"],
            "tier": job["tier"],
            "arm": job.get("arm") or job["engine"],
            "engine": job["engine"],
            "threads": job["threads"],
            "device_used": job["device"],
            "data": {"data_kind": "synthetic"},
        }
        for job in jobs
    ]
    keys = verify.accelerator_cells(as_records)
    computed = {
        (record["engine"], record["device_used"])
        for record in as_records
        if verify.is_oracle(record, keys)
    }
    check(
        computed == {("mojotrees", "cpu")},
        "verify.is_oracle computed from records must find the same cells "
        "run.py labelled, or a results file written before the cell_role "
        "field existed reads under different labels from one written after. "
        f"Found {sorted(computed)}",
    )
    check(
        verify.is_oracle({"cell_role": "measured", "engine": "mojotrees",
                          "device_used": "cpu"}, keys) is False,
        "cell_role on the record must WIN over the computed fallback. The "
        "runner knew the whole matrix and is_oracle sees one row",
    )

    # The headline label names all three reasons, because it is the one
    # sentence in the harness a reader could take as a cheat. A future edit
    # that trims it to "our GPU against their CPU" removes the argument and
    # leaves the claim, which is the worst of the available outcomes.
    import report

    for needle, why in (
        ("LightGBM runs on the cpu here", "the LightGBM refusal"),
        ("border_count", "the CatBoost quantization refusal"),
        ("CUDA", "the XGBoost no-such-backend refusal"),
        ("Apple silicon", "the machine this refusal is about"),
        ("BEST AVAILABLE BACKEND", "what the comparison actually is"),
    ):
        check(
            needle in report.HEADLINE_LABEL,
            f"report.HEADLINE_LABEL must state {why}: the phrase {needle!r} "
            "is missing. The headline pairs our accelerator against three "
            "competitors' cpu, and that is only honest while the label says "
            "why none of them can use this GPU",
        )


def check_accuracy_axes():
    """The two accuracy axes, and the rule that neither may suppress the other.

    Registered 2026-08-17 with the separation. Two things are asserted here and
    they fail in opposite directions, which is why both are needed.

    THE SUPPRESSION MUST NOT COME BACK. The frontier must rank an arm on speed
    whatever its accuracy says. The defect this replaced was silent: a real
    1.24x improvement simply did not appear in a table, and nothing anywhere
    reported that a row had been withheld. A test that only checked "the
    fastest arm is named" would have passed on the broken version too, because
    the broken version did name one. So this builds a fixture where the FASTEST
    arm is also the LEAST accurate and asserts it is ranked first.

    THE ACCURACY MUST NOT GO MISSING EITHER. Ranking everything on seconds is
    only honest while every row carries its metric, so the same fixture asserts
    the metric value is in the row and the caveat sentence is above the table.
    Those two assertions are the trade this design made, and dropping either
    half turns a fix into a different defect.
    """
    import report
    import verify

    check(
        verify.NOTE in verify.STATUSES and verify.NOTE != verify.WARN,
        "verify.NOTE must exist as a status distinct from WARN. It is what "
        "lets the peer accuracy comparison be reported on every run without "
        "being coloured as a problem with the run",
    )
    verdict = verify.Verdict()
    verdict.add(verify.NOTE, "x", "y", "z")
    check(
        verdict.counts()[verify.NOTE] == 1,
        "Verdict.counts does not count a note, so the summary line would "
        "under-report the run",
    )
    check(
        not verdict.failed,
        "a note must never reach Verdict.failed. The exit code is FAIL and "
        "nothing else, and the accuracy scoreboard must not be able to move it",
    )
    try:
        verdict.add("bogus", "x", "y", "z")
        check(False, "Verdict.add accepted an unknown status")
    except ValueError:
        pass

    # The peer comparison gates nothing, and the way that is enforced is that
    # there is no key to turn it back on. A dormant `gating` flag contradicting
    # a written ruling is an invitation.
    check(
        "gating" not in verify.DEFAULT_ACCURACY_PEER,
        "verify.DEFAULT_ACCURACY_PEER has a `gating` key. The peer comparison "
        "is a scoreboard and gates nothing after Andrew's 2026-08-17 ruling; a "
        "dormant switch that contradicts a ruling gets flipped",
    )
    check(
        verify.DEFAULT_ACCURACY_ANCHOR.get("gating") is True,
        "the self-anchored accuracy check must gate. It is the ONLY accuracy "
        "gate in this harness now that the peer comparison does not gate, so "
        "turning it off leaves accuracy ungated entirely",
    )
    check(
        verify.DEFAULT_ACCURACY_ANCHOR["max_worse_relative"]
        != verify.DEFAULT_ACCURACY_PEER["max_worse_relative"],
        "the anchor tolerance equals the peer band. They are different "
        "quantities: one is how much of OUR OWN accuracy a single change may "
        "silently cost, the other is a standing distance to a competitor. "
        "Reusing the number is the mistake this separation exists to undo",
    )
    check(
        0 < verify.DEFAULT_ACCURACY_ANCHOR["max_worse_relative"] < 0.01,
        "the anchor tolerance is outside (0, 1 percent). Zero would fail on "
        "any floating-point reassociation and 1 percent would let one change "
        "give away the whole competitive gap",
    )
    # The anchors file must exist, must parse, and must not be written by any
    # code path. The second assertion is the ratchet defence: if a run could
    # write this file, the anchor would become "whatever ran last".
    anchors_path = verify.ACCURACY_ANCHORS_PATH
    check(
        os.path.exists(anchors_path),
        f"{anchors_path} is missing, so every arm reads as unanchored and the "
        "accuracy gate covers nothing",
    )
    source = open(os.path.join(HERE, "verify.py")).read()
    check(
        "ACCURACY_ANCHORS_PATH" in source
        and 'open(ACCURACY_ANCHORS_PATH, "w")' not in source
        and "open(path, \"w\")" in source.split("def propose_anchors")[1],
        "verify.py must never open the live anchors file for writing. The "
        "anchor is a fixed point only while no run can move it; --propose-"
        "anchors writes a candidate somewhere else and a person moves it",
    )

    # A fixture where the FASTEST arm is the LEAST accurate. Under the old rule
    # this row was printed as `OUTSIDE, not ranked` and its speed vanished.
    def row(arm, engine, seconds, rmse, device="gpu"):
        return {
            "status": "ok", "scenario": "dense_regression", "tier": "standard",
            "engine": engine, "arm": arm, "threads": 4,
            "device_used": device, "device_requested": device,
            "cell_role": "measured",
            "primary_metric": "rmse", "quality": {"rmse": rmse},
            "params": {"num_boost_round": 100},
            "data": {"data_kind": "synthetic", "dataset": "generated:x"},
            "phases": {"train": {"elapsed_s": seconds}},
            "environment": {"cpu": {"arch": "arm64", "model": "Test"}},
        }

    oracle = row("mojotrees", "mojotrees", 1.046, 0.40, device="cpu")
    oracle["cell_role"] = "oracle"
    records = [
        row("mojotrees", "mojotrees", 0.839, 0.40),
        row("mojotrees_catboost_mode", "mojotrees_catboost_mode", 2.238, 0.30),
        row("catboost", "catboost", 0.557, 0.31, device="cpu"),
        row("lightgbm", "lightgbm", 0.760, 0.32, device="cpu"),
        oracle,
    ]
    config = json.load(open(os.path.join(HERE, "thresholds.json")))
    lines = []
    report._frontier(records, config, lines.append)
    text = "\n".join(lines)

    fastest = [
        line for line in lines
        if line.startswith("| 1 |") and "mojotrees" in line
    ]
    check(
        bool(fastest),
        "the fastest arm was not ranked 1. It is the least accurate arm in "
        "this fixture, which is exactly the case the old rule dropped from the "
        "table: a 1.24x improvement went unpublished on 2026-08-17 because the "
        "arm was behind CatBoost. Speed is ranked for every arm, always",
    )
    if fastest:
        check(
            "0.4" in fastest[0],
            "the fastest row does not carry its own metric value. A rank "
            "without the accuracy beside it is the defect this design traded "
            f"for, not an improvement on it. Row: {fastest[0]}",
        )
    # TABLE ROWS ONLY. The section preamble quotes the old `OUTSIDE, not
    # ranked` label on purpose, to say what this table used to be, and a naive
    # substring search over the whole output fails on that sentence. What must
    # never come back is the label in a ROW.
    table_rows = [line for line in lines if line.startswith("|")]
    check(
        not any("not ranked" in line for line in table_rows),
        "a frontier table row says `not ranked` for an accuracy reason. An "
        "accuracy verdict may label a row and may never remove it from the "
        "speed ranking. (`oracle` in the rank column is the one exclusion and "
        "it is a backend decision, not an accuracy one, so it says `oracle`.)",
    )
    check(
        report.RANKING_CAVEAT in text,
        "the ranking caveat is missing from the frontier output. A ranking "
        "that spans arms of differing accuracy has to say so where it cannot "
        "be missed, because speed is purchasable with accuracy",
    )
    for needle, why in (
        ("purchasable with accuracy", "what the reader is being warned about"),
        ("fewer trees", "how it is purchased"),
    ):
        check(
            needle in report.RANKING_CAVEAT,
            f"report.RANKING_CAVEAT no longer states {why}: {needle!r} is "
            "missing. Trimmed to 'ranked on seconds' it becomes a caption "
            "instead of a warning",
        )
    check(
        "vs anchor" in text and "vs best peer" in text,
        "the frontier table lost one of its two accuracy columns. The gate and "
        "the scoreboard are different questions and printing one of them is "
        "how they got conflated in the first place",
    )
    # The oracle exclusion is the ONE row that leaves the speed ranking, and it
    # must still leave it, for a backend reason that has nothing to do with
    # accuracy. Its accuracy columns must still be filled in, which is the
    # separation applied in the other direction.
    oracle_rows = [line for line in table_rows if line.startswith("| oracle |")]
    check(
        bool(oracle_rows),
        "the oracle row lost its `oracle` rank label. It is out of the SPEED "
        "ranking because the GPU is the product and the cpu backend is not "
        "optimized, which survives the accuracy separation untouched",
    )
    if oracle_rows:
        check(
            "behind" in oracle_rows[0] or "ahead" in oracle_rows[0],
            "the oracle row has no peer accuracy figure. Being out of the "
            "speed story is not a reason to be out of the accuracy story: the "
            f"separation cuts both ways. Row: {oracle_rows[0]}",
        )
    check(
        "| catboost |" in text and "| 1 |" in text,
        "no competitor row is ranked. Competitors are ranked on SPEED from "
        "2026-08-17; being the accuracy bar is a statement about the accuracy "
        "axis and was being used to remove them from the speed axis",
    )
    check(
        "pareto" in text and "dominated by" in text,
        "the pareto column is gone. It is what lets 'fastest overall' and "
        "'fastest that nothing beats on both axes' both be answered, which a "
        "single ordering cannot do",
    )

    # The gate itself, on the same fixture plus one adopted anchor. A regression
    # must FAIL and it must do so without any peer in the verdict.
    anchor_key = verify._anchor_key(records[0])
    good = {anchor_key: {
        "primary_metric": "rmse", "value": 0.40,
        "environment": {"arch": "arm64", "cpu_model": "Test"},
        # CURRENT by construction: the one parameter this fixture's arm left
        # unset is recorded at whatever the package ships today, so
        # `verify.anchor_staleness` returns None and the anchor gates. Added
        # 2026-08-17 with the staleness mechanism; without this block the
        # anchor is ANCHOR CURRENCY UNKNOWN and cannot pass or fail anything,
        # which is the behavior `check_stale_anchors` asserts separately.
        "configuration": {
            "passed": {"lambda_l2": None, "learning_rate": None},
            "followed_default": ["lambda_l2", "learning_rate"],
            "shipped_at_record": {
                "lambda_l2": verify.shipped_constant("_LAMBDA_L2"),
                "learning_rate": verify.shipped_constant("_LEARNING_RATE"),
            },
            "grow_policy": "lossguide",
        },
        "recorded_from": {"run_id": "fixture", "git_commit": "0" * 40,
                          "recorded_at": "2026-08-17"},
    }}
    held = verify.Verdict()
    verify.check_accuracy_anchor(records, config, held, anchors=good)
    statuses = {
        c["scope"]: c for c in held.checks if c["check"] == "accuracy_anchor"
    }
    check(
        statuses.get(anchor_key, {}).get("status") == verify.PASS,
        "an arm matching its own anchor exactly did not pass the accuracy gate",
    )
    regressed = dict(good[anchor_key])
    regressed["value"] = 0.30
    dropped = verify.Verdict()
    verify.check_accuracy_anchor(
        records, config, dropped, anchors={anchor_key: regressed}
    )
    hit = [
        c for c in dropped.checks
        if c["scope"] == anchor_key and c["status"] == verify.FAIL
    ]
    check(
        bool(hit),
        "a 33 percent accuracy regression against our own recorded anchor did "
        "not FAIL. This is the only accuracy gate in the harness",
    )
    if hit:
        check(
            "catboost" not in hit[0]["detail"]
            and "lightgbm" not in hit[0]["detail"],
            "the anchor verdict names a competitor. It is self-anchored by "
            f"construction and no peer belongs in it: {hit[0]['detail']}",
        )
    missing = verify.Verdict()
    verify.check_accuracy_anchor(records, config, missing, anchors={})
    check(
        any(
            c["status"] == verify.WARN and "NO ANCHOR" in c["detail"]
            for c in missing.checks
        ),
        "an arm with no recorded anchor did not WARN. Silently passing is how "
        "a bad number becomes the anchor: the first run of a new arm would "
        "establish whatever it happened to produce as correct",
    )
    check(
        not any(c["status"] == verify.FAIL for c in missing.checks),
        "a missing anchor FAILED. A legitimately new arm would then fail every "
        "run until somebody blessed it, and a harness that refuses to run a "
        "new arm is a harness people route around",
    )

    # The peer comparison must be incapable of failing anything.
    peer = verify.Verdict()
    verify.check_accuracy_peer(records, config, peer)
    check(
        not peer.failed,
        "the peer comparison produced a FAIL. It is a scoreboard and gates "
        "nothing: our accuracy against our own anchor is the gate",
    )
    check(
        all(
            c["status"] in (verify.NOTE, verify.SKIP, verify.WARN)
            for c in peer.checks if c["check"] == "accuracy_peer"
        ),
        "a peer scoreboard line came back as PASS. Reporting green for one arm "
        "and yellow for another is a gate wearing a different word",
    )


def check_pair_plan():
    """The pair run's own invariants, plus the two things a reader of its table
    must not be able to get wrong. Added 2026-08-17 with `pairs.py`.

    `pairs.check` already raises on an inconsistent plan and is called by
    `pairs.plan` and by `run.py`, so this runs it. What this adds on top is the
    two assertions that are about MEANING rather than about structure, and both
    are ones a plan can satisfy structurally while being wrong:

    THE TWO CLASSES MUST BE DISTINGUISHABLE IN THE OUTPUT. Class A's `mojotrees`
    arm is our leaf-wise implementation at LightGBM's regularizer and Class B's
    `shipped.lossguide` is our product. Reading the first as the second is the
    single most consequential misreading available in that table, and the only
    thing preventing it is that the two rows carry different `block` values and
    that the report prints a legend for them.

    THE TWO ARMS MUST ACTUALLY DIFFER. If `BASE_PARAMS['lambda_l2']` ever equals
    the shipped value again, the mirror arm and the product arm are the same fit
    and the run pays for two copies of one number while presenting them as a
    comparison. `pairs.check_shipped_is_default` raises on that; this asserts the
    resolved dicts differ, which is the same claim read off the translator rather
    than off the constants.
    """
    import pairs
    import report
    import scenarios
    import verify

    try:
        all_arms = pairs.arms()
        pairs.check(all_arms)
    except AssertionError as exc:
        FAILURES.append(f"pairs.check failed: {exc}")
        return

    blocks = {arm["block"] for arm in all_arms}
    check(
        blocks <= set(pairs.CLASSES),
        f"pairs emits blocks {sorted(blocks - set(pairs.CLASSES))} that "
        "pairs.CLASSES does not describe, so report._block_legend would print "
        "them as UNDECLARED",
    )
    check(
        any(b.startswith("A") for b in blocks)
        and any(b.startswith("B") for b in blocks),
        "the pair run does not carry both classes. Both were asked for in one "
        "table so that neither can be read as the other",
    )
    vocabulary = report._block_vocabulary()
    for block in sorted(blocks):
        check(
            block in vocabulary,
            f"report._block_vocabulary does not know pairs block {block!r}, so "
            "the frontier table would print the label with no legend and the "
            "reader supplies the meaning",
        )

    # The two arms must resolve to different models, through the translator that
    # actually builds them.
    spec = scenarios.resolve("dense_regression", "standard", "synthetic")
    mirror = scenarios.mojotrees_params(spec, "cpu")
    product = scenarios.mojotrees_params(spec, "cpu", dict(pairs.SHIPPED_LOSSGUIDE))
    check(
        mirror.get("lambda_l2") != product.get("lambda_l2"),
        "the Class A mirror arm and the Class B product arm resolve the same "
        f"lambda_l2 ({mirror.get('lambda_l2')!r}). They are then the same fit, "
        "the run pays twice for one number, and the table presents two "
        "identical rows as a comparison between a mirror and a product",
    )
    check(
        product.get("lambda_l2") is None,
        "the Class B product arm resolves a lambda_l2 VALUE rather than None. "
        "None is the only thing sklearn.py::_Base._l2_named reads as unset, and "
        "an arm that names the key does not get the package default it claims "
        f"to be measuring. Got {product.get('lambda_l2')!r}",
    )
    # Every key the two share must otherwise be equal, or the pair's headline
    # claim -- that they differ in exactly one parameter -- is false.
    differing = sorted(
        key for key in set(mirror) | set(product)
        if mirror.get(key) != product.get(key)
    )
    check(
        differing == ["lambda_l2"],
        "the Class A mirror arm and the Class B product arm differ in "
        f"{differing} rather than in lambda_l2 alone. pairs.py's headline claim "
        "is that one parameter separates them, and a reader who takes that "
        "claim while a second parameter moved is attributing the whole "
        "difference to the wrong thing",
    )
    # The symmetric arm must leave both automatic-rate gate keys unset, or it
    # RAISES at fit rather than degrading, and every one of its cells is an
    # infrastructure failure that withholds the whole run's quality verdict.
    symmetric = scenarios.mojotrees_params(
        spec, "cpu", dict(pairs.SHIPPED_SYMMETRIC)
    )
    check(
        symmetric.get("auto_learning_rate") is True
        and symmetric.get("learning_rate") is None
        and symmetric.get("lambda_l2") is None,
        "the symmetric shipped arm resolves auto_learning_rate=True beside a "
        f"named learning_rate ({symmetric.get('learning_rate')!r}) or a named "
        f"lambda_l2 ({symmetric.get('lambda_l2')!r}). "
        "sklearn.py::_Base._auto_learning_rate_knobs RAISES on either, so every "
        "cell of this arm would be an infrastructure failure and run.py "
        "withholds the quality verdict for the WHOLE matrix on one of those",
    )
    # And frontier's own auto-rate arms, which had exactly this defect at head.
    import frontier

    for arm in frontier.arms():
        if arm["skip"]:
            continue
        resolved = scenarios.mojotrees_params(
            spec, arm["device"], dict(arm["params"])
        )
        if not resolved.get("auto_learning_rate"):
            continue
        check(
            resolved.get("learning_rate") is None
            and resolved.get("lambda_l2") is None,
            f"frontier arm {arm['id']} resolves auto_learning_rate=True beside "
            "a named learning_rate or lambda_l2 and would RAISE at fit. "
            "Omitting a key from an arm's overrides does not unset it: "
            "scenarios.mojotrees_params copies both out of BASE_PARAMS. Pass "
            "None. See frontier.RESOLVED_SINCE"
            "['auto_rate_needed_explicit_none']",
        )
    check(
        float(verify.shipped_constant("_LAMBDA_L2") or -1.0)
        in [float(v) for v in pairs.L2_AXIS],
        "the lambda_l2 we ship is not a point on pairs.L2_AXIS, so the run "
        "cannot tell whether the value we ship is the value the registered "
        "decision rule would pick",
    )


def check_stale_anchors():
    """A STALE ANCHOR CANNOT GATE. Added 2026-08-17.

    An accuracy anchor is an absolute recorded value and the harness will not
    recompute one, which is the whole ratchet defence (`LANE_RULES.md` rule 3).
    That leaves one hole the rule does not close: an anchor can stop describing
    the model we ship without any number in it changing, because an arm that
    leaves a parameter UNSET follows a package default, and a default can move.
    It did, on 2026-08-17, when `lambda_l2` went from 0.0 to 1.0 under every
    non-symmetric growth policy.

    **The failure mode this guards is silence, not a wrong number.** A stale
    anchor produces a perfectly well-formed PASS. Nothing about the verdict looks
    unusual; it is simply a comparison against a model nobody ships, presented as
    this arm's accuracy gate holding. So the assertion here is not "the number is
    right", it is that a stale anchor may produce NEITHER a PASS NOR a FAIL. Both
    of those are statements about an arm against its own reference and a stale
    anchor is not that reference.

    Four things are asserted and they fail in different directions:

    1. an anchor whose unset parameter moved is detected as stale;
    2. it produces a WARN and never a PASS or a FAIL;
    3. an anchor with no `configuration` block is UNKNOWN, also WARNs, and also
       does not gate, on the precedent of C10's missing cpu twin;
    4. `propose_anchors` WRITES the `configuration` block, so the run a person
       adopts from is current by construction and the staleness clears itself.

    The fourth is the one that makes the mechanism maintainable. Without it,
    staleness would be a hand-kept list, and a hand-kept list about staleness
    goes stale.
    """
    import verify

    shipped = verify.shipped_constant("_LAMBDA_L2")
    check(
        shipped is not None,
        "verify.shipped_constant cannot read _LAMBDA_L2 out of "
        "python/mojotrees/sklearn.py, so nothing can tell whether an anchor "
        "still describes the lambda_l2 we ship and every anchor is UNKNOWN",
    )
    if shipped is None:
        return
    check(
        "lambda_l2" in verify.STALE_ANCHOR_PARAMETERS,
        "verify.STALE_ANCHOR_PARAMETERS does not track lambda_l2, which is the "
        "parameter whose default moved on 2026-08-17 and the reason this "
        "mechanism exists",
    )

    def anchor(followed, recorded, value=0.40, grow_policy="lossguide"):
        return {
            "primary_metric": "rmse", "value": value,
            "environment": {"arch": "arm64", "cpu_model": "Test"},
            "configuration": {
                "passed": {"lambda_l2": None if "lambda_l2" in followed else 0.0},
                "followed_default": list(followed),
                "shipped_at_record": {"lambda_l2": recorded},
                "grow_policy": grow_policy,
            },
            "recorded_from": {"run_id": "fixture", "git_commit": "0" * 40,
                              "recorded_at": "2026-08-16"},
        }

    moved = shipped + 1.0
    stale = verify.anchor_staleness(anchor(["lambda_l2"], moved))
    check(
        stale is not None and stale.get("parameter") == "lambda_l2",
        "an anchor whose arm left lambda_l2 UNSET, recorded at a value the "
        "package no longer resolves to, was not detected as stale. That anchor "
        "describes a model we do not ship and it would gate as though it were "
        f"this arm's reference. Got: {stale!r}",
    )
    if stale is not None:
        check(
            "cleared_by" in stale and "propose-anchors" in stale["cleared_by"],
            "the stale verdict does not say how it is cleared. The remedy is a "
            "RUN and a deliberate adoption, and a reader told only that "
            "something is stale will reach for the edit that installs the "
            "ratchet",
        )
    # NAMED, not unset: a value the arm typed is a property of the arm, and
    # moving a package default cannot invalidate it. If this came back stale the
    # mechanism would condemn every pinned arm in the harness, which is most of
    # them.
    check(
        verify.anchor_staleness(anchor([], moved)) is None,
        "an anchor whose arm NAMED lambda_l2 was reported stale because the "
        "package default moved. A named value does not follow the default, so "
        "this would invalidate every pinned arm in the harness for a change "
        "that cannot have touched them",
    )
    check(
        verify.anchor_staleness(anchor(["lambda_l2"], shipped)) is None,
        "an anchor recorded at exactly the value we ship was reported stale, so "
        "no anchor could ever be current and the gate would never run",
    )
    # THE FALSE POSITIVE THIS DESIGN HAD TO AVOID, and it would have hit every
    # plain mojotrees arm in the harness. `scenarios.mojotrees_params` copies
    # `lambda_l2` and `learning_rate` out of BASE_PARAMS onto EVERY mojotrees
    # arm, so an arm that names neither in its own overrides still PASSES both,
    # and a default change cannot touch it. `followed_default` is therefore read
    # off what was PASSED (None means unset) and not off the arm's override dict.
    passed_not_unset = {
        "primary_metric": "rmse", "value": 0.40,
        "configuration": {
            "passed": {"lambda_l2": moved},
            "followed_default": [],
            "shipped_at_record": {"lambda_l2": moved},
            "grow_policy": "lossguide",
        },
    }
    check(
        verify.anchor_staleness(passed_not_unset) is None,
        "an anchor whose harness PASSED a lambda_l2 value was reported stale "
        "because the package default moved. BASE_PARAMS passes that key to every "
        "mojotrees arm, so this would stale every plain arm in the harness for a "
        "change that cannot have altered one of them",
    )
    # The same claim one level down, on the DERIVATION rather than on a
    # hand-built dict, because the fixture above cannot catch
    # `anchor_configuration` putting a passed key into `followed_default`.
    def cfg(lambda_l2):
        return verify.anchor_configuration({
            "params": {"engine": {"lambda_l2": lambda_l2, "learning_rate": 0.1,
                                  "grow_policy": "lossguide"}},
            "arm_overrides": {"params": {}},
        })

    check(
        cfg(0.0)["followed_default"] == []
        and cfg(0.0)["passed"]["lambda_l2"] == 0.0,
        "verify.anchor_configuration put a PASSED parameter into "
        f"followed_default: {cfg(0.0)}. Only None means unset, and treating a "
        "passed value as a followed default stales every arm the harness hands "
        "BASE_PARAMS to, which is all of them",
    )
    check(
        cfg(None)["followed_default"] == ["lambda_l2"],
        "verify.anchor_configuration did not record an UNSET lambda_l2 as a "
        f"followed default: {cfg(None)}. Then a default change would never be "
        "detected and a stale anchor would gate silently, which is the whole "
        "failure this mechanism exists for",
    )
    # THE PER-POLICY CONSTANT. A symmetric arm's unset lambda_l2 resolves from
    # CatBoost's constant and not from _LAMBDA_L2 (`sklearn.py::_Base._params`,
    # `l2_default = _CATBOOST_L2_LEAF_REG if catboost_mode else _LAMBDA_L2`), so
    # pricing one against the other reports staleness that is not there.
    catboost_l2 = verify.shipped_constant("_CATBOOST_L2_LEAF_REG")
    check(
        catboost_l2 is not None and catboost_l2 != shipped,
        "_CATBOOST_L2_LEAF_REG is unreadable or equal to _LAMBDA_L2, so this "
        "check cannot tell whether the per-policy branch is being taken",
    )
    check(
        verify.stale_anchor_constant("lambda_l2", "symmetrictree")
        == "_CATBOOST_L2_LEAF_REG"
        and verify.stale_anchor_constant("lambda_l2", "lossguide")
        == "_LAMBDA_L2",
        "verify.stale_anchor_constant does not branch on the grow policy. An "
        "unset lambda_l2 resolves from CatBoost's constant under symmetrictree "
        "and from LightGBM's under every other policy, so one constant for both "
        "prices a symmetric arm against a leaf-wise default",
    )
    if catboost_l2 is not None:
        check(
            verify.anchor_staleness(
                anchor(["lambda_l2"], catboost_l2, grow_policy="symmetrictree")
            ) is None,
            "a SYMMETRIC arm's anchor recorded at CatBoost's l2_leaf_reg was "
            "reported stale, which means it was compared against _LAMBDA_L2. "
            "Every symmetric anchor would then be permanently stale and the "
            "gate would never run on the arm the documented defaults name",
        )
    # And the derived rate, which has no constant to compare against at all.
    check(
        verify.stale_anchor_constant("learning_rate", "symmetrictree") is None,
        "verify.stale_anchor_constant claims a constant for a symmetric arm's "
        "learning rate. Under symmetrictree an unset rate is DERIVED per fit "
        "from the row count, the iteration count and the loss, so there is no "
        "literal to read and comparing one would invent a verdict",
    )
    unknown = verify.anchor_staleness(
        {"primary_metric": "rmse", "value": 0.40}
    )
    check(
        unknown is not None and unknown.get("unknown"),
        "an anchor with no `configuration` block was treated as current. "
        "Nothing can tell whether it describes what we ship, and a check that "
        "cannot run must say so rather than pass",
    )
    declared = verify.anchor_staleness({
        "primary_metric": "rmse", "value": 0.40,
        "configuration": {"passed": {"lambda_l2": None},
                          "followed_default": ["lambda_l2"],
                          "shipped_at_record": {"lambda_l2": shipped},
                          "grow_policy": "lossguide"},
        "stale": {"since": "2026-08-17", "why": "retired by hand",
                  "cleared_by": "a run"},
    })
    check(
        declared is not None and declared.get("declared"),
        "an explicit `stale` block on an anchor entry was ignored. A person "
        "must be able to retire an anchor for a reason no comparison can see, "
        "and that declaration has to win over the comparison",
    )

    # Now the gate itself, on a record whose anchor is stale.
    record = {
        "status": "ok", "scenario": "dense_regression", "tier": "standard",
        "engine": "mojotrees", "arm": "mojotrees", "threads": 4,
        "device_used": "cpu", "device_requested": "cpu",
        "cell_role": "measured",
        "primary_metric": "rmse", "quality": {"rmse": 0.40},
        "params": {"num_boost_round": 100},
        "data": {"data_kind": "synthetic", "dataset": "generated:x"},
        "phases": {"train": {"elapsed_s": 1.0}},
        "environment": {"cpu": {"arch": "arm64", "model": "Test"}},
    }
    config = json.load(open(os.path.join(HERE, "thresholds.json")))
    key = verify._anchor_key(record)
    for label, entry, worse in (
        # Same value as the anchor, which would have PASSED.
        ("matching", anchor(["lambda_l2"], moved, value=0.40), False),
        # Far worse than the anchor, which would have FAILED.
        ("regressed", anchor(["lambda_l2"], moved, value=0.20), True),
        # No configuration block at all: UNKNOWN, and equally ungating.
        ("unknown", {"primary_metric": "rmse", "value": 0.20,
                     "environment": {"arch": "arm64", "cpu_model": "Test"}},
         True),
    ):
        verdict = verify.Verdict()
        verify.check_accuracy_anchor(
            [record], config, verdict, anchors={key: entry}
        )
        rows = [
            c for c in verdict.checks
            if c["check"] == "accuracy_anchor" and c["scope"] == key
        ]
        check(
            len(rows) == 1 and rows[0]["status"] == verify.WARN,
            f"a stale or uncheckable anchor ({label}) produced "
            f"{[r['status'] for r in rows]!r} rather than exactly one WARN. A "
            "stale anchor must not be able to PASS, because that reports this "
            "arm's accuracy gate as holding against a model nobody ships, and "
            "it must not be able to FAIL, because that stops a run on a "
            "comparison it was not entitled to make",
        )
        if rows:
            check(
                rows[0].get("stale") is True and not rows[0].get("anchored"),
                f"the {label} anchor row does not carry `stale` and "
                "`anchored: False`, so report.py cannot tell a stale anchor "
                "from a held one and would print a percentage against it",
            )
        check(
            not verdict.failed,
            f"a {label} anchor moved the run's exit code. Staleness is a "
            "reporting state, not a run failure: the run is what CLEARS it, so "
            "failing the run would make the remedy unreachable",
        )
        if worse:
            # And the run must not read as clean either. The uncovered note is
            # what says the gate covered nothing for this cell.
            check(
                any(
                    c["check"] == "accuracy_anchor" and c["scope"] == "run"
                    and c["status"] == verify.NOTE
                    for c in verdict.checks
                ),
                f"a {label} anchor left no run-level note, so a reader of the "
                "verdict summary sees no failures and concludes the accuracy "
                "gate ran. A gate that emits nothing is indistinguishable from "
                "a gate that passes (LANE_RULES rule 8)",
            )

    # And the writer. `propose_anchors` must emit the block, or every anchor a
    # person adopts is UNKNOWN and the gate is permanently unarmed.
    source = open(os.path.join(HERE, "verify.py")).read()
    proposer = source.split("def propose_anchors")[1]
    check(
        '"configuration": anchor_configuration(record)' in proposer,
        "verify.propose_anchors does not write the `configuration` block. "
        "Every anchor adopted from it would then be ANCHOR CURRENCY UNKNOWN "
        "and would never gate, which is a silently unarmed gate rather than a "
        "loud one",
    )
    # The report has to render the two states differently, because the remedies
    # differ: a missing anchor needs an adoption and a stale one needs a run.
    import report

    check(
        "STALE ANCHOR" in report._anchor_cell(
            {"status": verify.WARN, "stale": True,
             "stale_reason": "parameter_default_moved",
             "stale_parameter": "lambda_l2"},
            False,
        ),
        "report._anchor_cell does not render a stale anchor as STALE. Printed "
        "as `NO ANCHOR` it sends the reader to the wrong remedy, and printed as "
        "`held` it is a false green",
    )


def check_arm_keying():
    """Every per-cell key in the readers is the ARM, and every role test is
    the ENGINE. Added 2026-08-17 after three instances in one day of the same
    mistake: a check that looked green because it compared the wrong things.

    An `--arms` run takes its cells from a module rather than from the engine
    cross product, so one engine name covers dozens of arms at different tree
    counts and bin counts, and every arm id differs from its engine name. That
    is the shape under which the two directions of the mistake show:

    - a CELL key on the engine folds all those arms into one cell. It made
      `check_device_agreement` compare whichever cpu arm and whichever gpu arm
      were written last and report green; it would have made `check_baseline`
      check one arm of forty; `check_determinism` pool forty digests into one
      cell; `check_backend_proof` look for a gpu row's twin among every arm's
      cpu digest; and `report.build_cells` print one row with `reps 40`.
    - a ROLE test on the arm id matches nothing. `check_accuracy_peer` tested
      `engines_judged` and `competitors`, both lists of ENGINE names, against
      the arm id, so on an `--arms` run it built no bar and judged no arm,
      with no line saying so; `report._frontier` decided "competitor" from
      three literal arm names.

    The fixture below is that shape, in miniature: two mojotrees arms with a
    cpu twin and a gpu row each, two lightgbm arms, no plain arm anywhere. It
    runs no engine; the records are dicts.
    """
    import verify
    import report
    import summarize

    def row(arm, engine, device, trees, rmse, sha, repeat=0):
        return {
            "status": "ok", "scenario": "dense_regression", "tier": "standard",
            "engine": engine, "arm": arm, "threads": 4, "repeat": repeat,
            "device_used": device, "device_requested": device,
            "cell_role": (
                "oracle" if device == "cpu" and engine == "mojotrees"
                else "measured"
            ),
            "primary_metric": "rmse", "quality": {"rmse": rmse},
            "baseline_quality": {"rmse": 1.0},
            "params": {"num_boost_round": trees}, "predictions_sha256": sha,
            "data": {"data_kind": "synthetic", "dataset": "generated:x",
                     "train": {"digest": "d1"}, "test": {"digest": "d2"}},
            "phases": {"train": {"elapsed_s": 1.0 + trees / 100}},
            "environment": {"cpu": {"arch": "arm64", "model": "Test"}},
            "backend_proof": None,
        }

    small, big = "F.r.mojotrees.trees.25", "F.r.mojotrees.trees.200"
    records = [
        row(small, "mojotrees", "cpu", 25, 0.50, "same-small"),
        row(small, "mojotrees", "gpu", 25, 0.50, "same-small"),
        row(big, "mojotrees", "cpu", 200, 0.30, "big-cpu"),
        row(big, "mojotrees", "gpu", 200, 0.31, "big-gpu"),
        row(big, "mojotrees", "gpu", 200, 0.31, "big-gpu", repeat=1),
        row("F.r.lightgbm.trees.25", "lightgbm", "cpu", 25, 0.52, "l25"),
        row("F.r.lightgbm.trees.200", "lightgbm", "cpu", 200, 0.29, "l200"),
    ]
    config = json.load(open(os.path.join(HERE, "thresholds.json")))
    verdict = verify.Verdict()
    verify.check_determinism(records, config, verdict)
    verify.check_backend_proof(records, config, verdict)
    verify.check_baseline(records, config, verdict)
    verify.check_accuracy_peer(records, config, verdict)
    verify.check_device_agreement(records, config, verdict, os.devnull)

    def scopes(name):
        return {c["scope"] for c in verdict.checks if c["check"] == name}

    check(
        {f"dense_regression/{small}/t4/rmse", f"dense_regression/{big}/t4/rmse"}
        <= scopes("device_agreement"),
        "check_device_agreement did not emit one metric verdict PER ARM on an "
        "--arms-shaped run. It is keyed on the arm because an engine key "
        f"folds every arm into one cell: {sorted(scopes('device_agreement'))}",
    )
    check(
        any(
            c["check"] == "device_agreement" and c["status"] == verify.FAIL
            and big in c["scope"]
            for c in verdict.checks
        ) and not any(
            c["check"] == "device_agreement" and c["status"] == verify.FAIL
            and small in c["scope"]
            for c in verdict.checks
        ),
        "device_agreement paired the wrong arms: the 200-tree arm disagrees "
        "across devices and the 25-tree arm agrees, and the verdict must say "
        "exactly that, per arm",
    )
    check(
        f"dense_regression/{big}/gpu/t4" in scopes("determinism")
        and any(
            c["check"] == "determinism" and c["status"] == verify.PASS
            and big in c["scope"]
            for c in verdict.checks
        ),
        "check_determinism did not group repeats by ARM. Two arms of one "
        "engine pooled into one cell have different digests by construction "
        "and would fail a deterministic trainer for having been asked two "
        "questions",
    )
    proof = {c["scope"]: c for c in verdict.checks if c["check"] == "backend_proof"}
    check(
        proof.get(f"dense_regression/{small}/gpu/t4", {}).get("status") == verify.FAIL
        and proof.get(f"dense_regression/{big}/gpu/t4", {}).get("status") == verify.WARN,
        "check_backend_proof did not look up each gpu row's cpu twin BY ARM. "
        "The 25-tree arm's gpu digest equals its own cpu twin with no proof "
        "(A, B and C: FAIL); the 200-tree arm's does not (WARN). Any other "
        f"answer means the twins were pooled: {sorted(proof)}",
    )
    baseline = scopes("baseline")
    check(
        len(baseline) == 6,
        "check_baseline did not check every arm. Keyed on the engine it kept "
        "the FIRST record per engine and device and left the rest unchecked "
        f"with no line: {sorted(baseline)}",
    )
    peer = [c for c in verdict.checks if c["check"] == "accuracy_peer"]
    check(
        len(peer) == 4 and all(c["status"] == verify.NOTE for c in peer),
        "check_accuracy_peer judged nothing on an --arms-shaped run. "
        "`engines_judged` and `competitors` are ENGINE names and must be "
        "tested against record['engine'], not the arm id, or no arm is ever "
        f"judged and no bar is ever built: {[(c['status'], c['scope']) for c in peer]}",
    )
    check(
        all("F.r.lightgbm.trees" in c["detail"] for c in peer),
        "the peer scoreboard did not find the lightgbm arm as the bar at the "
        "matched tree count",
    )

    cells = report.build_cells(records, 0.2)
    check(
        len(cells) == 6
        and cells[
            ("dense_regression", "standard", "synthetic", big, "gpu", 4)
        ]["repeats"] == 2,
        "report.build_cells did not produce one cell PER ARM. An engine key "
        f"prints one row with the arms as its repeats: {sorted(cells)}",
    )
    # And the two fields `cell_key` grew on 2026-08-17, asserted by POSITION so
    # that a reader of this test can see the key's shape. Their absence let one
    # scenario at two tiers, or its generator beside its real dataset, share a
    # table row and print a median across both.
    check(
        all(key[1] == "standard" and key[2] == "synthetic" for key in cells),
        "report.cell_key no longer carries the tier and the data kind in "
        f"positions 1 and 2: {sorted(cells)[0]}. Without both, a run holding "
        "dense_regression at two tiers, or its generator and UCI "
        "YearPredictionMSD, averages them into one row",
    )
    lines = []
    report._frontier(records, config, lines.append)
    check(
        any("F.r.lightgbm.trees.200 | competitor" in line for line in lines)
        and not any("F.r.mojotrees" in line and "| competitor |" in line for line in lines),
        "report._frontier decided competitor status from the arm id rather "
        "than the engine. A lightgbm arm under a frontier id is still a "
        "competitor and a mojotrees arm never is",
    )
    agreement = summarize.build_device_agreement(records, "mojotrees")
    check(
        {entry["arm"] for entry in agreement} == {small, big},
        "summarize.build_device_agreement did not pair BY ARM: "
        f"{[(e['arm'], e['device']) for e in agreement]}",
    )
    peers_too = records + [row("catboost", "catboost", "cpu", 100, 0.3, "cb")]
    try:
        named = summarize.project_engine_name(peers_too)
    except ValueError as exc:  # pragma: no cover - the failure being guarded
        named = exc
    check(
        named == "mojotrees",
        "summarize.project_engine_name refused or misnamed a run that carries "
        "a peer engine. Peers and subject variants are ordinary rows of one "
        f"run and the subject set is verify.SUBJECT_ENGINES: {named!r}",
    )


def check_arm_keying_writer():
    """The WRITER side of `check_arm_keying`: `run.py`'s matrix keys a cell
    by the ARM and tests a role by the ENGINE. Added 2026-08-17.

    Every other fixture in this file that walks `run.build_matrix` does so
    without `--arms`, where arm == engine and the two directions of the
    keying mistake are invisible by construction. This one hands the builder
    an `--arms`-shaped list: two arms of one engine at different tree counts,
    each with a cpu and a gpu cell, a third arm of the same engine whose gpu
    cell the arm itself declares skipped, and a competitor arm under a
    frontier-style id. The engines are the two CORRECTNESS arms, because they
    exist to witness bugs a green suite could not see and a writer-side key
    that folded them would be a coverage gap inside the coverage fix.

    What must hold, and what an engine key would have done instead:

    - `_mark_oracle_cells` labels the cpu twin of EACH gpu arm an oracle and
      trims it, per arm. Under an engine key the arm with no gpu twin would
      have been folded with its siblings, labeled an oracle it is not, and
      trimmed from a measurement down to one repeat.
    - `label()` gives every arm its own filename stem, so two arms of one
      engine cannot overwrite each other's job, record and predictions.
    - `_engine_skip_reason` is a ROLE test and reads the engine, so a
      lightgbm arm under a frontier id is still refused on the gpu and a
      mojotrees arm under one is not.
    - Every job kind, runnable and skipped, carries `arm`, `axis` and
      `arm_block`, which is what `worker.py` and the skip/timeout/crash
      writers copy onto the record.

    Runs no cells, touches no data.
    """
    import run

    parser = run.build_parser()
    args = parser.parse_args(
        ["--tier", "smoke", "--threads", "4", "--repeats", "3", "--device", "cpu"]
    )
    args.scenario = args.scenario or []
    args.engine = args.engine or []
    args.device = args.device or ["cpu"]
    args.threads = args.threads or [4]

    def arm(engine, trees, device, block="axis", skip=None):
        # NOTE the id carries no device: a cpu cell and a gpu cell of one arm
        # share the arm id, which is what makes one the other's twin
        # everywhere the harness pairs them (`run._oracle_key`,
        # `verify._oracle_cell_key`, `verify.check_device_agreement`).
        return {
            "id": f"F.r.{engine}.trees.{trees}", "scenario": "dense_regression",
            "tier": "smoke", "variant": "synthetic", "engine": engine,
            "device": device, "axis": "trees", "axis_value": trees,
            "block": block, "params": {"n_estimators": trees},
            "dataset_params": {}, "env": {}, "repeats": 3, "skip": skip,
        }

    colsample, cosine = "mojotrees_symmetric_colsample", "mojotrees_cosine_leafwise"
    arms = [
        arm(colsample, 25, "cpu"), arm(colsample, 25, "gpu"),
        arm(colsample, 200, "cpu"), arm(colsample, 200, "gpu"),
        arm(cosine, 25, "cpu"), arm(cosine, 25, "gpu"),
        # The gpu leg declared skipped by the arm itself: this cpu cell has
        # NO twin and is the measurement.
        arm(cosine, 200, "cpu"),
        arm(cosine, 200, "gpu", skip="declared by the plan for this fixture"),
        arm("lightgbm", 25, "cpu", block="competitor"),
        arm("lightgbm", 25, "gpu", block="competitor"),
    ]
    jobs = run.build_matrix(args, arms)
    runnable = [job for job in jobs if "skip" not in job]
    skipped = [job for job in jobs if "skip" in job]

    def cells(arm_id, device):
        return [j for j in runnable if j["arm"] == arm_id and j["device"] == device]

    for engine, trees in ((colsample, 25), (colsample, 200), (cosine, 25)):
        arm_id = f"F.r.{engine}.trees.{trees}"
        cpu = cells(arm_id, "cpu")
        check(
            len(cpu) == 1 and cpu[0]["cell_role"] == "oracle",
            f"run._mark_oracle_cells did not label and trim {arm_id}'s cpu "
            "cell PER ARM on an --arms-shaped matrix: found "
            f"{[(j['repeat'], j.get('cell_role')) for j in cpu]}",
        )
        check(
            len(cells(arm_id, "gpu")) == 3
            and all(j["cell_role"] == "measured" for j in cells(arm_id, "gpu")),
            f"{arm_id}'s gpu cells must be measured at full repeats",
        )
    no_twin = cells(f"F.r.{cosine}.trees.200", "cpu")
    check(
        len(no_twin) == 3 and all(j["cell_role"] == "measured" for j in no_twin),
        "a cpu arm whose gpu leg is a declared skip has NO twin and is the "
        "measurement. Keyed by engine it would be folded with the sibling "
        "arms of the same engine, labeled oracle and trimmed to one repeat: "
        f"{[(j['repeat'], j.get('cell_role')) for j in no_twin]}",
    )
    stems = {run.label(j) for j in runnable}
    check(
        len(stems) == len(runnable),
        "run.label must give every (arm, device, threads, repeat) its own "
        "filename stem on an --arms matrix; a collision is one arm's record "
        "overwriting another's",
    )
    lightgbm_gpu = [j for j in skipped if j["engine"] == "lightgbm" and j["device"] == "gpu"]
    check(
        len(lightgbm_gpu) == 1 and "LightGBM" in lightgbm_gpu[0]["skip"],
        "run._engine_skip_reason is a ROLE test and must read the engine: a "
        "lightgbm arm under a frontier-style id is still refused on the gpu",
    )
    check(
        not any(
            j["engine"] in (colsample, cosine) and "LightGBM" in j["skip"]
            for j in skipped
        ),
        "a mojotrees arm under a frontier-style id was refused by a rule "
        "meant for another engine",
    )
    check(
        all(
            j.get("arm") and "axis" in j and "arm_block" in j
            for j in jobs
        ) and {j["arm_block"] for j in jobs if j["engine"] == "lightgbm"} == {"competitor"},
        "every job kind, runnable and skipped, must carry arm, axis and "
        "arm_block, so that every record kind can copy them; competitor arms "
        "must keep block 'competitor' through the matrix",
    )
    manifest_cells = run._oracle_manifest_block(jobs, 1)["cells"]
    check(
        len(manifest_cells) == 3
        and all(f"F.r.{colsample}.trees" in c or f"F.r.{cosine}.trees.25" in c
                for c in manifest_cells),
        "the manifest's oracle block must list oracle cells BY ARM: "
        f"{manifest_cells}",
    )


def check_coverage_guard():
    """`verify.check_coverage` states the silence and nothing else.

    Added 2026-08-17 beside `check_arm_keying`, which is the fixture for the
    defects that motivated it: a gate whose key or role test compares the
    wrong things emits ZERO lines, and zero lines read as a clean run. The
    guard runs last, over the verdict, and warns for every subject cell that
    no per-cell check named. Three fixtures, all dicts, no engine runs:

    A. the `check_arm_keying` shape through every per-cell check and then the
       guard: every subject cell is named, so no uncovered line and one PASS
       summary.
    B. a subject record run ONLY through `check_differential`, whose scope has
       no arm: exactly one WARN naming that cell.
    C. a competitor record with nothing naming it: no line at all. Roles are
       by ENGINE and a competitor is not a subject cell.
    """
    import verify

    def row(arm, engine, device, trees, rmse, sha, repeat=0):
        return {
            "status": "ok", "scenario": "dense_regression", "tier": "standard",
            "engine": engine, "arm": arm, "threads": 4, "repeat": repeat,
            "device_used": device, "device_requested": device,
            "cell_role": (
                "oracle" if device == "cpu" and engine == "mojotrees"
                else "measured"
            ),
            "primary_metric": "rmse", "quality": {"rmse": rmse},
            "baseline_quality": {"rmse": 1.0},
            "params": {"num_boost_round": trees}, "predictions_sha256": sha,
            "data": {"data_kind": "synthetic", "dataset": "generated:x",
                     "train": {"digest": "d1"}, "test": {"digest": "d2"}},
            "phases": {"train": {"elapsed_s": 1.0 + trees / 100}},
            "environment": {"cpu": {"arch": "arm64", "model": "Test"}},
            "backend_proof": None,
        }

    config = json.load(open(os.path.join(HERE, "thresholds.json")))

    def coverage(verdict):
        return [c for c in verdict.checks if c["check"] == "coverage"]

    # A. Every subject cell named.
    small, big = "F.r.mojotrees.trees.25", "F.r.mojotrees.trees.200"
    records = [
        row(small, "mojotrees", "cpu", 25, 0.50, "same-small"),
        row(small, "mojotrees", "gpu", 25, 0.50, "same-small"),
        row(big, "mojotrees", "cpu", 200, 0.30, "big-cpu"),
        row(big, "mojotrees", "gpu", 200, 0.31, "big-gpu"),
        row(big, "mojotrees", "gpu", 200, 0.31, "big-gpu", repeat=1),
        row("F.r.lightgbm.trees.25", "lightgbm", "cpu", 25, 0.52, "l25"),
        row("F.r.lightgbm.trees.200", "lightgbm", "cpu", 200, 0.29, "l200"),
    ]
    verdict = verify.Verdict()
    verify.check_determinism(records, config, verdict)
    verify.check_backend_proof(records, config, verdict)
    verify.check_differential(records, config, verdict, set())
    verify.check_accuracy_anchor(records, config, verdict)
    verify.check_accuracy_peer(records, config, verdict)
    verify.check_baseline(records, config, verdict)
    verify.check_device_agreement(records, config, verdict, os.devnull)
    verify.check_coverage(records, verdict)
    lines = coverage(verdict)
    check(
        len(lines) == 1 and lines[0]["status"] == verify.PASS
        and lines[0]["scope"] == "run" and lines[0]["detail"].startswith("4 of 4"),
        "check_coverage warned on a run where every subject cell was named by "
        "a per-cell check, or did not sum the covered cells into one PASS line: "
        f"{[(c['status'], c['scope'], c['detail'][:40]) for c in lines]}",
    )

    # B. A subject cell that only the arm-free differential saw.
    lone = "F.r.mojotrees.trees.50"
    records = [
        row(lone, "mojotrees", "gpu", 50, 0.40, "lone-gpu"),
        row("F.r.lightgbm.trees.50", "lightgbm", "cpu", 50, 0.41, "l50"),
    ]
    verdict = verify.Verdict()
    verify.check_differential(records, config, verdict, set())
    verify.check_coverage(records, verdict)
    lines = coverage(verdict)
    warned = [c for c in lines if c["status"] == verify.WARN and c["scope"] != "run"]
    check(
        len(warned) == 1
        and warned[0]["scope"] == f"dense_regression/{lone}/gpu/t4"
        and "indistinguishable from a passing gate" in warned[0]["detail"]
        and any(c["scope"] == "run" and c["status"] == verify.WARN for c in lines),
        "check_coverage did not warn for a subject cell that no per-cell check "
        "named. check_differential's scope carries no arm and must not cover: "
        f"{[(c['status'], c['scope']) for c in lines]}",
    )
    # A prefix of the arm id is not the arm id. Components, not substrings.
    verdict = verify.Verdict()
    verdict.add(verify.PASS, "baseline", f"dense_regression/{lone}.b/gpu", "x")
    verify.check_coverage(records, verdict)
    check(
        any(c["status"] == verify.WARN and c["scope"] != "run" for c in coverage(verdict)),
        "check_coverage matched an arm id as a substring of a longer arm id; "
        "it must match on delimited scope components",
    )

    # C. A competitor is not a subject cell.
    records = [row("F.r.lightgbm.trees.50", "lightgbm", "cpu", 50, 0.41, "l50")]
    verdict = verify.Verdict()
    verify.check_coverage(records, verdict)
    check(
        not coverage(verdict),
        "check_coverage wrote a line for a competitor-only run. Roles are by "
        f"engine and a competitor is never a subject cell: {coverage(verdict)}",
    )


#: Files whose prose citations are held to `path::symbol` form. LANE_RULES.md
#: rule 7 (2026-08-17): ten lanes edit the same files, so a pointer by line
#: number drifts within the hour while a pointer by enclosing symbol does not. Paths are relative to the repository root.
CITATION_SCAN = (
    "bench/real_data/*.py",
    "bench/results/LANE_RULES.md",
    "bench/results/PROFILE_PROTOCOL.md",
    "bench/results/SESSION_QUEUE.md",
    "docs/design/ACCURACY_GAP.md",
    "docs/design/ACCURACY_BUDGET.md",
)

#: Where a bare `binning.mojo::MAX_BINS` is looked up when the path is not
#: given from the repository root, in this order.
CITATION_SEARCH_DIRS = (
    "",
    "src/mojotrees",
    "bench/real_data",
    "python/mojotrees",
    "bindings",
    "bench",
    "tools",
)

#: Line-number pointers that are allowed to stay, as (path, line, reason).
#: Every entry is visible debt or a deliberate third-party pin; a pointer into
#: THIS repository never belongs here, it gets converted. Split into
#: (path, line) rather than written as one string so that scanning this file
#: does not find the pointer in its own allowlist.
CITATION_LINE_ALLOWLIST = (
    (
        "catboost/core.py",
        804,
        "third-party source pinned to catboost 1.2.10; the version, not the "
        "working tree, fixes the line",
    ),
)

#: `path::symbol` citations whose symbol no longer exists, as
#: (path, symbol, reason). Empty is the goal; an entry here is a citation
#: whose target was removed and whose surrounding claim still needs a reader.
CITATION_SYMBOL_ALLOWLIST = ()


def check_citations():
    """Every `path::symbol` citation in the scanned files names a symbol that
    exists in that file today, and no `path.py:NNN` or `path.mojo:NNN`
    line-number pointer survives in them. Added 2026-08-17 for LANE_RULES.md
    rule 7.

    A `.py` or `.mojo` symbol exists when the file has `def`, `fn`, `struct`,
    `class`, `trait`, `alias`, `var` or `comptime` followed by the name at any
    indentation, or the name at column 0 followed by `=` or `:`. Every dotted
    component of `Struct.method` is checked on its own. A `.md` target counts
    when a heading contains the symbol, and is otherwise not checked, since
    prose has no symbol table.

    A remaining line-number pointer is a defect and is listed by file and
    line, so one cannot come back silently. Third-party pins and removed
    symbols live in the two allowlists above with a reason each, which keeps
    the check green while keeping the debt visible.
    """
    import glob
    import re

    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    files = []
    for pattern in CITATION_SCAN:
        files.extend(sorted(glob.glob(os.path.join(root, pattern))))
    check(files, "check_citations found no files to scan")

    symbol_re = re.compile(
        r"([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|mojo|md))::([A-Za-z_][A-Za-z0-9_.]*)"
    )
    line_re = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|mojo)):(\d+)")

    def resolve(path):
        for base in CITATION_SEARCH_DIRS:
            candidate = os.path.join(root, base, path)
            if os.path.isfile(candidate):
                return candidate
        return None

    def defines(text, name, suffix):
        if suffix == ".md":
            return name in "\n".join(
                line for line in text.splitlines() if line.startswith("#")
            )
        pattern = (
            r"^\s*(?:def|fn|struct|class|trait|alias|var|comptime)\s+"
            + re.escape(name)
            + r"\b|^"
            + re.escape(name)
            + r"\s*[:=]"
        )
        return re.search(pattern, text, re.MULTILINE) is not None

    allowed_lines = {(p, n) for p, n, _ in CITATION_LINE_ALLOWLIST}
    allowed_symbols = {(p, s) for p, s, _ in CITATION_SYMBOL_ALLOWLIST}
    unresolved = []
    line_pointers = []
    checked = 0
    cache = {}
    for path in files:
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                for target, symbol in symbol_re.findall(line):
                    symbol = symbol.rstrip(".")
                    if (target, symbol) in allowed_symbols:
                        continue
                    checked += 1
                    resolved = resolve(target)
                    if resolved is None:
                        unresolved.append(f"{rel}:{number} {target}::{symbol} (no such file)")
                        continue
                    if resolved not in cache:
                        with open(resolved, encoding="utf-8", errors="replace") as h:
                            cache[resolved] = h.read()
                    text = cache[resolved]
                    suffix = os.path.splitext(resolved)[1]
                    if suffix == ".md":
                        # A heading match is evidence; its absence is not,
                        # since prose is cited by section and not by symbol.
                        continue
                    missing = [
                        part for part in symbol.split(".")
                        if part and not defines(text, part, suffix)
                    ]
                    if missing:
                        unresolved.append(
                            f"{rel}:{number} {target}::{symbol} "
                            f"(not defined in {os.path.relpath(resolved, root)}: "
                            f"{', '.join(missing)})"
                        )
                for target, digits in line_re.findall(line):
                    if (target, int(digits)) in allowed_lines:
                        continue
                    line_pointers.append(f"{rel}:{number} {target}:{digits}")

    check(
        checked > 0,
        "check_citations found no path::symbol citations in the scanned files, "
        "which means the pattern or the file list is wrong",
    )
    check(
        not unresolved,
        "path::symbol citations that name a symbol the file does not define "
        "(LANE_RULES.md rule 7; fix the citation or add it to "
        "CITATION_SYMBOL_ALLOWLIST with a reason):\n    "
        + "\n    ".join(unresolved),
    )
    check(
        not line_pointers,
        "line-number citations survive in the scanned files (LANE_RULES.md "
        "rule 7 says cite path::symbol; convert them, or for a third-party pin "
        "add (path, line, reason) to CITATION_LINE_ALLOWLIST):\n    "
        + "\n    ".join(line_pointers),
    )


def check_bayes_floor():
    """The excess-over-floor lens is reported where a floor is KNOWN and
    nowhere else. Added 2026-08-17 with `scenarios.BAYES_FLOOR_RMSE`.

    Three things, each of which fails silently if got wrong. The declared floor
    must equal the noise the generator actually adds at every tier, or the
    excess is measured against a number that is not the floor. On the
    generator variant a small raw gap must come out beside a large excess gap
    (a 1.7 percent RMSE gap on dense_regression is a 70 percent excess-MSE gap
    and a 30 percent excess-RMSE gap), in the note line, in the extra fields
    and in the report cell. And on the REAL variant of the same scenario, which
    has no known floor, no excess figure may appear at all, because a floor
    that was never known cannot be fabricated from the scenario name.
    """
    import inspect
    import numpy as np
    import generators
    import scenarios
    import verify
    import report

    # A. Declaration matches the generator, tier by tier.
    for scenario_id, spec in scenarios.SCENARIOS.items():
        floor = spec.get("bayes_floor")
        if not floor:
            continue
        gen = generators.GENERATORS.get(spec["generator"])
        default_noise = inspect.signature(gen).parameters["noise"].default
        check(
            floor["generator"] == spec["generator"] and floor["metric"] == "rmse",
            f"{scenario_id}: bayes_floor names generator {floor.get('generator')} "
            f"and metric {floor.get('metric')}; the scenario runs "
            f"{spec['generator']} on rmse",
        )
        for tier, kwargs in spec.get("generator_sizes", {}).items():
            noise = kwargs.get("noise", default_noise)
            check(
                abs(noise - floor["value"]) < 1e-12 and abs(floor["mse"] - noise ** 2) < 1e-12,
                f"{scenario_id}/{tier}: declared Bayes floor {floor['value']} but "
                f"the generator adds noise {noise}",
            )
    check(
        scenarios.bayes_floor("dense_regression", "real") is None
        and scenarios.bayes_floor("multiclass", "synthetic") is None
        and scenarios.bayes_floor("dense_regression", "synthetic", {"noise": 0.5}) is None,
        "scenarios.bayes_floor fabricated a floor for a real-data record, an "
        "undeclared scenario, or a stale declaration",
    )
    realized = generators.realized_noise_floor({"n_rows": 200_000, "n_features": 50})
    check(
        abs(realized ** 0.5 - 0.298252) < 5e-7,
        "generators.realized_noise_floor does not reproduce the standard-tier "
        f"held-out noise RMSE 0.298252 from ACCURACY_GAP.md section 1: {realized ** 0.5}",
    )

    # B. The fixture: standard-tier numbers from results/20260817T130309Z-lanecheck,
    # a 1.67 percent raw gap, on a small generator so the realized floor is cheap.
    def row(engine, device, rmse, kind, extra_data=None):
        data = {"data_kind": kind, "train": {"digest": "d1"}, "test": {"digest": "d2"}}
        if kind == "synthetic":
            data.update({
                "dataset": "generated:dense_regression", "generator": "dense_regression",
                "generator_kwargs": {"n_rows": 5_000, "n_features": 20},
                "split": {"kind": "hash", "train_fraction": 0.8, "seed": 1900},
            })
        else:
            data.update({"dataset": "year_prediction_msd"})
        data.update(extra_data or {})
        return {
            "status": "ok", "scenario": "dense_regression", "tier": "standard",
            "engine": engine, "arm": engine, "threads": 4, "repeat": 0,
            "device_used": device, "device_requested": device,
            "cell_role": "measured", "primary_metric": "rmse",
            "quality": {"rmse": rmse}, "baseline_quality": {"rmse": 1.0},
            "params": {"num_boost_round": 100}, "predictions_sha256": engine,
            "data": data, "phases": {"train": {"elapsed_s": 1.0}},
            "environment": {"cpu": {"arch": "arm64", "model": "Test"}},
            "backend_proof": None,
        }

    config = json.load(open(os.path.join(HERE, "thresholds.json")))
    small_floor = generators.realized_noise_floor({"n_rows": 5_000, "n_features": 20})
    ours, theirs = 0.310775, 0.305663
    want_raw = (ours - theirs) / theirs
    want_excess = ((ours ** 2 - small_floor) - (theirs ** 2 - small_floor)) / (theirs ** 2 - small_floor)
    for stamped in (False, True):
        extra = None
        if stamped:
            extra = {"bayes_floor": generators.with_realized_floor(
                scenarios.bayes_floor("dense_regression", "synthetic"),
                {"n_rows": 5_000, "n_features": 20}, None,
            )}
        records = [
            row("mojotrees", "gpu", ours, "synthetic", extra),
            row("catboost", "cpu", theirs, "synthetic", extra),
        ]
        verdict = verify.Verdict()
        verify.check_accuracy_peer(records, config, verdict)
        notes = [c for c in verdict.checks if c["check"] == "accuracy_peer"]
        check(
            len(notes) == 1 and notes[0]["status"] == verify.NOTE
            and abs(notes[0]["worse_relative"] - want_raw) < 1e-9
            and abs(notes[0].get("excess_worse_relative", -1) - want_excess) < 1e-9
            and abs(notes[0].get("bayes_floor_mse", -1) - small_floor) < 1e-12
            and notes[0].get("excess_root_worse_relative", 0) > want_raw * 10,
            "check_accuracy_peer on the generator variant "
            f"({'stamped' if stamped else 'resolved by scenario name'}) did not "
            "carry both the raw gap and the excess-over-floor gap in its fields: "
            f"{notes}",
        )
        check(
            len(notes) == 1
            and f"{abs(want_raw) * 100:.3f} percent" in notes[0]["detail"]
            and "EXCESS over the Bayes floor" in notes[0]["detail"]
            and f"{abs(want_excess) * 100:.3f} percent" in notes[0]["detail"],
            "the accuracy_peer note line does not print the raw gap and the "
            f"excess gap side by side: {notes and notes[0]['detail']}",
        )
        cell = report._peer_cell(notes[0], False) if notes else ""
        check(
            "1.67% behind catboost" in cell and "excess over floor" in cell
            and f"{abs(want_excess) * 100:.1f}% behind" in cell,
            f"report._peer_cell does not show the excess gap beside the raw one: {cell!r}",
        )
        lines = []
        report._ratios(report.build_cells(
            records + [row("lightgbm", "cpu", 0.313535, "synthetic", extra)], 0.2
        ), 1, lines.append)
        check(
            any("0.88% ahead, excess over floor" in line for line in lines),
            "report._ratios' accuracy gap cell does not carry the excess gap on "
            f"the generator variant: {[l for l in lines if l.startswith('| 4')]}",
        )

    # C. The real variant: raw gap only, no excess figure anywhere.
    records = [
        row("mojotrees", "gpu", ours, "real"),
        row("catboost", "cpu", theirs, "real"),
        row("lightgbm", "cpu", 0.313535, "real"),
    ]
    verdict = verify.Verdict()
    verify.check_accuracy_peer(records, config, verdict)
    notes = [c for c in verdict.checks if c["check"] == "accuracy_peer"]
    lines = []
    report._ratios(report.build_cells(records, 0.2), 1, lines.append)
    check(
        len(notes) == 1
        and "excess_worse_relative" not in notes[0]
        and "EXCESS" not in notes[0]["detail"]
        and abs(notes[0]["worse_relative"] - want_raw) < 1e-9
        and "excess" not in report._peer_cell(notes[0], False)
        and not any("excess" in line for line in lines if line.startswith("| 4")),
        "an excess-over-floor figure appeared on the REAL variant of "
        "dense_regression, which has no known floor. A floor is a fact about "
        f"the generator and never about the scenario name: {notes} {lines}",
    )

    # D. decompose.py recovers a planted bias share on arrays built here.
    # 20,000 rows of uniform features, the dense_regression signal, a smooth
    # bias on x0 of known variance and pure noise of known variance: the
    # systematic estimate must land within 10 percent of the planted variance
    # and the pure-noise residual must decompose to (nearly) all variance with
    # no resolution warning. Nothing is trained; the "predictions" are the
    # signal plus the planted terms.
    import decompose
    n = 20_000
    x = np.stack([generators._stream(7, tag, n) for tag in range(1, 7)], axis=1)
    signal = generators.dense_regression_signal(x)
    bias = 0.1 * np.sin(20.0 * x[:, 0])
    noise = generators._normal(7, 40, n) * 0.07
    result = decompose.decompose(signal + bias + noise, signal, x, 45)
    planted = float(np.var(bias))
    check(
        abs(result["systematic"] - planted) < 0.1 * planted
        and result["components"][0]["feature"] == "x0"
        and result["components"][0]["systematic"] > 0.9 * result["systematic"]
        and abs(result["excess_mse"] - np.mean((bias + noise) ** 2)) < 1e-12,
        "decompose.decompose did not recover a planted bias on x0: systematic "
        f"{result['systematic']} against planted {planted}, components "
        f"{[(c['feature'], round(c['systematic'], 6)) for c in result['components']]}",
    )
    pure = decompose.decompose(signal + noise, signal, x, 45)
    check(
        pure["systematic"] < 0.02 * pure["excess_mse"]
        and pure["resolution_warning"] is None,
        "decompose.decompose read bias into a pure-noise residual, or warned "
        f"about resolution on one: {pure['systematic']} of {pure['excess_mse']}, "
        f"{pure['resolution_warning']}",
    )
    spike = 1.0 * ((x[:, 4] > 0.7) & (x[:, 4] < 0.7 + 1.0 / 255))
    narrow = decompose.decompose(signal + spike + noise, signal, x, 45)
    check(
        narrow["resolution_warning"] is not None
        and narrow["resolution_ladder"][0]["systematic"] < 0.5 * narrow["systematic"],
        "decompose.decompose did not warn that 40 bins understate a bias that "
        "lives in a 1/255-wide window, which is the document's finding: "
        f"{narrow['resolution_ladder']} {narrow['resolution_warning']}",
    )


def main():
    check_compiles()
    documents = check_json()
    if not FAILURES:
        check_registry(documents)
        check_params()
        check_comparator()
        check_catboost_arm()
        check_no_row_count_injection()
        check_metrics()
        check_generators_are_pure()
        check_categorical_sources()
        check_pending_scenarios()
        check_correctness_arms()
        check_oracle_cells()
        check_accuracy_axes()
        check_stale_anchors()
        check_pair_plan()
        check_arm_keying()
        check_arm_keying_writer()
        check_coverage_guard()
        check_bayes_floor()
        check_citations()
        check_outputs()

    if FAILURES:
        for failure in FAILURES:
            print(f"FAIL {failure}")
        print(f"\n{len(FAILURES)} problems")
        # An import failure here is far more often the WRONG INTERPRETER than
        # a missing dependency, and the two look identical in the message.
        # This file is designed to run anywhere in under a second, which makes
        # `python3 bench/real_data/selfcheck.py` the natural thing to type --
        # and that interpreter lacks pandas, which the categorical fixture
        # needs. So the tool that is supposed to run anywhere fails in a way
        # that reads as "the repository is broken" rather than "you used the
        # wrong python". Two people lost time to it on 2026-08-16, one of them
        # reporting it upward as a pre-existing environment gap after checking
        # carefully that it was not their own change.
        #
        # The message that will be acted on should carry what it was computed
        # from, so say which interpreter this was.
        if any("No module named" in str(f) for f in FAILURES):
            print()
            print(f"  Interpreter: {sys.executable}")
            print(f"  Version:     {sys.version.split()[0]}")
            print(
                "  A missing module here is usually the wrong interpreter.\n"
                "  The bench environment is the one that has the dependencies:\n"
                "      pixi run -e bench python bench/real_data/selfcheck.py"
            )
        return 1
    print("harness self-check passed. Nothing was trained and nothing was downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
