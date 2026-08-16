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
and none of them runs here. This needs nothing but the standard library and
numpy, in any environment, in well under a second.
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
            # A not_reached key that names the scenarios which WOULD reach it
            # must not have any of them scheduled for this arm. This is the
            # coupling that stops `one_hot_max_size` staying unset the day
            # somebody turns a categorical scenario back on.
            for reachable in entry.get("required_when_scenarios", ()):
                live = (
                    scenarios.MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT.get(
                        reachable, "skipped"
                    )
                    is None
                )
                check(
                    not live,
                    f"{row['catboost']} is declared not_reached because no "
                    f"scenario this arm runs reaches it, and {reachable} now "
                    "runs it. Either set the matching key in "
                    "MOJOTREES_CATBOOST_MODE and move this row to matched, or "
                    "record why the two arms may differ on it",
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
    for engine in ("mojotrees", "lightgbm", "catboost"):
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
        check_pending_scenarios()
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
