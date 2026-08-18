"""The oblivious-tree question, as an arm list: what does the symmetric arm
cost, and does the device MVS solve pay for itself inside the accuracy anchor?

    python bench/real_data/run.py --arms arms_oblivious --tag obl

WHY THIS IS AN ARM LIST AND NOT TWO RUNS. `MOJOTREES_GPU_MVS_DEVICE` is a
per-cell environment variable, so the obvious way to measure it is to run the
matrix twice with the variable set the second time. That comparison would be
worthless here. This repository has measured the same benchmark drifting two
to three times across time windows, and the standing rule is that only
round-interleaved arms compare. So the switched cell is an ARM beside its own
baseline, in one run, at the same repeat indices, and the two numbers are a
difference rather than two facts about two afternoons.

The engine set is the one the 20260817T205811Z-e2e run used, unchanged, so
every row here can be read against that run as well as against its neighbors.
`catboost` is not optional in it: the CatBoost-mode arms cannot be built
without CatBoost's resolved learning rate, which a CatBoost cell in the SAME
run writes into catboost_readback.json.

WHY TWO SCENARIOS AND NOT THREE. `docs/design/ACCURACY_BUDGET.md` section 14
pre-registers `imbalanced_binary` and `multiclass` as the DISQUALIFYING
scenarios for a candidate that passes on `dense_regression`, so a measurement
on the dense arm alone does not clear that gate as written. This module runs
`imbalanced_binary` for that reason. It does NOT run `multiclass`, and the
reason is not a schedule. `scenarios.SCENARIOS["multiclass"]` declares no
`mojotrees_catboost_mode` arm at all, and independently
`GpuMvsSampler.draw` raises for `n_classes != 1` while `train_multiclass_gpu`
constructs no bootstrap bundle, so the switch under test CANNOT REACH a GPU
multiclass fit. Half of the pre-registered veto is therefore unsatisfiable
through this arm rather than unrun, which is a different thing and has to be
resolved in the budget document rather than in a run.
"""

#: The scenarios every arm below runs. `dense_regression` is the measurement
#: and `imbalanced_binary` is the pre-registered veto scenario; see the module
#: docstring for why `multiclass` is absent and why that absence is structural.
SCENARIOS = ("dense_regression", "imbalanced_binary")

#: The switch under test, default OFF at this commit. It moves the MVS keep
#: threshold solve onto the device, which is what lets a symmetric round stop
#: computing host Float64 derivatives every round. Measured at 2.07x on this
#: arm with an rmse delta of 0.0034 percent, which is 74x inside the veto, in
#: the lane that built it. This run is that measurement placed beside its
#: baseline in a harness rather than in a lane's own timing loop.
MVS_DEVICE_ENV = {"MOJOTREES_GPU_MVS_DEVICE": "1"}


def arms(caps=None):
    """Every cell of the run, in build order."""
    out = []
    for scenario in SCENARIOS:
        out.extend(_scenario_arms(scenario))
    return out


def _scenario_arms(scenario):
    """One scenario's cells, peers first."""
    out = []

    # The peers and the comparator. catboost first: the CatBoost-mode arms
    # below read its resolved rate.
    for engine in ("catboost", "lightgbm", "xgboost"):
        out.append({"id": engine, "scenario": scenario, "engine": engine,
                    "device": "cpu"})

    # Our three shipped arms, each on both backends. The cpu cells are marked
    # oracles by the runner, not here.
    for engine in ("mojotrees", "mojotrees_depthwise", "mojotrees_catboost_mode"):
        for device in ("cpu", "gpu"):
            out.append({"id": engine, "scenario": scenario, "engine": engine,
                        "device": device})

    # The cell this run exists for. Same engine, same parameters, one
    # environment variable, its own arm id so it cannot collide with the
    # baseline's record, and an axis so a reader can see what it moves.
    out.append({
        "id": "mojotrees_catboost_mode_mvs_device",
        "scenario": scenario,
        "engine": "mojotrees_catboost_mode",
        "device": "gpu",
        "axis": "mvs_solve",
        "axis_value": "device",
        "env": dict(MVS_DEVICE_ENV),
    })
    return out


def check(planned):
    """A plan that cannot be read correctly must not be run at all."""
    cells = [(a["scenario"], a["id"], a["device"]) for a in planned]
    if len(cells) != len(set(cells)):
        raise SystemExit(
            "arms_oblivious: two arms share a (scenario, id, device) cell"
        )
    for scenario in SCENARIOS:
        here = [a for a in planned if a["scenario"] == scenario]
        if not any(a["engine"] == "catboost" for a in here):
            raise SystemExit(
                f"arms_oblivious/{scenario}: the CatBoost-mode arms cannot be "
                "built without a catboost cell in the same run to write "
                "catboost_readback.json"
            )
        switched = [a for a in here if a.get("env")]
        if len(switched) != 1:
            raise SystemExit(
                f"arms_oblivious/{scenario}: exactly one arm is meant to carry "
                f"an environment override; {len(switched)} do"
            )
        base = [a for a in here
                if a["engine"] == "mojotrees_catboost_mode"
                and a["device"] == switched[0]["device"]
                and not a.get("env")]
        if not base:
            raise SystemExit(
                f"arms_oblivious/{scenario}: the switched arm has no baseline "
                "on its own device in this run, so its number would be a fact "
                "about an afternoon rather than a difference"
            )
