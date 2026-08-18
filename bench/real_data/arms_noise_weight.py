"""Does fanning out the `random_strength` noise draw pay?

    python bench/real_data/run.py --arms arms_noise_weight --tag noiseweight

`random_score_plane` passed its raw DRAW COUNT to `parallel.plan_tasks`, which
takes an op count in histogram-op equivalents. A symmetric level is 100
features by 255 bins, so `cells` is 25,500 against a `PARALLEL_MIN_OPS` of
65,536, and `plan_tasks` therefore answers ONE. The default-on parallel arm has
never fanned out on a shipped symmetric fit.
`MOJOTREES_NOISE_DRAW_WEIGHT=1` prices a draw at 32 histogram-ops, which
clears the crossover.

    REGISTERED PREDICTION, from the lane that found it: the draws are 174 to
    313 ms of the 457 ms of host CPU remaining on the symmetric arm after the
    device MVS solve, and fanning them out saves 145 to 260 ms per fit against
    12 to 48 ms of pool-wake overhead. On a 2.464 s train that is 1.06x to
    1.12x.

    WHAT REFUTES IT: indistinguishable, or slower. Slower would mean 600 pool
    wakes cost more than the draws they parallelize, which at 20 to 80 us a
    wake is 12 to 48 ms and should not reach 145 ms.

    rmse MUST be identical. `plan_tasks` documents its answer as a scheduling
    decision only, and a plane entry is a pure function of seven arguments
    with no counter and no accumulator, so the worker count cannot reach it.
    A moved rmse is a defect, not a result.

BOTH ARMS CARRY THE DEVICE MVS SOLVE. That is deliberate. The noise draw is a
fixed number of milliseconds either way, so measuring it against the host-solve
baseline would divide it by a wall clock 2.4x larger and understate it. The
device-solve arm is also the configuration this change is for.
"""

SCENARIO = "dense_regression"
DEVICE_MVS = {"MOJOTREES_GPU_MVS_DEVICE": "1"}
WEIGHT = {"MOJOTREES_NOISE_DRAW_WEIGHT": "1"}


def arms(caps=None):
    return [
        {"id": "catboost", "scenario": SCENARIO, "engine": "catboost",
         "device": "cpu"},
        _cell("mojotrees_catboost_mode_devmvs", dict(DEVICE_MVS), "off"),
        _cell("mojotrees_catboost_mode_devmvs_noiseweight",
              {**DEVICE_MVS, **WEIGHT}, "on"),
    ]


def _cell(arm_id, env, value):
    return {
        "id": arm_id,
        "scenario": SCENARIO,
        "engine": "mojotrees_catboost_mode",
        "device": "gpu",
        "axis": "noise_draw_weight",
        "axis_value": value,
        "env": dict(env),
    }


def check(planned):
    ours = [a for a in planned if a["engine"] == "mojotrees_catboost_mode"]
    if len(ours) != 2:
        raise SystemExit("arms_noise_weight: expected exactly one pair")
    if not any(a["engine"] == "catboost" for a in planned):
        raise SystemExit(
            "arms_noise_weight: the CatBoost-mode arms cannot be built without "
            "a catboost cell in the same run to write catboost_readback.json"
        )
    a, b = ours
    if set(b["env"]) - set(a["env"]) != {"MOJOTREES_NOISE_DRAW_WEIGHT"}:
        raise SystemExit(
            "arms_noise_weight: the two cells differ in more than the switch "
            "under test, so their ratio would not name one thing"
        )
    for arm in ours:
        if arm["env"].get("MOJOTREES_GPU_MVS_DEVICE") != "1":
            raise SystemExit(
                "arms_noise_weight: both arms must carry the device MVS solve; "
                "see the module docstring for why the host-solve baseline "
                "would understate this change"
            )
