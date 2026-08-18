"""How far does `MOJOTREES_GPU_MVS_DEVICE` actually reach, and does it hold up
where the switch's ONE non-numeric difference can bite?

    python bench/real_data/run.py --arms arms_mvs_reach --tag mvsreach

WHY THIS EXISTS, and it is a correction to a run I already did.

The device MVS solve was measured at 2.23x on `dense_regression` through the
`mojotrees_catboost_mode` arm, which is the oblivious grower. That is one
growth policy, and the switch does not key on growth policy at all. It is read
in `train_gpu.device_gradients` and in `train_gpu._train_gpu_rounds`, neither
of which inspects `grow_policy`; the policy is dispatched later. So the switch
reaches EVERY GPU dense single-output fit with `bootstrap_type=MVS` and
`subsample < 1`, at any policy, and a default flip on one policy's evidence
would be a flip on evidence that does not cover the change.

WHY THE VETO SCENARIO IS REACHED THROUGH A DIFFERENT ARM THAN THE MEASUREMENT.
`docs/design/ACCURACY_BUDGET.md` section 14 pre-registers `imbalanced_binary`
and `multiclass` as the disqualifying scenarios. Run 20260818T... with tag
obl-veto established that the CatBoost-mode arm CANNOT run on
`imbalanced_binary` on either backend: that scenario carries a categorical
column, the device declines the arm's Cosine plus random_strength plus
symmetrictree beside one (BLOCK_SCORE_FUNCTION, BLOCK_RANDOM_STRENGTH,
BLOCK_GROW_POLICY) and the CPU grower raises "grow_policy=oblivious is
implemented for numerical thresholds only". So the veto is not unrun through
that arm, it is UNREACHABLE through it.

It is reachable through the LEAF-WISE arm, which runs on `imbalanced_binary`
today, because the switch keys on the bootstrap and not on the policy. That is
what these arms do, and it closes the veto and the blast radius with one
matrix instead of two.

`multiclass` stays unreachable and no arm here pretends otherwise.
`train_multiclass_gpu` constructs no bootstrap bundle and
`GpuMvsSampler.draw` raises for `n_classes != 1`, so the switch cannot change
a GPU multiclass fit. A scenario a change cannot reach cannot be regressed by
it, which makes that half of the veto vacuous rather than unmet. That belongs
in the budget document as a ruling and not in a run.

THE `min_data_in_leaf` BLOCK, which is the part of this run that is not a
speed measurement at all
------------------------------------------------------------------------

The device arm and the host arm differ in one way that is not floating point.
The host draw COMPACTS: `sampling.bootstrap_round` replaces the round's row
list with the kept rows. The device draw does not; a dropped row keeps its
slot at weight zero. `min_data_in_leaf` counts ROWS AND NOT MASS, so the two
arms can disagree about whether a leaf is legal, and
`gpu_objectives_native.mojo` says so in as many words: "at the shipped
symmetric `min_data_in_leaf` of 1 that binds only on an empty leaf. It is not
inert in general."

Every measurement of this switch so far ran at a `min_data_in_leaf` where the
difference cannot bite, so every measurement so far is silent about it, and no
run at the default will ever say anything. The last block below sets
`min_data_in_leaf=20`, which is a value a user would plausibly pass, and runs
the switch off and on beside each other. If the compaction difference matters,
this is where it shows, and if it does not move the metric here then the
statement "not inert in general" has at least one datum under it instead of
none.

**A null here does not prove the difference is harmless.** It prices one value
on one scenario. It is worth running because the alternative is zero values on
zero scenarios.
"""

#: The scenarios these arms run. `dense_regression` because it is where every
#: other measurement of this switch lives, `imbalanced_binary` because it is
#: the reachable half of the pre-registered veto.
SCENARIOS = ("dense_regression", "imbalanced_binary")

#: The switch under test, default off at this commit.
MVS_DEVICE_ENV = {"MOJOTREES_GPU_MVS_DEVICE": "1"}

#: CatBoost's MVS contract, which this library follows: `subsample` IS the MVS
#: rate under `bootstrap_type="MVS"` and row bagging is off there. 0.8 is the
#: rate the CatBoost-mode default set uses, so the two measurements are of the
#: same sampler at the same rate and differ only in the grower.
MVS_PARAMS = {"bootstrap_type": "MVS", "subsample": 0.8}

#: A non-default leaf floor, high enough that a leaf can actually be refused
#: for holding too few ROWS. See the module docstring.
LEAF_FLOOR = 20


def arms(caps=None):
    """Every cell, in build order. Each switched arm follows its own baseline."""
    out = []
    for scenario in SCENARIOS:
        # Blast radius: the two growers the oblivious measurement did not cover.
        for engine in ("mojotrees", "mojotrees_depthwise"):
            out.extend(_pair(scenario, engine, engine + "_mvs", dict(MVS_PARAMS)))
    # The compaction probe. dense_regression only: it is a question about a
    # row-count rule, not about a task, and running it twice would not sharpen
    # it. Leaf-wise only, for the same reason.
    params = dict(MVS_PARAMS)
    params["min_data_in_leaf"] = LEAF_FLOOR
    out.extend(_pair("dense_regression", "mojotrees", "mojotrees_mvs_leaf20", params))
    return out


def _pair(scenario, engine, arm_id, params):
    """A baseline cell and the same cell with the switch on, in that order.

    Both on the gpu, because the switch has no effect anywhere else, and
    adjacent in build order so the runner's round interleaving puts them at
    the same repeat indices. A number from this file is a DIFFERENCE between
    two cells of one run and never a fact about an afternoon.
    """
    base = {
        "id": arm_id,
        "scenario": scenario,
        "engine": engine,
        "device": "gpu",
        "params": dict(params),
        "axis": "mvs_solve",
        "axis_value": "host",
    }
    switched = dict(base)
    switched["id"] = arm_id + "_device"
    switched["axis_value"] = "device"
    switched["env"] = dict(MVS_DEVICE_ENV)
    return [base, switched]


def check(planned):
    """A plan that cannot be read correctly must not be run at all."""
    cells = [(a["scenario"], a["id"], a["device"]) for a in planned]
    if len(cells) != len(set(cells)):
        raise SystemExit("arms_mvs_reach: two arms share a (scenario, id, device) cell")

    switched = [a for a in planned if a.get("env")]
    if not switched:
        raise SystemExit("arms_mvs_reach: nothing here turns the switch on")
    for arm in switched:
        if arm["env"] != MVS_DEVICE_ENV:
            raise SystemExit(
                f"arms_mvs_reach/{arm['id']}: carries an environment override "
                "that is not the switch under test. One switch per run, or the "
                "difference between two cells stops naming one thing"
            )
        base_id = arm["id"][: -len("_device")]
        base = [a for a in planned
                if a["id"] == base_id
                and a["scenario"] == arm["scenario"]
                and a["device"] == arm["device"]]
        if len(base) != 1:
            raise SystemExit(
                f"arms_mvs_reach/{arm['id']}: expected exactly one baseline "
                f"`{base_id}` on the same scenario and device in this run, "
                f"found {len(base)}. Without it the switched cell is a "
                "measurement of an afternoon"
            )
        if base[0]["params"] != arm["params"]:
            raise SystemExit(
                f"arms_mvs_reach/{arm['id']}: the switched cell and its "
                "baseline do not carry identical parameters, so their "
                "difference would name the parameters as well as the switch"
            )

    for arm in planned:
        if arm["params"].get("bootstrap_type") != "MVS":
            raise SystemExit(
                f"arms_mvs_reach/{arm['id']}: the switch is read only on an "
                "MVS fit, so an arm here that does not set bootstrap_type=MVS "
                "would render two identical rows and read as a null result"
            )
        if not 0.0 < float(arm["params"]["subsample"]) < 1.0:
            raise SystemExit(
                f"arms_mvs_reach/{arm['id']}: `samples_rows()` is False at "
                "subsample 1.0 and no sampler is constructed at all"
            )
