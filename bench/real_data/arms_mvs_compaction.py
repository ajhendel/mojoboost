"""Does the device MVS draw's MISSING ROW COMPACTION change the model, at a
`min_data_in_leaf` where it can actually bind?

    python bench/real_data/run.py --arms arms_mvs_compaction --tag compaction

THIS FILE REPLACES A PROBE THAT WAS NULL BY CONSTRUCTION

Run 20260818T115655Z-mvsreach carried an arm `mojotrees_mvs_leaf20` that set
`min_data_in_leaf=20` and compared the switch off against on. It came back
BIT IDENTICAL to the pair that did not set the parameter at all, rmse
9.097826 and 9.107779 on both. The override was applied; the record shows it.
The parameter simply could not bind: 463,715 rows over `num_leaves=31` leaves
about 15,000 rows in an average leaf, and a floor of 20 never refuses
anything. So that arm measured nothing, and read carelessly it would have
looked like evidence that compaction does not matter. It is withdrawn and this
module is the redo.

THE MECHANISM, and what a binding floor has to do to expose it

The host MVS draw COMPACTS. `sampling.bootstrap_round` replaces the round's
row list with the kept rows, so a tree grown after a `subsample=0.8` draw sees
about 0.8 n rows and `min_data_in_leaf` counts those.

The device MVS draw does not compact. A dropped row keeps its slot at weight
zero, and `min_data_in_leaf` COUNTS ROWS AND NOT MASS, so it counts the
dropped rows too. `gpu_objectives_native.mojo` states the consequence and does
not price it: "at the shipped symmetric `min_data_in_leaf` of 1 that binds
only on an empty leaf. It is not inert in general."

So at a binding floor the two arms enforce DIFFERENT EFFECTIVE FLOORS. A leaf
holding `k` kept rows out of `k / 0.8` slots is legal on the device arm when
`k / 0.8 >= floor` and illegal on the host arm until `k >= floor`. The device
arm therefore permits leaves the host arm refuses, by a factor of about
`1 / subsample`, and the trees are structurally different rather than
numerically perturbed.

    WHAT THIS PREDICTS: at a binding floor the device arm grows DEEPER or
    WIDER trees than the host arm, and the two disagree by more than the
    0.109 percent that the same switch cost on this scenario at a
    non-binding floor. `model` size in the record is the structural readout
    and the metric is the consequence.

    WHAT REFUTES IT: the two arms agree to within the same margin they showed
    at a non-binding floor, in which case the row-versus-mass difference is
    real in the source and inert in practice at these shapes, and that is
    worth recording as a measured fact rather than a hope.

TWO BLOCKS, because "binding" can be reached from either side
-------------------------------------------------------------

BLOCK A raises the floor into the leaves: `num_leaves=31` as shipped, floor
20,000, which is above the roughly 15,000 an average leaf holds, so the floor
refuses splits across the whole tree.

BLOCK B shrinks the leaves into the floor: `num_leaves=1024`, which leaves
about 450 rows in an average leaf, with floor 400. This is the sharper of the
two because MANY leaves sit near the floor at once rather than the tree simply
being cut short, so the `1 / subsample` gap applies to many decisions instead
of a few.

Both blocks are leaf-wise, which is the shipped growth policy and the one the
switch was measured to cost the most accuracy on (0.109 percent). Both are
`dense_regression`, because this is a question about a row counting rule and
not about a task.
"""

SCENARIO = "dense_regression"

MVS_DEVICE_ENV = {"MOJOTREES_GPU_MVS_DEVICE": "1"}

#: The MVS rate. The effective-floor gap between the two arms is about
#: `1 / subsample`, so this value is what sets the size of the effect and it
#: matches every other measurement of this switch.
SUBSAMPLE = 0.8

#: BLOCK A: the shipped leaf count with a floor above the average leaf.
BLOCK_A = {"num_leaves": 31, "min_data_in_leaf": 20000}

#: BLOCK B: many small leaves with a floor just under the average leaf. The
#: sharper probe; see the module docstring.
BLOCK_B = {"num_leaves": 1024, "min_data_in_leaf": 400}

#: A control at a floor that CANNOT bind, in the same run, so the binding
#: blocks are read against a same-window measurement of the switch rather than
#: against a number from another afternoon.
BLOCK_CONTROL = {"num_leaves": 1024, "min_data_in_leaf": 1}


def arms(caps=None):
    out = []
    out.extend(_pair("floor_high", BLOCK_A))
    out.extend(_pair("leaves_many", BLOCK_B))
    out.extend(_pair("leaves_many_nofloor", BLOCK_CONTROL))
    return out


def _pair(name, block):
    params = {"bootstrap_type": "MVS", "subsample": SUBSAMPLE}
    params.update(block)
    base = {
        "id": "mojotrees_" + name,
        "scenario": SCENARIO,
        "engine": "mojotrees",
        "device": "gpu",
        "params": dict(params),
        "axis": "mvs_solve",
        "axis_value": "host",
    }
    switched = dict(base)
    switched["id"] = base["id"] + "_device"
    switched["axis_value"] = "device"
    switched["env"] = dict(MVS_DEVICE_ENV)
    return [base, switched]


def check(planned):
    """A plan that cannot be read correctly must not be run at all."""
    cells = [(a["scenario"], a["id"], a["device"]) for a in planned]
    if len(cells) != len(set(cells)):
        raise SystemExit("arms_mvs_compaction: two arms share a cell")

    controls = [a for a in planned if a["params"]["min_data_in_leaf"] <= 1]
    if not controls:
        raise SystemExit(
            "arms_mvs_compaction: no non-binding control in this plan. Without "
            "one, a difference at a binding floor cannot be told from the "
            "difference the switch already makes at any floor"
        )

    for arm in planned:
        params = arm["params"]
        if params.get("bootstrap_type") != "MVS":
            raise SystemExit(
                f"arms_mvs_compaction/{arm['id']}: no MVS, no draw, no "
                "compaction question"
            )
        if not 0.0 < float(params["subsample"]) < 1.0:
            raise SystemExit(
                f"arms_mvs_compaction/{arm['id']}: at subsample 1.0 nothing is "
                "dropped, so the compacted and uncompacted row sets are the "
                "same set and this probe is vacuous"
            )
        # The whole point is a floor that can refuse a split. Catch a repeat of
        # the mvsreach mistake in the PLAN rather than in the results.
        floor = int(params["min_data_in_leaf"])
        leaves = int(params["num_leaves"])
        if floor > 1:
            rows_per_leaf = 463715 / float(leaves)
            if floor < 0.5 * rows_per_leaf:
                raise SystemExit(
                    f"arms_mvs_compaction/{arm['id']}: min_data_in_leaf="
                    f"{floor} against about {rows_per_leaf:.0f} rows in an "
                    f"average leaf at num_leaves={leaves}. That floor cannot "
                    "bind, so this arm would repeat the null-by-construction "
                    "probe this module exists to replace"
                )

    switched = [a for a in planned if a.get("env")]
    for arm in switched:
        base_id = arm["id"][: -len("_device")]
        base = [a for a in planned if a["id"] == base_id]
        if len(base) != 1 or base[0]["params"] != arm["params"]:
            raise SystemExit(
                f"arms_mvs_compaction/{arm['id']}: needs exactly one baseline "
                "at identical parameters, or its difference names the "
                "parameters as well as the switch"
            )

    print("arms_mvs_compaction: floors checked against the average leaf")
    for arm in planned:
        if arm.get("env"):
            continue
        p = arm["params"]
        print(f"  {arm['id']:36s} num_leaves={p['num_leaves']:5d} "
              f"floor={p['min_data_in_leaf']:6d} "
              f"rows/leaf~{463715 / p['num_leaves']:.0f}")
