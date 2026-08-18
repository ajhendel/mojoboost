"""Where does the oblivious tree's DEVICE time go? Two pre-registered
predictions, measured before anybody writes an optimization.

    python bench/real_data/run.py --arms arms_oblivious_attrib --tag attrib

WHY THIS RUNS BEFORE THE LANE AND NOT AFTER IT

Twelve consecutive optimizations of this plane have measured null, and every
one of them was reasoned statically from source and built before it was
priced. The one that worked, the device MVS solve, was the one that named its
mechanism first. So this module measures the ATTRIBUTION before the fix, and it
states what each arm should produce so the run can come back wrong.

The model under test: the level build reads the bin matrix through a SCATTERED
row index. `_batch_hist_atomic_subtract_kernel` reads
`bins[f * n_rows + r]` where `r` is an ORIGINAL row id, and a child at level L
holds rows spread with stride about 2^(L+1) across the whole extent. While that
stride is under a cache line, a child's read of one feature costs a full column
pass however few rows it owns. So a level of 2^L built children costs 2^L full
passes over the bin matrix rather than one pass over its own rows.

If that model is right, the tree's cost is driven by the NUMBER OF FULL PASSES,
which with sibling subtraction and skip-last-build on is
1 + 1 + 2 + 4 + 8 + 16 = 32 at depth 6.

THE TWO ARM BLOCKS
------------------

M1, the subtraction arm, re-anchors the attribution IN ONE TIME WINDOW.
Turning `MOJOTREES_GPU_OBLIVIOUS_SUBTRACT` off takes the pass count from 32 to
63, a ratio of 1.97. If the build is a fraction f of the tree then the measured
slowdown is 1 / ((1 - f) + f / 1.97). The recorded 1.58x to 1.78x range implies
f between 0.745 and 0.86, but those two readings were taken in different time
windows from the 24.6 ms figure, and this repository has measured the same
benchmark drifting two to three times across windows. This block puts the
ratio and the baseline in the same run.

    PREDICTION M1: the switched-off arm is 1.58x to 1.97x the baseline, and
    rmse is IDENTICAL, because sibling subtraction on fixed-point Int32
    accumulators is bit-identical by construction.

M2, the depth arms, is the falsifiable one and it is why this file exists.
Passes go 8, 16, 32 for depth 4, 5, 6. Fixed per-level costs go roughly with
depth, so 4, 5, 6.

    PREDICTION M2: after subtracting a fixed cost of about 5 ms per tree, the
    remainder roughly DOUBLES from depth 5 to depth 6 and quadruples from
    depth 4 to depth 6. Concretely, per-tree times near 10, 15, 24.6 ms.

    WHAT REFUTES IT: per-tree time roughly LINEAR in depth, near 17, 21,
    24.6 ms. That would mean the passes are not the driver, the estimate that
    the level build is three quarters of the tree is inflated, and the time is
    in the per-level fixed costs instead. In that case the compacted-read lane
    should NOT be opened.

A comparison of depth arms is a comparison of DIFFERENT MODELS, so their
accuracy columns are not each other's business and no anchor covers them. Read
the seconds. The rmse column here is a sanity check that a tree was grown, not
a quality judgment.

`catboost` is in the plan because the CatBoost-mode arms cannot be built
without CatBoost's resolved learning rate, which a CatBoost cell in the SAME
run writes into catboost_readback.json. It is not part of either prediction.
"""

SCENARIO = "dense_regression"

#: The escape hatch that turns sibling subtraction off. Default ON since
#: 2026-08-17, so the spelling is an equality against "0".
SUBTRACT_OFF_ENV = {"MOJOTREES_GPU_OBLIVIOUS_SUBTRACT": "0"}

#: The depths M2 sweeps. 6 is the shipped CatBoost default and the depth every
#: other measurement of this arm was taken at, so it is the anchor of the
#: sweep rather than one point in it.
DEPTHS = (4, 5, 6)

#: Passes over the bin matrix per tree at each depth, with sibling subtraction
#: and skip-last-build both on: root, then one built child per pair per level,
#: levels 0 through max_depth - 2.
def passes_at(depth):
    """The model's own prediction, as arithmetic rather than prose."""
    total = 1
    for level in range(depth - 1):
        total += 1 << level
    return total


def arms(caps=None):
    out = []
    out.append({"id": "catboost", "scenario": SCENARIO, "engine": "catboost",
                "device": "cpu"})

    # M1. The baseline is depth 6, which is also M2's anchor arm, so the two
    # blocks share one cell rather than measuring the same thing twice.
    out.append(_cell("mojotrees_catboost_mode_d6", {"max_depth": 6},
                     axis="depth", axis_value=6))
    out.append(_cell("mojotrees_catboost_mode_d6_nosubtract", {"max_depth": 6},
                     axis="subtract", axis_value="off", env=SUBTRACT_OFF_ENV))

    # M2. The other two depths, beside it.
    for depth in DEPTHS:
        if depth == 6:
            continue
        out.append(_cell(f"mojotrees_catboost_mode_d{depth}",
                         {"max_depth": depth}, axis="depth", axis_value=depth))
    return out


def _cell(arm_id, params, axis, axis_value, env=None):
    cell = {
        "id": arm_id,
        "scenario": SCENARIO,
        "engine": "mojotrees_catboost_mode",
        "device": "gpu",
        "params": dict(params),
        "axis": axis,
        "axis_value": axis_value,
    }
    if env:
        cell["env"] = dict(env)
    return cell


def check(planned):
    """A plan that cannot be read correctly must not be run at all."""
    cells = [(a["scenario"], a["id"], a["device"]) for a in planned]
    if len(cells) != len(set(cells)):
        raise SystemExit("arms_oblivious_attrib: two arms share a cell")
    if not any(a["engine"] == "catboost" for a in planned):
        raise SystemExit(
            "arms_oblivious_attrib: the CatBoost-mode arms cannot be built "
            "without a catboost cell in the same run to write "
            "catboost_readback.json"
        )

    ours = [a for a in planned if a["engine"] == "mojotrees_catboost_mode"]
    depths = sorted({a["params"]["max_depth"] for a in ours})
    if tuple(depths) != tuple(sorted(DEPTHS)):
        raise SystemExit(
            f"arms_oblivious_attrib: M2 needs depths {sorted(DEPTHS)} in one "
            f"run to read a slope; this plan has {depths}"
        )

    switched = [a for a in ours if a.get("env")]
    if len(switched) != 1:
        raise SystemExit(
            "arms_oblivious_attrib: M1 is one switched cell against one "
            f"baseline; {len(switched)} cells carry an override"
        )
    base = [a for a in ours
            if not a.get("env")
            and a["params"] == switched[0]["params"]
            and a["device"] == switched[0]["device"]]
    if len(base) != 1:
        raise SystemExit(
            "arms_oblivious_attrib: M1's switched cell has no unique baseline "
            "at identical parameters on its own device, so its ratio would "
            "name the parameters as well as the switch"
        )

    # The predictions, printed so that the plan states them before the run
    # produces anything to read them against.
    print("arms_oblivious_attrib predictions, registered before the run:")
    for depth in sorted(DEPTHS):
        print(f"  depth {depth}: {passes_at(depth)} passes over the bin matrix")
    print("  M1 subtract off: 1.58x to 1.97x slower, rmse IDENTICAL")
    print("  M2 pass model:   per-tree seconds scale with the pass counts above")
    print("  M2 refuted if:   per-tree seconds scale with depth instead")
