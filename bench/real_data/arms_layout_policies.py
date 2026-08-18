"""Does the row-major layout switch reach all three growers, or only one?

    python bench/real_data/run.py --arms arms_layout_policies --tag layoutpol

WHY THIS RUN EXISTS. `arms_cpu_locality` measured
`MOJOTREES_CPU_LAYOUT_BY_NODE=1` at 1.301x, 1.236x and 1.231x on real year
across two shapes, bit-identical. Every one of those cells used the
`mojotrees` engine, which resolves `grow_policy` to `lossguide`. So the
measurement covers LEAF-WISE and says nothing about the other two growers
this package ships.

THAT GAP IS NOT HYPOTHETICAL FOR THIS PARTICULAR SWITCH. `_env_layout_by_node`
did not reach the symmetric grower until 2026-08-17, and a measurement taken
before that recorded the switch as neutral. **Neutral is what a switch that
does not reach the code always measures**, which is why a null here has to be
read as "did not reach" until reach is established, not as "did not help".
A 1.30x that holds on all three growers and a 1.30x that holds on one are
different facts and they justify different defaults.

WHAT IS DELIBERATELY ABSENT, and it is a change of practice. There is no
LightGBM cell and no XGBoost cell. `arms_cpu_locality` carried a LightGBM cell
per block as a drift canary and it cost 20 percent of that run's wall clock
for information the three-repeat spread across our own interleaved arms
already provides. Competitor arms belong in a scoreboard run, not in a switch
measurement. See the two-tier rule in `bench/external/README.md`.

THERE IS A CATBOOST CELL, AND IT IS NOT A COMPETITOR ARM HERE. The symmetric
arm resolves its learning rate from CatBoost's, which cb-shipped no longer
pins, so a catboost cell has to run in the SAME run and write
`catboost_readback.json` or the symmetric cells are skipped entirely. The
first draft of this module banned every competitor engine and the harness
refused two of its six cells for exactly that reason, which is the harness
working. The cell is a build DEPENDENCY of an arm under test, not a
scoreboard row, and `check` below distinguishes the two rather than banning by
name.

WHY YEAR AND NOT COVERTYPE. The switch measured as a clear win on year and as
noise on covertype, swinging from 0.813x to 1.116x there. Adding a noisy
scenario to a reach question doubles the cost and adds nothing: if the switch
does not reach a grower, it will read null on the clean dataset too, and that
null is interpretable. Covertype's own problem is the rectangular histogram,
which is a different lane.

WHAT WOULD FALSIFY THE READING. If all three growers move together, the switch
is a property of the CPU histogram path and the default flips under LANE_RULES
rule 11, since it is bit-identical and therefore cannot change any user's
output. If leaf-wise moves and the others do not, the next question is reach
and not benefit, and the answer is a grep of the grower bodies rather than
another run.
"""

#: One scenario. Real year, 463,715 x 90 dense continuous columns, which is
#: where the switch measured cleanly.
SCENARIO = "dense_regression"

#: The three growers, by the engine name that selects each. `mojotrees`
#: resolves grow_policy to lossguide, so it IS the leaf-wise arm; it is named
#: here explicitly rather than left implicit, because the run it follows was
#: read as a symmetric run by someone reasonably assuming the default was
#: symmetric.
POLICIES = (
    ("leafwise", "mojotrees"),
    ("depthwise", "mojotrees_depthwise"),
    ("symmetric", "mojotrees_catboost_mode"),
)

#: The shape the third-party benchmark uses. Held fixed across all three
#: growers so the only thing varying is the grower.
SHAPE = {"num_leaves": 256, "max_depth": 8}

LAYOUT_ENV = {"MOJOTREES_CPU_LAYOUT_BY_NODE": "1"}


def arms(caps=None):
    """Baseline and switched cell for each grower, baselines first.

    The catboost cell is first because the symmetric arms read what it
    writes, and the runner builds in list order.
    """
    out = [{"id": "catboost_dependency", "scenario": SCENARIO,
            "engine": "catboost", "device": "cpu", "params": dict(SHAPE)}]
    for tag, engine in POLICIES:
        out.append({
            "id": f"{tag}_base", "scenario": SCENARIO, "engine": engine,
            "device": "cpu", "params": dict(SHAPE),
            "axis": "layout_reach", "axis_value": f"{tag}_off",
        })
        out.append({
            "id": f"{tag}_layout", "scenario": SCENARIO, "engine": engine,
            "device": "cpu", "params": dict(SHAPE),
            "axis": "layout_reach", "axis_value": f"{tag}_on",
            "env": dict(LAYOUT_ENV),
        })
    return out


def check(planned):
    """A plan that cannot be read correctly must not be run at all."""
    cells = [(a["scenario"], a["id"], a["device"]) for a in planned]
    if len(cells) != len(set(cells)):
        raise SystemExit(
            "arms_layout_policies: two arms share a (scenario, id, device) cell"
        )
    engines = {a["engine"] for a in planned if a["engine"] != "catboost"}
    if len(engines) != len(POLICIES):
        raise SystemExit(
            "arms_layout_policies: this run exists to compare growers, and "
            f"{len(engines)} distinct engines are planned for {len(POLICIES)} "
            "policies. A grower that dropped out would leave a gap that reads "
            "as an answer"
        )
    if not any(a["engine"] == "catboost" for a in planned):
        raise SystemExit(
            "arms_layout_policies: no catboost cell, so the symmetric arms "
            "cannot resolve their learning rate and would be SKIPPED. A run "
            "that silently drops one of the three growers answers the reach "
            "question with a gap"
        )
    for tag, engine in POLICIES:
        here = [a for a in planned if a["engine"] == engine]
        switched = [a for a in here if a.get("env")]
        base = [a for a in here if not a.get("env")]
        if len(switched) != 1 or len(base) != 1:
            raise SystemExit(
                f"arms_layout_policies/{tag}: needs exactly one switched cell "
                f"and one baseline; found {len(switched)} and {len(base)}"
            )
        if base[0]["params"] != switched[0]["params"]:
            raise SystemExit(
                f"arms_layout_policies/{tag}: the switched cell and its "
                "baseline are at different shapes, so their ratio would "
                "measure the shape and not the switch"
            )
    if any(a["engine"] in ("lightgbm", "xgboost") for a in planned):
        raise SystemExit(
            "arms_layout_policies: a scoreboard arm is planned. This is a "
            "switch measurement, and competitor arms belong in a scoreboard "
            "run; they cost wall clock here for information the interleaved "
            "spread already carries. catboost is exempt and is checked for "
            "above, because the symmetric arm cannot be BUILT without it"
        )
