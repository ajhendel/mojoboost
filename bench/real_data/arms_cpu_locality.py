"""The three bit-identical CPU switches, measured at the shape that should
show them, beside the shape that should not.

    python bench/real_data/run.py --arms arms_cpu_locality --tag cpuloc

WHY THIS RUN EXISTS. On 2026-08-18 our CPU arm became the fastest thing we
ship on a real competitor benchmark: covtype in NVIDIA's gbm-bench put
mojotrees-cpu at 28.077 s against our own GPU's 40.894 s, and LightGBM at
9.024 s. LANE_RULES rule 11 retired the convention that the CPU path is an
oracle and is never optimized, on exactly that evidence. This is the first
run taken under the new rule.

WHAT IS UNDER TEST, and all three are BIT-IDENTICAL by construction.

  MOJOTREES_CPU_LAYOUT_BY_NODE=1
      A node the plan will not row-block reads its bin ids from the
      contiguous row-major view instead of from scattered feature-major
      columns. Measured 2026-08-16 at 1.32x on small nodes and 4.52x on tiny
      ones, with root/large/medium untouched. `tests/test_row_major_bins.mojo`
      and `tests/test_grow_bin_layout.mojo` assert the identity through a
      whole grown tree.

  MOJOTREES_CPU_FEATURE_GROUP=<cache width>
      `_schedule_group` demands `TASK_BALANCE_FACTOR * cores` = 20 dispatch
      units, which at a non-blocking node clamps the feature interleave from
      the cache-derived width down to `n_active // 20`. One rung narrower is
      twice the re-walks of the row-id list and the gathered derivatives. The
      pool it is manufacturing those units for has been measured at about 3.5
      wide and FLAT from 4 to 16 tasks (DECLINED_OPTIMIZATIONS F6), so the
      clamp is buying nothing with the locality it spends. An explicit
      request bypasses all three clamps, and the output is bit-identical at
      every setting.

WHY THE SHAPE MATTERS MORE THAN THE SWITCH. Both switches act on the SMALL
NODE class, and whether a fit has small nodes is a function of its leaf
budget, not of the switch. `plan_row_block_count` stops row-blocking below
`2 * row_block_min_rows(255)` = 8,160 rows. At 464,809 rows and a complete
depth-8 tree, that threshold falls between depth 5 and depth 6, so 192 of
every 255 splits land in the non-blocking classes where the per-slot rate is
6x the root's at one worker and 11.65x at auto. At the harness default of 31
leaves a fit barely enters that regime at all.

So GBM_BENCH_SHAPE is the measurement and DEFAULT_SHAPE is the control, and
the control is not a formality: if the switches move the default shape as
much as they move the deep one, then the mechanism is not the small-node
locality it is supposed to be and the explanation above is wrong.

WHY BOTH SCENARIOS. `multiclass` is real covertype, 54 columns of which 44
are one-hot indicators, and it is the shape we lose 4.5x on. Its per-tree
cost should be dominated by the small-node regime. `dense_regression` is
year, 90 dense continuous columns, where we lose only 1.19x. If a switch
helps covtype and not year, that is the locality story. If it helps both
equally, it is something else and the covtype diagnosis needs revisiting.

WHY LIGHTGBM IS IN EVERY CELL BLOCK. Not for the comparison, which we
already have. It is the drift canary: this box has moved 22 percent between
repeats in one sitting, and a comparator that moves with our arms says the
window moved, while one that sits still says we did.
"""

#: Real covertype and real year. See the module docstring for why both.
SCENARIOS = ("multiclass", "dense_regression")

#: The shape a third-party benchmark actually uses, and the shape this run
#: exists to measure. `2 ** 8 == 256`, so this is a complete depth-8 tree and
#: the leaf budget cannot bind.
GBM_BENCH_SHAPE = {"num_leaves": 256, "max_depth": 8}

#: The control. The harness default, where row blocking still reaches most
#: splits and neither switch should have much to act on.
DEFAULT_SHAPE = {"num_leaves": 31, "max_depth": -1}

#: The cache-derived interleave width the schedule clamp throws away, per
#: scenario, because it is a function of the active feature count and the
#: hessian arm. 54 columns with a varying hessian resolves to 4 and is
#: clamped to 2; 90 columns under a constant hessian resolves to 8 and is
#: clamped to 4. Passing the value explicitly bypasses the clamp; passing the
#: value the clamp would have produced would measure nothing.
CACHE_GROUP = {"multiclass": "4", "dense_regression": "8"}

LAYOUT_ENV = {"MOJOTREES_CPU_LAYOUT_BY_NODE": "1"}


def arms(caps=None):
    """Every cell, deep shape first so a truncated run still answers."""
    out = []
    for scenario in SCENARIOS:
        out.extend(_block(scenario, GBM_BENCH_SHAPE, "deep"))
    for scenario in SCENARIOS:
        out.extend(_block(scenario, DEFAULT_SHAPE, "default"))
    return out


def _block(scenario, shape, shape_tag):
    """One scenario at one shape: the canary, our baseline, three switches."""
    group = CACHE_GROUP[scenario]
    out = [
        # The drift canary. Same shape, so it moves with the window if the
        # window moves.
        {"id": f"lightgbm_{shape_tag}", "scenario": scenario,
         "engine": "lightgbm", "device": "cpu", "params": dict(shape)},
        # The baseline every switched cell below is a difference against.
        {"id": f"mojotrees_cpu_{shape_tag}", "scenario": scenario,
         "engine": "mojotrees", "device": "cpu", "params": dict(shape),
         "axis": "cpu_locality", "axis_value": "baseline"},
    ]
    for arm_id, env, value in (
        ("layout", dict(LAYOUT_ENV), "layout_by_node"),
        ("group", {"MOJOTREES_CPU_FEATURE_GROUP": group}, f"group{group}"),
        ("both", dict(LAYOUT_ENV, MOJOTREES_CPU_FEATURE_GROUP=group), "both"),
    ):
        out.append({
            "id": f"mojotrees_cpu_{shape_tag}_{arm_id}",
            "scenario": scenario, "engine": "mojotrees", "device": "cpu",
            "params": dict(shape), "axis": "cpu_locality",
            "axis_value": value, "env": env,
        })
    return out


def check(planned):
    """A plan that cannot be read correctly must not be run at all."""
    cells = [(a["scenario"], a["id"], a["device"]) for a in planned]
    if len(cells) != len(set(cells)):
        raise SystemExit(
            "arms_cpu_locality: two arms share a (scenario, id, device) cell"
        )
    for scenario in SCENARIOS:
        for shape_tag, shape in (("deep", GBM_BENCH_SHAPE),
                                 ("default", DEFAULT_SHAPE)):
            here = [a for a in planned
                    if a["scenario"] == scenario
                    and a["params"] == shape]
            switched = [a for a in here if a.get("env")]
            if len(switched) != 3:
                raise SystemExit(
                    f"arms_cpu_locality/{scenario}/{shape_tag}: expected three "
                    f"switched cells, found {len(switched)}"
                )
            base = [a for a in here
                    if a["engine"] == "mojotrees" and not a.get("env")]
            if not base:
                raise SystemExit(
                    f"arms_cpu_locality/{scenario}/{shape_tag}: the switched "
                    "cells have no baseline at their own shape in this run, so "
                    "their numbers would be facts about an afternoon rather "
                    "than differences"
                )
            if not any(a["engine"] == "lightgbm" for a in here):
                raise SystemExit(
                    f"arms_cpu_locality/{scenario}/{shape_tag}: no comparator "
                    "at this shape, so a shared move could not be told from a "
                    "window that drifted"
                )
    # The control has to be a genuinely different shape or it is not a control.
    if GBM_BENCH_SHAPE == DEFAULT_SHAPE:
        raise SystemExit(
            "arms_cpu_locality: the measurement shape and the control shape "
            "are the same, so the run cannot distinguish a small-node effect "
            "from a general one"
        )
