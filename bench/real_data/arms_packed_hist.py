"""The compact histogram accumulator, at the shape that should show it and
the shape that should not.

    python bench/real_data/run.py --arms arms_packed_hist --tag packed

WHY THIS RUN EXISTS. The 2026-08-18 phase profile put 62.0 percent of the
covertype CPU round in the histogram accumulate, with the per-slot rate
degrading 19.4x from root nodes to tiny ones. The arithmetic that followed
said the mechanism is footprint: at 24 bytes a cell, a rectangular
54 x 255 histogram is 13,770 cells and 322.7 KB against a 64 KB L1, while the
cells any row can actually reach number `sum_f feature_bins[f]`, about 2,400,
or 56.2 KB, which fits. That was arithmetic over a profile and not an A/B.
This is the A/B.

WHAT IS UNDER TEST.

  MOJOTREES_CPU_PACKED_HIST=1
      `build_histogram_subset_into_scratch` accumulates into a compact buffer
      where feature f's cells start at `bin_offset[f]` and number
      `feature_bins[f]`, then expands once per node into the rectangular
      `f * n_bins + b` every consumer reads. BIT-IDENTICAL BY CONSTRUCTION:
      the same Float64 additions in the same order at different addresses,
      and the expansion is a copy. Verified as a prediction digest before this
      run was planned, and verified to be REACHED by sabotaging the expander
      and watching the digest move only with the switch on.

THE PREDICTION, written before the run, because a result that fits any
outcome measures nothing.

Covertype should win and year should not, and the reason is the data rather
than the code. Covertype is 54 columns of which 44 are binary: packed they
sit four to a 64-byte cache line, rectangular they are 4,080 bytes apart, and
the whole working set crosses from above L1 to below it. Year is 90 dense
continuous columns that each use most of their 255 bins, so its packed
footprint is close to its rectangular one and the only thing the switch adds
is the expansion pass. **If year wins too, the mechanism is not footprint and
the explanation above is wrong.** A loss on year of roughly the expansion's
cost is the expected, healthy outcome.

WHY THE DEEP SHAPE. The accumulate is partitioned by FEATURE, so at a large
node the tasks each walk a slice that fits anyway and there is nothing to
recover. One task walks all 54 columns only at a node small enough not to be
split across the pool, and `plan_row_block_count` stops row-blocking below
8,160 rows, which at 464,809 rows and a complete depth-8 tree falls between
depth 5 and depth 6. So 192 of every 255 splits land in the classes this is
aimed at. At the 31-leaf default they barely appear, which is the control.

WHY LIGHTGBM IS IN EVERY CELL BLOCK. The drift canary. This box has moved 22
percent between repeats in one sitting, and a comparator that moves with our
arms says the window moved while one that sits still says we did.

ONE KNOWN COST NOT YET REMOVED. The compact buffer is allocated per node
rather than carried on the caller's scratch the way `pairs` is. If this wins
it should win by more once that allocation is hoisted; if it loses, the
allocation is the first thing to rule out before the mechanism is.
"""

#: Real covertype, the shape the profile was taken on, and real year as the
#: control the prediction above distinguishes.
SCENARIOS = ("multiclass", "dense_regression")

#: A complete depth-8 tree, so the leaf budget cannot bind and the small-node
#: classes are actually populated.
GBM_BENCH_SHAPE = {"num_leaves": 256, "max_depth": 8}

#: The control shape, where row blocking still reaches most splits.
DEFAULT_SHAPE = {"num_leaves": 31, "max_depth": -1}

PACKED_ENV = {"MOJOTREES_CPU_PACKED_HIST": "1"}

#: THE CONTROL THE FIRST RUN LACKED, added 2026-08-18 after
#: `20260818T185452Z-packed` came back a loss with different covertype
#: digests. Packing forces row blocking off, and `MOJOTREES_CPU_ROW_BLOCKS`
#: is documented as one that MOVES BITS, because a block count is a summation
#: order. So the packed arm was never packing-versus-baseline on a shape the
#: planner blocks; it was packing-and-no-blocking. This arm turns blocking off
#: and packs nothing, which splits the two: packed against it isolates the
#: packing and must be bit-identical, and it against the baseline prices the
#: blocking on its own.
NOBLOCK_ENV = {"MOJOTREES_CPU_ROW_BLOCKS": "1"}


def arms(caps=None):
    """Every cell, deep shape first so a truncated run still answers."""
    out = []
    for scenario in SCENARIOS:
        out.extend(_block(scenario, GBM_BENCH_SHAPE, "deep"))
    for scenario in SCENARIOS:
        out.extend(_block(scenario, DEFAULT_SHAPE, "default"))
    return out


def _block(scenario, shape, shape_tag):
    """One scenario at one shape: the canary, our baseline, the switch."""
    return [
        {"id": f"lightgbm_{shape_tag}", "scenario": scenario,
         "engine": "lightgbm", "device": "cpu", "params": dict(shape)},
        {"id": f"mojotrees_cpu_{shape_tag}", "scenario": scenario,
         "engine": "mojotrees", "device": "cpu", "params": dict(shape),
         "axis": "packed_hist", "axis_value": "baseline"},
        {"id": f"mojotrees_cpu_{shape_tag}_noblock", "scenario": scenario,
         "engine": "mojotrees", "device": "cpu", "params": dict(shape),
         "axis": "packed_hist", "axis_value": "noblock",
         "env": dict(NOBLOCK_ENV)},
        {"id": f"mojotrees_cpu_{shape_tag}_packed", "scenario": scenario,
         "engine": "mojotrees", "device": "cpu", "params": dict(shape),
         "axis": "packed_hist", "axis_value": "packed",
         "env": dict(PACKED_ENV)},
    ]


def check(planned):
    """A plan that cannot be read correctly must not be run at all."""
    cells = [(a["scenario"], a["id"], a["device"]) for a in planned]
    if len(cells) != len(set(cells)):
        raise SystemExit(
            "arms_packed_hist: two arms share a (scenario, id, device) cell"
        )
    for scenario in SCENARIOS:
        for shape_tag, shape in (("deep", GBM_BENCH_SHAPE),
                                 ("default", DEFAULT_SHAPE)):
            here = [a for a in planned
                    if a["scenario"] == scenario and a["params"] == shape]
            switched = [a for a in here if a.get("env")]
            if len(switched) != 2:
                raise SystemExit(
                    f"arms_packed_hist/{scenario}/{shape_tag}: expected two "
                    f"switched cells, found {len(switched)}. The no-blocking "
                    "control is not optional: without it a covertype number "
                    "prices packing and the loss of row blocking together, "
                    "which is what the first run did"
                )
            if not [a for a in here
                    if a["engine"] == "mojotrees" and not a.get("env")]:
                raise SystemExit(
                    f"arms_packed_hist/{scenario}/{shape_tag}: the switched "
                    "cell has no baseline at its own shape in this run, so its "
                    "number would be a fact about an afternoon rather than a "
                    "difference"
                )
            if not any(a["engine"] == "lightgbm" for a in here):
                raise SystemExit(
                    f"arms_packed_hist/{scenario}/{shape_tag}: no comparator "
                    "at this shape, so a shared move could not be told from a "
                    "window that drifted"
                )
    if GBM_BENCH_SHAPE == DEFAULT_SHAPE:
        raise SystemExit(
            "arms_packed_hist: the measurement shape and the control shape "
            "are the same, so the run cannot distinguish a small-node effect "
            "from a general one"
        )
