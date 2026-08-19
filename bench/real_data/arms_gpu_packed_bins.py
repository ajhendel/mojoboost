"""Does narrowing the bin cell pay on a low-cardinality matrix?

    python bench/real_data/run.py --arms arms_gpu_packed_bins --tag packedbins

WHY THIS RUN EXISTS. `GpuActiveRows.set_packed_bins` was built, documented,
and had NO CALLER anywhere in src/, tests/ or bench/. It was wired into
`train_gpu` on 2026-08-19 behind `MOJOTREES_GPU_PACKED_BINS`, and this is the
first time it can be measured end to end.

WHAT IT DOES. One bit width per feature instead of a byte for every feature,
so a two-bin column costs 1 bit and not 8. The bins are unchanged; only their
storage narrows.

WHY IT IS THE RIGHT ARM TO TRY NEXT. Two row-compaction measurements went
negative this week, 1.535x slower leaf-wise and 0.757x symmetric, and both
moved the BINNED MATRIX. CatBoost never moves it: `partitions.h` keeps leaf
membership positional and permutes only the index and the stat columns. We
already have that half, since `gpu_active_rows`' partition is stable with both
sides ascending. What we do not have is their compressed index, which packs by
cardinality at 32 binary features per UInt32. Against their 480 MB a tree we
read 3.9 GB. Compaction rearranges the cell; this narrows it, and only one of
the two has been tested.

WHY COVERTYPE AND ONLY COVERTYPE. `set_packed_bins` REFUSES an all-width-8
table, because at eight bits throughout the packed layout is the dense matrix
already on the device. Covertype is 54 columns of which 44 are binary
indicators, so it packs. `dense_regression` is 90 continuous columns using
most of their 255 bins, so every width is 8 and the arm cannot run there at
all -- it raises rather than losing. That refusal is not a gap in this run, it
is the layout stating its own domain, and it is what the reach check below
used.

REACH, VERIFIED BEFORE ANY TIMING. A digest cannot distinguish a working
change from a no-op, and this repository shipped four no-ops past digest
checks in one day. So reach was established by the refusal rather than by a
number:

    MOJOTREES_GPU_PACKED_BINS=1  bench-train-gpu 20000 20 reg 5
        -> RAISES "a packed bin layout at eight bits throughout is the
           feature-major matrix already on the device"
    (unset)                      bench-train-gpu 20000 20 reg 5
        -> trains, gpu_train_s median 0.771

On/off differ in behavior on a shape the layout refuses, so the switch is
reaching the code. That is a positive control on the ENV PATH, not on the
covertype pack, and the digest equality below is what covers the second.

THE PREDICTION, written before the run.

  P1  ACCURACY IS IDENTICAL TO EVERY DIGIT. Packing changes where a bin id is
      stored, not what it is, and the GPU histogram accumulates in fixed-point
      Int32 whose sum does not depend on order. A model that moves means the
      pack is lossy and the run is a correctness bug report, not a timing one.

  P2  covertype is FASTER, and if the mechanism is bytes read it should land
      near the ratio of packed bytes to dense bytes per row. 44 binary columns
      at 1 bit plus 10 columns at 8 bits is 44 + 80 = 124 bits = 15.5 bytes,
      against 54 dense bytes: 3.5x fewer bin bytes per row. The fit will not
      move 3.5x, because the bin read is one term; a win in the 1.1x to 1.5x
      range is the honest expectation and anything above that wants explaining
      before it is believed.

  P3  IF IT LOSES, the first thing to rule out is the unpack cost per visit,
      not the mechanism. `test_bytes_per_visit_at_the_four_headline_
      cardinalities` already prices that in isolation.

WHY LIGHTGBM IS IN THE BLOCK. The drift canary. This box has moved 2 to 3x
across time windows and 22 percent between repeats in one sitting. A
comparator that moves with our arms says the window moved; one that sits still
says we did.

SHAPE. The same depth-8, 256-leaf shape the row-compaction run used
(`GPU_ROW_COMPACTION_2026-08-19.md`), so this run can be read against it
directly rather than only against its own neighbors.
"""

#: Covertype only. See the module docstring: the layout refuses the dense
#: regression matrix by construction rather than merely losing on it.
SCENARIOS = ("multiclass",)

#: Matches GPU_ROW_COMPACTION_2026-08-19.md so the two runs are comparable.
GBM_BENCH_SHAPE = {"num_leaves": 256, "max_depth": 8}

PACKED_ENV = {"MOJOTREES_GPU_PACKED_BINS": "1"}


def arms(caps=None):
    out = []
    for scenario in SCENARIOS:
        out.extend(_block(scenario, GBM_BENCH_SHAPE, "deep"))
    return out


def _block(scenario, shape, shape_tag):
    """The canary, our baseline, and the switch. Interleaved by the runner."""
    return [
        {"id": f"lightgbm_{shape_tag}", "scenario": scenario,
         "engine": "lightgbm", "device": "cpu", "params": dict(shape)},
        {"id": f"mojotrees_gpu_{shape_tag}", "scenario": scenario,
         "engine": "mojotrees", "device": "gpu", "params": dict(shape),
         "axis": "packed_bins", "axis_value": "baseline"},
        {"id": f"mojotrees_gpu_{shape_tag}_packed", "scenario": scenario,
         "engine": "mojotrees", "device": "gpu", "params": dict(shape),
         "axis": "packed_bins", "axis_value": "packed",
         "env": dict(PACKED_ENV)},
    ]


def check(planned):
    """A plan that cannot be read correctly must not be run at all."""
    cells = [(a["scenario"], a["id"], a["device"]) for a in planned]
    if len(cells) != len(set(cells)):
        raise SystemExit(
            "arms_gpu_packed_bins: two arms share a (scenario, id, device) cell"
        )
    switched = [a for a in planned if a.get("env")]
    if not switched:
        raise SystemExit(
            "arms_gpu_packed_bins: no arm sets MOJOTREES_GPU_PACKED_BINS, so"
            " the run measures the baseline against itself"
        )
    for a in switched:
        twin = [b for b in planned
                if b["scenario"] == a["scenario"]
                and b["device"] == a["device"]
                and b["params"] == a["params"]
                and not b.get("env")]
        if not twin:
            raise SystemExit(
                "arms_gpu_packed_bins: the packed arm has no baseline twin at"
                " the same scenario, device and shape, so its number would be"
                " a fact about an afternoon rather than a difference"
            )
