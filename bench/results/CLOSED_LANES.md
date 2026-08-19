# Closed lanes: measured, declined, do not rebuild

**Read this before proposing a performance change.** Everything below was
built or reached, measured against an interleaved control, and lost or tied.
Each line is a thing somebody could plausibly propose again tomorrow, which is
the only reason the file exists. A lane leaves this file only with a new
measurement that contradicts the one recorded here, not with an argument.

The rule this file enforces: **arithmetic over a profile ranks where to point
an experiment; it does not predict the experiment's result.** Every entry in
the "predicted a win" column below was a confident model.

## Memory layout and access density

| lane | result | evidence |
|---|---|---|
| Compact histogram addresses (footprint-by-address) | **1.006x, three interleaved runs.** Tried twice | the 22.3 ms / 38 percent prize rested on it and is withdrawn |
| Row-major bin layout | **1.15-1.35x SLOWER**, built and measured | fixes line utilization, forces every feature's histogram slice resident |
| 16-byte histogram cell | **run invalidated by its own pre-registered control** (multiclass moved 5.5 / 7.2 percent where the switch cannot fire) | not a null, an unmeasurable |
| GPU row compaction, per split | **1.535x SLOWER, ranges disjoint** | `GPU_ROW_COMPACTION_2026-08-19.md`, covtype, 3 interleaved repeats |
| CPU level-wise fold compaction, CatBoost cadence | **2.6x SLOWER, ranges disjoint** | covtype symmetric depth 8, 3 interleaved repeats, bit-identical |

**The density lane is closed.** Two independent implementations of the fix the
density diagnosis named, at the two different cadences (per split on GPU, per
level on CPU, which is CatBoost's own), both verified to execute by sabotage
or by a reach trace, both slower. The confound was checked and cleared: the
fold's loss of the row-major kernel for small nodes measures 0.923x and
OVERLAPS, so it explains none of the 2.6x.

What survives of the diagnosis is the observation that our kernel is fast per
update and the gather is where the time sits. What is refuted is that
compaction can collect it: the reorder costs more than the gather it removes,
at both cadences, at the shape it was designed for.

## GPU launch and scheduling

| lane | result |
|---|---|
| Removing 21.5 percent of launches | **1.004x** |
| Tile floor / raised occupancy | **22 percent SLOWER at 50 features**, reverted to opt-in |
| `SCAN_BLOCK=32` + global sort scratch | built, tested, **measured inside noise**, reverted |
| The "last waits" lane | **dropped before building**: `device_wait` 0 calls, 0 ns, and static attribution puts every synchronize in the fit under 3 percent |
| Unified memory: `host_direct`, `map_write` | `host_direct` WRONG, `map_write` slower; copy at 75-85 GB/s says transfer is not the cost |

## Policy

| lane | result |
|---|---|
| Multiclass GPU crossover rule | **withdrawn on evidence**: covtype GPU 40.894 s against CPU 28.077 s, 1.45x slower. While it stood, `device="auto"` returned the slower arm silently |

## Standing scoreboard the lanes are measured against

gbm-bench covtype, matched parameters, 581,012 x 54 over 7 classes, 100 trees:
LightGBM 8.96 s, XGBoost 12.15 s, CatBoost 17.99 s, **us CPU 25.40 s, us GPU
42.2 s** (the GPU number re-measured 2026-08-19; the box drifts, ratios inside
one run are what compare). Accuracy is tied with LightGBM, 0.88318 against
0.88490.
