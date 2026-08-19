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

## Open, measured, and NOT closed: exclusive feature bundling

Recorded here because it is the counterexample the rest of this file needs.
Every closed lane above was a memory-layout hypothesis. This one is a
difference in HOW MUCH WORK GETS DONE, and it is the only performance change
measured this week that went the right way.

LightGBM defaults `enable_bundle` **on**; we default it **off**. Covtype is
**44 binary columns of 54**, the exact shape bundling exists for.

| shape | result | accuracy |
|---|---|---|
| covtype, 581,012 x 54, 44 binary, leaf-wise CPU | **1.10x FASTER** (1.104 / 1.110 / 1.150 within-repeat, four windows) | **bit-identical** |
| year, 463,715 x 90, all continuous | **0.955x, 4.5 percent slower** (0.893 / 0.983 / 0.964) | **bit-identical** |

**Bit-identical in both directions**, because we accept only LightGBM's
default `max_conflict_rate = 0.0`, which makes bundling exactly lossless.

**Reach is proven, not assumed.** A third arm sets `min_reduction = 0.999`,
which builds the bundle plan and then declines to apply it. It measured
0.926 / 0.854 / 0.822 against the off arm, slower in every window: the win
tracks the plan being APPLIED, and building one you do not use costs 15
percent. That is also why the dense arm regresses.

**What is not yet resolved:** whether the dense regression is a fixed
plan-construction cost that amortizes away at realistic tree counts, or a
per-tree cost. The two imply different defaults and the measurement is cheap.

**Why the default cannot simply flip:** `efb.check_bundling_supported`
RAISES whenever the run is not CPU, so `enable_bundle = True` as an
unconditional default would start failing every `device="gpu"` fit. The
correct shape is a tri-state where an unset value resolves to on-where-
supported and an explicit `True` still raises loudly, so a caller who asked
for bundling never silently does not get it.

**And the structural note worth more than the switch:** LightGBM bundles at
Dataset construction regardless of device, so their GPU arm gets bundling and
ours structurally cannot. On covtype that is 54 columns against roughly 15,
on the arm that is our slowest at 42.2 s. Making the GPU histogram
bundle-aware is a project, not a switch, and it is not started.
