# The measurement queue, written down so an interruption costs nothing

Everything below is pending. It exists because this session's measurements are
blocked on a precondition (M1, quiet box) rather than on a decision, and a
blocked queue that lives only in a conversation is a queue that gets lost.

Rules referred to by name live in `PROFILE_PROTOCOL.md`: **S1-S4** were
registered before Sweep II, **M0-M6** before Session III. Read M0 before
recording any result, because it is the rule that says what may be called a
number.

**Precondition on every line below.** No lane, build, agent, or compile running.
Record `uptime` and the top processes before the first pair and after the last.
This project has lost two numbers to ignoring it, one of them a histogram A/B
taken at 18.6 percent spread with four agents compiling. It is a precondition,
not a preference.

A runbook that executes this in order is kept outside the repo during the
session; the authoritative list is this file.

---

## Session II leftovers: the one decision Sweep II could not close

### S1. Does the tree-resident plane become the default GPU plane?

Three conditions, all required. State as of now:

| condition | state |
|---|---|
| trees node-identical to the host plane | **satisfied**, `tests/test_gpu_tree_resident.mojo`, no tolerance, six configurations, and it asserts on the plane's own trace that the gate opened |
| faster at 250,000 **and** 1,000,000 rows | 1M is **consistent, not resolved** (M0). **250k has never been measured at all.** |
| no regression at 50,000 | **never measured** |

So S1 is two shapes short of decidable, and it has been two shapes short since
the plane landed. Until both are taken the plane stays opt-in, which is the
correct default and also the reason nobody noticed the gap.

```
# 250k and 50k, alternating processes, >= 5 pairs each, ON against OFF
MOJOTREES_GPU_TREE_RESIDENT=1 pixi run -e bench bench-train-gpu 250000 50 reg 5 gpu-device
MOJOTREES_GPU_TREE_RESIDENT=0 pixi run -e bench bench-train-gpu 250000 50 reg 5 gpu-device
MOJOTREES_GPU_TREE_RESIDENT=1 pixi run -e bench bench-train-gpu 50000 50 reg 5 gpu-device
MOJOTREES_GPU_TREE_RESIDENT=0 pixi run -e bench bench-train-gpu 50000 50 reg 5 gpu-device
```

Note the shapes are not arbitrary. 50,000 is below the `M4_MIN_NORMALIZED_WORK`
gate, so it takes the **host scan** and is where we currently lose; 1,000,000 x
50 x 255 bins x 31 leaves is exactly 50,000,000, which is not less than the
gate, so it takes the device search. 250,000 is the shape that already
contradicted the pure fixed-cost story once and is worth its own row for that
reason alone.

### S2, S3, S4

No measurement pending. S2 (depthwise is a benchmark row and an opt-in, never
the parity default) is a standing decision. S3 and S4 are rules applied to
results rather than work items. S4 in particular applies to M2.1 below and is
restated there.

---

## Session III: this round's list, in the order it must run

Order matters: the main claim goes first, while the box is coldest.

### M2.1 The upload collapse, ON against OFF

**Blocked on merge.** Five lanes are out; two of them are this item.

Every copy is a host wait on Metal, in both directions, at a **measured** ~458
microseconds regardless of byte count. What the lanes are removing, per tree:

- five uploads in `DeviceTreeTables.begin_tree`, replaced by a device reset
  kernel, since every byte it writes is a constant or a function of three
  scalars (`lane/tables-reset-kernel`)
- six downloads in `download_desc_tables`, packed toward one; one is genuine
  and correct, six is not (`lane/tables-reset-kernel`)
- four uploads in `GpuSplitSearcher._copy_tables`, scoped to what actually
  changes; the float parameter block and monotone vector are fit-constant, the
  feature table and allow mask move only when `feature_fraction` is active
  (`lane/search-tables-per-fit`)
- two uploads in `GpuNativeObjectives.update_raw_ranges`, packed into one if a
  32-bit bitcast is expressible (`lane/raw-update-copies`)

**Registered prediction, per M3:** roughly fourteen fewer waits per tree, at 458
microseconds over 100 trees, is an **estimated** 0.64 seconds, taking a 3.17
second leaf-wise fit to roughly 2.5. Recorded before the data so the measurement
can refute it rather than be fitted to it.

**Registered falsification, per S4 and M3:** if the collapse lands and the fit
does not move by at least 0.3 seconds, then a copy does not cost 458
microseconds in this position, and the per-synchronization constant that three
independent routes agreed on is wrong somewhere. That outcome is more
informative than the win and must be written up at least as loudly.

Arm names to be filled in from the lane reports. Both lanes were required to
keep two arms live in one binary rather than behind an environment variable,
because this machine drifts two to three times between time windows and only
interleaved arms compare.

### M2.2 The resident plane, re-taken

**Not blocked. Runs first if M2.1 is still unmerged when the box goes quiet.**

Currently **consistent, not resolved**: three pairs, one inverted, ON arm
spanning 3.136 to 3.612 against an effect of 0.57. Six alternating pairs at
1,000,000 x 50, which is one more than M0 requires, because re-opening this a
third time costs more than the spare pair does.

```
MOJOTREES_GPU_TREE_RESIDENT=1 pixi run -e bench bench-train-gpu 1000000 50 reg 5 gpu-device
MOJOTREES_GPU_TREE_RESIDENT=0 pixi run -e bench bench-train-gpu 1000000 50 reg 5 gpu-device
```

### M2.3 The histogram row unroll

**Unblocked.** The arms are wired and compile; nothing has been run.

Must run interleaved in ONE process. That is the entire reason `set_row_unroll`
is a runtime argument rather than a comptime knob, and a two-build comparison on
this machine would be worthless. It cannot change a histogram: both arms visit
the same rows and add the same fixed-point integers, and integer addition is
associative, so this is a speed question only. A smoke run at 2,000 x 8 had the
two arms agree on the training loss to the last digit and on the tree count,
which is that claim holding rather than a timing.

Two benchmarks, and **run both**:

```
pixi run bench-train-gpu 1000000 50 reg 5 row-unroll-on,row-unroll-off
pixi run bench-hist 1000000 50 20
```

The first is end to end through `train_gpu`; the second is the isolated
histogram A/B in `bench_histogram.mojo` and prints `row_unroll_on_*` /
`row_unroll_off_*` in milliseconds. They answer different questions and the
end-to-end one is the one that decides anything: **an isolated histogram win is
a hypothesis about a fit, not a result about one.** The row-tile floor measured
well in isolation and was a 22 to 36 percent regression across a whole fit,
because the isolated shape did not carry the partial traffic a real round does.
If the two disagree here, that disagreement is the finding and neither number
supersedes the other.

`gpu-unroll,gpu-nounroll`, the spelling this file carried while the arms did not
exist, still parses and resolves to the same pair under the same two labels.

The only prior attempt ran at 18.6 percent spread with lanes compiling and is
discarded, not superseded.

### M2.4 LightGBM, interleaved, with its own repeat spread

**The most valuable item on this list.** LightGBM's repeat-to-repeat spread on
this machine has **never been measured**, so every margin either way has been
claimed against an unknown noise floor.

```
pixi run -e bench bench-train-gpu 1000000 50 reg 5 gpu-device,lightgbm
pixi run -e bench bench-train-gpu 250000 50 reg 5 gpu-device,lightgbm
```

**Registered in advance, per M4, because it can go against the result we want:**
if LightGBM's own spread is wider than the margin being claimed against it, then
no margin either way is a result. The 6.5 percent depthwise win reverts to
indistinguishable along with everything else, until repeated enough to separate.

**Per M5:** putting LightGBM inside the interleaved loop also removed a
serialize-and-reload round trip from inside its timed call, which makes LightGBM
*faster* and moves the comparison *against* us. That is the correct direction
for a comparator and is why it stays, but no figure taken before that commit may
sit in a table beside one taken after it. Every LightGBM number is re-taken.

### M2.5 The parallel grain at the host-scan shape

**Not blocked. Costs no code change.**

`histogram_gpu.histogram_from_host` converts `3 * n_features * n_bins` cells,
which at 50 features and 255 bins is **38,250 ops**, against a default grain of
65,536 and a **measured** M4 parallel crossover near 40,000. It sits on the
knife edge and therefore stays serial by a hair. Its own docstring says moving
the grain is a measured decision and declines to take it.

```
MOJOTREES_PARALLEL_MIN_OPS=32768 pixi run -e bench bench-train-gpu 50000 50 reg 5 gpu-device
pixi run -e bench bench-train-gpu 50000 50 reg 5 gpu-device
```

Only 50,000 and other sub-gate shapes can move here. At the 1,000,000-row
headline this function is never called at all, so nothing measured here may be
reported as a headline improvement.

---

## What may be called a win when this is done, per M6

Leaf-wise, at 1,000,000 x 50, resolved by M0, against a LightGBM arm measured in
the same process in the same window with its own spread reported.

Nothing else. Not depthwise, which grows a different tree and is an opt-in under
S2. Not a projection from a wait count, however well the wait count has behaved
so far.

## Known-open, not scheduled here

- The wait audit was scoped to `grow_tree_device_resident`, so waits outside
  that function were invisible to it. `update_raw_ranges` had two nobody had
  counted. The gradient upload is described as two drains per round, which would
  be another 200 per fit, and is unexamined.
- Speculative single-candidate prebuild, to give leaf-wise the level-shaped
  launches that are the other half of depthwise's win. A census found the greedy
  pick is the top runner-up in 100 percent of 4,030 decisions, so K=1 suffices.
- One plane with `grow_policy` as the pick rule, rather than two host loops
  sharing kernels and a split-policy gate that is a lever for one policy and a
  no-op for the other.
- `main` is unpromoted. All of this is on `perf-round-2`.
