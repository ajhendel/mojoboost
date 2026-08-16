# The measurement queue, written down so an interruption costs nothing

> **STATUS, 2026-08-16: the Session III queue has been drained. Everything in
> it ran.** This banner was added by the instruction audit, which found the file
> still opening with "Everything below is pending" after every item in it had
> been measured and written up in
> [`session3_2026-08-16/RESULTS.md`](session3_2026-08-16/RESULTS.md).
>
> | item | verdict | where |
> |---|---|---|
> | S1, resident plane at 250k and 50k | **RUN**, both resolved, S1 **closed** | `session3` "S1, closed" |
> | M2.1, the upload collapse | **RUN**, five pairs, **REFUTED** at 0.016s against a registered 0.64s | `session3` M2.1 |
> | M2.2, the resident plane re-taken | **RUN**, six pairs, **resolved** | `session3` M2.2 |
> | M2.3, the histogram row unroll | **RUN twice**: indistinguishable slow, **resolved 10.8%** fast | `session3` M2.3 |
> | M2.4, LightGBM interleaved, at 1,000,000 | **RUN**; its spread measured for the first time | `session3` M2.4 |
> | M2.4, the same at **250,000** | **NEVER RUN** — still open | — |
> | M2.5, the parallel grain | **RUN**, three pairs, **NULL** | `session3` M2.5 |
>
> The commands below are kept, because they are how each item gets re-taken and
> several deserve to be. Read each item's status line before running it. The
> full audit is [`INSTRUCTION_AUDIT.md`](INSTRUCTION_AUDIT.md).

It exists because this session's measurements were blocked on a precondition
(M1, quiet box) rather than on a decision, and a blocked queue that lives only
in a conversation is a queue that gets lost.

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

> **Annotation, 2026-08-16. Session III has since run, and the cost model this
> queue was written against was refuted.** Results are in
> `session3_2026-08-16/RESULTS.md`; the withdrawal is
> `docs/GPU_PORTABILITY.md` section 6.1.1.
>
> **Nothing below this line has been edited.** The predictions are left exactly
> as they were registered, because a registered prediction that was refuted is
> the evidence that the process worked, and rewriting it to look right in
> hindsight destroys the only thing registering it was for.
>
> What was refuted, in one line: the queue prices every `enqueue_copy` at the
> ~458 microsecond per-synchronization constant. **Measured**, removing thirteen
> copies per tree bought 0.016 seconds against a registered prediction of 0.64,
> a null under M0; removing about thirty *round trips* per tree bought 0.75
> seconds, resolved. The mechanism is untouched: a copy really does drain the
> queue on Metal. What is withdrawn is the price. Count round trips to predict
> time; count copies to predict portability risk and ordering hazards.
>
> Read every "wait" below as "drain" and every wait-count projection as void.


---

> **HOLD, 2026-08-16: do not measure or quote any binning comparison right now.**
>
> The CPU campaign flipped `bin_construct_sample_cnt` to 200,000 and
> `min_data_in_bin` to 3 to match LightGBM stock. But `bench/real_data/
> engines.py` (~441 and ~454) still injects `bin_construct_sample_cnt` from the
> training row count for **LightGBM**.
>
> **That inverts the pin rather than dropping it.** Previously both sides binned
> every row. Now we bin a 200,000-row subsample and the comparator is still
> forced to bin all of them, so **the comparator does strictly more work than we
> do** and any binning ratio taken in this window is wrong in our favour.
>
> This is the fourth comparator-configuration hazard in three days, and the first
> one caught by its own campaign before anything was measured rather than after.
> The CPU campaign owns the fix.
>
> Also on hold: **regenerating the GPU bin-pinning test expectations**
> (`test_gpu_fma_consistency`'s `record.bin == 3` is the real one;
> `test_gpu_kernel_family`'s capacity ladder is a kernel-family property and
> should be unaffected). Six binning tests are currently failing on `cpu-round-1`
> including the sparse-equals-dense contract, so the numbers will move again.
> Wait for the green SHA.

---

## Session II leftovers: the one decision Sweep II could not close

### S1. Does the tree-resident plane become the default GPU plane?

Three conditions, all required. State as of now:

**CLOSED 2026-08-16.** All three conditions hold. The table below is the state
as of the night before, kept for the record, with the outcome beside it.

| condition | state when queued | outcome |
|---|---|---|
| trees node-identical to the host plane | **satisfied**, `tests/test_gpu_tree_resident.mojo`, no tolerance, six configurations, and it asserts on the plane's own trace that the gate opened | unchanged |
| faster at 250,000 **and** 1,000,000 rows | 1M was **consistent, not resolved** (M0). 250k had never been measured at all. | **both resolved**: 44% at 250k, 24% at 1M in the fast regime |
| no regression at 50,000 | never measured | **2.2x faster**, resolved, the largest relative win of the three |

The gate was proved open rather than assumed, with
`MOJOTREES_GPU_TREE_RESIDENT_TRACE=1`. Details in
[`session3_2026-08-16/RESULTS.md`](session3_2026-08-16/RESULTS.md).

**One gap remains and it is a code gap, not a measurement gap.** Under S1 the
resident plane becomes the default GPU plane. It is still opt-in in the source:
`src/mojotrees/gpu_resident_round.mojo` gates it on
`getenv("MOJOTREES_GPU_TREE_RESIDENT") == "1"`. The decision was recorded and
has not been shipped.

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

> **Annotation, 2026-08-16. RAN, and REFUTED.** 1,000,000 x 50, alternating
> processes, five pairs, medians 2.489 ON against 2.505 OFF: **0.016 seconds**
> against the 0.64 registered below. Not resolved under M0; the gap sits inside
> the ON arm's own 0.039 spread. The registered falsification two paragraphs
> down fired exactly as written, which is the outcome it said was more
> informative than the win. Full table in `session3_2026-08-16/RESULTS.md`,
> withdrawal in `docs/GPU_PORTABILITY.md` section 6.1.1. Text below unedited.

**Blocked on merge.** Five lanes are out; two of them are this item.

Every copy is a host wait on Metal, in both directions, at a **measured** ~458
microseconds regardless of byte count. What the lanes are removing, per tree:

- **LANDED**, five uploads in `DeviceTreeTables.begin_tree`, replaced by
  `_reset_tables_kernel`, since every byte it writes is a constant or a
  function of three scalars (`lane/tables-reset-kernel`)
- **LANDED**, six downloads in `download_desc_tables` packed into one by
  `_pack_tables_kernel`; the float plane crosses as bits and is bitcast back,
  and the decode below the fetch is shared by both arms so the snapshot cannot
  tell which ran (`lane/tables-reset-kernel`)
- four uploads in `GpuSplitSearcher._copy_tables`, scoped to what actually
  changes; the float parameter block and monotone vector are fit-constant, the
  feature table and allow mask move only when `feature_fraction` is active
  (`lane/search-tables-per-fit`)
- **LANDED**, two uploads in `GpuNativeObjectives.update_raw_ranges` packed into
  one; the step went into the descriptor's existing padding word, so it costs
  zero extra bytes and drops a whole device plane from that arm
  (`lane/raw-update-copies`)
- **LANDED but NOT IN THIS A/B**, the two-copy gradient upload fused into one
  (`lane/gradient-upload-drains`). `grad_dev` and `hess_dev` are now
  `create_sub_buffer` windows onto one `2 * n_rows` allocation and no file
  outside `histogram_gpu.mojo` changed. **It has no environment override and
  `train_gpu` does not forward its setter, so it is ON in both arms below and
  its effect is not measured by them.** That is deliberate rather than an
  oversight: the lane corrected my estimate from 0.137 to **0.046 seconds**,
  about 1.4 percent, which is inside this machine's noise and could never be
  resolved on its own under M0. It is recorded as an unmeasured saving rather
  than pretended into a table.

**Registered prediction, per M3:** roughly fourteen fewer waits per tree, at 458
microseconds over 100 trees, is an **estimated** 0.64 seconds, taking a 3.17
second leaf-wise fit to roughly 2.5. Recorded before the data so the measurement
can refute it rather than be fitted to it.

**Registered falsification, per S4 and M3:** if the collapse lands and the fit
does not move by at least 0.3 seconds, then a copy does not cost 458
microseconds in this position, and the per-synchronization constant that three
independent routes agreed on is wrong somewhere. That outcome is more
informative than the win and must be written up at least as loudly.

**Arms, as landed.** `lane/tables-reset-kernel` exposes setters
(`set_reset_on_device`, `set_packed_download`, both defaulting to the device
form) **and** environment overrides `MOJOTREES_GPU_TABLE_RESET=0` /
`MOJOTREES_GPU_PACKED_DOWNLOAD=0`. It needed the environment variables because
`DeviceTreeTables` is constructed inside `GpuHistogramBuilder.open_resident_
tables`, in a file that lane was not allowed to touch, so nothing between a
bench and the object exposes the setter.

**Consequence, stated rather than glossed: this A/B is alternating processes,
not interleaved arms in one process.** That is weaker, and it is the same shape
as the resident-plane A/B in M2.2, which is precedent rather than justification.
It is accepted here because the alternative is threading two more arguments
through `train_gpu` and adding fields to a struct three other lanes are working
next to. If this measurement lands ambiguous, closing that gap is the first
thing to try, not more repeats.

```
MOJOTREES_GPU_TABLE_RESET=0 MOJOTREES_GPU_PACKED_DOWNLOAD=0 \
  pixi run -e bench bench-train-gpu 1000000 50 reg 5 gpu-device      # off arm
pixi run -e bench bench-train-gpu 1000000 50 reg 5 gpu-device        # on arm
```

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
- **`compatibility/api_snapshot.json`: regenerated, and the gate is green.**
  This bullet used to say the snapshot was stale and missing
  `MOJOTREES_CONST_HESSIAN`, `MOJOTREES_GPU_TREE_RESIDENT` and its trace
  variables, `MOJOTREES_GPU_TABLE_RESET` and `MOJOTREES_GPU_PACKED_DOWNLOAD`.
  All five are present. `7a73a64`, "Regenerate the API snapshot once, for all
  six lanes at once", did it exactly as this bullet asked — once, at merge, with
  `--write` — and `python3 tools/api_snapshot.py --check` returns `ok`.

  **The remaining question is the one this bullet raised second and nobody has
  answered:** whether these knobs get declared in
  `docs/COMPATIBILITY_POLICY.md`. `environment.declared` holds **7** names
  against `environment.undeclared`'s **61**, so the drift is now the rule rather
  than the exception, and it is still worth settling deliberately.

  Two things the audit found about that file, so the next reader does not
  over-trust it. `environment.observed` is a **string-literal scan, not a read
  scan** (`tools/api_snapshot.py:839`), so membership is not evidence a variable
  does anything — one of the 68, `MOJOTREES_STARTUP_REPORT_FD`, is read by
  nothing at all and says so at `src/mojotrees/initialization.mojo:114`. And
  `ENV_SCAN_DIRS` excludes `bench/`, so the best-evidenced knobs in the
  repository, the `MOJOTREES_UM_*` family behind
  `apple_m4_unified_memory_2026-08-15.md`, are not in it. See
  [`INSTRUCTION_AUDIT.md`](INSTRUCTION_AUDIT.md) section 9.
- `stage_frontier` still uploads three tables. Untouched on purpose: it stages an
  arbitrary test-built frontier that no scalar describes, so the reset-kernel
  trick does not apply to it. It is not on the per-tree path.

### The copies that are still there, found by lanes and deliberately not taken

> **Annotation, 2026-08-16.** This list is still worth having, and its reason
> has changed. It was written as a list of unclaimed waits; after M2.1 it is a
> list of unclaimed **hazards** and portability items. Every entry below that
> says "wait" means "drain", and none of them should be picked up expecting a
> clock to move. The one item here that would be a *round trip* is none of them.
> Pick these up to reduce the places a stale byte can hide and the work a
> non-Metal backend will need, not to make a fit faster. See
> `docs/GPU_PORTABILITY.md` section 6.1.1. Text below unedited.

Listed so the next round starts from a list rather than from another audit. None
of these is on the default 1M leaf-wise path, which is why none was taken.

- `histogram_gpu.stage_from_device` — two copies plus its own `synchronize`,
  once per round per class, hybrid leaf scheduling only. Both planes are now
  adjacent in this direction too, so it is **one line** from costing one wait
  instead of two. Left alone so a second unmeasured arm would not confound the
  first in the same commit.
- `histogram_gpu.set_features` — `map_to_host`, which copies in **both**
  directions, per feature-set change, so per tree under column subsampling. The
  only `map_to_host` left in that file, and exactly the "per-tree constant that
  should cross once" case the portability doc names.
- `gpu_objectives_native.update_raw` — still writes through a bidirectional
  `map_to_host` on the non-compacted arm, strictly worse than the staged copy
  the range arms use. Different arm, so documented rather than churned.
- `gpu_sparse.GpuSparseHistogramBuilder` — carries the identical unfused
  gradient upload, three waits per round. Same fix, different file.
- `gpu_gradient_stream.HostGradientStage` — same two-copy upload and a
  `synchronize`, and it has **no callers anywhere** in `src/`, `bench/` or
  `tests/`. Dead code duplicating `stage_gradients`. Deleting it is a
  correctness-neutral simplification, not a performance change.
- The session ledger charges `SLOT_GRAD` and `SLOT_HESS` separately. Bytes are
  still right; the allocation count those entries imply is now one high.

---

# The wave window, in order, when `cheap-sync` reports and the CPU wave merges

One window. Interleaved where the harness supports it, five repeats, **medians**
per the amended M0, and a `conditions` line on every row -- which now records the
environment-driven arms, after a `gpu-device` arm with
`MOJOTREES_GPU_TREE_RESIDENT=0` was found printing a line identical to one with
it on, across a 0.75-second difference.

1. **Canary calibration.** `pixi run bench-canary 7`. It refused itself once
   already at 10.9 percent spread against its own 3 percent bar, which is the
   instrument working; do not record a baseline it declines to give.
2. **GPU headline, 1M / 250k / 50k against stock+det.** This is where 2.58
   seconds at 1M becomes a number or does not.
3. **Depthwise probe arm.**
4. **K1's four arms, resident plane ON.** Only meaningful since
   `enqueue_desc_child` began passing the tiling requests -- before that the tile
   arms reached one node in sixty-one. Note K3's arm design assumed
   `feature_group = 1` is shipping; **it is 2 on Metal**, so arm A is not the
   shipping configuration.
5. **K2 speculation on/off**, at 50,000 and 1,000,000 separately. The registered
   condition binds: the launch-shape gain must beat the wasted work in a **whole
   fit**, not a phase.
6. **`cheap-sync`'s readback arms.**
7. **Queue depth on/off.**
8. **Atomic-fraction arms.**

## After the window, before any new lane opens

Report: **merged / moved / blocked**, one line each, and the one number that
matters -- **GPU against stock+det at three shapes, with accuracy beside it**.
Speed and accuracy are reported together against one comparator or not at all.

## Bucket line, required on every brief from here

Every lane brief states its bucket, and nothing outside these three launches:

- **A -- speed**
- **B -- accuracy**
- **C -- comparison validity**

The third exists because four of this week's most consequential findings were in
it: the throttled comparator, the `force_row_wise` pin, the inverted binning pin,
and the conditions line that could not distinguish two arms differing by 0.75
seconds.


---

# The headline is now END-TO-END, and that has a blocker

Settled 2026-08-16: every published number is **binning plus training**, against
LightGBM stock+det (their Dataset construction plus their train), speed and
accuracy side by side, thresholds relative to that comparator, conditions line on
every row.

**Two consequences, and the second one blocks the window.**

**1. Every number in `session3_2026-08-16/RESULTS.md` is training-only.** The 2.58
seconds at 1M, the 0.75-second resident-plane result, the 10.8 percent unroll --
all of them exclude binning on both sides. They are not wrong and they are not
comparable to an end-to-end figure. They keep their labels and the end-to-end
headline is a new measurement, not a re-reading of an old one.

**2. The end-to-end headline CANNOT be taken until the inverted binning pin is
fixed**, and that fix is in the CPU campaign's fallout lane.

`bench/real_data/engines.py` still injects `bin_construct_sample_cnt` from the
training row count **for LightGBM**, while our own default moved to LightGBM's
stock 200,000. So we bin a 200,000-row subsample and the comparator is forced to
bin every row: **it does strictly more binning work than we do.**

That was tolerable while binning was reported separately and held under an
explicit void. **It is not tolerable once binning is inside the headline**, where
it would inflate the number in our favour and be invisible. Our binning advantage
is the largest single ratio this project has measured -- 6.3x at 1M -- and under
an end-to-end headline it flows directly into the result.

**RESOLVED, same hour, and nothing in the sequencing changes.** The fix already
exists on `cpu-round-1` -- `f37354b` "Bin at LightGBM's stock defaults, not at
ours" and `3b26481` "One comparator, stock+det, recorded on every run it
produces". **Verified rather than taken on report:** both commits exist in this
object store, and `git show cpu-round-1:bench/real_data/engines.py` reads *"No
`bin_construct_sample_cnt` here"* where `perf-round-2` still has the injection at
lines 441 and 454.

So `perf-round-2` carries the defect only because the wave-end merge has not
happened, and that merge already precedes the window in the agreed order.
**The end-to-end headline is taken on the merged tree**, which is the same
sequencing as before -- the blocker was real on this branch and had already been
fixed on the other one.

Worth keeping as a coordination note rather than deleting: a defect can be live
on your branch and fixed on your peer's, and a report of either state is only
true of the branch it was made about. This project has already made the mirror
error in the other direction, when an orphan-module claim was accurate for
`perf-round-2` and wrong for `cpu-round-1`.

# Closed with evidence. Do not reopen without a new number.

- **`DeviceGraph` on Metal.** Verified by execution: raises at builder creation;
  `MetalDeviceGraphBuilder.cpp` absent from a driver set containing the CUDA and
  HIP equivalents.
- **Device row-major bins.** Reopens on exactly one condition: the
  atomic-fraction probe showing the gather under ~10 percent of the histogram
  phase.
- **K >= 2 speculation.** A theorem, not a measurement: the children a step
  creates have no records until that step's own search writes them, and the
  pre-existing leaves' ranking provably cannot move across a commit.
- **Feature-blocked layout at 256 bins.** Two lanes, independently.
- **CPU fallback for wide features.** It would merge per-node histogram cells
  from a Float64 backend and a fixed-point one, which is not a hybrid split but a
  histogram with no single definition.

# On fixed point, stated so it stops drifting

**Neither a proven advantage nor a proven handicap on this platform.** XGBoost's
int64 quantiser is precedent for **determinism**, not validation of cost. Cite it
as precedent only. The one thing measured here is that the power-of-two scale
made dequantization exact and *tightened* the overflow bound by 64 units.
