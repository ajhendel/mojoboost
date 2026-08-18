# Path coverage

Which optimization reaches which execution body, which do not, and whether the
absence is DELIBERATE or UNCARRIED.

Written 2026-08-18 by reading source at head. **Nothing here was built,
compiled, run or measured by this lane.** Every number quoted is quoted from a
results file, from a docstring that records a measurement, or from
`docs/design/DECLINED_OPTIMIZATIONS.md`, and its source is named in the cell.
Everything else is a citation to a symbol.

> **CORRECTED LATER ON 2026-08-18, and the correction is the most useful thing
> in the file.** This document was read and written in the morning. In the
> afternoon a lane finished the one half-built mechanism section 3A.1 named,
> measured it a LOSS on real data, and deleted it and everything it touched
> before the day was out. Section 3A's G-SYM compacted-bin cell, section 3A.1,
> the `MOJOTREES_GPU_ROW_COMPACTION` cell in section 4, incident 1 in section 7
> and ranked item 2 in section 8 all described the pre-removal world and are
> edited in place, each saying what it used to say. Section 1 gained the
> DELIBERATE strengths that let a measured loss be recorded as the closed cell
> it is. **The lesson is not that the table was wrong. It is that a coverage
> document ages in hours, so every cell names the symbol that settles it.**

**Line numbers are not used.** This was read in a shared checkout with several
lanes editing it, and two files moved under this read while it was open
(`gpu_split_search.mojo` shifted a switch reader by 27 lines mid-session). Every
citation names `file::symbol`. If a symbol is not where a reader expects it,
grep the name.

---

## 1. How to read this, and what the two verdicts mean

The defect this file exists to make visible has one shape. **A mechanism is
built and measured on one code path and never carried to its twin.** Nobody
notices, because each path still produces a correct tree. The path that missed
out is only slower, and slow is invisible until somebody profiles the exact
configuration it governs.

Six separate incidents of that shape are catalogued in section 7. They were
found one at a time, by six different people, over three days.

A cell in the tables below is one of four things.

**HAS.** The body executes the mechanism. A symbol is cited so a reader can
check it in one grep.

**DELIBERATE.** The body does not execute the mechanism and a reason is
recorded in source or in a design document. The reason is quoted or cited. A
DELIBERATE cell is a **closed** question. Nobody needs to spend an afternoon on
it, and re-opening it needs a new argument rather than a fresh reading.

DELIBERATE has three strengths and the cell says which one it is, because
"nobody carried it and here is why" and "somebody carried it and it lost" are
not the same finding even though both are closed.

- **DELIBERATE**, plain. A structural or stated reason, argued and never
  built. `EFB bundled histogram` on the GPU bodies is this, refused by name at
  `efb.check_bundling_honored`.
- **DELIBERATE, and priced.** Built, measured, and came in a null. The arm was
  then deleted and the cell records the ratio. The constant-hessian and
  pre-quantized cells on G-SYM are this, at 1.016x and 1.018x.
- **DELIBERATE, AND MEASURED NEGATIVE.** Built, measured, and came in a loss.
  The compacted bin read on G-SYM is this, at 0.757x, and 3A.1 sets it out.

**There is no separate "tried and lost" verdict and there should not be.** A
fifth value would put the strongest closed cells in the table in a column of
their own, next to the UNCARRIED ones, when what they are is the most closed
thing this document can hold. An UNCARRIED cell means nobody has decided. A
measured-negative cell means somebody built the mechanism, ran it, and the
number decided. Running those two together would be the same error as running
DELIBERATE and UNCARRIED together, one level in.

**UNCARRIED.** The body does not execute the mechanism and **no reason is
recorded anywhere this lane could find.** That is the output this file exists
to produce. It is not an accusation that the cell is wrong, it is a statement
that nobody has yet decided whether it is wrong.

**N/A.** The mechanism is not expressible on that body. A host grower has no
device permutation plane to read from, so "compacted bin read" is not a
question that can be asked of it.

The whole value of the document is the line between DELIBERATE and UNCARRIED.
"The oblivious walker does not do X because an oblivious tree has no ragged
nodes" is finished. "The oblivious walker does not do X" with nothing after it
is an open item worth somebody's afternoon. A table that runs the two together
is a list of things to feel bad about.

**A reason was never invented for a cell.** Where this lane could not find one,
the cell is UNCARRIED, and where the reason found is thin or implied rather
than stated at the site, the cell says so.

### What this file is not

Three reach documents already exist and this one does not repeat them.

- `docs/design/GROWTH_POLICY_REACH.md` is GPU switches against the three growth
  policies. It is the direct ancestor of this file and its findings are treated
  as prior art here, including two rows it left marked unverified, which
  section 3 settles.
- `docs/design/BOOSTING_MODE_REACH.md` is boosting modes against public
  surfaces.
- `docs/design/ROW_SAMPLING_REACH.md` is row samplers against policy and
  backend.
- `docs/design/SWITCH_GRID.md` is every `MOJOTREES_*` variable, what reads it,
  and what stands behind it.

All four ask whether a **feature** works. This one asks whether a **performance
mechanism** landed everywhere it could have, which is a different question with
a different failure mode. A feature that misses a path raises or returns a
wrong number. A mechanism that misses a path returns the right number slowly,
which no test can see.

---

## 2. The axes, derived from source

The axes were enumerated before any cell was filled, because a table with the
wrong axes cannot be repaired by adding rows. Seven were found. Four were named
in the brief that opened this lane and three were not.

| # | axis | values | where the fork is |
| --- | --- | --- | --- |
| 1 | growth policy | leaf-wise, depth-wise, symmetric | `growth_policy.check_grow_policy`. Leaf-wise and depth-wise share a frontier and `GrowthSchedule` orders it; symmetric leaves the loop entirely at `tree.grow_tree_leaves_profiled` and runs in `tree._grow_oblivious_levels` |
| 2 | backend | CPU, GPU | `backend.build_histogram_on`, and at scale `boosting` against `train_gpu` |
| 3 | data plane | dense, sparse | `tree_sparse.grow_tree_sparse` and `train_gpu_sparse.grow_tree_gpu_sparse` are separate growers with separate accumulators |
| 4 | output shape | single output, multiclass, custom objective | `boosting._boost_rounds` against `_boost_rounds_multiclass`; `train_gpu._train_gpu_rounds` against `_train_multiclass_gpu_rounds` and `_train_custom_gpu_rounds` |
| 5 | **control plane** | host-stepped, device-owned | `train_gpu._grow_tree_gpu_device_search` picks between `_device_search_incremental`, `_device_search_resident`, and the two device-owned growers in `gpu_resident_round.mojo`. **This axis is not the same as the backend axis and it carries more mechanisms than any other** |
| 6 | **train against score** | grower, walker | `predict.mojo` holds three batch walkers, `gpu_predict.GpuPredictor` a fourth, `tree_sparse.predict_row_sparse` a fifth. A mechanism that lands in a grower has no reason to reach a walker and vice versa, so they are tabulated apart |
| 7 | **single node against distributed** | single, data parallel, feature parallel, voting parallel | `distributed._grow_tree_data_parallel`, `_grow_tree_feature_parallel`, `_grow_tree_voting_parallel`. Three further growers, each with its own body |

Axes 5, 6 and 7 were the additions. Axis 5 matters most, because it is the one
that carries the switches, and it is invisible if a reader thinks of the
backend as the only device question.

Two candidate axes were considered and **rejected** as axes, because the source
shows they do not fork a body.

- **Row sampler** (bagging, GOSS, MVS). Rejected. `ROW_SAMPLING_REACH.md`
  establishes that on both backends the sample is drawn in the round loop
  above the grower and reaches it as one argument, so no grower reads a bag
  differently by policy. One row list, every body.
- **Boosting mode** (gbdt, dart, rf). Rejected as a grower axis, because
  `boosting_dart` and `boosting_rf` both call `tree.grow_tree` and therefore
  reach all three policies. It **is** a real axis at the trainer level, where
  the round loop lives, and it is tabulated there in section 4. That
  distinction is what produced ranked item 3.

---

## 3. Training mechanisms against grower bodies

Nine bodies. Each is the code that actually loops, not the entry point that
forwards to it.

| tag | body |
| --- | --- |
| **C-FRONT** | `tree.grow_tree_leaves_profiled`, CPU dense, leaf-wise and depth-wise |
| **C-SYM** | `tree._grow_oblivious_levels`, CPU dense symmetric |
| **C-SPARSE** | `tree_sparse.grow_tree_sparse`, CPU sparse. Leaf-wise and depth-wise only |
| **G-HOST** | `train_gpu.grow_tree_gpu_profiled`, the host-scan arm |
| **G-INCR** | `train_gpu._device_search_incremental` |
| **G-BATCH** | `train_gpu._device_search_resident` |
| **G-LEAF** | `gpu_resident_round.grow_tree_device_resident`, device-owned leaf-wise |
| **G-SYM** | `gpu_resident_round.grow_tree_device_oblivious`, device-owned symmetric |
| **G-SPARSE** | `train_gpu_sparse.grow_tree_gpu_sparse` |

`tools/check_path_coverage.py` enumerates these mechanically and fails when a
tenth appears. See section 6.

### 3A. Histogram accumulation

| mechanism | C-FRONT | C-SYM | C-SPARSE | G-HOST | G-INCR | G-BATCH | G-LEAF | G-SYM | G-SPARSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Sibling subtraction** | HAS, `tree.grow_tree_leaves_profiled` calls `histogram.subtract_histogram_into` | HAS, `tree._grow_oblivious_levels` calls the same | HAS, `tree_sparse.grow_tree_sparse` calls `histogram.subtract_histogram` | HAS, `train_gpu.grow_tree_gpu_profiled` calls `subtract_histogram` | **DELIBERATE.** `_grow_tree_gpu_device_search` states it, "keeps nothing, so a split builds both children". Residency is the precondition and this loop is the fallback for when the pool declines | HAS, `train_gpu._enqueue_resident_split` calls `histogram_gpu.enqueue_resident_leaf_subtracting` | HAS, unconditional, folded into `histogram_gpu.enqueue_desc_child` | HAS, `gpu_leaf_batching.oblivious_subtract_requested`, default ON since 2026-08-17, measured 1.78x | HAS, `train_gpu_sparse.grow_tree_gpu_sparse` calls `subtract_histogram` on the host |
| **In-place subtraction, no per-node allocation** | HAS, `subtract_histogram_into` writes into a reused buffer | HAS, same | **UNCARRIED.** Calls `subtract_histogram`, whose body is `Histogram.zeroed(...)` then `subtract_histogram_into`. One full histogram allocated per split | **UNCARRIED**, same wrapper | N/A | N/A, device slots | N/A | N/A | **UNCARRIED**, same wrapper |
| **Constant-hessian two-plane elision** | HAS, `const_hessian` argument threaded from `boosting.round_has_constant_hessian` | HAS, same argument, same grower call | **DELIBERATE.** `tree_sparse.grow_tree_sparse` states it, "the sparse accumulator has no three-plane elision to switch off", and carries the fields anyway so a future elision finds them in place | HAS, via `histogram_gpu.GpuHistogramBuilder.set_constant_hessian` forwarding to `GpuActiveRows` | HAS, same | HAS, same | HAS, same | **DELIBERATE, and priced.** The forward to the batcher was built behind `MOJOTREES_GPU_BATCH_CONST_HESS`, **measured 1.016x, a null**, and removed. `DECLINED_OPTIMIZATIONS.md` row E14. The kernel arm survives in `gpu_leaf_batching._batch_hist_atomic_kernel` and `GpuLeafBatcher.set_constant_hessian` has no caller, which its own docstring records | **UNCARRIED.** No `const_hessian` argument and no `set_constant_hessian` anywhere in `gpu_sparse.mojo`; `train_gpu_sparse` names `const_hessian_verify` only. The CPU sparse reason in the column three to the left may well transfer, and it is not written here |
| **Pre-quantized gradient pair** | **DELIBERATE, refused by name.** `tree_parameters_extra.check_quantized_grad` raises on `use_quantized_grad=true`, "no trainer is wired to the quantized histogram in this build, so setting it would train a float model that silently ignored it". The accumulator exists at `quantized_gradient.build_histogram_subset_quantized_into_scratch` | DELIBERATE, same refusal, it fires above the policy fork | DELIBERATE, same refusal | HAS, `GpuActiveRows._ensure_quantized`, default ON via `MOJOTREES_GPU_QUANTIZED_GRADS` | HAS, same | HAS, same | HAS, same | **DELIBERATE, and priced.** Built behind `MOJOTREES_GPU_BATCH_QUANT`, **measured 1.018x, a null**, removed with its kernel and its buffer. `DECLINED_OPTIMIZATIONS.md` row E10, which ends "Do not rebuild it without a new reason" | N/A, own compressed accumulator |
| **Compacted bin read, permutation-ordered matrix** | N/A | N/A | N/A | HAS, `enqueue_leaf` reaches `GpuActiveRows.enqueue_range_histogram` reaches `_ensure_compacted` | HAS, `_search_leaf_device` reaches the same chain | HAS, `enqueue_resident_leaf_subtracting` reaches the same chain | HAS, `enqueue_desc_child` reaches `GpuActiveRows.enqueue_desc_histogram` reaches `_ensure_compacted` | **DELIBERATE, AND MEASURED NEGATIVE.** Built as `MOJOTREES_GPU_OBLIVIOUS_COMPACT_BINS`, **measured 0.757x on the device-MVS arm and 0.949x on the host-MVS one**, bit-identical models on both pairs, and removed the same day with its kernel scalar, its entry-point parameter and the compact-plane ping-pong it needed. `DECLINED_OPTIMIZATIONS.md` row C1. The level build now indexes the dataset's own matrix by row id and has no alternative. See 3A.1 | N/A |
| **EFB bundled histogram** | HAS, `tree._expand_bundled` calls `efb.expand_bundled_histogram` | HAS, `_grow_oblivious_levels` takes the same `bundling: BundledMatrix` argument | HAS, `tree_sparse._node_histogram` calls the same | DELIBERATE, refused by name at `efb.check_bundling_honored`, called from `train_gpu` | DELIBERATE, same | DELIBERATE, same | DELIBERATE, same | DELIBERATE, same | DELIBERATE, refused by name at `train_gpu_sparse._refuse_bundling` |

#### 3A.1. The compacted bin read on the symmetric device plane

This cell is set out at length because it was the live instance of the whole
defect class, because `GROWTH_POLICY_REACH.md` left it as "SUSPECTED
ACCIDENTAL, not fully traced", and because it is now the only cell in this
document that was closed by a number rather than by an argument. That file said
so in three separate places and none of them was edited when this section first
claimed to have traced it; all three are corrected as of 2026-08-18, so the two
documents now agree at head.

**AN EARLIER VERSION OF THIS SECTION SAID THE ARM WAS HALF BUILT AT HEAD AND
ASKED SOMEBODY TO FINISH IT. That was true when this lane read the tree on
2026-08-18 and false by the evening of the same day.** It is recorded here
rather than quietly replaced, because a document that describes a repair as
unfinished after the repair was built, measured and deleted is worse than one
that never noticed. What the section used to hold was a list of the surviving
halves. It named `gpu_leaf_batching.oblivious_compacted_bins_requested` with no
call sites, a `compacted: Int32` launch scalar on `_batch_hist_atomic_kernel` and
its subtracting twin, and a `compacted: Bool = False` parameter on
`enqueue_device_plan_batch_fused` and `..._subtracting` that no caller passed.
**None of those symbols exists now.** Grepping any of them returns prose only.

**The reach was real.** `GpuActiveRows` maintains `cbins`, a
permutation-ordered copy of the bin matrix, and rebuilds it in
`_ensure_compacted`, which has exactly two call sites,
`GpuActiveRows.enqueue_desc_histogram` and
`GpuActiveRows.enqueue_range_histogram`. The symmetric device grower reaches
neither. `grow_tree_device_oblivious` calls
`histogram_gpu.GpuHistogramBuilder.enqueue_desc_level_children`, which calls
`gpu_leaf_batching.GpuLeafBatcher.enqueue_device_plan_batch_fused` or its
subtracting twin, and neither of those touches `GpuActiveRows` at all. That
trace is unchanged and still checks out at head.

**The missing consumer was then built, and it lost.** A lane wired the dispatch
half behind `MOJOTREES_GPU_OBLIVIOUS_COMPACT_BINS`, default off, pointing the
level build at the permutation-ordered plane. Real data,
`year_prediction_msd`, 463,715 x 90, 100 trees, symmetric depth 6, Apple M4,
both arms interleaved inside ONE run, three repeats.

| arm | train s | ms per tree | model sha256 | ratio |
|---|---|---|---|---|
| host MVS, compact off | 6.009 | 60.09 | `2504284d1efa` | baseline |
| host MVS, compact on | 6.329 | 63.29 | `2504284d1efa` | **0.949x** |
| device MVS, compact off | 2.476 | 24.76 | `7614c64f8ca3` | baseline |
| device MVS, compact on | 3.270 | 32.70 | `7614c64f8ca3` | **0.757x** |

**The models are bit-identical on both sides of each pair**, the same
`file_sha256` within each MVS configuration, which is what the arm's own gate
demanded. So the implementation was correct and it was simply slower, and it
plainly engaged, because an arm that did nothing would have read 1.00x rather
than 0.757x. The device-solve arm is the one the switch was aimed at, because
that is where the level build is about 85 percent of the tree, and it is the
arm that lost 24 percent. The lane's own registered refutation condition was
met by more than its worst case, because it priced the maintenance at about
5.8 ms per tree and the measured regression is 7.9 ms.

**What this closes, beyond the cell.** The argument for the arm was that a
strided row index makes a child's read of one feature cost a full column pass,
so the level build was throwing away most of the bandwidth it consumed. If
that had been true, removing the gather would have been a large win. It was a
loss, so **the gather is not what the level build spends its time on**, and any
future proposal resting on "the scattered bin read is the cost" is refuted in
advance. `DECLINED_OPTIMIZATIONS.md` row C1 carries the full record and reads
MEASURED NEGATIVE. Do not rebuild this arm on the old reasoning.

**The two qualifications this section used to raise were both right, and
neither is what killed it.** The switch's own docstring priced the arm at 1.86x
on the level build's **bin traffic** rather than on the fit, and said the sign
was the whole question; the sign came back negative. And `row_compaction_live()`
requires the quantized gradient arm, which is a measured decline on this exact
plane (E10), so wiring the switch was never going to be sufficient on its own.
Recording that both cautions were correct is the cheap half of the lesson. The
expensive half is that the estimate behind the arm was wrong in sign, not in
size.

**A latent bug the removal exposed, and it is now refused.** With the level
read gone, nothing on the symmetric plane consumes a compacted plane except the
root build, yet `MOJOTREES_GPU_ROW_COMPACTION=1` still armed the maintenance
there, because `GpuActiveRows.__init__` reads that variable for every fit. An
oblivious tree paid one rebuild plus two launches per level, 1 + 12 = 13
command buffers, while 62 of the 63 histograms it builds read the dataset's own
matrix. `gpu_resident_round.oblivious_schedule_launches(6, 64, True)` is 55,
and 55 + 13 is 68 command buffers a tree.

This paragraph read "68 against a Metal queue that is 64 deep on the measured
machine and **does not raise when it is overrun**. Since the failure mode is a
silent overrun rather than a slow fit", and **that clause is retired as of
2026-08-18**. A full Metal queue blocks the thread enqueueing into it rather than
dropping a buffer, so there is no silent overrun to be anyone's failure mode, and
the leaf-wise plane in the column beside this one runs 2,303 buffers a tree
measured backpressured and is the fastest arm here (`docs/GPU_PORTABILITY.md`
6.2, `docs/design/SWITCH_GRID.md` section 6 item 8).

**The refusal keeps its other two grounds and needs no third.** 13 of those
buffers are maintenance for a plane where 62 of 63 histograms read the raw
matrix, which is waste countable from the source and independent of any queue
depth; and the consumer that would have justified them was built and **measured
0.757x**, the loss this whole section is about. On those,
`histogram_gpu.GpuHistogramBuilder.stage_desc_level_plan` **raises** on
`self.rows.row_compaction_requested()`, once per tree, off a host field. The
switch does not reach this plane inertly. It errors.

### 3B. Split search and level scheduling

| mechanism | C-FRONT | C-SYM | C-SPARSE | G-HOST | G-INCR | G-BATCH | G-LEAF | G-SYM | G-SPARSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Device split search** | N/A | N/A | N/A | DELIBERATE, this body is the host-scan arm by definition, selected at `train_gpu.split_search_decision_for` | HAS | HAS | HAS | HAS | **DELIBERATE.** `train_gpu_sparse` module prose assigns split selection to the CPU so "CPU and GPU sparse fits choose from identical rules" |
| **Wide scan** | N/A | N/A | N/A | N/A | HAS, `gpu_split_search.wide_scan_for` reads `wide_scan_requested`, default ON, measured 1.21x | HAS, same searcher field | HAS, same | HAS, its own twin `oblivious_wide_scan_requested` read at `_launch_oblivious_search`, default ON, measured 4.5% | N/A |
| **Terminal-child build elision** | **UNCARRIED. See 3B.1** | **UNCARRIED. See 3B.1** | **UNCARRIED. See 3B.1** | UNCARRIED, same argument | UNCARRIED, same argument | HAS, `train_gpu.skip_terminal_children_enabled`, default off | **DELIBERATE.** `GROWTH_POLICY_REACH.md` states it. On the device-owned plane the host never learns a child's depth or row count, so the test cannot be made there | HAS, `gpu_resident_round.oblivious_skip_last_build_requested`, default ON, measured 1.26x | UNCARRIED, same argument |
| **K=1 speculative prebuild** | N/A | N/A | N/A | N/A | DELIBERATE, structural. `GROWTH_POLICY_REACH.md`, the batch already is the speculation | DELIBERATE, same | HAS, `gpu_resident_round.speculative_build_enabled` | DELIBERATE, refused by name (`OBLIVIOUS_SPECULATION`) at `oblivious_device_supported` | N/A |
| **Partition tail fusion** | N/A | N/A | N/A | N/A | DELIBERATE, `GpuActiveRows.partition` has no descriptor and no deferred copy-back | DELIBERATE, same | HAS, switched, `GpuActiveRows.set_partition_fusion` | HAS, unconditional. `grow_tree_device_oblivious` hard-wires `set_partition_fusion(True)` because the level build is the only thing that can pay the deferred copy-back | N/A |
| **Table upload hoisting** | N/A | N/A | N/A | N/A | HAS, `gpu_split_search.table_upload_hoisting_requested` read once at `GpuSplitSearcher.__init__`, so every searcher gets it | HAS, same | HAS, same | HAS, same | N/A |
| **Parallel noise staging** | N/A | N/A | N/A | N/A | HAS, `gpu_split_search.random_score_plane_into` reads `noise_stage_parallel_requested` | HAS, same | HAS, same | HAS, same | N/A |

#### 3B.1. The terminal-child histogram build on the host growers

`tree.grow_tree_leaves_profiled` builds one child's histogram with
`_hist_subset` and derives the sibling with `subtract_histogram_into`, then
calls `tree._search` on each. `_search` refuses at the depth limit with
"A leaf at the depth limit yields no split", and refuses again when
`n_rows < 2 * params.min_data_in_leaf`. **Both refusals are evaluated after the
histogram has already been built.**

Both of those predicates are known before the build. `depth` is the loop's own
counter and the child row counts come out of the partition. This is the same
elision that measured **1.26x** on the symmetric device plane
(`MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD`) and was then carried to
`_device_search_resident` as `MOJOTREES_GPU_SKIP_TERMINAL_CHILDREN` on
2026-08-17. It reached two GPU bodies out of nine and no host body at all.

No reason is recorded in `tree.mojo`, in `tree_sparse.mojo`, in
`DECLINED_OPTIMIZATIONS.md`, or in `SWITCH_GRID.md`. The cell is UNCARRIED.

### 3C. Bodies not tabulated above

The three distributed growers, `distributed._grow_tree_data_parallel`,
`_grow_tree_feature_parallel` and `_grow_tree_voting_parallel`, agree with each
other on every mechanism in this section. All three call
`histogram.subtract_histogram`, the allocating wrapper, so they take the
UNCARRIED cell on in-place subtraction. None of the device mechanisms is
expressible on them. They are named here rather than given three identical
columns.

---

## 4. Trainer-level mechanisms

Some mechanisms are set in the round loop rather than in the grower, so they
key on the output shape and the boosting mode instead of on the growth policy.
Five trainers, plus the two alternate boosting modes.

| tag | trainer |
| --- | --- |
| **CPU-1** | `boosting._boost_rounds` and `boosting.train_with_valid` |
| **CPU-K** | `boosting._boost_rounds_multiclass` and `train_multiclass_with_valid` |
| **CPU-SP** | `boosting_sparse.train_sparse` and its two siblings |
| **DART** | `boosting_dart` |
| **RF** | `boosting_rf` |
| **GPU-1** | `train_gpu._train_gpu_rounds`, reached from both `train_gpu` overloads |
| **GPU-V** | `train_gpu._train_gpu_valid_rounds`, the GPU early-stopping loop |
| **GPU-K** | `train_gpu._train_multiclass_gpu_rounds` |
| **GPU-C** | `train_gpu._train_custom_gpu_rounds` |

| mechanism | CPU-1 | CPU-K | CPU-SP | DART | RF | GPU-1 | GPU-V | GPU-K | GPU-C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Score update by leaf membership rather than tree traversal** | HAS, `boosting._leaf_score_update_enabled`, default ON | HAS, same, at `_boost_rounds_multiclass` and `train_multiclass_with_valid` | HAS, structurally and unconditionally. `tree_sparse.grow_tree_sparse` returns the row-to-leaf assignment, which `boosting_sparse._add_tree_scores` consumes | **UNCARRIED.** Adds `w * trees[s].predict_row(data, r)` per row per tree | **UNCARRIED.** Adds `trees[i].predict_row(data, row)` per row per tree | HAS, `histogram_gpu.GpuHistogramBuilder.update_raw_device` reaching `gpu_objectives_native.GpuObjectiveState.update_raw_ranges`, one launch over the leaf range table | **UNCARRIED. See 4.1** | HAS, `update_raw_device`, and also from `_train_multiclass_gpu_batched` | **UNCARRIED.** `raw[r] += params.learning_rate * tree.predict_row(data, r)` |
| **Constant-hessian declaration** | HAS, `boosting.round_has_constant_hessian` | **DELIBERATE**, implied rather than stated at the site. `round_has_constant_hessian` opens "Whether a **single-output** round", and a softmax hessian is not constant. The reason is sound and it is not written where the multiclass loop is | DELIBERATE, the grower has no elision to switch off (3A) | not tabulated, `grow_tree` is shared | not tabulated | HAS, `builder.set_constant_hessian` in both `train_gpu` overloads | HAS, `builder.set_constant_hessian` in `train_gpu_with_valid` | DELIBERATE, same single-output argument | **DELIBERATE, stated at the site.** "No constant-hessian declaration on a custom objective ... the hessians here are whatever `grad_hess` returns" |
| **Row compaction as an explicit parameter** | N/A | N/A | N/A | N/A | N/A | HAS, `row_compaction: Bool` argument on both `train_gpu` overloads | UNCARRIED as a parameter. The **switch** reaches it on the leaf-wise and depth-wise planes, because `GpuActiveRows.__init__` reads `MOJOTREES_GPU_ROW_COMPACTION` itself, so this is an API asymmetry and not a lost optimization. **Under `grow_policy=oblivious` neither route reaches it any more, because since 2026-08-18 `histogram_gpu.GpuHistogramBuilder.stage_desc_level_plan` raises on `self.rows.row_compaction_requested()`**, which is the field both the parameter and the variable set, so a symmetric fit that asks for compaction by either route errors rather than paying 13 launches a tree for one root build. See 3A.1 | UNCARRIED as a parameter, same | UNCARRIED as a parameter, same |
| **Class batching, one launch across several classes** | N/A | **UNCARRIED at the trainer.** `gpu_multiclass_batch` is the GPU-side answer and has no CPU twin | N/A | N/A | N/A | N/A | N/A | Present but off by default. `gpu_output_planes.plan_class_batches` defaults to the sequential path "one class at a time, exactly what the trainer does today, until a caller or `MOJOTREES_GPU_CLASS_BATCH` asks for more" | N/A |
| **Sibling subtraction inside the batched multiclass kernels** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **UNCARRIED.** No subtraction symbol appears anywhere in `gpu_multiclass_batch.mojo`. Every batched class builds both children | N/A |

### 4.1. The host traversal in the GPU early-stopping loop

`train_gpu._train_gpu_rounds` folds a tree into the raw scores with one device
launch over the leaf range table. `train_gpu._train_gpu_valid_rounds`, which is
the same fit with early stopping turned on, holds `raw` as a host
`List[Float64]` and folds the tree in with

    for r in range(n):
        raw[r] += params.learning_rate * tree.predict_row(data, r)

which is a dependent walk down the tree per row per round on the host, while the
device that just grew the tree is idle.

The loop's docstring gives a reason for the raw scores living on the host, which
is bit-parity with `boosting.train_with_valid`, and that reason is sound. **It is
not a reason for the traversal.** `train_with_valid` itself keeps host raw scores
and still updates by leaf, through `_leaf_score_update_enabled`, and the two arms
are documented there as leaving bit-identical raw scores. What blocks the same
move here is that `train_gpu.grow_tree_gpu` returns a `Tree` and nothing else,
where `tree.grow_tree_leaves` returns a `LeafMembership` beside it. That is a
signature gap, and it is not written down anywhere as a decision.

The same applies to GPU-C, `_train_custom_gpu_rounds`, for the same reason and
with the same one-line fold.

---

## 5. Scoring mechanisms against walker bodies

| tag | walker |
| --- | --- |
| **P-FLAT-BIN** | `tree.predict_row` through `model.predict_batch`, binned input |
| **P-FLAT-RAW** | `predict.predict_raw_batch`, unbinned column-major input |
| **P-SYM-BIN** | `predict.predict_oblivious_batch` |
| **P-SYM-RAW** | `predict.predict_oblivious_raw_batch` |
| **P-SPARSE** | `tree_sparse.predict_row_sparse` and `predict_row_sparse_csc` |
| **P-GPU** | `gpu_predict.GpuPredictor`, binned input, one thread per (row, output) |

| mechanism | P-FLAT-BIN | P-FLAT-RAW | P-SYM-BIN | P-SYM-RAW | P-SPARSE | P-GPU |
| --- | --- | --- | --- | --- | --- | --- |
| **Branchless NaN identity** (no `isnan`, three-way select on `v <= edge` / `v > edge` / neither) | N/A, a binned walker has no NaN, the bin was resolved at binning time | HAS, in `predict.predict_raw_batch`, both the blocked and the remainder loops, resting on `predict._raw_split` refusing a NaN edge | N/A, binned | HAS, in `predict.predict_oblivious_raw_batch`, folded into the SIMD compare with `nan_left` hoisted to the level | N/A, binned | N/A, binned |
| **Raw unbinned plan, no bin lookup at score time** | DELIBERATE, this is the fallback the raw plan replaces, kept as the arm `MOJOTREES_RAW_PREDICT=0` selects | HAS, `predict.raw_plan` | DELIBERATE, same fallback role | HAS, `predict.oblivious_raw_plan` | UNCARRIED. No raw plan exists for a CSR or CSC row | **UNCARRIED.** `GpuPredictor` takes a binned matrix. No reason recorded |
| **Column-tiled row-block nest** | N/A | HAS, switched, `predict.predict_tile_enabled` read in `apply_tiled` | N/A | HAS, unconditional, `_OBLIVIOUS_TILE` is a `comptime` width with no switch. **Asymmetric by construction and it is the faster arm**, so the asymmetry is in the safe direction | N/A | N/A, the kernel's grid is the tiling |
| **Symmetric specialization, level-major with no dependent load** | N/A | N/A | HAS | HAS. Measured **0.0215 s against 0.0735 s** for the depth-wise flat walker at identical tree work, recorded in `predict.predict_raw_batch`, run `20260818T113255Z-obl` | UNCARRIED, no symmetric arm | **UNCARRIED.** `gpu_predict.mojo` contains no occurrence of "oblivious" or "symmetric". Every tree is a pointer chase over the flat node table |

---

## 6. The checker

`tools/check_path_coverage.py`, stdlib only, no build, no `mojo`, sub-second.
Baseline at `tools/path_coverage_baseline.txt`, in the style
`tools/check_gpu_guards.py` established, where existing state is recorded and
only new state fails.

**A checker is possible for part of this and not for all of it, and the split
is worth being exact about**, because a checker that appears to cover more than
it does would recreate the defect it exists to end.

### What it checks

**R1, body inventory.** Every function in `src/mojotrees/` whose name matches a
grower or batch-scorer shape must be classified in the baseline as a BODY, which
owes this document a column, or a NOTBODY, which is a forwarder or an unrelated
name the patterns caught. A name in neither fails. Thirty-one candidates are
classified today, fifteen of them bodies. **This is the rule that fires when a
tenth grower arrives and nobody tells the table**, which is the upstream half of
every incident in section 7.

**R2, mechanism reach ratchet.** Twenty-eight mechanisms, each with an entry
symbol and the exact recorded set of `file::owner` sites that reference it. The
set is recomputed and any difference fails, in either direction. Growing means
the mechanism reached a new body and this document's verdict for that cell is
stale. Shrinking means a body lost it. The check does not and cannot decide
which, it forces a human to look.

Both rules were verified to fail correctly by perturbing the baseline and
observing the two expected failures, then restoring it.

### What it deliberately does not check

**It cannot tell DELIBERATE from UNCARRIED.** That is a judgement about whether
a reason exists and is good, and no regex reads a reason. The verdict column
here is human and stays human. The checker only detects that a cell moved.

**It cannot see an inline mechanism.** The branchless NaN identity is fourteen
lines of arithmetic inside a closure with no symbol to count. A regex for
`v <= edge` would pass the day somebody renames the variable and fail the day
somebody writes the same idiom for an unrelated reason. Those rows are marked
NOT MECHANICALLY CHECKABLE in section 5 and the script makes no claim about
them.

**A third rule was prototyped and rejected as a gate.** R3 would flag a
docstring asserting that a symbol has no callers when the symbol does have
callers, which is exactly the shape of incident 5 in section 7. It was built
and run. It found 17 non-reach claims in the tree, of which **11 are flagged as
contradicted and at least 6 of those 11 are wrong**, because the claim's subject
is usually a symbol other than the enclosing one (`tree_parameters_extra.check_quantized_grad`
carries a true claim about `build_histogram_subset_quantized_into_scratch`), and
because Mojo methods on different structs share names that text cannot separate
(`GpuLeafBatcher.set_constant_hessian` genuinely has no caller, and a text scan
counts nine, all of them `GpuActiveRows.set_constant_hessian` and
`GpuHistogramBuilder.set_constant_hessian`). Resolving either would need type
information the script does not have. It ships as `--claims`, prints its
precision problem in its own output, and **never fails the build.** A gate at
40 percent precision teaches people to ignore the gate.

### Running it

    python3 tools/check_path_coverage.py             # R1 + R2, gates
    python3 tools/check_path_coverage.py --list      # what it sees
    python3 tools/check_path_coverage.py --claims    # advisory only
    python3 tools/check_path_coverage.py --write-baseline

`--write-baseline` refuses while any body candidate is unclassified, so the
judgement cannot be regenerated away.

---

## 7. What this would have caught, honestly

The six incidents that opened this lane, against the table and the checker as
they stand. **Three of six for the table. Two of six for the checker. One of
six for neither.** Those numbers are the value proposition and inflating them
would destroy the point of building any of it.

| # | incident | table | checker | why |
| --- | --- | --- | --- | --- |
| 1 | `GpuActiveRows` compaction never read by the oblivious level build | **YES** | **YES** | It is a mechanism-against-body cell. Section 3A carries it. **Closed on 2026-08-18 by measurement.** The missing consumer was built, came in at 0.757x on the arm it was aimed at, and was deleted the same day, which is the outcome the table is for. Section 3A.1 sets it out. R2 records `_ensure_compacted` at two sites and would fire the moment a third appears |
| 2 | Batched family missing the pre-quantized gradient and the constant-hessian elision | **YES** | partly | Two cells in section 3A. Both are DELIBERATE today, priced at 1.018x and 1.016x nulls by E10 and E14, which is precisely the outcome the table is for. R2 tracks `_ensure_quantized` and `set_constant_hessian` |
| 3 | Oblivious predict walker had the branchless NaN identity, the flat walker kept the branch | **YES** | **NO** | A row in section 5, and it is the row marked NOT MECHANICALLY CHECKABLE. Inline arithmetic, no symbol. This is the honest limit of the mechanical half |
| 4 | `noise_stage_parallel_requested` parallelized the draw and left the serial allocation and copy at both call sites | **NO** | **NO** | Both call sites are on the **same** path. The table's granularity is "does body B execute mechanism M", the cell would read HAS, and it would be right. The defect is partial implementation **inside** one cell, which is a different class and needs a different instrument |
| 5 | `set_constant_hessian` gained three callers while its docstring said nothing called it | **NO** | **advisory only** | Docstring staleness, not path coverage. R3 is exactly this check, and section 6 explains why it cannot be trusted enough to gate. It does appear in `--claims` output today |
| 6 | Four `*_params` translators in `bench/` and `real_data/` applied the mode dict after the caller's overrides | **NO** | **NO** | Same class, "it existed in four places because nobody had a list", but the wrong domain. Not a performance mechanism and not in `src/`. A list-of-N-places checker for the harness would catch it and this is not that checker |

The pattern in the misses is worth stating. **The table catches "one body has
it and its twin does not". It does not catch "the mechanism is half-built
inside one body", and it does not catch anything outside `src/`.** Incidents 4
and 6 are both of shapes this artifact does not cover, and pretending otherwise
would be the same failure one level up.

---

## 8. UNCARRIED cells ranked

**Everything in this section is my estimate from reading source. Nothing here
was measured by this lane and no number below is a measurement of the item it
sits under.** Where a figure appears it is a measurement of a *different*
mechanism, quoted to say why the item is worth an afternoon, and it is labelled
as such.

Three items. The list is short on purpose. **Item 2 closed on 2026-08-18, by measurement and against its own estimate**, and is kept in place and annotated rather than removed.

### 1. Terminal-child histogram builds on the host growers

**Cells.** Section 3B.1. C-FRONT, C-SYM, C-SPARSE, G-HOST, G-INCR, G-SPARSE.

**The argument.** Both host growers build a child's histogram and then discover
in `tree._search` that the child cannot be split, on either of two predicates
that were both known before the build. On a complete symmetric tree the last
level is half of every node in the tree, so at the shipped default of
`max_depth=6` roughly half the deepest level's accumulation work is built and
discarded. On the leaf-wise grower the share is smaller, because `num_leaves`
usually binds before `max_depth`, and because the derived sibling of a terminal
pair costs a per-cell subtraction rather than a per-row build, so only the built
half of each pair is a full saving.

**Why I rank it first.** The identical elision measured **1.26x** on the
symmetric device plane, and that is a measurement of the GPU twin, not of this.
It was carried to a second GPU body the same day and to no host body. The CPU
symmetric grower is the slowest shipped plane in the project, 3.8x behind
CatBoost on the standing scorecard, and section 3B.1 puts a structurally large
share of its deepest level's work in the discard pile. The predicates are
already computed in the loop that would use them.

**What I am not claiming.** I have not measured it, the leaf-wise share is
genuinely smaller than the symmetric one, and the CPU path is documented
elsewhere in this project as an oracle rather than an optimization target, which
may be a sufficient reason to decline all three CPU cells. If it is, that reason
belongs in the cell, and today the cell is empty.

### 2. Finishing the compacted bin read on the symmetric device plane. CLOSED 2026-08-18, MEASURED NEGATIVE

**Kept in the list rather than deleted from it, because the ranking is the
record of how this was judged and a silently shortened list would not show that
the item resolved, or how badly the estimate behind it read.**

**Cell.** Section 3A.1. G-SYM.

**What this item asked for.** It argued that "somebody built most of a thing
and stopped", listed the kernel argument, the `dense` branch, the launch
scalar, the entry-point parameter and a switch reader with a 60-line docstring
as all existing at head, and asked for the one missing dispatch site. It ranked
the item second because the cost to close was small and the outcome genuinely
uncertain, and it noted that a half-built mechanism is strictly worse than
either finishing it or deleting it.

**What happened.** The dispatch site was written the same day. The arm ran on
the exact measurement the switch had committed itself to, one interleaved run
at 463,715 x 90. It measured **0.757x on the device-MVS arm and 0.949x on the
host-MVS one**, with bit-identical models on both sides of each pair. Every
symbol listed above was deleted with it. `DECLINED_OPTIMIZATIONS.md` C1 reads
MEASURED NEGATIVE and the numbers are in 3A.1.

**What the item got right, and what it got wrong.** Right, that a half-built
mechanism is the worst of the three states, and that the way out is a
measurement rather than an opinion. It cost one afternoon and it is finished
forever, which is what the ranking was for. Wrong, nothing in this item, nor in
either docstring it quoted, nor in C1's own "somewhere between zero and +5.7 s"
estimate, allowed for the sign being negative. The mechanism the whole estimate
rested on, that the level build is bandwidth-bound on a strided bin read, is
dead. **Do not re-rank this item on a fresh reading of the same source.** It
needs a new mechanism, not a new afternoon.

### 3. The score update by tree traversal on four trainers

**Cells.** Section 4 and 4.1. GPU-V, GPU-C, DART, RF.

**The argument.** Every other trainer in the package folds a tree into the raw
scores by leaf, in O(n_rows) with one indirect load and one read-modify-write per
row. `boosting._leaf_score_update_enabled` documents the traversal arm as having
"no workload on which the traversal is the better route". Four trainers still
walk the tree per row per round.

**Do GPU-V first, and it is the only one of the four I would put real weight
on.** `_train_gpu_valid_rounds` is the shipped GPU early-stopping loop, which is
a mainstream configuration rather than a corner. Its sibling
`_train_gpu_rounds`, the same fit without early stopping, already does the fold
in one device launch. So a GPU fit that turns on early stopping silently swaps a
device launch for a host walk over every row of every round, and nothing in the
source says that trade was chosen. The `_LEAF_ROW_OPS` and `_TRAVERSAL_ROW_OPS`
constants in `boosting.mojo` put the two shapes at 2 and 8 scheduling units per
row, and both are flagged **there** as estimates nobody has measured, so they
argue the direction and not the size.

**Why it is third and not higher.** The fix is not one line. It needs
`grow_tree_gpu` to return a leaf membership beside the tree, which the host
grower already does and the device grower does not, and on the device-owned
plane that membership currently lives in `GpuActiveRows.ranges` rather than in a
host list. The other three cells are weaker still. DART must re-score its
dropped set every round whatever happens to the newcomer, which caps the win at
a fraction of a round; RF and GPU-C are cleaner but neither is on any published
benchmark, so the win lands on a surface nobody currently measures.

### Found, and deliberately not ranked

Recorded so the next reader does not re-derive them, with why each stayed off
the list.

- **A symmetric specialization of `GpuPredictor`.** The host symmetric walker
  measured 3.4x over the flat one at identical tree work. The same argument
  should transfer to a kernel, where the dependent load chain is worse. Not
  ranked because it is a new kernel family rather than a carry, and this
  document is about mechanisms that already exist somewhere.
- **In-place sibling subtraction on the five bodies that use the allocating
  wrapper.** One `Histogram.zeroed` per split on C-SPARSE, G-HOST, G-SPARSE and
  the three distributed growers. Real, and small enough per split that it needs
  a measurement rather than an argument.
- **Sibling subtraction inside the batched multiclass kernels.** Every other
  histogram family in the project has it. Not ranked because the batched
  multiclass path is off by default and reaches no shipped configuration, so
  the cell should be closed by deciding the path's future rather than by
  carrying a mechanism onto it.
- **`histogram_gpu.GpuHistogramBuilder.enqueue_resident_subtract` has no
  callers.** Superseded by `enqueue_resident_leaf_subtracting`. Dead code, not a
  coverage gap, and named here only so the R2 entry for it is not mistaken for
  a reach.

---

## 9. Provenance

Read at head on 2026-08-18 in the shared checkout. No build, no test, no
benchmark, and no `mojo` invocation of any kind. The only thing executed was
`tools/check_path_coverage.py`, which is pure Python over the source text and
was written by this lane.

Every HAS cell names a symbol. Every DELIBERATE cell quotes or cites the reason.
Every UNCARRIED cell means this lane searched `src/`, `docs/design/` and
`bench/results/` for a reason and did not find one, which is a statement about
the search and not a proof that no reason exists.

Two cells rest on inference rather than on a quoted line, and both say so where
they sit. The multiclass constant-hessian cells in section 4 are DELIBERATE on
the strength of `round_has_constant_hessian` opening with the words
"single-output round", which is a sound reason that is not written at the
multiclass trainer. Section 3B.1's claim that both terminal predicates are
computable before the build is read off the loop structure rather than
demonstrated by a patch.
