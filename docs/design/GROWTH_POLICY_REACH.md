# Which optimization reaches which growth policy

Written 2026-08-17 by reading the source at head. Nothing here was measured on
a run; every row names the line that decides it, so any one row can be checked
without re-deriving the table.

> **CORRECTED 2026-08-18, and the way this file went stale is worth more than
> the correction.** The `MOJOTREES_GPU_ROW_COMPACTION` row, item 3 of "Edits
> owed by other lanes" and item 6 of "What could not be verified without a
> compiler" all left the same question open in three places. It was answered
> the next day, in `docs/design/PATH_COVERAGE.md`, whose section 3A.1 opens by
> saying that THIS file "left it as SUSPECTED ACCIDENTAL, not fully traced. It
> is now traced" -- and then nobody came back and edited this file. So for a
> day the repository held a document that recorded a question as closed and a
> second document that still asked it, and a reader arriving at this one got
> the stale answer. **A trace written in a new file does not close an item in
> an old one. Close it where it was opened, in the same session, or it stays
> open to everyone who reads the old file first.** All three places are now
> edited in place and each quotes what it used to say.

The question this file answers was asked as "if a switch is good for one,
shouldn't it be good for all". The answer twice that day was yes, and both
times the switch had been sitting off on a plane nobody had checked it against.
This is the sweep for the rest of them.

## The three growers, and why a fix lands in one of them

There is no shared GPU grower. `train_gpu._grow_tree_gpu_device_search` routes
a tree to one of three loops, and the three do not share a body:

| policy | grower | file | round trips per tree |
| --- | --- | --- | --- |
| leaf-wise (lossguide) | `grow_tree_device_resident` | `gpu_resident_round.mojo` | 1 in the grower, 2 counting the round's own |
| depth-wise | `_device_search_resident` | `train_gpu.mojo` | 1 per batch, so ~7 at 31 leaves |
| symmetric (oblivious) | `grow_tree_device_oblivious` | `gpu_resident_round.mojo` | 1 |

The dispatch, in order, at `_grow_tree_gpu_device_search`:

1. `params.grow_policy == GROW_OBLIVIOUS` goes to `grow_tree_device_oblivious`
   unconditionally. It never consults `resident_round_enabled()` and it has no
   fallback: an unsupported symmetric configuration raises rather than
   degrading.
2. Otherwise, if the slot pool opens and `resident_round_enabled()`, then
   `resident_round_supported(...)` is asked. That calls
   `gpu_tree_tables.tree_resident_supported`, whose fourth test is

   ```
   if params.grow_policy != GROW_LEAFWISE:
       return TREE_RESIDENT_DEPTHWISE
   ```

   so depth-wise never reaches the device-owned plane.
3. Everything that got this far goes to `_device_search_resident`, and if the
   pool declined, to `_device_search_incremental`.

**That third step is the structural cause of the whole table below.**
Depth-wise has exactly one GPU grower and it is the host-stepped one, so every
optimization that lives inside `gpu_resident_round.grow_tree_device_resident`
is invisible to it, permanently, until the restriction moves.

## The matrix

REACHES means the grower executes the optimization. STRUCTURAL means it cannot
and the code says why. ACCIDENTAL means nothing prevents it and nobody ported
it, which is a defect.

| switch or optimization | read at | leaf-wise | depth-wise | symmetric |
| --- | --- | --- | --- | --- |
| `MOJOTREES_GPU_TREE_RESIDENT` | `gpu_resident_round.mojo:724` | REACHES (gates the plane) | STRUCTURAL, gated out one layer up by `tree_resident_supported` | STRUCTURAL, no second symmetric grower to fall back to |
| `MOJOTREES_GPU_SPLIT_RESIDENT` | `train_gpu.mojo:1593` | REACHES | REACHES | **DEFECT, now named.** It closed the pool and the symmetric route then raised `RESIDENT_NO_POOL`. The refusal now names the variable |
| `MOJOTREES_GPU_SPECULATION` (K=1 prebuild) | `gpu_resident_round.mojo:888` | REACHES | STRUCTURAL: under depth-wise the whole level is planned before any of it runs, so there is nothing to speculate about; the batch already is the speculation, resolved | STRUCTURAL, refused by name at `grow_tree_device_oblivious` (`OBLIVIOUS_SPECULATION`) |
| `MOJOTREES_GPU_FUSE_PARTITION_TAIL` | `gpu_resident_round.mojo:921` | REACHES | STRUCTURAL: `_device_search_resident` partitions through `apply_split` → `GpuActiveRows.partition`, which has no descriptor and no deferred copy-back | STRUCTURAL: `enqueue_desc_level_children` ends in an unconditional `mark_copy_back_fused()`, which **raises** when no debt is outstanding, so `=0` could only turn a symmetric fit into an error. The loop hard-wires `set_partition_fusion(True)` |
| `MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD` (default ON, measured 1.26x) | `gpu_resident_round.mojo:1013` | STRUCTURAL for the device-owned plane: the host never learns a child's depth or row count, so the test cannot be made there | **WAS ACCIDENTAL. Built as `MOJOTREES_GPU_SKIP_TERMINAL_CHILDREN`** | REACHES |
| `MOJOTREES_GPU_SKIP_TERMINAL_CHILDREN` (new, default off) | `train_gpu.mojo` | REACHES when the fit falls back to `_device_search_resident` | REACHES | STRUCTURAL, the symmetric switch above is its level-shaped twin |
| `MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST` | `gpu_resident_round.mojo:1073` | **ACCIDENTAL, and the bigger case.** The leaf-wise plane makes one `_copy_noise` per growth step, 30 per tree at the default budget against the symmetric 6, and moves two records apiece rather than one. Not built; see below | STRUCTURAL: `enqueue_frontier` already stages a whole batch before any table crosses, so a level pays one staging pass, not one per split | REACHES |
| `MOJOTREES_GPU_OBLIVIOUS_SUBTRACT` (sibling subtraction, **default ON, measured 1.78x**) | `gpu_leaf_batching.mojo` | REACHES, unconditionally and with no switch (`enqueue_desc_child` folds it in) | REACHES, unconditionally (`enqueue_resident_leaf_subtracting`) | REACHES. This cell read "behind the switch"; since 2026-08-17 the switch is `!= "0"` and the subtraction is what a symmetric fit does unless refused. This plane is the only one that ever dropped it |
| `MOJOTREES_GPU_SPLIT_WIDE` (default ON, measured 1.21x) | `gpu_split_search.mojo:2562` | REACHES | REACHES, same searcher field `wide_scan`, set once at construction | STRUCTURAL, has its own twin below |
| `MOJOTREES_GPU_OBLIVIOUS_WIDE` (default ON, measured 4.5%) | `gpu_split_search.mojo:4355` | STRUCTURAL, twin of the above | STRUCTURAL | REACHES |
| `MOJOTREES_GPU_SPLIT_TABLE_PACK` (four uploads to one) | `gpu_split_search.mojo` | REACHES | REACHES via `enqueue_frontier` → `_copy_tables` | REACHES |
| `MOJOTREES_GPU_TABLE_RESET` (five copies to one kernel) | `gpu_tree_tables.mojo` | REACHES | STRUCTURAL: `_device_search_resident` never opens the descriptor tables | REACHES |
| `MOJOTREES_GPU_PACKED_DOWNLOAD` (six downloads to one) | `gpu_tree_tables.mojo` | REACHES | STRUCTURAL, same reason | REACHES |
| `MOJOTREES_GPU_ROW_COMPACTION` | `gpu_active_rows.mojo:5910` | REACHES: `enqueue_desc_child` → `enqueue_desc_histogram` → `_ensure_compacted` | REACHES: `enqueue_leaf` → `enqueue_range_histogram` → `_ensure_compacted` | **TRACED AND REFUSED, 2026-08-18. It was accidental, and it was not inert.** This cell read "SUSPECTED ACCIDENTAL ... Not fully traced". The trace holds, and the batched level build goes through `gpu_leaf_batching.enqueue_device_plan_batch_fused`, and `_ensure_compacted` still has exactly two call sites, neither of them that one. What the cell got wrong is the word inert. `GpuActiveRows` still MAINTAINED the compaction on this plane, one rebuild plus two launches a level, **13 command buffers on a depth-6 tree**, while 62 of the 63 histograms the tree builds read the dataset's own matrix. `oblivious_schedule_launches(6, 64, True)` is 55, and 55 + 13 is 68 against a 64-deep Metal queue that DOES NOT RAISE when overrun. So `histogram_gpu.GpuHistogramBuilder.stage_desc_level_plan` now **raises** on `self.rows.row_compaction_requested()`, once per tree. The missing consumer was also built, as `MOJOTREES_GPU_OBLIVIOUS_COMPACT_BINS`, and **measured 0.757x**; see `docs/design/DECLINED_OPTIMIZATIONS.md` C1 |
| `MOJOTREES_GPU_HIST_STRATEGY`, `_MIN_TILES`, `_HIST_SPECIALIZATION` | `gpu_tiling.mojo`, `apple_histogram_policy.mojo` | REACHES | REACHES | partially: the batched level build resolves its own geometry |
| `MOJOTREES_GPU_VERIFY_ROWS` | `train_gpu.mojo:1397` | STRUCTURAL, refused by name (`_check_verify_rows_reachable`) | REACHES, through `apply_split(expected_left=...)` | STRUCTURAL, refused by name |
| `random_strength` device noise | `train_gpu.mojo`, `gpu_split_search.mojo` | REACHES | **WAS A HARD FAILURE.** Fixed; see below | REACHES |
| `PhaseProfile` phase axis (device phases) | `phase_profile.mojo` | STRUCTURAL absence, stated: fences would measure the instrument | REACHES, fully | STRUCTURAL absence, one opaque phase |
| `PhaseProfile` host-span axis | `phase_profile.mojo`, `gpu_resident_round.mojo` | REACHES, fully (2026-08-18): host clocks around host calls add no fence, so the refusal above does not cover them | DOES NOT REACH. `_device_search_resident` and `_device_search_incremental` are not bracketed on this axis; they are the fully instrumented loops on the PHASE axis, which is the instrument for them, and a host-span table from a `MOJOTREES_GPU_TREE_RESIDENT=0` run is empty rather than partial | REACHES, fully, with the level index as the step axis |

## The `random_strength` contradiction, settled

**Answer: yes, the device `random_strength` path is reached by a fit today.**
The two comments in `train_gpu.mojo` saying "NOT REACHED BY ANY FIT TODAY" were
wrong. `_device_search_unsupported_reason` and
`docs/design/OBLIVIOUS_WAIT_CENSUS.md` are right, and the census's six-drain
finding stands.

The proof is a call graph, checked link by link:

1. `_grow_tree_gpu_device_search` calls `_check_device_search_supported`.
2. That function does not read `ExtraTreeParams.is_active()` anywhere. It calls
   `check_scalars(min_data_in_leaf, scale_computed_per_tree=True)` and then
   `_device_search_unsupported_reason`.
3. `check_scalars` refuses a positive `random_strength` only when the scale is
   zero **and** `scale_computed_per_tree` is False. This caller passes True.
4. `_device_search_unsupported_reason` delegates to
   `ExtraTreeParams.device_unsupported_reason`, whose `random_strength` arm is
   `if has_categorical and self.random_strength > 0.0`. With no categorical
   column it returns the empty string.
5. `_device_search_semantics_supported` is the question form of the same
   predicate, so AUTO routes such a fit **onto** the device search.
6. Both arms of `_train_gpu_rounds` compute `random_score_scale` per round, so
   `random_score_stdev()` is positive.

So `cache.searchers[0].set_random_score(...)` executes. The comment claiming
otherwise has been replaced with the trace above. `check_scalars`'s own message
had also already been corrected to name both GPU arms, which the stale comment
denied as well.

**What the sweep found underneath it.** The staging reached two of the three
growers. `grow_tree_device_resident` supplies a node id through
`resident_child_node_base`, and `grow_tree_device_oblivious` keys by level
depth. `_device_search_resident` and `_device_search_incremental` supplied
none: every `SplitNodeRequest` was built without the `node` argument, so it
defaulted to -1, and `GpuSplitSearcher.stage_random_score` **raises** on a
negative node id. A depth-wise GPU fit with `random_strength` set therefore did
not train a quietly unnoised model, it failed with "random_strength keys its
draw by node id, which must be nonnegative". Both loops now pass the node id
they already hold, which is `tree._add_node`'s numbering and therefore the same
stream the CPU grower and `resident_child_node_base` draw from.

That fix is unconditional rather than switched. It cannot move a number in any
fit that runs today: `SplitNodeRequest.node` is read at exactly one site,
guarded by `noise_stdev > 0.0`, and every fit that reached that site with the
noise on raised instead of producing a tree.

## What lifting the depth-wise restriction is worth

**An estimate, arithmetic shown, and it is smaller than it looks.**

The restriction is `tree_resident_supported`'s
`if params.grow_policy != GROW_LEAFWISE: return TREE_RESIDENT_DEPTHWISE`, whose
stated reason is that a level is admitted as a gain-ordered prefix under the
budget, which is a different reduction from the device commit kernel's single
argmax. That reason is accurate but incomplete, and the fuller answer is that
**two** kernels are single-shaped, not one:

- `gpu_tree_tables._pick_and_commit_kernel` reduces to one winner over the
  whole frontier with no level restriction.
- `gpu_tree_tables._commit_level_kernel`, which does commit a whole level in
  one launch, commits **one split applied to every leaf of the level**. Its
  partition argument is explicit: "one stable partition of that prefix by the
  level's single routing rule". Depth-wise needs N routing rules per level, so
  neither `enqueue_desc_partition` nor `enqueue_desc_level_children` can be
  reused as they stand.

There is a cheap-looking lift that does not survive contact. Restricting the
existing single argmax to the shallowest eligible depth produces exactly the
depth-wise **set** of splits, because gains at a level are invariant to
splitting a sibling (disjoint rows), so greedy-by-gain within a level is the
same top-k that `admit_level`'s gain-ordered prefix takes. What it does not
produce is the depth-wise **order**: `GrowthSchedule` queues admitted leaves in
ascending node id and the argmax commits them in descending gain, so the node
ids differ. Predictions would be identical and the serialized model would not,
which fails the node-for-node comparison the harness runs. And it would still
need a `min_depth` argument threaded into a kernel in `gpu_tree_tables.mojo`.

The size of the prize, from the numbers already registered:

```
depth-wise today, 799,110 x 100                     3.245 s
round trips per depth-wise tree                     7   (root + one per level)
round trips a device-owned plane would make         2
round trips removed per tree                        5
trees in the measured fit                         100
round trips removed                               500

one round trip, measured, docs/METAL_TIMELINE.md:550   606 us
one round trip, derived figure used in gpu_resident_round 458 us

500 * 606us = 0.303 s   =  9.3% of 3.245
500 * 458us = 0.229 s   =  7.1% of 3.245
```

Add the launch side. The device plane's descriptor partition is 2 launches
fused where `apply_split` is 4, so 2 fewer per split; at 30 splits and the
measured ~20 us of enqueue per launch that is 1.2 ms per tree, 0.12 s per fit,
another 3.7 percent. Host work per batch (candidate list, per-node feature
draws, frontier rebuild) is not counted here because it partly overlaps device
work and nothing has measured it.

**Estimated total: 0.35 to 0.42 s, or 11 to 13 percent of a 3.245 s
depth-wise fit.** Against that: a new level commit kernel, a multi-rule level
partition, and a multi-rule batched child build, in three files
(`gpu_tree_tables.mojo`, `gpu_active_rows.mojo`, `gpu_leaf_batching.mojo`) that
this lane may not write. Compare with the two switch flips taken the same day,
which cost one line each and returned 1.21x and 1.26x. **This is not the next
thing to do.** It is the right thing to do eventually, and it should be
scheduled behind anything cheaper.

One number to keep in view while deciding: depth-wise at 3.245 s is already
faster than leaf-wise at 3.659 s on the same shape, on seven round trips
against two. Round trips are not what separates those two arms, which is
itself evidence that 7 to 9 percent is the right order for this lift and not
an underestimate.

## Built in this round

Both in `train_gpu.mojo`, both bit-identical, one switched and one not.

- **`MOJOTREES_GPU_SKIP_TERMINAL_CHILDREN`**, default off, spelled `== "1"`.
  The general twin of `MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD`. A child that
  `_apply_shape_rules` will refuse -- at `max_depth`, or below
  `2 * min_data_in_leaf` rows -- is known before its device work is enqueued,
  from the parent record's exact integer counts. Such a child's histogram is
  read by nothing (the only reader of a leaf's slot is that leaf's own search)
  and its record is read for two words that are already decided. So its search
  request is not staged, and when **both** children are terminal the histogram
  build goes too and the parent's slot is released rather than reassigned. A
  batch whose every request was skipped enqueues no search pair and makes no
  `download_frontier`, which under depth-wise growth with `max_depth` binding
  is the whole last level and one of the seven round trips.

  Bit-identity rests on three legs, written out at
  `SKIP_TERMINAL_CHILDREN_VAR`: leaf values come from the parent's record and
  not the child's; both schedules gate on `eligible` before they read `gain`,
  and a skipped child is filed with `GpuSplitRecord()`, whose `found` is False;
  and `apply_split` still runs, so the row permutation the trainer reads back
  does not move.

- **The node id on every `SplitNodeRequest`**, unconditional, for the reason
  above.

Also, no behavior change: `MOJOTREES_GPU_SPLIT_RESIDENT=0` on a symmetric fit
now raises a message that names the variable instead of reporting a pool that
was never asked to open.

## Not built, and why

**The leaf-wise noise hoist.** `MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST` collapses
6 per-level `_copy_noise` drains to 1. The leaf-wise device plane makes 30 of
them per tree, two records apiece, and has no such arm. Nothing structural
blocks it: the draw is a pure function of the node id, and every node id is
known in advance as `2 * step + 1`; all three device entry points the loop uses
take their record indices as arguments, so no kernel changes. It costs
`2 * (num_leaves - 1)` records above the leaf budget -- 91 rather than 33 at
the default -- and a noise buffer near 9 MB at 100 features and 256 bins. It is
not built because of **reach, not difficulty**: the leaf-wise noise path is off
in both shipped default sets, since CatBoost mode is symmetric and lossguide
mode mirrors LightGBM, which has no `random_strength`. Worth building for
whoever measures leaf-wise CatBoost-style regularization; worth nothing before
that.

## Edits owed by other lanes

1. **APPLIED 2026-08-17.** `gpu_split_policy.mojo`, `decide_split_search`. Its
   docstring stated two work thresholds ("device iff work >= 50,000,000", and a
   depth-wise superset) and stated that `grow_policy` "is built, tested and
   unreached until a caller passes it". Both were stale at head. The body has no
   threshold left after the 2026-08-16 removal, and
   `train_gpu.split_search_decision_for` does pass `params.grow_policy`. A
   reader deciding whether depth-wise routes to the device today got the wrong
   answer from that docstring. The docstring now states the four-branch rule the
   body has, quotes the withdrawn text so the change is visible, and says that
   `grow_policy` is carried for the record and read by nothing.
2. **CLOSED 2026-08-17, by the middle course rather than by deletion.**
   `gpu_tree_tables.tree_resident_requested` read `MOJOTREES_GPU_TREE_RESIDENT`
   as `== "1"` while the live gate read `!= "0"`, so two predicates over one
   variable disagreed about its default. That file now holds
   `tree_resident_enabled` (`!= "0"`) and
   `tree_resident_explicitly_requested` (`== "1"`) over one
   `comptime TREE_RESIDENT_VAR`, and `tree_resident_requested` survives only as
   a one-line deprecated alias for the diagnostic one, reading no variable of
   its own. It is kept because `tests/test_gpu_tree_tables.mojo:106` and `:581`
   import and call it by name. STILL OWED, and now recorded in both files
   rather than in one. `gpu_resident_round.resident_round_enabled` and
   `resident_round_explicitly_requested` each still call `getenv` themselves, so
   four reader bodies survive over one variable. They agree, so this is a
   cleanup and not a hazard. Delete the alias with the two test lines, and turn
   the two in `gpu_resident_round` into delegations, in whichever order.
3. **CLOSED 2026-08-18, and the guess in it was half wrong.**
   `gpu_leaf_batching.mojo`. This item read "The batched level build appears
   not to go through `GpuActiveRows._ensure_compacted`, which has two call
   sites and neither is it, so `MOJOTREES_GPU_ROW_COMPACTION` would be inert on
   the symmetric plane while reaching the other two. Confirm and classify." It
   was confirmed and classified. The two call sites are
   `GpuActiveRows.enqueue_desc_histogram` and
   `GpuActiveRows.enqueue_range_histogram`, and the level build still reaches
   neither. **But it was never inert.** Arming the switch armed the
   MAINTENANCE, which is one `_ensure_compacted` rebuild at the root build plus
   the scatter and the copy-back from `_maintain_compaction` on every level's
   descriptor partition, so 1 + 12 = 13 command buffers on a depth-6 tree,
   collecting only the root build out of 63 histograms. Nothing consumes the
   result and the queue overruns silently at 68 buffers against 64, so
   `histogram_gpu.GpuHistogramBuilder.stage_desc_level_plan` now raises rather
   than paying it. The missing consumer was separately built as
   `MOJOTREES_GPU_OBLIVIOUS_COMPACT_BINS` and measured 0.757x, so it is not
   coming back; `docs/design/DECLINED_OPTIMIZATIONS.md` row C1 holds the run.
4. `gpu_tree_tables.mojo`, for whoever takes the depth-wise lift: a `min_depth`
   argument on the pick, and a level commit that admits per-leaf splits rather
   than one split per level.

## What could not be verified without a compiler, ranked

1. **That the new code compiles at all.** In particular: reassigning
   `var recs = List[GpuSplitRecord]()` from `searcher.download_frontier(...)`;
   `var left_slot = -1` later assigned from a conditional expression; and the
   two new defaulted `Int` fields on `_GpuPendingSplit`, a `Movable`
   non-`Copyable` struct.
2. **That the switch is inert when off.** Argued line by line -- `left_live`
   and `right_live` are both True unless `skip_terminal`, and `left_rec` /
   `right_rec` then take exactly the values `2 * k` and `2 * k + 1` the
   positional rule computed -- but not executed.
3. **That the switch is bit-identical when on.** The three legs are argued
   above and each is checkable by reading. The check that would settle it is a
   fit with the switch off and on, forests compared node for node with `value`
   and `split_gain` as bit patterns, under `grow_policy=depthwise` with a
   binding `max_depth`, which is the shape
   `tests/test_gpu_speculation_build.mojo` already uses for its own arm.
4. **That the `random_strength` fix produces the same tree as the CPU
   grower.** The node ids agree by construction, so the draws agree; that the
   resulting model then matches is the standing bar
   `ExtraTreeParams.device_unsupported_reason` set on 2026-08-17 ("the device
   model must MOVE when the setting moves") and it has not been cleared for
   this path.
5. **The size of the terminal-child skip.** Counted, not measured. Under
   depth-wise growth at 31 leaves with a binding depth, 16 of 30 splits have
   both children terminal, and the sum of built rows over those 16 is roughly
   `n/2` against a whole-tree total near `3.5n`, so the arithmetic says about a
   14 percent cut in histogram row traffic plus one round trip. On a plane
   where histogram construction is the reported 86 percent of a symmetric fit,
   that is a double-digit expectation, and expectations of that shape have been
   wrong here before (the row-tile floor).
6. **CLOSED 2026-08-18, and it never needed a compiler.** This item read
   "Whether `MOJOTREES_GPU_ROW_COMPACTION` is inert on the symmetric plane.
   Traced to two `_ensure_compacted` call sites, neither in
   `gpu_leaf_batching`, but the batched kernel's own bin source was not read."
   The batched kernel's bin source has now been read.
   `gpu_leaf_batching._batch_hist_atomic_kernel` and its subtracting twin
   gather `bins[f * n_rows + rows[begin + j]]` from the dataset's own matrix
   and have no other source, so the compacted planes are never consumed on this
   plane. **The answer is that it is NOT inert.** It is worse than inert,
   because it costs 13 command buffers a tree in maintenance nothing reads, and the total
   of 68 overruns a 64-deep queue that reports nothing when overrun. It is now
   a refusal in
   `histogram_gpu.GpuHistogramBuilder.stage_desc_level_plan`. Reading source
   settled every part of this, which is why it should not have sat on a list
   headed "could not be verified without a compiler" for a day.
