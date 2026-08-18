# Resume note, written 2026-08-18 evening, before a compaction

For me, next session. This carries what a conversation summary loses: exact
numbers with their run ids, the things I got wrong and corrected, and the
decisions that are Andrew's rather than mine. `SESSION_2026-08-18_MEASURED.md`
is the measurement index and is the companion to this file; read that one for
what was measured and this one for what to do next.

## State

Branch `perf-round-2`, pushed through `0b300ba`. Roughly 55 commits today, all
mine except a handful from the CUDA session. **The full suite ran and is green**
behind all of them, which was the first time in 45 commits. Two failures were
found and both were mine, both fixed and re-verified.

Nothing is on `main`, and merging is a release judgment Andrew has not made.

## The four things worth knowing that a summary will flatten

**1. The accumulate finding, and I reversed myself twice getting to it.**
The covertype CPU round is 62 percent histogram accumulate. The per-slot rate
degrades **19.4x** from root to tiny nodes, where our own note said 8.7x. The
prize is **22.3 ms of a 36.8 ms phase, 38 percent of the whole CPU round.**

First I said compaction could not touch it, because the accumulate is
slot-proportional rather than cell-proportional. That is right about the pass
count and **wrong about the mechanism**, and I published it before checking.
What degrades is histogram WRITE REUSE: updates per cell fall 1822 to 295 to 71
to 9.1 to 1.4 across the classes, tracking the cost exactly. And the bytes
settle it: at 24 bytes per cell the rectangular histogram is 13,770 cells =
**322.7 KB against a 64 KB L1**, and a packed one is ~2,400 cells = **56.2 KB,
which fits**. So compaction reaches the 62 percent after all, by giving each
cell 5.7x more reuse and moving the working set from L2 to L1.

**This is arithmetic over a profile, not an A/B.** Nobody has built a packed
accumulate and timed it. The falsifier is written down: if a packed accumulate
does not move the small and tiny classes toward the root rate, the reuse model
is wrong too.

**2. The instrument has a hole and I misrepresented it for hours.** Every
per-cell profile row on the GPU arm reads **ZERO**. That instrumentation is
CPU-path only; the device reports host-step spans instead. So the
"discriminating measurement" settles things for the CPU arm alone, and **the
GPU's 4.5x on covertype still has no per-phase attribution.**

**3. Two accidents hardened into published limits, and both are fixed.** The
wheel was `cp314-cp314-macosx_26_0_arm64` because pixi's solver took the newest
Python and the compiler stamped the build machine's SDK. It is now
`py3-none-macosx_12_0_arm64`, verified by installing into a clean venv and
fitting. **No build matrix was needed**: the extension links no libpython and
resolves CPython by name, so one artifact already serves 3.10 through 3.14 and
only the tag was in the way. A `python_tag` option does nothing; overriding
`bdist_wheel.get_tag` is the only place the decision is made.

**4. Depth 8 works and my own raise had created the bug.** The oblivious device
grower now grows depth 8 bit-identically, rmse 0.319707 on both backends. The
bug was `_commit_level_kernel` phase 2 being `if tid < n_live:` with no stride
at `block_dim=64`, so a 128-parent level left half the windows unwritten. And
the kernel's overflow test `n_live > OBLIVIOUS_LEVEL_LEAVES` had been, by
coincidence, exactly the block-coverage check while that constant was 64.
**Raising it to 256 disarmed a guard nobody knew existed**, turning a loud
refusal into silent corruption. The bound was written down five times.

## Do not redo these

- The layout switch is default ON. Measured 1.269x and 1.224x on year, null on
  covertype, bit-identical by ONE prediction digest across twelve cells.
- The feature-group clamp is a NULL, 1.002x and 0.967x, and setting it WITH the
  layout switch is worse than layout alone.
- Small-node READ locality is not the layout switch's mechanism. It wins on 90
  dense continuous columns, not on small nodes. This does not contradict item 1
  above; read the cross-reference in the measured-session file.
- The three-arm GPU A/B is NULL. Morning, head, and head with
  `OBLIVIOUS_MAX_LEAVES` reverted to 64 are within 0.05 percent. No commit
  today slowed the GPU arm, and the gbm-bench 1.19x-to-1.41x move was
  cross-window drift. **Do not quote 1.13x for anything.**
- Launch-count reduction on the leaf-wise arm measured 1.004x after removing
  21.5 percent of launches, at 31 leaves. That null may NOT transfer to 256
  leaves; the regimes differ.

## Open, with who owns it

**Andrew's calls.** Merging to main. The MVS flip, which now needs only a
measurement since both gates are closed as patches at `/tmp/mvs_gate*.patch`
with a stated apply order. Whether to delete rule 12, which has one live
instance left, the relevance-label cap of 30 that is LightGBM's gain-table
length rather than any bound of ours.

**Mine, ranked.** The compile cache: CI takes the suite from 519 s to 78 s and
if the local runner misses that cache we pay seven times over on every run,
which is worth more than any test deletion. Then the packed accumulate from
item 1. Then the cgroup quota read, because `nproc` reporting 256 against a
27.2-CPU limit is every container and not one pod, so every derived task count
and crossover threshold is computed against a machine we do not have.

**Still unreached from Python:** nothing, as of today. Five refusal blocks that
existed and could not fire were wired. Verify with a call, never a predicate.

## THE FULL TO-DO LIST

Everything identified today, with who owns it. Nothing here is done unless it
says so. Ranked inside each group.

### A. Andrew's decisions, not mine

1. **Merge to main.** 55 commits, suite green, real behavior changes. Release
   judgment.
2. **The MVS device flip.** Both gates are CLOSED as patches at
   `/tmp/mvs_gate2_keep_plane.patch`, `/tmp/mvs_gate1_iteration_cap.patch`,
   `/tmp/mvs_gate2_train_gpu.patch`, `/tmp/mvs_gate_tests.patch`, with a stated
   apply order and a whole-file fallback at
   `/tmp/gpu_objectives_native.mojo.mvs_gates`. Worth 2.23x on the oblivious
   arm. Needs the patches applied, a build, `device_agreement` re-run at
   `min_data_in_leaf=20` on the BUILT artifact, and one interleaved
   measurement of the compaction cost against the subsample gain. The advisor's
   objective carve-out may be UNNECESSARY, because gate 2's compaction closes
   the renewal gate as a side effect.
3. **Delete rule 12 or not.** One live instance remains, the relevance-label cap
   of 30, which is LightGBM's gain-table length; the real exactness bound is 53.
4. **Whether to pull CatBoost's `model_evaluation_speed`.** Inference, the one
   competitor-owned suite our GPU can enter, and the place our local numbers say
   we win. Needs a 3.86 GB epsilon download.
5. **The reranker inference lane**, 100 to 1,000 rows, p50 and p99, against
   LightGBM, XGBoost, CatBoost, lleaves and ONNX Runtime. Strategically the
   strongest bet and the place we are already ahead.

### B. NVIDIA and CUDA, ordered by the session that owns the hardware

1. **FMA prescale. DONE** (`d01bf56`, verified their side; my fix was `92c90f7`).
2. **The suite at a valid job count**, then the determinism row, then the
   split-search arm bench that would let `SPLIT_REASON_UNKNOWN_HARDWARE` become
   a measured policy row rather than an evidence refusal, then widen the gate,
   then gbm-bench on Linux with `--variant real`.
3. **Hypothesis 2, unbounded in-flight work in the resident grower.** 278
   command buffers per tree at 31 leaves with one synchronize. Metal throttles
   when its 64-deep queue fills; CUDA's behavior there is unestablished. Test
   with one variable, `MOJOTREES_GPU_TREE_RESIDENT=0`. This is the cheapest of
   the three and the only one that also explains why a MAX-only repro with
   modest queue depth ran clean.
4. **The 48 KiB shared-memory path the M4 physically cannot reach.** Feature
   group 16 at 256 bins raises on Metal and is accepted on the 5090.
5. **No 5090 number is publishable** until 2 and 3 land.

### C. Mine, ranked, and the first two are cheap

1. ~~**The compile cache.**~~ **CLOSED 2026-08-18, AND IT WAS MY MISREADING.**
   The 519 s to 78 s figure is CI gaining a cache that LOCAL ALREADY HAS.
   `tools/run_tests.sh`'s own comment says it: the cache "survives locally and
   CI throws it away every run", and `.github/workflows/ci.yml` already has the
   restore step. Measured: **23 GB and 22,240 files** in
   `.pixi/envs/default/share/max/cache`. There is no local win here.
   The suite's seventeen minutes is not caching. It is GPU correctness tests
   doing real fits: `test_gpu_launch_fusion` alone is 690 s standalone and
   2,239 s under the suite's own contention, for seven tests that grow six real
   forests and compare two arms bit for bit. That file earns its time. **If
   suite duration is ever the target, the lever is the GPU tests' parallelism
   and the 4-job GPU cap, not a cache and not test deletion.**
2. **The cgroup quota read.** `MOJOTREES_NUM_WORKERS` may never reach MAX's pool,
   and `nproc` reports the host's cores against a container limit. Every derived
   task count, grain size and crossover threshold is computed against a machine
   we do not have. Read `/sys/fs/cgroup/cpu.max` and v1's
   `cpu.cfs_quota_us`/`cpu.cfs_period_us`. If we CANNOT size MAX's pool, then the
   fix is to stop claiming we can, at every docstring that implies it.
   Constrained by `DECLINED_OPTIMIZATIONS` F6: that pool measured ~3.5x on a
   10-core M4 and FLAT from 4 to 16 tasks.
3. **The packed accumulate.** The 38-percent prize. 269 sites in `src/` (57 host,
   212 device) plus 186 in tests, and ZERO in bindings, python, capi, the model
   format or `lgbm_model_io` — the histogram never leaves the trainer. Do it as
   the staged path: make the four cell-proportional passes offset-aware ONE AT A
   TIME on a pool invariant that tail cells are zero, output staying
   rectangular. Start with SUBTRACT, 60 percent of the avoidable traffic, three
   functions, no flags, unconditionally bit-identical. **Two conditions:**
   `apple_cpu_policy`'s `max_row_blocks_for_cells` must keep receiving
   `n_active * n_bins` or the block count changes and bits move, and the split
   scan must keep the full width under `random_strength` or `extra_trees`.
   **And `feature_bins`/`bin_offset` are EMPTY on every GPU fit** and their
   observed-max rule is unsafe as an output width; use `efb.dense_bin_counts`.
4. **Device-first compaction.** On the device the rectangle costs CAPACITY, not
   bandwidth: a covertype slot is 165 KB rectangular and 28.8 KB packed, so the
   resident pool holds 5.7x more leaves and the pool-not-fitting refusal stops
   firing. The only item found today that changes WHICH CODE PATH RUNS.
   Decompacting inside `histogram_from_host` keeps the in-run host-replica
   bit-identity check working unmodified.
5. **The fan-out fusion.** `parallel.dispatch_regions` has ZERO callers, was
   built for exactly this, and its docstring says to delete it if nobody
   sequences the wiring. Worth ~8.2 s on covertype at a measured 46 microseconds
   per wake, derived from our own interleaved thread-scaling run. Needs a
   three-line call-site guard for the case where both children sit below the
   parallel crossover. **The hazard no compiler catches:** `extra_trees` and
   `random_strength` key their draws on the node id and the two fused regions
   carry different node ids, so getting it wrong makes every right child draw
   from the left child's stream and produces a different tree that trains
   normally.
6. **The eval-set GPU route.** `BLOCK_VALIDATION_SET` refuses the device to every
   fit with an eval set, and `train_gpu_with_valid` is a complete GPU eval-set
   trainer with a device-resident scorer and no binding. **It is not a wiring
   job**: the Python path goes to `custom_metric.fit_with_metrics`, where Python
   owns the round loop and calls back per metric per round. Take the NARROW
   route, eval_set with a built-in metric only, callback path untouched, with two
   conditions: early-stopping semantics must match the CPU path exactly (same
   best-iteration, patience counting, truncation, `evals_result_` shape, proven
   by one fit both ways comparing the chosen round and the metric trace), and the
   route must announce itself. If (a) cannot be met, do not route it.
7. **`SPLIT_REASON_UNKNOWN_HARDWARE` has no message.** Every non-M4 GPU user is
   silently routed to the host split scan, which is 1.29x to 1.85x slower AND
   produces a different model, and is told nothing. Add it to the
   `_warn_about_device_decision` whitelist.
8. **`AUTO_GPU_MIN_ROWS = 250_000` is the last provisional threshold**, and it
   sits at a shape where the GPU measured LOSING by 14 percent. One interleaved
   pair in [250k, 1M) rows at 50 features settles it.
9. **Depth 9.** Needs a device capability query rather than the 16,384
   conservative fallback, or the two Cosine accumulators comptime-specialized
   out of an L2 level, which takes twelve words per leaf to ten. Mojo 1.0 does
   support the specialization and this repo already does it elsewhere.
10. **The two in-kernel silent clamps.** `gpu_split_search` has two
    `if nl > OBLIVIOUS_MAX_LEAVES: nl = OBLIVIOUS_MAX_LEAVES`. A level wider than
    the constant is TRUNCATED rather than refused, building a correct-looking
    tree from a fraction of the level. Unreachable today because the host
    refuses first. **Must become a raise before anyone moves the constant again**,
    and a raise is not expressible in a Mojo kernel so it has to be the host
    guard.
11. **The switch deletion.** 85 real switches in `src/` (106 names, 21
    prose-only tombstones). Inventory started, not finished. Bucket (a) only:
    default off by `== "1"`, with a recorded null or negative verdict, not
    appearing in any `bench/` arms module. An UNMEASURED off switch is not a
    candidate, because "measured null" and "never tried" look identical from the
    predicate. Never delete a `!= "0"` escape hatch; that removes a user's
    revert path.
12. **The shared-frontier multiclass class batch.** covertype is 700 trees
    because it is 7 classes at 100 rounds, each paying the full 2,303 launches.
    Lockstep would take it to 100 x 2,303, a true 7x. The step loop is built for
    it and says so. **But** the reachable gradient half is already MEASURED NULL
    (15.45 vs 15.30), the histogram half has zero call sites, and seven resident
    pools at 256 leaves is 297 MB against a 256 MiB budget. Do not open it before
    the accumulate mechanism is settled.

### D. Correctness and honesty items, small and each already diagnosed

1. **Extract the workload mapping into a named function** in
   `device_selection.py` that both the sender and
   `tests/test_objective_marshalling.mojo`'s fixture call. That fixture has
   drifted THREE times, once for 646 commits. Until it is extracted, drift is
   detectable rather than impossible.
2. **~110 dead handoff citations** remain in `src/`, `python/`, `bindings/` and
   `docs/`. `tools/mark_dead_handoff_citations.py --apply` marks them; the better
   fix is repointing to the live document, which needs judgment per site.
3. **`efb.mojo:70`** calls bundling "indistinguishable" while the same file at
   `:1899` admits it agrees only "to floating-point association". The second is
   right. **EFB is NOT bit-identical**, by two mechanisms, one structural
   (default-bin recovery by subtraction) and one removable (the row-block count
   changing with `n_active`). Tier 2 under rule 11.
4. **EFB's eligibility probe is three serial passes and can be one**,
   bit-identical, saving 50 M cell visits on covertype and 83 M on year where the
   answer is always no. And its 3.5x estimate is in DOUBT:
   `expand_bundled_histogram` writes the full rectangular output per node,
   serially, so the honest prediction is a big win on the top levels and possibly
   a loss on the leaf frontier.
5. **`objective_backends` over-claims** `SUPPORTS_GPU` for `QUERY_RMSE`,
   `PAIR_LOGIT` and `YETI_RANK`, asserting trainers that do not exist.
6. **`gpu_supports_outputs` is `return n_outputs >= 1`**, so `BLOCK_OUTPUT_LIMIT`
   cannot fire. A tautology is the second way a gate becomes unreachable.
7. **Two Python comments** (`device_selection.py:~109`, `sklearn.py:~416`) still
   say the policy blocks every non-L2 score selector. Untrue since 2026-08-17.
8. **`check_goss_honored` is not exported** from `src/mojotrees/__init__.mojo`.
9. **`predict.predict_tile_enabled`'s docstring** still says its measurement is
   owed. It was taken: 1.10x leaf-wise, 1.54x depth-wise. On its own terms that
   function, the variable and the `apply_row_major` closure are due for deletion.
10. **`DECLINED_OPTIMIZATIONS` C1 cites `docs/design/LANE_RULES.md`**, which does
    not exist; it is `bench/results/LANE_RULES.md`.
11. **Five docstrings still say the multiclass class batch is unmeasured.** It
    was measured null at this exact shape, 15.45 against 15.30.
12. **`phase_profile.HOST_HIST_DISPATCHES = 2` is wrong in both directions**: the
    zero pass it counts does not run at full feature sampling, and the fold and
    gather that do run are not counted. Three call-site citations in that file
    also point at moved lines.
13. **`HOST_STEP_SLOTS = 32`** means 223 of 255 steps collapse into
    `HOST_SLOT_OVERFLOW` at the benchmark shape, truncating the only curve that
    separates encoding from backpressure. Widen it before profiling that shape.
14. **`pixi.toml`'s claim** that turning off `--fp-mode contract=fast` "would
    cost speed rather than buy it" carries no measurement. Unpriced claim in a
    file that reads as a contract.

### E. Ranking, and the structural fix comes before the dataset

1. **A scenario that falls back must emit "not measured", never a scored
   verdict.** The ranking scenario has run SIX times, every one silently on a
   generator where both engines score 0.99 against a 0.02 absolute threshold.
   Without this, MSLR becomes the seventh run of a harness that would have passed
   on noise. Rule 13's shape.
2. **Split the 6.5x training gap before opening a lane on it**: what fraction of
   a lambdarank round is the gradient (host, per query, pairwise) versus the
   tree. Do NOT let a lane "reach" the device ranking path as the fix; those
   gradients cover QueryRMSE, PairLogit and YetiRank, and **lambdarank is not
   among them.**
3. **MSLR-WEB10K**, manual download, `~/gbm-datasets/Fold1` is waiting, then
   `fetch.py --pin` and `protect_datasets.sh lock`.
4. **A lambdarank row in `bench/lgbm_interop_matrix.py`.** The parser reading
   says it will pass; the real risk is `_synthesize_mapping` against
   MSLR-shaped features, not the objective, so it belongs after the dataset.
5. **XGBoost model import does not exist** in any form, JSON or UBJSON.

## Process facts that will not survive a summary

**Seven subagents died on server-side 529 errors.** Not on the work. Everything
after that was done directly and went faster than the coordination did. If lanes
keep failing, do the work.

**Rule 16a is new and it is Andrew's clarification:** subagents run local tests
only, never the suite or a benchmark or a timing or a profile. **The
orchestrator measures without asking per run.** The old reading cost three
measurement windows in one day.

**Rules are compressed.** `LANE_RULES.md` now opens with fourteen questions in a
table; the argument is below. Four have actually caught things: 12, 14, 11's
digest, and 16a. **No new rule without a new incident** — six were written today,
which is the shape of producing governance instead of results.

**Session topology, as agreed:** one session per machine, not per topic. The
CUDA session keeps the 5090. Everything on this M4 is me plus subagents. Peer
sessions sharing this checkout cost two handovers and a broken per-file
ownership boundary today, and the peers and I independently found the same
depth-8 bug within an hour.

**Data is protected.** `~/gbm-datasets` is flagged and vaulted, and so is
`bench/real_data/data`, which held 227 MB inside the git repo where
`git clean -xfd` would have taken covertype and year. `tools/protect_datasets.sh
lock` after every download. A locked file breaks `fetch.py --force`, and the
failure now says so.

**MSLR-WEB10K is not downloaded and cannot be fetched automatically.** Manual by
design; the URL redirects through a licence page. `~/gbm-datasets/Fold1` is
created and waiting. Before it lands, fix the harness so a scenario that falls
back emits "not measured" rather than a scored verdict, or MSLR becomes the
seventh run of a harness that would have passed on noise. The ranking scenario
has run six times, every one silently on a generator where both engines score
0.99.

## The one sentence I would want to read first

Every defect found today was a **consistency** failure rather than a reasoning
failure, eight of them, and the day's best work was finding places where this
repository disagreed with itself: twelve contradictions, three of which were
live bugs. Hunt contradictions, verify reach with a call, and never trust "it
builds."
