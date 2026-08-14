# Handoff: device-side split search (Apple task A2)

Best-split search over a GPU histogram, run on the device, returning one
compact record per node instead of a histogram. Implemented, tested against
both hand-computed arithmetic and the host `find_best_split`, and wired into
nothing. This file is the exact wiring for whoever integrates it.

The point of the lane: `grow_tree_gpu` downloads every node's histogram to
choose a split from it. That is `3 * n_features * n_bins` Int32 words plus a
full pipeline drain, per node. A 100-feature, 256-bin dataset pays 300 KB and
one host synchronization to produce a (feature, bin) pair, so the transfer is
three orders of magnitude larger than its answer, and it is paid on every node
of every tree of every round. Searching the histogram where it already lives
replaces that with a fixed 136-byte readback.

## Files this lane owns

| File | State |
|---|---|
| `src/mojoboost/gpu_split_search.mojo` | New. Three kernels, seven Float32 arithmetic helpers, `GpuSplitSearcher`, `GpuSplitRecord`, and a host reference implementation. ~1500 lines with docstrings. |
| `tests/parallel/test_gpu_split_search.mojo` | New. 16 tests. |
| `handoffs/apple_a2_split_search.md` | This file. |

Nothing else was touched. No edit to `split.mojo`, `categorical.mojo`,
`gain.mojo`, `monotone.mojo`, `histogram_gpu.mojo`, `train_gpu.mojo`,
`tree.mojo`, `__init__.mojo`, `pixi.toml`, or any existing test. Nothing was
committed or staged.

Note on `git status`: `src/mojoboost/gpu_split_search.mojo` shows as modified
rather than untracked, because commit `b5a9afc` captured a mid-session
snapshot of this same lane's file. Every line the working tree removes
relative to `HEAD` is this lane's own pre-fix code (the pointer-typed
`enqueue`, the `map_to_host` parameter upload, the inlined kernel launches).
No other lane's work is touched by it; `tests/parallel/api_snapshot_manifest.json`,
which another lane is editing, contains no reference to this module.

## Focused test

```sh
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/parallel/test_gpu_split_search.mojo
```

Result on this machine (Apple silicon GPU present, so nothing skipped):

```
Running 16 tests
    PASS [   0.015 ] test_numerical_scan_matches_hand_computed_gains
    PASS [   0.001 ] test_l1_soft_thresholds_every_gradient_sum
    PASS [   0.004 ] test_ties_go_to_the_lowest_bin_and_the_lowest_feature
    PASS [   0.004 ] test_min_child_hess_and_min_data_in_leaf_reject_candidates
    PASS [   0.009 ] test_no_split_still_reports_the_parent_leaf_value
    PASS [   0.001 ] test_missing_rows_choose_their_direction
    PASS [   0.002 ] test_top_ordinary_bin_is_a_candidate_only_with_missing_rows
    PASS [   0.004 ] test_monotone_constraint_rejects_a_candidate
    PASS [   0.003 ] test_interaction_mask_and_feature_subset_narrow_the_scan
    PASS [   0.004 ] test_categorical_onehot_partition
    PASS [   0.003 ] test_categorical_sorted_partition
    PASS [   0.001 ] test_categorical_and_numerical_compete_on_one_gain
    PASS [   0.001 ] test_record_round_trips_to_split_info
    PASS [ 105.961 ] test_device_search_matches_the_reference
    PASS [   2.749 ] test_device_search_is_deterministic
    PASS [  12.726 ] test_device_frontier_pick_best
--------
Summary [ 121.488 ] 16 tests run: 16 passed, 0 failed, 0 skipped
```

Two edits landed after that run: one docstring word (`pointer` to `buffer`)
and one line wrapped to stay inside 79 columns. Neither changes any
expression, but the run above predates them.

`git diff --check --` on the three assigned paths reports nothing. A direct
grep for trailing whitespace and tabs finds none, and no line exceeds 79
columns; those are the checks with content, since two of the three paths are
untracked.

## What the module provides

Everything is in `mojoboost.gpu_split_search`.

Kernels (private):

- `_scan_slot_kernel` — one threadgroup per active feature, writes that
  feature's best candidate as a per-slot record. Covers the ordinal threshold
  scan, the reserved missing bin and its direction, the top-threshold rule,
  `min_child_hess` / `min_data_in_leaf` rejection, monotone candidate
  rejection and output clamping, the interaction allow mask, one-vs-rest
  categories, and the sorted many-vs-many category walk.
- `_reduce_slots_kernel` — folds the per-slot records into one in ascending
  slot order and fills in child statistics, both child leaf values, and the
  parent's leaf value.
- `_pick_best_record_kernel` — folds several finished records into the
  best-gain one, ties to the lower index. The frontier selection
  `grow_tree_gpu` does on the host today.

Public surface:

```mojo
struct GpuSplitSearcher(Movable):
    def __init__(out self, n_features: Int, n_bins: Int,
                 missing_bins: List[Int] = [],
                 cats: CategoricalSpec = CategoricalSpec.none(),
                 max_records: Int = 1) raises
    def set_features(mut self, features: List[Int]) raises      # per tree
    def set_monotone(mut self, signs: List[Int] = []) raises     # per tree
    def set_allowed(mut self, allowed: List[Bool] = []) raises   # per node
    def enqueue(mut self, mut hist: DeviceBuffer[DType.int32],
                params: GpuSplitParams, g_scale: Float64, h_scale: Float64,
                bounds: OutputBounds = OutputBounds.unbounded(),
                record: Int = 0) raises
    def enqueue_pick_best(mut self, n_records: Int, record: Int = 0) raises
    def download(mut self, record: Int = 0) raises -> GpuSplitRecord
    def download_words(mut self, mut words_i: List[Int32],
                       mut words_f: List[Float32]) raises
    def upload_histogram(mut self, words: List[Int32]) raises    # test/bench
    def search(mut self, params, g_scale, h_scale, bounds, record) raises
        -> GpuSplitRecord                                        # test/bench
    def synchronize(self) raises
    def n_active(self) -> Int

struct GpuSplitRecord(Copyable, Movable, Writable):
    feature, bin, gain, found, default_left, is_categorical, cat_bitset,
    left: ChildStats, right: ChildStats, total: ChildStats,
    left_value, right_value, parent_value, ordinal
    def to_split_info(self) -> SplitInfo

struct GpuSplitParams(Copyable, Movable):
    lambda_l2, lambda_l1, min_child_hess, min_data_in_leaf, cat

def reference_search(...) raises -> GpuSplitRecord   # host mirror, see below
def decode_record(words_i, words_f, record) raises -> GpuSplitRecord
```

`GpuSplitRecord.to_split_info()` returns exactly the `SplitInfo` the growers
already consume, so the record is a drop-in for `_search`'s return value.

## Central integration, step by step

### 1. One shared `DeviceContext` (the only blocking change)

`GpuSplitSearcher.__init__` constructs its own `DeviceContext()`, the way
`GpuHistogramBuilder.__init__` does. Two contexts on one device are two
streams, so a searcher reading the builder's `out_dev` is only correct behind
an explicit fence.

The right fix is a constructor that adopts the builder's context. That needs
one edit inside `gpu_split_search.mojo` (this lane's file, but it is a shared
decision, so it is written here rather than guessed at) and one at the call
site:

```mojo
# gpu_split_search.mojo — add alongside the existing __init__
def __init__(out self, var ctx: DeviceContext, n_features: Int, n_bins: Int,
             missing_bins: List[Int] = [],
             cats: CategoricalSpec = CategoricalSpec.none(),
             max_records: Int = 1) raises:
    self.ctx = ctx^
    ...                       # body otherwise identical
```

and in `train_gpu.grow_tree_gpu`, construct with `builder.ctx.copy()` (or
whatever transfer `DeviceContext` supports — that is the one API fact this
lane has not verified, since nothing here passes a context across a
boundary). Until that lands, every integration must fence:

```mojo
builder.enqueue_leaf(node)
builder.synchronize()                 # required while the contexts differ
searcher.enqueue(builder.out_dev, ...)
```

which keeps one synchronization per node and still removes the 300 KB
transfer. Ship the fence first, delete it when the shared context lands.

No other field access needs an edit: `builder.out_dev`, `builder.ctx`,
`builder.g_scale`, `builder.h_scale`, `builder.missing_bin`, `builder.cats`,
`builder.n_features`, and `builder.n_bins` are all already public fields of
`GpuHistogramBuilder`.

### 2. Construct the searcher next to the builder

In `train_gpu`, `train_custom_gpu`, and `train_multiclass_gpu`, beside
`var builder = GpuHistogramBuilder(data)`:

```mojo
var searcher = GpuSplitSearcher(
    data.n_features, data.n_bins, data.missing_bin, data.cats,
    max_records=params.tree.num_leaves,
)
```

`max_records = num_leaves` is what the frontier stage below needs;
`max_records=1` is enough for the node-at-a-time stage. Both cost
`max_records * 136` bytes.

`grow_tree_gpu` must take the searcher: change its signature to

```mojo
def grow_tree_gpu(
    mut builder: GpuHistogramBuilder,
    mut searcher: GpuSplitSearcher,
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
) raises -> Tree:
```

and update the three call sites in `train_gpu.mojo`. `grow_tree_gpu` is
re-exported from `src/mojoboost/__init__.mojo` line 167, so that signature
change is public; if that matters, keep a one-argument wrapper that builds a
searcher per tree (it allocates nothing per node, so a per-tree searcher is
cheap, just wasteful of one construction).

### 3. Per tree

Right after `builder.set_features(tree_features)`:

```mojo
searcher.set_features(tree_features)
searcher.set_monotone(signs)
```

`set_features` resets the allow mask, so it must come before `set_allowed`.
Both are `map_to_host`-based and therefore synchronize; they run once per
tree, which is where the existing per-round `stage_gradients` synchronization
already sits.

### 4. Per node: replace the download and the host scan

Today:

```mojo
var root_hist = builder.build_leaf(root)
tree.value[root] = _leaf_value(root_hist, params.lambda_reg,
                               params.lambda_l1, value_feature)
var root_split = _search(root_hist, n_root, params, allowed, node_features,
                         depth=0, missing_bins=builder.missing_bin,
                         monotone=signs, cats=builder.cats)
```

After:

```mojo
var split_params = GpuSplitParams(
    params.lambda_reg, params.lambda_l1, params.min_child_hess,
    params.min_data_in_leaf, params.cat.copy(),
)
builder.enqueue_leaf(root)
builder.synchronize()                      # drop once the context is shared
searcher.set_features(node_features)       # per-node feature subsampling
searcher.set_allowed(allowed)
searcher.enqueue(builder.out_dev, split_params, builder.g_scale,
                 builder.h_scale, bounds)
var rec = searcher.download()
tree.value[root] = rec.parent_value
var root_split = rec.to_split_info()
```

Four host-side rules stay on the host, because they are properties of tree
shape rather than of the histogram, and `_search` applies them before it ever
looks at bins:

- `params.max_depth`: skip the search entirely for a leaf at the limit and
  record `SplitInfo(-1, -1, 0.0, False)`, exactly as `_search` returns.
- `n_rows < 2 * params.min_data_in_leaf` and `n_rows < 2`: same, skip.
- `parent_bounds.clamp(...)` on `rec.left_value` / `rec.right_value`, and the
  midpoint collapse when a rounding step inverts a constrained pair. The
  record carries the raw Newton values; the clamping logic in
  `grow_tree_gpu` lines 279-301 is unchanged and still applies.
- `child_bounds(...)` and `extend_branch(...)`: unchanged.

Two host helpers become dead in the GPU grower:

- `_count_left` — `rec.left.count` and `rec.right.count` are exact integers
  from the same histogram counts, so `n_left` / `n_right` come straight off
  the record. Leave `_count_left` in place for now; it has no other caller,
  so it can be deleted in the same commit or left for a cleanup pass.
- `_leaf_value` for GPU nodes — `rec.parent_value` is the same quantity. The
  CPU grower still needs it.

`value_feature` (`tree_features[0]`) disappears from the GPU path: the record
takes the parent's totals from slot 0, which *is* `tree_features[0]`.

### 5. The histogram-subtraction problem (read before benchmarking)

`grow_tree_gpu` builds the smaller child's histogram on the device and
derives the sibling on the host with `subtract_histogram`. That requires the
parent's histogram *on the host*, which is exactly the download this lane
exists to remove. The two cannot both be true, so the integration has to pick
an order:

- **Stage 1, correct and simple.** Build both children's histograms on the
  device (two `enqueue_leaf` calls) and search each there. Every histogram
  download disappears; the cost is one extra `O(n_rows * n_features)`
  histogram build per node in exchange for one removed
  `O(n_features * n_bins)` transfer plus a synchronization. On a wide, deep
  tree over few rows that is a loss; on the shapes the GPU backend is for it
  should be a win. **Measure it before claiming either.**
- **Stage 2, what the design wants.** A device-side subtraction kernel over
  the fixed-point buffer (exact Int32, so bit-identical to
  `subtract_histogram`) plus a small device histogram pool indexed by
  frontier leaf, so a parent's histogram stays on the device for its
  children. `GpuHistogramBuilder` currently owns exactly one `out_dev`, so
  this is a change to that struct's shape: `out_dev` becomes
  `n_pool_slots * 3 * hist_size` with an explicit slot argument on
  `enqueue_leaf`. That is a `histogram_gpu.mojo` change and belongs to
  whoever owns that file next.

Stage 1 is the honest first commit. Do not describe stage 1 as a speedup
until `bench/bench_train_gpu.mojo` says so.

### 6. The device-side frontier (what `max_records` is for)

Once stage 2 lands, the node-at-a-time loop collapses:

1. Each frontier leaf's search writes its own record slot
   (`searcher.enqueue(..., record=leaf_slot)`).
2. `searcher.enqueue_pick_best(n_frontier, record=scratch_slot)` selects the
   leaf to split next, on the device, with the same strictly-greater-gain,
   lower-index-wins rule the host loop uses.
3. The host downloads one record per *split*, not per *node*.

The remaining blocker for a fully device-side queue is the per-node
parameter staging: `set_allowed` and the float parameter block each go
through one pinned buffer that every node reuses, so a node's `enqueue` must
be followed by its `download` (or a `synchronize`) before the next node
stages. A device-resident queue needs a per-leaf slot in both, which is a
mechanical change to `allow_dev` (already indexed by slot; widen to
`max_records * n_features`) and `fparam_dev` (widen to
`max_records * PF_WORDS`), plus a slot argument on both kernels.

### 7. Re-exports

`src/mojoboost/__init__.mojo` is a shared hotspot; add, in the GPU block near
line 165:

```mojo
from .gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
)
```

Nothing else. `reference_search`, `decode_record`, and the `gpu_*` arithmetic
helpers are internal.

### 8. What needs no change

- **Serialization.** The record never reaches a model file. `serialize.mojo`,
  the C ABI, the CPython bindings, and the Python estimators are untouched.
- **`docs/LIGHTGBM_PARITY.md`.** No parameter's meaning changes and no
  behavior changes until the trainer is switched over. When it is, the entry
  to add is the Float32 note below, not a parity claim.
- **`device.mojo`.** Device selection is unaffected; `auto` still resolves to
  the CPU.
- **`gpu_tiling.mojo`.** The search launches `n_active` single-thread blocks
  and needs no tiling policy. It deliberately does not consult
  `HistogramTiling`.

## Semantics: what changes, and what the tests pin

The device search reproduces `find_best_split` candidate for candidate and
tie for tie. The tests assert that directly: every non-device test compares
`reference_search` against `find_best_split` on the same histogram, across
missing bins, both categorical searches, the interaction mask, feature
subsets, monotone constraints with and without output bounds, and both
rejection rules.

Two numeric differences are deliberate and will show up downstream:

1. **Float32.** Gains, hessian tests, leaf values, and the categorical sort
   keys are Float32, because Apple GPUs have no Float64 — the same constraint
   the histogram kernels already live under. Two candidates whose Float64
   gains differ below Float32 resolution can therefore resolve the other way,
   and a categorical sort key tie can order differently. **Consequence:**
   after integration, CPU/GPU tree shapes may differ on near-ties.
   `test_backend_equivalence.mojo` and `test_gpu_training.mojo` must stay
   tolerance-based on predictions and must not assert identical split
   choices. They already are tolerance-based on values; check the tree-shape
   assertions specifically.
2. **Exact accumulation.** Child sums accumulate in the histogram's
   fixed-point Int32 and dequantize once, where the host sums already
   dequantized Float64 values. Integer addition is associative and the scale
   bounds every partial sum, so the device sums are exact and reproducible —
   more accurate than the host's, not less, but not identical to them.

Both together: the device path is bit-deterministic run to run (asserted by
`test_device_search_is_deterministic`, and structurally guaranteed by the
absence of atomics and a fixed reduction order), and it is not bit-identical
to the host path.

## Benchmark counters to add (none were run)

`bench/bench_train_gpu.mojo` and `bench/bench_histogram.mojo` currently time
`stage_gradients`, `upload_staged`, `enqueue_leaf`, `download_raw`, and
`histogram_from_host`. The comparison this lane needs is:

- `download_raw` + `histogram_from_host` + host `find_best_split`, per node
  (the path being replaced), against
- `enqueue_search` (both kernels) + `download_record`, per node.
- Bytes transferred per node, both paths: `12 * n_features * n_bins` against
  `136`.
- Host synchronizations per node, both paths: 1 against 1 today, 1 against
  0 once the frontier stage lands.
- Extra histogram builds per node under stage 1 above: the count, and the
  wall clock they cost, so the subtraction tradeoff is a measured number and
  not an argument.

## Known limitations

- **The scan is one thread per feature.** Features are the parallel
  dimension; within a feature the scan is sequential, because the candidate
  order *is* the tie-breaking rule and a threshold scan is a prefix sum. The
  work is `O(n_features * n_bins)` against the histogram build's
  `O(n_rows * n_features)`, so this is the cheap half; parallelizing it (one
  thread per bin over a shared-memory prefix scan) cannot change a result,
  since the prefix sums are exact integers. Do that only if a profile says
  the scan is visible.
- **The categorical sort is an insertion sort**, `O(k^2)` on one thread, with
  `k` at most 255. Stable and ascending, matching `metrics._argsort`. Same
  reasoning: change it only if measured.
- **`max_cat_threshold` and `min_data_per_group` behave as on the host**, but
  the sorted walk's early `break`s are evaluated in Float32; a category set
  sitting exactly on a `min_child_hess` boundary can be admitted or rejected
  differently from the host.
- **Staging-buffer ordering contract**, described in section 6 above.
- **No GOSS or bagging interaction.** Both act on the leaf-assignment array
  before any histogram exists, so the search sees only their effect on the
  histogram, which is what the host scan sees too.
