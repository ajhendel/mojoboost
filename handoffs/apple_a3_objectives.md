# Handoff: device-side objectives (Apple task A3)

Per-row gradients, hessians, softmax probabilities, and the raw-score update,
computed on the device from device-resident labels, weights, and raw scores.
Implemented, tested, and wired into nothing. This file is the exact wiring
for whoever integrates it.

The point of the lane: `train_gpu` currently computes every round's
derivatives on the host and uploads `2 * n_rows` Float32 per round. Nothing
in that upload is new information. The labels and the weights never change,
and the raw scores change by an amount the device already knows, because the
leaf-assignment array left behind by tree growth is exactly "which leaf value
does each row get". Moving the derivatives onto the device removes the
upload entirely and replaces it with a 2 KB readback that a later step folds
into a transfer that already happens.

## Files this lane owns

| File | State |
|---|---|
| `src/mojoboost/gpu_objectives_native.mojo` | New. Six kernels plus `GpuObjectiveState`, 886 lines with docstrings. |
| `tests/parallel/test_gpu_objectives_native.mojo` | New. 12 tests, 767 lines. |
| `handoffs/apple_a3_objectives.md` | This file. |

Nothing else was touched. No edit to `objective.mojo`, `boosting.mojo`,
`histogram_gpu.mojo`, `train_gpu.mojo`, `device.mojo`, `pixi.toml`, or any
existing test. Nothing was committed or staged.

## Focused test

```sh
pixi run -q mojo run -I src tests/parallel/test_gpu_objectives_native.mojo
```

Result on this machine (Apple silicon GPU present, so no test skipped):

```
Running 12 tests
    PASS [   0.003 ] test_supports_device_objective
    PASS [ 108.005 ] test_device_grad_hess_matches_cpu
    PASS [  42.655 ] test_device_grad_hess_weights
    PASS [  32.942 ] test_device_gradient_finite_difference
    PASS [  19.649 ] test_device_hessian_finite_difference
    PASS [  20.679 ] test_device_softmax_matches_cpu
    PASS [  13.765 ] test_device_raw_update_matches_host
    PASS [   3.499 ] test_device_raw_update_is_per_class
    PASS [  15.632 ] test_magnitude_sums_match_host
    PASS [   0.020 ] test_device_fixed_scale_edges
    PASS [  10.531 ] test_device_exp_clamp_keeps_gradients_finite
    PASS [   5.579 ] test_device_objective_contract_errors
--------
Summary [ 272.959 ] 12 tests run: 12 passed, 0 failed, 0 skipped
```

`git diff --check --` on the three assigned paths reports nothing, though
all three are untracked, so that check inspects nothing. A direct grep for
trailing whitespace and tabs finds none in any of them, which is the claim
that actually has content.

Two notes on that run. It was launched directly rather than through
`tools/with_build_lock.sh`; use the lock wrapper when other lanes are
building. And the 273 s is kernel compilation, not arithmetic: each test
opens its own `DeviceContext` and the six kernels are compiled per context.
A trainer opens one context for the whole run, so this cost is a property of
the test file, not of the path.

It took two runs to get there, not one. The first failed to compile: the
kernels take `MutPointer`, so the borrowed `DeviceBuffer` arguments had to
become `mut grad_dev` / `mut hess_dev` / `mut leaf_dev`. That is why the
signatures below take them `mut`, and passing them immutably is a compile
error rather than a silent copy.

## What the module provides

```mojo
from mojoboost.gpu_objectives_native import (
    GpuObjectiveState, device_fixed_scale, supports_device_objective,
)

var state = GpuObjectiveState(ctx, target, sample_weight, n_classes, max_nodes)
state.init_raw(ctx, base_scores)                  # once per run
state.set_raw(ctx, raw)                           # init_score / continued training
state.fill_grad_hess(ctx, objective, alpha, grad_dev, hess_dev)
state.refresh_softmax(ctx)                        # multiclass, once per round
state.fill_softmax_grad_hess(ctx, k, grad_dev, hess_dev)
var sums = state.magnitude_sums(ctx, grad_dev, hess_dev)   # -> GradMagnitudes
state.update_raw(ctx, leaf_dev, tree.value, learning_rate, k)
var raw = state.download_raw(ctx)                 # renewal / metrics only
```

The `DeviceContext` is passed to every method rather than held, so these
buffers and `GpuHistogramBuilder`'s can be driven by the one context that
owns them both. `grad_dev` / `hess_dev` / `leaf_dev` are taken `mut` because
the kernel parameters are `MutPointer`; passing them immutably is a compile
error, not a silent copy.

Objectives with a device kernel: squared error, binary logistic, cross
entropy, poisson, gamma, tweedie, huber, quantile, L1, MAPE, fair, and
softmax multiclass. `supports_device_objective` is the single place that says
so. `CUSTOM` returns False and `fill_grad_hess` raises on it: the callback is
host code over host `List[Float64]`, there is no device image of it, and
`train_custom_gpu`'s contract is untouched by this lane. The host path is
preserved, not replaced.

## Buffer lifetimes

`GpuObjectiveState` owns seven device buffers and two pinned host buffers.
`n` is `n_rows`, `K` is `n_classes` (1 for single-output).

| Buffer | Size | Allocated | Written | Read | Freed |
|---|---|---|---|---|---|
| `target_dev` | `n` f32 | construction | construction only, through `map_to_host` | every gradient kernel | with the state |
| `weight_dev` | `n` f32, or 1 when unweighted | construction | construction only | every gradient kernel | with the state |
| `raw_dev` | `n*K` f32 | construction | `init_raw` / `set_raw` (host), `_update_raw_kernel` (device, once per tree) | gradient and softmax kernels, `download_raw` | with the state |
| `prob_dev` | `n*K` f32, or 1 when `K == 1` | construction | `_softmax_prob_kernel`, once per multiclass round | `_softmax_class_kernel`, once per class | with the state |
| `value_dev` | `max_nodes` f32 | construction | `update_raw`, once per tree, through `map_to_host` | `_update_raw_kernel` | with the state |
| `base_dev` | `K` f32 | construction | `init_raw` | `_init_raw_kernel` | with the state |
| `part_dev` | `2*256` f32 | construction | `_abs_sum_kernel`, once per round | copied to `host_part` | with the state |
| `host_part` | `2*256` f32 pinned | construction | device copy | host sum in `magnitude_sums` | with the state |
| `host_raw` | `n*K` f32 pinned | construction | device copy | `download_raw`, `download_grad_hess` | with the state |

Not owned, borrowed per call: `grad_dev` and `hess_dev` (the histogram
builder's, `n` f32 each) and `leaf_dev` (the builder's leaf-assignment array,
`n` i32).

Unchecked precondition: those three borrowed buffers must each hold at least
`n_rows` elements, and the module does not verify it, because a
`DeviceBuffer`'s length is not something these methods interrogate. A short
buffer is an out-of-bounds device read, not an error. Every caller in the
wiring below passes the builder's own buffers, which are allocated at
`n_rows` in `GpuHistogramBuilder.__init__`, so the precondition holds by
construction there; a new call site is where it could be broken.

Ordering rules that matter:

- `init_raw` or `set_raw` must run before any gradient kernel. `has_raw`
  enforces it; without the guard the first round would read an uninitialized
  `raw_dev`.
- `set_raw` calls `ctx.synchronize()` before mapping, because an enqueued
  `_update_raw_kernel` may still be writing `raw_dev`.
- `update_raw` must run **after** `grow_tree_gpu` returns and **before** the
  next round's `begin_tree`, which is what resets the leaf ids it reads.
- `refresh_softmax` must run before the round's first
  `fill_softmax_grad_hess`; every class of the round shares the one
  probability pass, exactly as on the host.
- `magnitude_sums` is the only method that synchronizes on the round's own
  work. `download_raw` and `download_grad_hess` also synchronize and are not
  part of a plain round.

Lifetime of the state itself: one per training run, constructed next to the
`GpuHistogramBuilder` and destroyed with it. It must not outlive the
`DeviceContext` its buffers came from.

## Exact integration

Three methods on `GpuHistogramBuilder` (histogram_gpu.mojo) and one import.
No kernel in that file changes, and no existing method changes.

```mojo
from .gpu_objectives_native import (
    DEFAULT_MAX_NODES, GpuObjectiveState, device_fixed_scale,
)

    def objective_state(
        mut self,
        target: List[Float64],
        sample_weight: List[Float64] = [],
        n_classes: Int = 1,
        max_nodes: Int = DEFAULT_MAX_NODES,
    ) raises -> GpuObjectiveState:
        """A device objective state on this builder's context, so its
        gradients land in this builder's buffers."""
        return GpuObjectiveState(
            self.ctx, target, sample_weight, n_classes, max_nodes
        )

    def fill_gradients_device(
        mut self,
        mut state: GpuObjectiveState,
        objective: Int,
        alpha: Float64,
    ) raises:
        """This round's gradients, computed on the device straight into the
        histogram buffers. Replaces `upload_gradients` for the built-in
        objectives; the fixed-point scales come from a device reduction
        instead of a host pass."""
        state.fill_grad_hess(
            self.ctx, objective, alpha, self.grad_dev, self.hess_dev
        )
        var sums = state.magnitude_sums(self.ctx, self.grad_dev, self.hess_dev)
        self.g_scale = Float64(device_fixed_scale(sums.grad))
        self.h_scale = Float64(device_fixed_scale(sums.hess))
        self.has_gradients = True

    def update_raw_device(
        mut self,
        mut state: GpuObjectiveState,
        values: List[Float64],
        learning_rate: Float64,
        k: Int = 0,
    ) raises:
        """Advance the device raw scores by the tree just grown, from the
        leaf assignments it left behind."""
        state.update_raw(self.ctx, self.leaf_dev, values, learning_rate, k)
```

A multiclass round adds `state.refresh_softmax(self.ctx)` once and
`state.fill_softmax_grad_hess(self.ctx, k, self.grad_dev, self.hess_dev)` per
class, with the same two scale lines after each.

Then `train_gpu` (train_gpu.mojo, lines 417 to 449) becomes:

```mojo
        var builder = GpuHistogramBuilder(data)
        var state = builder.objective_state(
            target, sample_weight, 1, 2 * params.tree.num_leaves
        )
        state.init_raw(builder.ctx, [base_score])
        # `raw` is no longer the host's running copy, so it stops being
        # allocated and filled with `base_score` up front; it is now just the
        # buffer the renewal branch downloads into.
        var raw = List[Float64]()
        for i in range(params.n_estimators):
            refresh_bag(bag, bagging, n, i)
            builder.fill_gradients_device(state, objective, alpha)
            var tree = grow_tree_gpu(builder, params.tree, bag, i)
            if renews:
                raw = state.download_raw(builder.ctx)
                _renew_leaf_values(
                    tree, data, target, raw, renew_w, renew_a, bag, signs
                )
            if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                ...
            builder.update_raw_device(state, tree.value, params.learning_rate)
            trees.append(tree^)
```

`_fill_grad_hess`, the `grad`/`hess` host lists, and the per-row `raw[r] +=
learning_rate * tree.predict_row(data, r)` loop all disappear from the round.
`raw` is only materialized where something host-side still reads it.

Deduplicating the scale, one edit in histogram_gpu.mojo so there is a single
definition rather than two that have to stay in step:

```mojo
def _fixed_scale(values: List[Float64]) raises -> Float32:
    var total = 0.0
    for i in range(len(values)):
        total += abs(values[i])
    return device_fixed_scale(total)
```

Behavior is unchanged: a non-finite element makes the total non-finite, which
is the condition `device_fixed_scale` raises on, and the floor and the
`Float32(FIXED_ONE / total)` expression are identical.

## Eliminating the per-round gradient upload

Per-round host-to-device traffic under the wiring above: **zero**. `target`
and `sample_weight` upload once at construction; `raw_dev` is updated by a
kernel, not a copy; the only host-to-device transfer left anywhere in the
loop is the node-value table, `(2 * num_leaves - 1) * 4` bytes once per tree
(244 bytes at the default `num_leaves = 31`), and it is per tree, not per
row. At 1M rows a round goes from 8 MB uploaded to 244 bytes.

`stage_g` and `stage_h` in `GpuHistogramBuilder` become dead for this path,
along with the Float64-to-Float32 conversion pass in `stage_gradients`. Leave
them allocated for `train_custom_gpu`, which still uploads, or make them lazy
and save `8 * n_rows` bytes of pinned host memory.

What is left is one **device-to-host** transfer per round: the 2 KB of
threadgroup partials `magnitude_sums` reduces on the host, plus the
`ctx.synchronize()` it needs, because the fixed-point scale is a scalar
kernel argument to the histogram kernels and so has to be known host-side
before the first `enqueue_leaf`. To remove that too:

1. Add a `scale_dev: DeviceBuffer[DType.float32]` of 2 elements to
   `GpuHistogramBuilder`, and a `_scale_kernel` (one threadgroup) that
   reduces `part_dev`'s 512 partials and writes `FIXED_ONE / total` for each
   plane, with the same floor and the same clamp `device_fixed_scale`
   applies.
2. Change `_hist_leaf_kernel` and `_hist_partial_kernel` to take
   `scales: MutPointer[Float32, MutAnyOrigin]` in place of the two
   `g_scale: Float32, h_scale: Float32` arguments, and read `scales[0]` and
   `scales[1]`. Both kernels already multiply by them per row; this only
   moves where the value comes from.
3. `histogram_from_host` still needs `1/g_scale` and `1/h_scale` to
   dequantize. It runs after `download_raw`, which already synchronizes once
   per node, so copy `scale_dev` in the same `download_raw` (extend
   `host_out` by 2 slots) and read the scales from there.

That leaves the round with no host synchronization of its own: every
transfer is folded into the per-node histogram download that already exists.
It is a change to histogram_gpu.mojo's kernel signatures, which is why this
lane did not make it.

## What still forces a host round trip

Honest list, so nobody integrates this expecting more than it gives:

- **Quantile and L1 leaf renewal.** `_renew_leaf_values` needs the host raw
  scores and the host target, so a renewing objective pays one
  `download_raw` (`4 * n_rows` bytes) per tree. Renewal is a per-leaf
  weighted percentile, i.e. a sort; putting it on the device is its own lane.
- **GOSS.** `goss_round` ranks rows by gradient magnitude on the host, so it
  needs the gradients back. `download_grad_hess` exists for exactly that,
  but it costs `8 * n_rows` per round and gives up the lane's whole benefit.
  Keep the host path when `goss.enabled` until there is a device-side
  ranking pass.
- **Row bagging.** `begin_tree(bag)` parks out-of-bag rows at `OUT_OF_BAG`,
  so tree growth never routes them and `update_raw` leaves their scores
  untouched, which is wrong: an out-of-bag row still gets a prediction from
  the tree. Two ways out. Either replay the tree device-side after growth
  (`builder.begin_tree([])`, then `builder.apply_split(...)` once per split
  in the order they were applied, recording them during growth, then
  `update_raw_device`), which costs `n_splits` partition-kernel launches over
  all rows; or score the out-of-bag rows on the host and `set_raw` the whole
  vector. The replay path is the one worth building, since it also gives the
  device a correct leaf assignment for every row at the end of a tree.
- **Validation sets, metrics, and early stopping.** Validation rows are not
  in the device state at all. They stay host-side.

## Verification status

Verified by running the focused test on this machine's Apple GPU:

- Every device objective matches `fill_grad_hess` row by row, on inputs
  rounded through Float32 first (`_f32`) so both backends see identical
  values and the comparison measures kernel arithmetic. Bound is `1e-5`
  relative with a `1e-6` floor.
- Every gradient matches a central difference of an independently written
  per-row loss (`_row_loss` in the test file, not reused from
  boosting.mojo), and the hessians of the six objectives whose hessian is
  the true second derivative match a second central difference. Huber,
  quantile, L1 and MAPE are excluded from the hessian check because
  LightGBM's hessian for them is the row weight, and poisson because its
  hessian is inflated by `max_delta_step`; those four are pinned against the
  CPU reference instead.
- Weights, including zero-weight rows, which produce an exactly zero
  gradient and hessian.
- Softmax against `_softmax_inplace` + `_fill_softmax_grad_hess`, all four
  classes.
- The raw update against the host loop, including unrouted rows and a second
  tree accumulating on the first, and the per-class multiclass update.
- The magnitude reduction against a host sum, its bit-identical repeat, and
  the scale derived from it.
- The exponent clamp at raw scores of +/-500, where Float32 `exp` would
  overflow.
- Twelve refusals: custom objectives, unknown codes, wrong lengths,
  out-of-range classes, oversized node tables, gradients before `init_raw`.

Not verified, and nobody should claim otherwise:

- **Nothing here has been run on CUDA or ROCm.** The kernels use only shared
  memory, `barrier()`, and plain global loads and stores; no warp shuffles,
  no float atomics, no vendor intrinsics. That is the same portable baseline
  histogram_gpu.mojo holds to, but it is an argument, not a measurement.
- **No end-to-end training runs through this module**, because the wiring
  above was not applied. CPU/GPU training parity per objective is still what
  `tests/test_gpu_objectives.mojo` measures, over host-computed gradients.
- **No benchmark.** The traffic accounting above is arithmetic on buffer
  sizes. Whether removing the upload moves `bench/bench_train_gpu.mojo` on
  any device is unmeasured, and the M4 GPU trainer is still slower than the
  CPU one for reasons this lane does not address (see the row-compaction
  note in histogram_gpu.mojo).
- **`Float32` gradients change training, not just transfer.** The device
  path computes derivatives in Float32 where the CPU computes them in
  Float64. Histograms were already Float32, but the gradients feeding them
  were not; whether that moves a split on real data is unmeasured. The
  place to measure it is `bench/bench_gpu_validation.mojo`.
- **The exponent clamp is a deliberate divergence** from both the CPU path
  and LightGBM. It only engages past `|raw| = 60`, where the model has
  diverged, and it turns an infinity into a large finite number so the
  fixed-point scale reports a real magnitude instead of raising. If it ships,
  it belongs in `docs/LIGHTGBM_PARITY.md`.

## Suggested next lane

1. Apply the three builder methods and the `train_gpu` rewrite above, behind
   nothing (the host path stays for CUSTOM and GOSS), and run
   `tests/test_gpu_training.mojo` and `tests/test_gpu_objectives.mojo`
   unchanged. They are the parity suites; if the device gradients are right
   they pass as they stand.
2. Add the device-side split replay so bagging works, since bagging is on by
   default in a lot of real configurations.
3. Then the scale-on-device change, which is what removes the last
   synchronization from a round.
