# Quantized gradient training, and mojoboost's numerical policy for it

Status: **implemented, not integrated, not publicly reachable.** The policy
lives in `src/mojoboost/quantized_gradient.mojo`. No trainer, histogram
builder, split search, parameter surface, binding, or Python entry point
reaches it, `quantized_gradient.CONNECTED` is `False`, and every decision
procedure in the module returns the float path while it is. Nothing in this
document describes behavior a user can obtain today. The ordered patch set
that would change that is `handoffs/remaining_06_quantized_gradients.md`.

This document is the numerical contract. It says what a quantized round
would compute, what it would bound, where it would lose accuracy, and what
it would refuse to do. It states no performance claim, because no benchmark
has been run.

---

## 1. What quantized gradient training is

Ordinary GBDT accumulates a Float64 gradient and a Float64 hessian into
every histogram bin. Quantized training rounds each row's gradient and
hessian onto a small integer lattice once per boosting round, accumulates
those integers, and multiplies the surviving sums back by the lattice step
when it needs a gain or a leaf value. LightGBM added it in 4.0 behind
`use_quantized_grad`, from the paper *Quantized Training of Gradient
Boosting Decision Trees* (Shi et al., NeurIPS 2022).

Two things follow, and only the second is about speed.

**Integer accumulation is associative.** A histogram summed in any order,
by any number of threads, blocks, or ranks, is bit-identical. Float64
accumulation is not, which is why the CPU histogram builder has to fix a
feature-slice ownership rule to stay reproducible and why a distributed
all-reduce over Float64 histograms depends on the rank count.

**Integers are narrower.** Two Float64 per row becomes two small integers,
and a histogram bin becomes 16 or 32 bits instead of 64. That is a memory
traffic and bandwidth argument, and this repository has not measured it.

mojoboost already relies on the first property and has since the GPU
histogram landed. `src/mojoboost/histogram_gpu.mojo` quantizes gradients to
a 2^30 fixed-point lattice and accumulates with Int32 integer atomics,
because Metal has no float atomic add and because integer accumulation is
the only order-independent option portable across CUDA, ROCm, and Metal. So
mojoboost has shipped quantized accumulation on one backend for as long as
it has had a GPU. What it has never had is a *choice* of lattice.
`quantized_gradient.mojo` is that choice, expressed once, for both backends.

---

## 2. Parameters

| Parameter | LightGBM name | Default | Meaning |
|---|---|---|---|
| `enabled` | `use_quantized_grad` | `false` | Turns the whole thing on |
| `num_grad_quant_bins` | `num_grad_quant_bins` | `4` | Lattice width. Gradients occupy `[-B/2, B/2]`, hessians `[-B, B]` |
| `stochastic_rounding` | `stochastic_rounding` | `true` | Unbiased rounding instead of round-to-nearest |
| `renew_leaf` | `quant_train_renew_leaf` | `false` | Recompute leaf values from unquantized sums after growth |
| `seed` | none | `11` | Seeds the stochastic rounding stream. mojoboost only |
| `scale_rule` | none | `SCALE_MAX_ABS` | Which lattice rule. mojoboost only |
| `max_width` | none | `WIDTH_64` | Widest integer accumulator this caller can provide |

The last three are mojoboost's. `seed` exists because LightGBM's stochastic
rounding draws from a per-thread engine and is therefore not reproducible
across thread counts, which would break the CPU/GPU agreement this package
tests. `scale_rule` exists because the GPU path already has a rule of its
own and one policy has to describe both. `max_width` exists because a
device with a 32-bit integer atomic has to be able to say so and get a
documented fallback rather than a wrapped accumulation.

One deliberate refusal: an **odd** `num_grad_quant_bins` is rejected.
LightGBM computes `num_grad_quant_bins_ / 2` in integer arithmetic, so an
odd count truncates and the positive and negative halves of the gradient
lattice stop matching. mojoboost raises instead.

---

## 3. The two scale rules

Both express a lattice as **units per unit of value**, so `q = round(x *
units)` and `x` is recovered as `q / units`. That is the direction the
shipped device kernel already multiplies in (`gq = Int32(round(grad *
g_scale))` in `gpu_leaf_batching.mojo`), and it is the reciprocal of
LightGBM's `grad_scale_`, which is a step size. `QuantScales.grad_step`
converts.

### `SCALE_MAX_ABS`, LightGBM's rule

```
grad units = (B / 2) / max|g|          clamp to [-B/2, B/2]
hess units =  B      / max|h|          clamp to [-B,   B  ]
```

The hessian gets the full width because hessians are one-sided: every
built-in objective floors them at zero (the logistic and softmax ones at
1e-16) and `objective.check_custom_grad_hess` rejects a negative one, so the
negative half of a symmetric lattice would never be reached. The clamp is
still symmetric, so a negative hessian that somehow arrived is represented
rather than folded to zero.

The bound is **per row**. A node of `n` rows therefore bounds each bin at
`n * B/2` for gradients and `n * B` for hessians, which is what lets a
narrow accumulator be chosen from the node's size.

### `SCALE_MAGNITUDE_SUM`, the rule the GPU already ships

```
grad units = 2^30 / sum|g|             no clamp is reachable
hess units = 2^30 / sum|h|
```

The bound is on the **total**, not per row: any node's rows are a subset of
all rows, so no partial sum of scaled values can exceed 2^30 plus the
rounding residue. This is `histogram_gpu._fixed_scale` and
`gpu_objectives_native.device_fixed_scale`, which are the same expression
written twice today;
`quantized_gradient.fixed_point_scale` is the single definition both are
asked to call.

### They are not interchangeable

`SCALE_MAX_ABS` at `B = 4` is lossy by design: it is a compression scheme
whose accuracy cost the LightGBM paper measures. `SCALE_MAGNITUDE_SUM` at
2^30 loses less than Float32 does and exists to make an accumulation
order-independent, not to make it small. A CPU replica of a GPU histogram
(`histogram_cache_policy.ORIGIN_CPU_REPLICA`) must use the second rule, at
round-to-nearest, with `max_width = WIDTH_32`. That configuration has a name:
`QuantGradParams.fixed_point()`.

---

## 4. Rounding, and the residue that follows from it

| Mode | Rule | Residue per row | Bias |
|---|---|---|---|
| `ROUND_NEAREST` | `std.math.round(x)` | `<= 1/2` unit | Systematic |
| `ROUND_STOCHASTIC` | `floor(x + u)`, `u` uniform in `[0, 1)` | `< 1` unit | None |

Stochastic rounding is what makes quantized training work at four bins: a
value 0.3 of the way to the next unit lands there with probability 0.3, so a
sum of many quantized values converges on the exact scaled sum instead of on
a systematically rounded one. It costs twice the residue bound, and that is
not cosmetic:

```
SCALE_MAGNITUDE_SUM, deterministic:  |bin sum| <= 2^30 + n/2
SCALE_MAGNITUDE_SUM, stochastic:     |bin sum| <= 2^30 + n
```

`histogram_gpu.MAX_ROWS` is `Int32.MAX`. The deterministic bound at that row
count is `2^30 + (2^31 - 1)/2`, which floors to exactly `Int32.MAX`: the
shipped Int32 accumulator holds it with **zero slack**. The stochastic bound
is `2^30 + 2^31 - 1`, which does not fit. So enabling stochastic rounding on
the magnitude-sum rule requires either `n_rows <= 2^30` or a wider
accumulator. `accumulation_bound` computes the number and `accumulator_width`
turns it into `WIDTH_16` / `WIDTH_32` / `WIDTH_64` / `WIDTH_NONE`; nothing
narrows an accumulator a bound does not fit.

The shipped GPU path is the deterministic one, which is why it has never had
to make this choice.

### Deterministic seeds

The stochastic dither is counter-based, exactly as `bagging.mojo`,
`goss.mojo`, and `sampling.mojo` draw their samples:

```
stream = mix64 over (seed, round_index, class_index, plane)
u(r)   = mix64(stream + r) >> 11, scaled by 2^-53        in [0, 1)
```

A row's draw depends on `(seed, round, class, plane, row)` and on nothing
else: not on how many rows were quantized before it, not on the thread
count, not on the bagged or GOSS-sampled subset, and not on the backend.
Gradients and hessians draw from separate streams, so a row does not get one
dither applied twice. A GPU kernel computing the same three integer
multiplies gets the same `u`, because the mixing is exact 64-bit integer
arithmetic on every device this package targets.

---

## 5. Accumulator width

`width_for_bound` picks the narrowest of Int16, Int32, Int64 that holds the
bound, and `accumulator_width` caps that by `max_width`, returning
`WIDTH_NONE` when nothing fits. `WIDTH_NONE` is a fallback condition, not an
error.

The width is a **per node** question, deliberately. LightGBM promotes its
histogram bit width dynamically for the same reason: the root of a
million-row dataset needs a wide accumulator and the leaves near the
frontier, holding a few hundred rows each, do not, and the narrow ones are
where the memory traffic actually is.

The gradient plane is signed and the hessian plane is not, but one width is
chosen for both. That keeps a histogram one buffer in the
`[grad | hess | count]` layout `histogram_gpu` already uses, which is what
makes a whole node's histogram one kernel launch and one copy instead of
three of each.

The host representation is always Int64 and the width travels with the
histogram as a declaration. Three parallel host histogram types would be
three accumulation loops, three subtraction kernels, and three dequantizers
to keep in step, against a saving that only matters where the bytes actually
move.

---

## 6. The order of operations

```
fill grad/hess  ->  GOSS scaling  ->  gradient_stats  ->  derive_scales
                ->  quantize      ->  accumulate      ->  reconstruct
```

Every arrow is load-bearing.

**Sample weights** are folded into the derivatives before anything else sees
them (`boosting._fill_grad_hess_into` multiplies by `w`), so the lattice is
derived from weighted magnitudes. There is no way around the cost: one row
weighted 10^6 times another sets `max|g|`, and at `B = 4` every ordinary row
then quantizes to zero. `count_underflow` reports exactly that, in rows.

**GOSS** multiplies the sampled small-gradient rows in place
(`goss.apply_goss_scaling`), so a lattice derived before it would be too
small by up to the GOSS multiplier and every sampled row would clamp. GOSS's
own row importance `|g * h|` is computed on the unquantized values, because
at four bins a quantized importance is mostly ties and the top-`k` threshold
stops selecting anything meaningful.

**The row subset** passed to `gradient_stats` is the one the tree will
actually be grown on: a bag, a GOSS selection, or every row. Measuring the
full set would still be correct, since a subset's magnitudes are bounded by
the full set's; it would only waste resolution whenever the sampler excludes
the extremes.

**Every row is quantized**, including the ones the sampler dropped. That
costs one pass and buys the property that the quantized arrays are indexed
by row id like the float ones: no compaction, no second index space, and a
bagged and an unbagged round produce the same integer for the same row.

---

## 7. Reconstruction

Gains and leaf values are reconstructed by dequantizing the **surviving
totals** and calling the existing arithmetic. The gain formula is not
restated anywhere in `quantized_gradient.mojo`: `gain.leaf_score` and
`gain.soft_threshold_l1` are called, and `quantized_leaf_output` delegates
to `tree_parameters_extra.raw_leaf_output`, which is the same Newton step a
float leaf gets. That is two or three divisions per split candidate, not per
row.

Dequantizing *before* the L1 soft threshold is required, not a convenience.
`lambda_l1` is in gradient units, so thresholding an integer sum against it
would compare a lattice count with a real number and the penalty would scale
with the lattice.

`min_sum_hessian_in_leaf` goes the other way: `hessian_units_at_least`
converts it into lattice units, rounded **up**, so a search can compare
integers. Rounding up keeps the integer test at least as strict as the float
one, so quantization can never admit a leaf the float path would have
rejected. It can reject one the float path would have admitted, by less than
one lattice unit, which is the direction to err in for a minimum-support
guard.

### What quantization does not change

- **Counts.** The count plane was always exact integers. `min_data_in_leaf`,
  the leaf-count-dependent `path_smooth`, and every count-based guard behave
  identically.
- **Sibling subtraction.** In Float64 the subtraction trick is exact only up
  to cancellation, which bites hardest near the frontier where the parent
  and child sums are closest. In integers `parent - child` is the sibling's
  accumulation bit for bit. This is the one histogram operation quantization
  makes *more* accurate, at every bin count.
- **The model format.** Trees carry Float64 thresholds and leaf values
  whether or not the gradients that produced them were quantized. Nothing in
  `serialize.mojo`, `lgbm_model_io.mojo`, or the model dump changes.

---

## 8. Multiclass

One lattice per class, never one shared. Softmax gradients for a rare class
are orders of magnitude smaller than for a common one, and a shared max-abs
lattice would put every rare-class row on zero and the class would stop
being fitted at all. The GPU multiclass path already carries a scale per
class for the same reason (`gpu_multiclass_batch.mojo`), so this is the
existing shape rather than a new one. The class index is mixed into the
rounding stream too, so two classes do not share a dither sequence.

---

## 9. Leaf renewal, and the two renewals that must not compose

`leaf_renewal_mode` resolves three cases:

| Case | Mode | What happens |
|---|---|---|
| Objective already renews (`mae`, `quantile`, `mape`) | `RENEW_BY_OBJECTIVE` | `boosting._renew_leaf_values` rewrites every leaf from residuals. Residuals are not gradients and were never quantized, so this supersedes everything |
| `quant_train_renew_leaf` set, objective does not renew | `RENEW_FROM_FLOAT` | After growth, each leaf's value is recomputed from the unquantized gradient and hessian sums of its rows |
| Neither | `RENEW_NONE` | Leaf values stand as the quantized histogram produced them |

The first case exists because applying `RENEW_FROM_FLOAT` under a renewing
objective computes a Newton step that is immediately overwritten: one
pointless pass over every leaf's rows, per tree, forever.

---

## 10. Fallback

`decide` returns `MODE_FLOAT` with a named reason rather than failing, in
this order:

| Reason | When |
|---|---|
| `REASON_NOT_REQUESTED` | `enabled` is false. The ordinary case |
| `REASON_NOT_CONNECTED` | `CONNECTED` is false. Every case today |
| `REASON_BACKEND` | The caller's accumulation path does not exist |
| `REASON_NO_ROWS` | Nothing to quantize |
| `REASON_NON_FINITE` | A gradient or hessian is not finite |
| `REASON_DEGENERATE` | Every magnitude is below `MAGNITUDE_FLOOR` |
| `REASON_OVERFLOW` | No allowed accumulator width holds the bound |

Every fallback keeps training, at the numerics that ship today.

`check_supported` is the other half and is deliberately separate: it
**raises** when quantized training was explicitly asked for and this build
cannot provide it. That is the rule
`unified_memory_policy.resolve_from_env` already applies to a transfer
route, for the same reason: an explicit request that cannot be honored is
refused where it was made, not quietly downgraded somewhere the user will
never look. A parameter surface calls `check_supported` at parse time;
`decide` never calls it, because a round that falls back because one class's
gradients collapsed is not a configuration error.

---

## 11. What is unverified

Nothing in this module has been compiled, run, benchmarked, or
differentially tested. This lane was static inspection only. Specifically
unverified:

1. That the file compiles.
2. That `std.math.round`'s tie rule is what the prose assumes it might be.
   The module deliberately does not depend on which rule it is, only on the
   host and the device both calling the same function, but the claim that
   they call the same function has not been checked by compiling both.
3. Every accuracy statement about `SCALE_MAX_ABS` at small bin counts. Those
   are the LightGBM paper's results, restated, not this repository's.
4. Any speed claim. There is none in this document, and there should be none
   anywhere until a benchmark exists.
5. That `build_quantized_histogram_into` agrees with
   `histogram.build_histogram_subset_into` after dequantization to within the
   lattice step. That is the first differential test the handoff asks for.

## 12. Relationship to the parity contract

`docs/LIGHTGBM_PARITY.md` currently carries one row for this whole family:

> `use_quantized_grad` / `num_grad_quant_bins` / `quant_train_renew_leaf` /
> `stochastic_rounding` | deferred | Quantized-gradient training.
> Interesting for the GPU path; nothing depends on it today

That row is still accurate and must not be upgraded by this lane. The
capability is `implemented` and nothing more: not integrated, not publicly
reachable, not tested. `handoffs/remaining_06_quantized_gradients.md` names
the exact wording change and the exact patch that would earn it.
