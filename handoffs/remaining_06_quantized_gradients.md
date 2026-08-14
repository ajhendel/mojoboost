# Handoff, remaining-parity task 06, quantized gradient training

Lane 06 of the remaining-parity round. Three files were written and nothing
else in the repository was touched. Nothing was compiled, run, benchmarked,
profiled, or committed; the round's rule was static inspection only and it
was kept.

| Path | What it is |
| --- | --- |
| `src/mojoboost/quantized_gradient.mojo` | New. The whole policy: parameters, the two scale rules, rounding and its seeds, overflow bounds and accumulator width, the integer histogram, reconstruction, renewal, multiclass, diagnostics, and the fallback decision. ~1850 lines, mostly docstrings |
| `docs/QUANTIZED_GRADIENTS.md` | New. The numerical contract in prose, plus what is unverified |
| `handoffs/remaining_06_quantized_gradients.md` | This file |

**Nothing reaches the module.** It is not exported from
`src/mojoboost/__init__.mojo`, no trainer imports it, and
`quantized_gradient.CONNECTED` is `False`, which makes `decide` return the
float path for every input regardless of parameters. That last one is the
gate: it is not a to-do marker, it is what guarantees no configuration can
route a training run through an unvalidated integer accumulator. Flipping it
is step 9 of section 3, after everything else and after the differential
test in section 5 passes.

**A note about the shared checkout.** This lane staged and committed
nothing, but another session ran a repository-wide commit while it was
writing, and `5085097` swept an **intermediate** copy of
`src/mojoboost/quantized_gradient.mojo` into history: in that snapshot
`QuantTotals` is declared after `QuantizedHistogram`, which uses it as a
return type. Read the working tree, or any commit after this handoff, not
`5085097`. The docs file and this handoff were swept in their finished
state.

Capability level, in the vocabulary of `docs/CAPABILITY_LEVELS.md`:
`implemented`, and nothing beyond it. Not integrated, not publicly
reachable, not focused-tested, not differential-tested, not
hardware-validated, not release-packaged. The parity row must not move.

---

## 1. What was inspected, and the four findings that shaped the design

Read before writing anything: `histogram.mojo` (the float accumulator and
its parallelization contract), `histogram_gpu.mojo` (the shipped fixed-point
path), `gpu_objectives_native.mojo` (the device-side scale), `gpu_leaf_batching.mojo`
(the actual quantizing kernel), `gpu_multiclass_batch.mojo` (per-class
scales), `histogram_cache_policy.mojo` (what invalidates a cached histogram),
`boosting.mojo` (gradient storage, the round loop, leaf renewal),
`goss.mojo`, `bagging.mojo`, `sampling.mojo`, `tree_parameters_extra.mojo`
(the four splitmix64 copies), `gain.mojo`, `split.mojo`, `tree.mojo`,
`distributed.mojo`, `params.mojo`, `bindings/_mojoboost.mojo`,
`python/mojoboost/__init__.py`, `python/mojoboost/basic.py`,
`docs/LIGHTGBM_PARITY.md`, `docs/INTEGRATION_INVENTORY.md`,
`tools/connectivity_audit.py`.

**F1. mojoboost has shipped quantized accumulation on the GPU since the GPU
landed, and calls it something else.** `histogram_gpu.mojo` quantizes to a
2^30 lattice (`_fixed_scale`), rounds with `std.math.round` inside the
kernel (`gpu_leaf_batching.mojo:380`, `:492`), accumulates Int32, and
dequantizes on download (`histogram_from_host`). That is quantized gradient
training with one hard-coded lattice. So this lane's job was never "add
quantization"; it was "make the lattice a parameter, and keep one policy
describing both the existing rule and LightGBM's". `SCALE_MAGNITUDE_SUM` is
the existing rule, named; `QuantGradParams.fixed_point()` is the shipped
configuration, spelled out.

**F2. The shipped Int32 bound is tight, with zero slack, and stochastic
rounding breaks it.** `histogram_gpu.mojo` argues that scaled accumulation
"cannot overflow" because a leaf's rows are a subset of all rows. That is
true for the exact scaled sum, but the accumulated values are *rounded*, and
`n` roundings add up to `n/2` on top of 2^30. At `MAX_ROWS = Int32.MAX` that
is `2^30 + (2^31 - 1)/2`, which floors to exactly `Int32.MAX`. It fits, and
it fits by nothing. Under stochastic rounding the residue is a full unit per
row, the bound becomes `2^30 + 2^31 - 1`, and it does not fit.
`accumulation_bound` and `residue_per_row` exist because of this, and
`decide` returns `REASON_OVERFLOW` rather than narrowing silently. This is
the single most consequential finding in the lane, and it is a latent
constraint on the *shipping* code, not only on the new module: the shipped
kernel is safe only because it rounds to nearest.

**F3. The fixed-point scale is written twice, in two modules a CPU-only
build cannot import.** `histogram_gpu._fixed_scale` and
`gpu_objectives_native.device_fixed_scale` are the same expression, and
`device_fixed_scale`'s own docstring says so and asks for the single
definition. Both live behind `from max.gpu.host import ...`, so no CPU path
could ever have called them. `quantized_gradient.fixed_point_scale` is that
single definition, GPU-free. Patch 3.2 makes the two call it. This is the
"fuse duplicate implementations" half of the round's final phase, and it is
the one fusion this lane can actually name a caller for.

**F4. splitmix64 now exists five times.** `bagging._splitmix64`,
`goss._splitmix64`, `sampling._splitmix64`,
`tree_parameters_extra._mix64`, and now `quantized_gradient._mix64`.
`tree_parameters_extra`'s copy carries a comment saying it is repeated
rather than imported so the module stays free of another module's private
names, and that the handoff asks for one shared copy. This lane made the
count worse rather than better, deliberately: the shared home has to be a
module all five can import, `bagging`/`goss`/`sampling` are all imported by
`boosting`, and picking the home is a decision that belongs to whoever owns
those files. Patch 3.10 states it; it is the lowest-priority item here and
changes no behavior.

---

## 2. What the module holds

Grouped, with the names a patch will reference.

- **Gate.** `CONNECTED`, `decide`, `check_supported`, `QuantDecision`,
  `describe_reason`, `describe_decision`.
- **Parameters.** `QuantGradParams` with `default()` (disabled, LightGBM's
  values), `lightgbm(...)`, and `fixed_point()` (the shipped GPU rule).
  `validate()` range-checks and refuses an odd `num_grad_quant_bins`.
- **Statistics.** `GradientStats`, `gradient_stats(grad, hess, rows)`,
  `combine_stats(a, b)` (associative, for threads and for ranks).
- **Scales.** `magnitude_sum`, `fixed_point_scale`, `QuantScales`,
  `derive_scales`, `derive_multiclass_scales`.
- **Rounding.** `_mix64`, `quant_uniform`, `quant_stream`, `QuantRoundKey`,
  `quantize_scalar`, `quantize_rows`.
- **Bounds.** `residue_per_row`, `accumulation_bound`, `width_for_bound`,
  `accumulator_width`.
- **Histogram.** `QuantizedHistogram`, `QuantTotals`, `check_same_lattice`,
  `subtract_quantized`, `accumulate_quantized`,
  `build_quantized_histogram_into`, `build_quantized_histogram`.
- **Reconstruction.** `quantized_leaf_score`, `quantized_split_gain`,
  `quantized_leaf_output`, `hessian_units_at_least`.
- **Renewal.** `leaf_renewal_mode`, `renewed_leaf_output`.
- **Diagnostics.** `count_underflow`, `lattice_resolution_bits`.

Imports, and the reason each one is safe: `.binning` (BinnedMatrix),
`.gain` (`leaf_score`, `soft_threshold_l1`), `.histogram` (`Histogram`,
`_zeroed_f64`, `_zeroed_int`), `.parallel` (`dispatch_feature_ranges`),
`.tree_parameters_extra` (`raw_leaf_output`). None of those import
`max.gpu.*`, so the module compiles in a CPU-only build, and none of them
import `boosting`, so `boosting` can import *this* module in patch 3.4
without a cycle. `objective_renews_leaves` is passed to
`leaf_renewal_mode` as a `Bool` rather than imported, for exactly that
reason.

---

## 3. Ready-to-apply integration patches

Ordered. Each step leaves the tree working, and every step before 3.9 leaves
quantized training unreachable. Every validation line is marked **UNRUN**:
this lane ran nothing.

### 3.1 Register the module with the connectivity audit (do first, no behavior)

**Target** `tools/connectivity_audit.py`, the `CLASSIFICATION` dict
(begins at line 208).
**Ownership** not this lane's. **Dependency** none.
**Patch** add, in the dict's alphabetical position:

```python
    "quantized_gradient": (
        PENDING,
        "remaining_06",
        "Quantized-gradient policy. CONNECTED is False and no trainer, "
        "histogram builder, or parameter surface reaches it.",
    ),
```

**Target** `docs/INTEGRATION_INVENTORY.md`, the "Orphan native modules"
table (begins line 50).
**Patch** add the row, keeping alphabetical order (between `lgbm_model_io`
and whatever follows):

```
| `quantized_gradient` | PENDING | remaining_06 | Quantized-gradient policy. `CONNECTED` is `False` and no trainer, histogram builder, or parameter surface reaches it |
```

**State flow** none. **Errors** none. **Fallback** none.
**Serialization effect** none. **Public API effect** none.
**Validation, UNRUN**: `python3 tools/audit_integration.py` reports no new
GAP; `python3 tools/connectivity_audit.py --section orphans` lists the
module with owner `remaining_06`.

Without this patch the module is a *new unclassified orphan*, which
`audit_integration.py` reports as a GAP and `--strict` turns into a failure.
That is the only way this lane's files can break an existing check, so it is
first.

### 3.2 Fuse the duplicated fixed-point scale (finding F3)

**Target A** `src/mojoboost/gpu_objectives_native.mojo`.
Delete `comptime FIXED_ONE` (line 117) and the body of
`device_fixed_scale` (line 155), and replace with a re-export:

```mojo
from .quantized_gradient import FIXED_ONE, fixed_point_scale

def device_fixed_scale(total: Float64) raises -> Float32:
    """The fixed-point histogram scale for a magnitude sum.

    `quantized_gradient.fixed_point_scale` is the definition; this name
    stays because the GPU modules import it and because it says which side
    of the boundary the magnitude sum came from.
    """
    return fixed_point_scale(total)
```

**Target B** `src/mojoboost/histogram_gpu.mojo`.
Delete `comptime _FIXED_ONE` (line 186) and replace `_fixed_scale`
(line 226) with:

```mojo
from .quantized_gradient import fixed_point_scale, magnitude_sum

def _fixed_scale(values: List[Float64]) raises -> Float32:
    """Fixed-point scale from a host-side list of values. The pass and the
    scale are `quantized_gradient.magnitude_sum` and
    `quantized_gradient.fixed_point_scale`; this is the two-call spelling
    the two call sites in `stage_gradients` want."""
    return fixed_point_scale(magnitude_sum(values))
```

**Signature** unchanged in both cases. **Call sites** unchanged:
`histogram_gpu.mojo:558-559` and `:624-625`, `:640-641`.
**Ownership** not this lane's (GPU files).
**Dependency** none beyond the new module existing.
**State flow** none; both functions are pure.
**Errors** the error *strings* change by one word. `fixed_point_scale`
raises `"gradient/hessian magnitudes are out of range for the fixed-point
histogram"`, dropping `" GPU"` from the middle, because the function is no
longer GPU-specific. If any test matches that string it must be updated;
grep for `out of range for the` before applying.
**Fallback** none. **Serialization effect** none.
**Public API effect** none (`device_fixed_scale` is not exported from
`__init__.mojo`).
**Risk** this pulls `quantized_gradient` (and therefore `binning`,
`histogram`, `parallel`, `gain`, `tree_parameters_extra`) into the import
closure of the GPU modules. All five are already in it via
`histogram_gpu`'s own `from .histogram import ...` and `from .binning import
...`, so the closure does not grow. Verify before applying.
**Validation, UNRUN**: `pixi run test-histogram-gpu` (or whichever pixi task
covers `tests/test_histogram_gpu.mojo`) still passes on a GPU host, and one
CPU-only build still compiles.

This patch is worth applying **on its own**, independent of everything
below. It removes a duplication the tree already documented as unwanted, and
it does not enable anything.

### 3.3 Parameters

**Target** `src/mojoboost/params.mojo`.
**Ownership** not this lane's. **Dependency** 3.4 (the field must exist on
`BoosterParams` before the parser can fill it).

1. Extend `SUPPORTED_KEYS` (line 66), appending to the last string segment:

```
    " tweedie_variance_power, device, use_missing, use_quantized_grad,"
    " num_grad_quant_bins, stochastic_rounding, quant_train_renew_leaf"
```

2. In `parse_params` (line 444), add four arms alongside the existing
   `_parse_bool` / `_parse_int` arms:

```mojo
        elif key == "use_quantized_grad":
            config.booster.quant.enabled = _parse_bool(key, value)
        elif key == "num_grad_quant_bins":
            config.booster.quant.num_grad_quant_bins = _parse_int(key, value)
        elif key == "stochastic_rounding":
            config.booster.quant.stochastic_rounding = _parse_bool(key, value)
        elif key == "quant_train_renew_leaf":
            config.booster.quant.renew_leaf = _parse_bool(key, value)
```

3. In `_validate` (line 378), after the existing checks:

```mojo
    check_supported(config.booster.quant)
```

   with `from .quantized_gradient import check_supported` at the top.

**State flow** the parsed bundle rides on `TrainConfig.booster.quant` and
reaches the trainer with the rest of `BoosterParams`. No new field on
`TrainConfig` itself.
**Errors** `QuantGradParams.validate` raises on a bin count outside
`[2, 1048576]` and on an odd one; `check_supported` raises
`"use_quantized_grad is not available in this build, quantized gradient
training is not connected to a trainer in this build"` for
`use_quantized_grad=true` while `CONNECTED` is `False`. That refusal is the
point: an explicit request is refused where it was made, which is the rule
`unified_memory_policy.resolve_from_env` already applies to a transfer
route.
**Fallback** none at this layer. Silent downgrade is what is being avoided.
**Serialization effect** none. `serialize.mojo` writes trees, the base
score, the objective, the bin mapper, and feature names; it does not write
`BoosterParams`, so a training-time parameter cannot change the model
format. Confirmed by reading `serialize.mojo`; re-confirm with
`grep -n BoosterParams src/mojoboost/serialize.mojo` before applying.
**Public API effect** four new accepted keys on the parameter-string API and
on the CLI that parses it. They are accepted and immediately refused while
`CONNECTED` is `False`, which is a deliberate intermediate state: it makes
the refusal discoverable instead of reporting the keys as unknown.
**Validation, UNRUN**: `tests/test_params.mojo` gains a case asserting that
`"use_quantized_grad=true"` raises with the not-connected message, and that
`"use_quantized_grad=false num_grad_quant_bins=8"` parses.

### 3.4 The trainer

**Target** `src/mojoboost/boosting.mojo`.
**Ownership** not this lane's. **Dependency** none (this is the root of the
chain).

1. `BoosterParams` (line 707). Append one field, so every positional caller
   keeps working:

```mojo
@fieldwise_init
struct BoosterParams(Copyable, Movable):
    var n_estimators: Int
    var learning_rate: Float64
    var tree: TreeParams
    var quant: QuantGradParams          # appended

    @staticmethod
    def default() -> BoosterParams:
        return BoosterParams(
            100, 0.1, TreeParams.default(), QuantGradParams.default()
        )
```

   Appending is what `TreeParams` did for `extra` and for `cat`, and for the
   same reason: `bindings/_mojoboost.mojo:362` constructs `BoosterParams`
   positionally with three arguments and must keep compiling until 3.8.
   **If `@fieldwise_init` in this Mojo version does not synthesize a default
   for a trailing field**, patch 3.8 becomes a hard dependency of this one
   and must be applied in the same change.

2. `_boost_rounds` (line 919). Inside the round loop, after
   `goss_round(...)` at line 975 and before `grow_tree` at line 976:

```mojo
        var q_stats = gradient_stats(grad, hess, bag)
        var q = decide(
            q_stats, params.quant, q_stats.n_rows, backend_supported=True
        )
        if q.is_quantized():
            quantize_rows(
                grad, hess, q.scales, params.quant,
                QuantRoundKey.single(params.quant.seed, round),
                qgrad, qhess,
            )
        var tree = grow_tree(
            data, grad, hess, params.tree, bag, round,
            quant=q, qgrad=qgrad, qhess=qhess,
        )
```

   with `var qgrad = List[Int64]()` and `var qhess = List[Int64]()` hoisted
   beside `grad`/`hess` at line 958, so the run allocates them once.

   **The order is the contract and must not be rearranged.** `gradient_stats`
   after `goss_round`, because `goss.apply_goss_scaling` multiplies the
   sampled rows in place and a lattice derived before it would clamp every
   one of them. `bag` as the row subset, because that is the set the tree
   will be grown on. `round` (the absolute index, already computed at line
   967) into the key, so a continued run dithers as an uninterrupted one
   would.

3. Leaf renewal. At line 977, the existing `if renews:` block becomes:

```mojo
        var renewal = leaf_renewal_mode(params.quant, renews)
        if renewal == RENEW_BY_OBJECTIVE:
            _renew_leaf_values(...)          # unchanged
        elif renewal == RENEW_FROM_FLOAT:
            _renew_leaf_values_from_gradients(
                tree, data, grad, hess, bag, params.tree, params.tree.extra
            )
```

   `_renew_leaf_values_from_gradients` does not exist and is this patch's
   only new function in `boosting.mojo`: one pass over the bag accumulating
   the unquantized `(g, h)` per leaf, then
   `quantized_gradient.renewed_leaf_output` per leaf, then the same
   `finish_leaf_output` and `bounds[node].clamp` tail
   `_renew_leaf_values` already applies (lines 694-704). It must reuse that
   tail rather than restate it.

4. `_boost_rounds_multiclass` (line 1482). The same three edits, with
   `derive_multiclass_scales` over the per-class stats and
   `QuantRoundKey(seed, round, k)` per class. One lattice per class, never a
   shared one; see section 8 of the design doc.

**State flow** `q` and the two integer buffers live for one round and are
handed down to `grow_tree`. Nothing is stored on the model.
**Errors** `gradient_stats` raises on a mismatched length or an
out-of-range row id; `quantize_rows` raises on an unusable lattice.
Non-finite gradients are *not* raised on: `GradientStats.finite` records
them and `decide` returns `REASON_NON_FINITE`, so the float path handles
them exactly as it does today.
**Fallback** total. When `decide` returns `MODE_FLOAT` the quantized buffers
are never filled and `grow_tree` takes the path it takes today, byte for
byte.
**Serialization effect** none.
**Public API effect** one appended field on `BoosterParams`, which is a
public Mojo type. A keyword-only construction stays source-compatible; a
four-positional one is new.
**Validation, UNRUN**: with `CONNECTED` still `False`, the full Mojo suite
passes unchanged, because `decide` returns `MODE_FLOAT` for every input and
the new code is a branch never taken. That is the point of gating on a
`comptime` bool: patches 3.4 through 3.8 can all land and be reviewed while
provably changing nothing.

### 3.5 The grower

**Target** `src/mojoboost/tree.mojo`, `grow_tree` (line 758).
**Ownership** not this lane's. **Dependency** 3.4.

**Signature** append three defaulted parameters, so every existing caller
(`boosting.mojo:976`, `distributed.mojo`, `train_gpu.mojo`, the tests)
compiles unchanged:

```mojo
def grow_tree(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
    quant: QuantDecision = QuantDecision.floating(REASON_NOT_REQUESTED),
    qgrad: List[Int64] = [],
    qhess: List[Int64] = [],
) raises -> Tree:
```

**Call site** inside the grower, wherever a node's histogram is built:
build `QuantizedHistogram` via `build_quantized_histogram_into` when
`quant.is_quantized()`, and the float `Histogram` otherwise. The sibling
subtraction switches to `subtract_quantized` in the same branch, and it is
*exact* there, which is a strict improvement over the float trick and worth
a comment at the call site.

**The decision is per tree, not per node.** `decide` is called once in
`_boost_rounds` against the root's row count, and every node of the tree
inherits it. Two nodes of one tree on two lattices would make sibling
subtraction meaningless, and `check_same_lattice` would (correctly) raise.
**Errors** `check_same_lattice` raises if a stale histogram from a previous
round reaches a subtraction. That is the failure
`histogram_cache_policy` describes for its own cached planes, and here it is
caught rather than silently wrong.
**Fallback** the default argument is a float decision, so an untouched
caller is untouched.
**Serialization effect** none. **Public API effect** three appended
defaulted parameters on an exported symbol.
**Validation, UNRUN**: the existing `tests/test_tree.mojo` passes unchanged
(every call takes the defaults).

### 3.6 The histogram builders

**Target** `src/mojoboost/histogram.mojo`.
**Ownership** not this lane's. **Dependency** 3.5.

Move `build_quantized_histogram_into` and `build_quantized_histogram` out of
`quantized_gradient.mojo` and into `histogram.mojo`, and put them under
`apple_cpu_policy.derive_accumulation_plan` like the float builder.

They live in `quantized_gradient.mojo` today only because this lane may not
edit `histogram.mojo`. They are deliberately *not* a copy of the float
builder: the pair gather and the two-features-per-inner-loop grouping are
tuned against Float64 pair loads with a Float64 read-modify-write, and an
Int64 pair is a different memory shape with a different crossover.
Restating the plan there would have been a second tuning policy to keep in
step with the first. The right end state is **one templated accumulation
kernel** over the element type, under one plan, with the quantized instance
and the float instance sharing the feature-slice ownership rule, the
zero-inside-the-task rule, and the work estimate.

After the move, `quantized_gradient.mojo` re-exports the two names so any
caller written against them keeps working, and the `.binning` and
`.parallel` imports there can go.

**Errors** unchanged. **Fallback** none. **Serialization effect** none.
**Public API effect** two new names in `histogram.mojo`, exported only if
`__init__.mojo` is patched (3.9).
**Validation, UNRUN**: the differential test in section 5.

### 3.7 The split search

**Target** `src/mojoboost/split.mojo`, `find_best_split` (line 287); and
`src/mojoboost/categorical.mojo`, `find_best_categorical_split`.
**Ownership** not this lane's. **Dependency** 3.6.

Two ways in, and the second is the one worth doing.

*Minimal:* call `QuantizedHistogram.dequantize()` before the existing
search. Correct, one allocation and one pass per node, and it throws away
the entire point of quantizing, which is that the scan never touches a
float. Acceptable as a first landing to get the differential test green.

*Real:* add a scan that walks `List[Int64]` prefix sums and calls
`quantized_split_gain` on the three surviving totals per candidate.
`min_sum_hessian_in_leaf` is compared with `hessian_units_at_least`, which
rounds up so the integer test is never laxer than the float one.
`min_data_in_leaf` is compared against the count plane, which was always
exact and is untouched. `lambda_l1` must be applied **after**
dequantization, never to an integer sum: it is in gradient units, so
thresholding a lattice count against it would make the penalty scale with
the lattice. `quantized_leaf_score` enforces that ordering by construction.

**Errors** `quantized_leaf_score` raises on an unusable lattice.
**Fallback** the float overload is untouched and is what a float decision
uses. **Serialization effect** none. **Public API effect** one new
overload on an exported symbol.
**Validation, UNRUN**: section 5's differential test, plus a check that the
integer scan and the dequantize-then-scan path pick the same split on a
fixture where no two candidate gains are within one lattice step.

### 3.8 The bindings and the Python surface

**Target** `bindings/_mojoboost.mojo`, `_parse_params` (line 343).
**Ownership** not this lane's. **Dependency** 3.3, 3.4.

```mojo
    return BoosterParams(
        Int(py=params["n_estimators"]),
        Float64(py=params["learning_rate"]),
        tree^,
        QuantGradParams(
            Int(py=params["use_quantized_grad"]) != 0,
            Int(py=params["num_grad_quant_bins"]),
            Int(py=params["stochastic_rounding"]) != 0,
            Int(py=params["quant_train_renew_leaf"]) != 0,
            Int(py=params["quant_seed"]),
            SCALE_MAX_ABS,
            WIDTH_64,
        ),
    )
```

Booleans cross as ints, which is the convention `_parse_use_missing`
(line 374) and the `goss` key already use: the boundary carries no Python
bool conversion.

**Target** `python/mojoboost/__init__.py`, the native-params dict
(line 1628) and the estimator `__init__` (line 1151).

```python
            # int, not bool: the binding reads it as an integer.
            "use_quantized_grad": int(bool(self.use_quantized_grad)),
            "num_grad_quant_bins": int(self.num_grad_quant_bins),
            "stochastic_rounding": int(bool(self.stochastic_rounding)),
            "quant_train_renew_leaf": int(bool(self.quant_train_renew_leaf)),
            "quant_seed": int(self.quant_seed),
```

with the five constructor keywords defaulting to
`False, 4, True, False, 11` and assigned in `__init__` beside
`self.bagging_seed` (line 1192).

**Target** `python/mojoboost/basic.py`. Nothing. These are `train()`
parameters, not `Dataset` binning parameters, so `_BINNING_DEFAULTS`
(line 110) and `_DATASET_PARAMS` (line 95) must **not** grow. A user passing
`use_quantized_grad` to `Dataset(params=...)` should keep getting the
existing "belongs to train()" error.

**Errors** the native `check_supported` refusal surfaces as the existing
`RuntimeError`-from-Mojo path, unchanged in mechanism.
**Fallback** none; the parameter is either honored or refused.
**Serialization effect** none.
**Public API effect** five new sklearn-style estimator keywords, which is
the first user-visible change in this entire sequence. It must land *after*
`CONNECTED` flips, or every one of them raises.
**Validation, UNRUN**: `python/tests/` gains a case asserting the four
LightGBM-named keywords are accepted and that
`use_quantized_grad=True` trains a model whose predictions are within a
stated tolerance of the float one on a fixture.

### 3.9 Flip the gate, then export

**Target** `src/mojoboost/quantized_gradient.mojo`, line 191
(this lane's file, and the only edit here that is this lane's to make).

```mojo
comptime CONNECTED = True
```

Only after 3.4 through 3.7 have landed **and** the differential test in
section 5 passes. Flipping it earlier makes every one of those patches live
without evidence.

**Target** `src/mojoboost/__init__.mojo`, after the `from .histogram import
(...)` block:

```mojo
from .quantized_gradient import (
    MODE_FLOAT,
    MODE_QUANTIZED,
    ROUND_NEAREST,
    ROUND_STOCHASTIC,
    SCALE_MAGNITUDE_SUM,
    SCALE_MAX_ABS,
    WIDTH_16,
    WIDTH_32,
    WIDTH_64,
    GradientStats,
    QuantDecision,
    QuantGradParams,
    QuantizedHistogram,
    QuantRoundKey,
    QuantScales,
    QuantTotals,
    accumulator_width,
    decide,
    derive_scales,
    gradient_stats,
    quantize_rows,
)
```

Export **last**, and only what a caller outside the package needs. Exporting
earlier makes `tools/check_parity.py` check 7 fire on the deferred parity
row, which is a false claim of reachability while nothing is reachable.
**Validation, UNRUN**: `python3 tools/check_parity.py` passes;
`python3 tools/audit_integration.py` no longer lists the module as an orphan
and the row from 3.1 is removed from both places.

### 3.10 Consolidate splitmix64 (finding F4, lowest priority, no behavior)

Five copies: `bagging._splitmix64`, `goss._splitmix64`,
`sampling._splitmix64`, `tree_parameters_extra._mix64`,
`quantized_gradient._mix64`. Pick one home every one of the five can import
without a cycle, export `mix64` and `uniform_from_counter` from it, and
delete the other four. `bagging`, `goss`, and `sampling` are all imported by
`boosting`, so the home cannot be any of them; a new leaf module
(`src/mojoboost/counter_rng.mojo`) importing nothing is the shape that
works. Bit-for-bit identical output in all five cases, so no test moves.

### 3.11 Documentation

**Target** `docs/LIGHTGBM_PARITY.md` line 385. Do **not** change the status
word until 3.9 lands. When it does:

```
| `use_quantized_grad` / `num_grad_quant_bins` / `quant_train_renew_leaf` / `stochastic_rounding` | partial | Quantized-gradient training. `src/mojoboost/quantized_gradient.mojo` holds one policy for both the LightGBM max-abs lattice and the fixed-point lattice the GPU histogram already used. CPU only; the device kernels still round to nearest at the fixed-point lattice, so `stochastic_rounding` is a CPU-path parameter. See `docs/QUANTIZED_GRADIENTS.md` |
```

`partial`, not `supported`, until the GPU kernel patch (section 4) lands.

---

## 4. The GPU patch, stated separately because it is a kernel change

**Target** `src/mojoboost/gpu_leaf_batching.mojo` (lines 380-381 and
492-493), `src/mojoboost/histogram_gpu.mojo` (`stage_gradients`,
`fill_gradients_device`, `enqueue_leaf`).
**Ownership** not this lane's. **Dependency** all of section 3.

The kernels quantize inline today:

```mojo
        var gq = Int32(round(grad[unsafe_offset = plane_base + r][0] * g_scale))
        var hq = Int32(round(hess[unsafe_offset = plane_base + r][0] * h_scale))
```

Three changes, in increasing order of cost.

1. **Nothing, for the deterministic max-abs lattice.** The kernel already
   computes `Int32(round(x * scale))`. Feeding it a max-abs `units` instead
   of a magnitude-sum one changes no kernel line. The host sets the scale;
   `derive_scales` is what produces it. This is the cheapest possible GPU
   support and should be the first thing tried.
2. **Stochastic rounding** needs `floor(x * scale + u)` with `u` from the
   same counter stream, which means three 64-bit integer multiplies and two
   shifts per value in the kernel, plus the stream base and the row's global
   index passed in. The arithmetic is exact integer arithmetic on every
   backend, so the CPU and the device agree bit for bit. This is the only
   way `stochastic_rounding=true` becomes a GPU parameter rather than a CPU
   one, and finding F2 applies with full force: the Int32 accumulator no
   longer holds the magnitude-sum bound at large row counts, so the kernel
   must either take the max-abs lattice or promote.
3. **Narrow accumulators** (`WIDTH_16`) would halve the histogram buffer and
   the download, and need a second kernel instantiation plus a promotion
   path when `accumulator_width` says 32. This is where the bandwidth
   argument actually lives and it is also the largest change; do not start
   here.

**Validation, UNRUN, and it needs hardware**: a GPU histogram built at a
max-abs lattice must equal a CPU one built at the same lattice, exactly, on
the integer planes, and the existing GPU/CPU float-histogram agreement test
must still hold at the fixed-point lattice.

---

## 5. The one test that has to exist first

**File** `tests/test_quantized_gradient.mojo` (does not exist; this lane may
not write tests). **UNRUN.**

The differential check, in order of what it proves:

1. `quantize_rows` then `build_quantized_histogram` then `dequantize`
   agrees with `build_histogram_subset` to within
   `n_rows * residue_per_row(mode) / units` per bin, at
   `SCALE_MAGNITUDE_SUM` and at `SCALE_MAX_ABS` with `B` in `{4, 16, 256}`.
   The tolerance is a formula, not a magic number; assert against the
   formula.
2. `subtract_quantized(parent, child)` equals the sibling's direct
   accumulation **exactly**, at every bin count. This is the strongest
   assertion in the suite and it should fail loudly if the accumulation
   order ever stops being irrelevant.
3. Rounding reproducibility: `quantize_rows` over a permuted row order,
   over `MOJOBOOST_NUM_WORKERS=1` and `=8`, and over a bagged subset,
   produces the identical array for the rows it has in common.
4. `accumulation_bound` at `(Int32.MAX, fixed_point lattice,
   ROUND_NEAREST)` equals `Int32.MAX` exactly, and at
   `ROUND_STOCHASTIC` exceeds it. This is finding F2 as an assertion, and
   it is the one that pins the shipping GPU path's safety argument.
5. `hessian_units_at_least` is never laxer than the float comparison:
   for random `(min_child_hess, lattice, hess_sum)`, integer-pass implies
   float-pass.
6. `check_same_lattice` raises when two rounds' histograms meet.

Per the memory note on this repository's test budget: this is **one** focused
test file, run once, never the full suite and never a build/bench loop.

---

## 6. Deliberate differences from LightGBM

| Thing | LightGBM | mojoboost | Why |
|---|---|---|---|
| Odd `num_grad_quant_bins` | Integer-divides by 2 and carries on | Raises | An asymmetric gradient lattice is a bug in every reported case, not a configuration |
| Stochastic rounding seed | Per-thread engine, not reproducible across thread counts | `seed` parameter, counter-based, reproducible everywhere | The package tests CPU/GPU agreement and rerun reproducibility; a thread-count-dependent dither breaks both |
| Scale rule | One (max-abs) | Two (max-abs, magnitude-sum) | The GPU histogram has shipped the second one since it landed. One policy has to describe both or there are two policies |
| Hessian lattice | `max|h| / B` | Same, but clamped symmetrically | A negative hessian cannot arrive (the custom-objective check rejects it), but if one did it is represented rather than folded to zero |
| Accumulator width | Promoted dynamically inside the histogram pool | Declared per node by `accumulator_width`, host representation always Int64 | Three host histogram types would be three accumulators, three subtractors, and three dequantizers to keep in step, for a saving that only matters where bytes move |
| Overflow | Not surfaced | `REASON_OVERFLOW`, float fallback | Finding F2 |

---

## 7. Open questions this lane could not answer without running anything

1. **`std.math.round`'s tie rule.** The module is deliberately independent
   of it (host and device call the same function), but the docstrings say
   "whatever tie rule that function implements" rather than naming it,
   because naming it would be a claim. One compiled probe settles it.
2. **Whether `@fieldwise_init` synthesizes a usable constructor when a
   trailing field is appended**, which decides whether 3.4 and 3.8 must land
   together. Read the other appended-field precedents
   (`TreeParams.extra`, `TreeParams.cat`) and how their callers were
   updated.
3. **Whether the integer scan in 3.7 is actually faster than
   dequantize-then-scan** on this repository's shapes. No benchmark exists
   and none should be quoted until one does.
4. **Whether `SCALE_MAX_ABS` at `B = 4` reproduces LightGBM's published
   accuracy** on any fixture here. The design doc restates the paper's
   result and labels it as the paper's; this repository has measured
   nothing.
5. **The distributed contract.** `combine_stats` is the reduction that makes
   every rank agree on one lattice before any rank quantizes, and
   `accumulate_quantized` makes the histogram all-reduce bit-identical
   across rank counts, which the Float64 one is not. Neither is wired.
   `distributed.allreduce_histogram` (line 227) is the call site, and it
   needs a stats all-reduce placed *before* the per-round quantization, not
   after. That is a real ordering constraint on a file this lane did not
   touch and could not test.
