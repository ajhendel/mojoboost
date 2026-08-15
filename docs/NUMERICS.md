# Floating point numerics and contraction policy

This document states what bit-level reproducibility mojotrees promises, names
the one optimizer transformation most likely to break that promise quietly,
records the two times it did break it during the perf-round-2 optimization
round, and sets the convention every multiply-add on the numerics path is
expected to follow.

It was written after those two incidents, on `perf-round-2` at commit
`cd5a5ea`, against Mojo 1.0.0 build `ed45d567`. The compiler behavior in
section 3 was measured on that build on an Apple M4, host and device. It has
not been measured on any other toolchain, any other host architecture, or any
other GPU vendor, and nothing here should be read as a claim about NVIDIA or
AMD device compilers.

## 1. The promise

mojotrees promises that training and prediction are **bit-identical across
runs**, subject to a scope that has to be stated exactly, because the scope is
what makes the promise testable.

Held fixed:

- the toolchain version, meaning the exact `mojo --version` build hash;
- the backend, meaning CPU or GPU, and on GPU the vendor and device;
- the compilation flags, including the floating point mode of section 3;
- the inputs, the parameters, and every seed.

Allowed to vary, with the results still required to be bit-identical:

- the worker count (`MOJOTREES_NUM_WORKERS`);
- the task count, block count, and the grain at which any parallel loop is
  split, including the parallel-versus-serial crossover
  (`MOJOTREES_PARALLEL_MIN_OPS`);
- the GPU launch geometry that `gpu_tiling.mojo` derives at runtime, insofar as
  it is derived from device attributes rather than from the data;
- machine load, thermal state, and wall clock.

The scope is deliberately narrow in one direction and deliberately wide in the
other. Wide, because a result that changes when the thread count changes is not
reproducible in any useful sense, and the whole parallel design of this package
is built so that the schedule cannot reach the arithmetic. Every parallel
accumulation on the numerics path either sums into per-row slots that no other
task touches, or accumulates in integers, where addition is associative.
`histogram.mojo` says this in its own docstrings, and `quantized_gradient.mojo`
is the fixed-point path that exists precisely to buy it on the GPU.

Narrow in the other direction, because **CPU and GPU are not bit-identical to
each other and are not intended to be.** The GPU histogram path carries
gradients in Float32 and accumulates in Int32 fixed point; the CPU path carries
Float64. `histogram_gpu.mojo` states the resulting contract directly, that
agreement with the CPU builder is to Float32 precision and not bit-exact, while
integer counts agree exactly. `hybrid_leaf_scheduler.mojo` builds its whole
`MODE_MIRROR` gate on the same point, that a host histogram is interchangeable
with a device one only once the two have been shown to agree bit for bit on the
target hardware, which is a hardware claim and never an assumption.

So the promise is **per backend**. Within a backend it is bitwise. Across
backends it is to Float32 precision on the float planes and exact on the
integer planes.

## 2. What contraction is, and why it is the hazard

An optimizer is allowed, under a relaxed floating point model, to replace

    round(round(a * b) + c)

with

    round(a * b + c)

by emitting a single fused multiply-add. Both are legitimate evaluations of the
source expression `a * b + c`. They differ because the first rounds twice and
the second rounds once, and on the values where the discarded bits matter they
land one unit in the last place apart. This is called **contraction**, and it is
on by default in Clang, in nvcc, in the Metal shading language compiler, and, as
section 3 shows, in Mojo.

Contraction is a hazard for this project specifically, and not merely an
academic one, because of three properties of gradient boosting.

**It compounds.** Raw scores feed the next round's gradients. A one-ulp
difference in a score update is not a one-ulp difference in the output. It is a
different gradient, which can be a different split, which is a different tree.
`boosting.mojo` records the observation directly, that hoisting the product out
of the score-update loop moved 98 of 600 rows by one ulp and produced a
different second tree.

**It is invisible to tolerance-based tests.** Every one of these incidents
passed a comparison at any reasonable epsilon. They were caught only because the
tests that cover the score update compare `to_bits()` rather than absolute
difference.

**It is a property of the optimizer, not of the source.** The same expression
contracts in one loop shape and not in another; contracts when an index is a
per-thread value and not when it is a launch argument; contracts in one function
and not in the next after inlining changes how many uses a product has. None of
this is stable across compiler versions, and none of it is written down in the
source unless somebody writes it down.

## 3. What Mojo 1.0.0 exposes, measured

The Modular documentation and the docs MCP are both unreliable for this project.
A lane in this same round found `docs.modular.com/gpu/block-and-warp.md` listing
warp primitives that do not exist in this release, and the docs MCP reporting a
module that does exist as not found. Everything in this section was therefore
taken out of the compiler, not out of the documentation.

### 3.1 There is a contraction control, and it works

`mojo build --help` and `mojo run --help` both list, in Mojo 1.0.0 build
`ed45d567`:

    --fp-mode <MODE>
        Controls floating-point behavior as a comma-separated list of
        `feature=value` items (may be given more than once). The only feature
        is `contract`, one of `fast` (default) or `off`. `contract=fast` is
        like Clang's `-ffp-contract=fast`: it fuses `a + b*c` into an FMA
        across statements and breaking strict IEEE compliance. `contract=off`
        disables contraction.

Probed for its edges, the compiler rejects everything else. `--fp-mode
contract=on` is rejected with "expected 'contract=fast' or 'contract=off'", and
`--fp-mode reassociate=off` is rejected with "the only supported feature is
'contract'". The compiler binary contains strings for a wider internal policy
covering reassociation, reciprocals, and exceptions, and an MLIR-level
`fastmathFlags` attribute on `pop.mul`, but none of that is reachable from the
command line or, on the spellings tried, from `__mlir_op` in Mojo source. Treat
it as absent.

The granularity is **whole compilation only**. There is no module-level pragma,
no function decorator, and no expression-level intrinsic. `mojo package` and
`mojo precompile` do not accept `--fp-mode` at all, and they do not need to: a
probe library precompiled with no flag and then linked into a program built with
`--fp-mode contract=off` came out unfused, so contraction is decided where the
final code is generated, not where the package is written.

### 3.2 The ground truth triple

Every claim below rests on one Float32 triple, chosen so that the fused and the
unfused evaluation of `a * b + c` differ in the last bit.

| symbol | value | bits (decimal) | bits (hex) |
|---|---|---|---|
| `a` | 0.6086544394493103 | 1058787527 | 0x3F1BD0C7 |
| `b` | 1.3038229942321777 | 1067901868 | 0x3FA6E3AC |
| `c` | -0.5372443199157715 | 3205073112 | 0xBF0988D8 |

    round(round(a * b) + c) = 1048788512   0x3E833E20   unfused
    round(a * b + c)        = 1048788511   0x3E833E1F   fused, one ulp lower

The rounded product `round(a * b)` is 1061890024, `0x3F4B27E8`.

That triple does not discriminate on the subtract form, because for these values
`c - a * b` rounds to the same Float32 either way. A second triple was
constructed for the subtract:

| symbol | value | bits (decimal) | bits (hex) |
|---|---|---|---|
| `a` | 1.1785693168640137 | 1066851164 | 0x3F96DB5C |
| `b` | 1.3396586179733276 | 1068202479 | 0x3FAB79EF |
| `c` | 1.6968423128128052 | 1071198753 | 0x3FD93221 |

    round(c - round(a * b)) = 1039242736   0x3DF195F0   unfused
    round(c - a * b)        = 1039242737   0x3DF195F1   fused, one ulp higher

The inputs are read from environment variables in every probe, because a first
attempt that wrote them as literals was constant folded away entirely. The
emitted assembly for that attempt contained zero `fmul` and zero `fadd`
instructions and exactly one `fmadd`, which was the explicit `fma` call. Compile
time folding evaluates the unfused semantics, so a folded probe silently reports
"no contraction" for every form. Any future probe must defeat folding the same
way and should check the assembly to confirm it did.

A second contamination is worth recording. A probe that computed `a * b` once
and then used it in several forms showed no contraction anywhere, because common
subexpression elimination gave the single product several uses, and LLVM
contracts only a product with exactly one use. Each form in the final probes
therefore gets its own three inputs.

### 3.3 CPU results

Run as `mojo run` on Apple M4, `-O3` default, one form per set of inputs.

| form written in the source | default (`contract=fast`) | `--fp-mode contract=off` |
|---|---|---|
| `a * b + c` | **fused** | unfused |
| `var p = a * b; p + c` | **fused** | unfused |
| `acc = c; acc += a * b` | **fused** | unfused |
| `fma(a, b, c)` | fused | fused |
| `c - a * b` | **fused** | unfused |
| `_prod(a, b) + c`, product returned from a callee | **fused** | unfused |
| product stored into a `List` slot, reloaded, then added | unfused | unfused |

The second row is the important one and it contradicts the intuition the
codebase has been operating on. **Binding the product to a named local does not
block contraction.** A `var` is an SSA value in LLVM, not a rounding barrier. If
a site depends on the product being rounded, a local variable does not buy that.

### 3.4 The loop shape result, which explains the CPU incident

Two loops, same operands, differing only in whether the product is invariant
across the loop:

| shape | default | `contract=off` |
|---|---|---|
| `out[i] = base[i] + lr * vals[i]`, product varies per iteration | **fused** | unfused |
| `out[i] = base[i] + lr * value`, product invariant across the loop | unfused | unfused |

This is the mechanism behind the CPU incident, reproduced in isolation. The
traversal update multiplies by a value that changes per row, so the product has
to be computed inside the loop, where the add can consume it and the two
contract. The leaf update multiplies by a value that is constant for a whole
leaf, so loop-invariant code motion hoists the multiply out of the row loop
before contraction is ever considered, and the hoisted product gets rounded on
its own.

Nothing about that is a language guarantee. It is the ordering of two optimizer
passes. The same source could contract next release if the passes ran the other
way around, or if the loop were unrolled first, or if the leaf loop were
vectorized.

### 3.5 GPU results

The same forms inside a device kernel, launched through `DeviceContext` on the
Apple M4 Metal backend, with the operands read from device memory so the device
compiler cannot fold them.

| form written in the kernel | default | `--fp-mode contract=off` |
|---|---|---|
| `a * b + c` | **fused** | unfused |
| `var p = a * b; p + c` | **fused** | unfused |
| `acc += a * b` | **fused** | unfused |
| `fma(a, b, c)` | fused | fused |
| `c - a * b` | **fused** | unfused |
| one operand derived from `global_idx` rather than a launch argument | **fused** | unfused |

Two conclusions. First, the device compiler contracts by default, as expected.
Second, and less obvious, **`--fp-mode` reaches the device compiler.** It is one
switch for both, not a host-only flag. That is a convenient property and it has
been checked only on Metal.

### 3.6 There is no reliable way to force the unfused form at a single site

`math.fma` reliably forces the **fused** form, at any optimization level, on
both backends. That is the tool the codebase already reaches for.

The opposite direction has no equivalent. Four candidate barriers, measured at
the default setting:

| candidate | result |
|---|---|
| bind the product to a `var` | **fused**, the barrier does not hold |
| round trip the product through `to_bits()` and `bitcast` back | **fused**, the bitcast pair folds away |
| widen to Float64, multiply, narrow back to Float32 | **fused**, the widen-multiply-narrow folds to a Float32 multiply and then contracts |
| store the product into a `List` slot and reload it | unfused |

Only the last one held, and it held for a bad reason. It is a store the
optimizer did not forward, which is another accident of the same kind as the
loop-invariance result, and it costs a round trip through memory. Do not treat
it as a supported idiom.

This asymmetry is the single most important practical fact in this document.
**Fused is expressible in the source. Unfused is not.** A site that needs the
unfused form has exactly two honest options: the global build flag, or a
restructuring that removes the multiply from the neighborhood of the add
entirely. The GPU incident took the second option and it is the better one,
because the result then holds independently of what any optimizer decides.

## 4. The three incidents from this round

### 4.1 CPU, the score update

`boosting.mojo` ends a round by adding `learning_rate * value` to every training
row's raw score. Before this round it did that by walking the whole tree once
per row to recover the leaf that row had already been partitioned into. The
optimization was to add to each leaf's own rows instead, using the membership
the grower already produced.

The obvious way to write the leaf version hoists `learning_rate * value` out of
the row loop, since it is constant for the leaf. That is the shape section 3.4
shows does not contract. The traversal version, which multiplies by a per-row
`predict_row` result, does contract. So the arithmetically identical rewrite
moved 98 of 600 rows by one ulp, and because raw scores feed the next round's
gradients, it produced a different second tree.

The fix is at `src/mojotrees/boosting.mojo:1201`, and it is the only `fma(` call
in `src/`:

    raw_p.unsafe_store(
        slot, fma(learning_rate, value, raw_p.unsafe_load(slot))
    )

The traversal update at `src/mojotrees/boosting.mojo:1135` was deliberately left
as the bare expression, so that it keeps tracking whatever the compiler does with
the expression the untouched trainers still write, and `test_round_overhead`
compares the two.

Note what this means. The explicit `fma` is not there because fused is more
correct. It is there because fused is what this project's existing bits are, and
the existing bits came from an expression the compiler happened to contract. The
fix preserves the bits; it does not justify them.

### 4.2 GPU, the leaf id that stopped being a launch argument

`gpu_objectives_native.mojo` has a per-leaf kernel that computes `raw[i] +
learning_rate * value[node]`, where `node` is a launch argument. One launch per
leaf. The optimization was a range-table kernel that does every live leaf in one
launch, finding its own segment by binary search, which turns `node` into a
per-thread value.

Writing the same expression in the new kernel produced a different last bit. In
the per-leaf kernel the product is uniform across the launch, so the device
compiler computes and rounds it on its own; in the range-table kernel it is
per-thread, so the compiler contracted it into the add. The kernel's own
docstring at `src/mojotrees/gpu_objectives_native.mojo:495` records the reasoning
in full.

The fix moved the multiply to the host. `src/mojotrees/gpu_objectives_native.mojo:1102`
computes `lr32 * Float32(values[i])` on the host and stages the step; the kernel
at `:537` does nothing but add. The kernel now contains no multiply at all, so
there is nothing left for any compiler to fuse, on any backend, at any release.

That is the strongest form of the fix available and it is the pattern to copy.

### 4.3 GPU, the score update and prediction, found by audit rather than by accident

The first two incidents were found because somebody restructured code and a
test that compared bits caught the consequence. The third was found the other
way round: this document's own fragility ranking named two sites as possibly
already inconsistent, and a test was written to settle it. They were
inconsistent.

The same arithmetic, `raw[i] + learning_rate * value[node]`, existed in
`gpu_objectives_native.mojo` in three spellings. `_update_raw_kernel` read
`node` per-thread out of the leaf-assignment array, so its product varied
across the launch and the Metal compiler contracted it. `_range_add_raw_kernel`
took `node` as a launch argument, so its product was uniform, was hoisted, and
was rounded on its own. `_range_table_add_raw_kernel` had no multiply at all,
because an earlier fix in the same round moved it to the host.

Measured on an Apple M4, one four-split tree over 3,000 rows, driving all three
arms over the same partition and the same node values: the per-thread arm
disagreed with both range arms on **2,225 of 3,000 rows**, every one by exactly
one unit in the last place. Classified against host references built so that the
unfused reference itself could not contract, the per-thread arm matched the
**fused** answer on all 2,225 separable rows and the unfused answer only on the
775 where the two roundings coincide. It was fused, not intermittently fused.

`gpu_predict.mojo`'s kernel had the same shape and disagreed on **992 of 2,000**
rows in the same way.

The consequence is what makes this an incident rather than a curiosity. A device
fit's raw scores depended on nothing but whether the run was bagged, because the
bagging and all-rows arm reached the contracting kernel and the device-resident
arm did not. And a device prediction disagreed with the raw-score update the
trainer had applied to that same tree. Neither is a defensible thing for a
library to do, whichever of the two answers is preferred.

Both kernels now take a precomputed step and contain no multiply, so there is
nothing left to fuse at either site on any backend at any release.
`_range_add_raw_kernel` deliberately keeps its inline product: no trainer
reaches it, it is the reference arm, and its device-computed unfused answer is
what pins the host-side multiply as bit-equal to the device one. Fixing it too
would leave the table arm compared only against itself.

This moved what the GPU computes, by one unit in the last place per tree, which
on the training arm compounds into a different model rather than a different
last digit. It was landed deliberately, on the argument that two paths which
must agree and do not is a defect, that the fix direction removes the dependence
on an optimizer rather than relocating it, and that the CPU golden fixture is
untouched. There is no GPU golden fixture yet; there should be, and that is the
obvious next piece of enforcement.

## 5. The convention at each kind of site

What follows is an audit of every place on the numerics path where a floating
point multiply feeds an add or a subtract, as of `cd5a5ea`. "Contracts today"
means at the default `contract=fast` on the toolchain and backend named at the
top of this document, and it is an inference from the form and the section 3
measurements unless the site is separately pinned by a test.

Two exemptions apply throughout and remove most of the code from consideration.
**Integer arithmetic cannot contract.** Every index computation, every Int32
histogram accumulation, every Int32 sibling subtraction, and the whole integer
cost model in `hybrid_leaf_scheduler.mojo` are exempt by type. **A multiply that
feeds anything other than an add or a subtract cannot contract.** A product
consumed by `round()`, by `abs()`, by a comparison, by a divide, or by a store is
not a contraction site, which is why the entire fixed-point quantization pipeline
is structurally immune.

### 5.1 The score update

| site | expression | today | intended | if the optimizer changed its mind |
|---|---|---|---|---|
| `boosting.mojo:1201` `_add_by_leaf` | explicit `fma(learning_rate, value, raw)` | fused, by construction | **fused** | nothing, `fma` is not a contraction |
| `boosting.mojo:1135` `_add_by_traversal` | `raw[slot] + learning_rate * tree.predict_row(...)` | fused | **fused** | **different model.** This is the reference the `fma` above reproduces |
| `boosting.mojo:916, 963, 980` | `s += learning_rate * predict_*` | fused | fused | different predictions, and different raw scores on continued training |
| `boosting.mojo:1555, 1852, 1881` | same shape, continued training and multiclass | fused | fused | different model on `train_more` |
| `objective.mojo:409, 476, 478` | `raw[r] += learning_rate * predict_row(...)` in the custom-objective trainers | fused | fused | **different model** |
| `ranking.mojo:740, 831` | same shape in the rankers | fused | fused | **different model** |
| `train_gpu.mojo:2847, 3065, 3509, 3744, 3977` | same shape, host side of GPU training | fused | fused | **different model** |
| `linear_tree.mojo:1045, 1061, 1915` | same shape for linear trees | fused | fused | **different model** |
| `gpu_objectives_native.mojo:537` `_range_table_add_raw_kernel` | add only, step precomputed on host | no multiply | **unfused product, computed on the host** | nothing, structurally immune |
| `gpu_objectives_native.mojo:445` `_range_add_raw_kernel` | inline product, `node` is a launch argument | unfused, because the product is launch-uniform and gets hoisted | unfused, to match the host-computed step | **different model.** Held only by launch uniformity |
| `gpu_objectives_native.mojo:413` `_update_raw_kernel` | takes a precomputed step; no multiply | none to fuse | unfused by construction | fixed. Was fused, and measurably disagreed; see section 4.3 |
| `gpu_predict.mojo:274` | accumulates a precomputed step; no multiply | none to fuse | unfused by construction | fixed. Was fused, and measurably disagreed; see section 4.3 |

The convention here is plain, even though it is not currently stated anywhere in
the source. **The score update is fused.** Every host-side variant of `raw[r] +=
learning_rate * value` contracts and the one that could not contract on its own
was given an explicit `fma` to match. On the device, the range-table path
achieves the same answer a different way, by doing the multiply on the host in
Float32 and shipping the step.

The two forms are not equal to each other. A host `fma(lr, value, raw)` in
Float64 and a device `raw + round(lr32 * value32)` in Float32 are different
numbers, which is exactly the per-backend scoping of section 1.

### 5.2 Gradients and hessians

Almost every built-in objective is structurally immune, because its gradient is
a single multiply of a pre-computed difference. `w * (p - y)` has the add inside
the multiplicand, not consuming the product, and nothing can fuse.

The exception is tweedie, in both implementations:

| site | expression | today | intended | if it changed |
|---|---|---|---|---|
| `boosting.mojo:523` | `w * (-y * e1 + e2)`, tweedie gradient | **fused** | unstated | **different model** |
| `boosting.mojo:524` | `w * (-y * (1 - a) * e1 + (2 - a) * e2)`, tweedie hessian, both operands of the add are products | **fused** | unstated | **different model** |
| `gpu_objectives_native.mojo:250, 252` | the device twins of the same two | **fused** | unstated | **different model** |

Everything else in `boosting.mojo`'s `_fill_grad_hess_into` and in
`gpu_objectives_native.mojo`'s `_grad_hess_kernel` is a bare multiply, a multiply
of a difference, or a divide. The GOSS importance at `boosting.mojo:1807`
computes `abs(grad * hess)`, and the `abs` sits between the product and the
accumulation, so it is immune. Multiclass softmax hessians are multiply chains
with no add consuming a product.

The convention is therefore accidental rather than chosen. Tweedie fused is
whatever this compiler produced, on both backends independently, and there is
nothing pinning host and device to the same choice.

### 5.3 Leaf values and the regularization terms

The Newton leaf output is `-T(G) / (H + lambda_l2)`. The soft threshold `T(G)`
is a subtract with no multiply, and the regularized denominator is an add with no
multiply, and the whole thing is a divide. `gain.mojo:35`, `tree.mojo:845`,
`split.mojo:615`, `gpu_split_search.mojo:349` and `:360` are all of this shape
and all structurally immune. The `t * t` products feed a divide, not an add.

Two leaf-value sites are not immune:

| site | expression | today | intended | if it changed |
|---|---|---|---|---|
| `tree_parameters_extra.mojo:231` | `value * (w / (w + 1.0)) + parent_output / (w + 1.0)`, path smoothing | **fused** | unstated | **different model**, whenever `path_smooth > 0` |
| `monotone.mojo:275` | `-(2.0 * grad_sum * output + (hess_sum + lambda_reg) * output * output)` | **fused** | unstated | **different model** on constrained or finished scans |

`monotone.mojo:275` is the densest contraction site in the CPU path. Both
operands of the add are product chains, it is `@always_inline`, and it inlines
directly into `split.mojo:284`. Its own docstring already concedes that it agrees
with the `G^2 / (H + lambda_l2)` form only "up to floating-point association",
which is the same class of caveat one level up.

### 5.4 Split gain

`split.mojo:264` is the hot unconstrained gain,

    left_g * left_g / (left_h + lambda_reg)
    + right_g * right_g / (right_h + lambda_reg)
    - parent_score

and it is **immune**, which is worth stating because it looks like the most
exposed line in the package. The products feed divides. The adds consume
quotients, and a divide cannot be fused into an add. The `+ lambda_reg` adds
have no multiply operand. `gpu_split_search.mojo:412` is the Float32 device twin
of the same expression and is immune for the same reason.

The constrained arm at `split.mojo:283` and its device counterpart at
`gpu_split_search.mojo:425` route through `output_score`, so they inherit
section 5.3's exposure.

One device site in the split scan is a genuine single-use inline product feeding
an add:

| site | expression | today | intended | if it changed |
|---|---|---|---|---|
| `gpu_split_search.mojo:678` | `lg * g_inv / (lh * h_inv + cat_smooth)`, many-vs-many categorical sort key | **fused** | unstated | **a different category ordering**, therefore a different split, therefore a different model |
| `gpu_split_search.mojo:3488` | the host twin of the same line inside `reference_search` | **fused** on this host | must match the device | the two are compiled by different optimizers and are pinned to each other by nothing |

### 5.5 Sibling subtraction

`histogram.mojo:1053` through `:1066` is a plain elementwise subtract of two
histograms with no multiply anywhere, so the whole CPU sibling subtraction is
immune. The device sibling subtractions at `gpu_active_rows.mojo:1121` and
`gpu_leaf_batching.mojo:599` are **Int32**, which is exact rather than merely
unfused, and is the reason the quantized path exists.

`gpu_split_search.mojo` performs its sibling subtraction on dequantized Float32,
in the pattern `var lgf = lg.cast[float32]() * g_inv` followed by `var rgf =
total_g - lgf`. Those products are not fusable in practice because each
dequantized local has two or more uses. That is a weaker guarantee than
integer exactness, and it depends on the use count surviving future edits to
those kernels.

### 5.6 Fixed-point quantization and dequantization

**Structurally immune, end to end, and by design rather than by luck.**

Quantization is `Int32(round(x * scale))`. The multiply is consumed by `round`,
which forces it to be materialized as a Float32 or Float64 of its own. This holds
at `quantized_gradient.mojo:934`, `histogram.mojo:922` and `:926`,
`gpu_active_rows.mojo:698`, `:701`, `:874`, `:875`, `:1038`, `:1039`,
`gpu_leaf_batching.mojo:409`, `:410`, `:521`, `:522`, `gpu_categorical.mojo:381`,
`:385`, `gpu_gradient_stream.mojo:721`, `:793`, `gpu_sparse.mojo:486`, `:604`,
and `gpu_multiclass_batch.mojo:559`. There is no add for a product to fuse into
at any of them.

Accumulation is integer. Dequantization is `Float64(q) * (1.0 / scale)` feeding
a store, at `histogram_gpu.mojo:1179` and following, at
`quantized_gradient.mojo:1334`, and at `histogram.mojo:953`. One value in, one
value out, no accumulation, no add.

The one float multiply anywhere in `quantized_gradient.mojo` that does feed an
add is the overflow bound at `:1053`, `FIXED_ONE + rows * residue_per_row(mode)`.
Both operations are exact at every reachable magnitude, since the product is an
integer times a power of two and the sum needs at most 32 mantissa bits of the
Float64's 53, so fused and unfused agree bit for bit. **Nothing.** It is listed
for completeness only.

`hybrid_leaf_scheduler.mojo` contains no floating point arithmetic at all. Its
cost model is integer nanoseconds throughout, which its own docstring gives as
the reason.

### 5.7 Prediction

`tree.mojo` and `model.mojo` contain **no floating point multiply feeding an add
anywhere**. `predict_row` and `predict_bins` return the stored leaf value with no
arithmetic; leaf values are stored unshrunk and `learning_rate` is applied only
at score-update and predict time, which puts every prediction site into section
5.1's table.

The linear-tree path is a different matter. `linear_tree.mojo:636` and `:649`
accumulate a dot product, `out += coef[j] * (v - center[j])`, and `:1274`,
`:1288`, `:1291`, `:1577`, `:1578`, `:1580` build and check the normal
equations. All are inline products feeding accumulations and all contract today.
A change there is a **different model**, because the coefficients themselves move.

### 5.8 Ranking

`ranking.mojo:307`, `:359`, `:360` accumulate DCG and ideal DCG as `total +=
gain * discount`, and `:562` accumulates `sum_lambdas -= 2.0 * p_lambda`. All
contract today. A change moves the NDCG deltas and therefore the lambdas and
therefore the model.

### 5.9 Metrics and losses

`boosting.mojo:625` through `:674` compute the training and validation losses,
and nine of those lines are inline products feeding a `total +=`. These are not
on the model path in the direct sense, but they are on it through **early
stopping**, since a loss that moves in the last bit can move a
`best_iteration` at a tie and change how many trees the model has. Rate the
consequence as "a different model, rarely, at ties" rather than "washes out".

## 6. What a contributor must do

When you add or move a floating point multiply that feeds an add or a subtract,
on any path that reaches a model, do all four of these.

**One. Decide which semantics you want, and say so in the code.** Not "whatever
the compiler does". Either the site should round the product on its own, or it
should not. If you cannot say which, you are not ready to write the line.

**Two. If you want fused, write `fma` explicitly.** Do not rely on the
expression contracting. `from std.math import fma`, and
`fma(a, b, c)` rather than `a * b + c`. This is the only construct measured to
hold at both settings on both backends. `boosting.mojo:1201` is the worked
example and its docstring is the model for the comment you should leave.

**Three. If you want unfused, restructure so there is no product next to the
add.** Compute the product somewhere the add cannot reach it, and pass in the
result. On the GPU, "somewhere" usually means the host, as
`gpu_objectives_native.mojo:1102` does. On the CPU it may mean a separate loop or
a separate function whose result crosses a boundary the optimizer will not
inline through, though be warned that section 3.3 measured a product crossing a
callee boundary and contracting anyway once inlined. Do **not** rely on binding
the product to a `var`, on a bits round trip, or on widening to Float64. Section
3.6 measured all three failing.

**Four. Pin it with a bit comparison, not a tolerance.** Add or extend a case in
`tests/test_golden_bits.mojo`, or in whichever suite already covers the site, and
compare `to_bits()`. A tolerance test would have passed for both incidents in
section 4.

Two additional rules follow from the audit.

**Prefer a form that cannot contract at all over a form that currently does not.**
`split.mojo:264` is safe because a divide sits between the product and the add,
and that is a property of the expression rather than of the optimizer. A site
that is safe because a value happens to have two uses, or because a product
happens to be loop invariant, is one inlining decision away from changing.

**When you add a host replica of a device computation, or a device twin of a host
one, write both halves in the same shape and say in a comment that they are
twins.** `gpu_split_search.mojo:678` and its host counterpart at `:3488` are the
same line compiled by two different optimizers, and nothing in the source records
that they have to agree.

## 7. Enforcement, and what to do on a toolchain bump

### 7.1 The golden-bits fixture

`tests/test_golden_bits.mojo` is the enforcement mechanism for everything above.
Its role is to hold a small set of trained models and predictions whose exact bit
patterns are recorded in the test itself, and to fail if any of them moves. Not
to assert that the values are correct in any absolute sense, which no fixture can
do, but to assert that they have not changed since the last time somebody looked
at them deliberately.

That is the only kind of test that catches contraction. A one-ulp move in a raw
score passes every tolerance and fails a bit comparison, and the two incidents in
section 4 were both caught that way and would both have been missed otherwise.

The fixture is being written on a sibling lane and its exact coverage, format, and
regeneration entry point are its own to define. This document deliberately does
not restate them, so that it does not go stale when they change. What this
document does commit to is the **relationship**: any change that moves a golden
value is a change to the model this package produces, and must be treated as
such, whatever the change was nominally about.

### 7.2 Toolchain bumps

Golden values are pinned to a toolchain version. They are expected to move when
that version changes, because contraction decisions are optimizer decisions and
optimizers change.

The procedure is:

1. **Expect movement, and do not treat it as a failure of the change.** A golden
   diff on a toolchain bump is information, not a bug, until it is examined.
2. **Do not regenerate as part of an unrelated change.** A toolchain bump is its
   own commit. Regenerating goldens inside a feature branch hides exactly the
   signal this whole document exists to preserve.
3. **Record both versions.** The old and the new `mojo --version`, including the
   build hash, go in the commit message. `ed45d567` is the build these values
   were established against.
4. **Record the diff, and characterize it.** How many values moved, by how many
   ulps, and on which sites. A handful of last-bit moves on a score update is one
   story. A gain that moved enough to change a split is a different story and
   needs the split investigated before the goldens are accepted.
5. **Re-run section 3's probes.** They live outside the repository, but they are
   cheap to rebuild from the tables above, and the answer to "did `--fp-mode`
   change" or "does binding to a local block contraction now" should be measured
   again rather than assumed to have carried over.
6. **Never regenerate silently.** A golden file whose history contains an
   unexplained bulk update has stopped being evidence of anything.

The same procedure applies to a GPU driver or vendor change, which recompiles
every kernel with a different device compiler.

## 8. The decision on `--fp-mode`

Section 3 found a contraction control, so this document owes a decision.

**The decision is to leave `--fp-mode` at its default, `contract=fast`, and to
add it to no build script or pixi task.** No task in `pixi.toml`, and no script
under `bindings/`, `capi/`, `cli/`, or `tools/`, passes it today, and none should
start.

The reasoning is not that fused is better arithmetic. In several places it is
better, since it rounds once instead of twice, but that is not why.

**The reason is that the project's existing golden bits were produced under
`contract=fast`.** Every recorded result, every model in the fixtures, every
comparison in `test_round_overhead`, and the entire GPU validation record in
`docs/GPU_VALIDATION.md` came out of a default build. Turning contraction off
globally would move all of them at once, which is a large deliberate change to
what this package computes, and it would need to be justified on its own terms
rather than smuggled in as a hardening measure.

**The second reason is that it would not actually deliver what it appears to
promise.** `contract=off` removes the optimizer's freedom to fuse, but it does
not remove the two mechanisms that actually bit us. `boosting.mojo:1201`'s
explicit `fma` is unaffected by the flag, correctly, since it is a fused
operation and not a contraction. And the GPU incident's per-leaf reference kernel
would keep its unfused answer under either setting, since it never contracted in
the first place. Turning the flag off would make the code more predictable and
would not make it self-documenting, and the audit above shows that most of the
exposure is at sites nobody has yet decided the semantics of.

**What it would take to adopt it globally**, if the project ever wants to:

1. A pass over section 5's tables converting every site whose intended semantics
   is fused into an explicit `fma`, so that the flag changes nothing there. That
   is `boosting.mojo:1135`, all of section 5.1's host score updates, and a
   decision on each of tweedie, path smoothing, `output_score`, and the linear
   tree normal equations.
2. `--fp-mode contract=off` added to every compilation entry point, which is
   `bench`, `test`, `build-python`, `build-capi`, `build-cli`, and the
   `run_tests.sh` invocations, because a partial adoption is worse than none. The
   flag is per compilation and not per module, so a program built without it
   picks up contraction in library code that was precompiled with it, as section
   3.1 measured.
3. A single commit regenerating every golden value under section 7.2's
   procedure, with the diff characterized.
4. A note in `docs/GPU_VALIDATION.md`, since the Metal record would no longer
   describe the shipped configuration.

Until all four are done, `contract=off` should be treated as a **diagnostic**
rather than a setting. Building a suite both ways and diffing the results is a
cheap way to enumerate which sites are contraction-sensitive, and it is the
recommended first move when investigating an unexplained bit difference.

**Because the flag is off, the per-site conventions in section 5 are
load-bearing rather than belt-and-braces.** There is no global setting standing
behind them. `boosting.mojo:1201`'s explicit `fma` and
`gpu_objectives_native.mojo:1102`'s host-side multiply are the only two places in
the package where the arithmetic is nailed down independently of the optimizer,
and everything else in section 5 holds because of what this compiler currently
chooses to do.

## 9. Sites this document considers fragile, in order

These are ranked by the product of how likely the optimizer is to change its mind
and how bad it would be if it did. None of them is asserted to be wrong today.
They are the places where "wrong tomorrow" is cheapest.

1. **Resolved, and it was not fragile, it was broken.** The first two entries in
   this list were `gpu_objectives_native.mojo:413` and `gpu_predict.mojo:274`,
   both flagged here as "may already be inconsistent rather than merely
   fragile". They were. `tests/test_gpu_fma_consistency.mojo` measured it on the
   M4: over one four-split tree on 3,000 rows, the per-thread arm disagreed with
   both range arms on 2,225 of them, every time by one unit in the last place,
   and matched the fused host reference on every separable row. The predict
   kernel disagreed on 992 of 2,000. Both now take a precomputed step and
   contain no multiply. See section 4.3.

   The lesson worth keeping is about this document rather than about those two
   sites: when an audit classifies a site as "fragile" from its source shape, it
   is stating a hypothesis, and the cost of testing it here was one test file.
   Two of the first two entries in a fragility ranking turned out to be live
   defects. Rank by cost of being wrong, then measure the top of the list.

3. **`gpu_objectives_native.mojo:445`, `_range_add_raw_kernel`.** The reference
   kernel the fixed one is pinned against. Its unfused answer is held **only** by
   `node` being a launch argument, which is precisely the property that stopped
   holding in the incident. If a future change makes that product non-uniform, or
   if the device compiler stops hoisting launch-uniform products, the reference
   itself moves and the pin moves with it.

4. **`monotone.mojo:275`, `output_score`.** Two product chains into one add, in an
   `@always_inline` function that inlines into the split gain at
   `split.mojo:284`. Whether it contracts depends on what the surrounding
   inlining does to the use counts, which is exactly the kind of thing that
   changes between releases. A change here moves split gains, which is a
   different model rather than a different last digit.

5. **`gpu_split_search.mojo:678` and its host twin at `:3488`.** The same
   categorical sort key compiled by two different optimizers, with nothing in the
   source recording that they must agree. A divergence changes a category
   ordering and therefore a split.

6. **`tree_parameters_extra.mojo:231`, `smooth_leaf_output`.** A single-use
   product feeding an add, producing the value that `Tree.value[node]` stores.
   Exposed only when `path_smooth > 0`, which is why it ranks below the
   unconditional sites, but a change there is a change to the stored model.

7. **`boosting.mojo:523` and `:524`, and their device twins at
   `gpu_objectives_native.mojo:250` and `:252`.** Tweedie gradient and hessian,
   the only built-in objective with a multiply feeding an add. Four sites, two
   compilers, no comment tying them together.

8. **`objective.mojo:409`, `:476`, `:478` and `ranking.mojo:740`, `:831`.** Score
   updates in the custom-objective and ranking trainers that never route through
   `_add_by_leaf` and therefore have **no `fma` counterpart**. They are correct
   today because they are still written as the contracting expression, but the
   asymmetry with `boosting.mojo:1201` is undocumented and invites somebody to
   optimize one of them the same way `_add_by_leaf` was optimized, and hit the
   same one-ulp wall without the same warning.

9. **`linear_tree.mojo:1274`, `:1288`, `:1291`.** The normal-equation
   accumulations. Fragile in the ordinary sense rather than the contraction
   sense, since they are also the sites most exposed to summation order, but a
   contraction change moves fitted coefficients directly.

10. **`gpu_split_search.mojo`'s dequantize-then-subtract pairs**, at `:885` and
    `:891`, `:1329` and `:1335`, `:1661` and `:1667`, and their host replicas in
    `reference_search`. Safe today only because each dequantized local has two or
    more uses. That is a use-count property, not an expression property, and it
    is one refactor away from becoming a single-use product feeding a subtract.

## 10. What was measured and what was not

Measured, on Mojo 1.0.0 build `ed45d567` on an Apple M4:

- **that contraction is load-bearing across the whole package, not only at the
  two sites this round patched.** `tests/test_golden_bits.mojo`, whose values
  were generated at the default `contract=fast`, fails **all six fixtures**
  when the identical source is run with `--fp-mode contract=off`. The command
  is one line and it is the cheapest diagnostic in this document:

      pixi run mojo run --fp-mode contract=off -I build -I tests tests/test_golden_bits.mojo

  What moved, and it is worth reading the shape rather than only the count.
  Three fixtures moved a final raw score by exactly one unit in the last
  place, which is the direct and expected effect of one fused product per row
  per round no longer being fused. Three moved a stored leaf value instead,
  and by more: 93 units in the last place for the bagged fixture, and for the
  plain regression and feature-fraction fixtures a value so small that the two
  answers straddle zero and the unit-in-last-place distance is not meaningful.
  Those two are near-zero leaf values, so a one-bit change upstream in an
  accumulation is amplified by cancellation into a large relative change in a
  quantity that is numerically nothing. That is not evidence of a defect; it
  is evidence that a leaf value on an already-fit residual is noise, and it is
  a reason to read this fixture's failures by array name before reacting to a
  distance.

  The consequence for section 1's promise is direct. Bit-determinism here is
  contingent on the default floating-point mode as well as on the toolchain
  version, and a build that sets `--fp-mode contract=off` is a different
  numerical build of this library, not a differently optimized one. Section 8's
  decision to stay at the default is therefore not a preference. Changing it
  would move every model this library has ever fitted.

- that `--fp-mode contract=fast|off` exists on `mojo build` and `mojo run`, is
  rejected for any other feature or value, and is absent from `mojo package` and
  `mojo precompile`;
- that it changes the emitted arithmetic on the host, for every form in section
  3.3's table;
- that it changes the emitted arithmetic inside a Metal device kernel, for every
  form in section 3.5's table;
- that it applies to code linked in from a separately precompiled package;
- that binding a product to a named local does not block contraction on either
  backend;
- that a loop-invariant product is not contracted while a per-iteration product
  in the same loop is, which reproduces the CPU incident's mechanism in
  isolation;
- that neither a bits round trip nor a Float64 widen-and-narrow blocks
  contraction.

Not measured, and therefore not claimed:

- **any behavior on NVIDIA or AMD.** Both device compilers are different
  optimizers. Every GPU statement here is a Metal statement.
- **any behavior on x86-64.** The CI matrix covers x86-64 and ARM64, and only
  ARM64 was probed. FMA availability and the cost model that drives contraction
  decisions differ.
- **any behavior at optimization levels other than the default `-O3`.**
- **whether the sites in section 5 that are marked "fused" actually contract in
  their real surroundings.** Every one of them was classified from its source
  form plus the section 3 measurements of the same form in isolation. Section 3.2
  and 3.3 both showed that surroundings matter, through constant folding and
  through use counts, so a site could be classified as fused here and be unfused
  in place. Only `boosting.mojo:1135` and `:1201` have been checked in place, by
  `test_round_overhead`. **This is the largest gap in the audit** and the honest
  way to close it is to build the suite twice, once at each `--fp-mode`, and diff.
- **whether item 1 in section 9 is currently producing a different answer from
  the range-table path.** It has the shape that would, it was not fixed when its
  sibling was, and nothing was run to find out.
- **any performance consequence of anything in this document.** No benchmark was
  run and none should be inferred.
