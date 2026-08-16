# Floating point numerics and contraction policy

This document states what bit-level reproducibility mojotrees promises and
what it deliberately does not promise, names the one optimizer transformation
most likely to move a result quietly, records the three times it did so during
the perf-round-2 optimization round, and sets the convention every
multiply-add on the numerics path is expected to follow.

It was written after the first two of those incidents, on `perf-round-2` at
commit `cd5a5ea`, against Mojo 1.0.0 build `ed45d567`. The compiler behavior in
section 3 was measured on that build on an Apple M4, host and device. It has
not been measured on any other toolchain, any other host architecture, or any
other GPU vendor, and nothing here should be read as a claim about NVIDIA or
AMD device compilers.

**Section 1 was rewritten during CPU round 1 to state the settled accuracy
policy, which is narrower than the one this document originally carried.**
Sections 4, 7, 8 and 10 were written under the stricter rule. Where a
paragraph's reasoning rested on the half of that rule which has now been
retired, the paragraph carries a note saying so rather than being deleted,
because the measurement it reports is still a measurement and the reasoning is
still worth reading with its date attached.

## 1. The promise

The policy has three parts. They are not one promise in three wordings, and
the difference between them is most of the content of this section. Part one
is about a single toolchain and is absolute. Part two is where accuracy is
actually defined, and it is defined against a comparator rather than against
this project's own history. Part three is a permission rather than a promise,
and it replaces a stricter rule this document used to state.

### 1.1 Deterministic on a given toolchain and across worker counts

**Training and prediction are bit-identical run to run, machine to machine,
and at every worker count, on one toolchain.** This part is not negotiable and
it is the property every test in this project asserts.

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

The second scope is deliberately wide, because a result that changes when the
thread count changes is not reproducible in any useful sense, and the whole
parallel design of this package is built so that the schedule cannot reach the
arithmetic. Every parallel accumulation on the numerics path either sums into
per-row slots that no other task touches, or accumulates in integers, where
addition is associative. `quantized_gradient.mojo` is the fixed-point path that
exists precisely to buy this on the GPU.

There is a consequence of part 1.1 that constrains CPU round 1 directly and is
easy to lose while reading part 1.3. A parallel decomposition is allowed to
change the arithmetic. What it changes the arithmetic *to* must be a function
of the data and of the resolved policy and of nothing else. A row-block
histogram fold is permitted under this policy; a row-block fold whose block
count is derived from the resolved worker count is not, because two runs at
different worker counts would then disagree and `require_identical_predictions`
in `bench/real_data/thresholds.json` would be the thing that caught it.
`docs/design/ACCURACY_BUDGET.md` section 4 states the specific rule, which is
to derive the block count from the node's row count alone and to fold the
partials in ascending block order regardless of which block finished first.

### 1.2 Accuracy is held-out parity with LightGBM, inside pre-registered thresholds

**Accuracy is defined against a comparator on real data, at tolerances written
down before any run.** It is not defined against this project's own previous
bytes. The tolerances live in `bench/real_data/thresholds.json`, and that
file's own preamble draws the distinction that makes it a gate rather than a
scoreboard. It is quoted here rather than paraphrased, because the paraphrase
loses the argument:

> Two kinds of number get confused in benchmark suites. A quality
> difference between two engines fitting the same objective on the same
> bins is small, stable, and reproducible, so a threshold on it is a real
> test: cross it and something is wrong. A timing is none of those things
> on a laptop with a thermal budget, so a threshold on it is a coin toss
> wearing a lab coat. verify.py reads this file and decides pass or fail.
> report.py prints the timings and decides nothing.

and, on the only legitimate way a threshold moves:

> Every value here was chosen before any run, from what the two
> implementations are known to differ by, and each carries its reasoning.
> Loosening one after seeing a result is allowed and has to be done in a
> commit that says so; editing one to make today's run green is how a
> suite stops meaning anything.

The same file restates part 1.1 as a test, through
`defaults.determinism.mojotrees.require_identical_predictions`, whose stated
rationale is that "a digest that moves is a regression in that property and is
worth failing over even when the metrics are unchanged."

### 1.3 Identity with past output is not a promise

**A change may move bits deliberately.** It is allowed to produce a different
model from the one this library produced yesterday, on three conditions, all of
which have to hold together.

1. **It stays deterministic under 1.1.** A change that moves bits and also
   makes the answer depend on the schedule is refused on the second ground
   whatever its merits on the first.
2. **It stays inside the accuracy thresholds of 1.2**, and the evidence for
   that is a before arm and an after arm of `bench/real_data` on one machine,
   not an argument from mechanism. An argument from mechanism is what decides
   whether the run is worth taking, not what discharges it.
3. **The golden fixture is re-baselined in the same commit, with the ulp
   movement stated.** How many values moved, on which arrays, by how many units
   in the last place, and for any that moved by more than a few, why. A commit
   that moves bits and leaves `tests/test_golden_bits.mojo` green did not move
   the bits it believes it moved. A commit that regenerates the fixture without
   characterizing the diff has spent the evidence rather than the budget.

`docs/design/ACCURACY_BUDGET.md` is the standing record of what each known
relaxation costs and who may authorize it.

**The retired half of the old rule was not a mistake and is not being called
one.** Until this round the promise was identity with past output as well as
determinism, and it earned its keep. It is what caught both of the incidents in
sections 4.1 and 4.2, neither of which any tolerance test would have seen, and
the third in section 4.3 was found by an audit that only got written because
the first two had happened. Three real defects, one invariant. It is being
traded, deliberately, under a directive that says the project optimizes for
speed and accuracy only, that accuracy is not negotiable, and that identity
with past output is. The trade is written down here rather than assumed so that
the next person to read a one-ulp diff knows which rule they are standing
under.

### 1.4 All three parts are scoped per backend

**CPU and GPU are not bit-identical to each other and are not intended to
be.** The GPU histogram path carries
gradients in Float32 and accumulates in Int32 fixed point; the CPU path carries
Float64. `histogram_gpu.mojo` states the resulting contract directly, that
agreement with the CPU builder is to Float32 precision and not bit-exact, while
integer counts agree exactly. `tests/test_host_replica.mojo` rests on the same
point, that a host histogram is interchangeable with a device one only once the
two have been shown to agree bit for bit on the target hardware, which is a
hardware claim and never an assumption. (The hybrid leaf scheduler built its
`MODE_MIRROR` gate on it too, until that module was deleted on 2026-08-16.)

So part 1.1 is **per backend**. Within a backend it is bitwise. Across backends
it is to Float32 precision on the float planes and exact on the integer planes,
and `thresholds.json` prices that separately under `defaults.device_agreement`
rather than folding it into the determinism gate.

### 1.5 Per-row derivatives are Float32, and two consequences follow

The CPU path stores per-row gradients and hessians at **Float32** precision,
matching LightGBM's `score_t`. Scores, leaf values, gains and histogram cells
stay Float64. The narrowing is applied at the objective and **again at every
read site**, which makes it idempotent and is what keeps the gathered and
ungathered accumulation paths adding identical Float64s.

This is the default and not the only setting; §1.6 has the measured trade that
made it a switch, and what `derivative_precision = "float64"` changes. Both
properties below are properties of the Float32 default.

Two properties follow that a reader should not have to discover from a test.

**Sample-weight scaling is exact only to Float32 precision.** The code forms
`score_t(g * w)`; anything computing `score_t(g) * w` instead gets a different
number, by up to two Float32 ulps. So doubling a sample weight does not
exactly double its gradient. This is a property of the product and not a
loosened test: `tests/test_objectives.mojo` asserts the identity to about ten
Float32 ulps because that is the bound the arithmetic actually supports.

**Any path that reads a derivative must narrow it too, or it stops agreeing
with the others.** This is not redundant with narrowing at the source: GOSS
reweighting and sample weights multiply a stored derivative, and the product
is a Float64 that generally is not Float32-representable. `histogram_sparse`
and the row-major kernels both had to be corrected for exactly this, and both
failures presented as a builder disagreeing with the dense reference rather
than as anything that looked like a precision problem.

### 1.6 It is a measured trade, and therefore a switch: `derivative_precision`

Float32 is not free, and the numbers below are the reason this is a parameter
rather than a constant. They are the first `bench/real_data` run after the
narrowing landed, comparing the run before it against the run after it on the
same scenarios, same seeds, same binning.

**What it bought.** Agreement between this package's own CPU and accelerator
arms, which is what makes a backend claim checkable at all:

| scenario | metric | before Float32 | after Float32 |
|---|---|---|---|
| dense_regression | rmse, CPU vs GPU | 1.6e-04 | **7.9e-09** |
| dense_regression | mae, CPU vs GPU | 2.0e-04 | 6.2e-09 |

Four orders of magnitude on RMSE. The device carries gradients in Float32 and
accumulates in Int32 fixed point (§1.4); a CPU path carrying Float64
derivatives was disagreeing with it for that reason and no other.

**What it cost.** Accuracy on the imbalanced binary scenario, on the CPU arm
alone. The accelerator arm barely moved, so this is a precision effect and not
two backends drifting:

| scenario | metric | before Float32 | after Float32 | change |
|---|---|---|---|---|
| imbalanced_binary | average_precision, CPU arm | 0.0136 | 0.0123 | **-9.4% relative** |
| imbalanced_binary | auc, CPU arm | 0.7339 | 0.7279 | -0.006 absolute |
| imbalanced_binary | average_precision, CPU vs GPU | 0.00497 | **0.0736** | wider |
| imbalanced_binary | auc, CPU vs GPU | 0.00317 | 0.0054 | wider |

The CPU-vs-GPU columns widen on this scenario while narrowing on
`dense_regression`, which is the same fact seen twice: the CPU arm moved and
the accelerator arm did not.

Provenance: **measured**, `bench/real_data`, one run per side, on the
scenario's synthetic variant at a 0.5 percent positive rate (the real variant,
`bank_marketing`, sits near an average precision of 0.68 and is not where this
shows). Two runs are two runs and not a distribution; average precision on a
rare class is dominated by the ordering of a few hundred rows and is the
noisiest number in the harness. Read the direction, not the third digit.

**The decision, and it did not change.** Float32 stays the default. It is
LightGBM's own profile (`typedef float score_t`, `include/LightGBM/meta.h`),
and on `imbalanced_binary` it puts this package *at* LightGBM's average
precision and AUC rather than above them, which is the parity the project is
aiming at. A number that moves toward the comparator is not a regression even
when it moves down.

**But a measured trade becomes a switch.** `derivative_precision` takes
`float32` (the default, everything above) or `float64`, and `float64` keeps
per-row gradients and hessians at full Float64 through the objective and every
read site. Reach it with `MOJOTREES_DERIVATIVE_PRECISION=float64`, which is
read once per fit by `histogram.ConstHessianSettings.resolve()` and once per
round by `boosting.fill_grad_hess`.

**What `float64` buys and what it costs, for a reader deciding.** It buys back
the accuracy in the second table on a rare-class ranking metric, and it buys
an exact sample-weight identity (§1.5's first property is a Float32 property
and does not apply). It costs, in order of size:

- **Agreement with the accelerator**, which is the first table run backwards.
  A `float64` CPU fit and a device fit are not comparable to the precision
  §1.4 claims, and `bench/real_data`'s `device_agreement` gate is calibrated
  for the Float32 default.
- **Speed.** The gathered `(gradient, hessian)` pair buffer packs two Float32
  into one Float64 word, so it cannot carry a Float64 derivative and does not
  run; and the row-blocked private histograms have that buffer as their only
  row source on the subset arm, so blocking is off in *both* builders (both,
  because one-sided blocking is what produced a four-ulp leaf-value
  disagreement between the bagged and whole-dataset paths). A timing taken
  under `float64` is not a timing of this package's CPU path.
- **Bit identity with any published number**, all of which are Float32.

What it does not cost: determinism across worker counts, which §1.1 promises
at both settings and `tests/test_derivative_precision.mojo` checks at both.

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

### 2.1 Which of that survives section 1.3

Part 1.3 retires identity with past output, and a reader could reasonably ask
whether contraction is still worth a document. It is, and the reason is worth
stating precisely, because it changes what the rest of this document is asking
for.

**Contraction never threatens part 1.1.** It is a compile-time decision. A
binary that contracts a site contracts it on every run, at every worker count,
on every machine on that toolchain. Determinism, the part that is not
negotiable, is not what contraction is a hazard to and never was.

What it is a hazard to is two other things, and both survive intact.

**Two paths that must agree with each other.** Section 4.3 is the whole
argument. A device trainer whose raw scores depended on whether the run was
bagged, and a device prediction that disagreed with the update the trainer had
applied to the same tree, are defects under any policy, because neither
version of the answer is being chosen. Nothing in part 1.3 makes it acceptable
for the tiled and atomic GPU strategies to disagree, or for
`hybrid_leaf_scheduler`'s host replica to stop matching the device it is
mirroring. Part 1.3 permits bits to move; it does not permit two things that
claim to be the same computation to be different computations.

**Bits moving without anybody deciding.** Read part 1.3 carefully and the
permission it grants is narrow. A change may move bits *deliberately*, priced,
with the fixture re-baselined and the movement stated. Contraction is the
mechanism by which bits move when nobody decided anything, because a refactor
that is arithmetically identical changes what the optimizer emits. That is the
one thing part 1.3 has no procedure for, since there is nothing to state and
nobody to state it. So the enforcement in section 7 is not weakened by the new
policy. Its job changes from "prove nothing moved" to "prove that whatever
moved, somebody meant it", and the fixture is the same fixture either way.

The practical effect is on section 6's four rules. Rule one, decide the
semantics and say so in the code, is now the load-bearing one. Rule four, pin
it with a bit comparison, is unchanged. What has gone is the unstated fifth
rule, that a diff in the golden fixture ends the discussion.

## 3. What Mojo 1.0.0 exposes, measured

The Modular documentation and the docs MCP are both unreliable for this project,
and so, it turns out, is a hasty probe. Everything in this section was taken out
of the compiler rather than out of the documentation, and one paragraph of it
had to be taken out of the compiler twice.

The correction is worth keeping because it is the more instructive half. A lane
in this round reported that `docs.modular.com/gpu/block-and-warp.md` lists warp
primitives that do not exist in this release. It had checked
`max.gpu.primitives.warp`, which indeed does not exist, and concluded that the
warp collectives were absent from the toolchain. **They are not.**
`std.gpu.primitives.warp` exists and exports the full set: `sum`, `max`, `min`,
`prefix_sum`, `shuffle_down`, `shuffle_up`, `shuffle_xor`, `shuffle_idx` and
`broadcast`, all of which compile. The docs page was right about the primitives
and wrong only about the namespace.

That mistake propagated for most of a day and reached several pieces of work as
"there are no warp primitives in Mojo 1.0", which is a much stronger and much
falser claim than the one that was actually verified. The lesson is narrow and
worth stating: a negative result from a probe bounds only what the probe
tested. `max.` not having a module is not evidence that `std.` does not, and an
absence should be reported with the namespace attached.

Separately, `max.gpu.host.device_graph.DeviceGraph` exists as a type. It is not
constructible from a `DeviceContext` directly, so it is presumably obtained from
a capture API that has not been located, and whether it is backed on Metal is
unverified. If it is, it captures exactly the fixed data-independent command
sequence a device-resident tree emits, which would make it worth finding.

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

**Under part 1.3 that justification has expired, and the call is deferred
rather than made here.** The only reason this `fma` exists is to reproduce
bytes the project no longer promises. Removing it would move the raw score of
every round on every CPU fit by up to one unit in the last place, which
compounds into a different model, so it is exactly the kind of change part 1.3
permits and exactly the kind it requires a priced run and a re-baseline for. It
also has a cost on the other side that is easy to miss: `test_round_overhead`
compares `_add_by_leaf` against `_add_by_traversal`, and the two agree only
because the `fma` matches what the traversal expression contracts to. Removing
the `fma` without settling the traversal arm's semantics turns that test from a
proof into a coincidence. The same question, one level up and for the whole
package at once, is section 8's.

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

Two of those three reasons still stand under part 1.3 and one has changed
status. Section 2.1's first hazard, two paths that must agree, is the reason
this was a defect, and it is untouched by the new policy. The fix direction is
still the right one. **"The CPU golden fixture is untouched" was a reason under
the old rule and is not one under the new rule**, and it is worth noticing that
it was the weakest of the three even then. It said only that the change had
been confined to a backend the fixture did not cover, which is a statement
about the fixture's coverage rather than about the change. Under part 1.3 the
change would still have landed, and would additionally have owed a priced
before-and-after arm, which it did not get. The absence of a GPU golden fixture
is why the third condition of part 1.3 currently has nothing to bind on the
device path.

## 5. The convention at each kind of site

What follows is an audit of every place on the numerics path where a floating
point multiply feeds an add or a subtract, as of `cd5a5ea`. "Contracts today"
means at the default `contract=fast` on the toolchain and backend named at the
top of this document, and it is an inference from the form and the section 3
measurements unless the site is separately pinned by a test.

Two exemptions apply throughout and remove most of the code from consideration.
**Integer arithmetic cannot contract.** Every index computation, every Int32
histogram accumulation, and every Int32 sibling subtraction are exempt by type
(so was the integer cost model in `hybrid_leaf_scheduler.mojo`, deleted
2026-08-16). **A multiply that
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
numbers, which is exactly the per-backend scoping of section 1.4.

The convention is descriptive rather than chosen, and section 4.1's note says
why that matters now. **Fused is what the compiler happened to do, and the
whole "intended" column of this table means "intended to keep matching what it
happened to do".** That was the right column to have under the old rule. Under
part 1.3 it is a set of open questions with a default answer, and the default
answer is worth keeping only because changing it costs a re-baseline and buys
nothing on its own. If a change in this round needs one of these sites moved,
move it and pay part 1.3's three conditions. Do not treat the word "intended"
here as a decision somebody made about the arithmetic, because at most two
sites in this whole table is that true.

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

`hybrid_leaf_scheduler.mojo` contained no floating point arithmetic at all: its
cost model was integer nanoseconds throughout, which its own docstring gave as
the reason. The module was deleted on 2026-08-16 and this paragraph is kept as
the record of what was audited.

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

**Five, which is new with part 1.3. If the pin then fails, say what moved
before you change it.** A red golden fixture is now the start of a decision
rather than the end of one. The decision has three questions and they go in
order. Is the change still deterministic across worker counts, which is the
part that is not negotiable. Does held-out parity survive `thresholds.json`,
which needs the harness run and not an argument. And what exactly moved, in
units in the last place, on which arrays. Answer all three in the commit
message and re-baseline in that same commit. Answer none of them and
re-baseline anyway and the fixture has stopped being evidence, which is the
failure mode section 7.2 item 6 was already written to prevent and which the
new policy makes cheaper to fall into rather than harder.

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

**Part 1.3 changes what "treated as such" means and it is worth being exact
about the difference, because it is easy to over-read in either direction.**
Under the old rule a golden diff was a stop. The change did not land, or it
landed with the diff explained away, and in practice it was a refusal. Under
the new rule a golden diff is a bill. The change may land, and what it owes is
the three conditions of part 1.3, of which the fixture re-baseline is only the
third. The fixture's job is unchanged and its authority is unchanged. It is
still the only instrument in the project that can tell you a model moved, since
no tolerance test can, and section 2.1's second hazard is precisely that
without it bits move with nobody deciding. What has changed is the default
verdict on the evidence it produces, and nothing else.

One thing that follows and should be said out loud, because a fixture whose
baseline is expected to move invites it. **The fixture is not weakened by being
re-baselined. It is weakened by being re-baselined without a statement.** Its
value has always been the git history of its values, not the values, and the
history is only readable if every movement in it carries what moved and why.

### 7.2 Toolchain bumps

Golden values are pinned to a toolchain version. They are expected to move when
that version changes, because contraction decisions are optimizer decisions and
optimizers change.

The procedure is:

1. **Expect movement, and do not treat it as a failure of the change.** A golden
   diff on a toolchain bump is information, not a bug, until it is examined.
2. **Do not regenerate as part of an unrelated change.** A toolchain bump is its
   own commit. Regenerating goldens inside a feature branch hides exactly the
   signal this whole document exists to preserve. Part 1.3 does not contradict
   this and the two rules are easy to read as contradicting, so the seam is
   worth naming. Part 1.3 requires a *deliberate* bit-moving change to
   re-baseline **in the same commit**, because the movement and its
   justification belong to one another. This item forbids re-baselining in a
   commit that was **not** about moving bits. The test is whether the commit
   message states the movement as one of the things the commit is for. If it
   does, same commit. If the movement is a surprise the author is absorbing on
   the way past, stop and split it out, and find out why it moved first.
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

**The first reason this section gave has expired and is retracted here rather
than edited out.** It read, in full, that "the project's existing golden bits
were produced under `contract=fast`", that every recorded result, every model
in the fixtures, every comparison in `test_round_overhead`, and the entire GPU
validation record in `docs/GPU_VALIDATION.md` came out of a default build, and
that turning contraction off globally would move all of them at once. Every
factual clause in that is still true. Section 10 measures it, and all six
golden fixtures fail under `contract=off`. What has gone is the inference.
Moving every model at once was a refusal under the old rule and is a priced
decision under part 1.3, and the price, by
`docs/design/ACCURACY_BUDGET.md`'s reckoning, is one unit in the last place per
site, which is thirteen orders of magnitude below where that document's
perturbation curve leaves zero. **The stated reason for this decision no longer
supports the decision.** `docs/design/ACCURACY_BUDGET.md` section 10 argues on
that basis that the question should be re-opened on its own terms, and it is
right that it should be, and this document is not the place it gets settled.

**The reason that survives is that it would not actually deliver what it
appears to promise.** `contract=off` removes the optimizer's freedom to fuse, but it does
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
3. A before-and-after run of `bench/real_data`, which is condition 2 of part
   1.3 and is the only thing that can establish that moving every model at once
   stays inside `thresholds.json`. Section 10 characterizes the movement on the
   fixtures but says nothing about a held-out metric, and a global flip of the
   floating-point mode is the largest single bit-moving change available to
   this project, so it is the last one that should be discharged by argument.
4. A single commit regenerating every golden value under section 7.2's
   procedure, with the diff characterized, which is condition 3.
5. A note in `docs/GPU_VALIDATION.md`, since the Metal record would no longer
   describe the shipped configuration.
6. An answer for `compatibility/fixtures/`, which is the item that has no
   answer today and is the reason this is harder than the other five put
   together. Those fixtures hold raw scores from released versions as bit
   patterns and their README states that their value comes from never being
   regenerated. Part 1.3 does not reach them, deliberately, because they are a
   promise to users with models on disk rather than a promise about our own
   history. A global change of the floating-point mode moves prediction bits
   and would break every one of them, and the choice would be between
   admitting the promise is broken and finding a way not to move prediction.
   `docs/design/ACCURACY_BUDGET.md` section 1 states the line.

Until all six are done, `contract=off` should be treated as a **diagnostic**
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

  The consequence for section 1.1 is direct and is unchanged by the new
  policy. The floating-point mode is one of the things held fixed, alongside
  the toolchain version, and a build that sets `--fp-mode contract=off` is a
  different numerical build of this library rather than a differently optimized
  one. Two builds at different settings are as unrelated to each other as CPU
  and GPU are, and section 1.4's per-backend scoping is the right analogy.

  The consequence for section 8 is where the new policy bites. Changing the
  flag would move every model this library has ever fitted, and under the old
  rule that sentence was the end of the argument. Under part 1.3 it is the
  size of the bill, not a refusal, and the size is what makes it worth a
  before-and-after run of `bench/real_data` rather than a paragraph. Note also
  what the shape of these six failures says about how such a run should be
  read. Three fixtures moved a raw score by one ulp, which is the
  expected effect. Three moved a leaf value, one of them by 93 ulps and two of
  them across zero. Read by ulp count alone, that looks like a second and
  larger effect. It is the same effect, amplified by cancellation in a
  quantity that is numerically nothing, and a held-out metric would not see
  it at all. **Ulp counts are the right unit for stating what moved and the
  wrong unit for deciding whether it matters.** That is the division of labor
  between part 1.3's condition 3 and its condition 2.

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

## The fixed-point scale cannot live on the device, and this is why

**A property, recorded because the next person will try the same dodge.**
Established 2026-08-16 by the `trip-count` lane while attempting exactly it.

The device fixed-point scale is currently folded on the host once per round, from
device partials, and handed to nine kernels as a `Float32` launch argument. That
readback is one of only two host round trips per round, so "compute the scale on
device and never read it back" is the obvious move.

**It does not work, and the obstruction is arithmetic rather than plumbing.**

The host needs the scale's *value* in three places. Two are mechanical: a launch
argument, and a staged table for the batching paths. The third is
`Float32(1.0 / g_scale)`, packed into the split searcher's parameter block.

**The gain `G^2 / (H + lambda)` and the leaf value `-G / (H + lambda)` are not
homogeneous in the scale, because `lambda` is not scaled.** Scale `G` and `H` by
`s` and the gain becomes `s^2 G^2 / (sH + lambda)`, which is not `s^k` times the
original for any `k`. So a searcher told the scale is 1.0 does not compute a
rescaled gain — it computes a **different** gain, and picks a **different split**.

### The dodge that looks like it works, and where it fails

Pre-scale the gradients on device and hand every kernel a scale of 1.0. With a
power-of-two scale this is genuinely exact -- `g * 2^k` is an exact `Float32`
product -- so the histogram is bit-identical and the first two sites are
satisfied.

**It fails on the third.** The searcher still needs the real `1/s` to weigh
`lambda` against sums that are now in scaled units. There is no value of the
parameter block that makes an unscaled `lambda` correct against scaled sums.

### What is available instead

Only **how often** the host asks, not **whether** it must. That is
`GpuHistogramBuilder.set_scale_refresh(N, H)`: the magnitudes are still reduced
every round into a pinned window, and only the readback is deferred. Its
amortization is verified rather than assumed -- every round under an outgoing
scale is checked against the overflow bound and a violation **raises**.

### One further fact, for whoever tries a device fold anyway

Apple GPUs have no `Float64`. A device fold would be `Float32` or double-float
(~2^-48) against the host's ~2^-53. Because `largest_power_of_two_at_most` is a
**step function**, the two agree except when the two quotients straddle a power
of two -- and then the scale differs by exactly 2x and every histogram cell
moves. The overflow bound survives either way. The bit-motion condition is
**dataset-dependent and unbounded**, not a probability, so it cannot be traded
away against an error budget.
