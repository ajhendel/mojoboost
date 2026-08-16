# Red team: the device-resident control plane plan

This document argues against a plan. It was commissioned adversarially, on the
stated understanding that an endorsement would be worth nothing, so where it
agrees with the plan it says so briefly and moves on, and where it disagrees
it works the arithmetic out in full. The disagreements are mostly arithmetic,
and arithmetic is checkable, which is the point.

Two framing rules were set for this review and are honored throughout.

The accuracy baseline is **LightGBM**, not this library's own past output.
Bit-identity with yesterday is being retired deliberately, so "the trees come
out different from before" is not an objection and is not raised as one. But
**determinism is being kept**, so "this result depends on the thread count, on
the launch geometry, or on which order the hardware finished in" is an
objection, and it is raised wherever it applies. As it turns out, that
distinction is load-bearing: one of the plan's three numerical relaxations
lands on the wrong side of it, and the plan appears not to know which promise
it is relaxing.

Everything below separates measurement from inference. Where a number is
quoted, its file is given. Where a number could not be found, that is said
plainly, because two of the figures in the plan's own justification do not
exist anywhere in this repository or its history.

## 0. The plan, and the verdict in one page

The plan:

1. Blow up the control plane, not the kernels. The host-driven per-split loop
   is 85.8 percent of a round and the kernels are 22.9 percent, so rewriting
   kernels re-earns correctness for a fifth of the prize. Move the tree onto
   the device, have kernels read the chosen split from device memory rather
   than take it as a launch argument, and reduce host waits from 31 per tree
   toward 1.
2. Take the numerical relaxations bit-identity was blocking: row-block private
   histograms on the CPU with a fixed fold, packed int16 gradient and hessian
   in one atomic on the GPU, a power-of-two device-computed scale.
3. Keep the existing kernels' arithmetic, which is proven correct and
   extensively tested.
4. Estimated outcome: a fully device-resident tree lands the GPU near 0.8 to
   1.0 seconds against LightGBM's 2.86 at one million rows by fifty features.

The verdict, in order of how much it should change what happens next.

**Step 4 is arithmetically unreachable by step 1.** The GPU's marginal cost is
2.24 microseconds per row and the Metal trace shows that marginal cost is
kernel execution. The control plane is a **fixed** cost of about 1.33 seconds,
because the synchronization count does not vary with rows. Delete all of it and
one million rows still takes 2.0 to 2.4 seconds. The plan's target requires the
kernels to get 2.3x to 2.8x faster, which is the work step 3 forbids.

**Most of the control plane prize may already be sitting behind a shipped
flag that nobody has benchmarked.** `train_gpu._device_search_resident` under
`grow_policy=depthwise` already batches a whole level into one host wait, which
is 5 waits per tree instead of 30. That is roughly 78 percent of the
synchronizations the plan proposes to remove, available today at the cost of a
parameter. `docs/design/GPU_LEVELWISE.md:25-27` states in plain language that
"No benchmark has been run on any of it."

**Relaxation two does not work at the target shape.** Packed int16 gradient and
hessian gives a row ceiling of 32,766 by the repository's own bound formula.
The target is 1,000,000. That is not a tuning problem, it is thirty times the
wrong side of a hard limit, and the failure mode is a silent carry from one
packed field into its neighbor.

**The plan has misidentified which promise it is relaxing.** There is no
CPU-versus-GPU bit-identity to give up; `docs/NUMERICS.md:48-49` says in bold
that the two backends "are not bit-identical to each other and are not intended
to be." What actually exists, and what relaxation one breaks, is bit-identity
**within** a backend across worker counts, which is published in the README and
in a shipped release artifact. That is a determinism promise, and determinism
is being kept.

**The plan is aimed at half the drains.** Disassembly of the shipped MAX Metal
driver shows that `enqueue_copy` is a **synchronous full-queue drain in both
directions**, host to device as well as device to host. So the per-split
staging uploads, which the code's own docstring calls asynchronous, are
blocking waits too. A device-resident tree that removes only the downloads
leaves the uploads in place, and MAX offers no asynchronous copy, no Metal
stream, and no Metal graph builder to replace them with.

> **Annotation, 2026-08-16: this verdict is withdrawn along with O6, which it
> summarizes.** The disassembly stands. "Blocking waits too" does not: a drain
> of a queue holding nothing costs nothing, and the uploads are behind nothing.
> **Measured**, removing thirteen copies per tree, nine of them uploads, bought
> 0.016 seconds against 0.64 predicted, a null under M0; removing about thirty
> **round trips** per tree bought 0.75 seconds, resolved
> (`bench/results/session3_2026-08-16/RESULTS.md`,
> `docs/GPU_PORTABILITY.md` section 6.1.1). Removing only the downloads was
> removing the round trips, so the plan was aimed at the half that pays. The
> paragraph is left in place because it is what the red team registered before
> the data.

**Step 1's direction is right and should be pursued anyway.** 32.1 blocking
readbacks per round is indefensible, the mechanism is correctly identified, and
the money is real. It is worth about 1.33 seconds at every shape. It is not
worth 2.6.

## 1. What was measured, and at what shape

### 1.1 The Metal System Trace

`docs/METAL_TIMELINE.md` and `bench/results/metal_timeline_2026-08-15/`. Four
captures at 50,000, 100,000, 200,000, and 400,000 rows by 50 features, squared
error, 100 rounds, Apple M4 with 10 GPU cores.

Every headline figure the plan quotes comes from the **200,000 row** capture:

```
span, first GPU start to last GPU end      2346.74 ms
GPU busy (union of intervals)               552.28 ms  (23.53% of span)
GPU idle inside the span                   1794.45 ms  (76.47% of span)
compute GPU time                            536.96 ms  (22.88% of span)
blit GPU time (bytes actually moving)        15.33 ms  ( 0.65% of span)
serialization points per round                 32.1
dispatches per round                            221
one blocking readback, commit to next commit  606.1 us median  (85.8% of round)
```

There is **no capture at 1,000,000 rows.** There is no capture of the
multiclass workload. The trace that justifies the plan was taken at one fifth
of the row count the plan is judged against, on a different objective from the
one the plan's second-best evidence uses. Two of the four captures ran entirely
at the device's Minimum GPU performance state and are, by the document's own
rule (`docs/METAL_TIMELINE.md:424-447`), unusable for any scaling claim.

### 1.2 The end-to-end benchmark

`bench/results/profile_2026-08-15/`. Three shapes, three repeats on the
mojotrees arms, **one repeat** on the LightGBM arm.

| shape | ours CPU | spread | ours GPU | spread | LightGBM | LGBM repeats |
|---|---|---|---|---|---|---|
| 50,000 x 50 | 0.563844 | 11.3% | 1.634874 | 2.1% | 0.594365 | 1 |
| 250,000 x 50 | 1.663785 | 11.3% | 1.892842 | 1.1% | 1.002939 | 1 |
| 1,000,000 x 50 | 6.981735 | 10.8% | 3.575185 | 2.5% | 2.858046 | 1 |

Serial reference at one million rows: ours 15.959379 at one worker
(`r1_ours_1w.txt`), LightGBM 8.823065 at one thread (`r1_lightgbm_1t.txt`).

Multiclass, synthetic 465,000 x 54 x 7: ours CPU 25.47218, ours GPU 15.303747,
interleaved, spreads 7.7 and 0.1 percent (`multiclass_batch1.txt`).

### 1.3 Two of the plan's numbers do not exist

Checked exhaustively rather than casually. `grep -rni roofline` over the
working tree returns nothing. `git log --all -S"roofline"` returns no commit.
No file contains a 0.4 second derivation, and no file pairs a per-row slope of
3.2 microseconds with one of 2.4.

The nearest thing on disk to a roofline is `bench/apple/bin_layout_plan.json`,
which is explicitly a plan with unmeasured inputs and which refuses to default
its own key constant, and `docs/METAL_TIMELINE.md:406-422`, which does the only
bandwidth arithmetic in the repository and then declines to turn it into a
roofline: "The true DRAM rate is somewhere between 21.6 and roughly 120 GB/s
and the trace cannot narrow it further."

The slope pair is worse than missing. It is **not reproducible from the data on
disk under any pairing of the three measured shapes**, and the real slopes point
the other way. Section 2.1 computes them: 2.24 microseconds per row for us and
2.47 for LightGBM over the upper band, which is our GPU being about 10 percent
**better** per marginal row. "3.2 against 2.4" says we are 33 percent worse. The
two readings are on opposite sides of the question and they recommend different
sizes of project. The only "3.2" in the profile record is
`bench/results/profile_2026-08-15/RESULTS.md:31`, "we bin 3.2x faster," a
different quantity entirely and a plausible source for a garbled restatement.

## 2. The arithmetic

### 2.1 The GPU's cost is a large constant plus a per-row term

Three GPU wall clocks are on disk. Fit a line through the upper two, which is
the band the competitive claim lives in:

```
slope     = (3.575185 - 1.892842) / 750,000  = 2.243 us per row
intercept = 1.892842 - 250,000 * 2.243e-6    = 1.332 s
```

So `gpu_seconds ~= 1.33 + 2.24 * n_millions` for 100 rounds. The intercept
agrees with what the repository already believed from a different direction:
the commit history records the GPU as carrying "about 1.5 seconds of fixed cost
per fit," which is why it loses to this library's own CPU below roughly a
million rows.

The same fit for LightGBM:

```
slope     = (2.858046 - 1.002939) / 750,000  = 2.473 us per row
intercept = 1.002939 - 250,000 * 2.473e-6    = 0.385 s
```

Two things follow, and both are the opposite of what the plan's missing slope
figures imply. **Our GPU is already marginally faster per row than LightGBM**,
2.24 against 2.47. We do not lose at one million rows because our kernels are
slow relative to LightGBM's inner loop. We lose because we carry 1.33 seconds
of fixed cost against LightGBM's 0.385, and at one million rows that difference
of about 0.95 seconds is very nearly the entire 0.72 second gap.

That is a strong argument for the plan's direction. It is, in the same breath,
a fatal argument against the plan's number.

### 2.2 The per-row term is entirely compute, so the control plane cannot touch it

The Metal captures give an independent route to the same slope and land in the
same place. This is the strongest single piece of evidence in this document.

Compute GPU time was 231.20 ms at 50,000 rows and 536.96 ms at 200,000 rows, at
comparable clock states. The dispatch count was identical to the last unit
across every capture, 22,107 command buffers, and the serialization count
identical at 3,208. So:

```
compute slope = (536.96 - 231.20) ms / 150,000 rows = 2.04 us per row over 100 rounds
```

The wall-clock slope from the end-to-end benchmark is 2.24. The GPU-clock
compute slope from the Metal trace is 2.04. Two completely different
instruments, measuring two different quantities, on two different days,
agreeing to within 10 percent.

**The marginal cost of a row on the GPU is the kernel executing.** The control
plane's cost is a constant, because the number of synchronizations is a
constant: 32.1 per round whether the round has 50,000 rows in it or 400,000.

Decompose the measured 3.575 seconds at one million rows:

```
kernels executing        ~2.04 to 2.24 s   (57% to 63%)
everything else          ~1.33 to 1.53 s   (37% to 43%)
```

Delete one hundred percent of the control plane, every readback, every host
wait, every scrap of submission latency, and the run still takes 2.04 to 2.24
seconds, because that is how long the kernels take. The realistic floor for a
perfect device-resident tree at one million rows is:

> **2.1 to 2.4 seconds, against LightGBM's 2.86.**

A genuine win of 1.2x to 1.36x, worth having, and two and a half times short of
the plan's estimate. To reach 0.8 to 1.0 seconds the per-row slope has to fall
from 2.24 microseconds to between 0.8 and 1.0, a factor of 2.3 to 2.8, and
there is exactly one place that factor can come from.

### 2.3 The 85.8 percent is a 200,000 row fact and it inverts at one million

The plan's central reasoning compares two percentages: the control plane is
85.8 percent of a round and compute is 22.9 percent, so the control plane is
roughly four times the prize. Both are correctly measured. Both are measured at
200,000 rows. Neither survives the trip to one million.

Using the fitted model of section 2.1:

| rows | measured GPU wall | fixed part | compute part | compute share |
|---|---|---|---|---|
| 50,000 | 1.635 s | 1.33 s | ~0.11 s | ~7% |
| 250,000 | 1.893 s | 1.33 s | ~0.56 s | ~30% |
| 1,000,000 | 3.575 s | 1.33 s | ~2.24 s | ~63% |
| 5,000,000 (extrapolated) | ~12.5 s | 1.33 s | ~11.2 s | ~89% |

"The kernels are a fifth of the prize" is true at 200,000 rows, false at one
million where they are roughly three fifths of it, and badly false at five
million where they are nearly all of it.

The uncomfortable consequence: **the plan's mechanism delivers its largest
multiple exactly where the competitive claim is not being made, and its
smallest multiple exactly where it is.** Removing 1.33 seconds of fixed cost
turns 1.635 seconds into about 0.3 at 50,000 rows, a 5x improvement, and turns
3.575 into about 2.24 at one million, a 1.6x improvement. The target shape is
where this mechanism pays least.

### 2.4 Where the missing factor actually is

If the slope is 2.24 microseconds per row and the kernels are the slope, it is
worth asking whether the kernels are good. They are not, and the reason is
already recorded in the trace document without anyone drawing the conclusion.

`docs/METAL_TIMELINE.md:406-412` reads the launch site and finds that on Metal
the feature group baseline is 2, so 50 features become 25 threadgroup blocks,
and each block walks every row of its tile reading a 4-byte row index, an
8-byte gradient and hessian pair, and one 1-byte bin per feature. Fourteen
bytes per row per block, across 25 blocks. Count the traffic at one million
rows for a root pass:

```
what the kernel loads      25 blocks * 1,000,000 rows * 14 bytes  = 350 MB
what the data actually is  1,000,000 * (4 + 8 + 50 bytes)         =  62 MB
```

**The row index and the gradient and hessian pair are re-read twenty five times
per histogram pass.** At 200,000 rows some of that is absorbed by cache, which
is exactly why the document's own bandwidth arithmetic produced the impossible
figure of 121.8 GB/s against a 120 GB/s memory system. At one million rows the
unique footprint is 62 MB, far past any cache on this part, so the re-reads
become real DRAM traffic.

The ratio 350 to 62 is 5.6x, and 2.24 seconds divided by 5.6 is 0.40 seconds.
That is almost certainly where the plan's missing roofline figure came from, and
**it is a statement about the histogram layout, not about the control plane.**

There is direct evidence for the superlinearity this predicts. The 100,000 and
400,000 row captures both ran at 100 percent Minimum performance state, so
unlike every other pair in that set they are directly comparable to each other.
Their root histogram kernels are 766.6 and 3386.0 microseconds: four times the
rows produced 4.42 times the time. The kernel is **superlinear in rows**,
exponent about 1.07, which is what losing cache reuse looks like. Every
extrapolation in section 2.2 is therefore a lower bound.

### 2.5 Steps 2 and 3 are in tension with each other

Step 2 takes packed int16 in one atomic. Step 3 declines to change the kernels.

The current histogram cell is three Int32 planes, 12 bytes
(`docs/design/GPU_LEVELWISE.md:319-321`). Threadgroup memory per block is
`group * bin_cap * 12` bytes, which is what caps the feature group width. A
narrower cell is precisely what would allow the group width to rise from 2 to 6
or 8, which is precisely the change that cuts the 25x re-read of section 2.4
down to 3x or 4x, which is where the missing factor lives.

So the plan proposes the enabler and declines the thing it enables. It takes
the change that carries the accuracy risk and refuses the change that carries
none, on the grounds that the kernels are "proven correct and extensively
tested." Widening a feature group does not change any arithmetic. It changes
which thread accumulates which cell, and there is already a five-rung compile
time ladder for it (`gpu_active_rows.mojo:2864-2954`, groups 2 through 16
instantiated). It is the lowest-risk, highest-value change on the table and it
is the one being ruled out.

## 3. Objections, ranked by probability times damage

### O1. The 0.8 to 1.0 second estimate is refuted by the project's own numbers

**P(right) 0.90. Damage: total. This is the plan's entire justification.**

*Argument.* The GPU's marginal cost is 2.24 microseconds per row and the Metal
trace says that marginal cost is kernel execution, not control plane. Removing
every host wait leaves the kernels, and the kernels are 2.0 to 2.24 seconds at
one million rows. The estimate requires them to get 2.3x to 2.8x faster, which
is the work step 3 excludes. Section 2.2 in full.

*Evidence.* Two independent instruments agree on the slope: 2.24 microseconds
per row from three end-to-end wall clocks in `bench/results/profile_2026-08-15/`
and 2.04 from GPU hardware timestamps in
`bench/results/metal_timeline_2026-08-15/`. The synchronization count is
invariant in rows by measurement (32.1 per round at 50,000 and at 400,000), so
the control plane contributes nothing to the slope by construction.

*The check.* One interleaved GPU-only sweep at 1M, 2M, and 5M rows in a single
process and time window, slope fitted over the upper two points. If it is at or
above 2.0 microseconds per row, the target is unreachable without kernel work
and must be restated as roughly 2.2 seconds. If it comes out below 1.0, this
objection is wrong and I want to know why. One benchmark run, no engineering.

### O2. Packed int16 does not work at the target shape, by the repository's own bound

**P(right) 0.95, the arithmetic is the repository's own. Damage: very high.
Accuracy is the hard constraint and the failure is silent.**

*Argument.* The shipped scheme accumulates in **Int32** with `FIXED_ONE = 2^30`
(`quantized_gradient.mojo:327`), shared and global atomics both 32-bit
(`gpu_active_rows.mojo:1232-1234`, `1270-1283`), three separate planes. The
repository's own row-ceiling formula, asserted in
`tests/test_gpu_portability.mojo:391-392`, is `safe_max_rows = 2 * (INT_MAX -
TARGET)`, which for a W-bit field collapses to `2^(W-1) - 2`:

| accumulator | target | headroom | max rows |
|---|---|---|---|
| Int32 (shipped) | 2^30 | 2^30 - 1 | **2,147,483,646** |
| int16 (proposed) | 2^14 | 16,383 | **32,766** |

**The row ceiling drops by a factor of 65,536, to 32,766, against a target of
1,000,000.** At one million rows the rounding residue alone, `n/2 = 500,000`,
is 15.3 times the entire 32,767 of headroom before a single gradient is added.
`tests/test_gpu_portability.mojo:400-403` asserts `safe_max_rows > 1_000_000_000`
with the message "fixed-point row ceiling has dropped into a realistic dataset
size," and it fails on a CPU-only CI runner with no GPU present.

*Three distinct failure modes, all reachable at the target shape.*

**Underflow collapse.** Under the shipped `SCALE_MAGNITUDE_SUM` rule at 16 bits
the mean per-row quantized magnitude at one million rows is `16384/1e6 = 0.0164`.
Round-to-nearest sends a row to zero unless its gradient is at least 30.5 times
the mean absolute residual, roughly 24 sigma for Gaussian residuals.
**Essentially every row quantizes to zero, every gain is zero, and every tree is
a root-only stump.** Break-even is at 16,384 rows.

**Unconditional hessian overflow.** For logloss the hessian is `p(1-p)` in
`(0, 0.25]`, one-sided, so there is **no cancellation at all** in the hessian
plane. A bin of `m` rows accumulates exactly `m * q_h`, so any bin holding
32,768 or more rows overflows the 16-bit field **whatever scale is chosen**. At
one million rows that is routine rather than exotic: the missing bin, a dominant
categorical level, any feature with fewer than about 31 effective distinct
values, and every configuration with a small `max_bin` (validation permits 2,
where a bin holds roughly 500,000 rows).

**Carry, not wrap, and this is the qualitative difference.** Two int16 fields in
one 32-bit atomic means overflow of the low field is not a wrap of that field,
it is a plus or minus one carry into its neighbor. Roughly 15 carries per bin at
one million rows, each silently adding a full lattice unit of hessian per
gradient overflow. This destroys three shipped properties at once. Sibling
subtraction stops being exact, and today it survives even wraparound
(`histogram.mojo:1216-1218`: "equal even under Int32 wraparound, since repeated
addition and multiplication agree modulo 2^32") because the modular argument
carries; a packed subtract borrows across the field boundary and the modular
argument does not. Order-independence dies, which
`tests/test_gpu_portability.mojo:388-390` names as the thing "the whole
bit-exactness argument rests on." And it fights the constant-hessian elision:
for squared error, L1, Huber, and quantile the hessian plane is currently **not
accumulated at all** (`histogram.mojo:117-131`), so packing reintroduces exactly
the traffic that elision removed, for a saving of one third rather than one half
because the count plane must stay Int32.

*The prior art in this repository points the other way.*
`docs/QUANTIZED_GRADIENTS.md:190-194`: "The width is a **per node** question,
**deliberately**. LightGBM promotes its histogram bit width dynamically for the
same reason: **the root of a million-row dataset needs a wide accumulator** and
the leaves near the frontier do not." A fixed packed int16 is precisely what
LightGBM does not do. The machinery to decide when 16 bits is safe already
exists in the repository, unused: `INT16_LIMIT`, the `WIDTH_16/32/64` ladder,
`width_for_bound`, `accumulator_width` (`quantized_gradient.mojo:243-248`,
`:360-362`, `:1067`).

*The check, and it must happen before any code.* Write the bound as an
inequality relating accumulator width, bin capacity, rows in the worst bin, and
the scale, and evaluate it at 1,000,000 and 5,000,000 rows with a skewed
single-bin feature and logloss. If int16 is still wanted after that, route it
through the existing per-node width ladder so the root and any dense bin get 32
bits, and **keep the two planes in separate words** so a bound miss wraps
instead of carrying. An afternoon of arithmetic.

### O3. Most of the control plane prize may already ship, behind a flag nobody has benchmarked

**P(right) 0.70. Damage: very high, because if right the plan's incremental
value is roughly a fifth of what it claims.**

*Argument.* The plan's headline is "reduce host waits from 31 per tree toward
1." The shipping resident device-search loop already takes one wait per **batch**
rather than per split (`gpu_split_search.mojo:3134-3140`, called from
`train_gpu.mojo:1589`), and under `grow_policy=depthwise` a batch is a whole
planned level. `docs/design/GPU_LEVELWISE.md:12-21`:

> "`growth_policy.plan_level` hands that loop a whole planned level, which it
> commits and enqueues back to back and then searches in one launch pair, so a
> level costs **one host wait instead of one per split** ... At the default 31
> leaves that is 5 waits for a tree rather than 30."

and immediately after, at `:25-27`:

> "**No benchmark has been run on any of it.** The counts above are counts."

Price it against the trace. There are 32.1 serialization points per round, of
which roughly 30 are the per-split waits and about 2 are outside the tree.
Depthwise batching takes the 30 to 5, so 32.1 becomes about 7. Section 2.1's
fitted fixed cost is 1.33 seconds; the sync accounting attributes about 450
microseconds of genuinely serial cost to each of 3,208 readbacks, or 1.44
seconds, which is the same number. So:

| step | syncs per round | fixed cost | GPU at 1M |
|---|---|---|---|
| today, leaf-wise | 32.1 | ~1.33 s | 3.575 s |
| `grow_policy=depthwise` on the shipped resident loop | ~7 | ~0.29 s | **~2.53 s** |
| full device-resident tree | ~1 | ~0.04 s | **~2.28 s** |

**The already-shipped flag captures roughly 78 percent of the synchronizations
the plan proposes to remove. The plan's marginal value over it is about 0.25
seconds at one million rows, not 2.6.**

*The honest caveats, because this objection is the one most likely to be wrong.*
Depthwise growth is a different model, not a faster route to the same trees, and
`GPU_LEVELWISE.md` sections 8 and 10 are emphatic about that: quality must be
compared as time to matched quality after tuning each mode separately, and fits
are not comparable seed for seed. LightGBM, the accuracy baseline, is leaf-wise.
It is also unverified that the depthwise resident path stays eligible at one
million rows rather than falling back. And a fourth complication: the two ideas
are currently **incompatible in the code**. `gpu_tree_tables.mojo:207-210`
refuses depth-wise for the tree-resident path. So the plan would have to unify
them or choose between them, and it has not noticed that it must.

*The check, and it is the cheapest experiment in this document.* Run
`grow_policy=depthwise` against the default on the existing GPU path at one
million rows, interleaved, reporting wall clock, sync count, and validation
metric. It uses only shipped code and requires no engineering.

**Run on 2026-08-15, and O3 is upheld.**
`bench/results/sweep2_2026-08-15/RESULTS.md`, five repeats interleaved:
`grow_policy=depthwise` on the shipped resident device search trains
1,000,000 x 50 in **2.587 seconds** against leaf-wise's 3.756, at a 0.3
percent spread. The table above predicted ~2.53 and the measurement landed
2 percent from it, so the pricing route was sound. The depth-wise path did
stay eligible at one million rows, which was the open question. Two of the
three things the check was asked to report did not come back: the sync count
per arm was not collected, and neither was any validation metric or training
loss, so the quality half of this objection is exactly as unresolved as it
was before the run. The incompatibility in `gpu_tree_tables.mojo` also stands,
and the tree-resident path was found in the same sweep to fail whenever it
actually executes, so the choice O3 says the plan must make is now between a
working batched path and a broken unbatched one.

### O4. Two of the four justifying numbers do not exist and the real ones point the other way

**P(right) 0.95, checked exhaustively. Damage: high.**

Section 1.3 has the detail. The reason this matters beyond bookkeeping: if the
plan's author believes we are 33 percent worse per marginal row than LightGBM,
then our fixed cost is bad and our slope is bad and a large structural change is
needed everywhere, which is the plan's shape. If the true reading is that our
slope is 10 percent better and our fixed cost is 3.5x worse, then the fixed cost
is the only defect, it is bounded, and the payoff is bounded with it. **The two
readings recommend different sizes of project.**

*The check.* Ask where the two numbers came from. If a derivation exists it
should be committed. If not, restate the justification without them and see
whether it still recommends the same scope. Zero cost.

### O5. Every LightGBM comparison is single-repeat, cross-process, and has no clock record

**P(right) 0.95. Damage: medium to high, because LightGBM is the target the
whole plan is aimed at.**

*Evidence.* `bench/bench_train_gpu.mojo` has no LightGBM arm; the LightGBM
figures come from the standalone `bench/bench_lightgbm.py`, which has no repeat
loop and no median. `bench/README.md:316-320` states the rule being broken:

> "Arms are compared only inside one invocation. This machine drifts by a factor
> of 2 to 3 across time windows, so a number from one run and a number from
> another are not comparable even when the two commands were identical."

`bench/results/PROFILE_PROTOCOL.md:36-40` requires five repeats and never fewer
than three. LightGBM got one.

Three specific casualties.

**The 50,000 row CPU win is inside its own noise.** 0.563844 against 0.594365 is
a 5.3 percent margin against a recorded mojotrees CPU spread of 11.3 percent,
with no spread at all on the LightGBM side. `PROFILE_PROTOCOL.md:39-40` says a
pair whose difference is inside the larger arm's spread is indistinguishable and
is recorded as such. It was instead promoted to "the first shape on which this
library is faster than LightGBM at anything." That claim is not established. At
250,000 rows our CPU is 1.66x behind.

**No GPU performance state was recorded for any one million row run.** The
thermal block in `bench/results/profile_2026-08-15/header.txt` is a dry run:
`run id thermal-PENDING`, "No thermal warning level has been recorded." The
Metal captures prove the device silently picks Minimum or Maximum run to run,
that Minimum is roughly 2.8x slower on the root kernel, and that this is the
most plausible mechanism for the documented 2-3x drift. The 3.575 and 2.858
figures have no clock record between them.

**The multiclass LightGBM figure is a quoted historical number, and it is
unstable.** "6.7 to 6.9 seconds" comes from
`bench/real_data/results/20260815T023123Z/records.csv`, about 18 hours earlier,
on real covertype rather than the synthetic 465,000 x 54 x 7 shape mojotrees was
measured on, through a different harness. The session 43 minutes before it,
`20260815T014842Z`, records the same LightGBM covertype arm at **4.45 to 4.68
seconds**. The reference point moves by 1.5x between adjacent windows.

*The check.* One interleaved run with LightGBM as an arm inside the same
process, five repeats, GPU performance state printed before and after, at 50k,
250k, and 1M. Until that exists there is no defensible statement about how far
behind LightGBM we are at any shape.

### O6. ~~Half the drains are uploads, and on Metal there is no asynchronous copy to replace them with~~ STRUCK 2026-08-16

> **STRUCK, 2026-08-16, and the reason is a measurement.** O6's mechanism is
> right and is not what is struck. Its **premise** is that because an upload
> drains, an upload is a wait, so the plan's failure to remove uploads leaves
> money on the table and removing them would be a prize. That premise was
> registered as a prediction in `bench/results/SESSION_QUEUE.md` and measured
> in Session III: removing thirteen copies per tree, nine of them uploads,
> **measured** 0.016 seconds at 1,000,000 x 50 against a predicted 0.64, which
> is a null under M0 (`bench/results/session3_2026-08-16/RESULTS.md`).
> `docs/GPU_PORTABILITY.md` section 6.1.1 records the withdrawal.
>
> There is no prize in the uploads. Draining a queue that holds nothing costs
> nothing, and nothing is queued behind a table upload. What costs is the
> **round trip**, which is what the plan was removing all along, and the same
> session measured that at 0.75 seconds, resolved. **So the plan's mechanism,
> which O6 called half-aimed, was aimed at the right half.** O6 is struck in
> the direction of the plan.
>
> What survives O6 and is not struck:
>
> - the disassembly, which is correct and is now section 6.1;
> - "the docstring is wrong", which was true and has since been fixed in
>   `gpu_split_search.mojo`;
> - the ordering consequence, that an upload fences the queue and therefore
>   bounds how deep a launch stream can get;
> - the portability consequence, that MAX offers no asynchronous copy, no
>   Metal stream and no `MetalDeviceGraphBuilder`, so a design that needs an
>   asynchronous upload has nothing to build it from here.
>
> Those are hazard and portability findings and they should still be honored.
> The check O6 proposed, splitting the blits by direction, is no longer worth
> running: the direction split does not change the answer, because neither
> direction of copy predicts time. **The text below is left unedited.** A
> refuted objection that was registered and then measured is the record of a
> working process, and editing it to look right in hindsight destroys the only
> thing writing it down was for.

**P(right) 0.85, from disassembly of the shipped runtime. Damage: medium to
high, and it cuts both ways.**

*The finding.* `libMGPRT.dylib`'s Metal driver implements
`enqueueCopyToDevice` and `enqueueCopyFromDevice` identically, and both are
synchronous:

```
[queue commandBuffer]      ; an empty command buffer, no encoder
[cmdbuf commit]
[cmdbuf waitUntilCompleted]
memcpy                     ; host memcpy into the shared MTLBuffer
```

The same shape appears in `enqueueSetMemory` and in `synchronize`. **On Metal,
`DeviceContext.enqueue_copy` is a full queue drain plus a host memcpy in either
direction, despite the name.** Those empty command buffers are precisely the
"3,225 carrying none" that `docs/METAL_TIMELINE.md:174` records without
explaining.

*Why it matters to the plan.* The plan is described entirely in terms of
removing the **readback**: kernels read the chosen split from device memory
rather than the host reading it out. But the resident split search also stages
host-authored per-node tables **to** the device every batch, in
`GpuSplitSearcher._stage_frontier` (`gpu_split_search.mojo:2773-2775`, three
copies) plus the four staging copies in `_copy_tables`. Each of those is a
drain. The docstring at `gpu_split_search.mojo:3230-3232` states that
`enqueue_frontier` "Does not transfer and does not synchronize," and **on Metal
that docstring is wrong**: it transfers and drains before it enqueues anything.

So the plan's mechanism, as stated, removes the downloads and leaves the
uploads. To remove the uploads, the device has to generate the per-node tables
itself: the feature set, the allow mask, the monotone bounds, and the histogram
slot index. And those are exactly the four things `gpu_tree_tables.mojo:186-210`
already **refuses** to support on a device-resident tree, because per-node
feature draws need an RNG the device does not have, interaction masks need the
ancestor chain, and the monotone midpoint needs Float64 that Apple GPUs do not
have.

*The way this cuts in the plan's favor.* At default parameters, with no
per-node feature fraction, no interaction constraints, and no monotone
constraints, those tables are constant across a tree and could be uploaded once
per tree rather than once per batch. If that is right, the prize is larger than
the plan claims and easier to take than this objection suggests. **Nobody has
counted how many of the 32.1 drains per round are uploads and how many are
downloads, and that is a one-line change to the reduction script over a trace
that already exists.**

*The way it cuts against.* There is no fallback if the tables must stay
host-authored. MAX exposes no asynchronous copy on Metal, `"Metal stream not
implemented"` is a literal string in the runtime so there is no second queue to
overlap with, and while the runtime ships `CUDADeviceGraphBuilder` and
`HIPDeviceGraphBuilder` there is **no `MetalDeviceGraphBuilder`**, so
capture-and-replay batching is unavailable too. That last clause was derived
from a symbol table when it was written and has since been **verified by
execution**: `DeviceGraph.create` raises on an M4 at builder creation, before
a node is added (`docs/GPU_PORTABILITY.md` section 6.5). The only escape is
writing
staging data directly into a `HostBuffer` the kernel reads through unified
memory, and `docs/APPLE_UNIFIED_MEMORY.md` already records that the host-direct
experiment was wrong and the map-write path was slower.

*The check.* Re-run `bench/apple/metal_timeline.py` over the existing 200,000
row trace, splitting the 3,206 blocking blits by direction. It costs nothing,
uses a capture already on disk, and it tells you whether the plan is aimed at 30
of the 32 drains or at 12 of them.

### O7. Indirect dispatch does not exist, so the counts that size the grids ride in the record the plan wants to stop downloading

**P(right) 0.85. Damage: high, and it is the plan's largest unpriced technical
dependency.**

*The finding.* `DeviceContext.enqueue_function` in this MAX version accepts only
host-side integer grid dimensions. Every one of the 60-plus call sites in this
repository passes a host `Int` or a tuple of them. There is no indirect
dispatch, no device-side grid sizing, no indirect command buffer support, and no
pre-recorded launch replay. `libMGPRT.dylib` does contain the string
`dispatchThreadgroupsWithIndirectBuffer:` but so does it contain the entire mesh
shader and tessellation family that MAX certainly does not use; that is
Objective-C protocol metadata pulled in by linking Metal, not evidence of use.

The strongest structural evidence is internal. `src/mojotrees/gpu_tree_tables.mojo`
was written specifically to remove the per-split host round trip, and its
solution is `_pick_and_commit_kernel` launched at **`grid_dim=1,
block_dim=PICK_THREADS`** (`:1503-1504`), a single fixed-size threadgroup doing
integer bookkeeping. The author went out of the way to make the device-resident
step grid-invariant. If a device-resident grid were available, that module would
not have needed that shape.

*Why it bites.* Two grids in the per-split loop depend on the current node's row
count: the row partition's `blocks`, `tiles`, and `copy_blocks`
(`gpu_active_rows.mojo:2294-2305`, `_partition_grid` at `:275-331`), and the
histogram's tile dimension `n_tiles` and its `rows_per_tile`
(`gpu_active_rows.mojo:2646-2651`, `gpu_tiling.resolve_tiling`). Today those
counts arrive on the host inside the **same 136-byte split record** that carries
the chosen feature and bin, as exact integers from the histogram counts, and the
host maintains `LeafRangeTable` from them. So a device-resident tree that stops
downloading the record loses the ability to size two of its own grids, and there
is no third option:

- **(i) keep downloading the record just for the counts.** That is the wait the
  plan set out to remove, so the plan collapses to a no-op.
- **(ii) fix the grids at the worst case with a device-side early exit.** This is
  the only real option and the plan does not mention it.

*Pricing option (ii), and here the news is better than I expected.* The
histogram grid is `(ceil(n_slots / GROUP), n_tiles)`, and the expensive
dimension is **features, not rows**: 25 blocks by 2 tiles at 200,000 rows, and
perhaps 25 by 8 at one million. Worst-casing the tile dimension costs a few
hundred threadgroups that read a counter and exit, on the order of single-digit
microseconds per dispatch. The partition's `copy_blocks = ceil(n / threads)` is
genuinely row-proportional and would be 3,907 blocks at one million rows, but
`blocks` is capped at 256 by `partition_block_cap`. Rough total for the
worst-case-grid tax: **tens of milliseconds per 100 rounds, not seconds.**
Tolerable. But it is unmeasured, unmentioned, and it is a direct tax on the
compute term, which section 2.2 shows is 63 percent of the run at the target
shape.

*The check.* Confirm with Modular whether device-side grid sizing exists or is
planned on Metal. If it does not, add option (ii) to the plan explicitly and
measure the early-exit dispatch cost with a microbenchmark before committing to
the design.

### O8. The plan has misidentified which promise it is relaxing, and one relaxation lands on determinism

**P(right) 0.95 on the framing. Damage: medium, but it changes what is
permitted under the owner's own constraint.**

*The framing error.* There is no CPU-versus-GPU bit-identity in this project to
relax. `docs/NUMERICS.md:48-49`, bold in the source:

> "Narrow in the other direction, because **CPU and GPU are not bit-identical to
> each other and are not intended to be.**"

and `:58-60`: "So the promise is **per backend**. Within a backend it is
bitwise. Across backends it is to Float32 precision on the float planes and
exact on the integer planes." The same contract is repeated in
`src/mojotrees/histogram_gpu.mojo:58-60`. Cross-backend tests are all
tolerance-based already; the count planes are the only exact comparisons and
they stay exact. **Zero tests fail from dropping a CPU-GPU bit-identity that
does not exist.**

*What actually gets relaxed, and why it matters here.* The promise that exists
is bit-identity **within** a backend across worker counts, and it is
user-facing. `README.md:287`: "Every path is **bit-identical to the serial one
at any worker count**: feature tasks keep each feature's sum inside one task,
**row blocks are used only where nothing is summed across rows**." That sentence
is a direct, published prohibition on relaxation one. The claim also ships in a
release artifact, `packaging/matrix/accelerators/index.toml:28`. And
`histogram.mojo:90-116` pre-refuses row blocks by name, at length, ending: "**Anyone
reaching for row blocks is proposing to give all of that up, and should say so
explicitly rather than discover it by breaking parity.**"

*The determinism question, stated precisely, because this is the part that is an
objection under the review's rules.* A fixed fold order over row blocks
**restores reproducibility only if the block count is a function of the row
count alone.** If the block count is derived from the worker count, which is the
natural implementation and which `MOJOTREES_NUM_WORKERS` makes user-visible,
then `histogram.mojo:108-110` is exactly right: "The block count would enter the
result, `MOJOTREES_NUM_WORKERS` would stop being a scheduling knob." That is a
determinism regression, shipped by accident, under a directive that says
determinism is kept.

**The plan is admissible on this point if and only if it states that the block
decomposition is a function of the data alone and workers claim blocks from it.**
It does not currently say so.

*One further cost, on accuracy rather than tidiness.* Row-block folding breaks
exact CPU sibling subtraction, because the parent's fold order over the whole
node differs from either child's fold order over its own rows, so `parent -
left` is no longer `right`. On the CPU that is Float64 rounding rather than an
integer error, so the magnitude is small, but it is a real accuracy degradation
rather than a bookkeeping one, and it compounds down a 31-leaf tree.

*Calibration on how much things move.* `docs/NUMERICS.md:799-817` records that
flipping `--fp-mode contract=off`, a strictly smaller perturbation than
reassociating every histogram sum, "fails all six fixtures," with one leaf value
moving **93 ulps** and two others straddling zero. Six golden fixtures
(`tests/test_golden_bits.mojo`, six tests at `:1780-1801`) plus
`test_histogram_reference`, `test_cpu_parallel`, `test_kernels`,
`test_cpu_feature_group`, and `test_cpu_dispatch` all move, and all of them run
on CPU-only CI. Under the fixture's own policy (`:82-95`) that is a permitted
regeneration, not a broken test, provided the ulp movement is stated and
reviewed. It is not free, but it is not an obstacle either.

*The third relaxation is the safe one and should be said so.* A device-computed
power-of-two scale is deterministic: a maximum reduction is exact and
order-independent, and the power-of-two part costs at most one bit of thirty.
Two caveats: the host must **read back** the scale rather than recompute it, or
the hybrid replica cannot match; and the final fold must stay in host Float64.
`gpu_objectives_native.mojo:1229-1234` explains that the threadgroup partials
come back to the host as 2 KB independent of row count and are summed in Float64
there, which is "more accurate than a Float32 device-side final reduction would
be." Moving that fold onto the device makes it a Float32 tree reduction whose
order depends on launch geometry, which breaks the one promise
`NUMERICS.md:33-36` genuinely makes.

### O9. The only arm-for-arm real multiclass measurement says the GPU loses

**The fact is on disk, P 1.0. P(it generalizes) about 0.4. Damage if it
generalizes: high, because multiclass is the plan's second-best evidence.**

`bench/real_data/results/20260815T023123Z/records.csv`, real covertype at
464,958 x 54 x 7, all three arms in one session: CPU 36.18, 36.86, 51.16; GPU
57.59, 56.08, 56.51; LightGBM 6.73, 6.85, 6.82. **On real covertype the GPU is
1.55x slower than our own CPU**, where on a synthetic shape of nearly identical
dimensions it is 1.63x faster. A 2.5x swing between synthetic and real data at
the same row count, feature count, and class count, unexplained.
`RESULTS.md:178-180` acknowledges it and does not resolve it.

*Why it matters to this plan.* If the gap is caused by real data having many
low-cardinality and categorical features, which covertype does, then the
small-node tail is far longer on real data than the synthetic profile suggests,
and the small-leaf objection in section 4 applies with much greater force. It
also means the plan's second-best evidence for the GPU direction may be a
property of the data generator.

*The check.* Run the multiclass arm interleaved on real covertype and on the
synthetic shape in one process, and take a Metal capture of the covertype run.
This is the single most anomalous number in the repository and nobody has looked
at it.

### O10. Reducing waits does not reduce launches, and the launch stream has its own floor

**P(right) 0.75. Damage: medium to high, and it lands on the shapes where the
plan's multiple is largest.**

A device-resident tree removes the host **waiting**. It does not remove the host
**submitting**. The plan keeps eight launches per split and 221 dispatches per
round, and every one is a command buffer the host must create, encode, and
commit. `docs/METAL_TIMELINE.md:462-473`: when the host is not blocked it commits
a new command buffer every **14.58 microseconds**, and this is a device and
driver property, confirmed by the median moving only from 12.67 to 12.62
microseconds between the 50,000 and 200,000 captures while compute changed by
2.3x.

```
22,107 command buffers * 14.58 us = 322 ms of pure host submission per 100 rounds
```

| rows | GPU compute per 100 rounds | host submission floor | which binds |
|---|---|---|---|
| 50,000 | 231 ms | 322 ms | **host** |
| 200,000 | 537 ms | 322 ms | GPU |
| 1,000,000 | ~2240 ms | 322 ms | GPU |

**At 50,000 rows a perfect device-resident tree is host-submission bound.** That
still beats LightGBM's 0.594 and still beats our own CPU, so it does not sink
the small-shape story, but it caps it at about 0.32 seconds and the cap is
invisible in the plan.

The sharper form concerns the small dispatches. Of the 18,703 compute dispatches
at 200,000 rows, 10,207 are shorter than 10 microseconds and carry 8.4 percent
of compute time. The GPU consumes a 5 microsecond kernel in about 8 microseconds
including the back-to-back gap; the host produces one every 14.58. **For the
small-leaf tail the GPU is roughly twice as fast at consuming launches as the
host is at producing them**, so that entire tail runs at host speed once the
waits are gone. Today that is invisible because the waits dominate. Tomorrow it
is the visible cost of the tail. Section 4 quantifies it.

*The check.* A microbenchmark that submits N trivial command buffers back to
back with no waits, at N of 100, 1,000, and 22,000, measuring wall clock per
buffer. It gives the submission floor directly, and section 4.3 says why the
curve will have a knee in it.

## 4. What the device-owned tree makes worse

A plan that counts only its benefits is the one that surprises people six weeks
in. These are costs incurred on success, not risks incurred on failure.

### 4.1 Debuggability, and this is the one that will hurt

Today every split passes through the host, so a developer can print the tree
after every split, check the device's actual left-row count against the count
predicted from the histogram, and compare a host replica of the histogram
against the device's. That last one is the project's oracle:
`docs/ARCHITECTURE.md:157` names the CPU trainer as "the reference implementation
the GPU path is verified against," and `histogram.build_histogram_subset_replica_into`
(`histogram.mojo:1167`) replays the device pipeline on the host bit for bit,
using the same Float32 inputs, the same scale, Int32 accumulation, and the same
dequantization.

A device-resident tree removes the observation point. Relaxation two makes the
replica far harder to sustain, because the host would have to reproduce packed
field carry and borrow semantics under a 32-bit atomic, which no vendor
specifies identically across Metal, CUDA, and ROCm. **Both halves of the current
debugging story are being retired in the same plan, and no replacement is
named.**

Auditing what specifically dies, most of the debug surface survives because most
of it is per tree rather than per split. Two things die outright and they are
the two that matter.

**The row-count cross-check.** `MOJOTREES_GPU_VERIFY_ROWS=1` downloads the
device's actual left-row count and raises if it disagrees with the count derived
from the histogram (`gpu_active_rows.mojo:2456-2461` for the readback,
`:2493-2501` for the check: "device left count disagrees with the histogram
count"). This is *the* check that the routing rule the host counted with is the
routing rule the device applied, and it is one blocking readback per split by
construction. It has to become a device-side comparison that sets a status word,
and `gpu_tree_tables.mojo` reserves **no status code for it** among its five.

**The near-tie margin.** `GpuSplitRecord.runner_gain`, `margin()`, and
`is_near_tie()` (`gpu_split_search.mojo:2030-2034`, `:2054-2067`) are decoded
from a downloaded record. They are the diagnostic for the Float32 versus Float64
tie sensitivity that `docs/NUMERICS.md` rests on. If records stay on the device
and only the finished tree comes home, no host code ever sees a per-split
margin and that diagnostic has no data source at all. This one deserves emphasis
because relaxation two makes near-ties **much more common**, not less: at
reduced gradient resolution, many more candidate splits score identically. The
plan simultaneously increases the tie rate and removes the instrument that
measures it.

Two more are hollowed rather than killed. `GpuActiveRows.check_frontier`
(`:2544-2572`, three distinct raises) validates the host frontier against the
host range table; if the device owns the frontier there is nothing to compare
against without downloading it. And the `fenced` profile modes work by inserting
`ctx.synchronize()` after each split (`train_gpu.mojo:1712`, `:1758`), so a
traced run becomes structurally unlike an untraced one in exactly the dimension
being optimized.

A trace mode is the obvious answer and it should be specified now rather than
after the first wrong tree at a customer site. What it must carry, per committed
split: node id, chosen feature, chosen bin, gain, runner-up gain, left and right
row counts, left and right gradient and hessian sums, and the stop reason. At 64
bytes per split, 30 splits, and 100 trees that is about 192 KB per fit and one
extra readback per tree, which is 100 readbacks at 606 microseconds, or 61
milliseconds. **The cost is negligible and the cost is not the point.** The point
is that a trace tells you what the device did and only a replica tells you
whether it was right. If the replica goes, the trace needs a tolerance-based
comparison instead, and somebody has to decide what tolerance means for a chosen
bin index, which is a discrete quantity where "close" has no meaning. Two splits
that differ by one bin are not approximately equal, they are different trees.

*Dated note, 2026-08-16.* The hybrid scheduler discussed in the next
paragraph has been deleted. The argument above is unaffected and was in fact
acted on: `histogram.build_histogram_subset_replica_into` was kept precisely
because this section names it as the project's oracle, and
`tests/test_host_replica.mojo` is where it is now asserted. The observation
point this section says a device-resident tree removes is therefore still
standing on the host side.

There is one piece of good news here worth recording. The hybrid scheduler
already fails safe rather than fails wrong. `MODE_MIRROR` compares and discards,
so a mismatch is a diagnostic; `MODE_REPLICA` is downgraded to mirror for its
first accepted leaf and, on mismatch, sets `REPLICA_REFUTED` and turns hybrid off
for the whole fit (`train_gpu.mojo:2274-2281`, `:2338-2343`). So a divergence
does not produce a wrong tree. It silently evaporates the measured 1.20x. That
is the right failure mode and any device-resident replacement should be built
with the same property.

### 4.2 The small-leaf tail, which is currently hidden behind the wait

This is plausibly the largest hole in the estimate and it has just been
independently confirmed. Batching seven multiclass classes into one launch
measured indistinguishable from one at a time. That was predictable, because
batching amortizes dispatch and dispatch is not the cost today. But it is
evidence about the wrong world: it says nothing about the world after the waits
are removed, and it will not be indistinguishable there.

Quantifying from the 200,000 row capture: 7,005 dispatches in the 2 to 5
microsecond class and 3,201 in the 5 to 10 class. A 2 microsecond kernel on a
10-core GPU is doing essentially nothing; it is the dispatch floor. Those 10,207
dispatches are 55 percent of all compute dispatches and 8.4 percent of compute
time. After the fix they cost the greater of their GPU time (45 ms) and their
share of the host submission stream (10,207 times 14.58 us = **149 ms**). So the
tail is a **149 millisecond floor per 100 rounds, invariant in rows**.

| post-fix run | tail cost | tail share |
|---|---|---|
| ~2.2 s at 1,000,000 rows | 149 ms | 7% |
| the plan's claimed 0.8 to 1.0 s | 149 ms | 15% to 19% |
| ~0.32 s at 50,000 rows | 149 ms | 47% |

The honest reading: **the tail does not eat the whole win, but it eats between a
tenth and a half of it depending on shape, and it only becomes visible after the
plan succeeds.** The fix for it is fewer launches, which is level-wise batching
or kernel fusion, not device residency. Note the interaction with O9: if real
data has a longer small-node tail than synthetic data, which the covertype
anomaly hints at, this cost is larger than the synthetic profile shows.

### 4.3 Command queue depth: the plan cannot have 200 launches in flight, it can have 64

This was checked rather than assumed, by disassembling the shipped runtime, and
the answer is worse than the plan's model.

MAX creates its Metal command queue with a bare `[device newCommandQueue]` and
no arguments (`MetalDeviceContext.cpp:397`). The selectors that would raise the
limit have **zero** load sites anywhere in the 38.6 MB binary:
`newCommandQueueWithMaxCommandBufferCount:` zero,
`setMaxCommandBufferCount:` zero, and no `MTLCommandQueueDescriptor` is
constructed on that path. There is one queue per `DeviceContext`, created once,
and every `DeviceContext` in the GPU path shares it. **So Apple's default of 64
in-flight command buffers applies, and there is no MAX-level knob, environment
variable, or `DeviceContext` parameter that changes it.**

The dispatch path is confirmed from the same disassembly:
`enqueueFunctionExecDirect` is a closed sequence of `[queue commandBuffer]`,
`computeCommandEncoder`, `setComputePipelineState:`, the argument loop,
`dispatchThreadgroups:`, `endEncoding`, `commit`. **One command buffer, one
encoder, one dispatch, immediate commit, per `enqueue_function`, with no seam to
batch at.** This confirms from the inside what `docs/METAL_TIMELINE.md:174-176`
observed from the outside.

So the plan's picture of 221 launches per round flying with no host wait is not
what happens. What happens is 64 in flight, then the host blocks on the 65th
until the first completes, and thereafter runs in lockstep at one in and one
out. That is a **throttled pipeline, not a queue overrun**, and it is not
catastrophic. But three things follow that the plan should own.

**The stall is invisible to every instrument in this repository.** Apple's
contract is that `MTLCommandQueue.commandBuffer` blocks the calling thread when
the queue is full; it does not return nil, so MAX's null check never fires. The
block happens inside `objc_msgSend`, which every profiler here counts as
"enqueue." It would show up as the host mysteriously taking longer to commit,
in the "completion signal to next commit" bucket of 164.8 microseconds median
that `docs/METAL_TIMELINE.md:618-620` already lists as uninvestigated host code.
That is inference from Apple's documented contract, not something readable from
the binary, and it is marked as such.

**There is no assist available.** No asynchronous copy, no Metal streams (the
runtime carries the literal string `"Metal stream not implemented"`), and no
`MetalDeviceGraphBuilder` although `CUDADeviceGraphBuilder` and
`HIPDeviceGraphBuilder` both ship. Whatever pipelining the plan wants, it has to
get from the 64-deep queue alone.

**The memory concern is a non-concern and I withdraw it.** I expected retained
references across 200 in-flight buffers to be a residency problem. It is not.
No device buffer is allocated per split (all 45 `enqueue_create_buffer` sites
are in constructors), MAX suballocates from chunks, and the whole 200,000 row
working set is about 20 MiB against an 11.84 GiB budget. Two hundred command
buffers retain references to the same ten long-lived buffers.

*The one residency thing that is real, and is different.* The runtime frees
device memory lazily behind queued flush handlers (`"Enqueued flush handler id:
"`, `"Completed free of N after queuing flush handler"`). Inference: those are
drained on completion or synchronization. A loop that never synchronizes never
drains them. That is harmless today because nothing is freed per split, but it
becomes a leak-shaped growth curve the moment a device-resident design allocates
or frees anything inside the launch window. Verify the drain trigger before
putting any allocation in that loop.

*The check.* The microbenchmark from O10: plot wall clock per command buffer
against N. Expect a knee at 64. If the curve is flat to 22,000, Apple's contract
does not apply the way I think it does and this objection is wrong.

### 4.4 Error handling and the loss of locality in time

Today a stop condition evaluated on the host from a downloaded value ends a tree
immediately and cleanly. Device-side, a flag word plus one download per tree
costs one readback and is fine. What is lost is locality: if an invariant is
violated at split 3 of 30, the device grows 27 more splits on a corrupt frontier
before the host is told. That is recoverable **only if the flag is checked before
the tree is used to update raw scores**, and the plan should commit to that
ordering explicitly, because checking once per fit instead of once per tree turns
one bad tree into one hundred.

What is not recoverable is the class of failure that sets no flag. Metal does not
bounds-check in release builds. A split index read from device memory that is out
of range today would very likely trip a host assertion when it is read back;
tomorrow it becomes an out-of-bounds device write into whatever is adjacent.
**The plan removes the last host-side sanity check on every value the device
produces and replaces it with nothing.**

The good news is that most of the encoding is already designed.
`gpu_tree_tables.mojo:502-528` defines `CTR_STATUS` with five values,
`TREE_RUNNING`, `TREE_BUDGET_SPENT`, `TREE_NO_CANDIDATE`, `TREE_POOL_FULL`, and
`TREE_OVERFLOW`, alongside a six-word device counter block designed so that "a
single download brings the whole state of a step home" (`:475-479`). Three gaps
remain and the plan should close them explicitly. `TREE_NO_CANDIDATE` merges
four distinguishable host conditions into one code (the found flag, the
positive-gain floor, the depth limit, and the row floor), so a tree that stops
early can no longer say why. There is no code reserved for the row-count
verification of section 4.1. And `growth_policy.stop_reason`
(`growth_policy.mojo:410`) is computed today but never read by the trainer, so
there is no host path a device status could even flow into yet;
`gpu_frontier.mojo:88-89` says outright that the trainer's loop "cannot currently
distinguish" budget exhaustion from a dry frontier.

One condition genuinely cannot be encoded as designed: `fixed_point_scale`
**raises** on non-finite input and applies a magnitude floor
(`quantized_gradient.mojo:747-757`), and a kernel cannot raise. Under relaxation
three, where the scale is computed on device, that becomes a sixth status code
plus a readback, or it becomes silence.

### 4.5 Fixed grids and generic launches, where the plan is safer than it looks

The coordinator asked specifically what compile-time specialization is given up.
The honest answer, from the source, is: **almost nothing, and this concern should
be retired.**

The split search kernels have **zero** compile-time parameters.
`_scan_slot_kernel`, `_scan_slot_wide_kernel`, `_scan_slot_wide_primitive_kernel`,
`_reduce_slots_kernel`, `_reduce_slots_block_kernel`, and `_pick_best_record_kernel`
(`gpu_split_search.mojo:491`, `:937`, `:1339`, `:1671`, `:1786`, `:1941`) are
plain functions. Everything about a chosen split, and everything about a node
(`min_data_in_leaf`, the lambdas, monotone bounds, the allow mask, the feature
set, the histogram slot) is already a runtime kernel argument or a per-record
device table. **Reading the chosen split from device memory costs nothing in
specialization.**

The comptime parameters that do exist are `GROUP` and `BIN_CAP` on the histogram
family (`gpu_active_rows.mojo:987-989`, `:1289-1291`, twenty instantiations
across a five-rung group ladder and a four-rung bin ladder) and `block_size` on
the primitive partition arm. All three are **dataset or device properties, not
node or split properties**, so they stay host-known and stay specialized under a
device-resident tree. The bin count itself is already a runtime `Int32`; only the
capacity rung is comptime.

So the generic-launch tax is close to zero. **The grid tax is the real one**, and
O7 prices it: tens of milliseconds per 100 rounds for worst-case grids with early
exit, tolerable, unmeasured, and unmentioned.

### 4.6 Two growers, permanently

There are already seven live tree-growing paths (host CPU dense, host CPU sparse,
GPU with host split search, GPU device-search incremental, GPU device-search
resident, GPU sparse, and three distributed variants), plus the wired-off
`gpu_tree_tables.mojo`, plus two modifiers in the hybrid scheduler and the
batched multi-leaf histogram, selected by roughly twenty `MOJOTREES_*`
environment variables. The plan adds an eighth.

The device split search already **refuses** CEGB and feature penalties,
`min_gain_to_split`, `max_delta_step`, `path_smooth`, `extra_trees`, and
`feature_fraction_bylevel` (`train_gpu.mojo:816-825`,
`gpu_split_search.mojo:2240-2242`). The device-resident tree additionally refuses
monotone constraints, interaction constraints, per-node feature draws, and
depth-wise growth (`gpu_tree_tables.mojo:186-210`). Linear trees, DART, and EFB
have no GPU wiring at all.

So the benefit accrues to plain regression at default parameters. The
maintenance cost accrues to every combination. Two specific rules would have to
exist in two places forever and stay in step: the Float64 monotone midpoint
collapse, which the device cannot do because Apple GPUs have no Float64, and the
ancestor-chain interaction allow mask.

## 5. Is leaf-wise the right algorithm to be optimizing

**P(level-wise is the better GPU target) about 0.5. Damage of ignoring it:
medium, and it is opportunity cost rather than failure.**

Leaf-wise growth is a dependent chain of 30 splits by construction, and it makes
nodes small fast, which is what produces the tail in section 4.2.
`docs/design/GPU_LEVELWISE.md:69-75` prices the alternative: launch groups and
host synchronizations per tree fall from `num_leaves` to `1 + effective_depth`,
which at the default 31 leaves is 6 instead of 31 and at 255 leaves is 9 instead
of 255. Critically, "rows touched per group" goes from one node's to the whole
active buffer, **so every launch in a level-wise tree is root-sized and fills the
device**, which is precisely the property the tail lacks. It attacks the launch
count and the occupancy together, where the plan attacks neither.

The dependency chain itself, incidentally, is not the problem. The trace shows
the median back-to-back gap is 3.33 microseconds and that two thirds of all gaps
are under 5 microseconds carrying 1.5 percent of the idle time. A chain of 240
dependent kernels per round costs 0.8 milliseconds of gap. Serial dependency is
cheap on this device. What is expensive is that the chain's later links are too
small to fill it.

What level-wise costs is that it is a different model.
`GPU_LEVELWISE.md` section 8 is explicit: at equal `num_leaves` leaf-wise should
fit training data better per tree, generalization is not settled in either
direction, and fits are not comparable seed for seed. Section 10 specifies that
the only valid comparison is time to matched quality after tuning each mode
separately. And the accuracy baseline for this project is LightGBM, which is
leaf-wise, so a level-wise default is a deliberate divergence from the thing
accuracy is measured against.

Oblivious trees, the CatBoost shape, take this further: one split shared by every
node at a depth, so a depth-6 tree is 6 splits and 6 full-sized launch groups. It
is the most GPU-friendly tree there is and it is why CatBoost's GPU path is fast.
It is also a different library, and proposing it here is proposing to stop being
LightGBM-compatible.

The recommendation is not to adopt level-wise. It is to **run it once**, since it
already ships, before committing weeks to a mechanism that overlaps with it.

## 6. Should the effort go to the CPU instead

**P(the CPU is the better bet outright) about 0.3. P(it is the better bet per
unit of engineering risk) about 0.55.**

*The case for.* Every hour spent on a Metal control plane is portable to nothing.
One-portable-source has been explicitly relaxed, which means this work is now, by
construction, a single-vendor investment in a laptop GPU. CPU work ships to every
user on every platform, runs in CI on x86-64 and ARM64, and is what the
overwhelming majority of installs execute.

The CPU gap decomposes cleanly at one million rows:

```
                     1 thread      10 threads    parallel speedup
mojotrees CPU        15.959 s       6.982 s          2.29x
LightGBM              8.823 s       2.858 s          3.08x
```

So the 2.44x deficit at 10 threads is **1.81x of serial inner loop and 1.35x of
parallel efficiency**. Closing the parallel term alone takes 6.98 to about 5.2
seconds.

*The case against, which I find stronger.* The 1.81x serial term is LightGBM's
histogram inner loop, which is a decade of tuning over uint8 bins, small
histogram cells, feature bundling, and multi-value dense bin layouts. Matching it
is not a project, it is a program. And the M4 has four performance cores plus six
efficiency cores, so 3.08x from ten threads is already near the practical ceiling
and 2.29x is not as far off as it looks. Meanwhile the GPU is 1.25x behind at one
million rows and the CPU is 2.44x behind, so on raw distance to the target the
GPU is the better bet.

*The observation that actually matters.* **The mojotrees CPU has never been
profiled at one million rows.** `bench/bench_profile.mojo` exists, covers every
stage from `bin_fit` through `partition` to `grow_tree`, and its output appears
nowhere on disk; its documented default shape is 100,000 by 100, not 1,000,000 by
50. The only decomposition anyone has of the 6.98 seconds is the two aggregate
thread-count points above.

So the honest statement is not "the CPU is a worse bet." It is: **the project is
about to commit weeks of engineering to the path that has a Metal System Trace,
and decline the path that has never been profiled at all, and those two facts are
not independent.** The GPU won the argument partly by being the thing that got
instrumented. One run of `bench_profile` at 1,000,000 by 50 would make that a
real decision rather than an artifact of what was measurable.

## 7. If the plan is right anyway

The plan's **direction** is right, and nothing above disputes it. The control
plane is the largest single structural defect in the GPU path, 32.1 blocking
readbacks per round is indefensible, the mechanism is correctly identified, and
fixing it is worth about 1.33 seconds at every shape.

The single part most likely to be wrong anyway, if I must pick one and be held to
it:

> **That the compute exposed by removing the waits will be as fast as the compute
> measured while it was hidden behind them.**

Every estimate in this document, my own 2.1 to 2.4 second floor included, assumes
2.24 microseconds per row is a property of the kernels that survives the
transition unchanged. Four named mechanisms in this document say it might not:
worst-case grids taxing every dispatch (O7), the small-leaf tail becoming
host-submission bound (O10 and 4.2), the 64-deep command queue throttling the
launch stream (4.3), and the staging uploads that remain synchronous whatever
happens to the downloads (O6, **struck 2026-08-16: measured at 0.016 seconds
for thirteen copies per tree, a null; it does not bite and it never could,
because a drain of an empty queue costs nothing**). Each is individually
plausible and each moves the
floor the wrong way. If two of the four bite, the device-resident tree lands at
2.6 to 3.0 seconds at one million rows, which is **no better than today**, and
the project will have spent weeks moving a bottleneck rather than removing it.

The cheapest thing that would reveal it, before any of the large work:

> **Take one Metal capture at 1,000,000 rows with `grow_policy=depthwise` on the
> shipped resident device-search path, and compare its GPU compute time per round
> against the same capture at the default policy.**

That path already batches a level's search into one wait and already reads its
frontier from device memory. It is a working, shipped, small-scale instance of
exactly the transformation the plan proposes. If its compute time per round is
unchanged, the grid and specialization concerns are unfounded and the floor is
2.1 seconds. If its compute time per round has gone up, the floor is higher than
2.4 and the estimate needs rebuilding before the engineering, not after. The same
capture simultaneously answers O3, which is the objection with the largest
consequence for scope.

## 8. The one measurement to take first

If only one thing is done before engineering time is committed:

> **One interleaved sweep, single process, single time window, five repeats, GPU
> performance state recorded before and after, arms {ours CPU, ours GPU, ours GPU
> with `grow_policy=depthwise`, LightGBM} at {250k, 1M, 2M} rows by 50 features.**

One run, and it settles simultaneously:

- the per-row slope of both libraries over a band where the fixed cost no longer
  dominates, which is O1 and is the decisive question;
- whether LightGBM's 2.858 is a real number or a single sample of a drifting
  window, which is O5;
- how much of the wait reduction the already-shipped depthwise resident path
  delivers for free, which is O3 and is the objection with the largest effect on
  scope;
- whether the clock state moved during the sweep, which is the precondition for
  any of the above meaning anything.

Everything else in this document is arithmetic over numbers we already have. That
sweep is the only new number that changes what should be built.

## 9. Summary

| # | Objection | P | Damage | The check |
|---|---|---|---|---|
| O1 | 0.8-1.0s is unreachable; the floor is 2.1-2.4s because the slope is all compute | 0.90 | total | slope fit at 1M/2M/5M |
| O2 | Packed int16 breaks at 32,766 rows, 30x below target; carry corrupts the neighbor field; underflow collapses the plane to zero | 0.95 | very high | write the bound inequality before any code |
| O3 | `grow_policy=depthwise` on the shipped resident loop already removes ~78% of the waits. Benchmarked 2026-08-15 and upheld: 2.587s against leaf-wise's 3.756s at 1,000,000 x 50, within 2 percent of the prediction, and ahead of LightGBM. Quality still unmeasured | 0.70 | very high | done, see O3 |
| O4 | The roofline and slope figures do not exist and the real slopes point the other way | 0.95 | high | ask for the derivation |
| O5 | Every LightGBM number is one repeat, cross-process, no clock record; the 50k win is inside noise | 0.95 | med-high | interleaved sweep with LightGBM as an arm |
| O6 | **STRUCK 2026-08-16.** `enqueue_copy` is a synchronous drain in both directions, so the staging uploads block too and the plan addresses only the downloads. The mechanism holds; the premise that an upload drain is a wait was measured and is a null (0.016s for thirteen copies per tree against 0.64 predicted). The plan was aimed at the right half. See O6 | ~~0.85~~ 0 | none | none; the direction split would not change the answer |
| O7 | No indirect dispatch, so the counts that size two grids ride in the record the plan wants to stop downloading | 0.85 | high | confirm with Modular; price worst-case grids |
| O8 | The relaxed promise is misidentified; row blocks break within-backend determinism unless the block count ignores worker count | 0.95 | medium | state the block decomposition rule |
| O9 | Real covertype says the GPU loses multiclass to our own CPU, opposite of the synthetic result | 1.0 (fact) | high | Metal capture of the covertype run |
| O10 | 322 ms of host command buffer submission is not removed and binds below 200k rows | 0.75 | med-high | back-to-back submission microbenchmark |
| 4.1 | The row-count cross-check and the near-tie margin both die; relaxation two raises the tie rate while removing its instrument | 0.90 | med-high | reserve a status code and specify the trace record now |
| 4.2 | The small-leaf tail is a 149 ms floor currently hidden behind the wait | 0.80 | medium | the same microbenchmark |
| 4.3 | The queue is 64 deep, never configured, with no async copy, no streams, and no Metal graph builder; the stall is invisible | 0.85 | medium | the same microbenchmark, look for a knee at 64 |
| 4.5 | Generic launches: **retracted**. The split search already has zero comptime parameters and the ones that exist are dataset properties | 0.10 | none | none needed |
| 4.3b | Retained-reference memory blowup: **retracted**. ~20 MiB against 11.84 GiB, nothing allocates per split | 0.05 | none | none needed |
| 6 | The CPU has never been profiled at 1M; the GPU won the argument by being instrumented | 0.55 | medium | run `bench_profile` at 1M x 50 |

## 10. What would change my mind

Stated in advance so this document can be falsified rather than argued with.

- A measured GPU per-row slope below 1.2 microseconds at one to five million rows
  destroys O1 and makes the plan's estimate reachable. I do not expect it,
  because two instruments already agree on 2.04 and 2.24.
- A depthwise resident run that is no faster than the default destroys O3 and
  makes the full device-resident tree the only route to the fixed cost. That
  would be a genuinely useful negative result and it costs one run.
- A per-node width ladder that keeps 32 bits at the root and any dense bin, with
  the two planes in separate words, answers O2 completely. Fixed packed int16
  does not.
- A back-to-back submission microbenchmark flat to 22,000 command buffers removes
  4.3 and softens O10 and 4.2.
- ~~A direction split of the 3,206 blocking blits showing that nearly all of them
  are downloads removes O6 and makes the plan's mechanism complete as stated.~~
  **Superseded 2026-08-16.** The direction split was never taken and is no
  longer worth taking. A stronger check was run instead: the uploads were
  actually removed and the fit was timed. Thirteen copies per tree, nine of
  them uploads, **measured** 0.016 seconds at 1,000,000 x 50, a null under M0.
  O6 is removed by the A/B rather than by the count, and the plan's mechanism
  is complete as stated.
- An interleaved LightGBM measurement at five repeats reproducing 2.858 within a
  few percent removes most of O5, and I would then argue only about the target
  rather than the baseline.

Two objections I raised and then withdrew on evidence, recorded so the record
shows what survived scrutiny and what did not. The loss of compile-time
specialization is not a real cost, because the split search kernels already have
no compile-time parameters and the histogram's `GROUP` and `BIN_CAP` are dataset
properties that stay host-known. And retained references across a deep command
queue are not a memory problem, because nothing is allocated per split and the
whole working set is about 20 MiB.

None of these takes more than a day. All of them together take less time than the
first week of the plan.
