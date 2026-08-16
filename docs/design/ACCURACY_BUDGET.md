# The accuracy budget: what bit-identity was hiding, and what each relaxation costs

Status (Aug 15 2026): a design document. Nothing described here is
implemented, and no production code was written or run to produce it. Every
number below comes from standalone numerical experiments in NumPy, written
for this document, described in section 3, and reproducible from the scripts
named there. Nothing here is a measurement of mojotrees. The distinction
matters throughout and is marked at every claim.

Toolchain and hardware are irrelevant to almost everything here, because
almost everything here is IEEE-754 arithmetic that any conforming
implementation performs identically. The two places where they are not
irrelevant are named in section 12.

## 1. What changed, and what the baseline now is

Until today this project held bit-identity as a hard constraint. Any change
had to reproduce the previous implementation's output bit for bit, enforced
by `tests/test_golden_bits.mojo` over six whole training runs. That
constraint has been lifted. The mandate is speed and accuracy.

Three consequences follow immediately, and getting them straight is most of
the work of reading this document.

**The baseline is LightGBM, not our own past output.** "This changes the
bits we shipped yesterday" is no longer a finding. It is not even evidence.
The question every candidate has to answer is whether held-out metric parity
with LightGBM survives, measured against the tolerances the project already
wrote down in `bench/real_data/thresholds.json`: three percent relative on
RMSE for dense regression, two points absolute on average precision for
imbalanced binary, five percent relative on multiclass logloss, one point
absolute on AUC for the sparse and categorical scenarios. Those tolerances
were chosen before any run and each carries its reasoning in the file. They
are the budget. This document spends against them.

**Determinism is not the same thing as identity, and we are keeping it.**
The rule to work within is *deterministic on a given toolchain, not
identical to the past*. Run-to-run reproducibility, and reproducibility
across worker counts and launch geometries, is nearly free wherever the
accumulation is integer or the fold order is fixed, and it is one of the few
properties this project has that LightGBM's CUDA path does not. Every
candidate in sections 4 through 9 is deterministic in that sense. The only
fast technique on the table that would cost determinism is floating-point
atomics, and Metal does not have them, so the question does not arise on the
backend that matters. Determinism costs us nothing here and should not be
traded away in exchange for anything.

**The existing quality evidence is thinner than the brief for this document
assumed, and that has to be said plainly.** The brief described current
held-out parity with LightGBM as "a few parts in ten thousand across five
real datasets, see `bench/real_data/results/`". That directory contains one
file, a README whose first line is "Empty, and not by accident. This harness
has never been run." Two runs are referenced elsewhere in the tree
(`20260815T014842Z` and `20260815T023123Z`) and neither left a record file
behind; the second is the run whose covertype rows turned out to be CPU
timings wearing a GPU label. The only head-to-head quality numbers that
survive anywhere are on synthetic data:

> **Correction, 2026-08-15, later the same day.** The paragraph above was
> true when written and is not true now. `bench/real_data/results/` now
> commits a `summary.json` from each of those two runs and a README that
> states the parity spread as a range: across the five scenarios that ran on
> pinned real data, the relative difference on each scenario's primary
> metric runs from 5.7e-06 (RCV1, auc) to 3.2e-03 (Bank Marketing, average
> precision), a factor of about 560. So held-out evidence exists, and the
> brief's "a few parts in ten thousand" is still the wrong form of the claim,
> for a different reason than this section gives: it is accurate for three of
> the five, understates RCV1 by two orders of magnitude, and overstates Bank
> Marketing by one. Quote the range with the metric attached. The rest of
> this section stands: 3.2e-03 is not precise enough to detect a half-percent
> regression, so no verdict below is discharged by pointing at it. All of that
> evidence is leaf-wise; nothing in it covers `grow_policy=depthwise`, whose
> accuracy no run has measured.

| source | comparison | delta |
| --- | --- | --- |
| `bench/README.md` | binary train logloss, 100k x 100 | 5.0 parts in ten thousand, mojotrees better |
| `bench/README.md` | regression train MSE, 100k x 100 | 4.8 percent, mojotrees better |
| `bench/results/apple_m4_large_scaling_2026-08-14.md` | training loss, 1M and 5M x 50, three seeds each | between 1.9 percent better and 4.3 percent worse, sign varying by seed |

Those are training losses, not held-out metrics. The project's own written
expectation, in the `dense_regression` rationale in `thresholds.json`, is
that a genuine engine difference "moves RMSE in the third decimal place",
which is percent-level and not ten-thousandths.

This is not a quibble about a phrase. It is the single most important
constraint on how this document can be used. **We do not currently have a
held-out parity measurement precise enough to detect a half-percent
regression, so no verdict below can be discharged by pointing at an existing
result.** Every "take it" in this document is conditional on the real-data
harness being run first, with a before arm and an after arm, at least three
repeats, on the same machine. Section 12 says what that costs. A relaxation
adopted without it is not a priced relaxation, it is an unpriced one with a
document attached.

## 2. The two channels, which is the framing everything else hangs on

Gradient boosting is self-correcting. An error in one tree's leaf values is
partly absorbed by the next tree, which fits the residual that the error
left behind. That is the reason LightGBM can ship four-bin quantized
gradients at all, and it is why a per-tree error bound that looks alarming
can be irrelevant after a hundred rounds.

The self-correction is not uniform. It applies to leaf **values** and not to
split **selection**. A leaf value that is one percent too small leaves one
percent of the residual on the table, and the next tree sees that residual
and takes another bite out of it, so the error decays geometrically in the
number of subsequent rounds. A split that goes to the wrong feature produces
a tree that partitions the data differently, and no later tree is given the
information that the partition was wrong. The wrong partition is baked into
the ensemble.

So the first question about every candidate is: which channel does it touch?
Anything that perturbs the histogram touches both, because the histogram is
read by the split search and then by the leaf value computation. Anything
that perturbs only the gain arithmetic touches only selection. Anything that
perturbs only the leaf value computation touches only values.

The framing is not just rhetoric, and experiment F (section 3) separates the
two channels empirically by quantizing the gradients that the split search
sees while leaving the leaf value computation exact, and then the reverse.
Six seeds, 20,000 training rows, 8,000 held out, 100 rounds, gradients
quantized to LightGBM's four-bin lattice with stochastic rounding:

| injection | held-out metric vs exact | per-seed signs |
| --- | --- | --- |
| logloss, split selection only | 0.85 percent | 5 of 6 the same way |
| logloss, leaf values only | 0.01 percent | 3 and 3 |
| RMSE, split selection only | 6.72 percent | 6 of 6 the same way |
| RMSE, leaf values only | 0.52 percent | 4 and 2 |

Read the sign column before the magnitude column. **The split-channel
injection produces an effect that points the same way in almost every seed.
The leaf-channel injection produces one that changes sign from seed to
seed, which is what noise looks like.** At sixteen bins the same pattern
holds with a smaller split-channel effect. Leaf value error is absorbed
almost completely. Split selection error is not absorbed at all. Every
verdict in this document is a judgment about how much of a candidate's error
lands in the selection channel.

One thing that table does *not* say, and must not be read as saying: the
sign of the split-channel effect here is *negative*, meaning the quantized
arms score better than the exact one. That is real and it is not an
endorsement. At 20,000 rows, 31 leaves, and 100 rounds with no early
stopping, the reference model is overfitting, and unbiased gradient dither
acts as regularization. The configuration decides the sign. Only the
magnitude and the seed-consistency transfer.

One qualification, stated here so it is not mistaken for a loophole. The
absorption of leaf value error is not free in the way "self-correcting"
makes it sound. It is paid for in rounds: the ensemble reaches the same
place, slightly later. At a fixed round budget with early stopping, a leaf
value perturbation shows up as a small metric loss, not as nothing. The
0.007 percent figure above is at a fixed hundred rounds with no early
stopping, which is the generous reading. It is still two orders of magnitude
under the selection channel.

## 3. Method

Five experiments, all standalone NumPy, none of them touching the package.
They live under `.exp/` in this worktree and are not committed, because a
committed benchmark script implies a committed result and there is no
committed result here. What follows describes each well enough to rebuild.

**Experiment A, the reassociation gap.** Sums a list of Float64 values in
the shipped ascending order and in a blocked order with a fixed fold, and
compares both against `math.fsum`, which is correctly rounded and therefore
exact. Reports the gap between the two orders relative to the sum of
absolute values in the cell, which is the right normalization because a
histogram cell's error matters against the gradient scale that the split
search compares it to, not against the cell's own value. Run over
well-conditioned, mean-zero, adversarially cancelling, and heavy-tailed
inputs at 256, 4,096, and 65,536 values per cell, plus a distributional
sweep over 2,000 independent cells per configuration.

**Experiment B, the flip curve.** The master curve that prices every other
candidate. Builds a synthetic node histogram (50 features, 255 bins, three
of the features carrying signal and the rest carrying only noise), computes
the exact argmax over all 12,700 prefix-split candidates, then multiplies
every bin's gradient sum by `(1 + delta * u)` with `u` uniform in `[-1, 1]`
and recomputes the argmax. Reports two numbers per `delta`: the fraction of
nodes whose winner moved, and, on the nodes where it moved, the exact gain
of the new winner divided by the exact gain of the old one. **The second
number is the one that matters and it is the one that gets left out of
analyses like this.** A flip to a candidate holding 99.99 percent of the
winner's gain is not a wrong split in any sense a metric can see. Swept at
200,000, 20,000, and 2,000 rows and at three signal strengths.

**Experiment C, the gain formula, withdrawn in part.** Verifies an algebraic
identity in exact rational arithmetic, which stands. It then compared three
Float32 evaluations of the split gain against a Float64 reference, and
**those numbers are withdrawn** because the generator was wrong: it drew
each feature's bin sums independently, so the fifty features of one node did
not share a node total. That is unphysical, since every feature of a node
partitions the same rows, and it made the shipped form look bad for a reason
that does not exist (a single Float32 parent constant subtracted from gains
whose exact parents genuinely differed). The error was caught by experiment
H, which disagreed with it, and the correct version is experiment I. This
paragraph stays in the document rather than being deleted because a
withdrawn measurement that leaves no trace is how a suite stops meaning
anything.

**Experiment I, the gain formula done from rows.** Generates row gradients
and hessians, assigns each row a bin per feature (three features correlated
with the gradient, the rest random), and histograms, so every feature of a
node shares `G` and `H` exactly. Quantizes to Int32 at the shipped `2^30 /
sum|g|` scale. Then runs four Float32 arms, the factorial of {shipped
subtractive form, cancellation-free cross form} by {right-hand sums by
Float32 subtraction, right-hand sums by Int32 subtraction}. Reports the
argmax error, the relative error of the computed gain over each node's top
200 candidates, and the fraction of ordered pairs among each node's top 50
candidates that the arm ranks the wrong way round. The last of these is the
sensitive measure: the argmax only moves when an inversion happens to reach
the very top of the ranking, so a form can be substantially wrong about the
ordering while still picking the same winner most of the time.

**Experiment J, the parameter that controls the gain's precision.**
Experiment I came back with no winner errors anywhere, which was as
suspicious as C's 24.5 percent had been. The reason is that I drew mean-zero
gradients, so a node's `G` was near zero, `parent_score` was near zero, and
there was no cancellation to find. J fixes that by sweeping the node's mean
gradient, which is what moves `parent_score / gain`, and reporting the
realized ratio next to each form's error. It is the experiment that should
have been written first.

**How to read the three of them together.** C manufactured a cancellation
that does not exist. I removed the cancellation entirely by accident. J
sweeps it. Only J's numbers are quoted for candidates 5 and 6; I's are quoted
only for the claim that the shipped form is not currently mispicking winners
at any row count from 10,000 to 1,000,000.

**Experiment D, the accumulator schemes.** Builds one node's histogram ten
different ways from the same rows and bin assignments (Float64 sequential,
Float64 blocked, Float32 sequential, Float32 blocked, Int32 at the current
scale, Int32 at a power-of-two scale, and four int16 variants), and scores
each against an `fsum` reference, normalized by the node's total gradient
magnitude. Plus the closed-form capacity arithmetic for int16, which is not
a simulation.

**Experiment E and F, the ensemble.** A small but complete GBDT in NumPy:
255 quantile bins fitted on train and applied to test, leaf-wise growth to
31 leaves with the gain-descending then node-id-ascending frontier order
this project uses, `min_data_in_leaf` of 20, `lambda_l2` of 1.0, learning
rate 0.1, 100 rounds, squared error and logistic objectives, 20,000 train
and 8,000 held-out rows, 12 features with genuine nonlinear and interaction
structure. Each candidate is injected as a transform on the per-round
gradients, with a switch for whether the split search sees the transform,
the leaf value computation sees it, or both. Experiment F adds two things
experiment E lacked: six seeds instead of three, and a control arm that is
numerically equivalent to the reference but scans the features in a permuted
order, so that near-ties resolve differently. **The control arm is the noise
floor.** It changes no gain and carries no information, so whatever spread
it produces is the level below which nothing else in the experiment can be
distinguished from tie-breaking.

**Experiment G, sibling subtraction.** Accumulates a parent cell and a small
child's cell in each summation order, derives the large child by
subtraction, and compares against the large child accumulated directly and
against an `fsum` reference.

**Experiment H, the device split search as written.** A faithful model of
`gpu_split_search`'s serial scan: Int32 fixed-point cells at the shipped
`2^30 / sum|g|` scale, Int32 node totals, `total_g = Float32(tg) * g_inv`,
right-hand sums by Float32 subtraction from the total, gain as a difference
of three Float32 quotients. Then the factorial of two changes against it.

**What none of these are.** None of them is a measurement of mojotrees. None
of them uses real data. None of them is a substitute for running
`bench/real_data`. They establish orders of magnitude and mechanisms, and
they are good for that, and they are not good for anything finer.

## 4. Candidate 1: row-block private histograms with a fixed-order fold (CPU)

### What is proposed

Today the CPU parallelizes histogram accumulation over feature *groups*.
`apple_cpu_policy._plan_group` picks an interleave width from three clamps,
narrowest wins, and `parallel._run_feature_ranges` splits the resulting
group count across tasks. The parallelism available is therefore
`n_features / group_width`, and the group width is itself bounded by an L1
budget: at 255 bins a feature's histogram slice is `255 * 24 = 6120` bytes,
the assumed budget is `65536 / 2 = 32768`, five slices fit, and the ladder
floors that to four. Fifty features at width four is twelve groups and
therefore at most twelve tasks, on a machine that reports ten dispatch cores
and would happily take forty.

The proposal is to cut a group's rows into blocks, give each block a private
histogram, and fold the partials in a fixed block order. Both the task count
and the interleave width can then rise together, which is the trade the
current design cannot make.

### One correction to the brief

The brief motivates this with the measurement that thirty of fifty features
were left unpaired. That number is real and it is quoted in three places
(`bench/apple/cpu_plan.json` line 56, `apple_cpu_policy.mojo` lines 39 to
46, `parallel.mojo` lines 57 to 62), but it describes the state of the code
*before* the feature-group ladder, when `plan_tasks` fanned fifty features
over forty tasks and thirty of those tasks held one feature each, giving a
nominal width of two an effective width of 1.25. Making the group rather
than the feature the dispatch unit already fixed that. The cap that remains
is the one described above, twelve groups from fifty features, and it is a
smaller problem than the one the brief describes. That does not make row
blocks a bad idea. It does mean the speed case for them has to be made
against the shipped ladder and not against the code the ladder replaced.

### The error introduced

Cell `(f, b)` becomes

    (sum of block 0's rows in b) + (sum of block 1's rows in b) + ...

instead of one ascending pass. Float64 addition is not associative, so this
is a different value.

The textbook bound says which direction. Sequential summation of `n` values
has error at most `(n - 1) * eps * sum|x|` with `eps = 2^-53`. Blocked
summation into `B` blocks has error at most `(n/B - 1 + B - 1) * eps *
sum|x|`, because each block accumulates `n/B` values and the fold accumulates
`B` of them. At `n = 1,000,000` and `B = 12` that is `1.11e-10` against
`9.25e-12`, a factor of twelve in favor of blocking. At `B = 64` it is a
factor of 64. **Blocking is not a loss of accuracy. It is a gain, and the
gain grows with the block count, up to `B = sqrt(n)` where the two terms
balance.**

The worst-case bounds are loose, so experiment A measures the gap directly.
Over 2,000 independent cells at 4,096 values per cell and 12 blocks, the gap
between the ascending answer and the blocked answer, relative to the cell's
`sum|x|`:

| cell size | blocks | median gap | p99 | max | worst-case bound |
| --- | --- | --- | --- | --- | --- |
| 256 | 8 | 2.2e-17 | 1.4e-16 | 2.1e-16 | 2.8e-14 |
| 4,096 | 8 | 2.2e-17 | 1.4e-16 | 2.9e-16 | 4.5e-13 |
| 4,096 | 12 | 2.4e-17 | 1.4e-16 | 3.0e-16 | 4.5e-13 |
| 65,536 | 12 | 2.2e-17 | 1.5e-16 | 3.2e-16 | 7.3e-12 |

Two or three ulps, flat in the cell size, three to four orders of magnitude
inside the worst-case bound. Experiment D, which builds a whole 255-bin
histogram rather than one cell, agrees: at a 1,000,000-row root the maximum
per-cell error is `7.1e-19` of `sum|g|` for the ascending pass and `1.4e-19`
for the blocked one, and the blocked answer is closer to exact at every node
size except the smallest, where the blocks hold three values each and
blocking has nothing left to do.

The adversarial case is worth stating because it is the only one where the
numbers move. When a cell's rows cancel almost exactly, so that `|sum|` is
`1e-8` of `sum|x|`, the relative error against `|sum|` rises to around
`1e-9` for both orders, and blocking is sometimes better and sometimes worse
because at that conditioning the ordering of two errors of the same size is
luck. Relative to `sum|x|`, which is the normalization the split search
cares about, nothing changes: both stay at a few times `1e-18`.

### Which channel, and does it compound

Both channels, because the histogram feeds both. Feeding `3e-16` into
experiment B's flip curve gives a flip rate indistinguishable from zero: the
curve does not leave zero until `delta` reaches `1e-3`, thirteen orders of
magnitude higher, and even there the flipped candidate holds 99.99 percent
of the winner's gain. There is no compounding to discuss, because there is
no first-round effect to compound.

### The claim in the docstring that does not survive

`histogram.mojo` lines 108 to 110 give three reasons to refuse row blocks:
the block count would enter the result, `MOJOTREES_NUM_WORKERS` would stop
being a scheduling knob, and sibling subtraction would stop being exact. The
first two are true and are exactly the bit-identity constraint that has been
lifted, subject to one condition stated below. The third is false, and it
was false before this change.

CPU sibling subtraction is `parent[c] - small[c]` in Float64, where the
parent was accumulated over a row order that interleaves both children. It
has never been exact. Experiment G measures it. Over 300 trials per setting,
maximum deviation relative to the cell's `sum|x|` between the derived child
and the same child accumulated directly.

| summation | cell size | small child | derived vs rebuilt | derived vs exact |
| --- | --- | --- | --- | --- |
| ascending (shipped) | 4,096 | 50 percent | 1.89e-16 | 2.01e-16 |
| ascending (shipped) | 65,536 | 10 percent | 1.45e-16 | 2.26e-16 |
| blocked, 12 (proposed) | 4,096 | 50 percent | 4.52e-17 | 3.92e-17 |
| blocked, 12 (proposed) | 65,536 | 10 percent | 4.07e-17 | 5.12e-17 |

Nonzero everywhere for the shipped path, and **four times smaller under
blocking**. The blocked fold makes sibling subtraction more consistent, not
less. Exactness of sibling subtraction is a property of the integer paths
(`gpu_active_rows`, `gpu_leaf_batching`, `quantized_gradient.subtract_quantized`),
which is what those modules actually claim, and `docs/NUMERICS.md` section
5.5 says so correctly. The CPU docstring overstates it and should be
corrected in the same commit that takes this candidate, whether or not
anyone believes the correction changes the decision.

### The condition that has to hold

The fold order must be a function of the block count and nothing else, and
the block count must be a function of the node's row count and the resolved
worker policy, not of scheduling. If two runs at the same settings can
disagree on how many blocks a node was cut into, they disagree on the
answer, and the determinism guarantee in `docs/NUMERICS.md` section 1 is
gone, along with the `require_identical_predictions` gate in
`thresholds.json`. That guarantee is worth more than the speed. The block
count must come from `DispatchSettings.resolve()`, snapshotted once per fit,
exactly as the task count does today, and the fold must walk block 0 to
block `B-1` in that order regardless of which block finished first.

Note what this does to `MOJOTREES_NUM_WORKERS`. If the block count is
derived from the resolved worker count, then changing the worker count
changes the answer, and the README's claim at line 287 that "every path is
bit-identical to the serial one at any worker count" stops being true for
the CPU histogram. That is a documented user-facing promise and it would
have to be rewritten, not quietly dropped. The alternative, which is
strictly better and costs nothing, is to derive the block count from the
node's row count alone, with a fixed target of rows per block, so that the
worker count still only decides how many blocks a task takes and never how
many there are. **Take the second option.** It keeps the worker-count
invariance, keeps the fold order fixed, and gives up only the ability to
tune blocking through the same knob that tunes threading, which nobody wants
to do anyway.

### Verdict

**Take it.** The reassociation is measured at two or three Float64 ulps, it
moves the answer *toward* exact rather than away from it, it improves
sibling subtraction consistency by a factor of four, and it lands thirteen
orders of magnitude below the perturbation at which experiment B's flip
curve leaves zero. It is deterministic provided the block count is derived
from the node's row count and the fold order is fixed. It is the cheapest
accuracy risk in this document by a wide margin, and it is the only
candidate here whose numerical error is *smaller* than what ships today.

The two obligations that come with it: derive the block count from row count
rather than worker count, and correct the sibling-subtraction claim in
`histogram.mojo`.

## 5. Candidate 2: packed int16 gradient and hessian in one 32-bit atomic (GPU)

### What is proposed

LightGBM's CUDA histogram constructor packs a quantized gradient and a
quantized hessian into one 32-bit word and accumulates with a single 32-bit
atomic, halving the shared-memory footprint and halving the atomic count.
The local sources confirm the shape: `__shared__ int16_t shared_hist[...]`,
and the unpack is
`(int64_t)((int16_t)(packed >> 16)) << 32 | (packed & 0xffff)`.

The project rejected this on two grounds. Under the global `2^30 / sum|g|`
scale an int16 accumulator overflows almost immediately, and Metal has no
64-bit threadgroup atomics, so the widen-on-overflow path that CUDA takes is
constrained.

### What the overflow arithmetic actually says

The brief states the int16 accumulator overflows "at any leaf holding more
than about 3 percent of gradient magnitude". The arithmetic gives a number a
thousand times smaller. The int16 signed limit is 32,767 and the scale bounds
any partial sum at `2^30 = 1,073,741,824`, so a bin overflows once it holds

    32767 / 2^30 = 3.05e-5

of the total gradient magnitude, which is three thousandths of one percent.
The correction makes the point far more strongly than the brief did.
Experiment D confirms it is not a theoretical bound: accumulating a
1,000,000-row node's 255-bin histogram at the shipped global scale overflows
int16 in **172 of 255 bins**, and at 100,000 rows in 228 of 255. There is no
node shape on which the current scale and an int16 accumulator coexist.

So the question is what scale makes int16 safe, and there are two families
of answer.

**The magnitude-sum family**, which is the rule the project already ships,
scaled down. Set `units = 2^k / sum_node|g|` so that any bin's exact scaled
sum is bounded by `2^k`, leaving `32767 - 2^k` of headroom for the rounding
residue of `n/2` under round-to-nearest.

| `k` | bin bound | residue headroom | maximum node rows |
| --- | --- | --- | --- |
| 14 | 16,384 | 16,383 | 32,766 |
| 13 | 8,192 | 24,575 | 49,150 |
| 12 | 4,096 | 28,671 | 57,342 |

**The per-row clamp family**, which is LightGBM's rule. With
`num_grad_quant_bins = B`, every row's quantized gradient is clamped into
`[-B/2, B/2]`, so a bin of `n` rows is bounded by `n * B/2`:

| `B` | per-row clamp | maximum node rows in int16 |
| --- | --- | --- |
| 4 | 2 | 16,383 |
| 8 | 4 | 8,191 |
| 16 | 8 | 4,095 |
| 32 | 16 | 2,047 |

Both families make int16 a **node-size-dependent representation**. There is
no fixed scale that is safe at a 1,000,000-row root and still useful at a
4,000-row leaf. That is not a defect of either rule, it is the arithmetic,
and it is exactly why LightGBM promotes bit width per leaf.
`cuda_single_gpu_tree_learner.cpp` line 198 calls
`GetHistBitsInLeaf(smaller_leaf_index_)` for every histogram it builds, and
the constructor branches on `num_bits_in_histogram_bins <= 16` in four
places. The mixed-width sibling subtraction at lines 925 to 944, with its
three separate width arguments for parent, smaller, and larger, is the cost
of that.

### The quantization error, and why the two families are not comparable

Experiment D scores each scheme's per-cell error against an exact reference,
normalized by the node's `sum|g|`:

| scheme | 1M rows | 100k rows | 10k rows | 1k rows |
| --- | --- | --- | --- | --- |
| Float64 ascending (CPU today) | 7.1e-19 | 1.2e-18 | 6.7e-19 | 1.1e-18 |
| Float32 ascending | 3.0e-10 | 3.7e-10 | 4.1e-10 | 5.6e-10 |
| Int32 at `2^30` (GPU today) | 4.9e-08 | 1.5e-08 | 4.9e-09 | 1.8e-09 |
| int16, magnitude-sum at `2^14` | 2.3e-04 | 7.1e-04 | 3.3e-04 | 1.1e-04 |
| int16, max-abs at `B = 16` | 4.0e-05 | 1.6e-04 | 4.6e-04 | 7.9e-04 |
| int16, max-abs at `B = 4` | 2.3e-04 | 5.5e-04 | 1.8e-03 | 4.4e-03 |

(maximum per-cell error; the RMS column runs about three times lower and
tells the same story.)

Three things to read off this table.

**int16 is four to six orders of magnitude coarser than the shipped Int32
path, under every rule.** Not a small step down. A different regime.

**The two rules trend in opposite directions with node size,** and that is
the important structural fact. The magnitude-sum rule's per-row step is
`sum_node|g| / 2^k`, which grows with the node's row count, so its per-row
noise relative to the gradient's own scale is proportional to `n`. The
max-abs rule's per-row step is `2 * max|g| / B`, which does not depend on
`n` at all. Write `r` for the ratio of the quantization noise standard
deviation to the gradient's own standard deviation. Under max-abs with
Gaussian gradients whose maximum is about `k` standard deviations,

    r = 2k / (B * sqrt(12)) = 0.58 * k / B

which for `k` around 4 or 5 is 0.15 at `B = 16` and 0.6 at `B = 4`, at every
node size. Under magnitude-sum at `2^14`, `r` is about `0.8 n / (2^14 *
sqrt(12))`, which is 0.14 at 10,000 rows and **14 at 1,000,000 rows**: the
quantization noise is fourteen times the gradient itself.

That gives the cleanest way to think about the max-abs rule's cost.
Quantization with unbiased rounding adds independent noise to each row's
gradient. A histogram cell is a sum, so both the true gradient and the added
noise average down at the same rate, and the net effect is a reduction in
effective sample size by a factor of `1 / (1 + r^2)`. At `B = 16` that is
`1 / 1.021`, or about two percent fewer rows. At `B = 4` it is `1 / 1.34`,
or about twenty-five percent fewer rows. **That is the honest way to price
LightGBM's default: it is not a small numerical error, it is a quarter of
your data, and it is why LightGBM's own paper measures an accuracy cost
rather than claiming there is none.**

The brief states LightGBM's `num_grad_quant_bins` default is 16. This
repository's own constant says otherwise: `quantized_gradient.mojo` line 342
sets `DEFAULT_NUM_GRAD_QUANT_BINS = 4` with the comment "LightGBM's
default". LightGBM's configuration source is not vendored here, so I could
not settle it. Both are priced below.

### What it does to the held-out metric

Experiment F, 100 rounds, held out, six seeds. A negative number means the
arm scored better than the exact Float64 reference, which as section 2
explains is a property of this overfitting configuration and not a
recommendation.

| arm | logloss vs exact | RMSE vs exact | per-seed signs consistent |
| --- | --- | --- | --- |
| control, permuted feature order | 0.00 percent | 0.00 percent | no effect at all |
| Float32 gradients | 0.00 percent | 0.00 percent | no |
| Float32 split gain | 0.00 percent | 0.00 percent | no |
| magnitude-sum `2^30` (GPU today) | +0.08 percent | 0.00 percent | no |
| magnitude-sum `2^30`, power of two | +0.03 percent | 0.00 percent | no |
| magnitude-sum `2^14` (int16, per node) | **+4.31 percent** | -0.61 percent | **yes, 6 of 6 on logloss** |
| max-abs `B = 16`, stochastic | -0.39 percent | -1.81 percent | 4 of 6, 6 of 6 |
| max-abs `B = 4`, stochastic | -1.34 percent | -8.88 percent | 6 of 6, 6 of 6 |

The control arm deserves a note, because it came out at exactly zero on
every seed. It scans the features in a permuted order, so it resolves exact
ties differently, and it produced no difference at all. That means there
were no exact ties, because with continuous gradients the tie-break channel
is empty. So the control does not establish a noise floor for a
rounding-magnitude perturbation, and the arms that do are the three
above it that are equivalent to a perturbation of `1e-9` or smaller. Those
give 0.00 to 0.08 percent in the mean with a worst single seed of 0.46
percent. **Read that as the harness's resolution. This experiment cannot
distinguish anything below roughly a tenth of a percent in the mean.**

Three readings.

The magnitude-sum rule at `2^14`, which is the "just narrow the existing
scheme to fit int16" option, is **plainly bad**, at 4.31 percent worse on
held-out logloss, the same sign on all six seeds, and above the 3 percent
relative tolerance `thresholds.json` sets for logloss. That is a rejection,
and it is exactly what the noise-ratio argument above predicts, because at
20,000 rows the ratio `r` is already near 1.

LightGBM's max-abs rule at `B = 16` and `B = 4` is inside the tolerances,
consistently. At the shapes the project actually runs at (hundreds of
thousands to millions of rows) the max-abs noise ratio is unchanged, so
these numbers should transfer, which is the one thing the magnitude-sum
numbers will not do.

Nothing at or above the current scheme's resolution (`2^30`, power of two,
Float32) is distinguishable from the reference.

### Which channel, and does it compound

Both channels, and the split channel dominates by two orders of magnitude,
which is the section 2 result. There is no compounding in the usual sense, because
the per-round quantization noise is redrawn every round and is
independent across rounds, so it does not accumulate in the leaf values. It
accumulates only through the split channel, where a wrong partition in round
three is still there in round one hundred.

There is one genuinely compounding effect and it is worth naming. Under
stochastic rounding the quantization is unbiased *per row*, but the split it
produces is a nonlinear function of the histogram, so the tree structure is
biased even though the gradients are not. Experiment F cannot separate that
from ordinary variance at six seeds. It is the reason the ensemble result
should be read as "no detectable degradation at this scale" and not as "no
degradation".

### The Metal constraint, and the tile bound that decides everything

Even granting a workable scale, the implementation has a problem the accuracy
analysis cannot fix on its own. LightGBM's packed accumulation works because
CUDA gives it a widening path. When a leaf is large it promotes the histogram
to 32 bits per plane, and the subtraction code carries three separate width
arguments so a parent at one width and children at another can still be
combined. On Metal there are no 64-bit threadgroup atomics, so the promotion
target for a packed 32-bit shared cell is not available in shared memory.

The Metal shape worth considering is therefore narrower than LightGBM's.
It would accumulate packed int16 pairs in **threadgroup** memory, and fold into the
**global Int32** planes we already have. That is very close to what
`_range_hist_atomic_kernel` does today, except the shared planes are half as
wide and there is one shared atomic per (row, feature) instead of two. The
saving is real. Threadgroup memory is what bounds resident blocks per core,
which is why the `BIN_CAP` ladder exists, and this halves the gradient and
hessian planes. And because the global planes stay Int32, this shape needs
**no per-leaf bit width and no mixed-width sibling subtraction**, which is
the whole of LightGBM's machinery, avoided.

The bound then applies to the threadgroup accumulator over the rows one block
walks, not to the whole node. And here is where an earlier draft of this
document was wrong, so the arithmetic is spelled out.

Under the shipped magnitude-sum scale a row's quantized magnitude is `|g_r| *
2^30 / sum|g|`, and there is no per-row clamp, so a tile's bin partial sum is
`2^30` times the share of total gradient magnitude that the tile's rows in
that bin carry. For that to stay under 32,767 the share must stay under
`3.05e-5`, and a tile of `T` rows out of `n` carries about `T/n`:

| dataset rows | maximum tile rows under the `2^30` scale |
| --- | --- |
| 100,000 | 3 |
| 1,000,000 | 30 |
| 10,000,000 | 305 |

**Three to three hundred rows per threadgroup tile is not a tile, so the
shipped scale does not fit an int16 shared accumulator at any useful launch
shape.** The earlier draft claimed otherwise and it was wrong; the error was
assuming a per-row bound that the magnitude-sum rule does not have. Nor can
it be patched by requantizing into the shared plane at a coarser scale: at
1,000,000 rows a typical row's quantized value is about `2^30 / n = 1073`, so
shifting right by the sixteen bits needed to fit sends almost every row to
zero.

The only per-row bound available is LightGBM's clamp, and with it the tile
bound is workable:

| `B` | per-row clamp | maximum tile rows in int16 |
| --- | --- | --- |
| 4 | 2 | 16,383 |
| 8 | 4 | 8,191 |
| 16 | 8 | 4,095 |
| 32 | 16 | 2,047 |

Four thousand rows per tile at `B = 16` is an ordinary launch shape.

So the candidate is genuinely a package and cannot be unbundled: **packed
int16 on Metal requires LightGBM's max-abs per-row rule, and therefore
requires paying its quantization cost.** That cost is the two percent of
effective sample size at `B = 16` derived above, and experiment F measures
its end-to-end effect at 0.39 percent on logloss and 1.81 percent on RMSE,
both inside the 3 percent tolerances and both real.

### Verdict

**Take it with a guard, and only as the whole package.**

The package is LightGBM's max-abs rule at `B = 16` (not the shipped
magnitude-sum rule, which does not fit), packed int16 pairs in threadgroup
memory, global planes and global sibling subtraction unchanged at Int32, and
a tile bound of `32767 / (B/2)` rows enforced at launch with a fallback to
the current two-plane Int32 shared layout when the resolved tile exceeds it.
The fallback is not optional. Without it this is a silent wraparound
producing a wrong histogram for one node on some tree shapes and not others,
which is the failure mode no fixture catches.

What it costs, stated plainly rather than buried: about two percent of
effective sample size, entering through the split channel, which section 2
shows is the channel that does not self-correct. Experiment F puts that at a
few tenths of a percent on held-out logloss at 20,000 rows, and the max-abs
noise ratio does not grow with row count, so it should transfer. That is
inside every tolerance in `thresholds.json` and it is not nothing, so this
one genuinely spends budget where candidates 1 and 3 do not. It should be the
last of the take-its to land and the first to be reverted if the real-data
harness moves.

**Not worth it in two other forms.** Narrowing the existing magnitude-sum
scale to `2^14` so it fits int16 is the obvious-looking move and it is the
worst option on the table: experiment F measures 4.31 percent worse held-out
logloss with the same sign on all six seeds, outside the project's own 3
percent tolerance, exactly as the noise-ratio argument predicts. And making
int16 the *global* histogram representation buys nothing here that the
threadgroup form does not, while requiring the per-leaf bit width promotion
and mixed-width subtraction that Metal's missing 64-bit shared atomics make
awkward.

**One free alternative that should be tried first.** For `SQUARED_ERROR`,
`L1`, `HUBER`, and `QUANTILE` without sample weights, the hessian plane
carries no information and `histogram.mojo` already documents the exact
elision, which the device kernel already supports through
`GpuActiveRows.set_constant_hessian`. Turning it on removes one of three
shared atomics per (row, feature) and one of three shared planes, for **zero
accuracy cost**, which is a better trade than int16 packing offers and does
not need this section's analysis at all. It is unclaimed because nothing
calls the predicate (section 10).

On 8-bit gradient schemes, which LightGBM also offers, the answer is no. At
`B = 4` the effective sample size loss is already twenty-five percent and
experiment F's split-only injection measures 0.85 percent on held-out logloss
and 6.72 percent on held-out RMSE, the latter with the same sign on all six
seeds and outside the 3 percent relative RMSE tolerance if the sign were the
other way. Eight-bit gradient values are a different risk class, this project
has no loss curve that would support them, and the analysis in this document
is not a substitute for one. Revisit only when someone produces the curve.

## 6. Candidate 3: a power-of-two fixed-point scale chosen on the device

### What is proposed

Today the scale is `Float32(2^30 / sum|g|)` with `sum|g|` reduced on the
host, which costs one host wait per boosting round. A power of two chosen
from the maximum exponent by a device reduction makes quantization a shift
rather than a multiply, makes dequantization exact rather than
approximately-inverse, and removes the wait.

### The error introduced

A power-of-two scale is at most a factor of two below the arbitrary scale it
replaces, so the quantization step is at most twice as large and the
rounding residue at most twice as large. Expressed as bits, the scheme
wastes up to one bit of the accumulator's dynamic range, uniformly
distributed in `log2` so the expected waste is about half a bit, or a factor
of about 1.44.

Experiment D measures the realized waste per node, since it depends on where
`sum|g|` falls between two powers of two:

| node | waste factor | max cell error, `2^30` scale | max cell error, power-of-two |
| --- | --- | --- | --- |
| 1M rows | 1.314 | 4.89e-08 | 6.87e-08 |
| 100k rows | 1.638 | 1.52e-08 | 2.89e-08 |
| 10k rows | 1.024 | 4.87e-09 | 4.75e-09 |
| 1k rows | 1.224 | 1.82e-09 | 2.96e-09 |

Worst observed penalty is 1.9x on the cell error. The bound is 2x. Both are
against a baseline that is already six orders of magnitude tighter than
anything that matters.

There is a second effect that runs the other way and is worth more than the
first. Under the current scheme the dequantization on read is
`Float64(q) * (1.0 / scale)`, and `1.0 / scale` is not exactly representable
for an arbitrary scale, so the round trip is not exact. Under a power-of-two
scale it is: both `scale` and `1/scale` are exactly representable, and the
multiply is an exponent adjustment with no mantissa rounding at all. **The
power-of-two scale makes dequantization exact, which the current scheme is
not.** That removes a rounding from the innermost read path of the split
search, on every cell, on every candidate.

### Which channel, and does it compound

Both, through the histogram, at a magnitude that experiment B's flip curve
cannot resolve. Experiment F's ensemble arm shows what a full training run
does with it.

There is one thing to be careful about that the accuracy analysis does not
cover. Removing the host wait changes *when* the scale is known, and if the
scale is derived from a device reduction over the same values in a different
order, the `sum|g|` it sees can differ from the host's in the last ulp.
`quantized_gradient.combine_stats` already documents this and prices it at a
relative `2^-52` on the scale, which moves a quantized value by at most one
unit at `2^30`. Under a power-of-two scale that becomes strictly better, not
worse: a last-ulp difference in `sum|g|` changes the chosen exponent only
when `sum|g|` sits within one ulp of a power of two, which happens with
probability about `2^-52` per round. When it does happen the scale moves by
a factor of two, which is a visible change, so the rule has to be stated as
a deterministic function of a deterministically-computed reduction, not as
"whatever the device reduced to this time". A device reduction over a fixed
tree shape is deterministic, so this is satisfiable, but it has to be
written down rather than assumed.

### Verdict

**Take it.** Up to one bit of mantissa is the stated cost and the measured
worst case is a factor of 1.9 on a cell error that is already `5e-8` of the
node's gradient magnitude. Against that it buys an exact dequantization,
which the current scheme does not have, a shift instead of a multiply in the
quantization, and the removal of a host synchronization per round. It is
deterministic. The guard is that the device reduction producing `sum|g|`
must itself be deterministic and the exponent must be a pure function of its
result, so that two runs cannot land on different powers of two.

This is the best speed-per-unit-of-risk item in the document among the
candidates the brief listed, because the risk is nearly zero and the saving
is a per-round synchronization, which is a fixed cost that does not shrink
as the data gets smaller and therefore matters most exactly where the
current GPU path is weakest.

## 7. Candidate 4: Float32 histogram accumulation instead of fixed-point Int32

### On the GPU, this is mostly moot, and the reason is worth stating precisely

Metal has no floating-point atomic add. `gpu_active_rows.mojo` says so at
line 45 and the whole fixed-point scheme exists because of it. So on Metal,
"Float32 accumulation" cannot mean atomics into a shared histogram. It can
only mean the tiled arm, where each block owns a private slice and no atomic
is involved (`_range_hist_partial_kernel`), followed by a fold.

If it did mean atomics, it would cost the determinism guarantee outright,
because float atomics commit in completion order and completion order is not
reproducible. That is the one trade in this document that should be refused
regardless of the accuracy number, and it is refused for us by the hardware.

For the tiled arm the question is real, and the answer is that Float32 is
*more* accurate than the shipped Int32 fixed point, not less. Experiment D,
maximum per-cell error relative to the node's `sum|g|`:

| scheme | 1M rows | 100k rows | 10k rows |
| --- | --- | --- | --- |
| Float32 ascending | 3.0e-10 | 3.7e-10 | 4.1e-10 |
| Float32 blocked, 12 | 8.0e-11 | 1.2e-10 | 2.8e-10 |
| Int32 at `2^30` (shipped) | 4.9e-08 | 1.5e-08 | 4.9e-09 |

Float32 is between one and two orders of magnitude tighter. That is not
surprising once stated: the fixed-point scheme spends its entire 31-bit
range on the *whole node's* gradient magnitude, so a cell holding a
thousandth of it gets ten bits of resolution and no more, while Float32
gives every cell 24 bits of its own magnitude. Fixed point buys
order-independence, and it pays for it in resolution.

But order-independence is the whole reason the fixed-point path exists.
Giving it up to gain resolution that nothing measures would be a bad
trade. It would break the bit-identity between the `tiled` and `atomic` GPU
strategies that `tests/test_gpu_strategies.mojo` asserts, it would break the
CPU replica path that `tests/test_host_replica.mojo` is built on (and that
`hybrid_leaf_scheduler`'s `MODE_MIRROR` was, until that module was deleted
on 2026-08-16),
and it would break the `require_identical_predictions` gate. All of that in
exchange for moving a `5e-8` error to a `3e-10` error, when the flip curve
says nothing happens until `1e-3`.

### On the CPU, there is a version of this worth taking, and it is not about accuracy

The CPU accumulates in Float64 and the histogram cell is 24 bytes
(`apple_cpu_policy.HISTOGRAM_BYTES_PER_CELL`). Float32 gradient, Float32
hessian, and Int32 count would be 12 bytes. Accuracy would get worse, from
`7e-19` to `3e-10` relative, which is still seven orders of magnitude below
where the flip curve moves.

The reason to care is not accuracy, it is the L1 clamp in
`apple_cpu_policy._cache_group`. At 255 bins:

| cell | slice bytes | budget | slices that fit | ladder width |
| --- | --- | --- | --- | --- |
| 24 bytes (today) | 6,120 | 32,768 | 5 | **4** |
| 12 bytes | 3,060 | 32,768 | 10 | **8** |

The interleave width doubles, and the docstring on `cache_feature_group`
says widening the interleave divides the gradient traffic by the same factor.
At the shape `bench/apple/cpu_plan.json` tabulates, 1,000,000 rows by 50
features, that is 208 MB of gradient traffic at width four against 104 MB at
width eight. And halving the histogram footprint also halves what the
`_HistPool` free list holds, which matters at 31 leaves.

The catch is that this candidate and candidate 1 compete for the same
resource. Widening the interleave from four to eight *halves* the group
count, from twelve groups to six, which halves the available task
parallelism and makes the cap that candidate 1 exists to remove worse. Taken
together they are complementary: row blocks restore the parallelism that the
wider interleave costs. Taken separately, Float32 cells alone would be a
regression in scheduling. **This candidate is conditional on candidate 1
landing first.**

### Which channel, and does it compound

Both channels, at `3e-10`, which is nothing. Experiment F's "Float32
gradients" arm confirms it at the ensemble level.

### Verdict

**Not worth it on the GPU.** Float32 accumulation there means either float
atomics, which Metal does not have and which would cost determinism if it
did, or a tiled fold that gains one to two orders of magnitude of resolution
that no metric can see, at the price of the cross-strategy bit-identity, the
CPU replica path, and the determinism gate. The current scheme was chosen
for order-independence and order-independence is still worth more than the
resolution.

**Take it with a guard on the CPU, after candidate 1.** Float32 cells halve
the histogram footprint and double the L1-clamped interleave width, which is
a direct traffic halving derived from the project's own policy formula. The
accuracy cost is `3e-10` relative, seven orders below the flip curve's knee.
The guard is that it must not ship before row blocks, because on its own it
halves the task parallelism it depends on.

One caveat that the accuracy analysis will not catch and that should be
checked before this is built: Float32 accumulation of a large node's cell is
a sequence of `n` roundings, and at `n` of a few million with same-sign
gradients the accumulator can start absorbing small addends entirely
(`fl(x + y) == x` once `y < eps * x / 2`). Blocking fixes this, which is
another reason the ordering matters, but the worst case is worth a bound
before anyone relies on it.

## 8. Candidate 5: the split gain, its cancellation, and the form that removes it

This candidate was not in the original brief. It is also the one where I got
the analysis wrong twice before getting it right, so the reasoning is set out
in full rather than just the conclusion.

### The observation, as offered

`gpu_split_search.gpu_split_gain` computes, per candidate, in Float32:

    left_g * left_g / (left_h + lambda_l2)
      + right_g * right_g / (right_h + lambda_l2)
      - parent_score

and `parent_score` is `gpu_leaf_score(total_g, total_h, ...)`, computed once
per node at line 599 and constant across every candidate the node scores.
The winner is chosen by `gain > best_gain` at lines 649, 743, and elsewhere.

The observation offered was that this subtraction is a catastrophic
cancellation at 1,000,000 rows, that the resolution is roughly `6e-8 *
parent_score` rather than a few ulps of the gain, and that since the parent
term is a per-node constant, the argmax does not need it and should drop it.

### What is right, what is wrong, and what the controlling parameter is

**The resolution claim is right, and it is the right thing to have noticed.**
The absolute error in the computed gain is a few times `eps * (left_score +
right_score)` with `eps = 2^-24`, and `left_score + right_score` is about
`parent_score`. So the relative error of the *gain* is about

    eps * parent_score / gain

and the controlling parameter is the ratio `parent_score / gain`, not the row
count. That is worth stating on its own, because it is the number a reader
can go and measure: when a node's best gain is a thousandth of its parent
score, the Float32 gain carries ten bits of information rather than 24.

**The proposed fix does not work.** Dropping the parent term recovers
nothing, because the information was not lost in the subtraction. It was lost
in forming `left_score + right_score`, whose rounding error is already
`eps * parent_score`. Subtracting an exactly-common constant afterward cannot
destroy information it does not have: rounding is monotone, so `x -> fl(x -
p)` preserves order, and when Sterbenz's condition holds (`p/2 <= x <= 2p`,
which is exactly the near-tie regime) the subtraction is exactly
representable and therefore lossless. An earlier version of experiment C
appeared to show the shipped form flipping up to 24.5 percent of node
winners; that generator gave each feature of a node a different node total,
which is unphysical, and those numbers are withdrawn (see section 3). With
the totals shared correctly, comparing `left_score + right_score` directly
picks the identical winner as the shipped form at every setting tested.
Dropping the parent term is a free micro-optimization of one Float32 subtract
per candidate. It is not an accuracy change and should not be described as
one.

### The form that does work

The cancellation is removable, exactly, by algebra. Writing `HL' = HL +
lambda_l2` and `HR' = HR + lambda_l2`, and using `HL + HR = H`:

    gain = (GL*HR' - GR*HL')^2 / (HL' * HR' * (H + 2*lambda_l2))
             -  lambda_l2 * G^2 / ((H + 2*lambda_l2) * (H + lambda_l2))

Experiment C verifies this in exact rational arithmetic over 300 random
inputs with `lambda_l2` in `[0, 10]`: zero mismatches. That part of C stands.
At `lambda_l2 = 0` it collapses to the familiar
`(GL*HR - GR*HL)^2 / (HL * HR * H)`.

The second term depends only on `G`, `H`, and `lambda_l2`, all node
constants. So the argmax over a node's candidates is the argmax of

    D^2 / (HL' * HR'),      D = GL*HR' - GR*HL'

which contains no subtraction of two large nearly-equal quantities, is
positive by construction, and whose relative error is a few ulps of its own
magnitude.

### What it is actually worth

Experiment J sweeps the node's mean gradient, which is what moves
`parent_score / gain`, at 200,000 rows and 20 features, 30 nodes per setting,
7,620 candidates per node. `mu` is the mean gradient in units of the
gradient's standard deviation: `mu = 0` is a perfectly centered node, `mu = 1`
is a node whose rows all pull the same way, which is what the child of a good
split looks like in an early round, and large `mu` is what a nearly-pure leaf
under logistic loss looks like.

| `mu` | parent / best gain | shipped, median rel err | cross form, median rel err | shipped, winner wrong |
| --- | --- | --- | --- | --- |
| 0.0 | 6.7e-05 | 8.9e-08 | 9.6e-08 | 0.0 percent |
| 0.1 | 0.19 | 1.0e-07 | 9.1e-08 | 0.0 percent |
| 0.3 | 1.7 | 2.0e-07 | 1.6e-07 | 0.0 percent |
| 1.0 | 18.6 | 8.4e-07 | 4.3e-07 | 0.0 percent |
| 3.0 | 167 | 7.3e-06 | 1.3e-06 | 0.0 percent |
| 10.0 | 1883 | 8.5e-05 | 4.4e-06 | 6.7 percent |

The shipped form's error tracks `eps * parent / gain` exactly as predicted,
rising linearly with the ratio once the ratio clears one. The cross form's
error rises far more slowly, and the gap between them opens from nothing at
`mu = 0` to a factor of 20 at `mu = 10`.

**And then the honest part: the shipped form's winner is correct everywhere
except the last row, and in that last row the winner it picks holds 0.999879
of the best candidate's exact gain.** So this is not a fix for a bug anyone
is currently suffering. It is a form that is uniformly at least as accurate,
markedly more accurate in the one-sided-gradient regime, and never worse.

Two things keep it on the "take it" list despite the modest measured effect.
It is a **split-selection** error, which is the channel section 2 shows does
not self-correct, so a small effect there is worth more than a large effect
in leaf values. And the regime where it matters, `parent / gain` in the
hundreds or thousands, is exactly the regime that logistic and softmax
objectives spend their late rounds in, which is where the metric is decided
and where nobody has looked.

### The obligations that come with it

The rewrite is not a drop-in, and three things move with it.

`best_gain` is initialized to `0.0`, encoding "no split beats not splitting".
Under the cross form the comparable quantity is not zero, so the
initialization becomes `-inf` and `min_gain_to_split` is applied once to the
winner after converting back to gain units. That conversion is one evaluation
of the full formula per node, not per candidate.

The monotone-constrained branch returns `0.0` as the sentinel for a violating
candidate, and that sentinel must become `-inf` too. That branch's gain is
`output_score(left) + output_score(right) - parent_score`, a different
expression the identity does not cover, so the constrained branch keeps the
subtraction. That is acceptable: it already clamps its outputs, which is a
far larger perturbation than the cancellation.

`SPLIT_TIE_RELATIVE` measures `gain - runner_gain` against a relative
tolerance. The margin is unchanged by dropping a common constant but the
denominator is not, so the relative test must be computed after converting
both back to gain units.

One numerical caveat, from `docs/NUMERICS.md` section 6. `GL*HR' - GR*HL'` is
a product feeding a subtract, which is the contractable shape, and the fused
and unfused forms give different last bits. Both are far more accurate than
the shipped form so the argument is unaffected, but the intent has to be
written down at the site rather than left to the optimizer, which is the rule
that document already sets.

### Verdict

**Take it.** It is an exact algebraic identity, verified in rational
arithmetic, that is never worse than the shipped form and is up to twenty
times better in the one-sided-gradient regime, at the cost of one extra
multiply and one fewer subtract per candidate. It is not a relaxation at all,
it is a strictly better answer in Float32 with no Float64 anywhere, and it
buys insurance in the channel that does not self-correct.

It is **not** the dramatic fix an earlier draft of this document claimed,
and the claim is retracted here rather than quietly softened. The measured
effect on winner selection is zero at every realistic `parent / gain` ratio
tested and 6.7 percent of nodes at an extreme ratio, where the wrong winner
holds 99.99 percent of the right one's gain.

## 9. Candidate 6: integer sibling subtraction inside the device split search

`docs/NUMERICS.md` section 5.5 already flags this as the weak point of the
GPU numerics, and bit-identity was the only thing keeping it.

`gpu_split_search`'s scan derives each candidate's right-hand sums by Float32
subtraction from the node total.

    var lgf = lg.cast[DType.float32]() * g_inv
    var rgf = total_g - lgf

where `total_g` is `tg.cast[DType.float32]() * g_inv` and `tg` is an Int32 sum
over the bins, computed at line 579. The integer totals are available and are
being thrown away. The same pattern appears at lines 643, 737, 806, 845, and
891. Computing `(tg - lg)` in Int32 and dequantizing once is exact in the
integer domain and costs the same instruction count.

Experiment J runs this as a factorial against candidate 5, and the result is
not the one I expected:

| `mu` | parent / gain | sub + f32 right (shipped) | sub + **int** right | cross + f32 right | cross + **int** right |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 6.7e-05 | 8.9e-08 | **6.4e-08** | 9.6e-08 | 8.5e-08 |
| 0.3 | 1.7 | 2.0e-07 | 2.1e-07 | 1.6e-07 | **1.1e-07** |
| 1.0 | 18.6 | 8.4e-07 | 1.8e-06 | 4.3e-07 | **2.5e-07** |
| 3.0 | 167 | 7.3e-06 | 1.5e-05 | 1.3e-06 | **7.4e-07** |
| 10.0 | 1883 | 8.5e-05 | 1.4e-04 | 4.4e-06 | **2.5e-06** |

(median relative error of the computed gain over each node's top 200
candidates.)

**Making the right-hand sum exact makes the shipped subtractive form worse,
by up to a factor of two, and makes the cross form better at every setting.**
That is not a measurement error, it has a mechanism, and the mechanism is
worth writing down because it is a trap.

Under Float32 subtraction, `rgf = total_g - lgf` inherits the error of `lgf`
with the opposite sign. In the subtractive form those two errors then flow
into `lgf^2/HL'` and `rgf^2/HR'`, which are added together and then have
`parent_score` removed, and the anti-correlation partially cancels. The
shipped form is accidentally benefiting from an error it introduces. Remove
the error and the cancellation goes with it. In the cross form the two terms
of `D = GL*HR' - GR*HL'` are genuinely subtracted rather than summed, so the
anti-correlation makes things worse rather than better, and removing it helps.

The practical consequence is a rule, not a caveat: **integer right-hand
subtraction and the cross form must be taken together.** Either alone is
defensible; integer subtraction on top of the shipped subtractive form is the
one combination in this table that is worse than doing nothing.

### Verdict

**Take it with a guard, where the guard is candidate 5.** Taken with the
cross form it is exact where the current form is approximate, it improves
every setting measured by 20 to 45 percent on the median relative error, it
costs nothing because the integer totals are already computed and already in
registers, and `docs/NUMERICS.md` has been asking for it. Taken without the
cross form it is a regression. Ship them in one change or ship neither.


## 10. Other things bit-identity was blocking

Five, in descending order of what they are worth.

**The `--fp-mode` decision is re-openable, and it should be re-opened.**
`docs/NUMERICS.md` section 8 decides to leave `--fp-mode` at its default
`contract=fast`, and the stated reason is explicit: "The reason is that the
project's existing golden bits were produced under `contract=fast`. Turning
contraction off globally would move all of them at once." That reason no
longer exists. Section 10 of the same document records that
`tests/test_golden_bits.mojo` fails all six fixtures under `contract=off`,
three by one ulp on a final raw score and three by more on a stored leaf
value. Under bit-identity that was a blocking finding. Under an accuracy
budget it is a one-ulp-per-tree question that experiment B's flip curve
prices at zero.

What makes this worth re-opening is not the accuracy, it is the maintenance
burden. Section 8 states the consequence of the current decision plainly.
"Because the flag is off, the per-site conventions in section 5 are
load-bearing rather than belt-and-braces. There is no global setting standing
behind them." The project is currently carrying an explicit `fma` call at
`boosting.mojo:1201` and a host-side multiply at
`gpu_objectives_native.mojo:1102` whose entire justification is preserving
bits that no longer need preserving, plus a documented rule that every
contributor must reason about contraction at every site. That is a
permanent tax paid for a property that has been retired. **This is not a
speed item and it should not be smuggled in as one; it is a decision that
should be re-litigated on its own terms now that its stated reason has
evaporated.** The accuracy analysis says the cost is at most one ulp per
site, which the flip curve cannot resolve.

**The constant-hessian elision can be turned on.** `histogram.mojo`
documents it fully, proves it exact (the accumulated series is `1.0 + 1.0 +
...`, every partial sum exactly representable below `2^53`, and the refill
writes `Float64(count)` which is the identical value), and then notes that
nothing in the package calls the predicate. It drops one of three planes'
read-modify-write per (row, feature) for four built-in objectives. It was
already exact, so bit-identity was not what blocked it, but it is sitting in
the same neighborhood and is worth naming because it is the cheapest
unclaimed win in the tree.

**SIMD accumulators in the histogram inner loop.** The accumulation is
scalar read-modify-write today, and `SIMD_LANES` is used only in the
elementwise loops. Multiple partial accumulators per cell, folded in a fixed
lane order, is the same reassociation class as candidate 1 and prices out the
same way: a few ulps, in the direction of exact. Worth having on the list
once row blocks land, because the two interact.

**Pairwise summation in `feature_totals`.** `histogram.feature_totals` lanes
then reduces then handles the scalar tail, and its docstring says the order
"must stay that way". That constraint was bit-identity. A pairwise or
blocked reduction there is strictly more accurate. Low value, because
`feature_totals` runs once per feature per node and not once per row, but
free.

**Extending the device split search downward.** The AUTO strategy currently
sends only large shapes to the device split search, and `thresholds.json`
says why: "Float32 gain comparisons flip near-tie splits, so equally good
trees can disagree on individual rows." That reasoning is sound but it names
the wrong parameter. Experiment I finds no winner errors at all from
1,000,000 rows down to 10,000, and experiment J shows the parameter that
actually controls the Float32 gain's resolution is `parent_score / gain`, not
the row count. **So the row-count crossover is not measuring the thing it was
put there to protect against.** Candidates 5 and 6 together drop the gain
error by 20 to 45 percent at every ratio and remove the only setting where a
winner moved at all. Lowering the crossover is then a decision about Float32
leaf values and device occupancy rather than about split flipping, and it is
worth more speed than anything else on this list, because it moves whole
workloads onto the device rather than shaving a constant off one that is
already there. It needs its own analysis, which this document does not
contain.

## 11. The budget in one table

| # | candidate | channel | error introduced | compounds | held-out estimate | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | row-block private histograms, fixed fold (CPU) | both | 2 to 3 Float64 ulps, *toward* exact | no | none detectable | **take it** |
| 2 | packed int16 in threadgroup memory, max-abs rule at `B = 16`, Int32 globally | both, split dominates | 4e-5 to 8e-4 relative; about 2 percent of effective sample size | through the split channel only | 0.4 percent logloss, 1.8 percent RMSE, inside tolerance | **take it with a guard** |
| 2' | int16 by narrowing the shipped magnitude-sum scale to `2^14` | both, split dominates | 1e-4 to 7e-4 relative; noise ratio grows with node size | same | +4.3 percent logloss, same sign on 6 of 6 seeds, outside tolerance | **not worth it** |
| 2'' | int16 as the global histogram representation | same as 2 | same as 2 | same | same as 2 | **not worth it** (needs per-leaf bit width for no extra saving) |
| 3 | power-of-two device scale | both | up to 1 bit, measured 1.9x worst; dequantization becomes exact | no | none detectable | **take it** |
| 4 | Float32 histogram accumulation on the GPU | both | 100x tighter than shipped, but costs order-independence | no | none, and breaks three invariants | **not worth it** |
| 4' | Float32 histogram cells on the CPU | both | 3e-10 relative | no | none detectable | **take it with a guard** (after 1) |
| 5 | cancellation-free gain form | split only | never worse; up to 20x tighter at high `parent / gain` | fixes an error in the channel that does not self-correct | improvement, not a cost | **take it** |
| 6 | integer right-hand subtraction in the device scan | split only | 20 to 45 percent tighter *with* candidate 5; up to 2x worse *without* it | same | improvement, not a cost | **take it with a guard** (only with 5) |

**Which one buys the most speed per unit of accuracy risk: candidate 1.** It
removes the CPU's structural parallelism cap (fifty features at the
L1-clamped interleave width of four is twelve groups, so twelve tasks on a
machine that wants forty), and its numerical error is *smaller* than what
ships today at every node size measured, four times smaller on sibling
subtraction consistency, and thirteen orders of magnitude below where the
flip curve moves. The risk is not small, it is negative.

**Which one to refuse outright: narrowing the existing magnitude-sum scale to
`2^14` so that it fits an int16 accumulator.** It is the move that looks
obvious from inside the current design, because it changes one constant, and
it is the only candidate in this document that experiment F measures outside
the project's own tolerance: 4.31 percent worse held-out logloss against a 3
percent gate, with the same sign on all six seeds. The mechanism is that the
magnitude-sum rule's per-row quantization step is proportional to the node's
gradient magnitude, so the noise it adds relative to the gradient grows with
the node's row count and is worst exactly at the root, where the tree's most
consequential splits are chosen. That is not a tuning problem. It is why
LightGBM uses a per-row max-abs clamp instead, and any int16 work here has to
start from that rule rather than from ours.

Eight-bit gradients are the same refusal held harder, and for a reason that
is about evidence rather than arithmetic: at four bins the effective sample
size is already down a quarter, and this project has no loss curve that would
justify going further. Produce the curve before asking again.

## 12. What is not measured, and what would settle it

**Nothing here was measured on mojotrees.** Every number is from a NumPy
model. The models are faithful to the arithmetic (experiments H, I, and J
reproduce `gpu_split_search`'s scan expression for expression) but they are
not the code, they do not use the project's binning, they do not use its
tie-break order, and they run on synthetic data.

**One experiment in this document was wrong and was caught by another.**
Experiment C's flip rates were an artifact of a generator that gave each
feature of a node a different node total, and they were believed long enough
to be written into a draft of section 8 as the headline finding. Experiment H
disagreed, experiment I found the reason, and experiment J found the
parameter that actually controls the effect. The withdrawal is recorded in
section 3 and section 8 rather than edited out. **The general lesson is that
a synthetic node generator has to be built from rows, because a node's
features are a partition of one row set and any generator that does not
respect that will manufacture or hide exactly the effects being looked
for.** Anyone extending this work should start there.

**The ensemble experiments are at 20,000 rows and 12 features.** Every effect
that scales with node size (the magnitude-sum noise ratio above all) is
*understated* there. The ensemble result is a lower bound on the interesting
cases, not a characterization of them. It also runs 100 rounds with no early
stopping on a problem the reference model overfits, which is why several arms
score better than exact; the sign of those numbers is a property of the
configuration and only their magnitude and seed-consistency transfer.

**Six seeds is not many, and the intended control arm did not work.** The
permuted-feature-order arm was meant to be the noise floor and produced
exactly zero difference on every seed, because with continuous gradients
there are no exact ties for it to break. The arms that do set the floor are
the ones equivalent to a perturbation of `1e-9` or smaller, and they put the
resolution at roughly a tenth of a percent in the mean with a worst single
seed near half a percent. Nothing below that in experiment F is a finding.

**The two places where toolchain and hardware matter.** First, FMA
contraction: everything in this document assumes IEEE-754 operations, and
`docs/NUMERICS.md` records that this project has been bitten three times by
the compiler contracting a multiply into an add. The cross form of candidate
5 contains a product feeding a subtract (`GL*HR' - GR*HL'`), which is exactly
the contractable shape, and the two contractions give different last bits.
This does not affect the argument, because both are more accurate than the
shipped form, but it means the cross form must be written with an explicit
intent and a comment, per NUMERICS section 6. Second, Metal's absence of
64-bit threadgroup atomics, which is what scopes candidate 2.

**Three things this document does not price and someone should.** The
Float32 leaf values the device already computes, which are a leaf-channel
error and therefore cheap by section 2's argument but have never been
measured. The interaction between candidate 1 and candidate 4', which pull
the interleave width in opposite directions and whose combined optimum is a
scheduling question, not a numerical one. And the AUTO crossover for the
device split search, which section 10 argues is keyed to the wrong parameter.

**What would settle all of it is one run of `bench/real_data`.** Six
scenarios, before and after, three repeats, one machine, pinned data. That
harness exists, its thresholds were chosen before any run, and it has never
produced a committed record. Until it does, the strongest statement this
document supports is that candidates 1, 3, 5, and 6 are very unlikely to move
a metric, and that is a statement about mechanisms, not a measurement.
