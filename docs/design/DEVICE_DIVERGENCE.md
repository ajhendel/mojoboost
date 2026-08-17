# Why the CPU and GPU backends disagree, and why the three growers disagree by different amounts

Written 2026-08-17 by a read-only diagnosis lane. **Nothing here was built,
compiled, timed or run.** Every claim is labelled **READ** (from the source or
stored artifact named, with file:line), **DERIVED** (algebra over read facts,
with the assumption stated), or **UNKNOWN**. Part 4 is the run plan and it
carries most of the weight, because the central question turns out to be
settled by a measurement nobody has taken rather than by a mechanism nobody
has found.

## 0. The headline, before the detail

**The asymmetry in the brief is not established to be a property of the three
growers, and the strongest reading of the evidence already in the repository
is that it is a parameter confound dominated by the learning rate.** The three
numbers come from three harness arms that differ in eight parameters, four of
which set directly how much of a shared per-tree perturbation reaches a
prediction. The relevant three are `learning_rate` 0.1 / 0.3 / 0.5,
`min_data_in_leaf` 20 / 1 / 1, and leaves 31 / 64 / 64.

**READ**, `docs/design/RANDOM_STRENGTH_UNITS.md:63-79`, which is the run these
numbers come from (`20260817T124906Z-postflip`, `dense_regression`, standard
tier), and which already contains the depth-wise control and reaches this
conclusion in its Finding A:

| arm | policy | `random_strength` | lr | depth / leaves | `min_data_in_leaf` | max abs diff | mean abs diff | rmse cpu | rmse gpu |
|---|---|---|---|---|---|---|---|---|---|
| `mojotrees` | leaf-wise | 0 | 0.1 | -1 / 31 | 20 | 0.116 | 0.014 | 0.310775 | 0.310847 |
| `mojotrees_depthwise` | depth-wise | 0 | 0.3 | 6 / 64 | 1 | 0.435 | 0.036 | 0.325803 | 0.324934 |
| `mojotrees_catboost_mode` | symmetric | 1.0 | 0.5 | 6 / 64 | 1 | 0.493 | 0.029 | 0.308262 | 0.307693 |

The brief's three metric figures are these rows, **DERIVED** by division:
0.000072 / 0.310775 = 2.32e-04, 0.000869 / 0.325803 = 2.67e-03, and
0.000569 / 0.308262 = 1.85e-03. So the provenance is confirmed and the
comparison is same data, same seed, and **not** same parameters.

The scaling is the finding:

| quantity | leaf-wise | depth-wise | symmetric |
|---|---|---|---|
| learning rate (**READ**) | 0.1 | 0.3 | 0.5 |
| max abs row diff (**READ**) | 0.116 | 0.435 | 0.493 |
| row diff / leaf-wise row diff (**DERIVED**) | 1.0 | 3.75 | 4.25 |
| lr / leaf-wise lr (**DERIVED**) | 1.0 | 3.0 | 5.0 |
| metric gap / leaf-wise metric gap (**DERIVED**) | 1.0 | 11.5 | 8.0 |
| (lr ratio)^2 (**DERIVED**) | 1.0 | 9.0 | 25.0 |

Row-level divergence tracks the learning rate almost linearly and the
primary-metric gap tracks it roughly quadratically, which is what a
fixed per-tree perturbation multiplied by the shrinkage predicts. The residual
(depth-wise over-runs its lr^2 prediction, symmetric under-runs it) is what a
grower-specific mechanism would have to explain, and it is about 1.3x and
0.3x, not 8x to 12x.

**The 5e-09 figure in the brief is not comparable to the other three and
should not be reasoned from.** `bench/results/RESUME_2026-08-17.md:115-116` is
the sentence the brief inherits, and it pairs a LARGE-tier leaf-wise number
with STANDARD-tier symmetric and depth-wise numbers. The nearest large-tier
leaf-wise cell that has been read is
`bench/results/COMPARISON_RUN_2026-08-16.md:88-92`: RMSE 9.10607 gpu against
9.10383 cpu, a relative gap of **2.46e-04**, with `max |gpu - cpu| = 9.67`.
That is the same order as the standard tier's 2.32e-04, not five orders
smaller. **UNKNOWN**: which cell produced 5.8e-09, and whether that cell's
`device_used` was actually `gpu` and its split search actually the device one.
Experiment 0 in Part 4 settles it, and until it does, "five to six orders of
magnitude" is a comparison between two different runs at two different tiers.

Likewise the brief's max-abs figures (9.67, 22.5, 10.5) are from a different
run than its metric figures: the run that produced the metric figures has max
abs diffs of 0.116, 0.435 and 0.493 (**READ**, table above). 9.67 is the
large-tier leaf-wise value (**READ**, `COMPARISON_RUN_2026-08-16.md:101`).
Mixing them makes the row-level and metric-level stories look inconsistent
when they are not.

## 1. The comparison table, per decision point, per policy

Growers. Leaf-wise device path `gpu_resident_round.mojo:2417`
(`grow_tree_device_resident`); symmetric device path
`gpu_resident_round.mojo:3318` (`grow_tree_device_oblivious`); depth-wise
device path `train_gpu.mojo:2744` (`_device_search_resident`). Host oracle
paths: `split.mojo:582` (`find_best_split`, per node, leaf-wise and depth-wise)
and `split.mojo:1280` (`find_best_split_shared`, per level, symmetric);
schedule `growth_policy.mojo:659` (`next_leaf`), `:439` (`rank_level`), `:497`
(`admit_level`), `:739` (`plan_level`).

All cells **READ** unless marked.

| decision point | leaf-wise | depth-wise | symmetric (oblivious) |
|---|---|---|---|
| **candidate enumeration order** | features in active-slot order, bins ascending, missing-left before missing-right; `_scan_slot_kernel` `gpu_split_search.mojo:2005`, host `find_best_split` `split.mojo:582`. Identical rules both sides. | same kernel, same order; `_device_search_resident` batches a level into one launch pair but "the scan order inside a node is unchanged" (`train_gpu.mojo:2744` docstring). | same order, one extra loop: leaves ascending by record, innermost. Device `_scan_slot_oblivious_kernel` `gpu_split_search.mojo:3618`; host `find_best_split_shared` `split.mojo:1620` ("THE CROSS-LEAF REDUCTION. Outer loop over leaves, ascending"). |
| **candidate SET identical across backends?** | yes. Top-threshold break `if b == n_scan - 1 and miss_c == 0` present in host (`split.mojo:1064`) and in all three per-node kernels. | yes, same break. | **now yes, was no until 2026-08-17.** Both oblivious kernels walked `range(n_scan)` and scored one candidate the host never enumerates (host bound `split.mojo:1596`). Fixed at `gpu_split_search.mojo:4106` (`n_top = n_scan - 1` when `not any_missing`). Inert with the noise off; a live divergence with it on. **READ**, `RANDOM_STRENGTH_UNITS.md:223-256`. |
| **gain expression and precision** | host Float64 over Float64 histogram sums; device Float32 over dequantized fixed-point Int32. Default arm is `GAIN_FORM_CROSS` (`gpu_split_search.mojo:557`), which evaluates the same gain through an identity that never forms the parent-score sum, so the resolution is about `eps * sqrt(parent_score * gain)` instead of `eps * parent_score`. | identical to leaf-wise; same kernel, same default form. | **different, and this is the one real per-policy precision difference.** L2 arm is `total += gpu_split_gain(...)` in Float32 over leaves (`gpu_split_search.mojo:4242`, accumulator declared `:4181`), which does use the cross form. **The Cosine arm cannot**: `total = gpu_cosine_score(cos_num, cos_den) - level_parent` (`:4265`), a Float32 subtraction of two level-wide quantities that are nearly equal when the split is weak, with no cross-form identity available because the score is a ratio. Host does the same subtraction in Float64 with Float64 accumulators (`split.mojo:1522` `acc_left`, `:1535` `den_left`). |
| **which score function each shipped arm actually uses** | `SCORE_L2`. `score_function="Cosine"` was silently ignored on every leaf-wise GPU fit until it was fixed, and the only arm that sets Cosine grows symmetric trees (`bench/real_data/scenarios.py:5100-5109`). | `SCORE_L2`; `MOJOTREES_DEPTHWISE` sets no `score_function` (`scenarios.py:1499-1507`). | `SCORE_COSINE`; `MOJOTREES_CATBOOST_MODE["score_function"] = "cosine"` (`scenarios.py:1574`). |
| **tie-break between candidates inside a node/level** | strict `>` over an ascending candidate ordinal on both sides; the cross-feature fold is `block.max` on gain then `block.min` on slot, which is the same rule in two halves. `gpu_split_search.mojo:80-104`. Verified by construction and by test: `tests/test_split_tie_parity.mojo`, and 1,200 pseudo-random nodes agreed 1,200 of 1,200 given the same histogram (**READ**, `bench/results/session3_2026-08-16/RESULTS.md:870-877`). | same. | same, plus the cross-slot fold reused unchanged (`_reduce_slots_kernel` `gpu_split_search.mojo:5088` over `n_records = 1`). |
| **tie-break between NODES, i.e. which leaf splits next** | argmax over the frontier one pick at a time, ties to the lower frontier slot (`growth_policy.mojo:659`). Reported but not decided by `frontier_margin` (`gpu_split_search.mojo:5670`). | **a whole-level ranking**: `rank_level` sorts gain descending then node id ascending with a bare `>` and an exact `==` (`growth_policy.mojo:459-470`), and `admit_level` admits a gain-ranked PREFIX bounded by `num_leaves` (`:497`). Membership, not order. Its own docstring records that exact gain ties "are common in practice, not a corner case". | **no per-node ranking exists**; a level splits entirely or not at all, and `num_leaves` does not bind (`growth_policy.mojo:118-127`). One decision per level, applied to every leaf of it. |
| **the gain VALUE the ranking compares** | device record gain, computed in Float32 and widened to Float64 (`gpu_split_search.mojo:8789` region); host gain Float64. `train_gpu.mojo:2636-2645` builds `LeafCandidate` from `frontier[i].rec.gain`. | same source, `train_gpu.mojo:3009-3018`. So `rank_level`'s Float64 `>` and `==` are applied to Float32-derived numbers on the GPU and to Float64 numbers on the CPU. | not applicable. |
| **leaf value, and its precision** | `gpu_leaf_value` in Float32 from fixed-point sums (`gpu_split_search.mojo:499`, written at `:5191-5197`); host Newton step Float64 over Float64 sums. Stated as not bit-identical at `docs/design/OBLIVIOUS.md:45-49`. | same. | same function, computed rather than copied because there is no per-leaf record: `_commit_level_kernel` (`gpu_tree_tables.mojo`, docstring "the leaf values are the same `gpu_leaf_value` a leaf-wise commit copies out of the record"). |
| **`min_data_in_leaf`** | exact Int32 counts on the device, exact integers on the host. Applied per child in the scan on both sides. | same. | applied **per leaf against that leaf's own sums**, and a leaf that fails contributes zero (L2) or its unsplit terms (Cosine) instead of vetoing the candidate. Device `gpu_split_search.mojo:4207-4218`; host `split.mojo:1684-1706`. The rule is ours and has no CatBoost referent (`split.mojo:1339-1346`, `growth_policy.mojo:119-126`). |
| **`min_child_hess`** | compared against a dequantized Float32 hessian sum on the device, a Float64 sum on the host. For squared error the hessian is 1 per row and the quantized sum is exact (**DERIVED**: `sum h = n`, so the power-of-two scale makes each row's quantized hessian an integer and the sum exact below 2^24), so the comparison agrees. For logistic and softmax it does not. | same. | same, and it is the one threshold comparison that can convert a tiny perturbation into a whole leaf's worth of change in the level score, because a leaf flips between contributing zero and contributing its gain. See H4. |
| **`lambda_l2`** | applied in the denominator on both sides, Float32 vs Float64. `BASE_PARAMS` sets 0.0 for the harness's mirror pair (`scenarios.py:236`); the shipped library default is 1.0 (`tree.mojo:284`). | 1.0 (`scenarios.py:1503`). | 3.0 (`scenarios.py:1572`). |
| **monotone / interaction constraints** | reproduced candidate for candidate; clamp applied before the accumulators are built on both sides (`gpu_split_search.mojo:80-104`). Not exercised by any arm here. | same. | monotone rejection folded into `gpu_cosine_level_terms` and returns the leaf's unsplit terms rather than a flag (`gpu_split_search.mojo:3728-3760`), matching `find_best_split_shared`'s `cn = ct.num if ct.ok else pt.num`. |
| **sibling subtraction** | used on both backends. Device subtracts Int32 in place and it is exact (`train_gpu.mojo:2762-2766`). Host subtracts Float64 planes (`histogram.mojo:3594`, `_subtract_histogram_arrays`), which is not exact but is Float64. | same. | same on both sides; device arm is `oblivious_subtract_requested`, default ON since 2026-08-17 (`gpu_leaf_batching.mojo`); host builds the smaller child of each pair and subtracts (`tree.mojo:2269`, `_grow_oblivious_levels` docstring). |
| **what the histogram itself holds** | device: Int32 fixed point at `2^30 / sum|g|` rounded down to a power of two (`histogram_gpu.mojo:343`, `quantized_gradient.fixed_point_scale_pow2`). Host: Float64 sums of Float64 gradients. **The two backends never read the same histogram on any policy.** | same. | same. |
| **where the gradients come from** | device objective plane when reachable: raw scores, labels, weights, gradients and hessians all **Float32** on the device against Float64 on the host (`gpu_objectives_native.mojo:64-72`). | same. | **host gradients**, because `bootstrap_type=MVS` routes the fit to the host-gradient arm of `_train_gpu_rounds` (**READ**, `RANDOM_STRENGTH_UNITS.md:118-124`). So the device objective plane is exonerated for the symmetric arm specifically. |
| **`random_strength`** | 0 on every arm here. | 0 (`scenarios.py:1499-1507` sets none). | 1.0 (`scenarios.py:1604`). Draw is host-computed Float64 and uploaded as a Float32 plane, so the addend agrees to Float32; every other link was verified equal (**READ**, `RANDOM_STRENGTH_UNITS.md:98-121`). Decays to 6e-6 of strength by round 48 at lr 0.5, so it is arithmetically absent for 70 of 100 trees (**DERIVED** there, `:36-45`). |
| **row sampling** | none. | none. | MVS at 0.8 (`scenarios.py:1601-1608`). **UNKNOWN**: whether the sampled row set is bit-identical across backends on this arm. `RESUME_2026-08-17.md:124-126` records that any DEVICE MVS draw would be equivalent in distribution and not bit-identical; the host-gradient routing above suggests the draw is host-side and shared, but that was not read end to end. |

## 2. The asymmetry, and the mechanisms ranked by how much of it they can carry

The framing correction first, because it changes what has to be explained.

**The brief's premise that the fixed-point histogram is not a divergence
source is true of the reduction and false of the inputs.** Integer addition is
associative so the device's accumulation is exact and order-independent, and
that is a real designed property. But the *values* being accumulated are
quantized gradients, and the host accumulates unquantized Float64 ones, so the
two backends read different histograms on every policy at every size
(`gpu_split_search.mojo:104-108` says exactly this: "the two backends do not
read the same histogram at all"). The repository has already localized the
measured divergence to precisely that shared plane, twice and by two
independent routes:

- **READ**, `COMPARISON_RUN_2026-08-16.md:135-152`. The same shape run three
  ways in one window: CPU, GPU on the host scan, GPU forced onto the device
  scan. GPU-host-scan against GPU-device-scan is **0.0000 max, 0.00 percent of
  rows differing**, while both miss the CPU by an identical 9.6719. Two
  entirely different code paths, one issuing 15,100 dispatches with 3,100 host
  histogram downloads and the other 100 transfers, produce the same
  predictions. So the divergence lives in what they share, which is the
  fixed-point histogram and the Float32 leaf value, and it is not the split
  search, not the launch structure and not the near-tie resolution.
- **READ**, `session3_2026-08-16/RESULTS.md:870-889`. Given the *same*
  histogram, the device replica and the host scan chose the identical split
  1,200 times out of 1,200, and every disagreement that does occur is a near
  tie. The lane's own conclusion: "The two backends do not read the same
  histogram."

So the object to explain is not "why does the GPU diverge" but "why does a
fixed per-tree perturbation of the same size on all three paths land 8x to
12x harder on two of them". Ranked answers.

### H1. Learning rate. Can carry 9x of the 11.5x and 25x of the 8x. LEADING.

**Mechanism.** The perturbation is per tree and per leaf value. A prediction
is `base + lr * sum over trees of leaf_value`, so a fixed relative error in
every leaf value produces a prediction difference proportional to `lr`, and a
prediction difference `d` that is not aligned with the residual raises RMSE by
about `d^2 / (2 * RMSE)`, so the metric gap goes as `lr^2`. **DERIVED.**

**Direction.** Correct: the two divergent arms are the two arms with the
larger rates, and they are divergent in the same order as their rates.

**Magnitude.** `lr^2` ratios of 9.0 and 25.0 against observed metric-gap
ratios of 11.5 and 8.0. Row-diff ratios of 3.75 and 4.25 against `lr` ratios
of 3.0 and 5.0. Both quantities land within about 1.3x of the prediction in
one direction and 3x in the other, on two arms, with no free parameter.
Nothing else on this list comes close to that.

**Falsifier.** Run the three policies at a matched `learning_rate`. If the
three metric gaps do not converge to within about 2x of each other, H1 is
insufficient and something grower-specific is live. Experiment 1.

**Note the direction of the residual, because it is informative.** Depth-wise
diverges *more* than `lr^2` predicts (11.5 against 9.0) and symmetric *less*
(8.0 against 25.0). H2 explains depth-wise's excess. Symmetric's shortfall is
consistent with `lambda_l2 = 3.0` damping every leaf value relative to
depth-wise's 1.0 and leaf-wise's 0.0, and with `random_strength` being
arithmetically dead after round 48 while the leaf-value perturbation is not.
Both are **DERIVED** and neither is measured.

### H2. Leaf freedom: `min_data_in_leaf` 20 against 1, and 31 leaves against 64. Can carry 2x to 5x. STRONG, and it is H1's co-factor.

**Mechanism, two halves.** First, averaging. A leaf's Newton value is
`-T(G) / (H + lambda)`, and `G` is a sum of `m` quantized gradients whose
roundings are independent, so the quantization error in `G` grows about as
`sqrt(m)` while `|G|` grows the same way for a well-mixed leaf: the relative
error is roughly flat in `m`. But at `m = 1` there is no cancellation to help
and no averaging at all, and the single row's rounding lands undamped in the
leaf value. `min_data_in_leaf = 20` against `1` therefore changes the *worst*
leaf on the tree from one row to twenty. Second, count. 64 leaves against 31
doubles the number of independent perturbed values a row's prediction sums
over, and puts half of them at depths leaf-wise never reaches.

**Direction.** Correct, and it is the same partition as H1: the two divergent
arms are exactly the two arms with `min_data_in_leaf = 1` and 64 leaves.

**Magnitude.** `sqrt(20) = 4.5x` on the relative error of the smallest leaf,
which squares to about 20x in the metric if the smallest leaves dominate and
to nothing if they do not. **DERIVED**, and the spread between those two ends
is why this is ranked below H1 rather than beside it. `RANDOM_STRENGTH_UNITS.md:83-87`
reaches the same conclusion from the same table: "`min_data_in_leaf=1` with 64
leaves gives many near-ties, and a learning rate of 0.3 to 0.5 multiplies one
different split into a visibly different ensemble over 100 rounds ...
divergence grows with learning rate and with leaf freedom".

**Falsifier.** Experiment 1 varies both together and Experiment 2 separates
them. If setting `min_data_in_leaf = 20` on the symmetric and depth-wise arms
at their own learning rates does not move the gap, H2 is dead.

### H3. Depth-wise only: `admit_level`'s gain-ranked prefix cut turns a gain comparison into MEMBERSHIP. Can carry the 1.3x residual. REAL, and it does NOT fire on the measured arm.

**Mechanism.** `rank_level` (`growth_policy.mojo:439`) totally orders a
level's candidates by gain descending then node id ascending, and `admit_level`
(`:497`) takes the highest-gain prefix that fits the remaining `num_leaves`
budget. When the budget cuts a level, an inversion across the cut changes
*which node is split at all*, not the order in which two nodes are split. On
the GPU the ranking key is a Float32-computed gain widened to Float64
(`train_gpu.mojo:3009-3018`); on the CPU it is a Float64 gain. And exact ties,
which `rank_level`'s own docstring says are common because "two siblings split
on the same feature over identical bin totals score identically", are resolved
by node id on the CPU and by quantization noise on the device, because
quantization essentially never leaves two gains bitwise equal. So on the
device the two backends are using different tie rules in practice, not
different last bits.

Leaf-wise has nothing equivalent. Its 30 picks are all taken; an inversion
between two nearly equal frontier leaves changes the order and usually
converges to the same set, because both leaves get split. Only the final pick
is a membership decision, and it is a max over a broad frontier rather than a
rank-15-against-rank-16 comparison among siblings at one depth.

**Whether it fires.** At the library default `num_leaves = 31` with
`max_depth = -1` (`tree.mojo:284`), depth-wise reaches 16 leaves after level 3
and level 4 offers 16 candidates against a budget of 15, so **exactly one node
per tree is dropped by a gain ranking, in every tree**. **DERIVED**, from
`leaf_budget` and `admit_level`. But `MOJOTREES_DEPTHWISE` sets
`num_leaves = 64` and `max_depth = 6` (`scenarios.py:1499-1507`), which makes
level 5 offer 32 candidates against a budget of 32: **the cut does not fire on
the measured arm.** So H3 cannot be the mechanism for the 2.67e-03 figure, and
it is registered here because it fires for a *user at defaults* and nothing in
the harness covers that configuration.

**Falsifier.** Compare depth-wise at `num_leaves = 31, max_depth = -1` against
`num_leaves = 32, max_depth = 5` at one learning rate. The first cuts every
tree, the second fits exactly. If the two agree, H3 is dead as a live
mechanism and stays only as a defaults hazard. Experiment 3.

### H4. Symmetric only: a per-leaf THRESHOLD comparison with no near-tie damping, inside a level score. Can carry a large factor on logistic objectives and about nothing on squared error. REAL, and out of scope for the measured arm.

**Mechanism.** Under a shared split a leaf that fails `min_data_in_leaf` or
`min_child_hess` contributes zero (L2) or its unsplit terms (Cosine) instead
of its gain (`gpu_split_search.mojo:4207-4218`, `split.mojo:1684-1706`). That
is a step function of a threshold comparison, so a leaf sitting within
quantization distance of the threshold flips the level score by a whole leaf's
worth, roughly 1/64 of the level score, rather than by an ulp. Every other
mechanism on this list is a gain comparison, which `ACCURACY_BUDGET.md`
section 6.1 records as damped: "Experiment B's flip curve does not leave zero
until a perturbation of 1e-3". A threshold comparison has no such damping, and
that same section flags it as unpriced: "nothing has looked at what a 4.9e-08
cell error does to `min_data_in_leaf` and `min_sum_hessian_in_leaf` at the
boundary, where a constraint check is a comparison against a threshold rather
than a gain comparison".

**Why it does not explain the measured number.** On `dense_regression` the
objective is squared error, so counts are exact integers on both backends and
the hessian sum is exact under the power-of-two scale (**DERIVED**, Part 1),
which leaves nothing at the boundary to flip. `min_child_hess = 0.0` on the
symmetric arm removes the hessian test entirely.

**Where it would show.** Binary and multiclass. `SESSION_QUEUE.md:583-586`
records `device_agreement` on `imbalanced_binary` going WARN to FAIL at
`max |gpu - cpu| = 0.231` with `average_precision` 7.4 percent relative
against a 0.5 percent limit, and the registered 2x2 that would separate its
two candidate causes is dated "no code moves on this until the 2x2 reads".
**UNKNOWN**: whether that 2x2 was ever run. Experiment 5.

### H5. Symmetric only: the Cosine level score is a Float32 cancellation that GAIN_FORM_CROSS cannot reach. Can carry an order of magnitude on the SPLIT CHOICE, and it did not show. REAL, RANKED LOW, and worth one control.

**Mechanism, and it is the one genuine per-policy precision difference in the
table.** The leaf-wise and depth-wise per-node scans default to
`GAIN_FORM_CROSS` (`gpu_split_search.mojo:557`), whose entire purpose is that
it "never forms the large sum", moving the resolution from about
`eps * parent_score` to about `eps * sqrt(parent_score * gain)`
(`gpu_split_search.mojo:44-56`). The symmetric arm scores with Cosine, and the
Cosine level score is computed as
`gpu_cosine_score(cos_num, cos_den) - level_parent` at
`gpu_split_search.mojo:4265`. That subtraction *is* the cancellation the cross
form exists to remove, taken in Float32 against a level-wide parent score, and
there is no cross-form identity for a ratio, so the remedy is structurally
unavailable. The module's own measured table puts `parent_score / gain` "in
the thousands" and ranging to 293, which puts the resolution of a level
decision at roughly `1e-7 * 1e3 = 1e-4` relative instead of leaf-wise's
`1e-7`. **DERIVED**, and it is three orders of magnitude, which is the right
size to flip a level decision.

**Why it is ranked below H1 and H2 anyway, and this is the discipline the
brief asks for.** A flipped level decision is the strongest structural
amplifier available: one level choice reroutes every row below it, so a flip
at depth 0 or 1 makes the whole tree different for 100 percent of rows,
whereas a leaf-wise flip moves only that node's rows. But the measured
evidence is against it carrying the observed gap. The noise-free depth-wise
arm, which scores with L2 under the cross form and has no level-wide
cancellation at all, diverges **more** than the symmetric arm on the relative
metric (2.67e-03 against 1.85e-03). If the Cosine cancellation were the
dominant term, symmetric would be worse than depth-wise and it is not.
`RANDOM_STRENGTH_UNITS.md:75-79` makes the same argument for a different
parameter and it applies here unchanged.

**Falsifier, and it is cheap.** Run the symmetric arm at
`score_function="l2"`, everything else unchanged. Under L2 the level score is
a Float32 sum of per-leaf cross-form gains with no level-parent subtraction
(`gpu_split_search.mojo:4242`, and `level_parent` is only set under Cosine at
`:4121`), so the cancellation disappears while every other property of the
policy is held. If the gap drops by an order of magnitude, H5 is the story and
H1 is a coincidence. If it does not move, H5 is real and inert. Experiment 4.

### H6. The Float32 cross-leaf accumulation itself. About 1e-6. CANNOT explain 1e-3, and is listed so it is not proposed again.

The oblivious kernels sum up to 64 leaves' contributions into a Float32
accumulator (`gpu_split_search.mojo:4181`, `:4186`) where the host uses Float64
planes (`split.mojo:1522`, `:1535`), and `gpu_cosine_level_terms`'s docstring
records a further deliberate association divergence: the two accumulator adds
are explicit `fma` where the host's `+=` leaves two roundings. Both are real,
both are documented as divergences, and both are bounded by about
`64 * eps_f32` relative, which is 8e-6. **A mechanism that can only produce
8e-6 does not explain a 1.85e-03 gap no matter how real it is.** The same
applies to the Float32 leaf value (6e-8 relative), the Float32 noise plane
(one ulp of the noise), the sqrt in `gpu_cosine_score` (correctly rounded), and
the host's Float64 sibling subtraction (about 1e-16).

### H7. The device objective plane's Float32 raw scores. Not the symmetric arm at all, and unmeasured elsewhere.

`gpu_objectives_native.mojo:64-72` **READ**: the device carries raw scores and
labels as Float32 where the host carries Float64, and `update_raw` accumulates
`raw[r] += learning_rate * value[leaf_id[r]]` in Float32. On a target whose
offset dwarfs its residual scale, that is a cancellation site the host does
not have: `dense_regression`'s label spread is about 1.8 and its RMSE about
0.31, but YearPredictionMSD's labels sit near 2000 with an RMSE of 9.1, where
a Float32 label carries up to 6.1e-05 of absolute representation error and 100
rounds of Float32 accumulation add a random walk on top. **DERIVED**: that
reaches about 1e-3 absolute, which is far short of the 0.46 median row
difference measured on that dataset, so it is not the whole story there
either. It is the standing candidate for the open question
`COMPARISON_RUN_2026-08-16.md:155-160` leaves: "what data conditions amplify
fixed-point histogram error by five orders". It is **not** a candidate for the
symmetric arm, which runs host gradients.

### Answers to the four items the brief asked to be confirmed or ruled out

- **(a) Does the symmetric path's level-wide single choice amplify one near
  tie into every node at that level?** **Yes, structurally, and it is
  confirmed by reading.** One `(feature, threshold, direction)` per level
  applied to every leaf (`growth_policy.mojo:105-117`), committed by
  `_commit_level_kernel` onto all `L` parents at once. Leaf-wise structurally
  cannot have it. But the measured ordering (depth-wise worse than symmetric)
  says this amplifier is not what is dominating, and the reason it is not is
  that its per-decision flip probability is *lower*, not higher: a level score
  is a sum over 64 leaves whose quantization errors are independent, so its
  relative noise is about `1/sqrt(64)` of one leaf's, while the number of
  decisions per tree falls from 30 to 6. Fewer, better-conditioned decisions
  with catastrophic consequences, against many low-consequence ones.
- **(b) Is sibling subtraction used on some paths and not others, and is
  fixed-point subtraction exact?** Used on **all six** paths, host and device,
  all three policies (Part 1 row). Device subtraction is exact because
  accumulation is Int32 under one scale per tree, so a parent's bins are the
  exact integer sum of its children's; there is no residue
  (`train_gpu.mojo:2762-2766`). Host subtraction is Float64 and inexact at
  about 1e-16. **Ruled out as a source of the asymmetry.**
- **(c) Is leaf-wise's small figure genuine tightness or luck?** **Neither,
  on the evidence available: the figure is not comparable.** At the same
  tier, same run and same dataset as the other two, leaf-wise is 2.32e-04, and
  the nearest read large-tier leaf-wise cell is 2.46e-04. There is no measured
  leaf-wise cell anywhere in `bench/results/` that is at float noise on a
  regression scenario, and there is one that is explicitly not
  (`COMPARISON_RUN_2026-08-16.md:110-120`: 100.00 percent of test rows differ
  by more than 1e-06). Multiclass predictions *are* sha256-identical across
  backends (`session3_2026-08-16/RESULTS.md:884-886`), so a 5.8e-09 figure is
  what a scenario whose metric is insensitive to leaf-value perturbation looks
  like, not what a tight leaf-wise regression looks like. Leaf-wise is neither
  safer nor luckier; it is *less amplified*, by a factor its arm's parameters
  set.
- **(d) Does tie amplification interact with DEPTH?** For symmetric, yes and
  structurally: a level choice at depth `k` propagates to every one of the
  `2^(d-k)` subtrees below it, so the earlier the level the more prediction
  mass one flip moves, and the total number of level decisions is `max_depth`.
  For depth-wise the interaction is different and is through H3: the budget cut
  fires at the level where `2^k` first exceeds the remaining `num_leaves`
  budget, so it is `max_depth` and `num_leaves` jointly that decide whether it
  fires at all. Neither interaction is measured. Experiment 6 measures both.

## 3. `host_rescan_recommended`: what it is, and the verdict

**What it computes.** `gpu_split_search.mojo:5609`. Given one
`GpuSplitRecord`, it answers whether that node's decision fell inside Float32's
resolution, by delegating to `GpuSplitRecord.is_near_tie` (`:5522`). The test
is the margin `gain - runner_gain` against the wider of two widths: a relative
one, `SPLIT_TIE_RELATIVE = 1e-6` of the gain (`:436`), and
`resolution_floor()` (`:5498`), which is
`8 * eps_f32 * max(parent_score_bound, sqrt(parent_score_bound * gain))` and
is the absolute width the scan's own arithmetic could not see past on that
node. It reads nothing from the device, costs a few arithmetic operations on a
record the caller already holds, and cannot change which candidate won:
`runner_gain` is tracked by one extra compare per candidate. It raises on a
nonpositive tolerance. Tested at `tests/test_split_tie_parity.mojo:498-499`.

**What a caller would do with a True.** Download that node's histogram
(`GpuHistogramBuilder.download_raw`, then `histogram_from_host`) and take
`_search`'s answer for that node instead of the record's, keeping the device
decision everywhere else. Per node, not per tree.

**Does it address the mechanism in Part 2? No, and this is not a judgement
call.** Two reasons, both **READ**.

1. **A host rescan reads the same quantized histogram.**
   `histogram_gpu.histogram_from_host` (`:2109`) is documented cell for cell
   as `Float64(Int32) * (1.0 / scale)` with "no accumulation, no
   reassociation, and no cross-cell dependence". So the rescan replaces
   Float32 arithmetic with Float64 arithmetic **over the same fixed-point
   gradients**. It cannot recover the host's Float64-gradient histogram,
   which is the plane the divergence was measured to live in.
2. **The measurement that matters has already been taken, and it read zero.**
   `COMPARISON_RUN_2026-08-16.md:135-152`: GPU on the host scan against GPU on
   the device scan, same shape, same window, is **0.0000 max and 0.00 percent
   of rows differing**, while both miss the CPU by an identical 9.6719. The
   GPU host-scan arm *is* `device_exact_ties` with the knob turned to 100
   percent of nodes rather than to the flagged handful. It made no difference
   at all. So wiring `host_rescan_recommended` would reduce the divergence by
   an amount already measured to be zero on the one shape it was measured on,
   and the flagged subset can only be a subset of that zero.

**So: detector, not fix.** It correctly detects nodes whose Float32 decision
was a coin flip, and that is a genuinely useful instrument. It does not reduce
CPU/GPU divergence, because the divergence is not in the Float32 decision.

**Two live contradictions this creates, which should be corrected in whichever
lane owns those files.** `gpu_split_search.mojo:64-70` and `:97-99` state that
`host_rescan_recommended` "is the answer to both" the Float32 scan and the
different histograms; it is the answer to the first only, since a rescan reads
the second. And `tests/test_split_tie_parity.mojo:428-433` asserts that "a
prediction gap two orders of magnitude past the agreement limit is the Float32
near-tie resolution this module documents, whose remedy is
`host_rescan_recommended`", which the three-arm window contradicts directly.
Both were written before that window and neither has been updated.

**Expected speed cost, from the code and without measuring.** Per flagged
node: one `download_raw` of `3 * n_features * n_bins` Int32 words, one host
synchronization, one `histogram_from_host` conversion over the same cells, and
one host `_search`. `histogram_from_host`'s own docstring prices the conversion
at about 10 ns per cell from the retired hybrid scheduler's calibrated model,
against a modeled device fixed cost per node of roughly 263 microseconds. At
100 features and 255 bins that is 76,500 cells, about 300 KB across the bus
plus a full pipeline drain, against the 136 bytes a device record costs. So
the cost per flagged node is roughly the whole per-node saving the device
search exists to capture, and the total cost is linear in the flag rate.
**UNKNOWN**: the flag rate. Nothing has counted how many nodes
`is_near_tie` answers True for on a real fit, which is the number that decides
whether this is free or ruinous, and it is countable without changing a bit
because the records already carry `runner_gain`.

**Which of the two governing rules this is.** Neither, as it stands. It is not
an internal choice costing accuracy that a user cannot reach, because it
changes nothing today: it has no caller
(`session3_2026-08-16/RESULTS.md:906-909`, "Built, documented, dead"), so there
is no accuracy being paid for speed anywhere. And it does not qualify as an
exposed knob trading about 1 percent of accuracy for at least 2x speed,
because it goes the wrong way on both axes: it costs speed and, on the one
measurement available, buys **zero** accuracy. **Recommendation: do not ship it
as `device_exact_ties`.** Ship it as an *instrument* instead, a counter that
reports the near-tie rate per tree under a trace flag, which costs nothing, is
already computed, and produces the one number Part 4 needs. If the rate turns
out to be high on some scenario, the knob argument can be reopened with
evidence; today it has none.

## 4. Experiments, ranked, for the orchestrator

Every one of these is a device-agreement question, so every cell needs its CPU
oracle twin (`verify.check_device_agreement` skips a GPU row with no CPU twin
and only warns, `bench/real_data/verify.py:1958-1983`). Record for every cell:
`primary_metric` on both devices, the relative gap, `max |gpu - cpu|`, mean
and p99 absolute row difference, `device_used`, `backend_proof`, and the
`MOJOTREES_GPU_SPLIT_TRACE=1` line naming the resolved split path. Save
predictions, or the row-level half of every conclusion is unavailable.

### Experiment 0. Provenance of 5.8e-09. Zero fits if the artifact exists.

**Do this first and do not run anything else until it answers**, because three
of the four numbers this whole investigation is about are consistent with each
other and the fourth is not.

Find the record behind `RESUME_2026-08-17.md:115`. Report its scenario, tier,
arm, `device_used`, `device_requested`, the resolved split path, and whether
its predictions were saved. **Falsifies the premise** if it turns out to be a
multiclass or otherwise perturbation-insensitive scenario, or a cell whose
`device_used` was `cpu`, or a cell with no device histogram. If no such record
exists, the figure should be withdrawn from `RESUME` and the brief's "five
orders of magnitude" retired, because the comparable large-tier leaf-wise
number that *is* on disk is 2.46e-04.

### Experiment 1. Matched-parameter control. THE decisive one. 6 cells.

Three policies, one parameter set, standard tier `dense_regression`:
`learning_rate=0.1`, `num_leaves=31`, `max_depth=6`, `min_data_in_leaf=20`,
`min_child_hess=1e-3`, `lambda_l2=1.0`, `lambda_l1=0.0`,
`score_function="l2"`, `random_strength=0`, no bootstrap, `n_estimators=100`.
CPU and GPU per policy. `max_depth=6` on all three so leaf-wise is bounded the
same way; note that symmetric will produce 64 leaves regardless because
`num_leaves` does not bind under it, and that is a residual confound this
experiment cannot remove.

- **Confirms H1 and H2** if the three relative metric gaps land within about
  2x of each other, all near 2e-04.
- **FALSIFIES H1 and H2** if symmetric or depth-wise stays 5x or more above
  leaf-wise at matched parameters. That result would mean a grower-specific
  mechanism is live, and H5 (symmetric) and H3 (depth-wise) become the
  candidate set, discriminated by Experiments 3 and 4.
- Note this is not a re-measurement of the gap. It is the only run in the
  repository that would compare the three growers rather than three arms.

### Experiment 2. Separate the two confounds. 8 cells, only if Experiment 1 confirms.

On the depth-wise arm only, a 2x2 over `learning_rate` in {0.1, 0.3} and
`min_data_in_leaf` in {1, 20}, everything else at the arm's own values, CPU
and GPU each.

- Row-level divergence should scale about linearly in `learning_rate` at fixed
  `min_data_in_leaf`, and the metric gap about quadratically. **Falsifies H1**
  if the gap is flat in the learning rate.
- **Falsifies H2** if the `min_data_in_leaf` column is flat.
- The interaction cell is what says whether the two multiply or one dominates,
  which no existing run can answer.

### Experiment 3. Does depth-wise's budget cut matter? 4 cells. NO CODE CHANGE.

Depth-wise at `learning_rate=0.1`, `min_data_in_leaf=20`, two shapes:
(a) `num_leaves=31, max_depth=-1`, where the level-4 cut drops exactly one
node of sixteen in every tree; (b) `num_leaves=32, max_depth=5`, where every
level fits its budget exactly and `admit_level` never cuts. CPU and GPU each.

- **Confirms H3** if (a) diverges materially more than (b).
- **FALSIFIES H3** if they agree, which would retire the ranked-prefix cut as
  a live mechanism and leave it only as a defaults hazard worth documenting.
- Worth running even if Experiment 1 confirms H1, because (a) is the *library
  default* shape (`num_leaves=31`, `max_depth=-1`) and no arm covers it. A
  user asking for `grow_policy="depthwise"` and nothing else gets it.

### Experiment 4. Does the Cosine level cancellation matter? 4 cells. NO CODE CHANGE.

The symmetric arm as shipped, against the same arm with
`score_function="l2"` and nothing else changed. CPU and GPU each.

- **Confirms H5** if the L2 cell's gap drops by roughly an order of magnitude.
  That is the result that would overturn H1 as the leading explanation, so this
  experiment is a genuine discriminator and not a consistency check.
- **FALSIFIES H5** if the gap does not move, which retires the level-wide
  Float32 cancellation as real-but-inert and leaves H1 and H2 unopposed.
- Second reading for free: it also prices the `random_strength` units defect,
  since a 1.0 under L2 is weaker than a 1.0 under Cosine by a factor in the
  hundreds (`RANDOM_STRENGTH_UNITS.md` section 3).

### Experiment 5. The registered 2x2 that was never read. 8 cells.

`imbalanced_binary`, standard tier, leaf-wise, the 2x2 over `lambda_l2` in
{0.0, 1.0} and `derivative_precision` in {f64, f32}, exactly as pre-registered
at `bench/results/session3_2026-08-16/RESULTS.md:889-916`, whose readings are
already written down so the interpretation cannot be chosen after the fact.
This is the only experiment on this list that targets a logistic objective,
and it is the only one that can exercise H4, because on squared error the
count and hessian thresholds are exact on both backends.

- **Confirms H4** if the failure tracks `lambda_l2 = 0` rather than the CPU's
  derivative precision, since that is the Float32 gain near the admission
  boundary. The pre-registered readings cover all four outcomes.
- Register before running that H4 predicts the *symmetric* arm on
  `imbalanced_binary` should be worse than the leaf-wise one there by more
  than the learning-rate ratio alone explains, because only symmetric has the
  per-leaf zero-contribution threshold. That is a second, independent
  falsifier for H4 and needs two more cells.

### Experiment 6. The depth interaction. 8 cells, lowest priority.

Symmetric at `max_depth` in {3, 4, 5, 6} at matched `learning_rate=0.1`,
`min_data_in_leaf=20`, `score_function="l2"`, CPU and GPU each.

- **Confirms (d)** if divergence rises with depth faster than the leaf count
  does, which is what a level choice propagating to `2^(d-k)` subtrees
  predicts.
- **FALSIFIES (d)** if divergence is flat in depth, which would say the
  amplifier is per-leaf-value and not per-level-decision, and would further
  strengthen H1 and H2 against H5.

### Experiment 7. Count the near ties. Zero extra fits, instrument only.

Under a trace flag, report per tree the number of records for which
`is_near_tie()` answers True, with and without `resolution_aware`, and
`frontier_margin` per growth step, for all three policies. The records already
carry `runner_gain` and both widths are pure arithmetic on a record the host
already holds, so this changes no bit and adds no launch. It produces the one
number Part 3 says is missing and it settles whether the near-tie population
is even large enough for a per-node fallback to be affordable.

- If the rate is near zero on all three, `host_rescan_recommended` is
  confirmed dead and can be deleted or demoted to a pure instrument.
- If the rate is high on symmetric and low on leaf-wise, that is independent
  support for H5 and the knob argument reopens with evidence.

## 5. What reading could not settle

- **Whether the residual after the learning rate and leaf freedom are matched
  is zero.** This is the whole question and only Experiment 1 answers it. My
  ranking says H1 and H2 carry most of the 8x to 12x, but "most" is
  `lr^2` arithmetic against two data points, not a measurement.
- **The provenance of 5.8e-09.** Experiment 0.
- **Why YearPredictionMSD amplifies the shared perturbation by five orders
  over well-conditioned synthetic data.** `COMPARISON_RUN_2026-08-16.md:155-160`
  left this open and it is still open. H7 is the standing candidate and is
  about 3 orders short of it by my own arithmetic.
- **The near-tie flag rate on any real fit.** Experiment 7.
- **Whether the symmetric arm's MVS row sample is bit-identical across
  backends.** If it is not, that divergence is first-order and everything in
  this document is downstream of it. One read of the MVS draw's routing under
  `bootstrap_type=MVS` on the GPU path settles it and I did not complete it.
- **Whether the registered `imbalanced_binary` 2x2 was ever run.** Experiment 5
  assumes it was not.

## 6. One correction to a shipped statement, unrelated to the ranking

`bench/real_data/thresholds.json`, `defaults/device_agreement/rationale`, says
the row bound "holds when both devices grow the same trees (the host split
search guarantees that)" and attributes the breach to "the workload-aware AUTO
strategy" sending large shapes to the device split search. Both halves are
now false. The host split search does not guarantee it, because a GPU host
scan reads the same quantized histogram as the device scan and the two were
measured bit-identical against each other and equally distant from the CPU
(`COMPARISON_RUN_2026-08-16.md:135-152`). And AUTO no longer routes on
workload at all: both work thresholds were withdrawn on 2026-08-16 and
`gpu_split_policy.decide_split_search` (`:310`) sends every eligible shape on
measured hardware to the device search at every size. The gate's *number* is
right and should not be tightened, for the reason the brief gives. Its stated
reason is wrong, and a gate whose rationale names the wrong mechanism will be
loosened or tightened for the wrong reason by whoever reads it next.
