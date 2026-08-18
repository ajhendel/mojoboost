# What was measured on 2026-08-18, and what it retired

This file exists so nobody measures these again.

It is not a narrative of the day. It is the list of questions that were
CLOSED by a run, with the number, the run id, and the thing the number
retires. Everything here was previously scattered across commit messages and
source docstrings, which is the right place for why a line of code is the way
it is and the wrong place for what we already know and should not pay for
twice. `docs/design/DECLINED_OPTIMIZATIONS.md` remains the register for
measured nulls and losses; this is the session's index into it and into the
run records.

Read the two protocol facts first, because every number below depends on
them. This machine drifts two to three times across thermal windows, so only
arms INTERLEAVED INSIDE ONE RUN compare. And a spread is part of a result: a
median with no min and max is not a measurement here.

---

## 1. Per-node bin layout, `MOJOTREES_CPU_LAYOUT_BY_NODE`. CLOSED, FLIPPED ON.

Run `20260818T143944Z-cpuloc`, `bench/real_data/arms_cpu_locality.py`. Four
arms interleaved, three whole-process repeats, real data, two shapes.

| scenario | shape | baseline s | with the rule | ratio |
|---|---|---|---|---|
| year, 463,715 x 90 | 256 leaves, depth 8 | 11.86 | 9.35 | **1.269x** |
| year, 463,715 x 90 | 31 leaves | 4.77 | 3.90 | **1.224x** |
| covertype, 464,809 x 54 x 7 | 256 leaves, depth 8 | 36.90 | 34.37 | 1.074x |
| covertype, 464,809 x 54 x 7 | 31 leaves | 19.61 | 19.58 | 1.001x |

Baseline spreads were 0.9 and 3.6 percent on the two year cells, so those
resolve. The covertype deep baseline spread was 19.6 percent and does NOT
resolve. The covertype default cell is a null.

**Bit-identity was measured, not argued.** Across all twelve switched cells
the run recorded ONE distinct `predictions_sha256` per (scenario, shape),
shared with the baseline and with both other switched arms. One digest over
the predictions, not one metric value.

**WHAT THIS RETIRES.** The prediction was that this rule fires on the small
and tiny node classes, so it should have been loudest on covertype's complete
depth-8 tree where row blocking stops below 8,160 rows and 192 of every 255
splits land in those classes. It is loudest on 90 dense continuous columns at
EVERY shape, including 31 leaves where those classes barely appear, and it
does nothing on 54 mostly-binary columns at either shape. **The small-node
locality explanation is dead.** Do not open a lane on it. What the switch is
actually doing is not yet mechanistic and should not be quoted as though it
were.

**READ THIS WITH SECTION 6, BECAUSE THE TWO LOOK LIKE THEY DISAGREE AND DO
NOT.** Section 6 finds that small-node degradation IS the story for the
accumulate, at 19.4x. That is not a contradiction of the sentence above, and
the distinction is the whole mechanism. What is dead here is small-node
**read** locality as the explanation for THIS SWITCH: the switch changes which
address a bin id is loaded from, and it wins on 90 dense continuous columns
rather than on small nodes. What section 6 identifies is small-node **write**
reuse in the histogram, which this switch does not touch at all. Same node
classes, opposite side of the memory traffic.

Written down explicitly because a reader skimming for "is small-node locality
the problem" would find both sentences and pick one, and picking the wrong one
costs a lane.

## 2. Feature-group schedule clamp, `MOJOTREES_CPU_FEATURE_GROUP`. CLOSED, NULL.

Same run, same interleaving. Setting the group width to the cache-derived
value the schedule clamp throws away read **1.002x** on year deep and
**0.967x** on covertype default.

**And setting BOTH switches is worse than the layout switch alone at every
cell that resolved**, 1.201x against 1.269x on year deep and 1.142x against
1.224x on year default. The two interact negatively.

**WHAT THIS RETIRES.** The argument that `_schedule_group` manufacturing 20
dispatch units for a pool measured at about 3.5 wide is costing locality worth
having. It is not, at these shapes. Do not reopen without a new mechanism.

## 3. LightGBM model import. CLOSED, THIRTEEN OF FOURTEEN BIT-IDENTICAL.

`bench/lgbm_interop_matrix.py`, which trains through lightgbm 4.7.0, saves
through lightgbm, reads back here, and compares `raw_score` on every row.

Bit-identical, exactly equal on every row: binary, regression, regression_l1,
poisson, five-class multiclass, depth-12, depth-1 stumps, 1023 bins, 15 bins,
L1 plus L2, categorical, ninety-percent-sparse, single-tree.

Refused by name: a model trained on data containing NaNs carries a non-finite
threshold. Missing-value models are unsupported and are rejected rather than
mispredicted.

**WHAT THIS RETIRES.** The status string that said "no file written by a real
LightGBM build has ever been read". It stood for as long as nobody tried. Also
retires the open question of whether CatBoost's `model_evaluation_speed`
benchmark is enterable: it is, because our arm can predict from THEIR checked-in
LightGBM model rather than one we fit, which removes the different-model
objection before a reader raises it.

**STILL OPEN, and the reason the word experimental stays**: nothing this
writer produced has been read back BY LightGBM. Only half the round trip has
evidence.

## 4. Symmetric trees at depth 7 and 8. CLOSED AS A REACHABILITY FACT.

Symmetric, 60,000 x 24, eight trees, one fit at each depth on each backend.
Not a timing.

    depth 6  cpu ok  gpu ok, rmse bit-identical at 0.332497
    depth 7  cpu ok  gpu raises at the batcher
    depth 8  cpu ok  gpu raises at the batcher
    depth 9  cpu ok  gpu DeviceUnavailableError, which is correct

**WHAT THIS RETIRES.** The claim, in a commit message of mine, that raising
`OBLIVIOUS_MAX_LEAVES` removed the depth ceiling. The bound is written down
FOUR times and two were raised. Depth 9 refusing is intended: the wide scan's
twelve shared arrays are 24,588 bytes at 512 leaves against a conservative
16,384-byte budget, which is the first bound in this family that genuinely
binds.

**STILL OPEN**: `gpu_leaf_batching.OBLIVIOUS_MAX_ITEMS` is still 64 and is
what refuses now.

## 5. gbm-bench year, 500 trees. TAKEN, AND IT DOES NOT RESOLVE.

`bench/results/gbm_bench_year_2026-08-18_113620.json`. One run of each arm.

    lgbm-cpu        22.30 s   MSE 79.7905
    mojotrees-cpu   32.02 s   MSE 80.2766
    mojotrees-gpu   31.51 s   MSE 79.9967

**Do not compare these to this morning's run at
`bench/results/gbm_bench_2026-08-18/`.** The box moved: LightGBM did not
change and went from 28.94 to 22.30. One run of each arm is not a
measurement, which the runner says in its own closing text.

**No ratio derived from this run should be quoted.** Using the comparator as
a clock assumes drift is uniform across arms and it is not; memory-bound and
compute-bound arms drift differently under thermal state. The direction is
all that survives.

**ONE OBSERVATION WORTH KEEPING, at the same weak confidence.** Our CPU came
in at 32.02 against our GPU's 31.51. This morning that gap was 47.18 to 34.41.
If that holds under repeats, year joins covertype as a shape where our two
backends are within noise of each other, and the case for routing year to the
accelerator gets much weaker.

---

## 6. The covertype CPU round, per phase. CLOSED, AND IT REVERSED TWICE.

`MOJOTREES_PHASE_PROFILE=async`, covertype shape 464,809 x 54 x 7 classes,
`num_leaves=256 max_depth=8`, CPU arm, ten rounds.

    histogram (accumulate)   62.0 pct
    split_search             15.1
    partition                15.0
    subtract                  7.7
    hist_alloc                0.2

    histogram ns per thousand slots, by node class
      root      234.1     large 311.1     medium 653.8
      small    1705.0     tiny 4544.3       degradation 19.4x

Our own prior note had that degradation at 8.7x. Nobody had measured it on
this dataset.

**THE PRIZE.** 57,706,074 slots at the root rate of 0.2521 ns per slot would
cost 14.5 ms. The measured histogram phase is 36.8 ms. So **22.3 ms is
addressable, 61 percent of the phase and 38 percent of the whole CPU round.**

**FIRST READING, AND IT WAS WRONG.** The accumulate is proportional to SLOTS,
not cells, so a packed histogram cannot touch it and compaction only attacks
the 7.7 percent subtract plus part of the scan. That is correct about the PASS
COUNT and wrong about the MECHANISM, and I published it before checking the
mechanism.

**WHAT ACTUALLY DEGRADES: histogram WRITE REUSE, and it tracks the cost
exactly.** Updates per histogram cell per node, against ns per thousand slots:

    class    updates/cell   ns/kslot
    root           1822.8        252
    large           295.2        290
    medium           71.3        653
    small             9.1       1257
    tiny              1.4       3240

At 1.4 updates per cell there is no reuse at all; every update is a cold
access. The bin READ is not the story, which is independently confirmed by the
per-node row-major layout switch measuring 1.074x and 1.001x on this exact
dataset while winning 1.269x on year.

**AND THE BYTES ARE DECISIVE.** At 24 bytes per cell:

    rectangular   54 x 255 = 13,770 cells = 322.7 KB   SPILLS L1 BY 5.0x
    packed          ~2,400 cells          =  56.2 KB   FITS L1

The M4 has 64 KB of L1 data cache per core. **The rectangular histogram does
not fit and the packed one does.** So compaction reaches the 62 percent after
all, not by doing fewer passes but by giving every cell 5.7x more reuse and
moving the working set from L2 into L1.

**WHAT THIS RETIRES.** My own claim, made an hour earlier in this same session,
that the cell-count theory was dead. It is not. It was the right refutation of
the wrong mechanism. Anyone quoting "compaction cannot touch the accumulate"
should read this row instead.

**STILL NOT MEASURED**, and this is arithmetic over a profile rather than an
A/B: nobody has built a packed histogram and timed it. The prediction is
specific enough to falsify, which is the point of writing it down: if a packed
accumulate does not move the small and tiny classes toward the root rate, the
reuse model is wrong too.

**AND THE INSTRUMENT HAS A HOLE.** Every per-cell row on the GPU arm reads
ZERO. The instrumentation is CPU-path only and the device reports host-step
spans instead. So this discriminates for the CPU arm only, and the GPU's 4.5x
still has no per-phase attribution. Claiming otherwise, which I did for several
hours, was wrong.

## Questions this session OPENED and did not close

Listed so they are not mistaken for answered.

**The covertype 4.5x has three competing explanations and none has been
measured on covertype.** Host-bound per-split encode (the 85.74 percent figure
is from a DIFFERENT shape), rectangular-versus-packed histogram cells, and
per-tree launch overhead. The per-cell instrumentation to discriminate them
already exists in `phase_profile.mojo`. **Nothing should get a lane on
covertype before that profile runs.**

**Whether any of today's twenty commits slowed the GPU arm.** Its ratio to the
comparator moved 18 percent across two runs. A three-arm interleaved A/B is
queued: this morning's commit, head, and head with `OBLIVIOUS_MAX_LEAVES`
reverted to 64. Note the mechanism caution from the advisor, which is better
than the hypothesis: Metal threadgroup memory belongs to each pipeline state
rather than to the library, so a lossguide kernel's occupancy should not move
because a symmetric kernel grew its static allocation. If the A/B is null,
bisect rather than stop.

**Whether the layout win reaches the depth-wise and symmetric growers.**
`bench/real_data/arms_layout_policies.py` is written and has not been run. It
matters for this switch specifically, because `_env_layout_by_node` did not
reach the symmetric grower until 2026-08-17 and a measurement taken before
that recorded neutral. Neutral is what a switch that does not reach the code
always measures.

**EFB is not bit-identical**, by two mechanisms, one of them structural
(default-bin recovery by subtraction). It is a Tier 2 change under rule 11 and
needs an anchor and regenerated goldens, not a flip. And the 3.5x estimate is
in doubt: `expand_bundled_histogram` writes the full rectangular output per
node, serially, so the honest prediction is a large win on the top levels and
possibly a loss on the leaf frontier.

## 7. The multiclass serial passes. CLOSED, 1.21x, and it is the evening's only win.

Run `20260818T230849Z-postfix`, covertype and year, three repeats, LightGBM in
every cell as the drift canary.

**Compared by RATIO TO THE CANARY, not by seconds.** The two changes are
unconditional rather than switched, so they cannot be interleaved against
their own baseline in one process, and raw seconds across windows are
worthless on a box that drifts 2 to 3x. The ratio is what survives.

    covertype, 31 leaves, mojotrees / LightGBM

    earlier tonight, four runs   4.14x  4.39x  4.45x  4.57x
    after the two changes        3.62x

    absolute: 16.93 s against 18.85, 19.88, 21.75, 22.32 earlier

**1.21x, and the control holds.** `dense_regression` reads 1.36x against a
prior range of 1.31x to 1.51x, unchanged, which is what it must be: the
parallel softmax is multiclass-only. A run where year had also improved would
have been a window effect rather than a change effect.

**WHAT WAS ACTUALLY WRONG.** A softmax round has two whole-dataset passes and
both ran on one core while the rest of the round used the pool: the
probability pass (`n_classes` exponentials per row, 325 M `exp` per covertype
fit) and the per-class derivative fill (a `List.append` loop run seven times
per round, 700 times per fit).

They were not disabled for multiclass. They were UNREACHABLE from it:
`_fill_softmax_grad_hess` took no `DispatchSettings`, so there was nothing to
dispatch with, while the single-output twin had been parallel for a long time.
Both functions were correct, so nothing failed and nothing said so.

The audit predicted 16 to 27 percent of the fit. Measured 21 percent.

**`_RowPool` is in this number and cannot be separated from it.** It landed in
the same window and is bit-identical, and on a 56-tree synthetic it read
inside noise. Whether it contributes at 700 trees is not established by this
run; the multiclass passes are sufficient to explain the whole 1.21x.

**WHAT THIS SAYS ABOUT THE THREE NULLS BEFORE IT.** Compact histogram
addresses, compact addresses in the right kernel, and the 16-byte cell all
measured nothing. This one, which removed no bytes and moved no cells and only
put existing work on the pool, is worth 21 percent. The evening's lesson is
that the time was not where the phase profile's 62 percent pointed.
