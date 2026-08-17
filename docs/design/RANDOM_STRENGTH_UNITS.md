# `random_strength`: what its number means, and what to measure before changing it

Written 2026-08-17 by the `random_strength` lane. Four defects were assigned.
Two were fixed in code, one is a units problem this note prices and does not
change, and one is an experiment this note specifies. Read the CODE for the
first two; this file is the argument, the arithmetic, and the run plan.

Everything below separates **read** (from the source named), **measured**
(from a record in `bench/real_data/results/`, cited by run id), and
**derived** (algebra or an order-of-magnitude estimate, with the assumption
stated). Nothing here was built or run: the lane had no compiler and no
measurement budget.

---

## 0. The parameter in one paragraph

A seeded normal is added to every candidate split's score before the argmax,
so the winner comes from among the near-ties rather than always being the
exact maximum. Same family as `extra_trees` and far gentler, since that one
randomizes the threshold outright and this only perturbs the ranking. The
standard deviation is CatBoost's `CalcScoreStDev`:

```
scoreStDev = random_strength
           * RMS over the whole learn set of the per-row weighted derivatives
           * n / (n + exp(iteration * learning_rate))
```

READ from `catboost/private/libs/algo/greedy_tensor_search.cpp` and
`rand_score.h`; the full transcription with line numbers is the comment block
headed `random_strength: seeded noise on a candidate's gain (CatBoost)` in
`src/mojotrees/tree_parameters_extra.mojo`, which remains the reference text.

**The decay term is stronger than it reads and it matters to every experiment
below.** `n / (n + exp(iteration * learning_rate))` is exactly `1/2` at
`iteration = log(n) / learning_rate` and falls off exponentially after that.
DERIVED, for the shipped CatBoost-mode arm on `dense_regression` at the
standard tier (159,649 training rows, resolved `learning_rate` 0.5, 100
trees): half strength at round 24, `6e-6` of strength by round 48. **So on
that arm the regularizer is alive for roughly the first thirty trees of a
hundred and is arithmetically absent for the rest.** At CatBoost's constant
rate of 0.03 the same crossing is round 399. Any sweep over `n_estimators` or
over `learning_rate` is therefore also a sweep over how much of the run the
regularizer exists for, and a sweep that does not say so will read as a
`random_strength` result when it is a decay-schedule result.

---

## 1. Defect 2, the headline: what the 0.493 CPU/GPU divergence is

### The claim under test

Run `20260817T124906Z-postflip`, `dense_regression`, standard tier, symmetric
arm (`mojotrees_catboost_mode`), which passes `random_strength: 1.0`
EXPLICITLY on both backends. CPU rmse 0.308262, GPU rmse 0.307693, and
`max |gpu - cpu|` over the 40,351 test predictions is 0.493, the largest of
any arm in that run. If the noise is seeded and the histograms are
fixed-point exact, that should be far smaller.

### Finding A, MEASURED: most of the 0.493 is not the noise

The same run carries a control nobody had read as one. The
`mojotrees_depthwise` arm sets **no** `random_strength` (the mode rule gives
CatBoost's 1.0 only under `grow_policy=oblivious`; `depthwise` inherits
LightGBM's absence), no `score_function`, and no bootstrap. Its parameters
and its CPU/GPU divergence, computed from the committed prediction vectors in
that run's `predictions/` directory:

| arm | `random_strength` | lr | depth / leaves | `min_data_in_leaf` | max abs diff | mean abs diff | rmse cpu | rmse gpu |
|---|---|---|---|---|---|---|---|---|
| `mojotrees` (lossguide) | 0 | 0.1 | -1 / 31 | 20 | **0.116** | 0.014 | 0.310775 | 0.310847 |
| `mojotrees_depthwise` | **0** | 0.3 | 6 / 64 | 1 | **0.435** | 0.036 | 0.325803 | 0.324934 |
| `mojotrees_catboost_mode` | **1.0** | 0.5 | 6 / 64 | 1 | **0.493** | 0.029 | 0.308262 | 0.307693 |

A noise-free arm on the same data at the same depth diverges by 0.435, and
its RELATIVE rmse gap between backends (0.27 percent) is LARGER than the
noisy arm's (0.18 percent). So the bulk of the 0.493 is a property of the
free-growing shape at a large learning rate and not of `random_strength`.

The mechanism, DERIVED and consistent with the table: the device histograms
are fixed-point Int32 and the device scan is Float32 while the host scan is
Float64, so a near-tie between two candidates can resolve differently.
Downstream, `min_data_in_leaf=1` with 64 leaves gives many near-ties, and a
learning rate of 0.3 to 0.5 multiplies one different split into a visibly
different ensemble over 100 rounds. The ordering in the table is exactly what
that predicts: divergence grows with learning rate and with leaf freedom, and
the constrained lossguide arm at lr 0.1 with `min_data_in_leaf=20` is five
times smaller. This is the existing "documented near-tie divergence"
explanation, and the depthwise control is the evidence for it that was
missing.

Note also that `max` over 40,351 rows is a tail statistic. Mean absolute
divergence on the noisy arm is 0.029 and p99 is 0.124, against a label spread
that puts rmse at 0.308.

### Finding B, a REAL noise-specific bug, found by reading and now fixed

Everything about the noise ITSELF checks out between the backends, and I
verified each link rather than assuming it:

- **Same key construction.** `tree_parameters_extra.score_stream_in` and
  `gpu_split_search.gpu_score_stream_in` are the same five splitmix64 folds
  with the same irregular `+1` pattern.
- **Same domain constant.** The symmetric CPU path calls
  `oblivious_score_noise` (`split.mojo:1917`) and the device plane is drawn by
  `oblivious_score_plane`, both in `_OBLIVIOUS_SCORE_DOMAIN` /
  `OBLIVIOUS_SCORE_DOMAIN`, which are the same word. This is the divergence
  fixed earlier the same day and it is fixed.
- **Same site.** Both key the level by its DEPTH. `stage_random_score_level`
  passes `depth`; `find_best_split_shared` passes `depth`.
- **Same feature index.** The device plane is keyed by the GLOBAL feature id
  read back out of the staged feature table (`_record_features`), not by the
  slot, so a narrowed or reordered feature set permutes the plane and changes
  no value in it. The host keys by `f`, also global.
- **Same standard deviation.** Both backends of this arm run the same host
  function. `bootstrap_type=MVS` routes the GPU fit to the HOST-gradient arm
  of `_train_gpu_rounds`, whose call is
  `_round_random_score_scale(params.tree.extra, grad, n, i, learning_rate)` at
  `train_gpu.mojo:4484` -- byte for byte `boosting._boost_rounds`'s call. The
  device-gradient arm's separate `_device_round_random_score_scale` is not
  reached by this arm at all.
- **Same placement.** Both add the draw to the level's aggregate score after
  the cross-leaf reduction and after the single Cosine ratio, once per
  (feature, bin), shared by the two routing directions.
- The only residual is the Float32 rounding of the plane, which is one ulp of
  the noise and is inside the Float32 difference the gain already carries.

**And then the candidate SETS differ by one candidate.** The host level scan
takes

```
n_top = n_scan if level_miss_c > 0 else n_scan - 1     # split.mojo:1596
```

because the top threshold puts every ordinary bin left and is only a split at
all when missing rows are there to fill the right child. Every per-NODE scan
in the package has the same rule as a `break`: `split.find_best_split` at
`split.mojo:1064`, and the three device kernels `_scan_slot_kernel`,
`_scan_slot_wide_kernel` and `_scan_slot_wide_primitive_kernel`, each spelled
`if b == n_scan - 1 and miss_c == 0: break`.

The two OBLIVIOUS kernels did not have it. Both walked `range(n_scan)`.

With the noise **off** that is inert, which is why it survived review: the
extra candidate is refused by every leaf (`tc - lc == 0 < min_data_in_leaf`),
so under Cosine each leaf contributes its unsplit terms, the ratio comes back
as exactly `level_parent`, `total` is exactly `0.0`, and `0.0 > best_gain` is
false at a `best_gain` that starts at `0.0`. Under L2 the illegal
contribution is `0.0` and the same argument holds.

With the noise **on** it scores `0.0 + noise` and wins on any positive draw
that beats the slot's real candidates. Two consequences, both bad:

1. It is a candidate the CPU cannot elect at all. That is a genuine
   CPU/GPU divergence caused by `random_strength`, of exactly the kind
   defect 2 hypothesized.
2. If it wins the level, the record it writes is internally inconsistent:
   `cand_c`, `cand_gf` and `cand_hf` are never incremented on a candidate no
   leaf accepted, so the slot reports `IREC_LEFT_COUNT = 0` while the
   threshold routes every row left, and the elected split has an empty child.

There is a second, smaller case with the noise OFF, which is why the fix is
written as the host's rule rather than as "skip the zero-score candidate".
`min_data_in_leaf` is only required to be nonnegative
(`validation.check_booster_ranges`), so at `min_data_in_leaf = 0` beside
`min_child_hess = 0` the empty right child passes both legality tests, the
candidate is scored for real, and its score is the level's own minus the
level's own -- about zero, but not exactly zero in Float32, so it can win a
level where nothing else scored positive. The host still never enumerates it.
No shipped configuration sets `min_data_in_leaf = 0`; the CatBoost-mode arm
sets 1.

The docstring on `_scan_slot_oblivious_kernel` asserted that this kernel,
`_scan_slot_kernel` and `find_best_split_shared` "all agree" about a
candidate every leaf refuses. They agree about every candidate the host
enumerates. This one the host does not enumerate.

**FIXED, as a bug and not behind a switch. Bit-identical for every fit with
the noise off and `min_data_in_leaf >= 1` or `min_child_hess > 0`, which is
every shipped configuration; bits move only where the two backends disagreed.**
In both oblivious kernels:
`_scan_slot_oblivious_kernel` now hoists `any_missing` out of its direction
loop and bounds its bin loop by `n_top`, and `_scan_slot_oblivious_wide_kernel`
narrows `n_scan` by one before it partitions candidates across threads. The
two remain bit-identical to each other by construction, and the wide kernel's
identity argument survives the changed partition because its cross-leaf sums
are private per candidate, its cross-bin prefix is Int32, and its winner is
chosen by `block.min` over the ascending ordinal rather than by thread index.

**Frequency: UNMEASURED, and I will not guess it.** Reachability is proven by
construction (the shipped CatBoost-mode arm has no missing bin, so
`any_missing` is false on every feature of every level, and the draw is
positive half the time); how often the degenerate candidate beat the slot's
real winner is a question only a run answers. See section 4.4 for the
measurement that answers it.

### What this means for the 0.493 number

MEASURED: at least 0.435 of it, and probably nearly all of it, is the
Float32/fixed-point near-tie divergence that the noise-free depthwise arm
shows at the same shape. DERIVED: the bug above can only widen that, never
narrow it. The honest statement after the fix is that `device_agreement` on
the symmetric arm should come back at or below the depthwise arm's number,
and that if it does NOT, the remaining gap is a second defect and this note's
Finding A is where to start.

---

## 2. Defect 1: the CPU-only mode default

`src/mojotrees/params.mojo` supplied CatBoost mode's `random_strength = 1.0`
only when `config.device == CPU_DEVICE`. The same parameter string therefore
built a different model on `device=gpu` -- and on `device=auto`, which is
neither -- than on `device=cpu`, with nothing said. It also poisons
`device_agreement`, which compares a GPU row against its CPU twin and would
report a parameter difference as a backend divergence.

**The condition was stale, verified from history and from the call graph.**
It was written at `e3cfb47` on 2026-08-16, when the accelerator genuinely
could not honor the parameter. The capability landed the next day: both arms
of `train_gpu._train_gpu_rounds` compute the per-tree scale, the oblivious
level launch stages and reads the noise plane (`c775959`), and
`ExtraTreeParams.device_unsupported_reason`'s `random_strength` arm refuses
only `random_strength` beside a categorical feature.

Three other surfaces had already retired the identical device test and only
this one was left behind, which is the strongest evidence it was an oversight
rather than a rule:

- `bindings/_mojotrees.mojo` widened `random_strength_ok` to `True` for `fit`
  and to `not d[].is_sparse` for `train_dataset`, with a comment dated
  2026-08-17 saying so. This is why the bench harness's GPU arm carries
  `random_strength: 1.0` today.
- `src/mojotrees/device_policy.mojo`'s `BLOCK_RANDOM_STRENGTH` narrowed to
  `random_strength > 0.0 and request.categorical`.
- `ExtraTreeParams.check_random_strength`'s own refusal message names both
  arms of the GPU round loop as computing a scale.

**FIXED as a bug, no switch.** Two edits in `params.mojo`:

1. The mode default drops the device test and keeps the multiclass one.
2. `_validate` passes `scale_computed_per_tree=(not config.is_multiclass())`
   instead of `(config.device == CPU_DEVICE)`. Without this the first edit
   would break every GPU CatBoost-mode string, because the mode defaults are
   applied BEFORE validation and `check_scalars` would then refuse the value
   it had just supplied. The new predicate is the routing question
   `model.fit` versus `model.fit_multiclass` actually answers, and it is
   strictly better on the multiclass axis too: a CPU multiclass fit with a
   typed `random_strength` used to pass validation and raise mid-fit at
   `split.mojo`'s noise read, which the old comment recorded as a known open
   gap. It is now refused at parameter time with the cause named.

**Bits move** for one configuration: a CatBoost-mode parameter string on
`device=gpu` or `device=auto` that names no `random_strength` now gets 1.0
where it used to get 0.0. That is the defect, not a side effect of the fix,
and the fix is the correct behavior.

**Still not covered, and cannot be from a parameter string:** a SPARSE matrix
resolves onto `boosting_sparse.train_sparse`, which computes no scale, and a
string carries no data. The Python surface, which knows the dataset, refuses
it there.

---

## 3. Defect 3: the units are Cosine's and our default score is L2

### The arithmetic, DERIVED and exact

CatBoost's `Cosine` score for one node is `G / sqrt(H + l2)`. This package's
gain is `G**2 / (H + lambda)`. So

```
L2_gain = Cosine ** 2
```

exactly, for a single node, before the parent subtraction. `scoreStDev` is a
derivative RMS, which is dimensionally the numerator's per-row scale and
therefore pairs with `Cosine`. Adding the same `sigma` to a squared quantity
instead of to the quantity is weaker by the derivative of the square:

```
delta needed on L2 to match a sigma on Cosine   =   2 * Cosine * sigma
```

so the shortfall factor is `2 * Cosine = 2 * G / sqrt(H + lambda)`.

ORDER OF MAGNITUDE, with the assumption stated: for a level of `n` rows whose
candidate separates gradients of typical magnitude `d`, `G` is about `n * d`
over the separated side and `H` is about `n * h_row`, so
`Cosine ~ d * sqrt(n / h_row)`. On `dense_regression` at the standard tier
(128k rows in the level at the root, unit hessians for squared error, first
round derivative RMS near the label spread of about 1.8) that puts `Cosine`
in the hundreds, so the shortfall factor is in the hundreds. **UNMEASURED**;
the exact number depends on the split and on the round, and only a run gives
it. The order is what the argument needs.

### What that means, and it is worse inside our library than against CatBoost

Against CatBoost we are not wrong. CatBoost applies one `scoreStDev` to
whichever score function is configured, and this is exactly its behavior
under `score_function=L2`.

Inside our library the same value means two different strengths, because the
two shipped modes score differently: **the symmetric CatBoost mode scores
with Cosine and gets CatBoost's intended strength; a `lossguide` fit that
sets `random_strength` scores with L2 and gets a regularizer weaker by a
factor in the hundreds.** That is the argument for fixing it rather than
documenting it, and it is why the internal comment was not enough.

### What was done, and what was NOT, and why

DONE, and it is the "document where a USER will see it" option taken
properly rather than as a fallback:

- A full `random_strength` paragraph now exists in the estimator docstring in
  `python/mojotrees/sklearn.py`, next to `score_function`, stating the
  formula, the decay, where it is honored, and then in bold that a 1.0 here
  is not CatBoost's 1.0 unless `score_function="Cosine"` is also set, with
  the factor and the reason. Before this the estimator docstring described
  `random_strength_seed` and never described `random_strength`.
- `docs/DEVICE_SELECTION.md` says the value does not transfer between the two
  score functions, and its two stale rows claiming the device refuses Cosine
  and refuses `random_strength` outright are corrected to the narrowed
  refusals that actually exist.
- The internal comment in `tree_parameters_extra.mojo` now states the SQUARE
  and the size instead of "a different size", labelled derived.
- `params.mojo` says it at the `random_strength` key on the string surface,
  whose default score function is L2.

NOT DONE, deliberately, with the price stated as rule 6 requires:

- **Rescaling the noise per score function.** The clean form is to compare
  `sqrt(gain) + noise` instead of `gain + noise` on the L2 arm, which is
  exactly "noise it in Cosine units" because `sqrt(L2) = Cosine`, and which
  is inert with the noise off. It costs one `sqrt` per candidate on the noisy
  L2 path only. It has to land in SEVEN places at once or it reintroduces
  defect 2: `split.find_best_split`, `split.find_best_split_shared`, and the
  five device kernels. The blocker is not the arithmetic, it is the switch:
  the kernels learn "noisy" through an `Int32` launch argument whose value is
  computed at call sites in `gpu_resident_round.mojo`, which this lane may
  not write, so a default-off switch needs either that file or a widening of
  the `noisy` argument from a flag to a code inside `gpu_split_search.mojo`'s
  own launchers. Deliverable spec: widen `noisy` to `0 = off`, `1 = raw
  units`, `2 = score units`, resolve the code inside `_launch_search` and
  `_launch_oblivious_search` from `self.noise_stdev > 0` and the searcher's
  score function and one `getenv`, and leave every caller's Bool argument
  alone. Then the CPU sites take the same `getenv`.
- **Rescaling the STANDARD DEVIATION instead**, which would be one line and
  no kernel change, is the tempting version and it does not work. The
  conversion factor `2 * Cosine` is per candidate; the best round-level stand
  in needs `sqrt(sum of hessians)`, which the host-gradient arm has and the
  device-gradient arm would need a new device reduction for, and both round
  loops are in files this lane may not write.
- **Refusing `random_strength > 0` beside `score_function=L2`** was
  considered and rejected. It would kill the depth-wise GPU noise wiring
  landed on 2026-08-17 (`_enqueue_resident_split` and `_search_leaf_device`
  now pass node ids specifically so that combination works), and it refuses a
  pairing CatBoost itself allows.

None of the three is a bug fix. All three move bits on every noisy fit, so
all three take rule 3's real-data gate, and section 4 is the run that would
settle the first of them at the same time as the default question.

---

## 4. Defect 4: the experiment, written to be run without further design

### 4.0 What is being asked, and the rule it has to satisfy

The suspicion is that `random_strength = 1.0` is a pure accuracy cost on our
shapes: `dense_regression` is clean synthetic data with little to overfit, so
a regularizer that fires there buys nothing and pays.

**Two standing rules bind this and both must be answered in the write-up.**

1. `bench/results/LANE_RULES.md` rule 3, changed 2026-08-17: **our own
   accuracy is the gate and peers are a scoreboard**, judged against a
   recorded ABSOLUTE anchor per arm and per scenario, not against the
   previous run. No such anchor file exists in `bench/results/` yet. **This
   run must therefore write one**, or the change it proposes has nothing to
   be judged against later. Recording the anchor is part of the experiment,
   not an afterthought.
2. The standing mode rule: **CatBoost mode mirrors CatBoost.** Proposing that
   our CatBoost mode drop a CatBoost default needs an explicit argument, not
   a silent recommendation. Section 4.6 states what that argument would have
   to be and what result would and would not support it.

### 4.1 A PREREQUISITE, or every GPU cell of this run is silently skipped

`bench/real_data/run.py` carries a `DEVICE_PARAMETER_DIVERGENCE` entry that
declares `random_strength` a GPU divergence:

```python
{
    "parameter": "random_strength",
    "device": "gpu",
    "applies": lambda params: float(params.get("random_strength", 0.0)) > 0.0,
    ...
    "why": (
        "the per-split draw is staged on the device but its per-tree "
        "SCALE is computed only by the dense CPU round loops "
        "(boosting._round_random_score_scale), so `_parse_params` "
        "declares random_strength_ok as `device == cpu and not sparse`. "
        ...
```

Every clause of that `why` is false at this head: both arms of
`_train_gpu_rounds` compute the scale, and `_parse_params` declares
`random_strength_ok=True` for `fit` and `not d[].is_sparse` for
`train_dataset`. The entry does not fire for the built-in
`mojotrees_catboost_mode` arm, because that arm's overrides are merged
downstream and `arm["params"]` is empty when `backend_divergence` reads it --
which is why the GPU cell in `20260817T124906Z-postflip` ran at all. **It
DOES fire for any `--arms` module, which puts its parameters on the arm.** So
the sweep in this section would come back with every positive-value GPU cell
recorded as a skip, carrying a reason that is no longer true, and the run
would look like it answered the question.

**Delete that entry before running.** The `derivative_precision` entry beside
it is correct and stays: Float64 on the device is a datatype the hardware does
not have. That leaves the tuple with one element, which is the right shape for
one real divergence.

Two neighbours are stale for the same reason and matter less, but a reader of
this run will hit them: the header comment above the tuple
(`run.py:204-213`), which says `random_strength` "is refused on the GPU when
it is NAMED and declines to 0.0 when it arrives as a CatBoost-mode default",
and `frontier.py`'s Base A `device_reason`, which gives two GPU refusals that
have both been retired -- MVS (the postflip run trained a GPU cell with
`bootstrap_type=MVS`) and symmetric trees under Cosine.

### 4.2 The vehicle

An `--arms` module, because the sweep is over a parameter that is not a CLI
axis. Create `bench/real_data/random_strength_sweep.py` exposing `arms()`
returning the dicts below; `bench/real_data/frontier.py` is the template for
the dict shape (`_arm` at `frontier.py:740` lists every required key: `id`,
`scenario`, `tier`, `variant`, `engine`, `device`, `axis`, `axis_value`,
`params`, `dataset_params`, `env`, `repeats`, `skip`).

```
python bench/real_data/run.py \
    --arms random_strength_sweep \
    --repeats 3 --oracle-repeats 1 \
    --tag rstrength
```

Quiet box per `bench/results/PROFILE_PROTOCOL.md`, on mains power, and the
timing lock in `MACHINE_LOCK.md` held for the whole window. Accuracy is the
verdict here and accuracy does not drift with the box, but the run also
produces train seconds and those are only readable under the protocol.

### 4.3 The arms

Base for every arm below: `MOJOTREES_CATBOOST_MODE` exactly as
`scenarios.py` defines it -- `grow_policy=symmetrictree`, `max_depth=6`,
`num_leaves=64`, `min_data_in_leaf=1`, `min_child_hess=0.0`, `lambda_l1=0`,
`lambda_l2=3.0`, `score_function=cosine`, `leaf_estimation_iterations=1`,
`max_cat_to_onehot=2`, `bootstrap_type=MVS`, `subsample=0.8`, over
`BASE_PARAMS`. **One axis moves and it is `random_strength`.**

Two things about that base that will bite if they are left implicit.

**The CatBoost cell is not optional.** `mojotrees_catboost_mode` takes
CatBoost's resolved learning rate for the same cell and RAISES without a
CatBoost cell in the same run; `_engine_skip_reason` declares that dependency
as a skip precisely because a raising cell withholds the quality verdict for
the whole matrix. So every row must schedule the `catboost` engine too, which
the scoreboard column wants anyway.

**Pin the learning rate per row, and say that you did.** Taking CatBoost's
per-cell readback is what the shipped arm does, and it makes the decay term of
section 0 an unknown that differs by row. Read the rate off a first pass, then
pin it with `auto_learning_rate: False` and an explicit `learning_rate` for
the sweep proper, so that `random_strength` is the only thing moving and the
number of rounds the noise survives is a constant you can state. Record the
pinned value on every row; a pinned rate is a deviation from the shipped arm
and the write-up has to own it.

**Axis values: `0.0, 0.25, 0.5, 1.0, 2.0, 4.0`.** Six points. `0.0` is the
counterfactual and is the point the whole question turns on; `1.0` is the
shipped default; the two above it exist because a sweep that only goes down
from the default cannot distinguish "the default is too strong" from "the
regularizer does nothing at any strength", and those two conclusions have
different consequences.

**Scenarios, four rows. The first two are the shipped shapes and the last two
are the overfitting-pressure rows, and the run is rigged without them.**

| row | scenario | tier | variant | devices | rows x features | metric | why |
|---|---|---|---|---|---|---|---|
| R1 | `dense_regression` | `large` | synthetic | cpu, gpu | 1,000,000 x 100 | rmse | the decision row; clean data, little to regularize. This is where the suspicion says the noise is a pure cost |
| R2 | `dense_regression` | `large` | real | cpu, gpu | 463,715 x 90 (YearPredictionMSD) | rmse | real data. Real residual structure is not the synthetic generator's, and a regularizer can behave differently on it |
| R3 | `dense_regression` | `smoke` | synthetic | cpu, gpu | 5,000 x 20 | rmse | **genuine overfitting pressure.** 64 leaves and `min_data_in_leaf=1` on 4,000 training rows is about 62 rows a leaf. If the regularizer helps anywhere it helps here. Read the resolved `learning_rate` off the record rather than assuming the 0.5 the standard tier got: this arm takes CatBoost's per-cell readback, which is data dependent, and the rate sets how many rounds the noise survives |
| R4 | `ordered_boosting_small` | `standard` | synthetic | cpu | 50,000 | rmse | **the second pressure row, and an independent one.** The generator is "the dense recipe at a higher noise level" (`bench/real_data/README.md`), so the pressure comes from label noise rather than from row scarcity. CPU only: the scenario declares no accelerator |

R3 and R4 are the answer to "testing a regularizer only where it cannot help
is a rigged test". They are two DIFFERENT kinds of pressure, few rows per
leaf and noisy labels, and a regularizer that helps on neither is not being
denied its case.

**Tree count, and this is not optional.** Run each (row, value) pair at
`n_estimators` in `{100, 400}` on R3 and R4 only. Reason, DERIVED in section
0: the decay term puts the noise at half strength by round `log(n)/lr` and
near zero shortly after, so at 100 trees on a 4,000-row problem the
regularizer is alive for a small prefix of the run. A verdict taken only at
100 trees is a verdict about the prefix. On R1 and R2 keep 100 trees, which
is what every published number in this campaign uses; adding a tree axis
there multiplies the run and answers a different question.

**Cell count.** R1 and R2: 6 values x 2 devices x 1 tree count = 24 cells
(plus their CPU oracle twins, which the CPU device arm already is). R3: 6 x 2
x 2 = 24. R4: 6 x 1 x 2 = 12. Total 60 subject cells at 3 repeats. Add
LightGBM `stock+det` and CatBoost as-shipped once per row for the scoreboard
column, 8 more cells. R1 and R2 at the large tier dominate the wall clock;
`frontier.py`'s `MEASURED_TRAIN_S_AT_100` puts a symmetric GPU cell at 3.6 s
of training and `FIXED_PER_CELL_S` at 10 to 40 s of everything else, so
budget the large rows at roughly 30 to 60 minutes and the small rows at
minutes.

### 4.4 What each cell must record beyond the standard record

Three things the standard record does not carry, and the first is what makes
this run also settle Finding B:

1. **`device_agreement` on every (row, value) pair that has both devices.**
   Report `max |gpu - cpu|` per value. The prediction of the section 1 fix is
   that the symmetric arm's number lands at or below the noise-free depthwise
   arm's on the same shape, and that it no longer grows with
   `random_strength`. If it still grows with `random_strength` after the fix,
   there is a second noise divergence and this note's section 1 is wrong.
2. **The `random_strength = 0.0` cell's prediction digest against the same
   cell run with the parameter absent entirely.** They must be byte
   identical. If they are not, `random_strength = 0` is not an off switch and
   nothing else in the run can be read.
3. **Train seconds.** The noise costs a host draw and, on the device, a
   `random_score_plane` upload per level per tree.
   `docs/design/OBLIVIOUS_WAIT_CENSUS.md` counts six `_copy_noise` enqueues
   per tree for a depth-6 symmetric fit. If the accuracy answer comes back
   "no difference", the speed column decides it on its own.

### 4.5 The verdict rule, written before the data

Primary metric is the scenario's own (`rmse` on all four rows here). For each
row, take the median over the 3 repeats per cell, and compare each value's
median against the `random_strength = 0.0` cell of the SAME row, SAME device,
SAME tree count.

- **The regularizer HELPS on a row** if its best positive value beats `0.0`
  by more than the within-cell spread on that row, where the spread is the
  max-minus-min over the three repeats of the `0.0` cell. Spread rather than
  a fixed percentage, because `PROFILE_PROTOCOL.md`'s M0 vocabulary is
  resolved / consistent / indistinguishable and this is the accuracy form of
  the same test.
- **The regularizer COSTS on a row** if `0.0` beats every positive value by
  more than that spread.
- **INDISTINGUISHABLE** otherwise, and that is a real answer, not a failed
  run. It would mean the noise is doing nothing detectable on our shapes at
  any strength, which is itself grounds to look at the speed column.

### 4.6 What result would justify changing the CatBoost-mode default, and what
would not

The default is `random_strength = 1.0` under `grow_policy=symmetrictree`
because CatBoost's is, and the standing rule is that CatBoost mode mirrors
CatBoost. Beating that rule needs an argument about the RULE, not only a
number.

**Sufficient to change it:** the regularizer COSTS on R1 or R2 by the rule
above **and** is INDISTINGUISHABLE or COSTS on both pressure rows R3 and R4,
at every positive value. That combination says the mechanism does not help
our shapes even where it should, which is an argument that we are mirroring a
default whose premise does not hold here, and that is an argument about the
rule. The change to propose then is NOT dropping the parameter: it is
lowering the CatBoost-mode default to the best measured value and recording
the deviation from CatBoost in `scenarios.CATBOOST_DELIBERATE_DIVERGENCE`
with this run as its citation, the same way the harness already records
CatBoost deviations.

**Not sufficient, and each of these is a way this could be misread:**

- It costs on R1 and R2 but HELPS on R3 or R4. That is a regularizer working
  as designed. It means our published rows have no overfitting for it to
  suppress, which is a fact about the scenarios and not about the parameter,
  and the default should stand for the users whose data is not the dense
  generator.
- It costs by less than the spread. That is INDISTINGUISHABLE and changing a
  mirrored default on it would be changing it on noise.
- It costs only at 100 trees on R3 and R4 and not at 400. Section 0 says why:
  at 100 trees the decay may have switched it off for most of the run, and
  that is a finding about the SCHEDULE. The follow-up is a sweep over the
  decay, not a change to the strength.
- The speed column favors `0.0`. Speed is a separate case and it is a good
  one, but it is rule 2 (a trade behind a switch, A/B'd) and not rule 3, and
  it must be argued as speed rather than smuggled in as accuracy.

**Whatever the verdict, the run writes the anchor.** For each of the four
rows, record the shipped-default arm's median primary metric as the absolute
accuracy anchor for that arm and scenario, in a file under `bench/results/`,
with the run id, the commit, and the conditions line. Rule 3 requires an
anchor that moves only by a deliberate act visible in a diff, and there is
not one today.
