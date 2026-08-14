# Linear trees

Status: **implemented, not reachable.** The algorithm core lives in
`src/mojoboost/linear_tree.mojo`. Nothing public reaches it:
`params.mojo` refuses `linear_tree` and `linear_lambda` by name through
`tree_parameters_extra.check_extra_option_supported`, `lgbm_model_io.mojo`
refuses `is_linear=1` / `leaf_const` / `leaf_coeff` on the way in, and
`linear_tree.check_linear_tree_public` raises with the list of connections
that are still outstanding. `handoffs/remaining_03_linear_trees.md` carries
each of them as a ready-to-apply patch.

This document is the normative statement of what a linear leaf is in
mojoboost, what it refuses to do, and what the model format has to gain.

## What a linear tree is

An ordinary tree routes an example to a leaf and emits the constant stored
there. A linear tree routes identically and then emits an affine function of
a few raw features:

```
leaf(x) = intercept + sum_j coef[j] * (x[feature[j]] - center[j])
```

Routing is untouched: same bins, same thresholds, same missing directions,
same category sets. Only the leaf changes. That is why this is a leaf
representation and not a boosting mode, and why it composes with GBDT, GOSS,
DART, and random-forest boosting without any of them knowing about it.

## The model representation

### Why `Tree` cannot hold it

`tree.Tree` stores one `Float64` per node in `value`. Every other array on
it (`feature`, `threshold_bin`, `left`, `right`, `default_left`,
`missing_bin`, `cat_offset`, `count`) is consumed by routing or by
attribution. There is no field a coefficient vector can live in, and no
honest encoding of one in the fields that exist. Any claim that today's
`Tree` "already supports" linear leaves is false.

### The sidecar

Widening `Tree` is the expensive answer. It is constructed positionally in
seven places (`tree.grow_tree`, `tree_sparse`, the two `train_gpu` growers,
the distributed merge, `serialize._read_trees`, and the LightGBM importer)
and read by `contrib.mojo`, `model_dump.mojo`, `gpu_predict.mojo`, and the C
ABI. All of that would change for a feature most models never use.

The representation mojoboost defines instead is a **sidecar** keyed by
`(tree index, leaf ordinal)`:

| Type | Holds |
| --- | --- |
| `LinearLeaf` | `intercept`, `feature[]`, `coef[]`, `center[]` |
| `LinearTree` | one `LinearLeaf` per leaf ordinal of one tree |
| `LinearEnsemble` | one `LinearTree` per tree of a `Booster`, plus `n_features` |

Leaf **ordinals**, not node ids. `Tree.leaf_ordinals()` numbers a tree's
leaves in node-array order; `tree.mojo` documents that the numbering is
fixed once a tree is grown and survives save and load unchanged, because
serialization writes nodes in array order. Node ids number internal nodes
and leaves together and are explicitly an implementation detail.

So the whole model-representation extension is:

1. one optional field on `Booster` and on `MulticlassBooster`
   (`var linear: LinearEnsemble`, empty meaning constant leaves), and
2. one optional serialized section.

`Tree` itself does not change, and every consumer that does not know about
linear leaves keeps compiling.

### The constant fallback is exact

`LinearLeaf.intercept` is, by construction, the value the ordinary grower
already wrote into `Tree.value` for that leaf: `refit_linear_tree` takes it
rather than recomputing it. Combined with centering, that means a consumer
which ignores the sidecar predicts the linear leaf evaluated at its own
training centroid. That is a well-defined, less accurate, constant-leaf
model rather than garbage.

`linear_leaf_reduces_to_constant` is that property as a predicate. It exists
so that `contrib.mojo`, `gpu_predict.mojo`, and the LightGBM exporter can
say what they would be predicting instead of the real model. **They are
still expected to refuse**, not to degrade silently; the property makes the
choice informed rather than accidental.

## The algorithm

### Where it runs

Growth is unchanged. Splits are found from gradient histograms exactly as
today, the tree is grown to completion with constant leaves, and then
`refit_linear_tree` replaces each leaf's constant with an affine function
fitted on the rows that reached it.

This is an intentional difference from LightGBM; see "LightGBM differences"
below.

### The solve

For one leaf, with rows `R`, per-row gradient `g_r` and hessian `h_r`, the
intercept `b` already fixed by the grower, and candidate features `S`:

```
center[j] = (sum_r h_r x_rj) / (sum_r h_r)
xt_rj     = x_rj - center[j]
A[j][k]   = sum_r h_r xt_rj xt_rk        (m by m, symmetric PSD)
q[j]      = sum_r g_r xt_rj

(A + linear_lambda * I) c = -q
```

The leaf's second-order objective is

```
F(b, c) = sum_r [ g_r f_r + h_r f_r^2 / 2 ]
        + reg(b) + linear_lambda ||c||^2 / 2 ,     f_r = b + c . xt_r
```

Centering makes `sum_r h_r xt_rj = 0` for every `j`, so the cross term
`b * sum_r h_r xt_rj` vanishes and

```
dF/dc_j = q[j] + (A c)[j] + linear_lambda * c[j]
```

does not mention `b` at all. Three consequences, all load-bearing:

1. **The coefficient solve is correct whatever the intercept is.** The
   grower's intercept is not the plain Newton step in general: it has been
   through `max_delta_step`, `path_smooth`, and the monotone clamp. Taking
   it as given costs nothing.
2. **`linear_lambda -> infinity` gives back today's model exactly.** `c` goes
   to zero and the leaf is bit-for-bit the constant leaf, because the
   intercept was never recomputed.
3. **The improvement is closed-form.** The training-objective reduction over
   the same leaf with `c = 0` is

   ```
   improvement = -( q . c + c^T A c / 2 + linear_lambda ||c||^2 / 2 )
   ```

   which equals `c^T (A + linear_lambda I) c / 2 >= 0` in exact arithmetic.
   The implementation computes it from the returned `c` rather than from that
   identity, so a solve that lost too much to roundoff to be worth keeping is
   caught and the leaf falls back to its constant.

`linear_lambda` regularizes the coefficients only. The intercept keeps the
`lambda_l1` and `lambda_l2` the grower already applied through
`tree._leaf_value`; those are separable terms in `F` and do not disturb the
decoupling.

### Leaf feature selection

`eligible_leaf_features` starts from the leaf's **branch set** (the features
split on between the root and it, LightGBM's rule) and removes, in order:

| Removed | Why |
| --- | --- |
| categorical features | a coefficient on an integer category code is meaningless; bins are set members, not magnitudes |
| features with a non-finite raw value on any row of the leaf | see "Missing values" |
| features whose in-leaf hessian-weighted variance is at or below `1e-12` of their scale | a constant column is a zero row and column of `A` |
| features beyond `max_leaf_features` | mojoboost extension, off by default |
| features the leaf has too few rows to afford (`min_data_per_linear_feature`) | mojoboost extension, off by default |

The last two trim from the bottom of one ranking: the univariate objective
reduction `q[j]^2 / (A[j][j] + linear_lambda)`, descending, ties broken by
ascending feature index. The ranking is a function of the statistics alone,
so two runs on the same data pick the same features whatever order the rows
were accumulated in.

The surviving set is ascending by feature index, which is what serialization
and inspection read.

### Numerical stability

- **Two passes, not one.** Weighted means first, then centered second
  moments. The one-pass `sum(xy) - sum(x)sum(y)/n` identity is algebraically
  the same and numerically much worse on a feature whose spread is small
  relative to its magnitude, which is the common case for a feature the tree
  has already split on twice.
- **`A` is filled on and above the diagonal and mirrored**, so the matrix is
  exactly symmetric rather than symmetric up to summation order. Cholesky
  reads the lower triangle; an asymmetry of one ulp would make the
  factorization describe a different matrix than the improvement check
  scores.
- **Equilibrated Cholesky.** `M = A + linear_lambda I` is scaled
  symmetrically by `D = diag(1/sqrt(M_jj))` so every diagonal is 1 before the
  factorization runs. The scaling is exact and is undone on the way out, and
  it is applied *after* the ridge term, so `linear_lambda` keeps its meaning
  in the feature's own scale rather than in a normalized one.
- **`ridge_eps`** adds a relative jitter `eps * trace(A) / m` to the diagonal
  (default `1e-10`), so an exactly singular `A` with `linear_lambda = 0` still
  factorizes. It is a numerical device and is deliberately left out of the
  improvement score.
- **Failure drops features, it does not return a plausible number.** A
  nonpositive diagonal, a pivot that falls to `1e-11` of its equilibrated
  start, or a non-finite coefficient removes one feature and restarts. Every
  retry strictly shrinks the active set, so the loop runs at most `m` times.
  If nothing survives, the leaf stays constant.
- **`max_linear_deviation`** (off by default) discards a fit whose linear
  part exceeds a bound in absolute value on any row of the leaf. It is
  discarded whole rather than clipped, because clipping would make the stored
  coefficients describe a function the leaf does not evaluate. This is the
  only check that costs a second pass over the rows.

Every failure path records a reason (`LINEAR_FIT_*`), and
`linear_fit_summary` counts them across the ensemble. A run whose leaves are
mostly `rank_deficient` or `no_eligible_features` has a data problem
(constant columns, missing values on the split features) rather than a
modelling one, and the difference is invisible from the fitted model alone.

## Restrictions

### Missing values

Two rules, and they are not the same rule.

**Training.** A feature with a non-finite value on any row reaching the leaf
is not eligible for that leaf at all. Imputing during the fit would put a
made-up number into `A`; excluding only the offending rows would fit
different features on different row sets, which is not a least-squares
problem.

The test is on the **raw** values, not on `BinnedMatrix.missing_bin`.
`use_missing=False` bins `NaN` as `0.0` and reserves no bin, so the binned
matrix cannot answer the question.

**Prediction.** An eligible feature can still be handed a `NaN` or an
infinity at prediction time, on a row unlike anything in training. Such a
value is replaced by that leaf's `center[j]`, so the term contributes exactly
zero and the leaf falls back to its intercept along that one axis. This is
why `center` is stored rather than folded into the intercept.

Routing is untouched by both. A missing value still follows `default_left`
to a leaf exactly as in a constant-leaf tree.

Both rules are **mojoboost-defined**. LightGBM's behaviour here has not been
read off its source; see "Parity-unverified" below.

### Categorical features

Categorical features route normally and are never linear terms. A model may
mix them freely with linear leaves. LightGBM excludes them too.

### Monotonic constraints

**Refused together.** `check_monotone_compatible` raises when any constraint
sign is nonzero.

This is not conservatism. `monotone.mojo` proves monotonicity by giving every
node a closed output interval and clamping the leaf's output into it, so that
two examples separated at a node splitting on the constrained feature satisfy
`pred(x) <= mid <= pred(x')`. A leaf whose output is an affine function of
`x` has no single output to clamp, and requiring the *function* to lie in a
bounded interval for all `x` forces every coefficient to zero, since a
nonconstant affine function on an unbounded domain is unbounded. The
mechanism as written admits only constant leaves.

#### The refinement that would work

A leaf's region **is** an axis-aligned box in raw feature space: it is the
intersection of its ancestors' bin thresholds, and a bin threshold maps back
through the bin edges to a half-line in the raw feature. An affine function
on a box attains its extrema at a corner, so the interval test becomes a test
on two corner values rather than on one constant. Concretely, a linear leaf
under constraints is admissible when, for a leaf with box
`[lo_f, hi_f]` per feature and monotone signs `s_f`:

1. `s_f * coef_f >= 0` for every constrained feature `f` that is a linear
   term (the leaf is itself monotone in `f` in the required direction), and
2. `min_corner >= bounds.lower` and `max_corner <= bounds.upper`, where
   `min_corner` and `max_corner` are the leaf function's values at the box
   corners chosen per feature by the sign of `coef_f`, and `bounds` is the
   node's monotone interval from `monotone.mojo`.

Condition 2 requires the box to be **bounded** in every feature that is a
linear term with a nonzero coefficient. A feature never split on above the
leaf, or split on only on the outer side, has an unbounded box side, so it
must be excluded from the leaf's linear terms whenever the corresponding
bound is finite.

What that needs and does not have: bin edges at the leaf (the grower is
handed a `BinnedMatrix`, which carries none), per-feature box bounds carried
down the frontier alongside `OutputBounds`, and a rejection path inside
`select_leaf_features`. None of it is built. Until it is, the combination
raises.

### Objectives

| Objective | Linear leaves |
| --- | --- |
| `SQUARED_ERROR`, `BINARY_LOGISTIC`, `POISSON`, `HUBER`, `GAMMA`, `TWEEDIE`, `FAIR`, `CROSS_ENTROPY` | allowed |
| `CUSTOM` | allowed; `check_custom_grad_hess` already guarantees finite, nonnegative hessians |
| `QUANTILE`, `L1`, `MAPE` | **refused** |
| `LAMBDARANK` | **refused** |

`QUANTILE`, `L1`, and `MAPE` renew their leaves after growth
(`boosting.objective_renews_leaves`), replacing the Newton value with a
weighted residual percentile. That percentile is not the minimizer of any
quadratic, so it does not decouple from the coefficients the way the
intercept above does, and fitting the two independently would give a leaf
that is neither the renewed constant nor the linear least-squares fit.

`LAMBDARANK` produces gradients from query-group pairs. The leaf solve is
well-defined on them, but a per-leaf regression on pairwise gradients is not
a combination LightGBM documents or mojoboost has checked. Refused rather
than guessed at.

The four codes above are **mirrored** in `linear_tree.mojo` as `_QUANTILE`,
`_L1`, `_LAMBDARANK`, and `_MAPE` rather than imported. `Booster` is what
holds the sidecar, so `boosting.mojo` will import `linear_tree.mojo`, and
`src/mojoboost` has no mutual imports anywhere; the reverse edge would be
the first one. The codes are part of a stable public numbering (serialized
in every model file, crossing the C ABI), so they do not move, and the
handoff asks the boosting lane for a compile-time cross-check anyway. This
is the same trade `model_dump.mojo` makes with `categorical._MAX_CATEGORY`.

## Multiclass

Nothing special. `LinearEnsemble.trees` is parallel to `Booster.trees`, and
`MulticlassBooster.trees` is round-major, so class `k`'s tree in iteration
`i` is at `i * n_classes + k` in both. Each class's leaves are fitted from
that class's own gradients and hessians, and each class's leaf may end up
with a different feature set.

The per-class **raw** score is piecewise affine in the raw features. The
softmax probabilities are **not**, for the same reason they are not monotone
under monotone constraints: each depends on every class's raw score.

## Continued training

Two obligations.

1. **The resume pass must go through the linear leaves.**
   `linear_tree.resume_raw_scores` is that pass. `boosting.train_more`
   recomputes the raw scores from the model on every call; resuming from the
   constant fallback would restart boosting from a model the ensemble does
   not actually predict, and every subsequent tree would be fitted to the
   wrong residual.
2. **Continued training needs the raw feature matrix.** `train_more` today
   takes a `BinnedMatrix`. A linear continuation needs the raw values both
   for the resume pass and for the new trees' leaf fits.

`check_continuation_compatible` enforces what can be checked from the two
ensembles: the sidecar must be as long as the tree list, its feature count
must match, and an ensemble cannot change halfway from constant leaves to
linear ones or back. A `Booster` holds one sidecar for all of its trees.

## Per-tree weights (DART)

`alternate_boosting.fold_weights_into_trees` multiplies each tree's node
values by that tree's drop weight, so an ensemble with per-tree weights
becomes one a single-shrinkage `Booster` represents exactly.
`LinearEnsemble.scale_all` is its other half: it multiplies intercepts and
coefficients by the same weights. Centres are untouched, being feature-space
quantities rather than output-space ones.

Folding the constants without the coefficients would leave a leaf whose
affine function no longer passes through the value its tree carries, so the
sidecar and the trees would describe two different models and
`linear_leaf_reduces_to_constant` would start returning False. The two calls
belong together, with the same weight vector, in the same place.

## Shrinkage

`Booster` applies `learning_rate` at prediction time
(`s += learning_rate * tree.predict_row(...)`) rather than baking it into
leaf values. So leaf values are unshrunk Newton steps, coefficients are
unshrunk too, and the caller scales the whole affine function by one factor.
Nothing in the sidecar needs to know the learning rate.

## Serialization

### Version

A model carrying linear leaves must be written at **model format version 5**.
`serialize._VERSION` is `v4` today (split gains, a cover presence flag, and
optional feature names); the `linear` section is what makes a file v5.

The bump is conditional: `linear_model_format_version` returns 5 only when
the sidecar is active, so a constant-leaf model written by a build that knows
about linear trees stays v4 and stays readable by a build that does not. The
version is a statement about the file, not about the binary that wrote it.

v4 was claimed by another lane while this module was written. That is exactly
the hazard the conditional bump contains: if `_VERSION` moves again before
this is wired, `LINEAR_MODEL_FORMAT_VERSION` moves with it and nothing else
changes.

### The section

Written after the trees, in `serialize.mojo`'s own encoding: whitespace
separated tokens, floats as their raw IEEE-754 bit patterns in decimal.

```
linear <revision> <n_trees> <n_features>
ltree <n_leaves> <n_linear_leaves>                        (once per tree)
<ordinal> <m> <feature x m> <coef x m> <center x m>       (per linear leaf)
```

- `revision` is `1`. It is the section's own number, so a later change to
  what a leaf stores (a per-leaf scale, say) does not need another
  model-format bump.
- **Constant leaves are not written.** A tree with three constant leaves and
  one linear leaf writes `ltree 4 1` and a single leaf record.
- **A model with no linear leaves writes no section at all**, so a v5 file
  for a constant-leaf model is byte for byte what v4 wrote.
- **Intercepts are not stored.** A leaf's intercept is `Tree.value` at that
  leaf, which the tree section already carries and which
  `refit_linear_tree` guarantees it equals. Reading it from the tree rather
  than writing it twice removes the only way the two could disagree.

`linear_section_text` writes it and `read_linear_section` reads it, both in
`linear_tree.mojo`, so the change in `serialize.mojo` is one call in each
direction rather than a second parser.

### Validation on load

`read_linear_section` validates everything against the trees it is given
before any of it is used: the tree count, the per-tree leaf count, each
ordinal's range and uniqueness, each feature index against the mapper's
width, the ascending-and-distinct feature order within a leaf, and the
finiteness of every stored float. A section that does not describe its trees
is a corrupt file, not a model to predict from.

### Backward and forward reading

v1 through v4 files load exactly as they do today and carry no section. A v5
reader that meets an unknown section revision raises rather than skipping,
because a leaf it cannot reconstruct is a leaf it cannot predict from.

## Interoperability

`lgbm_model_io.mojo` refuses LightGBM linear trees on the way in today
(`is_linear=1`, `leaf_const`, `leaf_coeff`) and that does not change here:
mojoboost's leaf function is centered and LightGBM's is not, so an imported
`leaf_coeff` would need `leaf_const` reinterpreted against a centroid the
file does not carry. Export is refused for the mirror-image reason.

Making either direction work is a separate piece of work, not a wiring
change. The conversion is arithmetic (`intercept = leaf_const + sum_j
coef_j * center_j` in one direction, `center = 0` in the other) but it moves
the missing-value rule too, since a LightGBM linear leaf substitutes nothing
for a missing value.

## Parameters

| Parameter | LightGBM name | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | `linear_tree` | `false` | |
| `linear_lambda` | `linear_lambda` | `0.0` | L2 on the coefficients only |
| `max_leaf_features` | none | `-1` | mojoboost extension; `-1` is no cap |
| `min_data_per_linear_feature` | none | `0` | mojoboost extension; `0` is off |
| `ridge_eps` | none | `1e-10` | mojoboost extension; relative diagonal jitter |
| `max_linear_deviation` | none | `0.0` | mojoboost extension; `0.0` is off |

All four extensions default to inactive, so a default `LinearParams` is the
LightGBM-shaped configuration.

## LightGBM differences

- **Split gains ignore the linear fit.** LightGBM evaluates candidate splits
  under the per-leaf regression, so its trees differ in *shape* from
  constant-leaf ones. mojoboost grows the constant-leaf tree and refits its
  leaves. The fitted leaves are optimal for the tree that was grown; the tree
  is not the one LightGBM would have grown. This is the largest and most
  visible difference and it is not a rounding matter.
  `accumulate_leaf_stats` and `solve_leaf_coefficients` are the pieces a
  linear-aware split search would reuse; the cost is one solve per candidate
  rather than one per leaf.
- **Centered parameterization.** mojoboost stores `center[]` and expresses
  the leaf as `intercept + sum coef * (x - center)`. LightGBM stores
  `leaf_const` and `leaf_coeff` in raw coordinates. The two describe the same
  function family; the difference is why import and export are refused
  (above) and why the constant fallback is exact here and would not be there.
- **Regularization placement.** `linear_lambda` is applied to the
  coefficients and not to the intercept.
- **Missing values.** Both the eligibility rule and the centre substitution
  are mojoboost-defined.
- **`ridge_eps`, `max_leaf_features`, `min_data_per_linear_feature`,
  `max_linear_deviation`** have no LightGBM counterpart.

## Parity-unverified

Three statements above are derived from LightGBM's documented behaviour and
from first principles, not from a line-by-line reading of its source. They
must be checked against a LightGBM run (`bench/compare_*.py` is the existing
pattern) before any parity row moves off `deferred`:

1. **Whether LightGBM's `linear_lambda` penalizes its intercept row.** It
   adds the value to the diagonal of an `(m+1)`-square system; whether the
   intercept position is included has not been confirmed. mojoboost does not
   penalize it.
2. **What LightGBM does with missing values in a linear leaf**, both at fit
   time and at prediction time.
3. **Whether LightGBM refuses `linear_tree` with monotone constraints,**
   with categorical features as linear terms, or with the leaf-renewing
   objectives, and what it does when it does not refuse.

Until those are checked, the `linear_tree` and `linear_lambda` rows in
`docs/LIGHTGBM_PARITY.md` stay `deferred` regardless of how much of this
document is implemented.

## Later validation

Every command below is **UNRUN**. Nothing in this round was built, run, or
tested.

- `pixi run mojo build src/mojoboost/linear_tree.mojo` -- compiles the module
  alone. **UNRUN.**
- A focused test that a leaf fitted with `linear_lambda` large reproduces
  `Tree.value` to the last bit. **UNRUN.**
- A focused test that `linear_section_text` followed by
  `read_linear_section` round-trips a two-tree sidecar exactly. **UNRUN.**
- A focused test that `check_monotone_compatible`,
  `check_objective_compatible`, and `check_continuation_compatible` each
  raise on the case they name. **UNRUN.**
- `python3 tools/check_parity.py` after any parity row moves. **UNRUN.**
