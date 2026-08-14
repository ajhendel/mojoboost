# mojoboost

[![CI](https://github.com/ajhendel/mojoboost/actions/workflows/ci.yml/badge.svg)](https://github.com/ajhendel/mojoboost/actions/workflows/ci.yml)

Gradient boosted decision trees in [Mojo](https://www.modular.com/mojo).

mojoboost is a from-scratch GBDT library in the LightGBM family. It uses
histogram-based split finding and leaf-wise (best-first) tree growth. Its
benchmark configuration aligns important parameters with LightGBM for
reproducible comparisons.

## Why Mojo

LightGBM and XGBoost are excellent, mature C++ libraries. The bet here is
that Mojo enables a simpler codebase that is at least as fast, because the
hot loop of gradient boosting (histogram accumulation over binned features)
is exactly what Mojo is built for. Explicit SIMD with per-chip vector widths,
compile-time specialization of the training loop for bin count and dtype
(no virtual dispatch or runtime branching), precise control of memory layout
for cache tiling, and structured parallelism, all in one language that also
targets GPUs from the same source.

## Status

Early development. Training works end to end. What works today, each piece
with tests

- quantile (equal-frequency) feature binning into `uint8` bins, LightGBM
  style, with stored edges so trained models predict on raw, unseen data
- histogram accumulation with the sibling subtraction trick
- best-split search with the standard second-order gain formula,
  `min_data_in_leaf` and hessian constraints
- leaf-wise (best-first) tree growth with `num_leaves` cap and Newton-step
  leaf values
- `max_depth` depth limiting that preserves leaf-wise growth: trees stay
  unbalanced and the best-gain leaf still splits first, leaves at the limit
  simply stop offering splits; enforced identically on the CPU and GPU
  trainers
- L1 and L2 split regularization (`lambda_l1`, `lambda_l2`) with LightGBM's
  soft-thresholding of gradient sums, applied consistently to split gains
  and leaf values on both the CPU and GPU trainers
- feature interaction constraints (`interaction_constraints`), LightGBM's
  per-branch allowed-feature rule, enforced identically on the CPU and GPU
  trainers and checked structurally on every root-to-leaf path
- objectives: squared error, binary logistic, poisson, huber, quantile,
  L1 (mean absolute error), and multiclass softmax; quantile and L1
  renew leaf values with residual percentiles the way LightGBM does, and
  `alpha` follows LightGBM's meaning for huber and quantile
- custom objectives (`train_custom`) through a compile-time callable, so
  the boosting loop stays direct-call; validated gradients and hessians,
  framework-applied sample weights, and a Python callback path with a
  measured per-round cost (see [Custom objectives](#custom-objectives))
- learning to rank with LambdaRank (`train_ranker`, `fit_ranker`,
  `MojoBoostRanker`): LightGBM's NDCG-weighted pairwise lambdas and
  hessians, `lambdarank_truncation_level`, `sigmoid`, and `lambdarank_norm`;
  query boundaries from LightGBM's `group` array or from a query id column,
  which is rejected when a query's rows are not contiguous; NDCG at any
  cutoff; query-aware validation and early stopping; and bagging that
  samples whole queries rather than rows
- validation-set early stopping with `min_delta` for every objective,
  truncating to the best round
- evaluation metrics: RMSE, log loss, accuracy, and ROC AUC with
  sklearn-matching tie handling
- sample weights for every objective, LightGBM semantics (weighted
  gradients, hessians, and base scores; zero-weight rows are ignored)
- seeded row bagging (`bagging_fraction`, `bagging_freq`, `bagging_seed`)
  from a counter-based RNG, so a bag depends only on its seed and index and
  the CPU and GPU trainers grow every round on identical rows
- Gradient-based One-Side Sampling (`boosting="goss"`, `top_rate`,
  `other_rate`), LightGBM's rule down to the `|grad * hess|` importance and
  the warmup rounds: high-gradient rows kept, low-gradient rows sampled and
  compensated, on either backend and for every objective including
  multiclass (see [GOSS](#gradient-based-one-side-sampling))
- seeded feature subsampling (`feature_fraction`, `feature_fraction_bynode`,
  `feature_fraction_seed`), per tree and optionally per node, without
  replacement and drawn from a counter-based RNG, with excluded features
  skipped in histogram accumulation as well as split search on both the CPU
  and GPU trainers
- feature importance, split counts and total gain, matching LightGBM's
  two importance types
- SIMD histogram kernels (pointer-based scatter accumulation, vectorized
  sibling subtraction and split scans)
- multicore CPU histogram accumulation across independent features
- experimental portable GPU histogram accumulation that scales past one
  threadgroup per feature: a 2D grid of (feature, row tile) threadgroups,
  a shared-memory partial histogram per threadgroup, and a choice of two
  ways to combine them, a deterministic reduction kernel or the original
  global integer atomics, tiled from the device's own reported capabilities
  (see [GPU histogram scaling](#gpu-histogram-scaling)). Tested for
  correctness on Apple Metal only. No NVIDIA or AMD device has ever run
  this code.
  [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md) holds the reproducible
  procedure, the CI workflow, and the record of what has and has not been
  executed; `tests/test_gpu_portability.mojo` pins the launch limits and
  numeric bounds that CUDA and HIP impose, without needing either device
- one device vocabulary across Mojo and Python (`cpu`, `gpu`, `auto`),
  where `gpu` raises rather than falling back silently (see
  [Device selection](#device-selection))
- model serialization: `save_model`/`load_model` with a versioned text
  format that stores floats as raw bit patterns, so loaded models predict
  bit-exactly; multiclass models via `save_multiclass_model` and
  `load_multiclass_model`
- scikit-learn style Python API (`MojoBoostRegressor`,
  `MojoBoostClassifier`, `MojoBoostRanker`) backed by a CPython extension
  module built from the same Mojo code, with sample weights and exact
  save/load
- LightGBM-native parameter names as the canonical Python vocabulary, with
  the aliases used by LightGBM's scikit-learn estimators accepted for easy
  migration; conflicting aliases raise instead of silently taking precedence
- multiclass end to end on raw data: `fit_multiclass` returns a
  `MulticlassModel` with `predict_proba` and `predict_class`

```mojo
from mojoboost import BINARY_LOGISTIC, BoosterParams, TreeParams, fit

def main() raises:
    # features is column-major: features[f * n_rows + r]
    var params = BoosterParams(100, 0.1, TreeParams.default())
    var model = fit(features, n_rows, n_features, labels,
                    BINARY_LOGISTIC, params)
    var p = model.predict(row)   # raw feature values in, probability out
```

Lower-level entry points `train`, `train_with_valid`, and
`train_multiclass` operate on pre-binned matrices.

## Python API

Build the extension once with `bindings/build.sh`, then use the
scikit-learn style estimators in `python/mojoboost`:

```python
from mojoboost import MojoBoostRegressor, MojoBoostClassifier

model = MojoBoostRegressor(num_leaves=31, n_estimators=100).fit(X, y)
pred = model.predict(X)          # numpy in/out when numpy is available
model.save("model.mbst")
model = MojoBoostRegressor.load("model.mbst")

clf = MojoBoostClassifier().fit(X, labels)   # binary or multiclass by labels
proba = clf.predict_proba(X)
```

`fit` accepts `sample_weight`, hyperparameters mirror the Mojo defaults,
and saved models round-trip bit-exactly. numpy is optional; plain Python
sequences work without it. The regressor takes LightGBM objective names
(`objective="regression"`, `"huber"`, `"quantile"`, or `"mae"`) with
`alpha` as the quantile level or huber transition point, or a callable for
a custom objective (see [Custom objectives](#custom-objectives)). Both
estimators take `device` and record the backend that ran on `device_`; see
[Device selection](#device-selection).

`pixi run test-wheel` builds a self-contained wheel (`pixi run build-wheel`)
and validates it in a clean venv. The wheel bundles the Mojo runtime
dylibs the extension links (delocate-style, with an `@loader_path` rpath),
so installing it requires no Mojo or MAX toolchain. Wheels currently
target macOS on Apple silicon; Linux wheels need a manylinux build.

## Device selection

The same three values everywhere, in Mojo (`fit(..., device=CPU_DEVICE)`)
and in Python (`MojoBoostRegressor(device="cpu")`):

| `device` | Behavior |
|---|---|
| `cpu` | The default and the dependable backend: Float64, every objective, every entry point |
| `gpu` | Device-resident tree growth. Raises when no accelerator is present, or on a workload the GPU path does not cover, instead of falling back |
| `auto` | The GPU when it is available, covers the workload, and the size heuristic selects it; the CPU otherwise |

The GPU trainers cover every objective that shares the per-row
gradient/hessian interface. `train_gpu` handles single-output training
(squared error, binary logistic, poisson, huber, quantile, L1),
`train_custom_gpu` a caller-supplied gradient callback, and
`train_multiclass_gpu` softmax, growing one tree per class per round
through one device-resident builder. Quantile and L1 renew their leaf
values on the host after the tree is grown, exactly as on the CPU.

The `device` routing above is narrower than the trainers underneath it:
`fit_multiclass` still resolves to the CPU, so reaching GPU multiclass
means calling `train_multiclass_gpu` directly. GOSS reaches both trainers
and every objective they cover: the sample is chosen on the host from
Float64 gradients and handed to the device as the tree's row list, so the
two backends sample identically.

`auto`'s size heuristic ships disabled, so `auto` currently always
resolves to the CPU. No benchmark on any device has established a
workload size where end-to-end GPU training beats the CPU trainer, and
shipping a crossover threshold before then would be a performance claim
with nothing behind it. `MOJOBOOST_AUTO_MIN_CELLS` enables it as an
integer cell count (`n_rows * n_features`) at or above which `auto`
chooses the GPU, which is the knob for running the crossover benchmark
that would justify a default. `MOJOBOOST_DISABLE_GPU=1` makes the library
report no accelerator, so `gpu` raises and `auto` chooses the CPU on a
machine that has one; it pins a mixed fleet to the CPU and exercises the
unavailable-GPU path in tests.

Fitted Python estimators record the backend that actually ran on
`device_`. The device is a training choice rather than part of the model,
so the serialization format is unchanged and a loaded estimator carries no
`device_`.

LightGBM difference: LightGBM spells this `device_type` with `cpu`, `gpu`,
and `cuda`, and has no `auto`. mojoboost has one portable GPU backend
rather than separate OpenCL and CUDA ones, so `gpu` covers every supported
accelerator, and `auto` is an addition.

### GPU histogram scaling

Histogram accumulation launches a 2D grid: `grid.x` is the active feature,
`grid.y` a tile of rows. A device therefore gets `n_active * n_tiles`
threadgroups rather than one per feature, which is what lets a wide GPU stay
busy on a dataset with few features. Each threadgroup accumulates a partial
histogram for its (feature, row tile) in shared memory, filtering rows by
the target leaf id.

Two strategies combine those partials:

| Strategy | How partials combine | Cost |
|---|---|---|
| `tiled` | each partial written to its own global slot, then a reduction kernel sums slots in ascending tile order | one extra kernel launch, one partial buffer |
| `atomic` | each partial folded into the output with global integer atomics | contention on hot bins, one output memset |

`atomic` is the original implementation, kept as the fallback for hardware
where the tiled path has not been validated. Both accumulate the same exact
fixed-point Int32 values and integer addition is associative, so the two
produce **bit-identical** histograms; `tests/test_gpu_strategies.mojo`
asserts that directly rather than to a tolerance, on full-dataset builds,
leaf-filtered builds, and under feature subsampling.

The launch geometry is derived at runtime from the device's own reported
capabilities (`src/mojoboost/gpu_tiling.mojo`), not fixed at compile time,
because the same source targets Metal, CUDA, and HIP across multiprocessor
counts spanning more than an order of magnitude. Threads per group come
from the device maximum rounded to a warp; row tiles per feature are the
tightest of three bounds: enough threadgroups to fill the multiprocessors,
enough rows per tile to pay for writing the partial, and a memory budget for
the partial buffer. A capability a backend does not implement falls back to a
conservative portable constant rather than failing, which matters because
Metal rejects several of the CUDA-derived attribute queries outright.

The kernels themselves use only what all three backends provide: shared
memory, `barrier()`, integer atomics on shared memory, and plain global
loads and stores. No warp shuffles, no float atomics (Metal has none), no
vendor intrinsics, and no per-architecture code paths. Only the tiling
numbers differ per device.

Three environment variables override the policy, for benchmarking and for
tests that must force one path, matching the `MOJOBOOST_` contract in
`parallel.mojo`:

| Variable | Effect |
|---|---|
| `MOJOBOOST_GPU_HIST_STRATEGY` | `atomic` or `tiled` forces that strategy; `auto` or unset lets the policy decide |
| `MOJOBOOST_GPU_ROW_TILE` | rows per tile, still clamped to the memory budget |
| `MOJOBOOST_GPU_BLOCK_THREADS` | threads per threadgroup, still clamped to the device maximum and rounded to a warp |

These are tuning and test knobs rather than model parameters: they change
how a histogram is computed, never what it equals, so they are deliberately
absent from the Python API and from the serialization format.

`pixi run bench-hist-scaling` reports the two strategies side by side with
kernel time separated from conversion, upload, download, and setup time. Every
measurement taken so far is Apple Metal; see
[docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md) for what has and has not
been run.

LightGBM difference: LightGBM's GPU histogram builder is written against
specific vendor toolchains and tunes itself with vendor-specific constants.
mojoboost has one kernel source for every backend and moves all
device-specific choice into the tiling policy, which is why the strategy and
tile size are runtime values here rather than compile-time ones.

## Roadmap

1. Integrate the GPU histogram backend into end-to-end training while keeping
   intermediate state device-resident
2. Validate the same GPU source on NVIDIA and AMD hardware
   ([procedure](docs/GPU_VALIDATION.md); neither has been run). The kernels
   already scale past one threadgroup per feature and tile themselves from
   device capabilities, but every measurement so far is Apple Metal
3. Publish the Python API to PyPI (macOS arm64 wheels build and validate
   today; Linux needs a manylinux build)
4. Broader benchmark suite (XGBoost and real datasets)

## Defaults

Matched to LightGBM so comparisons are apples to apples.

| Parameter | Default |
|---|---|
| `num_leaves` | 31 |
| `max_depth` | -1 (unlimited, LightGBM's default) |
| `learning_rate` | 0.1 |
| `n_estimators` | 100 |
| `min_data_in_leaf` | 20 |
| `max_bin` | 255 |
| `lambda_l2` | 1.0 (LightGBM's own default is 0; benchmarks set both to 1.0) |
| `lambda_l1` | 0.0 (LightGBM's default) |
| `bagging_fraction` | 1.0 (LightGBM's default) |
| `bagging_freq` | 0, meaning no bagging (LightGBM's default) |
| `bagging_seed` | 3 (LightGBM's default) |
| `boosting` | `gbdt` (LightGBM's default; `goss` is the other value) |
| `top_rate` | 0.2 (LightGBM's default, GOSS only) |
| `other_rate` | 0.1 (LightGBM's default, GOSS only) |
| `goss_seed` | 3 (LightGBM draws GOSS from `bagging_seed`, whose default is 3) |
| `feature_fraction` | 1.0 (LightGBM's default) |
| `feature_fraction_bynode` | 1.0 (LightGBM's default) |
| `feature_fraction_seed` | 2 (LightGBM's default) |

### Maximum depth

`max_depth` bounds how deep any leaf may sit, counted in edges from the
root, so the root is depth 0 and `max_depth=1` grows stumps. Values of 0 or
less mean unlimited, matching LightGBM, and that is the default.

The limit does not change how trees are grown. Growth stays leaf-wise: the
highest-gain leaf anywhere in the tree still splits first, and a leaf that
has reached the limit simply stops offering a split, so it is never chosen.
Depth-bounded trees are therefore still unbalanced and usually have fewer
than `2**max_depth` leaves, unlike a level-wise grower's. `max_depth` and
`num_leaves` compose: whichever binds first stops growth.

Because the limit depends only on tree shape, and never on histogram
values, the CPU and GPU trainers cut growth at exactly the same leaves;
both go through the same `_search` entry point, which is where the check
lives.

`max_depth` is a training parameter only. It constrains which trees get
built, but adds nothing to a fitted model, so the serialization format is
unchanged and models saved before it round-trip as they always did.

### L1 regularization

`lambda_l1` follows LightGBM's `ThresholdL1`: every gradient sum is shrunk
toward zero by `lambda_l1` and clamped there,

    T(G) = sign(G) * max(0, |G| - lambda_l1)

and `T` is applied to the parent, left, and right sums of the split gain and
to the gradient sum of the Newton leaf value `-T(G) / (H + lambda_l2)`. A
leaf whose gradients all fall inside the threshold gets value zero, and a
split whose children both fall inside it has no gain and is never taken. The
penalty acts on absolute gradient sums, so sample weights scale what the
threshold removes: a leaf of heavily weighted rows keeps more of its
gradient than the same rows at weight one.

The Python estimators also accept the spellings LightGBM's own scikit-learn
estimators use, `reg_alpha` for `lambda_l1` and `reg_lambda` for
`lambda_l2`, under the alias rule in [Python API](#python-api).

Two intentional consequences, both matching LightGBM:

- for the `mae`/`quantile` objectives, leaf values are replaced afterwards by
  residual percentiles (LightGBM's `RenewTreeOutput`), so `lambda_l1` shapes
  which splits are chosen but not the final leaf value
- `lambda_l1` is a training parameter only; it changes the trees, not the
  model format, so serialized models are unaffected

### Row bagging

`bagging_fraction` and `bagging_freq` are LightGBM's row bagging. Every
`bagging_freq` rounds a new bag is drawn, each row kept independently with
probability `bagging_fraction`, and the trees of the rounds in between are
grown on that bag. `bagging_freq=0` or `bagging_fraction=1` means no
bagging, which is the default.

```python
MojoBoostRegressor(
    bagging_fraction=0.8, bagging_freq=1, bagging_seed=3
).fit(X, y)
```

```mojo
var booster = train(
    data, target, SQUARED_ERROR, params, sample_weight=[], alpha=0.9,
    bagging=BaggingParams(0.8, 1, 3),
)
```

Within tree growth the bag is the dataset: histograms, row counts,
`min_data_in_leaf`, leaf values, and quantile/L1 leaf renewal see bagged
rows only, and a tree grown on a bag is bit-for-bit the tree the grower
would produce on a dataset physically holding just those rows (asserted in
`tests/test_bagging.mojo`). Outside tree growth nothing is bagged, again as
in LightGBM: the base score comes from every row, and every row's raw score
is updated after each tree, so out-of-bag rows carry correct gradients into
later rounds.

Sampling is uniform over rows and ignores `sample_weight`. A heavy row is no
likelier to be drawn, it just carries its weight into the gradients of
whichever bag holds it; a zero-weight row contributes nothing whether it is
drawn or not.

Bags are drawn from a counter-based splitmix64 stream keyed by
`(bagging_seed, bag index, row index)` rather than from a running RNG. A
given bag therefore depends on nothing that happened before it, which is
what lets the CPU and GPU trainers grow round *i* on identical rows, and
what makes a run reproducible from the seed alone. On the GPU the bag rides
the existing leaf-assignment array: out-of-bag rows are parked at a leaf id
no histogram build can target, so nothing is compacted, copied, or
reuploaded.

Two intentional differences from LightGBM:

- LightGBM draws from a 15-bit LCG seeded per 1024-row block. splitmix64
  carries 53 bits and needs no blocking, so bags do not match LightGBM's
  row for row at equal seeds; the distribution and the resampling schedule
  do match
- a draw that selects no rows falls back to the single row with the
  smallest draw value, so a bag is never empty and an unlucky draw on a
  tiny dataset cannot end training early

Like the other regularizers, bagging is a training parameter only: it
changes which trees get built, not the model format.

### Gradient-based One-Side Sampling

`boosting="goss"` replaces uniform row bagging with LightGBM's GOSS. Each
round keeps the `top_rate` share of rows with the largest gradient
magnitude, samples `other_rate` of the remaining rows, and scales the
sampled rows' gradients and hessians by `(n - top_k) / other_k` so that the
smaller sample still estimates the full-data histogram. Ordinary GBDT
remains the default.

```python
MojoBoostRegressor(
    boosting="goss", top_rate=0.2, other_rate=0.1, goss_seed=3
).fit(X, y)
```

```mojo
var booster = train(
    data, target, SQUARED_ERROR, params, sample_weight=[], alpha=0.9,
    bagging=BaggingParams.disabled(),
    goss=GossParams.enable(top_rate=0.2, other_rate=0.1),
)
```

The sample is the row list the tree is grown on, the same list bagging
fills, so everything downstream follows from that: histograms, row counts,
`min_data_in_leaf`, leaf values, and quantile/L1 leaf renewal see sampled
rows only, and the compensation multiplier reaches the histograms because
it is applied to the gradients before they are accumulated (or uploaded, on
the GPU). Sample weights ride in through the gradients too, so a heavy row
is ranked by its weighted contribution and a zero-weight row ranks last and
contributes nothing if drawn. Multiclass draws one sample per round from
the per-row importance summed over the round's trees, so every class's tree
is grown on the same rows. Both backends rank rows on the host from the
same Float64 gradients, so CPU and GPU sample identically given identical
gradients.

Row importance is `|grad * hess|`, which is what LightGBM computes, not the
`|grad|` of the GOSS paper. Sampling starts only after
`int(1 / learning_rate)` rounds of full-data training, again LightGBM's
behavior; `goss_warmup_rounds` overrides that count, and `-1` (the default)
keeps LightGBM's rule.

Three intentional differences from LightGBM:

- LightGBM splits the rows into blocks and gives each block its own 15-bit
  LCG, so its sample depends on the block layout and consecutive rounds
  start their streams at nearby states. mojoboost draws from the same
  counter-based splitmix64 scheme bagging uses, keyed by
  `(goss_seed, round, row index)`, so a row's draw depends on nothing that
  happened before it. The threshold rule, the sampled counts, and the
  multiplier are the same; the individual small-gradient rows drawn are not
- `top_rate + other_rate <= 0` is rejected. LightGBM's `max(1, top_k)`
  would train that configuration on a single row
- GOSS and row bagging together are rejected. LightGBM silently disables
  bagging in that case

GOSS is a training parameter only: it changes which trees get built, not
the model format, so serialized models are unaffected.

### Feature subsampling

`feature_fraction` is LightGBM's per-tree feature sampling and
`feature_fraction_bynode` its per-node sampling. Every tree draws its own
feature set once; with `feature_fraction_bynode < 1` each node then draws
again from that set, so the effective per-node share is the product of the
two. Both draws are without replacement and both default to 1.0, which
selects every feature and makes the seed irrelevant.

```python
MojoBoostRegressor(
    feature_fraction=0.8, feature_fraction_bynode=0.5, feature_fraction_seed=2
).fit(X, y)
```

```mojo
var params = BoosterParams(
    100, 0.1,
    TreeParams(
        31, 20, 1.0, 1e-3, 0.0,
        feature_fraction=0.8,
        feature_fraction_bynode=0.5,
        feature_fraction_seed=2,
    ),
)
```

The count follows LightGBM's `ColSampler::GetCnt`: `round(total * fraction)`,
never fewer than 2 features (or than the number available, when that is
smaller) and never more than the total. Fractions outside `(0, 1]` raise.

Excluded features cost nothing rather than being masked late. A tree
accumulates histograms for its own feature set only, so an excluded feature
is never read out of the binned matrix, on either backend: the CPU builders
loop over the selected ids and the GPU kernel gets one threadgroup column per
selected feature. The binned matrix itself is never copied, re-indexed, or
re-uploaded; histograms keep their full `n_features * n_bins` shape with the
excluded slices left at zero, which is what keeps the sibling-subtraction
trick exact. Per-node sampling then narrows only the split search, since a
node's set is a subset of its tree's already-accumulated set.

Both trainers draw from the same sampler and assign node ids in the same
order, so the CPU and GPU growers work from identical feature sets
(`tests/test_gpu_training.mojo`).

Two intentional differences from LightGBM:

- LightGBM draws from a single linear-congruential stream that advances as
  training proceeds, so a tree's feature set depends on how many draws came
  before it. mojoboost keys an independent counter-based splitmix64 stream on
  `(feature_fraction_seed, tree index, node id)`, so a selection is
  reproducible per tree and per node regardless of history or backend, and
  sets do not match LightGBM's at equal seeds. The count formula and the
  selection algorithm (Knuth's Algorithm S, as in LightGBM's
  `Random::Sample`) are the same, so the distribution matches
- LightGBM builds each node's histograms over that node's by-node set;
  mojoboost builds over the tree's set and applies the by-node set to the
  split search. Split decisions are identical, and building the superset is
  what keeps histogram subtraction exact when the by-node sets differ
  between a parent and its children
- with interaction constraints also configured, LightGBM draws the by-node
  set from the features the branch already allows; mojoboost draws from the
  tree's set and then applies the allow mask, so a node can end up with fewer
  candidates than LightGBM would give it. Both restrict to the same features;
  only how many survive the draw differs

Feature importance follows the sampled sets by construction: a tree can only
credit features its draw allowed, which is what spreads split and gain
importance onto features an unsubsampled ensemble would never reach.
Subsampling is a training parameter only; it changes which trees get built,
not the model format.

### Feature interaction constraints

`interaction_constraints` is a list of feature groups. Two features may
appear together on the same root-to-leaf path only if a single group holds
both, and transitively everything else already on that path. That is what
restricts a model to additive or low-order-interaction structure.

```python
MojoBoostRegressor(interaction_constraints=[[0, 1], [2, 3]]).fit(X, y)
```

```mojo
var constraints = InteractionConstraints.from_groups([[0, 1], [2, 3]], 4)
var params = BoosterParams(
    100, 0.1, TreeParams(31, 20, 1.0, 1e-3, 0.0, constraints^)
)
```

The rule follows LightGBM's `ColSampler::GetByNode`. For a node whose branch
features (those split on between the root and it) are `B`, the features it
may split on are

    allowed(B) = B  union  ( union of every group G with B subset-of G )

so the root allows the union of all groups, each split narrows the allowed
set to the groups that still contain the whole branch, and a feature already
used on the branch stays available (re-splitting on it adds no interaction).
The invariant that falls out, and the one the tests check by walking every
root-to-leaf path of every tree, is that a path's feature set is always
contained in at least one configured group.

**Overlapping groups** are supported and are the point of the union: with
`[[0, 1], [1, 2]]`, feature 1 may pair with 0 or with 2, but no path may
carry 0, 1, and 2 together, because no single group holds all three.

**Unconstrained features do not exist.** Since the root allows only the
union of the groups, a feature listed in no group is never split on and
drops out of the model entirely. This matches LightGBM and is the sharp
edge: constraining 2 of 50 features silently discards the other 48. To leave
a feature free to interact with everything, add it to every group. Putting
it in a group of its own does the opposite of what it looks like, allowing
it at the root but nowhere below another group's split.

Three further notes:

- an empty constraint set means no constraints, the default, and costs
  nothing at run time (no allow mask is built and split search is untouched)
- the CPU and GPU trainers share the branch tracking, the allow masks, and
  the split-search entry point, so constraints are enforced identically on
  both, independent of the GPU's Float32 histogram precision
- constraints are a training parameter only; they change which trees get
  grown, not how a grown tree is evaluated, so the model format is
  unchanged and serialized models carry no constraint record, the same way
  they carry no `num_leaves`

### Custom objectives

An objective is a callable that takes the current raw predictions and the
labels and fills gradient and hessian arrays. In Mojo it is a compile-time
callable parameter, so `train_custom` specializes on your callable and the
per-round call is direct and inlinable, with nothing dynamic in the boosting
loop:

```mojo
from mojoboost import BoosterParams, TreeParams, mean_label, train_custom

def my_objective(
    raw: List[Float64],
    target: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
) raises:
    grad.clear()
    hess.clear()
    for r in range(len(target)):
        grad.append(raw[r] - target[r])   # squared error
        hess.append(1.0)

def main() raises:
    var params = BoosterParams(100, 0.1, TreeParams.default())
    var booster = train_custom(
        data, target, my_objective, params,
        base_score=mean_label(target, []),
    )
```

That example is the built-in squared-error objective written out, and with
the label mean as the base score it reproduces it bit for bit, weighted or
unweighted (`tests/test_custom_objective.mojo`). Closures work too, so
hyperparameters ride along in a capture list. `train_custom_gpu` is the same
thing with GPU tree growth; `fit_custom` takes raw features instead of a
binned matrix; `train_custom_with_valid` adds early stopping and takes a
second callable, the validation loss (lower is better), since a custom
objective carries no metric the framework could infer.

The contract, and where it intentionally differs from LightGBM:

- **sample weights are applied for you.** The callback returns unweighted
  per-row derivatives and never sees the weights; the trainer multiplies
  both arrays afterwards, exactly as the built-in objectives do. LightGBM
  hands the raw scores to the callback and expects it to fold the weights in
  itself. Ours keeps a custom objective and its built-in twin bit-identical
  under weights, and makes a weight-unaware callback correct by construction
- **the returned arrays are validated** every round: length, non-finite
  values, and negative hessians all raise, naming the offending row. Zero
  hessians are allowed, since that is what a zero-weight row produces.
  LightGBM does not check, and will train on NaN. The check is one pass over
  the rows per round
- **the base score is yours.** Custom objectives start from `base_score`
  (default 0.0), because the framework does not know the link function.
  LightGBM's `boost_from_average` likewise does not apply to custom
  objectives. Pass `mean_label(target, weights)` to start where the built-in
  mean-link objectives do
- **predictions are raw scores.** The fitted model carries the `CUSTOM`
  objective code and `predict` returns the raw score, since the inverse link
  is yours to apply. This matches LightGBM
- **single output only.** There is no multiclass custom objective:
  LightGBM's (n_rows x n_classes) gradient matrix is deliberately not
  supported, `train_multiclass` has no custom entry point, and passing
  `CUSTOM` to `train`, `train_gpu`, or `fit` raises and points at
  `train_custom`

From Python, pass a callable as the objective. It is called once per
boosting round with the whole array of raw predictions, never per row:

```python
import numpy as np
from mojoboost import MojoBoostRegressor

def squared_error(raw, y):
    return raw - y, np.ones_like(raw)

model = MojoBoostRegressor(
    objective=squared_error, base_score="mean", n_estimators=100
).fit(X, y)
pred = model.predict(X)          # raw scores: apply your own link
```

`base_score` accepts a number or `"mean"` for the weighted label mean.
`MojoBoostClassifier` takes no objective and says so; use the regressor and
apply your own link. The Python callback is convenient, not free:
`bench/bench_custom_objective.py` measures it, and on 100,000 rows x 20
features x 100 rounds (Apple M4, best of 3) it added 8.9 ms per round, 36%
over the same fit with the built-in objective, for bit-identical
predictions. That is a fixed per-round cost plus a per-row copy, so it
shrinks as a fraction of a bigger fit and dominates a small one. Use the
native Mojo interface when the objective is on a hot path.

### Learning to rank

LambdaRank, LightGBM's `objective="lambdarank"`. Ranking data is a row
matrix plus query boundaries: rows of one query are the documents retrieved
for one search, and the label is graded relevance, an integer in [0, 30]
with 0 meaning irrelevant. Rows of a query must be contiguous.

```python
from mojoboost import MojoBoostRanker, group_from_query_ids, ndcg_score

# `group` is LightGBM's: the number of rows in each query, in row order.
model = MojoBoostRanker(n_estimators=100).fit(X, y, group=[6, 4, 9])
scores = model.predict(X)                  # rank each query by these
print(model.score(X, y, group=[6, 4, 9]))  # mean NDCG@5

group = group_from_query_ids(qids)         # raises on a split-up query
print(ndcg_score(scores, y, group, at=10))
```

```mojo
from mojoboost import BoosterParams, groups_from_counts, ndcg, train_ranker

var groups = groups_from_counts(group_counts)
var booster = train_ranker(data, labels, groups, params)
```

Within a query, every pair of documents with different labels contributes a
lambda proportional to the NDCG the ranking would gain by swapping them,
weighted by a pairwise logistic on the score difference.
`lambdarank_truncation_level` (30), `sigmoid` (1.0), and `lambdarank_norm`
(on) carry LightGBM's meanings, and the truncation level is also the cutoff
of the maxDCG the lambdas are normalized by, as in LightGBM.

Nothing crosses a query boundary. Lambdas compare only documents of the
same query, NDCG is averaged over queries after ranking each one within
itself, `train_ranker_with_valid` early-stops on the validation set's own
per-query NDCG, and bagging samples whole queries (LightGBM's
`bagging_by_query=true`) because a half-sampled query would be normalized
against a maxDCG that no served ranking ever had. Malformed groups
(nonpositive counts, counts that do not cover every row) and noncontiguous
query ids are rejected rather than silently reinterpreted.

A ranker serializes as an ordinary single-output model, since query
boundaries are training data and not model state. `predict` returns raw
scores whose order is the only meaningful thing about them: they are not
comparable between queries.

`src/mojoboost/ranking.mojo` documents the intentional differences from
LightGBM (an exactly evaluated pairwise sigmoid instead of a lookup table,
a fixed `label_gain`, query-level bagging, and no unbiased-lambdarank
extensions). `pixi run -e bench compare-ranking` checks the NDCG metric
against LightGBM's on identical scores and compares the two libraries'
ranking quality on held-out queries.

## Development

Requires [pixi](https://pixi.sh).

```sh
pixi install
pixi run test
```

The test command includes CPU/GPU equivalence checks. They run when a
supported accelerator is present and skip cleanly on CPU-only machines.

## Benchmarks

Reproducible from `bench/` (methodology, exact parameters, and caveats in
[bench/README.md](bench/README.md)). Both drivers generate bit-identical
synthetic data from the same splitmix64 stream and train with matched
parameters. The table below preserves the original single-thread baseline;
rerun the commands for current multicore results. 100,000 rows x 100
features, 100 rounds, Apple M4:

| | mojoboost (1 thread) | LightGBM (1 thread) |
|---|---|---|
| Regression: training | 3.53 s | 2.41 s |
| Regression: binning | 0.55 s | 0.81 s |
| Regression: train MSE | 0.003615 | 0.003797 |
| Binary: training | 3.50 s | 2.32 s |
| Binary: train logloss | 0.267034 | 0.267168 |

The original implementation was within 1.5x of single-threaded LightGBM on
training and faster at binning, before multicore histogram accumulation.

```sh
pixi run bench                 # mojoboost
pixi run -e bench bench-lgbm --threads 1
pixi run bench-hist            # CPU/GPU histogram microbenchmark
pixi run bench-hist-scaling    # GPU strategy and phase breakdown
pixi run gpu-validate          # per-device GPU validation report
```

The GPU microbenchmark separates first-use setup from repeated builds. It is
a kernel-development measurement, not an end-to-end GPU-training claim.

`gpu-validate` prints the device identity, the launch geometry, and a phase
breakdown (setup, transfers, kernels, total training) across several dataset
shapes. It is the driver the cross-vendor procedure in
[docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md) is built around. It has been
run on Apple Metal; no NVIDIA or AMD numbers exist, and none should be quoted
until that document's status table says otherwise.

## License

Apache-2.0
