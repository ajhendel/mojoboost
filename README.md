# mojotrees

[![PyPI](https://img.shields.io/pypi/v/mojotrees.svg)](https://pypi.org/project/mojotrees/)
[![CI](https://github.com/mojotrees/mojotrees/actions/workflows/ci.yml/badge.svg)](https://github.com/mojotrees/mojotrees/actions/workflows/ci.yml)

Native gradient-boosted trees accelerated by the GPU already inside Apple
Silicon Macs, written in [Mojo](https://www.modular.com/mojo).

> [!IMPORTANT]
> mojotrees is an experimental public alpha. Its feature surface is broad,
> but it is not yet a production replacement for LightGBM or XGBoost. Treat
> every capability according to the evidence in
> [docs/LIGHTGBM_PARITY.md](docs/LIGHTGBM_PARITY.md) and
> [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md), report failures, and do
> not rely on unvalidated hardware or parameter combinations in production.
> On speed specifically, [Where the speed stands](#where-the-speed-stands)
> gives the current numbers against LightGBM in both directions.

mojotrees is a from-scratch GBDT library in the LightGBM family. It uses
histogram-based split finding and leaf-wise (best-first) tree growth. Its
benchmark configuration aligns important parameters with LightGBM for
reproducible comparisons.

## Creator and stewardship

mojotrees was created by [Andrew Hendel](https://github.com/ajhendel), who
originated the project, set its product and technical direction, and serves as
lead maintainer. The project is developed in public under Apache-2.0 and
welcomes contributors without transferring their copyright.

Development has made extensive use of AI-assisted coding tools under human
direction and review. Andrew remains accountable for what the project claims,
merges, releases, and supports. The project treats reproducible evidence,
not who or what typed a change, as the standard for correctness and
performance.

- [Authors and attribution](AUTHORS.md)
- [Project origin and stewardship](docs/PROJECT_ORIGIN.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Technical article draft](docs/technical/BUILDING_MOJOTREES.md)
- [How to cite mojotrees](CITATION.cff)
- [Governance](GOVERNANCE.md)

## Installation

The first public alpha is on PyPI:

```sh
pip install --pre mojotrees
```

The current `0.1.0a2` wheel supports **CPython 3.14 on Apple Silicon running
macOS 26 or newer**. It is self-contained: using the wheel does not require
Mojo, MAX, Pixi, or a compiler. Other interpreter and platform combinations
will correctly receive `No matching distribution found` until matching
wheels have been built and validated.

| State | What you type | Status today |
|---|---|---|
| Published alpha | `pip install --pre mojotrees` | **Available:** `0.1.0a2`, CPython 3.14, Apple Silicon, macOS 26+ |
| Release wheel file | `pip install ./mojotrees-<version>-<tags>.whl` | **Available** from the release workflow |
| Source checkout with Pixi | the four commands below | **Available** for contributors and development |

[docs/INSTALLATION.md](docs/INSTALLATION.md) is the full version of this,
with what each state will require, the wheel filename for each platform, the
first five minutes step by step, and the error messages an install can
produce with what each one means.

mojotrees publishes no source distribution, deliberately, so `pip install
mojotrees` can never turn into a Mojo compile on a machine with no Mojo
toolchain. It resolves to a wheel that matches your machine or it refuses.

### Source installation for contributors

A Mojo or MAX installation is not required separately; pixi resolves the
versions pinned by this repository.

```sh
git clone https://github.com/mojotrees/mojotrees.git
cd mojotrees
pixi install
pixi run build-python
```

```sh
PYTHONPATH=python python - <<'PY'
from mojotrees import MojoTreesRegressor

X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]

model = MojoTreesRegressor(n_estimators=20, num_leaves=7, min_data_in_leaf=1)
model.fit(X, y)
print(model.predict([[1.5], [4.5]]))
print("backend:", model.device_)
PY
```

`min_data_in_leaf=1` is there because the default of 20 is larger than that
toy dataset, and a model that cannot split predicts one constant.

### The first five minutes

Import and diagnostics, a tiny regression, a validation set with early
stopping, a bit-exact save and load, and what each device value does on your
machine, in one script that prints each result.

```sh
PYTHONPATH=python python examples/install_smoke.py
```

It uses only the standard library, so it runs in the default pixi
environment with no numpy installed. Step by step, with the same seven steps
written out, in [docs/INSTALLATION.md](docs/INSTALLATION.md#the-first-five-minutes).

### Devices, in one paragraph

`device="cpu"` is the default and the dependable backend. `device="gpu"`
requires an available supported accelerator and raises on unsupported
hardware or workloads rather than silently falling back. `device="auto"`
still resolves to the CPU in practice, but no longer because the crossover
table is empty: `crossover_rules()` now holds one rule, scoped to Metal on
an Apple M4 for unweighted squared error at 1,000,000 rows by 50 features
and above. The reason a defaulted `fit(device="auto")` does not reach it is
narrower and is a real gap rather than a policy: a hardware-scoped rule can
match only a profile that names the hardware, and `DeviceCapabilities.detect()`
opens no device, so `resolve_device` sees an unidentified machine, warns,
and keeps the CPU. Read [Device selection](#device-selection) before using
it.

The two backends are not a primary and a port. On an accelerator the GPU
owns the data plane, and the CPU owns the control plane, small data, and
verification. Every per-row structure of a fit (binned features, gradients
and hessians, the leaf assignment, histogram accumulation, row partitioning,
and for the built-in objectives the gradient generation and the score
update) lives on the device and stays there for the whole fit; the CPU
coordinates boosting, selects splits over histograms of `n_features x
n_bins` cells, holds the tree model, and is the bit-exact reference the GPU
path is verified against. Below the launch-cost crossover (small datasets
and the small leaves deep in a tree) the CPU is also the faster place to do
the work, which is why the CPU trainer is a permanent part of the design
rather than a fallback. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#4-host-to-device)
states the division precisely, and
[bench/results](bench/results/apple_m4_large_scaling_2026-08-14.md) holds
the measurement behind it.

If any of this fails, open a
[bug report](https://github.com/mojotrees/mojotrees/issues/new?template=bug_report.yml),
which covers installation problems, and paste the output of
`mojotrees.show_versions()` plus, from a source checkout,
`pixi run mojo --version`. `show_versions()` reports what this install is,
which environment variables are in effect, and whether a GPU path was
compiled into the build, which is decided on the build machine and cannot be
recovered from anything else.

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

Experimental public alpha. Training works end to end and the repository has
a broad LightGBM-shaped feature surface. That is not the same as verified
behavioral or production parity: combinations, edge cases, installation
targets, and non-Apple accelerators remain less mature. The list below is
what works today, each piece with tests; for what LightGBM has that mojotrees does not,
and for the semantics that differ deliberately, read the parity contract in
[docs/LIGHTGBM_PARITY.md](docs/LIGHTGBM_PARITY.md), which is authoritative
where this list and it disagree.

One caution that the size of the list makes worth stating plainly. A file
in `src/mojotrees/` is not a feature. Several capabilities in this
repository are written, compile, and are reached by no entry point, so
nothing you can call behaves differently because they exist.
[docs/INTEGRATION_INVENTORY.md](docs/INTEGRATION_INVENTORY.md) is the
current list of those, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) is the
map of what does reach what, and
[docs/CAPABILITY_LEVELS.md](docs/CAPABILITY_LEVELS.md) is the vocabulary
all three use. The list below is not in that state: every item names
something a user can reach, and where one capability has two
implementations the item says which one runs.

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
- monotonic constraints (`monotone_constraints`) for numerical features,
  LightGBM's `basic` method: predictions are nondecreasing or nonincreasing
  along a constrained feature at any feature value, on both the CPU and GPU
  trainers, for every single-output objective including quantile and L1 leaf
  renewal; recorded in the model file, and free when unused (an all-zero
  vector fits bit-identically to no vector at all)
- native categorical features (`categorical_feature`) with no one-hot
  expansion: an integer-coded column keeps one column and k + 1 bins, and
  splits are searched as category *sets* by LightGBM's gradient/Hessian
  ordering, so codes carry no implied order; one-vs-rest below
  `max_cat_to_onehot`, the sorted many-vs-many walk above it, with
  `max_cat_threshold`, `cat_smooth`, `cat_l2`, and `min_data_per_group`
  carrying LightGBM's meanings and defaults. Missing (negative or `NaN`),
  unseen, and dropped codes share a reserved bin 0 that is in no split set,
  so all three route right, as in LightGBM. Category sets are stored per node
  as a 256-bit set, applied identically by CPU growth, prediction, and the
  GPU partition kernel, and written to the model file; models with no
  categorical feature serialize to exactly the bytes they did before. See
  [Categorical features](#categorical-features)
- objectives: squared error, binary logistic, poisson, huber, quantile,
  L1 (mean absolute error), gamma, tweedie, MAPE, fair, cross entropy
  (labels anywhere in [0, 1]), and multiclass softmax. Quantile, L1, and
  MAPE renew leaf values with residual percentiles the way LightGBM does,
  MAPE under its own `1 / max(1, |y|)` label weights. One trainer slot
  carries each objective's scalar parameter under LightGBM's name for it:
  `alpha` for huber and quantile, `fair_c` for fair,
  `tweedie_variance_power` for tweedie. LightGBM objectives that are *not*
  implemented (`cross_entropy_lambda`, `multiclassova`, `rank_xendcg`) are
  reported by name, with what to use instead, rather than as unknown
- custom objectives (`train_custom`) through a compile-time callable, so
  the boosting loop stays direct-call; validated gradients and hessians,
  framework-applied sample weights, and a Python callback path with a
  measured per-round cost (see [Custom objectives](#custom-objectives))
- learning to rank with LambdaRank (`train_ranker`, `fit_ranker`,
  `MojoTreesRanker`): LightGBM's NDCG-weighted pairwise lambdas and
  hessians, `lambdarank_truncation_level`, `sigmoid`, and `lambdarank_norm`;
  query boundaries from LightGBM's `group` array or from a query id column,
  which is rejected when a query's rows are not contiguous; NDCG at any
  cutoff; query-aware validation and early stopping; and bagging that
  samples whole queries rather than rows
- validation-set early stopping with `min_delta` for every objective,
  truncating to the best round
- custom validation metrics (`train_with_metrics`, `fit_with_metrics`,
  `MojoTreesRegressor.fit(eval_set=..., eval_metric=...)`), independent of
  the objective: each metric has a name, a direction, and a flag for
  whether early stopping watches it, several may run against several
  validation sets, and an explicit primary metric picks the round the
  ensemble is truncated to (see
  [Custom validation metrics](#custom-validation-metrics))
- evaluation metrics: L2/RMSE, L1, quantile, huber, MAPE, fair, the
  poisson, gamma, and tweedie deviances, binary and multiclass log loss,
  cross entropy and KL divergence for soft labels, accuracy and error
  rates, ROC AUC and average precision with sklearn-matching tie handling,
  and NDCG and MAP per query. Selectable by LightGBM's names from Python
  (`eval_metric="auc"`) and callable directly from Mojo
- class weighting: `class_weight="balanced"` or a per-class dict on the
  Python classifier, and LightGBM's `scale_pos_weight` / `is_unbalance`
  under their own names in `src/mojotrees/class_weight.mojo`. Each is
  folded into the row weights before training rather than into the
  objective, so it composes with `sample_weight` instead of competing with
  it. Two implementations of that folding exist today, one in Mojo and one
  in the Python estimator, and only the Python one runs for a
  `MojoTreesClassifier`; see
  [docs/INTEGRATION_INVENTORY.md](docs/INTEGRATION_INVENTORY.md)
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
- multicore CPU work across independent features (histogram accumulation,
  split search, bin fitting, bin transform) and across contiguous row blocks
  (gradient generation, row partitioning), scheduled from a work estimate
  rather than an item count so cheap elementwise stages are not fanned out
  below the point where scheduling costs more than the work. Every path is
  bit-identical to the serial one at any worker count: feature tasks keep each
  feature's sum inside one task, row blocks are used only where nothing is
  summed across rows, and the split search folds its per-feature winners in
  ascending order so the tie-break is the serial one. Two grains govern the
  fan-out and they are separate: `MOJOTREES_PARALLEL_MIN_OPS` is the
  whole-loop crossover below which a loop stays serial, and
  `MOJOTREES_PARALLEL_MIN_TASK_OPS` is the smallest work a single task may be
  given above it. `MOJOTREES_NUM_WORKERS` overrides both and pins the
  scheduler for reproducible runs; `pixi run bench-profile` times each stage
  serial against parallel
- a data-parallel distributed training prototype: rows partitioned across
  ranks, one histogram all-reduce per tree node, and every split decision
  taken from global histograms so the tree stays identical on every rank
  without broadcasting a single split. Equivalence with the single-node
  trainer is bit-exact wherever the arithmetic is exact. Every rank runs in
  one process and nothing has run over a network, so no distributed
  performance is claimed and this is a prototype rather than a feature; the
  transport and collective contract, the determinism and failure semantics,
  and what is deliberately unsupported are in
  [docs/distributed.md](docs/distributed.md)
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
- scikit-learn style Python API (`MojoTreesRegressor`,
  `MojoTreesClassifier`, `MojoTreesRanker`) backed by a CPython extension
  module built from the same Mojo code, with sample weights and exact
  save/load
- LightGBM-native parameter names as the canonical Python vocabulary, with
  the aliases used by LightGBM's scikit-learn estimators accepted for easy
  migration; conflicting aliases raise instead of silently taking precedence
- multiclass end to end on raw data: `fit_multiclass` returns a
  `MulticlassModel` with `predict_proba` and `predict_class`
- a small stable C ABI (`capi/`) with opaque model handles, dense training,
  prediction, save/load, error retrieval, and destruction, meant as the base
  for bindings in other languages (see [C API](#c-api))
- a command line tool (`cli/`) that trains and predicts from documented text
  files, with no code to write (see [Command line tool](#command-line-tool))
- sparse training and prediction that never densifies the matrix, in Mojo
  (`fit_csc`, `train_sparse`, `predict_csr`) and from Python (any SciPy
  sparse matrix passed to `fit` or `predict`), see
  [Sparse input](#sparse-input)

```mojo
from mojotrees import BINARY_LOGISTIC, BoosterParams, TreeParams, fit

def main() raises:
    # features is column-major: features[f * n_rows + r]
    var params = BoosterParams(100, 0.1, TreeParams.default())
    var model = fit(features, n_rows, n_features, labels,
                    BINARY_LOGISTIC, params)
    var p = model.predict(row)   # raw feature values in, probability out
```

Lower-level entry points `train`, `train_with_valid`, and
`train_multiclass` operate on pre-binned matrices.

### Where the speed stands

Correctness, determinism, portability, memory, and accuracy are the parts
of this project that are good and measured. Speed has been the part that is
behind. As of 2026-08-15 there is exactly one shape where it is not, and
most of this section is what that single result does and does not license.

This project separates measurement from inference on purpose, so every
figure below is labeled **measured**, **fitted**, **derived**, or
**estimated**, and the label is part of the claim.

Every number is Apple M4, Mojo 1.0.0, 100 rounds, 31 leaves, 255 bins,
squared error, seconds of training time with binning excluded. Our own arms
are interleaved inside one process; the LightGBM column comes from a
separate single run of `bench/bench_lightgbm.py` on the same generated data,
which is the asymmetry condition 4 below is about. LightGBM is 4.7 at 10
threads. The second
sweep runs five repeats and reports the median with its spread; the earlier
profile run is a median of three. The records are
[`bench/results/sweep2_2026-08-15/RESULTS.md`](bench/results/sweep2_2026-08-15/RESULTS.md)
and
[`bench/results/profile_2026-08-15/RESULTS.md`](bench/results/profile_2026-08-15/RESULTS.md),
both taken under the rules committed beforehand in
[`bench/results/PROFILE_PROTOCOL.md`](bench/results/PROFILE_PROTOCOL.md).

**Measured**, second sweep:

| shape | our CPU | our GPU, leaf-wise | our GPU, depth-wise | LightGBM, 10 threads |
|---|---|---|---|---|
| 250,000 x 50 | 1.649 | 1.967 | 1.909 | **1.023** |
| 1,000,000 x 50 | 5.942 | 3.756 | **2.587** | 2.767 |
| 2,000,000 x 50 | 13.483 | 6.093 | 5.417 | **5.228** |

Spreads in the same order: our GPU 1.1 / 1.8 / 12.5 percent, depth-wise
6.7 / 0.3 / 10.8, our CPU 9.3 / 13.5 / 35.6. The two-million-row CPU arm at
35.6 percent is too noisy to carry a verdict and is reported rather than
relied on.

#### The first measured win, and the four ways it could mislead

**At 1,000,000 x 50, `grow_policy="depthwise"` on Metal trains in 2.587
seconds against LightGBM's 2.767 on ten CPU cores. That is 6.5 percent
faster at a 0.3 percent spread**, the tightest arm in the sweep. At
2,000,000 x 50 the same arm is 3.6 percent behind (5.417 against 5.228),
which is parity. This is the first measured win this project has against
LightGBM on training speed at a large shape, and it is measured, not
projected.

Four conditions travel with that sentence. Without them it is misleading,
and in four separate ways.

**1. It is a different model.** Depth-wise growth grows a different tree
than leaf-wise growth does, from the same data and the same parameters.
LightGBM grows leaf-wise and has no `grow_policy`; XGBoost grows depth-wise
by default, so this is a growth order users legitimately choose, but the
comparison above is *our depth-wise against LightGBM's leaf-wise* and it is
not a like-for-like race. Our own leaf-wise arm is still behind LightGBM at
every shape measured: 1.92x at 250,000, 1.36x at 1,000,000, 1.17x at
2,000,000. For that reason depth-wise is not the default and does not
become one. Answering a speed question by silently changing the model is
not something this project does.

**2. Accuracy has not been shown for these arms, and no accuracy parity is
claimed for depth-wise.** `bench_train_gpu.mojo` computes a training loss
per arm on the first repeat and prints it as `<arm>_train_loss`, and
`bench_lightgbm.py` prints `train_mse` for the same generated data; the
committed sweep record kept only the timing lines, so **no training loss
survives for any arm of this sweep, ours or LightGBM's**. Nobody has
measured whether a depth-wise tree at these shapes is as accurate as the
leaf-wise tree it is being timed against, and a speed win from a worse model
is not a win. What would establish it is two runs that already have
harnesses: re-run the same arms and keep the `_train_loss` lines beside the
times, and run `bench/real_data/` with a depth-wise arm for held-out metrics
on real datasets rather than a training loss on a generator. Until both
exist, read the table as a timing result only.

The accuracy parity this project does have is **leaf-wise only**, on
LightGBM-matched defaults, and is stated as a range rather than a single
figure: across five pinned real datasets the relative difference on each
scenario's primary metric runs from 5.7e-06 (RCV1, auc) to 3.2e-03 (Bank
Marketing, average precision), a factor of about 560 between the closest and
the furthest. The record, including why the widest cell is the noisiest
metric in the suite, is
[`bench/real_data/results/README.md`](bench/real_data/results/README.md).

**3. It is one machine, one shape family, one dataset generator.** Apple M4,
synthetic dense regression from the splitmix64 stream both drivers share, 50
features, squared error. No real dataset, no second machine, and no NVIDIA
or AMD hardware has ever executed this code
([docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md)).

**4. Our side is repeated and LightGBM's is not.** Our arms are five repeats
interleaved inside one process, and the 0.3 percent figure is the spread of
those five. `bench/bench_lightgbm.py` trains once, in a separate process,
and reports no median and no spread, so 2.767 is a single sample and its
noise floor is unknown rather than zero. This machine is documented as
drifting by a factor of two to three across time windows, which is why the
Mojo harness interleaves at all. A 6.5 percent margin measured that way is
the best evidence in the tree and is still weaker than the 0.3 percent
suggests. What would settle it is a LightGBM arm inside the same interleaved
loop, which the harness does not currently have.

#### What the same sweep says about everything else

**Our marginal cost per row now equals LightGBM's on ten CPU cores, and the
whole remaining deficit is fixed cost.** **Fitted** from the measured
points, three shapes and two segments each:

| segment | our GPU | LightGBM, 10 threads |
|---|---|---|
| 250,000 to 1,000,000 | 2.385 us/row | 2.325 us/row |
| 1,000,000 to 2,000,000 | 2.337 us/row | 2.461 us/row |

The four fitted slopes span 2.33 to 2.46 microseconds per row and they
interleave, so on the marginal cost of a row this library and LightGBM are
even. Two older claims do not survive that: the cost per row is not
superlinear above one million rows, and we are not about 10 percent better
per row than LightGBM.

Extrapolating each fit back to zero rows, our intercept is about 1.42
seconds and LightGBM's about 0.44, so the gap is roughly **1.0 second of
fixed cost that does not grow with the data** (**derived** from the fits
above, not measured directly). That is the same finding the Metal timeline
reached in a different unit.

The consequence is the part worth carrying, and it is an **estimate**:
remove the entire second of excess fixed cost and 1,000,000 rows lands near
2.8 seconds, which is **parity with LightGBM, not a win**. Depth-wise beat
that number, so it did not only remove fixed cost. Batching a level into one
host wait also fills the GPU better per launch, which moves the **slope**
rather than only the intercept. That reframes the plan: the histogram kernel
is not a phase behind the control plane, it is the entire margin.

**The one-line status, which is the honest form of all of the above: our
per-row cost is even with LightGBM's, our fixed cost is about one second
behind, and the win, if it comes, comes from the histogram kernel.**

#### The automatic split-search gate was protecting the losing arm, and is withdrawn

The GPU's automatic split search compared `normalized_split_work`
(`n_rows * active_features * (n_bins/255) * (num_leaves/31)`) against a
threshold of 50,000,000 and sent anything below it to the **host scan**:
download the node's whole `3 * n_features * n_bins` histogram, block on the
device, dequantize it to Float64, and scan it on the CPU. A second, lower
threshold of 12,500,000 applied to depth-wise growth.

**Measured 2026-08-16**, four shapes, the two arms interleaved in one process,
five repeats each, box verified quiet at every shape boundary:

| shape | normalized work | host scan | device search | ratio |
|---|---|---|---|---|
| 100,000 x 50 | 5.0M | 1.695 | **0.918** | **1.85x** |
| 250,000 x 100 | 25.0M | 2.494 | **1.862** | **1.34x** |
| 463,715 x 90 | 41.7M | 3.041 | **2.309** | **1.32x** |
| 700,000 x 100 | 70.0M | 4.041 | **3.128** | **1.29x** |

Every shape resolved with disjoint ranges: at each one the device search's
slowest repeat beat the host scan's fastest. The range spans both retired
thresholds, so neither is a boundary the sweep failed to bracket.

**The threshold's premise was inverted, which is the finding rather than the
margin.** A profitability gate assumes the cheaper path is not worth its fixed
cost below some size. The device search's advantage is *largest at the
smallest shape* and shrinks as the work grows -- which is what you see when
the thing being avoided is per node rather than per row: fewer rows means the
per-node download, block and dequantization dominate more, not less. There is
no size at which the host scan becomes the better choice, so both thresholds
were withdrawn and neither was replaced. There is no measured value to install
because there is no crossover to record.

What survives the withdrawal is **eligibility**, which is a different question
and has the same answer at every size: the device search is declined for a
configuration it cannot express (`TreeParams.extra`,
`feature_fraction_bylevel`), for a resident frontier that does not fit, and on
hardware nobody has measured it on. Those three still route to the host scan,
which is why the host scan is still here.

Read the scope precisely: this compares two **GPU** arms. It says nothing
about the CPU/GPU crossover, which is a separate gate and is unchanged.

#### From the earlier profile run, at other shapes

These are the median-of-three figures at 50,000 x 50 and the shapes the
second sweep did not repeat. They are a different run and are not comparable
cell by cell with the table above.

- **At 50,000 rows on the CPU we may be ahead**, 0.564 against LightGBM's
  0.594. Read that as indistinguishable rather than as a win: the 5.3 percent
  margin sits inside our own arm's 11.3 percent spread, and the LightGBM
  number is a single unrepeated sample. This library's earlier description of
  it as the first shape it was faster than LightGBM at anything was not
  established by the run that produced it.
- **We bin 3.2x faster**, 0.377s against LightGBM's 1.207s, on the same
  data. Binning is excluded from every training figure here, so this is a
  column we win outright.
- **The GPU wins multiclass**, 15.30 against the CPU's 25.47 at 465,000 rows
  by 54 features over 7 classes, so 1.63x. That is the first honest
  multiclass measurement in the project; every earlier one was a CPU fit
  wearing a GPU label, and the dispatch bug behind that is fixed.
- **Serially we are 1.81x behind**, 15.96 at one worker against LightGBM's
  8.82 at one thread, so the deficit is not only threading. We do get 2.29x
  from ten cores where LightGBM gets 3.08x, so both are true at once: the
  inner loop is slower and the parallel scaling is worse.

The dominant GPU cost is located and not yet fixed. The first Metal
timeline this project has taken
([docs/METAL_TIMELINE.md](docs/METAL_TIMELINE.md)) says the GPU is idle for
76.5% of a training span at 200,000 rows and 87.5% at 50,000, at the
device's Maximum clock rather than a downclock. The host blocks on 94.1% of
blits and on 2 compute kernels out of 18,701, which is 32.1 serialization
points per round. One blocking readback costs 606 microseconds of wall
clock, of which 3.7 microseconds is the GPU moving bytes. Compute of every
kind is 22.9% of a round, so even an infinitely fast histogram kernel leaves
most of the round in place. That last sentence and the fitted slopes above
are in tension worth keeping in view: the timeline prices the fixed cost and
the fits say removing all of it reaches parity and stops, so both the wait
count and the kernel have to move.

Nothing anywhere in this section has been measured on any device other than
one Apple M4.

## Sparse input

A sparse matrix is trained on as it is, without ever materializing the dense
one. Internally the representation is CSC, because histogram accumulation is
feature-oriented; prediction takes CSR, because it is row-oriented, and the
two convert into each other in O(nnz).

```mojo
from mojotrees import SQUARED_ERROR, BoosterParams, CscMatrix, fit_csc
from mojotrees import predict_csr

def main() raises:
    # Feature f owns entries [col_offsets[f], col_offsets[f + 1]).
    var csc = CscMatrix(row_index, values, col_offsets, n_rows, n_features)
    var model = fit_csc(csc, target, SQUARED_ERROR, BoosterParams.default())
    var scores = predict_csr(model, csc.to_csr())
```

An **absent entry is the numerical value 0.0**, not a missing value, which is
LightGBM's default `zero_as_missing=false`: a sparse fit is the dense fit of
the same matrix with its gaps filled with zeros. Zeros take part in the
quantile bin edges like any other value, and explicitly stored zeros (which
SciPy keeps until `eliminate_zeros()`) mean exactly what the gaps mean. `NaN`
is still the missing marker wherever it is stored, and gets the same reserved
bin and learned default direction it gets on the dense path. LightGBM's
`zero_as_missing=true` is not implemented.

The two paths are separate implementations rather than one abstraction over
both: `histogram_sparse.mojo` visits only the stored entries of a node and
assigns each feature's leftover to the bin holding 0.0, so a node costs
O(nnz_in_node) instead of O(rows * features), and the dense kernels are
untouched. Everything above the accumulator is shared verbatim, so the two
agree on split search, leaf values, constraints, missing-value routing, and
categorical partitioning.

The fitted model is an ordinary `Model`: the serialization format is
unchanged, a sparse-trained model loads and predicts like any other, and
`Model.predict` still takes a dense row.

Agreement with the dense path is to floating-point rounding, not bit-exact,
because the sparse accumulator derives each feature's zero bin by
subtraction. Counts and bin ids agree exactly, and bin edges are bit-exact.
The one caveat is tied split gains: when two candidates score equal to
within that rounding, the two paths can pick different (equally good)
winners and their ensembles then separate. `tests/test_sparse.mojo` pins all
of this down.

Not implemented for sparse input: the GPU backend (there is no sparse
kernel; `device="gpu"` raises), custom objectives, `eval_set` from Python,
ranking, and `pred_leaf`/`pred_contrib`/iteration slicing at predict time.
Each raises rather than densifying silently.

## Python API

Build the extension once with `pixi run build-python` (which runs
`bindings/build.sh`; see [Installation](#installation)), then use the
scikit-learn style estimators in `python/mojotrees`:

```python
from mojotrees import MojoTreesRegressor, MojoTreesClassifier

model = MojoTreesRegressor(num_leaves=31, n_estimators=100).fit(X, y)
pred = model.predict(X)          # numpy in/out when numpy is available
model.save("model.mbst")
model = MojoTreesRegressor.load("model.mbst")

clf = MojoTreesClassifier().fit(X, labels)   # binary or multiclass by labels
proba = clf.predict_proba(X)
```

`fit` accepts `sample_weight`, hyperparameters mirror the Mojo defaults,
and saved models round-trip bit-exactly. numpy is optional; plain Python
sequences work without it. The regressor takes LightGBM objective names
(`objective="regression"`, `"huber"`, `"quantile"`, `"mae"`, `"poisson"`,
`"gamma"`, `"tweedie"`, `"mape"`, `"fair"`, or `"cross_entropy"`) with
`alpha` as the quantile level or huber transition point, `fair_c` and
`tweedie_variance_power` for the two objectives that take them, or a
callable for a custom objective (see
[Custom objectives](#custom-objectives)). The classifier takes
`class_weight`, `"balanced"` or a `{label: weight}` dict. Both
estimators take `device` and record the backend that ran on `device_`; see
[Device selection](#device-selection).

### Dataset, Booster, and train()

The estimators are one door. The other is LightGBM's functional API, which
gives you the model object itself:

```python
import mojotrees as mb

train_set = mb.Dataset(X, label=y)
booster = mb.train({"objective": "regression", "num_leaves": 31},
                   train_set, num_boost_round=100)

booster.predict(X_test)                  # response scale
booster.predict(X_test, raw_score=True)  # link scale
booster.eval(train_set, "training")      # [(name, metric, value, higher?)]
booster.feature_importance("gain")
booster.save_model("model.mbst")
```

A `Dataset` owns the training data and everything that describes its rows:
`label`, `weight`, `group` for ranking, `init_score`, `feature_name`, and
`categorical_feature`. Binning is the expensive part of starting a run, so
it happens once, on `construct()` or on the first `train()` that uses the
dataset, and every later run on it reuses those bins:

```python
train_set = mb.Dataset(X, label=y, weight=w, params={"max_bin": 63})
shallow = mb.train({"objective": "regression", "num_leaves": 7}, train_set, 100)
deep = mb.train({"objective": "regression", "num_leaves": 63}, train_set, 100)
```

A dataset is immutable once constructed. LightGBM's `set_label`,
`set_field`, and `set_categorical_feature` are deliberately absent: bin
edges are fitted from the data and from the categorical declaration, so
changing either afterwards would leave the binned matrix describing data
the dataset no longer holds. Construct another dataset instead.

A `Booster` grows. It starts at zero iterations, `update(n)` adds rounds,
and continued training resumes from where the existing trees left off, so
40 rounds and then 60 more are the 100-round model, tree for tree:

```python
booster = mb.Booster({"objective": "regression"}, train_set)
while not booster.update():      # returns True when the objective converges
    pass

warm = mb.train(params, train_set, 60, init_model=booster)   # copies, then grows
```

That holds for the single-output objectives and for softmax multiclass.
Ranking is the exception: LambdaRank gradients are computed within a query
from state the fitted ensemble does not carry, so a ranking booster raises
rather than appending trees that would be wrong.

`init_score` is training state, not model state. Boosting starts from it
and the fitted model predicts the trees alone, so scoring new data means
adding your own offset back, which is what LightGBM does too.

The estimators hold the same object on `booster_`:

```python
model = MojoTreesRegressor(n_estimators=100).fit(X, y)
model.booster_.num_trees()
model.booster_.model_to_string()
```

There is one model object in the package rather than one per API, and the
test suite asserts that a model trained through `train()` and the same
model trained through an estimator predict identically, value for value.
What `booster_` cannot do is `update()`: an estimator bins the matrix it
was handed and does not keep the `Dataset` that continued training grows
on. Train through `mb.train()` when you want that.

`docs/LIGHTGBM_PARITY.md` section 5 is the row-by-row statement of what
these two types do and do not carry over from LightGBM.

### scikit-learn conventions

The estimators implement `get_params`, `set_params`, `fit`, `predict`,
`predict_proba` (classifier), and `score` (R^2 for the regressor, accuracy
for the classifier), and set `n_features_in_`, `feature_names_in_`,
`classes_`, `n_classes_`, `feature_importances_`, `best_iteration_`, and
`device_` when fitted. Methods that need a model raise `NotFittedError`
before that, which is also a `RuntimeError` and, when scikit-learn is
installed, its `NotFittedError` too. `clone`, `Pipeline`, `GridSearchCV`,
and `cross_val_score` work; scikit-learn itself stays optional, and
nothing imports it except the `__sklearn_tags__` hook scikit-learn calls.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("scale", StandardScaler()), ("gbdt", MojoTreesRegressor())])
search = GridSearchCV(pipe, {"gbdt__num_leaves": [15, 31]}, cv=3).fit(X, y)
```

Classifier labels may be of any single comparable type, as in
scikit-learn: they are sorted onto `classes_` and encoded to the
0..n_classes-1 the trainer needs, and `predict` returns labels from
`classes_`. `feature_importances_` reports what `importance_type` selects,
`"split"` (LightGBM's default) or `"gain"`. `feature_names_in_` is
recorded when `X` carries string column names, and a later `predict` with
different names raises.

Input validation follows LightGBM's own scikit-learn wrapper, which
validates with `force_all_finite="allow-nan"`: `X` may hold `NaN`, the
missing-value marker, but not infinities, while `y` and `sample_weight`
must be finite, and weights must be nonnegative and not all zero. Sparse
input is rejected rather than densified behind your back.

### Prediction options

`predict`, and the classifier's `predict_proba`, take LightGBM's prediction
keywords.

```python
model.predict(X, raw_score=True)              # before the inverse link
model.predict(X, num_iteration=10)            # the first 10 iterations
model.predict(X, start_iteration=10)          # everything after them
model.predict(X, pred_leaf=True)              # leaf ordinals, one per tree
model.predict(X, pred_contrib=True)           # exact TreeSHAP contributions
model.predict(X, validate_features=True)      # names must match, not warn
```

`raw_score` skips the inverse link. It changes nothing for the regressor's
objectives or the ranker, which have none; the binary classifier returns
log-odds of shape `(n_samples,)`, one score per row rather than one per
class, and the multiclass classifier the pre-softmax scores of shape
`(n_samples, n_classes)`, both as in LightGBM.

`start_iteration` and `num_iteration` slice the boosting iterations, and
clamp the way LightGBM's `GBDT::InitPredict` does: a negative start becomes
0, a start past the end selects nothing, and `num_iteration=None` or any
value <= 0 means every iteration from the start on. The base score belongs
to iteration 0, so a slice starting there carries it and a later slice does
not. Two consequences are worth stating, and both are tested: predicting
with `num_iteration=k` reproduces a `k`-round fit exactly, and `[0, k)` and
`[k, n)` sum to the whole raw score. Out-of-range slices clamp rather than
raise; a non-integer bound is a `TypeError`.

`num_iteration=None` predicts with `best_iteration_` iterations, LightGBM's
documented default. mojotrees gets there structurally: early stopping
truncates the ensemble at its best iteration, so the trees the model still
holds are the best iteration and there is nothing later to exclude.

`pred_leaf` returns integers, shape `(n_samples, num_iteration)` for the
single-output estimators (regressor, ranker, binary classifier) and
`(n_samples, num_iteration * n_classes)` for the multiclass classifier,
whose column `i * n_classes + k` holds class k's tree in iteration i. A leaf
is named by its ordinal within its own tree, in `[0, num_leaves)`, numbered
in node order; the numbering is fixed once a tree is grown and survives
`save`/`load` and pickling. It is mojotrees's own numbering and not
LightGBM's leaf id.

`pred_contrib` returns exact per-feature contributions: shape
`(n_samples, n_features + 1)` for the single-output estimators and
`(n_samples, n_classes * (n_features + 1))` in class-major blocks for the
multiclass classifier, both LightGBM's shapes. See the next section.

`validate_features` turns the feature-name checks from warnings into
errors. `raw_score=True`, `pred_leaf=True`, and `pred_contrib=True` ask for
different dtypes and shapes, so passing more than one raises; LightGBM picks
a winner silently, which the output does not reveal.

### Feature contributions

`predict(X, pred_contrib=True)` returns exact TreeSHAP values, not a
split-gain heuristic. The last column of each row (of each class block, for
multiclass) is the expected value, and **every row's entries sum to that
row's raw score**, exactly:

```python
contrib = model.predict(X, pred_contrib=True)
assert np.allclose(contrib.sum(axis=1), model.predict(X, raw_score=True))
```

That identity is not a normalization step. The numbers are the Shapley
values of `v(S) = E[f(x) | x_S]`, so summing them is the Shapley efficiency
property. The conditional expectation is taken over each tree's own node
covers, which is the "path-dependent" convention LightGBM, XGBoost, and the
reference `shap` package use for a model with no supplied background data.
The algorithm is the polynomial-time recursion of
[Lundberg, Erion, and Lee (2018)](https://arxiv.org/abs/1802.03888), which
costs `O(L·D²)` per tree rather than the `O(2^M)` a subset enumeration
would.

Contributions always explain the **raw** score, whatever the objective's
link, so `raw_score=True` cannot be combined with them. For a binary
classifier they explain the log-odds; pushing them through the sigmoid is
not meaningful, because the sigmoid is not additive.

Missing values, categorical set splits, and features split on more than once
along one path are all handled: the recursion asks only which child a row
takes, which is the same routing prediction uses, and it unwinds a repeated
feature before re-extending it. A depth-zero tree contributes to the
expected value alone, since it says nothing about any feature.

Node covers are recorded during training and travel with the model from
format v3 on, because they cannot be recovered from a fitted tree. A model
saved by an older build (v1 or v2) still loads and predicts exactly as
before, but asking it for contributions raises rather than guessing at the
covers; retrain or re-save from a current build. Split gains travel the
same way from v4 on, for the same reason.

How this is checked: `tests/test_contrib.mojo` and
`python/tests/test_contrib.py` each carry an independent implementation that
enumerates all `2^M` feature subsets straight from the Shapley definition,
sharing no code with the recursion (the Python one goes further and parses
the saved model file). The two agree to `1e-9` on trained models with
missing values, categorical splits, repeated features, iteration slices, and
multiclass. Hand-worked tiny trees pin down the values the sum property
alone cannot distinguish.

Estimators pickle. The trained model is an opaque handle, so pickling
serializes it with the same versioned format `save()` writes and restores
it on unpickling; everything else is ordinary Python state. That makes
pickle the way to keep a whole estimator, and `save()`/`load()` the way to
keep a model: a model file holds the ensemble, its binning, its split
gains, and its feature names, but neither the class labels nor the
constructor hyperparameters, so a loaded classifier reports `classes_` as
0..n_classes-1. A model saved before format v4 carries no gains and
reports zero gain importance (with a warning), which
`dump_model()["has_split_gain"]` tells apart from a measured zero.

Two documented differences from what a scikit-learn estimator is supposed
to be, both of which are why mojotrees says "scikit-learn style" and does
not claim `check_estimator` passes:

- the estimators forward their shared hyperparameters through `**kwargs`,
  so `get_params()` lists them all but `inspect.signature` does not
- `best_iteration_` is always set, where LightGBM sets it only when early
  stopping ran. With validation it is the iteration the primary metric
  peaked at, which is also the number the model kept whenever early
  stopping was on; without an `eval_set` it is the number of iterations
  trained, `n_estimators` unless the objective converged first. `n_iter_`
  is always that trained count

### Tests and wheels

Two Python suites. `pixi run test-python` runs `python/test_python_api.py`,
which covers training behavior and deliberately needs nothing but the
extension module, so it also runs against a bare wheel install.
`pixi run -e pytest test-estimators` runs `python/tests`, the estimator
suite, under pytest with scikit-learn and pandas; those tests skip
themselves when either is missing.

`pixi run test-wheel` builds a self-contained wheel (`pixi run build-wheel`)
and validates it in two clean venvs: one with the wheel alone, exercising
the stdlib fallback through `packaging/smoke_test.py` and the
dependency-free suite, and one with numpy, pytest, scikit-learn, and
pandas, running the estimator suite against the installed package from a
neutral directory. The wheel bundles the Mojo runtime dylibs the extension
links (delocate-style, with an `@loader_path` rpath), so installing it
requires no Mojo or MAX toolchain.

That wheel carries whatever platform tag your build machine produced, and it
is for you rather than for redistribution. No wheel has been published
anywhere, on any index or any release page. What a published one would be
called, which platforms are targeted, and what evidence stands behind each
row is in [docs/PLATFORM_MATRIX.md](docs/PLATFORM_MATRIX.md); what installing
one will look like is in [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Device selection

The same three values everywhere, in Mojo (`fit(..., device=CPU_DEVICE)`)
and in Python (`MojoTreesRegressor(device="cpu")`):

| `device` | Behavior |
|---|---|
| `cpu` | The default and the dependable backend: Float64, every objective, every entry point |
| `gpu` | Device-resident tree growth. Raises when no accelerator is present, or on a workload the GPU path does not cover, instead of falling back |
| `auto` | The GPU when it is available, covers the workload, and the size heuristic selects it; the CPU otherwise |

The GPU trainers cover every objective that shares the per-row
gradient/hessian interface. `train_gpu` handles single-output training
(squared error, binary logistic, poisson, huber, quantile, L1, gamma,
tweedie, MAPE, fair, cross entropy),
`train_custom_gpu` a caller-supplied gradient callback, and
`train_multiclass_gpu` softmax, growing one tree per class per round
through one device-resident builder. Quantile, L1, and MAPE renew their leaf
values on the host after the tree is grown, exactly as on the CPU.

For the built-in objectives without row sampling, gradients are generated on
the device from device-resident labels and raw scores, and each grown tree
advances those raw scores from its leaf ranges, so nothing per-row crosses
the host/device boundary in a plain round; bagging and GOSS rank their
samples host-side and keep the host gradient path. Per-node split selection
runs on the host over downloaded histograms, which is what keeps CPU and GPU
split decisions identical, on every shape except the largest.
`MOJOTREES_GPU_SPLIT_STRATEGY=device` moves the scan onto the device (one
136-byte record per node instead of the histogram, Float32 gains that can
flip near-tie decisions, bit-deterministic run to run), growing over a
device-resident frontier so that a split builds one histogram and subtracts
for the sibling, and and since 2026-08-16 the automatic policy selects that
path for **every eligible shape** on an observed Apple M4, at any size. It
used to require `normalized_split_work >= 50,000,000`; that threshold and its
depth-wise sibling were withdrawn when the sweep they were owed found no
crossover at any shape from 5.0M to 70.0M, with the device search winning
everywhere and winning by more at the smaller shapes. See
[Where the speed stands](#where-the-speed-stands) for the numbers. The host
scan still runs for a configuration the device kernel cannot express, for a
resident frontier that does not fit, and on hardware nobody has measured --
eligibility, not size.
To repeat or refute any of that, run
`pixi run bench-train-gpu 50000 100 reg 5 gpu-host,gpu-device`, which
alternates the two arms inside one process and prints each arm's own spread
next to the delta between them. Only adjacent samples are comparable on a
machine that throttles, so the harness calls a gap narrower than that spread
indistinguishable instead of reporting it as a number.
Batched prediction can also run on the device through
`Model.predict_batch(..., device=GPU_DEVICE)`; binning stays host-side, so
both devices route every row to the same leaf.

One intentional difference in where training stops. Both trainers end early
when a round produces a single leaf whose value is under 1e-12, meaning the
objective has converged. The CPU sums gradients in Float64 and hits exactly
zero on an already-solved objective; the GPU sums fixed-point gradients, so
rounding leaves a residue of a few quantization units and its leaf value
lands nearer 1e-9. On a degenerate dataset (every feature constant, squared
error already solved by the base score) the CPU therefore stops with no
trees while the GPU emits a few more single-leaf trees first. Those trees
are zero to within the device's gradient resolution and the two models
agree; only the tree count differs. Real data never reaches an exactly zero
gradient sum, so this shows up only in degenerate cases, and
`tests/test_gpu_objectives.mojo` pins it.

`fit_multiclass` routes by `device` exactly as `fit` does, so `gpu` grows
multiclass on the device through `train_multiclass_gpu` rather than
raising, and `auto` may select it. GOSS reaches both trainers
and every objective they cover: the sample is chosen on the host from
Float64 gradients and handed to the device as the tree's row list, so the
two backends sample identically.

`crossover_rules()` holds exactly one rule, and it is the smallest claim
the records support. `auto` selects the GPU when the API is Metal, the
Apple generation is M4, the objective is squared error, the output count is
one, and the shape is at least 1,000,000 rows by 50 features. That rule
rests on two interleaved CPU against GPU records at exactly that shape
(`bench/results/apple_m4_large_scaling_2026-08-14.md` and the 2026-08-15
section of [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md)), and every
other backend, Apple generation, objective, smaller shape, and multiclass
falls through to the CPU because nothing measured them.

Below that shape the CPU is the measured winner rather than merely the
untested one: at 250,000 x 50 it is 1.66s against the GPU's 1.89s and at
50,000 x 50 it is 0.564s against 1.63s, because the device carries about
1.5s of fixed cost per fit. The crossover itself lies somewhere in
(250,000, 1,000,000] rows and has not been bracketed, so the rule's floor
sits at the top of that interval.

Two things about how `auto` identifies the hardware. The API and Apple
generation come from the accelerator this binary was *compiled for*, through
the comptime `_accelerator_arch()`, because probing costs a device open that
a decision cannot afford; the decision reports `profile_source=build-target`
and carries `WARN_BUILD_TARGET_HARDWARE`, since a wheel built for one chip
and run on another will believe it is still on the first. A caller holding an
open `DeviceContext` can pass what it read to
`decide_device_report_reported`, and that reading outranks the build target.

And a defaulted `fit(device="auto")` still gets the CPU at every shape, for
one remaining reason: every rule is scoped to the objective it was measured
on, and `resolve_device`'s four-argument form declares no objective, so it
cannot inherit a squared-error measurement. That gate is deliberate.
`resolve_device_full`, `decide_device_report`, and the Python
`device_selection.select_device` all take an objective and all reach the rule
today; the trainer entry points hold one and do not yet pass it.

`MOJOTREES_AUTO_MIN_CELLS` is the escape hatch, an
integer cell count (`n_rows * n_features`) at or above which `auto`
chooses the GPU. A run that reaches the GPU through it is reported with
`EVIDENCE_ENV` and a warning, never as a validated choice.
`MOJOTREES_DISABLE_GPU=1` makes the library
report no accelerator, so `gpu` raises and `auto` chooses the CPU on a
machine that has one; it pins a mixed fleet to the CPU and exercises the
unavailable-GPU path in tests.

Fitted Python estimators record the backend that actually ran on
`device_`. The device is a training choice rather than part of the model,
so the serialization format is unchanged and a loaded estimator carries no
`device_`.

LightGBM difference: LightGBM spells this `device_type` with `cpu`, `gpu`,
and `cuda`, and has no `auto`. mojotrees has one portable GPU backend
rather than separate OpenCL and CUDA ones, so `gpu` covers every supported
accelerator, and `auto` is an addition.

### GPU histogram scaling

Histogram accumulation launches a 2D grid: `grid.x` is the active feature,
`grid.y` a tile of rows. A device therefore gets `n_active * n_tiles`
threadgroups rather than one per feature, which is what lets a wide GPU stay
busy on a dataset with few features. Each threadgroup accumulates a partial
histogram for its (feature, row tile) in shared memory, reading only the
target node's own rows: a device-resident, order-preserving permutation of
the row indices keeps every live leaf's rows in one contiguous range, so a
node's histogram costs that node's rows rather than a scan of the whole
dataset, and the launch grid is sized per node from the rows it actually
owns.

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
capabilities (`src/mojotrees/gpu_tiling.mojo`), not fixed at compile time,
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

Four environment variables override the policy, for benchmarking and for
tests that must force one path, matching the `MOJOTREES_` contract in
`parallel.mojo`:

| Variable | Effect |
|---|---|
| `MOJOTREES_GPU_HIST_STRATEGY` | `atomic` or `tiled` forces that strategy; `auto` or unset lets the policy decide |
| `MOJOTREES_GPU_ROW_TILE` | rows per tile, still clamped to the memory budget |
| `MOJOTREES_GPU_BLOCK_THREADS` | threads per threadgroup, still clamped to the device maximum and rounded to a warp |
| `MOJOTREES_GPU_SPLIT_WIDE` | `1` scans each feature's bins on a threadgroup rather than on one thread; unset keeps the one-thread scan. Ignored for a dataset with any categorical feature, whose search the wide kernel does not implement |

These are tuning and test knobs rather than model parameters: they change
how a histogram is computed, never what it equals, so they are deliberately
absent from the Python API and from the serialization format. The same is
true of the wide scan: it returns the one-thread scan's records bit for bit,
which `tests/test_gpu_split_search.mojo` asserts, and it is off by default
only because no benchmark has priced it.

`pixi run bench-hist-scaling` reports the two strategies side by side with
kernel time separated from conversion, upload, download, and setup time. Every
measurement taken so far is Apple Metal; see
[docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md) for what has and has not
been run.

LightGBM difference: LightGBM's GPU histogram builder is written against
specific vendor toolchains and tunes itself with vendor-specific constants.
mojotrees has one kernel source for every backend and moves all
device-specific choice into the tiling policy, which is why the strategy and
tile size are runtime values here rather than compile-time ones.

## Roadmap

1. Connect what is already written. The repository has grown faster than
   its call graph: several capability families are implemented, compile,
   and are reached by no entry point at all, including packed-bin GPU
   layout, the sparse and categorical GPU kernels,
   DART and random forest, CEGB, and LightGBM model file interop.
   Hybrid CPU and GPU leaf placement left this list by deletion rather than
   by connection on 2026-08-16: the device-resident tree plane beats the
   host path at every measured shape, so there is no leaf left for the host
   to usefully take.
   Class-batched multiclass rounds have since gained a call site behind
   `MOJOTREES_GPU_CLASS_BATCH`, and batching seven classes measured
   indistinguishable from the sequential schedule (15.45 against 15.30
   seconds), so that one is reached, measured, and not worth turning on. The
   current list, with what blocks each one, is
   [docs/INTEGRATION_INVENTORY.md](docs/INTEGRATION_INVENTORY.md).
   Registering the four auxiliary modules in `bindings/` is the single
   largest item in it
2. Close the remaining v1 gaps in the parity contract
   ([docs/LIGHTGBM_PARITY.md](docs/LIGHTGBM_PARITY.md)). Sparse input
   landed for training and prediction; the gaps left there are a reachable
   sparse GPU kernel, custom objectives, and `eval_set` from Python.
   `device="auto"` now has one measured crossover rule; what it still needs
   is a device profile a hardware-scoped rule can match, because
   `DeviceCapabilities.detect()` opens no device and so the rule cannot fire
   through `resolve_device`
3. Validate the same GPU source on NVIDIA and AMD hardware
   ([procedure](docs/GPU_VALIDATION.md); neither has been run). The kernels
   already scale past one threadgroup per feature and tile themselves from
   device capabilities, but every measurement so far is Apple Metal
4. Expand the published wheel matrix beyond the validated CPython 3.14,
   Apple Silicon, macOS 26+ alpha. Linux needs a measured glibc floor before
   any `manylinux` claim, and additional Python versions require their own
   build and clean-install evidence. See
   [docs/INSTALLATION.md](docs/INSTALLATION.md) and
   [docs/PLATFORM_MATRIX.md](docs/PLATFORM_MATRIX.md)
5. Broader benchmark suite (XGBoost and real datasets)
6. R bindings on top of the C ABI in `capi/`, which exists so that the R
   package (and any other language binding) never has to track a mojotrees
   internal layout

## Defaults

Matched to LightGBM so comparisons are apples to apples. Not "mostly
matched": every default below is LightGBM's own stock value, read from
`include/LightGBM/config.h` at tag v4.7.0, and `tools/check_parity.py`
asserts each one against that table on every run. A default that drifts off
stock fails a gate rather than being noticed later in a result.

`lambda_l2` was the last exception. It defaulted to 1.0 here against
LightGBM's 0.0 until 2026-08-16, which meant every benchmark had to pin
`lambda_l2` on both sides to compare the two libraries at all, and an arm
labelled "LightGBM stock" was not on LightGBM's regularizer. It is 0.0 now.
**Fits that did not set `lambda_l2` explicitly produce different leaf values
and can take different splits than they did before that date**, because the
parameter sits in the denominator of the Newton step `-T(G) / (H + l2)`.

| Parameter | Default |
|---|---|
| `num_leaves` | 31 |
| `max_depth` | -1 (unlimited, LightGBM's default) |
| `learning_rate` | 0.1 |
| `n_estimators` | 100 |
| `min_data_in_leaf` | 20 |
| `max_bin` | 255 |
| `lambda_l2` | 0.0 (LightGBM's default; was 1.0 here until 2026-08-16) |
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
| `monotone_constraints` | none (LightGBM's default) |
| `use_missing` | `True` (LightGBM's default) |

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

### Growth policy

`grow_policy` selects how a tree spends its leaf budget. It is XGBoost's
parameter (LightGBM has no equivalent), with XGBoost's spellings:

| value | growth |
| --- | --- |
| `leafwise` (default; alias `lossguide`) | LightGBM's: the highest-gain leaf anywhere in the tree splits next |
| `depthwise` | every eligible leaf at one depth splits before any deeper one, so the tree fills level by level |

`num_leaves` stays a hard bound under both. A depth-wise level that would
overrun it is admitted as its highest-gain prefix (ties broken by node id),
so at the default `num_leaves=31` with unlimited `max_depth` a depth-wise
tree fills four levels and half of a fifth; set `max_depth` deliberately
for depth-wise runs, since it is the natural control there. Leaves that
run out of rows or gain drop out of a level independently, so depth-wise
trees can still be ragged at the bottom, only never deeper on one branch
while a shallower leaf still had a split to offer.

The mode changes only which leaf is split next: partitioning, sibling
subtraction, leaf values, and every constraint go through the same code, so
the CPU dense and sparse growers and all three GPU growers make the same
choice on the same inputs (`tests/test_grow_policy.mojo`). The distributed
prototype rejects it.

On the GPU, half of the launch batching now ships: the device-resident split
search commits a planned level and searches it in one launch pair, so a
level costs one host wait instead of one per split, about 5 waits per tree
rather than 30. The row partition and the histogram build are still per
node (`docs/design/GPU_LEVELWISE.md` describes what remains). Batching only
exists on the device split search, and since the size threshold was withdrawn
on 2026-08-16 every eligible shape gets it; a fit the policy declines on
eligibility grounds still does not.

That is why depth-wise is the fastest GPU arm this project has measured.
**Measured** on an Apple M4, 1,000,000 x 50, five repeats: 2.587 seconds
against leaf-wise's 3.756 and LightGBM's 2.767 on ten threads, the only
shape where this library is ahead of LightGBM on training time. Read
[Where the speed stands](#where-the-speed-stands) before quoting that
anywhere, because it compares our depth-wise trees against LightGBM's
leaf-wise ones, and because no accuracy measurement of any kind has been
taken on a depth-wise arm. Choose `depthwise` because you want depth-wise
trees, and if you choose it for the speed, measure your own loss.

```python
MojoTreesRegressor(grow_policy="depthwise", max_depth=6, num_leaves=64)
```

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
MojoTreesRegressor(
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
what makes a run reproducible from the seed alone. On the GPU the bag seeds
the root's device-side row range directly: rows outside the bag are simply
not inside any leaf's range, so no kernel ever iterates them and a bagged
tree costs one staged Int32 copy of the bag.

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
MojoTreesRegressor(
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
  start their streams at nearby states. mojotrees draws from the same
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
MojoTreesRegressor(
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
  before it. mojotrees keys an independent counter-based splitmix64 stream on
  `(feature_fraction_seed, tree index, node id)`, so a selection is
  reproducible per tree and per node regardless of history or backend, and
  sets do not match LightGBM's at equal seeds. The count formula and the
  selection algorithm (Knuth's Algorithm S, as in LightGBM's
  `Random::Sample`) are the same, so the distribution matches
- LightGBM builds each node's histograms over that node's by-node set;
  mojotrees builds over the tree's set and applies the by-node set to the
  split search. Split decisions are identical, and building the superset is
  what keeps histogram subtraction exact when the by-node sets differ
  between a parent and its children
- with interaction constraints also configured, LightGBM draws the by-node
  set from the features the branch already allows; mojotrees draws from the
  tree's set and then applies the allow mask, so a node can end up with fewer
  candidates than LightGBM would give it. Both restrict to the same features;
  only how many survive the draw differs

Feature importance follows the sampled sets by construction: a tree can only
credit features its draw allowed, which is what spreads split and gain
importance onto features an unsubsampled ensemble would never reach.
Subsampling is a training parameter only; it changes which trees get built,
not the model format.

### Missing values

`NaN` is the missing marker for numerical features, and it is routed rather
than compared. A feature whose training column holds any `NaN` reserves one
extra bin for them, above its ordinary bins, at the cost of one bin out of the
`max_bin` budget. `NaN` is dropped before the quantile edges are fit and
routed to the reserved bin before any binary search runs, so it never takes
part in an ordinary quantile comparison.

At every node the split search scores each threshold twice, once with the
missing rows in the left child and once with them in the right, and stores the
winner as that node's default direction. Training, raw prediction, binned
prediction, and GPU row partitioning all apply that one rule, so a missing row
follows the same path whichever of them is running. Because missingness is
routed rather than imputed, a pattern of missingness that predicts the target
is learnable on its own.

```python
X = np.array([[1.0], [np.nan], [3.0]])
model = MojoTreesRegressor().fit(X, y)
model.predict(np.array([[np.nan]]))     # follows each node's default direction
```

Set `use_missing=False` to switch it off, in which case every `NaN` is binned
as the value 0.0.

| Value | Treated as |
|---|---|
| `NaN`, feature had missing values in training | missing: the reserved bin, routed by the node's default direction |
| `NaN`, feature had none in training | the value 0.0, matching LightGBM's `missing_type=None` |
| `NaN`, `use_missing=False` | the value 0.0 |
| `+inf` | the feature's highest ordinary bin, never missing |
| `-inf` | bin 0, never missing |

The infinity rows describe the Mojo core. The Python estimators are stricter
and reject a non-finite `X` value other than `NaN` outright, so an `inf`
raises there rather than binning.

Bin edges are clamped to +/-1e300 (LightGBM's `Common::AvoidInf`), which is
what keeps an infinite training value from producing an infinite edge.

Differences from LightGBM, all deliberate:

- LightGBM's `zero_as_missing` is not implemented; only `NaN` is missing.
- Categorical features have no reserved missing bin. They keep the rule in
  `categorical.mojo`, where `NaN`, negative codes, and unseen codes share bin
  0 and route right, which is LightGBM's `CategoricalDecision` behavior.

`pixi run -e bench compare-missing` is the reproducible comparison: it prints
mojotrees's and LightGBM's routing decisions side by side on the same data.
Against LightGBM 4.7 every one of them matches.

### Categorical features

Mark the columns whose integer codes are unordered categories with
`categorical_feature` (LightGBM's parameter name; the plural
`categorical_features` is accepted as an alias). It takes column indices,
column names, a mix of the two, or LightGBM's default `"auto"`:

```python
model = MojoTreesRegressor(categorical_feature=[0, 3]).fit(X, y)
model = MojoTreesRegressor(categorical_feature=["city", "device"]).fit(df, y)
model = MojoTreesRegressor().fit(df, y)   # "auto": every category column
```

or, from Mojo:

```mojo
var model = fit(
    features, n_rows, n_features, target, SQUARED_ERROR, params,
    categorical_features=[0, 3],
)
```

The parameter lives on the constructor rather than on `fit`, unlike
LightGBM, because scikit-learn's clone contract keeps hyperparameters on the
estimator. It works the same on `MojoTreesRegressor`,
`MojoTreesClassifier`, and `MojoTreesRanker`, and on the Mojo `fit`,
`fit_multiclass`, `fit_custom`, and `fit_ranker`.

#### pandas categorical columns

`"auto"`, the default, means every pandas `category` column of `X` and
nothing else; on a numpy matrix that is no columns, so an estimator that
never sees a frame behaves exactly as it would have. A `category` column is
encoded by its **labels**, and the label table is kept on the fitted
estimator, so a prediction frame that orders or extends its categories
differently still lands on the categories the model was fitted with:

```python
train = pd.DataFrame({"city": pd.Categorical(["nyc", "sfo", ...])})
model = MojoTreesRegressor().fit(train, y)
model.categorical_feature_          # [0]
model.predict(other_frame)          # 'nyc' is 'nyc' whatever its code there
```

Two consequences worth stating outright:

- A `category` column left out of an explicit `categorical_feature` raises.
  LightGBM quietly feeds its codes to the numerical scan, and a declared
  category is the one thing that must never become an ordered number.
- Only a frame carries labels. A model fitted on `category` columns raises
  rather than predict on a plain array, and a model fitted on integer codes
  raises rather than read a frame's own codes. Pickle keeps the label
  tables; `save()` / `load()` does not, because the model file holds the
  category tables but not the labels (`load()` still restores
  `categorical_feature_`, so a loaded model splits and routes exactly as it
  did, on codes).

#### Category codes

Outside a pandas `category` column, the values of a categorical feature are
integer codes:

| Value | Treated as |
|---|---|
| a code seen in training | its own category bin |
| a code not seen in training | unseen: bin 0, routes right |
| a code dropped for cardinality | unseen: bin 0, routes right |
| any negative value | missing: bin 0, routes right |
| `NaN` | missing: bin 0, routes right |
| a fractional value | rejected by the Python estimators; truncated toward zero by the Mojo core |
| `>= 2**31` | rejected: codes must stay representable as Int32, as in LightGBM |

The Python estimators check the last two at fit **and** at predict, rather
than leaving them to `bin_of`, which truncates a fractional code and reads an
oversized one as unseen: either would answer a caller who encoded the column
differently than they did at fit with a prediction instead of an error.

Those columns skip quantile binning entirely. Their distinct codes are
collected at fit time, sorted ascending, and mapped to bins 1..k; bin 0 is
reserved. A node splitting such a feature holds a *set* of category bins that
route left, searched with LightGBM's algorithm: one-vs-rest when the feature
has at most `max_cat_to_onehot` categories, otherwise categories are ordered
by `sum_grad / (sum_hess + cat_smooth)` and prefixes of that order are
accumulated from both ends, up to `max_cat_threshold` per side. Categorical
and numerical candidates then compete on gain in the same search, so nothing
about a numerical column changes when a categorical one is added.

**Default routing.** Bin 0 collects three kinds of row, and none of them is
ever a member of a split set, so all three take the **right** branch at every
categorical node, matching LightGBM's `CategoricalDecision`:

- missing values: a negative code, or `NaN`
- unseen categories: codes absent from the training column
- dropped categories: codes present in training but not kept, when the column
  had more distinct codes than `max_bins - 1`

**Intentional differences from LightGBM.**

- Row counts are exact. LightGBM estimates a bin's count from its Hessian sum
  because its histograms carry no counts; mojotrees's do, so
  `min_data_in_leaf`, `min_data_per_group`, and the `cat_smooth` count filter
  use exact counts.
- One-vs-rest is selected on the number of categories, matching the
  documented meaning of `max_cat_to_onehot`, rather than on an internal bin
  count that may or may not include the unknown bin.
- LightGBM keeps the most frequent categories covering 99% of rows and drops
  categories below `min_data_in_bin`; mojotrees keeps the `max_bins - 1` most
  frequent and drops nothing else, leaving rare categories to the
  `cat_smooth` filter during split search.
- LightGBM's `kEpsilon` (1e-15) Hessian nudges are omitted.
- Monotonic constraints on a categorical feature are rejected rather than
  silently applied.

**Both backends.** Categorical features are not a CPU-only feature. The GPU
trainer searches category partitions on the same downloaded histograms and
routes rows by the node's 256-bit set in its partition kernel, so a
categorical model trains on either device and the two agree to the Float32
tolerance every other backend-equivalence test uses
(`test_gpu_categorical_splits_match_cpu` in tests/test_categorical.mojo,
which skips itself without an accelerator). The paths that are CPU-only are
CPU-only for other reasons: ranking and Python objective or metric callbacks
raise for `device="gpu"` before training starts, whether or not a feature is
categorical. Multiclass is no longer among them; it trains on the device
through `train_multiclass_gpu` and, at 465,000 rows by 54 features over
7 classes, wins by 1.63x
([`bench/results/profile_2026-08-15/RESULTS.md`](bench/results/profile_2026-08-15/RESULTS.md)).

`bench/compare_categorical_lightgbm.py` fits both libraries on the same
matrix, with and without the columns marked categorical, and reports held-out
RMSE for all four:

```
pixi run build-python
pixi run -e bench compare-categorical
```

### Feature interaction constraints

`interaction_constraints` is a list of feature groups. Two features may
appear together on the same root-to-leaf path only if a single group holds
both, and transitively everything else already on that path. That is what
restricts a model to additive or low-order-interaction structure.

```python
MojoTreesRegressor(interaction_constraints=[[0, 1], [2, 3]]).fit(X, y)
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

### Monotonic constraints

`monotone_constraints` takes one entry per numerical feature: `1` for
nondecreasing, `-1` for nonincreasing, `0` for unconstrained.

```python
MojoTreesRegressor(monotone_constraints=[1, 0, -1]).fit(X, y)
```

```mojo
var monotone = MonotoneConstraints.from_signs([1, 0, -1], 3)
var params = BoosterParams(
    100,
    0.1,
    TreeParams(31, 20, 1.0, 1e-3, 0.0, monotone=monotone^),
)
```

The guarantee is global and exact. For any two examples differing only in a
constrained feature, the model's predictions are ordered the way the
constraint says, at any raw feature value and not merely on the training
data. It holds on the response scale too, since the logistic and poisson
links are increasing.

This is LightGBM's `monotone_constraints_method="basic"`. Each node carries
an interval its output must lie in; a split on a constrained feature divides
that interval between its children at the midpoint of their values, so bounds
only tighten with depth, and every leaf value is clamped into its own
interval. Candidates are scored from clamped outputs, and a candidate whose
children would run against the constraint is discarded whatever its gain.
Two examples that differ only in feature `f` diverge at some node splitting
on `f`, and every leaf below its low child is capped at that node's midpoint
while every leaf below the high child is floored at it, which is what makes
the ordering hold for the whole tree, and so for a positively weighted sum
of trees.

Notes and deliberate differences:

- LightGBM decides constraints are in play by checking that the vector is
  non-empty; mojotrees also treats an all-zero vector as inactive, so split
  search keeps its unconstrained path and the fit is bit-identical to one
  with no vector at all (there is a test for exactly this)
- only the `basic` method is implemented; LightGBM's `intermediate` and
  `advanced` methods buy back some of the accuracy `basic` gives up, and are
  not attempted here
- quantile and L1 rewrite every leaf value after the tree is grown, which
  knows nothing about monotonicity, so mojotrees clamps the renewed value
  back into its leaf's interval. That keeps the guarantee for those
  objectives at the cost of biasing the renewed percentile. We have not
  verified LightGBM's behavior at this step, so treat the clamp as
  mojotrees-defined rather than matched
- for multiclass, constraints apply to every per-class tree, so each class's
  **raw score** is monotone. Softmax probabilities are **not** guaranteed
  monotone, because a class's probability also moves with the other classes'
  scores
- a categorical feature cannot carry a nonzero constraint; the combination is
  rejected rather than ignored
- the CPU and GPU trainers share the split search, the interval bookkeeping,
  and the leaf clamping, so the constraint is enforced identically on both;
  only the histogram sums the decisions are made from carry the GPU's Float32
  precision
- unlike the other training-time restrictions, the constraint vector is
  recorded in the model file, because it is a property the fitted trees
  satisfy rather than only a knob that shaped them. The section is written
  only when there is a vector to write, so files for unconstrained models are
  unchanged, and a file without it loads as unconstrained

### Custom objectives

An objective is a callable that takes the current raw predictions and the
labels and fills gradient and hessian arrays. In Mojo it is a compile-time
callable parameter, so `train_custom` specializes on your callable and the
per-round call is direct and inlinable, with nothing dynamic in the boosting
loop:

```mojo
from mojotrees import BoosterParams, TreeParams, mean_label, train_custom

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
from mojotrees import MojoTreesRegressor

def squared_error(raw, y):
    return raw - y, np.ones_like(raw)

model = MojoTreesRegressor(
    objective=squared_error, base_score="mean", n_estimators=100
).fit(X, y)
pred = model.predict(X)          # raw scores: apply your own link
```

`base_score` accepts a number or `"mean"` for the weighted label mean.
`MojoTreesClassifier` takes no objective and says so; use the regressor and
apply your own link. The Python callback is convenient, not free:
`bench/bench_custom_objective.py` measures it, and on 100,000 rows x 20
features x 100 rounds (Apple M4, best of 3) it added 8.9 ms per round, 36%
over the same fit with the built-in objective, for bit-identical
predictions. That is a fixed per-round cost plus a per-row copy, so it
shrinks as a fraction of a bigger fit and dominates a small one. Use the
native Mojo interface when the objective is on a hot path.

### Custom validation metrics

A custom metric scores validation predictions against validation labels and
returns one scalar. Metrics are independent of objectives: a built-in
objective can be watched by custom metrics (`train_with_metrics`), a custom
objective can be watched by them (`train_custom_with_metrics`), and either
can still use the older single-loss path (`train_with_valid`,
`train_custom_with_valid`).

```mojo
from mojotrees import (
    SQUARED_ERROR, BoosterParams, TreeParams, CustomMetric, MetricSuite,
    ValidSet, rmse, train_with_metric, train_with_metrics,
)

def my_rmse(pred: List[Float64], target: List[Float64]) raises -> Float64:
    return rmse(pred, target)

var valid_sets: List[ValidSet] = [ValidSet("holdout", valid_data, valid_y)]
var result = train_with_metric(
    data, target, valid_sets, SQUARED_ERROR, params, "rmse", my_rmse,
    early_stopping_rounds=10,
)
result.booster            # truncated to the best round
result.best_iteration     # trees kept
result.history.series(0, 0)   # the metric, round by round
```

Several metrics go through one dispatching callable, because two Mojo
callables of the same signature are still different types and cannot share
a `List`:

```mojo
def evaluate(
    metric: Int, valid: Int, pred: List[Float64], y: List[Float64]
) raises -> Float64:
    if metric == 0:
        return my_rmse(pred, y)
    return binary_auc(pred, y)

var suite = MetricSuite(
    [CustomMetric("rmse"), CustomMetric("auc", higher_is_better=True)],
    evaluate,
    primary=1,
)
```

The contract, and where it intentionally differs from LightGBM:

- **predictions are raw scores**, before any inverse link, as LightGBM's
  `feval` receives them: log-odds for binary logistic, log-mean for
  poisson. `response_scale(objective, raw)` converts a vector when a metric
  wants probabilities
- **the direction is declared, not returned.** LightGBM's `feval` returns
  `(name, value, is_higher_better)` per call, so the direction is only
  known after the first evaluation. Here `higher_is_better` belongs to
  `CustomMetric`, which is what lets the primary metric and the
  early-stopping set be validated before training starts
- **metric values must be finite.** A NaN or infinity raises, naming the
  metric and the round. A NaN would otherwise look like an unending run of
  non-improving rounds, since every comparison against it is False.
  LightGBM does not check
- **early stopping watches every (validation set, flagged metric) pair**
  and stops as soon as one of them goes `early_stopping_rounds` rounds
  without improving, which is LightGBM's behavior. The ensemble is then
  truncated to the best round of the *primary* metric on the *first*
  validation set, where LightGBM truncates to the best iteration of the
  pair that triggered the stop: which model you keep should not depend on
  which pair ran out of patience first
- **a tie is not an improvement**, and with a nonzero `min_delta` an
  improvement has to clear it, in whichever direction the metric runs.
  This is LightGBM's rule
- **`early_stopping_rounds = 0` disables stopping** and keeps recording:
  every metric still runs every round, the full ensemble comes back, and
  `best_iteration` still reports where the primary metric peaked
- **the history starts at round 0**, the base-score-only model, so
  `value(i, ...)` is the score after `i` trees. LightGBM's `evals_result_`
  starts at the first iteration
- **CPU only**: there is no GPU trainer with a validation loop

`train_multiclass_with_metrics` and `train_ranker_with_metrics` extend the
same machinery to the softmax and LambdaRank trainers, with the same
early-stopping, truncation, and history rules. A round is one tree per class
for multiclass, and its metrics receive row-major raw scores,
`pred[r * n_classes + k]`; a ranking metric holds its validation set's own
query boundaries, since the trainer has no use for them.

From Python, pass `eval_set` and friends to `fit`, in LightGBM's spelling:

```python
model = MojoTreesRegressor(n_estimators=500).fit(
    X, y,
    eval_set=[(X_valid, y_valid)],   # or (X_valid, y_valid), or eval_X/eval_y
    eval_names=["holdout"],          # valid_0, valid_1, ... by default
    eval_sample_weight=[w_valid],
    eval_metric=["l2", "l1"],
    early_stopping_rounds=20,
)
model.best_iteration_        # where the primary metric peaked
model.n_iter_                # rounds actually trained
model.best_score_            # the primary metric's best value
model.evals_result_["holdout"]["l2"]   # index 0 is the base score alone
model.stopped_early_
```

`eval_metric` takes LightGBM's metric names, callables, or both, and
defaults to the objective's own loss. The names are `l2`, `rmse`, `l1`,
`quantile`, `huber`, `binary_logloss`, `binary_error`, `auc`,
`multi_logloss`, `multi_error`, and `ndcg`, plus LightGBM's aliases
(`mse`, `mae`, `l2_root`, `binary`, `multiclass`, ...). Each belongs to one
task, so `auc` on a regressor is an error rather than a number nobody can
read; their values come from `src/mojotrees/metrics.mojo`, so Python and
Mojo cannot disagree, and `eval_sample_weight` weights them.

A callable is `f(y_true, y_pred) -> float` (LightGBM's
`(name, value, is_higher_better)` return is accepted, and only its value
read), called once per metric per validation set per round with raw scores
in `y_pred`. Declare its direction with `("name", f, True)`, or pass a dict,
`{"name": ..., "func": ..., "higher_is_better": ..., "early_stopping":
...}`, which is how a metric is recorded without letting it stop training.
A callable is handed unweighted predictions, so combining one with
`eval_sample_weight` raises rather than dropping the weights quietly.

The classifier encodes validation labels through the `classes_` it recorded,
so a label absent from training raises; the ranker takes `eval_group`, one
group array per validation set, and rejects `eval_sample_weight` because
NDCG has no weighted LightGBM definition to match. Combining a Python
*objective* callback with validation is not wired up yet; the Mojo API pairs
them with `train_custom_with_metrics`.

### Training callbacks

`fit(..., callbacks=[...])` runs code around every boosting round, with
LightGBM's contract, so a callback list written for LightGBM runs here:

```python
from mojotrees import MojoTreesRegressor
from mojotrees.callback import (
    early_stopping, log_evaluation, record_evaluation, reset_parameter,
)

history = {}
model = MojoTreesRegressor(n_estimators=500).fit(
    X, y,
    eval_set=[(X_valid, y_valid)],
    eval_metric="l2",
    callbacks=[
        reset_parameter(learning_rate=lambda i: 0.1 * 0.995 ** i),
        log_evaluation(period=25),
        record_evaluation(history),
        early_stopping(20),
    ],
)
```

A callback is any callable taking one `CallbackEnv`, the same namedtuple
LightGBM passes: `model`, `params`, `iteration`, `begin_iteration`,
`end_iteration`, and `evaluation_result_list`. Callbacks with a truthy
`before_iteration` attribute run before the round's tree is grown, the rest
after it is grown and scored, and within each group they run in ascending
`order`, ties in the order you listed them.

Callbacks need an `eval_set`: the hook lives in the trainer that scores
validation metrics. They run for the regressor and the binary classifier;
the softmax and LambdaRank trainers have no hook yet and refuse a callback
list rather than accepting it and doing nothing. `early_stopping_rounds=`
works for every task either way.

`reset_parameter` schedules the nine hyperparameters the loop re-reads each
round (`mojotrees.callback.RESETTABLE`); anything else raises, rather than
being ignored the way LightGBM ignores it. A learning-rate schedule cannot
be represented by the single rate `Booster` applies at predict time, so the
shrinkage is baked into the leaf values instead and the stored rate becomes
1.0, which is what LightGBM does for every model. That switch happens only
once a schedule actually moves the rate: without one, the model is
bit-identical to the same fit with no callbacks.

`early_stopping()` configures the trainer's own stopper rather than
reimplementing the rule in Python, so it agrees with
`fit(early_stopping_rounds=...)` exactly; passing both raises. A callback
can also stop a run by raising `EarlyStopException`, which rolls the
ensemble back to the best round as LightGBM does. Any other exception
propagates with its own type, and the estimator is left unfitted rather than
half-updated.

The training loop is Mojo, so a callback costs one crossing of the Python
boundary per phase per round and nothing per row;
`bench/bench_callbacks.py` measures that and asserts the crossing count.
With no callbacks the bridge does not cross the boundary at all. See
`python/mojotrees/callback.py` for the full contract and
`src/mojotrees/callback.mojo` for the loop's side of it, including
`train_with_callbacks` for native callers.

### Learning to rank

LambdaRank, LightGBM's `objective="lambdarank"`. Ranking data is a row
matrix plus query boundaries: rows of one query are the documents retrieved
for one search, and the label is graded relevance, an integer in [0, 30]
with 0 meaning irrelevant. Rows of a query must be contiguous.

```python
from mojotrees import MojoTreesRanker, group_from_query_ids, ndcg_score

# `group` is LightGBM's: the number of rows in each query, in row order.
model = MojoTreesRanker(n_estimators=100).fit(X, y, group=[6, 4, 9])
scores = model.predict(X)                  # rank each query by these
print(model.score(X, y, group=[6, 4, 9]))  # mean NDCG@5

group = group_from_query_ids(qids)         # raises on a split-up query
print(ndcg_score(scores, y, group, at=10))
```

```mojo
from mojotrees import BoosterParams, groups_from_counts, ndcg, train_ranker

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

`src/mojotrees/ranking.mojo` documents the intentional differences from
LightGBM (an exactly evaluated pairwise sigmoid instead of a lookup table,
a fixed `label_gain`, query-level bagging, and no unbiased-lambdarank
extensions). `pixi run -e bench compare-ranking` checks the NDCG metric
against LightGBM's on identical scores and compares the two libraries'
ranking quality on held-out queries.

## C API

`capi/` is a small C ABI over the same trainer, meant as the base for
bindings in any language that speaks C. Full documentation, including the
parameter string both it and the CLI take, is in
[capi/README.md](capi/README.md); the header
[capi/mojotrees.h](capi/mojotrees.h) is the contract.

```c
MojoTreesError *err = mojotrees_error_create();
MojoTreesModel *model = NULL;
if (mojotrees_train_dense(x, n_rows, n_features, y, NULL,
                          "objective=binary num_iterations=200",
                          &model, err) != MOJOTREES_OK) {
    fprintf(stderr, "%s\n", mojotrees_error_message(err));
}
mojotrees_predict(model, x, n_rows, n_features, pred, n_rows, err);
mojotrees_save_model(model, "model.mbst", err);
mojotrees_model_free(model);
mojotrees_error_free(err);
```

Only C scalars, C strings, caller-owned buffers, and opaque handles cross
the boundary, so no mojotrees type is exposed and internal layouts can
change without breaking a compiled caller. Hyperparameters travel as a
LightGBM style parameter string rather than a struct for the same reason.
Errors go to an explicit error object instead of a thread-local global,
which is a deliberate difference from LightGBM's `LGBM_GetLastError`, so
concurrent use is well defined.

```sh
pixi run build-capi     # capi/libmojotrees.{dylib,so}
pixi run test-capi      # Mojo tests: ABI matches the Mojo API exactly
pixi run test-c         # C tests: lifecycle, invalid input, handle churn
```

## Command line tool

`cli/` trains and predicts from text files, so a model can be fit and used
without writing code. The data format, column roles, and exit statuses are
documented in [cli/README.md](cli/README.md).

```sh
pixi run build-cli
cli/mojotrees train --data train.csv --model model.mbst \
    --params "objective=binary num_iterations=200"
cli/mojotrees predict --model model.mbst --data test.csv --output pred.csv
cli/mojotrees info --model model.mbst
```

Data files are comma separated numbers, one example per line, with `#`
comments, an optional header, and an empty field or `nan`/`na`/`?` for a
missing value. `--label` and `--weight` name column indices; everything
else is a feature, in file order.

## Development

Requires [pixi](https://pixi.sh).

```sh
pixi install
pixi run test          # all suites
pixi run test-cpu      # everything that needs no accelerator
pixi run test-gpu      # the accelerator subset
pixi run test-list     # what would run, without running it
```

`pixi run test` builds the package once with `mojo precompile` and then runs
`tests/test_*.mojo` through a bounded pool of processes, reporting every
failure rather than stopping at the first. `tools/run_tests.sh` is the runner
and documents the knobs; `MOJOTREES_TEST_JOBS` is the one most people want.

During ordinary development, run only the smallest test file covering the
change. The full suite is an integration or release check and should not be
launched repeatedly or concurrently on a shared development machine. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the focused-test workflow.

The test command includes CPU/GPU equivalence checks. They run when a
supported accelerator is present and skip cleanly on CPU-only machines.

```sh
pixi run check-parity     # or: python3 tools/check_parity.py
```

`check-parity` holds [docs/LIGHTGBM_PARITY.md](docs/LIGHTGBM_PARITY.md) to
the code: it fails when a row that claims support is deleted or downgraded,
when the contract cites a file that no longer exists, when a public Python or
Mojo symbol those rows depend on disappears, when a test suite the
contract offers as evidence is not run by any task, or when a row claims a
capability is out of reach after its name landed in `mojotrees.__all__`. It
builds nothing and needs only the standard library, so it also runs in CI.

```sh
python3 tools/connectivity_audit.py    # what is connected to what
python3 tools/audit_integration.py     # is the written inventory still true
```

Those two answer a different question: not "is this claim about LightGBM
right" but "does anything call this code at all". The first computes the
import graph from the four entry points and reports orphan modules, unused
imports, duplicate policies, public parameters nothing reads, and native
functions Python asks for that no binding exports. The second checks
[docs/INTEGRATION_INVENTORY.md](docs/INTEGRATION_INVENTORY.md) against that
graph. Neither imports mojotrees or builds anything, so both run on a bare
checkout.

## Benchmarks

Reproducible from `bench/` (methodology, exact parameters, and caveats in
[bench/README.md](bench/README.md)). Both drivers generate bit-identical
synthetic data from the same splitmix64 stream and train with matched
parameters.

The current numbers, and the honest reading of them, are under
[Where the speed stands](#where-the-speed-stands) above. The records behind
them are
[`bench/results/sweep2_2026-08-15/RESULTS.md`](bench/results/sweep2_2026-08-15/RESULTS.md)
and
[`bench/results/profile_2026-08-15/RESULTS.md`](bench/results/profile_2026-08-15/RESULTS.md).
In one line: our per-row cost is even with LightGBM's, our fixed cost is
about a second behind, our leaf-wise GPU arm is 1.36x behind at 1,000,000 x
50, our depth-wise GPU arm is 6.5 percent ahead of LightGBM at that shape
while growing a different tree and with its accuracy unmeasured, and we are
3.2x ahead on binning.

The table below is the original single-threaded baseline at a different
shape and a different revision. It is kept because it is what several
earlier statements in this repository were derived from, not because it
describes the library today. 100,000 rows x 100 features, 100 rounds,
Apple M4:

| | mojotrees (1 thread) | LightGBM (1 thread) |
|---|---|---|
| Regression: training | 3.53 s | 2.41 s |
| Regression: binning | 0.55 s | 0.81 s |
| Regression: train MSE | 0.003615 | 0.003797 |
| Binary: training | 3.50 s | 2.32 s |
| Binary: train logloss | 0.267034 | 0.267168 |

That baseline was within 1.5x of single-threaded LightGBM on training and
faster at binning, before multicore histogram accumulation. The serial ratio
measured since, at 1,000,000 x 50, is 1.81x rather than 1.5x, so do not
carry the 1.5x forward to any other shape.

```sh
pixi run bench                 # mojotrees
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
