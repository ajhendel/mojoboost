# Five minutes with mojotrees on Apple Silicon

**Native gradient-boosted trees accelerated by the GPU already inside every
Apple Silicon Mac.**

### Read this before you quote that line

That headline is the goal of the Apple Silicon work, and part of it is real
today. mojotrees trains a complete model on the Metal GPU through
`device="gpu"`, with device resident tree growth and bit deterministic
histograms.

The acceleration is now measured, and it is real at one end of the size
range and absent at the other. On an Apple M4, 100 rounds, 31 leaves, 255
bins, squared error, seconds of training with binning excluded, median of
three interleaved arms
(`bench/results/profile_2026-08-15/RESULTS.md`):

| shape | our CPU | our GPU |
| --- | --- | --- |
| 1,000,000 x 50 | 6.98 | **3.58** |
| 250,000 x 50 | **1.66** | 1.89 |
| 50,000 x 50 | **0.564** | 1.63 |

So the GPU is 1.85x the CPU at a million rows and loses to it below that,
because it carries roughly 1.5 seconds of fixed cost per fit that does not
scale with rows. Multiclass is the clearer win: 15.30s against the CPU's
25.47s at 465,000 rows by 54 features over 7 classes, which is 1.63x.

Two things this does not say. It does not say mojotrees is broadly fast
against LightGBM, which at 1,000,000 x 50 trains the same model in about 2.8s
and is still ahead of both the backends in that table. There is now one
exception, and it is narrow: at that shape `grow_policy="depthwise"` on the
GPU trains in 2.587s against LightGBM's 2.767s, which is a real measured win
at a 0.3 percent spread over five repeats
(`bench/results/sweep2_2026-08-15/RESULTS.md`), but depth-wise growth builds a
different tree than LightGBM's leaf-wise growth, nobody has measured whether
that tree is as accurate, and our own leaf-wise arm is still behind at every
shape. Read the caveats in the README's "Where the speed stands" before
repeating it. And it says nothing about any Apple part other than this one
M4. `TIMINGS.md` next to this file is the per-machine table, and every cell
in it is still empty until somebody runs it on their own hardware.

With that in hand, treat the headline as accurate at the large end and as a
statement of direction at the small end, and use the tour below for what it
actually demonstrates, which is a working GPU capable GBDT that runs on the
Mac you already own.

The whole tour is one script.

```sh
PYTHONPATH=python python examples/apple_silicon/five_minute_tour.py
```

It uses only the standard library, so it runs in the default pixi environment
with no numpy installed, and it finishes in well under five minutes on a
laptop. `--time` adds one optional fit per available backend at the end.

---

## What you need

- An Apple Silicon Mac (M series). Intel Macs are covered under
  [troubleshooting](#unsupported-hardware).
- A macOS release supported by the pinned MAX and Mojo versions in
  `pixi.toml`. The wheel this repository builds is tagged
  `macosx_26_0_arm64`, taken from the Mojo toolchain's own minimum OS
  version, so an older macOS needs a source install or an older toolchain.
- The Metal toolchain, for the GPU path only. The CPU path needs nothing
  beyond pixi.

```sh
xcodebuild -downloadComponent MetalToolchain
```

---

## Install

### Placeholder, not published yet

```sh
pip install mojotrees          # PLACEHOLDER. Not on PyPI. Do not run.
```

The package is not published. The name is not reserved, no release exists,
and there is no manylinux build, so nothing installs this way today. This
block becomes real at the first release, and not before.

### Placeholder, a wheel exists but is not distributed

```sh
pip install mojotrees-0.1.0-cp314-cp314-macosx_26_0_arm64.whl   # PLACEHOLDER
```

`pixi run build-wheel` really does build a self contained arm64 wheel with
the MAX runtime dylibs bundled, and `pixi run test-wheel` validates it in
clean virtual environments. There is no published artifact to download, so
this is a placeholder for a release asset that does not exist yet.

### Works today, from source

```sh
git clone https://github.com/mojotrees/mojotrees.git
cd mojotrees
pixi install
pixi run build-python
PYTHONPATH=python python examples/apple_silicon/five_minute_tour.py
```

---

## Step 1. Does this build have a GPU

```python
import mojotrees

mojotrees.gpu_available()
```

`True` means an accelerator was present when the extension was compiled and
`MOJOTREES_DISABLE_GPU=1` is not set. Two things about that answer are worth
knowing before you rely on it.

It is a property of the build and not of the machine. Mojo resolves
`has_accelerator()` at compile time, so a wheel built on a Mac with a GPU
reports one wherever it is installed. On a redistributed build the mismatch
surfaces when the device is actually opened rather than when the request is
resolved.

It is also the switch for pinning a mixed fleet. `MOJOTREES_DISABLE_GPU=1`
makes this build report no accelerator at all, so `device="gpu"` raises and
`device="auto"` chooses the CPU on a machine that does have one.

## Step 2. `device="auto"`

```python
from mojotrees import MojoTreesRegressor

model = MojoTreesRegressor(n_estimators=100, device="auto").fit(X, y)
model.device_          # the backend that actually ran
```

Estimators take `device="cpu"`, `device="gpu"`, or `device="auto"`, and
`device_type` is accepted as the LightGBM spelling of the same parameter.
Fitting records the backend that ran on `device_`.

`auto` resolves to the CPU today, and the reason is worth getting right
because it is no longer "there is no evidence". One crossover rule is
installed, scoped to Metal on an M4 for unweighted squared error at
1,000,000 rows by 50 features and above, on the records quoted at the top of
this file. What stops it firing is that `DeviceCapabilities.detect()` opens
no device, so the machine is reported as unidentified and a rule scoped to
an Apple generation cannot match it. `auto` therefore warns and keeps the
CPU, even on the M4 the rule was measured on. That is a wiring gap and it is
written down rather than hidden.

`MOJOTREES_AUTO_MIN_CELLS` is the escape hatch: a number of cells
(`n_rows * n_features`) at or above which `auto` should pick the GPU. `0`
means whenever the GPU path covers the workload. A GPU chosen through it is
reported as resting on the environment rather than on evidence.

`device="gpu"` either runs on the GPU or raises. It never falls back
quietly, which is why an unexpected CPU fit is not something that can happen
behind your back.

## Step 3. Validation and early stopping

```python
model = MojoTreesRegressor(n_estimators=300, learning_rate=0.05, device="cpu")
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_names=["holdout"],
    eval_metric=["l2", "l1"],
    early_stopping_rounds=20,
)
model.n_iter_, model.best_iteration_, model.best_score_, model.stopped_early_
model.evals_result_["holdout"]["l2"]
```

Metric names are LightGBM's and are computed by `src/mojotrees/metrics.mojo`,
so they agree with the Mojo API by construction. `evals_result_[set][metric][i]`
is the score after `i` trees, so index `0` is the base score alone.

Note the device in that snippet. Validation is scored on the CPU, so an
`eval_set` together with `device="gpu"` raises rather than falling back. Early
stopping on the GPU path is one of the capabilities listed in the handoff.

## Step 4. Predict

```python
model.predict(X_valid)
model.predict(X_valid, raw_score=True, num_iteration=model.best_iteration_)
```

`predict` takes LightGBM's prediction keywords, `raw_score`,
`start_iteration`, `num_iteration`, `pred_leaf`, `pred_contrib`, and
`validate_features`. With `num_iteration=None` it uses `best_iteration_`,
which is the ensemble early stopping left behind.

**NOT AVAILABLE YET.** Prediction always runs on the CPU. There is no
`device` argument on `predict` and no GPU prediction kernel wired into the
estimator.

```python
model.predict(X_valid, device="gpu")     # NOT IMPLEMENTED
```

## Step 5. Why that device

The tour prints the inputs that decide the backend, in the vocabulary of
`src/mojotrees/device.mojo`.

| Value | Meaning |
| --- | --- |
| `cpu` | The dependable path. Float64 throughout, every objective, every entry point. |
| `gpu` | Device resident tree growth. Raises when no accelerator is present or when the workload is outside the GPU path. Never falls back silently. |
| `auto` | The GPU when the complete GPU path covers the workload and the size heuristic selects it, the CPU otherwise. The heuristic is off by default. |

What the GPU path covers today is single output training with squared error,
binary logistic, poisson, huber, quantile, and L1, and multiclass. Multiclass
grows one tree per class per round through a device resident builder, so
`device="gpu"` trains it on the accelerator rather than raising, and on an M4
it is the shape the GPU wins by the largest margin.

A structured, machine readable version of that explanation, with the
crossover rules it applied and the reason for the answer, is available. It
never raises: a request that would fail comes back with `would_raise=True`
and the message in `error`.

```python
from mojotrees import explain_device_choice

report = explain_device_choice(X, y, device="auto")
report.to_dict()["resolved"]        # "cpu" or "gpu"
[r.code for r in report.reasons]    # the stable reason codes
report.validated                    # True only for a GPU chosen by a rule
```

## Step 6. Save and load

```python
model.save("model.mbst")
restored = MojoTreesRegressor.load("model.mbst")
restored.predict(X_valid)        # identical to the original, bit for bit
```

The file holds the model and not the estimator, in mojotrees's versioned text
format. Hyperparameters, feature names, split gains, and the training device
do not travel with it, and a loaded estimator has no `device_` for that
reason. The trained ensemble is the same whichever backend produced it.
Pickle the estimator when you want everything else, including the class
labels a classifier learned.

## Step 7. The timing table

```sh
PYTHONPATH=python python examples/apple_silicon/five_minute_tour.py --time
```

That prints wall clock seconds for one CPU fit and, where the build has an
accelerator, one GPU fit, on this machine and this dataset. One run of one
shape on one machine is not a benchmark, and the script says so.

The real table is [`TIMINGS.md`](TIMINGS.md). Every cell in it is empty and
stays empty until an actual run fills it in under the measurement protocol.
No number in this directory may come from anywhere else.

---

## Troubleshooting

### The Metal toolchain is missing

`Metal Compiler failed to compile metallib` during a GPU run means the Metal
toolchain component is not installed. This happened on the development Mac
and is the single most likely first failure.

```sh
xcodebuild -downloadComponent MetalToolchain
xcodebuild -runFirstLaunch      # sometimes needed afterwards
```

Xcode or the Command Line Tools must be selected first
(`xcode-select --install`, then `sudo xcode-select -s /Applications/Xcode.app`
if you have the full Xcode).

### macOS is too old

A wheel built by this repository is tagged `macosx_26_0_arm64` because that
is the Mojo toolchain's minimum OS version, and pip refuses it on anything
older. Install from source with pixi, which resolves the pinned toolchain,
or pin an older MAX and Mojo. The GPU path additionally depends on whatever
macOS version the pinned MAX release supports for Metal, so check the Modular
release notes rather than assuming this example's requirements are the whole
story.

### Unsupported hardware

Intel Macs have no Metal backend in this project. `gpu_available()` returns
`False`, `device="gpu"` raises, and `device="auto"` and `device="cpu"` train
normally. Everything in this tour except Step 7's GPU row runs on an Intel
Mac and on Linux.

Discrete GPUs from other vendors are a separate question that no measurement
in this repository answers. What has actually been exercised on which device,
and which device attributes Metal refuses to answer at all, is recorded in
[`docs/GPU_VALIDATION.md`](../../docs/GPU_VALIDATION.md). Read that before
assuming anything about hardware other than the Apple M4 it was written from.

### `device="gpu"` raised and I wanted it to just work

That is deliberate. An explicit GPU request either runs on the GPU or raises,
so a silent CPU fit cannot be mistaken for an accelerated one. The message
names the reason. Common ones are no accelerator in this build, an
`eval_set` (validation scores on the CPU), a sparse matrix from the Python
estimators, and a Python objective callback. Multiclass used to be on this
list and is not any more; it trains on the device.

Ask for `device="auto"` when you want the library to choose, or `device="cpu"`
when you want the dependable path.

### I want the CPU on a machine that has a GPU

```sh
export MOJOTREES_DISABLE_GPU=1
```

This build then reports no accelerator at all. It is also how the unavailable
GPU path gets exercised in tests.

### The CPU path feels single threaded

`MOJOTREES_NUM_WORKERS` controls it. `1` forces serial, `N > 1` forces the
work into at most that many chunks whatever its size, and `0` or unset means
automatic. `MOJOTREES_PARALLEL_MIN_OPS` overrides the amount of work below
which the automatic path stays serial. Both reach every parallel dispatch in
the library, which is binning, histogram accumulation, gradient generation,
row partitioning, and prediction.

Neither changes a result. Every dispatch keeps floating point summation order
independent of the task count, so output is bit identical to the serial path
at every worker setting. They are for reproducible benchmarking and for
pinning a machine, not for tuning accuracy.

### `ModuleNotFoundError: mojotrees`

The extension module is built into `python/mojotrees`, so a source checkout
needs `pixi run build-python` and `PYTHONPATH=python`. From an installed
wheel neither is needed, and running from inside the checkout root can
shadow the installed package.

### numpy is not installed

Nothing here needs it. The estimators fall back to `array.array` and lists,
and this example is written that way on purpose so it runs in the default
pixi environment. numpy arrays, pandas frames, and SciPy sparse matrices all
work when they are present.

---

## What is not integrated yet

| Shown as a placeholder | State |
| --- | --- |
| `pip install mojotrees` | Not published. No PyPI release, no downloadable wheel. |
| `predict(..., device="gpu")` | Implemented. `predict`, `predict_proba`, and the ranker's `predict` take `device=`, and the device entry points are registered in the extension. Contributions and sparse input have no device path and refuse an explicit `"gpu"`. This row is retired. |
| `explain_device_choice(X)` | Implemented and re-exported from `mojotrees`. This row is retired. |
| `device="auto"` choosing the GPU | One measured rule exists and cannot fire, because `DeviceCapabilities.detect()` opens no device and a hardware-scoped rule cannot match an unidentified profile. `MOJOTREES_AUTO_MIN_CELLS` is the escape hatch until a trainer passes the profile it reads. |
| Early stopping with `device="gpu"` | Raises. Validation is scored on the CPU. |
| Sparse input on the GPU from Python | The estimators keep the CPU. `train_gpu_sparse` exists and is reachable from the Mojo API; its crossover is unmeasured, so `auto` keeps the CPU there too. |
| Any Apple GPU speedup claim beyond one M4 | Measured on exactly one machine, and `TIMINGS.md` is empty for every other. On that machine the GPU is 1.85x our CPU at 1,000,000 x 50 and slower than it below about a million rows. |

Every row above is a real limit today, checked against the code rather than
guessed at. The rows disappear from this table as the work lands, and this
example is worth pointing anyone at once the first four are gone.
