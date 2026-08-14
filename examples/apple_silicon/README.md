# Five minutes with mojoboost on Apple Silicon

**Native gradient-boosted trees accelerated by the GPU already inside every
Apple Silicon Mac.**

### Read this before you quote that line

That headline is the goal of the Apple Silicon work, and part of it is real
today. mojoboost trains a complete model on the Metal GPU through
`device="gpu"`, with device resident tree growth and bit deterministic
histograms. What is not established is the acceleration. The only end to end
Apple measurement in this repository (Apple M4, `bench/bench_train_gpu.mojo`)
came out **slower** than the multicore CPU trainer at every shape that was
tried, so `device="auto"` deliberately resolves to the CPU and this example
makes no speed claim anywhere. `TIMINGS.md` next to this file is the table
that would change that, and every cell in it is empty until somebody runs it.

Until those cells are filled, treat the headline as a statement of direction
and use the tour below for what it actually demonstrates, which is a working
GPU capable GBDT that runs on the Mac you already own.

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
pip install mojoboost          # PLACEHOLDER. Not on PyPI. Do not run.
```

The package is not published. The name is not reserved, no release exists,
and there is no manylinux build, so nothing installs this way today. This
block becomes real at the first release, and not before.

### Placeholder, a wheel exists but is not distributed

```sh
pip install mojoboost-0.1.0-cp314-cp314-macosx_26_0_arm64.whl   # PLACEHOLDER
```

`pixi run build-wheel` really does build a self contained arm64 wheel with
the MAX runtime dylibs bundled, and `pixi run test-wheel` validates it in
clean virtual environments. There is no published artifact to download, so
this is a placeholder for a release asset that does not exist yet.

### Works today, from source

```sh
git clone https://github.com/ajhendel/mojoboost.git
cd mojoboost
pixi install
pixi run build-python
PYTHONPATH=python python examples/apple_silicon/five_minute_tour.py
```

---

## Step 1. Does this build have a GPU

```python
import mojoboost

mojoboost.gpu_available()
```

`True` means an accelerator was present when the extension was compiled and
`MOJOBOOST_DISABLE_GPU=1` is not set. Two things about that answer are worth
knowing before you rely on it.

It is a property of the build and not of the machine. Mojo resolves
`has_accelerator()` at compile time, so a wheel built on a Mac with a GPU
reports one wherever it is installed. On a redistributed build the mismatch
surfaces when the device is actually opened rather than when the request is
resolved.

It is also the switch for pinning a mixed fleet. `MOJOBOOST_DISABLE_GPU=1`
makes this build report no accelerator at all, so `device="gpu"` raises and
`device="auto"` chooses the CPU on a machine that does have one.

## Step 2. `device="auto"`

```python
from mojoboost import MojoBoostRegressor

model = MojoBoostRegressor(n_estimators=100, device="auto").fit(X, y)
model.device_          # the backend that actually ran
```

Estimators take `device="cpu"`, `device="gpu"`, or `device="auto"`, and
`device_type` is accepted as the LightGBM spelling of the same parameter.
Fitting records the backend that ran on `device_`.

`auto` resolves to the CPU today. The size heuristic that would send large
workloads to the GPU is disabled by default, because no measurement on any
device has established a shape where the GPU trainer wins and a shipped
crossover threshold would be a performance claim with nothing behind it. To
run the experiment that would justify one, set `MOJOBOOST_AUTO_MIN_CELLS` to
a number of cells (`n_rows * n_features`) at or above which `auto` should
pick the GPU. `0` means whenever the GPU path covers the workload.

`device="gpu"` either runs on the GPU or raises. It never falls back
quietly, which is why an unexpected CPU fit is not something that can happen
behind your back.

## Step 3. Validation and early stopping

```python
model = MojoBoostRegressor(n_estimators=300, learning_rate=0.05, device="cpu")
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

Metric names are LightGBM's and are computed by `src/mojoboost/metrics.mojo`,
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
`src/mojoboost/device.mojo`.

| Value | Meaning |
| --- | --- |
| `cpu` | The dependable path. Float64 throughout, every objective, every entry point. |
| `gpu` | Device resident tree growth. Raises when no accelerator is present or when the workload is outside the GPU path. Never falls back silently. |
| `auto` | The GPU when the complete GPU path covers the workload and the size heuristic selects it, the CPU otherwise. The heuristic is off by default. |

What the GPU path covers today is single output training with squared error,
binary logistic, poisson, huber, quantile, and L1. Multiclass grows one tree
per class per round on the CPU only, so `device="gpu"` raises for it and
`auto` chooses the CPU.

**NOT AVAILABLE YET.** A structured, machine readable version of that
explanation, with the crossover rules it applied and the reason for the
answer, is separate work.

```python
from mojoboost import explain_device_choice     # NOT IMPLEMENTED
report = explain_device_choice(X)               # NOT IMPLEMENTED
```

## Step 6. Save and load

```python
model.save("model.mbst")
restored = MojoBoostRegressor.load("model.mbst")
restored.predict(X_valid)        # identical to the original, bit for bit
```

The file holds the model and not the estimator, in mojoboost's versioned text
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
names the reason. Common ones are no accelerator in this build, a multiclass
objective, an `eval_set` (validation scores on the CPU), a sparse matrix
(there is no sparse GPU kernel), and a Python objective callback.

Ask for `device="auto"` when you want the library to choose, or `device="cpu"`
when you want the dependable path.

### I want the CPU on a machine that has a GPU

```sh
export MOJOBOOST_DISABLE_GPU=1
```

This build then reports no accelerator at all. It is also how the unavailable
GPU path gets exercised in tests.

### The CPU path feels single threaded

`MOJOBOOST_NUM_WORKERS` controls it. `1` forces serial, `N > 1` forces the
work into at most that many chunks whatever its size, and `0` or unset means
automatic. `MOJOBOOST_PARALLEL_MIN_OPS` overrides the amount of work below
which the automatic path stays serial. Both reach every parallel dispatch in
the library, which is binning, histogram accumulation, gradient generation,
row partitioning, and prediction.

Neither changes a result. Every dispatch keeps floating point summation order
independent of the task count, so output is bit identical to the serial path
at every worker setting. They are for reproducible benchmarking and for
pinning a machine, not for tuning accuracy.

### `ModuleNotFoundError: mojoboost`

The extension module is built into `python/mojoboost`, so a source checkout
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
| `pip install mojoboost` | Not published. No PyPI release, no downloadable wheel. |
| `predict(..., device="gpu")` | Not implemented. Prediction is CPU only. |
| `explain_device_choice(X)` | Not implemented. |
| `device="auto"` choosing the GPU | Implemented but disabled. Needs `MOJOBOOST_AUTO_MIN_CELLS`, and a measured crossover before any default changes. |
| Early stopping with `device="gpu"` | Raises. Validation is scored on the CPU. |
| Multiclass on the GPU | Raises. One tree per class per round is CPU only. |
| Sparse input on the GPU | Raises. There is no sparse GPU kernel. |
| Any Apple GPU speedup claim | Unmeasured except on one M4, where the GPU trainer was slower. `TIMINGS.md` is empty. |

Every row above is a real limit today, checked against the code rather than
guessed at. The rows disappear from this table as the work lands, and this
example is worth pointing anyone at once the first four are gone.
