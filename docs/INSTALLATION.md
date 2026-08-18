# Installing mojotrees

> [!IMPORTANT]
> mojotrees is an experimental public alpha. It trains, predicts, saves, and
> loads today, and people can use it for real work on the platform below.
> It is not yet a drop-in production replacement for LightGBM or XGBoost.
> The first wheel is published for the narrow platform described below. Read
> [docs/LIGHTGBM_PARITY.md](LIGHTGBM_PARITY.md) for what the library
> promises and [docs/GPU_VALIDATION.md](GPU_VALIDATION.md) before believing
> anything about an accelerator.

The first public alpha is available from PyPI:

```sh
pip install --pre mojotrees
```

The current release is `0.1.0a2`. Its wheel supports **CPython 3.14 on Apple
Silicon running macOS 26 or newer**. No Mojo installation, MAX installation,
Pixi environment, or compiler is required to use that wheel.

## Three installation paths

| State | What you type | Status today |
|---|---|---|
| 1. Published alpha from PyPI | `pip install --pre mojotrees` | **Available** for CPython 3.14, Apple Silicon, macOS 26+ |
| 2. A release wheel file | `pip install ./mojotrees-<version>-<tags>.whl` | **Available** from the release workflow |
| 3. Source checkout with Pixi | `git clone`, `pixi install`, `pixi run build-python` | **Available** for contributors and unsupported targets |

Pick by what you are trying to do.

- You use CPython 3.14 on an Apple Silicon Mac with macOS 26 or newer. State
  1 is the ordinary install path.
- You use another interpreter or platform. No matching public wheel exists
  yet; use state 3 if the pinned Mojo toolchain supports your system.
- You are contributing, or you want the Mojo API, the C ABI, or the CLI.
  State 3 is the only one that gives you those at all.

One thing that will never appear on this page is a command that looks like a
plain pip install and quietly turns into a Mojo compile on a machine with no
Mojo toolchain. mojotrees publishes no source distribution, deliberately, so
pip cannot fall back to building from source and cannot fail an hour later
with a compiler error. See
[Why there is no sdist](#why-there-is-no-sdist).

---

## State 1. Published alpha from PyPI

Install the published binary wheel directly from PyPI.

```sh
python -m pip install --pre --only-binary=:all: mojotrees
```

`--only-binary=:all:` is not decoration. It tells pip to refuse anything that
is not a prebuilt wheel, which is exactly the guarantee this project wants to
make, and it makes an install fail fast and legibly on a platform we do not
publish for.

What that install contains.

- The `mojotrees` Python package and one compiled extension module,
  `_mojotrees`, built from the same Mojo sources as the rest of the library.
- The Mojo and MAX runtime libraries the extension links, bundled inside the
  wheel with loader-relative paths. **No Mojo installation, no MAX
  installation, no conda, and no compiler are required to run it.**
- Nothing else. numpy is optional, scikit-learn is optional, scipy is never
  imported.

What it requires today.

| Requirement | Value today | Where it is decided |
|---|---|---|
| Python | 3.10 | `requires-python` in `python/pyproject.toml`, measured in [docs/PYTHON_SUPPORT.md](PYTHON_SUPPORT.md) section 10 |
| Platform | macOS on Apple silicon first, Linux x86_64 and aarch64 after that | [docs/PLATFORM_MATRIX.md](PLATFORM_MATRIX.md) |
| numpy | optional | plain Python sequences work without it; `pip install "mojotrees[numpy]"` pulls it in |

The Python floor is 3.10, and [docs/PYTHON_SUPPORT.md](PYTHON_SUPPORT.md)
is where it was established rather than assumed. Run M1a there took one
unmodified extension, built under CPython 3.14.6, and imported it under
3.14, 3.13, 3.12, 3.11 and 3.10. `test_python_api.py` passed on all five,
and `gpu_available()` returned true on each, so the Metal path is not
3.14-specific either. Run M1b puts the floor exactly at 3.10: 3.9 aborts
during module initialization on `symbol not found: Py_NewRef`, which CPython
added in 3.10, and `mojo 1.0.0` and `mojo-python 1.0.0` independently
require `python >=3.10`.

Two things that measurement did not change. The wheel published today is
labeled `cp314` because a 3.14 compiled it, there is no `abi3` build, and one
wheel serves exactly one CPython minor version, so **an index install on 3.10
through 3.13 needs a wheel that does not exist yet**; pip matches the tag
before it reads `requires-python`. And every interpreter above was exercised
on osx-arm64 only.

An earlier version of this section said `Py_GetConstantBorrowed`, a CPython
3.13 addition present in the extension's entry point table, left 3.12 and
earlier blocked. That was a static prediction, and M1a contradicted it.
Section 9.1 of PYTHON_SUPPORT.md records why the entry point table is not
the floor.

### What `pip install mojotrees` does right now

Nothing, and that is deliberate. `0.1.0a2` is a **pre-release**, and pip
ignores pre-releases unless you ask for one. Plain `pip install mojotrees`
reports `No matching distribution found` on every machine, including a
supported one, because the only version on the index is an alpha.

To install the alpha, ask for it by name:

```sh
pip install --pre mojotrees        # newest pre-release
pip install mojotrees==0.1.0a2     # or pin it exactly
```

On the supported target that downloads and installs `0.1.0a2`. On any other
target, pip reports `No matching distribution found` because no compatible
wheel exists. That refusal is intentional; pip never falls back to compiling
Mojo source.

This is the property that makes claiming the name safe. Nobody who types
`pip install mojotrees` on a whim ends up running an experimental alpha they
did not ask for. When the first final version ships, plain
`pip install mojotrees` starts working and this section goes away.

### Why there is no sdist

A source distribution would let pip download mojotrees on any machine, start
a build, and then fail deep inside a toolchain the user never asked for and
does not have. The compile needs Mojo and MAX from a pinned Pixi environment,
so a source build on an ordinary Python machine cannot succeed, and an
install that fails after eight minutes of confusing output is worse than one
that refuses in a second.

So `pip install mojotrees` will only ever resolve to a wheel that matches the
machine, or to a clean "no matching distribution found". If you want to build
from source, that is state 3, and it is a deliberate, documented act rather
than a silent fallback.

---

## State 2. Install a release wheel file

This is the path for anyone who wants the artifact before or without PyPI,
or who mirrors artifacts internally.

The macOS release workflow has produced and published the first alpha
artifact through PyPI trusted publishing. Release workflows also produce a
wheel, SHA-256 manifest, provenance, and software bill of materials from a
tagged commit. Linux artifacts remain unvalidated and unpublished.

### Pick the wheel for your exact platform

A wheel filename is a promise about the machine it runs on, and pip enforces
it. There is one wheel per row, and no row is a near enough match for another.

| Your machine | The wheel | Where that stands |
|---|---|---|
| Apple silicon Mac, Python 3.14 | `mojotrees-0.1.0a2-cp314-cp314-macosx_26_0_arm64.whl` | Published and clean-install validated |
| Linux x86_64, Python 3.14 | `mojotrees-0.1.0a2-cp314-cp314-linux_x86_64.whl` | Matrix row `linux-x86_64-cp314`, not index-publishable. See below |
| Linux aarch64, Python 3.14 | `mojotrees-0.1.0a2-cp314-cp314-linux_aarch64.whl` | Matrix row `linux-aarch64-cp314`, not index-publishable. See below |
| Intel Mac | none, and there will not be one | The pinned channel ships no Intel macOS toolchain |
| Windows | none | No toolchain in the pinned channel |
| Free-threaded Python (`3.14t`) | none | A different ABI tag; the extension cannot load |

Read the filename left to right. `cp314` is the interpreter,
`macosx_26_0_arm64` is the operating system floor and the processor
architecture. If either half does not describe your machine, that wheel will
not install, and forcing it is a way to turn a clean refusal into a crash at
import time.

The macOS floor of 26.0 is a real constraint, not a typo, and it is higher
than it should be. It comes from the SDK on the build machine rather than
from the code, and lowering it is a known, scoped piece of work described in
[docs/PLATFORM_MATRIX.md](PLATFORM_MATRIX.md).

**The Linux tag is the open question, and it is not cosmetic.** The Linux
release workflow defaults to a `plain` tag policy, which produces
`linux_x86_64` and `linux_aarch64`. PyPI and TestPyPI reject those tags
outright, so a plain wheel cannot be published to an index at all; it can
only be handed to named testers. Worse, a musl system matches the plain tag,
so pip there accepts the wheel and the import then fails, because the objects
need glibc. Promoting to a `manylinux_2_28` tag is a deliberate, separate
decision that requires measuring the highest `GLIBC_` symbol version the
shipped objects actually reference, not inheriting the floor Pixi happened to
solve against. Until that measurement exists, treat the Linux rows above as
artifact names rather than as an install anyone should be pointed at.

That decision is now written down rather than left implicit. The plain wheels
are matrix rows `linux-x86_64-cp314` and `linux-aarch64-cp314`, both carrying
`publishable = false`; the promoted wheels are the separate rows
`linux-x86_64-cp314-manylinux` and `linux-aarch64-cp314-manylinux`, which the
same builder produces under `tag_policy=manylinux`. The workflow's TestPyPI job
refuses to run on a plain tag, so the rule above is enforced rather than merely
recommended.

### Install it

```sh
# 1. Verify the download against the checksum manifest the release run
#    published. Both platforms publish a manifest rather than a bare digest,
#    so check it with -c and let the tool compare, instead of reading two
#    hex strings side by side.
#    macOS:  SHA256SUMS, from packaging/macos/hash_artifacts.sh
#    Linux:  SHA256SUMS-x86_64.txt or SHA256SUMS-aarch64.txt, one per
#            architecture, written next to the wheels by release-linux.yml
shasum -a 256 -c SHA256SUMS                    # macOS
sha256sum -c SHA256SUMS-x86_64.txt             # Linux x86_64

# 2. Install into a fresh virtual environment, from the file itself.
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install ./mojotrees-0.1.0a2-cp314-cp314-macosx_26_0_arm64.whl

# 3. Confirm it imports and can train, from a directory that is not a
#    mojotrees checkout, so a stray source tree cannot make this pass.
cd ~
python -c "import mojotrees; print(mojotrees.__version__, mojotrees.__file__)"
```

Installing the file path rather than the package name is the point. pip
either accepts that exact wheel for this interpreter and platform or refuses
it, and there is nothing for it to substitute.

### Uninstalling

```sh
python -m pip uninstall mojotrees
```

The bundled runtime libraries live inside the package directory, so they go
with it. Nothing is installed outside the environment, and nothing is
registered with the system.

---

## State 3. Source checkout with Pixi

This is the contributor and unsupported-target path. It needs [Pixi](https://pixi.sh) and
about a gigabyte of toolchain, and it gives you the Python API, the Mojo API,
the C ABI, the CLI, the tests, and the benchmarks. You do not need to install
Mojo or MAX separately; Pixi resolves the exact versions this repository pins.

```sh
git clone https://github.com/mojotrees/mojotrees.git
cd mojotrees
pixi install
pixi run build-python
```

`pixi run build-python` compiles the CPython extension into
`python/mojotrees/_mojotrees.so`. It is the step that takes the time, and it
has to be rerun whenever the Mojo sources under `src/` or `bindings/` change.

The package is then importable with the `python` directory on the path.

```sh
PYTHONPATH=python python -c "import mojotrees; print(mojotrees.__version__)"
```

If you would rather not set `PYTHONPATH` on every command, install the built
package into a virtual environment in editable-ish fashion by building a
wheel from the same checkout.

```sh
pixi run build-wheel      # writes python/dist/*.whl
```

That wheel is self-contained in the same way a release wheel would be, and
installing it into a plain virtual environment is the closest thing to a
preview of state 2. It carries whatever platform tag your build machine
produced, so it is for you and not for redistribution.

### What state 3 does not give you

- A redistributable artifact. The checkout is a build environment; use the
  release workflow for published wheels.
- Any accelerator guarantee. Whether the GPU path is compiled in at all is
  decided on the machine that builds, because Mojo resolves
  `has_accelerator()` at compile time. See
  [docs/DEVICE_SELECTION.md](DEVICE_SELECTION.md).
- Windows. `pixi.toml` declares macOS arm64 and Linux, and the pinned channel
  ships no Windows toolchain.

---

## The first five minutes

Everything below assumes mojotrees imports, by whichever state got you there.
If you are in state 3, prefix each command with `PYTHONPATH=python`.

All of it runs in one script, which prints each step with its result.

```sh
python examples/install_smoke.py            # installed package
PYTHONPATH=python python examples/install_smoke.py   # source checkout
```

The script uses only the standard library, so it runs in the default Pixi
environment where numpy is not installed. Read it alongside this section; it
covers the same ground as the steps below, and its own error handling names
the two import failures a bad install produces.

### 1. Import, and know what you imported

```python
import mojotrees
from mojotrees import MojoTreesRegressor, gpu_available

print(mojotrees.__version__)   # 0.1.0a2
print(mojotrees.__file__)      # where this package actually came from
print(gpu_available())         # True if this build has an accelerator path
```

`mojotrees.__file__` is worth printing once. A source checkout on
`PYTHONPATH` shadows an installed wheel, and the two can be different builds.

### 2. Train a tiny regression

```python
X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]

model = MojoTreesRegressor(n_estimators=20, num_leaves=7, min_data_in_leaf=1)
model.fit(X, y)
print(model.predict([[1.5], [4.5]]))
print(model.device_)           # the backend that actually ran
```

Lists of rows are fine. numpy arrays, pandas frames, polars frames, and SciPy
sparse matrices all work too, and numpy is used for the return type when it
is installed.

### 3. Add a validation set and early stopping

```python
model = MojoTreesRegressor(n_estimators=500, learning_rate=0.05, device="cpu")
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_names=["holdout"],
    eval_metric=["l2", "l1"],
    early_stopping_rounds=20,
)

model.n_iter_            # rounds actually trained
model.best_iteration_    # the round the primary metric peaked at
model.best_score_        # its value there
model.stopped_early_     # whether patience ran out
model.evals_result_["holdout"]["l2"]   # index 0 is the base score alone
```

These are LightGBM's spellings, and they mean what they mean there. The
ensemble is rolled back to `best_iteration_`, so the model you predict with
is the one that scored best rather than the one that trained last.

Validation metrics are scored on the CPU, so `device="gpu"` with an
`eval_set` raises rather than falling back. That is why this step passes
`device="cpu"` explicitly.

### 4. Save and load

```python
model.save("model.mbst")
restored = MojoTreesRegressor.load("model.mbst")
assert list(restored.predict(X_valid)) == list(model.predict(X_valid))
```

The predictions are bit-identical, not merely close; floats are stored as raw
bit patterns. The file holds the model and not the estimator, so
hyperparameters, feature names, and the training device do not travel with
it. Pickle the estimator when you want those.

### 5. Choose a device

```python
MojoTreesRegressor(device="cpu")    # default, dependable, every objective
MojoTreesRegressor(device="gpu")    # accelerator or an exception, never a fallback
MojoTreesRegressor(device="auto")   # picks for you, and picks the CPU unless a rule covers the run
```

`device="gpu"` is a request that gets honored or refused. It never quietly
trains on the CPU while you believe you used the GPU. `device="auto"` picks
the GPU only where a benchmark says it is faster, which today is one rule:
Metal on an Apple M4, unweighted squared error, single output, 1,000,000 rows
by 50 features and above, where the GPU trains in 3.58s against the CPU's
6.98s. Everything below that keeps the CPU, and below that shape the CPU
really is the right answer on this hardware: at 250,000 rows the GPU takes
1.89s against the CPU's 1.66s and at 50,000 it takes 1.63s against 0.564s.
[docs/DEVICE_SELECTION.md](DEVICE_SELECTION.md) has the whole policy,
including what the rule is scoped to and why, and what a redistributed wheel
assumes about the hardware it is running on.

### 6. Print the diagnostics

One call, and it answers most of what an installation bug report would
otherwise need a conversation to establish.

```python
import mojotrees

mojotrees.show_versions()
```

```text
mojotrees 0.1.0a2
  package                /.../site-packages/mojotrees/__init__.py
  install                wheel
  extension              /.../site-packages/mojotrees/_mojotrees.so
  bundled runtime        4 in /.../site-packages/mojotrees/.dylibs
  gpu path compiled in   yes
  gpu_available()        True

python 3.14.0 (CPython)
  executable             /.../bin/python
  platform               macOS-26.0-arm64-arm-64bit
  machine                arm64

optional dependencies
  numpy                  2.3.1
  pandas                 not installed
  ...

environment
  (none set)             MOJOTREES_* and MODULAR_* are unset
```

`mojotrees.build_info()` returns the same facts as a JSON-serializable dict
when you want to attach them to something rather than read them.

Three of those rows are worth knowing how to read.

**`gpu path compiled in`** is the one that cannot be recovered any other way.
Whether an accelerator is usable is decided when the extension is compiled,
not on the machine that runs it, so one wheel carries one answer to every
user who installs it. `gpu_available()` alone cannot tell you which answer
yours has, because it returns `False` both for a build with no GPU path and
for a perfectly good GPU build with `MOJOTREES_DISABLE_GPU=1` set. This row
separates them, and prints `unknown` rather than guessing when the variable
is masking the answer. A `no` here means `device="gpu"` will raise on every
machine this wheel is installed on, including machines with a working
accelerator, and that is a property of the artifact rather than a fault on
your side.

**`install`** is `wheel`, `source`, or `absent`, read off the filesystem. A
`source` install resolves its runtime libraries through an absolute path into
the Pixi environment that built it; a `wheel` carries them inside the package.

**`environment`** lists every `MOJOTREES_*` and `MODULAR_*` variable that is
set, discovered by scanning rather than from a fixed list, so a knob added
after this page was written still shows up in your report.

A `build` block appears above it when the artifact recorded its own
provenance, which is where the Mojo and MAX versions, the git tag, the build
host, and the Metal toolchain would show. No build writes that file yet, so
today the report says so instead, and a source install asks you for
`pixi run mojo --version` in its place.

`show_versions()` adds a short note at the end whenever there is something to
say: a missing runtime library, an installed distribution whose version
disagrees with the imported package (a checkout shadowing a wheel, which is
the most common reason a fix appears not to take effect), or a masked GPU
answer. A clean install prints no notes rather than a row of reassurances.

In a source checkout, add the toolchain version, which the extension does not
carry and `show_versions()` therefore asks you for.

```sh
pixi run mojo --version
```

`examples/install_smoke.py` prints all of this at the top of its output, so
running it and pasting the result is the fastest way to fill in the
Environment field of a bug report.

### 7. Read the two documents that bound the claims

[docs/LIGHTGBM_PARITY.md](LIGHTGBM_PARITY.md) is authoritative wherever a
README and it disagree about behavior.
[docs/GPU_VALIDATION.md](GPU_VALIDATION.md) is the record of what hardware
has actually executed this code, and most of its rows still say **not run**.

---

## When something goes wrong

pip's exact wording moves between versions, so match the shape of the message
rather than the punctuation.

### Unsupported Python

```text
ERROR: Ignored the following versions that require a different python version: 0.1.0a2 Requires-Python >=3.14
ERROR: Could not find a version that satisfies the requirement mojotrees
ERROR: No matching distribution found for mojotrees
```

Your interpreter is older than the declared floor. Check with
`python -c "import sys; print(sys.version)"`. The floor is where it is
because 3.14 is the only interpreter anything here has ever run on, and
because one entry point the extension uses was added in CPython 3.13. It is
not a stylistic preference, it is not a toolchain requirement, and it is
under review rather than fixed forever, with the evidence for each
interpreter in [docs/PYTHON_SUPPORT.md](PYTHON_SUPPORT.md).

Installing a wheel file directly on the wrong interpreter gives the tag
mismatch below instead, because the filename carries `cp314`.

### Wrong architecture, wrong operating system, or too old a macOS

```text
ERROR: mojotrees-0.1.0a2-cp314-cp314-macosx_26_0_arm64.whl is not a supported wheel on this platform.
```

pip compared the filename tags against the machine and they did not match.
The usual causes, in order.

- An Intel Mac. The `arm64` half of the tag does not match, and there will be
  no Intel wheel; the pinned channel ships no Intel macOS toolchain.
- A macOS older than the `macosx_26_0` floor. Check with `sw_vers`.
- A Linux machine handed a macOS wheel, or x86_64 handed an aarch64 wheel.
  Check with `uname -sm`.
- Free-threaded Python. `cp314t` is a different ABI tag from `cp314` and
  cannot load this extension.

Do not force it. `--force-reinstall`, renaming the file, or
`--implicit-namespace-packages` style workarounds turn a refusal that costs
one second into a segfault or an `ImportError` at some later point. The tag
is a statement about the binary inside.

### Missing runtime library

The install succeeded and the import does not.

```text
ImportError: dlopen(.../site-packages/mojotrees/_mojotrees.so, 0x0002):
  Library not loaded: @rpath/libKGENCompilerRTShared.dylib
  Reason: tried: '.../site-packages/mojotrees/.dylibs/libKGENCompilerRTShared.dylib' (no such file)
```

or, on Linux,

```text
ImportError: libKGENCompilerRTShared.so: cannot open shared object file: No such file or directory
```

The extension found no MAX runtime library to link against at load time. A
release wheel bundles those libraries inside the package and points the
extension at them with a loader-relative path, so seeing this from an
installed wheel means the wheel is broken and is worth a bug report with the
full message.

From a source checkout it means something more ordinary. Either the extension
was never built, which produces a different message,

```text
ImportError: cannot import name '_mojotrees' from 'mojotrees' (.../python/mojotrees/__init__.py)
```

and the fix is `pixi run build-python`; or it was built and you are running
it outside the Pixi environment that owns the runtime libraries, in which
case run through `pixi run`, or build a self-contained wheel with
`pixi run build-wheel` and install that instead.

Those two are worth telling apart, because they arrive as the same exception
type and mean opposite things. A missing `_mojotrees.so` is reported as
"cannot import name", not as "no module named", because the import machinery
swallows the underlying `ModuleNotFoundError` for a submodule named in a
`from . import ...` list and the attribute lookup fails afterward
(`_handle_fromlist` in `importlib/_bootstrap.py`). A `.so` that exists but
cannot resolve its runtime libraries fails later, inside `dlopen`, and keeps
the loader's own message. `examples/install_smoke.py` distinguishes them and
prints the right fix for each.

### GPU requested and unavailable

```text
RuntimeError: device 'gpu' requested but no accelerator is available; use device 'cpu' or 'auto'
```

The build has no accelerator path in it, or `MOJOTREES_DISABLE_GPU=1` is set.
`mojotrees.show_versions()` tells you which, on the `gpu path compiled in`
row; `gpu_available()` alone cannot, because it returns `False` for both.

The part that surprises people is that availability is a property of the
build and not of the machine. Mojo resolves `has_accelerator()` at compile
time, so a wheel built where no accelerator was visible reports none on every
machine that installs it, including machines with a perfectly good GPU, and a
wheel built where one was visible reports one everywhere. On a redistributed
build whose host cannot actually open a device, the failure arrives when the
device is opened rather than when it is resolved.

There is no fallback, by design. A silent fallback would turn "my GPU run"
into "a CPU run that took the same wall clock and I never knew".

### GPU requested for a workload the GPU path does not cover

```text
RuntimeError: validation metrics are scored on the CPU; use device='cpu' or device='auto'
```

```text
RuntimeError: custom objectives train on the CPU; use device='cpu' or device='auto'
```

Each one names the specific thing that is not covered and what to do instead.
The rule is the same as above; an explicit `device="gpu"` runs on the
accelerator or raises, and never densifies your matrix or moves your metric
computation without telling you. `device="auto"` accepts all of these and
resolves to the CPU.

A mistyped device name fails earlier and differently.

```text
ValueError: unknown device 'metal'; expected one of cpu, gpu, auto
```

There is one device vocabulary across the Mojo API, the C ABI, and Python,
and `gpu` covers every accelerator rather than naming a vendor.

### `device="auto"` chose the CPU and said nothing

This is not an error and there is no message. It is the current, deliberate
behavior on every machine and every workload.

```python
model = MojoTreesRegressor(device="auto").fit(X, y)
model.device_        # "cpu" below the crossover, "gpu" above it
```

The crossover table that `auto` consults holds one rule. It is scoped to
Metal on an Apple M4 for unweighted squared error, single output, at
1,000,000 rows by 50 features and above, where the GPU has been measured at
3.58s against the CPU's 6.98s. Anything the rule does not cover keeps the
CPU, and the report says which half of "no rule covered this" applied, rather
than implying the shape was too small.

Below that shape the CPU really is the right answer on this hardware, and
that too is measured rather than assumed: at 250,000 rows the GPU takes 1.89s
against the CPU's 1.66s, and at 50,000 it takes 1.63s against 0.564s, because
the device carries about 1.5 seconds of fixed cost per fit. The crossover
itself is somewhere between 250,000 and 1,000,000 rows and has not been
bracketed, so the rule's floor sits at the top of that interval.

One caveat specific to installed wheels. `auto` identifies the accelerator
from the one this binary was *compiled for*, because probing costs a device
open that a decision cannot afford. On a wheel built for an M4 and run on
different Apple silicon, `auto` will believe it is on an M4. The report says
so (`profile_source=build-target`, warning `build-target-hardware`);
`device="cpu"` and `MOJOTREES_DISABLE_GPU=1` both opt out.

Two ways forward, depending on what you want.

```python
MojoTreesRegressor(device="gpu")     # force it, and get an exception if it cannot
```

```sh
MOJOTREES_AUTO_MIN_CELLS=10000000 python your_benchmark.py
```

`MOJOTREES_AUTO_MIN_CELLS` is the cell count (`n_rows * n_features`) at or
above which `auto` selects the GPU, and it is the knob for running the
crossover benchmark that would justify a shipped default. It is device
independent on purpose; there are no per vendor or per chip special cases
anywhere in the policy. If you run that benchmark, the result belongs in an
issue, filed with the
[accelerator validation template](https://github.com/mojotrees/mojotrees/issues/new?template=hardware_validation.yml).

### Anything else

Open a bug report with the
[bug template](https://github.com/mojotrees/mojotrees/issues/new?template=bug_report.yml)
and include the output of `mojotrees.show_versions()`, the complete error
text, and the device you requested. Installation problems are explicitly in scope for that
template. An alpha with quiet failures is worse than one with loud ones, so a
report about a confusing install is useful, not noise.

---

## Where to go next

| Document | What it settles |
|---|---|
| [docs/LIGHTGBM_PARITY.md](LIGHTGBM_PARITY.md) | What mojotrees supports, what LightGBM has that it does not, and which differences are deliberate. Authoritative on behavior |
| [docs/GPU_VALIDATION.md](GPU_VALIDATION.md) | Which hardware has actually run this code, and the procedure for adding a device |
| [docs/PLATFORM_MATRIX.md](PLATFORM_MATRIX.md) | Every install target, its artifact name, and the evidence behind its status |
| [docs/PYTHON_SUPPORT.md](PYTHON_SUPPORT.md) | Why the floor is 3.14, what each older interpreter would take, and what has actually run |
| [docs/DEVICE_SELECTION.md](DEVICE_SELECTION.md) | The full `cpu` / `gpu` / `auto` policy and the report it produces |
| [docs/COMPATIBILITY_POLICY.md](COMPATIBILITY_POLICY.md) | What may change between versions and what may not |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | How to build, which tests to run, and what a pull request has to say |
| [examples/apple_silicon/](../examples/apple_silicon/) | A longer tour of the Apple silicon path, with the timing table that is still empty |

Issue templates.

- [Bug report](https://github.com/mojotrees/mojotrees/issues/new?template=bug_report.yml),
  including installation problems.
- [Accelerator validation report](https://github.com/mojotrees/mojotrees/issues/new?template=hardware_validation.yml),
  for correctness, determinism, or performance evidence from real hardware.
  Failures are useful evidence too.

## What has been run for this document

Nothing. No wheel was built, no package was installed, no example was
executed, and no error message on this page was produced by running the
command above it in this session.

The messages quoted verbatim were read out of the source that raises them,
`src/mojotrees/device.mojo` and `python/mojotrees/__init__.py`. The filenames
and platform tags were read out of
[docs/PLATFORM_MATRIX.md](PLATFORM_MATRIX.md) and
`packaging/matrix/platform_matrix.toml`. The pip and dynamic-loader messages
are shapes rather than transcripts, because their exact wording depends on
the pip and operating system versions involved. The unrun commands that would
turn any of this into a record are listed in
[handoffs/release_04_install_ux.md](../handoffs/release_04_install_ux.md).
