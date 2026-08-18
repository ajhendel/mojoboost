# Python version support

Why mojotrees declares `requires-python = ">=3.14"`, how much of that floor
is real, and what would have to run before it could honestly be lowered.

## Status of this document

**The floor is measured, and it is CPython 3.10.** The compiled extension
imports and passes the full Python API suite on 3.10, 3.11, 3.12, 3.13, and
3.14. On 3.9 it does not fail, it aborts, naming the symbol it wanted. The
verbatim runs are in section 10.

This document still does not change `requires-python`. Applying the value to
`python/pyproject.toml` is a separate step.

What has been run, in full: the existing built extension was imported under
five interpreters and `python/test_python_api.py` was run against it under
each. Nothing was rebuilt, no wheel was produced, no Mojo compiled, no CI
invoked, and no benchmark run. The measurement is one platform only,
osx-arm64, which is stated again in every row that depends on it.

Everything in sections 3 through 8 is static inspection of files already in
the repository or of build artifacts already on disk, and each claim names
where it came from so it can be re-checked when the toolchain moves. Those
sections are what predicted where to look. Section 9.1 records where they
predicted wrong.

## Which vocabulary this is

Three documents describe support and they use three vocabularies on purpose.

- [`docs/PLATFORM_MATRIX.md`](PLATFORM_MATRIX.md) and
  `packaging/matrix/platform_matrix.toml` use `validated`, `tested`,
  `designed`, and `unsupported`. Those words describe an **artifact**, and
  that matrix is the authority on what is installable where.
- Section 10 of [`docs/COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md)
  uses tiers 1, 2, and 3. Those describe **test evidence for a platform**.
- This document uses `proven`, `expected`, `blocked`, and `unsupported`.
  These describe an **interpreter**, which is a third question, and they are
  defined in the matrix in section 8 below.

The vocabularies are not interchangeable and should not be merged. Where this
document and the platform matrix disagree about an artifact, the matrix wins
and this document is the bug. Where they disagree about why an interpreter is
or is not a candidate, section 9 lists the specific rows that need correcting
and why.

## 1. The question

`python/pyproject.toml` says `requires-python = ">=3.14"`. Section 6.1 of the
compatibility policy calls that "a hard floor rather than a preference". If
that is right, mojotrees's first release reaches only the interpreter that
shipped most recently, which for a library whose users are scikit-learn and
LightGBM users is a small fraction of the installed base.

The same policy section, two clauses later, says the extension "links no
libpython". Those two statements are in tension. A binary that links no
libpython has no link-time ABI dependency on any CPython, and a hard floor
has to come from somewhere. This document goes and finds where.

## 2. What was inspected

| Source | What it gave |
|---|---|
| `python/mojotrees/*.py` | Every version-gated construct in the pure-Python half, by `ast` |
| `python/pyproject.toml`, `python/setup.py` | The declared floor, the classifiers, and how the wheel tag is decided |
| `pixi.toml`, `pixi.lock` | Which conda packages pin an interpreter, and which do not |
| `.pixi/envs/*/conda-meta/*.json` | Which package owns which installed file |
| The local rattler repodata cache | Which `max` build variants the pinned channel publishes |
| `python/mojotrees/_mojotrees.so` | Linkage, load commands, and the CPython entry points named inside it |
| `python/mojotrees/.dylibs/*.dylib` | Linkage of the four bundled MAX runtime libraries |
| `.github/workflows/ci.yml` | Which interpreters CI has ever run |

The repodata cache is the one source that is not in the repository. It is the
on-disk cache under `~/Library/Caches/rattler/cache/repodata/` that the solve
producing `pixi.lock` wrote, so it is a record of the same channel state the
lock was solved against, read from a file rather than fetched. Section 10
gives the command that reproduces it over the network for anyone who wants to
confirm it directly.

## 3. Finding: the Python source imposes no floor above 3.7

An `ast` scan of every module in `python/mojotrees/` finds:

- no `match` statement, no `except*`, no PEP 695 type parameters, no walrus
  operator, and no positional-only parameters
- no `from __future__ import annotations` anywhere, and no need for one,
  because there are no function or variable annotations outside dataclass
  fields
- no PEP 604 `X | Y` and no PEP 585 `list[int]` in any annotation. The
  dataclass fields in `dask.py` annotate with bare `tuple`, `dict`, `int`,
  `str`, `bytes`, and `bool`
- no import of `typing` in any module
- no `str.removeprefix`, no `zip(strict=)`, no `functools.cache`, no
  `itertools.pairwise`, no `tomllib`, no `graphlib`, no `zoneinfo`

The newest construct in the package is `@dataclass(frozen=True)` in
`dask.py`, which is 3.7. The test suites in `python/tests/` and
`python/test_python_api.py` scan the same way and find nothing newer.

The standard library surface is `array`, `collections`, `dataclasses`,
`hashlib`, `importlib`, `inspect`, `json`, `math`, `operator`, `os`,
`platform`, `random`, `struct`, `subprocess`, `sys`, `sysconfig`,
`tempfile`, and `warnings`. The newest call in it is `inspect.signature`,
which is 3.3.

**Consequence.** Nothing in `requires-python = ">=3.14"` is about the Python
language or the standard library.

This section is the one part of the audit that goes stale on its own, because
it describes a tree that changes with every commit. It is written down here
for the argument it supports and not as a fact to be trusted later.
`tools/audit_python_compat.py` re-derives the floor on demand and fails when
the source outruns the declared value, which is the durable form of this
finding. Run the tool; do not cite this paragraph.

## 4. Finding: the extension links no libpython, and resolves CPython by name

This is the load-bearing finding.

`otool -L python/mojotrees/_mojotrees.so` lists three dependencies:
`@rpath/libKGENCompilerRTShared.dylib`, `@rpath/libAsyncRTMojoBindings.dylib`,
and `/usr/lib/libSystem.B.dylib`. No libpython. `nm -u` on the same file
lists no undefined `Py*` symbol at all: the only undefined symbols are the
Modular runtime's own `AsyncRT_*` and `KGEN_CompilerRT_*` entry points plus
libc. `otool -L` on all four bundled dylibs in `python/mojotrees/.dylibs/`
finds no libpython either.

The binary does contain the names. `strings -a -t d` finds 109 of them, 107 of
which sit in one contiguous 16-byte-aligned run from `Py_GetConstantBorrowed`
at offset 828864 through `Py_GetVersion` at 831584. The two that do not are
`PyExc_TypeError` and `PyExc_Exception`, which are data objects rather than
functions and live elsewhere in the binary.
`libKGENCompilerRTShared.dylib` carries the code that consumes it, including
the strings `MOJO_PYTHON`, `MOJO_PYTHON_LIBRARY`, `PYTHONEXECUTABLE`,
`Failed to load libpython from `, and the fragment
`binary = f"libpython{pyver}{abiflags}.{ext}"` alongside
`for libpython in [Path(get_config_var(p)) / binary for p in ["LIBPL", "LIBDIR"]]:`.

That is a runtime that asks the running interpreter where its own libpython
is, opens it, and looks its entry points up by name. It is not a binary
compiled against a CPython header set.

**Three consequences follow, and they are the reason this document exists.**

1. The extension has no compile-time CPython ABI. Whatever constrains which
   interpreters it can serve is a property of the name table and of how
   absences are handled, not of anything the linker recorded.
2. The name table comes from `lib/mojo/std.mojoc`, which is shipped by
   `mojo-compiler 1.0.0-release`. That package has exactly one build, with no
   per-interpreter variants. So the same table is compiled into the extension
   no matter which CPython is in the environment when `bindings/build.sh`
   runs, and an extension built under 3.10 would carry the identical set.
3. Therefore the interpreter question can be answered by taking the extension
   that already exists and importing it under a different interpreter. It
   does not require a rebuild. Section 10 makes that experiment M1.

## 5. Finding: the toolchain publishes 3.10 through 3.14, not 3.14 alone

`pixi.toml` declares `mojo = ">=1.0.0,<2"` and `max = ">=26.5.0,<27"` and
pins no Python at all. Every environment in `pixi.lock`, including the `bench`
feature that asks only for `python >=3.11`, resolved to CPython 3.14.6,
because with no pin the solver takes the newest.

Reading the lock's dependency blocks rather than its solution:

| Package in `pixi.lock` | Build | Python dependency |
|---|---|---|
| `mojo 1.0.0` | `release` | `python >=3.10` |
| `mojo-python 1.0.0` | `release`, noarch | `python >=3.10` |
| `mojo-compiler 1.0.0` | `release` | none, via `mojo-python` only |
| `max-core 26.5.0` | `release` | **none at all** |
| `max 26.5.0` | `3.14release` | `python 3.14.*`, `python-gil` |

Only one row pins an interpreter, and it pins it because the build string
says which variant of itself it is. The repodata the solve was run against
lists, for `max 26.5.0`, on each of `osx-arm64`, `linux-64`, and
`linux-aarch64`:

```
max-26.5.0-3.10release
max-26.5.0-3.11release
max-26.5.0-3.12release
max-26.5.0-3.13release
max-26.5.0-3.14release
```

Five variants of the exact pinned version, on all three of the repository's
platforms. The 3.13 variant on `osx-arm64` depends on `python 3.13.*`,
`python-gil`, and `max-core ==26.5.0`, which is the same `max-core` every
other variant depends on.

**Consequence.** `python 3.14.*` is not a toolchain requirement. It is the
variant pixi selected because nothing told it otherwise. The pinned toolchain
supports CPython 3.10 through 3.14 on every platform this project targets.

### 5.1 And the pinned Mojo code does not use the package that pins it

`src/` and `bindings/` import `max.gpu.host`, `max.gpu.memory`,
`max.gpu.sync`, and `max.algorithm`. Those are Mojo packages. The file list in
`.pixi/envs/default/conda-meta/max-core-26.5.0-release.json` includes
`lib/mojo/max.mojoc` and `lib/mojo/algorithm.mojoc`; the file list for
`max-26.5.0-3.14release.json` includes no `lib/mojo/` entry whatsoever. The
`max` conda package is the MAX Python API and nothing in this repository
imports it, other than `bench/real_data/envinfo.py:119`, which runs
`import max; print(max.__version__)` to record a version string.

So the dependency that carries the entire Python pin is a dependency on a
Python API mojotrees does not call. Whether the right fix is to pin Python
explicitly or to depend on `max-core` is a `pixi.toml` decision and belongs to
whoever owns that file; the handoff states it as a question rather than as an
instruction, because dropping `max` also drops the environment `bench` and
`envinfo.py` were written against.

## 6. Finding: the `cp314` wheel tag is setuptools, not an ABI binding

`python/setup.py` sets `has_ext_modules()` to True on a `Distribution`
subclass. That is what makes `bdist_wheel` emit an interpreter-and-ABI tag
instead of `py3-none-any`, and the tag it emits is the tag of the interpreter
running the build. There is no `Extension()` object, no `py_limited_api`, and
nothing that binds the wheel to 3.14 other than which Python invoked
`python -m build`.

The wheel on disk, `mojotrees-0.1.0-cp314-cp314-macosx_26_0_arm64.whl`, is
therefore labeled `cp314` because it was built by a 3.14, not because
anything in it is 3.14-specific. The same source built in an environment
solved against 3.13 would produce `cp313-cp313-macosx_26_0_arm64.whl` with no
change to `setup.py`.

The `plat_name` override in `setup.py` is a separate and real constraint
(`macosx_26_0`, from the Mojo compile step's deployment target). It is the
platform matrix's business, not this document's.

## 7. Finding: `Py_LIMITED_API` is not a lever this repository has

`Py_LIMITED_API` is a C preprocessor macro defined before including
`Python.h`. mojotrees has no C source in the extension path, no `Python.h`
include, and no compile step where the macro could be set: the extension is
emitted by `mojo build --emit shared-lib` from `bindings/_mojotrees.mojo`, and
the CPython glue is `std.python.bindings` inside Modular's `std.mojoc`.
`setup.py` has no `Extension()` to pass `py_limited_api=True` to. Whether an
abi3 build is ever possible is a question for Modular's toolchain, and no
answer to it can be implemented here.

More usefully: **the limited API is not the mechanism mojotrees would need
anyway.** abi3 exists so that one binary can serve many interpreters despite
being linked against libpython. This binary is not linked against libpython.
If one extension can serve several interpreters, it is already able to, and
the thing standing between that and a shippable artifact is the wheel tag,
which section 6 shows is a build-environment choice.

The row in `packaging/matrix/platform_matrix.toml` that marks `abi3` as
`unsupported` reaches the right conclusion by a route that does not hold, and
its advice not to tag a wheel abi3 to widen its reach is correct and should
stay. Section 9 has the wording fix.

## 8. Finding: free-threaded builds are genuinely excluded

Every `max 26.5.0` variant, at every Python version and on every platform,
depends on `python-gil`. A free-threaded interpreter does not provide it, so
the environment cannot be solved and no extension can be built. This is the
one constraint in this document that is a real toolchain requirement rather
than an artifact of how the environment was resolved.

The failure mode matters. pip refuses a `cp314` wheel on a `cp314t`
interpreter by tag mismatch, which is correct and needs no code. A source
checkout gets no such protection, and CPython does not refuse to load an
extension that declares no free-threaded support; it re-enables the GIL and
warns. `python/mojotrees/_compat.py` exists for exactly that case, and for
the below-the-floor case that a source checkout can also reach.

## 9. The proposed support matrix

Status vocabulary, which is this document's and applies to interpreters only:

| Status | Meaning |
|---|---|
| `proven` | An extension was built for this interpreter, imported, and exercised, and the run is recorded in the file named under Evidence. |
| `expected` | The toolchain publishes a build for it and nothing in this audit contradicts it. Nothing has run. Not a claim. |
| `blocked` | A specific named thing prevents it. The thing is named, and so is the test that would clear it. |
| `unsupported` | Deliberately out of scope. Not a gap to be closed. |

Every row below is `proven` on **osx-arm64 only**, against a source checkout
with the extension built. Linux is a separate measurement that has not been
made, and no wheel has been produced for any interpreter but 3.14.

| Interpreter | Status | Basis | Evidence, or what is missing |
|---|---|---|---|
| CPython 3.14 | `proven` | Imports, fits, predicts, round-trips, and passes `test_python_api.py` | Section 10, run M1a. Also `.github/workflows/ci.yml`, `python` job on `ubuntu-latest` and `ubuntu-24.04-arm`, every push, which is the only Linux evidence any row has |
| CPython 3.13 | `proven` | Same suite, same artifact | Section 10, run M1a. osx-arm64 only |
| CPython 3.12 | `proven` | Same suite, same artifact | Section 10, run M1a. osx-arm64 only |
| CPython 3.11 | `proven` | Same suite, same artifact | Section 10, run M1a. osx-arm64 only |
| CPython 3.10 | `proven` | Same suite, same artifact. This is the floor, and two independent constraints put it here: the extension aborts below it, and `mojo 1.0.0` and `mojo-python 1.0.0` both require `python >=3.10` | Section 10, run M1a. osx-arm64 only |
| CPython 3.9 and earlier | `blocked` | The extension aborts at load with `symbol not found: Py_NewRef`, which CPython added in 3.10. Also below the toolchain's own floor | Section 10, run M1b. Nothing short of a different toolchain changes this |
| CPython 3.15 and later | `expected` | Nothing in the source, the metadata, or the extension caps an upper bound, and `requires-python` is deliberately uncapped | Needs a `max` variant to exist, then M2 through M4 |
| Free-threaded, any version | `blocked` | Every `max 26.5.0` variant depends on `python-gil` | Not measured, because there is no artifact to measure. Only Modular can change it. Do not work around it |
| PyPy, GraalPy, any non-CPython | `unsupported` | The runtime resolves CPython C entry points by name out of libpython | Out of scope |
| One abi3 wheel for many interpreters | `unsupported` | Section 7. Not a lever this repository has, and not the mechanism it would need | Ship one wheel per interpreter instead |

### 9.1 Where the static audit predicted wrong, and why that matters

Before anything ran, this document put 3.10, 3.11, and 3.12 at `blocked`, on
the following reasoning. `Py_GetConstantBorrowed` was added in CPython 3.13.
It is in the extension's entry point table. The Mojo runtime demonstrably
resolves *some* entry points conditionally, because the binary contains
exactly two diagnostics of the form `<name> is not available in this Python
version`, for `PyErr_GetRaisedException` (3.12) and `PyType_GetName` (3.11).
`Py_GetConstantBorrowed` has no such string, so it looked required.

**That was wrong.** 3.12 runs the whole suite. The diagnostic string is not
the only mechanism the runtime has for tolerating a missing entry point, and
its absence proves nothing.

The prediction failed in the safe direction, which is worth naming: a byte
scan can establish that a symbol is *present* in a table and can never
establish that it is *needed*. Every floor this repository states about the
extension has to come from an interpreter that ran, and
`tools/audit_python_compat.py` now carries a `MEASURED_LAZY` table whose
entries each cite the run that put them there, rather than inferring from the
absence of a string.

The real floor is set by a different symbol entirely, and the runtime names
it on the way down:

```
ABORT: oss/modular/mojo/stdlib/std/ffi/__init__.mojo:762:18:
symbol not found: Py_NewRef
```

`Py_NewRef` is a CPython 3.10 addition. Three other 3.10 additions sit in the
same table (`Py_Is`, `PyModule_AddObjectRef`, and `Py_NewRef` itself), so
3.10 is where the extension stops regardless of which one is reached first.

That failure mode is the reason `python/mojotrees/_compat.py` checks the
interpreter *before* importing the extension rather than catching around it.
`ABORT` ends the process. There is no exception to catch, and buffered stdout
is lost with it.

### 9.2 Optional dependencies

Nothing in `python/mojotrees/` touches a numpy C API. `_arrays.py:41` takes
buffer addresses through `numpy.ndarray.ctypes.data`, which is a Python-level
attribute, and every other numpy call is `asarray`, `ascontiguousarray`,
`asfortranarray`, `empty`, `unique`, `isfinite`, `isinf`, `floor`, and
`flatnonzero`. There is no numpy ABI dependency, no `oldest-supported-numpy`
build pin to think about, and no numpy 1.x versus 2.x split to navigate.
numpy is optional throughout and `python/tests/test_no_numpy.py` plus the
bare-venv install in `packaging/test_wheel.sh` keep it that way.

scikit-learn is used through `__sklearn_tags__`, which is scikit-learn 1.6 and
newer, and scikit-learn 1.6 supports CPython 3.9 and newer. Lowering
mojotrees's floor to any version in this document's matrix does not put the
tags hook out of reach.

For reference, the stack the pinned environments resolved on 3.14 is numpy
2.5.2, scipy 1.18.0, scikit-learn 1.9.0, pandas 3.0.5, pyarrow 25.0.0, polars
1.43.2, and lightgbm 4.7.0, all from conda-forge. A lower interpreter widens
what is available rather than narrowing it, so the optional dependency stack
is not an argument for or against any floor in the matrix. What it does
change is which *version* of each a user gets, and section 10 makes recording
that per interpreter experiment M5.

### 9.3 The range for a first release

**`requires-python = ">=3.10"`.**

It is the floor the artifact actually has, measured on five interpreters, and
it is simultaneously the floor the toolchain imposes from the other side:
`mojo 1.0.0` and `mojo-python 1.0.0` both require `python >=3.10`, and the
channel publishes no `max 26.5.0` variant below 3.10. Two independent
constraints landing on the same number is the strongest form this answer
could have taken, and there is nothing below it to argue about.

It costs no source change. Section 3 found nothing in the Python to change
and section 4 found nothing in the extension to rebuild differently: the
artifact that shipped these runs is the one already in the working copy,
built by a 3.14, and it served 3.10 unmodified.

What it does cost is **one wheel per interpreter per platform**, because the
wheel tag is per-interpreter (section 6) and abi3 is not available (section
7). That is five macOS wheels instead of one. Whoever owns the release
decides whether to publish all five or a subset; `requires-python` is a floor
and does not oblige a wheel to exist for every version above it. Publishing
fewer wheels than `requires-python` admits is normal and is what
`packaging/matrix/platform_matrix.toml` is for.

`>=3.14` is no longer defensible on any grounds. It is not what the toolchain
requires (section 5), not what the extension requires (section 9.1), and not
what the language requires (section 3).

## 10. The evidence

M1 has run and is recorded below. M2 through M6 have not, and each says what
it would settle. Record failures verbatim; a diagnostic naming a missing C
symbol is the most informative result any of these can produce, and it is
exactly what M1b returned.

### M1. Import the existing extension under another interpreter

The decisive one, and the cheapest. It needs no rebuild, because section 4
established that the entry point table does not vary with the interpreter the
build ran under. The interpreter must not be the pixi one, and it needs the
four MAX runtime dylibs reachable, which they are when
`python/mojotrees/.dylibs/` is populated.

**Host.** Apple M4, macOS 26, osx-arm64. Artifact:
`python/mojotrees/_mojotrees.so` as built by `bindings/build.sh` under
CPython 3.14.6, 1453392 bytes, unmodified between runs. Interpreters:
homebrew for 3.12, 3.13, and 3.14; `pixi exec` cached environments for 3.9,
3.10, and 3.11.

#### M1a. Supported interpreters

```
cd python
python3.14 test_python_api.py
python3.13 test_python_api.py
python3.12 test_python_api.py
pixi exec --spec "python==3.11.*" --spec numpy -- python test_python_api.py
pixi exec --spec "python==3.10.*" --spec numpy -- python test_python_api.py
```

All five ended with the suite's own last line and exit status 0:

```
all python API tests passed
```

A fit, a predict, `gpu_available()`, `num_trees()`, and a
`model_to_string()` round trip were also run standalone on 3.10 through 3.13
before the suite, and all four behaved identically across them.
`gpu_available()` returned `True` on every interpreter, so the Metal path is
reached from all of them and is not 3.14-specific either.

#### M1b. The floor

```
pixi exec --spec "python==3.9.*" -- python -c "import mojotrees"
```

```
ABORT: oss/modular/mojo/stdlib/std/ffi/__init__.mojo:762:18:
symbol not found: Py_NewRef
```

Not an exception. The process aborts during module initialization and
buffered stdout is discarded with it, which is why the guard in
`python/mojotrees/_compat.py` runs in front of the import rather than around
it. `Py_NewRef` is a CPython 3.10 addition.

#### M1c. A failure that is not about interpreter version

`python/test_python_api.py` fails on any interpreter with no numpy
installed:

```
TypeError: X is a sparse matrix, which needs numpy; install numpy or pass a
dense sequence
```

raised from `_arrays.py` `check_X_sparse`. This was confirmed to be
version-orthogonal by running it on CPython 3.14 in a `venv --without-pip`,
where it fails at the identical line. It is a real gap and it is not this
document's to fix; see section 12, item 4.

### M2. Toolchain resolution per interpreter

Not run. M1 measured what the built artifact tolerates, which is a different
question from what the toolchain will build for. Both matter: a wheel for
3.11 has to be built by a 3.11 environment to carry a `cp311` tag.

```
for v in 3.10 3.11 3.12 3.13 3.14; do
    d=$(mktemp -d)
    pixi init "$d" \
        --channel https://conda.modular.com/max --channel conda-forge
    pixi add --manifest-path "$d/pixi.toml" \
        "python==$v.*" "mojo>=1.0.0,<2" "max>=26.5.0,<27" \
        && echo "$v: resolves" || echo "$v: does not resolve"
done
```

Settles: whether the pinned toolchain co-resolves with each interpreter, on
this platform, today. Run it on each of `osx-arm64`, `linux-64`, and
`linux-aarch64`. This is also the network confirmation of the repodata
reading in section 5; the direct form of that is:

```
pixi search --channel https://conda.modular.com/max max
```

### M3. Build the extension per interpreter

On a branch, with `python = "==3.13.*"` added to `[dependencies]` in
`pixi.toml`:

```
pixi install
pixi run build-python
python3 tools/audit_python_compat.py python/mojotrees/_mojotrees.so
otool -l python/mojotrees/_mojotrees.so | grep -A4 LC_BUILD_VERSION   # macOS
readelf -d python/mojotrees/_mojotrees.so                             # Linux
```

Settles: whether the build succeeds, and whether the entry point table is
identical to the 3.14 build. Diff the audit tool's symbol lines between the
two builds; section 4 predicts they are the same, and a difference would
invalidate M1's shortcut and is the more interesting outcome.

Not run. M1 makes this less urgent than it looked, because the artifact that
served all five interpreters was built once by a 3.14. What M3 still decides
is the wheel tag, not whether the code works.

### M4. The suite, then the wheel, per interpreter

```
pixi run test-python
pixi run -e pytest test-estimators
pixi run -e pkg build-wheel
ls python/dist/
pixi run -e pkg test-wheel
python3 packaging/matrix/validate_artifact.py python/dist/*.whl
```

Settles: whether the interpreter is `proven` rather than `expected`. Record
the exact wheel filename; section 6 predicts `cp313-cp313-...` from a 3.13
environment, and the filename is the check on that prediction.

Note that `packaging/test_wheel.sh` builds its venvs with whichever `python`
is on PATH, which inside `pixi run` is the pixi environment's. That is the
correct interpreter here by construction, and it is also the reason the
script has never tested any other one.

### M5. The optional dependency stack per interpreter

```
python -m venv /tmp/mb-check && \
    /tmp/mb-check/bin/pip install numpy scipy scikit-learn pandas && \
    /tmp/mb-check/bin/python -c \
    "import numpy,scipy,sklearn,pandas as p;print(numpy.__version__,scipy.__version__,sklearn.__version__,p.__version__)"
```

Settles: which versions of each optional dependency a user of that
interpreter actually gets from PyPI, which is the number that belongs in a
release note. scikit-learn must be 1.6 or newer for `__sklearn_tags__`.

### M6. The audit tool under every candidate

```
python3.10 tools/audit_python_compat.py
python3.11 tools/audit_python_compat.py
python3.12 tools/audit_python_compat.py
python3.13 tools/audit_python_compat.py
python3.14 tools/audit_python_compat.py
```

Settles: that no module in `python/mojotrees/` fails to parse under any
candidate interpreter, which is a direct measurement rather than a scan. The
tool is written to run on CPython 3.8 and newer so it can be the thing that
runs first on a machine where nothing else can.

### What a row needs before it moves

The five `proven` rows are proven for a **source install on osx-arm64** and
for nothing else. To extend any of them to a shipped wheel, on one platform:

1. M2 resolves for that interpreter.
2. M3 builds, and the audit tool reports no contradiction.
3. M4's suites pass and the wheel carries the expected tag.
4. The output of all three is written into this document's section 10 and
   named in the Evidence column.

To extend them to Linux, M1 is enough and is the same three commands against
a Linux-built extension.

`blocked` moves only on a run that shows the named blocker does not apply,
never on reasoning. Section 9.1 is the worked example of why.

## 11. What must not be done

- Do not raise `requires-python` back above 3.10 without a run that shows
  3.10 failing. The current value rests on five interpreters that passed and
  one that aborted, not on inference.
- Do not treat the absence of a `is not available in this Python version`
  string as evidence that an entry point is required. Section 9.1.
- Do not tag a wheel `abi3` to widen its reach. Section 7.
- Do not remove `python-gil` from consideration or try to force a
  free-threaded build. Section 8.
- Do not add a `<3.15` cap to `requires-python`. It would have to be edited
  on every release the toolchain permits, and `python/pyproject.toml` already
  says why it is absent.
- Do not add shims to `python/mojotrees/_compat.py` for constructs the source
  does not use. Section 3 found none, and the module's docstring says what
  would justify a new one.
- Do not read the five `proven` rows as claims about Linux or about wheels.
  They are one platform and one source install.

## 12. Corrections owed to other documents

Three statements elsewhere in the repository do not survive this audit. None
of the files involved is this lane's to edit, so each is listed below with
its replacement.

1. `packaging/matrix/platform_matrix.toml`, the `cp313` row: "No MAX build
   for these interpreters in the pinned channel." Five variants of
   `max 26.5.0` are published on all three platforms, 3.10 through 3.14.
2. `packaging/matrix/platform_matrix.toml`, `[toolchain]`, and the same claim
   restated in `docs/PLATFORM_MATRIX.md` and `packaging/linux/README.md`:
   that the `python 3.14.*` pin "is what makes cp314 the only interpreter
   mojotrees builds against". It is the variant the solver chose with no pin
   present, not a constraint the toolchain imposes.
3. `docs/COMPATIBILITY_POLICY.md` section 6.1: "`requires-python = ">=3.14"`
   today, and that is a hard floor rather than a preference." It is not a
   hard floor. The extension runs on 3.10.
4. `pixi.toml`'s comment on `test-python`, that it "stays dependency-free on
   purpose, so it also runs against a bare wheel install". It does not.
   `python/test_python_api.py` needs numpy for its sparse cases and raises
   `TypeError` without it, on every interpreter including 3.14. It passes
   under `pixi run test-python` only because the default pixi environment has
   numpy, which arrives as a dependency of `max` rather than by anyone asking
   for it. The consequence is that the `bare install (no numpy)` step of
   `packaging/test_wheel.sh` would fail on its first run, and that step has
   evidently never run. This is a real bug in a path the project believes is
   covered, and it is the most load-bearing thing this audit found that was
   not about interpreter versions.

The `abi3` row in the platform matrix reaches a conclusion this document
agrees with, by reasoning it does not. Its conclusion should stay and its
reasoning should be replaced.

Corrections 1, 2, and the `abi3` row have been applied to
`packaging/matrix/platform_matrix.toml` and `validate_matrix.py`. Corrections
3 and 4, and the prose restatements, are still owed.
