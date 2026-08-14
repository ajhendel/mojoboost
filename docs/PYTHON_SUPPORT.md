# Python version support

Why mojoboost declares `requires-python = ">=3.14"`, how much of that floor
is real, and what would have to run before it could honestly be lowered.

## Status of this document

Nothing in this document has been executed. No extension was built, no
interpreter other than the pinned 3.14 was installed, no wheel was produced,
and no import was attempted. Every claim below is a reading of files that are
already in the repository or of build artifacts already on disk, and each one
names where it came from so it can be re-checked when the toolchain moves.

This document does not change `requires-python`. It says what the evidence
supports, what it does not, and which single experiment separates the two.
Until that experiment runs, `>=3.14` stays.

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
that is right, mojoboost's first release reaches only the interpreter that
shipped most recently, which for a library whose users are scikit-learn and
LightGBM users is a small fraction of the installed base.

The same policy section, two clauses later, says the extension "links no
libpython". Those two statements are in tension. A binary that links no
libpython has no link-time ABI dependency on any CPython, and a hard floor
has to come from somewhere. This document goes and finds where.

## 2. What was inspected

| Source | What it gave |
|---|---|
| `python/mojoboost/*.py` | Every version-gated construct in the pure-Python half, by `ast` |
| `python/pyproject.toml`, `python/setup.py` | The declared floor, the classifiers, and how the wheel tag is decided |
| `pixi.toml`, `pixi.lock` | Which conda packages pin an interpreter, and which do not |
| `.pixi/envs/*/conda-meta/*.json` | Which package owns which installed file |
| The local rattler repodata cache | Which `max` build variants the pinned channel publishes |
| `python/mojoboost/_mojoboost.so` | Linkage, load commands, and the CPython entry points named inside it |
| `python/mojoboost/.dylibs/*.dylib` | Linkage of the four bundled MAX runtime libraries |
| `.github/workflows/ci.yml` | Which interpreters CI has ever run |

The repodata cache is the one source that is not in the repository. It is the
on-disk cache under `~/Library/Caches/rattler/cache/repodata/` that the solve
producing `pixi.lock` wrote, so it is a record of the same channel state the
lock was solved against, read from a file rather than fetched. Section 10
gives the command that reproduces it over the network for anyone who wants to
confirm it directly.

## 3. Finding: the Python source imposes no floor above 3.7

An `ast` scan of all ten modules in `python/mojoboost/` finds:

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

The standard library surface is `array`, `hashlib`, `importlib`, `inspect`,
`json`, `math`, `operator`, `os`, `platform`, `random`, `struct`,
`subprocess`, `tempfile`, and `warnings`. The newest call in it is
`inspect.signature`, which is 3.3. `os.sysconf` in
`device_selection.py:830` is POSIX-only and is already wrapped in a handler
that catches `AttributeError`, so it is a platform question and not a version
one.

**Consequence.** Nothing in `requires-python = ">=3.14"` is about the Python
language or the standard library. `tools/audit_python_compat.py` re-derives
this floor on demand and fails if the source ever outruns the declared value.

## 4. Finding: the extension links no libpython, and resolves CPython by name

This is the load-bearing finding.

`otool -L python/mojoboost/_mojoboost.so` lists three dependencies:
`@rpath/libKGENCompilerRTShared.dylib`, `@rpath/libAsyncRTMojoBindings.dylib`,
and `/usr/lib/libSystem.B.dylib`. No libpython. `nm -u` on the same file
lists no undefined `Py*` symbol at all: the only undefined symbols are the
Modular runtime's own `AsyncRT_*` and `KGEN_CompilerRT_*` entry points plus
libc. `otool -L` on all four bundled dylibs in `python/mojoboost/.dylibs/`
finds no libpython either.

The binary does contain the names. A byte scan finds 109 CPython entry point
names laid out as one contiguous 16-byte-aligned table, from
`Py_GetConstantBorrowed` at offset 828864 through `Py_GetVersion` at 831584.
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
Python API mojoboost does not call. Whether the right fix is to pin Python
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

The wheel on disk, `mojoboost-0.1.0-cp314-cp314-macosx_26_0_arm64.whl`, is
therefore labeled `cp314` because it was built by a 3.14, not because
anything in it is 3.14-specific. The same source built in an environment
solved against 3.13 would produce `cp313-cp313-macosx_26_0_arm64.whl` with no
change to `setup.py`.

The `plat_name` override in `setup.py` is a separate and real constraint
(`macosx_26_0`, from the Mojo compile step's deployment target). It is the
platform matrix's business, not this document's.

## 7. Finding: `Py_LIMITED_API` is not a lever this repository has

`Py_LIMITED_API` is a C preprocessor macro defined before including
`Python.h`. mojoboost has no C source in the extension path, no `Python.h`
include, and no compile step where the macro could be set: the extension is
emitted by `mojo build --emit shared-lib` from `bindings/_mojoboost.mojo`, and
the CPython glue is `std.python.bindings` inside Modular's `std.mojoc`.
`setup.py` has no `Extension()` to pass `py_limited_api=True` to. Whether an
abi3 build is ever possible is a question for Modular's toolchain, and no
answer to it can be implemented here.

More usefully: **the limited API is not the mechanism mojoboost would need
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
warns. `python/mojoboost/_compat.py` exists for exactly that case, and for
the below-the-floor case that a source checkout can also reach.

## 9. The proposed support matrix

Status vocabulary, which is this document's and applies to interpreters only:

| Status | Meaning |
|---|---|
| `proven` | An extension was built for this interpreter, imported, and exercised, and the run is recorded in the file named under Evidence. |
| `expected` | The toolchain publishes a build for it and nothing in this audit contradicts it. Nothing has run. Not a claim. |
| `blocked` | A specific named thing prevents it. The thing is named, and so is the test that would clear it. |
| `unsupported` | Deliberately out of scope. Not a gap to be closed. |

| Interpreter | Status | Basis | Evidence, or what is missing |
|---|---|---|---|
| CPython 3.14 | `proven` | The only interpreter anything has ever run on | `.github/workflows/ci.yml`, `python` job on `ubuntu-latest` and `ubuntu-24.04-arm`, every push. The wheel and the `.so` in this working copy are `cp314`. macOS is local only, per section 10.2 of the compatibility policy |
| CPython 3.13 | `expected` | `max-26.5.0-3.13release` is published on all three platforms. No entry point in the extension's table postdates 3.13 | Nothing has run. Needs M1, then M2 and M3 |
| CPython 3.12 | `blocked` | `Py_GetConstantBorrowed`, added in CPython 3.13, is in the extension's entry point table and carries no conditional-resolution diagnostic | M1 clears or confirms this in one command |
| CPython 3.11 | `blocked` | Same as 3.12 | Same as 3.12 |
| CPython 3.10 | `blocked` | Same as 3.12. This is the lowest the toolchain could ever reach: `mojo 1.0.0` and `mojo-python 1.0.0` both require `python >=3.10` | Same as 3.12 |
| CPython 3.9 and earlier | `unsupported` | Below the toolchain's own floor of `python >=3.10`, and no `max 26.5.0` variant exists for them | Nothing would change this short of a different toolchain |
| CPython 3.15 and later | `expected` | Nothing in the source, the metadata, or the extension caps an upper bound, and `requires-python` is deliberately uncapped | Needs a `max` variant to exist, then M1 through M4 |
| Free-threaded, any version | `blocked` | Every `max 26.5.0` variant depends on `python-gil` | Only Modular can change this. Do not work around it |
| PyPy, GraalPy, any non-CPython | `unsupported` | The runtime resolves CPython C entry points by name out of libpython | Out of scope |
| One abi3 wheel for many interpreters | `unsupported` | Section 7. Not a lever this repository has, and not the mechanism it would need | Ship one wheel per interpreter instead |

### 9.1 The one thing standing between `blocked` and `expected`

`Py_GetConstantBorrowed` was added in CPython 3.13. It is present in the
extension's entry point table, at the head of it, adjacent to
`_Py_NoneStruct` and `Py_Is`, which is what a constants group with a modern
path and a legacy fallback would look like. That reading is a guess.

What is not a guess is that the Mojo runtime resolves some entry points
conditionally and says so. The binary contains exactly two diagnostics of the
form `<name> is not available in this Python version`, for
`PyErr_GetRaisedException` (CPython 3.12) and `PyType_GetName` (CPython
3.11). `Py_GetConstantBorrowed` has no such string.

Two readings fit that:

- the runtime requires it unconditionally, and the extension has a hard floor
  of CPython 3.13 on every build, or
- it is resolved lazily and falls back, and the floor is CPython 3.10.

A byte scan cannot tell these apart, and `tools/audit_python_compat.py` says
so in its own output rather than picking one. Experiment M1 in section 10
settles it in a single command, and it is the highest-value thing anybody can
do to this question.

### 9.2 Optional dependencies

Nothing in `python/mojoboost/` touches a numpy C API. `_arrays.py:41` takes
buffer addresses through `numpy.ndarray.ctypes.data`, which is a Python-level
attribute, and every other numpy call is `asarray`, `ascontiguousarray`,
`asfortranarray`, `empty`, `unique`, `isfinite`, `isinf`, `floor`, and
`flatnonzero`. There is no numpy ABI dependency, no `oldest-supported-numpy`
build pin to think about, and no numpy 1.x versus 2.x split to navigate.
numpy is optional throughout and `python/tests/test_no_numpy.py` plus the
bare-venv install in `packaging/test_wheel.sh` keep it that way.

scikit-learn is used through `__sklearn_tags__`, which is scikit-learn 1.6 and
newer, and scikit-learn 1.6 supports CPython 3.9 and newer. Lowering
mojoboost's floor to any version in this document's matrix does not put the
tags hook out of reach.

For reference, the stack the pinned environments resolved on 3.14 is numpy
2.5.2, scipy 1.18.0, scikit-learn 1.9.0, pandas 3.0.5, pyarrow 25.0.0, polars
1.43.2, and lightgbm 4.7.0, all from conda-forge. A lower interpreter widens
what is available rather than narrowing it, so the optional dependency stack
is not an argument for or against any floor in the matrix. What it does
change is which *version* of each a user gets, and section 10 makes recording
that per interpreter experiment M5.

### 9.3 The highest-value realistic range for a first release

**`>=3.13`.**

It is one command away. M1 either clears `Py_GetConstantBorrowed` or confirms
it, and if it confirms it, 3.13 is still reachable because no entry point in
the table postdates 3.13. It costs no source change, because section 3 found
nothing in the Python to change and section 4 found nothing in the extension
to rebuild differently. It costs one more wheel per platform, built by an
environment solved against `python==3.13.*`, using the build script that
already exists.

`>=3.10` is the larger prize and it is not this project's to take. It needs
Modular to guard one more entry point, and section 7 explains why mojoboost
cannot route around that with the limited API. Ask for it upstream; do not
engineer around it.

`>=3.14` is defensible only as the current state of the evidence, and only
until M1 runs. It should not be defended on the grounds that the toolchain
requires it, because section 5 shows it does not.

## 10. What evidence would be needed, per interpreter and per platform

Nothing below has been run. Each experiment says what it settles and what to
record. Record failures verbatim; a traceback naming a missing C symbol is
the most informative result any of these can produce.

### M1. Import the existing extension under another interpreter

The decisive one, and the cheapest. It needs no rebuild, because section 4
established that the entry point table does not vary with the interpreter the
build ran under.

```
# From a checkout with python/mojoboost/_mojoboost.so already built.
cd python
/path/to/python3.13 -c "import mojoboost; print(mojoboost.__version__)"
/path/to/python3.12 -c "import mojoboost; print(mojoboost.__version__)"
/path/to/python3.10 -c "import mojoboost; print(mojoboost.__version__)"
```

The interpreter must not be the pixi one, and it needs the four MAX runtime
dylibs reachable, which they are when `python/mojoboost/.dylibs/` is
populated by `packaging/build_wheel.sh`.

Settles: whether `Py_GetConstantBorrowed` is a hard floor. Record the exact
output or the exact traceback for each, per platform.

### M2. Toolchain resolution per interpreter

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
python3 tools/audit_python_compat.py python/mojoboost/_mojoboost.so
otool -l python/mojoboost/_mojoboost.so | grep -A4 LC_BUILD_VERSION   # macOS
readelf -d python/mojoboost/_mojoboost.so                             # Linux
```

Settles: whether the build succeeds, and whether the entry point table is
identical to the 3.14 build. Diff the audit tool's symbol lines between the
two builds; section 4 predicts they are the same, and a difference would
invalidate M1's shortcut and is the more interesting outcome.

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

Settles: that no module in `python/mojoboost/` fails to parse under any
candidate interpreter, which is a direct measurement rather than a scan. The
tool is written to run on CPython 3.8 and newer so it can be the thing that
runs first on a machine where nothing else can.

### What a row needs before it moves

To move an interpreter from `expected` to `proven`, on one platform:

1. M2 resolves.
2. M3 builds, and the audit tool reports no contradiction.
3. M4's suites pass and the wheel carries the expected tag.
4. The output of all three is written to a file in the repository and named
   in this document's Evidence column.

`blocked` moves to `expected` when M1 or M3 shows the named blocker does not
apply. `blocked` never moves on reasoning alone, and neither does anything
else here.

## 11. What must not be done

- Do not lower `requires-python` before M1 has run. The evidence in this
  document is sufficient to make 3.14 look accidental and insufficient to
  make anything else look safe.
- Do not tag a wheel `abi3` to widen its reach. Section 7.
- Do not remove `python-gil` from consideration or try to force a
  free-threaded build. Section 8.
- Do not add a `<3.15` cap to `requires-python`. It would have to be edited
  on every release the toolchain permits, and `python/pyproject.toml` already
  says why it is absent.
- Do not add shims to `python/mojoboost/_compat.py` for constructs the source
  does not use. Section 3 found none, and the module's docstring says what
  would justify a new one.
- Do not claim an interpreter is supported because the toolchain publishes a
  variant for it. Publication is `expected`. Running is `proven`.

## 12. Corrections owed to other documents

Three statements elsewhere in the repository do not survive this audit. None
of the files involved is this lane's to edit, so each is written out with its
replacement in
[`handoffs/release_05_python_versions.md`](../handoffs/release_05_python_versions.md).

1. `packaging/matrix/platform_matrix.toml`, the `cp313` row: "No MAX build
   for these interpreters in the pinned channel." Five variants of
   `max 26.5.0` are published on all three platforms, 3.10 through 3.14.
2. `packaging/matrix/platform_matrix.toml`, `[toolchain]`, and the same claim
   restated in `docs/PLATFORM_MATRIX.md` and `packaging/linux/README.md`:
   that the `python 3.14.*` pin "is what makes cp314 the only interpreter
   mojoboost builds against". It is the variant the solver chose with no pin
   present, not a constraint the toolchain imposes.
3. `docs/COMPATIBILITY_POLICY.md` section 6.1: "`requires-python = ">=3.14"`
   today, and that is a hard floor rather than a preference." The same
   paragraph's own next clause, that the extension links no libpython, is
   what makes the first clause doubtful.

The `abi3` row in the platform matrix reaches a conclusion this document
agrees with, by reasoning it does not. Its conclusion should stay and its
reasoning should be replaced.
