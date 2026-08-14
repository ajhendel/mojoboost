# Handoff: the Python version support contract

Lane 05 of a parallel round. This document is the part of the work that
touches files this lane did not own. Edits marked **PENDING** are written out
exactly and have not been applied. Edits marked **APPLIED** were applied at
the repository owner's request after the audit produced a measurement.

The reasoning behind all of it is [`docs/PYTHON_SUPPORT.md`](../docs/PYTHON_SUPPORT.md).
This handoff does not restate it; it states what to change and where.

## The answer Task 01 asked for

**`requires-python = ">=3.10"`. Resolved, measured, not inferred.**

The extension already in the working copy, built once by CPython 3.14,
imports and passes `python/test_python_api.py` unmodified on CPython 3.10,
3.11, 3.12, 3.13, and 3.14. On 3.9 it aborts during module initialization
with `symbol not found: Py_NewRef`, a CPython 3.10 addition. The verbatim
runs are in section 10 of `docs/PYTHON_SUPPORT.md`.

3.10 is also where the toolchain stops from the other side: `mojo 1.0.0` and
`mojo-python 1.0.0` both require `python >=3.10`, and the pinned channel
publishes no `max 26.5.0` variant below 3.10. Two independent constraints
landing on the same number.

**Scope, which matters.** This is measured on osx-arm64 only, against a
source install. It says nothing about Linux and nothing about wheels. It is
sufficient to set `requires-python`, which is a floor on the interpreter, and
insufficient to promise an artifact for any interpreter but 3.14.

The earlier draft of this handoff said the answer was unresolved and
recommended `>=3.13` as a target. Both are superseded. The reasoning that
produced them is preserved in section 9.1 of `docs/PYTHON_SUPPORT.md` as a
worked example of a byte scan overestimating a floor.

## What this lane added

| Path | What it is |
|---|---|
| `docs/PYTHON_SUPPORT.md` | The audit, the measured support matrix, and the verbatim runs behind it |
| `tools/audit_python_compat.py` | Standard-library-only scanner: source syntax, packaging metadata, `pixi.lock` toolchain pins, and the CPython entry points inside a built `_mojoboost.so`. Imports nothing and loads no extension |
| `python/mojoboost/_compat.py` | Two interpreter checks and one guarded import. No shims: the audit found nothing to shim |

## What was executed

The existing built extension was imported under six interpreters and
`python/test_python_api.py` was run against it under five, on osx-arm64.
`tools/audit_python_compat.py` and `packaging/matrix/validate_matrix.py` were
run. Nothing was rebuilt: no Mojo compiled, no wheel produced, no CI invoked,
no benchmark run. Interpreters came from homebrew (3.12, 3.13, 3.14) and from
`pixi exec` cached environments (3.9, 3.10, 3.11); the repository's own pixi
environment was not modified and `pixi.toml` was not touched.

## Edit 1: `python/pyproject.toml` (Task 01) — PENDING

**1a. The value.**

```toml
requires-python = ">=3.10"
```

**1b. The comment above `classifiers`.** Replace the paragraph beginning
"`requires-python` is a floor, not a range" with:

```toml
# `requires-python` is a floor, not a range, which is what section 6.1 of
# docs/COMPATIBILITY_POLICY.md states and what this project keeps. A capped
# `<3.15` would add nothing and would have to be edited on every release the
# toolchain permits.
#
# 3.10 is measured, not assumed. The compiled extension links no libpython
# and resolves CPython entry points by name at runtime, so one build serves
# several interpreters: 3.10 through 3.14 all import it and pass
# python/test_python_api.py, and 3.9 aborts on Py_NewRef, a 3.10 addition.
# mojo 1.0.0 independently requires python >=3.10. docs/PYTHON_SUPPORT.md
# section 10 has the runs.
#
# This floor is NOT a promise that a wheel exists for every version above it.
# The wheel tag is per-interpreter and there is no abi3 build, so each
# interpreter needs its own wheel. packaging/matrix/platform_matrix.toml is
# the authority on which ones exist.
```

**1c. Classifiers.** Add the four rows the floor now admits, next to the
existing `3.14`:

```toml
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
```

`tools/audit_python_compat.py` fails when a classifier claims a version
`requires-python` excludes, so this pair stays honest mechanically once the
task in Edit 3 exists.

Nothing else in `pyproject.toml` changes.

## Edit 2: the two unguarded extension imports — PENDING

Five modules reach the extension. Only two do it at module scope.

| Site | Kind | Needs the guard |
|---|---|---|
| `python/mojoboost/__init__.py:253` | module scope | **Yes.** First line `import mojoboost` reaches, seven before it imports `basic` |
| `python/mojoboost/basic.py:65` | module scope | **Yes.** `mojoboost.basic` is a supported import path (section 2 of the compatibility policy) so it can be reached first |
| `python/mojoboost/device_selection.py`, in `_native_gpu_available` | deferred, already guarded | No. Treats absence as non-fatal on purpose. Correct as written |
| `python/mojoboost/dask.py:1258` | deferred | No. Unreachable until `__init__` has succeeded |
| `python/mojoboost/inspection.py:152` and `:677` | deferred | No, same reason |

**Why in front of the import and not around it.** On 3.9 the extension does
not raise, it aborts:

```
ABORT: oss/modular/mojo/stdlib/std/ffi/__init__.mojo:762:18:
symbol not found: Py_NewRef
```

A `try` around the import would never reach its handler, and buffered stdout
is discarded with the process. `_compat.import_extension()` therefore checks
first and imports second.

**2a. `python/mojoboost/__init__.py`.** Line 253 currently reads:

```python
from . import _arrays, _eval, _mojoboost, callback as _callback
```

Replace with:

```python
from . import _arrays, _compat, _eval, callback as _callback

# Not `from . import _mojoboost`. This is the first line `import mojoboost`
# reaches, and on an interpreter older than the extension's floor the loader
# does not raise, it aborts the process naming a CPython C symbol. _compat
# checks in front of the import for that reason.
# See docs/PYTHON_SUPPORT.md sections 9.1 and 10.
_mojoboost = _compat.import_extension()
```

The two `_mojoboost.*` call sites at lines 2118 and 2120 are unchanged.

**2b. `python/mojoboost/basic.py`.** Line 65 currently reads:

```python
from . import _arrays, _eval, _mojoboost
```

Replace with:

```python
from . import _arrays, _compat, _eval

# See the same call in __init__.py. Repeated here because
# `import mojoboost.basic` is a supported import path (section 2 of
# docs/COMPATIBILITY_POLICY.md) and can be the first thing a program runs.
_mojoboost = _compat.import_extension()
```

The 25 `_mojoboost.*` call sites below it are unchanged. Doing this twice
costs nothing: the second call gets the module from `sys.modules`.

`_compat.EXTENSION_FLOOR` is a measured property of the artifact and is
deliberately a different name from `requires-python`, which is what the
project chooses to declare. They are equal today at 3.10. If a future
toolchain raises the artifact's floor above the declared one, the audit tool
fails on the pair.

## Edit 3: `pixi.toml` — PENDING

One task, in the default `[tasks]` table next to `check-parity`, which it
resembles: standard library only, builds nothing, imports nothing.

```toml
# Checks the declared Python floor against the source, the packaging
# metadata, pixi.lock, and any built extension. Standard library only, it
# builds nothing and imports no extension, so it is cheap enough to run on
# every change and it fails when requires-python and the tree disagree.
audit-python-compat = "python3 tools/audit_python_compat.py"
```

### The open pixi question, which is not this lane's to decide

`pixi.toml` depends on `max = ">=26.5.0,<27"`, which is the MAX Python API.
The Mojo packages `src/` actually imports (`max.gpu.host`, `max.gpu.memory`,
`max.gpu.sync`, `max.algorithm`) ship as `lib/mojo/max.mojoc` and
`lib/mojo/algorithm.mojoc` inside `max-core`, which has no Python dependency
at all. The `max` package's only appearance outside the manifest is
`bench/real_data/envinfo.py:119`, which prints `max.__version__`.

So the dependency carrying the entire Python pin is on an API this project
does not call. Two ways to move it:

- Pin the interpreter explicitly, `python = "==3.13.*"` or similar in
  `[dependencies]`. Keeps `max`, keeps `envinfo.py`, makes the interpreter a
  decision instead of a solver outcome. **Recommended**, and it is what
  experiment M3 needs anyway.
- Depend on `max-core` instead of `max`. Removes the pin entirely, at the
  cost of `envinfo.py` and of any future use of the MAX Python API.

Either way `pixi.lock` is regenerated and the `[toolchain]` block in
`platform_matrix.toml` has to be re-derived from it.

**Note the wheel consequence before choosing.** The interpreter pinned here
decides the wheel tag. Pinning 3.13 means the default wheel is `cp313` and
the 3.14 users lose theirs. If more than one wheel is to be published, the
release workflow needs one environment per interpreter, which is Edit 4's
territory rather than this one's.

## Edit 4: `.github/workflows/ci.yml` — PENDING

One job, modeled on the existing `parity` job:

```yaml
  # The declared Python floor against the tree. Standard library only and no
  # build, so it runs on a bare runner in seconds. It does not prove any
  # interpreter works; it fails when requires-python, the classifiers, the
  # source, and pixi.lock stop agreeing with each other.
  python-compat:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Python compatibility audit
        run: python3 tools/audit_python_compat.py
```

Deliberately not a matrix over interpreters: without a built extension the
tool's extension section reports "not built" and skips, so a matrix here
would look like evidence and be none.

**What would be worth a matrix.** The five `tested` rows in the interpreter
table are measured on osx-arm64 only. The same measurement on Linux is three
commands against a Linux-built extension, and CI already builds one in the
`python` job. Adding a step there that runs `python/test_python_api.py` under
a non-pixi 3.10 and 3.12 would move those rows to two platforms for almost no
runner time. That is the highest-value CI change available and it belongs to
whoever owns the workflow.

## Edit 5: `packaging/matrix/platform_matrix.toml` — APPLIED

Three statements in this file were contradicted by the audit and have been
corrected. The file's own rule, that every fact in `[toolchain]` comes from
`pixi.lock`, is what produced the errors: the lock records the variant that
was *solved*, and the file read that as the variant that *exists*.

1. **`[toolchain]`.** The comment claiming the `python 3.14.*` pin is what
   makes cp314 the only interpreter now says that the `python-gil` half is a
   real requirement and the `python 3.14.*` half is a solver outcome. Added
   `python_toolchain_floor = "3.10"`, with both derivations.
2. **The interpreter table.** The single `cp313 and earlier = unsupported`
   row, whose reason read "No MAX build for these interpreters in the pinned
   channel", was false and is gone. In its place: five `tested` rows,
   cp310 through cp314, each citing `docs/PYTHON_SUPPORT.md`; a `cp39` row at
   `unsupported` quoting the abort; the unchanged `cp314t` row, now saying
   explicitly that it is the one row nobody measured.
3. **The `abi3` row.** Conclusion kept, reasoning replaced. The old text
   implied building against the limited API was an option this repository
   has. It is not, and it is also not the mechanism it would need.

A new `build_target = true` key marks the one interpreter that compiles the
extension, which is the invariant the old "exactly one supported row" check
was really reaching for.

## Edit 6: `packaging/matrix/validate_matrix.py` — APPLIED

`check_python_rows` required exactly one interpreter row whose status was not
`unsupported`, with the message "The toolchain pins one; a second row means
the pin moved or the matrix is wrong." That premise was the one Edit 5
corrects, and the check would have failed on a true matrix.

It now enforces the two invariants separately:

- exactly one row carries `build_target = true`, and its version matches
  `python_pin`, and it is not `unsupported`
- at least one row is runnable, every runnable row is at or above
  `python_toolchain_floor`, and no runnable row claims to work without the
  GIL

Both the positive and the negative cases were exercised. `validate_matrix.py`
passes against the corrected matrix, and synthetic rows confirm that two
build targets, zero build targets, a runnable row below the floor, a runnable
row without the GIL, and an all-unsupported table each fail.

Docstring item 5 was updated to match.

## Edit 7: prose that restates the corrected claim — PENDING

Three documents repeat what Edit 5 corrected. None needs new reasoning.

**7a. `docs/PLATFORM_MATRIX.md`, the Interpreters section.** It opens "One,
and it is not a simplification", which is now wrong twice over, and its
`cp314` row ends "so the interpreter the wheel targets is the interpreter MAX
pins". The table should become the five-row shape of the corrected
`platform_matrix.toml`, with the heading changed to something like "One build
target, five runnable". The sentence below the table, "All four facts come
from `pixi.lock`, which is the only place the toolchain is pinned", should
gain the qualifier that the lock records the variant that was solved and not
the variants that exist, because that gap is what produced the error.

**7b. `packaging/linux/README.md`.** The "What is deliberately not supported"
table has a row whose Why column reads "`max 26.5.0` pins `python 3.14.*` and
depends on `python-gil`." Split it; the halves are different kinds of fact:

| System | Behavior | Why |
| --- | --- | --- |
| Free-threaded builds | `pip` refuses by tag | `max 26.5.0` depends on `python-gil`, which no free-threaded build provides. A real toolchain limit |
| CPython 3.9 and earlier | `pip` refuses by tag | The extension aborts on `Py_NewRef`. See `docs/PYTHON_SUPPORT.md` |
| CPython 3.10 to 3.13 | no wheel published yet | Not a limit. The code runs there; no wheel has been built by those interpreters |

**7c. `docs/COMPATIBILITY_POLICY.md` section 6.1.** The paragraph opens
"`requires-python = ">=3.14"` today, and that is a hard floor rather than a
preference." Suggested replacement, keeping the three policy bullets below it
unchanged:

> **Interpreter support.** `requires-python = ">=3.10"`. That floor is
> measured rather than assumed: the extension links no libpython and resolves
> CPython entry points by name at runtime, so one build serves several
> interpreters, and 3.10 through 3.14 all import it and pass the Python API
> suite while 3.9 aborts. See [docs/PYTHON_SUPPORT.md](PYTHON_SUPPORT.md).
> There is no `abi3` build and none is available to this project, so a wheel
> still serves exactly one CPython minor version and the floor is not a
> promise that a wheel exists for every version above it.

The three bullets that follow are correct as written. The first, that adding
an interpreter is additive and lands in a minor release, is what makes
publishing wheels for 3.10 through 3.13 a later release rather than a
blocker for the first one.

**7d. `docs/INSTALLATION.md`.** The requirements table's Python row cites
`requires-python` "which follows the interpreter the pinned MAX build
targets". Change the value to 3.10 and the parenthetical to "measured across
five interpreters; see `docs/PYTHON_SUPPORT.md`". The paragraph below it
saying the floor "is the most likely of these to move before the first
release" has now happened and should say so.

## Edit 8: a bug this audit found that is not about interpreters — PENDING

**`python/test_python_api.py` is not dependency-free, and the bare wheel test
has never run.**

`pixi.toml` says of `test-python` that it "stays dependency-free on purpose,
so it also runs against a bare wheel install", and `packaging/test_wheel.sh`
has a `bare install (no numpy)` step that runs exactly that file in a venv
with only the wheel in it. On any interpreter with no numpy, including 3.14,
that file fails:

```
TypeError: X is a sparse matrix, which needs numpy; install numpy or pass a
dense sequence
```

raised from `check_X_sparse` in `_arrays.py`. Confirmed version-orthogonal by
running it under CPython 3.14 in a `venv --without-pip`, where it fails at
the identical line.

It passes under `pixi run test-python` only because the default pixi
environment has numpy, which arrives as a dependency of `max` rather than
because anyone asked for it. So the guard against a numpy-free regression has
been reporting green while testing the numpy path.

Two ways to fix, and the choice belongs to whoever owns the suite:

- skip the sparse cases when `_arrays.have_numpy()` is false, which is what
  the rest of the suite's optional-dependency handling does
- move them to `python/tests/`, which is the suite that is allowed
  dependencies

Either way, `packaging/test_wheel.sh`'s bare step should then be run once for
real, because it never has been. This is the most load-bearing thing the
audit turned up that was not about interpreter versions, and it is
independent of everything else in this handoff.

## The release gate — PENDING

Two items for `docs/COMPATIBILITY_POLICY.md` section 12:

- **A8.** `python3 tools/audit_python_compat.py` green, so `requires-python`,
  the Python classifiers, the source, and `pixi.lock` still agree.
- **B6.** `EXTENSION_FLOOR` in `python/mojoboost/_compat.py` is at or below
  the lower bound of `requires-python` in `python/pyproject.toml`.

## What is still worth measuring

M1 is done. What remains, in order of value:

1. **Linux.** The five `tested` rows are osx-arm64 only. CI already builds a
   Linux extension; running the suite there under a non-pixi 3.10 and 3.12
   would extend every row to a second platform. Edit 4 has the shape.
2. **One non-3.14 wheel.** M3 and M4 in `docs/PYTHON_SUPPORT.md` section 10.
   This is the only thing standing between `tested` and `validated` for
   cp310 through cp313, and it also confirms the section 6 prediction that
   the tag follows the building interpreter.
3. **The optional dependency stack per interpreter.** M5. Which numpy,
   scipy, scikit-learn, and pandas a user of 3.10 actually gets, which is the
   number that belongs in a release note.
