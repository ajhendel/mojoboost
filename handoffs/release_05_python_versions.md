# Handoff: the Python version support contract

Lane 05 of a parallel round. This document is the part of the work that
touches files this lane was not allowed to edit. Every edit below is written
out exactly, ready to apply, and **none of it has been applied**.

The reasoning behind all of it is [`docs/PYTHON_SUPPORT.md`](../docs/PYTHON_SUPPORT.md).
This handoff does not restate it; it states what to change and where.

## What this lane added

| Path | What it is |
|---|---|
| `docs/PYTHON_SUPPORT.md` | The audit, the support matrix, and the experiments that would move a row |
| `tools/audit_python_compat.py` | Standard-library-only scanner: source syntax, packaging metadata, `pixi.lock` toolchain pins, and the CPython entry points named inside a built `_mojoboost.so`. Imports nothing, builds nothing, loads no extension |
| `python/mojoboost/_compat.py` | Two interpreter checks and one guarded import helper. No shims: the audit found nothing to shim |

## What was executed

Nothing. No Python, no Mojo, no pixi, no pytest, no pip, no build, no CI, no
network query, no benchmark. `tools/audit_python_compat.py` was written but
has never been run, so its output appears nowhere in this repository and no
line in `docs/PYTHON_SUPPORT.md` quotes it.

The audit is static inspection of files already on disk: the Python sources,
`pyproject.toml`, `setup.py`, `pixi.toml`, `pixi.lock`, the conda-meta file
lists in `.pixi/envs/`, the local rattler repodata cache the solve wrote, the
built `_mojoboost.so` and the four dylibs beside it, and `ci.yml`.

## The answer Task 01 asked for

**`requires-python` remains unresolved. Leave it at `>=3.14` for now.**

Three reasons, in order of how much they matter.

1. **The 3.14 pin is almost certainly accidental, and "almost certainly" is
   not a basis for lowering a floor.** `pixi.toml` pins no Python. Every
   environment in `pixi.lock`, including `bench`, which asks only for
   `python >=3.11`, resolved to 3.14.6 because with no pin the solver takes
   the newest. `mojo 1.0.0` and `mojo-python 1.0.0` require only
   `python >=3.10`, `max-core 26.5.0` has no Python dependency at all, and
   the pinned channel publishes `max-26.5.0` as five variants, 3.10 through
   3.14, on all three of the repository's platforms. The pin comes from the
   `max` metapackage, whose Python API this repository never imports.
2. **One thing in the built extension could still be a real 3.13 floor, and
   it has not been tested.** `Py_GetConstantBorrowed`, a CPython 3.13
   addition, is in the extension's entry point table with no
   conditional-resolution diagnostic beside it, while two other entry points
   in the same table do carry one. Section 9.1 of `docs/PYTHON_SUPPORT.md`
   has the detail. A byte scan cannot tell an eagerly resolved name from a
   lazily resolved one.
3. **Lowering a floor is additive under section 6.1 of the compatibility
   policy and lands in a minor release.** There is no cost to being right
   later and a real cost to shipping a floor that turns out to be a lie in
   either direction.

### The value to write when the evidence arrives

Experiment M1 in section 10 of `docs/PYTHON_SUPPORT.md` is one command and it
decides this. Its three possible outcomes map onto three values:

| M1 outcome | Write | Also |
|---|---|---|
| Imports under 3.13, fails under 3.12 | `requires-python = ">=3.13"` | Add the `3.13` classifier. This is the expected outcome and the recommended target for a first release |
| Imports under 3.10 | `requires-python = ">=3.10"` | Add classifiers `3.10` through `3.14`. `3.10` is the floor `mojo 1.0.0` imposes and nothing below it is reachable |
| Fails under 3.13 | `requires-python = ">=3.14"` stays | Record the traceback and say in the release note that the floor was measured, not assumed |

Do not write any of these before M1 runs. Do not add a `<3.15` cap in any
case; `python/pyproject.toml` already carries the comment explaining why one
is absent, and that comment is correct.

## Edit 1: `python/pyproject.toml` (Task 01)

Two changes, and only one of them is conditional.

**1a, unconditional.** The comment block above `classifiers` currently
explains `requires-python` as a floor and says nothing about where the floor
came from. Replace the paragraph that begins "`requires-python` is a floor,
not a range" with this, which keeps every sentence that is still true and
adds the pointer:

```toml
# `requires-python` is a floor, not a range, which is what section 6.1 of
# docs/COMPATIBILITY_POLICY.md states and what this project keeps. It is
# not the mechanism that stops a wrong install: the wheel carries a
# `cp314-cp314` tag and no abi3 build exists, so pip refuses a mismatched
# interpreter by tag before `requires-python` is consulted. A capped
# `<3.15` would add nothing and would have to be edited on every release
# the toolchain permits.
#
# The value itself is under review and is not settled. docs/PYTHON_SUPPORT.md
# is the authority: it finds that the pinned toolchain publishes builds for
# CPython 3.10 through 3.14 and that this project pinned none of them, so
# 3.14 is the interpreter the solver chose rather than one the toolchain
# requires. It stays here until the experiment in section 10 of that
# document has run. Do not lower it on the strength of the audit alone.
```

**1b, conditional on M1.** The `classifiers` list currently claims
`"Programming Language :: Python :: 3.14"` and nothing else per-minor. Every
minor version at or above whatever `requires-python` ends up saying gets a
row, and no row is added for a version `requires-python` excludes.
`tools/audit_python_compat.py` fails on that mismatch in either direction, so
the check is mechanical once the task in Edit 3 exists.

Nothing else in `pyproject.toml` changes. `dependencies = []`, the optional
extras, `include-package-data = false`, and the `package-data` list are all
independent of the interpreter question.

## Edit 2: the two unguarded extension imports

Five modules reach the extension. Only two of them do it at module scope, and
those two are the ones `python/mojoboost/_compat.py` is for.

| Site | Kind | Needs the guard |
|---|---|---|
| `python/mojoboost/__init__.py:253` | module scope | **Yes.** This is the first line `import mojoboost` reaches, seven lines before it imports `basic` |
| `python/mojoboost/basic.py:65` | module scope | **Yes.** Section 2 of the compatibility policy makes `mojoboost.basic` a supported import path, so it can be reached without `__init__` having run first |
| `python/mojoboost/device_selection.py`, in `_native_gpu_available` | deferred, already guarded | No. It treats the absence as non-fatal on purpose, so the policy layer stays usable on a machine that has not built the extension. Correct as written |
| `python/mojoboost/dask.py:1258` | deferred | No. Unreachable until `__init__` has already succeeded |
| `python/mojoboost/inspection.py:152` and `:677` | deferred | No, same reason |

**2a. `python/mojoboost/__init__.py`.** As of this reading, line 253 is:

```python
from . import _arrays, _eval, _mojoboost, callback as _callback
```

Replace it with:

```python
from . import _arrays, _compat, _eval, callback as _callback

# Not `from . import _mojoboost`. This is the first line `import mojoboost`
# reaches, and on an interpreter this build does not serve it fails with a
# dynamic loader error naming a CPython C symbol, which does not tell the
# reader which interpreter to use instead. _compat adds that.
# See docs/PYTHON_SUPPORT.md sections 8 and 9.
_mojoboost = _compat.import_extension()
```

Two `_mojoboost.*` call sites follow in that file, at lines 2118 and 2120.
Neither changes: `_mojoboost` stays a module-level name with the same binding.

**2b. `python/mojoboost/basic.py`.** As of this reading, lines 62 to 65 are:

```python
import os as _os
import tempfile as _tempfile

from . import _arrays, _eval, _mojoboost
```

Replace the last of those with:

```python
from . import _arrays, _compat, _eval

# See the same call in __init__.py. Repeated here because
# `import mojoboost.basic` is a supported import path (section 2 of
# docs/COMPATIBILITY_POLICY.md) and can be the first thing a program runs.
_mojoboost = _compat.import_extension()
```

The 25 `_mojoboost.*` call sites below it are unchanged.

`_compat.import_extension()` re-raises the original `ImportError` untouched
when the interpreter is one this build does serve, so neither edit hides a
genuine loader failure. Doing this twice costs nothing at runtime: the second
call gets the module out of `sys.modules`.

If `_compat.PYTHON_FLOOR` and `requires-python` are ever allowed to disagree,
both guards report the wrong number. They move in the same commit. See the
release gate item at the end of this handoff.

## Edit 3: `pixi.toml`

One task, in the default `[tasks]` table next to `check-parity`, which it
resembles on purpose: standard library only, builds nothing, imports nothing.

```toml
# Checks the declared Python floor against the source, the packaging
# metadata, pixi.lock, and any built extension. Standard library only, it
# builds nothing and imports no extension, so it is cheap enough to run on
# every change and it fails when requires-python and the tree disagree.
audit-python-compat = "python3 tools/audit_python_compat.py"
```

### The open pixi question, which is not this lane's to decide

`pixi.toml` depends on `max = ">=26.5.0,<27"`. That package is the MAX Python
API. The Mojo packages `src/` actually imports (`max.gpu.host`,
`max.gpu.memory`, `max.gpu.sync`, `max.algorithm`) ship as
`lib/mojo/max.mojoc` and `lib/mojo/algorithm.mojoc` inside `max-core`, which
has no Python dependency at all. The `max` package's only appearance anywhere
in this repository outside the manifest is `bench/real_data/envinfo.py:119`,
which runs `import max; print(max.__version__)` to record a version string.

So the dependency carrying the entire Python pin is a dependency on an API
this project does not call. Two ways to move it, and the choice belongs to
whoever owns `pixi.toml`:

- Pin the interpreter explicitly, `python = "==3.13.*"` in `[dependencies]`.
  Keeps `max`, keeps `envinfo.py` working, makes the interpreter a decision
  instead of a solver outcome. **Recommended**, and it is what experiment M3
  needs anyway.
- Depend on `max-core` instead of `max`. Removes the pin entirely and lets
  the interpreter float, at the cost of `envinfo.py` and of any future use of
  the MAX Python API.

Whichever is chosen, `pixi.lock` is regenerated and every derived claim in
`packaging/matrix/platform_matrix.toml` has to be re-derived from it. That is
what Edit 5 is.

## Edit 4: `.github/workflows/ci.yml`

One job, modeled on the existing `parity` job, which already runs a
standard-library checker on a bare runner:

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

Deliberately **not** a matrix over interpreters. A matrix here would look like
evidence about 3.10 through 3.13 and would be nothing of the kind: without a
built extension the tool's extension section reports "not built" and skips.
The interpreter matrix that would be evidence is experiments M2 through M4 in
`docs/PYTHON_SUPPORT.md` section 10, and it needs a build, not a checkout.

One thing worth knowing about the current CI: the `test` and `python` jobs
install the pixi environment, which is 3.14 and only 3.14. There is no
interpreter in CI today that could disagree with the declared floor, which is
most of why the floor has gone unexamined.

## Edit 5: `packaging/matrix/platform_matrix.toml`

Three statements in this file are contradicted by the audit. The file's own
rule, that every fact in `[toolchain]` comes from `pixi.lock`, is what caught
them: `pixi.lock` records the variant that was solved, and the file read that
as the variant that exists.

**5a. The `cp313` row.** Its `reason` currently opens "No MAX build for these
interpreters in the pinned channel." That is false. Replace the `reason` with:

```toml
reason = """
The pinned channel publishes max 26.5.0 for CPython 3.10 through 3.14 on all
three platforms, so this row is not a toolchain limit. It is `unsupported`
because no extension has ever been built for these interpreters and no import
has ever been attempted on one, and because one entry point in the built
extension's table (Py_GetConstantBorrowed, CPython 3.13) may be a hard floor.
docs/PYTHON_SUPPORT.md section 9 holds the matrix and section 10 the
experiment that would move this row.
"""
```

**5b. The `[toolchain]` block.** The comment above `python_pin` says the pair
of `python 3.14.*` and `python-gil` "is what makes cp314 the only interpreter
mojoboost builds against". Half of that is right. Replace the comment and add
one key:

```toml
# max 26.5.0 depends on `python-gil`, which is a real toolchain requirement and
# is what excludes every free-threaded build. The `python 3.14.*` pin beside it
# is not: max 26.5.0 is published as five per-interpreter variants, 3.10release
# through 3.14release, on all three platforms, and pixi.toml pins no python, so
# the solver took the newest. cp314 is the interpreter this project happens to
# build against, not the only one it could. See docs/PYTHON_SUPPORT.md.
python_pin = "3.14.*"
requires_gil = true
# The lowest interpreter any pinned toolchain package will accept: mojo 1.0.0
# and mojo-python 1.0.0 both depend on `python >=3.10`. Nothing below this is
# reachable with this toolchain, whatever else changes.
python_toolchain_floor = "3.10"
```

`validate_matrix.py` reads `[toolchain]` by key and ignores unknown ones, so
adding `python_toolchain_floor` is inert until something reads it. Adding a
check that it still follows from `pixi.lock` belongs with whoever owns that
script.

**5c. The `abi3` row.** Its conclusion is right and should stay. Its reason,
"The extension is not built against the limited API, so there is no abi3
wheel to ship", implies that building against the limited API is an option
this repository has. It is not. Replace the `reason` with:

```toml
reason = """
Py_LIMITED_API is a C macro set before including Python.h, and this project
has no C source, no Python.h, and no compile step where it could be set: the
extension is emitted by `mojo build --emit shared-lib` and its CPython glue
comes from Modular's std.mojoc. setup.py has no Extension() to pass
py_limited_api to. abi3 is not a lever this repository has.

It is also not the lever it would need. abi3 exists so a binary linked against
libpython can serve several interpreters; this binary links no libpython and
resolves CPython entry points by name at runtime. One wheel per interpreter is
the shape of this project. Do not tag a wheel abi3 to widen its reach.
"""
```

### The check that blocks a second interpreter row

`packaging/matrix/validate_matrix.py`, around line 161, requires exactly one
`python` row whose status is not `unsupported`:

```
expected exactly one build interpreter, found [...]. The toolchain pins one;
a second row means the pin moved or the matrix is wrong.
```

That premise is the one 5b corrects. The day a second interpreter is
`validated` or `tested`, this check fails on a true matrix. It should become a
check that every non-`unsupported` row's `version` is consistent with the
`[toolchain]` block and with `requires-python`, rather than a count. Owner of
that script decides the form; the count is the part that has to go.

## Edit 6: prose that restates the same claim

Three documents repeat the toolchain claim 5b corrects. None of them needs new
reasoning, only the corrected sentence.

**6a. `docs/PLATFORM_MATRIX.md`, the Interpreters table.** The `cp314` row
reads "`max 26.5.0` ships as a 3.14 build and depends on `python 3.14.*`. The
extension is compiled inside that environment, so the interpreter the wheel
targets is the interpreter MAX pins." The last clause is the error. Suggested
replacement for that cell:

> `max 26.5.0` is published as five per-interpreter variants, 3.10 through
> 3.14, and `pixi.toml` pins no python, so the solver took the newest. The
> extension is compiled inside that environment and the wheel is tagged for
> it. This is the interpreter the project builds against, not the only one it
> could. See [docs/PYTHON_SUPPORT.md](PYTHON_SUPPORT.md)

The `cp313 and earlier` row's "No MAX build for them in the pinned channel"
becomes the 5a wording, shortened. The sentence below the table, "All four
facts come from `pixi.lock`, which is the only place the toolchain is
pinned", should gain the qualifier that the lock records the variant that was
solved and not the variants that exist, because that gap is exactly what
produced these two rows.

**6b. `packaging/linux/README.md`, line 226.** The "What is deliberately not
supported" table has a row whose Why column reads "`max 26.5.0` pins
`python 3.14.*` and depends on `python-gil`." Split it, because the two
halves are not the same kind of fact:

| System | Behavior | Why |
| --- | --- | --- |
| 3.14 free-threaded | `pip` refuses by tag | `max 26.5.0` depends on `python-gil`, which no free-threaded build provides. A real toolchain limit |
| CPython 3.13 and earlier | `pip` refuses by tag | No wheel is built for them. Not a toolchain limit: see `docs/PYTHON_SUPPORT.md` |

**6c. `docs/COMPATIBILITY_POLICY.md` section 6.1.** The paragraph opens
"`requires-python = ">=3.14"` today, and that is a hard floor rather than a
preference. The extension module is built by the Mojo toolchain against a
specific CPython, links no libpython, and carries a `cp314-cp314` tag."
Those two sentences contradict each other, and the second one is the accurate
half. Suggested replacement for the paragraph, keeping the policy bullets
below it unchanged:

> **Interpreter support.** `requires-python = ">=3.14"` today. How much of
> that floor is real is an open question rather than a settled one: the
> extension links no libpython and resolves CPython entry points by name at
> runtime, and the pinned toolchain publishes builds for CPython 3.10 through
> 3.14. [docs/PYTHON_SUPPORT.md](PYTHON_SUPPORT.md) is the authority on what
> the evidence supports and on what would have to run before the floor moves.
> There is no `abi3` build and none is available to this project, so a wheel
> serves exactly one CPython minor version.

The three policy bullets that follow ("Adding support ... is additive",
"Dropping one follows section 3.1", "The `requires-python` floor and the wheel
tags shipped are stated in every release note") are all correct as written and
should not change. The first one is what makes lowering the floor a minor
release rather than a decision that has to be made before 1.0.

**6d. `docs/INSTALLATION.md`.** The requirements table's Python row cites
`requires-python` "which follows the interpreter the pinned MAX build
targets". Change the parenthetical to "which follows the interpreter the
project's pixi environment resolved to". The paragraph below it, which already
says the floor "is an artifact of the toolchain pin rather than a language
requirement" and points at `docs/PYTHON_SUPPORT.md`, is correct and needs
nothing.

## Edit 7: the clean-install fixtures

`packaging/matrix/smoke/clean_install_macos.sh:49` and
`clean_install_linux.sh:51` both default to `PY=${PYTHON:-python3.14}` and
refuse with a message naming 3.14. Both already honor a `PYTHON` environment
variable, so both already work for experiment M4 without any change. Note
only, and low priority: when the floor moves, the default and the refusal
message move with it, and a fixture that hardcodes the old floor will refuse
to test the new one.

## The release gate

Nothing in `docs/COMPATIBILITY_POLICY.md` section 12 covers this. Two items
would, if whoever owns that document wants them:

- **A8.** `python3 tools/audit_python_compat.py` green, so `requires-python`,
  the Python classifiers, the source, and `pixi.lock` still agree.
- **B6.** `PYTHON_FLOOR` in `python/mojoboost/_compat.py` equals the lower
  bound of `requires-python` in `python/pyproject.toml`.

B6 is the pair Edit 2 introduces, and it is the same kind of item as B1.

## The one experiment worth doing first

Everything above is bookkeeping except this. From a checkout with the
extension already built, and with an interpreter that is not the pixi one:

```
cd python
/path/to/python3.13 -c "import mojoboost; print(mojoboost.__version__)"
/path/to/python3.12 -c "import mojoboost; print(mojoboost.__version__)"
/path/to/python3.10 -c "import mojoboost; print(mojoboost.__version__)"
```

It needs no rebuild. The extension's CPython entry point table is compiled in
from `mojo-compiler 1.0.0-release`, which has one build and no per-interpreter
variants, so the table does not vary with the interpreter the build ran under.
Section 4 of `docs/PYTHON_SUPPORT.md` is the argument for that and section 10
lists the four experiments that follow from whatever this one returns.

Record the output verbatim, including the failures. A traceback naming a
missing C symbol is the most informative result any of this can produce, and
it is the difference between a floor that was measured and one that was
inherited.
