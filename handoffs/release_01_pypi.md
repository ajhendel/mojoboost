# Task 01 handoff: PyPI metadata and release contract

Lane scope was `python/pyproject.toml`, `python/setup.py`,
`python/MANIFEST.in`, and `docs/PYPI_RELEASE.md`. Nothing outside that set
was touched. No test was written or run, nothing was built, nothing was
uploaded anywhere, nothing was committed.

Several lanes landed files while this one was in progress. The last
section, [Reconciliation](#reconciliation-with-work-that-landed-mid-lane),
records what changed as a result and what those lanes still need from
this one.

## Files changed

### `python/pyproject.toml` (rewritten in place)

Kept unchanged: `name`, `version = "0.1.0"`, `description`, `readme`,
`requires-python = ">=3.14"`, `license = "Apache-2.0"`,
`license-files = ["LICENSE"]`, `authors`, and `packages = ["mojoboost"]`.
All of those were already correct and evidence-backed.

Added or changed:

| Change | Evidence |
|---|---|
| `dependencies = []` stated explicitly | Section 6.1 of `docs/COMPATIBILITY_POLICY.md` promises numpy is never required; `packaging/test_wheel.sh` installs into a bare venv. It was implicit before, so nothing could be read as a broken promise. |
| `include-package-data = false` | setuptools defaults this to **true** under pyproject.toml configuration, so the wheel's contents were partly decided by sdist file finders. Off, the wheel is exactly `packages` plus `package-data`. This is what check C9 of `packaging/macos/inspect_wheel.py` ("no source-tree members") depends on being true by construction rather than by luck. |
| `package-data` gained `.libs/*.so` and `.libs/*.so.*` | Blocker C1 of `packaging/linux/check_metadata_ready.py`, whose required edit is quoted verbatim there. `packaging/linux/build_wheel_linux.sh` stages the ELF closure into `python/mojoboost/.libs` and sets RPATH to `$ORIGIN/.libs`. Both patterns are needed because ELF sonames carry version suffixes and `*.so` does not match `libfoo.so.1`. A missed pattern is dropped silently and surfaces as an ImportError on a user's machine. |
| Classifiers `Programming Language :: Python :: 3.14`, `:: 3 :: Only`, `:: Implementation :: CPython` | The `[[python]]` rows of `packaging/matrix/platform_matrix.toml`: cp314 only, with cp313, abi3 and cp314t all `unsupported`, and `requires_gil = true`. |
| Classifier `Operating System :: MacOS` replaced with `Operating System :: MacOS :: MacOS X` | The more specific registered classifier. |
| Classifiers `Intended Audience :: Developers` and `Topic :: Software Development :: Libraries :: Python Modules` | Non-claims, uncontroversial. |
| Extras `pandas`, `scipy`, `scikit-learn`, `dask`, `all` | One per guarded optional import: `mojoboost/_sklearn.py` (the sklearn tags hook), `mojoboost/inspection.py` and `__init__.py` (pandas), `mojoboost/_arrays.py` (scipy sparse), `mojoboost/dask.py` `_import_dask()`. The `dask` extra is spelled `dask[distributed]` to match the ImportError that module raises verbatim. `all` is self-referential so the union cannot drift. |
| Keywords `lightgbm`, `gpu` | `docs/LIGHTGBM_PARITY.md`, `docs/GPU_VALIDATION.md` (Apple M4). |
| `project.urls`: `Source` replaced by `Repository`, added `Issues` and `Documentation` | `Homepage` and `Source` were the same URL twice. `Documentation` points at `docs/` in the repository, which is where the docs are. No `Changelog` entry, because there is no CHANGELOG file to point at. |

Deliberately **not** claimed, each of which would have been a lie:

- Any `License ::` classifier. PEP 639 forbids one alongside the `license`
  expression, and setuptools>=77 rejects the pair outright.
- `Typing :: Typed`. There is no `py.typed` and no `.pyi` in
  `python/mojoboost/`.
- Any Linux `Operating System ::` classifier. Linux is a tier 1 tested
  *source* install, but both Linux wheel targets are `designed` with
  `builder = "does not exist yet"` at the time of writing. This matches
  what check C4 of `packaging/linux/check_metadata_ready.py` asks for: it
  warns rather than blocks, and says the classifier "should stay wrong
  until a Linux wheel is real". Add
  `"Operating System :: POSIX :: Linux"` at the same time as the first
  Linux artifact is published, not before.
- `Environment :: GPU :: NVIDIA CUDA` or the AMD equivalent. No such
  device has run a kernel.
- Any version floor on an extra. There is no measurement behind one.
- A capped `requires-python`. Section 6.1 states the floor form, and the
  `cp314-cp314` wheel tag is what actually refuses a wrong interpreter.

### `python/setup.py` (rewritten in place)

Behavior changes, not just comments:

1. **The macOS platform tag is overridable** through
   `MOJOBOOST_MACOS_DEPLOYMENT_TARGET`, defaulting to the previous
   hardcoded `26.0`. This is what the `macos-arm64-cp314-lowered` target
   in `packaging/matrix/platform_matrix.toml` needs to be buildable at
   all, and it is the variable
   `packaging/macos/build_release_wheel.sh` already exports.
2. **`plat_name` is only set on Darwin.** It was set unconditionally, so a
   build on Linux would have produced a wheel named
   `...-macosx_26_0_x86_64.whl` containing an ELF object, which pip on
   macOS accepts and then fails to import. This is blocker C2 of
   `packaging/linux/check_metadata_ready.py`, and it also removes the need
   for `packaging/linux/build_wheel_linux.sh` to fight the default with
   `--plat-name`, though that override still works and is still checked.
3. **macOS x86_64 is a hard stop**, with a message pointing at the
   `macos-x86_64` matrix row, instead of silently minting
   `macosx_26_0_x86_64`. This also catches a build running under Rosetta.
4. `MACOSX_DEPLOYMENT_TARGET` is documented as deliberately **not**
   consulted, because conda-style environments (which is what pixi
   provides here) export it at values unrelated to what the Mojo toolchain
   emitted, and inheriting it would tag a wheel with a floor its binary
   does not honor. `packaging/macos/build_release_wheel.sh` documents the
   same split from the other side, independently.

There was no metadata duplicated between `setup.py` and `pyproject.toml`
to remove. `setup.py` declared none and still declares none. It carries
only the two things PEP 621 has no field for: `has_ext_modules`, and the
platform tag.

### `python/MANIFEST.in` (new)

Exists for one purpose: to make an accidentally produced sdist harmless.
It excludes `mojoboost/_mojoboost.so` and `mojoboost/.dylibs/`, prunes
`tests`, `build`, `dist`, `.pytest_cache`, and drops bytecode.

It does **not** affect the wheel, because `include-package-data = false`.
That coupling is stated in both files. Turning `include-package-data` back
on without deleting this file would ship a wheel with no extension module
in it.

### `docs/PYPI_RELEASE.md` (new)

Owner-only procedure, deferring to `packaging/matrix/platform_matrix.toml`
for which artifacts exist and to section 12 of
`docs/COMPATIBILITY_POLICY.md` for whether a release may be cut.

Structure: status; authority table; the rule that a published artifact
comes from a release workflow and never a laptop, with the reason
(attributability, not correctness); a four-step sequence that separates
**binding the name** (pending trusted publishers, no upload, no version
number) from **proving the pipeline** (TestPyPI, `0.1.0.dev1`) from
**claiming the name** (PyPI, `0.1.0a1`, a pre-release pip will not install
without `--pre`) from **the production release** (gated on section 12);
preflight with exact commands; what "the identical artifact" does and does
not mean across two workflow runs; hash and attestation verification;
yanking, with never-delete and the reason; how to have no API token at
all; a summary of what `.github/workflows/release-provenance.yml`
guarantees; and what to do if the name is already taken.

## Reconciliation with work that landed mid-lane

Four lanes' files appeared during this one and change what this lane
should say. Everything below has already been applied.

**R1. The release workflow exists.** It is
`.github/workflows/release-provenance.yml`, not `release.yml`.
`docs/PYPI_RELEASE.md` originally specified a workflow to be written; it
now describes the one that exists and defers to
`docs/RELEASE_SECURITY.md` and `handoffs/release_10_security.md` for its
reasoning. Concretely changed:

- The pending-publisher table now says **Workflow name
  `release-provenance.yml`**. Configuring the publisher with the wrong
  filename produces a publisher that never matches and an upload that is
  rejected with a confusing error, so this was the one thing that had to
  be right.
- The workflow is `workflow_dispatch` only, with a `publish` choice input.
  Every step that said "push a tag and the workflow fires" now says "tag
  the commit, then run the workflow by hand", because the tag is what
  `packaging/macos/build_release_wheel.sh` requires, not what triggers
  anything.
- The environment names `testpypi` and `pypi` come from that input, so
  they are the only two possible spellings.

**R2. One run publishes to one index, so the TestPyPI and PyPI wheels are
two different builds.** The original draft claimed the promoted artifact
would be byte-identical, which is what a single-run two-upload workflow
would give and is not what this one gives. Rewritten to say plainly that
within a run the digest chain holds end to end (build job output, checked
again before attestation, downloaded unmodified by the publish job) and
that across runs it does not, because macOS wheels are not reproducible:
`packaging/linux/build_wheel_linux.sh` sets `SOURCE_DATE_EPOCH` from the
commit and `packaging/macos/build_release_wheel.sh` does not. A TestPyPI
to PyPI digest comparison is therefore **not** a valid check today, and
the document says so rather than instructing an owner to run a check that
will fail for an innocent reason. The two ways to close it, both other
lanes': set `SOURCE_DATE_EPOCH` in the macOS builder, or give the workflow
a publish mode that uploads one build to both indexes.

**R3. `packaging/linux/check_metadata_ready.py` addresses this lane by
name.** Its blocker C1 (package-data) is applied verbatim. Its blocker C2
(conditional `plat_name`) is applied in substance. Its warning C4
(classifiers) is deliberately left as it is, on that script's own
instruction. Its C5 (`requires-python`) already passed.

**R4. Env var naming converged without contact.** This lane chose
`MOJOBOOST_MACOS_DEPLOYMENT_TARGET`, and
`packaging/macos/build_release_wheel.sh` and
`packaging/macos/inspect_wheel.py` were written against a variable of that
exact name with the same rationale for ignoring
`MACOSX_DEPLOYMENT_TARGET`. Nothing needed changing; recording it because
a later reader will assume coordination happened.

## Edits other lanes must make (not made here)

**H1. Blocker C2 of `packaging/linux/check_metadata_ready.py` is now a
false positive.** It blocks when the string `macosx` appears anywhere in
`python/setup.py`, and the tag is built from that string, so the check
fires whether the assignment is conditional or not. `check_plat_name` is
now testing for the presence of a substring rather than for the defect.
Suggested predicate, which is still a text scan but distinguishes the two
cases:

```python
if "macosx" not in text or ('sys.platform' in text and '"darwin"' in text):
    rep.ok("C2", "the macOS platform tag is conditional on sys.platform")
    return
```

Nothing is unblocked by fixing it alone: C3 (patchelf absent from every
pixi environment) is a separate blocker owned by the integration owner,
and both must clear before `build_wheel_linux.sh` will run. Owned by the
Linux packaging lane.

**H2. `python/LICENSE` is gitignored and copied at build time.**
`.gitignore` has the line `python/LICENSE`, and `packaging/build_wheel.sh`
does `cp LICENSE python/LICENSE`. `license-files = ["LICENSE"]` therefore
resolves only after that script has run, and a `python -m build` from a
fresh clone fails on a missing license file with a message that does not
say "run build_wheel.sh". Check C10b of
`packaging/macos/inspect_wheel.py` catches the resulting wheel, which is
good, but catches it late. Two ways out, both outside this lane: un-ignore
and commit `python/LICENSE` (`.gitignore` owner), or leave it and accept
that the build scripts are the only supported entry point. Recommend the
first. It makes the wheel buildable from a checkout and removes a
build-order dependency for nothing.

**H3. `packaging/test_wheel.sh` does
`WHEEL=$(ls python/dist/mojoboost-*.whl)`.** That breaks the moment
`dist/` holds more than one wheel, which is what happens when the
lowered-deployment-target wheel of H4 is built alongside the default one.
`packaging/macos/build_release_wheel.sh` already handles the same
situation correctly by asserting exactly one. Owned by the packaging lane.

**H4. The `macos-arm64-cp314-lowered` matrix row is now buildable.** Run
the release builder with `MOJOBOOST_MACOS_TARGET=12.0`, which exports both
halves. The matrix says the four bundled MAX dylibs are already built for
macOS 11.0, so only the Mojo-compiled extension imposes the 26.0 floor.
Whether the Mojo compiler honors a lowered target is untested; check C1 of
`packaging/macos/inspect_wheel.py` is what turns the request into a fact.
Do not create a wheel with the lowered filename until it passes. Owned by
the matrix and macOS packaging lanes; the env var exists now so neither
has to edit `setup.py`.

**H5. `python/mojoboost/__init__.py` line 265 carries
`# Keep in sync with python/pyproject.toml.`** Still accurate, left alone
on purpose. See the first unresolved decision below.

**H6. Section 6.1 of `docs/COMPATIBILITY_POLICY.md` says only `mojoboost`
and `mojoboost.callback` are supported import paths**, and names
`mojoboost.inspection` as the open case. `python/mojoboost/` now also
ships `dask.py`, `cv.py`, and `device_selection.py`, reachable only as
submodule imports. The `dask` extra added here implies
`import mojoboost.dask` is something users do. Either the policy's list
needs extending or those modules need re-exporting; release gate item C5
already covers `inspection` and should probably cover all four. Owned by
the compatibility-policy lane. Nothing in this lane is blocked on it.

## Unresolved metadata decisions

1. **Version source of truth.** The version is in four places:
   `pixi.toml`, `python/pyproject.toml`, `python/mojoboost/__init__.py`,
   and `packaging/matrix/platform_matrix.toml`. `pyproject.toml` could
   read it from the module with `dynamic = ["version"]` plus
   `[tool.setuptools.dynamic] version = {attr = "mojoboost.__version__"}`,
   which setuptools resolves by AST without importing, so it works with no
   `.so` present. **Not done**, because it deletes the `[project] version`
   row that section 1.1 of the compatibility policy names as one of three
   locations that must agree, and that document belongs to another lane.
   If that lane wants it, the change here is three lines.
2. **Author email.** `authors = [{ name = "Andrew Hendel" }]` carries no
   email; `pixi.toml` has `ajhendel@gmail.com`. PyPI renders author email
   publicly on the project page and in METADATA permanently. Left off,
   with `Issues` in `project.urls` as the contact route. Owner's call.
3. **`arrow` and `polars` extras.** Not added. The only evidence in the
   tree is duck-typed slicing in `mojoboost/cv.py` lines 169 and 172 and a
   docstring mention in `mojoboost/device_selection.py` line 418, while
   `pixi.toml`'s `pytest` feature does install `pyarrow` and `polars`. If
   the table-input lane lands first-class support, add
   `arrow = ["pyarrow"]` and `polars = ["polars"]` alongside the others
   and extend the `all` self-reference.
4. **Extra version floors.** Every extra is unpinned. Nothing in the
   repository measures a minimum version of numpy, pandas, scipy,
   scikit-learn, or dask, so a floor would be invented. Add one when
   something fails below it.
5. **`Development Status :: 3 - Alpha`.** Correct for 0.1.0, and it has to
   be revisited at the release that stops being an alpha. A stale
   Development Status is the most common piece of dishonest packaging
   metadata there is, and the release note is where it should be stated.
6. **Whether to publish an sdist at all.** The matrix says no, and the
   reasoning (pip falls back to a source build the user cannot perform)
   holds. Nothing here changes it. Revisit only if a source build without
   the Mojo toolchain ever becomes possible, which today it is not.

## Validation commands for later

None were run. All require a built extension, which requires the Mojo
toolchain, which this lane did not invoke.

Metadata only, no upload:

```
pixi run -e pkg python -m pip install validate-pyproject[all]
pixi run -e pkg validate-pyproject python/pyproject.toml
python3 packaging/linux/check_metadata_ready.py
pixi run -e pkg build-wheel
pixi run -e pkg python -m twine check python/dist/*.whl
python3 packaging/macos/inspect_wheel.py python/dist/mojoboost-*.whl
unzip -l python/dist/mojoboost-*.whl
unzip -p python/dist/mojoboost-*.whl 'mojoboost-*.dist-info/METADATA'
```

What each is checking, specifically:

- `validate-pyproject` catches a schema error in the file.
- `check_metadata_ready.py` should now report C1 ok and C4 as the
  intended warning. It will still report C2 as a blocker until H1 is
  fixed, and that report is a false positive.
- `twine check` catches an unregistered trove classifier. The three new
  `Programming Language ::` rows and the new `Topic ::` row have **not**
  been checked against PyPI's registered list, and an unregistered
  classifier is rejected at upload time, after a version number has
  already been committed to.
- `inspect_wheel.py` covers C9 (no source-tree members, which is what
  `include-package-data = false` is for), C10a and C10b (metadata and
  license file present), and C13 (`Root-Is-Purelib: false`, which is what
  the `BinaryDistribution` subclass is for). Those four are the checks
  that would fail first if anything in this lane is wrong.
- `unzip -l` must show `mojoboost/__init__.py` plus every other module
  in `python/mojoboost/` at the time of the build (the set is still
  growing in other lanes, so count it against the directory rather than
  against a number written here), `mojoboost/_mojoboost.so`, four
  `.dylibs/*.dylib`, and `dist-info/licenses/LICENSE`, and nothing else.
  The wheel currently sitting in `python/dist/` predates most of the
  package and contains only `__init__.py` and the binaries. It is stale
  and proves nothing about the current metadata.
- `unzip -p ... METADATA` must show every `Requires-Dist` carrying an
  `; extra == "..."` marker. A bare one breaks the dependency-free
  contract of section 6.1.

The platform tag, which no check in this lane automates:

```
otool -l python/mojoboost/_mojoboost.so | grep -A4 LC_BUILD_VERSION
```

`minos` must equal `DEFAULT_MACOS_TARGET` in `python/setup.py`, currently
`26.0`, and must equal the version in the wheel filename. Check C1 of
`packaging/macos/inspect_wheel.py` is the automated form.

The behavior change in `setup.py` that wants exercising once:

```
MOJOBOOST_MACOS_TARGET=12.0 bash packaging/macos/build_release_wheel.sh
```

should produce `mojoboost-0.1.0-cp314-cp314-macosx_12_0_arm64.whl`, and
check C1 is what says whether the binary agrees. Do not publish it if it
does not (H4).

Install behavior, which needs a second machine or a container:

```
pixi run -e pkg test-wheel
```

## Actions deliberately not executed

- No Mojo, Pixi, Python, pytest, pip, `build`, `twine`, `auditwheel`, or
  `delocate` invocation of any kind. No benchmark, no CI run, no
  background job, no polling loop. Repository inspection was `rg`, `sed`,
  `ls`, `git status`, and `unzip -l` on the stale wheel.
- No test was written and none was run.
- Nothing was uploaded to PyPI or TestPyPI, and no account, project,
  publisher, environment, or token was created or inspected. Whether the
  name `mojoboost` is currently available on PyPI was **not checked**.
  `docs/PYPI_RELEASE.md` has a section on what to do if it is taken, and
  checking availability is the first thing the owner should do.
- `otool` was not run, so the `26.0` deployment target is carried forward
  from the previous `setup.py` and from `platform_matrix.toml` rather than
  re-measured. It is documented as a fact that needs verifying, not as one
  that was verified here.
- The stale wheel in `python/dist/` was listed but not deleted, rebuilt,
  or used to justify any claim. It contains a single-module version of the
  package and is evidence of nothing.
- No file outside the four in the lane scope was edited. In particular
  `README.md`, `python/README.md`, `packaging/` including everything that
  appeared there mid-lane, `.github/workflows/`, `.gitignore`,
  `pixi.toml`, `docs/COMPATIBILITY_POLICY.md`, `docs/PLATFORM_MATRIX.md`,
  `packaging/matrix/platform_matrix.toml`, and every source module were
  left alone. Everything that wants changing in them is in the handoff
  list above.
- Nothing was committed. `git status` was clean when this lane started;
  the other modified and untracked paths in it now belong to concurrent
  lanes.
