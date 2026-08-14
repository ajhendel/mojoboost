# Task 04 handoff: pip-first installation experience

Lane: `04_PIP_INSTALL_UX.txt`. Documentation and one example script only.
Nothing in this lane was executed. No test was written or run, no build was
started, no package was installed, and no command in any file below was run
to produce the output shown next to it.

## Files changed

| File | Change |
|---|---|
| `docs/INSTALLATION.md` | New. The full installation contract: three states, the first five minutes, diagnostics, and every error message with what it means |
| `README.md` | `## Five-minute start` replaced by `## Installation`, which leads with `pip install mojoboost` and says plainly that it does not work yet. Added the first-five-minutes pointer and a devices paragraph. Corrected the tiny example to `min_data_in_leaf=1`. Updated the wheel paragraph in `### Tests and wheels` to stop implying a published macOS wheel. `## Python API` now names `pixi run build-python` |
| `python/README.md` | The PyPI long description. Added the alpha warning, an `## Installing` section with the three states, `## The first five minutes`, and `## When something goes wrong` with a message table. Rewrote `## Platform support` to stop claiming wheels exist. Fixed a stale multiclass-is-CPU-only claim in `## Device selection`. Expanded `## Links` |
| `examples/install_smoke.py` | New. Standard-library-only script that walks the same seven steps, prints the diagnostics block a bug report needs, and names the common import failures in its own error handling |

Not touched, by ownership rule: `python/pyproject.toml`, `python/setup.py`,
`packaging/`, `.github/workflows/`, `.github/ISSUE_TEMPLATE/`, anything under
`src/`, `bindings/`, or `python/mojoboost/`, and
`examples/apple_silicon/`.

## Claims that depend on another lane before they can be published

Every row is written in `docs/INSTALLATION.md`, `README.md`, or
`python/README.md` today. Each is either already hedged as unavailable or is
a forward-looking contract. The **Action** column is what has to happen
before the claim can be stated as fact.

### Depends on Task 01 (PyPI metadata and release contract)

| Claim | Where | Action |
|---|---|---|
| The public package name is `mojoboost` | all three docs, the `pip install mojoboost` line | Confirm the name is reservable and reserved. If it is taken, every install command in all three files changes |
| No source distribution is ever published, so pip cannot fall back to a Mojo compile | `INSTALLATION.md` "Why there is no sdist", `README.md`, `python/README.md` | Task 01 must keep the sdist out of the release contract. `docs/PLATFORM_MATRIX.md` already lists `sdist` as `unsupported`; if that changes, the `--only-binary=:all:` guidance and the whole "no silent compile" promise are wrong |
| `pip install "mojoboost[numpy]"` pulls numpy in | `INSTALLATION.md` state 1 | The `numpy` extra exists in `python/pyproject.toml` today. Confirm Task 01 keeps that name |
| Version `0.1.0` appears in every example wheel filename | `INSTALLATION.md` state 2 | Confirm the first published version. If Task 01 moves the version source or the first release is not `0.1.0`, the filename table needs one edit |
| The alpha wording, license, and URLs on the PyPI page | `python/README.md` | `python/README.md` is the `readme` in `python/pyproject.toml`, so it renders on PyPI. GitHub admonition syntax does not render there, which is why this file uses a plain blockquote |
| **The "it does not work yet" paragraph itself** | `python/README.md` `## Installing` | This is the one that must not survive publication. The moment a wheel is on PyPI, the sentence "It does not work yet" is false on the page a user reads after installing it. Swapping that section is a release-checklist item, not a follow-up |

### Depends on Task 02 (macOS ARM64 wheels)

| Claim | Where | Action |
|---|---|---|
| The macOS wheel is `mojoboost-0.1.0-cp314-cp314-macosx_26_0_arm64.whl` | `INSTALLATION.md` state 2 table and install block | Confirm the tag the release build actually produces. If the deployment-target lowering in `docs/PLATFORM_MATRIX.md` lands, the tag becomes `macosx_12_0_arm64` and both the table and the "macOS floor of 26.0" paragraph change |
| Wheels are attached to a GitHub release with a `SHA256SUMS` file next to them | `INSTALLATION.md` state 2 | Confirm the artifact layout and the checksum filename. The `shasum -a 256` step assumes both |
| A release wheel bundles its MAX runtime libraries with loader-relative paths, so no toolchain is needed | `INSTALLATION.md` state 1 and 2, `python/README.md` | True of `packaging/build_wheel.sh` output today. It becomes a published promise only once a release wheel passes `packaging/matrix/validate_artifact.py` rules R5c and R5e |
| Seeing a `Library not loaded` error from an installed wheel means the wheel is broken | `INSTALLATION.md` "Missing runtime library" | Depends on the release wheel genuinely being self-contained. Confirm with `packaging/matrix/smoke/clean_install_macos.sh`, which refuses to run inside the pixi environment |
| Intel Macs get a tag mismatch and there will never be an Intel wheel | `INSTALLATION.md`, `python/README.md` | Confirm Task 02 does not attempt `universal2` |

### Depends on Task 03 (Linux x86_64 and ARM64 wheels)

| Claim | Where | Action |
|---|---|---|
| The Linux wheels are `manylinux_2_28_x86_64` and `manylinux_2_28_aarch64` | `INSTALLATION.md` state 2 table | The `manylinux_2_28` in `docs/PLATFORM_MATRIX.md` is the glibc floor pixi solved against, not a measurement of the shipped objects. Task 03 measures the real floor with `readelf`. Until then this row is a plan, and the tag in the table may move |
| Linux is a supported platform at all, second after macOS | `INSTALLATION.md` state 1 requirements table, `python/README.md` `## Platform support` | Confirm Task 03 produces a redistributable artifact and not only the source install CI already runs |
| The Linux missing-runtime error is `ImportError: libKGENCompilerRTShared.so: cannot open shared object file` | `INSTALLATION.md` "Missing runtime library" | The four Linux runtime library names in `docs/PLATFORM_MATRIX.md` are the macOS set with a different extension, and the Linux runtime layout has not been inspected. Verify the real `DT_NEEDED` names with `readelf -d` and correct the message if they differ |
| `sha256sum` is the Linux verification command | `INSTALLATION.md` state 2 | Cosmetic, but confirm the checksum file format Task 03 emits matches Task 02's |

### Depends on Task 05 (Python version support contract)

| Claim | Where | Action |
|---|---|---|
| Python 3.14 is the floor | `INSTALLATION.md` state 1 requirements table and "Unsupported Python", `python/README.md` `## Platform support` | This is the single most likely claim on the page to change. It is currently written as "the current declared value, not a settled decision" and points at `docs/PYTHON_SUPPORT.md`. When Task 05 lands a decision, the requirements table, the error section, and the `cp314` half of every wheel filename all move together |
| `cp314` appears in every wheel filename | `INSTALLATION.md` state 2 table | Same dependency. A widened floor multiplies the filename table by the number of supported interpreters |
| Free-threaded `3.14t` is unsupported and cannot load the extension | `INSTALLATION.md` state 2 table and "Wrong architecture" | Task 05 should confirm the `python-gil` dependency reasoning in `docs/PLATFORM_MATRIX.md` still holds against the current pin |
| The pip message shape `Requires-Python >=3.14` | `INSTALLATION.md` "Unsupported Python", `python/README.md` table | Update the literal string if the floor changes |
| `docs/PYTHON_SUPPORT.md` exists and is where the floor is worked out | `INSTALLATION.md` state 1 | Task 05 owns that file. It is referenced by name and did not exist when this lane wrote the reference |

`docs/PYPI_RELEASE.md` (Task 01) is deliberately **not** linked from any
user-facing file. It is an owner-only release procedure and does not belong
in an installation guide.

## Code changes this lane could not make

Recorded rather than made, per the ownership rule. Each names the file that
would change and who plausibly owns it.

1. **There is no `mojoboost.show_versions()`.** The diagnostics block in
   `docs/INSTALLATION.md` step 6 and in `examples/install_smoke.py` is seven
   lines of hand-assembled `platform`, `sys`, and `os.environ` reads, which
   is exactly the thing every library eventually ships as one function.
   `python/mojoboost/__init__.py` would gain `show_versions()` printing the
   package version, the extension path, the interpreter, the platform,
   `gpu_available()`, and the two `MOJOBOOST_*` environment variables. Both
   documents would then say "run `mojoboost.show_versions()`" and the bug
   report template would ask for its output.
2. **The extension carries no build provenance.** `bindings/_mojoboost.mojo`
   exports no version or build-info function, so an installed wheel cannot
   say which Mojo built it, which OS it was built on, or, critically,
   whether an accelerator was visible at compile time. That last one changes
   what the artifact does on the user's machine
   (`docs/PLATFORM_MATRIX.md` calls it `has_accelerator_at_build`), and it
   is currently only recoverable from a provenance sidecar next to the
   release. A `build_info()` in the bindings, surfaced by `show_versions()`,
   would make "is this wheel GPU-enabled" answerable from the installed
   package. Owner: bindings plus whoever writes the provenance sidecar.
3. **`gpu_supports()` in `src/mojoboost/device.mojo` returns
   `n_outputs >= 1`, which is always true**, so the
   `device 'gpu' does not support multiclass training` error at line 150 is
   unreachable. Three places still describe it as reachable: that module's
   own docstring (lines 14 to 17 and 8 to 10), the estimator docstring at
   `python/mojoboost/__init__.py:2656`, and
   `examples/apple_silicon/five_minute_tour.py` in `explain_device`. The
   root `README.md` states the opposite, that `fit_multiclass` routes to
   `train_multiclass_gpu` and `auto` may select it. This lane fixed only the
   copy it owns (`python/README.md`) and deliberately left the unreachable
   message out of `docs/INSTALLATION.md`. Somebody who owns `src/` should
   decide which is true and correct the other three.
4. **`python/mojoboost/device_selection.py` is not exported.**
   `explain_device_choice`, `select_device`, and `detect_capabilities` are
   implemented and documented in `docs/DEVICE_SELECTION.md`, but
   `python/mojoboost/__init__.py` does not import the module and `__all__`
   does not list it, so nothing in it is reachable through the public API a
   user is told about. `docs/INSTALLATION.md` therefore documents no
   explanation API at all, which understates what the repository can do. If
   Task 06 wires it up, the "device='auto' chose the CPU and said nothing"
   section should show `explain_device_choice(...)` instead of describing
   the silence.
5. **`examples/apple_silicon/five_minute_tour.py` says
   `explain_device_choice` is "not implemented"** in a printed placeholder.
   It is implemented, just unexported. Same fix as item 4, different file,
   and that file is not in this lane's ownership.
6. **`.github/ISSUE_TEMPLATE/bug_report.yml` asks for
   `pixi run mojo --version`** in its Environment field, which a user who
   installed a wheel cannot run and does not have. Once wheels ship, that
   field should ask for `mojoboost.show_versions()` output, or for both with
   the pixi one marked source-checkout-only.

## Exact commands a later coordinated validation pass should run

None of these has been run. Grouped by what they would establish.

Static checks on what this lane wrote, safe on any machine:

```sh
python3 -m py_compile examples/install_smoke.py
python3 -m pyflakes examples/install_smoke.py          # if pyflakes is present
git diff --check
```

The example script, source checkout, which is the only currently possible way
to run it:

```sh
pixi install
pixi run build-python
PYTHONPATH=python python examples/install_smoke.py
```

Expected: exit status 0, five step blocks printed, `device='auto'` reporting
`cpu`, and `device='gpu'` either refused or reporting `gpu` depending on the
build. A nonzero exit means the bit-exact save and load check failed, which
would be a real defect and not a documentation problem.

The example script against an installed wheel, which is what state 2 promises
and cannot be checked until Task 02 or Task 03 produces one:

```sh
python3.14 -m venv /tmp/mbst-check
. /tmp/mbst-check/bin/activate
python -m pip install ./mojoboost-0.1.0-cp314-cp314-macosx_26_0_arm64.whl
cd /tmp                                   # not a mojoboost checkout
python ~/path/to/mojoboost/examples/install_smoke.py
python -m pip uninstall -y mojoboost
```

The two failure paths the script's own error handling claims to catch, which
need deliberate setup and are worth one manual pass each:

```sh
# 1. Package present, extension missing. Expect the "build it with
#    pixi run build-python" message, not a traceback.
PYTHONPATH=python python examples/install_smoke.py     # in a checkout that
                                                        # has never been built

# 2. Extension present, runtime libraries unreachable. Expect the
#    "failed to load" message, not a traceback.
PYTHONPATH=python python examples/install_smoke.py     # outside pixi run, in a
                                                        # shell with no CONDA_PREFIX
```

Every error message quoted in `docs/INSTALLATION.md` that has **not** been
produced by running anything, and the command that would produce it:

| Message | Command that would produce it |
|---|---|
| `No matching distribution found for mojoboost` | `python -m pip install mojoboost` on any machine, today |
| `Requires-Python >=3.14` | `python3.13 -m pip install mojoboost`, once the package is on PyPI |
| `is not a supported wheel on this platform` | `pip install` a macOS arm64 wheel on Linux or on an Intel Mac |
| `dlopen ... Library not loaded: @rpath/libKGENCompilerRTShared.dylib` | Delete `python/mojoboost/.dylibs` from an installed wheel and import |
| `ModuleNotFoundError: No module named 'mojoboost._mojoboost'` | Import from a checkout with no `_mojoboost.so` |
| `device 'gpu' requested but no accelerator is available` | `MOJOBOOST_DISABLE_GPU=1 python -c "..."` with `device='gpu'` |
| `validation metrics are scored on the CPU` | Fit with `eval_set=[...]` and `device='gpu'` on a GPU build |
| `sparse input trains on the CPU` | Fit a SciPy CSR matrix with `device='gpu'` |
| `custom objectives train on the CPU` | Fit with a callable `objective` and `device='gpu'` |
| `unknown device 'metal'` | `MojoBoostRegressor(device='metal').fit(X, y)` |

The first three are shapes rather than transcripts, because pip's and dyld's
exact wording depends on their versions. The remaining seven are quoted
verbatim from the source that raises them and only need one run each to be
promoted from "read out of the code" to "observed".

Documentation link checking, once the files other lanes own exist:

```sh
# Every relative link in the three documents this lane owns.
grep -o '](\.\./[^)]*)\|](docs/[^)]*)\|](\./[^)]*)' docs/INSTALLATION.md README.md
```

`docs/INSTALLATION.md` links `docs/PYTHON_SUPPORT.md` (Task 05), which did
not exist when the link was written. `docs/PYPI_RELEASE.md` (Task 01) is
intentionally not linked.

## Deliberately not executed in this lane

- No Mojo, Pixi, Python, pytest, or pip command of any kind.
- `examples/install_smoke.py` was never run, imported, or syntax-checked by
  an interpreter. It is written against the estimator API as it reads in
  `python/mojoboost/__init__.py` and the idioms in
  `examples/apple_silicon/five_minute_tour.py`, both read statically.
- No wheel was built, downloaded, installed, inspected, or uninstalled.
- No error message was reproduced. The verbatim ones were read out of
  `src/mojoboost/device.mojo` and `python/mojoboost/__init__.py`; the pip and
  dynamic-loader ones are shapes.
- No network request, no PyPI or TestPyPI query, no check that the name
  `mojoboost` is available.
- No test file was written and no existing test was modified.
- No commit.

## One judgment call worth flagging

`docs/INSTALLATION.md` leads with `pip install mojoboost` and then says it
does not work. The alternative was to lead with the pixi source install,
which is the only thing that works, and mention pip at the end.

Leading with pip is the right call for the same reason the lane exists: the
command a Mac user will type is `pip install mojoboost`, and a page that
buries it makes them try it anyway and get a confusing pip error with no
context. Saying it plainly in the first three lines costs one paragraph and
answers the question before it is asked. The risk is that a skimmer copies
the first code block and it fails; that is mitigated by putting **It does not
work yet** in bold immediately under it, and by the three-state table being
the next thing on the page.
