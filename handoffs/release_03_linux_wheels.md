# Handoff: Linux x86_64 and ARM64 wheels (release lane 03)

**Nothing in this lane was executed.** No wheel was built, no container was
pulled, no workflow was dispatched, no script in `packaging/linux/` was run, and
no Python, Mojo, pixi, pip, docker or auditwheel command was invoked. Every
scripted step described here is a hypothesis written from a static reading of
`pixi.lock`, `packaging/build_wheel.sh`, `packaging/matrix/`, `.github/workflows/ci.yml`,
and the ELF and wheel formats. Treat the first run of anything below as an
experiment whose outcome is genuinely unknown.

## Files added, all inside this lane's ownership

| File | What it is |
| --- | --- |
| `packaging/linux/README.md` | The plan: the wheel-versus-source-install contract, the staged tag policy, build host options, what goes in the wheel, size, licensing, unsupported systems |
| `packaging/linux/build_wheel_linux.sh` | The Linux builder. Discovers the runtime closure instead of hard-coding it, rewrites RUNPATH with patchelf, tags honestly, writes a SHA-256 manifest and a provenance sidecar, refuses to leave a bad artifact behind |
| `packaging/linux/check_metadata_ready.py` | Static preflight over `python/pyproject.toml`, `python/setup.py`, `pixi.toml`, `.gitignore`. Three blockers today. Stdlib only |
| `packaging/linux/inspect_wheel.py` | Wheel and ELF inspector. Parses ELF section headers, the dynamic section and `.gnu.version_r` itself, so it runs on macOS against a Linux wheel. Stdlib only |
| `packaging/linux/inspect_elf.sh` | binutils inspection of installed objects on the target: `readelf -d`, `readelf -V`, `ldd -r`, host-path strings |
| `packaging/linux/container_elf_report.sh` | The container-side driver: install into a throwaway venv, then run `inspect_elf.sh` against the installed package |
| `packaging/linux/images.env` | Container image references. Every digest field is empty on purpose, and both the workflow and the builder refuse container mode until they are filled |
| `.github/workflows/release-linux.yml` | Manual release workflow. `plain` tags by default, native runners on both architectures, clean-install verification in a container, TestPyPI publishing disabled three ways |

Not touched, as required: Python metadata, `packaging/matrix/*`, `packaging/build_wheel.sh`,
README files, source, `pixi.toml`, `ci.yml`, `gpu-validation.yml`.

## The one distinction this lane exists to make

`ci.yml` runs the full Mojo suite, the Python API suite, and the estimator suite
on `ubuntu-latest` and `ubuntu-24.04-arm`, on every push and pull request, and
`packaging/matrix/platform_matrix.toml` records both as `status = "tested"`.
That is a supported **source install**: the code builds and passes with the
pinned pixi environment present.

A **wheel** is a promise to a machine with no pixi, no conda, no Mojo and no
MAX. CI cannot make that promise, because in CI the toolchain is on the library
path whether the wheel bundles it or not. Every green Linux CI run in this
repository is compatible with a Linux wheel that fails to import on a user's
machine.

Anyone reading the release notes will collapse these two into "Linux is
supported". The matrix rows, the workflow comments, and
`packaging/linux/README.md` all say the distinction out loud in the same words
on purpose.

## Decisions, and why

**The first Linux wheel is not tagged manylinux.** Default `tag_policy` is
`plain`: `linux_x86_64` and `linux_aarch64`. PyPI and TestPyPI reject those tags
on upload, so an unaudited Linux binary cannot reach an index by accident, and
the file is still installable from a path or a GitHub release asset by the
people being asked to test it. `manylinux_2_28` appears in the matrix rows today
because that is the `__glibc` virtual package `pixi.lock` was solved against, and
that is a property of the environment resolver rather than a measurement of any
shipped object. Promotion to a manylinux tag is a dispatch input, gated on the
four conditions in `packaging/linux/README.md`.

**The bundled library list is discovered, not copied.** `packaging/build_wheel.sh`
names four dylibs. The matrix's Linux rows carry those four names with `.so`
substituted and its own note calls that a guess. `build_wheel_linux.sh` walks
`DT_NEEDED` to a fixed point from the built extension instead, stages everything
that resolves inside `$CONDA_PREFIX`, leaves the C library and the loader to the
target, and fails loudly on anything that is neither. The real Linux set is
still unknown and this is how it gets found out. For reference, on macOS the
extension links two of the four directly (`libKGENCompilerRTShared`,
`libAsyncRTMojoBindings`, per `otool -L`) and the other two arrive transitively.

**Native runners, not emulation, and 22.04 rather than 24.04.** `ubuntu-22.04`
and `ubuntu-22.04-arm`. The build host's glibc is the ceiling on how low the
artifact's floor can be, so 2.35 beats 2.39. `ci.yml` uses `ubuntu-24.04-arm`
and should keep doing so: it is testing source, not producing an artifact.

**The `manylinux_2_28` container is the better build host in principle and is
not the default.** It has the oldest glibc the tag would permit, so the ceiling
and the claim coincide. It is not known whether the pinned Mojo toolchain runs
on AlmaLinux 8 with glibc 2.28, and this repository has never tried. That
question is unresolved and it is the single highest-value thing to find out.

**Bundled libraries are not hash-renamed.** auditwheel appends a content hash to
each vendored soname so two packages vendoring the same library cannot collide
in one process. mojoboost bundles one proprietary runtime that no other PyPI
package vendors, so the rename buys nothing and costs byte-for-byte
comparability between two builds. Revisit if that stops being true.

**auditwheel is not used at all.** Its `repair` step would vendor and rename in
one move, and it is the standard answer. It is also a black box at exactly the
step where this project needs to learn what the runtime consists of, and it
assumes a build inside a manylinux image, which is the unresolved question
above. Once the toolchain is known to run in the container, replacing steps 4
and 5 of the builder with `auditwheel repair` is a reasonable simplification
and should be reconsidered then.

**The build host must have no accelerator.** `has_accelerator()` in
`src/mojoboost/device.mojo` resolves at compile time, so a wheel built on a
machine with a working CUDA or ROCm stack is a different product under the same
filename: it reports a GPU as available and fails when the device is opened.
The builder refuses when `nvidia-smi` or `rocm-smi` is on `PATH`, and the
provenance sidecar records the answer either way. This matters far more on Linux
than on macOS, because GPU Linux machines are the ordinary case.

## Required edits outside this lane's ownership

This lane started from a tree where three things blocked a Linux wheel. **Task 01
fixed two of them while this lane was running**, in files this lane may not
touch. Both were re-read in the working tree at the time of writing and both are
correct. They are recorded here anyway, because the integration owner has to
confirm they survive the round, and because `build_wheel_linux.sh` and
`check_metadata_ready.py` were written against the broken state and will keep
detecting a regression.

### 1. `python/pyproject.toml`, package data (Task 01). **Fixed concurrently, verify it holds.**

Was `mojoboost = ["*.so", ".dylibs/*.dylib"]`, which matches nothing this lane
stages. Now:

```toml
[tool.setuptools.package-data]
mojoboost = ["*.so", ".dylibs/*.dylib", ".libs/*.so", ".libs/*.so.*"]
```

Both `.libs` patterns are required and neither is redundant, because sonames
carry version suffixes and `*.so` does not match `libfoo.so.1`. Without them the
bundled runtime is dropped from the wheel silently and the failure appears as an
`ImportError` on a user's machine rather than at build time. Check `C1` in
`check_metadata_ready.py` enforces it against the exact names the builder stages.

### 2. `python/setup.py`, hard-coded macOS platform tag (Task 01). **Fixed concurrently, verify it holds.**

Was an unconditional `plat_name = "macosx_26_0_" + platform.machine().lower()`,
which on Linux produces `mojoboost-0.1.0-cp314-cp314-macosx_26_0_x86_64.whl`: a
macOS tag on an ELF binary, which pip on a Mac accepts and then fails to import.
It is now guarded by `if sys.platform == "darwin"`, so on Linux the tag comes
from `--plat-name` or the default.

`build_wheel_linux.sh` still passes `--plat-name` explicitly and still verifies
the resulting filename before accepting the artifact. That belt-and-braces is
deliberate: the check costs nothing and it is the difference between a caught
error and a mislabeled file. Check `C2` fails if the unconditional form comes
back.

### 3. `pixi.toml`, patchelf (integration owner, no lane owns this file). **Still a blocker.**

```toml
[feature.pkg.target.linux-64.dependencies]
patchelf = "*"

[feature.pkg.target.linux-aarch64.dependencies]
patchelf = "*"
```

Target-scoped so macOS builds do not acquire a dependency they cannot use.
patchelf is to a Linux wheel what `install_name_tool` is to a macOS one: without
it the extension keeps an absolute RPATH into the build machine's pixi
environment. **This changes `pixi.lock`,** which is why it is not a casual
mid-round edit.

### 4. `pixi.toml`, tasks (integration owner). Optional, and deliberately not proposed as default.

If the integration owner wants task entries:

```toml
[feature.pkg.tasks]
build-wheel-linux = "packaging/linux/build_wheel_linux.sh"
inspect-wheel = "python3 packaging/linux/inspect_wheel.py"
```

No `test-wheel-linux` counterpart is proposed. `packaging/test_wheel.sh` installs
into a venv on the build machine, which on Linux is the machine with the
toolchain on it, and a Linux wheel that passes there has demonstrated nothing
about the property it needs to have. The Linux equivalent is
`packaging/matrix/smoke/clean_install_linux.sh` in a container, and it should
stay a deliberate act rather than a pixi task.

### 5. `packaging/matrix/platform_matrix.toml` (Task 18's file, matrix owner).

The two Linux target rows say `builder = "does not exist yet"`. When this lane
lands, that becomes `builder = "packaging/linux/build_wheel_linux.sh"`. Both rows
stay `status = "designed"` with `evidence = ""`, because nothing has been built
or run. Also worth splitting each row in two, since `wheel_tag` is a single
string and the staged policy produces two different filenames per architecture:

| Row | wheel_tag |
| --- | --- |
| `linux-x86_64-cp314-plain` | `cp314-cp314-linux_x86_64` |
| `linux-x86_64-cp314-manylinux` | `cp314-cp314-manylinux_2_28_x86_64` |
| `linux-aarch64-cp314-plain` | `cp314-cp314-linux_aarch64` |
| `linux-aarch64-cp314-manylinux` | `cp314-cp314-manylinux_2_28_aarch64` |

with the plain rows reachable first and the manylinux rows blocked on a measured
floor. The `bundled_dylibs` lists in the Linux rows should be emptied rather
than left as the macOS names, since they are a guess and the builder is what
will answer the question.

### 6. `packaging/matrix/validate_artifact.py` (matrix owner).

Its `_check_elf` is explicitly a byte scan and its own comment defers to a real
parse. `packaging/linux/inspect_wheel.py` is that parse. Either call it from
there, or leave both and run both; do not delete the byte scan, which catches
the build-host strings in members that are not ELF objects.

Its `PROVENANCE_KEYS` require `metal_toolchain`, which a Linux build cannot have.
`build_wheel_linux.sh` writes `"n/a (linux)"`, which is non-empty and therefore
passes rule R7. If R7 ever gets stricter, it needs a per-OS key set.

### 7. `.github/workflows/release-linux.yml` action pinning (Task 10).

Every `uses:` in the new workflow carries a `# TODO(pin)` comment and a version
tag rather than a commit SHA. **No SHA was invented**, because a wrong pin is
worse than an unpinned tag: it fails opaquely or, if it happens to resolve,
pins something nobody chose. Task 10 owns the pinning policy and should
substitute real SHAs for `actions/checkout`, `actions/upload-artifact`,
`actions/download-artifact`, `prefix-dev/setup-pixi`, and
`pypa/gh-action-pypi-publish` before the workflow is enabled.

### 8. GitHub settings that no file can express (Task 10 / organization owner).

- Repository variable `MOJOBOOST_TESTPYPI_ENABLED`, unset today. The publish job
  is skipped while it is anything other than `true`.
- Environment `testpypi` with required reviewers.
- TestPyPI trusted publisher for the project, bound to this repository and this
  workflow filename. Not configured; the publish job cannot work until it is.
- A PyPI project does not exist and this lane did not reserve one.

### 9. README claims (Task 04, which owns README.md).

The README's "Tests and wheels" section says Linux needs a manylinux build.
After this lane that sentence is half wrong: the machinery exists, the artifact
does not. Suggested replacement, for Task 04 to place:

> Wheels are built by `packaging/linux/build_wheel_linux.sh` and
> `.github/workflows/release-linux.yml` on Linux, and by `packaging/build_wheel.sh`
> on macOS. **No Linux wheel has been built or published.** The workflow is
> manual, produces `linux_x86_64` and `linux_aarch64` artifacts that no package
> index accepts by design, and a `manylinux` tag is only claimed once the glibc
> floor has been measured on a real artifact. Installing from source with pixi
> is the supported path on Linux today.

## Exact commands for the later validation pass

None of these has been run. They are listed in the order they must happen.

### Step 0. Preflight, anywhere, no toolchain needed

```bash
python3 packaging/linux/check_metadata_ready.py
```

Expected at the time of writing: C1 and C2 pass, because Task 01 fixed them
during this round; C3 fails, because no pixi environment provides patchelf. Exit
1. Nothing else in this list can run until it exits 0, and if C1 or C2 comes
back the fix landed and was then lost in the merge.

### Step 1. Build, on a Linux host with no accelerator

```bash
# x86_64 host
pixi run -e pkg packaging/linux/build_wheel_linux.sh

# ARM64 host, same command; the tag follows uname -m
pixi run -e pkg packaging/linux/build_wheel_linux.sh
```

Produces, in `python/dist/`:

```
mojoboost-0.1.0-cp314-cp314-linux_x86_64.whl
mojoboost-0.1.0-cp314-cp314-linux_x86_64.whl.sha256
mojoboost-0.1.0-cp314-cp314-linux_x86_64.whl.provenance.json
```

Read the closure the builder prints. It is the first time anyone will have seen
which libraries the Linux MAX runtime actually consists of, and it should be
copied into the matrix rows and into this handoff's successor.

### Step 2. Inspect the artifact, from anywhere including macOS

```bash
python3 packaging/linux/inspect_wheel.py python/dist/mojoboost-0.1.0-cp314-cp314-linux_x86_64.whl
python3 packaging/matrix/validate_artifact.py python/dist/mojoboost-0.1.0-cp314-cp314-linux_x86_64.whl
```

Rule L9b prints the artifact's real glibc floor. That number, not `pixi.lock`,
is what any future manylinux tag has to be at or above.

### Step 3. Clean container install, x86_64

```bash
docker run --rm -v "$PWD:/io" -w /io \
    -e PYTHON=/opt/python/cp314-cp314/bin/python3.14 \
    quay.io/pypa/manylinux_2_28_x86_64@sha256:<fill from images.env> \
    packaging/matrix/smoke/clean_install_linux.sh \
        python/dist/mojoboost-0.1.0-cp314-cp314-linux_x86_64.whl \
        records/linux-x86_64-clean-install.txt
```

### Step 4. Clean container install, ARM64

Run on an ARM64 host or an ARM64 runner. Do **not** substitute
`--platform linux/arm64` emulation on an x86 machine: qemu changes what the
loader and the CPU feature detection see, and a pass under emulation is not
evidence about an ARM64 machine.

```bash
docker run --rm -v "$PWD:/io" -w /io \
    -e PYTHON=/opt/python/cp314-cp314/bin/python3.14 \
    quay.io/pypa/manylinux_2_28_aarch64@sha256:<fill from images.env> \
    packaging/matrix/smoke/clean_install_linux.sh \
        python/dist/mojoboost-0.1.0-cp314-cp314-linux_aarch64.whl \
        records/linux-aarch64-clean-install.txt
```

### Step 5. ELF report on the target, both architectures

```bash
docker run --rm -v "$PWD:/io" -w /io \
    -e PYTHON=/opt/python/cp314-cp314/bin/python3.14 \
    quay.io/pypa/manylinux_2_28_x86_64@sha256:<digest> \
    packaging/linux/container_elf_report.sh \
        python/dist/mojoboost-0.1.0-cp314-cp314-linux_x86_64.whl \
        /io/records/linux-x86_64-elf.txt
```

### Step 6. A second, modern container

The old image tests the floor. An ordinary current distribution tests what users
have. A wheel has to pass both before a manylinux tag is honest, and Ubuntu
24.04 needs a CPython 3.14 installed first, which the manylinux image already
carries:

```bash
docker run --rm -v "$PWD:/io" -w /io docker.io/library/ubuntu:24.04@sha256:<digest> \
    bash -c 'apt-get update && apt-get install -y python3.14-venv binutils file \
        && PYTHON=python3.14 packaging/matrix/smoke/clean_install_linux.sh \
             /io/python/dist/mojoboost-0.1.0-cp314-cp314-linux_x86_64.whl \
             /io/records/linux-x86_64-ubuntu2404.txt'
```

Whether `python3.14-venv` exists in Ubuntu 24.04's archive has not been checked.
A deadsnakes PPA or a python-build-standalone tarball is the fallback.

### Step 7. Install, import, fit, predict, serialize, by hand

The fixture in step 3 covers install, import provenance, a fit and predict on
plain Python lists, pickling, and device behavior. It does **not** cover the
versioned model format. Run this inside the same container, in a directory that
is not the source tree, after the fixture's venv or a fresh one:

```bash
cd /tmp && /tmp/venv/bin/python - <<'PY'
import mojoboost
from mojoboost import MojoBoostRegressor, MojoBoostClassifier

print("version:", mojoboost.__version__)
print("install:", mojoboost.diagnostics.describe_install())
print("environment:", mojoboost.diagnostics.environment_snapshot())

X = [[i / 40.0, (i % 5) / 5.0] for i in range(40)]
y = [3.0 * r[0] + r[1] for r in X]

reg = MojoBoostRegressor(n_estimators=20, min_data_in_leaf=2).fit(X, y)
before = list(reg.predict(X))
print("n_features_in_:", reg.n_features_in_, "best_iteration_:", reg.best_iteration_)
print("device_:", reg.device_)
print("r2:", reg.score(X, y))

reg.save("/tmp/model.mbst")
again = MojoBoostRegressor.load("/tmp/model.mbst")
assert list(again.predict(X)) == before, "save/load changed the predictions"
print("save/load: identical predictions")

labels = ["lo" if r[0] < 0.5 else "hi" for r in X]
clf = MojoBoostClassifier(n_estimators=20, min_data_in_leaf=2).fit(X, labels)
proba = clf.predict_proba(X)
assert all(abs(sum(row) - 1.0) < 1e-9 for row in proba)
print("classifier classes_:", list(clf.classes_))
PY
```

Then, since a wheel that cannot be removed cleanly is its own problem:

```bash
/tmp/venv/bin/pip uninstall -y mojoboost
/tmp/venv/bin/python -c "import mojoboost" 2>&1 | tail -1   # expect ModuleNotFoundError
```

### Step 8. Only then, the workflow

```
Actions -> Release wheels (Linux) -> Run workflow
  ref: <the release tag>
  arch: both
  tag_policy: plain
  publish_testpypi: false
```

`plain` first, always. A `manylinux` dispatch is a claim, and steps 2 through 6
are what earn it.

## Expected failure behavior on unsupported systems

None of this has been observed. It is what the tags and the metadata should
produce.

| Situation | Expected behavior |
| --- | --- |
| `pip install mojoboost` on Alpine or any musl distribution | `ERROR: Could not find a version that satisfies the requirement mojoboost`, then `No matching distribution found`. No musl build exists and none can be made without a musl toolchain |
| Direct install of a `linux_x86_64` wheel on musl | pip accepts it by tag (musl systems match the plain tag) and the import fails at runtime, because the objects need glibc. **This is the plain tag's one sharp edge**, and it is a reason to distribute stage 0 wheels only to named testers rather than linking them publicly |
| glibc older than a `manylinux_2_XX` tag | `ERROR: ... is not a supported wheel on this platform` for a direct file install; no candidate at all from an index. Correct and automatic, and the entire reason the tag must be measured |
| 32 bit, ppc64le, s390x, riscv64 | No matching distribution. `pixi.toml` declares no such platform and the channel ships no toolchain |
| CPython 3.13 or earlier | No matching distribution, and `requires-python >= 3.14` gives pip a clearer message |
| CPython 3.14 free-threaded (`cp314t`) | Tag mismatch, no install. `max 26.5.0` depends on `python-gil`, so no extension can be built for it |
| Wrong architecture wheel installed by hand | pip refuses on the tag. If forced, the import fails with an ELF class or machine error from the loader |
| Correct wheel, missing bundled runtime (the package-data blocker above) | `ImportError` on `import mojoboost`, naming the missing `.so`. Caught by rule L7b of `inspect_wheel.py` long before a user sees it |
| `device="gpu"` on a Linux host with no accelerator | Raises. Which exception, and whether it raises at fit time or at device open, depends on whether an accelerator was visible when the wheel was compiled. The fixture records which of the two happened and the provenance sidecar records why |
| `device="gpu"` on a Linux host with an NVIDIA or AMD GPU | **Unknown.** No NVIDIA or AMD device has ever executed a kernel from this project (`packaging/matrix/accelerators/index.toml`). A Linux wheel does not change that, and nothing in this lane should be read as GPU support on Linux |

## Unresolved licensing questions

These block publication to any index. They do not block building an artifact and
inspecting it.

1. **The MAX runtime is proprietary.** Every `max`, `max-core`, `mojo`, and
   `mojo-compiler` record in `pixi.lock` carries
   `license: LicenseRef-Modular-Proprietary`. A wheel that bundles their shared
   objects redistributes proprietary binaries inside an Apache-2.0 package.
   Nobody on this project has read Modular's license terms and confirmed that
   redistribution is permitted. **This is the highest-priority open question in
   this lane and it applies to the macOS wheel exactly as much as to the Linux
   one.** If the answer is no, the shape of the product changes: the fallback is
   a wheel that requires a separate MAX installation, which is a different thing
   to ship and a different install story.
2. **Attribution inside the wheel.** Even where redistribution is permitted,
   the bundled components' license texts have to travel in the wheel's
   `dist-info`. Today the wheel carries only the project's Apache-2.0 `LICENSE`.
   Rule L5 of `inspect_wheel.py` reports what is there; it cannot know what
   should be.
3. **`libstdc++` and `libgcc_s`.** If the closure includes conda-forge's copies,
   those are GPL-3 with the GCC Runtime Library Exception. The exception is
   written for this case and is very likely fine; their license texts still have
   to ship. The builder logs loudly when it bundles either. Whether they are in
   the closure at all is unknown until step 1 runs.
4. **`LicenseRef-Modular-Proprietary` is not an SPDX identifier**, so the
   wheel's `License-Expression: Apache-2.0` describes the project's own code and
   says nothing about what is bundled beside it. Whatever the answer to (1) is,
   the metadata should end up saying it. That is Task 01's file.

## Every check that has not been run

The scripts, none of which has ever executed:

- `packaging/linux/build_wheel_linux.sh`. Never run. Specific parts most likely
  to be wrong on first contact: the `readelf -d` `sed` pattern for `DT_NEEDED`
  (format varies between binutils versions), the `--config-setting=--build-option`
  spelling for overriding `plat_name`, whether `patchelf --set-rpath` produces
  `DT_RUNPATH` rather than `DT_RPATH` with the version that ends up installed,
  and the `max_version` extraction from `pixi.lock`.
- `packaging/linux/check_metadata_ready.py`. Never run. Its `_fnmatch_one`
  approximates setuptools package-data globbing and has not been checked against
  setuptools' real behavior.
- `packaging/linux/inspect_wheel.py`. Never run, on any wheel, on any platform.
  It parses ELF section headers, the dynamic section, and `.gnu.version_r` from
  format documentation rather than against a file. The `.gnu.version_r` parse is
  the riskiest part and falls back to a dynamic-string-table scan when it throws,
  which means a silent downgrade to a weaker answer is possible. **Run it against
  any known-good Linux wheel first**, for example a numpy manylinux wheel, and
  confirm the output matches `readelf`, before trusting a word it says about a
  mojoboost artifact.
- `packaging/linux/inspect_elf.sh` and `container_elf_report.sh`. Never run.
- `.github/workflows/release-linux.yml`. Never dispatched, never linted, never
  parsed by anything but a human. Unverified assumptions inside it: that
  `ubuntu-22.04-arm` is available to this repository, that Docker is present on
  ARM64 GitHub-hosted runners, that `setup-pixi` works on `ubuntu-22.04-arm`
  (`ci.yml` only proves it on `ubuntu-24.04-arm`), and that the guard job's
  matrix JSON round-trips through `fromJSON` as written.

The facts, none of which has been measured:

- Whether the Mojo toolchain runs at all inside `manylinux_2_28`.
- Which shared objects the Linux MAX runtime consists of.
- The real glibc floor of a built Linux extension, on either architecture.
- Whether `GLIBCXX_` or `CXXABI_` versions appear at all, and from where.
- The size of the resulting wheel, and which side of PyPI's 100 MB per-file
  limit it lands on.
- Whether cp314 is present in the current `manylinux_2_28` images at the path
  `images.env` assumes.
- Whether a Linux wheel imports, fits, predicts, or serializes anywhere.
- Whether `device="gpu"` behaves as described on any Linux machine, with or
  without a device.

Nothing in `packaging/matrix/platform_matrix.toml` may be promoted out of
`designed` on the strength of this handoff. A status change needs an evidence
file naming a machine that ran something.
