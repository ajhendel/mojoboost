# Handoff: platform, wheel, and accelerator release design

Lane 18 of a parallel round. This document is the part of the work that touches
files this lane was not allowed to edit: the pixi tasks, the CI workflows, the
build scripts, and `README.md`. Every edit below is written out exactly, ready
to apply, and **none of it has been applied**.

## What this lane added

| Path | What it is |
|---|---|
| `packaging/matrix/platform_matrix.toml` | Declarative matrix: targets, expected wheel tags, interpreters, toolchain floor derived from `pixi.lock` |
| `packaging/matrix/accelerators/index.toml` | One row per device, 25 of them, 24 at `not-run` |
| `packaging/matrix/accelerators/TEMPLATE_{apple,nvidia,amd}.md` | What a device record must contain, per vendor |
| `packaging/matrix/validate_matrix.py` | Checks the metadata against itself, against `pixi.lock`, and against the doc |
| `packaging/matrix/validate_artifact.py` | Checks a built wheel against the matrix. Parses Mach-O load commands with the standard library |
| `packaging/matrix/smoke/clean_install_{macos,linux}.sh` | Clean-install fixtures that refuse to run inside pixi |
| `packaging/matrix/smoke/probe_platform.py` | Prints the platform facts a record has to contain |
| `packaging/matrix/README.md` | What runs, what does not, and why the metadata is separate from the prose |
| `docs/PLATFORM_MATRIX.md` | The prose half, including the macOS install story and the failure table |

## What was executed

One command, once:

```
$ python3 packaging/matrix/validate_matrix.py
release matrix ok: 6 targets, 3 source installs, 25 devices, 1 with any recorded evidence
```

Nothing else. No wheel was built, no artifact validated, no platform tested, no
device exercised, and no benchmark run. `validate_artifact.py` and both smoke
fixtures have never executed, because each needs something that does not exist
yet (a wheel, and a machine with no toolchain on it).

## Facts this design rests on, and where they came from

Recorded here because the design is only as good as these, and each one is
cheap to re-check when the toolchain moves.

| Fact | Source |
|---|---|
| `max 26.5.0` is a 3.14 build, depends on `python 3.14.*` and `python-gil` | `pixi.lock` |
| `mojo 1.0.0`, platforms `osx-arm64`, `linux-64`, `linux-aarch64` | `pixi.toml`, `pixi.lock` |
| The lock solves against `__glibc=2.28` and `__osx=13.0` | `pixi.lock` header |
| `python/mojoboost/_mojoboost.so` is Mach-O arm64, `minos 26.0`, `sdk 26.5` | `otool -l` on the file in the working copy |
| The four bundled MAX dylibs are `minos 11.0` | `otool -l` on the files in `.pixi/envs/*/lib` |
| A `...-cp314-cp314-macosx_26_0_arm64.whl` exists in `python/dist` | `ls`, from an earlier local build |
| `capi/libmojoboost.dylib` has `minos 26.0` and one absolute rpath into the build machine's pixi environment | `otool -l` and `otool -L` on the file in the working copy |
| `has_accelerator()` is resolved at compile time | `src/mojoboost/device.mojo` docstring |
| GitHub-hosted Apple silicon runners have no Metal toolchain | comment at the top of `.github/workflows/ci.yml` |

The last two together are why a published macOS wheel is a product decision and
not just a build step. See "Metal toolchain requirements" in
`docs/PLATFORM_MATRIX.md`.

## Edit 1: pixi tasks

`pixi.toml`. Add one task to the default `[tasks]` table, next to
`check-parity`, which it deliberately resembles: standard library only, builds
nothing, runs in under a second on a bare checkout.

```toml
# Checks the release matrix in packaging/matrix against the repository and
# against pixi.lock. Standard library only, and it builds nothing, so it is
# cheap enough to run on every change and it fails when a platform status
# outruns its evidence.
check-matrix = "python3 packaging/matrix/validate_matrix.py"
```

And two to `[feature.pkg.tasks]`, which is where the wheel already lives:

```toml
# Checks a built wheel's tag, bundled libraries, rpaths, deployment target,
# code signatures, and provenance sidecar against packaging/matrix. Runs after
# build-wheel; does not install anything.
validate-wheel = { cmd = "python3 packaging/matrix/validate_artifact.py python/dist/*.whl", depends-on = [
    "build-wheel",
] }
```

Do **not** add a pixi task for the clean-install fixtures. They refuse to run
with `CONDA_PREFIX` set, which is exactly what a `pixi run` gives them, and
that refusal is the point of the fixture. They are run by hand, or by a release
workflow step outside the pixi environment.

## Edit 2: CI

`.github/workflows/ci.yml`. Add one job, modeled on the existing `parity` job,
which already runs a standard-library checker on a bare runner:

```yaml
  # The release matrix contract. Standard library only and no build, so it runs
  # on a bare runner in seconds and fails when a platform status, a wheel tag,
  # or a cited file stops agreeing with the repository.
  matrix:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Release matrix contract
        run: python3 packaging/matrix/validate_matrix.py
```

Nothing else in `ci.yml` changes. In particular, do not add a macOS runner to
build a wheel there: a GitHub-hosted Apple silicon runner has no Metal
toolchain, so a wheel built on it would have a different answer to
`has_accelerator()` than the wheel built locally, under the same filename. Two
artifacts with the same name and different behavior is the worst outcome
available here.

## Edit 3: the deployment target, which gates the whole macOS story

This is the one experiment that decides whether `pip install mojoboost` works
for Mac users generally or only for those on the newest macOS. Everything else
in this handoff is bookkeeping by comparison.

**Run this first, before writing any of the code below:**

```sh
MACOSX_DEPLOYMENT_TARGET=12.0 bindings/build.sh
otool -l python/mojoboost/_mojoboost.so | grep -A 4 LC_BUILD_VERSION
```

Three possible outcomes and what each means:

- `minos 12.0`: the Mojo compiler honors the variable. Apply edits 3a and 3b,
  and the `macos-arm64-cp314-lowered` row becomes reachable.
- `minos 26.0` unchanged: the variable is ignored. Look for a Mojo build flag
  next, and if none exists, this is a Modular feature request rather than a
  mojoboost change. Record the finding in `docs/PLATFORM_MATRIX.md` under the
  easy macOS path so the next person does not repeat the experiment.
- Build failure: record the error verbatim, same place.

Do not relabel the wheel on the strength of the variable being set. The tag has
to follow the binary, which is what edit 3b enforces.

### 3a. `bindings/build.sh`

Pass a deployment target through, opt-in, defaulting to today's behavior so
this cannot silently change what a dev build produces:

```sh
#!/bin/sh
# Build the CPython extension module into python/mojoboost/_mojoboost.so.
# Run from anywhere; requires pixi.
#
# MOJOBOOST_MACOS_TARGET sets the macOS deployment target of the built
# extension, which becomes the wheel's platform tag and therefore the oldest
# macOS pip will install it on. Unset means whatever the local SDK defaults to,
# which on a current Xcode is macOS 26 and is far newer than the bundled MAX
# runtime requires. Verify the result rather than trusting the variable:
#   otool -l python/mojoboost/_mojoboost.so | grep -A 4 LC_BUILD_VERSION
set -e
cd "$(dirname "$0")/.."
if [ -n "${MOJOBOOST_MACOS_TARGET:-}" ]; then
    MACOSX_DEPLOYMENT_TARGET="$MOJOBOOST_MACOS_TARGET"
    export MACOSX_DEPLOYMENT_TARGET
fi
pixi run mojo build --emit shared-lib -I src \
    bindings/_mojoboost.mojo -o python/mojoboost/_mojoboost.so
echo "built python/mojoboost/_mojoboost.so"
```

### 3b. `python/setup.py`

Stop hardcoding the platform tag. Read `minos` out of the binary that was
actually built, so the tag cannot disagree with the object even in principle,
and fail the build rather than guess.

```python
"""Wheel configuration for the prebuilt Mojo extension.

setuptools does not compile anything here; the extension is built by
`bindings/build.sh` (Mojo) before packaging. has_ext_modules is overridden so
the wheel is tagged for the interpreter and platform instead of py3-none-any.

The platform tag is read out of the built extension's LC_BUILD_VERSION rather
than hardcoded, because that number is what actually constrains where the
binary runs, and a tag that disagrees with it produces the worst failure
available: pip installs the wheel and the import fails.
"""

import struct
import sys
from pathlib import Path

from setuptools import setup
from setuptools.dist import Distribution

SO = Path(__file__).parent / "mojoboost" / "_mojoboost.so"
MH_MAGIC_64 = 0xFEEDFACF
LC_BUILD_VERSION = 0x32
LC_VERSION_MIN_MACOSX = 0x24


def deployment_target(path):
    """(major, minor) from the object's load commands, or None."""
    blob = path.read_bytes()
    if struct.unpack_from("<I", blob, 0)[0] != MH_MAGIC_64:
        return None
    ncmds = struct.unpack_from("<I", blob, 16)[0]
    off = 32
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", blob, off)
        if cmd in (LC_BUILD_VERSION, LC_VERSION_MIN_MACOSX):
            packed = struct.unpack_from(
                "<I", blob, off + (12 if cmd == LC_BUILD_VERSION else 8)
            )[0]
            return (packed >> 16, (packed >> 8) & 0xFF)
        off += cmdsize
    return None


def plat_name():
    if sys.platform != "darwin":
        raise SystemExit(
            "no wheel build for this platform yet; see docs/PLATFORM_MATRIX.md"
        )
    if not SO.exists():
        raise SystemExit(f"{SO} is missing; run bindings/build.sh first")
    target = deployment_target(SO)
    if target is None:
        raise SystemExit(f"cannot read a deployment target from {SO}")
    import platform

    return f"macosx_{target[0]}_{target[1]}_{platform.machine().lower()}"


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    distclass=BinaryDistribution,
    options={"bdist_wheel": {"plat_name": plat_name()}},
)
```

Watch the offsets. `LC_BUILD_VERSION` is `cmd, cmdsize, platform, minos, sdk,
ntools`, so `minos` sits at `off + 12`, while `LC_VERSION_MIN_MACOSX` is
`cmd, cmdsize, version, sdk` and puts its version at `off + 8`.
`packaging/matrix/validate_artifact.py` parses the same two commands and is the
cross-check on this arithmetic.

## Edit 4: provenance and validation in the wheel build

`packaging/build_wheel.sh`. Append after the existing `ls -l python/dist/`:

```sh
# Provenance. None of this is recoverable from the wheel afterwards, and
# has_accelerator_at_build changes what the artifact does on the user's
# machine (src/mojoboost/device.mojo resolves it at compile time), so two
# wheels with the same filename and different answers are different products.
WHEEL=$(ls python/dist/mojoboost-*.whl)
cat >"$WHEEL.provenance.json" <<EOF
{
  "mojo_version": "$(pixi run mojo --version 2>/dev/null | head -1)",
  "max_version": "$(pixi list --environment default 2>/dev/null | awk '$1=="max"{print $2; exit}')",
  "pixi_lock_sha256": "$(shasum -a 256 pixi.lock | cut -d' ' -f1)",
  "git_commit": "$(git rev-parse HEAD)",
  "git_dirty": $(test -n "$(git status --porcelain)" && echo true || echo false),
  "build_host_os": "$(sw_vers -productVersion)",
  "build_host_arch": "$(uname -m)",
  "xcode": "$(xcodebuild -version 2>/dev/null | tr '\n' ' ')",
  "metal_toolchain": "$(xcrun --find metal 2>/dev/null || echo absent)",
  "has_accelerator_at_build": "$(pixi run mojo run -I src tools/report_accelerator.mojo 2>/dev/null || echo unknown)"
}
EOF

python3 packaging/matrix/validate_artifact.py "$WHEEL"
```

Two things that need deciding before this is applied:

- **`has_accelerator_at_build` has no reporter yet.** `tools/report_accelerator.mojo`
  does not exist. It is a four-line program that prints
  `has_accelerator()` and exits, and it is worth writing, because this is the
  single field that changes the artifact's runtime behavior. Until it exists,
  either write the field by hand or leave it `unknown`, and note that
  `validate_artifact.py` rule R7 treats an empty value as a failure while
  accepting `unknown` as a recorded answer.
- **`python/dist/` is in `.gitignore`**, so the sidecar is ignored along with
  the wheel. That is correct for the repository and wrong for a release: both
  files have to be uploaded together, or the provenance is lost exactly when it
  starts mattering.

## Edit 5: a release workflow

New file, `.github/workflows/release.yml`. It is `workflow_dispatch` only and
targets a self-hosted Apple silicon runner, because the wheel must be built on
a machine with a Metal toolchain and GitHub-hosted macOS runners do not have
one. No such runner is registered today, so this job will not run until one is,
which is the same situation `gpu-validation.yml` is in and for a related
reason.

```yaml
# macOS wheel release. Manual, and self-hosted, because a GitHub-hosted Apple
# silicon runner reports an accelerator at compile time and has no Metal
# toolchain: a wheel built there would carry a different GPU story than the
# same filename built locally.
#
# Runner label expected:
#   [self-hosted, macos, arm64, metal]   an Apple silicon Mac with Xcode and
#                                        the Metal toolchain installed
name: Release wheel

on:
  workflow_dispatch:
    inputs:
      macos_target:
        description: macOS deployment target for the extension (empty = SDK default)
        required: false
        default: ""

jobs:
  macos-arm64:
    runs-on: [self-hosted, macos, arm64, metal]
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.10.1
        with:
          cache: true
          environments: default pkg

      # A wheel from a machine without the Metal toolchain is a different
      # product. Fail here rather than publish it.
      - name: Metal toolchain
        run: |
          xcodebuild -version
          xcrun --find metal
          xcrun metal --version

      - name: Matrix contract
        run: python3 packaging/matrix/validate_matrix.py

      - name: Build and test the wheel
        env:
          MOJOBOOST_MACOS_TARGET: ${{ inputs.macos_target }}
        run: pixi run -e pkg test-wheel

      - name: Deployment target of the built extension
        run: otool -l python/mojoboost/_mojoboost.so | grep -A 4 LC_BUILD_VERSION

      - name: Validate the artifact
        run: pixi run -e pkg validate-wheel

      # Outside pixi on purpose: the fixture refuses to run with CONDA_PREFIX
      # set, because a wheel tested next to the toolchain it must not need has
      # not been shown to be self-contained.
      - name: Clean install smoke test
        run: |
          env -u CONDA_PREFIX bash packaging/matrix/smoke/clean_install_macos.sh \
            "$(ls python/dist/mojoboost-*.whl)" clean-install.txt

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: wheel-macos-arm64
          path: |
            python/dist/*.whl
            python/dist/*.provenance.json
            clean-install.txt
          if-no-files-found: warn
```

The `env -u CONDA_PREFIX` there is necessary but not sufficient: the runner's
`PATH` must also not have `mojo` on it, or the fixture refuses. Getting that
right on a self-hosted Mac that is also a development machine is the fiddly
part of this job, and the fixture failing loudly is preferable to it passing
meaninglessly.

## Edit 6: README

`README.md` is being edited by other lanes in this round, so this is written as
a replacement for a specific paragraph rather than as a diff. In the
"Tests and wheels" section, the sentence

> Wheels currently target macOS on Apple silicon; Linux wheels need a manylinux
> build.

becomes

> Wheels currently target macOS on Apple silicon; Linux wheels need a manylinux
> build. The full release matrix, the expected wheel tags, the artifact
> validation rules, and what has and has not been validated on which hardware
> are in [docs/PLATFORM_MATRIX.md](docs/PLATFORM_MATRIX.md). No wheel has been
> published, and no platform in that matrix is marked validated.

And roadmap item 4,

> Publish the Python API to PyPI (macOS arm64 wheels build and validate today;
> Linux needs a manylinux build)

becomes

> Publish the Python API to PyPI. The macOS wheel builds and installs from a
> clean venv today, but its platform tag is `macosx_26_0_arm64`, so pip refuses
> it on any older macOS; lowering the extension's deployment target is the
> first thing to try and it is one experiment
> ([docs/PLATFORM_MATRIX.md](docs/PLATFORM_MATRIX.md)). Linux needs a manylinux
> build, which is a new builder rather than a flag, because `install_name_tool`
> and `codesign` have no Linux counterpart

`python/pyproject.toml` also needs a look before the first publish. Its
classifiers say `Operating System :: MacOS` and no Python version, which is
right today and wrong the moment a Linux wheel exists. Left alone deliberately:
changing it now would make the metadata claim a target that does not exist.

## Sequenced plan

1. Apply edits 1 and 2. Cheap, no risk, and they make every later status change
   checkable.
2. Run the deployment-target experiment at the top of edit 3. One command, and
   it decides the shape of the Apple story.
3. Depending on the result, apply 3a and 3b, rebuild, and check the tag with
   `validate_artifact.py`.
4. Apply edit 4, and write `tools/report_accelerator.mojo` while doing it.
5. Run `packaging/matrix/smoke/clean_install_macos.sh` by hand, from a plain
   shell, on a Mac that is not the build machine if one is available. Paste the
   output into a record and only then move `macos-arm64-cp314` off `designed`.
6. Apply edit 5 when a self-hosted Mac runner exists. Not before; an
   unrunnable workflow that looks like a release pipeline is worse than no
   workflow.
7. Before the R bindings work starts, decide the bundling story for
   `capi/libmojoboost.dylib`. It is built by `capi/build.sh` with no
   delocate-style step, so today it carries an absolute rpath into the build
   machine's pixi environment and loads nowhere else. The wheel's four-line fix
   (copy the runtime dylibs, rewrite the rpath to `@loader_path`, re-sign)
   ports directly, and `validate_artifact.py` rules R5b, R5c, R5d, and R6 apply
   to it unchanged. An R package that ships a library with somebody else's home
   directory baked into it is not distributable, so this is a prerequisite of
   roadmap item 6 rather than a detail of it.
8. Linux is a separate piece of work and does not block any of the above. It
   needs a new builder (`patchelf` plus `auditwheel repair`, or an explicit
   `$ORIGIN` RUNPATH), a measured glibc floor, and an inspection of what the
   Linux MAX runtime actually consists of, since the bundled library list in
   the Linux rows is currently the macOS list with a different extension.

## Cross-lane reconciliation

Another lane in this round added `docs/COMPATIBILITY_POLICY.md`, which contains
a support tier table (section 10.2) and a distribution paragraph (10.3) that
overlap this work. Read at the time of writing, the two agree on every fact:
the `cp314-cp314` tag, no `abi3`, the `macosx_26_0_arm64` platform tag, the
four bundled dylibs, the extension linking no `libpython`, no manylinux build,
the M4 being the only device with evidence, and NVIDIA and AMD never having
executed anything. Two documents derived independently and landing on the same
numbers is a good sign for both.

Two things for whoever integrates the round:

- **A vocabulary collision on one word.** That document's tier table lists
  "linux-64 or linux-aarch64 wheels" as `Unsupported`, meaning no artifact
  exists. `packaging/matrix/platform_matrix.toml` marks the same two targets
  `designed`, and reserves `unsupported` for things that are deliberately out
  of scope and will not be built (`macos-x86_64`, `sdist`). Both readings are
  defensible in isolation and together they are misleading: a reader sees Linux
  wheels called unsupported in one place and planned in the other. The matrix
  vocabulary is the one a script enforces, so the cheaper fix is to reword that
  row to "no artifact yet" rather than to loosen `unsupported` here. Either
  way, pick one before both documents ship.
- **Decide which document owns what.** The natural split, and the one that
  needs no content moved: `COMPATIBILITY_POLICY.md` owns the policy, the tiers,
  and what a version bump promises; `PLATFORM_MATRIX.md` owns the artifact
  level detail, the expected tags, the validation rules, and the metadata a
  script checks. Add a cross-reference in each direction once the round is
  merged. Deliberately not added here, because that file belongs to another
  lane and may still move.

## Open questions

- Does the Mojo compiler honor `MACOSX_DEPLOYMENT_TARGET`? Unknown, and it is
  the gate on the whole macOS install story.
- What is the real glibc floor of a Linux build? The `manylinux_2_28` in the
  matrix is what `pixi.lock` was solved against, not a measurement of any
  shipped object.
- Which libraries does the Linux MAX runtime need bundled? The four names in
  the Linux target rows are the macOS set with `.so` substituted, and that is a
  guess.
- Does the built extension's CPython ABI dependency match the `cp314` tag it
  carries? The `.so` links no `libpython`, so the tag is a conservative choice
  rather than a measured one. An audit of which C API symbols it resolves at
  load time would tell us whether a wider tag is honest. Do not widen it on the
  strength of "it imported once".
- Should the C ABI library and the CLI binary be release artifacts at all, or
  stay development build products? The matrix deliberately covers neither, and
  `docs/PLATFORM_MATRIX.md` records that as a scoping decision. If they become
  artifacts, they need the same bundling step, the same provenance, and rows of
  their own.
- Should a second Apple chip be validated before the first release? An M4 Pro
  or M4 Max changes `multiprocessor_count` and nothing else about the backend,
  which makes it the cheapest test of whether the tiling policy scales on Metal.

## Deliberately not done

- No wheel was built and no artifact was validated.
- No workflow, pixi file, build script, or `README.md` was edited. Everything
  in this handoff is text, not an applied change.
- No platform is marked validated anywhere, including the macOS one that has a
  wheel sitting in `python/dist`.
- No accelerator row moved off `not-run` except the M4, which was already
  recorded in `docs/GPU_VALIDATION.md` before this lane started, and which is
  recorded here as `partial` rather than upgraded.
