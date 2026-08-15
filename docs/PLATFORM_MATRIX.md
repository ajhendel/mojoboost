# Platform and release matrix

Where mojotrees can be installed, what the artifact for each place is called,
which interpreter it targets, and, for every row, what evidence exists that any
of it is true.

The rule this document is built around, and the reason it exists at all:

> A platform is **validated** when hardware ran the artifact and somebody wrote
> down what happened. Not when it builds. Not when CI is green. Not when the
> matrix says so.

Nothing below has been validated. No wheel has been published, no accelerator
other than one Apple M4 has executed a single kernel, and no NVIDIA or AMD
device has ever run this code.

The machine-readable half of this document is
[`packaging/matrix/platform_matrix.toml`](../packaging/matrix/platform_matrix.toml),
and `python3 packaging/matrix/validate_matrix.py` checks the two against each
other. If the prose here and the metadata there disagree, the metadata is
authoritative and one of them is a bug.

## Status vocabulary

The same four words in the prose and in the metadata.

| Status | Means |
|---|---|
| `validated` | Hardware ran the artifact and the run is recorded in the file the row names |
| `tested` | CI or a local run exercised the source here, but no distributable artifact was produced or installed from it |
| `designed` | The path is specified and believed correct. Nothing has run, and there is no artifact |
| `unsupported` | Deliberately out of scope. The failure behavior is specified below |

`validate_matrix.py` rejects a row that says `validated` with no evidence file,
which is the one automated defense against this document aging into marketing.

## Targets

| Target id | Artifact | Expected filename | Status | Index-publishable |
|---|---|---|---|---|
| `macos-arm64-cp314` | wheel | `mojotrees-0.1.0-cp314-cp314-macosx_26_0_arm64.whl` | `designed` | yes |
| `macos-arm64-cp314-lowered` | wheel | `mojotrees-0.1.0-cp314-cp314-macosx_12_0_arm64.whl` | `designed` | yes |
| `linux-x86_64-cp314` | wheel | `mojotrees-0.1.0-cp314-cp314-linux_x86_64.whl` | `designed` | no |
| `linux-aarch64-cp314` | wheel | `mojotrees-0.1.0-cp314-cp314-linux_aarch64.whl` | `designed` | no |
| `linux-x86_64-cp314-manylinux` | wheel | `mojotrees-0.1.0-cp314-cp314-manylinux_2_28_x86_64.whl` | `designed` | yes, once measured |
| `linux-aarch64-cp314-manylinux` | wheel | `mojotrees-0.1.0-cp314-cp314-manylinux_2_28_aarch64.whl` | `designed` | yes, once measured |
| `macos-x86_64` | none | none | `unsupported` | n/a |
| `sdist` | none | none | `unsupported` | n/a |

The last column is the `publishable` field, and it is the one that stops a
plausible mistake. The two plain Linux rows are the wheels the builder produces
when every input is left at its default. Their tags promise nothing about glibc
and every index rejects them, so they are honest artifacts that ship as files
and never as an upload. The two `-manylinux` rows are the same builder run with
`tag_policy=manylinux`, and they are reached by measuring a floor, never by
renaming a file.

And the source install, which is how every current user actually installs
mojotrees:

| Source install | Platform | Status | Evidence |
|---|---|---|---|
| `src-linux-x86_64` | `linux-64` | `tested` | `ci.yml`, `test` and `python` jobs on `ubuntu-latest` |
| `src-linux-aarch64` | `linux-aarch64` | `tested` | `ci.yml`, `test` and `python` jobs on `ubuntu-24.04-arm` |
| `src-macos-arm64` | `osx-arm64` | `tested` | local only, plus the M4 record in [GPU_VALIDATION.md](GPU_VALIDATION.md) |

A wheel with the `macos-arm64-cp314` filename exists in `python/dist` in at
least one working copy, from an earlier local `pixi run build-wheel`. That is
an artifact on a disk, not a record. There is no clean-install run behind it,
no host provenance, and no recorded output, so the row stays `designed`.

## Interpreters

One build target, five runnable. Those are different questions and this table
used to answer only the first.

The extension links no libpython. It resolves CPython entry points by name out
of the running interpreter at load time, so the interpreter that *compiles* it
does not determine the interpreters it *runs* on. What the build interpreter
does determine is the wheel's tag, because setuptools tags a wheel for
whichever interpreter ran the build.

| Tag | Status | Why |
|---|---|---|
| `cp314` | `tested`, build target | The interpreter the pixi environment resolved to, so the one the Mojo toolchain compiles against and the one the wheel is tagged for. A consequence of `pixi.toml` pinning no python, not of a toolchain requirement |
| `cp313`, `cp312`, `cp311`, `cp310` | `tested` | The cp314-built extension imports and passes `python/test_python_api.py` on each, unmodified, on osx-arm64. No wheel has been built by any of these environments, so no artifact carries their tag |
| `cp39` and earlier | `unsupported` | The extension aborts at load with `symbol not found: Py_NewRef`, a CPython 3.10 addition. Also below the toolchain's own floor, so there is nothing to build with either |
| free-threaded, any version | `unsupported` | `max 26.5.0` depends on `python-gil`, which no free-threaded build provides. The environment cannot resolve, so no extension can be built. The one row here nobody has measured |
| `abi3` | `unsupported` | Not a lever this project has: there is no C source and no compile step where `Py_LIMITED_API` could be set. Also not the lever it would need, since the binary links no libpython. Do not tag a wheel `abi3` to widen its reach |

The toolchain facts come from `pixi.lock`, and `validate_matrix.py` re-derives
them on every run. **The lock records the variant that was solved, not the
variants that exist**, and reading it as the latter is what previously put
"No MAX build for them in the pinned channel" in this table. It was false:
`max 26.5.0` is published for 3.10 through 3.14 on all three platforms.

The `tested` rows are measured on osx-arm64 against a source install, and are
recorded in [docs/PYTHON_SUPPORT.md](PYTHON_SUPPORT.md) section 10. Linux is
not measured, and a floor is not a promise that a wheel exists for every
version above it.

## The easy macOS wheel path

The Apple product story is a Mac user typing `pip install mojotrees` and
getting a working GBDT library that uses the GPU already in their machine, with
no Mojo, no MAX, no conda, and no compiler. Two thirds of that already works.
The remaining third is a version number.

**What works.** `packaging/build_wheel.sh` builds the extension, copies the
four MAX runtime dylibs into `mojotrees/.dylibs`, rewrites the extension's
rpath from the build machine's pixi environment to `@loader_path/.dylibs`, and
re-signs everything, because `install_name_tool` invalidates a signature on
arm64. That is a delocate-style self-contained wheel, and it is the reason an
install needs no toolchain.

**What does not.** The wheel's platform tag is `macosx_26_0_arm64`, so pip
refuses to install it on macOS 15 and everything older. Most Macs in the world
are running something older.

That floor is not a property of the code. It is measured, and the measurement
splits cleanly in two:

- the Mojo-built extension carries `minos 26.0`, `sdk 26.5` in its
  `LC_BUILD_VERSION`, inherited from the SDK on the build machine
- the four bundled MAX runtime dylibs carry `minos 11.0`

So the runtime that ships inside the wheel is built for macOS 11. Only the
compile step imposes 26.0, and `python/setup.py` then hardcodes the tag to
match it. The easy path follows directly:

1. Lower the deployment target on the Mojo build step, most likely through
   `MACOSX_DEPLOYMENT_TARGET` in the environment `bindings/build.sh` runs in.
   **Whether the Mojo compiler honors it has not been tested.** This is the
   entire risk in the plan, and it is one command to find out.
2. Confirm with `otool -l python/mojotrees/_mojotrees.so | grep -A 4
   LC_BUILD_VERSION` that `minos` actually moved. If it did not, stop. Do not
   relabel a binary whose `minos` says otherwise: the wheel would install and
   then fail to load, which is worse than pip refusing it.
3. Stop hardcoding the platform tag. `setup.py` should read `minos` back out of
   the built extension and derive `plat_name` from it, so the tag cannot
   disagree with the binary even in principle.
4. Test on an actual older Mac, or at minimum on the oldest macOS available,
   and record it. A tag is a promise about machines you do not own.

`macos-arm64-cp314-lowered` is the row for the result. Its target floor is
macOS 12.0 rather than 11.0 to leave the bundled libraries a margin, and the
number moves to whatever step 2 measures. The exact edits are in
[`handoffs/task18_platform.md`](../handoffs/task18_platform.md).

Two things this path does not buy:

- **Intel Macs.** `pixi.toml` lists no `osx-64` platform and the pinned channel
  ships no Intel macOS toolchain, so there is nothing to build with, and
  `universal2` is out for the same reason: a fat wheel needs an x86_64 half.
  Intel Macs get a tag mismatch from pip, which is the correct failure.
- **A GPU guarantee.** See the next section.

## Metal toolchain requirements

Three separate things get confused here, and the wheel story depends on keeping
them apart.

1. **Having an Apple GPU.** Every Apple silicon Mac has one.
2. **Being able to build GPU code.** That needs the Metal compiler, which is an
   Xcode component and a separate installation from the Command Line Tools.
3. **The build deciding whether to compile the GPU path in at all.** Mojo
   resolves `has_accelerator()` at compile time
   (`src/mojotrees/device.mojo`), so this is decided on the build machine and
   frozen into the artifact.

The consequences, in order of how much trouble they cause:

- **`.github/workflows/ci.yml` has no macOS runner on purpose.** A
  GitHub-hosted Apple silicon runner reports an accelerator at compile time and
  has no Metal toolchain, so the GPU equivalence tests cannot build there. This
  is not a runner we have not gotten around to adding. It is a combination that
  does not work, and `osx-arm64` is validated locally instead.
- **A release wheel must be built on a machine with the Metal toolchain**, and
  the build must record that it had one. Verify with `xcrun --find metal` and
  `xcrun metal --version`, and record both. The component's exact name has
  moved between Xcode releases, so confirm it against the Xcode on the build
  machine rather than against this file.
- **GPU availability is baked in at build time, and the wheel cannot re-decide
  it.** A wheel built on a Mac with a working Metal toolchain reports a GPU as
  available on every machine that installs it. On a machine that then cannot
  open a device, the failure arrives when the device is opened rather than when
  it is resolved. `auto` selects the GPU on its own only when the device
  probe reports the hardware a crossover rule names (an Apple M4 today), which
  a host with no usable device cannot report, so the gap is reachable through
  `device="gpu"` and through `MOJOTREES_AUTO_MIN_CELLS` on a redistributed
  build. `MOJOTREES_DISABLE_GPU=1` is the way to pin such a build to the CPU.
- **A wheel built where no accelerator was visible has no GPU path in it at
  all**, and `device="gpu"` raises everywhere it is installed, including on
  machines with a perfectly good GPU. This is why the provenance sidecar records
  `has_accelerator_at_build`. Two wheels with identical filenames and different
  answers to that question are different products.

Until a per-machine runtime device query exists, publishing one macOS wheel
means choosing one answer for every user. The build machine is a Mac with a
Metal toolchain, so the answer is GPU-enabled, and the failure mode on a
machine that cannot open a device is a raised exception rather than a wrong
number.

## Linux

A Linux wheel builder exists. `packaging/linux/build_wheel_linux.sh` is a
separate program from the macOS one rather than a port of it, which is what the
platforms require: `packaging/build_wheel.sh` calls `install_name_tool` and
`codesign`, neither of which has a Linux counterpart, and the ELF side instead
walks the extension's `DT_NEEDED` closure, stages it into `mojotrees/.libs`, and
sets an `$ORIGIN` RUNPATH. `.github/workflows/release-linux.yml` runs it on a
runner per architecture, with a clean-install job in a container that is
deliberately not the build image.

What that builder does **not** do is decide the tag for you. It reads
`MOJOTREES_TAG_POLICY`, defaults to `plain`, and emits `linux_<arch>`. The
workflow's `tag_policy` input defaults to `plain` for the same reason. This is
the point of the split rows in the table above, and it was worth writing down
because this document previously claimed the opposite in both directions: it
said no builder existed, and it named a `manylinux_2_28` file that no default
build has ever produced.

Two numbers still have to be measured before either `-manylinux` row is
published:

- **The glibc floor.** `manylinux_2_28` in those rows is the floor `pixi.lock`
  was solved against, which is a statement about the build environment, not a
  measurement of the shipped objects. The real floor is the highest `GLIBC_`
  symbol version any bundled object references. `validate_artifact.py` rule R5f
  reads it from the wheel, and
  `packaging/matrix/smoke/clean_install_linux.sh` reads it with `readelf` on
  the target. The plain rows say `unmeasured` for their installer floor rather
  than repeating the solved number as though it were a result.
- **The bundled library set.** The four names in the Linux rows are the macOS
  set with a different extension. The Linux runtime layout has not been
  inspected, so treat that list as a guess until `readelf -d` on a real build
  says otherwise. Nothing currently catches an error in it:
  `validate_artifact.py` rule R2b compares `bundled_dylibs` on macOS only, so
  on Linux that list is documentation rather than a checked contract.

Both Linux platforms already run the full test suite in CI on every push, on
`ubuntu-latest` and `ubuntu-24.04-arm`, so the source install is `tested`. What
is missing on Linux is no longer the builder or the workflow. It is a build that
has actually been run: every Linux row is `designed`, no wheel has been produced
by either tag policy, and no clean install has happened.

## Artifact classes this matrix does not cover

The matrix above is about wheels, because the Python API is the only thing this
project currently intends to publish. Two other build scripts produce
redistributable-looking artifacts, and leaving them unmentioned would make this
document read as more complete than it is.

| Artifact | Built by | Covered here |
|---|---|---|
| Python wheel | `packaging/build_wheel.sh` (macOS), `packaging/linux/build_wheel_linux.sh` (Linux) | yes, the whole document |
| C ABI shared library | `capi/build.sh` | no |
| Command line tool | `cli/build.sh` | no |

Neither of the other two is redistributable today, and the reason is worth
writing down because it is the same reason the wheel needed a bundling step.
`capi/build.sh` runs `mojo build --emit shared-lib` and stops there. The
resulting `capi/libmojotrees.dylib` in a working copy carries `minos 26.0` and
a single rpath pointing at an absolute path inside the build machine's pixi
environment, so it resolves its MAX runtime dependencies only on the machine
that built it, at that exact path. Copy it anywhere else and it fails to load.

That is not a defect in `capi/build.sh`, which is a development build script and
works as one. It is the observation that the delocate-style step in
`packaging/build_wheel.sh` (copy the four runtime dylibs, rewrite the rpath to
`@loader_path`, re-sign) is what turns a Mojo build product into something that
can leave the machine, and neither the C library nor the CLI binary has an
equivalent.

This matters ahead of the R bindings on the roadmap, which are meant to sit on
top of the C ABI. An R package that links a library with an absolute rpath into
somebody else's home directory is not distributable, so the bundling question
has to be answered before the binding work is worth starting, not after. The
same `validate_artifact.py` rules apply almost unchanged: R5b, R5c, R5d, and R6
are exactly the checks a shippable `libmojotrees.dylib` would need to pass.

Out of scope until the wheel path is real, and recorded here so that scoping is
a decision rather than an oversight.

## Accelerators

Summary only. [`docs/GPU_VALIDATION.md`](GPU_VALIDATION.md) is the record of
truth and
[`packaging/matrix/accelerators/index.toml`](../packaging/matrix/accelerators/index.toml)
is its index.

| Vendor | API | Devices with any recorded evidence |
|---|---|---|
| Apple | Metal | one, an M4, correctness and determinism only |
| NVIDIA | CUDA | none |
| AMD | HIP | none |

There is one GPU source for all three backends and no vendor branch anywhere in
it. That is a design commitment worth exactly as much as the evidence behind
it, and the evidence today is one chip. No NVIDIA and no AMD device has ever
executed this code, on a laptop or in CI, so nothing in this repository should
be read as a claim about their behavior or performance.

Record templates, one per vendor, with the fields a record must contain and the
vendor-specific things worth checking while the hardware is in front of you:

- [`TEMPLATE_apple.md`](../packaging/matrix/accelerators/TEMPLATE_apple.md)
  covers M1 through M5, and starts with the Metal toolchain check, because a
  machine that cannot build the GPU path is a different result from a chip that
  fails a test.
- [`TEMPLATE_nvidia.md`](../packaging/matrix/accelerators/TEMPLATE_nvidia.md)
- [`TEMPLATE_amd.md`](../packaging/matrix/accelerators/TEMPLATE_amd.md)

The index carries a row per device we would say anything about, every one of
them `not-run` except the M4. A row is a fill-in target, not a support claim.

## Unsupported combinations and how they fail

The general principle, the same one `device="gpu"` follows: fail where the
mistake is, loudly, rather than somewhere convenient and quietly.

| Combination | What happens | Where it is decided |
|---|---|---|
| macOS wheel on an older macOS than the tag | pip: "not a supported wheel on this platform". Nothing installs | platform tag |
| macOS wheel on an Intel Mac | Same tag mismatch | platform tag, `arm64` |
| The cp314 wheel on Python 3.10 to 3.13 | Tag mismatch, so pip declines. The code would have run: this is a missing artifact, not an incompatibility, and the fix is to build a wheel per interpreter | `cp314` tag |
| Any wheel on Python 3.9 or older | Tag mismatch, and `requires-python = ">=3.10"` refuses it too. Here the code genuinely would not run: the extension aborts on `Py_NewRef` | `cp3XX` tag, and the floor |
| Any wheel on a free-threaded interpreter | Same tag mismatch. The ABI tags differ | `cp314` vs `cp314t` |
| `pip install mojotrees` on Linux, today | "no matching distribution found". No sdist is published, so pip cannot fall back to a source build that would fail with a compiler error instead | no sdist |
| `device="gpu"` with no accelerator in the build | Raises. It never falls back to the CPU silently | `src/mojotrees/device.mojo` |
| `device="gpu"` on a redistributed build whose host has no usable device | Raises when the device is opened, later than the resolve | `has_accelerator()` is compile time |
| `device="gpu"` for multiclass | Raises. Multiclass grows one tree per class per round on the CPU | `device.mojo`, `fit_multiclass` |
| `device="auto"` | Chooses the GPU for dense single-output regression and binary classification on an Apple M4 from 25 million cells and 200,000 rows; the CPU everywhere else, since no other device has a measured crossover | `crossover_rules()` in `device_policy.mojo` |
| GPU tests on a machine with no accelerator | Print `skipped: no accelerator` and pass. A pass that says `skipped` is not a validation | test suites |
| A wheel whose tag no target declares | `validate_artifact.py` rule R1 fails the release | release check |

The last row is the one to resist fixing the easy way. A tag that no target
declares means either the build is wrong or the matrix is out of date, and
adding the tag the build happened to produce resolves the symptom while
publishing the mistake.

## Artifact validation rules

`python3 packaging/matrix/validate_artifact.py <wheel>` checks a built wheel
against the matrix. It has not been run, because it needs a wheel. Standard
library only, and it parses the Mach-O load commands itself, so it needs
neither delocate nor auditwheel nor an install of the thing it inspects.

| Rule | Checks |
|---|---|
| R1 | The filename's tag matches a declared target, and the name and version match the matrix |
| R2 | Exactly one extension module (R2a), and exactly the declared bundled runtime libraries (R2b, macOS only) |
| R3 | No `__pycache__`, test, or build directory members |
| R4 | `Requires-Python` matches the target's interpreter, and a license file ships |
| R5a | One architecture, the one on the label. No fat binaries |
| R5b | Every object's `minos` is at or below the floor the platform tag promises |
| R5c | Every rpath is `@`-relative. No absolute path into the build machine |
| R5d | Every object carries a code signature, because `install_name_tool` invalidates it |
| R5e | Every `@rpath/` dependency resolves to a library that is actually in the wheel |
| R5f | Linux: the highest `GLIBC_` symbol version referenced, for comparison against the tag |
| R6 | No build-host strings anywhere in the bytes, not only in the load commands |
| R7 | A provenance sidecar exists and is complete |

The provenance sidecar is the part that cannot be recovered later. A wheel does
not know which Mojo built it, which macOS it was built on, whether a Metal
toolchain was present, or whether an accelerator was visible at compile time,
and that last one changes what the artifact does on the user's machine. Write
it at build time or lose it.

Clean-install smoke tests live in `packaging/matrix/smoke/`. They differ from
`packaging/test_wheel.sh` in the only way that matters for a release: they
refuse to run inside the pixi environment. A wheel tested with the Mojo
toolchain on `PATH` and `CONDA_PREFIX` set has not been shown to be
self-contained, because the thing it must not need is sitting right there.

## Changing a status

1. Run the thing. The smoke fixture for a target, the procedure in
   [GPU_VALIDATION.md](GPU_VALIDATION.md) for a device.
2. Paste the output, verbatim, into a record. Device records go in
   GPU_VALIDATION.md, artifact records go next to the release.
3. Point the row's `evidence` or `record` field at that file.
4. Change the status, and only then.
5. `python3 packaging/matrix/validate_matrix.py`, which fails if 3 and 4 were
   done in the other order.

## What has been run for this document

Nothing. No wheel was built, no artifact was validated, no platform was tested,
and no device was exercised while writing any of this.

The macOS facts quoted above were read with `otool` and `ls` out of files
already present in a working copy, all of them build products of earlier local
runs:

- `python/mojotrees/_mojotrees.so`: Mach-O arm64, `minos 26.0`, `sdk 26.5`
- the four bundled MAX dylibs in the pixi environment: `minos 11.0`
- `capi/libmojotrees.dylib`: `minos 26.0`, one rpath, absolute, into the build
  machine's pixi environment
- `python/dist/mojotrees-0.1.0-cp314-cp314-macosx_26_0_arm64.whl` exists

The toolchain facts were read out of `pixi.lock`. Everything else is design.
