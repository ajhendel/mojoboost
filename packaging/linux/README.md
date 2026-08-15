# packaging/linux

The Linux wheel: what a redistributable Linux artifact has to be, the builder
that produces it, and what has to be measured before its filename is allowed to
make a promise.

**Nothing in this directory has been executed.** No Linux wheel exists, none has
been built by the change that added these files, and no command below has been
run. Every number that is not read out of `pixi.lock` is marked as unmeasured,
because it is.

```
README.md                  this file: the plan and the contract
build_wheel_linux.sh       the builder, run by release-linux.yml. Never executed
check_metadata_ready.py    static preflight: is the repo's Python metadata Linux-ready
inspect_wheel.py           stdlib wheel and ELF inspector, no binutils required
inspect_elf.sh             binutils inspection of installed objects, on the target
container_elf_report.sh    the container-side driver for inspect_elf.sh
images.env                 container image references, digests deliberately empty
```

Related, and owned elsewhere:

| File | What it does |
| --- | --- |
| [`packaging/matrix/platform_matrix.toml`](../matrix/platform_matrix.toml) | Four Linux target rows, all `designed`: `linux-{x86_64,aarch64}-cp314` are the plain-tag wheels this builder produces by default and are `publishable = false`; the two `-manylinux` rows are the same builder with `MOJOTREES_TAG_POLICY=manylinux` and are gated on a measured glibc floor |
| [`packaging/matrix/smoke/clean_install_linux.sh`](../matrix/smoke/clean_install_linux.sh) | The clean-install acceptance fixture. This directory produces the wheel it is waiting for |
| [`packaging/matrix/validate_artifact.py`](../matrix/validate_artifact.py) | Matrix conformance for a built wheel. Its ELF branch is a byte scan and says so; `inspect_wheel.py` here is the dynamic-section parse it defers to |
| [`packaging/build_wheel.sh`](../build_wheel.sh) | The macOS builder. Not portable, see below |
| [`.github/workflows/release-linux.yml`](../../.github/workflows/release-linux.yml) | The manual release workflow that drives everything here |

## Two different things, one of which exists

The repository already supports Linux. It does not ship a Linux artifact. These
are separate claims and the difference is the whole subject of this directory.

**Source install, supported today.** `pixi.toml` lists `linux-64` and
`linux-aarch64`, and `.github/workflows/ci.yml` runs the Mojo suite, the Python
API suite, and the estimator suite on `ubuntu-latest` and `ubuntu-24.04-arm` on
every push and pull request. That is real, continuous, and it is what
`platform_matrix.toml` records as `status = "tested"` for `src-linux-x86_64` and
`src-linux-aarch64`. What it proves is that the source builds and passes on a
GitHub runner with the pinned pixi environment installed.

**Redistributable wheel, does not exist.** A wheel is a promise made to a
machine that has no pixi, no conda environment, no Mojo, and no MAX, that a
binary compiled somewhere else will load and run. CI proves nothing about that,
because CI runs the code with the toolchain sitting right there on the library
path. Every green Linux CI run in this repository is compatible with a Linux
wheel that fails to import on a user's machine.

Do not let the first sentence be used as evidence for the second one.

## Why the macOS builder does not port

`packaging/build_wheel.sh` copies four MAX dylibs into `python/mojotrees/.dylibs`,
rewrites the extension's `LC_RPATH` with `install_name_tool`, and re-signs
everything with `codesign`. None of those three steps has a Linux equivalent
that is a flag away:

- `install_name_tool -rpath` becomes `patchelf --set-rpath '$ORIGIN/.libs'`, a
  different program that is not currently in any pixi environment in this repo.
- `codesign` has no counterpart and needs none. Linux loaders do not check
  signatures on an ELF shared object.
- The four dylib names in the macOS list are Mach-O objects. **The Linux runtime
  set is unknown.** `platform_matrix.toml` carries the macOS names with `.so`
  substituted, and its own note says that is a guess. Discovering the real set is
  step 4 of `build_wheel_linux.sh` and it is a discovery, not a copy list.
- On macOS the extension links exactly two of the four directly
  (`libKGENCompilerRTShared`, `libAsyncRTMojoBindings`, per `otool -L`); the other
  two are transitive. The Linux closure has to be walked, not assumed.

## The tag policy, and why the first wheel is not manylinux

A platform tag is a claim about the oldest system the file works on. There are
two ways to get one wrong and only one of them is loud.

**Stage 0, `linux_x86_64` and `linux_aarch64`.** This is the honest tag for a
binary whose glibc floor has not been measured. It says "Linux, this
architecture, no promise about the distribution". Consequences, both intended:

- PyPI and TestPyPI **reject** these tags on upload. There is no way to publish
  an unaudited Linux wheel to an index by accident.
- `pip install ./mojotrees-...-linux_x86_64.whl` from a local file or a GitHub
  release asset works normally.

So stage 0 artifacts are distributed as GitHub release assets, and only to people
who are being asked to test them. This is the default of
`release-linux.yml` (`tag_policy: plain`).

**Stage 1, `manylinux_2_XX_<arch>`.** Legal only when all of the following have
been measured on the actual artifact rather than assumed:

1. The highest `GLIBC_` symbol version referenced by the extension **and** by
   every bundled runtime object is at or below `2.XX`. `inspect_wheel.py`
   reports this per object; the number the tag uses is the maximum across all of
   them, not the extension's.
2. Every `DT_NEEDED` entry is either bundled inside the wheel or is on the
   PEP 600 external allowlist (`libc`, `libm`, `libdl`, `librt`, `libpthread`,
   `ld-linux`, `libgcc_s`, `libstdc++`, and the handful of X/GL libraries that do
   not apply here). Anything else is a library the user's machine is being
   silently required to have.
3. `RUNPATH` on every shipped object is `$ORIGIN`-relative and contains no
   absolute path from the build machine.
4. A clean container of the *claimed* floor, not of the build host, ran
   `packaging/matrix/smoke/clean_install_linux.sh` against the file and passed.
   A pass on glibc 2.39 is not evidence for a `manylinux_2_28` tag. This is the
   check that catches the other three when they are wrong.

`2.28` appears in the Linux rows of `platform_matrix.toml` because that is the
`__glibc` virtual package `pixi.lock` was solved against (lock header,
`linux-64` and `linux-aarch64`). That is the floor the *environment resolver*
was told to respect. It is not a measurement of the shipped ELF objects, and
`validate_artifact.py` rule R4 makes the same point. Treat it as a hypothesis
with a plausible value, not as a result.

## Build hosts

Three options, in the order the workflow prefers them. None has been tried.

**A. GitHub-hosted native runners (the workflow's default).** `ubuntu-22.04`
for x86_64 and `ubuntu-22.04-arm` for ARM64. Native on both architectures, so no
emulation and no cross build. The build host's glibc becomes the artifact's
ceiling on how low the floor can be: Ubuntu 22.04 ships glibc 2.35, so a wheel
built there cannot honestly claim `manylinux_2_28` no matter what the symbol
scan says, unless the toolchain happens to reference nothing newer than 2.28,
which is exactly what has to be measured. Ubuntu 24.04 would put the ceiling at
2.39, which is worse; `ci.yml` uses `ubuntu-24.04-arm` because it is testing
source, not producing an artifact. If `ubuntu-22.04-arm` is not available to
this repository, the ARM64 job's floor rises and the tag has to rise with it.

**B. A `manylinux_2_28` container.** The right build host in principle: it has
the oldest glibc the tag would permit, so the ceiling and the claim coincide.
**Unverified and possibly a dead end:** it is not known whether the pinned Mojo
toolchain runs on AlmaLinux 8 with glibc 2.28. Modular publishes its own
supported-distribution list and this repository has never tested against the
container. Find out before building the workflow around it. `images.env` holds
the image references with their digest fields empty on purpose.

**C. A self-hosted machine.** Only if A and B both fail. It reintroduces the
problem `gpu-validation.yml` already has: a job that cannot run until someone
registers a runner.

**Whichever host is used, it must have no accelerator visible.**
`has_accelerator()` in `src/mojotrees/device.mojo` resolves at compile time, so a
wheel built on a machine with a working CUDA or ROCm stack ships a different
product under the same filename: it reports a GPU as available and then fails
when the device is opened. This is recorded in the provenance sidecar as
`has_accelerator_at_build` and it is the single most important field in that
file for Linux.

## What goes inside the wheel

```
mojotrees/
    __init__.py, estimator modules, ...
    _mojotrees.so             the Mojo-built extension, RUNPATH $ORIGIN/.libs
    .libs/
        lib*.so*              the MAX runtime closure, RUNPATH $ORIGIN
mojotrees-0.1.0.dist-info/
    METADATA, WHEEL, RECORD, licenses/LICENSE
```

`.libs/` rather than auditwheel's `mojotrees.libs/` because
`clean_install_linux.sh` already looks in both and `.libs/` sits inside the
package, which keeps the `$ORIGIN` relationship trivial. Bundled libraries are
**not** renamed with a content hash. auditwheel does that to prevent two
independently vendored copies of the same soname from colliding in one process.
mojotrees bundles one proprietary runtime that nothing else on PyPI vendors, so
the hash suffix would buy nothing and cost the ability to compare two wheels
byte for byte. Revisit if a second package ever ships the MAX runtime.

**Three things in the Python metadata and the environment have to be right
before any of this works, and none of them is this directory's file to fix.**
`check_metadata_ready.py` checks all three and the builder refuses to run while
one is wrong:

1. `python/setup.py` must not apply its macOS `plat_name` on Linux. Applied
   unconditionally it produces a wheel tagged for macOS holding an ELF object,
   which is the most confidently wrong artifact this project could make. Guarded
   by `sys.platform == "darwin"` as of Task 01's work in this round. The builder
   also passes `--plat-name` explicitly and verifies the filename it got, which
   costs nothing and turns a regression into a caught error.
2. `python/pyproject.toml` package-data must match what gets staged. `["*.so",
   ".dylibs/*.dylib"]` matches nothing under `.libs/`, so the bundled runtime
   would be dropped from the wheel silently and the failure would surface as an
   `ImportError` on a user's machine. Both `.libs/*.so` and `.libs/*.so.*` are
   needed, and both are there as of Task 01's work in this round.
3. `pixi.toml` must provide `patchelf` on Linux. It does not, and this one is
   still open: it changes `pixi.lock`, so it goes through the integration owner.

All three are written out as exact edits in
[`handoffs/release_03_linux_wheels.md`](../../handoffs/release_03_linux_wheels.md).

## Size

`max-core` for `linux-64` is 117 MB compressed in the pinned channel, and 82 MB
for `linux-aarch64` (`pixi.lock`). Only the runtime objects the extension
actually needs get bundled, not the package, so the wheel will be far smaller
than that. How much smaller is unmeasured. PyPI's per-file limit is 100 MB
without an approved exception, and it is worth knowing which side of that line
the artifact lands on before the upload attempt rather than during it.

## Licensing, unresolved

The MAX and Mojo conda packages are published under
`license: LicenseRef-Modular-Proprietary` (`pixi.lock`, every `max`, `max-core`,
`mojo`, and `mojo-compiler` record). A wheel that bundles their shared objects is
redistributing proprietary binaries inside an Apache-2.0 package. **Nobody on
this project has read the Modular license terms and confirmed that
redistribution is permitted.** That question blocks publication to any index; it
does not block building an artifact and inspecting it, which is what this
directory is for. If the answer turns out to be no, the Linux (and macOS) wheel
story changes shape entirely and the fallback is a wheel that requires a MAX
installation, which is a different product.

Second, smaller question: if the closure ends up including conda-forge's
`libstdc++` or `libgcc_s`, those carry GPL-3 with the GCC Runtime Library
Exception. The exception is written for exactly this case and is very likely
fine, and the wheel still has to carry their license texts. The builder logs
loudly when it bundles either one.

## What is deliberately not supported

| System | Behavior | Why |
| --- | --- | --- |
| musl (Alpine) | `pip` finds no matching distribution | No musl build of the toolchain exists. A `musllinux` wheel would have to be built against musl, and cannot be |
| glibc older than the tag | `pip` refuses the file by tag | Correct and automatic once the tag is honest, which is the entire point of stage 0 |
| 32 bit, ppc64le, s390x, riscv64 | no wheel | `pixi.toml` lists no such platform and the channel ships no toolchain |
| CPython 3.13 and earlier, and 3.14 free-threaded | `pip` refuses by tag | `max 26.5.0` pins `python 3.14.*` and depends on `python-gil`. See the `python` rows in `platform_matrix.toml` |
| sdist fallback | none published | An sdist cannot build without the Mojo toolchain, and publishing one turns "no matching distribution" into a compiler error |

## Running any of this later

Nothing here is wired into `pixi.toml` or into `ci.yml`, on purpose: a task that
builds a wheel is a release operation and should not be one keystroke away from a
test run. The exact commands, the container invocations for both architectures,
and the full list of what has never been executed are in
[`handoffs/release_03_linux_wheels.md`](../../handoffs/release_03_linux_wheels.md).
