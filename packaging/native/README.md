# Native artifact layout

How the C ABI shared library, its header, and the command line tool are laid
out in a release, what they are named, where their loader looks, and what
still has to be true before any of it can be distributed.

**Nothing here has been built or executed.** This directory specifies a
layout; it does not produce one. Every status in
[layout.toml](layout.toml) is `designed`, using the vocabulary in
`packaging/matrix/platform_matrix.toml`. Where a fact below was read off a
binary, it says so, and where it was not, it says that too.

The Python wheel is a separate artifact with its own pipeline
(`packaging/build_wheel.sh`, `packaging/linux/`, `packaging/macos/`). This
directory is about the artifact a C, R, Julia, or Go consumer links against,
which is not shipped inside a wheel and does not follow wheel rules.

## The tree

```
mojoboost-0.1.0-<platform>/
  include/mojoboost/mojoboost.h
  lib/libmojoboost.2.dylib          (macOS)   libmojoboost.so.2   (Linux)
  lib/libmojoboost.dylib -> ...     symlink for `-lmojoboost`
  lib/mojoboost-runtime/            the Mojo runtime this library needs
  bin/mojoboost                     the command line tool
  share/doc/mojoboost/              LICENSE, NOTICE, the SBOM
```

A GNU-style prefix, so `-I$PREFIX/include -L$PREFIX/lib -lmojoboost` works
with no mojoboost-specific knowledge and a distribution can drop the tree
under `/usr/local` or `/opt` without rearranging it.

The header sits one directory deep, so an installed consumer writes
`#include <mojoboost/mojoboost.h>`. In-tree it is `capi/mojoboost.h` and
`#include "mojoboost.h"`, which is what `capi/test_capi.c` does. Both
spellings have to keep working, which is why the header includes nothing but
`<stdint.h>` and has no sibling headers to find.

## Library names carry the ABI version, not the release version

`libmojoboost.so.2` and `libmojoboost.2.dylib`, from
`MOJOBOOST_ABI_VERSION`. A caller links against an ABI, not a release: two
mojoboost releases with the same ABI version must be swappable without
relinking, and the version in the name is what makes the loader enforce
that. The unversioned symlink exists only so `-lmojoboost` resolves at link
time.

ABI versions are cumulative — each adds declarations, none removes or
changes any — so a caller built against version 1 keeps working against a
library reporting 2. A change that genuinely broke a compiled caller would
ship under a different library name rather than as a version number that
silently means something else. See [docs/C_API.md](../../docs/C_API.md).

## Loader paths

The library must find the Mojo runtime beside itself, never through an
absolute path or an environment variable:

| Platform | Artifact | Search path |
|---|---|---|
| macOS | `lib/libmojoboost.2.dylib` | `@loader_path/mojoboost-runtime` |
| macOS | `bin/mojoboost` | `@executable_path/../lib/mojoboost-runtime` |
| Linux | `lib/libmojoboost.so.2` | `$ORIGIN/mojoboost-runtime` |
| Linux | `bin/mojoboost` | `$ORIGIN/../lib/mojoboost-runtime` |

`DYLD_LIBRARY_PATH` and `LD_LIBRARY_PATH` are not part of the contract. An
artifact that needs either is not relocatable, and telling a consumer to set
one leaks the build machine's layout into their environment.

## The Mojo runtime has to ship

A mojoboost shared library is not self-contained. Read off the built macOS
library with `otool -L`, transitively:

- Direct: `libKGENCompilerRTShared.dylib`, `libAsyncRTMojoBindings.dylib`
- Through those: `libMSupportGlobals.dylib`, `libAsyncRTRuntimeGlobals.dylib`

All four come from the Mojo toolchain, all four are referenced by `@rpath`,
and none is a system library. They go in `lib/mojoboost-runtime/`. Their own
install names have to be rewritten as well, because they reference each
other by `@rpath` too.

Everything else the library needs is resolved by the OS and must not be
shipped: `libSystem`, `libc++`, `libobjc`, and the CoreFoundation,
Foundation (weak), IOKit, CoreGraphics, and Metal frameworks. The Metal
dependency is worth noting — it comes in through the Mojo runtime and is
present whether or not this build has an accelerator.

The Linux closure has **not** been read off anything, because no Linux
shared library has been built in this checkout. The names in `layout.toml`
under `[runtime.linux]` are the expected ELF counterparts and must be
replaced with `readelf -d` output before a Linux artifact is published.

## What blocks distribution today

These are properties of the artifact `capi/build.sh` emits right now, read
off the built library, not hypotheticals. Each needs a build-script change
or a post-link step; none is fixed by this directory.

1. **The install name is a relative build path.** `LC_ID_DYLIB` is
   `capi/libmojoboost.dylib`. Anything linking it records that path as the
   thing to load, so it resolves only when the process runs from the
   directory above `capi/`. Needs
   `install_name_tool -id @rpath/libmojoboost.2.dylib`.
2. **The rpath points into the developer checkout.** The built library
   carries an absolute `LC_RPATH` into `.pixi/envs/default/lib`. A shipped
   copy would search a directory that exists only on the machine that built
   it. Strip it, add `@loader_path/mojoboost-runtime`.
3. **No runtime is staged.** `capi/build.sh` copies none of the four
   libraries above.
4. **No Linux artifact exists**, so every Linux row is a design.

`capi/run_c_tests.sh` passes today despite 1 and 2 because it adds an
absolute `-rpath` to the build tree for the test binary. That is correct for
an in-tree test and tells you nothing about whether the artifact is
relocatable.

## Distribution requirements

Before a native artifact is published:

- A per-platform SBOM, from `packaging/security/sbom_supplement.py`, listing
  the staged Mojo runtime libraries as vendored components. They are
  redistributed binaries, so they belong in the SBOM and their license goes
  in `share/doc/mojoboost/`.
- Recorded hashes, from `packaging/macos/hash_artifacts.sh`, over the
  staged tree rather than over the loose library.
- On macOS, codesigning and notarization. An unsigned dylib downloaded from
  a browser is quarantined and will not load; this is not optional for a
  published artifact and no signing identity is configured in this
  repository.
- A build-tree-freedom check: no absolute path from the build machine may
  appear in any shipped binary. This is what would have caught defect 2.
- Confirmation that the ABI version in `layout.toml` matches
  `MOJOBOOST_ABI_VERSION` in the shipped header.

## Files here

- `layout.toml` — the machine-readable layout, names, loader paths, runtime
  closure, and the blocking defect list.
- `README.md` — this file.

There is deliberately no build script here. Building is `capi/build.sh` and
`cli/build.sh`; a staging step that assembles the tree from already-built
inputs belongs next to them once the four defects above are fixed, and
writing it before then would produce a tree that does not load.
