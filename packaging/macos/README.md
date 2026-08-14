# packaging/macos

Release machinery for the self-contained macOS arm64 wheel.

`packaging/` already contains the developer-facing build (`build_wheel.sh`,
`test_wheel.sh`) and the release contracts (`packaging/matrix/`). This directory
is the third thing: what a *release* does on top of a build, which is
provenance, verification, hashes, and artifact retention.

```
build_release_wheel.sh    the release build: preflight, build, verify, hash
provenance.sh             writes the wheel's provenance sidecar
inspect_wheel.py          describes the wheel and checks the release-only rules
hash_artifacts.sh         writes and verifies SHA256SUMS
check_action_pins.py      every third-party action in the workflow is SHA pinned
report_accelerator.mojo   prints the build's compile-time has_accelerator()
```

## Execution status

**Nothing in this directory has been executed.** No wheel was built, no wheel
was inspected, no hash was computed, no workflow was dispatched, and nothing was
published to any index. Every script here is written to be read and reviewed
before it is trusted, and every claim it will make is a claim about a future
run.

The same is true of `.github/workflows/release-macos.yml`, which is the caller
for all of this and which cannot run today for two independent reasons: the
runner it targets does not exist, and its action pins are placeholders.
`handoffs/release_02_macos_wheels.md` lists both, with the exact commands to
resolve them.

## What this reuses rather than replaces

The existing contracts are the contracts. This directory calls them and does not
copy them.

| Reused | Where it comes from | What it answers |
|---|---|---|
| `pixi run -e pkg build-wheel` | `packaging/build_wheel.sh` | builds the extension, bundles the MAX runtime, rewrites the rpath, re-signs, builds the wheel |
| `pixi run -e pkg test-wheel` | `packaging/test_wheel.sh` | does the wheel work, in two clean venvs |
| `validate_matrix.py` | `packaging/matrix/` | does the release matrix still agree with the repository |
| `validate_artifact.py` | `packaging/matrix/` | does the wheel match the target the matrix declares |
| `smoke/clean_install_macos.sh` | `packaging/matrix/` | does the wheel install and run on a machine with no toolchain |
| `macho_info()` | `packaging/matrix/validate_artifact.py` | the repository's only Mach-O load command parser, imported by `inspect_wheel.py` |

There is deliberately no second wheel builder here. Two builders producing
artifacts with the same filename is the failure this whole packaging tree exists
to prevent.

## What `inspect_wheel.py` adds over `validate_artifact.py`

`validate_artifact.py` checks a wheel against `platform_matrix.toml`: is the tag
declared, is the architecture right, is the deployment target no newer than the
tag claims, is everything signed, is every `@rpath` dependency bundled. Those
rules stand and are not re-implemented.

The release adds four questions that a matrix check does not ask.

1. **Is the platform tag exactly right, rather than merely not-a-lie?**
   `validate_artifact.py` rule R5b passes when the extension's `minos` is *at or
   below* the tag's floor. A wheel built for macOS 12 and tagged
   `macosx_26_0_arm64` passes that rule and is still wrong at release time,
   because pip then refuses it on every Mac between 12 and 26. The release check
   is equality.
2. **Is the bundle minimal?** "Bundle only what is required" has two failure
   directions. A missing library is caught by R5e. A library that nothing in the
   wheel loads is not caught anywhere, and shipping one means shipping bytes,
   and a license obligation, for no reason.
3. **What do the bundled libraries declare about themselves?** Their
   `LC_ID_DYLIB` install name and their own `LC_RPATH` entries are load-time
   inputs that no existing check reads.
4. **Is anything from the source tree in the artifact?** Caches, test data,
   scratch files, and secret-shaped strings, across every member of the zip
   rather than only the Mach-O objects.

## Code signing and notarization

**Ad-hoc signing is necessary. Developer ID signing and notarization are not,
for this artifact.** The reasoning, so the decision can be re-argued when the
artifact changes rather than re-derived from scratch:

- On Apple silicon every Mach-O image must carry a valid signature to be mapped,
  and `install_name_tool` invalidates the signature of any object it rewrites.
  `packaging/build_wheel.sh` already re-signs the extension and each bundled
  dylib with `codesign --force --sign -` (ad-hoc). That is not optional and it
  is not a Gatekeeper measure: it is what makes the rewritten objects loadable
  at all. `inspect_wheel.py` and `validate_artifact.py` rule R5d both check that
  the signature survived into the wheel.
- Notarization is enforced against quarantined code. The quarantine attribute is
  applied by the downloader, and pip is not one: it fetches a zip over HTTPS and
  extracts members with `zipfile`, which writes plain files with no extended
  attributes. An ad-hoc signed library that is not quarantined loads.
- A `.whl` is also not a container that notarization can staple to. The Apple
  path expects a `.app`, `.pkg`, `.dmg`, or a signed zip, and the stapled ticket
  would be discarded by pip's extraction even if one existed.
- The cost is not zero and lands on the part of the release that should stay
  boring. Developer ID signing in CI means a long-lived signing identity plus
  App Store Connect credentials for `notarytool` living as secrets next to a
  self-hosted runner, which is exactly the credential posture
  `docs/RELEASE_SECURITY.md` is being written to avoid.

Two situations flip this decision, and neither is true today. Shipping a
downloadable installer or a standalone binary (a `.pkg`, a `.dmg`, or the
`cli/mojoboost` executable offered as a release download) puts a quarantine
attribute on something a user launches directly, and that needs Developer ID and
notarization. And a user who downloads the `.whl` in a browser gets a quarantined
*wheel file*; the assumption here is that the attribute does not propagate to the
members pip extracts from it. That assumption is stated in
`handoffs/release_02_macos_wheels.md` with the command that would confirm it, and
it has not been run.

## The one thing to check before trusting any of this

`has_accelerator()` is resolved at compile time (`src/mojoboost/device.mojo`), so
a wheel built on a machine with a visible accelerator is a different product from
the same commit built on a machine without one, under the same filename.
`provenance.sh` records the answer, and refuses to certify a build whose
accelerator answer and Metal toolchain disagree. Read that section of the handoff
before registering a runner.
