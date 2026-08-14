# packaging/matrix

The release matrix: which platforms mojoboost claims, what the artifact for
each is called, and what evidence exists behind every claim.

[`docs/PLATFORM_MATRIX.md`](../../docs/PLATFORM_MATRIX.md) is the prose. This
directory is the data and the checks.

```
platform_matrix.toml        the matrix: targets, interpreters, toolchain floor
accelerators/index.toml     one row per device, every one of them not-run bar the M4
accelerators/TEMPLATE_*.md  what a device record has to contain, per vendor
validate_matrix.py          checks the metadata against itself and the repository
validate_artifact.py        checks a built wheel against the matrix
smoke/clean_install_*.sh    clean-install fixtures, one per operating system
smoke/probe_platform.py     prints the platform facts a record has to contain
```

## What runs and what does not

`validate_matrix.py` runs anywhere, in under a second, with the standard
library. It builds nothing and imports no part of mojoboost, so it is cheap
enough to run on every change and it works on a bare checkout. Wiring it into
pixi and CI is specified in
[`handoffs/task18_platform.md`](../../handoffs/task18_platform.md).

Everything else in this directory has **not been executed**. `validate_artifact.py`
needs a wheel, and the smoke fixtures need a target machine with no toolchain on
it. They are written down now so the acceptance criteria are settled and
reviewable before the first artifact is published, rather than being invented
afterwards to fit whatever the build produced.

Nothing here has built a wheel, validated a platform, or run anything on an
accelerator.

## Why the metadata is separate from the prose

Because a status is a claim and a claim needs a check. `validate_matrix.py`
enforces one rule above all others:

> A target that says `validated` must name an evidence file that exists, and a
> device that says `validated` must have all four recorded steps passing.

A document alone cannot enforce that. The failure it prevents is the ordinary
one, where a platform is added to a table because the build worked on the
author's machine, and six months later the table is the only thing anybody
reads.

## Relationship to the rest of packaging/

`build_wheel.sh` and `test_wheel.sh` build a macOS wheel and check that it
works, from inside the pixi environment. They are not superseded by anything
here.

This directory answers the two questions they cannot:

- does the artifact say true things about where it can be installed
- does it carry everything it needs and nothing from the machine that built it

A wheel that passes `test_wheel.sh` and fails `validate_artifact.py` is a wheel
that works on the developer's machine and lies on its label.
