# Contributing to mojotrees

mojotrees is an experimental public alpha. Contributions that improve
correctness, integration, portability, documentation, and reproducible
performance evidence are welcome.

## Before changing code

1. Read `docs/LIGHTGBM_PARITY.md` for the public behavior contract.
2. Read `docs/GPU_VALIDATION.md` before making accelerator claims.
3. Check open issues and current code before adding a second implementation.
4. Keep the Python layer thin. Training logic belongs in Mojo.
5. Preserve unrelated work in the tree.

## Tests without overwhelming a development machine

During implementation, run exactly the smallest relevant test file. Naming
files after the mode does that, and builds the package first so the run does
not recompile `src/mojotrees` from source:

```sh
tools/run_tests.sh cpu test_sparse
tools/run_tests.sh cpu test_callbacks test_binning
tools/run_tests.sh gpu test_gpu_training
python3 tools/check_parity.py
git diff --check
```

The package build is what makes this cheap, so it is worth keeping. Once
`build/mojotrees.mojopkg` is current, a single file runs against it directly:

```sh
mojo run -I build -I tests tests/test_sparse.mojo
```

`-I tests` is not optional: the shared data generators live in
`tests/support.mojo`, which every suite imports rather than carrying its own
copy of `_splitmix64` and `_uniform`.

Do not run `pixi run test`, all Python suites, wheel tests, benchmarks, or
retry loops as part of every edit. Do not start polling, watch, or background
test loops. Broad suites belong in CI or an explicitly coordinated integration
pass after focused tests succeed. `tools/run_tests.sh` runs its files
concurrently, which is why the default job count is two fewer than the CPU
count; `MOJOTREES_TEST_JOBS` moves it, and `tools/with_build_lock.sh` is the
separate lock that serializes a whole run against other sessions in the same
checkout.

Adding a test file is the whole wiring step. The runner discovers
`tests/test_*.mojo` by glob, so a new suite is in `pixi run test` by
existing. It did not used to be: `pixi run test` was sixty `mojo run`
commands chained with `&&`, and `tests/test_gpu_split_policy.mojo` sat in
the tree passing and named by no task at all.

When Python bindings change, build once and run the narrowest relevant Python
test. Avoid multiple concurrent extension builds; they compete for the same
artifacts.

## Pull requests

Describe:

- the exact behavior added or fixed;
- intentional differences from LightGBM;
- files and public contracts changed;
- the focused commands run and their results;
- checks not run and why;
- CPU/GPU coverage and unsupported paths;
- remaining risks.

Update code, tests, Python bindings, documentation, serialization, and the
parity matrix together whenever a public contract changes.

Never claim performance, correctness, portability, or feature parity without
a reproducible command and recorded environment. An explicit unsupported error
is better than silent fallback or a placeholder implementation.

## Useful contribution areas

- Apple GPU profiling and active-row compaction
- device-side split selection and objectives
- M1 through M5 validation reports
- NVIDIA and AMD correctness validation on real hardware
- differential tests against LightGBM
- packaging and clean-install wheel testing
- sparse, categorical, ranking, and missing-value edge cases
- API and serialization stability

Use the hardware validation issue template when contributing accelerator
results.

## Contributor access

The public repository accepts pull requests from anyone without an invitation.
Sustained contributors may be promoted automatically through Triage, Write,
and Maintain according to the public, auditable policy in
[docs/AUTOMATED_ACCESS.md](docs/AUTOMATED_ACCESS.md). The policy uses merged
pull requests over time and review/reliability work, not raw commit or line
counts. Admin access is always granted explicitly by an existing Admin.
