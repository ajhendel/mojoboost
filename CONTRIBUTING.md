# Contributing to mojoboost

mojoboost is an experimental public alpha. Contributions that improve
correctness, integration, portability, documentation, and reproducible
performance evidence are welcome.

## Before changing code

1. Read `docs/LIGHTGBM_PARITY.md` for the public behavior contract.
2. Read `docs/GPU_VALIDATION.md` before making accelerator claims.
3. Check open issues and current code before adding a second implementation.
4. Keep the Python layer thin. Training logic belongs in Mojo.
5. Preserve unrelated work in the tree.

## Tests without overwhelming a development machine

During implementation, run exactly the smallest relevant test file. Examples:

```sh
mojo run -I src tests/test_sparse.mojo
mojo run -I src tests/test_callbacks.mojo
mojo run -I src tests/test_gpu_training.mojo
python3 tools/check_parity.py
git diff --check
```

Do not run `pixi run test`, all Python suites, wheel tests, benchmarks, or
retry loops as part of every edit. Do not start polling, watch, or background
test loops. Broad suites belong in CI or an explicitly coordinated integration
pass after focused tests succeed.

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
