<!--
No human approval is required to merge ordinary alpha contributions. If
continuous integration is green and this pull request touches no reserved
path, it merges automatically. See GOVERNANCE.md for reserved paths and the
rest of the merge policy.

This template is a prompt, not a form. Delete what does not apply. An
unfinished template never blocks a merge.
-->

## What this changes

<!-- The exact behavior added or fixed. One or two sentences is usually right. -->

## How it was checked

<!--
The focused commands you ran and what they printed. CONTRIBUTING.md explains
why this is the smallest relevant test file rather than the full suite.

  mojo run -I src tests/test_<area>.mojo
  python3 tools/check_parity.py

Also worth saying: what you did not run and why, and whether this was
exercised on CPU, GPU, or both. "Untested on GPU, no Apple hardware" is a
useful sentence, not an admission.
-->

## Claims

<!--
Skip this unless the change asserts something about performance, parity,
portability, or correctness. If it does, the claim needs a reproducible
command and the recorded environment (chip, OS, `pixi run mojo --version`).
An explicit unsupported error is better than a silent fallback.
-->

## Public contract

<!--
Delete if nothing public moved. If it did, confirm that code, tests, Python
bindings, docs, serialization, and docs/LIGHTGBM_PARITY.md moved together,
and name any intentional difference from LightGBM.
-->

## Anything unresolved

<!-- Known gaps, risks, follow-up work, or a question you want answered. -->
