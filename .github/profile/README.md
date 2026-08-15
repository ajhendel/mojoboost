<!--
STAGED COPY. This file is the GitHub organization profile for mojoboost-ml.

GitHub renders an organization profile from a repository named `.github`
owned by the organization, at the path `profile/README.md`. It is NOT
rendered from this path inside the mojoboost repository, where nothing
displays it.

To publish it, create `mojoboost-ml/.github` (public) and copy this file to
`profile/README.md` there. Keep this staged copy as the source of truth so the
profile text is reviewable alongside the project it describes.
-->

# mojoboost

**Gradient-boosted decision trees, written natively in [Mojo](https://www.modular.com/mojo), with first-class support for the GPU already inside an Apple Silicon Mac.**

[**mojoboost**](https://github.com/mojoboost-ml/mojoboost) is the project. It
is a from-scratch library in the LightGBM family, using histogram-based split
finding and leaf-wise tree growth, with a familiar Python API on top and
histogram accumulation that runs on the CPU or on the GPU from the same source.

> **Experimental public alpha.** The feature surface is broad and training
> works end to end. That is not the same as production parity with LightGBM or
> XGBoost. Feature combinations, edge cases, packaging, and hardware beyond
> Apple Silicon are less mature, and the repository says exactly where.

## The bet

LightGBM and XGBoost are excellent, mature C++ libraries, and reimplementing
them is only interesting if the language buys something. The hot loop of
gradient boosting is histogram accumulation over binned features, which is
precisely what Mojo is built for. Explicit SIMD at the chip's vector width,
compile-time specialization of the training loop on bin count and dtype with no
virtual dispatch, direct control of memory layout for cache tiling, structured
parallelism, and GPU targeting from the same source in the same language.

The claim being tested is that this produces a substantially simpler codebase
at competitive speed. It is not yet settled, and the repository does not
pretend otherwise.

## How this project talks about itself

Three rules, and they apply to maintainers exactly as they apply to a
first-time contributor.

- **No claim without a command.** Performance, parity, portability, and
  correctness claims ship with a reproducible command and a recorded
  environment, or they do not ship. `docs/GPU_VALIDATION.md` is the procedure.
- **The parity contract is authoritative.** `docs/LIGHTGBM_PARITY.md` states
  what is supported and which differences from LightGBM are deliberate. Where
  it and any other document disagree, it wins.
- **Fail loudly.** An explicit unsupported error beats a silent fallback.
  Asking for `device="gpu"` on hardware that cannot deliver it raises rather
  than quietly training on the CPU and reporting a number you would misread.

Apple Silicon correctness has been exercised on an M4. NVIDIA and AMD have not
been validated on real hardware, which is stated plainly in the repository and
is one of the most useful things a contributor could change.

## Contributing

The alpha is deliberately low friction. **No contribution waits on a human
approval.** Contributors with Write access push directly, and a pull request
from anyone else merges automatically once continuous integration is green,
with the single exception of changes to workflows, dependencies, packaging, and
release tooling, which a maintainer reads first. Write access is the normal
outcome of contributing a few substantive changes, and asking for it is the
expected way to get it.

Areas where help is most wanted right now.

| Area | What it looks like |
| --- | --- |
| NVIDIA and AMD validation | Running a capture script on hardware the maintainers do not own, and filing the record. `docs/HARDWARE_CONTRIBUTORS.md` is written for people who do not want to learn Mojo |
| Apple GPU performance | Profiling Metal kernels, order-preserving active-row compaction, split search on device |
| Differential testing | Comparing against LightGBM on real datasets and turning the gaps into issues |
| Packaging | Clean-install wheel testing across platforms |
| Edge cases | Sparse, categorical, ranking, and missing-value behavior |

Start with `CONTRIBUTING.md` for how to make a change, `GOVERNANCE.md` for how
decisions and access work, and `SUPPORT.md` for where to ask a question.

## What this organization is

A home for the repository, so that more than one person can hold
administrative access and the project outlives any single account.

It is not a company. There is no legal entity, no funding, nobody employed, and
nobody on call. Everything here is Apache-2.0 and provided as-is, and every
response time is an intention rather than a commitment. If that is the whole
truth of a project, it is better said on the front page than discovered later.
