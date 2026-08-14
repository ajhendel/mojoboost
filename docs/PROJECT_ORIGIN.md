# Project origin and stewardship

## Why mojoboost exists

Andrew Hendel created mojoboost to explore a concrete systems question:

> Can a modern, Mojo-native gradient-boosted-tree implementation provide a
> familiar Python experience while using the CPU and GPU already present in
> Apple Silicon Macs?

That question shaped the project more strongly than feature-count parity. The
core implementation belongs in Mojo; Python should provide familiar estimator
ergonomics rather than becoming a second training engine. CPU execution must
remain a dependable fallback. Explicit GPU requests must not silently execute
on the CPU. Performance claims must describe the complete user operation and
must survive comparison with mature, optimized libraries.

## Direction established by the creator

The project's defining choices include:

- histogram-based, leaf-wise tree growth in the LightGBM family;
- a LightGBM-familiar API where compatibility benefits users;
- intentional rejection of deprecated or file-configuration compatibility
  that would add maintenance without improving the Python library;
- Mojo ownership of training, prediction, objectives, metrics, model state,
  serialization, and accelerator policy;
- CPU and portable accelerator foundations alongside first-class Apple Silicon
  work;
- explicit capability levels separating code existence from integration,
  testing, hardware evidence, and release packaging;
- no silent device fallback for an explicit GPU request;
- reproducible evidence before correctness, parity, or performance claims;
- an Apache-2.0 public project with low-friction contribution.

Andrew Hendel originated this direction and remains responsible for deciding
how the project evolves during its alpha period. Governance and contributor
rights are documented separately in [GOVERNANCE.md](../GOVERNANCE.md) and
[CONTRIBUTING.md](../CONTRIBUTING.md).

## AI-assisted engineering

The repository has been built with extensive AI-assisted engineering. Andrew
has used coding agents to investigate APIs, implement bounded workstreams,
draft documentation, and accelerate parallel development. That method is part
of the project's history and should be described directly rather than hidden.

The use of agents does not turn unexecuted code into evidence. It makes
integration discipline more important: duplicate implementations must be
fused, public paths must be traced into native calls, model state must survive
serialization, and focused validation must establish what actually works.
Andrew is accountable for coordinating that process and for every capability
the project ultimately advertises or releases.

## What recognition should attach to

The project should be evaluated on more than its size or development speed.
The meaningful contribution is the creation and stewardship of a coherent ML
systems project spanning:

- tree-learning algorithms and objectives;
- native data structures and serialization;
- Python, Mojo, C, and command-line interfaces;
- CPU, Apple GPU, and portable accelerator architecture;
- packaging and compatibility contracts;
- transparent capability and benchmark evidence.

If mojoboost becomes useful, credit belongs both to its creator and to the
contributors whose work makes it reliable. Git history records individual
changes; [AUTHORS.md](../AUTHORS.md) records project-level attribution.

## Current status

mojoboost remains an experimental alpha. Broad implementation work and an
ambitious architecture are not substitutes for clean installation, focused
correctness evidence, reproducible performance results, or operating history.
The repository deliberately records those distinctions in
[CAPABILITY_LEVELS.md](CAPABILITY_LEVELS.md),
[LIGHTGBM_PARITY.md](LIGHTGBM_PARITY.md), and
[GPU_VALIDATION.md](GPU_VALIDATION.md).
