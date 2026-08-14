# Building mojoboost: gradient-boosted trees for the GPU inside a Mac

Status: publication draft. Statements about architecture describe project
intent and implementation. Statements about performance require results from
the repository's benchmark protocol before publication.

By Andrew Hendel, creator and lead maintainer of mojoboost

## The premise

Gradient-boosted decision trees remain one of the strongest general-purpose
tools for structured data. LightGBM and XGBoost demonstrate what mature,
highly optimized implementations can achieve. They also establish a demanding
baseline: a new library needs more than familiar algorithms and a large API.
It needs a reason to exist.

The premise behind mojoboost is that the accelerator already present in every
Apple Silicon Mac could become that reason. A developer should be able to
write familiar Python, request an accelerator explicitly, and know whether the
GPU actually ran. Small workloads should retain a fast CPU path. The same core
architecture should leave room for CUDA and other backends rather than making
Metal support a disconnected fork.

That is the product idea. The engineering problem is harder.

## Why tree training is not ordinary GPU work

A dense neural-network layer exposes a large, regular operation. Tree growth
repeatedly partitions rows, constructs histograms for candidate leaves, scans
split gains, and chooses an irregular next action. A naive GPU implementation
can spend more time allocating buffers, synchronizing with the host, and
examining irrelevant rows than doing useful arithmetic.

The original GPU histogram path in mojoboost exposed the central problem: if
every leaf scans every row and filters by leaf identifier, deeper trees waste
increasing amounts of bandwidth. Returning every candidate histogram to the
CPU for split selection adds another synchronization point.

The architecture therefore targets a device-oriented flow:

1. Bin feature values once and keep the prepared matrix resident.
2. Maintain compact active-row ranges for each leaf.
3. Partition a leaf's rows into compact child ranges on the device.
4. Batch compatible leaves into fewer histogram launches.
5. Reduce histograms and scan split gains on the device when numerical policy
   permits it.
6. Keep predictions, gradients, Hessians, and validation data resident across
   boosting rounds.
7. Reuse contexts, kernels, queues, and buffers across an estimator's fit.

Each stage needs a conservative fallback because architectural elegance does
not establish equivalence with the CPU model.

## Mojo as the implementation boundary

mojoboost follows one architectural rule: Mojo decides; Python asks and
formats.

Tree growth, objectives, metrics, prediction, model state, serialization, and
device selection live in the native implementation. The Python layer provides
estimators, ecosystem-container handling, callbacks, and scikit-learn-style
ergonomics. This avoids maintaining two subtly different definitions of an
objective or parameter depending on which API a user enters through.

The rule also creates a useful diagnostic: when Python contains a complete
policy that native Mojo does not expose, the project has an integration gap.
The repository tracks those gaps explicitly rather than treating the presence
of a source file as proof of a feature.

## CPU and GPU are different schedules, not different products

The CPU path favors feature-oriented loops, contiguous SIMD accumulation,
cache-aware histograms, and bounded multicore scheduling. The GPU path favors
large batches of active rows and leaves, coalesced packed-bin reads, local
histogram reduction, and limited synchronization.

The schedules differ because the hardware differs. The model semantics should
not. Both paths must agree on missing directions, categorical sets, sample and
feature selection, regularization, constraints, tree order, and serialized
state. An explicit GPU request should fail when that agreement cannot be
provided; silently switching to CPU makes performance debugging impossible.

## Apple Silicon and unified memory

Unified physical memory is promising, but the phrase "zero copy" can hide
page migration, synchronization, mapping, and contention costs. mojoboost's
policy work keeps explicit-copy and shared or mapped strategies distinct. The
useful question is empirical: for a particular matrix and boosting workload,
which representation minimizes complete-operation time and peak memory?

The benchmark must include binning, context and kernel initialization,
allocation, transfer or mapping, training, validation, prediction, memory, and
model quality. Kernel-only timing cannot answer whether a Python user should
select the GPU.

## A familiar API with visible device decisions

The intended experience is deliberately ordinary:

```python
from mojoboost import MojoBoostClassifier

model = MojoBoostClassifier(device="auto")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=20,
)

print(model.device_)
```

The unusual part should be transparency. Automatic selection should be based
on versioned crossover evidence and should explain its decision. An explicit
GPU request should either use the GPU or raise a clear error. Until measured
crossover rules exist, choosing the CPU automatically is more honest than
guessing.

## AI-assisted development and integration debt

I have built mojoboost with extensive assistance from coding agents. They make
it possible to investigate and implement many bounded areas quickly. They also
make it easy to produce multiple plausible implementations that are not
connected to the same production path.

The remedy is not to conceal the method. It is to make reachability and
evidence first-class. The project records whether a capability is implemented,
integrated, public, focused-tested, differential-tested, hardware-validated,
and release-packaged. Static tooling traces entry points through bindings to
native modules. Integration rounds fuse duplicate registries and policies
before more features are added.

This approach treats orchestration, architecture, review, and accountability
as engineering work rather than equating authorship with keystrokes.

## What would constitute success

The near-term goal is not to claim universal replacement of mature libraries.
It is to establish a compelling, reproducible local workflow:

- a clean `pip install mojoboost` on a supported Mac;
- familiar regression, classification, ranking, validation, inspection, and
  serialization behavior;
- correct CPU and GPU results on representative tabular workloads;
- a measured CPU/GPU crossover rather than a hard-coded preference;
- an end-to-end Apple result compared fairly with optimized LightGBM;
- raw benchmark data and environment metadata anyone can inspect.

If that result is strong, mojoboost can offer something concrete: native
gradient-boosted trees accelerated by the GPU already inside a Mac. If it is
not strong, publishing the architecture and measurements still provides a
useful account of where irregular tree learning benefits from—or fights—the
hardware.

## Follow the evidence

The live project is available at
[github.com/ajhendel/mojoboost](https://github.com/ajhendel/mojoboost).
Architecture, parity, hardware records, and benchmark methodology live in the
repository. The software is Apache-2.0 and experimental; contributions and
independent hardware results are welcome.
