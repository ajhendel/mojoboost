# Architecture

Written: 2026-08-14

What this repository is shaped like, where each decision is made, and which
paths a user can actually reach. It is a map of the *call graph*, not of the
directory listing: a file existing in `src/mojotrees/` says nothing about
whether anything runs it, and this document is careful never to imply that
it does.

Three companion files carry the parts that move:

- `docs/CAPABILITY_LEVELS.md` defines the seven words this file and the
  parity contract both use for how far a capability has got.
- `docs/INTEGRATION_INVENTORY.md` is the current, tool-checked answer to
  "what is reachable today", including the list of modules that are not.
- `docs/LIGHTGBM_PARITY.md` scores capabilities against LightGBM.

## The one rule

**Mojo decides; Python asks and formats.**

Training, prediction, objectives, metrics, the data representation, trees,
serialization, and device policy are decided in `src/mojotrees/`. The
Python package validates its inputs, marshals buffers, calls the extension
module, and renders what comes back. A rule that exists in
`python/mojotrees/` and not in `src/mojotrees/` is a rule the Mojo API, the
C ABI, and the command line tool do not have, which means the same
parameters produce different behavior depending on which door the caller
came through.

The rule has exceptions today. They are not design; they are unfinished
work, and each one is listed in `docs/INTEGRATION_INVENTORY.md` under
"Policy that exists twice" with the native module that is meant to win.

## Entry points

Five roots. Everything reachable is reachable from one of these, and
everything else in `src/mojotrees/` is written but not connected.

| Root | Surface | Public per `docs/COMPATIBILITY_POLICY.md` |
|---|---|---|
| `python/mojotrees/__init__.py` | the Python package | the names in `__all__` |
| `bindings/_mojotrees.mojo` | the CPython extension module | not itself public; its `def_function` table is the entire Python-to-native surface |
| `src/mojotrees/__init__.mojo` | the native Mojo package | the names it re-exports |
| `capi/mojotrees_capi.mojo` | the C ABI | the declarations in `capi/mojotrees.h` |
| `cli/mojotrees_cli.mojo` | the command line tool | its arguments and exit statuses |

The three native roots are peers. `capi/` and `cli/` do not go through the
Python package, so a capability that only Python knows about is a
capability the C ABI and the CLI silently lack.

## The layers

Bottom to top, within `src/mojotrees/`. A layer may call downward and
sideways within itself, never upward.

**Data.** `binning` (quantile bin edges and the `uint8` binned matrix),
`categorical` (category specs and 256-bit split sets), `sparse` (CSC and
CSR), `raw_data`, `trainset` (the `Dataset` object the functional API
binds).

**Arithmetic.** `gain` (the second-order split gain and leaf score),
`histogram` (SIMD accumulation and sibling subtraction), `split` (best
split search), `metrics`, `objective` (gradients, hessians, links, and the
custom-objective entry point), `monotone`, `interaction`,
`tree_parameters_extra`.

**Growth.** `tree` (leaf-wise growth), `tree_sparse`, `boosting` (the round
loop, objective codes, leaf renewal), `boosting_sparse`, `bagging`, `goss`,
`sampling`, `class_weight`.

**Model.** `model` and `model_sparse` (`Model`, `MulticlassModel`, and the
`fit*` family), `ranking`, `contrib` (TreeSHAP), `importance`, `serialize`
(the versioned text format), `params` (the LightGBM-style parameter
string), `inspection` and `model_dump` (the structured dump).

**Device.** `device_policy` (the authoritative decision engine), `device`
(a thin facade over it), `apple_gpu_policy` (device profiles and the Apple
tuning derivations), `unified_memory_policy` (transfer routes),
`initialization` (session warmup phases).

**Accelerator.** `gpu_tiling` (launch geometry from reported device
capabilities), `gpu_runtime` (session, residency ledger, staging),
`histogram_gpu`, `gpu_active_rows`, `gpu_objectives_native`,
`gpu_split_search`, `gpu_predict`, `train_gpu`, and a set of kernels and
planners that are written but not yet reached from a trainer.

**Distributed.** `collective` (the all-reduce contract), `distributed`
(row-partitioned training), `distributed_transport`.

Cycles are avoided by copying constants rather than importing them.
`device_policy` mirrors a handful of codes from `ranking` and
`histogram_gpu` for exactly this reason, with a pinning test named in its
own docstring; a mirror that drifts is a defect, and the mirror is always
the copy, never the source.

## The four seams

A seam is where one language, ABI, or process boundary hands work to
another. Every seam is narrow on purpose, and every seam is a place a
capability can be implemented on one side and unreachable from the other.

### 1. Python to native

`bindings/_mojotrees.mojo` builds one `PythonModuleBuilder` and registers
one flat table of functions plus three types (`Model`, `MulticlassModel`,
`Dataset`). Buffers cross as an integer address plus a length, never as a
Python object the native side has to understand. Errors cross as raised
Mojo errors and are re-raised as `ValueError` or `RuntimeError`.

**The table is the surface.** A native function that is not in the
`def_function` table cannot be called from Python, no matter how public it
is in Mojo. The Python package handles that gracefully in several places:
it probes for a hook with `getattr(_mojotrees, name, None)` and falls back
to a slower path when the hook is absent. A probe is not the same thing as
a disconnection, and the difference is which extension you built. Most of
these hooks are registered as of this revision: `_mojotrees.mojo` imports
the inspection, objective-registry, dataset, and distributed binding
modules, so a package built from this tree takes the native route and the
probe is what keeps it working against an extension built before the
change. One hook still finds nothing on any build, `split_gains`, and it
is a missing implementation rather than an unregistered one.
`docs/INTEGRATION_INVENTORY.md` says which is which, one row at a time.

### 2. Native to C

`capi/mojotrees_capi.mojo` exports C-ABI functions declared in
`capi/mojotrees.h`, over opaque handles, an error object, and a documented
ownership table (`capi/README.md`). It is mojotrees's own interface and is
deliberately not source compatible with LightGBM's `c_api.h`.

### 3. Native to the command line

`cli/mojotrees_cli.mojo` reads CSV and the same LightGBM-style parameter
string the C ABI takes, through `params.mojo`. It reads no configuration
file.

### 4. Host to device

`gpu_tiling` turns reported device capabilities into launch geometry, and
every kernel launch derives its geometry there rather than hard-coding it.
Device selection is decided before any of that, by `device_policy`. A
`gpu` request that the GPU path cannot cover raises; it never falls back
silently, because a silent fallback turns "my GPU run" into "a CPU run that
took the same wall clock and I never knew".

The division of labor across this seam is fixed, and it is not "GPU first,
CPU as a fallback." **The GPU owns the data plane. The CPU owns the control
plane, small data, and verification.**

| Owner | What | Why it lives there |
|---|---|---|
| GPU | binned matrix, gradients and hessians, active-row permutation and leaf ranges, histogram accumulation, stable row partitioning, native objective evaluation and score advancement, device split search when selected | every one of these scales with `n_rows`, and none of it crosses the boundary during a fit (`train_gpu.mojo`, `gpu_active_rows.mojo`, `gpu_objectives_native.mojo`, `gpu_frontier.mojo`) |
| CPU | boosting coordination, split selection over histograms of `n_features x n_bins` cells, the tree model, leaf-value renewal, prediction, host row sampling for bagging and GOSS | latency-bound scalar work over data that does not scale with `n_rows`; the host scan of a 50 x 255 histogram costs microseconds and a device scan of it costs a launch and a synchronization |
| CPU | the entire fit below the launch-cost crossover (`device_policy.mojo`) | a kernel launch plus a synchronization is a fixed cost per node, and below the crossover the whole fit is cheaper where no launch is paid. Per-leaf hybrid placement above the crossover was tried and deleted on 2026-08-16; see the note below |
| CPU | the reference implementation the GPU path is verified against | the host fixed-point replica (`histogram.build_histogram_subset_replica_into`) has been shown bit-identical to the device build, and the CPU trainer is the oracle every GPU test compares to |

**Per-leaf hybrid placement, and why it is gone.** Until 2026-08-16 a hybrid
CPU/GPU leaf scheduler (`hybrid_leaf_scheduler.mojo`, double opt-in behind
`MOJOTREES_HYBRID_LEAVES` and `MOJOTREES_HYBRID_COSTS=apple-m4`) elected
individual small leaves onto the host, on the premise that the host could
usefully take the leaves the device was slow at. It measured 1.20x on a
bagged 20,000-row fit, and it only ever reached host-gradient runs. The
premise is gone: the device-resident tree plane now beats the host path at
every shape measured, all resolved under rule M0 — 2.2x at 50,000 rows, 44
percent at 250,000, 24 percent at 1,000,000
(`bench/results/session3_2026-08-16/RESULTS.md`). What survives the deletion
is the host replica builder itself, which was never the scheduler's: it is
the oracle in the row above. The design record is
`docs/design/HYBRID_TRAINING.md` and the calibration is
`bench/results/apple_m4_hybrid_costs_2026-08-15.md`.

Two things follow. First, the CPU trainer is a permanent part of the design,
not a compatibility layer to be retired once the GPU path is complete; it is
where the small end of the workload runs and where correctness is
established. Second, what remains host-side in the data plane is a short,
explicit list rather than a legacy: row sampling under bagging and GOSS,
which draws its ranked sample on the host and uploads gradients per round;
validation scoring, which walks the tree on the host; and the binning pass
itself. Each is named in `train_gpu.mojo`'s module docstring with the switch
that selects it.

The end-to-end measurement of the split as it stands is
`bench/results/profile_2026-08-15/RESULTS.md`. On an Apple M4 at 100 rounds,
31 leaves, 255 bins, squared error, the GPU is **1.85x** the CPU trainer at
1,000,000 x 50 (3.58s against 6.98s), loses to it at 250,000 (1.89 against
1.66) and at 50,000 (1.63 against 0.564), and wins multiclass by 1.63x
(15.30 against 25.47 at 465,000 x 54 over 7 classes).
`bench/results/apple_m4_large_scaling_2026-08-14.md` recorded 2.6x at one
million and 3.3x at five million at half the resident memory; that ratio has
since fallen to 1.85x because the CPU trainer got 1.63x faster in the round
that followed, not because the device got slower.

`bench/results/sweep2_2026-08-15/RESULTS.md` extends that to 2,000,000 rows
and fits the two libraries' cost per row against each other. The finding that
bears on this division of labor is that our GPU's marginal cost per row and
LightGBM's on ten CPU cores are even, 2.33 to 2.46 microseconds across four
fitted segments, and the remaining deficit is roughly one second of fixed
cost derived from the intercepts. A GPU whose per-row cost equals a laptop
CPU's is not yet earning its data plane, so the control plane is a necessary
fix and not a sufficient one: removing all of the fixed cost is **estimated**
to reach parity at 1,000,000 rows and stop there.

Where the device's time goes is measured too, and it is the mechanism behind
the third row of the table above. The Metal timeline in
`docs/METAL_TIMELINE.md` finds the GPU idle for 76.5% of a training span at
200,000 rows at the device's Maximum clock, with the host blocking on 94.1%
of blits and on 2 of 18,701 compute kernels. That is 32.1 serialization
points per round at 606 microseconds each, and compute of every kind is
22.9% of a round. The control plane's per-split host round trip is therefore
the dominant cost of a GPU fit today, which is a cost of this division of
labor rather than an argument against it, and it is what a device-owned tree
would remove.

## Where each policy lives

One authoritative implementation per question. Where a second one exists
today, the table names it and says which is meant to win.

| Question | Authoritative | Second implementation, if any |
|---|---|---|
| Which backend runs this training job | `src/mojotrees/device_policy.mojo` | `python/mojotrees/device_selection.py` is a formatter over the native decision, and degrades to a narrower report when the full binding is absent |
| What launch geometry a kernel gets | `src/mojotrees/gpu_tiling.mojo` | `src/mojotrees/apple_gpu_policy.mojo` derives Apple-shaped geometry, consulted only by the opt-in histogram specializations |
| What an objective or metric is called, and what it accepts | `src/mojotrees/objective_registry.mojo` | `python/mojotrees/_eval.py` carries mirror tables it uses whenever the registry is not bound |
| How class weights become row weights | `src/mojotrees/class_weight.mojo` | `python/mojotrees/__init__.py` computes them in Python for the estimators |
| What a model dump contains | `src/mojotrees/inspection.mojo` and `model_dump.mojo` | `python/mojotrees/inspection.py` parses `Booster.model_to_string()` |
| What the model file contains | `src/mojotrees/serialize.mojo` | none |
| What a parameter string means | `src/mojotrees/params.mojo` | none |

Every "second implementation" in that table is a disconnection, not a
design, and they are in two different states. For the dump, the registry,
and the device report the binding has landed, so the second implementation
is now a compatibility path rather than the path, and the risk has moved
with it: keeping a fallback after its binding lands is how two answers to
one question start, and the deletion is what is owed next. Two things keep
them alive for now. `python/mojotrees/inspection.py` still parses for split
gains, because no `split_gains` hook exists to ask, and
`_Base._resolve_device` still calls `resolve_device` directly rather than
the fuller `decide_device` the same extension now registers, so a report
and a `fit` remain two decisions. `class_weight` is the one row where no
binding is even open: the Mojo module has no caller at all, and the Python
arithmetic is what every estimator runs.

## Reachability, and how to check it

"Implemented" and "reachable" are different claims, and the gap between
them is wide enough in this repository to need a tool rather than a
reading. Two scripts check it, neither of which imports or builds the
package:

- `tools/connectivity_audit.py` computes the import graph from the five
  roots and reports orphan modules, imports that are never used, duplicate
  registries, public parameters with no consumer, binding functions with no
  Python caller, and native functions Python reaches for that the binding
  does not export. It is the authoritative graph engine.
- `tools/audit_integration.py` checks `docs/INTEGRATION_INVENTORY.md`
  against that graph, so the written inventory cannot drift away from the
  tree without failing.
- `tools/check_parity.py` checks `docs/LIGHTGBM_PARITY.md` and
  `docs/CAPABILITY_LEVELS.md`. It owns the claims; the other two own the
  graph. Two parity checkers would be exactly the duplication all three
  exist to find.

A module appearing under `src/mojotrees/` is never evidence of anything.
The evidence is a chain of imports from a root, plus a call site, plus a
test, and the three are independent.
