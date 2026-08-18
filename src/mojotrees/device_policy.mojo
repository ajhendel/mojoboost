"""The device-policy contract: one authoritative decision engine in Mojo.

Every question "should this training run go to the accelerator, and if not,
why not" is answered here and nowhere else. Before this module the answer
existed twice: once in `device.mojo` (the vocabulary, the availability
probe, and a size heuristic) and once again, larger and with a different
set of rules, in `python/mojotrees/device_selection.py`. Two engines that
answer the same question will disagree eventually, and the disagreement is
invisible to the user until a run lands on a backend they did not expect.

So this module owns all of it:

- the device vocabulary (`cpu`, `gpu`, `auto`) and its codes,
- what the GPU training path supports (objectives, outputs, row and bin
  limits),
- the memory estimate for one GPU training session,
- how detected hardware capabilities are turned into a plan,
- the crossover evidence table and the rule that reads it,
- the fallback semantics for unknown hardware,
- the refusal an explicit `gpu` request gets when it cannot run,
- which transfer route each device buffer is on, by carrying the plan
  unified_memory_policy.mojo resolves,
- how much of the one-time startup cost the process has already paid, by
  carrying the `SessionState` initialization.mojo defines.

The last two are carried, not decided. A route is chosen by
unified_memory_policy.mojo against its own evidence ladder and a session
state is observed by initialization.mojo; this module folds both into the
decision so that one report answers "where does this run, on what hardware,
moving bytes how, having already paid what". Neither one may select a
backend: a transfer route and a warm session are performance facts, and
`crossover_rules()` is the only place a performance fact is allowed to
change a device.

`device.mojo` is a thin compatibility facade over this module.
`python/mojotrees/device_selection.py` is reduced to extracting plain
workload metadata from `X` and `y` and formatting the decision this module
returns. Neither one decides anything.

Callers that are not Mojo
-------------------------
`decide_device` takes structs, which no CPython binding, C API, or CLI can
construct. `decide_device_report` and `decide_device_report_reported` are
the flat forms of the same call: plain scalars in, the serialized decision
out. They are marshallers and nothing else, and they exist so that each of
those callers does not grow its own. `resolve_device_full` is the raising
form for a trainer that knows its whole workload, as against `resolve_device`,
which knows only the shape and says so.

The contract
------------
`DeviceRequest` is the serializable input: the requested device, the shape
(rows, features, outputs, bins), the objective code, and the flags that
change what the GPU path can do (sparse input, categorical features,
missing values, validation use). `DeviceCapabilities` is the other input:
what the build and the device in front of us can do, either detected here
or injected by a caller that already has a `DeviceContext` open.

`DeviceDecision` is the serializable output: the selected backend, the
ordered blocking reasons, the warnings, the memory estimate, the policy
version, and the identifier of the evidence a GPU selection rests on.
`DeviceDecision.serialize()` renders it as `key=value` lines, which is the
form the bindings hand to Python: Python parses those lines and formats
them, it does not recompute them.

Three requested devices, and what each one means
------------------------------------------------
- `cpu`: the dependable path. Float64 throughout, every objective, every
  entry point. It always resolves to itself.
- `gpu`: an explicit request. It runs on the accelerator or it raises.
  There is no silent fallback, because a fallback turns "my GPU run" into
  "a CPU run that took the same wall clock and I never knew".
- `auto`: the GPU only when the GPU path covers the workload and evidence
  says the GPU is the faster choice for that shape on that device. With no
  such evidence it chooses the CPU and says so.

Where `auto` reaches the GPU, and where it still does not
---------------------------------------------------------
`crossover_rules()` holds two rules, one single-output and one multiclass,
and each is as narrow as the records behind it.

The single-output rule's two records both measure end-to-end training on an
Apple M4 over Metal at 1,000,000 rows by 50 dense features, 255 bins, 31
leaves, 100 rounds, squared error, in interleaved CPU/GPU arms:
`bench/results/apple_m4_large_scaling_2026-08-14.md` (GPU 4.289-4.382 s
against CPU 11.094-11.706 s over three seeds) and the 2026-08-15 section of
`docs/GPU_VALIDATION.md` (GPU 4.10 s against CPU 11.36 s over five
repeats). So it fires on that device, for that objective, at 50 or more
features, and nowhere else.

The multiclass rule's one record is softmax on the same machine at 465,000
rows by 54 dense features over 7 classes, same bins, leaves and rounds:
`bench/results/profile_2026-08-15/RESULTS.md`, GPU 15.30 s against CPU 25.47
s, medians of three at 0.1 and 7.7 percent spread, so the GPU wins by 1.63x
with the two spreads nowhere near touching. It is scoped by trees per round
rather than by objective code, because trees-per-round is what the trainers
branch on and is the one fact every multiclass entry point declares; see
`crossover_rules()` for why that is the exact scope and not a weakened one.
Until 2026-08-16 there was no rule here at all, so `auto` handed every
softmax fit the slower backend at every size.

Every other backend, every other Apple generation, and every other objective
still return "no rule covered this", because nothing here measured them.

The *row floor* is a different kind of thing and is labelled as one. Both
rules read `AUTO_GPU_MIN_ROWS`, a plain provisional constant at 250,000
rows, set below the smallest shape either record covers, on a stated trade
rather than on a measurement. Read the comment on that constant before
changing it; the measured crossover is scheduled work and this is the
placeholder standing in for it. One number for both is itself a decision and
`crossover_rules()` argues it: a K-class fit does K times the tree work per
round against the same one-time upload and session cost, so the multiclass
crossover sits at or below the single-output one and reusing the constant
cannot over-reach relative to that.

That the table was empty for longer than the evidence warranted was a bug,
not conservatism: `auto` selected the CPU at every size on every machine,
so the GPU trainer was reachable only by asking for it by name. What has
not changed is the standard. A rule is a benchmarking result, not a code
change: it carries its `evidence_id` and its `measured_on`, and
`POLICY_VERSION` is bumped with it.

How the rule became reachable, 2026-08-16, and what still is not
--------------------------------------------------------------
The rule above was installed and then could not fire, anywhere, ever. Three
independent things stopped it, each sufficient on its own, and all three had
to be found before any of them was worth fixing:

1. `DeviceCapabilities.detect` returned `GpuProfile.generic()`, whose `api`
   is `API_UNKNOWN`. `CrossoverEvidence.matches` tests the API first, so the
   rule was declined before the shape was ever compared. Fixed by
   `PROFILE_BUILD_TARGET`: identity now comes from the comptime
   `_accelerator_arch()`, which opens no device and so does not drag
   `max.gpu.host` into `params.mojo`'s import graph. Numbers are still the
   portable fallback.
2. `parse_apple_generation` could not parse `4-metal4`, which is the string a
   Metal device actually reports. "metal4" ends in "l4", not "m4". So even
   `capabilities_from_reported` and `decide_device_report_reported`, the two
   entry points built for a caller holding an open `DeviceContext`, produced
   `APPLE_GEN_UNKNOWN` from a real M4 reading, and the generation scope
   declined the rule a second time. Fixed in apple_gpu_policy.mojo. The
   reason nobody noticed is worth keeping: `apple_m4_observed()` builds its
   profile with the fieldwise constructor and hands `APPLE_GEN_M4` in
   directly, so every test asserting that the rule fires was asserting it
   against a profile no detection path in this repository could construct.
3. `resolve_device`, which is what all six Mojo trainer entry points call,
   declares no objective, and every rule is objective-scoped. This is NOT
   fixed here and must not be "fixed" by weakening the scope. It is fixed at
   the call sites, which hold the objective already; `resolve_device` now
   takes one, with a backward-compatible default, so each site is one word.
   Until those land, `auto` reaches the GPU through `resolve_device_full`,
   `decide_device`, `decide_device_report` and `decide_device_report_reported`
   and not through `model.fit`.

`PROFILE_REPORTED` still outranks everything and is still the only
authoritative source. What changed is that the absence of a reading is no
longer the absence of an identity.

`MOJOTREES_AUTO_MIN_CELLS` is the escape hatch for running that benchmark:
an integer cell count (`n_rows * n_features`) at or above which `auto`
selects the GPU, `0` meaning "whenever the GPU path covers the workload",
unset or negative meaning the heuristic is off. A run that reaches the GPU
through it is reported with `EVIDENCE_ENV` and a warning, never as a
validated choice.

`MOJOTREES_DISABLE_GPU=1` makes this module report that no accelerator is
present: `gpu` raises and `auto` chooses the CPU on a machine that does
have one. It exists to exercise the unavailable-GPU path and to pin a
mixed fleet to the CPU backend.

Parameters the accelerator cannot honor, and why they are blocks
----------------------------------------------------------------
A block is not only "the kernels cannot index this many rows". Three
*training parameters* are blocks too, and all three arrived on 2026-08-16
from a sweep whose whole point was that nobody had checked:

- `enable_bundle` (`BLOCK_FEATURE_BUNDLING`),
- `linear_tree` (`BLOCK_LINEAR_TREE`),
- a forced-split document (`BLOCK_FORCED_SPLITS`).

Each was accepted by the GPU trainers, applied by none of them, and
reported as success. The pattern that hid all three, and that hid
`leaf_estimation_iterations` and `MOJOTREES_DERIVATIVE_PRECISION=float64`
before them, is an *aggregate guard assumed to cover a knob it does not*:
`ExtraTreeParams.is_active()` names `forced`, which reads as coverage and
is coverage only for the non-default `MOJOTREES_GPU_SPLIT_STRATEGY=device`
path; the shipping host split scan refuses only
`ExtraTreeParams.needs_grower_support()`, a strictly smaller set. Do not
read an aggregate as an answer here. Check the knob against the code that
would have to read it.

The two device requests get different answers, and that asymmetry is the
whole design: an explicit `device='gpu'` raises, because a caller who named
a backend is entitled to be told it cannot honor what they asked for, while
`device='auto'` selects the CPU with `DECISION_AUTO_CPU_BLOCKED`, because a
caller who asked us to pick asked for the backend that *can*. Blocks are
collected before `crossover_rules()` is consulted, so no shape is compared
and `AUTO_GPU_MIN_ROWS` never enters the answer for any of them.

WHAT THIS MODULE CANNOT CLOSE ON ITS OWN. `resolve_device`, which is what
`model.fit` calls, describes the shape and the objective and nothing else,
so it cannot see any of these three. `resolve_device_full` takes all of
them and is the entry point a trainer holding its own `BoosterParams`
should call; until each call site passes them (one argument each, in
model.mojo and trainset.mojo, which are other lanes' files) an `auto` fit
that sets one of the three still reaches `train_gpu` and is refused *there*
rather than routed. That is the trainers' refusal, not a fallback, and it
is a strictly better outcome than the silent ignore it replaces -- but it
is not the outcome this policy specifies, and it is written down here
rather than left to be rediscovered.

Unknown hardware
----------------
A device whose API this module cannot name gets `GpuProfile.generic()`, the
conservative portable profile in apple_gpu_policy.mojo. It is deliberately
not Apple-shaped and not NVIDIA-shaped. Nothing in this module infers
hardware from an operating system name or from a marketing chip string. The
ladder, strongest first: `PROFILE_REPORTED`, attributes a caller read off an
open `DeviceContext`; `PROFILE_BUILD_TARGET`, the api and generation this
binary was compiled for, identity only and every number still the fallback;
`PROFILE_DECLARED`, an operator naming an API through `MOJOTREES_GPU_BACKEND`
that contradicts the build target; `PROFILE_FALLBACK`, nothing at all. A
declaration that agrees with the build target adds nothing and is not
recorded; one that disagrees removes the generation rather than installing a
different one, so hardware cannot be misdeclared into a crossover rule. No
source below `PROFILE_REPORTED` ever supplies a capability number.

Availability, and now identity, are build properties
----------------------------------------------------
Mojo resolves `has_accelerator()` at compile time, so a binary built where
an accelerator was present reports one as available. On a redistributed
build (a wheel) a `gpu` request therefore fails when the device is opened
rather than when it is resolved. `WARN_BUILD_TIME_AVAILABILITY` marks every
decision that rests on that comptime answer, and `MOJOTREES_DISABLE_GPU=1`
is the way to pin such a build to the CPU.

`_accelerator_arch()` is comptime in the same way and for the same reason,
which is what makes `PROFILE_BUILD_TARGET` free. It is also wrong in the same
way: on a wheel built for an M4 and run on an M3, it says M4. That is a
stronger error than a wrong availability answer, because it can *select* a
backend rather than merely fail one, so it gets its own
`WARN_BUILD_TARGET_HARDWARE` on top of the availability warning, and the
remedy is the same: `device='cpu'`, `MOJOTREES_DISABLE_GPU=1`, or a caller
that opens a `DeviceContext` and goes through
`decide_device_report_reported`, whose `PROFILE_REPORTED` outranks this.
The alternative was for `auto` to keep selecting the CPU on every machine
including the ones this project has measured, which is not the safer choice,
only the quieter one.

LightGBM difference: LightGBM spells this `device_type` and takes `cpu`,
`gpu`, or `cuda`, with no `auto`. mojotrees has a single portable GPU
backend rather than separate OpenCL and CUDA ones, so the value is `gpu`
for every accelerator, and `auto` is an addition.
"""

from std.os import getenv
from std.sys import has_accelerator

# The accelerator this binary was compiled for, as `<api>:<arch>` (measured
# on the development machine: `metal:4-metal4`). It is comptime, like
# `has_accelerator()` beside it, and it opens no device, which is the whole
# reason this module may call it: the purity contract below forbids importing
# `max.gpu.host`, and identifying hardware by opening a `DeviceContext` in
# `detect()` would have done exactly that, through `device.mojo` and into
# `params.mojo`'s import graph.
#
# It is underscore-private in the standard library, which is a real risk and
# is taken deliberately: it is the only zero-cost source of device identity
# available here, and if it disappears the failure is a compile error at this
# import rather than a silent wrong answer. `build_accelerator_target` below
# is the only caller.
from std.sys.info import _accelerator_arch

from .apple_gpu_policy import (
    API_METAL,
    API_UNKNOWN,
    APPLE_GEN_M4,
    APPLE_GEN_UNKNOWN,
    BYTES_PER_PARTIAL_CELL,
    CROSSOVER_DISABLED,
    GpuProfile,
    api_name,
    apple_generation_name,
    parse_api,
    parse_apple_generation,
    partial_budget_bytes,
)
# `CUSTOM` only. The other objective codes were imported to spell out the
# built-in list here, and that list now comes from objective_registry.mojo,
# so naming them again would be re-establishing the duplicate by hand.
# `CUSTOM` stays because `_collect_blocks` refuses it by name and gives a
# reason specific to it.
from .boosting import CUSTOM

# The derivative-precision rule, imported rather than restated.
# `MOJOTREES_DERIVATIVE_PRECISION` has one home and it is histogram.mojo;
# this module reads its answer into `DeviceCapabilities` so that
# `decide_device` can stay pure. Safe to import: histogram.mojo reaches only
# binning, objective_registry, apple_cpu_policy and parallel, none of which
# come back here, and `boosting` (already imported above) reaches it anyway.
from .histogram import const_hessian_verify, derivative_precision_narrows
# `SCORE_L2` is imported rather than mirrored here. A mirrored constant is a
# default expressed in two places, which has already produced two silent wrong
# answers in this campaign; if `split.mojo` ever renumbers, this must fail to
# compile rather than start refusing the wrong selector. The edge is
# cycle-free: nothing in `split.mojo`'s import closure reaches `device.mojo`,
# `gpu_predict.mojo` or this module, which are the only three importers of the
# policy.
#
# `SCORE_L2` alone, and not `SCORE_COSINE` beside it: the block below tests
# `!= SCORE_L2` so that a selector added later refuses the device instead of
# reaching it, which means the Cosine constant is never named here. Importing
# it for symmetry would be an unused import that reads like a coupling.
from .split import SCORE_COSINE, SCORE_L2
from .tree_parameters_extra import DERIV_PRECISION_FLOAT32

# `GROW_OBLIVIOUS` imported rather than mirrored, on the same argument the
# `SCORE_L2` note above makes: a mirrored code is a second definition that
# nothing keeps equal to the first. `growth_policy` imports nothing from this
# package, so this adds no cycle -- the constraint `cegb.mojo` learned the
# hard way.
from .growth_policy import GROW_LEAFWISE, GROW_OBLIVIOUS
from .initialization import SessionState, warmup_level_name

# The one table of objective facts. Imported rather than restated: this
# module used to carry its own copy of the built-in objective list, its own
# copy of which objectives have a device-side derivative kernel, and its own
# `LAMBDARANK = 7`, which made three lists that had to be edited together and
# nothing that would notice if they were not.
#
# Safe to import: objective_registry.mojo imports only `boosting` (which this
# module already imports) and `metrics` (which imports nothing local), so it
# reaches neither `model.mojo` nor the GPU kernel stack and closes no cycle.
from .objective_registry import (
    LAMBDARANK,
    MULTICLASS,
    SQUARED_ERROR,
    objective_gradients_on_device,
    objective_is_builtin,
)

# Aliased on the way in. unified_memory_policy.mojo has its own
# `block_reason_name` and its own `EVIDENCE_NONE`, and both mean something
# different here: a transfer-route block is not a device block, and its
# evidence ladder is not the crossover evidence identifier. Importing them
# under their own names would shadow this module's, which is how a report
# ends up naming a route refusal as the reason a GPU request was denied.
from .unified_memory_policy import (
    SessionMemoryPlan,
    evidence_name as transfer_evidence_name,
    plan_session_routes,
    retire_event_name,
    route_name as transfer_route_name,
    transfer_block_name,
    transfer_role_name,
)


# --- Mirrors. NOT pinned by any test in this repository. ---
#
# Three constants, all from histogram_gpu.mojo, copied rather than imported
# because importing that module pulls the whole `max.gpu.*` kernel stack into
# a layer that has to stay compilable and testable on a machine with no
# accelerator. Same reason objective_registry.mojo does not import
# gpu_objectives_native.mojo.
#
# The objective mirrors that used to sit here are gone: `LAMBDARANK` and the
# built-in objective list now come from objective_registry.mojo, which is the
# one table of objective facts and which this module can import without
# closing a cycle.
#
# There is no `tests/parallel/test_device_policy.mojo`. Nothing asserts that
# the three below still equal their source, so each is a copy that can drift
# silently, and a drifted `MAX_GPU_ROWS` or `MAX_GPU_BINS` admits a workload
# the kernels cannot index. handoffs/connect_05_device_policy.md specifies
# that test; it is unwritten and unrun, and this comment says so rather than
# claiming a guarantee that does not exist.

# `MAX_ROWS` in histogram_gpu.mojo: the histogram and partition kernels
# index rows as Int32, which is also where the fixed-point accumulator
# stops being exact.
comptime MAX_GPU_ROWS = Int(Int32.MAX)

# `max_bins must be in [2, 256]` in binning.mojo, and `MAX_BINS` in
# histogram_gpu.mojo, which reserves shared memory for that many.
comptime MIN_GPU_BINS = 2
comptime MAX_GPU_BINS = 256

# --- End mirrors. ---


comptime CPU_DEVICE = 0
comptime GPU_DEVICE = 1
comptime AUTO_DEVICE = 2

# `DeviceDecision.selected_device` when the request was refused. An
# explicit `gpu` that cannot run selects nothing; it does not select the
# CPU, because falling back is exactly what this policy refuses to do.
comptime NO_DEVICE = -1

# `DeviceRequest.objective` when the caller did not declare one. The
# objective gates are skipped for it and the decision carries
# `WARN_INCOMPLETE_REQUEST`, which is how the compatibility entry point in
# device.mojo keeps its four-argument signature without this module
# inventing an objective the caller never named.
comptime OBJECTIVE_UNSPECIFIED = -2

# `DeviceRequest.n_bins` when the caller did not declare one. Same
# treatment: the bin-limit gate is skipped, the histogram terms of the
# memory estimate are unknown, and the estimate is marked partial.
comptime BINS_UNSPECIFIED = 0

# Cells (`n_rows * n_features`) at or above which `auto` chooses the GPU
# with no rule behind it. Negative disables the heuristic, which is the
# default and stays the default now that a rule exists: this knob is how a
# crossover benchmark reaches the GPU on a device no rule covers, and a
# measured rule for one device is not a reason to start guessing on the
# rest. Deliberately the same sentinel apple_gpu_policy.mojo reports on
# `CrossoverInputs.min_cells`.
comptime AUTO_MIN_CELLS = CROSSOVER_DISABLED

# Bumped whenever a crossover rule is added, removed, or retuned, or
# whenever a gate below changes what it admits. A decision carries it so a
# report from one release can be told apart from a report from another.
#
# 2: the first crossover rule, `apple-m4-metal-dense-regression`. Version 1
# shipped an empty table, so `auto` was the CPU at every size on every
# machine.
# 3: the rule became reachable. No threshold moved and no rule was added; what
# changed is that `DeviceCapabilities.detect` can now name the hardware
# (`PROFILE_BUILD_TARGET`) and that `parse_apple_generation` can parse the
# architecture string a Metal device actually reports. Under version 2 the
# table was non-empty and unreachable, so a version-2 report saying "no rule
# covered this" and a version-3 report saying the same thing mean different
# things, which is precisely what this number is for.
# 4: the row floor moved down, from the measured 1,000,000 to a provisional
# 250,000 (`AUTO_GPU_MIN_ROWS`), and the 50,000,000-cell term was removed so
# that one named constant decides. No rule was added and no scope widened.
# This is the first version whose threshold is set ahead of its evidence
# rather than at it, and `crossover_rules()` states the trade and what would
# reverse it; a version-4 GPU selection between 250,000 and 1,000,000 rows is
# a provisional selection and a report carrying this number says so.
# 5: three gates were added and no rule moved. `enable_bundle`, `linear_tree`,
# and a forced-split document are now blocks (`BLOCK_FEATURE_BUNDLING`,
# `BLOCK_LINEAR_TREE`, `BLOCK_FORCED_SPLITS`); under version 4 and every
# version before it they were accepted by the GPU trainers and silently not
# applied. A version-4 report saying a GPU selection was unblocked and a
# version-5 report saying the same thing therefore mean different things for
# a request that set any of the three, which is what this number is for.
# 6: the multiclass crossover rule, `apple-m4-metal-dense-multiclass`. The
# table had one rule and it was scoped to a single output, so `auto` selected
# the CPU for every softmax fit at every size on every machine, including the
# machine where the GPU is measured 1.63x faster at it. No existing rule
# moved, no threshold moved, and no single-output request can match the new
# rule (it requires two or more trees per round). A version-4 report saying
# "no rule covered this" for a multiclass workload and a version-6 report
# saying it therefore mean different things.
# 7: two gates were added and no rule moved, for the same reason version 5
# existed -- a CatBoost parameter that the GPU trainers accepted and silently
# did not apply. `boosting_type='ordered'` (`BLOCK_ORDERED_BOOSTING`) and
# `score_function=Cosine` (`BLOCK_SCORE_FUNCTION`). Both were harmless while
# the Python surface refused them outright; the CatBoost reachability work of
# 2026-08-16 made both reachable, and neither has a device implementation. A
# version-6 report saying a GPU selection was unblocked and a version-7 report
# saying the same thing mean different things for a request that set either.
#
# Worth recording as a mechanism rather than as two entries: a lane that wires
# a feature can open a hole in a refusal that a different lane owns, and no
# reachability walk finds it, because the gate is imported, is called on every
# fit, and is merely BLIND to the parameter it would refuse on. Both halves of
# this version were found that way and neither was found by the lane that
# built the feature.
#
# VERSION 8, 2026-08-16 evening. `BLOCK_RANDOM_STRENGTH`. `random_strength`
# was an UNROUTED refusal, which is a third shape beside the two above: the
# bindings refused it on the GPU at `_mojotrees.mojo`, no policy block
# existed, and `Workload` had no field for it -- so `device='auto'` selected
# the accelerator on shape and the fit then raised inside the grower, which is
# the one outcome `auto` exists to prevent.
#
# **Read this before retiring any block in this file.** A block does two jobs.
# It refuses a configuration, and it is often the ONLY thing making `auto`
# route to the CPU rather than fail. So a block may be retired only when no
# downstream refusal would still fire for the same fit. This one exists
# because of that rule rather than in spite of it: the shipped default is
# `score_function=cosine` AND `random_strength=1`, `BLOCK_SCORE_FUNCTION` is
# scheduled for retirement the moment the device learns Cosine, and retiring
# it with no block here would have left the default selecting the GPU on shape
# and raising in the grower on `ExtraTreeParams.is_active()`. The capability
# landing is not sufficient grounds to remove the gate; the absence of every
# other reason to refuse the same fit is.
# The deepest oblivious tree the device plane grows, and the ONE place this
# bound is written down.
#
# **IT WAS WRITTEN DOWN TWICE, AND THAT IS WHY RAISING IT ON 2026-08-18 DID
# NOT WORK THE FIRST TIME.** `gpu_resident_round.oblivious_device_supported`
# carried `max_depth > 6` and so did the predicate below, and a lane raised
# the first to 8, built clean, corrected two user-visible messages, and
# shipped. Symmetric fits at depth 7 and 8 still refused, because this copy
# was untouched.
#
# It was worse than a missed edit. `grow_policy` and `max_depth` had just been
# wired across the Python boundary in the same session, which made
# `BLOCK_MAX_DEPTH` reachable for the first time. So the two changes composed
# into a block that now WORKS and still refuses, and the lane's own commit
# message said the ceiling was gone. "It builds" is the same class of evidence
# as "the switch is set": neither says the code was reached. What caught it
# was one fit at each depth on each backend, which is four seconds of work.
#
# 7 RATHER THAN 8, AND THE DIFFERENCE IS THE WHOLE POINT OF THIS COMMENT.
#
# The memory arithmetic supports 8: the wide oblivious scan's twelve shared
# arrays are 12,300 bytes at 256 leaves against a conservative 16,384-byte
# threadgroup budget, and 9 is where that genuinely binds at 24,588. Three
# separate constants were raised to 256 on the strength of it and every one of
# those raises is correct.
#
# The bound is 7 because 7 is what a fit actually reaches. Measured 2026-08-18,
# symmetric, 60,000 x 24, eight trees, one fit at each depth on each backend:
#
#     depth 6  cpu ok  gpu ok, rmse bit-identical at 0.332497
#     depth 7  cpu ok  gpu ok, rmse bit-identical at 0.322253
#     depth 8  cpu ok  gpu raises "active-row ranges do not cover the active
#                      prefix"
#     depth 9  cpu ok  gpu refused here, which is correct
#
# Depth 8 does not hit a capacity refusal. It hits an INVARIANT in the
# active-row range machinery, which is a real defect in the level plane at 256
# leaves and not a sizing question. Advertising a depth whose fits raise an
# internal invariant message is worse than advertising one less depth, because
# the message is not something a user can act on.
#
# **This is the rule applied to itself.** A lifted bound is verified by a fit
# at the new bound, or it is not lifted. That rule was written today after
# this same bound was raised to 8, built clean, and shipped with a commit
# message saying the ceiling was gone while depth 7 and 8 both still refused.
# Raising it to 8 a second time on arithmetic alone would have been the same
# error with better arithmetic.
#
# WHAT MOVES IT TO 8: fix the range containment at 256 leaves, then one fit at
# depth 8 on both backends in the same commit. The constants are already
# sized. See `gpu_split_search.OBLIVIOUS_MAX_LEAVES` for the account of the
# three justifications that turned out not to bind.
comptime OBLIVIOUS_DEVICE_MAX_DEPTH = 7

comptime POLICY_VERSION = 10

# `DeviceDecision.evidence_id` values that are not a crossover rule name.
comptime EVIDENCE_NONE = String("none")
comptime EVIDENCE_EXPLICIT = String("explicit-request")
comptime EVIDENCE_ENV = String("MOJOTREES_AUTO_MIN_CELLS")


# --- Where a capability profile came from -----------------------------

comptime PROFILE_NONE = 0
"""No accelerator, so no profile was built."""

comptime PROFILE_FALLBACK = 1
"""`GpuProfile.generic()`: nobody has opened a device, so the conservative
portable profile stands in. Not Apple-shaped and not NVIDIA-shaped."""

comptime PROFILE_DECLARED = 2
"""An operator named the API through `MOJOTREES_GPU_BACKEND`. The numbers
are still the portable fallback; only the API name is theirs."""

comptime PROFILE_REPORTED = 3
"""Read from an open `DeviceContext`. Authoritative."""

comptime PROFILE_SYNTHETIC = 4
"""A named fixture (`apple_synthetic`), never a reading. Only tests and
benchmarks should produce this."""

comptime PROFILE_BUILD_TARGET = 5
"""The API and Apple generation of the accelerator this binary was *compiled
for*, from the comptime `_accelerator_arch()`. Identity only: every capability
number is still the portable fallback, exactly as `PROFILE_DECLARED` takes
only an API name.

Weaker than `PROFILE_REPORTED` and stronger than `PROFILE_DECLARED`. It is a
property of the build, not of the machine, which is the same standing caveat
`WARN_BUILD_TIME_AVAILABILITY` already attaches to `gpu_available` for the
same reason; a decision that *selects a backend* on it carries
`WARN_BUILD_TARGET_HARDWARE` as well, because selecting on a build property
is a stronger claim than reporting availability on one."""


def profile_source_name(source: Int) -> String:
    if source == PROFILE_NONE:
        return String("none")
    if source == PROFILE_FALLBACK:
        return String("fallback")
    if source == PROFILE_DECLARED:
        return String("declared")
    if source == PROFILE_REPORTED:
        return String("reported")
    if source == PROFILE_SYNTHETIC:
        return String("synthetic")
    if source == PROFILE_BUILD_TARGET:
        return String("build-target")
    return String("unknown")


# --- Blocking reasons -------------------------------------------------
#
# A block is something that will actually fail on the GPU path. Each one
# corresponds to a rule enforced somewhere else: the trainer's guards in
# train_gpu.mojo and boosting.mojo, the binner's range in binning.mojo, or
# the kernels' indexing limits in histogram_gpu.mojo. Codes are stable and
# travel into serialized reports, so they are appended to, never renumbered.

comptime BLOCK_NO_ACCELERATOR = 1
comptime BLOCK_GPU_DISABLED_ENV = 2
comptime BLOCK_SPARSE_INPUT = 3
comptime BLOCK_CUSTOM_OBJECTIVE = 4
comptime BLOCK_RANKING_OBJECTIVE = 5
comptime BLOCK_UNKNOWN_OBJECTIVE = 6
comptime BLOCK_VALIDATION_SET = 7
comptime BLOCK_ROW_LIMIT = 8
comptime BLOCK_BIN_LIMIT = 9
comptime BLOCK_OUTPUT_LIMIT = 10
comptime BLOCK_MEMORY_BUDGET = 11
comptime BLOCK_DERIVATIVE_PRECISION = 12

# --- The 2026-08-16 refusal sweep -------------------------------------
#
# Three parameters that the GPU growers accepted and then did not apply.
# Each was found the same way the derivative-precision defect above was
# found: by checking a knob against the code that would have to read it,
# rather than against an aggregate guard that was assumed to cover it.
#
# `ExtraTreeParams.is_active()` names `forced` and so looks like coverage,
# and it is coverage for exactly one path: `_check_device_search_supported`
# refuses the whole bundle under `MOJOTREES_GPU_SPLIT_STRATEGY=device`. The
# host split scan is the *default*, it routes through `tree._search`, and
# `tree._search` refuses only `needs_grower_support()` -- which is
# `max_delta_step`, `path_smooth`, `extra_trees`, and `random_strength`, and
# is not `forced`. So a forced-split document reached the shipping GPU
# grower, was never read, and produced an unforced tree reported as success.
# distributed.mojo refuses the same parameter for the same reason
# (`_UNSUPPORTED_FORCED_SPLITS`), which is the precedent these follow.
#
# The fourth is an environment knob rather than a parameter, and it is the
# most instructive of the four. `MOJOTREES_CONST_HESSIAN` and
# `MOJOTREES_CONST_HESSIAN_VERIFY` are one pair: the first enables a
# histogram shortcut, the second audits it. The GPU honors the first through
# its own read in `gpu_active_rows.GpuActiveRows.__init__` and does not
# implement the second at all, because the audit walks a host hessian array
# and the device's hessians are on the device. So a GPU fit under the audit
# took the shortcut and reported it as checked, which is a worse failure than
# a knob that merely does nothing: the user has stopped looking.
comptime BLOCK_FEATURE_BUNDLING = 13
comptime BLOCK_LINEAR_TREE = 14
comptime BLOCK_FORCED_SPLITS = 15
comptime BLOCK_CONST_HESSIAN_VERIFY = 16
comptime BLOCK_ORDERED_BOOSTING = 17
comptime BLOCK_SCORE_FUNCTION = 18
comptime BLOCK_RANDOM_STRENGTH = 19

# --- The oblivious device route ---------------------------------------
#
# TWO CODES, ONE CONDITION, DELIBERATELY. Both of these fire only under
# `grow_policy=oblivious`: the depth bound at `gpu_resident_round:1645`
# (`params.max_depth < 1 or params.max_depth > OBLIVIOUS_DEVICE_MAX_DEPTH ->
# OBLIVIOUS_DEPTH`) exists
# because the oblivious plane sizes every table from `1 << max_depth`, and
# there is no depth bound at all on the leaf-wise or depth-wise device paths.
# So `BLOCK_MAX_DEPTH` is a SUB-REASON of `BLOCK_GROW_POLICY` and not a
# parallel one, and that is worth saying here rather than leaving to be
# discovered when one is retired and the other silently stops firing.
#
# They are two codes anyway, because a fit refused for its depth should be
# told about its depth. The shipped default sits exactly ON the bound at
# `max_depth=6`, so the difference between "your depth is the problem" and
# "your growth policy is the problem" is the difference between a user
# changing one number and a user changing backends.
#
# WHY EITHER EXISTS, given train_gpu already refuses. It refuses by RAISING
# (`train_gpu:1780`), and its own comment explains that it raises rather than
# falls back because no other GPU grower implements a symmetric tree. That is
# correct for the trainer and it is a CLIFF for `auto`: the CPU grower does
# implement this mode, so `auto` has a working path it was never offered.
# Under the standing ruling these route -- `auto` falls back to the CPU,
# explicit `device='gpu'` raises -- and neither may silently decline.
comptime BLOCK_GROW_POLICY = 20
comptime BLOCK_MAX_DEPTH = 21


def block_reason_name(code: Int) -> String:
    if code == BLOCK_NO_ACCELERATOR:
        return String("no-accelerator")
    if code == BLOCK_GPU_DISABLED_ENV:
        return String("gpu-disabled-env")
    if code == BLOCK_SPARSE_INPUT:
        return String("sparse-input")
    if code == BLOCK_CUSTOM_OBJECTIVE:
        return String("custom-objective")
    if code == BLOCK_RANKING_OBJECTIVE:
        return String("ranking-objective")
    if code == BLOCK_UNKNOWN_OBJECTIVE:
        return String("unknown-objective")
    if code == BLOCK_VALIDATION_SET:
        return String("validation-set")
    if code == BLOCK_ROW_LIMIT:
        return String("row-limit")
    if code == BLOCK_BIN_LIMIT:
        return String("bin-limit")
    if code == BLOCK_OUTPUT_LIMIT:
        return String("output-limit")
    if code == BLOCK_MEMORY_BUDGET:
        return String("memory-budget")
    if code == BLOCK_DERIVATIVE_PRECISION:
        return String("derivative-precision")
    if code == BLOCK_FEATURE_BUNDLING:
        return String("feature-bundling")
    if code == BLOCK_LINEAR_TREE:
        return String("linear-tree")
    if code == BLOCK_FORCED_SPLITS:
        return String("forced-splits")
    if code == BLOCK_CONST_HESSIAN_VERIFY:
        return String("const-hessian-verify")
    if code == BLOCK_ORDERED_BOOSTING:
        return String("ordered-boosting")
    if code == BLOCK_SCORE_FUNCTION:
        return String("score-function")
    if code == BLOCK_RANDOM_STRENGTH:
        return String("random-strength")
    if code == BLOCK_GROW_POLICY:
        return String("grow-policy")
    if code == BLOCK_MAX_DEPTH:
        return String("max-depth")
    return String("unknown-block")


# --- Warnings ---------------------------------------------------------
#
# A warning never changes the selected backend. It records something the
# reader should know about how much the decision is worth.

comptime WARN_BUILD_TIME_AVAILABILITY = 1
comptime WARN_UNKNOWN_HARDWARE = 2
comptime WARN_SYNTHETIC_CAPABILITIES = 3
comptime WARN_UNIFIED_MEMORY_BUDGET = 4
comptime WARN_INCOMPLETE_REQUEST = 5
comptime WARN_EXPLICIT_GPU_UNMEASURED = 6
comptime WARN_ENV_THRESHOLD_UNVALIDATED = 7
comptime WARN_HOST_GRADIENT_PATH = 8
comptime WARN_MEMORY_BUDGET_UNKNOWN = 9
comptime WARN_COLD_SESSION = 10
comptime WARN_UNPROVEN_TRANSFER_ROUTE = 11
comptime WARN_KERNEL_RETIREMENT_ROUTE = 12
comptime WARN_BUILD_TARGET_HARDWARE = 13


def warning_name(code: Int) -> String:
    if code == WARN_BUILD_TIME_AVAILABILITY:
        return String("build-time-availability")
    if code == WARN_UNKNOWN_HARDWARE:
        return String("unknown-hardware")
    if code == WARN_SYNTHETIC_CAPABILITIES:
        return String("synthetic-capabilities")
    if code == WARN_UNIFIED_MEMORY_BUDGET:
        return String("unified-memory-budget")
    if code == WARN_INCOMPLETE_REQUEST:
        return String("incomplete-request")
    if code == WARN_EXPLICIT_GPU_UNMEASURED:
        return String("explicit-gpu-unmeasured")
    if code == WARN_ENV_THRESHOLD_UNVALIDATED:
        return String("env-threshold-unvalidated")
    if code == WARN_HOST_GRADIENT_PATH:
        return String("host-gradient-path")
    if code == WARN_MEMORY_BUDGET_UNKNOWN:
        return String("memory-budget-unknown")
    if code == WARN_COLD_SESSION:
        return String("cold-session")
    if code == WARN_UNPROVEN_TRANSFER_ROUTE:
        return String("unproven-transfer-route")
    if code == WARN_KERNEL_RETIREMENT_ROUTE:
        return String("kernel-retirement-route")
    if code == WARN_BUILD_TARGET_HARDWARE:
        return String("build-target-hardware")
    return String("unknown-warning")


# --- How the decision was reached -------------------------------------

comptime DECISION_EXPLICIT_CPU = 0
comptime DECISION_EXPLICIT_GPU = 1
comptime DECISION_GPU_REFUSED = 2
comptime DECISION_AUTO_CPU_BLOCKED = 3
comptime DECISION_AUTO_CPU_NO_EVIDENCE = 4
comptime DECISION_AUTO_CPU_BELOW_EVIDENCE = 5
comptime DECISION_AUTO_CPU_BELOW_ENV_THRESHOLD = 6
comptime DECISION_AUTO_GPU_EVIDENCE = 7
comptime DECISION_AUTO_GPU_ENV_THRESHOLD = 8


def decision_name(code: Int) -> String:
    if code == DECISION_EXPLICIT_CPU:
        return String("explicit-cpu")
    if code == DECISION_EXPLICIT_GPU:
        return String("explicit-gpu")
    if code == DECISION_GPU_REFUSED:
        return String("gpu-refused")
    if code == DECISION_AUTO_CPU_BLOCKED:
        return String("auto-cpu-blocked")
    if code == DECISION_AUTO_CPU_NO_EVIDENCE:
        return String("auto-cpu-no-evidence")
    if code == DECISION_AUTO_CPU_BELOW_EVIDENCE:
        return String("auto-cpu-below-evidence")
    if code == DECISION_AUTO_CPU_BELOW_ENV_THRESHOLD:
        return String("auto-cpu-below-env-threshold")
    if code == DECISION_AUTO_GPU_EVIDENCE:
        return String("auto-gpu-evidence")
    if code == DECISION_AUTO_GPU_ENV_THRESHOLD:
        return String("auto-gpu-env-threshold")
    return String("unknown-decision")


# --- Vocabulary -------------------------------------------------------


def parse_device(name: String) raises -> Int:
    """Device code for a public device name ("cpu", "gpu", or "auto").

    Names are canonical lowercase here. The Python wrapper lowercases what
    the user passes before calling in, which is how LightGBM treats
    `device_type`."""
    if name == "cpu":
        return CPU_DEVICE
    if name == "gpu":
        return GPU_DEVICE
    if name == "auto":
        return AUTO_DEVICE
    raise Error(
        "unknown device '", name, "'; expected 'cpu', 'gpu', or 'auto'"
    )


def device_name(device: Int) raises -> String:
    """Public device name for a device code."""
    if device == CPU_DEVICE:
        return String("cpu")
    if device == GPU_DEVICE:
        return String("gpu")
    if device == AUTO_DEVICE:
        return String("auto")
    if device == NO_DEVICE:
        return String("none")
    raise Error("unknown device code ", device)


# --- What the GPU training path supports ------------------------------


def is_builtin_objective(objective: Int) -> Bool:
    """Whether `objective` is one of the built-in single-output objectives
    the boosted trainers implement.

    The same membership `_check_objective` in boosting.mojo tests before it
    raises "unknown objective", expressed as a predicate so the device
    policy can answer without raising. `CUSTOM` is excluded (its gradients
    come from a caller-supplied callable) and so is `LAMBDARANK` (its
    gradients come from query groups) and `MULTICLASS` (one tree per class
    per round, through its own trainer).

    Delegates to `objective_is_builtin` in objective_registry.mojo, which is
    the one table of objective facts. Kept as a name here because it is the
    spelling the gates below read and because a device question should not
    have to know which module the answer lives in; it holds no list of its
    own."""
    return objective_is_builtin(objective)


def gpu_trains_objective(objective: Int) -> Bool:
    """Whether `train_gpu` covers this objective.

    Every built-in objective, which is what `train_gpu` accepts: it runs
    the same `_check_objective` the CPU trainer does and then grows trees
    on the device.

    Deliberately narrower than "does any GPU trainer exist for this". It
    does not for `LAMBDARANK`, which is CPU only; it does for `CUSTOM`
    (`train_custom_gpu`), which the `device` setting does not route to
    because reaching it is an explicit call; and it does for multiclass
    (`train_multiclass_gpu`), which the `device` setting *does* route to,
    through `model.fit_multiclass`, `trainset.train_dataset_multiclass`, and
    `external_memory.train_external_multiclass`. So this predicate answering
    False for `MULTICLASS` is not the whole answer for a multiclass run and
    must not be read as one: `_collect_blocks` gives multiclass its own
    branch and does not block it, and `crossover_rules()` gives it its own
    rule. `objective_backends` in objective_registry.mojo answers the wider
    question this predicate is narrower than.

    `OBJECTIVE_UNSPECIFIED` answers True: a caller that did not name an
    objective is not asserting an unsupported one, and the decision carries
    `WARN_INCOMPLETE_REQUEST` so the gap is visible rather than silently
    resolved either way."""
    if objective == OBJECTIVE_UNSPECIFIED:
        return True
    return is_builtin_objective(objective)


def gpu_objective_is_device_resident(objective: Int) -> Bool:
    """Whether this objective's gradients are generated on the device.

    The objectives with a closed-form per-row derivative the kernels in
    gpu_objectives_native.mojo implement, which today is every built-in
    one. When it is False, `train_gpu` still trains the objective, but it
    fills gradients on the host and uploads them once per round, which is
    slower and worth reporting; that report is `WARN_HOST_GRADIENT_PATH`,
    never a block.

    Delegates to `objective_gradients_on_device` in objective_registry.mojo,
    which is the one table of objective facts and which
    `gpu_objectives_native.supports_device_objective` already defers to. The
    three predicates therefore now agree by construction rather than by
    three lists being edited together.

    `OBJECTIVE_UNSPECIFIED` answers True for the same reason
    `gpu_trains_objective` does: a caller that did not name an objective is
    not asserting one without a kernel, and the gap is reported through
    `WARN_INCOMPLETE_REQUEST` rather than resolved silently either way."""
    if objective == OBJECTIVE_UNSPECIFIED:
        return True
    return objective_gradients_on_device(objective)


def gpu_supports_outputs(n_outputs: Int) -> Bool:
    """Whether the complete GPU training path covers this many trees per
    boosting round. `n_outputs` is 1 for single-output training and for
    binary classification, and the class count beyond that.

    Every workload the device vocabulary routes is covered: multiclass
    grows one tree per class per round through `train_multiclass_gpu`, on
    the same device-resident builder the single-output trainer uses. The
    check stays as the one place to reject a future workload the GPU path
    does not implement, which is why the engine still consults it rather
    than assuming coverage."""
    return n_outputs >= 1


# --- Environment ------------------------------------------------------


def build_has_accelerator() -> Bool:
    """Whether an accelerator was present when this build was compiled.

    Mojo resolves `has_accelerator()` at compile time, so this is a
    property of the binary, not of the machine running it. See the module
    docstring on redistributed builds."""
    comptime if has_accelerator():
        return True
    else:
        return False


def gpu_disabled_by_env() -> Bool:
    """Whether `MOJOTREES_DISABLE_GPU=1` pins this process to the CPU."""
    return getenv("MOJOTREES_DISABLE_GPU") == "1"


def gpu_available() -> Bool:
    """True when training can run on an accelerator: one was present when
    this build was compiled and `MOJOTREES_DISABLE_GPU=1` is not set."""
    if not build_has_accelerator():
        return False
    return not gpu_disabled_by_env()


def env_derivative_precision_is_float64() -> Bool:
    """Whether this process asked for Float64 per-row derivatives, through
    `MOJOTREES_DERIVATIVE_PRECISION=float64`.

    The one read of that variable on this side of the boundary, and it is
    not a second copy of the rule: `histogram.derivative_precision_narrows`
    is the rule, and this is its negation folded into `DeviceCapabilities`
    so that `decide_device` stays pure. A typo in the value is diagnosed by
    `histogram.check_derivative_precision`, which every entry point that can
    raise already calls; anything unrecognized here means the Float32
    default, which is the setting the GPU path implements, so a typo can
    only ever leave the device reachable, never make it unreachable.

    Why the device policy cares at all. `gpu_gradient_stream.stage_gradients`
    narrows every per-row derivative to Float32 as it uploads, and the GPU
    histogram runs whatever else is configured, so the accelerator has never
    been able to honor a Float64 derivative request. Until 2026-08-16 it
    accepted the flag and produced the Float32 answer anyway, reported as
    success. The growers now refuse it, which is correct for `device='gpu'`;
    this function is what lets `auto` route around it instead.
    """
    return not derivative_precision_narrows()


def env_const_hessian_verify() -> Bool:
    """Whether this process asked for the constant-hessian audit, through
    `MOJOTREES_CONST_HESSIAN_VERIFY=1`.

    The one read of that variable on this side of the boundary, and not a
    second copy of the rule: `histogram.const_hessian_verify` is the rule, and
    this is it folded into `DeviceCapabilities` so that `decide_device` stays
    pure. Exactly the arrangement `env_derivative_precision_is_float64` above
    has, for exactly the same reason.

    Why the device policy cares. The audit is a host walk over the host
    hessian array (`histogram._check_constant_hessian`), and every CPU builder
    performs it; no GPU builder can, because the hessians it accumulates from
    are the device's. Until 2026-08-16 a GPU fit accepted the flag, took the
    constant-hessian shortcut, and reported an audited result. The trainers
    now refuse it, which is correct for `device='gpu'`; this function is what
    lets `auto` route around it instead.
    """
    return const_hessian_verify()


def env_auto_min_cells() -> Int:
    """The `auto` size threshold in cells. Unset, negative, or unparsable
    means disabled, in which case `auto` never selects the GPU on size
    alone."""
    var s = getenv("MOJOTREES_AUTO_MIN_CELLS")
    if s.byte_length() == 0:
        return AUTO_MIN_CELLS
    try:
        return Int(s)
    except:
        return AUTO_MIN_CELLS


def build_accelerator_target() -> String:
    """The accelerator this binary was compiled for, as `<api>:<arch>`, or
    empty when it was compiled without one.

    MEASURED on the development machine, 2026-08-16: `metal:4-metal4`. The
    same reading `DeviceContext().arch_name()` gives as `4-metal4`, with the
    API in front of it.

    Comptime, exactly like `has_accelerator()` above, and guarded the way
    every accelerator entry point in this repository is guarded: the call
    sits inside `comptime if has_accelerator()` with an `else` covering the
    whole body, so a CPU-only build never elaborates it. That pattern is not
    decoration here. A build with no GPU architecture fails at compile time
    rather than at run time, and an Apple machine never reproduces it, so the
    guard is the only thing standing between this module and the x86-64 half
    of CI.
    """
    comptime if has_accelerator():
        return String(_accelerator_arch())
    else:
        return String("")


def _target_api_text(target: String) -> String:
    """The API half of `<api>:<arch>`, or the whole string when there is no
    separator. A backend that reports a bare API name is not an error; it is
    an arch this module cannot name, which is what `API_UNKNOWN` and
    `APPLE_GEN_UNKNOWN` are for."""
    var sep = target.find(":")
    if sep < 0:
        return target.copy()
    return String(target[byte=0:sep])


def _target_arch_text(target: String) -> String:
    """The arch half of `<api>:<arch>`, or empty when there is no
    separator."""
    var sep = target.find(":")
    if sep < 0:
        return String("")
    return String(target[byte = sep + 1 : target.byte_length()])


def build_target_profile() raises -> GpuProfile:
    """A `GpuProfile` carrying the build target's *identity* and the portable
    fallback's *numbers*.

    Identity only, and the split matters. `api` and `apple_generation` come
    from the toolchain's own record of what this binary was compiled for, so
    they are as trustworthy as `has_accelerator()` and no more. Every
    capability number stays `GpuProfile.generic()`'s, because the build target
    says nothing about core counts, threadgroup memory, or a memory budget,
    and inventing them would put a fabricated number into the memory gate.
    A zero memory budget cannot block a run (`_collect_blocks` requires
    `memory_budget_known()`), so this profile can only ever widen what is
    admitted through identity, never through capacity.

    `synthetic` stays True for the same reason: the numbers were constructed.
    `unified_memory` follows `GpuProfile.from_reported`'s rule, Metal implies
    unified, which today changes nothing at all because
    `plan_session_routes` returns the all-staged plan under an empty evidence
    ledger whichever way it is answered.

    Raises never in practice; it is `raises` because `GpuProfile.generic()`
    is.
    """
    var target = build_accelerator_target()
    var base = GpuProfile.generic()
    if target.byte_length() == 0:
        return base^
    var api = parse_api(_target_api_text(target))
    if api == API_UNKNOWN:
        # An accelerator whose API this module cannot name. The fallback
        # profile is the honest answer and no rule will match it.
        return base^
    var generation = APPLE_GEN_UNKNOWN
    if api == API_METAL:
        generation = parse_apple_generation(_target_arch_text(target))
    return GpuProfile(
        api,
        generation,
        base.core_count,
        base.max_threads_per_block,
        base.max_shared_memory_per_block,
        base.memory_budget_bytes,
        api == API_METAL,
        True,
    )


def env_declared_api() -> Int:
    """The GPU API an operator named through `MOJOTREES_GPU_BACKEND`, or
    `API_UNKNOWN`.

    A declaration, not a detection: it names the API for reporting and for
    scoping a crossover rule, and it never supplies a capability number.
    Nothing here reads an operating system name or a marketing chip string
    to guess at hardware."""
    var s = getenv("MOJOTREES_GPU_BACKEND")
    if s.byte_length() == 0:
        return API_UNKNOWN
    return parse_api(s)


def _bool_text(value: Bool) -> String:
    """How a Bool is spelled in the serialized decision."""
    if value:
        return String("true")
    return String("false")


# --- Reason lists -----------------------------------------------------


struct ReasonList(Copyable, Movable):
    """An ordered list of (stable code, prose) pairs.

    Two parallel lists rather than a list of pairs so the serialized form
    stays flat and a consumer that only matches on codes never has to parse
    the prose."""

    var codes: List[Int]
    var messages: List[String]

    def __init__(out self):
        self.codes = List[Int]()
        self.messages = List[String]()

    def add(mut self, code: Int, var message: String):
        self.codes.append(code)
        self.messages.append(message^)

    def count(self) -> Int:
        return len(self.codes)

    def is_empty(self) -> Bool:
        return len(self.codes) == 0

    def first_message(self) raises -> String:
        if len(self.messages) == 0:
            raise Error("no reasons recorded")
        return self.messages[0].copy()


# --- Workload and request ---------------------------------------------


struct DeviceRequest(Copyable, Movable):
    """One training run, as the policy needs to see it.

    Every field is plain data. Python builds one of these by reading `X`
    and `y`, which is inspection, not policy: nothing here needs the
    dataset itself, and nothing about the decision depends on values it
    could not serialize.

    - `requested_device`: `CPU_DEVICE`, `GPU_DEVICE`, or `AUTO_DEVICE`.
    - `n_rows`, `n_features`: the shape of the training matrix.
    - `n_outputs`: trees grown per boosting round, 1 for single-output
      training and for binary classification, the class count beyond that.
    - `n_bins`: the estimator's `max_bin`, or `BINS_UNSPECIFIED`.
    - `objective`: a boosting.mojo objective code, or
      `OBJECTIVE_UNSPECIFIED`.
    - `sparse`: the input is a sparse matrix. `train_gpu_sparse` grows on
      the compressed matrix, so an explicit `gpu` runs it; `auto` keeps the
      CPU, because the sparse path's crossover is unmeasured. Reported and
      routed, not blocked. Prediction is a different question: there is
      no sparse device predictor, and `gpu_predict_support` blocks it.
    - `categorical`: the run declares categorical features. The GPU
      grower routes them (`split.is_categorical` in train_gpu.mojo), so
      this is reported, not blocked.
    - `has_missing`: the run uses LightGBM's `use_missing` handling. The
      GPU grower carries a missing bin per feature, so this is reported,
      not blocked.
    - `uses_validation`: the run has an eval set. Validation metrics are
      scored on the CPU, so a run with one trains there too.

    The last three are the 2026-08-16 refusal sweep, and they are here for
    one reason: each names a parameter that the GPU growers accepted and
    silently did not apply, and routing `auto` around a parameter requires
    the policy to be told about it. They default False, which is each
    parameter's own default, so a request that does not mention them is the
    request this struct has always described.

    - `bundling`: `BoosterParams.bundling.enabled`, LightGBM's
      `enable_bundle`. Only the dense CPU trainers in boosting.mojo build a
      bundled matrix; `train_gpu` builds its histograms from the unbundled
      binned matrix and read the setting nowhere.
      `train_gpu_sparse._refuse_bundling` already refused it, which is what
      made the dense gap visible.
    - `linear_tree`: `BoosterParams.linear.enabled`, LightGBM's
      `linear_tree`. Linear leaves need the raw feature matrix and the GPU
      trainers take a binned one, exactly as `boosting.train` does; that
      trainer refuses it (`check_linear_tree_unconnected`) and the GPU ones
      did not.
    - `forced_splits`: a non-empty `TreeParams.extra.forced`, LightGBM's
      `forcedsplits_filename`. Applied by `tree.grow_tree` and by nothing
      else. See `BLOCK_FORCED_SPLITS` for why the aggregate guard everyone
      reads did not cover it.
    """

    var requested_device: Int
    var n_rows: Int
    var n_features: Int
    var n_outputs: Int
    var n_bins: Int
    var objective: Int
    var sparse: Bool
    var categorical: Bool
    var has_missing: Bool
    var uses_validation: Bool
    var bundling: Bool
    var linear_tree: Bool
    var forced_splits: Bool
    var ordered_boosting: Bool
    var score_function: Int
    var random_strength: Float64
    var derivative_precision: Int
    # `TreeParams.grow_policy` and `TreeParams.max_depth`. `max_depth` is
    # carried unconditionally rather than only under the oblivious policy,
    # because a request is a description of the fit and not a pre-filtered
    # argument list; the block decides what to do with it.
    var grow_policy: Int
    var max_depth: Int

    def __init__(
        out self,
        requested_device: Int,
        n_rows: Int,
        n_features: Int,
        n_outputs: Int = 1,
        n_bins: Int = BINS_UNSPECIFIED,
        objective: Int = OBJECTIVE_UNSPECIFIED,
        sparse: Bool = False,
        categorical: Bool = False,
        has_missing: Bool = False,
        uses_validation: Bool = False,
        bundling: Bool = False,
        linear_tree: Bool = False,
        forced_splits: Bool = False,
        ordered_boosting: Bool = False,
        score_function: Int = SCORE_L2,
        random_strength: Float64 = 0.0,
        derivative_precision: Int = DERIV_PRECISION_FLOAT32,
        grow_policy: Int = GROW_LEAFWISE,
        max_depth: Int = 0,
    ):
        self.requested_device = requested_device
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_bins = n_bins
        self.objective = objective
        self.sparse = sparse
        self.categorical = categorical
        self.has_missing = has_missing
        self.uses_validation = uses_validation
        self.bundling = bundling
        self.linear_tree = linear_tree
        self.forced_splits = forced_splits
        self.ordered_boosting = ordered_boosting
        self.score_function = score_function
        self.random_strength = random_strength
        self.derivative_precision = derivative_precision
        self.grow_policy = grow_policy
        self.max_depth = max_depth

    def cells(self) -> Int:
        """`n_rows * n_features`, the size measure the crossover rules and
        `MOJOTREES_AUTO_MIN_CELLS` are written in."""
        return self.n_rows * self.n_features

    def bins_known(self) -> Bool:
        return self.n_bins != BINS_UNSPECIFIED

    def objective_known(self) -> Bool:
        return self.objective != OBJECTIVE_UNSPECIFIED

    def is_complete(self) -> Bool:
        """Whether the caller declared everything the policy can gate on.
        An incomplete request still gets a decision; it gets
        `WARN_INCOMPLETE_REQUEST` with it."""
        return self.bins_known() and self.objective_known()


# --- Capabilities -----------------------------------------------------


struct DeviceCapabilities(Copyable, Movable):
    """What the build and the device can do, as one serializable value.

    Never a probe once constructed, so a caller can inject a machine it
    does not have and the engine cannot tell the difference. That is what
    makes every gate below testable without an accelerator.

    - `gpu_available`: training can run on an accelerator.
    - `built_with_accelerator`: an accelerator was present at compile
      time, which is a different question (see the module docstring).
    - `disabled_by_env`: `MOJOTREES_DISABLE_GPU=1` was set.
    - `profile`: the hardware capabilities, from apple_gpu_policy.mojo.
    - `profile_source`: one of the `PROFILE_*` codes, which is how a
      reader tells a reading from a fallback.
    - `max_rows`, `min_bins`, `max_bins`: the kernel and binner limits.
      Fields rather than constants so a build that widens one can say so.
    - `auto_min_cells`: the `MOJOTREES_AUTO_MIN_CELLS` value in effect.
      Negative means the heuristic is off.
    - `derivative_precision_float64`: this process asked for Float64 per-row
      derivatives (`MOJOTREES_DERIVATIVE_PRECISION=float64`), which no GPU
      grower can honor. A *capability* rather than a preference, which is
      why it lands here beside the kernel limits and not near the crossover
      rules: it blocks, and a block is consulted before any threshold.
    - `const_hessian_verify`: this process asked for the constant-hessian
      audit (`MOJOTREES_CONST_HESSIAN_VERIFY=1`), which walks the host
      hessian array and which no GPU builder performs. Here for the same
      reason as the line above, and it is the same shape of defect: the flag
      was accepted and the audit never ran.
    - `session`: how much of the one-time startup cost this process has
      already paid, from initialization.mojo. Reported and warned on, never
      selected on: see `SessionState`.
    - `transfer`: the per-role transfer routes in effect, from
      unified_memory_policy.mojo. The staged copy for every role in every
      context this repository controls.

    The last two are why the environment is read *here* and not in
    `decide_device`: both of them fold an environment variable into a value,
    and keeping every such read on this side of the boundary is what lets
    `decide_device` stay pure and injectable.
    """

    var gpu_available: Bool
    var built_with_accelerator: Bool
    var disabled_by_env: Bool
    var profile: GpuProfile
    var profile_source: Int
    var max_rows: Int
    var min_bins: Int
    var max_bins: Int
    var auto_min_cells: Int
    var derivative_precision_float64: Bool
    var const_hessian_verify: Bool
    var session: SessionState
    var transfer: SessionMemoryPlan

    def __init__(
        out self,
        gpu_available: Bool,
        built_with_accelerator: Bool,
        disabled_by_env: Bool,
        var profile: GpuProfile,
        profile_source: Int,
        auto_min_cells: Int,
        var session: SessionState,
        var transfer: SessionMemoryPlan,
        max_rows: Int = MAX_GPU_ROWS,
        min_bins: Int = MIN_GPU_BINS,
        max_bins: Int = MAX_GPU_BINS,
        derivative_precision_float64: Bool = False,
        const_hessian_verify: Bool = False,
    ):
        self.gpu_available = gpu_available
        self.built_with_accelerator = built_with_accelerator
        self.disabled_by_env = disabled_by_env
        self.profile = profile^
        self.profile_source = profile_source
        self.max_rows = max_rows
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.auto_min_cells = auto_min_cells
        # Defaults False rather than reading the environment here, because
        # every other environment read on this struct happens in the named
        # constructors below and a fixture built by hand must stay a value
        # the caller wrote down. `detect` and `from_profile` fill it in.
        self.derivative_precision_float64 = derivative_precision_float64
        self.const_hessian_verify = const_hessian_verify
        self.session = session^
        self.transfer = transfer^

    @staticmethod
    def detect(
        var session: SessionState = SessionState.cold(),
    ) raises -> DeviceCapabilities:
        """Capabilities for the build and process running right now.

        Opens no device, and that constraint is structural rather than a
        preference: this module is reached from `params.mojo`'s import graph
        through `device.mojo`, so importing `max.gpu.host` here would put the
        GPU host runtime in front of every CPU-only compile, and
        handoffs/migration_20_device_policy.md already declined a smaller
        version of that change for the same reason.

        Identity therefore comes from the build target
        (`build_target_profile`, `PROFILE_BUILD_TARGET`), which is comptime
        and free. Capability *numbers* remain the portable fallback in every
        case; only a caller holding an open `DeviceContext` can supply real
        ones, through `from_profile` or `capabilities_from_reported`, which
        are still the only way to reach `PROFILE_REPORTED`. Such a caller
        should also hand its own `SessionState` rather than taking the cold
        default.

        WHY THIS IS NOT A FALLBACK ANY MORE, AND WHAT IT COST TO BE ONE.
        Until 2026-08-16 this returned `GpuProfile.generic()` unconditionally,
        whose `api` is `API_UNKNOWN` and whose `apple_generation` is
        `APPLE_GEN_UNKNOWN`. The one installed crossover rule is scoped to
        Metal on an M4, and `CrossoverEvidence.matches` tests the API before
        it tests anything else, so the rule could not fire on any request
        resolved through here. `auto` was the CPU at every shape on every
        machine including the one the rule was measured on, which is a
        shipped-default bug and was documented as a structural consequence.

        Precedence, and the one case where a declaration outranks the build
        target. `MOJOTREES_GPU_BACKEND` names an API and nothing else. When it
        agrees with the build target it adds nothing and the build target
        stands, keeping the generation. When it *disagrees*, the operator is
        asserting that this binary is running somewhere other than where it
        was built, which is exactly the redistributed-build case
        `WARN_BUILD_TIME_AVAILABILITY` exists for; the build target's identity
        is then wrong and is dropped, leaving `PROFILE_DECLARED` with the
        operator's API and no generation. A disagreement can only ever remove
        identity, never install a different one, so no rule can be reached by
        misdeclaring hardware.

        Raises for an unparsable `MOJOTREES_GPU_TRANSFER`, which is
        `plan_session_routes`' rule and the same one this module applies to
        `device`: a misspelled knob must not resolve to a different path.
        """
        var built = build_has_accelerator()
        var disabled = gpu_disabled_by_env()
        var available = built and not disabled
        var declared = env_declared_api()
        var profile = GpuProfile.generic()
        var source: Int = PROFILE_NONE
        if available:
            source = PROFILE_FALLBACK
            var target = build_target_profile()
            if target.api != API_UNKNOWN:
                source = PROFILE_BUILD_TARGET
                profile = target^
            if declared != API_UNKNOWN and declared != profile.api:
                source = PROFILE_DECLARED
                profile = GpuProfile(
                    declared,
                    APPLE_GEN_UNKNOWN,
                    profile.core_count,
                    profile.max_threads_per_block,
                    profile.max_shared_memory_per_block,
                    profile.memory_budget_bytes,
                    declared == API_METAL,
                    profile.synthetic,
                )
        var transfer = plan_session_routes(profile.unified_memory)
        return DeviceCapabilities(
            available,
            built,
            disabled,
            profile^,
            source,
            env_auto_min_cells(),
            session^,
            transfer^,
            derivative_precision_float64=env_derivative_precision_is_float64(),
            const_hessian_verify=env_const_hessian_verify(),
        )

    @staticmethod
    def from_profile(
        var profile: GpuProfile,
        profile_source: Int = PROFILE_REPORTED,
        var session: SessionState = SessionState.cold(),
    ) raises -> DeviceCapabilities:
        """Capabilities carrying a profile a caller already has.

        `PROFILE_REPORTED` is the default because the intended caller is
        one that read the attributes off an open `DeviceContext`. Pass
        `PROFILE_SYNTHETIC` for a fixture; the source is recorded and
        surfaces as `WARN_SYNTHETIC_CAPABILITIES`, and it never changes a
        gate.

        That same caller is the one that can answer `session` truthfully: it
        holds the context, so `context_open` is True for it and cold for
        everybody else. The default stays cold because guessing warm on
        behalf of a caller that did not say so reports a paid cost that was
        not paid.
        """
        var built = build_has_accelerator()
        var disabled = gpu_disabled_by_env()
        var transfer = plan_session_routes(profile.unified_memory)
        return DeviceCapabilities(
            built and not disabled,
            built,
            disabled,
            profile^,
            profile_source,
            env_auto_min_cells(),
            session^,
            transfer^,
            derivative_precision_float64=env_derivative_precision_is_float64(),
            const_hessian_verify=env_const_hessian_verify(),
        )

    @staticmethod
    def unavailable() raises -> DeviceCapabilities:
        """A machine with no accelerator. The conservative answer, and the
        one an injected fixture wants when it is testing the CPU path.

        Reads no environment at all, including the transfer knob: a fixture
        asking what the CPU path does should not be affected by a variable
        about device buffers.
        """
        return DeviceCapabilities(
            False,
            False,
            False,
            GpuProfile.generic(),
            PROFILE_NONE,
            AUTO_MIN_CELLS,
            SessionState.cold(),
            SessionMemoryPlan.staged(),
        )

    def memory_budget_known(self) -> Bool:
        return self.profile.memory_budget_bytes > 0


# --- Memory estimate --------------------------------------------------


@fieldwise_init
struct MemoryEstimate(Copyable, Movable):
    """What one GPU training session is estimated to allocate.

    One term per buffer `GpuHistogramBuilder.__init__` creates in
    histogram_gpu.mojo:

        binned matrix     n_rows * n_features * 1     uint8
        leaf ids          n_rows * 4                  int32
        gradients         n_rows * 4 * n_outputs      float32
        hessians          n_rows * 4 * n_outputs      float32
        histograms        n_features * n_bins * 12    3 int32 planes
        feature ids       n_features * 4              int32

    plus, on the host, two pinned float32 staging planes of `n_rows` and
    one pinned copy of the histogram buffer.

    Invariants. No test asserts them today; the handoff specifies the one
    that should, and marks it unwritten. They are stated here because the
    memory gate below compares this estimate against a device budget, and
    invariant 4 in particular is what makes that comparison sound:

    1. Every term is nonnegative.
    2. `device_bytes()` is the sum of the six device terms and nothing
       else, and `host_bytes()` the sum of the two host terms.
    3. `upper_bound_bytes()` equals `device_bytes() + partial_budget`, so
       it is never below `device_bytes()`.
    4. The estimate is nondecreasing in each of `n_rows`, `n_features`,
       `n_outputs`, and `n_bins`. No term may ever shrink as a workload
       grows, because the memory gate compares against a budget and a
       shrinking estimate would admit a run that does not fit.
    5. `bins_known` False means the histogram terms are zero rather than
       guessed. An estimate with `bins_known` False must never be used to
       block a run, because it is a lower bound on an unknown quantity.

    This is an estimate and it is labeled one everywhere it appears. It
    counts the training buffers, not the allocator's own overhead, and the
    `n_outputs` factor on the gradient planes is an upper bound that
    assumes every class plane is resident at once.

    The tiled accumulation strategy also allocates a partial-histogram
    buffer whose size depends on device attributes. `partial_budget_bytes`
    in apple_gpu_policy.mojo derives the ceiling from the reported memory
    budget, tighter when memory is unified, capped by the portable 64 MiB
    ceiling, and that ceiling is what `upper_bound_bytes()` adds.
    """

    var binned_matrix_bytes: Int
    var leaf_id_bytes: Int
    var gradient_bytes: Int
    var hessian_bytes: Int
    var histogram_bytes: Int
    var feature_id_bytes: Int
    var host_staging_bytes: Int
    var host_readback_bytes: Int
    var partial_budget: Int
    var bins_known: Bool

    def device_bytes(self) -> Int:
        """Device allocations excluding the tiled partial buffer."""
        return (
            self.binned_matrix_bytes
            + self.leaf_id_bytes
            + self.gradient_bytes
            + self.hessian_bytes
            + self.histogram_bytes
            + self.feature_id_bytes
        )

    def upper_bound_bytes(self) -> Int:
        """Device allocations with the partial-histogram budget included."""
        return self.device_bytes() + self.partial_budget

    def host_bytes(self) -> Int:
        """Pinned host staging buffers."""
        return self.host_staging_bytes + self.host_readback_bytes


def estimate_gpu_memory(
    request: DeviceRequest, profile: GpuProfile
) raises -> MemoryEstimate:
    """The `MemoryEstimate` for one request on one device.

    See `MemoryEstimate` for the terms and the invariants. Raises for a
    shape with no rows or no features, which is not a workload."""
    if request.n_rows < 1 or request.n_features < 1:
        raise Error(
            "a workload needs at least one row and one feature; got n_rows=",
            request.n_rows,
            ", n_features=",
            request.n_features,
        )
    var planes = request.n_outputs
    if planes < 1:
        planes = 1
    var histogram_bytes: Int = 0
    var readback_bytes: Int = 0
    if request.bins_known():
        var cells = request.n_features * request.n_bins
        histogram_bytes = cells * BYTES_PER_PARTIAL_CELL
        readback_bytes = histogram_bytes
    return MemoryEstimate(
        request.n_rows * request.n_features,
        request.n_rows * 4,
        request.n_rows * 4 * planes,
        request.n_rows * 4 * planes,
        histogram_bytes,
        request.n_features * 4,
        request.n_rows * 8,
        readback_bytes,
        partial_budget_bytes(profile),
        request.bins_known(),
    )


# --- Crossover evidence -----------------------------------------------

# The one installed rule, spelled out as constants so the numbers in it and
# the numbers in `crossover_rules()` cannot drift apart, and so a test can
# assert against the same values the rule was built from.
#
# Two records, both Apple M4 over Metal, both end-to-end training with the
# CPU and GPU arms alternated on an otherwise idle machine, both at
# 1,000,000 rows x 50 dense Float64 features, 255 bins, 31 leaves, 100
# rounds, squared error, single output:
#
#   bench/results/apple_m4_large_scaling_2026-08-14.md
#     three seeds, three alternating CPU/GPU repeats each, minimum reported
#     GPU  4.289, 4.382, 4.378 s
#     CPU 11.094, 11.346, 11.706 s
#
#   docs/GPU_VALIDATION.md, "Apple M4, 2026-08-15: the five-lane GPU
#   performance round"
#     five repeats, minimum reported, each figure reproduced twice in the
#     same window
#     GPU  4.10 s (spread 5.8 percent)
#     CPU 11.36 s (spread 3.8 percent)
#
# So the GPU is between 2.6 and 2.8 times the CPU at this shape, measured
# twice a day apart. That margin is what makes the rule installable at the
# measured point rather than at a multiple of it: gpu_split_policy.mojo
# doubles its threshold because its win was about 2 percent, smaller than
# this machine's noise, and this one is not.
#
# The same 2026-08-14 record has 5,000,000 x 50 at 17.162 s against 56.902
# s, 3.32x, so the advantage grows with rows rather than closing. These two
# records are what scope the rule to Metal, to an M4, to squared error, and
# to single output. They are NOT what sets the row floor: the floor is
# `AUTO_GPU_MIN_ROWS`, which sits below every shape either record covers and
# carries its own reasoning.
comptime M4_TRAINING_RULE_NAME = String("apple-m4-metal-dense-regression")
comptime M4_TRAINING_EVIDENCE_ID = String(
    "bench/results/apple_m4_large_scaling_2026-08-14.md +"
    " docs/GPU_VALIDATION.md 'Apple M4, 2026-08-15'"
)
comptime M4_TRAINING_MEASURED_ON = String(
    "Apple M4, 10 GPU cores, Metal, macOS 26.5.2 arm64, Mojo 1.0.0, MAX"
    " 26.5.0; 1,000,000 x 50 dense, 255 bins, 31 leaves, 100 rounds,"
    " squared error, single output"
)

# THE THRESHOLD, AND IT IS A PLAIN PROVISIONAL CONSTANT.
#
# `AUTO_GPU_MIN_ROWS` is the row count at or above which `auto` selects the
# GPU. One number, written down, with nothing computed into it: not a fitted
# crossover, not a work estimate, not a function of features or bins or
# leaves or core count, and not the cell product that used to sit beside it.
# A reader who wants to know where `auto` switches reads this line and is
# finished.
#
# IT IS PROVISIONAL AND IT IS AHEAD OF ITS EVIDENCE. The real crossover is a
# measurement nobody has taken and it is scheduled for the end of the current
# feature work, not for this lane; `crossover_rules()` below states exactly
# what would move this number and in which direction. Two things are true at
# once and both belong on the record:
#
#   - end to end against LightGBM stock+det at 1,000,000 x 50 the GPU arm is
#     1.18x ahead and the CPU arm is 1.75x behind, so which backend `auto`
#     reaches is worth more than any kernel in either campaign, and a user
#     who asked for `auto` and silently got the losing arm is the failure
#     this constant exists to end;
#   - `bench/results/profile_2026-08-15/RESULTS.md` measured 250,000 x 50 as
#     a GPU *loss* against our own CPU (1.89 s against 1.66 s, training only,
#     binning excluded). This constant is set at that shape anyway,
#     deliberately, and the reasoning is written out in `crossover_rules()`
#     under WHY THE FLOOR SITS BELOW ITS OWN EVIDENCE. If that trade turns
#     out to be wrong the fix is to raise this one number.
#
# HOW THIS RELATES TO THE CELLS MACHINERY, AND WHICH ONE WINS. There are two
# other size measures in this module and neither of them decides this any
# more:
#
#   - `M4_TRAINING_MIN_CELLS`, which was 50,000,000 and was the product of
#     1,000,000 rows and 50 features. It is DELETED. A rule that gated on
#     rows and on cells gated twice on the same fact and made the shipped
#     threshold a thing you had to derive by division; at 250,000 x 50 the
#     cell product is 12,500,000, so leaving it installed would have kept
#     `auto` on the CPU while this constant said otherwise. The rule's
#     `min_cells` field survives on `CrossoverEvidence` because a future
#     rule measured on cells may want it, and is passed 0 (does not
#     constrain) by the one installed rule.
#   - `MOJOTREES_AUTO_MIN_CELLS` / `AUTO_MIN_CELLS`, the operator override.
#     It is disabled by default and is not a shipped threshold; it exists so
#     a crossover benchmark can reach the GPU on hardware no rule covers.
#     When it *is* set it outranks this constant, because `decide_device`
#     tests it before it consults the rule table, and the decision it
#     produces says so (`DECISION_AUTO_GPU_ENV_THRESHOLD` plus
#     `WARN_ENV_THRESHOLD_UNVALIDATED`). So: the env knob wins when set,
#     this constant wins otherwise, and the cell product decides nothing at
#     all.
comptime AUTO_GPU_MIN_ROWS = 250_000

# The rest of the rule's scope, which is not the threshold. `min_features`
# is the feature count the record was taken at and stays a scope bound
# rather than a second threshold: a 5,000,000 x 10 matrix is a different
# ratio of per-node launch cost to per-node work and the record says nothing
# about it. `max_outputs` keeps multiclass out of *this* rule, which grows
# one tree per class per round through a different trainer and now has a
# rule of its own below.
comptime M4_TRAINING_MIN_FEATURES = 50
comptime M4_TRAINING_MAX_OUTPUTS = 1


# --- The multiclass rule, and the record behind it --------------------
#
# One record, Apple M4 over Metal, end-to-end softmax training with the CPU
# and GPU arms alternated on an otherwise idle machine, at 465,000 rows x 54
# dense features over 7 classes, 255 bins, 31 leaves, 100 rounds:
#
#   bench/results/profile_2026-08-15/RESULTS.md, "Multiclass, measured for
#   the first time, 465,000 x 54, 7 classes"
#     medians of three, spread beside each
#     GPU 15.30 s (0.1 percent)
#     CPU 25.47 s (7.7 percent)
#
#   docs/GPU_VALIDATION.md carries the same table.
#
# So the GPU is 1.63x the CPU at this shape, resolved well outside the noise
# floor: the two spreads do not come close to touching, which is a stronger
# separation than the single-output record's and far stronger than the 2
# percent gpu_split_policy.mojo doubles its threshold for.
#
# WHY THIS RECORD IS THE FIRST HONEST MULTICLASS NUMBER, which matters
# because there are older ones on disk that say the opposite.
# `trainset.train_dataset_multiclass` resolved a device and then discarded
# the answer, so every multiclass "GPU" timing before 2026-08-15 was a CPU
# fit wearing a GPU label. That is provable rather than suspected: the
# covertype CPU and GPU records in
# `bench/real_data/results/20260815T023123Z` carry byte-identical
# `predictions_sha256` while the single-output scenarios in the same run do
# not. The retraction is written out in the RESULTS.md section above, and
# the 44-to-56-second covertype figure it withdraws must not be cited
# against this rule.
comptime M4_MULTICLASS_RULE_NAME = String("apple-m4-metal-dense-multiclass")
comptime M4_MULTICLASS_EVIDENCE_ID = String(
    "bench/results/profile_2026-08-15/RESULTS.md 'Multiclass, measured for"
    " the first time, 465,000 x 54, 7 classes' + docs/GPU_VALIDATION.md"
    " 'Apple M4, 2026-08-15'"
)
comptime M4_MULTICLASS_MEASURED_ON = String(
    "Apple M4, 10 GPU cores, Metal, macOS 26.5.2 arm64, Mojo 1.0.0, MAX"
    " 26.5.0; 465,000 x 54 dense, 7 classes, 255 bins, 31 leaves, 100"
    " rounds, softmax"
)

# The feature count the record was taken at, and a scope bound for exactly
# the reason `M4_TRAINING_MIN_FEATURES` is one: nothing here says what a
# multiclass fit over ten features does. It is 54 rather than 50 because
# that is the shape that was measured, and rounding a scope bound down to
# match another rule's is how a rule quietly widens past its record.
comptime M4_MULTICLASS_MIN_FEATURES = 54

# Trees per round at or above which this rule applies. Two, which is what
# "multiclass" means: `n_classes` is refused below 2 by every multiclass
# trainer in this package.
comptime M4_MULTICLASS_MIN_OUTPUTS = 2


struct CrossoverEvidence(Copyable, Movable):
    """One benchmark-derived rule saying "the GPU wins from here up".

    A rule is a claim about measured performance, so it carries the
    measurement with it. `evidence_id` cites where the numbers live (a
    document section, a benchmark file, a commit) and is what a decision
    reports; `measured_on` names the device they came from. A rule without
    an evidence identifier is not a rule, and `crossover_rules()` is empty
    for exactly that reason.

    Scope narrows a rule to what was actually measured. `api` and
    `apple_generation` limit it to one device family or generation,
    `objective` to one objective, and `min_rows`, `min_features`, and
    `min_cells` are the thresholds themselves. `API_UNKNOWN`,
    `OBJECTIVE_UNSPECIFIED`, and a zero minimum do not constrain. A rule
    matches only when every set field matches, so widening a rule to
    hardware nobody measured takes a deliberate edit.
    """

    var name: String
    var evidence_id: String
    var measured_on: String
    var api: Int
    var apple_generation: Int
    var objective: Int
    var min_rows: Int
    var min_features: Int
    var min_cells: Int
    var max_outputs: Int
    """Trees per round the measurement covered. Zero does not constrain."""
    var min_outputs: Int
    """Trees per round at or above which the rule applies. Zero does not
    constrain.

    The lower bound `max_outputs` is the upper bound of, and it exists for
    the same reason: a rule measured on one tree per round and a rule
    measured on seven are measurements of different work, and neither may
    inherit the other's number. A single-output rule sets `max_outputs=1`
    and leaves this at zero; a multiclass rule sets `min_outputs=2` and, on
    the evidence installed today, leaves `max_outputs` at zero. See
    `crossover_rules()` for why that asymmetry is deliberate rather than an
    omission."""

    def __init__(
        out self,
        var name: String,
        var evidence_id: String,
        var measured_on: String,
        api: Int = API_UNKNOWN,
        apple_generation: Int = 0,
        objective: Int = OBJECTIVE_UNSPECIFIED,
        min_rows: Int = 0,
        min_features: Int = 0,
        min_cells: Int = 0,
        max_outputs: Int = 0,
        min_outputs: Int = 0,
    ) raises:
        if evidence_id.byte_length() == 0:
            raise Error(
                "a crossover rule needs an evidence identifier; cite the"
                " benchmark that measured it"
            )
        self.name = name^
        self.evidence_id = evidence_id^
        self.measured_on = measured_on^
        self.api = api
        self.apple_generation = apple_generation
        self.objective = objective
        self.min_rows = min_rows
        self.min_features = min_features
        self.min_cells = min_cells
        self.max_outputs = max_outputs
        self.min_outputs = min_outputs

    def matches(
        self, caps: DeviceCapabilities, request: DeviceRequest
    ) -> Bool:
        """Whether this rule covers the (device, workload) pair."""
        if self.api != API_UNKNOWN and caps.profile.api != self.api:
            return False
        if (
            self.apple_generation != 0
            and caps.profile.apple_generation != self.apple_generation
        ):
            return False
        if (
            self.objective != OBJECTIVE_UNSPECIFIED
            and request.objective != self.objective
        ):
            return False
        if request.n_rows < self.min_rows:
            return False
        if request.n_features < self.min_features:
            return False
        if request.cells() < self.min_cells:
            return False
        if self.max_outputs != 0 and request.n_outputs > self.max_outputs:
            return False
        if self.min_outputs != 0 and request.n_outputs < self.min_outputs:
            return False
        return True

    def cite(self) -> String:
        """Where the numbers came from and what they were taken on.

        The form the deleted `HybridCosts.cite` used, for the same reason:
        a decision
        that says "the GPU, on evidence" is worth what the reader can go
        and check, so the identifier and the device travel together.
        """
        return String(self.evidence_id, " on ", self.measured_on)


def crossover_rules() raises -> List[CrossoverEvidence]:
    """The benchmark-derived crossover rules, in priority order.

    Two rules, and they are disjoint: the first is single output, the second
    is multiclass, and no request can match both. They share one row floor,
    `AUTO_GPU_MIN_ROWS`, which is a provisional constant for both.

    The single-output rule, as arithmetic. `auto` selects the GPU when *all*
    of:

        profile.api             == metal
        profile.apple_generation == m4
        request.objective       == squared error
        request.n_outputs       <= 1
        request.n_rows          >= AUTO_GPU_MIN_ROWS   (250,000)
        request.n_features      >= 50

    The multiclass rule, as arithmetic. `auto` selects the GPU when *all*
    of:

        profile.api             == metal
        profile.apple_generation == m4
        request.n_outputs       >= 2
        request.n_rows          >= AUTO_GPU_MIN_ROWS   (250,000)
        request.n_features      >= 54

    and in both cases the GPU path is otherwise unblocked and the input is
    dense. There is no cell-count term in either: `min_cells` is passed 0 and
    the comment on `AUTO_GPU_MIN_ROWS` says why. Every other (device,
    workload) pair falls through to `DECISION_AUTO_CPU_BELOW_EVIDENCE` and
    keeps the CPU.

    THE MULTICLASS RULE, AND THE FOUR CHOICES IN IT. It rests on one record,
    quoted above `M4_MULTICLASS_RULE_NAME`: 465,000 x 54 over 7 classes on an
    M4, GPU 15.30 s against CPU 25.47 s, medians of three with spreads of 0.1
    and 7.7 percent. The GPU wins by 1.63x and the two spreads are nowhere
    near touching, so the *direction* of this rule is the best-resolved
    performance fact either backend has. Multiclass is also the one workload
    where the GPU beats our CPU and the CPU is the arm that loses to
    LightGBM, so an `auto` with no rule here was handing every softmax user
    the slower backend at every size.

    1. `objective` is left unconstrained, and that is not the objective gate
       being weakened. The scope this rule needs is "this is a softmax fit",
       and `n_outputs >= 2` states it more exactly than the objective code
       does: trees-per-round is the fact the trainers actually branch on
       (`train_multiclass_gpu` against `train_multiclass`), no single-output
       objective can present it, and it is there whether or not the caller
       named an objective. Scoping on `objective == MULTICLASS` instead would
       have narrowed nothing and would have made the rule unreachable from
       `model.fit_multiclass`, `trainset.train_dataset_multiclass`, and
       `external_memory.train_external_multiclass`, none of which declares
       one -- all three pass `n_classes` as `n_outputs` and stop there. A
       caller that *does* declare `MULTICLASS` (Python's `binding_params`
       does) matches this rule too, which is the point: one rule, both
       spellings, no third place where the two disagree.
    2. `min_outputs` is 2 and `max_outputs` is left at 0, so the class count
       is bounded below and not above. The lower bound is what makes the rule
       disjoint from the single-output one. The upper bound is deliberately
       absent, and this is the one place the rule reaches past its record:
       the measurement is at seven classes, and a class is one more
       independent `grow_tree_gpu` call over the same already-uploaded binned
       matrix, so a larger K is more of the work that was measured rather
       than a different kind of work, and it amortizes the device's fixed
       per-fit cost over more of it. Capping at 7 would have sent a 10-class
       fit to the CPU while a 7-class fit of the same shape went to the
       device, which is a discontinuity with nothing behind it. If a large
       class count is ever measured *losing*, the fix is to set
       `max_outputs` here and bump `POLICY_VERSION`.
    3. `min_features` is 54, not 50. It is the feature count the record was
       taken at, which is the same convention `M4_TRAINING_MIN_FEATURES`
       follows, and rounding it down to match that rule's number would widen
       this one past its own evidence to make two constants look tidy.
    4. `min_rows` is `AUTO_GPU_MIN_ROWS`, THE SAME PROVISIONAL CONSTANT the
       single-output rule uses, and it is provisional here for the same
       reason and to a larger degree. The multiclass record is a single point
       at 465,000 rows; nothing below it has been measured, so the multiclass
       crossover is unmeasured exactly as the single-output one is. A second
       number was not invented, and the reason is not tidiness: the one
       structural argument available says a K-class fit does K times the tree
       work per round against the same one-time upload, binning, and session
       cost, so the device's roughly 1.5 s of fixed cost per fit is amortized
       over K times as much work and the multiclass crossover should sit at
       or *below* the single-output one, never above. Reusing 250,000
       therefore cannot over-reach relative to that argument. It is still a
       constant set ahead of its evidence and it is labelled one, and if a
       measurement in [250,000, 465,000) shows the GPU losing at some class
       count, the honest fix is a `M4_MULTICLASS_MIN_ROWS` of its own rather
       than moving the shared one.

    The evidence for the *hardware and objective* scope is the two Apple M4
    records named in `M4_TRAINING_EVIDENCE_ID` and quoted above this
    function: interleaved CPU/GPU arms at 1,000,000 x 50, a day apart, GPU
    2.6x to 2.8x the CPU both times, and 3.3x at five million rows.

    WHY THE FLOOR SITS BELOW ITS OWN EVIDENCE, WHICH IS THE ONE THING TO
    READ HERE BEFORE CHANGING IT. Until 2026-08-16 this floor was
    1,000,000 rows, the measured point, on the standing principle that a
    threshold placed inside an unmeasured interval is a guess with a number
    on it. It was lowered to 250,000 anyway, and the reasoning is a trade
    rather than a measurement:

    - The two backends now sit on opposite sides of the comparator. At
      1,000,000 x 50 against LightGBM stock+det, end to end, the GPU arm is
      1.18x ahead and the CPU arm is 1.75x behind. Which backend `auto`
      reaches is therefore a 2.1x swing decided by dispatch, larger than
      anything either performance campaign has open, and the cost of getting
      it wrong is asymmetric: an `auto` that under-reaches hands a user the
      losing arm silently and forever, while an `auto` that over-reaches
      costs them a measurable but bounded amount on one shape.
    - At 250,000 x 50 that bounded amount is known and it is small. Our GPU
      measured 1.89 s against our CPU's 1.66 s
      (`bench/results/profile_2026-08-15/RESULTS.md`, training only, binning
      excluded, medians of three, arms interleaved): a 14 percent loss, on a
      machine the same protocol records drifting by a factor of two between
      windows. At 50,000 x 50 the loss is 2.9x, which is not small, and that
      is why the floor is at 250,000 and not lower.
    - The number is a constant on purpose. It is not fitted, and no
      expression should grow here in its place. When the crossover is
      measured, this line changes and `POLICY_VERSION` is bumped.

    So the honest statement of the floor's status is: the hardware scope,
    the objective scope, and the output scope rest on measurement; the row
    floor is a deliberate provisional setting that is 250,000 rows below the
    smallest shape at which the GPU has been measured to win, taken because
    the loss it risks is smaller than the loss it prevents. It is scheduled
    to be replaced by a measured crossover at the end of the current feature
    work.

    Why every other scope field is set, and what stays "no evidence":

    - `api` and `apple_generation`. Metal on an M4 is the only device this
      code has ever run on (docs/GPU_VALIDATION.md, whose CUDA and HIP rows
      are `not run`). An M3 or an M5 shares neither the core count nor the
      synchronization costs the round was tuned against, and a rule that
      inherited to them would be a performance claim about hardware nobody
      owns. `gpu_split_policy._is_observed_m4` draws the same line and for
      the same reason.
    - `objective`. The record's target is a synthetic regression; squared
      error is the objective both arms ran. Other objectives differ in
      where their gradients are computed, and
      `WARN_HOST_GRADIENT_PATH` exists precisely because some of them come
      back to the host every round. Unmeasured, so declined.
    - `max_outputs`. Multiclass grows one tree per class per round through
      a different trainer, so it is not what this record measured and this
      rule still declines it. It is no longer *unmeasured*, which is what
      this bullet used to say: the multiclass rule above carries its own
      record and its own scope, and the two rules are disjoint rather than
      one inheriting the other's numbers.
    - `min_features`, and no cell term beside it. A 5,000,000 x 10 matrix
      would clear any cell floor the measured shape clears while carrying a
      tenth of its features, which is a different ratio of per-node launch
      cost to per-node work. The record has nothing to say about it, so the
      feature count is bounded directly and the product is not bounded at
      all.

    WHAT THE MEASUREMENTS ACTUALLY SAY, in full, so that the trade above is
    checkable. From `bench/results/profile_2026-08-15/RESULTS.md`, taken
    under the pre-registered rules in
    `bench/results/PROFILE_PROTOCOL.md`, arms interleaved, median of three,
    seconds of training with binning excluded, 100 rounds, 31 leaves, 255
    bins, squared error (MEASURED, and reproduced at five repeats for the top
    two shapes in `bench/results/sweep2_2026-08-15/RESULTS.md`):

        shape             our CPU   our GPU
        1,000,000 x 50       6.98      3.58     GPU 1.85x
          250,000 x 50       1.66      1.89     CPU by 14 percent
           50,000 x 50      0.564      1.63     CPU by 2.9x

    The floor now sits at the middle row, which is a shape where the GPU
    lost. It does not sit at the bottom row, where it lost by 2.9x: the
    device's roughly 1.5 s of fixed cost per fit is a growing fraction of a
    smaller tree, and at 50,000 rows that fraction is most of the fit. The
    crossover point itself is still unmeasured and lies somewhere in
    (50,000, 1,000,000] rows at 50 features.

    `bench/results/session3_2026-08-16/RESULTS.md` does not move any of
    this. Its 250,000 and 50,000 figures (1.126 s and 0.789 s) are GPU
    against GPU, the resident plane against the shipping device loop with
    the device split search forced on both arms, and that file's own
    same-night correction withdraws the inference that was drawn from them
    about a gate. Its 1,000,000 x 50 GPU arm at 2.584 s in a fast window
    corroborates this rule's direction. It also measured this machine
    drifting by a factor of two between windows on the GPU and 2.2 on the
    CPU, with a verdict flipping from "indistinguishable" to "resolved" on
    identical code, which is why the 14 percent figure at 250,000 is treated
    as within the noise of this box rather than as a settled loss.

    WHAT WOULD FALSIFY THE RULE, split by which half it hits. The scope and
    the floor fail differently and should be fixed differently:

    - the scope. An interleaved CPU/GPU pair at 1,000,000 x 50 on an M4
      where the GPU is not faster, run in one window on an idle machine; the
      same at 5,000,000 x 50, where the record claims a larger margin; or a
      bin count or leaf budget far from the measured 255 and 31 that
      reverses the sign. Any of these and the rule is wrong and should be
      narrowed or withdrawn rather than patched.
    - the floor. An interleaved pair anywhere in [250,000, 1,000,000) rows
      at 50 features, medians of five, in a single window, showing the GPU
      losing by more than this box's window-to-window drift. That does not
      withdraw the rule; it raises `AUTO_GPU_MIN_ROWS` to the smallest shape
      the GPU still wins, which is the measurement scheduled for the end of
      the current feature work.

    Either way it moves by an edit to one constant and a `POLICY_VERSION`
    bump, the way it went in.

    WHAT WOULD FALSIFY THE MULTICLASS RULE, and it splits the same way:

    - the scope. An interleaved CPU/GPU pair at 465,000 x 54 over 7 classes
      on an M4 where the GPU is not faster, run in one window on an idle
      machine. The record's 0.1 percent spread against 7.7 makes that a hard
      result to reverse, which is why the rule is installed at all.
    - the class count. An interleaved pair at a large K showing the GPU
      losing where it wins at 7. That does not withdraw the rule; it sets
      `max_outputs` on it, which is the one bound left open above.
    - the floor. An interleaved pair anywhere in [250,000, 465,000) rows at
      54 features over some class count, showing the GPU losing by more than
      this box's drift. That gives multiclass a `M4_MULTICLASS_MIN_ROWS` of
      its own; it must not move `AUTO_GPU_MIN_ROWS`, which the single-output
      rule reads for a different reason.

    Do not add a *rule* from reasoning. Add one from a recorded sweep, cite
    it in `evidence_id`, name the device in `measured_on`, and bump
    `POLICY_VERSION`.
    """
    var rules = List[CrossoverEvidence]()
    rules.append(
        CrossoverEvidence(
            M4_TRAINING_RULE_NAME,
            M4_TRAINING_EVIDENCE_ID,
            M4_TRAINING_MEASURED_ON,
            API_METAL,
            APPLE_GEN_M4,
            SQUARED_ERROR,
            AUTO_GPU_MIN_ROWS,
            M4_TRAINING_MIN_FEATURES,
            # min_cells: 0, which does not constrain. The threshold is
            # `AUTO_GPU_MIN_ROWS` and only `AUTO_GPU_MIN_ROWS`; see the
            # comment on it for why the 50,000,000-cell gate was removed
            # rather than lowered alongside the row floor.
            0,
            M4_TRAINING_MAX_OUTPUTS,
            # min_outputs: 0, which does not constrain. `max_outputs=1` is
            # already the whole of this rule's output scope.
            0,
        )
    )
    # THE MULTICLASS RULE WAS WITHDRAWN ON 2026-08-18. Its constants stay
    # below this comment so the record of what was claimed survives the
    # claim, and so a future sweep can reinstate it by restoring one
    # `rules.append` rather than by reconstructing a scope from prose.
    #
    # WHAT FALSIFIED IT. The clause above asked for "an interleaved CPU/GPU
    # pair at 465,000 x 54 over 7 classes on an M4 where the GPU is not
    # faster, run in one window on an idle machine".
    # `bench/results/gbm_bench_2026-08-18/RESULTS.md` is an interleaved
    # six-arm run at 581,012 x 54 over 7 classes on an M4, three repeats,
    # 2.9 percent spread, and the GPU arm took 40.894 s against the CPU
    # arm's 28.077 s. The GPU is not faster. It is 1.45x slower.
    #
    # WHY WITHDRAWN RATHER THAN NARROWED. Two differences from the clause
    # invite narrowing instead: the falsifying run used a leaf budget of 256
    # where the record used 31, and it used real covertype where the record
    # used a synthetic imitation of it. Narrowing on the leaf budget would
    # keep the rule alive on the strength of the same single synthetic
    # record, and that record is the whole of the rule's evidence. Every
    # real-data multiclass number this project has ever taken at this shape
    # has the GPU behind, and there is no measured real-data multiclass GPU
    # win to lose by withdrawing. The docstring's own standing instruction
    # is "do not add a rule from reasoning, add one from a recorded sweep",
    # and narrowing this one to survive its own falsification would be
    # adding it back from reasoning.
    #
    # WHAT THIS COST A USER WHILE IT STOOD. `n_outputs >= 2` and
    # `n_rows >= 250,000` and `n_features >= 54` are all satisfied by
    # covertype, which clears the feature bound exactly. So
    # `MojoTreesClassifier(device="auto")` on that dataset resolved to the
    # GPU and handed back the slower of the two arms we ship, silently, with
    # no prompt to check. `auto` was worse than `device="cpu"` on a real,
    # named, public dataset.
    #
    # WHAT WOULD REINSTATE IT. One interleaved CPU/GPU pair on REAL
    # covertype, 464,958 x 54 over 7 classes, at the leaf budget the record
    # claims (`num_leaves=31`, `max_depth=-1`, 100 rounds), three or five
    # repeats in one window. That configuration already exists as the
    # `multiclass` scenario in `bench/real_data`, so it needs no new code,
    # no new dataset and no new arm. It holds the leaf budget fixed and
    # swaps synthetic data for real, which is the one variable the
    # falsification clause above did not consider. If the GPU wins it, the
    # rule comes back with a `max_leaves` bound. If it loses, the rule was
    # never right at any leaf budget and these constants should be deleted.
    return rules^


# --- The engine -------------------------------------------------------


def _collect_blocks(
    request: DeviceRequest,
    caps: DeviceCapabilities,
    memory: MemoryEstimate,
) raises -> ReasonList:
    """Everything that will actually fail if this workload goes to the GPU.

    Ordered from the cheapest and most fundamental to the most
    workload-specific, so the first entry is the one worth putting in a
    refusal message. Availability short-circuits: on a machine with no
    accelerator, nothing else about the workload matters.
    """
    var blocks = ReasonList()

    if not caps.gpu_available:
        if caps.disabled_by_env:
            blocks.add(
                BLOCK_GPU_DISABLED_ENV,
                String(
                    "MOJOTREES_DISABLE_GPU=1 pins this process to the CPU"
                    " backend, so no accelerator is available"
                ),
            )
        else:
            blocks.add(
                BLOCK_NO_ACCELERATOR,
                String("no accelerator is available to this build"),
            )
        return blocks^

    if (
        caps.derivative_precision_float64
        or request.derivative_precision != DERIV_PRECISION_FLOAT32
    ):
        # TWO ENTRIES, ONE ANSWER, and until 2026-08-16 evening only one of them
        # reached here. `caps.derivative_precision_float64` is read from
        # `MOJOTREES_DERIVATIVE_PRECISION` at capability detection, so the
        # ENVIRONMENT entry routed correctly: `auto` to the CPU, explicit `gpu`
        # told. The PARAMETER entry supplied nothing, so a fit that asked for
        # float64 as a parameter selected the accelerator on shape and then
        # raised inside the grower at `histogram.check_device_derivative_precision`
        # -- a late raise where the same request through the environment got an
        # early route. `ExtraTreeParams.is_active()`'s own docstring already
        # recorded that this setting has two entries which once diverged; that
        # divergence was fixed for the gain pass and survived here.
        #
        # `!= DERIV_PRECISION_FLOAT32` rather than `== DERIV_PRECISION_FLOAT64`,
        # for the reason `BLOCK_SCORE_FUNCTION` was written the same way: a third
        # precision added later and not taught to the device must refuse rather
        # than silently receive the Float32 answer under its own name. A Bool
        # here would have foreclosed that, because a third value would have to be
        # squeezed into "is it float64" and either resolution is wrong for one
        # of them.
        # Precision is a capability, not a preference, so it sits here with
        # the kernel limits and above every shape question. `stage_gradients`
        # in gpu_gradient_stream.mojo narrows each per-row derivative with a
        # literal `Float32(g)` on upload and the device histogram runs
        # whatever else is configured, so no GPU grower has ever produced the
        # Float64 answer this setting asks for. Until 2026-08-16 it produced
        # the Float32 one and reported success.
        #
        # Being a block rather than a rule scope is what gives the two device
        # requests their different answers, and both are deliberate: an
        # explicit `device='gpu'` raises this message, because a backend the
        # user named cannot honor what they asked for and a silent downgrade
        # is the defect itself; `device='auto'` selects the CPU with
        # `DECISION_AUTO_CPU_BLOCKED`, because a user who asked us to pick
        # asked for the backend that *can* honor it. Forwarding instead
        # would have the host compute Float64 derivatives for the device to
        # narrow, which is worse than either.
        blocks.add(
            BLOCK_DERIVATIVE_PRECISION,
            String(
                "derivative_precision=float64 asks for Float64 per-row"
                " derivatives -- set as a parameter, or through"
                " MOJOTREES_DERIVATIVE_PRECISION -- and the GPU gradient"
                " upload narrows every derivative to Float32, so no"
                " accelerator path can produce that answer"
            ),
        )

    if caps.const_hessian_verify:
        # The same treatment and for the same reason as the line above: an
        # audit the accelerator does not run is a capability question, not a
        # size question, so it sits above every shape gate and `auto` never
        # compares a row count before answering.
        #
        # It is worth being precise about what is missing, because the two
        # halves of this diagnostic pair are NOT in the same state.
        # `MOJOTREES_CONST_HESSIAN` is honored on the device
        # (`GpuActiveRows.const_hessian_allowed`, ANDed with the trainer's
        # declaration), so the shortcut itself obeys its knob. What no device
        # path has is `histogram._check_constant_hessian`, which walks the
        # host hessian array. A GPU fit therefore took an *unaudited*
        # shortcut under a flag whose entire purpose is the audit.
        blocks.add(
            BLOCK_CONST_HESSIAN_VERIFY,
            String(
                "MOJOTREES_CONST_HESSIAN_VERIFY=1 asks for the declared"
                " constant hessian to be audited against the data, which is a"
                " walk over the host hessian array that only the CPU"
                " histogram builders perform, so no accelerator path can"
                " report an audited result"
            ),
        )

    # An impossible shape is not a block. `estimate_gpu_memory` raises for
    # it before this function is reached, because a workload with no rows
    # or no features is a caller error and not something the CPU path
    # would have run either.

    # Sparse input is not a block: `train_gpu_sparse` grows on the
    # compressed matrix through gpu_sparse.mojo. `BLOCK_SPARSE_INPUT` is
    # kept in the vocabulary because prediction still has no sparse device
    # kernel (`gpu_predict_support` raises it) and a serialized decision
    # from an earlier build may name it. Under `auto`, sparse input keeps
    # the CPU: see `decide_device`.

    if request.objective == CUSTOM:
        # `train_custom_gpu` does exist, and it grows trees on the device
        # while calling the gradient callback on the host. The `device`
        # vocabulary does not route to it: `fit` and `fit_multiclass` send
        # a custom objective to `train_custom`, and `train_gpu` itself
        # rejects the code. So this block is about the path the request
        # will actually take, not a claim that no GPU custom trainer
        # exists. Reaching that trainer is an explicit call, not a device
        # setting.
        blocks.add(
            BLOCK_CUSTOM_OBJECTIVE,
            String(
                "a custom objective's gradients come from a caller-supplied"
                " callable, and the device setting routes it to train_custom"
                " on the CPU; call train_custom_gpu directly to grow its"
                " trees on the device"
            ),
        )
    elif request.objective == LAMBDARANK:
        blocks.add(
            BLOCK_RANKING_OBJECTIVE,
            String(
                "objective 'lambdarank' takes its gradients from query"
                " groups and trains on the CPU only"
            ),
        )
    elif request.objective == MULTICLASS:
        # No block. `gpu_trains_objective` answers False for MULTICLASS, and
        # correctly: it asks whether `train_gpu` *itself* accepts the code,
        # and it does not. That is not the question this gate asks. The
        # question here is whether any backend covers the run, and
        # `objective_backends(MULTICLASS)` is CPU|GPU because
        # `train_multiclass_gpu` covers it -- reached through the `device`
        # setting, not around it: `model.fit_multiclass`,
        # `trainset.train_dataset_multiclass`, and
        # `external_memory.train_external_multiclass` all take `device` and
        # branch to it.
        #
        # This branch is what makes `-1` safe to declare. Without it, a
        # request that honestly says "softmax" is refused with "objective
        # code -1 is not one the built-in trainers implement", which is the
        # failure recorded in the comments in model.mojo and trainset.mojo
        # and measured by bench/real_data. Those call sites worked around it
        # by sending OBJECTIVE_UNSPECIFIED instead; the workaround costs
        # every multiclass run its objective gate. Fixed here so the honest
        # declaration is also the working one.
        pass
    elif not gpu_trains_objective(request.objective):
        blocks.add(
            BLOCK_UNKNOWN_OBJECTIVE,
            String(
                "objective code ",
                request.objective,
                " is not one the built-in trainers implement, so no backend"
                " covers it",
            ),
        )

    if request.uses_validation:
        blocks.add(
            BLOCK_VALIDATION_SET,
            String(
                "validation metrics are scored on the CPU, so a run with an"
                " eval set trains there too"
            ),
        )

    # --- The refusal sweep, 2026-08-16 ---
    #
    # Three parameters no GPU grower applies. They sit together, above the
    # kernel limits and below the objective gates, because each is a fact
    # about what the device path *implements* rather than about the shape:
    # the answer is the same at every size, so a report that named a
    # threshold would send the reader to the wrong knob. `decide_device`
    # reads the blocks before the crossover table, so `auto` takes the CPU
    # for these without any shape being compared, exactly as it does for
    # `BLOCK_DERIVATIVE_PRECISION`.
    #
    # The trainers refuse the same three by name (`_check_gpu_params` in
    # train_gpu.mojo and train_gpu_sparse.mojo). That is not a duplicate
    # policy: this module decides *where a run goes* and the trainers refuse
    # *a run that arrived anyway*, which is the only protection a caller who
    # reaches `train_gpu` directly has.
    if request.bundling:
        blocks.add(
            BLOCK_FEATURE_BUNDLING,
            String(
                "enable_bundle is applied by the dense CPU trainers in"
                " boosting.mojo, which build a bundled matrix before they"
                " grow; the GPU trainers build their histograms from the"
                " unbundled binned matrix and cannot apply it"
            ),
        )

    if request.linear_tree:
        blocks.add(
            BLOCK_LINEAR_TREE,
            String(
                "linear_tree fits an affine model in each leaf from the raw"
                " feature matrix, and the GPU trainers take a binned matrix"
                " only, so no accelerator path can produce those leaves"
            ),
        )

    if request.forced_splits:
        blocks.add(
            BLOCK_FORCED_SPLITS,
            String(
                "forced splits are applied by tree.grow_tree and by no other"
                " grower; the GPU growers never read the document, so a GPU"
                " fit would return an unforced tree"
            ),
        )

    # --- The second refusal sweep, 2026-08-16 evening ---
    #
    # Two CatBoost parameters, added the day they became reachable rather than
    # the day they were built. Both were harmless until then: `params.mojo`
    # refused `score_function` other than L2 by name, and `boosting_type`
    # never reached a fit. The CatBoost reachability work removed both
    # refusals at the Python surface, and neither has a device implementation,
    # so without these two blocks a `device='auto'` fit above the crossover
    # would have returned plain-boosting L2 answers under CatBoost labels.
    #
    # They are blocks and not rule scopes, for the reason spelled at
    # `BLOCK_DERIVATIVE_PRECISION`: an explicit `device='gpu'` raises, because
    # a backend the user named cannot honor what they asked for; `device=
    # 'auto'` takes the CPU with `DECISION_AUTO_CPU_BLOCKED`, because a user
    # who asked us to pick asked for the backend that *can* honor it.
    if request.ordered_boosting:
        # `ordered_boosting.mojo` holds the rung planes and the per-permutation
        # score ladder, and it is reached from `boosting.mojo` only. Neither
        # `train_gpu.mojo` nor `gpu_split_search.mojo` mentions `boosting_type`
        # at all, so a device fit does not partially implement this -- it does
        # plain boosting and reports success, which is precisely the leakage
        # ordered boosting exists to remove.
        #
        # Ordered boosting's own five refusals (bagging, GOSS, balanced
        # bagging, renewing objectives, continued training) are all about the
        # sampler and the objective. None of them is about the device, which
        # is why this one has to live here.
        blocks.add(
            BLOCK_ORDERED_BOOSTING,
            String(
                "boosting_type='ordered' fits each row from a model that never"
                " saw it, through the per-permutation rung planes in"
                " ordered_boosting.mojo, and no accelerator path builds those"
                " planes; a GPU fit would return ordinary plain boosting"
            ),
        )

    # --- score_function, narrowed 2026-08-17 ---
    #
    # This refused every non-L2 selector. The device now evaluates the Cosine
    # ratio directly: `gpu_cosine_score` at `gpu_split_search:986` with five
    # call sites, per node, and `_scan_slot_oblivious_kernel` carrying a
    # level's two cross-leaf accumulators and its single root. So the wide
    # refusal was withdrawing a path that works.
    #
    # Retired against the standing rule and not against the capability alone:
    # no downstream refusal still fires for the same fit.
    # `_device_search_semantics_supported` used to decline any active
    # `ExtraTreeParams`, which `score_function != L2` sets, so a narrowed
    # block here would have routed a Cosine fit to an accelerator that then
    # scanned on the host and reported `device='gpu'` truthfully. That gate
    # asks per parameter now, which is what makes this narrowing earnable.
    #
    # TWO CONDITIONS SURVIVE, and they are different in kind.
    if (
        request.score_function != SCORE_L2
        and request.score_function != SCORE_COSINE
    ):
        # An UNKNOWN selector, and this arm is why the old test was written
        # `!= SCORE_L2` rather than `== SCORE_COSINE`. The scan kernels
        # compute `var cosine = score_function == SCORE_COSINE` and treat
        # everything else as L2, so a third selector added later and not
        # taught to them would silently receive an L2 answer under its own
        # label. `check_score_function` raises on it, but only once a grower
        # reads it, and this gate answers before a backend is chosen.
        blocks.add(
            BLOCK_SCORE_FUNCTION,
            String(
                "score_function selects which functional of the children's"
                " sums is maximized, and this value is neither L2 nor Cosine."
                " The device scan kernels test for Cosine and treat every"
                " other code as L2, so an accelerator would answer under the"
                " wrong functional's name rather than refuse"
            ),
        )
    elif request.score_function == SCORE_COSINE and request.categorical:
        # Cosine beside a categorical column. The category partition search
        # scores with the L2 gain (`GpuSplitSearcher.set_score_function`, and
        # `split.find_best_split` before it), so allowing the pair puts two
        # score functions inside one argmax -- the winner of the partition
        # search chosen under L2, compared against numerical candidates
        # scored under Cosine.
        #
        # The declaration question rather than the searchable one, matching
        # `train_gpu._device_search_unsupported_reason`. Over-refusing a
        # CTR-replaced column costs a fit the accelerator it could have used;
        # under-refusing costs a tree scored by two functionals.
        # THE MESSAGE IS THE NARROWED REASON AND NOT THE WIDE ONE, and the old
        # text was wrong twice rather than once.
        #
        # (1) Until 2026-08-17 this string said the device accepted Cosine and
        # silently dropped it, which was true of the WIDE refusal this arm is
        # the remnant of. The guard narrowed to `and request.categorical` at
        # 820c06b and the text did not move, so a user with a categorical
        # column was told the accelerator ignores the setting, when the
        # accelerator applies it and the PAIR is what cannot be scored. A
        # refusal that misnames its cause sends the reader to the wrong knob:
        # the old text invites "wait for the device to implement Cosine",
        # which is already done.
        #
        # (2) It also ended "the CPU backend applies it", and the CPU does
        # not. `split.mojo:843` raises on exactly this pair with exactly this
        # reason, so `device='cpu'` is not an exit and telling a user it was
        # sent them to a fit that raises. This is a device-policy block, so it
        # never fires for a CPU request and nothing here ever caught that.
        #
        # The only real exits are the two named below, and CTR replacement is
        # the one CatBoost uses: a replaced column is not OFFERED to the scan,
        # which is what both refusals actually test.
        blocks.add(
            BLOCK_SCORE_FUNCTION,
            String(
                "score_function=Cosine cannot be combined with a categorical"
                " column, on either backend. The device kernels do evaluate"
                " the Cosine ratio, but a categorical candidate is a category"
                " SET chosen by a partition search that scores with the L2"
                " gain, so honoring the pair would put two functionals inside"
                " one argmax: the partition search's winner chosen under L2,"
                " then compared against numerical candidates scored under"
                " Cosine. Replace the categorical column with its CTR"
                " columns, which makes it numerical and is what CatBoost"
                " itself does, or set score_function=L2. Selecting"
                " device='cpu' does NOT lift this: split.find_best_split"
                " raises on the same pair for the same reason"
            ),
        )

    # --- random_strength, narrowed 2026-08-17 ---
    #
    # This refused every positive `random_strength`. What it stood in for was
    # the per-tree SCALE: the noise plane, its draw and its consumption were
    # staged, but no GPU round loop computed the standard deviation the draw
    # is multiplied by, so the product was zero on every fit and a device fit
    # would either refuse in the grower or train an unregularized model that
    # reported success.
    #
    # Both arms of `_train_gpu_rounds` compute it now -- the host-gradient arm
    # from the round's user-weighted derivatives before any sampler rewrites
    # them, the device-gradient arm from
    # `GpuObjectiveState.derivative_sum_squares` -- so the thing this block
    # stood in for is written, which is the only reason a refusal in this
    # package may be retired.
    #
    # THE DOWNSTREAM REFUSALS THAT MADE THIS WIDE ARE GONE, EACH BY NAME,
    # which is the standing rule rather than a summary of it:
    #
    #   the device split search declining any active ExtraTreeParams
    #       -> `_device_search_unsupported_reason` asks per parameter.
    #   the oblivious grower refusing a level draw
    #       -> retired; the site is the level depth in its own hash domain
    #          and the scale reaches the searcher before the route decision.
    #   the device-gradient arm beside a Bayesian bootstrap
    #       -> `ROUND_BAYESIAN_NOISE_SCALE` routes it to the host-gradient
    #          arm rather than raising, so it is a resolution and not a
    #          cliff this block was covering.
    #
    # ONE CONDITION SURVIVES.
    if request.random_strength > 0.0 and request.categorical:
        # `gpu_split_search` refuses the noise beside a categorical feature by
        # name: a categorical candidate is a category SET chosen by a
        # partition search, so only that search's winner would be noised
        # while every numerical feature had every candidate noised. That is a
        # different regularizer wearing the same parameter's name, which is
        # the same class of error as scaling the noise by a sampler.
        #
        # `> 0.0` rather than `!= 0.0`: a negative value is not a weaker
        # request, it is invalid, and `check_random_strength` is the thing
        # that says so with the right message.
        # Rewritten 2026-08-17 for the reasons given in full at the
        # `BLOCK_SCORE_FUNCTION` arm above, which had the identical defect in
        # the identical two ways. The old text claimed the device accepts the
        # noise and drops it, which described the WIDE refusal that c775959
        # narrowed, and it closed with "the CPU backend applies it", which is
        # false beside a categorical column: `split.mojo:816` raises on this
        # pair with this reason. Both halves are corrected here.
        blocks.add(
            BLOCK_RANDOM_STRENGTH,
            String(
                "random_strength cannot be combined with a categorical"
                " column, on either backend. The noise plane, its draw and"
                " the per-tree scale are all built and the device applies"
                " them to numerical candidates, but a categorical candidate"
                " is a category SET chosen by a partition search, and only"
                " that search's single winner would be noised while every"
                " numerical feature had every candidate noised. That is a"
                " different regularizer wearing this parameter's name."
                " Replace the categorical column with its CTR columns, which"
                " makes it numerical and is what CatBoost itself does, or set"
                " random_strength=0. Selecting device='cpu' does NOT lift"
                " this: split.find_best_split raises on the same pair for the"
                " same reason"
            ),
        )

    # --- The oblivious device route ---
    #
    # SCOPED TO WHAT THIS FUNCTION CAN SEE, AND NO WIDER. It is tempting to
    # refuse every `grow_policy=oblivious` request here, since `train_gpu`
    # raises on this route today. That would be wrong and it would be the
    # `BLOCK_SCORE_FUNCTION` mistake again: `oblivious_device_supported`
    # returns `OBLIVIOUS_OK` for a plain oblivious fit, so the device grows
    # symmetric trees and refusing them all would withdraw a working path.
    #
    # So each block below mirrors ONE named reason from that predicate, and
    # only the reasons a `DeviceRequest` actually carries.
    if request.grow_policy == GROW_OBLIVIOUS and (
        request.max_depth < 1 or request.max_depth > OBLIVIOUS_DEVICE_MAX_DEPTH
    ):
        # `gpu_resident_round:1645`, `OBLIVIOUS_DEPTH`. The oblivious plane
        # sizes every table it owns from `1 << max_depth` -- the slot pool,
        # the tree tables, the searcher's records, the plan -- so the bound
        # is an allocation fact rather than a policy choice, and there is no
        # depth bound at all on the leaf-wise or depth-wise device paths.
        #
        # The shipped default is `max_depth=6`, which is INSIDE this bound
        # and at its edge. So this block does not fire for the default; it
        # fires the moment anyone moves the default depth, which is exactly
        # when a cliff would otherwise appear.
        blocks.add(
            BLOCK_MAX_DEPTH,
            String(
                "grow_policy=oblivious sizes every device table from 1 <<",
                " max_depth, so the device plane takes max_depth in [1, 6]",
                " and this fit asks for ",
                request.max_depth,
                "; the CPU grower grows the same symmetric tree at any depth",
            ),
        )
    if request.grow_policy == GROW_OBLIVIOUS and request.categorical:
        # `gpu_resident_round:1647`, `OBLIVIOUS_CATEGORICAL`. Named
        # separately from the depth bound because a fit refused for its data
        # should not be told about its depth.
        blocks.add(
            BLOCK_GROW_POLICY,
            String(
                "grow_policy=oblivious cannot search a categorical column at"
                " any node on the device: a symmetric level commits one"
                " (feature, bin) split for the whole level and the device"
                " level search evaluates ordinal thresholds only; the CPU"
                " grower grows the same symmetric tree with categorical"
                " splits",
            ),
        )
    # WHAT IS STILL A CLIFF, NAMED RATHER THAN LEFT TO BE FOUND. The rest of
    # `oblivious_device_supported`'s refusals -- monotone constraints,
    # interaction constraints, `feature_fraction_bynode`,
    # `feature_fraction_bylevel`, an active `ExtraTreeParams` bundle,
    # speculative build, a level histogram that does not fit, and the record
    # count -- are not fields of a `DeviceRequest`, so no block here can see
    # them and `train_gpu:1780` still RAISES for them under `auto`.
    #
    # None of them is in the shipped default, and the two that were
    # (`score_function`, `random_strength`, both through
    # `ExtraTreeParams.is_active()`) have their own blocks above. So the
    # default does not cliff. A fit that combines oblivious growth with a
    # monotone constraint still does, and closing that means carrying those
    # fields on the request rather than widening either block above.

    if not gpu_supports_outputs(request.n_outputs):
        blocks.add(
            BLOCK_OUTPUT_LIMIT,
            String(
                "the GPU path does not cover ",
                request.n_outputs,
                " trees per boosting round",
            ),
        )

    if request.n_rows > caps.max_rows:
        blocks.add(
            BLOCK_ROW_LIMIT,
            String(
                request.n_rows,
                " rows is past the ",
                caps.max_rows,
                " the GPU kernels can index",
            ),
        )

    if request.bins_known() and (
        request.n_bins < caps.min_bins or request.n_bins > caps.max_bins
    ):
        blocks.add(
            BLOCK_BIN_LIMIT,
            String(
                "max_bin=",
                request.n_bins,
                " is outside the [",
                caps.min_bins,
                ", ",
                caps.max_bins,
                "] the binner and the GPU histogram kernels support",
            ),
        )

    # Invariant 5 on `MemoryEstimate`: an estimate built without a bin
    # count is a lower bound on an unknown quantity, so it may never
    # block. A budget of zero means unreported, which is not a budget.
    if caps.memory_budget_known() and memory.bins_known:
        if memory.device_bytes() > caps.profile.memory_budget_bytes:
            blocks.add(
                BLOCK_MEMORY_BUDGET,
                String(
                    "the estimated ",
                    memory.device_bytes(),
                    " bytes of training buffers does not fit the reported ",
                    caps.profile.memory_budget_bytes,
                    " byte budget",
                ),
            )

    return blocks^


def _collect_warnings(
    request: DeviceRequest, caps: DeviceCapabilities
) raises -> ReasonList:
    """Everything a reader should know about how much this decision is
    worth. Nothing here changes the selected backend.

    Everything below the availability check is about the accelerator, so
    an explicit `cpu` request collects none of it: a machine that happens
    to have a GPU should not narrate its capabilities at every CPU run.
    The incomplete-request warning is the exception, because a caller that
    under-described its workload wants to know that whatever device it got.
    """
    var warnings = ReasonList()

    if not request.is_complete():
        var missing = String("")
        if not request.objective_known():
            missing += String(" an objective")
        if not request.bins_known():
            if missing.byte_length() > 0:
                missing += String(" and")
            missing += String(" a bin count")
        warnings.add(
            WARN_INCOMPLETE_REQUEST,
            String(
                "the caller did not declare",
                missing,
                ", so the gates that depend on it were skipped and the"
                " memory estimate is partial",
            ),
        )

    if not caps.gpu_available or request.requested_device == CPU_DEVICE:
        return warnings^

    if caps.built_with_accelerator:
        warnings.add(
            WARN_BUILD_TIME_AVAILABILITY,
            String(
                "accelerator availability is resolved at compile time, so a"
                " redistributed build reports the device its builder had;"
                " set MOJOTREES_DISABLE_GPU=1 to pin such a build to the CPU"
            ),
        )

    if caps.profile_source == PROFILE_FALLBACK:
        warnings.add(
            WARN_UNKNOWN_HARDWARE,
            String(
                "no device attributes were read, so the conservative"
                " portable profile is in use and no crossover rule can be"
                " scoped to this hardware"
            ),
        )
    elif caps.profile_source == PROFILE_DECLARED:
        warnings.add(
            WARN_UNKNOWN_HARDWARE,
            String(
                "MOJOTREES_GPU_BACKEND declared api '",
                api_name(caps.profile.api),
                "' but no device attributes were read, so the capability"
                " numbers are still the portable fallback",
            ),
        )
    elif caps.profile_source == PROFILE_BUILD_TARGET:
        # Two separate facts, and a reader needs both. The identity is a
        # build property, which is what may select a backend and what may
        # therefore be wrong on a redistributed wheel; the numbers were never
        # read at all, which is what keeps the memory gate silent.
        warnings.add(
            WARN_BUILD_TARGET_HARDWARE,
            String(
                "the hardware identity used here (api '",
                api_name(caps.profile.api),
                "', apple generation '",
                apple_generation_name(caps.profile.apple_generation),
                "') is the accelerator this binary was compiled for, not one"
                " that was read from this machine; on a redistributed build"
                " set device='cpu' or MOJOTREES_DISABLE_GPU=1, or open a"
                " DeviceContext and go through"
                " decide_device_report_reported",
            ),
        )
        warnings.add(
            WARN_SYNTHETIC_CAPABILITIES,
            String(
                "only the api and generation came from the build target; the"
                " core count, threadgroup memory, and memory budget are the"
                " portable fallback and were not read from a device"
            ),
        )
    elif caps.profile_source == PROFILE_SYNTHETIC or caps.profile.synthetic:
        warnings.add(
            WARN_SYNTHETIC_CAPABILITIES,
            String(
                "these capabilities were constructed rather than read from a"
                " device, so they describe a fixture and not the hardware in"
                " front of you"
            ),
        )

    if not caps.memory_budget_known():
        warnings.add(
            WARN_MEMORY_BUDGET_UNKNOWN,
            String(
                "no device memory budget was reported, so memory cannot"
                " block this run and the partial-histogram budget falls back"
                " to the portable ceiling"
            ),
        )
    elif caps.profile.unified_memory:
        warnings.add(
            WARN_UNIFIED_MEMORY_BUDGET,
            String(
                "device memory is host memory on this backend, so the budget"
                " shown is shared with the dataset the host is holding"
            ),
        )

    # MULTICLASS is excluded for the same reason it is excluded from the
    # block above and for the same reason `objective_gradients_on_device`
    # answers False for it: that predicate is about the *single-output*
    # path's `fill_gradients_device`. Softmax derivatives have a device
    # kernel of their own inside `train_multiclass_gpu`, so warning that a
    # declared multiclass run fills gradients on the host would be a false
    # statement in the report, printed on every softmax fit.
    if (
        request.objective_known()
        and request.objective != MULTICLASS
        and not gpu_objective_is_device_resident(request.objective)
    ):
        warnings.add(
            WARN_HOST_GRADIENT_PATH,
            String(
                "objective code ",
                request.objective,
                " has no device-side derivative kernel, so its gradients are"
                " filled on the host and uploaded once per boosting round",
            ),
        )

    if caps.session.paid_nothing():
        warnings.add(
            WARN_COLD_SESSION,
            String(
                "no device context or kernel has been created in this"
                " process, so a GPU run here also pays context creation and"
                " first-launch kernel creation; that cost is startup, not"
                " training, and comparing it against a warm CPU run measures"
                " the wrong thing (warm-up level ",
                warmup_level_name(caps.session.warmup_level),
                ")",
            ),
        )

    # The transfer plan never blocks. A role that cannot take the requested
    # route falls back to the staged copy inside unified_memory_policy.mojo,
    # and the allocation site is what refuses; what belongs in a device
    # decision is that the run is not on the shipped route.
    if caps.transfer.any_unproven():
        warnings.add(
            WARN_UNPROVEN_TRANSFER_ROUTE,
            String(
                "MOJOTREES_GPU_TRANSFER_UNPROVEN=1 put at least one buffer on"
                " transfer route '",
                transfer_route_name(caps.transfer.requested),
                "', which has no installed evidence; any number measured"
                " under this must be reported with the flag",
            ),
        )
    if caps.transfer.needs_kernel_retirement():
        warnings.add(
            WARN_KERNEL_RETIREMENT_ROUTE,
            String(
                "at least one buffer is on a route the kernels read directly,"
                " so the host cannot refill it until every kernel that read"
                " it has retired and the staging ring's overlap does not"
                " apply; this is a synchronization change, not only an"
                " allocation one",
            ),
        )

    return warnings^


struct DeviceDecision(Copyable, Movable):
    """The decision, everything it rested on, and how to serialize it.

    `selected_device` is `CPU_DEVICE` or `GPU_DEVICE` when a backend was
    chosen, and `NO_DEVICE` when the request cannot run, in which case
    `blocked` is True and `message` holds what `raise_if_blocked` raises.

    `blocking_reasons` holds the reasons the GPU path is unavailable for
    this workload, whatever was requested. It is populated even when
    `cpu` was asked for and the answer was never in doubt, because "why
    would the GPU not have worked here" is a question a report should be
    able to answer without being asked twice.
    """

    var request: DeviceRequest
    var capabilities: DeviceCapabilities
    var selected_device: Int
    var blocked: Bool
    var decision_code: Int
    var message: String
    var blocking_reasons: ReasonList
    var warnings: ReasonList
    var memory: MemoryEstimate
    var policy_version: Int
    var evidence_id: String
    var crossover_citation: String
    """`CrossoverEvidence.cite()` for the rule that selected the GPU, and
    empty for every decision no rule decided.

    `evidence_id` alone says which record; this says which record *and*
    what device and shape it was taken on, which is the difference between
    a reader being able to check the claim and having to go looking for the
    measurement that backs it."""

    def __init__(
        out self,
        var request: DeviceRequest,
        var capabilities: DeviceCapabilities,
        selected_device: Int,
        blocked: Bool,
        decision_code: Int,
        var message: String,
        var blocking_reasons: ReasonList,
        var warnings: ReasonList,
        var memory: MemoryEstimate,
        var evidence_id: String,
        var crossover_citation: String = String(""),
    ):
        self.request = request^
        self.capabilities = capabilities^
        self.selected_device = selected_device
        self.blocked = blocked
        self.decision_code = decision_code
        self.message = message^
        self.blocking_reasons = blocking_reasons^
        self.warnings = warnings^
        self.memory = memory^
        self.policy_version = POLICY_VERSION
        self.evidence_id = evidence_id^
        self.crossover_citation = crossover_citation^

    def validated(self) -> Bool:
        """Whether a GPU selection rests on benchmark-derived evidence.
        False for every CPU selection, and False for a GPU reached through
        `MOJOTREES_AUTO_MIN_CELLS` or requested explicitly."""
        if self.selected_device != GPU_DEVICE:
            return False
        return self.decision_code == DECISION_AUTO_GPU_EVIDENCE

    def raise_if_blocked(self) raises:
        """Raise when the request cannot run, and return otherwise. This
        is what turns a report into the refusal an explicit `gpu` gets."""
        if self.blocked:
            raise Error(self.message)

    def serialize(self) raises -> String:
        """The decision as `key=value` lines.

        The wire format between this module and any formatter, Python
        included. One key per line, `=` separating key from value, values
        free of newlines. Repeated keys are lists in order: `block` for
        each blocking reason, `warning` for each warning, each rendered as
        `<code>:<name>:<message>`.

        A consumer that only cares about the answer reads `selected`.
        A consumer that wants to explain it reads the rest.
        """
        var out = String("policy_version=", self.policy_version, "\n")
        out += String("evidence_id=", self.evidence_id, "\n")
        # How many rules this build ships, which is a property of the build
        # and not of this decision, and so is reported for every decision
        # including the ones that never consulted the table. It is what
        # tells a report reading `no-validated-rule` whether the table was
        # empty or whether it was consulted and declined.
        out += String(
            "crossover_rules_installed=", len(crossover_rules()), "\n"
        )
        # Present only when a rule decided this run. An empty line here
        # would read as "a rule fired and cited nothing", which is exactly
        # what `CrossoverEvidence` refuses to let happen.
        if self.crossover_citation.byte_length() > 0:
            out += String("crossover_rule=", self.crossover_citation, "\n")
        out += String(
            "requested=", device_name(self.request.requested_device), "\n"
        )
        out += String("selected=", device_name(self.selected_device), "\n")
        out += String("blocked=", _bool_text(self.blocked), "\n")
        out += String(
            "decision=", decision_name(self.decision_code), "\n"
        )
        out += String("message=", self.message, "\n")
        out += String("validated=", _bool_text(self.validated()), "\n")

        out += String("n_rows=", self.request.n_rows, "\n")
        out += String("n_features=", self.request.n_features, "\n")
        out += String("n_outputs=", self.request.n_outputs, "\n")
        out += String("cells=", self.request.cells(), "\n")
        out += String("n_bins=", self.request.n_bins, "\n")
        out += String(
            "bins_known=", _bool_text(self.request.bins_known()), "\n"
        )
        out += String("objective=", self.request.objective, "\n")
        out += String(
            "objective_known=",
            _bool_text(self.request.objective_known()),
            "\n",
        )
        out += String("sparse=", _bool_text(self.request.sparse), "\n")
        out += String(
            "categorical=", _bool_text(self.request.categorical), "\n"
        )
        out += String(
            "has_missing=", _bool_text(self.request.has_missing), "\n"
        )
        out += String(
            "uses_validation=",
            _bool_text(self.request.uses_validation),
            "\n",
        )
        out += String("bundling=", _bool_text(self.request.bundling), "\n")
        out += String(
            "linear_tree=", _bool_text(self.request.linear_tree), "\n"
        )
        out += String(
            "forced_splits=", _bool_text(self.request.forced_splits), "\n"
        )
        # Serialized beside the other three request flags, not left out of the
        # record. A refusal a reader cannot see the input for is a refusal
        # they have to reproduce to understand, and this report exists so that
        # `device='gpu'` can be asked "what would you do here" without one.
        out += String(
            "ordered_boosting=",
            _bool_text(self.request.ordered_boosting),
            "\n",
        )
        out += String(
            "score_function=", self.request.score_function, "\n"
        )
        out += String(
            "random_strength=", self.request.random_strength, "\n"
        )
        out += String(
            "derivative_precision=", self.request.derivative_precision, "\n"
        )
        out += String("grow_policy=", self.request.grow_policy, "\n")
        out += String("max_depth=", self.request.max_depth, "\n")

        out += String(
            "gpu_available=",
            _bool_text(self.capabilities.gpu_available),
            "\n",
        )
        out += String(
            "built_with_accelerator=",
            _bool_text(self.capabilities.built_with_accelerator),
            "\n",
        )
        out += String(
            "disabled_by_env=",
            _bool_text(self.capabilities.disabled_by_env),
            "\n",
        )
        out += String(
            "profile_source=",
            profile_source_name(self.capabilities.profile_source),
            "\n",
        )
        out += String("api=", api_name(self.capabilities.profile.api), "\n")
        out += String(
            "apple_generation=",
            apple_generation_name(
                self.capabilities.profile.apple_generation
            ),
            "\n",
        )
        out += String(
            "core_count=", self.capabilities.profile.core_count, "\n"
        )
        out += String(
            "max_threads_per_block=",
            self.capabilities.profile.max_threads_per_block,
            "\n",
        )
        out += String(
            "max_shared_memory_per_block=",
            self.capabilities.profile.max_shared_memory_per_block,
            "\n",
        )
        out += String(
            "memory_budget_bytes=",
            self.capabilities.profile.memory_budget_bytes,
            "\n",
        )
        out += String(
            "unified_memory=",
            _bool_text(self.capabilities.profile.unified_memory),
            "\n",
        )
        out += String("max_rows=", self.capabilities.max_rows, "\n")
        out += String("min_bins=", self.capabilities.min_bins, "\n")
        out += String("max_bins=", self.capabilities.max_bins, "\n")
        out += String(
            "auto_min_cells=", self.capabilities.auto_min_cells, "\n"
        )
        out += String(
            "derivative_precision_float64=",
            _bool_text(self.capabilities.derivative_precision_float64),
            "\n",
        )
        out += String(
            "const_hessian_verify=",
            _bool_text(self.capabilities.const_hessian_verify),
            "\n",
        )

        out += String(
            "session_context_open=",
            _bool_text(self.capabilities.session.context_open),
            "\n",
        )
        out += String(
            "session_kernels_ready=",
            _bool_text(self.capabilities.session.kernels_ready),
            "\n",
        )
        out += String(
            "session_warmup_level=",
            warmup_level_name(self.capabilities.session.warmup_level),
            "\n",
        )

        out += String(
            "transfer_requested=",
            transfer_route_name(self.capabilities.transfer.requested),
            "\n",
        )
        out += String(
            "transfer_all_default=",
            _bool_text(self.capabilities.transfer.all_default()),
            "\n",
        )
        out += String(
            "transfer_ack_unproven=",
            _bool_text(self.capabilities.transfer.any_unproven()),
            "\n",
        )
        out += String(
            "transfer_unified_memory=",
            _bool_text(self.capabilities.transfer.unified_memory),
            "\n",
        )
        # One `transfer` line per buffer role, in role order, always all
        # present, rendered `<role>:<requested>:<selected>:<reason>:<evidence>
        # :<retire_on>`. A repeated key is a list in order, the same
        # convention `block` and `warning` use below, so a consumer that only
        # wants the headline reads `transfer_all_default` and one that wants
        # to explain a per-role fallback reads these.
        for i in range(len(self.capabilities.transfer.decisions)):
            var d = self.capabilities.transfer.decisions[i].copy()
            out += String(
                "transfer=",
                transfer_role_name(d.role),
                ":",
                transfer_route_name(d.requested),
                ":",
                transfer_route_name(d.selected),
                ":",
                transfer_block_name(d.reason),
                ":",
                transfer_evidence_name(d.evidence_level),
                ":",
                retire_event_name(d.contract.retire_event),
                "\n",
            )

        out += String(
            "memory_device_bytes=", self.memory.device_bytes(), "\n"
        )
        out += String(
            "memory_upper_bound_bytes=",
            self.memory.upper_bound_bytes(),
            "\n",
        )
        out += String("memory_host_bytes=", self.memory.host_bytes(), "\n")
        out += String(
            "memory_partial_budget_bytes=", self.memory.partial_budget, "\n"
        )
        out += String(
            "memory_estimate_complete=",
            _bool_text(self.memory.bins_known),
            "\n",
        )
        out += String(
            "memory_binned_matrix_bytes=",
            self.memory.binned_matrix_bytes,
            "\n",
        )
        out += String(
            "memory_leaf_id_bytes=", self.memory.leaf_id_bytes, "\n"
        )
        out += String(
            "memory_gradient_bytes=", self.memory.gradient_bytes, "\n"
        )
        out += String(
            "memory_hessian_bytes=", self.memory.hessian_bytes, "\n"
        )
        out += String(
            "memory_histogram_bytes=", self.memory.histogram_bytes, "\n"
        )
        out += String(
            "memory_feature_id_bytes=", self.memory.feature_id_bytes, "\n"
        )

        for i in range(self.blocking_reasons.count()):
            out += String(
                "block=",
                self.blocking_reasons.codes[i],
                ":",
                block_reason_name(self.blocking_reasons.codes[i]),
                ":",
                self.blocking_reasons.messages[i],
                "\n",
            )
        for i in range(self.warnings.count()):
            out += String(
                "warning=",
                self.warnings.codes[i],
                ":",
                warning_name(self.warnings.codes[i]),
                ":",
                self.warnings.messages[i],
                "\n",
            )
        return out^


def decide_device(
    request: DeviceRequest, caps: DeviceCapabilities
) raises -> DeviceDecision:
    """Resolve one request against one set of capabilities.

    Pure: it opens no device, reads no environment (the environment was
    already folded into `caps`), and touches no dataset. Every input is a
    value a caller could have serialized, which is what makes the whole
    policy testable on a machine with no accelerator and injectable with
    hardware nobody here owns.

    Never raises for a GPU request that cannot run. The decision carries
    `blocked` and `message` instead, and `raise_if_blocked` turns it into
    the refusal. That split is what lets a caller ask "what would
    device='gpu' do here" without handling an exception.

    Raises only for a device code outside the vocabulary and for a shape
    with no rows or no features, both of which are caller errors rather
    than policy outcomes.
    """
    if (
        request.requested_device != CPU_DEVICE
        and request.requested_device != GPU_DEVICE
        and request.requested_device != AUTO_DEVICE
    ):
        raise Error("unknown device code ", request.requested_device)

    var memory = estimate_gpu_memory(request, caps.profile)
    var blocks = _collect_blocks(request, caps, memory)
    var warnings = _collect_warnings(request, caps)

    # Filled in by one of the branches below, then handed to the single
    # constructor at the end. One construction site rather than nine keeps
    # the ownership transfers of `blocks`, `warnings`, and `memory` in one
    # place, where they can be read against the fields they land in.
    var selected: Int = CPU_DEVICE
    var blocked: Bool = False
    var code: Int = DECISION_EXPLICIT_CPU
    var message = String("")
    var evidence: String = EVIDENCE_NONE
    var citation = String("")

    if request.requested_device == CPU_DEVICE:
        message = String(
            "device='cpu' was requested, and the CPU path covers every"
            " objective and every input"
        )
    elif request.requested_device == GPU_DEVICE:
        if not blocks.is_empty():
            selected = NO_DEVICE
            blocked = True
            code = DECISION_GPU_REFUSED
            message = String(
                "device 'gpu' requested but ",
                blocks.first_message(),
                "; use device 'cpu' or 'auto'",
            )
        else:
            selected = GPU_DEVICE
            code = DECISION_EXPLICIT_GPU
            evidence = EVIDENCE_EXPLICIT
            message = String(
                "device='gpu' was requested and nothing blocks it, so"
                " training runs on the accelerator; an explicit request"
                " never falls back to the CPU"
            )
            warnings.add(
                WARN_EXPLICIT_GPU_UNMEASURED,
                String(
                    "device='gpu' was requested explicitly, so this run"
                    " rests on no crossover measurement and may be slower"
                    " than the CPU"
                ),
            )
    elif not blocks.is_empty():
        code = DECISION_AUTO_CPU_BLOCKED
        message = String(
            "auto chose the CPU because the GPU path cannot run this"
            " workload: ",
            blocks.first_message(),
        )
    elif caps.auto_min_cells >= 0:
        # An operator's threshold outranks the rule table: it exists so the
        # crossover benchmark can reach the GPU on a machine no rule covers.
        evidence = EVIDENCE_ENV
        if request.cells() >= caps.auto_min_cells:
            selected = GPU_DEVICE
            code = DECISION_AUTO_GPU_ENV_THRESHOLD
            message = String(
                "MOJOTREES_AUTO_MIN_CELLS=",
                caps.auto_min_cells,
                " and this workload has ",
                request.cells(),
                " cells, so the size heuristic selects the GPU",
            )
            warnings.add(
                WARN_ENV_THRESHOLD_UNVALIDATED,
                String(
                    "MOJOTREES_AUTO_MIN_CELLS is the knob for running the"
                    " crossover benchmark, not a validated threshold; this"
                    " GPU choice rests on no measurement"
                ),
            )
        else:
            code = DECISION_AUTO_CPU_BELOW_ENV_THRESHOLD
            message = String(
                "MOJOTREES_AUTO_MIN_CELLS=",
                caps.auto_min_cells,
                " and this workload has only ",
                request.cells(),
                " cells, so auto keeps the CPU",
            )
    elif request.sparse:
        # The crossover table is measured on dense matrices and the sparse
        # GPU trainer reads a different structure with different per-node
        # costs (docs/GPU_SPARSE_CATEGORICAL_DESIGN.md section 10), so no
        # dense rule transfers to it and none is claimed. Explicit `gpu`
        # runs it; `MOJOTREES_AUTO_MIN_CELLS` above reaches it, which is
        # what a crossover benchmark needs.
        code = DECISION_AUTO_CPU_NO_EVIDENCE
        message = String(
            "auto keeps the CPU for sparse input: the sparse GPU trainer's"
            " crossover against the CPU sparse trainer is unmeasured. Set"
            " device='gpu' to force it, or MOJOTREES_AUTO_MIN_CELLS to run"
            " that benchmark"
        )
    else:
        var rules = crossover_rules()
        var matched = -1
        for i in range(len(rules)):
            if rules[i].matches(caps, request):
                matched = i
                break
        if matched >= 0:
            selected = GPU_DEVICE
            code = DECISION_AUTO_GPU_EVIDENCE
            evidence = rules[matched].evidence_id.copy()
            citation = rules[matched].cite()
            message = String(
                "crossover rule '",
                rules[matched].name,
                "' covers this device and workload, so auto selects the"
                " GPU; measured on ",
                rules[matched].measured_on,
                " and recorded in ",
                rules[matched].evidence_id,
            )
        elif len(rules) > 0:
            code = DECISION_AUTO_CPU_BELOW_EVIDENCE
            message = String(
                "none of the ",
                len(rules),
                " crossover rule(s) in policy version ",
                POLICY_VERSION,
                " covers this device and workload, so auto keeps the CPU",
            )
            # Which half of "does not cover" applies is the first thing a
            # reader wants and the last thing they can work out from the
            # sentence above, because a fallback profile fails the hardware
            # scope of every rule before the shape is ever compared.
            if (
                caps.profile_source == PROFILE_FALLBACK
                or caps.profile_source == PROFILE_DECLARED
            ):
                message += String(
                    ". No device attributes were read (profile source '",
                    profile_source_name(caps.profile_source),
                    "'), so every hardware-scoped rule was out of reach"
                    " whatever the shape; a caller holding an open"
                    " DeviceContext should read its attributes and go"
                    " through decide_device_report_reported",
                )
            else:
                message += String(
                    ". The device was identified (api '",
                    api_name(caps.profile.api),
                    "', apple generation '",
                    apple_generation_name(caps.profile.apple_generation),
                    "'), so what no rule covers is this device with this"
                    " objective at ",
                    request.n_rows,
                    " x ",
                    request.n_features,
                    " and ",
                    request.n_outputs,
                    " tree(s) per round",
                )
        else:
            code = DECISION_AUTO_CPU_NO_EVIDENCE
            message = String(
                "the crossover table (policy version ",
                POLICY_VERSION,
                ") is empty: no benchmark has established a workload size"
                " where GPU training beats CPU training, so auto"
                " conservatively keeps the CPU. Set MOJOTREES_AUTO_MIN_CELLS"
                " to run that benchmark, or device='gpu' to force the"
                " accelerator",
            )

    return DeviceDecision(
        request.copy(),
        caps.copy(),
        selected,
        blocked,
        code,
        message^,
        blocks^,
        warnings^,
        memory^,
        evidence^,
        citation^,
    )


def resolve_device(
    device: Int,
    n_rows: Int,
    n_features: Int,
    n_outputs: Int = 1,
    objective: Int = OBJECTIVE_UNSPECIFIED,
    ordered_boosting: Bool = False,
    score_function: Int = SCORE_L2,
    random_strength: Float64 = 0.0,
    derivative_precision: Int = DERIV_PRECISION_FLOAT32,
    grow_policy: Int = GROW_LEAFWISE,
    max_depth: Int = 0,
) raises -> Int:
    """Resolve a requested device to the backend that will actually run:
    `CPU_DEVICE` or `GPU_DEVICE`, never `AUTO_DEVICE`.

    The narrow entry point the trainers use, in terms of the engine above:
    it builds a request the caller has only partly described (no bin count,
    no input flags), detects capabilities, and raises rather than returning a
    refusal. Callers that can describe the whole workload should build a
    `DeviceRequest` and call `decide_device` directly, or call
    `resolve_device_full`, which is what gets them the memory gate, the
    bin-limit gate, the sparse and validation blocks, and a report.

    WHY `objective` IS HERE, AND WHY IT DEFAULTS TO UNDECLARED. Every
    installed crossover rule is scoped to the objective it was measured on,
    and `CrossoverEvidence.matches` declines an `OBJECTIVE_UNSPECIFIED`
    request rather than letting it inherit a squared-error measurement. That
    is the right rule and it is not being weakened: a caller that did not say
    what it is training has not earned a claim measured on something else. It
    does mean that a caller which leaves this defaulted can never reach the
    GPU on evidence, whatever the shape and whatever the hardware.

    WHAT `auto` DOES UNDER `MOJOTREES_DERIVATIVE_PRECISION=float64`, WHICH IS
    NOT A THRESHOLD COMPARISON. It returns `CPU_DEVICE`, at every shape, on
    every machine, including shapes far above `AUTO_GPU_MIN_ROWS`. The GPU
    gradient upload narrows every per-row derivative to Float32
    (`gpu_gradient_stream.stage_gradients`), so a Float64 request is
    something the accelerator cannot do rather than something it does
    slowly. Precision is a capability, so it enters through
    `_collect_blocks`, and `decide_device` consults the blocks before it
    consults the crossover table: the shape is never compared and the
    250,000-row floor never comes into it. A reader looking for the
    threshold that produced this answer will not find one, and that is
    correct.

    `device='gpu'` under the same setting raises instead, and the asymmetry
    is the point. A caller who named a backend is told it cannot honor the
    request; a caller who asked us to pick is given the backend that can.
    Softening the explicit refusal into a fallback would restore the defect
    this block exists for, which was an accepted flag and a silently
    Float32 answer.

    Every caller in the tree leaves it defaulted today, and each of them is
    holding the objective when it does. `model.fit` takes `objective` as a
    parameter and drops it here; so do `model.fit_multiclass`,
    `trainset.train_dataset`, `trainset.train_dataset_multiclass`,
    `external_memory.train_external`, `external_memory.train_external_
    multiclass`, and, on the Python side, `_Config.binding_params`, which
    writes `params["objective"]` two statements after it resolves the device
    without one. The parameter is added here with a backward-compatible
    default so that closing that is one word at each site rather than a
    signature change; the sites themselves belong to other lanes.
    """
    var request = DeviceRequest(
        device,
        n_rows,
        n_features,
        n_outputs,
        BINS_UNSPECIFIED,
        objective,
        # Keyword-passed, and the reason is on the record. `gain_form` was
        # dropped from `GpuSplitSearcher.search` for months because it sat in
        # a positional list that something was later inserted above. These two
        # are the last parameters of a thirteen-argument constructor, which is
        # the position that mistake happens in.
        ordered_boosting=ordered_boosting,
        score_function=score_function,
        random_strength=random_strength,
        derivative_precision=derivative_precision,
        grow_policy=grow_policy,
        max_depth=max_depth,
    )
    var caps = DeviceCapabilities.detect()
    var decision = decide_device(request, caps)
    decision.raise_if_blocked()
    return decision.selected_device


# --- Flat entry points ------------------------------------------------
#
# `decide_device` takes structs, which is right for a Mojo caller and wrong
# for every other caller this policy has. A CPython binding builds arguments
# out of `PythonObject`s and cannot construct a `DeviceRequest`; a C API and
# a CLI are in the same position. Without a flat seam each of them grows its
# own argument marshalling, and the one that gets it slightly wrong is a
# second policy nobody meant to write.
#
# So the flat forms live here, beside the engine, and every one of them is a
# marshaller: it normalizes sentinels, builds the two structs, and calls
# `decide_device`. None of them decides anything, and none of them may.


def _normalized_bins(n_bins: Int) -> Int:
    """A bin count off a flat boundary, with "undeclared" normalized.

    Anything nonpositive is undeclared. The boundary carries plain ints, so
    an undeclared value needs a value, and Python sends `-1`
    (`_BINS_UNSPECIFIED` in python/mojotrees/device_selection.py). Zero
    arrives from a caller that left a field default. Neither is a bin count
    the binner would accept, and treating either as one would put a fabricated
    number into the memory estimate.
    """
    if n_bins <= 0:
        return BINS_UNSPECIFIED
    return n_bins


def _normalized_objective(objective: Int) -> Int:
    """An objective code off a flat boundary, with "undeclared" normalized.

    Below `-1` is undeclared. `-1` is deliberately excluded: it is the
    multiclass marker and a real code, so folding it into the undeclared
    sentinel would silently skip the objective gate for every multiclass run.

    That `-1` branch was dead from Python until 2026-08-16.
    `decide_device_workload` in bindings/basic_bindings.mojo folded every
    negative objective to `OBJECTIVE_UNSPECIFIED` one statement before
    calling `decide_device_report`, so this function never saw a `-1` that
    came from a Python caller and the two marshallers disagreed on exactly
    that value. The binding no longer folds; this is the only normalizer,
    and it is now the only place the `-1`/`-2` distinction is decided.
    """
    if objective < -1:
        return OBJECTIVE_UNSPECIFIED
    return objective


def capabilities_from_reported(
    reported_api: String,
    reported_arch: String,
    core_count: Int,
    max_threads_per_block: Int,
    max_shared_memory_per_block: Int,
    memory_budget_bytes: Int = 0,
    context_open: Bool = True,
    kernels_ready: Bool = False,
    warmup_level: Int = 0,
) raises -> DeviceCapabilities:
    """Capabilities from attributes a caller read off an open `DeviceContext`.

    The call site apple_gpu_policy.mojo's `GpuProfile.from_reported`
    docstring promises and this module's `PROFILE_REPORTED` describes, in one
    function, so an owner of a `DeviceContext` needs no knowledge of how the
    two layers fit together: query the attributes, pass them here, hand the
    result to `decide_device`.

    `context_open` defaults True because a caller with attributes to report
    has a context open by construction. `kernels_ready` defaults False
    because having a context says nothing about whether any kernel has been
    created, and reporting an unpaid cost as paid is the failure
    `SessionState` exists to prevent.
    """
    var profile = GpuProfile.from_reported(
        reported_api,
        reported_arch,
        core_count,
        max_threads_per_block,
        max_shared_memory_per_block,
        memory_budget_bytes,
    )
    return DeviceCapabilities.from_profile(
        profile^,
        PROFILE_REPORTED,
        SessionState(context_open, kernels_ready, warmup_level),
    )


def decide_device_report(
    device: String,
    n_rows: Int,
    n_features: Int,
    n_outputs: Int = 1,
    n_bins: Int = BINS_UNSPECIFIED,
    objective: Int = OBJECTIVE_UNSPECIFIED,
    sparse: Bool = False,
    categorical: Bool = False,
    has_missing: Bool = False,
    uses_validation: Bool = False,
    bundling: Bool = False,
    linear_tree: Bool = False,
    forced_splits: Bool = False,
    ordered_boosting: Bool = False,
    score_function: Int = SCORE_L2,
    random_strength: Float64 = 0.0,
    derivative_precision: Int = DERIV_PRECISION_FLOAT32,
    grow_policy: Int = GROW_LEAFWISE,
    max_depth: Int = 0,
) raises -> String:
    """The whole contract across a flat boundary: workload in, serialized
    decision out.

    This is the entry point `python/mojotrees/device_selection.py` calls
    `decide_device` and the one that moves it off its `"narrow"` contract.
    The exact binding is in handoffs/connect_05_device_policy.md.

    Never raises for a workload it refuses. A refusal is `blocked=true` in the
    returned lines with `message=` saying why, so a caller can ask "what would
    `device='gpu'` do here" without handling an exception, and the caller that
    wants the exception calls `resolve_device_full` instead. It raises only
    for a device name outside the vocabulary, a shape with no rows or no
    features, and an unparsable `MOJOTREES_GPU_TRANSFER`: caller and operator
    errors, not policy outcomes.

    Capabilities are detected, so the hardware profile is the portable
    fallback and the decision says so through `profile_source=fallback`. A
    caller holding an open `DeviceContext` should use
    `decide_device_report_reported`, which is the same function over
    attributes it actually read.
    """
    var request = DeviceRequest(
        parse_device(device),
        n_rows,
        n_features,
        n_outputs,
        _normalized_bins(n_bins),
        _normalized_objective(objective),
        sparse,
        categorical,
        has_missing,
        uses_validation,
        bundling,
        linear_tree,
        forced_splits,
        ordered_boosting=ordered_boosting,
        score_function=score_function,
        random_strength=random_strength,
        derivative_precision=derivative_precision,
        grow_policy=grow_policy,
        max_depth=max_depth,
    )
    var caps = DeviceCapabilities.detect(SessionState.from_env())
    return decide_device(request, caps).serialize()


def decide_device_report_reported(
    device: String,
    n_rows: Int,
    n_features: Int,
    n_outputs: Int,
    n_bins: Int,
    objective: Int,
    sparse: Bool,
    categorical: Bool,
    has_missing: Bool,
    uses_validation: Bool,
    reported_api: String,
    reported_arch: String,
    core_count: Int,
    max_threads_per_block: Int,
    max_shared_memory_per_block: Int,
    memory_budget_bytes: Int = 0,
    context_open: Bool = True,
    kernels_ready: Bool = False,
    warmup_level: Int = 0,
    bundling: Bool = False,
    linear_tree: Bool = False,
    forced_splits: Bool = False,
    ordered_boosting: Bool = False,
    score_function: Int = SCORE_L2,
    random_strength: Float64 = 0.0,
    derivative_precision: Int = DERIV_PRECISION_FLOAT32,
    grow_policy: Int = GROW_LEAFWISE,
    max_depth: Int = 0,
) raises -> String:
    """`decide_device_report` for a caller that has read the device.

    The same engine over `PROFILE_REPORTED` capabilities instead of the
    portable fallback, which is what makes the memory gate, the unified-memory
    warning, and any future hardware-scoped crossover rule mean anything. The
    decision it returns carries `profile_source=reported`, so a report from
    this entry point can be told from one that guessed.
    """
    var request = DeviceRequest(
        parse_device(device),
        n_rows,
        n_features,
        n_outputs,
        _normalized_bins(n_bins),
        _normalized_objective(objective),
        sparse,
        categorical,
        has_missing,
        uses_validation,
        bundling,
        linear_tree,
        forced_splits,
        ordered_boosting=ordered_boosting,
        score_function=score_function,
        random_strength=random_strength,
        derivative_precision=derivative_precision,
        grow_policy=grow_policy,
        max_depth=max_depth,
    )
    var caps = capabilities_from_reported(
        reported_api,
        reported_arch,
        core_count,
        max_threads_per_block,
        max_shared_memory_per_block,
        memory_budget_bytes,
        context_open,
        kernels_ready,
        warmup_level,
    )
    return decide_device(request, caps).serialize()


def resolve_device_full(
    device: Int,
    n_rows: Int,
    n_features: Int,
    n_outputs: Int = 1,
    n_bins: Int = BINS_UNSPECIFIED,
    objective: Int = OBJECTIVE_UNSPECIFIED,
    sparse: Bool = False,
    categorical: Bool = False,
    has_missing: Bool = False,
    uses_validation: Bool = False,
    bundling: Bool = False,
    linear_tree: Bool = False,
    forced_splits: Bool = False,
    ordered_boosting: Bool = False,
    score_function: Int = SCORE_L2,
    random_strength: Float64 = 0.0,
    derivative_precision: Int = DERIV_PRECISION_FLOAT32,
    grow_policy: Int = GROW_LEAFWISE,
    max_depth: Int = 0,
) raises -> Int:
    """Resolve a fully described workload to `CPU_DEVICE` or `GPU_DEVICE`,
    raising the refusal when it cannot run.

    What `resolve_device` should have been, and what a trainer that knows its
    own objective, bin count, and input flags should call. `resolve_device`
    describes only the shape, so it skips the objective gate, the bin-limit
    gate, the sparse and validation blocks, and the memory gate, and the
    decision it produces carries `WARN_INCOMPLETE_REQUEST` saying so. That is
    not a second policy, it is the same engine asked a smaller question, but
    the smaller question is one an explicit `device='gpu'` can pass and then
    fail deeper in, which is precisely the outcome the refusal exists to
    prevent.

    The exact call-site edits for `model.mojo` and `trainset.mojo` are in
    handoffs/connect_05_device_policy.md. They are one lane over, so they are
    a patch request rather than an edit.
    """
    var request = DeviceRequest(
        device,
        n_rows,
        n_features,
        n_outputs,
        _normalized_bins(n_bins),
        _normalized_objective(objective),
        sparse,
        categorical,
        has_missing,
        uses_validation,
        bundling,
        linear_tree,
        forced_splits,
        ordered_boosting=ordered_boosting,
        score_function=score_function,
        random_strength=random_strength,
        derivative_precision=derivative_precision,
        grow_policy=grow_policy,
        max_depth=max_depth,
    )
    var caps = DeviceCapabilities.detect(SessionState.from_env())
    var decision = decide_device(request, caps)
    decision.raise_if_blocked()
    return decision.selected_device


def describe_device_decision(decision: DeviceDecision) raises -> String:
    """One line for benchmark output and bug reports.

    The prose report belongs to whoever is formatting for a human;
    `serialize()` is what they should parse. This is the terse form for a
    log line that has one line to spend.

    `evidence=` is the bare identifier for every decision no crossover rule
    decided, and the full citation (identifier plus the device and shape it
    was measured on) for the ones a rule did. A GPU chosen automatically is
    the one answer here a reader is entitled to check, so it is the one
    that spends the extra characters."""
    var cited = decision.evidence_id.copy()
    if decision.crossover_citation.byte_length() > 0:
        cited = decision.crossover_citation.copy()
    var out = String(
        "device ",
        device_name(decision.request.requested_device),
        " -> ",
        device_name(decision.selected_device),
        " (",
        decision_name(decision.decision_code),
        ") policy=",
        decision.policy_version,
        " evidence=",
        cited,
        " rows=",
        decision.request.n_rows,
        " features=",
        decision.request.n_features,
        " api=",
        api_name(decision.capabilities.profile.api),
        " profile=",
        profile_source_name(decision.capabilities.profile_source),
        " session=",
        decision.capabilities.session.describe(),
    )
    if not decision.capabilities.transfer.all_default():
        out += String(
            " transfer=",
            transfer_route_name(decision.capabilities.transfer.requested),
        )
    if decision.capabilities.transfer.any_unproven():
        out += String(" ack_unproven=1")
    if not decision.blocking_reasons.is_empty():
        out += String(
            " blocked_by=",
            block_reason_name(decision.blocking_reasons.codes[0]),
        )
    return out^
