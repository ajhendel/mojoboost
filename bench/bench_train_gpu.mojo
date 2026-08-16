"""End-to-end CPU vs GPU training benchmark for mojotrees.

Generates the same deterministic synthetic dataset as bench_train.mojo
(counter-based splitmix64), bins it once, then times complete boosted
training — including GPU initialization and all transfers — through any
combination of the CPU trainer (`train`) and the GPU trainer (`train_gpu`)
under a chosen split-search strategy, with the library defaults
(LightGBM-matched). Also reports each model's training loss so throughput is
never read apart from fit quality.

Arms alternate inside a single process rather than running blocked, because
adjacent samples are the only comparable ones on a thermally variable
machine: back-to-back repeats here have agreed to a fraction of a percent
while the same binary minutes apart drifted by multiples. Two arms timed in
separate invocations cannot settle a few-percent question no matter how many
decimal places they print, so the summary reports each arm's own spread and
refuses to call a gap smaller than that spread a result.

LightGBM is one of those arms. It used to be the one comparison in this
repository that broke the rule the rest of the file is built on: every
LightGBM number recorded under bench/results was a single sample taken by
`bench/bench_lightgbm.py` in a separate process at a different moment, so
the headline margin against it was quoted with our spread beside it and the
comparator's spread not measured at all. A single sample has a spread of
zero by construction rather than a noise floor of zero, and this machine's
*measured* drift across time windows runs to factors of two and three, which
is two orders of magnitude above the margin being claimed. The `lightgbm`
arm closes that: it reaches LightGBM through Python interop, in this
process, in this loop, alternating with the mojotrees arms, under the same
repeat count, the same minimum-with-spread reduction, and the same
resolved-versus-indistinguishable verdict. It needs the bench environment,
which is where LightGBM lives, and it is the same task with the environment
named:

    pixi run -e bench bench-train-gpu 1000000 50 reg 5 \\
        gpu-device-depth,lightgbm

What that makes comparable is the time window, which was the hole. What it
does not make comparable is worth stating in the same breath. The two
engines still generate the dataset separately, from the same counter-based
splitmix64 sequence and to bit-identical values, but through different code.
Only the boosting run is inside the clock on both sides: our arms are handed
an already binned matrix and LightGBM trains on an already constructed
Dataset, with both binning times reported separately and neither folded in.
LightGBM's thread count is pinned by MOJOTREES_LGBM_THREADS or, failing
that, MOJOTREES_NUM_WORKERS, and nothing here can check that a given number
buys the same amount of machine on both sides, so the resolved value is
printed rather than assumed. Loading LightGBM into this process also changes
this process, by the memory its regenerated feature matrix holds and by the
thread pool it parks between repeats, so an arm timing from a run containing
a `lightgbm` arm is a *derived* quantity of that run and is not
interchangeable with the same arm's timing from a run without one. Compare
inside a run, which is what this file has always asked for.

The arm is single output. `reg` and `binary` are what it takes, because the
multiclass arms bucket the signal with a rule that exists only here, and
`lightgbm-depth` is refused because LightGBM has no depth-wise grow policy
to select.

Each arm names its split-search strategy as an argument to `train_gpu`, not
through MOJOTREES_GPU_SPLIT_STRATEGY, so a mistyped or word-split shell
export cannot leave one arm running the other arm's code path under the
wrong label. `gpu` is whatever SPLIT_SEARCH_AUTO resolves to for the
workload; `gpu-host` and `gpu-device` pin the choice.

CPU-side threading honors MOJOTREES_NUM_WORKERS / MOJOTREES_PARALLEL_MIN_OPS
(see parallel.mojo), so pin those for reproducible comparisons.

The first repeat of the first GPU arm also pays one-time device setup, which
is why the summary leads with the minimum rather than the mean.

Every arm, LightGBM included, reports its own repeats back three ways: the
interleaved `run N <arm> train_s` lines as they happen, an
`<arm>_train_s_samples` line holding that arm's repeats alone in the order
they ran, and a one-line `json_summary` record carrying the same lists.
`MOJOTREES_BENCH_JSON=<path>` writes that record to a file as well. The
duplication is on purpose: the interleaved lines are the only ones that show
the alternation, and they are also the ones a reader has to unpick n lines at
a time to recover a single arm's dispersion, which is how a spread stops
being copied into a table. Two spreads are printed per arm and they answer
different questions -- `<arm>_spread_pct` is (max - min) / min, unchanged,
and is what the verdict below is computed against; `<arm>_spread_pct_of_median`
is (max - min) / median, which is the arm's dispersion rather than its
excursion above its own best sample and is the one to quote beside a median.

Usage: mojo run -I src bench/bench_train_gpu.mojo \\
    [n_rows] [n_features] [reg|binary|multi[:classes]] [repeats] [arms] [seed]

`multi` is softmax multiclass and defaults to 7 classes, which is
covertype's shape; `multi:3` or `multi:10` names another count. The class
count rides on the objective word rather than taking a positional argument
of its own so that every documented reg and binary invocation keeps its
argument positions. It exists because there was no way to measure whether
GPU multiclass is faster or slower than CPU multiclass: the two trainers
are `train_multiclass` and `train_multiclass_gpu`, and until
`trainset.train_dataset_multiclass` was fixed, a caller reaching multiclass
through a `Dataset` got the CPU trainer whatever it asked for, so a GPU
multiclass timing taken through that path was a CPU timing wearing a GPU
label. That question is open and this arm is what settles it.

Read the multiclass arms with two differences from the single-output ones
in mind. Softmax grows one tree per class per round on both backends, so
`n_trees` is rounds times classes and is not comparable to a reg or binary
arm's; and the reported loss is multiclass log loss, not squared error or
binary log loss, so it is comparable across arms of the same run and across
nothing else.

Every run is also bracketed by the **regime canary** (`bench/canary.mojo`): a
fixed CPU probe and a fixed GPU probe, run once before the first arm and once
after the last, reported as `canary_cpu_ratio` and `canary_gpu_ratio` against
the baselines in `bench/canary_baseline.json` and never averaged together.
The probes touch nothing this file varies -- no dataset, no thread count, no
trainer -- so they move only when the machine does. That is the point: this
file's whole design rests on adjacent samples being comparable, and until now
nothing in it could tell you whether that assumption held. When the start and
end readings disagree by more than the canary's threshold, the output says so
in capitals, because it means the arms in between were not all taken on the
same machine and the verdict lines below are not usable. Sessions before this
existed inferred their regime by hand from effect sizes, and one such
attribution was made and retracted the same night.

Two consequences worth stating rather than discovering. **With no baseline
recorded the canary prints raw milliseconds and says the ratio is
unavailable**, which is the designed behavior; `bench/bench_canary.mojo`
establishes the baselines and must be run in a certified quiet window.
**Opening a device for the canary before the first arm shifts some one-time
GPU setup out of the first GPU repeat**, which the summary already leads with
the minimum to defend against, but it does mean an arm timing from a
canary-bearing run is not interchangeable with one recorded before the canary
existed. `MOJOTREES_CANARY=0` turns it off for exactly that comparison, and
the off state is printed and recorded rather than left silent.

`MOJOTREES_PHASE_PROFILE=async` on any run prints one `phase_profile` block
per arm per repeat: wall time, call counts, kernel or dispatch counts, host
synchronizations, rows, row slots, and histogram cells, per phase and per
node size class (phase_profile.mojo). It covers the `cpu` arm and the `gpu*`
arms under the same phase names, so the two backends' histograms, partitions,
score updates, and split searches sit on the same lines and can be subtracted.
Use `fenced` instead to separate device execution from enqueue, at the cost of
two extra waits per split; that changes the schedule, so a fenced number and
an async number are not comparable and no arm timing should be quoted from a
fenced run. Both are off by default, and an off run reads no clock, writes no
counter, and prints nothing.

`row-unroll-on` and `row-unroll-off` are the end-to-end pair for the
histogram row walk: how many rows one thread keeps in flight inside the
histogram row loop, `HIST_ROW_UNROLL` against one
(`GpuActiveRows.set_row_unroll`, reached through `train_gpu`'s `row_unroll`
argument). Both run under SPLIT_SEARCH_AUTO, so the pair holds every
condition but the knob constant. `gpu-unroll` and `gpu-nounroll` are accepted
as the same two arms under their older names, and any GPU arm takes a
`-unroll` or `-nounroll` suffix when the strategy needs pinning too
(`gpu-device-nounroll`, `gpu-device-depth-nounroll`). A `-unroll` suffix
prints under the plain arm's name, because on *is* the default and inventing
a second key for an identical configuration would put one condition in the
record twice; the pair aliases keep their own labels because there the two
names are the point.

**The arm cannot change a model.** Both walks visit the same rows of the same
range and add the same fixed-point integers into the same bins, so the
histograms are identical, so is every split chosen from them, and so is the
fit. What differs is instruction count and how many memory requests a thread
has outstanding, against a higher live register count and a threadgroup
residency this backend does not let anyone query.

`bench_histogram.mojo` already A/Bs the same knob in isolation, and this pair
exists **because an isolated histogram win is a hypothesis about a fit and
not a result about one**. The row-tile floor is the case in point: it
measured well in isolation on this repo and was a 22 to 36 percent regression
across a whole fit, because the isolated shape did not carry the partial
traffic a real round does. Run both. If the two benchmarks disagree, the
disagreement is the finding and neither number supersedes the other.

Three more launch-shape pairs came off the K1 hist-latency lane and are
wired the same way, each against its own control and **each measurable on
its own**, because three changes timed together cannot be attributed to any
one of them:

    narrow-index-on   / narrow-index-off    the two data-dependent indices
                                            in the histogram row loop, Int32
                                            against Int (`narrow_index`)
    pair-align-on     / pair-align-off      the width-2 load of the quantized
                                            gradient pair, with the 8-byte
                                            alignment its address has stated
                                            or left at align 4
                                            (`pair_alignment`)
    row-tiles-default / row-tiles-1 / row-tiles-2 / row-tiles-4 /
    row-tiles-8                             the per-node row-tile geometry
                                            (`rows_per_tile`)

None of them can change a model, for the reason the row-walk pair cannot:
every arm accumulates the same fixed-point Int32 values into the same bins,
and integer addition is associative and commutative. Each arm prints its
training loss, which is how a reader confirms that rather than taking it.
The index-width arm is the one exception worth naming: it is exact *under a
bound on the dataset shape*, `n_features * n_rows` and `2 * n_rows` both
below `Int32.MAX`, and a shape outside that bound is refused here rather
than run.

**The tile arms and their direction.** `N` in `row-tiles-N` is the tile
count at the **root**: the arm sets a fixed tile length of
`ceil(n_rows / N)`, so a node of `m` rows gets `ceil(m / length)` tiles.
Larger N is more and shorter tiles, which is the direction an earlier
device-wide floor of 80 tiles went; that experiment *measured* 22 percent
slower at 50 features and 36 percent slower at 100 across a whole fit and
was reverted. Smaller N is fewer and longer tiles and **has never been
tested**, which is what `row-tiles-1` is for. The sweep moves one parameter
only: `min_tiles` is zero on every arm, because it is a floor with
`ceil(target_blocks / n_slots)` underneath it and therefore cannot express
one tile at all, and a sweep whose two ends come from different mechanisms
cannot attribute what it finds. `row-tiles-default` passes zeros on both
parameters, which is byte for byte the geometry the trainer produced before
they existed; it is a separate arm from `row-tiles-2` even at the shape
where the default resolves to two tiles at the root, because the default
fixes a tile *count* per node while these arms fix a tile *length*, and the
two rules diverge as soon as a node is smaller than the dataset.

**What each arm refuses rather than measuring.** A knob that reaches no
kernel would still print a number under a label, which is the failure this
file exists to prevent, so `_check_arm_reachability` refuses before any data
is generated:

- Any launch-shape arm under `MOJOTREES_GPU_HIST_SPECIALIZATION=batched`.
  The batched kernels in `gpu_leaf_batching.mojo` carry their own row loop
  and their own tile arithmetic and read none of the four fields.
- Any launch-shape arm on a multiclass objective. `train_multiclass_gpu`
  takes none of the four arguments.
- A `narrow-index` arm on a shape outside the Int32 bound above.
- A `pair-align` arm on an objective with a constant hessian, which
  includes `reg`. The width-2 pair load exists only where the hessian plane
  is live; squared error guarantees a constant hessian, so an unweighted
  non-GOSS `reg` round elides that plane and gathers a single word, and both
  halves of the pair would be the same run. `binary` is where this arm is
  live. `MOJOTREES_CONST_HESSIAN=0` also makes it live, at the cost of
  measuring a configuration the library does not ship.
- A `row-tiles` arm on `gpu` or `gpu-device` while the device-owned growth
  plane is on, which is the default. That plane builds every non-root
  histogram through `gpu_active_rows.enqueue_desc_child`, which derives its
  tiling without the tile requests, so the arm would reach the root node and
  nothing else. `MOJOTREES_GPU_TREE_RESIDENT=0` forces the host-driven split
  loop, which honors the requests at every node; `gpu-host` never routes to
  the resident plane and is not refused.

Every arm also prints one `arm_conditions:` line before the loop starts,
holding its trainer, split strategy, grow policy and all four launch shapes
with the tile request resolved to the `rows_per_tile` actually passed. The
same string goes into each arm's object in `json_summary` as `conditions`. A
label is not a record, and this file has twice had to discard a number whose
conditions were written down only somewhere else.

`arms` is a comma-separated list of cpu, gpu, gpu-host, gpu-device,
lightgbm, row-unroll-on, row-unroll-off, narrow-index-on, narrow-index-off,
pair-align-on, pair-align-off, row-tiles-default, row-tiles-1, row-tiles-2,
row-tiles-4, row-tiles-8, in the order they should run; the
first is the baseline every other arm is compared against. Underscores read
as hyphens, so an output key copied out of a transcript is a valid arm name.
Any arm but lightgbm takes a `-depth` suffix
(`cpu-depth`, `gpu-device-depth`, ...) to train the same configuration under
`grow_policy=depthwise`; the
leaf budget is unchanged, so a depth-wise arm commits the same number of
splits and issues the same number of GPU launch groups as its leaf-wise
twin, and the pair measures the per-split cost of the order alone. That is
the comparison the depth-wise lane owes before any per-level launch
batching is written (docs/design/GPU_LEVELWISE.md): if the twins are
indistinguishable, batching has only the launch count to attack, and
`MOJOTREES_GPU_PHASE_TRACE=1` on a single arm at two row counts gives the
row-independent share of a tree's time, which is the ceiling of that
attack. Defaults: 100000 rows, 100 features, reg, 1 repeat, `cpu,gpu`.

    pixi run bench-train-gpu 50000 100 reg 5 gpu-host,gpu-device
    pixi run bench-train-gpu 250000 50 reg 5 gpu-device,gpu-device-depth
    pixi run bench-train-gpu 100000 54 multi:7 5 cpu,gpu
    pixi run -e bench bench-train-gpu 1000000 50 reg 5 lightgbm,gpu-device
    pixi run bench-train-gpu 1000000 50 reg 5 row-unroll-on,row-unroll-off
    pixi run bench-train-gpu 1000000 50 reg 5 narrow-index-off,narrow-index-on
    pixi run bench-train-gpu 1000000 50 binary 5 pair-align-on,pair-align-off
    MOJOTREES_GPU_TREE_RESIDENT=0 pixi run bench-train-gpu 1000000 50 reg 5 \\
        row-tiles-default,row-tiles-1
"""

from std.collections import Optional
from std.math import exp, log
from std.os import getenv
from std.python import Python, PythonObject
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from canary import (
    CanaryReading,
    canary_disabled_json,
    canary_json,
    enabled as canary_enabled,
    load_baseline,
    print_baseline,
    print_reading,
    print_session_verdict,
    take_reading,
)

from mojotrees.binning import fit_bins
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    MulticlassBooster,
    train,
    train_multiclass,
)
from mojotrees.binning import BinnedMatrix
from mojotrees.growth_policy import GROW_DEPTHWISE
# The three predicates the launch-shape refusals are read from, imported
# rather than restated so that this file and the trainer cannot drift into
# disagreeing about the same fact. `narrow_index_fits` is the shape bound
# `set_narrow_index` enforces; `objective_has_constant_hessian` is what
# decides whether a round carries a hessian plane for `pair_alignment` to
# annotate; `resident_round_enabled` is the gate `train_gpu` routes the
# device-owned growth plane on. All three are host arithmetic or a getenv
# and none of them opens a device.
from mojotrees.gpu_active_rows import narrow_index_fits
from mojotrees.gpu_resident_round import resident_round_enabled
from mojotrees.histogram import objective_has_constant_hessian
from mojotrees.train_gpu import (
    SPLIT_SEARCH_AUTO,
    SPLIT_SEARCH_DEVICE,
    SPLIT_SEARCH_HOST,
    train_gpu,
    train_multiclass_gpu,
)


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _sigmoid(x: Float64) -> Float64:
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    var e = exp(x)
    return e / (1.0 + e)


def _train_loss(
    booster: Booster,
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
) -> Float64:
    var loss = 0.0
    for r in range(data.n_rows):
        if objective == BINARY_LOGISTIC:
            var p = booster.predict_row(data, r)
            if p < 1e-15:
                p = 1e-15
            if p > 1.0 - 1e-15:
                p = 1.0 - 1e-15
            if target[r] > 0.5:
                loss -= log(p)
            else:
                loss -= log(1.0 - p)
        else:
            var d = booster.predict_row(data, r) - target[r]
            loss += d * d
    return loss / Float64(data.n_rows)


def _multiclass_loss(
    booster: MulticlassBooster,
    data: BinnedMatrix,
    labels: List[Int],
) -> Float64:
    """Mean softmax log loss over the training rows.

    The multiclass counterpart of `_train_loss`, and outside the timed
    region for the same reason. Row by row through `predict_proba_bins`
    rather than through `predict_batch`, because a batched predict fans out
    over `dispatch_rows` and would put a parallel prediction inside a
    function whose only job is to say that two arms fit the same model.
    """
    var loss = 0.0
    var bins = List[Int](capacity=data.n_features)
    for r in range(data.n_rows):
        bins.clear()
        for f in range(data.n_features):
            bins.append(data.bin_at(r, f))
        var probs = booster.predict_proba_bins(bins)
        var p = probs[labels[r]]
        if p < 1e-15:
            p = 1e-15
        loss -= log(p)
    return loss / Float64(data.n_rows)


def _class_labels(values: List[Float64], n_classes: Int) -> List[Int]:
    """Bucket a continuous signal into `n_classes` near-equal classes.

    The cuts are sample quantiles rather than an equal split of the
    signal's range, because the range split would be badly unbalanced: the
    generator's signal is a sum of four terms and is unimodal, so equal-
    width buckets would leave the two extreme classes nearly empty and
    their per-class trees would commit no splits worth timing. A benchmark
    whose classes are empty measures the empty classes.

    The quantiles come from a bounded sample rather than a full sort. A
    thousand-odd points place a cut to within about a percent of the true
    quantile, which is far tighter than the balance this needs, and it
    keeps setup out of the way of the measurement. Deterministic, since the
    sample is a fixed stride over deterministic data.
    """
    var n = len(values)
    var probe = 1024 if n > 1024 else n
    var stride = n // probe
    if stride < 1:
        stride = 1
    var sample = List[Float64](capacity=probe)
    var i = 0
    while i < n and len(sample) < probe:
        sample.append(values[i])
        i += stride
    for j in range(1, len(sample)):
        var v = sample[j]
        var k = j - 1
        while k >= 0 and sample[k] > v:
            sample[k + 1] = sample[k]
            k -= 1
        sample[k + 1] = v

    var cuts = List[Float64](capacity=n_classes - 1)
    for c in range(1, n_classes):
        var idx = c * len(sample) // n_classes
        if idx >= len(sample):
            idx = len(sample) - 1
        cuts.append(sample[idx])

    var out = List[Int](capacity=n)
    for r in range(n):
        var cls = 0
        while cls < len(cuts) and values[r] >= cuts[cls]:
            cls += 1
        out.append(cls)
    return out^


# An arm is a trainer plus, for the GPU trainer, an explicit split-search
# strategy. ARM_GPU leaves the choice to SPLIT_SEARCH_AUTO and so measures
# what a caller actually gets; the two pinned arms are what a comparison
# between the strategies has to use.
comptime ARM_CPU = 0
comptime ARM_GPU = 1
comptime ARM_GPU_HOST = 2
comptime ARM_GPU_DEVICE = 3
# LightGBM, in this process, in this loop. Not a mojotrees trainer at all:
# it reaches `bench/bench_lightgbm.py` through Python interop and trains on
# a Dataset built once before the loop, so what it contributes per repeat is
# one boosting run and nothing else. See `_run_lgbm` and the module
# docstring for what this does and does not make comparable.
comptime ARM_LGBM = 4
# Added to any of the above: the same arm under `grow_policy=depthwise`.
comptime ARM_DEPTHWISE = 8
# Added to a GPU arm: pass `row_unroll=False` to the trainer, so the histogram
# row loop walks one row per iteration instead of keeping `HIST_ROW_UNROLL`
# rows in flight. A launch shape and nothing else; see `_run_arm`.
comptime ARM_NOUNROLL = 16
# Marks the two arms of the named row-unroll pair, so they label themselves
# `row-unroll-on` and `row-unroll-off` rather than `gpu` and `gpu-nounroll`.
# The pair earns its own labels because it is the comparison the measurement
# queue names, and because the ON arm is byte-for-byte the default: an arm
# that prints as plain `gpu` invites the reader to treat it as a baseline
# that happened to be lying around rather than as a declared condition.
comptime ARM_UNROLL_PAIR = 32

# --- the K1 hist-latency lane's three arms ---------------------------------
#
# `train_gpu` grew four more launch-shape arguments (`narrow_index`,
# `pair_alignment`, `min_tiles`, `rows_per_tile`) and nothing here could
# reach any of them, which on this machine is the same as their not having
# landed: only interleaved arms in one process resolve anything, so a knob a
# rebuild is the only way to move is a knob nobody can measure.
#
# **Three arms and not one.** They are wired as three independent pairs, each
# against its own control, rather than as a single "K1 on" arm, because three
# changes measured together cannot be attributed to any one of them. That is
# not a hypothetical: this repository has already produced two
# correct-but-incomplete wait counts by moving more than one thing between
# two timings.
#
# **None of them can change a model**, and every one of the arms below prints
# its training loss so a reader can confirm that rather than take it. The
# argument is in `train_gpu`'s docstring and in `GpuActiveRows`: accumulation
# is fixed-point Int32 and integer addition is associative and commutative,
# so a different index width, a different alignment assertion, and a
# different tile geometry all sum the same integers into the same bins in a
# different order to the same value. The index width is the one that is exact
# only *under a bound* on the dataset shape, which is why `narrow_index_fits`
# is checked here before a run rather than left to fail at the first launch.

# Added to a GPU arm: pass `narrow_index=True`. OFF is what the trainer
# defaults to, so unlike ARM_NOUNROLL this flag names the non-default
# direction. Reversed polarity is a hazard worth stating once: `_arm_name`
# and `_arm_conditions` both derive the printed word from the flag rather
# than restating it, so there is one place the polarity lives.
comptime ARM_NARROW = 64
# Marks the two arms of the named index-width pair, so they label themselves
# `narrow-index-on` / `narrow-index-off` rather than `gpu` and `gpu-narrow`.
comptime ARM_NARROW_PAIR = 128
# Added to a GPU arm: pass `pair_alignment=False`. ON is the trainer's
# default, so this flag names the OFF direction as ARM_NOUNROLL does.
comptime ARM_NOALIGN = 256
comptime ARM_ALIGN_PAIR = 512
# Marks the named row-tiling arms, so they label themselves `row-tiles-4`
# rather than `gpu-tiles-4`.
comptime ARM_TILE_PAIR = 1024
# The row-tiling request this arm carries, as a small code in three bits
# above ARM_TILE_SHIFT. A code and not the tile count itself, because the
# control arm has to be distinguishable from every count: it passes zeros,
# and zero is not "one tile" but "no request at all".
comptime ARM_TILE_SHIFT = 11
comptime ARM_TILE_MASK = 7 << ARM_TILE_SHIFT
comptime TILE_NONE = 0
comptime TILE_DEFAULT = 1
comptime TILE_1 = 2
comptime TILE_2 = 3
comptime TILE_4 = 4
comptime TILE_8 = 5


struct ArmRun(Copyable, Movable):
    """One timed training run: wall seconds, training loss, tree count."""

    var seconds: Float64
    var loss: Float64
    var n_trees: Int

    def __init__(out self, seconds: Float64, loss: Float64, n_trees: Int):
        self.seconds = seconds
        self.loss = loss
        self.n_trees = n_trees


def _arm_base(arm: Int) -> Int:
    return arm & (ARM_DEPTHWISE - 1)


def _arm_depthwise(arm: Int) -> Bool:
    return (arm & ARM_DEPTHWISE) != 0


def _arm_row_unroll(arm: Int) -> Bool:
    """The `row_unroll` this arm passes to `train_gpu`. True is the default."""
    return (arm & ARM_NOUNROLL) == 0


def _arm_touches_unroll(arm: Int) -> Bool:
    """Whether this arm names the row-walk knob at all, either way.

    Both members of the pair count, not only the OFF one. `row-unroll-on`
    passes the value the trainer would have defaulted to, so it cannot fail
    for want of the argument; it is refused wherever its twin is refused
    because half a pair measures nothing and would still print a number.
    """
    return (arm & (ARM_NOUNROLL | ARM_UNROLL_PAIR)) != 0


def _arm_narrow_index(arm: Int) -> Bool:
    """The `narrow_index` this arm passes to `train_gpu`. False is the
    default, so the flag names the ON direction."""
    return (arm & ARM_NARROW) != 0


def _arm_touches_narrow(arm: Int) -> Bool:
    """Whether this arm names the index-width knob at all, either way.

    Both members of the pair count, for the reason `_arm_touches_unroll`
    gives: the arm that passes the trainer's own default still declares a
    condition, and half a pair measures nothing while still printing a
    number."""
    return (arm & (ARM_NARROW | ARM_NARROW_PAIR)) != 0


def _arm_pair_alignment(arm: Int) -> Bool:
    """The `pair_alignment` this arm passes. True is the default."""
    return (arm & ARM_NOALIGN) == 0


def _arm_touches_align(arm: Int) -> Bool:
    """Whether this arm names the pair-load alignment knob at all."""
    return (arm & (ARM_NOALIGN | ARM_ALIGN_PAIR)) != 0


def _arm_tile_code(arm: Int) -> Int:
    """This arm's row-tiling request as one of the TILE_* codes."""
    return (arm & ARM_TILE_MASK) >> ARM_TILE_SHIFT


def _arm_touches_tiles(arm: Int) -> Bool:
    """Whether this arm names the row-tile geometry at all, control
    included. The control arm counts because it is the half of the pair that
    passes zeros, and a pair with only one half is not a pair."""
    return _arm_tile_code(arm) != TILE_NONE


def _tile_root_count(code: Int) -> Int:
    """The tile count a TILE_* code asks for **at the root node**, or 0 for
    the control and for a non-tiling arm.

    Root, not per node, and the distinction is the whole reason this
    function is named the way it is. The arms request a tile *length*
    (`rows_per_tile`), so a node holding `m` rows gets `ceil(m / length)`
    tiles: the number in the arm's name is what the root gets, and a node
    smaller than one tile length gets one tile whatever the arm asked for.
    See `_arm_rows_per_tile` for why the length is the knob that is swept.
    """
    if code == TILE_1:
        return 1
    if code == TILE_2:
        return 2
    if code == TILE_4:
        return 4
    if code == TILE_8:
        return 8
    return 0


def _tile_word(code: Int) -> String:
    """The TILE_* code as it appears in an arm's name."""
    if code == TILE_DEFAULT:
        return String("default")
    return String(_tile_root_count(code))


def _arm_rows_per_tile(arm: Int, n_rows: Int) -> Int:
    """The `rows_per_tile` this arm passes to `train_gpu`, or 0 for none.

    **One knob for all four tile counts, on purpose.** `min_tiles` and
    `rows_per_tile` are two different mechanisms and the sweep uses only the
    second, because a sweep whose ends are reached by different mechanisms
    cannot attribute what it finds to the tile count. `min_tiles` is a
    *floor* underneath which `gpu_tiling.row_tile_floor`'s occupancy term
    `ceil(target_blocks / n_slots)` still applies, so it cannot express one
    tile at all on a shape whose occupancy term is 2 -- which is exactly the
    shape being tested and exactly the direction that has never been tried.
    `rows_per_tile` is the only parameter that can ask for FEWER tiles than
    the occupancy term gives. So it is the one that carries the whole sweep,
    and `min_tiles` is passed as zero by every arm here.

    The value is `ceil(n_rows / N)` for an arm asking for N tiles, which
    makes these arms a **fixed tile length** policy: one threadgroup scans
    at most that many rows, whatever node it is on. The shipped default is
    the other kind of policy, a fixed tile *count* per node whose length
    therefore shrinks with the node. The two are not the same rule even
    where they agree at the root, which is why `row-tiles-2` is not the
    control and why the control passes zeros instead of passing 2.
    """
    var n = _tile_root_count(_arm_tile_code(arm))
    if n < 1:
        return 0
    return (n_rows + n - 1) // n


def _arm_touches_launch_shape(arm: Int) -> Bool:
    """Whether this arm names any GPU histogram launch shape.

    The four knobs together: the row walk, the index width, the pair load's
    alignment, and the tile geometry. Used by the refusals that apply to all
    of them at once -- a trainer with no such loop to reshape (`cpu`,
    `lightgbm`), an entry point that takes none of the arguments
    (`train_multiclass_gpu`), and a kernel family that reads none of the
    fields (`gpu_leaf_batching`)."""
    return (
        _arm_touches_unroll(arm)
        or _arm_touches_narrow(arm)
        or _arm_touches_align(arm)
        or _arm_touches_tiles(arm)
    )


def _arm_name(arm: Int) -> String:
    if (arm & ARM_UNROLL_PAIR) != 0:
        if _arm_row_unroll(arm):
            return String("row-unroll-on")
        return String("row-unroll-off")
    if (arm & ARM_NARROW_PAIR) != 0:
        if _arm_narrow_index(arm):
            return String("narrow-index-on")
        return String("narrow-index-off")
    if (arm & ARM_ALIGN_PAIR) != 0:
        if _arm_pair_alignment(arm):
            return String("pair-align-on")
        return String("pair-align-off")
    if (arm & ARM_TILE_PAIR) != 0:
        return String("row-tiles-", _tile_word(_arm_tile_code(arm)))
    var base = _arm_base(arm)
    var name = String("gpu")
    if base == ARM_CPU:
        name = String("cpu")
    elif base == ARM_GPU_HOST:
        name = String("gpu-host")
    elif base == ARM_GPU_DEVICE:
        name = String("gpu-device")
    elif base == ARM_LGBM:
        name = String("lightgbm")
    if _arm_depthwise(arm):
        name += "-depth"
    if not _arm_row_unroll(arm):
        name += "-nounroll"
    # Only the non-default direction earns a suffix, which is the rule the
    # `-unroll` suffix already follows: an arm that declares the trainer's
    # own value is the same configuration as the plain arm and must not
    # enter the record under a second key.
    if _arm_narrow_index(arm):
        name += "-narrow"
    if not _arm_pair_alignment(arm):
        name += "-nopalign"
    var tcode = _arm_tile_code(arm)
    if tcode != TILE_NONE and tcode != TILE_DEFAULT:
        name += "-tiles-" + _tile_word(tcode)
    return name


def _arm_key(arm: Int) -> String:
    """`_arm_name` with the hyphens replaced, for `key: value` output lines."""
    return _arm_name(arm).replace("-", "_")


def _parse_arms(spec: String) raises -> List[Int]:
    """A comma-separated arm list, in the order the arms should run.

    Underscores are read as hyphens, so `row_unroll_off` and
    `row-unroll-off` are the same arm. The output keys use underscores and
    the command line uses hyphens, which is a distinction nobody should have
    to hold in their head while copying a key out of a transcript back into a
    command.

    `gpu-unroll` and `gpu-nounroll` are accepted as the pair's older names,
    because that is the spelling `bench/results/SESSION_QUEUE.md` recorded
    while the arms did not yet exist. They resolve to the same two arms and
    print under the same two labels, so a queued command and a fresh one
    produce the same keys.
    """
    var arms = List[Int]()
    for part in spec.split(","):
        var given = String(part)
        if given.byte_length() == 0:
            continue
        var name = given.replace("_", "-")

        # The named pair, matched whole and before any suffix stripping, so
        # that `gpu-nounroll` is the OFF arm of the pair rather than a plain
        # `gpu` carrying a suffix. Both members run under SPLIT_SEARCH_AUTO:
        # the pair holds every condition but the knob constant, and AUTO is
        # what a caller actually gets. Pin the strategy with the composable
        # suffix form (`gpu-device-nounroll`) when that is the question.
        if name == "row-unroll-on" or name == "gpu-unroll":
            arms.append(ARM_GPU | ARM_UNROLL_PAIR)
            continue
        if name == "row-unroll-off" or name == "gpu-nounroll":
            arms.append(ARM_GPU | ARM_NOUNROLL | ARM_UNROLL_PAIR)
            continue

        # The K1 lane's three named pairs, matched whole and before any
        # suffix stripping, for the reason the row-walk pair is: the name is
        # the comparison. Each runs under SPLIT_SEARCH_AUTO so the pair holds
        # every condition but its own knob constant; pin the strategy with
        # the composable suffix form when that is the question.
        if name == "narrow-index-on":
            arms.append(ARM_GPU | ARM_NARROW | ARM_NARROW_PAIR)
            continue
        if name == "narrow-index-off":
            arms.append(ARM_GPU | ARM_NARROW_PAIR)
            continue
        if name == "pair-align-on":
            arms.append(ARM_GPU | ARM_ALIGN_PAIR)
            continue
        if name == "pair-align-off":
            arms.append(ARM_GPU | ARM_NOALIGN | ARM_ALIGN_PAIR)
            continue
        # The tile sweep. `row-tiles-default` is the control and passes zeros
        # on both tile parameters, which is byte for byte the geometry this
        # trainer produced before the parameters existed. It is a separate
        # arm from `row-tiles-2` rather than the same one, even at the shape
        # where the default resolves to two tiles at the root, because the
        # two are different rules: see `_arm_rows_per_tile`.
        if name == "row-tiles-default":
            arms.append(
                ARM_GPU | ARM_TILE_PAIR | (TILE_DEFAULT << ARM_TILE_SHIFT)
            )
            continue
        if name == "row-tiles-1":
            arms.append(ARM_GPU | ARM_TILE_PAIR | (TILE_1 << ARM_TILE_SHIFT))
            continue
        if name == "row-tiles-2":
            arms.append(ARM_GPU | ARM_TILE_PAIR | (TILE_2 << ARM_TILE_SHIFT))
            continue
        if name == "row-tiles-4":
            arms.append(ARM_GPU | ARM_TILE_PAIR | (TILE_4 << ARM_TILE_SHIFT))
            continue
        if name == "row-tiles-8":
            arms.append(ARM_GPU | ARM_TILE_PAIR | (TILE_8 << ARM_TILE_SHIFT))
            continue

        # Suffixes, stripped in a loop so they compose in either order:
        # `gpu-device-depth-nounroll` and `gpu-device-nounroll-depth` are the
        # same arm.
        var flags = 0
        var named_shape = False
        while True:
            if name.endswith("-nounroll"):
                flags |= ARM_NOUNROLL
                named_shape = True
                var trimmed = String(name[byte= : name.byte_length() - 9])
                name = trimmed^
            elif name.endswith("-unroll"):
                # Sets no flag: on is the default. It is worth being able to
                # write anyway, because an A/B whose ON arm inherits its
                # condition has one arm's condition undeclared, and this file
                # already refuses to let a strategy be inherited for the same
                # reason.
                named_shape = True
                var trimmed = String(name[byte= : name.byte_length() - 7])
                name = trimmed^
            elif name.endswith("-nonarrow"):
                # The trainer's default. Written out for the reason
                # `-unroll` is, and checked before `-narrow` so that the two
                # cannot be confused by a suffix match.
                named_shape = True
                var trimmed = String(name[byte= : name.byte_length() - 9])
                name = trimmed^
            elif name.endswith("-narrow"):
                flags |= ARM_NARROW
                named_shape = True
                var trimmed = String(name[byte= : name.byte_length() - 7])
                name = trimmed^
            elif name.endswith("-nopalign"):
                flags |= ARM_NOALIGN
                named_shape = True
                var trimmed = String(name[byte= : name.byte_length() - 9])
                name = trimmed^
            elif name.endswith("-palign"):
                named_shape = True
                var trimmed = String(name[byte= : name.byte_length() - 7])
                name = trimmed^
            elif name.endswith("-tiles-default"):
                flags |= TILE_DEFAULT << ARM_TILE_SHIFT
                named_shape = True
                var trimmed = String(name[byte= : name.byte_length() - 14])
                name = trimmed^
            elif name.endswith("-tiles-1"):
                flags |= TILE_1 << ARM_TILE_SHIFT
                named_shape = True
                var trimmed = String(name[byte= : name.byte_length() - 8])
                name = trimmed^
            elif name.endswith("-tiles-2"):
                flags |= TILE_2 << ARM_TILE_SHIFT
                named_shape = True
                var trimmed = String(name[byte= : name.byte_length() - 8])
                name = trimmed^
            elif name.endswith("-tiles-4"):
                flags |= TILE_4 << ARM_TILE_SHIFT
                named_shape = True
                var trimmed = String(name[byte= : name.byte_length() - 8])
                name = trimmed^
            elif name.endswith("-tiles-8"):
                flags |= TILE_8 << ARM_TILE_SHIFT
                named_shape = True
                var trimmed = String(name[byte= : name.byte_length() - 8])
                name = trimmed^
            elif name.endswith("-depth"):
                flags |= ARM_DEPTHWISE
                var trimmed = String(name[byte= : name.byte_length() - 6])
                name = trimmed^
            else:
                break

        if name == "cpu":
            if named_shape:
                # Every one of these four knobs is a GPU histogram launch
                # shape. The CPU trainer has no such loop to reshape, so
                # `cpu-nounroll` or `cpu-tiles-4` would be plain `cpu` under
                # a label claiming an arm.
                raise Error(
                    "the row-walk, index-width, pair-alignment and row-tile"
                    " knobs are all GPU histogram launch shapes and the CPU"
                    " trainer has none, so 'cpu-nounroll' or 'cpu-tiles-4'"
                    " would be plain cpu under a misleading label; use 'cpu'"
                )
            arms.append(ARM_CPU | flags)
        elif name == "gpu":
            arms.append(ARM_GPU | flags)
        elif name == "gpu-host":
            arms.append(ARM_GPU_HOST | flags)
        elif name == "gpu-device":
            arms.append(ARM_GPU_DEVICE | flags)
        elif name == "lightgbm":
            if flags != 0 or named_shape:
                # `grow_policy` is a mojotrees and XGBoost switch. LightGBM
                # grows leaf-wise and has no depth-wise mode to ask for, so
                # `lightgbm-depth` would silently be plain `lightgbm` under a
                # label claiming otherwise. Compare a depth-wise mojotrees arm
                # against `lightgbm` and read the label as what it says. All
                # four launch-shape knobs are ours and reach nothing on that
                # side at all, so they are refused on the same grounds.
                raise Error(
                    "lightgbm has neither a depth-wise grow policy nor any of"
                    " our histogram launch-shape knobs to select, so"
                    " 'lightgbm-depth', 'lightgbm-nounroll' or"
                    " 'lightgbm-tiles-4' would be plain lightgbm under a"
                    " misleading label; use 'lightgbm'"
                )
            arms.append(ARM_LGBM)
        else:
            raise Error(
                String(
                    "unknown arm '",
                    given,
                    "'; use cpu, gpu, gpu-host, gpu-device, lightgbm, or one"
                    " of the four named launch-shape pairs: row-unroll-on /"
                    " row-unroll-off, narrow-index-on / narrow-index-off,"
                    " pair-align-on / pair-align-off, and the row-tile sweep"
                    " row-tiles-default / row-tiles-1 / row-tiles-2 /"
                    " row-tiles-4 / row-tiles-8. Any arm but lightgbm takes a"
                    " -depth suffix, and any GPU arm takes -unroll,"
                    " -nounroll, -narrow, -nonarrow, -palign, -nopalign, or"
                    " -tiles-default / -tiles-1 / -tiles-2 / -tiles-4 /"
                    " -tiles-8, in any order.",
                )
            )
    if len(arms) == 0:
        raise Error("no arms selected")
    return arms^


def _arm_conditions(arm: Int, n_rows: Int) -> String:
    """Every condition this arm passes to its trainer, on one line.

    Printed once per arm before the loop starts, because an arm name is a
    label and a label is not a record. Two things in this file have already
    had to be discarded for want of exactly this line: a timing whose split
    strategy was inherited from an environment variable, and a pair of
    figures whose conditions were written down only in a commit message.

    It also carries the direction a tile arm moves in, in the one place a
    reader of the output can see it without opening this file: the resolved
    `rows_per_tile` is here beside the arm's name, and a smaller number is
    more and shorter tiles.
    """
    var out = String(_arm_name(arm))
    var base = _arm_base(arm)
    if base == ARM_LGBM:
        return out + " trainer=lightgbm (none of the mojotrees knobs apply)"
    if base == ARM_CPU:
        out += " trainer=cpu split_search=n/a"
    else:
        out += " trainer=gpu split_search="
        if base == ARM_GPU_HOST:
            out += "host"
        elif base == ARM_GPU_DEVICE:
            out += "device"
        else:
            out += "auto"
    out += " grow=" + (
        String("depthwise") if _arm_depthwise(arm) else String("leafwise")
    )
    if base == ARM_CPU:
        return out^
    out += " row_unroll=" + (
        String("on") if _arm_row_unroll(arm) else String("off")
    )
    out += " narrow_index=" + (
        String("on") if _arm_narrow_index(arm) else String("off")
    )
    out += " pair_alignment=" + (
        String("on") if _arm_pair_alignment(arm) else String("off")
    )
    out += String(
        " min_tiles=0 rows_per_tile=", _arm_rows_per_tile(arm, n_rows)
    )
    var tiles = _tile_root_count(_arm_tile_code(arm))
    if tiles > 0:
        out += String(" root_tiles_requested=", tiles)
    elif _arm_touches_tiles(arm):
        out += " root_tiles_requested=none(control,zeros)"
    return out^


def _tile_legend() -> String:
    """What the number in `row-tiles-N` means, for a reader of the output.

    Written out rather than left to the arm names, because the names alone
    do not say which direction the earlier experiment went. It went **up**:
    an 80-tile device-wide floor was tried on this repository and *measured*
    22 percent slower at 50 features and 36 percent slower at 100 in a whole
    fit, and was reverted (`gpu_tiling.row_tile_floor`). The downward
    direction has never been tried, which is what `row-tiles-1` is for.
    """
    return String(
        "row_tiles_legend: N in row-tiles-N is the tile count at the ROOT"
        " node. The arm sets a fixed tile length rows_per_tile ="
        " ceil(n_rows / N), so a node holding m rows gets ceil(m / length)"
        " tiles and a node shorter than one tile gets one. LARGER N = more,"
        " shorter tiles: the direction an earlier device-wide floor of 80"
        " went, which measured 22 to 36 percent slower across a fit and was"
        " reverted. SMALLER N = fewer, longer tiles: never tested."
        " row-tiles-default passes zeros on both tile parameters and is byte"
        " for byte the geometry this trainer produced before the parameters"
        " existed; it is NOT the same rule as row-tiles-2 even where the two"
        " agree at the root, because the default fixes a tile count per node"
        " and these arms fix a tile length."
    )


def _check_arm_reachability(
    arms: List[Int],
    n_rows: Int,
    n_features: Int,
    objective: Int,
    n_classes: Int,
) raises:
    """Refuse, before any data is generated, every arm combination in which
    a knob would reach nothing.

    The failure this guards against is not a crash. It is two arms that run
    identical code, print two labels and two timings, and produce a delta
    that a reader will attribute to a knob that was never in the picture.
    This file already refuses `cpu-nounroll` and `lightgbm-depth` on exactly
    that ground; these are the same rule applied to the paths the K1 arms
    have to travel to reach a kernel.

    Every predicate here is read from the same source the trainer reads, or
    from the same environment variable spelled the same way, rather than
    restated: `narrow_index_fits` is the bound `set_narrow_index` enforces,
    `resident_round_enabled` is the gate `train_gpu` routes on, and
    `objective_has_constant_hessian` is what `round_has_constant_hessian`
    binds. Restating any of them here would put a second predicate over one
    fact into the tree, which is the hazard `resident_round_enabled`'s own
    docstring is currently warning about.
    """
    var batched = getenv("MOJOTREES_GPU_HIST_SPECIALIZATION") == "batched"
    var quant_off = getenv("MOJOTREES_GPU_QUANTIZED_GRADS") == "0"
    var const_hess_off = getenv("MOJOTREES_CONST_HESSIAN") == "0"
    var resident_on = resident_round_enabled()

    for a in range(len(arms)):
        var arm = arms[a]
        if not _arm_touches_launch_shape(arm):
            continue
        var label = _arm_name(arm)

        if n_classes > 0:
            # `train_multiclass_gpu` takes none of the four arguments, so
            # both halves of any launch-shape pair would be the same run.
            raise Error(
                String(
                    "arm '",
                    label,
                    "' is single output: train_multiclass_gpu takes no"
                    " row_unroll, narrow_index, pair_alignment or"
                    " rows_per_tile argument, so both halves of the pair"
                    " would be the same run under two labels",
                )
            )

        if batched:
            # `gpu_leaf_batching`'s kernel family has its own row loop and
            # its own tile arithmetic and reads none of the four fields, so
            # under a batched specialization every launch-shape arm is inert
            # wherever a batch is actually taken. `batching_declined_reason`
            # can still decline a given frontier, which makes the inertness
            # partial and per node rather than total -- and a knob that
            # reaches some nodes and not others is worse to time than one
            # that reaches none, not better.
            raise Error(
                String(
                    "arm '",
                    label,
                    "' cannot be measured under"
                    " MOJOTREES_GPU_HIST_SPECIALIZATION=batched: the batched"
                    " histogram kernels in gpu_leaf_batching.mojo have their"
                    " own row loop and tile arithmetic and read none of"
                    " row_unroll, narrow_index, pair_alignment or the tile"
                    " requests, so the arm would be inert on every batched"
                    " frontier. Unset the variable, or measure the batched"
                    " path against itself.",
                )
            )

        if _arm_touches_narrow(arm) and not narrow_index_fits(
            n_rows, n_features
        ):
            # The bound `set_narrow_index` enforces. Checked here so the
            # refusal lands before a million rows are generated rather than
            # at the first histogram launch, and so the message can name the
            # shape rather than the setter.
            raise Error(
                String(
                    "arm '",
                    label,
                    "' needs n_features * n_rows and 2 * n_rows to fit a"
                    " signed 32-bit integer and this shape (",
                    n_rows,
                    " x ",
                    n_features,
                    ") exceeds it; the wide index arm is the only correct"
                    " one for it, so there is no pair to run",
                )
            )

        if _arm_touches_align(arm):
            if quant_off:
                # The Float32 gradient arm gathers two separate planes and
                # issues no width-2 load at all, so there is no alignment to
                # assert.
                raise Error(
                    String(
                        "arm '",
                        label,
                        "' has nothing to align under"
                        " MOJOTREES_GPU_QUANTIZED_GRADS=0: the Float32"
                        " gradient arm gathers two separate planes and"
                        " issues no width-2 pair load, so both halves of the"
                        " pair would be the same run",
                    )
                )
            if (
                objective_has_constant_hessian(objective, False)
                and not const_hess_off
            ):
                # **The one that matters, and it rules out `reg`.** The pair
                # load exists only on the quantized arm with a live hessian
                # plane (`_hist_rows_step`, `comptime if QUANT: comptime if
                # CELIDE`). Squared error guarantees a constant hessian, so
                # an unweighted non-GOSS `reg` round elides the hessian plane
                # and gathers one word, and `pair_alignment` is read by
                # nothing. `binary` is where the arm is live, because
                # logistic curvature is p(1-p).
                raise Error(
                    String(
                        "arm '",
                        label,
                        "' is inert on this objective: it guarantees a"
                        " constant hessian, so the histogram row loop elides"
                        " the hessian plane, gathers a single word, and"
                        " issues no width-2 pair load for pair_alignment to"
                        " annotate. Run the pair on 'binary', whose"
                        " curvature is p(1-p) and whose rounds therefore"
                        " carry a live hessian plane; or force the"
                        " three-plane path with MOJOTREES_CONST_HESSIAN=0,"
                        " which measures the knob on a configuration the"
                        " library does not ship.",
                    )
                )

        if _arm_touches_tiles(arm) and resident_on:
            var base = _arm_base(arm)
            if base == ARM_GPU or base == ARM_GPU_DEVICE:
                # `gpu_active_rows.enqueue_desc_child` -- the device-owned
                # tree's per-node histogram -- calls `derive_tiling` without
                # `min_tiles_request` / `rows_per_tile_request`, so under the
                # resident plane only the root histogram (which goes through
                # `enqueue_leaf`) honors a tile request and every other node
                # ignores it. That is a partly-inert arm, which is the worst
                # kind to time: the delta would be real and would be
                # attributed to a geometry most of the tree never saw.
                #
                # The resident plane is reached only from the device
                # split-search path, so `gpu-host` arms are unaffected and
                # are not refused.
                raise Error(
                    String(
                        "arm '",
                        label,
                        "' cannot be measured while the device-owned growth"
                        " plane is on. It builds every non-root histogram"
                        " through gpu_active_rows.enqueue_desc_child, which"
                        " derives its tiling without the tile requests, so"
                        " the arm would reach the root node and nothing"
                        " else. Set MOJOTREES_GPU_TREE_RESIDENT=0 to force"
                        " the host-driven split loop, which honors the"
                        " requests at every node, or pin the arm to"
                        " 'gpu-host', which never routes to that plane.",
                    )
                )


def _lgbm_threads() -> Int:
    """The thread count the LightGBM arm pins, or 0 for LightGBM's own default.

    MOJOTREES_LGBM_THREADS if set, otherwise MOJOTREES_NUM_WORKERS, which is
    what the CPU arm is already pinned by. The two variables mean the same
    thing here by coincidence of their zero cases rather than by design: 0 or
    unset is auto on both sides, one thread per core on LightGBM's and a
    core-count-derived task count on ours. Nothing in this file can verify
    that a pinned number means the same amount of machine on both sides, so
    the resolved value is printed and the reader is owed that check.

    The separate MOJOTREES_LGBM_THREADS exists for the case the comparison
    actually wants: the GPU arms do not read MOJOTREES_NUM_WORKERS in any way
    that makes it the right number for LightGBM, so a GPU-versus-LightGBM run
    has to say what LightGBM got rather than inherit it.
    """
    var s = getenv("MOJOTREES_LGBM_THREADS")
    if s.byte_length() == 0:
        s = getenv("MOJOTREES_NUM_WORKERS")
    if s.byte_length() == 0:
        return 0
    try:
        var n = Int(s)
        return n if n > 0 else 0
    except:
        return 0


def _lgbm_arm(
    n_rows: Int,
    n_features: Int,
    obj_name: String,
    seed: Int,
    params: BoosterParams,
    max_bin: Int,
) raises -> PythonObject:
    """Build the LightGBM arm's state: import, generate, bin, all untimed.

    Everything expensive that is not a boosting run happens here, before the
    interleaved loop starts, so that the arm's per-repeat call contains only
    the work the mojotrees arms are also timing. LightGBM's own binning is
    reported separately for the same reason ours is.

    The mojotrees defaults are handed over so the Python side can refuse to
    run when they have drifted off `bench/real_data/scenarios.py`. They are a
    cross-check and not a configuration: LightGBM is configured from that
    file, and this call is what proves the two descriptions still agree.

    Resolved from MOJOTREES_BENCH_DIR, defaulting to `bench`, because the
    module is found relative to the working directory and the pixi task runs
    from the repository root.
    """
    var bench_dir = getenv("MOJOTREES_BENCH_DIR")
    if bench_dir.byte_length() == 0:
        bench_dir = String("bench")
    Python.add_to_path(bench_dir)
    var module: PythonObject
    try:
        module = Python.import_module("bench_lightgbm")
    except e:
        raise Error(
            String(
                "the lightgbm arm needs bench/bench_lightgbm.py and the bench"
                " environment's LightGBM; run under `pixi run -e bench` from"
                " the repository root, or point MOJOTREES_BENCH_DIR at the"
                " bench directory. Underlying error: ",
                String(e),
            )
        )
    return module.InterleavedArm(
        n_rows,
        n_features,
        obj_name,
        _lgbm_threads(),
        seed,
        rounds=params.n_estimators,
        learning_rate=params.learning_rate,
        num_leaves=params.tree.num_leaves,
        min_data_in_leaf=params.tree.min_data_in_leaf,
        lambda_l2=params.tree.lambda_reg,
        lambda_l1=params.tree.lambda_l1,
        min_child_hess=params.tree.min_child_hess,
        max_depth=params.tree.max_depth,
        max_bin=max_bin,
    )


def _run_lgbm(state: PythonObject, want_loss: Bool) raises -> ArmRun:
    """One LightGBM boosting run, timed by this process's clock.

    The clock is Mojo's `perf_counter_ns`, the same one every other arm is
    timed by, rather than Python's, so no arm's number carries a different
    clock's offset. What sits inside it is one `lgb.train` call plus a couple
    of attribute lookups, which is microseconds against a run measured in
    seconds.

    The loss is read afterwards, outside the timed region, on the first
    repeat only, exactly as the mojotrees arms do it. It is the mean squared
    residual for `reg` and the mean log loss with the same 1e-15 clip for
    `binary`, so it is read directly against `_train_loss` above rather than
    merely alongside it.
    """
    var t0 = perf_counter_ns()
    var n_trees = Int(py=state.train())
    var seconds = Float64(perf_counter_ns() - t0) / 1e9
    var loss = -1.0
    if want_loss:
        loss = Float64(py=state.loss())
    return ArmRun(seconds, loss, n_trees)


def _run_arm(
    arm: Int,
    data: BinnedMatrix,
    target: List[Float64],
    labels: List[Int],
    n_classes: Int,
    objective: Int,
    want_loss: Bool,
    lgbm: Optional[PythonObject],
) raises -> ArmRun:
    """Time one complete training run on one arm.

    The loss pass sits outside the timed region and is requested on the first
    repeat only: the fit is deterministic, so scoring it again on every
    repeat would only add prediction time to a wall clock that is meant to
    measure training.

    `n_classes` of 0 means single output and `objective` decides; anything
    else is softmax and `objective` is not read at all, because there is no
    single-output objective code for softmax to pass. `labels` is empty in
    the first case and `target` is the bucketed signal in the second, and
    each is ignored on the arm that does not use it.

    `lgbm` holds the LightGBM arm's state and is empty unless a `lightgbm`
    arm was selected. It is an `Optional` rather than a `PythonObject`
    standing in for absence, because constructing any `PythonObject` starts
    the interpreter, and a `cpu,gpu` run must not require a Python at all.
    """
    if _arm_base(arm) == ARM_LGBM:
        return _run_lgbm(lgbm.value(), want_loss)

    var params = BoosterParams.default()
    if _arm_depthwise(arm):
        params.tree.grow_policy = GROW_DEPTHWISE
    var base = _arm_base(arm)
    # Stated on every GPU arm rather than defaulted on the ON one, for the
    # reason the split strategy is stated: a condition that is inherited is a
    # condition nobody wrote down, and this file has already been burned once
    # by an arm running under a label it did not earn.
    var unroll = _arm_row_unroll(arm)
    # The K1 lane's three, stated on every GPU arm for the same reason. The
    # tile length is derived from this matrix's own row count rather than
    # from the command line's, so an arm cannot be handed a length that
    # belongs to a different dataset; `min_tiles` is zero on every arm here
    # and `_arm_rows_per_tile` says why.
    var narrow = _arm_narrow_index(arm)
    var palign = _arm_pair_alignment(arm)
    var rows_per_tile = _arm_rows_per_tile(arm, data.n_rows)

    if n_classes > 0:
        # Softmax on both backends: one tree per class per round, so the
        # tree count reported below is rounds times classes. The split
        # strategy reaches `train_multiclass_gpu` as an argument for the
        # same reason it does on the single-output arms: a mistyped
        # environment export cannot then relabel an arm.
        var mc_strategy = SPLIT_SEARCH_AUTO
        if base == ARM_GPU_HOST:
            mc_strategy = SPLIT_SEARCH_HOST
        elif base == ARM_GPU_DEVICE:
            mc_strategy = SPLIT_SEARCH_DEVICE
        if _arm_touches_launch_shape(arm):
            # `train_multiclass_gpu` has none of the four launch-shape
            # arguments, so such an arm reaching here would train under the
            # trainer's own defaults whatever it asked for, and print a
            # number under a label claiming otherwise. `main` refuses this
            # combination before any data is generated; this is the second
            # door on the same room, because the first one is a check
            # somebody can move.
            raise Error(
                "the launch-shape arms are single output:"
                " train_multiclass_gpu takes no row_unroll, narrow_index,"
                " pair_alignment or rows_per_tile argument, so a multiclass"
                " launch-shape arm would run the defaults under an arm's"
                " label"
            )
        if base == ARM_CPU:
            var mc_cpu_t0 = perf_counter_ns()
            var mc_cpu_model = train_multiclass(
                data, labels, n_classes, params
            )
            var mc_cpu_s = Float64(perf_counter_ns() - mc_cpu_t0) / 1e9
            var mc_cpu_loss = -1.0
            if want_loss:
                mc_cpu_loss = _multiclass_loss(mc_cpu_model, data, labels)
            return ArmRun(
                mc_cpu_s, mc_cpu_loss, len(mc_cpu_model.trees)
            )
        var mc_gpu_t0 = perf_counter_ns()
        var mc_gpu_model = train_multiclass_gpu(
            data, labels, n_classes, params, split_search=mc_strategy
        )
        var mc_gpu_s = Float64(perf_counter_ns() - mc_gpu_t0) / 1e9
        var mc_gpu_loss = -1.0
        if want_loss:
            mc_gpu_loss = _multiclass_loss(mc_gpu_model, data, labels)
        return ArmRun(mc_gpu_s, mc_gpu_loss, len(mc_gpu_model.trees))

    if base == ARM_CPU:
        var cpu_t0 = perf_counter_ns()
        var cpu_model = train(data, target, objective, params)
        var cpu_s = Float64(perf_counter_ns() - cpu_t0) / 1e9
        var cpu_loss = -1.0
        if want_loss:
            cpu_loss = _train_loss(cpu_model, data, target, objective)
        return ArmRun(cpu_s, cpu_loss, len(cpu_model.trees))

    var strategy = SPLIT_SEARCH_AUTO
    if base == ARM_GPU_HOST:
        strategy = SPLIT_SEARCH_HOST
    elif base == ARM_GPU_DEVICE:
        strategy = SPLIT_SEARCH_DEVICE
    # Includes GpuHistogramBuilder construction and every transfer.
    var gpu_t0 = perf_counter_ns()
    var gpu_model = train_gpu(
        data,
        target,
        objective,
        params,
        split_search=strategy,
        row_unroll=unroll,
        narrow_index=narrow,
        pair_alignment=palign,
        min_tiles=0,
        rows_per_tile=rows_per_tile,
    )
    var gpu_s = Float64(perf_counter_ns() - gpu_t0) / 1e9
    var gpu_loss = -1.0
    if want_loss:
        gpu_loss = _train_loss(gpu_model, data, target, objective)
    return ArmRun(gpu_s, gpu_loss, len(gpu_model.trees))


def _min_of(values: List[Float64]) -> Float64:
    var m = values[0]
    for i in range(1, len(values)):
        if values[i] < m:
            m = values[i]
    return m


def _max_of(values: List[Float64]) -> Float64:
    var m = values[0]
    for i in range(1, len(values)):
        if values[i] > m:
            m = values[i]
    return m


def _median_of(values: List[Float64]) -> Float64:
    var s = List[Float64](capacity=len(values))
    for i in range(len(values)):
        s.append(values[i])
    for i in range(1, len(s)):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    var n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pct(fraction: Float64) -> Float64:
    """A fraction as a percentage rounded to one decimal place."""
    var scaled = fraction * 1000.0
    if scaled >= 0.0:
        return Float64(Int(scaled + 0.5)) / 10.0
    return Float64(Int(scaled - 0.5)) / 10.0


def _join_floats(values: List[Float64]) -> String:
    """Space-separated, full precision, in the order the repeats ran."""
    var out = String("")
    for i in range(len(values)):
        if i > 0:
            out += " "
        out += String(values[i])
    return out^


def _json_floats(values: List[Float64]) -> String:
    """The same list as a JSON array."""
    var out = String("[")
    for i in range(len(values)):
        if i > 0:
            out += ","
        out += String(values[i])
    out += "]"
    return out^


def _json_string(s: String) -> String:
    """A JSON string literal. Escapes the two characters that can appear.

    Only `"` and `\\` are escaped, because everything this file puts through
    here is an arm name, an objective word, or LightGBM's own resolved
    parameter line, and none of those carries a control character. A value
    that did would produce invalid JSON rather than wrong JSON, which is the
    failure mode to prefer for a record nobody re-reads by hand.
    """
    return String("\"", s.replace("\\", "\\\\").replace("\"", "\\\""), "\"")


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator present; GPU training benchmark skipped")
    else:
        var n_rows = 100_000
        var n_features = 100
        var objective = SQUARED_ERROR
        var obj_name = String("reg")
        # 0 is single output. Anything else selects softmax and makes
        # `objective` unread; see `_run_arm`.
        var n_classes = 0
        var repeats = 1
        var seed = 0
        var arms = List[Int]()
        arms.append(ARM_CPU)
        arms.append(ARM_GPU)
        var args = argv()
        if len(args) > 1:
            n_rows = Int(String(args[1]))
        if len(args) > 2:
            n_features = Int(String(args[2]))
        if len(args) > 3:
            obj_name = String(args[3])
            if obj_name == "binary":
                objective = BINARY_LOGISTIC
            elif obj_name == "multi":
                n_classes = 7
            elif obj_name.startswith("multi:"):
                n_classes = Int(
                    String(obj_name[byte=6 : obj_name.byte_length()])
                )
                if n_classes < 2:
                    raise Error(
                        "multi needs at least 2 classes; a one-class softmax"
                        " has nothing to separate"
                    )
            elif obj_name != "reg":
                raise Error(
                    "objective must be 'reg', 'binary', or"
                    " 'multi' / 'multi:<classes>'"
                )
        if len(args) > 4:
            repeats = Int(String(args[4]))
            if repeats < 1:
                raise Error("repeats must be at least 1")
        if len(args) > 5:
            arms = _parse_arms(String(args[5]))
        if len(args) > 6:
            seed = Int(String(args[6]))
        if n_features < 4:
            raise Error("need at least 4 features")

        var want_lgbm = False
        for a in range(len(arms)):
            if _arm_base(arms[a]) == ARM_LGBM:
                want_lgbm = True
        if want_lgbm and n_classes > 0:
            # The multiclass arms bucket the signal with `_class_labels`, a
            # stride-sampled quantile rule that exists only here. LightGBM
            # would have to be handed the same labels to be training on the
            # same problem, and it is not, so the arm refuses rather than
            # comparing two different problems.
            raise Error(
                "the lightgbm arm is single output only; multiclass would"
                " need `_class_labels` replicated on the Python side and it"
                " has not been"
            )

        # Every launch-shape arm's reachability, checked before a million
        # rows are generated rather than at the first repeat. Covers the
        # multiclass refusal the row-walk pair used to carry alone, plus the
        # index-width shape bound, the pair-load objective, the tile
        # geometry's growth plane, and the batched kernel family. See
        # `_check_arm_reachability`.
        _check_arm_reachability(arms, n_rows, n_features, objective, n_classes)

        # Same data as bench_train.mojo: column-major features, target from
        # features 0..3 plus a noise stream at counters >= n_rows * n_features.
        var features = List[Float64](capacity=n_rows * n_features)
        var seed_offset = UInt64(seed) * 0x9E3779B97F4A7C15
        for k in range(n_rows * n_features):
            features.append(_uniform(seed_offset + UInt64(k)))
        var noise_base = seed_offset + UInt64(n_rows * n_features)
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var x0 = features[0 * n_rows + r]
            var x1 = features[1 * n_rows + r]
            var x2 = features[2 * n_rows + r]
            var x3 = features[3 * n_rows + r]
            var signal = (
                5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)
            )
            var u = _uniform(noise_base + UInt64(r))
            if n_classes > 0:
                # The same signal, perturbed and then bucketed below. The
                # noise is wider than the regression arm's because a class
                # boundary needs rows on both sides of it to be worth a
                # split: a label that is a step function of the signal
                # alone is separable by four features and the trees stop
                # growing early, which measures the early stop and not the
                # backend.
                target.append(signal + 0.5 * (u - 0.5))
            elif objective == BINARY_LOGISTIC:
                var p = _sigmoid(2.0 * (signal - 3.0))
                target.append(1.0 if u < p else 0.0)
            else:
                target.append(signal + 0.1 * (u - 0.5))

        var labels = List[Int]()
        if n_classes > 0:
            labels = _class_labels(target, n_classes)

        print(
            "mojotrees gpu-vs-cpu bench:",
            n_rows,
            "rows x",
            n_features,
            "features,",
            obj_name,
            "seed",
            seed,
        )
        if n_classes > 0:
            print("classes:", n_classes)

        var t0 = perf_counter_ns()
        var mapper = fit_bins(features, n_rows, n_features, 255)
        var data = mapper.transform(features, n_rows)
        var t1 = perf_counter_ns()
        var binning_s = Float64(t1 - t0) / 1e9
        print("binning_s:", binning_s)

        var n_arms = len(arms)
        var arm_list = String("")
        for a in range(n_arms):
            if a > 0:
                arm_list += ","
            arm_list += _arm_name(arms[a])
        print("arms:", arm_list)
        # One line per arm holding every condition it passes, so that the
        # record a reader copies out carries the conditions and not only the
        # labels. Printed before the repeats for the same reason the canary's
        # verdict is printed before the summaries: it is what the numbers
        # below have to be read against.
        var any_tile_arm = False
        for a in range(n_arms):
            print("arm_conditions:", _arm_conditions(arms[a], n_rows))
            if _arm_touches_tiles(arms[a]):
                any_tile_arm = True
        if any_tile_arm:
            print(_tile_legend())
        print("repeats:", repeats)
        if repeats < 3:
            print(
                "warning: fewer than 3 repeats cannot separate a real"
                " difference from machine drift; pass a repeat count of 3 or"
                " more before reporting a delta"
            )

        # Built before the loop and outside every clock: the import, the
        # numpy regeneration of the same splitmix64 dataset, and LightGBM's
        # Dataset construction. Its binning time is printed beside ours
        # rather than folded into either arm's training number.
        var lgbm = Optional[PythonObject]()
        var lgbm_threads = 0
        var lgbm_binning_s = -1.0
        var lgbm_params = String("")
        if want_lgbm:
            var state = _lgbm_arm(
                n_rows,
                n_features,
                obj_name,
                seed,
                BoosterParams.default(),
                255,
            )
            lgbm_threads = Int(py=state.resolved_threads())
            lgbm_binning_s = Float64(py=state.binning_s)
            lgbm_params = String(py=state.summary())
            print("lightgbm_threads:", lgbm_threads)
            print("lightgbm_binning_s:", lgbm_binning_s)
            print("lightgbm_params:", lgbm_params)
            lgbm = Optional[PythonObject](state)

        # The regime canary's first reading. Taken here rather than at the top
        # of `main` on purpose: everything above -- data generation, binning,
        # the LightGBM import and its Dataset construction -- has already run,
        # so the process the canary reads is the process the arms will run in,
        # with the same memory resident and the same thread pools parked. A
        # reading taken before all that would describe a different machine
        # state than any arm experiences.
        var canary_on = canary_enabled()
        var baseline = load_baseline()
        var canary_start = CanaryReading(-1.0, 0, -1.0, -1.0, String("off"))
        var canary_end = CanaryReading(-1.0, 0, -1.0, -1.0, String("off"))
        if canary_on:
            print_baseline(baseline)
            canary_start = take_reading()
            print_reading(String("start"), canary_start, baseline)
        else:
            print(
                "canary: disabled by MOJOTREES_CANARY; this run carries no"
                " regime label and its arm timings cannot be placed in a"
                " window"
            )

        # Interleaved, not blocked: one repeat runs every arm before the next
        # repeat starts, so the samples being compared sit next to each other
        # in time. Running all of one arm and then all of the other puts the
        # two arms in different thermal windows and makes the difference
        # between them unreadable.
        var samples = List[Float64](capacity=repeats * n_arms)
        var losses = List[Float64](capacity=n_arms)
        var tree_counts = List[Int](capacity=n_arms)
        for _ in range(n_arms):
            losses.append(-1.0)
            tree_counts.append(0)
        for rep in range(repeats):
            for a in range(n_arms):
                var run = _run_arm(
                    arms[a],
                    data,
                    target,
                    labels,
                    n_classes,
                    objective,
                    rep == 0,
                    lgbm,
                )
                samples.append(run.seconds)
                if rep == 0:
                    losses[a] = run.loss
                    tree_counts[a] = run.n_trees
                print(
                    "run", rep + 1, _arm_name(arms[a]), "train_s:", run.seconds
                )

        # The canary's second reading, immediately after the last arm and
        # before any reduction, so that nothing between the two readings but
        # the arms themselves. The verdict is printed before the per-arm
        # summaries rather than after, because if the machine moved during the
        # run then every line below it is a comparison between arms in
        # different regimes and the reader should know that first.
        var regime = String("unmeasured")
        if canary_on:
            canary_end = take_reading()
            print_reading(String("end"), canary_end, baseline)
            regime = print_session_verdict(canary_start, canary_end, baseline)

        # The minimum leads because it is the sample least contaminated by
        # thermal drift and by the one-time device setup the first GPU run
        # pays. The spread beside it is what says whether the minimum can be
        # trusted at all.
        #
        # Every arm also prints its own repeats back, in the order they ran,
        # on one `<arm>_train_s_samples` line, and the same list goes into
        # the JSON record below. That is deliberate duplication of the `run N`
        # lines above: those interleave the arms, so reading one arm's
        # dispersion off them means picking every n-th line out of a
        # transcript by hand, and a spread that is only recoverable by hand is
        # a spread that gets dropped when a result is copied into a table.
        # The comparator is the reason this matters. LightGBM's own repeat
        # spread on this machine has never been recorded, so there is no
        # measured noise floor to read a few-percent margin against, and a
        # margin quoted against an unmeasured floor is not a result.
        #
        # Two spreads are reported per arm and they are not interchangeable.
        # `_spread_pct` is (max - min) / min, which is the one the verdict
        # below uses, kept unchanged so figures recorded before this line
        # existed still mean what they meant. `_spread_pct_of_median` is
        # (max - min) / median, which is the dispersion of the arm as a whole
        # rather than the excursion above its best sample; it is the smaller
        # of the two whenever the minimum is an outlier low, and it is the
        # honest one to quote beside a median. Both are printed rather than
        # one being chosen here, because the choice belongs to whoever quotes
        # the number.
        var mins = List[Float64](capacity=n_arms)
        var meds = List[Float64](capacity=n_arms)
        var maxs = List[Float64](capacity=n_arms)
        var spreads = List[Float64](capacity=n_arms)
        var spreads_med = List[Float64](capacity=n_arms)
        var per_arm = List[String](capacity=n_arms)
        for a in range(n_arms):
            var vals = List[Float64](capacity=repeats)
            for rep in range(repeats):
                vals.append(samples[rep * n_arms + a])
            var lo = _min_of(vals)
            var hi = _max_of(vals)
            var med = _median_of(vals)
            mins.append(lo)
            meds.append(med)
            maxs.append(hi)
            spreads.append((hi - lo) / lo)
            spreads_med.append((hi - lo) / med)
            var key = _arm_key(arms[a])
            print(key + "_train_s_samples:", _join_floats(vals))
            print(key + "_train_s:", lo)
            print(key + "_train_s_median:", med)
            print(key + "_train_s_max:", hi)
            print(key + "_spread_pct:", _pct(spreads[a]))
            print(key + "_spread_pct_of_median:", _pct(spreads_med[a]))
            print(key + "_n_trees:", tree_counts[a])
            print(key + "_train_loss:", losses[a])
            per_arm.append(
                String(
                    "{\"name\":",
                    _json_string(_arm_name(arms[a])),
                    ",\"samples_s\":",
                    _json_floats(vals),
                    ",\"min_s\":",
                    lo,
                    ",\"median_s\":",
                    med,
                    ",\"max_s\":",
                    hi,
                    ",\"spread_pct_of_min\":",
                    _pct(spreads[a]),
                    ",\"spread_pct_of_median\":",
                    _pct(spreads_med[a]),
                    ",\"n_trees\":",
                    tree_counts[a],
                    ",\"train_loss\":",
                    losses[a],
                    ",\"conditions\":",
                    _json_string(_arm_conditions(arms[a], n_rows)),
                    "}",
                )
            )

        # Every delta is reported against the noise floor that produced it,
        # taken as the wider of the two arms' own spreads. A gap smaller than
        # that floor is not a result however many decimals it carries, and
        # saying so here is the whole point of the repeat count.
        #
        # **Median, not minimum, and this was a real bug until 2026-08-16.**
        # This block computed both the delta and the floor from the arms'
        # *minima* while `PROFILE_PROTOCOL.md` requires the median and says
        # why: the minimum is the luckiest sample, and contention on this
        # machine is the finding rather than noise, so a statistic that
        # discards contention hides exactly what is being measured.
        #
        # It was not academic. At 50,000 rows against LightGBM the two
        # statistics disagreed on the verdict word: **resolved** on minima
        # (13.9 percent against a 10.8 percent floor), **consistent, not
        # resolved** on medians. That shape is this project's one claimed win
        # over LightGBM, at the size a user meets first, and it was resting on
        # the statistic the protocol had rejected in writing. Found by the CPU
        # campaign reading its own output rather than by anything here.
        #
        # `_spread_pct` (max-min)/min is still printed unchanged so figures
        # recorded before this commit still mean what they meant; what changed
        # is only which pair of numbers the verdict is computed from.
        var base_key = _arm_key(arms[0])
        var comparisons = List[String](capacity=n_arms)
        for a in range(1, n_arms):
            var key = _arm_key(arms[a])
            var delta = (meds[a] - meds[0]) / meds[0]
            var magnitude = delta if delta >= 0.0 else -delta
            var floor = (
                spreads_med[0] if spreads_med[0]
                > spreads_med[a] else spreads_med[a]
            )
            var verdict = String("unresolvable")
            # Median here too, so the printed ratio and the verdict beside
            # it cannot come from different statistics.
            print(key + "_speedup_x:", meds[0] / meds[a])
            if repeats == 1:
                # A single sample per arm has a spread of zero by
                # construction, not a noise floor of zero, so no delta can be
                # called resolved against it.
                print(
                    key + "_vs_" + base_key + ":",
                    "unresolvable delta_pct",
                    _pct(delta),
                    "noise_floor_pct unmeasured-at-1-repeat",
                )
            else:
                verdict = String(
                    "indistinguishable"
                ) if magnitude <= floor else String("resolved")
                print(
                    key + "_vs_" + base_key + ":",
                    verdict,
                    "delta_pct",
                    _pct(delta),
                    "noise_floor_pct",
                    _pct(floor),
                )
            comparisons.append(
                String(
                    "{\"arm\":",
                    _json_string(_arm_name(arms[a])),
                    ",\"baseline\":",
                    _json_string(_arm_name(arms[0])),
                    ",\"speedup_x\":",
                    meds[0] / meds[a],
                    ",\"delta_pct\":",
                    _pct(delta),
                    ",\"noise_floor_pct\":",
                    -1.0 if repeats == 1 else _pct(floor),
                    ",\"verdict\":",
                    _json_string(verdict),
                    "}",
                )
            )

        # One machine-readable record of the whole run, printed on one line
        # and, when MOJOTREES_BENCH_JSON names a path, written there too.
        # Printed unconditionally because the transcript is the artifact that
        # actually survives: this project has twice had to discard a number
        # because the conditions it was taken under were not written down
        # beside it, and the per-repeat samples are the first thing a
        # hand-copied summary loses. `noise_floor_pct` is -1 at one repeat,
        # which is the null and not a floor of zero.
        var record = String(
            "{\"n_rows\":",
            n_rows,
            ",\"n_features\":",
            n_features,
            ",\"objective\":",
            _json_string(obj_name),
            ",\"n_classes\":",
            n_classes,
            ",\"seed\":",
            seed,
            ",\"repeats\":",
            repeats,
            ",\"binning_s\":",
            binning_s,
            ",\"arms\":[",
        )
        for i in range(len(per_arm)):
            if i > 0:
                record += ","
            record += per_arm[i]
        record += "],\"comparisons\":["
        for i in range(len(comparisons)):
            if i > 0:
                record += ","
            record += comparisons[i]
        record += "]"
        if want_lgbm:
            record += String(
                ",\"lightgbm\":{\"threads\":",
                lgbm_threads,
                ",\"binning_s\":",
                lgbm_binning_s,
                ",\"params\":",
                _json_string(lgbm_params),
                "}",
            )
        # One new key, added rather than replacing anything: every key above
        # is read by committed results under `bench/results/` and by other
        # sessions. Absent baselines come through as `null` ratios, which is
        # the record correctly saying the window is unlabelled rather than
        # claiming it was nominal.
        record += ",\"canary\":"
        if canary_on:
            record += canary_json(canary_start, canary_end, baseline, regime)
        else:
            record += canary_disabled_json()
        record += "}"
        print("json_summary:", record)
        var json_path = getenv("MOJOTREES_BENCH_JSON")
        if json_path.byte_length() > 0:
            with open(json_path, "w") as handle:
                handle.write(record + "\n")
            print("json_summary_path:", json_path)
