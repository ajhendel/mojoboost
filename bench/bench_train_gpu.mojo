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

`arms` is a comma-separated list of cpu, gpu, gpu-host, gpu-device,
lightgbm, row-unroll-on, row-unroll-off, in the order they should run; the
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
"""

from std.collections import Optional
from std.math import exp, log
from std.os import getenv
from std.python import Python, PythonObject
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

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


def _arm_name(arm: Int) -> String:
    if (arm & ARM_UNROLL_PAIR) != 0:
        if _arm_row_unroll(arm):
            return String("row-unroll-on")
        return String("row-unroll-off")
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

        # Suffixes, stripped in a loop so they compose in either order:
        # `gpu-device-depth-nounroll` and `gpu-device-nounroll-depth` are the
        # same arm.
        var flags = 0
        var named_unroll = False
        while True:
            if name.endswith("-nounroll"):
                flags |= ARM_NOUNROLL
                named_unroll = True
                var trimmed = String(name[byte= : name.byte_length() - 9])
                name = trimmed^
            elif name.endswith("-unroll"):
                # Sets no flag: on is the default. It is worth being able to
                # write anyway, because an A/B whose ON arm inherits its
                # condition has one arm's condition undeclared, and this file
                # already refuses to let a strategy be inherited for the same
                # reason.
                named_unroll = True
                var trimmed = String(name[byte= : name.byte_length() - 7])
                name = trimmed^
            elif name.endswith("-depth"):
                flags |= ARM_DEPTHWISE
                var trimmed = String(name[byte= : name.byte_length() - 6])
                name = trimmed^
            else:
                break

        if name == "cpu":
            if named_unroll:
                # `row_unroll` is a GPU histogram launch shape. The CPU
                # trainer has no such loop to reshape, so `cpu-nounroll`
                # would be plain `cpu` under a label claiming an arm.
                raise Error(
                    "the row-walk knob is a GPU histogram launch shape and"
                    " the CPU trainer has none, so 'cpu-nounroll' would be"
                    " plain cpu under a misleading label; use 'cpu'"
                )
            arms.append(ARM_CPU | flags)
        elif name == "gpu":
            arms.append(ARM_GPU | flags)
        elif name == "gpu-host":
            arms.append(ARM_GPU_HOST | flags)
        elif name == "gpu-device":
            arms.append(ARM_GPU_DEVICE | flags)
        elif name == "lightgbm":
            if flags != 0 or named_unroll:
                # `grow_policy` is a mojotrees and XGBoost switch. LightGBM
                # grows leaf-wise and has no depth-wise mode to ask for, so
                # `lightgbm-depth` would silently be plain `lightgbm` under a
                # label claiming otherwise. Compare a depth-wise mojotrees arm
                # against `lightgbm` and read the label as what it says. The
                # row-walk knob is ours and reaches nothing on that side at
                # all, so it is refused on the same grounds.
                raise Error(
                    "lightgbm has neither a depth-wise grow policy nor our"
                    " histogram row-walk knob to select, so 'lightgbm-depth'"
                    " or 'lightgbm-nounroll' would be plain lightgbm under a"
                    " misleading label; use 'lightgbm'"
                )
            arms.append(ARM_LGBM)
        else:
            raise Error(
                String(
                    "unknown arm '",
                    given,
                    "'; use cpu, gpu, gpu-host, gpu-device, lightgbm, or the"
                    " row-walk pair row-unroll-on / row-unroll-off. Any arm"
                    " but lightgbm takes a -depth suffix, and any GPU arm"
                    " takes -unroll or -nounroll.",
                )
            )
    if len(arms) == 0:
        raise Error("no arms selected")
    return arms^


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
        if _arm_touches_unroll(arm):
            # `train_multiclass_gpu` has no `row_unroll` argument, so an
            # unroll arm reaching here would train under the trainer's own
            # default whatever the arm asked for, and print a number under a
            # label claiming otherwise. `main` refuses this combination
            # before any data is generated; this is the second door on the
            # same room, because the first one is a check somebody can move.
            raise Error(
                "the row-walk arms are single output: train_multiclass_gpu"
                " takes no row_unroll argument, so a multiclass unroll arm"
                " would run the default under an arm's label"
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

        var want_unroll_arm = False
        for a in range(len(arms)):
            if _arm_touches_unroll(arms[a]):
                want_unroll_arm = True
        if want_unroll_arm and n_classes > 0:
            # `row_unroll` is threaded through `train_gpu` and not through
            # `train_multiclass_gpu`, so a multiclass unroll arm would train
            # under the trainer's default whichever arm it claimed to be, and
            # both arms of the pair would be the same run. Refused here,
            # before a million rows are generated, rather than at the first
            # repeat.
            raise Error(
                "the row-walk arms are single output: train_multiclass_gpu"
                " takes no row_unroll argument, so both arms of the pair"
                " would run the same code under two labels"
            )

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
                    "}",
                )
            )

        # Every delta is reported against the noise floor that produced it,
        # taken as the wider of the two arms' own spreads. A gap smaller than
        # that floor is not a result however many decimals it carries, and
        # saying so here is the whole point of the repeat count.
        var base_key = _arm_key(arms[0])
        var comparisons = List[String](capacity=n_arms)
        for a in range(1, n_arms):
            var key = _arm_key(arms[a])
            var delta = (mins[a] - mins[0]) / mins[0]
            var magnitude = delta if delta >= 0.0 else -delta
            var floor = spreads[0] if spreads[0] > spreads[a] else spreads[a]
            var verdict = String("unresolvable")
            print(key + "_speedup_x:", mins[0] / mins[a])
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
                    mins[0] / mins[a],
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
        record += "}"
        print("json_summary:", record)
        var json_path = getenv("MOJOTREES_BENCH_JSON")
        if json_path.byte_length() > 0:
            with open(json_path, "w") as handle:
                handle.write(record + "\n")
            print("json_summary_path:", json_path)
