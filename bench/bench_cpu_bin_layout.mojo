"""CPU bin-layout A/B, arms interleaved in one process.

Why this file exists rather than another arm on `bench_train_gpu.mojo`. The
layout a fit runs is a *per-fit snapshot* (`tree.GrowScratch` reads
`MOJOTREES_CPU_BIN_LAYOUT` once, in its constructor), so the two arms are
selected by a `setenv` between fits and not by an argument threaded through a
trainer. That is a different mechanism from every arm in that file, which
passes its condition as a parameter, and mixing the two there would put an
environment-selected arm beside parameter-selected ones under one label.

**Interleaved, because this box drifts.** An A-block then a B-block measures
the drift and not the change: the campaign has a recorded case of the same
command on the same code measuring `indistinguishable` in a busy window and
`resolved` in a quiet one. Arms alternate inside one repeat, so a drift that
lands mid-run hits every arm alike.

**LightGBM is an arm here for the same reason it is one there**, reached
through `bench/bench_lightgbm.py` in this process. It is the anchor a
cross-invocation reading is normalized against; without it the two mojotrees
arms are comparable to each other and to nothing else.

**The digest is not decoration.** The two layouts are documented to be
bit-identical -- same plan, same rows, same order, only the address the bin id
is loaded from differs -- and this file prints a digest of each arm's model so
that the claim is checked on every run rather than asserted. The digest is
over `predict_row` outputs, bitwise, so two models that differ in the last ulp
of one leaf produce different digests. `MOJOTREES_NUM_WORKERS` moves the task
count and must not move the digest; running this file twice at different
worker counts is the determinism check, and it is a check rather than a
tolerance.

Usage:

    bench-cpu-bin-layout [rows] [features] [repeats] [arms] [seed]

`arms` is a comma-separated list of `feature`, `row`, `auto`, `group8`,
`group16` and `lightgbm`, defaulting to all six. The first named arm is the
baseline every other is reported against. Regression objective only: the
layout question is about the accumulate, and one objective is enough to ask
it.

The six arms answer three questions in one window, which is the point: which
layout is faster (`feature` against `row`), whether the shipped default
reaches it (`auto`), and whether the derived interleave width is the right
one (`group8` and `group16` against `feature`). All five mojotrees arms are
documented to be scheduling-only and must therefore print the same digest;
that check is free here and is printed as a verdict.

At 799,110 x 100 with twelve repeats this is roughly eleven minutes of
machine. Name fewer arms if the window is tighter, and name `feature,row,
auto,lightgbm` if only the layout question matters.

**`auto` is an arm and not a convenience.** It is what a user who sets
nothing actually gets, and it runs the timed probe
(`tree.GrowScratch.resolve_layout_timed`) rather than either pinned layout.
Without it a run could establish which layout is faster and say nothing about
whether the shipped default reaches it, which is this repository's most
frequently repeated defect: five settings were found in one day that were
accepted and then quietly ignored. `auto` landing on the slower of the two
pinned arms is a defect report, and it is one this file can produce.

The traffic table printed before the loop is a **derived bound**, arithmetic
over bytes and counts and nothing else. It is what the decomposition claim
about streamed traffic has to be checked against, and it is printed whether
or not the row-major arm is selected, because a bound that only appears when
the change is on is a bound nobody can falsify.
"""

from std.collections import Optional
from std.memory import bitcast
from std.os import getenv, setenv
from std.python import Python, PythonObject
from std.sys import argv
from std.time import perf_counter_ns

from mojotrees.apple_cpu_policy import (
    CpuProfile,
    feature_group_count,
    plan_feature_group,
    plan_row_block_count,
)
from mojotrees.binning import BinnedMatrix, fit_bins
from mojotrees.boosting import (
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    train,
)


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


comptime ARM_FEATURE = 0
comptime ARM_ROW = 1
comptime ARM_AUTO = 2
comptime ARM_LGBM = 3
# Feature-major at a wider interleave than the derived width chooses. These
# are the *same layout* as `feature` with one number changed, and they are
# here because that number is the multiplier on the term the traffic table
# says dominates: the feature-major kernel re-walks the node's row ids and
# gathered derivatives once per feature group, and the group count is
# `ceil(n_active / width)`. Doubling the width halves that term.
#
# The width is derived rather than named, and its binding clamp at 255 bins
# is `cache_feature_group`, which divides `ASSUMED_L1D_BYTES` by two and by
# a 24-byte cell. `ASSUMED_L1D_BYTES` is 64 KiB -- a stated portable floor,
# not this machine's cache, and `apple_cpu_policy`'s own docstring says the
# measurement that would move it does not exist. These two arms are that
# measurement, taken through the knob the module points at
# (`MOJOTREES_CPU_FEATURE_GROUP`) rather than by editing the constant, so a
# result here is evidence about the floor without anybody having shipped a
# change to it first.
comptime ARM_GROUP4 = 4
comptime ARM_GROUP8 = 5
comptime ARM_GROUP16 = 6


def _arm_name(arm: Int) -> String:
    if arm == ARM_FEATURE:
        return String("feature")
    if arm == ARM_ROW:
        return String("row")
    if arm == ARM_AUTO:
        return String("auto")
    if arm == ARM_GROUP4:
        return String("group4")
    if arm == ARM_GROUP8:
        return String("group8")
    if arm == ARM_GROUP16:
        return String("group16")
    return String("lightgbm")


def _arm_layout_word(arm: Int) -> String:
    """What this arm exports as `MOJOTREES_CPU_BIN_LAYOUT`.

    The three layout arms are named after the value they export, so a
    transcript line and the environment it was taken under cannot disagree
    about which layout ran. The feature-group arms are feature-major with a
    different interleave, so they export `feature`: leaving the layout at
    `auto` there would confound a width result with a layout result.
    """
    if arm == ARM_GROUP4 or arm == ARM_GROUP8 or arm == ARM_GROUP16:
        return String("feature")
    return _arm_name(arm)


def _arm_group_word(arm: Int) -> String:
    """What this arm exports as `MOJOTREES_CPU_FEATURE_GROUP`.

    Empty on every arm but the two width arms, and exported as empty rather
    than left alone, because these arms run in one process and an unset
    variable that a previous arm set is the classic way an A/B runs one arm
    under the other's label.
    """
    if arm == ARM_GROUP4:
        return String("4")
    if arm == ARM_GROUP8:
        return String("8")
    if arm == ARM_GROUP16:
        return String("16")
    return String("")


def _arm_from_name(word: String) raises -> Int:
    if word == "feature":
        return ARM_FEATURE
    if word == "row":
        return ARM_ROW
    if word == "auto":
        return ARM_AUTO
    if word == "group4":
        return ARM_GROUP4
    if word == "group8":
        return ARM_GROUP8
    if word == "group16":
        return ARM_GROUP16
    if word == "lightgbm" or word == "lgbm":
        return ARM_LGBM
    raise Error(
        String(
            'unknown arm "',
            word,
            '"; expected feature, row, auto, group4, group8, group16 or'
            " lightgbm",
        )
    )


def _model_digest(booster: Booster, data: BinnedMatrix) -> UInt64:
    """A bitwise digest of this model's predictions on a fixed row stride.

    Bitwise and not approximate: the question this answers is whether two
    accumulations produced the *same* Float64, and a tolerance would answer a
    different question. The stride keeps it to a few thousand predictions so
    that it is free against a training run, and it is a fixed stride rather
    than a sample so that two runs digest the same rows.
    """
    var step = data.n_rows // 4096
    if step < 1:
        step = 1
    var h = UInt64(0xCBF29CE484222325)
    var r = 0
    while r < data.n_rows:
        var bits = bitcast[DType.uint64, 1](booster.predict_row(data, r))
        h = _splitmix64(h ^ bits)
        r += step
    return h


def _train_loss(
    booster: Booster, data: BinnedMatrix, target: List[Float64]
) -> Float64:
    """Mean squared residual over every training row.

    **Called after the last repeat, never between two of them.** It is a
    serial pass of `n_rows` predictions through 100 trees, which at this
    shape is seconds of machine, and seconds of untimed machine landing in
    the middle of an interleaved sequence is the drift this file exists to
    avoid. `_model_digest` is what runs inside the sequence, and it is four
    thousand predictions rather than eight hundred thousand.
    """
    var loss = 0.0
    for r in range(data.n_rows):
        var d = booster.predict_row(data, r) - target[r]
        loss += d * d
    return loss / Float64(data.n_rows)


def _median(values: List[Float64]) -> Float64:
    var s = values.copy()
    for i in range(1, len(s)):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    var n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _plateau(values: List[Float64], drop: Int) -> List[Float64]:
    """The repeats past `drop`, which is where both engines plateau.

    Both engines climb for roughly eight repeats on this machine before
    levelling, so a median over every repeat is a median over two regimes.
    This is the tail, and the head is reported beside it rather than
    discarded silently.
    """
    var out = List[Float64]()
    for i in range(len(values)):
        if i >= drop:
            out.append(values[i])
    if len(out) == 0:
        return values.copy()
    return out^


def _traffic_report(
    data: BinnedMatrix, n_active: Int, const_h: Bool
) raises:
    """Derived bound on the bytes one histogram build streams, both layouts.

    Counts, per (row, feature) or per row, what the two accumulate kernels
    load and store outside their private histogram, which is the quantity the
    two layouts differ on. It does not count the private histogram's own
    traffic: that lives in L1 on the feature-major kernel and in L2 on the
    row-major one, it is the same number of accumulates either way, and
    pretending to price a cache hit here would bury the one term that is
    genuinely different. The two working-set figures printed at the end are
    what a reader should hold that omission against.

    - **row ids**: 8 bytes per row per accumulate pass. The feature-major
      kernel makes `ceil(n_active / group_width)` passes over the node's rows,
      one per feature group; the row-major kernel makes one.
    - **gathered derivatives**: 8 bytes per row per pass, and 8 rather than 4
      under a constant hessian on purpose. The gather buffer holds a packed
      `(g, h)` pair of Float32 whatever the objective is, so a `const_h`
      kernel reads only the gradient half but strides over both and touches
      every line either way. Charging 4 would be counting the loads and not
      the traffic.
    - **bin ids**: the feature-major kernel reads each active feature's binned
      column, charged at the whole column because a node holding half the
      dataset touches every line of it. The row-major kernel reads each row's
      record once, and a record covers every feature.
    - **zero and fold** of the private partials: written once and read once,
      `blocks * cells * stride * 8 * 2`.

    The group width and the block count are read from the same policy
    functions the kernels read, so this table cannot describe a dispatch the
    kernels are not making. That is the whole reason it is computed here
    rather than written into a document: a bound copied into prose goes stale
    the first time a constant moves, and this one recomputes itself on every
    run.

    Reported for **a child node holding half the dataset's rows**, not for the
    root. The root is where the two layouts differ least (its row list is the
    identity, so both read sequentially) and it is also the one node the
    row-major kernel cannot reach at all: an unbagged root goes through
    `build_histogram_into_scratch`, which has no row-major twin. Children are
    what a fit spends its time on and are where the question lives.

    Two counts are given for the row-major record stream, because they bound
    the answer from both sides: the **line** figure charges a whole
    `ASSUMED_CACHE_LINE_BYTES` per record touched, which is what a scattered
    read costs when the records are far apart, and the **byte** figure
    charges only `row_stride`, which is what it costs when consecutive row ids
    land in the same lines. A real node is between them.
    """
    var rows = data.n_rows // 2
    var n_bins = data.n_bins
    var stride = 2 if const_h else 3
    var pair_bytes = 8
    var blocks = plan_row_block_count(0, rows, n_bins, n_active, True)
    # `const_h` is passed, and this is the line that makes the table describe
    # the dispatch the kernel is actually making. Without it this reported
    # width 4 while a squared-error fit ran width 8, which is precisely the
    # "instrument describes a dispatch the kernels are not making" failure
    # this function's docstring claims it cannot have.
    var group = plan_feature_group(
        CpuProfile.detect(), n_bins, n_active, blocks, const_h
    )
    var groups = feature_group_count(n_active, group)
    var cells = n_active * n_bins
    # Zeroed once and read back once by the fold, both contiguous.
    var fold = blocks * cells * stride * 8 * 2
    # The feature-major kernel re-walks the node's row ids and its gathered
    # derivative pairs once per feature group, and reads each active feature's
    # binned column once. The column read is charged at the whole column,
    # since a node holding half the rows touches every line of it.
    var fm = (
        rows * groups * (8 + pair_bytes) + data.n_rows * n_active + fold
    )
    var compact = data.compact_bin_count()
    if compact == 0:
        compact = cells
    var rm_fold = blocks * compact * stride * 8 * 2
    var rm_line = rows * (8 + pair_bytes) + rows * 64 + rm_fold
    var rm_byte = rows * (8 + pair_bytes) + rows * data.row_stride + rm_fold
    print("traffic_bound_note: derived bound, bytes for one half-dataset node")
    print("traffic_bound_node_rows:", rows)
    print("traffic_bound_group_width:", group, "groups:", groups)
    print("traffic_bound_row_blocks:", blocks)
    print("traffic_bound_row_stride:", data.row_stride)
    print("traffic_bound_feature_major_mb:", Float64(fm) / (1024.0 * 1024.0))
    print(
        "traffic_bound_row_major_line_mb:",
        Float64(rm_line) / (1024.0 * 1024.0),
    )
    print(
        "traffic_bound_row_major_byte_mb:",
        Float64(rm_byte) / (1024.0 * 1024.0),
    )
    if rm_line > 0:
        print("traffic_bound_ratio_vs_line:", Float64(fm) / Float64(rm_line))
    if rm_byte > 0:
        print("traffic_bound_ratio_vs_byte:", Float64(fm) / Float64(rm_byte))
    # The counter-force, printed beside the win so that a reader who sees a
    # 3x traffic ratio and no 3x speedup has the reason in the same block.
    # The feature-major kernel's working set is one group's slices, sized to
    # fit L1 by construction; the row-major kernel's is the whole compact
    # histogram, which at this shape is far past it.
    print(
        "traffic_bound_feature_major_working_set_kb:",
        Float64(group * n_bins * stride * 8) / 1024.0,
    )
    print(
        "traffic_bound_row_major_working_set_kb:",
        Float64(compact * stride * 8) / 1024.0,
    )


def _lgbm_arm(
    n_rows: Int, n_features: Int, seed: Int, params: BoosterParams
) raises -> PythonObject:
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
                " the repository root. Underlying error: ",
                String(e),
            )
        )
    var threads = 0
    var s = getenv("MOJOTREES_LGBM_THREADS")
    if s.byte_length() == 0:
        s = getenv("MOJOTREES_NUM_WORKERS")
    if s.byte_length() > 0:
        try:
            var n = Int(s)
            threads = n if n > 0 else 0
        except:
            threads = 0
    return module.InterleavedArm(
        n_rows,
        n_features,
        String("reg"),
        threads,
        seed,
        rounds=params.n_estimators,
        learning_rate=params.learning_rate,
        num_leaves=params.tree.num_leaves,
        min_data_in_leaf=params.tree.min_data_in_leaf,
        lambda_l2=params.tree.lambda_reg,
        lambda_l1=params.tree.lambda_l1,
        min_child_hess=params.tree.min_child_hess,
        max_depth=params.tree.max_depth,
        max_bin=255,
    )


def main() raises:
    var args = argv()
    var n_rows = 799110
    var n_features = 100
    var repeats = 12
    var seed = 0
    var arm_words = String("feature,group4,row,auto,lightgbm")
    if len(args) > 1:
        n_rows = Int(String(args[1]))
    if len(args) > 2:
        n_features = Int(String(args[2]))
    if len(args) > 3:
        repeats = Int(String(args[3]))
        if repeats < 1:
            raise Error("repeats must be at least 1")
    if len(args) > 4:
        arm_words = String(args[4])
    if len(args) > 5:
        seed = Int(String(args[5]))
    if n_features < 4:
        raise Error("the generator needs at least 4 features")

    var arms = List[Int]()
    for part in arm_words.split(","):
        var given = String(part)
        if given.byte_length() == 0:
            continue
        arms.append(_arm_from_name(given))
    if len(arms) == 0:
        raise Error("no arms selected")

    # **The same sequence as `bench/bench_train_gpu.mojo` and
    # `bench/bench_lightgbm.py`, counter for counter.** Column-major features
    # over counters `[0, n_rows * n_features)` and the noise stream starting
    # at `n_rows * n_features`, all from one `seed_offset`. This is not a
    # detail to paraphrase: the LightGBM arm regenerates the dataset on the
    # Python side from that same counter sequence and never receives ours, so
    # a generator that drifted by one multiply would have the two engines
    # fitting different problems and the ratio would mean nothing. It is
    # copied rather than imported because a `bench/` module cannot be
    # imported from another `bench/` entry point without a package.
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
        var signal = 5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)
        var u = _uniform(noise_base + UInt64(r))
        target.append(signal + 0.1 * (u - 0.5))

    print(
        "mojotrees cpu bin-layout bench:",
        n_rows,
        "rows x",
        n_features,
        "features, reg seed",
        seed,
    )
    var t0 = perf_counter_ns()
    var mapper = fit_bins(features, n_rows, n_features, 255)
    var data = mapper.transform(features, n_rows)
    var t1 = perf_counter_ns()
    print("binning_s:", Float64(t1 - t0) / 1e9)
    print("has_row_major:", data.has_row_major())
    print("row_major_mb:", Float64(data.row_major_bytes()) / (1024.0 * 1024.0))
    print("packed_feature_count:", data.packed_feature_count())
    print("compact_bin_count:", data.compact_bin_count())
    _traffic_report(data, n_features, True)

    var arm_list = String("")
    for a in range(len(arms)):
        if a > 0:
            arm_list += ","
        arm_list += _arm_name(arms[a])
    print("arms:", arm_list)
    print("repeats:", repeats)
    if repeats < 3:
        print(
            "warning: fewer than 3 repeats cannot separate a real difference"
            " from machine drift"
        )

    var params = BoosterParams.default()
    var lgbm = Optional[PythonObject]()
    for a in range(len(arms)):
        if arms[a] == ARM_LGBM:
            var state = _lgbm_arm(n_rows, n_features, seed, params)
            print("lightgbm_threads:", Int(py=state.resolved_threads()))
            print("lightgbm_binning_s:", Float64(py=state.binning_s))
            print("lightgbm_params:", String(py=state.summary()))
            lgbm = Optional[PythonObject](state)

    var samples = List[List[Float64]]()
    var digests = List[UInt64]()
    var losses = List[Float64]()
    for _ in range(len(arms)):
        samples.append(List[Float64]())
        digests.append(UInt64(0))
        losses.append(-1.0)

    for rep in range(repeats):
        for a in range(len(arms)):
            var arm = arms[a]
            var seconds: Float64
            if arm == ARM_LGBM:
                var t = perf_counter_ns()
                _ = Int(py=lgbm.value().train())
                seconds = Float64(perf_counter_ns() - t) / 1e9
            else:
                # Both variables are set on every mojotrees arm, including to
                # the empty string, so no arm can inherit the previous arm's
                # setting. `GrowScratch` reads them in its constructor, which
                # `train` runs once per fit, so the value in force is the one
                # set here and a mid-fit change is not observed.
                _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", _arm_layout_word(arm))
                _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", _arm_group_word(arm))
                var t = perf_counter_ns()
                var model = train(data, target, SQUARED_ERROR, params)
                seconds = Float64(perf_counter_ns() - t) / 1e9
                if rep == 0:
                    digests[a] = _model_digest(model, data)
            samples[a].append(seconds)
            print("run", rep + 1, _arm_name(arm), "train_s:", seconds)

    # Everything from here is after the clock has stopped for good. The two
    # losses are the only cross-engine accuracy statement this file makes and
    # they each cost a full prediction pass, so they are taken here rather
    # than between two repeats where they would land inside somebody's
    # measurement.
    #
    # One extra untimed fit rather than a model kept from the loop: keeping
    # one would mean holding a `Booster` alive across the whole interleaved
    # sequence, and the arms are supposed to run in as close to the same
    # memory state as this file can arrange.
    var first_ours = -1
    for a in range(len(arms)):
        if arms[a] != ARM_LGBM and first_ours < 0:
            first_ours = a
    if first_ours >= 0:
        _ = setenv(
            "MOJOTREES_CPU_BIN_LAYOUT", _arm_layout_word(arms[first_ours])
        )
        _ = setenv(
            "MOJOTREES_CPU_FEATURE_GROUP", _arm_group_word(arms[first_ours])
        )
        var final_model = train(data, target, SQUARED_ERROR, params)
        losses[first_ours] = _train_loss(final_model, data, target)
    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "")
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "")
    for a in range(len(arms)):
        if arms[a] == ARM_LGBM:
            losses[a] = Float64(py=lgbm.value().loss())

    # Bit identity across the mojotrees arms, checked in the run that
    # measured them rather than in a separate one. Every arm here is
    # documented to be scheduling-only -- the layout changes which array a
    # bin id is loaded from, the interleave width changes how many features
    # share one walk of the rows, and neither changes the order in which any
    # one feature's bins are summed -- so all of them must produce the same
    # model to the bit. A mismatch is a defect and is printed as one.
    #
    # This does NOT establish determinism across `MOJOTREES_NUM_WORKERS`,
    # which is a different claim: every arm in one run shares one worker
    # count. Run this file twice at two worker counts and compare these same
    # lines for that.
    var ref_digest = UInt64(0)
    var have_ref = False
    var digest_ok = True
    for a in range(len(arms)):
        if arms[a] == ARM_LGBM:
            continue
        if not have_ref:
            ref_digest = digests[a]
            have_ref = True
        elif digests[a] != ref_digest:
            digest_ok = False
    if have_ref:
        print("digest_reference:", ref_digest)
        if digest_ok:
            print("digest_verdict: identical across every mojotrees arm")
        else:
            print(
                "digest_verdict: MISMATCH -- an arm documented as"
                " scheduling-only moved a bit; see the per-arm digests below"
            )

    var drop = repeats // 3
    print("plateau_drops_first:", drop)
    for a in range(len(arms)):
        var name = _arm_name(arms[a])
        var line = String(name, "_samples:")
        for i in range(len(samples[a])):
            line += String(" ", samples[a][i])
        print(line)
        var tail = _plateau(samples[a], drop)
        var lo = tail[0]
        var hi = tail[0]
        for i in range(len(tail)):
            if tail[i] < lo:
                lo = tail[i]
            if tail[i] > hi:
                hi = tail[i]
        var med = _median(tail)
        var spread_pct = 0.0
        if med > 0.0:
            spread_pct = 100.0 * (hi - lo) / med
        print(name, "plateau_median_s:", med)
        print(name, "plateau_min_s:", lo, "plateau_max_s:", hi)
        print(name, "plateau_spread_pct_of_median:", spread_pct)
        if arms[a] != ARM_LGBM:
            print(name, "digest:", digests[a])
        # `-1` means not measured on this arm, which is every mojotrees arm
        # but the first: the digest already says they fit the same model to
        # the bit, so a second full prediction pass per arm would buy a
        # number that is equal by construction.
        print(name, "train_loss:", losses[a])

    # The verdict, computed under M0 on medians: resolved when the medians
    # differ by more than the WIDER arm's own plateau spread. Printed as the
    # word rather than left to the reader, and printed as
    # `indistinguishable` when it is, because this campaign's failure mode is
    # a mean quoted as a result.
    for a in range(1, len(arms)):
        var t0m = _plateau(samples[0], drop)
        var t1m = _plateau(samples[a], drop)
        var m0 = _median(t0m)
        var m1 = _median(t1m)
        var lo0 = t0m[0]
        var hi0 = t0m[0]
        for i in range(len(t0m)):
            if t0m[i] < lo0:
                lo0 = t0m[i]
            if t0m[i] > hi0:
                hi0 = t0m[i]
        var s0 = hi0 - lo0
        var lo1 = t1m[0]
        var hi1 = t1m[0]
        for i in range(len(t1m)):
            if t1m[i] < lo1:
                lo1 = t1m[i]
            if t1m[i] > hi1:
                hi1 = t1m[i]
        var s1 = hi1 - lo1
        var widest = s0 if s0 > s1 else s1
        var delta = m0 - m1
        var mag = delta if delta > 0.0 else -delta
        var verdict = (
            String("resolved") if mag > widest else String("indistinguishable")
        )
        print(
            _arm_name(arms[0]),
            "vs",
            _arm_name(arms[a]),
            "median_delta_s:",
            delta,
            "widest_plateau_spread_s:",
            widest,
            "verdict:",
            verdict,
            "ratio:",
            m0 / m1 if m1 > 0.0 else 0.0,
        )
