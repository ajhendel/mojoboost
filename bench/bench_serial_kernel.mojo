"""CPU serial-histogram-kernel A/B, arms interleaved in one process.

The arms are the four spellings of `histogram._accumulate_blocked_at`'s inner
scatter, selected by `MOJOTREES_CPU_SERIAL_KERNEL`, plus LightGBM. They are
**cumulative**, so each adjacent pair isolates one change:

| arm      | cell stride | cell write  | inactive lane of a full group |
|----------|-------------|-------------|-------------------------------|
| `base`   | runtime     | two scalars | tested, once per row          |
| `stride` | comptime    | two scalars | tested, once per row          |
| `packed` | comptime    | one 16-byte | tested, once per row          |
| `full`   | comptime    | one 16-byte | not tested                    |

`full` is the shipped default. `base` is the kernel as it was before this
round and is kept in the source for exactly this reason: a decomposition that
cannot be re-taken is a decomposition nobody can check.

**Why this is a file and not four commits measured one at a time.** This
machine drifts by factors of two and three across time windows, and this
repository has a recorded case of one command on one revision measuring
`indistinguishable` in a busy window and `resolved` in a quiet one. Four
sequential builds would measure the box. The arms alternate inside one
repeat, so a drift that lands mid-run hits every arm alike.

**Why LightGBM is an arm.** Two reasons, and the second is the one that made
it worth the minutes. First, it is the anchor a reading taken here is
normalized against, so a number from this file can be compared with a number
from another window. Second, this lane's target is stated as a *ratio* --
`1.405x behind at one thread` -- and a ratio needs both halves measured in
the same window or it is two numbers from two windows wearing one label. It
is reached through `bench/bench_lightgbm.py`, in this process, under this
process's `MOJOTREES_NUM_WORKERS`.

**The digest is the correctness gate and it is bitwise.** Every arm here is
documented to perform the same Float64 additions in the same order: the
stride is the same number whether it is a parameter or a variable, the packed
write is one SIMD add over two independent lanes where the scalar write was
two adds of the same two lanes, and the branch removal deletes a test whose
answer is already fixed for the whole dispatch unit. So all four must produce
the same model to the bit. A mismatch means the order moved and is a defect,
not a tolerance question.

This file does **not** establish determinism across `MOJOTREES_NUM_WORKERS`:
every arm in one run shares one worker count. Run it twice at two worker
counts and compare the `digest_reference` lines for that. That is the run the
lane owes and it is two invocations, not a flag.

**No traffic table.** Byte-count arguments are retired as predictors for this
kernel at this shape: one change cut streamed bytes 1.5x and measured zero,
and a separate one cut them 3.4x and measured a resolved regression. The
group-at-a-time builder serves most of the re-walk from cache. A number this
file cannot measure does not get printed beside numbers it can.

Usage:

    bench-serial-kernel [rows] [features] [repeats] [arms] [seed]

`arms` is a comma-separated list of `base`, `stride`, `packed`, `full` and
`lightgbm`, defaulting to all five. The first named arm is the baseline every
other is reported against. Regression objective only: the question is about
the accumulate, and one objective is enough to ask it.

Defaults are 799,110 rows x 100 features x 12 repeats, which is the decision
row this campaign has been comparing on. At one thread that is roughly ninety
minutes of machine with all five arms; name fewer arms if the window is
tighter, and `base,full,lightgbm` is the pair the lane's headline needs.

The canary is not taken here. Take it with `bench-canary` immediately before
the first invocation of a window and immediately after the last, which is
what bookends a *session* rather than one process.
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


comptime ARM_BASE = 0
comptime ARM_STRIDE = 1
comptime ARM_PACKED = 2
comptime ARM_FULL = 3
comptime ARM_LGBM = 4


def _arm_name(arm: Int) -> String:
    if arm == ARM_BASE:
        return String("base")
    if arm == ARM_STRIDE:
        return String("stride")
    if arm == ARM_PACKED:
        return String("packed")
    if arm == ARM_FULL:
        return String("full")
    return String("lightgbm")


def _arm_kernel_word(arm: Int) -> String:
    """What this arm exports as `MOJOTREES_CPU_SERIAL_KERNEL`.

    Every mojotrees arm is named after the value it exports, so a transcript
    line and the environment it was taken under cannot disagree about which
    kernel ran. It is exported on every mojotrees arm, never left alone,
    because an unset variable that a previous arm set is the classic way an
    A/B runs one arm under the other's label.
    """
    return _arm_name(arm)


def _arm_from_name(word: String) raises -> Int:
    if word == "base":
        return ARM_BASE
    if word == "stride":
        return ARM_STRIDE
    if word == "packed":
        return ARM_PACKED
    if word == "full":
        return ARM_FULL
    if word == "lightgbm" or word == "lgbm":
        return ARM_LGBM
    raise Error(
        String(
            'unknown arm "',
            word,
            '"; expected base, stride, packed, full or lightgbm',
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

    Called after the last repeat, never between two of them: it is a serial
    pass of `n_rows` predictions through a hundred trees, and seconds of
    untimed machine landing inside an interleaved sequence is the drift this
    file exists to avoid.
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

    Both engines climb for several repeats on this machine before levelling,
    so a median over every repeat is a median over two regimes. This is the
    tail; the head is printed beside it rather than discarded silently.
    """
    var out = List[Float64]()
    for i in range(len(values)):
        if i >= drop:
            out.append(values[i])
    if len(out) == 0:
        return values.copy()
    return out^


def _spread(values: List[Float64]) -> Float64:
    var lo = values[0]
    var hi = values[0]
    for i in range(len(values)):
        if values[i] < lo:
            lo = values[i]
        if values[i] > hi:
            hi = values[i]
    return hi - lo


def _lo(values: List[Float64]) -> Float64:
    var lo = values[0]
    for i in range(len(values)):
        if values[i] < lo:
            lo = values[i]
    return lo


def _hi(values: List[Float64]) -> Float64:
    var hi = values[0]
    for i in range(len(values)):
        if values[i] > hi:
            hi = values[i]
    return hi


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
                " environment's LightGBM; run under the bench environment from"
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
    var arm_words = String("base,stride,packed,full,lightgbm")
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

    # The same sequence as `bench/bench_train_gpu.mojo`,
    # `bench/bench_cpu_bin_layout.mojo` and `bench/bench_lightgbm.py`, counter
    # for counter: column-major features over counters `[0, n_rows *
    # n_features)` and the noise stream starting at `n_rows * n_features`, all
    # from one `seed_offset`. The LightGBM arm regenerates the dataset on the
    # Python side from that same counter sequence and never receives ours, so
    # a generator that drifted by one multiply would have the two engines
    # fitting different problems and the ratio would mean nothing.
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
        "mojotrees cpu serial-kernel bench:",
        n_rows,
        "rows x",
        n_features,
        "features, reg seed",
        seed,
    )
    var workers = getenv("MOJOTREES_NUM_WORKERS")
    print(
        "num_workers_env:",
        workers if workers.byte_length() > 0 else String("(unset, auto)"),
    )
    var t0 = perf_counter_ns()
    var mapper = fit_bins(features, n_rows, n_features, 255)
    var data = mapper.transform(features, n_rows)
    var t1 = perf_counter_ns()
    print("binning_s:", Float64(t1 - t0) / 1e9)

    # The dispatch the kernel will actually make, printed so that a reading
    # carries the shape it was taken at. Reported for a child node holding
    # half the dataset, which is where a fit spends its time; the root is one
    # node in sixty-one.
    var half = data.n_rows // 2
    var blocks = plan_row_block_count(0, half, data.n_bins, n_features, True)
    var group = plan_feature_group(
        CpuProfile.detect(), data.n_bins, n_features, blocks, True
    )
    var groups = feature_group_count(n_features, group)
    print("dispatch_node_rows:", half)
    print("dispatch_row_blocks:", blocks)
    print("dispatch_group_width:", group, "groups:", groups)
    print("dispatch_units:", blocks * groups)
    print("dispatch_full_groups:", n_features // group, "of", groups)

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
                # Set on every mojotrees arm, so no arm can inherit the
                # previous arm's setting. `histogram.serial_kernel_arm` reads
                # it once per node histogram build, so the value in force is
                # the one set here.
                _ = setenv("MOJOTREES_CPU_SERIAL_KERNEL", _arm_kernel_word(arm))
                var t = perf_counter_ns()
                var model = train(data, target, SQUARED_ERROR, params)
                seconds = Float64(perf_counter_ns() - t) / 1e9
                if rep == 0:
                    digests[a] = _model_digest(model, data)
            samples[a].append(seconds)
            print("run", rep + 1, _arm_name(arm), "train_s:", seconds)

    # Everything from here is after the clock has stopped for good.
    var first_ours = -1
    for a in range(len(arms)):
        if arms[a] != ARM_LGBM and first_ours < 0:
            first_ours = a
    if first_ours >= 0:
        _ = setenv(
            "MOJOTREES_CPU_SERIAL_KERNEL", _arm_kernel_word(arms[first_ours])
        )
        var final_model = train(data, target, SQUARED_ERROR, params)
        losses[first_ours] = _train_loss(final_model, data, target)
    _ = setenv("MOJOTREES_CPU_SERIAL_KERNEL", "")
    for a in range(len(arms)):
        if arms[a] == ARM_LGBM:
            losses[a] = Float64(py=lgbm.value().loss())

    # Bit identity across the mojotrees arms, checked in the run that measured
    # them rather than in a separate one.
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
                "digest_verdict: MISMATCH -- an arm documented to preserve"
                " every summation order moved a bit; see the per-arm digests"
                " below"
            )

    var drop = repeats // 3
    print("plateau_drops_first:", drop)
    var lgbm_median = -1.0
    for a in range(len(arms)):
        var name = _arm_name(arms[a])
        var line = String(name, "_samples:")
        for i in range(len(samples[a])):
            line += String(" ", samples[a][i])
        print(line)
        var tail = _plateau(samples[a], drop)
        var med = _median(tail)
        if arms[a] == ARM_LGBM:
            lgbm_median = med
        var spread_pct = 0.0
        if med > 0.0:
            spread_pct = 100.0 * _spread(tail) / med
        print(name, "plateau_median_s:", med)
        print(name, "plateau_min_s:", _lo(tail), "plateau_max_s:", _hi(tail))
        print(name, "plateau_spread_pct_of_median:", spread_pct)
        if arms[a] != ARM_LGBM:
            print(name, "digest:", digests[a])
        print(name, "train_loss:", losses[a])

    # The verdict, on medians: resolved when the medians differ by more than
    # the WIDER arm's own plateau spread. Printed as the word rather than left
    # to the reader, and printed as `indistinguishable` when it is, because
    # this campaign's failure mode is a mean quoted as a result.
    for a in range(1, len(arms)):
        var t0m = _plateau(samples[0], drop)
        var t1m = _plateau(samples[a], drop)
        var m0 = _median(t0m)
        var m1 = _median(t1m)
        var s0 = _spread(t0m)
        var s1 = _spread(t1m)
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

    # Adjacent pairs, which is what a cumulative ladder is for: `base` against
    # `stride` prices the comptime stride alone, `stride` against `packed` the
    # 16-byte cell alone, `packed` against `full` the deleted lane test alone.
    # Printed only for pairs that are actually adjacent in the named order.
    for a in range(1, len(arms)):
        if arms[a] == ARM_LGBM or arms[a - 1] == ARM_LGBM:
            continue
        var ta = _plateau(samples[a - 1], drop)
        var tb = _plateau(samples[a], drop)
        var ma = _median(ta)
        var mb = _median(tb)
        var sa = _spread(ta)
        var sb = _spread(tb)
        var widest = sa if sa > sb else sb
        var delta = ma - mb
        var mag = delta if delta > 0.0 else -delta
        print(
            "step",
            _arm_name(arms[a - 1]),
            "->",
            _arm_name(arms[a]),
            "median_delta_s:",
            delta,
            "widest_plateau_spread_s:",
            widest,
            "verdict:",
            String("resolved") if mag > widest else String(
                "indistinguishable"
            ),
            "ratio:",
            ma / mb if mb > 0.0 else 0.0,
        )

    # The cross-window anchor. Every mojotrees arm against LightGBM measured
    # in this same process, which is the only form in which a number here may
    # be compared with a number from another window.
    if lgbm_median > 0.0:
        for a in range(len(arms)):
            if arms[a] == ARM_LGBM:
                continue
            var med = _median(_plateau(samples[a], drop))
            print(
                "vs_lightgbm",
                _arm_name(arms[a]),
                "ratio_behind:",
                med / lgbm_median,
            )
