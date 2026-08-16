"""The oblivious level engine A/B, arms interleaved inside one process.

Two producers of one level's leaf statistics, the same everything else:

  level     `MOJOTREES_OBLIVIOUS_LEVEL_ENGINE=1`, CatBoost's arrangement --
            fan out over candidate features, one contiguous pass over the
            level's kept documents per feature, every leaf folded into a
            private `[slot][feature][bin]` stripe
  leafwise  `MOJOTREES_OBLIVIOUS_LEVEL_ENGINE=0`, the builder that shipped,
            which walks the whole `n_features * n_rows` bin matrix once per
            LEAF of the level

**The arms alternate inside one process, one after the other, every repeat.**
`bench/results/MACHINE_LOCK.md` is explicit about why and the evidence is in
it: this machine measured the same configuration at 6.115, 9.655 and 13.047
seconds across twelve minutes under an uncontested timing lock, a 2.1x
degradation from another session's compiles, while the within-pair RATIO held
to within 1.5 points. A blocked A-then-B measures the drift. Adjacent arms
survive it.

Twelve repeats, not five, because both arms climb for about eight before
plateauing and a five-repeat median measures the warm-up rather than the
plateau. The report prints every repeat, the median and range of each arm,
and the spread, and it says **indistinguishable** rather than picking a
winner when the ranges overlap.

Usage:
  mojo run -I src bench/bench_oblivious_level.mojo
      [rows] [features] [depth] [trees] [repeats]

Defaults are the decision row of `bench/results/COMPARISON_RUN_2026-08-16.md`
Block A: 799,110 rows by 100 features, depth 6, 100 trees.

What this arm is NOT
--------------------
It is the CatBoost SHAPE, not the CatBoost-mode arm of the comparison run.
Two of that arm's twelve keys are deliberately absent, and their absence is
declared here rather than discovered from a number that does not reproduce:

- `random_strength=1.0`. It needs the per-round noise scale that only the
  boosting loops compute, and it is a per-candidate cost that lands on both
  arms identically, so leaving it out changes the level and not the ratio.
- `bootstrap_type=MVS, subsample=0.8`. Mojo-API-only, and it would shrink
  every level's document count by a fifth. Again both arms alike, so it moves
  the level and not the ratio -- but it means the absolute seconds here are
  NOT comparable with Block A's 14.427 and must not be put in that table.

The ratio is what this file is for. The absolute seconds are a sanity check
on the shape and nothing more.
"""

from std.os import setenv
from std.sys import argv
from std.time import perf_counter_ns

from mojotrees.binning import fit_bins
from mojotrees.boosting import SQUARED_ERROR, train
from mojotrees.params import parse_params
from mojotrees.trainset import to_column_major


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _median(values: List[Float64]) -> Float64:
    var sorted = values.copy()
    for i in range(1, len(sorted)):
        var v = sorted[i]
        var j = i - 1
        while j >= 0 and sorted[j] > v:
            sorted[j + 1] = sorted[j]
            j -= 1
        sorted[j + 1] = v
    var n = len(sorted)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return sorted[n // 2]
    return 0.5 * (sorted[n // 2 - 1] + sorted[n // 2])


def _lo(values: List[Float64]) -> Float64:
    var out = values[0]
    for i in range(1, len(values)):
        if values[i] < out:
            out = values[i]
    return out


def _hi(values: List[Float64]) -> Float64:
    var out = values[0]
    for i in range(1, len(values)):
        if values[i] > out:
            out = values[i]
    return out


def main() raises:
    var n_rows = 799_110
    var n_features = 100
    var max_depth = 6
    var n_trees = 100
    var repeats = 12
    var args = argv()
    if len(args) > 1:
        n_rows = Int(String(args[1]))
    if len(args) > 2:
        n_features = Int(String(args[2]))
    if len(args) > 3:
        max_depth = Int(String(args[3]))
    if len(args) > 4:
        n_trees = Int(String(args[4]))
    if len(args) > 5:
        repeats = Int(String(args[5]))
    if n_features < 4:
        raise Error("need at least 4 features")

    # The same counter stream bench_train.mojo and bench_lightgbm.py use, so
    # a number here and a number there describe the same data.
    var raw = List[Float64](unsafe_uninit_length=n_rows * n_features)
    for r in range(n_rows):
        for f in range(n_features):
            raw[r * n_features + f] = _uniform(UInt64(f * n_rows + r))
    var features = to_column_major(raw, n_rows, n_features)
    _ = raw^

    var noise_base = UInt64(n_rows * n_features)
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        var x3 = features[3 * n_rows + r]
        var signal = 5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)
        target.append(signal + 0.1 * (_uniform(noise_base + UInt64(r)) - 0.5))

    var mapper = fit_bins(features, n_rows, n_features, 255)
    var data = mapper.transform(features, n_rows)

    var spec = String(
        "grow_policy=symmetrictree max_depth=",
        String(max_depth),
        " num_leaves=",
        String(1 << max_depth),
        " min_data_in_leaf=1 min_child_weight=0.0 lambda_l1=0.0",
        " lambda_l2=3.0 score_function=cosine learning_rate=0.1",
        " n_estimators=",
        String(n_trees),
    )
    var config = parse_params(spec)

    print("mojotrees oblivious level A/B")
    print("shape:", n_rows, "rows x", n_features, "features")
    print("depth:", max_depth, " trees:", n_trees, " repeats:", repeats)
    print("params:", spec)
    print("")

    var level_s = List[Float64](capacity=repeats)
    var leafwise_s = List[Float64](capacity=repeats)
    var loss_level = 0.0
    var loss_leafwise = 0.0

    for rep in range(repeats):
        _ = setenv("MOJOTREES_OBLIVIOUS_LEVEL_ENGINE", "1")
        var t0 = perf_counter_ns()
        var b_level = train(data, target, SQUARED_ERROR, config.booster)
        var t1 = perf_counter_ns()

        _ = setenv("MOJOTREES_OBLIVIOUS_LEVEL_ENGINE", "0")
        var t2 = perf_counter_ns()
        var b_leafwise = train(data, target, SQUARED_ERROR, config.booster)
        var t3 = perf_counter_ns()
        _ = setenv("MOJOTREES_OBLIVIOUS_LEVEL_ENGINE", "")

        level_s.append(Float64(t1 - t0) / 1e9)
        leafwise_s.append(Float64(t3 - t2) / 1e9)
        # One row's prediction from each arm, printed once, so a run that
        # accidentally trained the same arm twice is visible rather than
        # averaged into a clean-looking null.
        if rep == 0:
            loss_level = b_level.predict_row(data, 0)
            loss_leafwise = b_leafwise.predict_row(data, 0)
        print(
            "repeat",
            rep,
            " level_s:",
            level_s[rep],
            " leafwise_s:",
            leafwise_s[rep],
        )

    var m_level = _median(level_s)
    var m_leafwise = _median(leafwise_s)
    print("")
    print(
        "level    median:",
        m_level,
        " [",
        _lo(level_s),
        ",",
        _hi(level_s),
        "]",
    )
    print(
        "leafwise median:",
        m_leafwise,
        " [",
        _lo(leafwise_s),
        ",",
        _hi(leafwise_s),
        "]",
    )
    print("speedup (leafwise / level):", m_leafwise / m_level)
    print("row 0 prediction, level arm:   ", loss_level)
    print("row 0 prediction, leafwise arm:", loss_leafwise)

    # The verdict rule, and it is the one this repository already uses: the
    # comparison is RESOLVED only when the ranges do not overlap. Anything
    # else is indistinguishable and gets called that, rather than having its
    # medians differenced and reported as a percentage.
    if _hi(level_s) < _lo(leafwise_s):
        print("verdict: RESOLVED, the level engine is faster")
    elif _hi(leafwise_s) < _lo(level_s):
        print("verdict: RESOLVED, the level engine is SLOWER")
    else:
        print("verdict: INDISTINGUISHABLE, the ranges overlap")
