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

Each arm names its split-search strategy as an argument to `train_gpu`, not
through MOJOTREES_GPU_SPLIT_STRATEGY, so a mistyped or word-split shell
export cannot leave one arm running the other arm's code path under the
wrong label. `gpu` is whatever SPLIT_SEARCH_AUTO resolves to for the
workload; `gpu-host` and `gpu-device` pin the choice.

CPU-side threading honors MOJOTREES_NUM_WORKERS / MOJOTREES_PARALLEL_MIN_OPS
(see parallel.mojo), so pin those for reproducible comparisons.

The first repeat of the first GPU arm also pays one-time device setup, which
is why the summary leads with the minimum rather than the mean.

Usage: mojo run -I src bench/bench_train_gpu.mojo \\
    [n_rows] [n_features] [reg|binary] [repeats] [arms] [seed]

`arms` is a comma-separated list of cpu, gpu, gpu-host, gpu-device, in the
order they should run; the first is the baseline every other arm is compared
against. Any arm takes a `-depth` suffix (`cpu-depth`, `gpu-device-depth`,
...) to train the same configuration under `grow_policy=depthwise`; the
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
"""

from std.math import exp, log
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from mojotrees.binning import fit_bins
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    train,
)
from mojotrees.binning import BinnedMatrix
from mojotrees.growth_policy import GROW_DEPTHWISE
from mojotrees.train_gpu import (
    SPLIT_SEARCH_AUTO,
    SPLIT_SEARCH_DEVICE,
    SPLIT_SEARCH_HOST,
    train_gpu,
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


# An arm is a trainer plus, for the GPU trainer, an explicit split-search
# strategy. ARM_GPU leaves the choice to SPLIT_SEARCH_AUTO and so measures
# what a caller actually gets; the two pinned arms are what a comparison
# between the strategies has to use.
comptime ARM_CPU = 0
comptime ARM_GPU = 1
comptime ARM_GPU_HOST = 2
comptime ARM_GPU_DEVICE = 3
# Added to any of the above: the same arm under `grow_policy=depthwise`.
comptime ARM_DEPTHWISE = 4


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


def _arm_name(arm: Int) -> String:
    var base = _arm_base(arm)
    var name = String("gpu")
    if base == ARM_CPU:
        name = String("cpu")
    elif base == ARM_GPU_HOST:
        name = String("gpu-host")
    elif base == ARM_GPU_DEVICE:
        name = String("gpu-device")
    if _arm_depthwise(arm):
        name += "-depth"
    return name


def _arm_key(arm: Int) -> String:
    """`_arm_name` with the hyphens replaced, for `key: value` output lines."""
    return _arm_name(arm).replace("-", "_")


def _parse_arms(spec: String) raises -> List[Int]:
    """A comma-separated arm list, in the order the arms should run."""
    var arms = List[Int]()
    for part in spec.split(","):
        var name = String(part)
        if name.byte_length() == 0:
            continue
        var flags = 0
        if name.endswith("-depth"):
            flags = ARM_DEPTHWISE
            var trimmed = String(name[byte= : name.byte_length() - 6])
            name = trimmed^
        if name == "cpu":
            arms.append(ARM_CPU | flags)
        elif name == "gpu":
            arms.append(ARM_GPU | flags)
        elif name == "gpu-host":
            arms.append(ARM_GPU_HOST | flags)
        elif name == "gpu-device":
            arms.append(ARM_GPU_DEVICE | flags)
        else:
            raise Error(
                String(
                    "unknown arm '",
                    name,
                    "'; use cpu, gpu, gpu-host, or gpu-device, each with an"
                    " optional -depth suffix",
                )
            )
    if len(arms) == 0:
        raise Error("no arms selected")
    return arms^


def _run_arm(
    arm: Int,
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    want_loss: Bool,
) raises -> ArmRun:
    """Time one complete training run on one arm.

    The loss pass sits outside the timed region and is requested on the first
    repeat only: the fit is deterministic, so scoring it again on every
    repeat would only add prediction time to a wall clock that is meant to
    measure training.
    """
    var params = BoosterParams.default()
    if _arm_depthwise(arm):
        params.tree.grow_policy = GROW_DEPTHWISE
    var base = _arm_base(arm)
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
        data, target, objective, params, split_search=strategy
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


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator present; GPU training benchmark skipped")
    else:
        var n_rows = 100_000
        var n_features = 100
        var objective = SQUARED_ERROR
        var obj_name = String("reg")
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
            elif obj_name != "reg":
                raise Error("objective must be 'reg' or 'binary'")
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
            if objective == BINARY_LOGISTIC:
                var p = _sigmoid(2.0 * (signal - 3.0))
                target.append(1.0 if u < p else 0.0)
            else:
                target.append(signal + 0.1 * (u - 0.5))

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

        var t0 = perf_counter_ns()
        var mapper = fit_bins(features, n_rows, n_features, 255)
        var data = mapper.transform(features, n_rows)
        var t1 = perf_counter_ns()
        print("binning_s:", Float64(t1 - t0) / 1e9)

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
                    arms[a], data, target, objective, rep == 0
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
        var mins = List[Float64](capacity=n_arms)
        var spreads = List[Float64](capacity=n_arms)
        for a in range(n_arms):
            var vals = List[Float64](capacity=repeats)
            for rep in range(repeats):
                vals.append(samples[rep * n_arms + a])
            var lo = _min_of(vals)
            var hi = _max_of(vals)
            mins.append(lo)
            spreads.append((hi - lo) / lo)
            var key = _arm_key(arms[a])
            print(key + "_train_s:", lo)
            print(key + "_train_s_median:", _median_of(vals))
            print(key + "_train_s_max:", hi)
            print(key + "_spread_pct:", _pct(spreads[a]))
            print(key + "_n_trees:", tree_counts[a])
            print(key + "_train_loss:", losses[a])

        # Every delta is reported against the noise floor that produced it,
        # taken as the wider of the two arms' own spreads. A gap smaller than
        # that floor is not a result however many decimals it carries, and
        # saying so here is the whole point of the repeat count.
        var base_key = _arm_key(arms[0])
        for a in range(1, n_arms):
            var key = _arm_key(arms[a])
            var delta = (mins[a] - mins[0]) / mins[0]
            var magnitude = delta if delta >= 0.0 else -delta
            var floor = spreads[0] if spreads[0] > spreads[a] else spreads[a]
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
                print(
                    key + "_vs_" + base_key + ":",
                    "indistinguishable" if magnitude <= floor else "resolved",
                    "delta_pct",
                    _pct(delta),
                    "noise_floor_pct",
                    _pct(floor),
                )
