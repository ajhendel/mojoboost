"""Early-stopping training on the GPU: where the gradients and the
validation scores live, measured against each other and against the CPU.

`train_gpu_with_valid` has two independent switches. `objective_source`
says whether the round's gradients are generated on the device from
device-resident raw scores or computed on the host and uploaded, with one
`predict_row` per training row per tree to keep the host raw scores
current. `valid_scoring` says whether the running validation scores are a
host list walked by `predict_row` or a device-resident vector advanced by
one kernel per round. This driver times every combination interleaved, in
one process, and prints each arm's best iteration and validation loss next
to its wall clock, because a faster arm that stops at a different round is
a different run and not a speedup.

The dataset is bench_train_gpu.mojo's synthetic regression, split into a
training block and a validation block drawn from the same stream. Arms:

  cpu     `train_with_valid` (the CPU trainer; the reference stopping rule)
  hg-hs   host gradients, host scorer (what `train_gpu_with_valid` shipped)
  dg-hs   device gradients, host scorer
  hg-ds   host gradients, device scorer
  dg-ds   device gradients, device scorer

Usage: mojo run -I src bench/bench_valid_scoring.mojo \\
    [n_rows] [n_valid] [n_features] [reg|binary] [repeats] [arms] [seed]

Defaults: 500000 rows, 100000 validation rows, 50 features, reg,
3 repeats, `cpu,hg-hs,dg-hs,hg-ds,dg-ds`, seed 0. `n_estimators` is 200
with `early_stopping_rounds` 10 so that most runs stop on the rule rather
than at the cap.

    pixi run bench-valid-scoring 500000 100000 50 reg 3
"""

from std.math import exp, log
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from mojotrees.binning import BinnedMatrix, fit_bins
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    train_with_valid,
)
from mojotrees.train_gpu import (
    OBJECTIVE_SOURCE_AUTO,
    OBJECTIVE_SOURCE_DEVICE,
    OBJECTIVE_SOURCE_HOST,
    VALID_SCORE_DEVICE,
    VALID_SCORE_HOST,
    train_gpu_with_valid,
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


def _loss(
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


comptime ARM_CPU = 0
comptime ARM_HG_HS = 1
comptime ARM_DG_HS = 2
comptime ARM_HG_DS = 3
comptime ARM_DG_DS = 4


def _arm_name(arm: Int) -> String:
    if arm == ARM_CPU:
        return String("cpu")
    if arm == ARM_HG_HS:
        return String("hg-hs")
    if arm == ARM_DG_HS:
        return String("dg-hs")
    if arm == ARM_HG_DS:
        return String("hg-ds")
    return String("dg-ds")


def _arm_key(arm: Int) -> String:
    if arm == ARM_HG_HS:
        return String("hg_hs")
    if arm == ARM_DG_HS:
        return String("dg_hs")
    if arm == ARM_HG_DS:
        return String("hg_ds")
    if arm == ARM_DG_DS:
        return String("dg_ds")
    return String("cpu")


def _parse_arms(spec: String) raises -> List[Int]:
    var arms = List[Int]()
    for part in spec.split(","):
        var name = String(part)
        if name.byte_length() == 0:
            continue
        if name == "cpu":
            arms.append(ARM_CPU)
        elif name == "hg-hs":
            arms.append(ARM_HG_HS)
        elif name == "dg-hs":
            arms.append(ARM_DG_HS)
        elif name == "hg-ds":
            arms.append(ARM_HG_DS)
        elif name == "dg-ds":
            arms.append(ARM_DG_DS)
        else:
            raise Error(
                String(
                    "unknown arm '",
                    name,
                    "'; use cpu, hg-hs, dg-hs, hg-ds, or dg-ds",
                )
            )
    if len(arms) == 0:
        raise Error("no arms selected")
    return arms^


struct ArmRun(Copyable, Movable):
    var seconds: Float64
    var valid_loss: Float64
    var n_trees: Int

    def __init__(out self, seconds: Float64, valid_loss: Float64, n_trees: Int):
        self.seconds = seconds
        self.valid_loss = valid_loss
        self.n_trees = n_trees


def _run_arm(
    arm: Int,
    data: BinnedMatrix,
    target: List[Float64],
    valid: BinnedMatrix,
    valid_target: List[Float64],
    objective: Int,
    params: BoosterParams,
    patience: Int,
    want_loss: Bool,
) raises -> ArmRun:
    var t0 = perf_counter_ns()
    var model: Booster
    if arm == ARM_CPU:
        model = train_with_valid(
            data, target, valid, valid_target, objective, params, patience
        )
    else:
        var source = OBJECTIVE_SOURCE_HOST
        if arm == ARM_DG_HS or arm == ARM_DG_DS:
            source = OBJECTIVE_SOURCE_DEVICE
        var scoring = VALID_SCORE_HOST
        if arm == ARM_HG_DS or arm == ARM_DG_DS:
            scoring = VALID_SCORE_DEVICE
        model = train_gpu_with_valid(
            data,
            target,
            valid,
            valid_target,
            objective,
            params,
            patience,
            valid_scoring=scoring,
            objective_source=source,
        )
    var seconds = Float64(perf_counter_ns() - t0) / 1e9
    var loss = -1.0
    if want_loss:
        loss = _loss(model, valid, valid_target, objective)
    return ArmRun(seconds, loss, len(model.trees))


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


def _pct(fraction: Float64) -> Float64:
    var scaled = fraction * 1000.0
    if scaled >= 0.0:
        return Float64(Int(scaled + 0.5)) / 10.0
    return Float64(Int(scaled - 0.5)) / 10.0


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator present; validation scoring benchmark skipped")
    else:
        var n_rows = 500_000
        var n_valid = 100_000
        var n_features = 50
        var objective = SQUARED_ERROR
        var obj_name = String("reg")
        var repeats = 3
        var seed = 0
        var arms = List[Int]()
        arms.append(ARM_CPU)
        arms.append(ARM_HG_HS)
        arms.append(ARM_DG_HS)
        arms.append(ARM_HG_DS)
        arms.append(ARM_DG_DS)
        var args = argv()
        if len(args) > 1:
            n_rows = Int(String(args[1]))
        if len(args) > 2:
            n_valid = Int(String(args[2]))
        if len(args) > 3:
            n_features = Int(String(args[3]))
        if len(args) > 4:
            obj_name = String(args[4])
            if obj_name == "binary":
                objective = BINARY_LOGISTIC
            elif obj_name != "reg":
                raise Error("objective must be 'reg' or 'binary'")
        if len(args) > 5:
            repeats = Int(String(args[5]))
            if repeats < 1:
                raise Error("repeats must be at least 1")
        if len(args) > 6:
            arms = _parse_arms(String(args[6]))
        if len(args) > 7:
            seed = Int(String(args[7]))
        if n_features < 4:
            raise Error("need at least 4 features")

        # One stream, training block first, then the validation block, so
        # the validation rows are fresh draws from the same distribution.
        var total = n_rows + n_valid
        var features = List[Float64](capacity=total * n_features)
        var seed_offset = UInt64(seed) * 0x9E3779B97F4A7C15
        for k in range(total * n_features):
            features.append(_uniform(seed_offset + UInt64(k)))
        var noise_base = seed_offset + UInt64(total * n_features)
        var all_target = List[Float64](capacity=total)
        for r in range(total):
            var x0 = features[0 * total + r]
            var x1 = features[1 * total + r]
            var x2 = features[2 * total + r]
            var x3 = features[3 * total + r]
            var signal = (
                5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)
            )
            var u = _uniform(noise_base + UInt64(r))
            if objective == BINARY_LOGISTIC:
                var p = _sigmoid(2.0 * (signal - 3.0))
                all_target.append(1.0 if u < p else 0.0)
            else:
                all_target.append(signal + 0.1 * (u - 0.5))

        # Column-major split: feature f's training rows are the first
        # n_rows of its column, its validation rows the rest.
        var train_x = List[Float64](capacity=n_rows * n_features)
        var valid_x = List[Float64](capacity=n_valid * n_features)
        for f in range(n_features):
            for r in range(n_rows):
                train_x.append(features[f * total + r])
            for r in range(n_valid):
                valid_x.append(features[f * total + n_rows + r])
        var target = List[Float64](capacity=n_rows)
        var valid_target = List[Float64](capacity=n_valid)
        for r in range(n_rows):
            target.append(all_target[r])
        for r in range(n_valid):
            valid_target.append(all_target[n_rows + r])

        print(
            "mojotrees validation-scoring bench:",
            n_rows,
            "train rows,",
            n_valid,
            "valid rows x",
            n_features,
            "features,",
            obj_name,
            "seed",
            seed,
        )
        var mapper = fit_bins(train_x, n_rows, n_features, 255)
        var data = mapper.transform(train_x, n_rows)
        var valid = mapper.transform(valid_x, n_valid)

        var params = BoosterParams.default()
        params.n_estimators = 200
        var patience = 10
        print("n_estimators:", params.n_estimators)
        print("early_stopping_rounds:", patience)

        var n_arms = len(arms)
        var arm_list = String("")
        for a in range(n_arms):
            if a > 0:
                arm_list += ","
            arm_list += _arm_name(arms[a])
        print("arms:", arm_list)
        print("repeats:", repeats)

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
                    valid,
                    valid_target,
                    objective,
                    params,
                    patience,
                    rep == 0,
                )
                samples.append(run.seconds)
                if rep == 0:
                    losses[a] = run.valid_loss
                    tree_counts[a] = run.n_trees
                print(
                    "run",
                    rep + 1,
                    _arm_name(arms[a]),
                    "s:",
                    run.seconds,
                    "n_trees:",
                    run.n_trees,
                )

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
            print(key + "_s:", lo)
            print(key + "_spread_pct:", _pct(spreads[a]))
            print(key + "_n_trees:", tree_counts[a])
            print(key + "_valid_loss:", losses[a])

        var base_key = _arm_key(arms[0])
        for a in range(1, n_arms):
            var key = _arm_key(arms[a])
            var delta = (mins[a] - mins[0]) / mins[0]
            var magnitude = delta if delta >= 0.0 else -delta
            var floor = spreads[0] if spreads[0] > spreads[a] else spreads[a]
            print(key + "_speedup_x:", mins[0] / mins[a])
            if repeats == 1:
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
