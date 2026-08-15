"""End-to-end CPU vs GPU training benchmark on sparse (CSC) input.

The sparse twin of bench_train_gpu.mojo, and it keeps that driver's one
non-negotiable property: every arm is timed inside a single process, one
repeat running every arm before the next repeat starts, because benchmarks
on this machine drift by multiples across time windows and only samples
that sit next to each other in time may be compared. Two arms timed in
separate invocations cannot settle this question no matter how many
decimals they print, so the summary reports each arm's own spread and
refuses to call a gap smaller than that spread a result.

This is the `w4_sparse` workload of docs/APPLE_GPU_BENCHMARK_PROTOCOL.md:
`train_sparse` on the CPU against `train_gpu_sparse` on the device, over
the same `SparseBinnedMatrix`, binned once outside the timed region. The
sparse crossover is what `device_policy.crossover_rules()` has no entry for
and what keeps `auto` on the CPU for sparse input, and this driver is the
measurement that gap is waiting on.

Why its own file rather than two more arms in bench_train_gpu.mojo: the
dense driver's arms all take a `BinnedMatrix` and differ only in the
split-search strategy handed to `train_gpu`, while these take a
`SparseBinnedMatrix`, come from a different generator, and have no
split-search choice at all (`grow_tree_gpu_sparse` always searches splits
on the host). Sharing one file would mean a dense arm and a sparse arm
could be selected in one run, and that comparison is a different question
with its own driver (bench_sparse.mojo). What is shared is the protocol,
which is copied deliberately rather than abstracted, so neither driver can
lose the interleaving.

Density is a command-line argument because the crossover moves with it: the
sparse accumulator costs O(nnz_in_node) per node against the dense
O(rows * features), so where the device starts winning depends on how many
entries a node actually holds, not only on the row count.

What this harness must never configure, because `train_gpu_sparse` refuses
each one by name rather than approximating it:
  - exclusive feature bundling (`enable_bundle` stays off, `bundling` stays
    `none()`), since the device split kernel knows no bundle's local-bin
    table;
  - custom objectives, whose gradients come from a host callable;
  - an eval set, since `train_gpu_sparse_with_valid` scores validation on
    the host and would put host row walks inside a device measurement.
So the arms run `reg` or `binary` with the library defaults
(LightGBM-matched) and nothing else.

Data: the generator of bench_sparse.mojo, entry for entry. Each row holds a
fixed small number of nonzeros picked by a row-dependent stride, so the
nonzeros spread over every column, and at `seed 0` the dataset is
bit-identical to the one bench_sparse.mojo builds at the same shape. The
CSC arrays are built directly by counting sort; no dense matrix is ever
materialized, which is what makes the wide shapes in the protocol runnable.

CPU-side threading honors MOJOTREES_NUM_WORKERS / MOJOTREES_PARALLEL_MIN_OPS
(see parallel.mojo), so pin those for reproducible comparisons.

The first repeat of the first GPU arm also pays one-time device setup,
which is why the summary leads with the minimum rather than the mean.

Usage: mojo run -I src bench/bench_train_gpu_sparse.mojo \\
    [n_rows] [n_features] [density_pct] [reg|binary] [repeats] [arms] [seed]

`density_pct` is the fraction of stored entries per row, as a percentage:
`2.0` at 500 features is 10 nonzeros per row, which is the protocol's
`w4_sparse` shape. The realized density is printed beside the requested one
(a row's last strided feature can fall past the last column and is skipped,
exactly as in bench_sparse.mojo).

`arms` is a comma-separated list of cpu, gpu, in the order they should run;
the first is the baseline every other arm is compared against. Either takes
a `-depth` suffix (`cpu-depth`, `gpu-depth`) to train the same
configuration under `grow_policy=depthwise`; the leaf budget is unchanged,
so a depth-wise arm commits the same number of splits and issues the same
number of device launch groups as its leaf-wise twin. Defaults: 200000
rows, 500 features, 2.0% density, reg, 1 repeat, `cpu,gpu`.

    pixi run bench-train-gpu-sparse 200000 500 2.0 reg 5 cpu,gpu
    pixi run bench-train-gpu-sparse 1000000 500 0.4 reg 5 cpu,gpu
"""

from std.math import exp, log
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from mojotrees.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
)
from mojotrees.boosting_sparse import train_sparse
from mojotrees.growth_policy import GROW_DEPTHWISE
from mojotrees.sparse import (
    CscMatrix,
    SparseBinnedMatrix,
    SparseBinnedRows,
    fit_bins_csc,
    transform_csc,
)
from mojotrees.train_gpu_sparse import train_gpu_sparse
from mojotrees.tree_sparse import predict_row_sparse


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
    rows: SparseBinnedRows,
    target: List[Float64],
    objective: Int,
) -> Float64:
    """Training loss, scored by the sparse row walk rather than by
    densifying: each node's test binary-searches one row's own entries.

    Outside every timed region. A booster grown on the device and one grown
    on the host are both ordinary `Booster`s, so the same walk scores both
    and a fit difference between the arms shows up here rather than being
    invisible behind a wall clock."""
    var loss = 0.0
    for r in range(rows.n_rows):
        var raw = booster.base_score
        for i in range(len(booster.trees)):
            raw += booster.learning_rate * predict_row_sparse(
                booster.trees[i], rows, r
            )
        if objective == BINARY_LOGISTIC:
            var p = booster.response(raw)
            if p < 1e-15:
                p = 1e-15
            if p > 1.0 - 1e-15:
                p = 1.0 - 1e-15
            if target[r] > 0.5:
                loss -= log(p)
            else:
                loss -= log(1.0 - p)
        else:
            var d = booster.response(raw) - target[r]
            loss += d * d
    return loss / Float64(rows.n_rows)


# An arm is a trainer. Unlike the dense driver there is no split-search
# choice to pin: `grow_tree_gpu_sparse` searches splits on the host off
# downloaded histograms, always, so `gpu` is the only device arm there is.
comptime ARM_CPU = 0
comptime ARM_GPU = 1
# Added to either of the above: the same arm under `grow_policy=depthwise`.
comptime ARM_DEPTHWISE = 2


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
    var name = String("gpu")
    if _arm_base(arm) == ARM_CPU:
        name = String("cpu")
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
        else:
            raise Error(
                String(
                    "unknown arm '",
                    name,
                    "'; use cpu or gpu, each with an optional -depth suffix."
                    " The sparse GPU grower searches splits on the host, so"
                    " there is no gpu-host / gpu-device choice to pin here",
                )
            )
    if len(arms) == 0:
        raise Error("no arms selected")
    return arms^


def _run_arm(
    arm: Int,
    data: SparseBinnedMatrix,
    rows: SparseBinnedRows,
    target: List[Float64],
    objective: Int,
    want_loss: Bool,
) raises -> ArmRun:
    """Time one complete training run on one arm.

    The loss pass sits outside the timed region and is requested on the
    first repeat only: the fit is deterministic, so scoring it again on
    every repeat would only add prediction time to a wall clock that is
    meant to measure training.

    Every optional argument of both trainers is left at its default, which
    is what keeps bundling off, the objective built in, and the eval set
    absent -- the three things `train_gpu_sparse` refuses.
    """
    var params = BoosterParams.default()
    if _arm_depthwise(arm):
        params.tree.grow_policy = GROW_DEPTHWISE
    if _arm_base(arm) == ARM_CPU:
        var cpu_t0 = perf_counter_ns()
        var cpu_model = train_sparse(data, target, objective, params)
        var cpu_s = Float64(perf_counter_ns() - cpu_t0) / 1e9
        var cpu_loss = -1.0
        if want_loss:
            cpu_loss = _train_loss(cpu_model, rows, target, objective)
        return ArmRun(cpu_s, cpu_loss, len(cpu_model.trees))

    # Includes GpuSparseHistogramBuilder construction, the compressed
    # matrix upload, and every transfer.
    var gpu_t0 = perf_counter_ns()
    var gpu_model = train_gpu_sparse(data, target, objective, params)
    var gpu_s = Float64(perf_counter_ns() - gpu_t0) / 1e9
    var gpu_loss = -1.0
    if want_loss:
        gpu_loss = _train_loss(gpu_model, rows, target, objective)
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
        print("no accelerator present; sparse GPU training benchmark skipped")
    else:
        var n_rows = 200_000
        var n_features = 500
        var density_pct = 2.0
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
            density_pct = Float64(String(args[3]))
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
        if n_features < 8:
            raise Error("need at least 8 features")
        if density_pct <= 0.0 or density_pct > 100.0:
            raise Error("density_pct must be in (0, 100]")

        # Density is the parameter; nonzeros per row is what the generator
        # takes, so it is derived here and both are reported.
        var nnz_per_row = Int(
            density_pct / 100.0 * Float64(n_features) + 0.5
        )
        if nnz_per_row < 1:
            nnz_per_row = 1
        if nnz_per_row > n_features:
            nnz_per_row = n_features
        var stride = n_features // nnz_per_row

        # The generator of bench_sparse.mojo: each row picks `nnz_per_row`
        # features by stepping through the feature space with a
        # row-dependent stride, so the nonzeros spread over every column.
        # Built straight into CSC by counting sort -- pass one counts each
        # column, pass two fills it -- because the dense matrix that driver
        # materializes is what the wide protocol shapes cannot afford.
        var seed_offset = UInt64(seed) * 0x9E3779B97F4A7C15
        var col_offsets = List[Int](capacity=n_features + 1)
        col_offsets.resize(n_features + 1, 0)
        for r in range(n_rows):
            var offset = Int(
                _splitmix64(seed_offset + UInt64(r)) % UInt64(stride)
            )
            for j in range(nnz_per_row):
                var f = offset + j * stride
                if f < n_features:
                    col_offsets[f + 1] += 1
        for f in range(n_features):
            col_offsets[f + 1] += col_offsets[f]
        var nnz = col_offsets[n_features]

        var pos = List[Int](capacity=n_features)
        for f in range(n_features):
            pos.append(col_offsets[f])
        var row_index = List[Int](capacity=nnz)
        row_index.resize(nnz, 0)
        var values = List[Float64](capacity=nnz)
        values.resize(nnz, 0.0)
        var target = List[Float64](capacity=n_rows)
        # Rows are visited in ascending order, so each column's row indices
        # come out ascending, which is what `CscMatrix.validate` wants.
        for r in range(n_rows):
            var offset = Int(
                _splitmix64(seed_offset + UInt64(r)) % UInt64(stride)
            )
            var signal = 0.0
            for j in range(nnz_per_row):
                var f = offset + j * stride
                if f >= n_features:
                    continue
                var v = (
                    4.0
                    * _uniform(
                        seed_offset
                        + UInt64(r) * UInt64(n_features)
                        + UInt64(f)
                    )
                    - 2.0
                )
                var p = pos[f]
                row_index[p] = r
                values[p] = v
                pos[f] = p + 1
                # The target reads the first eight features, whose absent
                # entries are numerical zeros and so contribute nothing.
                if f < 8:
                    signal += (1.0 + 0.37 * Float64(f)) * v
            var u = _uniform(seed_offset + UInt64(7_000_000) + UInt64(r))
            var value = signal / 4.0
            if objective == BINARY_LOGISTIC:
                target.append(1.0 if u < _sigmoid(2.0 * value) else 0.0)
            else:
                target.append(value + 0.05 * (u - 0.5))

        var csc = CscMatrix(
            row_index^, values^, col_offsets^, n_rows, n_features
        )

        print(
            "mojotrees sparse gpu-vs-cpu bench (w4_sparse):",
            n_rows,
            "rows x",
            n_features,
            "features,",
            obj_name,
            "seed",
            seed,
        )
        print("nnz:", nnz)
        print("nnz_per_row_requested:", nnz_per_row)
        print("density_pct_requested:", density_pct)
        print("density_pct_realized:", 100.0 * csc.density())

        var t0 = perf_counter_ns()
        var mapper = fit_bins_csc(csc, 255)
        var data = transform_csc(mapper, csc)
        var t1 = perf_counter_ns()
        print("binning_s:", Float64(t1 - t0) / 1e9)

        # Row-oriented view for the loss pass only. Built once, outside
        # every timed region, and never handed to a trainer.
        var rows = data.to_rows()

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

        # Interleaved, not blocked: one repeat runs every arm before the
        # next repeat starts, so the samples being compared sit next to each
        # other in time. Running all of one arm and then all of the other
        # puts the two arms in different thermal windows and makes the
        # difference between them unreadable.
        var samples = List[Float64](capacity=repeats * n_arms)
        var losses = List[Float64](capacity=n_arms)
        var tree_counts = List[Int](capacity=n_arms)
        for _ in range(n_arms):
            losses.append(-1.0)
            tree_counts.append(0)
        for rep in range(repeats):
            for a in range(n_arms):
                var run = _run_arm(
                    arms[a], data, rows, target, objective, rep == 0
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
        # taken as the wider of the two arms' own spreads. A gap smaller
        # than that floor is not a result however many decimals it carries,
        # and saying so here is the whole point of the repeat count.
        var base_key = _arm_key(arms[0])
        for a in range(1, n_arms):
            var key = _arm_key(arms[a])
            var delta = (mins[a] - mins[0]) / mins[0]
            var magnitude = delta if delta >= 0.0 else -delta
            var floor = spreads[0] if spreads[0] > spreads[a] else spreads[a]
            print(key + "_speedup_x:", mins[0] / mins[a])
            if repeats == 1:
                # A single sample per arm has a spread of zero by
                # construction, not a noise floor of zero, so no delta can
                # be called resolved against it.
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
