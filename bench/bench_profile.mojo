"""Per-stage CPU profile of a boosting round.

Times every stage the CPU backend spends real work in, each one twice on
identical data: once with `MOJOTREES_NUM_WORKERS=1` (forced serial) and once
in auto mode (the shipped scheduler). The ratio of the two is the multicore
speedup attributable to that stage alone, which is what tuning the scheduler
moves; the serial column is what single-core optimization moves.

Stages, in the order a boosting round touches them:

- `bin_fit`        quantile edge fitting, per feature (sort-bound)
- `bin_transform`  raw values to bin ids, per feature (binary-search-bound)
- `grad_hess`      per-row gradients and hessians (elementwise)
- `hist_full`      root histogram over every row (scatter-bound)
- `hist_subset50`  child histogram over half the rows
- `hist_subset10`  child histogram over a tenth of the rows
- `hist_subtract`  sibling histogram by subtraction (SIMD elementwise)
- `split_scan`     best-split search over a built histogram
- `partition`      routing a node's rows to its two children
- `grow_tree`      one whole tree, as an integration check on the above
- `predict_batch`  the shipped batch scorer, `Booster.predict_batch_range`
- `predict_row_serial`  this benchmark's own serial loop over
  `Tree.predict_row` for one tree

Then a second table, the **node-size ladder**: the same node histogram build
the grower runs, at the row counts a real tree's nodes actually hold. See
"The ladder" below for why it is a separate table and what its extra columns
prove.

What each stage is, exactly, and where it is not the shipped path
-----------------------------------------------------------------
A stage's name is not a claim that the library calls that code. Three of
these stages are microbenchmarks of a kernel in isolation and cost more than
the grower's call to the same kernel does, in ways that matter at small node
sizes:

- `hist_full`, `hist_subset50`, and `hist_subset10` call `build_histogram`
  and `build_histogram_subset`, each of which allocates a fresh `Histogram`
  (three arrays of `n_features * n_bins`) per call, and the subset form also
  allocates the gradient/hessian pair buffer per call. The grower does
  neither: it takes a recycled buffer from `tree._HistPool` and passes one
  `pairs` list across the whole tree
  (`histogram.build_histogram_subset_into_scratch`). At 100 features and 256
  bins a `Histogram` is 25,600 cells at 24 bytes, so every one of those calls
  carries 614,400 bytes of allocation the grower amortizes away. The ladder
  below does *not* repeat that mistake; these three stages keep it so that
  numbers already recorded under these names stay comparable.
- `hist_subtract` calls `subtract_histogram`, which allocates its result;
  the grower calls `subtract_histogram_into`. Since the subtraction itself is
  one SIMD sweep over the same 614,400 bytes, this stage is plausibly
  allocation-dominated, and nothing here has established what share is which.
- `partition` calls `partition_split_rows`, which is `partition_rows_into`
  plus two freshly allocated `List[Int]`; the grower reuses its buffers.

`split_scan` is honest about its kernel (`tree._search` delegates to
`find_best_split`) but runs it at `find_best_split`'s own defaults rather
than at `TreeParams.default()`'s regularization, so its absolute time is a
kernel time and not a grower time.

Dataset generation matches bench_train.mojo and bench_lightgbm.py, so a
profile and an end-to-end run describe the same data.

The ladder, and the question it exists to answer
-------------------------------------------------
`hist_full`, `hist_subset50` and `hist_subset10` are one node size apart by
factors of two and ten -- at 1,000,000 rows that is 1,000,000, 500,000 and
100,000. A default 31-leaf tree spends most of its nodes far below the
smallest of those, so this table said nothing at all about them, and the gap
between a kernel that scales well in isolation and a `grow_tree` that does
not has two explanations that this file could not tell apart: per-node fixed
cost, or nodes falling below the scheduler's grain floor and running serial.

The ladder table times the node histogram at 100,000 / 30,000 / 10,000 /
3,000 / 1,000 / 300 rows, through
`histogram.build_histogram_subset_into_scratch` with a recycled output
buffer and a recycled pair buffer, which is the call the grower makes.
Alongside serial, auto, and the speedup it prints the **resolved task
count** of the accumulation dispatch in each arm, from
`parallel.plan_tasks` on `parallel.plan_row_blocks` with the same arguments
the kernel passes. That column is the point. A 1.0x speedup is ambiguous
between "fanned out and gained nothing" and "never fanned out"; a resolved
task count of 1 settles it, and a resolved count of 1 in the *serial* arm is
what proves the serial arm was serial rather than assumed to be.

The row lists are strided (`n_rows // size` apart) rather than contiguous,
which follows `hist_subset50` and `hist_subset10` above. A real node's rows
are ascending but clustered by the split feature's bin, so a strided list is
a pessimistic cache model of a real node and the ladder's absolute times are
an upper bound on a real node's, not an estimate of it.

This table is a microbenchmark and remains one. The in-run complement --
real trees, every node, a size breakdown, and the share of a whole fit --
is `src/mojotrees/phase_profile.mojo`, reached with
`MOJOTREES_PHASE_PROFILE=async` on any run that trains. Neither replaces the
other, and phase_profile cannot answer the resolved-task-count question at
all, because its host `dispatches` column is a structural constant charged
per call site rather than the count the scheduler actually chose.

Usage: mojo run -I src bench/bench_profile.mojo [n_rows] [n_features] [reps]
       [pred_trees]

Report the machine header with any numbers taken from this tool. The stage
times are wall-clock on a shared machine: run it on an idle box, and treat a
single run as an estimate rather than a measurement.
"""

from std.os import setenv
from std.sys import argv
from std.sys.info import (
    CompilationTarget,
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
    simd_width_of,
)
from std.time import perf_counter_ns

from mojotrees.apple_cpu_policy import (
    cpu_profile,
    derive_accumulation_plan,
    split_scan_ops,
)
from mojotrees.binning import BinMapper, BinnedMatrix, fit_bins
from mojotrees.boosting import (
    Booster,
    IterationRange,
    SQUARED_ERROR,
    fill_grad_hess,
)
from mojotrees.histogram import (
    Histogram,
    SIMD_LANES,
    build_histogram,
    build_histogram_subset,
    build_histogram_subset_into_scratch,
    subtract_histogram,
)
from mojotrees.parallel import (
    PARALLEL_MIN_OPS,
    TASKS_PER_CORE,
    env_parallel_min_ops,
    plan_row_blocks,
    plan_tasks,
)
from mojotrees.split import find_best_split
from mojotrees.tree import Tree, TreeParams, grow_tree, partition_split_rows

# How much work one row of a batch prediction is worth, in the histogram-op
# equivalents `parallel.plan_tasks` compares against its grain. This is
# `boosting._TRAVERSAL_ROW_OPS`, copied rather than imported because it is
# private to that module; it is used here only to *report* the task count
# `Booster.predict_batch_range` will resolve to, never to change one. If that
# constant moves and this does not, the reported task count for
# `predict_batch` goes wrong and nothing else does.
comptime TRAVERSAL_ROW_OPS = 8

# Reps for a ladder rung, as `LADDER_REP_BUDGET // node_rows` times the
# caller's `reps`, so a 300-row rung is not one clock tick wide. Scheduling
# only; it divides out of every number reported.
comptime LADDER_REP_BUDGET = 300_000


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _serial():
    _ = setenv("MOJOTREES_NUM_WORKERS", "1")


def _auto():
    _ = setenv("MOJOTREES_NUM_WORKERS", "0")


def _report(name: String, serial_s: Float64, auto_s: Float64):
    var speedup = serial_s / auto_s if auto_s > 0.0 else 0.0
    print(name, "|", serial_s, "|", auto_s, "|", speedup)


def _accum_tasks(plan_group_count: Int, active_ops: Int) -> Int:
    """Tasks the accumulation dispatch resolves to under the environment as
    it stands right now.

    `histogram._accumulate_subset_at` ends in
    `dispatch_feature_ranges(accumulate_groups, n_groups, active_ops)`, and
    `dispatch_feature_ranges` is `plan_tasks(n_features, total_ops)` followed
    by an even split. So this is that call's answer, taken with the same two
    arguments. It reads `MOJOTREES_NUM_WORKERS` and
    `MOJOTREES_PARALLEL_MIN_OPS` on every call, which is exactly why it is
    called once per arm rather than hoisted.
    """
    return plan_tasks(plan_group_count, active_ops)


def _report_ladder(
    node_rows: Int,
    serial_s: Float64,
    auto_s: Float64,
    serial_tasks: Int,
    auto_tasks: Int,
    gather_blocks: Int,
    group_width: Int,
    group_count: Int,
    active_ops: Int,
    compact: Bool,
):
    var speedup = serial_s / auto_s if auto_s > 0.0 else 0.0
    print(
        node_rows,
        "|",
        serial_s,
        "|",
        auto_s,
        "|",
        speedup,
        "|",
        serial_tasks,
        "|",
        auto_tasks,
        "|",
        gather_blocks,
        "|",
        group_width,
        "|",
        group_count,
        "|",
        active_ops,
        "|",
        "yes" if compact else "no",
    )


def _strided_rows(n_rows: Int, want: Int) -> List[Int]:
    """`want` ascending row ids spread evenly over `[0, n_rows)`.

    Strided rather than contiguous, following `hist_subset50` and
    `hist_subset10`. A real node's rows are ascending but clustered by the
    split feature's bin, so this is a pessimistic cache model of one: the
    times it produces bound a real node's from above and do not estimate it.
    """
    var stride = n_rows // want
    if stride < 1:
        stride = 1
    var rows = List[Int](capacity=want)
    var r = 0
    while r < n_rows and len(rows) < want:
        rows.append(r)
        r += stride
    return rows^


def _machine_header() raises:
    var isa = String("unknown")
    if CompilationTarget.has_avx512f():
        isa = String("x86-64 avx512f")
    elif CompilationTarget.has_avx2():
        isa = String("x86-64 avx2")
    elif CompilationTarget.has_neon():
        isa = String("arm64 neon")
    print("isa:", isa)
    print(
        "cores: physical",
        num_physical_cores(),
        "logical",
        num_logical_cores(),
        "performance",
        num_performance_cores(),
    )
    print(
        "simd: float64 width",
        simd_width_of[DType.float64](),
        "kernel lanes",
        SIMD_LANES,
    )
    print(
        "scheduler: tasks_per_core",
        TASKS_PER_CORE,
        "parallel_min_ops",
        env_parallel_min_ops(),
        "(default",
        PARALLEL_MIN_OPS,
        ")",
    )


def main() raises:
    var n_rows = 100_000
    var n_features = 100
    var reps = 3
    var pred_trees = 8
    var args = argv()
    if len(args) > 1:
        n_rows = Int(String(args[1]))
    if len(args) > 2:
        n_features = Int(String(args[2]))
    if len(args) > 3:
        reps = Int(String(args[3]))
    if len(args) > 4:
        pred_trees = Int(String(args[4]))
    if n_features < 4:
        raise Error("need at least 4 features")
    if reps < 1:
        raise Error("reps must be positive")
    if pred_trees < 1:
        raise Error("pred_trees must be positive")

    _machine_header()
    print(
        "shape:",
        n_rows,
        "rows x",
        n_features,
        "features, reps",
        reps,
        ", pred_trees",
        pred_trees,
    )
    print(
        "auto-mode task counts: features",
        plan_tasks(n_features, n_features * n_rows),
        "rows",
        plan_tasks(n_rows, n_rows),
    )
    print("")

    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(UInt64(k)))

    var noise_base = UInt64(n_rows * n_features)
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        var x3 = features[3 * n_rows + r]
        var signal = 5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)
        target.append(signal + 0.1 * (_uniform(noise_base + UInt64(r)) - 0.5))

    print("stage | serial_s | auto_s | speedup")

    # --- binning -----------------------------------------------------------
    var t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var m = fit_bins(features, n_rows, n_features, 255)
        _ = m.n_features
    var t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var m2 = fit_bins(features, n_rows, n_features, 255)
        _ = m2.n_features
    var t2 = perf_counter_ns()
    _report(
        "bin_fit",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    var mapper = fit_bins(features, n_rows, n_features, 255)

    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var d = mapper.transform(features, n_rows)
        _ = d.n_rows
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var d2 = mapper.transform(features, n_rows)
        _ = d2.n_rows
    t2 = perf_counter_ns()
    _report(
        "bin_transform",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    var data = mapper.transform(features, n_rows)

    # --- gradient generation ----------------------------------------------
    var raw = List[Float64](capacity=n_rows)
    raw.resize(n_rows, 0.5)
    var grad = List[Float64]()
    var hess = List[Float64]()
    var no_weights = List[Float64]()
    fill_grad_hess(raw, target, SQUARED_ERROR, no_weights, 0.9, grad, hess)

    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        fill_grad_hess(raw, target, SQUARED_ERROR, no_weights, 0.9, grad, hess)
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        fill_grad_hess(raw, target, SQUARED_ERROR, no_weights, 0.9, grad, hess)
    t2 = perf_counter_ns()
    _report(
        "grad_hess",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    # --- histograms --------------------------------------------------------
    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var h = build_histogram(data, grad, hess)
        _ = h.n_bins
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var h2 = build_histogram(data, grad, hess)
        _ = h2.n_bins
    t2 = perf_counter_ns()
    _report(
        "hist_full",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    var half = List[Int]()
    for r in range(0, n_rows, 2):
        half.append(r)
    var tenth = List[Int]()
    for r in range(0, n_rows, 10):
        tenth.append(r)

    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var h = build_histogram_subset(data, grad, hess, half)
        _ = h.n_bins
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var h2 = build_histogram_subset(data, grad, hess, half)
        _ = h2.n_bins
    t2 = perf_counter_ns()
    _report(
        "hist_subset50",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var h = build_histogram_subset(data, grad, hess, tenth)
        _ = h.n_bins
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var h2 = build_histogram_subset(data, grad, hess, tenth)
        _ = h2.n_bins
    t2 = perf_counter_ns()
    _report(
        "hist_subset10",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    var parent = build_histogram(data, grad, hess)
    var child = build_histogram_subset(data, grad, hess, half)

    # Subtraction is a single SIMD sweep, so one rep is far too short to
    # time; scale the rep count up rather than reporting quantization noise.
    var sub_reps = reps * 200
    t0 = perf_counter_ns()
    _serial()
    for _ in range(sub_reps):
        var s = subtract_histogram(parent, child)
        _ = s.n_bins
    t1 = perf_counter_ns()
    _auto()
    for _ in range(sub_reps):
        var s2 = subtract_histogram(parent, child)
        _ = s2.n_bins
    t2 = perf_counter_ns()
    _report(
        "hist_subtract",
        Float64(t1 - t0) / 1e9 / Float64(sub_reps),
        Float64(t2 - t1) / 1e9 / Float64(sub_reps),
    )

    # --- split scanning ----------------------------------------------------
    var scan_reps = reps * 50
    t0 = perf_counter_ns()
    _serial()
    for _ in range(scan_reps):
        var s = find_best_split(parent)
        _ = s.feature
    t1 = perf_counter_ns()
    _auto()
    for _ in range(scan_reps):
        var s2 = find_best_split(parent)
        _ = s2.feature
    t2 = perf_counter_ns()
    _report(
        "split_scan",
        Float64(t1 - t0) / 1e9 / Float64(scan_reps),
        Float64(t2 - t1) / 1e9 / Float64(scan_reps),
    )

    # --- row partitioning --------------------------------------------------
    var best = find_best_split(parent)
    if not best.found:
        raise Error("profile needs a splittable root histogram")
    var all_rows = List[Int](capacity=n_rows)
    for r in range(n_rows):
        all_rows.append(r)
    var missing = data.missing_bin[best.feature]

    var part_reps = reps * 10
    t0 = perf_counter_ns()
    _serial()
    for _ in range(part_reps):
        var p = partition_split_rows(data, all_rows, best, missing)
        _ = len(p.left)
    t1 = perf_counter_ns()
    _auto()
    for _ in range(part_reps):
        var p2 = partition_split_rows(data, all_rows, best, missing)
        _ = len(p2.left)
    t2 = perf_counter_ns()
    _report(
        "partition",
        Float64(t1 - t0) / 1e9 / Float64(part_reps),
        Float64(t2 - t1) / 1e9 / Float64(part_reps),
    )

    # --- whole tree and prediction ----------------------------------------
    var params = TreeParams.default()
    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var t = grow_tree(data, grad, hess, params)
        _ = t.n_leaves
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var t2b = grow_tree(data, grad, hess, params)
        _ = t2b.n_leaves
    t2 = perf_counter_ns()
    _report(
        "grow_tree",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    var tree = grow_tree(data, grad, hess, params)

    # The shipped batch scorer. `Booster.predict_batch_range` is what
    # `model.Model.predict` and every binding reach, and it fans out over row
    # blocks with `parallel.dispatch_rows` (boosting.mojo). Nothing in this
    # file measured it until this stage existed.
    #
    # The ensemble is `pred_trees` copies of the one grown tree. A copy of a
    # tree costs a row exactly what a distinct tree of the same shape costs
    # it -- the traversal reads the same node arrays and takes the same
    # depth -- so the ensemble is honest for timing even though its
    # predictions are not a fitted model's. Growing `pred_trees` real trees
    # instead would put the grower's time inside a prediction stage's setup.
    var pred_trees_list = List[Tree](capacity=pred_trees)
    for _ in range(pred_trees):
        pred_trees_list.append(tree.copy())
    # LightGBM's default shrinkage. The value scales every leaf output and
    # cannot change how long a traversal takes, so it is a placeholder here
    # rather than a parameter of the measurement.
    var booster = Booster(pred_trees_list^, 0.0, 0.1, SQUARED_ERROR)
    var pred_rng = IterationRange.slice(pred_trees, 0, pred_trees)
    var batch_ops = n_rows * (
        n_features + pred_trees * TRAVERSAL_ROW_OPS
    )

    # The block counts `dispatch_rows` will resolve to, read off outside the
    # timed windows in each arm's environment. Printed after the tables.
    _serial()
    var batch_serial_blocks = plan_row_blocks(n_rows, batch_ops).n_blocks
    _auto()
    var batch_auto_blocks = plan_row_blocks(n_rows, batch_ops).n_blocks

    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var p = booster.predict_batch_range(data, pred_rng)
        _ = len(p)
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var p2 = booster.predict_batch_range(data, pred_rng)
        _ = len(p2)
    t2 = perf_counter_ns()
    _report(
        "predict_batch",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    # This benchmark's own serial loop over one tree, which is what the stage
    # named `predict` was before this table distinguished the two. It is not
    # a path the library takes: it is serial in both columns by construction,
    # so its speedup is always about 1.0 and that is a property of this loop
    # and not of the library. Kept because the serial column is a clean
    # single-core traversal cost, which is what single-core work moves.
    var pred_reps = reps * 5
    t0 = perf_counter_ns()
    _serial()
    for _ in range(pred_reps):
        var acc = 0.0
        for r in range(n_rows):
            acc += tree.predict_row(data, r)
        _ = acc
    t1 = perf_counter_ns()
    _auto()
    for _ in range(pred_reps):
        var acc2 = 0.0
        for r in range(n_rows):
            acc2 += tree.predict_row(data, r)
        _ = acc2
    t2 = perf_counter_ns()
    _report(
        "predict_row_serial",
        Float64(t1 - t0) / 1e9 / Float64(pred_reps),
        Float64(t2 - t1) / 1e9 / Float64(pred_reps),
    )

    # --- the node-size ladder ---------------------------------------------
    #
    # See the module docstring. Same kernel the grower calls, same recycled
    # buffers, at the sizes a 31-leaf tree's nodes actually hold, with the
    # resolved task count of each arm printed beside its time so a 1.0x can
    # be told apart from a dispatch that never fanned out.
    print("")
    _auto()
    print(
        "split_scan resolved tasks (auto, per node, independent of node"
        " rows):",
        plan_tasks(
            n_features,
            split_scan_ops(n_features, data.n_bins, True),
        ),
    )
    print(
        "ladder: node histogram via build_histogram_subset_into_scratch,"
        " pooled output + reused pairs"
    )
    print(
        "node_rows | serial_s | auto_s | speedup | serial_tasks |"
        " auto_tasks | gather_blocks | group_width | group_count |"
        " active_ops | compact"
    )

    # The sizes the ladder walks. Chosen to bracket the scheduler's grain
    # floor from both sides at the two shapes this round cares about: with
    # 256 bins the accumulation's work estimate `n_active * (n_bins +
    # n_rows)` clears `PARALLEL_MIN_OPS` (65,536) at about 1,055 rows on 50
    # features and at about 400 rows on 100. So 300 and 1,000 are expected
    # below the floor at 50 features and 3,000 above it, and the
    # `serial_tasks`/`auto_tasks` columns are what say whether they were.
    var ladder_sizes: List[Int] = [
        100_000, 30_000, 10_000, 3_000, 1_000, 300
    ]

    var rung_hist = Histogram.zeroed(data.n_features, data.n_bins)
    var rung_pairs = List[Float64]()
    var no_features = List[Int]()
    for si in range(len(ladder_sizes)):
        var size = ladder_sizes[si]
        if size > n_rows:
            continue
        var rows = _strided_rows(n_rows, size)
        var n_sub = len(rows)

        # The plan the kernel will derive for itself. It reads
        # MOJOTREES_CPU_* and the machine, never MOJOTREES_NUM_WORKERS, so it
        # is the same object in both arms; only the task count off it moves.
        var plan = derive_accumulation_plan(
            cpu_profile(), n_features, n_features, data.n_bins, n_sub, True
        )

        _serial()
        var serial_tasks = _accum_tasks(plan.group_count, plan.active_ops)
        _auto()
        var auto_tasks = _accum_tasks(plan.group_count, plan.active_ops)
        var gather_blocks = 0
        if plan.compact_rows:
            gather_blocks = plan_row_blocks(n_sub, plan.gather_ops).n_blocks

        var rung_reps = reps * (LADDER_REP_BUDGET // size)
        if rung_reps < reps:
            rung_reps = reps

        # Warm both recycled buffers at this rung's size before either arm is
        # timed, so the first rep does not pay a resize the rest do not.
        build_histogram_subset_into_scratch(
            rung_hist, rung_pairs, data, grad, hess, rows, 0, n_sub,
            no_features, False,
        )

        t0 = perf_counter_ns()
        _serial()
        for _ in range(rung_reps):
            build_histogram_subset_into_scratch(
                rung_hist, rung_pairs, data, grad, hess, rows, 0, n_sub,
                no_features, False,
            )
        t1 = perf_counter_ns()
        _auto()
        for _ in range(rung_reps):
            build_histogram_subset_into_scratch(
                rung_hist, rung_pairs, data, grad, hess, rows, 0, n_sub,
                no_features, False,
            )
        t2 = perf_counter_ns()
        _report_ladder(
            n_sub,
            Float64(t1 - t0) / 1e9 / Float64(rung_reps),
            Float64(t2 - t1) / 1e9 / Float64(rung_reps),
            serial_tasks,
            auto_tasks,
            gather_blocks,
            plan.group_width,
            plan.group_count,
            plan.active_ops,
            plan.compact_rows,
        )

    print("")
    print(
        "predict_batch resolved row blocks: serial",
        batch_serial_blocks,
        "auto",
        batch_auto_blocks,
    )

    _ = setenv("MOJOTREES_NUM_WORKERS", "")
