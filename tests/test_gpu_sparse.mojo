"""Sparse GPU training against the CPU sparse trainer.

Every test needs an accelerator and prints `skipped` without one. Two kinds
of check:

- the device path's own consistency, which holds regardless of rounding:
  the leaf the device partition assigned every row is the leaf a host walk
  of the same tree reaches (`row_leaf` against `predict_row_sparse`), for
  numerical, missing, and categorical splits alike, and two runs are
  bit-identical;
- agreement with `train_sparse` to the tolerance the dense GPU trainer is
  held to against `train` in tests/test_gpu_training.mojo (Float32
  histograms on the device, Float64 on the host), which is a statement
  about fit quality and predictions, not tree structure: a split whose gain
  ties another to the last bits may fall the other way on the device.
"""

from std.math import log
from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.bagging import BaggingParams
from mojotrees.binning import fit_bins
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    IterationRange,
)
from mojotrees.boosting_sparse import (
    train_multiclass_sparse,
    train_sparse,
    train_sparse_with_valid,
)
from mojotrees.device import CPU_DEVICE, GPU_DEVICE
from mojotrees.efb import EfbSettings
from mojotrees.gpu_categorical import CatSetPool
from mojotrees.gpu_sparse import GpuSparseHistogramBuilder
from mojotrees.gpu_sparse_layout import sparse_gpu_training_is_wired
from mojotrees.histogram_sparse import build_histogram_sparse
from mojotrees.model_sparse import fit_csc, fit_multiclass_csc, predict_csr
from mojotrees.sparse import (
    SparseBinnedRows,
    csc_from_dense,
    fit_bins_csc,
    transform_csc,
)
from mojotrees.train_gpu_sparse import (
    grow_tree_gpu_sparse,
    train_gpu_sparse,
    train_gpu_sparse_with_valid,
    train_multiclass_gpu_sparse,
)
from mojotrees.tree import Tree, TreeParams
from mojotrees.tree_sparse import (
    SparseBundling,
    SparseTreeResult,
    grow_tree_sparse,
    predict_row_sparse,
)


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _sparse_dense(
    n_rows: Int, n_features: Int, density: Float64, seed: UInt64
) -> List[Float64]:
    var out = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        var u = _uniform(seed + UInt64(k))
        if u < density:
            out.append(4.0 * (u / density) - 2.0)
        else:
            out.append(0.0)
    return out^


def _target(dense: List[Float64], n_rows: Int, seed: UInt64) -> List[Float64]:
    var n_features = len(dense) // n_rows
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var total = 0.0
        for f in range(n_features):
            total += (1.0 + 0.37 * Float64(f)) * dense[f * n_rows + r]
        out.append(total / 4.0 + 0.05 * (_uniform(seed + UInt64(r)) - 0.5))
    return out^


def _grads(n_rows: Int, seed: UInt64) -> List[Float64]:
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        g.append(2.0 * _uniform(seed + UInt64(r)) - 1.0)
    return g^


def _hessians(n_rows: Int, seed: UInt64) -> List[Float64]:
    var h = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        h.append(_uniform(seed + UInt64(r)) + 0.01)
    return h^


def _nan() -> Float64:
    var zero = 0.0
    return zero / zero


def _assert_leaves_match_walk(
    grown: SparseTreeResult, data: SparseBinnedRows, n_rows: Int
) raises:
    """Every row the tree was grown on landed, on the device, in the leaf a
    host walk of the tree reaches; that is the device partition, numerical
    and categorical routing included, checked against the tree it built."""
    for r in range(n_rows):
        if grown.row_leaf[r] < 0:
            continue
        assert_equal(
            grown.tree.value[grown.row_leaf[r]],
            predict_row_sparse(grown.tree, data, r),
        )


def _worst_gap(
    a: Booster, b: Booster, rows: SparseBinnedRows, n_rows: Int
) raises -> Float64:
    var worst = 0.0
    for r in range(n_rows):
        var d = abs(
            a.predict_bins(rows.bins_of_row(r)) - b.predict_bins(rows.bins_of_row(r))
        )
        if d > worst:
            worst = d
    return worst


def test_wired_flag_and_capability_record() raises:
    assert_true(sparse_gpu_training_is_wired())
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var dense = _sparse_dense(200, 4, 0.2, UInt64(1))
        var csc = csc_from_dense(dense, 200, 4)
        var sparse = transform_csc(fit_bins_csc(csc, 16), csc)
        var builder = GpuSparseHistogramBuilder(sparse, 8)
        assert_true(builder.capability.histograms)
        assert_true(builder.capability.training)


def test_gpu_sparse_histogram_matches_cpu_sparse_histogram() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 1000
        var n_features = 12
        var dense = _sparse_dense(n_rows, n_features, 0.12, UInt64(3))
        var csc = csc_from_dense(dense, n_rows, n_features)
        var sparse = transform_csc(fit_bins_csc(csc, 32), csc)
        var grad = _grads(n_rows, UInt64(100_000))
        var hess = _hessians(n_rows, UInt64(200_000))
        var want = build_histogram_sparse(sparse, grad, hess)
        var builder = GpuSparseHistogramBuilder(sparse, 4)
        var got = builder.build(grad, hess)
        assert_equal(got.n_features, want.n_features)
        assert_equal(got.n_bins, want.n_bins)
        for i in range(len(want.count)):
            assert_equal(got.count[i], want.count[i])
            assert_true(abs(got.grad[i] - want.grad[i]) <= 1e-3)
            assert_true(abs(got.hess[i] - want.hess[i]) <= 1e-3)


def test_gpu_sparse_tree_matches_cpu_sparse_tree() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 800
        var n_features = 12
        var dense = _sparse_dense(n_rows, n_features, 0.12, UInt64(8))
        var csc = csc_from_dense(dense, n_rows, n_features)
        var sparse = transform_csc(fit_bins_csc(csc, 32), csc)
        var rows = sparse.to_rows()
        var grad = _grads(n_rows, UInt64(5_000_000))
        var hess = _hessians(n_rows, UInt64(6_000_000))
        var params = TreeParams(15, 20, 1.0, 1e-3)

        var want = grow_tree_sparse(sparse, grad, hess, params)
        var builder = GpuSparseHistogramBuilder(sparse, 2 * params.num_leaves)
        var pool = CatSetPool(builder.ctx, params.num_leaves)
        var got = grow_tree_gpu_sparse(
            builder, pool, sparse, grad, hess, params
        )
        assert_equal(got.tree.n_leaves, want.tree.n_leaves)
        _assert_leaves_match_walk(got, rows, n_rows)
        for r in range(n_rows):
            assert_true(got.row_leaf[r] >= 0)
            assert_true(
                abs(
                    want.tree.value[want.row_leaf[r]]
                    - got.tree.value[got.row_leaf[r]]
                )
                <= 1e-3
            )


def test_gpu_sparse_bagged_tree_leaves_the_bag_out() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 600
        var n_features = 8
        var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(21))
        var csc = csc_from_dense(dense, n_rows, n_features)
        var sparse = transform_csc(fit_bins_csc(csc, 32), csc)
        var rows = sparse.to_rows()
        var grad = _grads(n_rows, UInt64(7_000_000))
        var hess = _hessians(n_rows, UInt64(8_000_000))
        var params = TreeParams(7, 10, 1.0, 1e-3)
        var bag = List[Int]()
        for r in range(0, n_rows, 3):
            bag.append(r)

        var want = grow_tree_sparse(sparse, grad, hess, params, bag)
        var builder = GpuSparseHistogramBuilder(sparse, 2 * params.num_leaves)
        var pool = CatSetPool(builder.ctx, params.num_leaves)
        var got = grow_tree_gpu_sparse(
            builder, pool, sparse, grad, hess, params, bag
        )
        _assert_leaves_match_walk(got, rows, n_rows)
        var inside = 0
        for r in range(n_rows):
            if got.row_leaf[r] >= 0:
                inside += 1
                assert_equal(r % 3, 0)
                assert_true(
                    abs(
                        want.tree.value[want.row_leaf[r]]
                        - got.tree.value[got.row_leaf[r]]
                    )
                    <= 1e-3
                )
            else:
                assert_equal(want.row_leaf[r], -1)
        assert_equal(inside, len(bag))


def test_gpu_sparse_categorical_and_missing_splits_route_as_the_tree_says() raises:
    """Categorical columns and a column with NaN entries: the pooled
    category-set routing and the missing-bin default direction on the
    device agree with the host walk of the grown tree, and the fit is the
    CPU sparse trainer's to tolerance."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 500
        var n_features = 5
        var dense = List[Float64](capacity=n_rows * n_features)
        dense.resize(n_rows * n_features, 0.0)
        for r in range(n_rows):
            # Two categorical columns with a strong per-code signal, one
            # numerical column with NaN in a tenth of its rows, two plain
            # sparse numerical columns.
            dense[0 * n_rows + r] = Float64((r * 7) % 6)
            dense[1 * n_rows + r] = Float64((r * 3) % 4)
            var u = _uniform(UInt64(40_000 + r))
            if r % 10 == 3:
                dense[2 * n_rows + r] = _nan()
            elif u < 0.4:
                dense[2 * n_rows + r] = 3.0 * u - 1.0
            var v = _uniform(UInt64(50_000 + r))
            if v < 0.25:
                dense[3 * n_rows + r] = v * 8.0 - 1.0
            var w = _uniform(UInt64(60_000 + r))
            if w < 0.25:
                dense[4 * n_rows + r] = 1.0 - w * 8.0
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var t = 0.0
            var c0 = Int(dense[0 * n_rows + r])
            var c1 = Int(dense[1 * n_rows + r])
            t += Float64((c0 * 5) % 6) * 0.5 - Float64((c1 * 3) % 4) * 0.7
            var x2 = dense[2 * n_rows + r]
            if x2 == x2:
                t += 0.8 * x2
            else:
                t += 1.1
            t += 0.3 * dense[3 * n_rows + r] - 0.6 * dense[4 * n_rows + r]
            target.append(t + 0.02 * (_uniform(UInt64(70_000 + r)) - 0.5))
        var csc = csc_from_dense(dense, n_rows, n_features)
        var cat_features: List[Int] = [0, 1]
        var mapper = fit_bins_csc(csc, 16, cat_features)
        assert_true(mapper.cats.any_categorical())
        var sparse = transform_csc(mapper, csc)
        var rows = sparse.to_rows()

        var params = BoosterParams(8, 0.2, TreeParams(8, 10, 1.0, 1e-3))
        var cpu = train_sparse(sparse, target, SQUARED_ERROR, params)
        var gpu = train_gpu_sparse(sparse, target, SQUARED_ERROR, params)
        assert_equal(len(gpu.trees), len(cpu.trees))
        var used_categorical = False
        for t in range(len(gpu.trees)):
            for i in range(len(gpu.trees[t].feature)):
                if (
                    gpu.trees[t].feature[i] >= 0
                    and gpu.trees[t].cat_offset[i] >= 0
                ):
                    used_categorical = True
        assert_true(used_categorical)

        var grad = _grads(n_rows, UInt64(9_000_000))
        var hess = _hessians(n_rows, UInt64(9_500_000))
        var builder = GpuSparseHistogramBuilder(
            sparse, 2 * params.tree.num_leaves
        )
        var pool = CatSetPool(builder.ctx, params.tree.num_leaves)
        var grown = grow_tree_gpu_sparse(
            builder, pool, sparse, grad, hess, params.tree
        )
        _assert_leaves_match_walk(grown, rows, n_rows)

        var cpu_sse = 0.0
        var gpu_sse = 0.0
        for r in range(n_rows):
            var pc = cpu.predict_bins(rows.bins_of_row(r))
            var pg = gpu.predict_bins(rows.bins_of_row(r))
            cpu_sse += (pc - target[r]) * (pc - target[r])
            gpu_sse += (pg - target[r]) * (pg - target[r])
        assert_true(abs(cpu_sse - gpu_sse) <= 2e-2 * (cpu_sse + 1e-12))


def test_gpu_sparse_training_matches_cpu_sparse() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 600
        var n_features = 8
        var dense = _sparse_dense(n_rows, n_features, 0.15, UInt64(11))
        var csc = csc_from_dense(dense, n_rows, n_features)
        var sparse = transform_csc(fit_bins_csc(csc, 32), csc)
        var rows = sparse.to_rows()
        var target = _target(dense, n_rows, UInt64(600_000))
        var params = BoosterParams(12, 0.1, TreeParams(15, 20, 1.0, 1e-3))

        var cpu = train_sparse(sparse, target, SQUARED_ERROR, params)
        var gpu = train_gpu_sparse(sparse, target, SQUARED_ERROR, params)
        assert_equal(len(gpu.trees), len(cpu.trees))
        assert_true(_worst_gap(cpu, gpu, rows, n_rows) <= 1e-3)
        var cpu_sse = 0.0
        var gpu_sse = 0.0
        for r in range(n_rows):
            var pc = cpu.predict_bins(rows.bins_of_row(r))
            var pg = gpu.predict_bins(rows.bins_of_row(r))
            cpu_sse += (pc - target[r]) * (pc - target[r])
            gpu_sse += (pg - target[r]) * (pg - target[r])
        assert_true(abs(cpu_sse - gpu_sse) <= 1e-3 * (cpu_sse + 1e-12))



def test_gpu_sparse_binary_training_matches_cpu_sparse() raises:
    """Binary logistic on a larger matrix than the regression check above.
    On 600 rows this dataset holds a split whose gain ties another feature's
    to seven digits, and the device's Float32 histogram breaks that tie the
    other way, after which the two trees legitimately diverge; at this size
    no such tie occurs and the fits agree pointwise."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 1500
        var n_features = 12
        var dense = _sparse_dense(n_rows, n_features, 0.15, UInt64(11))
        var csc = csc_from_dense(dense, n_rows, n_features)
        var sparse = transform_csc(fit_bins_csc(csc, 32), csc)
        var rows = sparse.to_rows()
        var target = _target(dense, n_rows, UInt64(600_000))
        var params = BoosterParams(12, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var labels = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            labels.append(1.0 if target[r] > 0.0 else 0.0)
        var cpu_bin = train_sparse(sparse, labels, BINARY_LOGISTIC, params)
        var gpu_bin = train_gpu_sparse(sparse, labels, BINARY_LOGISTIC, params)
        assert_equal(len(gpu_bin.trees), len(cpu_bin.trees))
        assert_true(_worst_gap(cpu_bin, gpu_bin, rows, n_rows) <= 1e-3)
        var cpu_loss = 0.0
        var gpu_loss = 0.0
        for r in range(n_rows):
            var pc = cpu_bin.predict_bins(rows.bins_of_row(r))
            var pg = gpu_bin.predict_bins(rows.bins_of_row(r))
            cpu_loss -= labels[r] * log(pc) + (1.0 - labels[r]) * log(1.0 - pc)
            gpu_loss -= labels[r] * log(pg) + (1.0 - labels[r]) * log(1.0 - pg)
        assert_true(abs(cpu_loss - gpu_loss) <= 1e-3 * (cpu_loss + 1e-12))


def test_gpu_sparse_training_is_deterministic() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 500
        var n_features = 8
        var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(9))
        var csc = csc_from_dense(dense, n_rows, n_features)
        var sparse = transform_csc(fit_bins_csc(csc, 32), csc)
        var rows = sparse.to_rows()
        var target = _target(dense, n_rows, UInt64(700_000))
        var params = BoosterParams(5, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var a = train_gpu_sparse(sparse, target, SQUARED_ERROR, params)
        var b = train_gpu_sparse(sparse, target, SQUARED_ERROR, params)
        assert_equal(len(a.trees), len(b.trees))
        assert_equal(_worst_gap(a, b, rows, n_rows), 0.0)


def test_gpu_sparse_bagged_training_matches_cpu_sparse() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 600
        var n_features = 8
        var dense = _sparse_dense(n_rows, n_features, 0.15, UInt64(13))
        var csc = csc_from_dense(dense, n_rows, n_features)
        var sparse = transform_csc(fit_bins_csc(csc, 32), csc)
        var rows = sparse.to_rows()
        var target = _target(dense, n_rows, UInt64(800_000))
        var params = BoosterParams(8, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var bagging = BaggingParams(0.6, 1, 7)
        var cpu = train_sparse(
            sparse, target, SQUARED_ERROR, params, bagging=bagging
        )
        var gpu = train_gpu_sparse(
            sparse, target, SQUARED_ERROR, params, bagging=bagging
        )
        assert_equal(len(gpu.trees), len(cpu.trees))
        assert_true(_worst_gap(cpu, gpu, rows, n_rows) <= 1e-3)


def test_gpu_sparse_early_stopping_matches_cpu_sparse() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 500
        var n_features = 8
        var dense = _sparse_dense(n_rows, n_features, 0.15, UInt64(17))
        var csc = csc_from_dense(dense, n_rows, n_features)
        var mapper = fit_bins_csc(csc, 32)
        var sparse = transform_csc(mapper, csc)
        var target = _target(dense, n_rows, UInt64(900_000))
        var n_valid = 150
        var vdense = _sparse_dense(n_valid, n_features, 0.15, UInt64(18))
        var vcsc = csc_from_dense(vdense, n_valid, n_features)
        var vsparse = transform_csc(mapper, vcsc)
        var vrows = vsparse.to_rows()
        var vtarget = _target(vdense, n_valid, UInt64(950_000))
        var params = BoosterParams(60, 0.3, TreeParams(15, 20, 1.0, 1e-3))
        var cpu = train_sparse_with_valid(
            sparse, target, vsparse, vtarget, SQUARED_ERROR, params, 5
        )
        var gpu = train_gpu_sparse_with_valid(
            sparse, target, vsparse, vtarget, SQUARED_ERROR, params, 5
        )
        # Both stopped, and where they stopped is decided by the same rule
        # over losses that agree to Float32 level.
        assert_true(len(cpu.trees) < params.n_estimators)
        assert_true(len(gpu.trees) < params.n_estimators)
        assert_true(abs(len(cpu.trees) - len(gpu.trees)) <= 2)
        var shared = min(len(cpu.trees), len(gpu.trees))
        var head = IterationRange(0, shared)
        for r in range(n_valid):
            var bins = vrows.bins_of_row(r)
            assert_true(
                abs(
                    cpu.predict_bins_range(bins, head)
                    - gpu.predict_bins_range(bins, head)
                )
                <= 1e-3
            )


def test_gpu_sparse_multiclass_matches_cpu_sparse() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 500
        var n_features = 8
        var n_classes = 3
        var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(23))
        var csc = csc_from_dense(dense, n_rows, n_features)
        var sparse = transform_csc(fit_bins_csc(csc, 32), csc)
        var rows = sparse.to_rows()
        var target = _target(dense, n_rows, UInt64(1_000_000))
        var labels = List[Int](capacity=n_rows)
        for r in range(n_rows):
            if target[r] < -0.2:
                labels.append(0)
            elif target[r] < 0.2:
                labels.append(1)
            else:
                labels.append(2)
        var params = BoosterParams(5, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var cpu = train_multiclass_sparse(sparse, labels, n_classes, params)
        var gpu = train_multiclass_gpu_sparse(
            sparse, labels, n_classes, params
        )
        assert_equal(len(gpu.trees), len(cpu.trees))
        # The rule tests/test_gpu_objectives.mojo holds the dense multiclass
        # trainer to: mean absolute probability gap, and the same accuracy
        # to within a percent of the rows.
        var total_diff = 0.0
        var cpu_correct = 0
        var gpu_correct = 0
        for r in range(n_rows):
            var pc = cpu.predict_proba_bins(rows.bins_of_row(r))
            var pg = gpu.predict_proba_bins(rows.bins_of_row(r))
            var cpu_arg = 0
            var gpu_arg = 0
            for k in range(n_classes):
                total_diff += abs(pc[k] - pg[k])
                if pc[k] > pc[cpu_arg]:
                    cpu_arg = k
                if pg[k] > pg[gpu_arg]:
                    gpu_arg = k
            if cpu_arg == labels[r]:
                cpu_correct += 1
            if gpu_arg == labels[r]:
                gpu_correct += 1
        assert_true(total_diff / Float64(n_rows * n_classes) <= 1e-3)
        assert_true(abs(cpu_correct - gpu_correct) <= n_rows // 100)


def test_gpu_sparse_refuses_bundling() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var dense = _sparse_dense(300, 6, 0.2, UInt64(29))
        var csc = csc_from_dense(dense, 300, 6)
        var sparse = transform_csc(fit_bins_csc(csc, 16), csc)
        var target = _target(dense, 300, UInt64(1_100_000))
        var params = BoosterParams(3, 0.1, TreeParams(7, 10, 1.0, 1e-3))
        params.bundling = EfbSettings(True)
        var raised = False
        try:
            _ = train_gpu_sparse(sparse, target, SQUARED_ERROR, params)
        except:
            raised = True
        assert_true(raised)


def test_fit_csc_dispatches_to_the_device() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 400
        var n_features = 8
        var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(31))
        var csc = csc_from_dense(dense, n_rows, n_features)
        var csr = csc.to_csr()
        var target = _target(dense, n_rows, UInt64(1_200_000))
        var params = BoosterParams(5, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var cpu = fit_csc(csc, target, SQUARED_ERROR, params, 32)
        var gpu = fit_csc(
            csc, target, SQUARED_ERROR, params, 32, device=GPU_DEVICE
        )
        var pc = predict_csr(cpu, csr)
        var pg = predict_csr(gpu, csr)
        assert_equal(len(pg), n_rows)
        for r in range(n_rows):
            assert_true(abs(pc[r] - pg[r]) <= 1e-3)
        var labels = List[Int](capacity=n_rows)
        for r in range(n_rows):
            labels.append(0 if target[r] < 0.0 else 1)
        var mc = fit_multiclass_csc(
            csc, labels, 2, params, 32, device=GPU_DEVICE
        )
        assert_equal(mc.booster.n_classes, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
