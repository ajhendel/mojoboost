"""Whether the three device spellings of the score update round the same.

WHY THIS FILE EXISTS
--------------------
`docs/NUMERICS.md` section 9 ranks `gpu_objectives_native.mojo`'s
`_update_raw_kernel` as the package's most exposed contraction site and says
of it that it "may already be inconsistent rather than merely fragile". That
is a claim about what a particular device compiler chose to do, so it is
answerable only by running the kernels, which is what this file does.

The same arithmetic, `raw[i] + learning_rate * value[node]`, is written three
ways in that module and reached from three different arms of a GPU round:

  `_update_raw_kernel`        `node` comes per-thread from the leaf-assignment
                              array. Reached from the bagging and all-rows
                              arms (`train_gpu.mojo`, `gpu_fused_round.mojo`).
  `_range_add_raw_kernel`     `node` is a launch argument, so the product is
                              uniform across the launch. Reached only from
                              `update_raw_ranges_per_leaf`, the reference arm.
  `_range_table_add_raw_kernel`  no multiply at all: the host computes
                              `Float32(lr) * Float32(value)` and ships the
                              step. Reached from the device-resident path.

A product that is uniform across a launch is hoisted and rounded on its own;
a product that varies per thread is free to contract into the add. Both are
legitimate Float32 evaluations and they differ by one unit in the last place,
which is enough to make one model not byte-identical to another.

WHAT IS ASSERTED
----------------
Two things, and the second is the one that matters.

First, that the three arms agree bit for bit over the same tree, the same
leaf ranges, the same node values, and the same starting scores. Float32
addition is exact given identical operands, and every row belongs to exactly
one leaf, so this is equality and not a tolerance.

Second, that they agree on the *unfused* answer specifically, by naming both
candidates on the host and asserting which one the device produced. Arm
agreement alone would still pass if some future compiler contracted all three
at once; naming the rounding pins the convention rather than the coincidence.
The unfused host reference is built by storing the products into a `List`
first and adding loaded values afterwards, so the reference itself contains
no multiply next to an add and cannot contract (NUMERICS section 3.6: fused
is expressible in the source, unfused is not, so the only sound way to write
an unfused reference is to keep the multiply out of the neighborhood). The
fused candidate is `math.fma`, which is a fused operation rather than a
contraction and is therefore stable at any optimization level.

The prediction kernel gets the same treatment. `_predict_kernel` accumulates
`learning_rate * values[node]` per tree with `node` walked per thread, which
is the same shape, and it is on the batch prediction path and on the resident
validation path. Two trees are enough to separate the two roundings, and the
ensemble is built by hand rather than fitted, so nothing here trains.

ONE THING HERE IS NOT ABOUT ROUNDING
------------------------------------
`test_split_searcher_allocates_its_histogram_lazily` is the lane's other
change and has nothing to do with contraction. `GpuSplitSearcher` owns a
`3 * n_features * n_bins` Int32 histogram that neither device search path
reads, and it is now allocated on first use. The assertion lives here
because the lane owns one test file and an allocation that a fit never
touches has no other home; it is separated by name and by this paragraph
rather than by a file.

Skips (passing) with no accelerator, so the suite stays green on CPU-only
machines. Nothing in this file measures anything.
"""

from std.math import fma
from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.binning import bin_equal_width
from mojotrees.boosting import IterationRange
from mojotrees.gpu_predict import GpuPredictor, flatten_trees
from mojotrees.categorical import CategoricalParams
from mojotrees.gpu_split_search import GpuSplitParams, GpuSplitSearcher
from mojotrees.histogram_gpu import GpuHistogramBuilder
from mojotrees.tree import Tree

from support import _make_features, _uniform


def _bits32(x: Float32) -> UInt32:
    """A Float32's IEEE 754 bit pattern as a plain `UInt32`."""
    return x.to_bits().cast[DType.uint32]()


def _f32_bits(x: Float64) -> UInt32:
    """The Float32 bit pattern of a score that was accumulated in Float32 and
    widened on the way back. The narrowing is exact, so this recovers the
    bits the device wrote."""
    return _bits32(Float32(x))


def _ulps(a: Float64, b: Float64) -> Int:
    """Signed distance in units in the last place between two same-sign
    finite Float32 values, as the difference of their bit patterns. Only
    meaningful when the two do not straddle zero, which is asserted by the
    callers through the scores they choose."""
    return Int(_f32_bits(a)) - Int(_f32_bits(b))


def _full_mantissa(seed: UInt64) -> Float64:
    """A value in (-1, 1) whose Float32 form uses its whole significand.

    A round number like 0.5 or 0.125 multiplies exactly, and an exact product
    rounds the same whether it is fused or not, so a test built out of round
    numbers cannot tell the two apart. Every value and every learning rate
    below comes from here for that reason."""
    return 2.0 * _uniform(seed) - 1.0


def _split_tree_ranges(mut builder: GpuHistogramBuilder) raises:
    """Four splits on the builder's row set, leaving five live leaves and
    four emptied internal nodes among nodes 0 through 8.

    The emptied ones are what the range arms skip and what the per-row leaf
    id array below must never name, so the three arms are looking at exactly
    the same partition of the rows."""
    builder.begin_tree()
    builder.apply_split(0, 15, 0, 1, 2)
    builder.apply_split(1, 15, 1, 3, 4)
    builder.apply_split(2, 15, 2, 5, 6)
    builder.apply_split(3, 15, 3, 7, 8)


def _leaf_ids_from_ranges(
    mut builder: GpuHistogramBuilder, n_rows: Int, n_nodes: Int
) raises -> List[Int]:
    """The per-row leaf assignment that the leaf ranges describe.

    `_update_raw_kernel` reads a row's node out of a per-row array while the
    range kernels read it out of a table of windows into the active-row
    permutation. Deriving the first from the second is what makes the three
    arms comparable: every row gets the node its range already gave it, so
    the arms differ in nothing but how the node reached the kernel."""
    var perm = List[Int](capacity=n_rows)
    builder.synchronize()
    with builder.rows.rows_dev.map_to_host() as host:
        var src = host.unsafe_ptr()
        for i in range(n_rows):
            perm.append(Int(src.unsafe_load(i)))

    var leaf_of = List[Int](capacity=n_rows)
    for _ in range(n_rows):
        leaf_of.append(-1)
    var covered = 0
    for node in range(n_nodes):
        var window = builder.rows.ranges.get(node)
        for slot in range(window.begin, window.end):
            leaf_of[perm[slot]] = node
            covered += 1
    if covered != n_rows:
        raise Error("leaf ranges do not tile the rows")
    return leaf_of^


def test_score_update_arms_round_the_same_way() raises:
    """The three device spellings of `raw + lr * value`, over one partition.

    This is the fact the NUMERICS audit asked for. If the arms disagree the
    failure prints the row, both bit patterns, and the distance in units in
    the last place, because that distance is the whole finding.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 4
        var n_nodes = 9
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            target.append(_uniform(UInt64(11_000 + r)))

        var builder = GpuHistogramBuilder(data)
        _split_tree_ranges(builder)
        var leaf_of = _leaf_ids_from_ranges(builder, n_rows, n_nodes)

        var values = List[Float64](capacity=n_nodes)
        for node in range(n_nodes):
            values.append(_full_mantissa(UInt64(900 + node)))
        var learning_rate = _uniform(UInt64(4_242)) + 0.5
        var base = _full_mantissa(UInt64(77))

        var leaf_dev = builder.ctx.enqueue_create_buffer[DType.int32](n_rows)
        with leaf_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for r in range(n_rows):
                dst.unsafe_store(r, Int32(leaf_of[r]))

        var per_thread = builder.objective_state(target)
        per_thread.init_raw(builder.ctx, [base])
        var per_leaf = builder.objective_state(target)
        per_leaf.init_raw(builder.ctx, [base])
        var table = builder.objective_state(target)
        table.init_raw(builder.ctx, [base])

        per_thread.update_raw(builder.ctx, leaf_dev, values, learning_rate)
        per_leaf.update_raw_ranges_per_leaf(
            builder.ctx, builder.rows, values, learning_rate
        )
        table.update_raw_ranges(
            builder.ctx, builder.rows, values, learning_rate
        )

        var got_thread = per_thread.download_raw(builder.ctx)
        var got_leaf = per_leaf.download_raw(builder.ctx)
        var got_table = table.download_raw(builder.ctx)

        # The two host candidates. `steps` is a separate loop into a `List`,
        # so the row loop below adds two loaded values and has no multiply
        # in it to contract; `fma` is the fused candidate by construction.
        var lr32 = Float32(learning_rate)
        var base32 = Float32(base)
        var steps = List[Float32](capacity=n_nodes)
        for node in range(n_nodes):
            steps.append(lr32 * Float32(values[node]))

        var disagreed = 0
        var fused_rows = 0
        var unfused_rows = 0
        var worst = 0
        var first_report = True
        for r in range(n_rows):
            var node = leaf_of[r]
            var unfused = base32 + steps[node]
            var fused = fma(lr32, Float32(values[node]), base32)
            var seen = _f32_bits(got_thread[r])
            if seen == _bits32(unfused):
                unfused_rows += 1
            elif seen == _bits32(fused):
                fused_rows += 1
            if (
                seen != _f32_bits(got_leaf[r])
                or seen != _f32_bits(got_table[r])
            ):
                disagreed += 1
                var d = _ulps(got_thread[r], got_table[r])
                if d < 0:
                    d = -d
                if d > worst:
                    worst = d
                if first_report:
                    first_report = False
                    print(
                        "raw update arms disagree at row ", r,
                        " node ", node,
                        ": per-thread ", hex(seen),
                        " per-leaf ", hex(_f32_bits(got_leaf[r])),
                        " table ", hex(_f32_bits(got_table[r])),
                        " ulps(per-thread, table) ",
                        _ulps(got_thread[r], got_table[r]),
                        sep="",
                    )
        if disagreed != 0:
            print(
                "rows disagreeing: ", disagreed, " of ", n_rows,
                ", worst ", worst, " ulp",
                "; per-thread matched the unfused host answer on ",
                unfused_rows, " rows and the fused one on ", fused_rows,
                sep="",
            )
        assert_equal(disagreed, 0)

        # Every arm on the unfused answer, not merely on each other's.
        var wrong_rounding = 0
        for r in range(n_rows):
            var node = leaf_of[r]
            var unfused = base32 + steps[node]
            if _f32_bits(got_table[r]) != _bits32(unfused):
                wrong_rounding += 1
        assert_equal(wrong_rounding, 0)

        # Without this the test would pass on three arms that all did
        # nothing: no node value here is zero, so every row must have moved
        # off the base score, and the two roundings must be separable on at
        # least some rows or the fixture proves nothing.
        var moved = 0
        var separable = 0
        for r in range(n_rows):
            if _f32_bits(got_thread[r]) != _bits32(base32):
                moved += 1
            var node = leaf_of[r]
            var unfused = base32 + steps[node]
            var fused = fma(lr32, Float32(values[node]), base32)
            if _bits32(unfused) != _bits32(fused):
                separable += 1
        assert_equal(moved, n_rows)
        assert_true(separable > 0)


def _stump(
    feature: Int, threshold_bin: Int, left: Float64, right: Float64
) -> Tree:
    """A one-split tree over `feature`: bins at or below `threshold_bin` take
    `left`, the rest take `right`. Built rather than fitted, because nothing
    in this file trains."""
    return Tree(
        [feature, -1, -1],
        [threshold_bin, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [0.0, left, right],
        [0.0, 0.0, 0.0],
        2,
    )


def test_predict_kernel_rounds_the_same_way() raises:
    """`_predict_kernel`'s per-tree accumulation against the same two
    candidates.

    Two stumps, so the accumulator takes two steps and the roundings
    compound; one tree would exercise a single add and would understate what
    a real ensemble does. The host reference again stores the two products
    before adding either of them.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 2
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 32)

        var v0_left = _full_mantissa(UInt64(31))
        var v0_right = _full_mantissa(UInt64(32))
        var v1_left = _full_mantissa(UInt64(33))
        var v1_right = _full_mantissa(UInt64(34))
        var learning_rate = _uniform(UInt64(555)) + 0.5
        var base = _full_mantissa(UInt64(56))

        var trees: List[Tree] = [
            _stump(0, 15, v0_left, v0_right),
            _stump(1, 15, v1_left, v1_right),
        ]
        var flat = flatten_trees(trees, [base], 1, learning_rate)

        var predictor = GpuPredictor(n_features, 1)
        predictor.upload_ensemble(flat)
        var got = predictor.raw_scores(data, IterationRange.slice(2, 0, 2))
        assert_equal(len(got), n_rows)

        var lr32 = Float32(learning_rate)
        var base32 = Float32(base)
        var step = List[Float32](capacity=4)
        step.append(lr32 * Float32(v0_left))
        step.append(lr32 * Float32(v0_right))
        step.append(lr32 * Float32(v1_left))
        step.append(lr32 * Float32(v1_right))

        var unfused_rows = 0
        var fused_rows = 0
        var neither = 0
        var separable = 0
        var worst = 0
        var first_report = True
        for r in range(n_rows):
            var i0 = 0 if Int(data.bins[0 * n_rows + r]) <= 15 else 1
            var i1 = 2 if Int(data.bins[1 * n_rows + r]) <= 15 else 3
            var unfused = (base32 + step[i0]) + step[i1]
            var mid = fma(lr32, Float32(_leaf_value(i0, v0_left, v0_right,
                v1_left, v1_right)), base32)
            var fused = fma(
                lr32,
                Float32(
                    _leaf_value(i1, v0_left, v0_right, v1_left, v1_right)
                ),
                mid,
            )
            if _bits32(unfused) != _bits32(fused):
                separable += 1
            var seen = _f32_bits(got[r])
            if seen == _bits32(unfused):
                unfused_rows += 1
            elif seen == _bits32(fused):
                fused_rows += 1
            else:
                neither += 1
            if seen != _bits32(unfused):
                var d = Int(seen) - Int(_bits32(unfused))
                if d < 0:
                    d = -d
                if d > worst:
                    worst = d
                if first_report:
                    first_report = False
                    print(
                        "predict kernel is not the unfused answer at row ", r,
                        ": device ", hex(seen),
                        " unfused ", hex(_bits32(unfused)),
                        " fused ", hex(_bits32(fused)),
                        " ulps(device, unfused) ", Int(seen)
                        - Int(_bits32(unfused)),
                        sep="",
                    )
        if unfused_rows != n_rows:
            print(
                "predict rounding: unfused ", unfused_rows,
                " fused ", fused_rows, " neither ", neither,
                " of ", n_rows, ", worst ", worst, " ulp", sep="",
            )
        assert_true(separable > 0)
        assert_equal(unfused_rows, n_rows)


def _leaf_value(
    slot: Int,
    v0_left: Float64,
    v0_right: Float64,
    v1_left: Float64,
    v1_right: Float64,
) -> Float64:
    """The leaf value the two stumps' four leaves hold, by the slot index the
    reference above uses."""
    if slot == 0:
        return v0_left
    if slot == 1:
        return v0_right
    if slot == 2:
        return v1_left
    return v1_right


def test_split_searcher_allocates_its_histogram_lazily() raises:
    """A searcher's own histogram buffer appears on first use and not before.

    Both halves matter. That a fresh searcher has not allocated it is the
    saving, since the trainer's two device search paths read the histogram
    builder's buffer and never this one. That `upload_histogram` and
    `search` still work afterwards is what says the saving cost nothing: the
    standalone path is the reason the buffer exists at all, so it has to
    keep producing the same record it produced when the allocation was
    unconditional.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_features = 2
        var n_bins = 8
        var size = n_features * n_bins
        var words = List[Int32](capacity=3 * size)
        for _ in range(3 * size):
            words.append(Int32(0))
        # One clean split on feature 0 between bins 3 and 4: the gradient
        # sign flips there and nowhere else, so the best split is knowable
        # without a reference implementation.
        for b in range(n_bins):
            var g = -100 if b < 4 else 100
            words[0 * n_bins + b] = Int32(g)
            words[size + 0 * n_bins + b] = Int32(50)
            words[2 * size + 0 * n_bins + b] = Int32(10)
            words[1 * n_bins + b] = Int32(0)
            words[size + 1 * n_bins + b] = Int32(50)
            words[2 * size + 1 * n_bins + b] = Int32(10)

        var searcher = GpuSplitSearcher(n_features, n_bins)
        assert_false(searcher.hist_owned)
        searcher.set_monotone([])
        searcher.set_allowed([])
        assert_false(searcher.hist_owned)

        searcher.upload_histogram(words)
        assert_true(searcher.hist_owned)
        var params = GpuSplitParams(
            1.0, 0.0, 0.0, 0, CategoricalParams.default()
        )
        var record = searcher.search(params, 1.0, 1.0)
        assert_true(record.found)
        assert_equal(record.feature, 0)
        assert_equal(record.bin, 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
