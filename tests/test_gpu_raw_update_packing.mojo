"""Whether packing the step into the range descriptor changed any value.

WHY THIS FILE EXISTS
--------------------
`GpuObjectiveState.update_raw_ranges` used to send two buffers to the device
per tree: a Float32 plane of per-node steps and an Int32 table of range
descriptors. Section 6.1 of `docs/GPU_PORTABILITY.md` establishes **by
measurement** that on Metal an `enqueue_copy` drains the whole queue in both
directions, so its cost is a host wait rather than a function of the byte
count, and two buffers of a few hundred bytes each were therefore buying two
waits. The step now rides inside the descriptor, in the word that was
padding, as the Float32's own bit pattern reinterpreted as an Int32. That is
one `enqueue_copy` per tree instead of two, **counted in source**; what it is
worth in seconds is a measurement this file does not attempt and does not
estimate.

The only thing that could have gone wrong is a changed value, and there are
exactly two ways for that to happen. This file closes both.

WHAT IS ASSERTED
----------------
`test_step_bits_survive_the_descriptor_round_trip` is the host half. A
Float32 written into an Int32 word and read back out must be the same
Float32, for every bit pattern a step can take, including the ones that a
conversion rather than a reinterpretation would destroy: negative zero, a
subnormal, the smallest and largest normals, and values whose significand is
full. A signed 32-bit integer cannot represent any of those as a number, so a
conversion would be visibly wrong here while a reinterpretation is exact by
construction. The assertion itself opens no device; it lives in a `test_gpu_`
file because it imports the module whose layout constants it is checking, and
`tools/run_tests.sh` classifies a file that reaches this module as
accelerator-only whatever any one function inside it does.

`test_packed_table_arm_matches_the_per_leaf_arm` is the device half. It runs
`update_raw_ranges` (packed, one launch, one copy) and
`update_raw_ranges_per_leaf` (the reference arm: one launch per leaf, node
value looked up on the device, learning rate applied in the kernel) over the
same rows, the same partition, the same node values, and the same starting
scores, and requires the resulting raw scores to be equal bit for bit. Not
close: equal. Every row belongs to exactly one leaf and Float32 addition is
exact given identical operands, so any difference at all would be a real one.

The node values and the learning rate come from `_full_mantissa`, whose
values use their whole Float32 significand. A round number like 0.5
multiplies exactly and would round the same however it were evaluated, so a
fixture built out of round numbers cannot detect the failure this file is
looking for.

WHAT THIS FILE DOES NOT DUPLICATE
---------------------------------
`tests/test_gpu_fma_consistency.mojo` already asserts that all three update
arms agree *and* that they agree on the unfused rounding specifically, which
is the stronger statement and pins the convention. This file is narrower on
purpose: it is the lane's own check that the packing is value-preserving,
and its host half is the part that no existing test covers, since no other
test looks at the descriptor encoding at all.

The device half skips (passing) with no accelerator, so the suite stays green
on CPU-only machines. Nothing here measures anything and nothing here trains.
"""

from std.memory import bitcast
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.binning import bin_equal_width
from mojotrees.gpu_objectives_native import (
    SEG_BEGIN,
    SEG_START,
    SEG_STEP,
    SEG_WORDS,
)
from mojotrees.histogram_gpu import GpuHistogramBuilder

from support import _make_features, _uniform


def _bits32(x: Float32) -> UInt32:
    """A Float32's IEEE 754 bit pattern as a plain `UInt32`."""
    return x.to_bits().cast[DType.uint32]()


def _f32_bits(x: Float64) -> UInt32:
    """The Float32 bit pattern of a score that was accumulated in Float32 and
    widened on the way back. The narrowing is exact, so this recovers the
    bits the device wrote."""
    return _bits32(Float32(x))


def _full_mantissa(seed: UInt64) -> Float64:
    """A value in (-1, 1) whose Float32 form uses its whole significand.

    An exact product rounds the same however it is evaluated and however it
    is carried, so a fixture of round numbers would pass whatever the
    encoding did."""
    return 2.0 * _uniform(seed) - 1.0


def test_step_bits_survive_the_descriptor_round_trip() raises:
    """A Float32 step stored in an Int32 descriptor word comes back
    unchanged, for every kind of bit pattern a step can have.

    This is the whole risk the packing introduces, isolated from the device.
    The word is written and read with the same `std.memory.bitcast` the
    module uses on each side of the transfer, so this asserts the actual
    encoding rather than a paraphrase of it; if it were a numeric conversion
    instead, every case below except the two ordinary ones would fail, and
    negative zero and the subnormal would fail loudest.
    """
    var steps: List[Float32] = [
        Float32(0.0),
        Float32(-0.0),
        Float32(1.0),
        Float32(-1.0),
        Float32(1.401298464324817e-45),  # smallest positive subnormal
        Float32(1.1754943508222875e-38),  # smallest positive normal
        Float32(3.4028234663852886e38),  # largest finite normal
        Float32(-3.4028234663852886e38),
    ]
    for seed in range(24):
        steps.append(
            Float32(_full_mantissa(UInt64(6_100 + seed)))
            * Float32(_full_mantissa(UInt64(seed)))
        )

    # A stand-in for one descriptor per step, laid out exactly as
    # `update_raw_ranges` lays `stage_seg` out, so the round trip is over the
    # real stride and the real word index rather than over a bare scalar.
    var words = List[Int32](capacity=len(steps) * SEG_WORDS)
    for _ in range(len(steps) * SEG_WORDS):
        words.append(Int32(0))
    for i in range(len(steps)):
        var base = i * SEG_WORDS
        words[base + SEG_START] = Int32(i)
        words[base + SEG_BEGIN] = Int32(2 * i)
        words[base + SEG_STEP] = bitcast[DType.int32, 1](steps[i])

    var differed = 0
    for i in range(len(steps)):
        var base = i * SEG_WORDS
        var got = bitcast[DType.float32, 1](words[base + SEG_STEP])
        if _bits32(got) != _bits32(steps[i]):
            differed += 1
            print(
                "step ", i, " did not survive the descriptor: in ",
                hex(_bits32(steps[i])), " out ", hex(_bits32(got)),
                sep="",
            )
        # The neighbouring words must be undisturbed, which is what says the
        # step occupies its own word and not part of another one.
        assert_equal(Int(words[base + SEG_START]), i)
        assert_equal(Int(words[base + SEG_BEGIN]), 2 * i)
    assert_equal(differed, 0)

    # Without this the test would pass on an all-zero fixture. The negative
    # zero case is the one that proves the comparison is over bits and not
    # over numeric equality: -0.0 == 0.0 is true and their bit patterns
    # differ, so a value comparison here would be blind to a sign flip.
    assert_true(len(steps) > 8)
    assert_equal(_bits32(steps[1]), UInt32(0x8000_0000))
    assert_true(_bits32(steps[0]) != _bits32(steps[1]))


def _split_tree_ranges(mut builder: GpuHistogramBuilder) raises:
    """Four splits on the builder's row set, leaving five live leaves and
    four emptied internal nodes among nodes 0 through 8.

    The emptied ones are exactly what both arms must skip, so the partition
    the two see is the same partition."""
    builder.begin_tree()
    builder.apply_split(0, 15, 0, 1, 2)
    builder.apply_split(1, 15, 1, 3, 4)
    builder.apply_split(2, 15, 2, 5, 6)
    builder.apply_split(3, 15, 3, 7, 8)


def test_packed_table_arm_matches_the_per_leaf_arm() raises:
    """The packed one-copy arm and the per-leaf reference arm land on the
    same raw scores, bit for bit.

    The two differ in everything the packing touches: the reference arm
    ships node values in a Float32 plane and multiplies by the learning rate
    inside the kernel with `node` as a launch argument, while the packed arm
    ships nothing but descriptors and adds a step the host already rounded.
    Agreement over a partition of several thousand rows is therefore a
    statement about the encoding and not about the fixture.
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
            target.append(_uniform(UInt64(21_000 + r)))

        var builder = GpuHistogramBuilder(data)
        _split_tree_ranges(builder)

        var values = List[Float64](capacity=n_nodes)
        for node in range(n_nodes):
            values.append(_full_mantissa(UInt64(1_900 + node)))
        var learning_rate = _uniform(UInt64(8_242)) + 0.5
        var base = _full_mantissa(UInt64(177))

        var packed = builder.objective_state(target)
        packed.init_raw(builder.ctx, [base])
        var per_leaf = builder.objective_state(target)
        per_leaf.init_raw(builder.ctx, [base])

        packed.update_raw_ranges(
            builder.ctx, builder.rows, values, learning_rate
        )
        per_leaf.update_raw_ranges_per_leaf(
            builder.ctx, builder.rows, values, learning_rate
        )

        var got_packed = packed.download_raw(builder.ctx)
        var got_leaf = per_leaf.download_raw(builder.ctx)
        assert_equal(len(got_packed), n_rows)
        assert_equal(len(got_leaf), n_rows)

        var disagreed = 0
        var first_report = True
        for r in range(n_rows):
            if _f32_bits(got_packed[r]) != _f32_bits(got_leaf[r]):
                disagreed += 1
                if first_report:
                    first_report = False
                    print(
                        "packed and per-leaf arms disagree at row ", r,
                        ": packed ", hex(_f32_bits(got_packed[r])),
                        " per-leaf ", hex(_f32_bits(got_leaf[r])),
                        sep="",
                    )
        assert_equal(disagreed, 0)

        # Without this the test would pass on two arms that both did
        # nothing. No node value here is zero, so every row must have moved
        # off the base score it started at.
        var base32 = Float32(base)
        var moved = 0
        for r in range(n_rows):
            if _f32_bits(got_packed[r]) != _bits32(base32):
                moved += 1
        assert_equal(moved, n_rows)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
