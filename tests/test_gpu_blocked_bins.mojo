"""The `[block][row][G]` bin layout's address arithmetic, and its round trip.

`gpu_blocked_bins.mojo` publishes one closed-form address, `offset(f, r) =
(f / G) * block_stride + (f mod G) + r * G`, and the histogram row loop in
`gpu_active_rows._hist_rows_step` evaluates it. Everything the layout claims
rests on that address being right, and it fails in the one way nothing else in
this package would catch: a wrong offset reads a **legal bin id belonging to a
different row**, so the histogram is well formed, the split is plausible, no
bound is violated, and no tolerance is exceeded. There is no assertion
anywhere downstream that would fire.

So the checks here are the two independent ways of stating the same thing, and
they are independent on purpose:

- against `gpu_binned_layout.BinLayoutPlan`, which computes the same address
  by walking a per-block offset table it built from `packed_stream_bytes` and
  `_align_up`. The two derivations share no code, so agreeing is evidence and
  not a tautology, and the plan is also the object the cost model prices --
  which is what makes `layout_node_cost` a statement about the layout that
  ships rather than about one somebody described.
- against the dense matrix, cell for cell, through `blocked_roundtrips`. This
  is the `matches_dense` invariant of the packed family, applied to the one
  member that has a kernel.

Everything in this file is host arithmetic over `List[UInt8]`. It opens no
device, which is deliberate rather than a limitation: the addresses are what
can be wrong, and they are the same integers on both sides of the PCIe bus.
The device pass is a byte copy that computes these offsets and nothing else,
and `blocked_relayout_host` is that pass written out on the host, so what the
kernel could still get wrong beyond this file is its grid, which is
`(ceil(n_rows / threads), n_blocks)` and covers every cell by construction.

The shapes are chosen for the cases the closed form is most likely to be wrong
on rather than for coverage:

- `n_features` not a multiple of `G`, so the pad columns exist and the last
  block is a full block of which only part is named by a feature id;
- `n_rows * G` not a multiple of `BLOCK_ALIGN_BYTES`, so the block stride is
  strictly larger than the block's data and `b * n_rows * G` is the wrong
  answer. A shape where alignment happens to be free is the shape that would
  let a missing alignment term pass;
- `n_rows` of 1 and `n_features` of 1, where the pad is the whole block;
- every rung of the group ladder that a small matrix admits.
"""

# run_tests: cpu-safe -- opens no device; see tools/run_tests.sh gpu_by_content.

from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.testing import TestSuite

from mojotrees.binning import BinnedMatrix
from mojotrees.gpu_binned_layout import BLOCK_ALIGN_BYTES
from mojotrees.gpu_blocked_bins import (
    BLOCKED_STRIDE_NONE,
    blocked_alignment_fraction,
    blocked_block_stride,
    blocked_bytes,
    blocked_column_base,
    blocked_decode_host,
    blocked_group_valid,
    blocked_is_identity,
    blocked_n_blocks,
    blocked_offset,
    blocked_padded_features,
    blocked_plan,
    blocked_relayout_host,
    blocked_roundtrips,
    check_blocked_group,
    check_blocked_group_matches,
    check_blocked_matches_plan,
)


def _matrix(n_rows: Int, n_features: Int) -> BinnedMatrix:
    """A dense matrix whose every cell is distinguishable from every other.

    `(f * 7 + r * 3) % 251` gives a cell a value that depends on both indices
    and repeats with a period coprime to any block width used here, so a
    layout that swapped two rows, two features, or a row for a feature would
    change the decoded value rather than land on an equal one. A constant
    matrix, or one whose cells depend on a single index, would round-trip
    under a wrong address.
    """
    var bins = List[UInt8](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            bins.append(UInt8((f * 7 + r * 3) % 251))
    return BinnedMatrix(bins^, n_rows, n_features, 256)


def test_group_ladder() raises:
    """`G` is a rung of the feature-group ladder and nothing else."""
    assert_true(blocked_group_valid(1))
    assert_true(blocked_group_valid(2))
    assert_true(blocked_group_valid(4))
    assert_true(blocked_group_valid(8))
    assert_true(blocked_group_valid(16))
    assert_false(blocked_group_valid(3))
    assert_false(blocked_group_valid(32))
    assert_false(blocked_group_valid(0))
    with assert_raises():
        check_blocked_group(3)
    # A rung that is not the histogram's own group is refused, in both
    # directions, because both produce a correct histogram at a cost nobody
    # asked for and a measurement would attribute to the layout.
    check_blocked_group_matches(4, 4)
    with assert_raises():
        check_blocked_group_matches(4, 2)
    with assert_raises():
        check_blocked_group_matches(2, 4)
    # G = 1 is the feature-major arrangement, which is why it is never
    # uploaded.
    assert_true(blocked_is_identity(1))
    assert_false(blocked_is_identity(2))
    assert_equal(BLOCKED_STRIDE_NONE, 1)


def test_padding_and_size() raises:
    """The padded feature axis, the block count, and the buffer size."""
    assert_equal(blocked_padded_features(50, 4), 52)
    assert_equal(blocked_padded_features(52, 4), 52)
    assert_equal(blocked_padded_features(1, 4), 4)
    assert_equal(blocked_padded_features(50, 1), 50)
    assert_equal(blocked_n_blocks(50, 4), 13)
    assert_equal(blocked_n_blocks(1, 8), 1)

    # The block stride carries the plan's 16-byte alignment, so it is
    # strictly larger than the block's own data whenever `n_rows * G` is not
    # a multiple of it. 7 * 4 = 28 rounds to 32; 8 * 4 = 32 does not move.
    assert_equal(blocked_block_stride(7, 4), 32)
    assert_equal(blocked_block_stride(8, 4), 32)
    assert_equal(blocked_block_stride(1, 2), BLOCK_ALIGN_BYTES)

    # The last block's alignment tail is never allocated, so the buffer is
    # one stride short of `blocks * stride` by exactly that tail.
    assert_equal(blocked_bytes(7, 5, 4), 32 + 28)
    assert_equal(blocked_bytes(8, 8, 4), 32 + 32)

    with assert_raises():
        _ = blocked_padded_features(0, 4)
    with assert_raises():
        _ = blocked_block_stride(0, 4)


def test_offset_is_the_plan_offset() raises:
    """The closed form agrees with `BinLayoutPlan.byte_of`, cell for cell.

    Two independent derivations of the same address: this module multiplies,
    the plan walks a table of block offsets it accumulated with
    `packed_stream_bytes` and `_align_up`. Run over shapes where the padding
    and the alignment both bind, since a shape where neither does would pass
    with either term missing.
    """
    check_blocked_matches_plan(7, 5, 256, 4)
    check_blocked_matches_plan(8, 8, 256, 4)
    check_blocked_matches_plan(5, 3, 64, 2)
    check_blocked_matches_plan(1, 1, 32, 2)
    check_blocked_matches_plan(9, 17, 256, 8)
    check_blocked_matches_plan(3, 2, 256, 16)

    # And the plan really is the blocked one: uniform G-wide blocks over the
    # padded axis, not the passthrough plan wearing a different name.
    var plan = blocked_plan(7, 5, 256, 4)
    assert_equal(plan.n_features, 8)
    assert_equal(plan.uniform_block_size(), 4)
    assert_equal(plan.uniform_width(), 8)
    assert_false(plan.is_passthrough())
    assert_equal(plan.n_blocks(), 2)
    assert_equal(plan.bytes(), blocked_bytes(7, 5, 4))


def test_offset_by_hand() raises:
    """The address, written out, on a shape small enough to check.

    7 rows, 5 features, `G = 4`: two blocks, stride 32 (28 data bytes rounded
    up to 16). Feature 4 is lane 0 of block 1, so its column base is 32 and
    its row 3 is at 32 + 12.
    """
    assert_equal(blocked_block_stride(7, 4), 32)
    assert_equal(blocked_column_base(0, 7, 4), 0)
    assert_equal(blocked_column_base(3, 7, 4), 3)
    assert_equal(blocked_column_base(4, 7, 4), 32)
    assert_equal(blocked_offset(0, 0, 7, 4), 0)
    assert_equal(blocked_offset(3, 0, 7, 4), 3)
    assert_equal(blocked_offset(0, 1, 7, 4), 4)
    assert_equal(blocked_offset(4, 3, 7, 4), 44)
    # One row of one block is G adjacent bytes. That is the entire layout.
    for f in range(3):
        assert_equal(
            blocked_offset(f + 1, 2, 7, 4) - blocked_offset(f, 2, 7, 4), 1
        )
    with assert_raises():
        _ = blocked_offset(0, 7, 7, 4)


def test_roundtrip() raises:
    """The blocked buffer decodes to the dense matrix, cell for cell.

    If this is ever false a histogram built from the blocked buffer is a
    histogram of different data, and it is data that looks exactly as valid
    as the right data.
    """
    assert_true(blocked_roundtrips(_matrix(7, 5), 4))
    assert_true(blocked_roundtrips(_matrix(8, 8), 4))
    assert_true(blocked_roundtrips(_matrix(5, 3), 2))
    assert_true(blocked_roundtrips(_matrix(1, 1), 2))
    assert_true(blocked_roundtrips(_matrix(9, 17), 8))
    assert_true(blocked_roundtrips(_matrix(33, 50), 4))

    # The pad columns are written, not left as whatever the allocation held,
    # so a host-built and a device-built buffer are comparable everywhere and
    # not only on the cells a feature id can name. Features 5, 6, 7 of the
    # 7x5 matrix are pad.
    var buf = blocked_relayout_host(_matrix(7, 5), 4)
    assert_equal(len(buf), blocked_bytes(7, 5, 4))
    for f in range(5, 8):
        for r in range(7):
            assert_equal(blocked_decode_host(buf, f, r, 7, 4), 0)

    # And a real cell is what the dense matrix holds there, read through the
    # same address the kernel forms.
    var data = _matrix(7, 5)
    for f in range(5):
        for r in range(7):
            assert_equal(
                blocked_decode_host(buf, f, r, 7, 4), data.bin_at(r, f)
            )


def test_roundtrip_detects_a_shifted_address() raises:
    """The round trip is a real check, not one every matrix passes.

    Decoding at the neighbouring lane must disagree with the dense matrix on
    at least one cell; if it did not, the fixture would be one whose cells do
    not distinguish addresses and every assertion above would be vacuous.
    """
    var data = _matrix(7, 5)
    var buf = blocked_relayout_host(data, 4)
    var disagreements = 0
    for f in range(4):
        for r in range(7):
            if blocked_decode_host(buf, f + 1, r, 7, 4) != data.bin_at(r, f):
                disagreements += 1
    assert_true(disagreements > 0)


def test_alignment_fraction() raises:
    """The share of an active feature set that lands in whole blocks.

    The quantity that says which regime a subsampled run was in. Correctness
    does not depend on it -- an address is computed from a feature id -- but
    the win does, and a result taken at 0.25 that does not say so is the
    error this project has made in four separate places.
    """
    var all4: List[Int] = [0, 1, 2, 3]
    assert_equal(blocked_alignment_fraction(all4, 8, 4), 1.0)
    # One feature per block: every block is touched, none is consumed whole.
    var spread: List[Int] = [0, 5, 9, 13]
    assert_equal(blocked_alignment_fraction(spread, 16, 4), 0.0)
    # Half and half: block 0 whole, block 1 with a single feature.
    var half: List[Int] = [0, 1, 2, 3, 4]
    assert_equal(blocked_alignment_fraction(half, 8, 4), 0.8)
    with assert_raises():
        var bad: List[Int] = [9]
        _ = blocked_alignment_fraction(bad, 8, 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
