"""The bit-packed bin layout's addresses, its round trip, and its refusals.

`gpu_packed_bins.mojo` stores feature `f` as a stream of `n_rows` values at
`ceil(log2(bins_used(f)))` bits beginning at `base[f]`, and
`gpu_active_rows._hist_rows_step` decodes it inline. Everything the layout
claims rests on two integers per cell being right -- a byte and a shift -- and
they fail in the way nothing else in this package would catch: a wrong byte or
a wrong shift decodes a **legal bin id belonging to a different row**, so the
histogram is well formed, the split is plausible, no bound is violated, no
tolerance is exceeded, and no assertion anywhere downstream fires.

**Every fixture here has mixed cardinalities, and that is the whole design of
this file.** On a matrix whose features all share one width, a stride error
and a shift error are indistinguishable: both displace every cell of every
column by the same amount, so one wrong constant passes both. The widths below
are 8, 7, 6, 4, 2 and 1 in one matrix, which makes them independent -- a
stride error moves the columns by different amounts and a shift error moves
the rows within a column by different amounts, and no single wrong constant
reproduces the right answer on all six.

The shapes are chosen for the cases the arithmetic is most likely to be wrong
on rather than for coverage:

- `n_rows` odd and coprime to nothing in particular (37, 13, 1), so a stream's
  data length is not a multiple of anything and the last element of a narrow
  column ends mid-byte;
- widths that do not divide 8 (7, 6, 3), where elements straddle byte
  boundaries and the two-byte decode window is load-bearing. A fixture at 1, 2,
  4 and 8 only would let a straddle bug through entirely;
- width 8 mixed in among them, which is the reader's fast path and the one
  width whose stream must have no pad;
- the all-width-8 table, which is the passthrough plan and is refused rather
  than packed.

Three independent derivations of the same bytes are compared, and independent
is the point rather than a bonus:

- `packed_relayout_host`, which writes cell by cell with
  `gpu_bin_packing.pack_value` (a read-modify-write of a two-byte window);
- `packed_pack_host_bytewise`, which composes each destination byte from the
  cells overlapping it, using the *same* three helpers
  `_packed_pack_kernel` uses. Equality with the first is what makes this a
  test of the device transform's arithmetic rather than of a paraphrase of it;
- `gpu_binned_layout.BinLayoutPlan`, which computes the addresses by walking a
  per-block offset table it built from `packed_stream_bytes` and `_align_up`.

Everything here is host arithmetic over `List[UInt8]`. It opens no device,
which is deliberate rather than a limitation: the addresses are the same
integers on both sides of the bus, and what a CPU-only run cannot reach is the
pack launch's grid, which is `(ceil(max_stream / threads), n_features)` and
covers every byte of every stream by construction.
"""

# run_tests: cpu-safe -- opens no device; see tools/run_tests.sh gpu_by_content.

from std.testing import assert_equal, assert_false
from std.testing import assert_raises, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix
from mojotrees.categorical import CategoricalSpec
from mojotrees.gpu_bin_packing import packed_stream_bytes, width_for_bins
from mojotrees.gpu_binned_layout import BLOCK_ALIGN_BYTES
from mojotrees.gpu_packed_bins import (
    PACKED_TABLE_STRIDE,
    PACKED_WIDTH_OFF,
    check_packed_matches_plan,
    check_packed_widths,
    check_packed_widths_cover,
    packed_bin_bytes_per_visit,
    packed_bin_bytes_ratio,
    packed_byte_first_elem,
    packed_byte_last_elem,
    packed_bytes,
    packed_column_bases,
    packed_decode_host,
    packed_is_identity,
    packed_max_stream_bytes,
    packed_offset,
    packed_pack_host_bytewise,
    packed_place_bits,
    packed_plan,
    packed_relayout_host,
    packed_roundtrips,
    packed_shift,
    packed_table,
    packed_widths_from_bin_counts,
    packed_widths_from_matrix,
    packed_window_bytes,
)


# --- Fixtures -------------------------------------------------------------
#
# One mixed-cardinality matrix, built so that each feature's *observed*
# extents pin its width exactly. A feature meant for width `w` uses bin
# `2^w - 1` on at least one row and never uses a higher one, so
# `packed_widths_from_matrix` derives the intended table rather than a
# coincidence of the fixture's row count.


def _mixed_bin_counts() -> List[Int]:
    """Six features at 256, 100, 64, 16, 4 and 2 bins: widths 8, 7, 6, 4, 2,
    1. Three of the six do not divide 8, which is where the straddle lives."""
    return [256, 100, 64, 16, 4, 2]


def _mixed_matrix(n_rows: Int) raises -> BinnedMatrix:
    """The fixture. Feature `f`'s column cycles through `0 .. top(f)` with a
    per-feature phase, and row 0 of every column is that feature's top bin, so
    the observed extent is the declared one on any row count.

    The phase matters: a column that is the same sequence in every feature
    would let a *column* mix-up pass, because the wrong column would hold the
    right value. With the phases, feature `f` and feature `g` disagree on
    almost every row.
    """
    var counts = _mixed_bin_counts()
    var n_features = len(counts)
    var bins = List[UInt8](capacity=n_rows * n_features)
    bins.resize(n_rows * n_features, 0)
    for f in range(n_features):
        var top = counts[f] - 1
        for r in range(n_rows):
            var v: Int
            if r == 0:
                v = top
            else:
                v = (r * (f + 3) + f) % counts[f]
            bins[f * n_rows + r] = UInt8(v)
    return BinnedMatrix(bins^, n_rows, n_features, 256)


def _mixed_widths() raises -> List[Int]:
    return packed_widths_from_bin_counts(_mixed_bin_counts())


# --- Widths ---------------------------------------------------------------


def test_widths_are_ceil_log2_of_the_bin_count() raises:
    """The headline arithmetic: 256 bins need 8 bits, 64 need 6, 16 need 4, 2
    need 1. A 255-bin feature needs 8 and gains nothing, which is the case
    this layout is honest about rather than the one it is built for."""
    var w = _mixed_widths()
    assert_equal(len(w), 6)
    assert_equal(w[0], 8)
    assert_equal(w[1], 7)
    assert_equal(w[2], 6)
    assert_equal(w[3], 4)
    assert_equal(w[4], 2)
    assert_equal(w[5], 1)
    assert_equal(width_for_bins(255), 8)
    assert_equal(width_for_bins(256), 8)
    assert_equal(width_for_bins(257), 0)


def test_widths_from_matrix_match_widths_from_counts() raises:
    """The two derivations agree on the fixture, which is what makes the
    matrix's own extents a usable source. They can disagree in general -- a
    column whose top bin never appears observes narrower -- and the fixture
    puts every feature's top bin on row 0 so that they do not here."""
    var data = _mixed_matrix(37)
    var observed = packed_widths_from_matrix(data)
    var declared = _mixed_widths()
    assert_equal(len(observed), len(declared))
    for f in range(len(declared)):
        assert_equal(observed[f], declared[f])


def test_missing_bin_widens_a_narrow_feature() raises:
    """A feature whose reserved missing bin is above every id it actually
    holds still gets a width that covers the sentinel.

    This is the failure that has no bound to violate: at a width that cannot
    hold `missing_bin[f]`, every missing row decodes to a truncated id, routes
    by the threshold instead of by `default_left`, and produces a plausible
    tree. Nothing downstream would catch it.
    """
    var bins = List[UInt8](capacity=8)
    bins.resize(8, 0)
    # One boolean feature, ids 0 and 1, which alone would need one bit.
    for r in range(8):
        bins[r] = UInt8(r % 2)
    var missing = List[Int]()
    missing.append(9)
    var data = BinnedMatrix(
        bins^, 8, 1, 16, CategoricalSpec.none(), missing^
    )
    var w = packed_widths_from_matrix(data)
    assert_equal(w[0], 4)
    check_packed_widths_cover(data, w)


def test_a_width_too_narrow_for_a_cell_is_refused() raises:
    """`check_packed_widths_cover` is the refusal, and it has to be a refusal
    rather than a truncation: a truncated bin id is a legal bin id."""
    var data = _mixed_matrix(37)
    var w = _mixed_widths()
    w[2] = 5
    with assert_raises(contains="does not fit the width"):
        check_packed_widths_cover(data, w)
    with assert_raises(contains="does not fit the width"):
        _ = packed_relayout_host(data, w)


def test_invalid_widths_are_refused() raises:
    var w = _mixed_widths()
    w[0] = 9
    with assert_raises(contains="bin width must be in"):
        check_packed_widths(w)
    w[0] = 0
    with assert_raises(contains="bin width must be in"):
        check_packed_widths(w)
    with assert_raises(contains="at least one feature"):
        check_packed_widths(List[Int]())


# --- Geometry -------------------------------------------------------------


def test_bases_are_aligned_and_carry_every_streams_pad() raises:
    """`base[f+1] = align_up(base[f] + stream(f), 16)`, and `stream(f)` is
    `ceil(n_rows * w / 8)` plus one pad byte below width 8 and none at width
    8.

    The pad is what keeps the last element's two-byte decode window inside the
    allocation and what keeps two adjacent streams from sharing a writable
    byte. Dropping it is a bug that shows up on exactly one cell of one column
    and only when the data length happens to be flush.
    """
    var n_rows = 37
    var w = _mixed_widths()
    var bases = packed_column_bases(n_rows, w)
    assert_equal(len(bases), 6)
    assert_equal(bases[0], 0)
    for f in range(len(bases)):
        assert_equal(bases[f] % BLOCK_ALIGN_BYTES, 0)
    for f in range(len(bases) - 1):
        var stream = packed_stream_bytes(n_rows, w[f])
        assert_true(bases[f] + stream <= bases[f + 1])
        assert_true(bases[f + 1] - bases[f] < stream + BLOCK_ALIGN_BYTES)
    # Width 8 takes no pad; every narrower width takes exactly one.
    assert_equal(packed_stream_bytes(n_rows, 8), 37)
    assert_equal(packed_stream_bytes(n_rows, 1), 5 + 1)
    assert_equal(packed_stream_bytes(n_rows, 6), 28 + 1)


def test_total_bytes_stop_at_the_last_stream() raises:
    """The buffer is the last base plus the last stream, with no trailing
    alignment tail: nothing addresses past the last feature's pad byte."""
    var n_rows = 37
    var w = _mixed_widths()
    var bases = packed_column_bases(n_rows, w)
    var total = packed_bytes(n_rows, w)
    assert_equal(total, bases[5] + packed_stream_bytes(n_rows, w[5]))
    # And it is genuinely smaller than the matrix it shadows, which is the
    # residency claim this layout has and the blocked one does not.
    assert_true(total < n_rows * 6)


def test_bytes_per_visit_at_the_four_headline_cardinalities() raises:
    """What the lane exists to move, at the four bin counts the mandate
    names. `packed_window_bytes` is the other side of the trade and is
    reported next to it rather than folded into it."""
    # Every one of these is an exact binary fraction, so exact equality is the
    # right comparison and a tolerance would be hiding something.
    assert_equal(packed_bin_bytes_per_visit(width_for_bins(255)), 1.0)
    assert_equal(packed_bin_bytes_per_visit(width_for_bins(64)), 0.75)
    assert_equal(packed_bin_bytes_per_visit(width_for_bins(16)), 0.5)
    assert_equal(packed_bin_bytes_per_visit(width_for_bins(2)), 0.125)
    assert_equal(packed_window_bytes(8), 1)
    assert_equal(packed_window_bytes(7), 2)
    assert_equal(packed_window_bytes(1), 2)
    # The whole fixture: (8 + 7 + 6 + 4 + 2 + 1) / 48, which is not one.
    var ratio = packed_bin_bytes_ratio(_mixed_widths())
    assert_true(ratio > 0.5833 and ratio < 0.5834)


def test_max_stream_sizes_the_pack_launch() raises:
    var n_rows = 37
    var w = _mixed_widths()
    assert_equal(packed_max_stream_bytes(n_rows, w), 37)


def test_the_device_table_is_base_then_width_per_feature() raises:
    var n_rows = 37
    var w = _mixed_widths()
    var bases = packed_column_bases(n_rows, w)
    var tab = packed_table(n_rows, w)
    assert_equal(len(tab), PACKED_TABLE_STRIDE * 6)
    for f in range(6):
        assert_equal(Int(tab[PACKED_TABLE_STRIDE * f]), bases[f])
        assert_equal(Int(tab[PACKED_TABLE_STRIDE * f + 1]), w[f])
    # Zero is the reader's "not packed" marker, so no real width may be it.
    for f in range(6):
        assert_true(Int(tab[PACKED_TABLE_STRIDE * f + 1]) != PACKED_WIDTH_OFF)


# --- Against the plan -----------------------------------------------------


def test_addresses_match_the_layout_plan() raises:
    """The closed form here and `BinLayoutPlan`'s table walk agree on every
    byte *and* every shift, for three row counts.

    Two independently written pieces of address arithmetic, sharing only
    `gpu_bin_packing`'s primitives, so agreeing is evidence rather than a
    tautology -- and the plan is also the object `layout_node_cost` prices,
    which is what makes the cost model a statement about the layout that
    ships.
    """
    var w = _mixed_widths()
    check_packed_matches_plan(37, 256, w)
    check_packed_matches_plan(13, 256, w)
    check_packed_matches_plan(1, 256, w)


def test_the_plan_is_feature_major_packed_and_not_passthrough() raises:
    var w = _mixed_widths()
    var plan = packed_plan(37, 256, w)
    assert_equal(plan.n_blocks(), 6)
    assert_equal(plan.uniform_block_size(), 1)
    assert_equal(plan.uniform_width(), -1)
    assert_false(plan.is_passthrough())
    plan.check_blocks_disjoint()


def test_all_width_eight_is_the_passthrough_plan_and_is_refused() raises:
    """A matrix with no low-cardinality column has nothing to pack: its plan's
    bases are `f * n_rows` and its buffer is `BinnedMatrix.bins` itself.

    Refused rather than handled, and the refusal has a second job: with it in
    force the block alignment is unconditionally `BLOCK_ALIGN_BYTES`, so the
    closed form above has no branch in it.
    """
    var eight = List[Int]()
    for _ in range(6):
        eight.append(8)
    assert_true(packed_is_identity(eight))
    assert_false(packed_is_identity(_mixed_widths()))
    var plan = packed_plan(37, 256, eight)
    assert_true(plan.is_passthrough())
    plan.check_passthrough_offsets()
    with assert_raises(contains="passthrough"):
        check_packed_matches_plan(37, 256, eight)


# --- The round trip -------------------------------------------------------


def test_packed_buffer_decodes_to_the_dense_matrix() raises:
    """The invariant everything rests on. If this is ever false, a histogram
    built from the packed buffer is a histogram of different data."""
    var w = _mixed_widths()
    assert_true(packed_roundtrips(_mixed_matrix(37), w))
    assert_true(packed_roundtrips(_mixed_matrix(13), w))
    assert_true(packed_roundtrips(_mixed_matrix(1), w))
    assert_true(packed_roundtrips(_mixed_matrix(256), w))


def test_decode_reads_the_address_the_closed_form_publishes() raises:
    """`packed_decode_host` and the published `packed_offset` / `packed_shift`
    are the same two integers, cell by cell.

    Stated separately from the round trip because the round trip would still
    pass if *both* the writer and the reader used the same wrong address. This
    check is against the addresses the kernel is written to evaluate.
    """
    var n_rows = 37
    var w = _mixed_widths()
    var data = _mixed_matrix(n_rows)
    var buf = packed_relayout_host(data, w)
    var mask_hits = 0
    for f in range(data.n_features):
        for r in range(n_rows):
            var at = packed_offset(f, r, n_rows, w)
            var sh = packed_shift(f, r, n_rows, w)
            var window = Int(buf[at])
            if w[f] != 8:
                window |= Int(buf[at + 1]) << 8
            var got = (window >> sh) & ((1 << w[f]) - 1)
            assert_equal(got, data.bin_at(r, f))
            assert_equal(
                packed_decode_host(buf, f, r, n_rows, w), data.bin_at(r, f)
            )
            if sh != 0:
                mask_hits += 1
    # A fixture where no cell is ever shifted would not be exercising the
    # shift at all, which is half of what can be wrong here.
    assert_true(mask_hits > 0)


def test_width_eight_columns_are_the_dense_column_at_an_aligned_base() raises:
    """The reader's fast path, as a property of the bytes: at width 8 the
    stream is the feature's own column, byte for byte, displaced to
    `base[f]`. That is what licenses the one-load, no-shift, no-mask branch
    in `_hist_rows_step`."""
    var n_rows = 37
    var w = _mixed_widths()
    var data = _mixed_matrix(n_rows)
    var buf = packed_relayout_host(data, w)
    var bases = packed_column_bases(n_rows, w)
    assert_equal(w[0], 8)
    for r in range(n_rows):
        assert_equal(Int(buf[bases[0] + r]), data.bin_at(r, 0))
        assert_equal(packed_shift(0, r, n_rows, w), 0)
        assert_equal(packed_offset(0, r, n_rows, w), bases[0] + r)


# --- The device transform's arithmetic ------------------------------------


def test_the_kernels_bytewise_pack_equals_the_cellwise_pack() raises:
    """The device transform composes each destination byte from the cells that
    overlap it, because below width 8 two adjacent cells share a byte and a
    thread per cell would have two writers on one byte.

    `packed_pack_host_bytewise` is that composition on the host, through the
    same three helpers the kernel calls, so this equality is a check of the
    kernel's arithmetic rather than of a restatement of it. What it cannot
    reach is the launch grid, which covers every byte of every stream by
    construction.
    """
    var w = _mixed_widths()
    for n_rows in [1, 13, 37, 64, 256]:
        var data = _mixed_matrix(n_rows)
        var a = packed_relayout_host(data, w)
        var b = packed_pack_host_bytewise(data, w)
        assert_equal(len(a), len(b))
        for i in range(len(a)):
            assert_equal(Int(a[i]), Int(b[i]))


def test_the_overlap_range_covers_exactly_the_cells_in_a_byte() raises:
    """`packed_byte_first_elem` and `packed_byte_last_elem` bracket the
    elements whose bits reach a byte, checked against the definition rather
    than against a table.

    An element `i` at width `w` occupies bits `[i*w, i*w + w)` and byte `b`
    covers `[8b, 8b + 8)`; they overlap iff `i*w < 8b + 8` and
    `i*w + w > 8b`. The bracket has to be tight on both ends: too narrow drops
    a cell's bits from the byte, and too wide is harmless only because
    `packed_place_bits` shifts non-overlapping cells out of range, which is
    not a property to rely on.
    """
    for width in range(1, 9):
        for b in range(0, 12):
            var lo = packed_byte_first_elem(b, width)
            var hi = packed_byte_last_elem(b, width)
            assert_true(lo >= 0)
            assert_true(hi >= lo)
            # Every element in the bracket really does overlap.
            for i in range(lo, hi + 1):
                assert_true(i * width < 8 * b + 8)
                assert_true(i * width + width > 8 * b)
            # And the two just outside it do not.
            if lo > 0:
                assert_false((lo - 1) * width + width > 8 * b)
            assert_false((hi + 1) * width < 8 * b + 8)


def test_place_bits_handles_the_element_that_began_in_the_previous_byte() raises:
    """The negative shift, which is the case a one-directional implementation
    gets wrong and which only exists at widths that do not divide 8.

    At width 3, byte 1 covers bits 8..15 and element 2 covers bits 6..8, so
    its top bit is bit 0 of this byte and the shift is -2.
    """
    assert_equal(packed_byte_first_elem(1, 3), 2)
    # Element 2 holding the value 5 (binary 101) contributes its high bit.
    assert_equal(packed_place_bits(5, 2, 1, 3), 1)
    # Element 3 at bits 9..12 contributes its whole value at shift 1.
    assert_equal(packed_place_bits(5, 3, 1, 3), 5 << 1)
    # At width 8 the shift is always zero and the byte is the value.
    assert_equal(packed_place_bits(200, 4, 4, 8), 200)
    # Masking, not smearing: a value too wide for its width is truncated
    # inside its own field and never reaches its neighbour's bits.
    assert_equal(packed_place_bits(255, 0, 0, 2), 3)


# --- Refusals -------------------------------------------------------------


def test_shape_mismatches_are_refused() raises:
    var w = _mixed_widths()
    var data = _mixed_matrix(37)
    var short = List[Int]()
    short.append(4)
    with assert_raises(contains="one entry per feature"):
        check_packed_widths_cover(data, short)
    with assert_raises(contains="one entry per feature"):
        _ = packed_relayout_host(data, short)
    with assert_raises(contains="row index out of range"):
        _ = packed_offset(0, 37, 37, w)
    with assert_raises(contains="feature index out of range"):
        _ = packed_offset(6, 0, 37, w)
    with assert_raises(contains="at least one row"):
        _ = packed_column_bases(0, w)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
