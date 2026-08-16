"""The bit-packed feature-major bin layout: its geometry, and the device pass
that builds it.

`gpu_binned_layout.mojo` describes a family of layouts and refuses to pick
one. `gpu_blocked_bins.mojo` implements one corner of it, `[block][row][G]` at
one byte per cell. This module implements a **different** corner, the one that
narrows the cell instead of rearranging it: one feature per block, row-major
inside the block (which for a one-feature block is the feature's own column),
stored at `ceil(log2(bins_used(f)))` bits per cell instead of eight. It is
what that file calls `LAYOUT_FEATURE_MAJOR_PACKED`, and `packed_plan` below
returns the very `BinLayoutPlan` this module's closed form implements, so the
layout that ships and the layout that is priced are one object.

The two are orthogonal and are not composable here. Blocking removes the
row-side traffic (`G` features share one row index and one gradient pair);
packing removes the bin-side traffic. `GpuActiveRows` refuses to have both on
at once, because the reader would need a block base, a lane, a row stride and
a bit width in one address and nothing has measured either arm yet.

Why this is worth building now, and what changed
------------------------------------------------
`gpu_blocked_bins.mojo`'s docstring argues against compression, and its
arithmetic was right for what it knew: it counted the bin as one byte of
thirteen moved per (row, feature) -- one bin, four for the row index, eight
for the quantized gradient pair -- so halving it removes under four percent of
the traffic.

That accounting is a *bytes-requested* model, and the wave that opened this
lane measured the phase instead of modelling it. At 1,000,000 x 50 at 256 bins
the histogram decomposes as **gather 56.7 percent**, atomics 15.9, launch
15.5, residual 12.0. The gather is the phase. The row index and the gradient
pair are read at a per-thread stride that consecutive threads walk together,
while the bin is read once per (row, feature) from a column the node's rows
sample sparsely; they are not thirteen interchangeable bytes.

What this module does **not** claim is that packing helps at 256 bins. It
cannot: a 255-bin feature needs eight bits and this layout stores it in eight,
byte for byte and address for address in its own column. The claim is
narrower and it is arithmetic rather than a hope: a feature that uses `k` bins
is fetched at `ceil(log2(k))/8` of the bytes it is fetched at today, and real
matrices are full of low-cardinality columns. `packed_bin_bytes_per_visit`
is that number and `packed_is_identity` is the refusal that says when it is
one.

The layout
----------
Feature `f` occupies a packed stream of `n_rows` elements at `width[f]` bits
beginning at byte `base[f]`, and the streams are laid out in feature order
with every base aligned to `BLOCK_ALIGN_BYTES`:

    base[0]   = 0
    base[f+1] = align_up(base[f] + packed_stream_bytes(n_rows, width[f]), 16)
    total     = base[F-1] + packed_stream_bytes(n_rows, width[F-1])

    offset(f, r) = base[f] + ((r * width[f]) >> 3)
    shift(f, r)  =           ((r * width[f]) &  7)
    bin(f, r)    = ((buf[offset] | (buf[offset + 1] << 8)) >> shift) & mask

`gpu_bin_packing.mojo` owns those last three lines and this module does not
restate them: `packed_offset` and `packed_shift` call
`element_byte_offset` and `element_bit_shift`, and the histogram row loop
evaluates the same two expressions inline because a kernel cannot call into a
host `List`. `check_packed_matches_plan` is what proves the closed form here,
the plan's table walk, and the primitives all agree.

The tail pad, and why every stream has one below width 8
--------------------------------------------------------
The decode window is two bytes, so the last element of a stream *reads* the
byte one past its data even when it does not straddle into it. Every stream
below width 8 therefore carries one pad byte, which is what keeps that read
inside the allocation and what keeps two adjacent streams from sharing a
writable byte. At width 8 there is no second window byte and no pad, so a
width-8 stream is exactly `n_rows` bytes and is byte for byte the feature's
column in `BinnedMatrix.bins`, displaced to an aligned base.

That last sentence is the reason the reader has a width-8 fast path at all:
inside a mixed-width matrix the 8-bit features cost exactly what they cost
today, one byte load with no shift and no mask, and only the narrow features
pay the decode.

Alignment, and why the all-8 case is refused rather than handled
----------------------------------------------------------------
`BinLayoutPlan` aligns block bases to `BLOCK_ALIGN_BYTES` **except** in its
passthrough case -- one feature per block, width 8 throughout -- where it
takes alignment 1 so that the plan's buffer is `BinnedMatrix.bins` itself with
no pads and no copy. A plan whose every width is 8 is therefore that
passthrough plan, its bases are `f * n_rows`, and uploading a second copy of
the matrix under it would cost residency and buy nothing at all.
`packed_is_identity` reports it and `GpuActiveRows.set_packed_bins` refuses
it, which is exactly what `gpu_blocked_bins.blocked_is_identity` does for
`G = 1`. With that case refused, the alignment is 16 unconditionally and the
closed form above has no branch in it.

Exactness
---------
**This layout cannot change a histogram, by construction.** The argument is
the same one the blocked layout makes and it is the strongest kind available
here: *the bin id is not touched*. `gpu_bin_packing` never renumbers a value
-- packing at width `w` and unpacking at width `w` returns the integer that
went in, which `column_roundtrips` states as a function -- so for every
`(feature, row)` the id this reader decodes is the id
`bins[f * n_rows + r]` held. The same id then selects the same shared cell,
the same three quantized values are added to it by the same atomics, and the
set of `(row, feature)` visits, the tiling, the slot assignment and the flush
are all untouched. Accumulation is fixed-point Int32 and integer addition is
associative and commutative, so even a reordering could not have moved a bit,
and this arm does not reorder anything either.

A re-encoding that yields the same bin id per (row, feature) produces the same
histogram by construction. That is the whole claim, and it holds only while
the encoding is lossless, which is the one thing a width choice can break:

- a width narrower than a feature's largest bin id truncates the id;
- a width narrower than `BinnedMatrix.missing_bin[f]` loses the missing
  sentinel, so missing rows route by the threshold instead of by
  `default_left`;
- a width narrower than a categorical feature's highest category bin drops a
  category out of its 256-bit split set.

None of the three violates a bound, exceeds a tolerance, or produces an
illegal bin id, so nothing downstream would catch any of them. They are
refused up front instead: `packed_widths_from_matrix` derives widths from the
observed extents *including* the reserved missing bin,
`check_packed_widths_cover` re-checks a width table against a matrix cell by
cell, and `gpu_binned_layout.check_markers_preserved` is run inside
`packed_plan`'s descriptor path. `packed_roundtrips` is the whole invariant as
one executable function, and if it is ever false a histogram built from this
buffer is a histogram of different data.

What the transform costs
------------------------
One kernel launch per fit, not per tree and not per node. It reads
`n_rows * n_features` bytes and writes `packed_bytes` of them, both streamed.
The packed buffer does **not** replace the feature-major one: `_flag_scan_kernel`
and `_scatter_kernel` in `gpu_active_rows.mojo`, `gpu_predict`, `gpu_sparse`,
`gpu_categorical` and the CPU builder all still index `bins[f * n_rows + r]`,
and a half-converted matrix is a much worse state than either layout. So both
stay resident and the device pays `n_rows * n_features + packed_bytes` for
bins; on a low-cardinality matrix the second term is a fraction of the first,
which is the one residency argument this layout has that the blocked one does
not.

Rejected: one width for the whole matrix
----------------------------------------
A single global width is what a uniform ELLPACK does, and it would let the
reader hoist the width out of the slot loop entirely. It was rejected because
the width would have to be the *widest* feature's, so one 255-bin column in a
matrix of booleans would put every column back at eight bits and the layout
would be the identity again. Per-feature widths cost one Int per owned slot,
read once outside the row walk on exactly the footing the column base is
already on.

Rejected: rounding widths to 1, 2, 4, 8
---------------------------------------
Widths that divide 8 never straddle a byte, so the decode would be one load
rather than two. It was rejected because it gives up the range that matters
most: a 64-bin feature needs 6 bits and would be stored at 8, and 33 to 64
bins is an extremely common cardinality. The straddle costs a second byte
*load* but almost never a second memory *sector*, because the two bytes are
adjacent; the rounding would cost a third of the bits on the widths where
there are the most bits to save. `packed_window_bytes` reports the load count
so a measurement can hold the two against each other rather than trusting
this paragraph.
"""

from std.gpu import block_idx, global_idx
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from .binning import BinnedMatrix
from .gpu_bin_packing import (
    BIN_WIDTH_FALLBACK,
    BIN_WIDTH_MAX,
    BIN_WIDTH_MIN,
    BITS_PER_BYTE,
    bin_width_mask,
    bins_representable,
    check_bin_width,
    element_bit_shift,
    element_byte_offset,
    pack_value,
    packed_data_bytes,
    packed_stream_bytes,
    unpack_value,
)
from .gpu_binned_layout import (
    BLOCK_ALIGN_BYTES,
    BinLayoutPlan,
    LAYOUT_MAX_INDEX,
    check_layout_support,
    observed_bins_from_matrix,
    plan_feature_major,
    widths_from_bin_counts,
)


# The per-slot width value that means "this slot is not packed", so the
# histogram row loop takes the ordinary `bins[base + row]` byte gather. Zero
# is not a legal packed width (`BIN_WIDTH_MIN` is 1), which is what makes it
# usable as the off marker in the same way `BLOCKED_STRIDE_NONE` is.
comptime PACKED_WIDTH_OFF = 0

# Two entries per feature in the device table: the stream's byte base, then
# its width in bits. Interleaved rather than two buffers so a launch takes one
# pointer, and so a slot's two values share a cache line.
comptime PACKED_TABLE_STRIDE = 2


def _align_up(value: Int, alignment: Int) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


# --- Widths ---------------------------------------------------------------


def packed_widths_from_bin_counts(bin_counts: List[Int]) raises -> List[Int]:
    """Per-feature storage widths from per-feature bin *counts*.

    `ceil(log2(count))` bits, which is `gpu_bin_packing.width_for_bins`: a
    feature using 256 bins needs 8, one using 64 needs 6, one using 16 needs
    4, and a boolean needs 1. A count of one still needs one bit, because a
    stream of width zero has no addressable elements.

    The count is a *count*, not a maximum id, and getting that wrong by one is
    the error this layout cannot survive: a feature whose largest bin id is 63
    uses 64 bins and needs 6 bits, and passing 63 here would give it 6 as well
    only by luck. `packed_widths_from_matrix` is the derivation that cannot be
    off by one because it is written against ids.
    """
    return widths_from_bin_counts(bin_counts, True)


def packed_widths_from_matrix(data: BinnedMatrix) raises -> List[Int]:
    """Per-feature widths derived from the matrix that will be packed.

    `gpu_binned_layout.observed_bins_from_matrix` scans each column for one
    past its largest id **and** keeps room for the feature's reserved missing
    bin whether or not any row of this matrix is missing, so a later matrix
    from the same mapper does not overflow the width on its first NaN. That
    reserved room is not optional and it is not conservatism: a width that
    cannot hold `missing_bin[f]` silently reroutes every missing row, which
    nothing downstream would catch.

    A full `n_rows * n_features` host pass, run once per fit. It is valid for
    *this* matrix; `check_packed_widths_cover` is how a second matrix is
    checked against the same table, and deriving from a `BinMapper` instead
    (`gpu_binned_layout.declared_bins_from_mapper`) is how the question is
    avoided.
    """
    return widths_from_bin_counts(observed_bins_from_matrix(data), True)


def packed_widths_valid(widths: List[Int]) -> Bool:
    """Whether every entry is a width `gpu_bin_packing` can address."""
    if len(widths) < 1:
        return False
    for f in range(len(widths)):
        if widths[f] < BIN_WIDTH_MIN or widths[f] > BIN_WIDTH_MAX:
            return False
    return True


def check_packed_widths(widths: List[Int]) raises:
    """Raising form of `packed_widths_valid`."""
    if len(widths) < 1:
        raise Error("a packed bin layout needs at least one feature")
    for f in range(len(widths)):
        check_bin_width(widths[f])


def packed_is_identity(widths: List[Int]) raises -> Bool:
    """Whether packing at these widths reproduces the feature-major buffer,
    so uploading a second copy of it would buy nothing.

    True exactly when every width is 8. That is the `BinLayoutPlan`
    passthrough case, whose bases are `f * n_rows` and whose buffer *is*
    `BinnedMatrix.bins`; see the module docstring for why it is refused rather
    than handled, and note the second consequence, which is that with it
    refused the block alignment below is unconditionally
    `BLOCK_ALIGN_BYTES`.
    """
    check_packed_widths(widths)
    for f in range(len(widths)):
        if widths[f] != BIN_WIDTH_FALLBACK:
            return False
    return True


# --- Geometry -------------------------------------------------------------


def packed_column_bases(n_rows: Int, widths: List[Int]) raises -> List[Int]:
    """The byte each feature's stream begins at.

    A prefix sum rather than a closed form, because the streams have different
    lengths: `base[f+1] = align_up(base[f] + stream(f), BLOCK_ALIGN_BYTES)`.
    That is `BinLayoutPlan.__init__`'s own loop at `block_size = 1`, and
    `check_packed_matches_plan` proves the two agree rather than asserting it.

    The table is computed on the host once per fit and uploaded as
    `packed_table`, because a kernel cannot walk a `List[Int]`; the row loop
    reads one entry per owned slot, on the same footing the feature-major
    multiply `f * n_rows` is on today.
    """
    check_packed_widths(widths)
    if n_rows < 1:
        raise Error("a packed bin layout needs at least one row")
    var out = List[Int](capacity=len(widths))
    var offset = 0
    for f in range(len(widths)):
        out.append(offset)
        var stream = packed_stream_bytes(n_rows, widths[f])
        if stream > LAYOUT_MAX_INDEX - offset:
            raise Error(
                "packed bin layout is larger than an Int32 byte offset holds"
            )
        offset = _align_up(offset + stream, BLOCK_ALIGN_BYTES)
    return out^


def packed_bytes(n_rows: Int, widths: List[Int]) raises -> Int:
    """Bytes the packed buffer occupies.

    The last stream's base plus its own length, so the final alignment tail is
    never allocated -- nothing addresses past the last feature's pad byte.
    Exactly `BinLayoutPlan.total_bytes` for the same plan.
    """
    var bases = packed_column_bases(n_rows, widths)
    var last = len(widths) - 1
    var total = bases[last] + packed_stream_bytes(n_rows, widths[last])
    if total > LAYOUT_MAX_INDEX:
        raise Error(
            "packed bin layout is larger than an Int32 byte offset holds"
        )
    return total


def packed_dense_bytes(n_rows: Int, widths: List[Int]) -> Int:
    """What the same matrix costs as the `UInt8` buffer it does not replace."""
    return n_rows * len(widths)


def packed_offset(
    feature: Int, row: Int, n_rows: Int, widths: List[Int]
) raises -> Int:
    """The byte the decode window starts at for `(feature, row)`.

        offset(f, r) = base[f] + ((r * width[f]) >> 3)

    The first of the two bytes the window reads, and the only one at width 8.
    """
    var bases = packed_column_bases(n_rows, widths)
    if feature < 0 or feature >= len(widths):
        raise Error("feature index out of range")
    if row < 0 or row >= n_rows:
        raise Error("row index out of range")
    return bases[feature] + element_byte_offset(row, widths[feature])


def packed_shift(
    feature: Int, row: Int, n_rows: Int, widths: List[Int]
) raises -> Int:
    """`(r * width[f]) & 7`, the low bit's position in the first window byte.
    Always zero at width 8, which is what the reader's fast path rests on."""
    if feature < 0 or feature >= len(widths):
        raise Error("feature index out of range")
    if row < 0 or row >= n_rows:
        raise Error("row index out of range")
    return element_bit_shift(row, widths[feature])


def packed_table(n_rows: Int, widths: List[Int]) raises -> List[Int32]:
    """The device table the histogram row loop reads: `[base, width]` per
    feature, interleaved.

    The whole of what a kernel needs to know about this layout. Everything
    else -- the alignment, the pad, the prefix sum -- is host arithmetic that
    has already happened by the time the table exists, which is the point of
    having one: the row loop forms an address from two integers and a shift
    and never re-derives a geometry.
    """
    var bases = packed_column_bases(n_rows, widths)
    var out = List[Int32](capacity=PACKED_TABLE_STRIDE * len(widths))
    for f in range(len(widths)):
        out.append(Int32(bases[f]))
        out.append(Int32(widths[f]))
    return out^


def packed_max_stream_bytes(n_rows: Int, widths: List[Int]) raises -> Int:
    """The longest single stream, which is what sizes the pack launch's row
    grid: every feature's threads walk the same `grid.x` and drop out past
    their own stream."""
    check_packed_widths(widths)
    var top = 0
    for f in range(len(widths)):
        var stream = packed_stream_bytes(n_rows, widths[f])
        if stream > top:
            top = stream
    return top


# --- What it saves --------------------------------------------------------


def packed_bin_bytes_per_visit(width: Int) raises -> Float64:
    """Bin bytes a single (row, feature) visit *wants* at this width:
    `width / 8`.

    One at width 8, and this is the number the lane exists to move. It is a
    useful-bytes figure and not a fetched-bytes one: what a device actually
    moves for a scattered read is a memory sector, so this is the streaming
    regime's cost exactly and the scattered regime's cost only insofar as
    narrower columns put more of the matrix in cache. `packed_window_bytes` is
    the other half of the accounting and the two are reported separately on
    purpose.
    """
    check_bin_width(width)
    return Float64(width) / Float64(BITS_PER_BYTE)


def packed_window_bytes(width: Int) raises -> Int:
    """Byte *loads* the decode issues per visit: one at width 8, two below it.

    The cost side of the trade, and the reason the module docstring refuses to
    call this change free. The two bytes are adjacent, so they are one memory
    sector except when the pair happens to straddle a sector boundary; against
    that, the column is `width / 8` as long. Which term wins is a measurement
    and this function exists so the measurement has both numbers.
    """
    check_bin_width(width)
    if width == BIN_WIDTH_FALLBACK:
        return 1
    return 2


def packed_bin_bytes_ratio(widths: List[Int]) raises -> Float64:
    """Bin bytes the whole matrix wants under this width table, over what it
    wants at one byte per cell. One when `packed_is_identity`."""
    check_packed_widths(widths)
    var bits = 0
    for f in range(len(widths)):
        bits += widths[f]
    return Float64(bits) / Float64(BITS_PER_BYTE * len(widths))


# --- The plan it implements -----------------------------------------------


def packed_plan(
    n_rows: Int, n_bins: Int, widths: List[Int]
) raises -> BinLayoutPlan:
    """This layout as the `BinLayoutPlan` `gpu_binned_layout.mojo` prices.

    One feature per block at the given per-feature widths, which is
    `plan_feature_major`. Returning a real plan rather than a description is
    what lets the cost model be applied to the layout that actually ships, and
    what lets `check_packed_matches_plan` compare two independently written
    pieces of address arithmetic instead of one against a comment.
    """
    check_packed_widths(widths)
    check_layout_support(n_rows, len(widths), n_bins)
    var copy = List[Int](capacity=len(widths))
    for f in range(len(widths)):
        copy.append(widths[f])
    return plan_feature_major(n_rows, len(widths), n_bins, copy^)


def check_packed_matches_plan(
    n_rows: Int, n_bins: Int, widths: List[Int]
) raises:
    """The closed form equals the plan's, for every cell.

    The invariant the module rests on, as a check rather than as a claim. If
    the two ever disagree the kernel decodes a **legal bin id belonging to a
    different row**: the histogram is well formed, the split is plausible, no
    bound is violated and no tolerance is exceeded, so there is no assertion
    anywhere downstream that would fire. Quadratic in the matrix and meant for
    tests and small shapes.

    Both the byte and the shift are compared. A stride error and a shift error
    are the two ways this address can be wrong, and on a matrix whose features
    all share one width they are indistinguishable -- both move every cell of
    a column by the same amount -- which is why the fixture that exercises
    this has features at several different widths.
    """
    if packed_is_identity(widths):
        raise Error(
            "an all-width-8 packed layout is the passthrough plan, whose"
            " bases are f * n_rows and whose alignment is 1; it is refused"
            " rather than compared"
        )
    var plan = packed_plan(n_rows, n_bins, widths)
    if plan.n_features != len(widths):
        raise Error("packed plan does not cover the feature axis")
    if plan.uniform_block_size() != 1:
        raise Error("packed plan's blocks are not one feature wide")
    if plan.is_passthrough():
        raise Error("packed plan resolved to the passthrough layout")
    if plan.bytes() != packed_bytes(n_rows, widths):
        raise Error("packed plan's size does not match the closed form")
    var bases = packed_column_bases(n_rows, widths)
    for f in range(len(widths)):
        if plan.block_offset[plan.block_of[f]] != bases[f]:
            raise Error("packed column base disagrees with the layout plan")
        for r in range(n_rows):
            if plan.byte_of(f, r) != packed_offset(f, r, n_rows, widths):
                raise Error(
                    "packed address arithmetic disagrees with the layout plan"
                    " it claims to implement"
                )
            if plan.shift_of(f, r) != packed_shift(f, r, n_rows, widths):
                raise Error(
                    "packed bit shift disagrees with the layout plan it"
                    " claims to implement"
                )


def check_packed_widths_cover(
    data: BinnedMatrix, widths: List[Int]
) raises:
    """Every cell of `data` fits the width its feature is stored at.

    The one way a width choice corrupts a dataset without violating a bound.
    Run against a second matrix -- a validation set, a prediction batch -- a
    width table derived from the training matrix's observed extents can be too
    narrow, and the failure is silent: a truncated id is a legal id.
    """
    check_packed_widths(widths)
    if data.n_features != len(widths):
        raise Error("width table must be one entry per feature")
    if len(data.bins) != data.n_rows * data.n_features:
        raise Error("binned matrix size must equal n_rows * n_features")
    var src = data.bins.unsafe_ptr()
    for f in range(data.n_features):
        var limit = bins_representable(widths[f])
        var col = f * data.n_rows
        for r in range(data.n_rows):
            if Int(src.unsafe_load(col + r)) >= limit:
                raise Error(
                    "bin id does not fit the width it is being packed at"
                )
        var mb = data.missing_bin[f]
        if mb >= 0 and mb >= limit:
            raise Error(
                "a feature's missing bin does not fit its storage width;"
                " packing it would reroute every missing row"
            )


# --- The per-byte composition, shared by the host and the kernel ----------
#
# The pack transform is written per *destination byte* rather than per cell,
# and that is not a style choice. Below width 8 two adjacent elements share a
# byte, so a thread per cell would have two writers on one byte and would lose
# updates -- `gpu_bin_packing`'s module docstring states the rule and this is
# the shape that respects it without a read-modify-write and without an
# atomic. One thread owns one byte, reads the handful of cells that overlap
# it, and writes it once.
#
# The two functions below are the element range that overlaps a byte. They are
# plain integer arithmetic with no `List` in them, so the kernel and the host
# reference call the *same* code and cannot drift; that is what makes a
# CPU-only test of the byte composition a test of the kernel's arithmetic and
# not of a paraphrase of it.


@always_inline
def packed_byte_first_elem(byte_in_stream: Int, width: Int) -> Int:
    """Lowest element index whose bits reach into this byte.

    Element `i` occupies bits `[i*w, i*w + w)` and the byte covers
    `[8b, 8b + 8)`, so the overlap condition on the low side is
    `i*w + w > 8b`, i.e. `i > 8b/w - 1`, whose least integer solution is
    `floor(8b/w)`. It is never negative and it may be an element that started
    in an earlier byte, which is the case the composition's negative shift
    handles.
    """
    return (BITS_PER_BYTE * byte_in_stream) // width


@always_inline
def packed_byte_last_elem(byte_in_stream: Int, width: Int) -> Int:
    """Highest element index whose bits reach into this byte.

    `i*w < 8b + 8` gives `i <= floor((8b + 7)/w)`. At width 8 this equals
    `packed_byte_first_elem` and the byte holds exactly one element, which is
    the specialization that keeps a width-8 stream the feature's own column.
    """
    return (BITS_PER_BYTE * byte_in_stream + BITS_PER_BYTE - 1) // width


@always_inline
def packed_place_bits(value: Int, elem: Int, byte_in_stream: Int, width: Int) -> Int:
    """`value`'s contribution to this byte, already positioned.

    The shift is `elem * width - 8 * byte`, which is negative for an element
    that began in the previous byte; a negative shift is a right shift of the
    element's high bits down to bit 0 of this one. The value is masked to
    `width` bits first, so a caller that has not range-checked its ids cannot
    smear one cell into its neighbour -- it merely stores a truncated id,
    which `check_packed_widths_cover` is what refuses.

    Both directions are needed and the pair is exhaustive: `width <= 8` and
    the byte covers 8 bits, so an element overlapping this byte begins at most
    7 bits before it and at most 7 bits into it.
    """
    var v = value & bin_width_mask(width)
    var sh = elem * width - BITS_PER_BYTE * byte_in_stream
    if sh >= 0:
        return (v << sh) & 255
    return (v >> (-sh)) & 255


# --- Host reference -------------------------------------------------------


def packed_relayout_host(
    data: BinnedMatrix, widths: List[Int]
) raises -> List[UInt8]:
    """The device pass, on the host.

    The reference implementation and the thing a test compares against. It is
    not on the training path: `GpuActiveRows` builds the buffer with the
    kernel below out of the feature-major copy already on the device, which
    costs no host work and no second upload. This exists so the address
    arithmetic can be checked without an accelerator, which is the half of CI
    an Apple M4 structurally cannot run.

    Written with `gpu_bin_packing.pack_value` per cell -- one ascending writer
    per stream, which is what the overlap rule requires -- rather than with
    the per-byte composition the kernel uses. Two independent derivations of
    the same bytes is the point: `packed_pack_host_bytewise` is the third, and
    a test that finds all three equal has checked the kernel's arithmetic and
    not a restatement of it.

    Pad bytes are left zero, so a buffer built here and a buffer built by the
    kernel are comparable everywhere and not only on the cells a feature id
    can name.
    """
    check_packed_widths(widths)
    if data.n_features != len(widths):
        raise Error("width table must be one entry per feature")
    if len(data.bins) != data.n_rows * data.n_features:
        raise Error("binned matrix size must equal n_rows * n_features")
    check_packed_widths_cover(data, widths)
    var total = packed_bytes(data.n_rows, widths)
    var bases = packed_column_bases(data.n_rows, widths)
    var out = List[UInt8](capacity=total)
    out.resize(total, 0)
    var src = data.bins.unsafe_ptr()
    for f in range(data.n_features):
        var base = bases[f]
        var w = widths[f]
        var col = f * data.n_rows
        for r in range(data.n_rows):
            pack_value(out, base, r, w, Int(src.unsafe_load(col + r)))
    return out^


def packed_pack_host_bytewise(
    data: BinnedMatrix, widths: List[Int]
) raises -> List[UInt8]:
    """The kernel's per-byte composition, run on the host.

    Byte for byte what `_packed_pack_kernel` writes, using the same three
    helpers, so a CPU-only test can check the transform the device performs
    rather than a description of it. The only thing left over that this cannot
    show is the launch grid, which is `(ceil(max_stream / threads),
    n_features)` and covers every byte of every stream by construction.
    """
    check_packed_widths(widths)
    if data.n_features != len(widths):
        raise Error("width table must be one entry per feature")
    if len(data.bins) != data.n_rows * data.n_features:
        raise Error("binned matrix size must equal n_rows * n_features")
    var total = packed_bytes(data.n_rows, widths)
    var bases = packed_column_bases(data.n_rows, widths)
    var out = List[UInt8](capacity=total)
    out.resize(total, 0)
    var src = data.bins.unsafe_ptr()
    for f in range(data.n_features):
        var base = bases[f]
        var w = widths[f]
        var col = f * data.n_rows
        var data_bytes = packed_data_bytes(data.n_rows, w)
        var stream = packed_stream_bytes(data.n_rows, w)
        for b in range(stream):
            if b >= data_bytes:
                out[base + b] = UInt8(0)
                continue
            var lo = packed_byte_first_elem(b, w)
            var hi = packed_byte_last_elem(b, w)
            var acc = 0
            for i in range(lo, hi + 1):
                if i >= data.n_rows:
                    break
                acc |= packed_place_bits(
                    Int(src.unsafe_load(col + i)), i, b, w
                )
            out[base + b] = UInt8(acc & 255)
    return out^


def packed_decode_host(
    buf: List[UInt8], feature: Int, row: Int, n_rows: Int, widths: List[Int]
) raises -> Int:
    """One cell out of a packed buffer, decoded the way the kernel decodes it.

    `gpu_bin_packing.unpack_value` is the decode and this only supplies the
    base, so the host reference and the device reader evaluate the same
    expression by construction rather than by agreement.
    """
    var bases = packed_column_bases(n_rows, widths)
    if feature < 0 or feature >= len(widths):
        raise Error("feature index out of range")
    if row < 0 or row >= n_rows:
        raise Error("row index out of range")
    var w = widths[feature]
    var at = bases[feature] + element_byte_offset(row, w)
    var window_end = at
    if w != BIN_WIDTH_FALLBACK:
        window_end = at + 1
    if at < 0 or window_end >= len(buf):
        raise Error("packed decode window outside the buffer")
    return unpack_value(buf, bases[feature], row, w)


def packed_roundtrips(data: BinnedMatrix, widths: List[Int]) raises -> Bool:
    """Whether the packed buffer decodes to the dense matrix, cell for cell.

    The exactness argument as a function. If this is ever false, a histogram
    built from the packed buffer is a histogram of different data and no
    tolerance anywhere would notice: the bins would be legal bin ids belonging
    to other rows or to other features.
    """
    var buf = packed_relayout_host(data, widths)
    for f in range(data.n_features):
        for r in range(data.n_rows):
            if (
                packed_decode_host(buf, f, r, data.n_rows, widths)
                != data.bin_at(r, f)
            ):
                return False
    return True


# --- The device pass ------------------------------------------------------


def _packed_pack_kernel(
    src: MutPointer[UInt8, MutAnyOrigin],
    dst: MutPointer[UInt8, MutAnyOrigin],
    tab: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_features: Int32,
):
    """Write one byte of one feature's packed stream.

    `grid.y` is the feature and `grid.x` walks that feature's stream, so one
    thread owns one destination byte and is its only writer. That is what the
    overlap rule requires: below width 8 adjacent elements share a byte, so a
    thread per *cell* would have two writers on one byte and would lose
    updates with nothing to catch it.

    The reads are the price of it. A thread composing byte `b` at width `w`
    reads the `ceil(8/w) + 1` or so source cells that overlap it, and those
    cells are adjacent in the feature-major column, so consecutive threads
    read overlapping contiguous runs -- a coalesced stream at 8/w times the
    width of the write. One launch per fit over a matrix already on the
    device, so this is not a cost worth a shared-memory stage.

    The per-feature geometry comes out of the same `tab` the histogram row
    loop reads, so the pack and the gather cannot disagree about where a
    stream begins or how wide its cells are. Everything else is derived here
    with the same two formulas `packed_stream_bytes` and `packed_data_bytes`
    use, spelled out because a kernel cannot call a raising host function.

    Pad bytes past the data are written as zero rather than skipped, so the
    buffer has no uninitialized bytes and a host-built copy and a device-built
    copy are comparable everywhere.
    """
    var f = Int(block_idx.y)
    var nf = Int(n_features)
    if f >= nf:
        return
    var nr = Int(n_rows)
    var base = Int(tab[unsafe_offset = PACKED_TABLE_STRIDE * f][0])
    var w = Int(tab[unsafe_offset = PACKED_TABLE_STRIDE * f + 1][0])
    if w < BIN_WIDTH_MIN or w > BIN_WIDTH_MAX:
        return
    var data_bytes = (nr * w + BITS_PER_BYTE - 1) // BITS_PER_BYTE
    var stream = data_bytes
    if w != BIN_WIDTH_FALLBACK:
        stream += 1
    var b = Int(global_idx.x)
    if b >= stream:
        return
    if b >= data_bytes:
        dst[unsafe_offset = base + b] = UInt8(0)
        return
    var lo = packed_byte_first_elem(b, w)
    var hi = packed_byte_last_elem(b, w)
    var acc = 0
    var col = f * nr
    for i in range(lo, hi + 1):
        if i >= nr:
            break
        acc |= packed_place_bits(
            Int(src[unsafe_offset = col + i][0]), i, b, w
        )
    dst[unsafe_offset = base + b] = UInt8(acc & 255)


def enqueue_packed_pack[
    src_origin: MutOrigin,
    dst_origin: MutOrigin,
    tab_origin: MutOrigin, //
](
    ctx: DeviceContext,
    src: MutPointer[UInt8, src_origin],
    dst: MutPointer[UInt8, dst_origin],
    tab: MutPointer[Int32, tab_origin],
    n_rows: Int,
    n_features: Int,
    max_stream: Int,
    threads: Int,
) raises:
    """Build the packed buffer on the device from the feature-major one.

    Once per fit. `src` is the `bins` pointer every histogram launch already
    receives and `dst` is the buffer `GpuActiveRows` allocated for the layout;
    they are different allocations, which is required -- a launch may not be
    handed the same buffer twice.

    `grid.x` is sized by the *longest* stream, so a feature stored at a
    narrower width has threads that return immediately. That wastes at most
    `(1 - min_width/max_width)` of one launch's threads, once per fit, against
    the alternative of a per-feature launch; it is not a trade worth eight
    hundred command encodings.

    Enqueued, not synchronized: it is ordered before the histogram that reads
    it by the queue, exactly as `_ensure_quantized`'s pass is ordered before
    the row loop that gathers from it.
    """
    comptime if not has_accelerator():
        raise Error(
            "the packed bin layout needs an accelerator; this build has none"
        )
    else:
        if n_rows < 1 or n_features < 1:
            raise Error("a packed pack needs a nonempty matrix")
        if max_stream < 1:
            raise Error("a packed pack needs a positive stream length")
        if threads < 1:
            raise Error("a packed pack needs a positive block width")
        var byte_blocks = (max_stream + threads - 1) // threads
        ctx.enqueue_function[_packed_pack_kernel](
            src,
            dst,
            tab,
            Int32(n_rows),
            Int32(n_features),
            grid_dim=(byte_blocks, n_features),
            block_dim=threads,
        )
