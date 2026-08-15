"""Bit-level packing primitives for binned data.

Pure integer arithmetic on byte buffers. Nothing here opens a
`DeviceContext`, allocates a device buffer, or launches a kernel, and nothing
here knows what a layout or a feature is: it packs and unpacks a *sequence of
small unsigned integers* into a byte stream, and it is the only place in
mojotrees where a bin id is stored as anything other than one `UInt8`.
`gpu_binned_layout.mojo` builds the per-feature plans that decide which
sequences get packed at which width; this module is the layer under it and
imports nothing from it.

The one representation
----------------------
A packed stream of `n` values at width `w` bits stores value `i` in the `w`
bits starting at bit `i * w`, counting from the least significant bit of byte
0, then byte 1, and so on:

    bit_offset(i) = i * w
    byte_of(i)    = (i * w) >> 3          # index of the byte holding bit 0
    shift_of(i)   = (i * w) & 7           # position of bit 0 inside that byte
    value(i)      = (stream16(byte_of(i)) >> shift_of(i)) & ((1 << w) - 1)

where `stream16(b)` is the two-byte little-end-first window
`buf[b] | (buf[b + 1] << 8)`, composed by *arithmetic* rather than by
reinterpreting memory as a 16-bit word. That distinction is the whole reason
this form was chosen over a 32-bit-word bit stream:

- **No endianness.** The window is built with a shift and an or, so the
  representation is byte-order independent. A `UInt32` bit stream would
  encode the host's endianness into the device buffer.
- **No alignment.** Every access is a byte load. A `UInt32` bit stream needs
  32-bit loads at arbitrary byte offsets, which is not portably expressible
  across Metal, CUDA, and HIP.
- **Two bytes always suffice.** `w <= 8` and `shift <= 7` give
  `shift + w <= 15`, so a value crosses at most one byte boundary and the
  two-byte window always contains it. This is the straddle invariant, and it
  is what bounds the decode to a fixed instruction count with no loop and no
  branch.

The `w = 8` specialization
--------------------------
At width 8 every shift is zero, the mask is the identity, and the second
window byte contributes nothing that the mask does not immediately discard.
So width 8 is not merely a member of the family, it is a *provable*
one-byte-load, no-ALU, no-slack path, and both `unpack_value` and
`pack_value` take it explicitly:

    value(i) = buf[i]        store(i, v) = buf[i] = v

That branch is on a loop-invariant width, so it costs nothing in a host loop
and disappears entirely in a kernel specialized on the width. It is what
makes the uncompressed layout the *same code* as the packed one rather than
a parallel implementation, and it is what lets a width-8 stream be byte for
byte the existing `BinnedMatrix.bins` column with no padding and no
reformatting at all.

Padding
-------
Below width 8 the window reads and writes `buf[b + 1]`, so a stream needs one
byte of slack past its last data byte; at width 8 it needs none:

    packed_stream_bytes(n, w) = ceil(n * w / 8) + (0 if w == 8 else 1)

The pad byte is never read for its contents, only for its address, so its
value is unspecified; `pack_column` leaves it zero. Dropping it below width 8
would make the last element of a stream touch one byte past the allocation.

Writing
-------
`pack_value` is a read-modify-write of both window bytes, including the case
where the value does not straddle. That keeps one code path, but it has a
consequence a caller has to respect:

    Packing two values whose windows overlap is safe only if the writes are
    ordered. Adjacent elements of one stream overlap whenever w < 8, so a
    stream must be packed by a single writer.

The subtle half of that rule is the *last* element. Below width 8 a stream's
writable footprint runs one byte past its data, because the last element's
read-modify-write covers its second window byte. Two such streams laid out
back to back would share that byte, and packing them concurrently would lose
updates even though neither stream's *data* overlaps the other's.

Two streams are therefore independent only when their footprints are
disjoint, which is what `writable_footprint_end` reports and
`streams_disjoint` tests, and why `gpu_binned_layout.BinLayoutPlan` pads
every block rather than only the last one. Alignment alone does not supply
it: a block whose data is a multiple of the alignment ends flush against the
next. At width 8 the footprint is the data and no pad is needed, which is the
case that has to stay free.

What this module refuses to do
------------------------------
It never renumbers a bin. A value packed at width `w` comes back as exactly
the integer that went in, so a missing bin keeps its id, a categorical bin
keeps its position in the 256-bit split set, and a threshold comparison
keeps its meaning. Any scheme that remapped ids to shrink a width would
silently invalidate `BinnedMatrix.missing_bin`, `Tree.threshold_bin`, the
`CatBitset` masks, and every serialized model, so there is no such scheme
here and there is no parameter that would enable one.
"""


# --- Width limits ---------------------------------------------------------

# Bins are unsigned integers of at most 8 bits, because the unpacked
# representation is `UInt8` and `binning.fit_bins` caps `max_bins` at 256.
comptime BIN_WIDTH_MIN = 1
comptime BIN_WIDTH_MAX = 8

# The uncompressed width. A plan that resolves every feature to this width is
# byte-identical to the existing `BinnedMatrix.bins` column layout.
comptime BIN_WIDTH_FALLBACK = 8

# Slack byte past the end of a packed stream, so the two-byte decode window
# of the last element stays inside the allocation. Applies below width 8
# only; `tail_pad_for` is the width-aware form and the one to call.
comptime PACK_TAIL_PAD = 1

# Bits per byte, named so the formulas below read as formulas.
comptime BITS_PER_BYTE = 8


def bin_width_valid(width: Int) -> Bool:
    """Whether `width` is a width this module can pack."""
    return width >= BIN_WIDTH_MIN and width <= BIN_WIDTH_MAX


def check_bin_width(width: Int) raises:
    """Raising form of `bin_width_valid`."""
    if not bin_width_valid(width):
        raise Error("bin width must be in [1, 8]")


def bin_width_mask(width: Int) -> Int:
    """`(1 << width) - 1`, the mask an unpack applies. Zero for an invalid
    width, so a mis-specified width decodes to zero rather than to garbage
    from a neighbouring element."""
    if not bin_width_valid(width):
        return 0
    return (1 << width) - 1


def bins_representable(width: Int) -> Int:
    """How many distinct bin ids a width can hold: ids `0 .. 2^width - 1`."""
    if not bin_width_valid(width):
        return 0
    return 1 << width


def width_for_bins(n_used_bins: Int) -> Int:
    """The narrowest width that can hold bin ids `0 .. n_used_bins - 1`.

    `n_used_bins` is a *count*, not a maximum id: a feature whose largest bin
    id is 5 uses 6 bins and needs 3 bits. Counts at or below 1 still need one
    bit, because a stream of width 0 has no addressable elements. Counts
    above 256 are not representable and return 0, which
    `check_width_for_bins` turns into an error.
    """
    if n_used_bins > bins_representable(BIN_WIDTH_MAX):
        return 0
    var width = BIN_WIDTH_MIN
    while bins_representable(width) < n_used_bins:
        width += 1
    return width


def check_width_for_bins(n_used_bins: Int) raises -> Int:
    """Raising form of `width_for_bins`."""
    var width = width_for_bins(n_used_bins)
    if width == 0:
        raise Error("a feature with more than 256 bins cannot be packed")
    return width


# --- Byte accounting ------------------------------------------------------


def packed_data_bytes(n_elems: Int, width: Int) raises -> Int:
    """Bytes of *data* a stream of `n_elems` values at `width` occupies,
    excluding the tail pad: `ceil(n_elems * width / 8)`."""
    check_bin_width(width)
    if n_elems < 0:
        raise Error("element count must not be negative")
    return (n_elems * width + BITS_PER_BYTE - 1) // BITS_PER_BYTE


def tail_pad_for(width: Int) raises -> Int:
    """Slack bytes a stream at `width` needs past its data.

    `PACK_TAIL_PAD` below width 8, where the decode window reads a second
    byte, and zero at width 8, where the specialization proves it does not.
    Zero here is what lets a width-8 stream be the existing `UInt8` column
    with nothing appended.
    """
    check_bin_width(width)
    if width == BIN_WIDTH_FALLBACK:
        return 0
    return PACK_TAIL_PAD


def packed_stream_bytes(n_elems: Int, width: Int) raises -> Int:
    """Bytes a stream of `n_elems` values at `width` must have allocated:
    its data bytes plus `tail_pad_for(width)`.

    At `width = 8` this is exactly `n_elems`, so an uncompressed stream is
    the uncompressed column and not a byte more.
    """
    return packed_data_bytes(n_elems, width) + tail_pad_for(width)


def writable_footprint_end(n_elems: Int, width: Int) raises -> Int:
    """Last byte index, relative to a stream's base, that reading or packing
    the stream may touch.

    One past the data below width 8, because the final element's
    read-modify-write covers its second window byte; the last data byte at
    width 8, where there is no second byte. Two streams may be packed
    concurrently only when their `[base, base + this]` intervals are
    disjoint; `streams_disjoint` is that test.
    """
    if n_elems <= 0:
        return -1
    return packed_stream_bytes(n_elems, width) - 1


def streams_disjoint(
    base_a: Int,
    n_a: Int,
    width_a: Int,
    base_b: Int,
    n_b: Int,
    width_b: Int,
) raises -> Bool:
    """Whether two streams can be packed by two writers without one losing
    the other's update. Compares writable footprints, not data ranges."""
    var end_a = base_a + writable_footprint_end(n_a, width_a)
    var end_b = base_b + writable_footprint_end(n_b, width_b)
    if base_a <= base_b:
        return end_a < base_b
    return end_b < base_a


def packed_bits_ratio(width: Int) raises -> Float64:
    """Packed bits per element over the uncompressed 8, ignoring the tail
    pad and any per-column alignment. Reported, never thresholded on: the
    handoff's cost model weighs bytes against decode instructions, and this
    ratio is only one of its inputs."""
    check_bin_width(width)
    return Float64(width) / Float64(BIN_WIDTH_FALLBACK)


# --- Addressing -----------------------------------------------------------
#
# The three functions below are the device contract. A kernel computes the
# same byte index and the same shift from the same element index, so a host
# unpack and a device decode cannot disagree without one of these changing.


@always_inline
def element_bit_offset(index: Int, width: Int) -> Int:
    """Bit position of element `index` inside its stream."""
    return index * width


@always_inline
def element_byte_offset(index: Int, width: Int) -> Int:
    """Index of the first of the two window bytes holding element `index`."""
    return (index * width) >> 3


@always_inline
def element_bit_shift(index: Int, width: Int) -> Int:
    """Position of element `index`'s low bit inside its first window byte,
    in `[0, 7]`."""
    return (index * width) & 7


def element_straddles(index: Int, width: Int) -> Bool:
    """Whether element `index` crosses a byte boundary, so its decode
    genuinely needs the second window byte.

    Never true at `width = 8` and never true for any element whose shift is
    zero. A layout policy uses it to count how much of a stream decodes with
    one byte load instead of two.
    """
    return element_bit_shift(index, width) + width > BITS_PER_BYTE


def check_element_index(index: Int, n_elems: Int) raises:
    """Bounds check for an element index against a stream length."""
    if index < 0 or index >= n_elems:
        raise Error("packed element index out of range")


# --- Scalar pack and unpack ----------------------------------------------


@always_inline
def unpack_value(
    buf: List[UInt8], base: Int, index: Int, width: Int
) -> Int:
    """Element `index` of the stream starting at byte `base` of `buf`.

    Unchecked: the caller guarantees `width` is valid, `index` is in range,
    and `buf` holds the stream's tail pad. `unpack_value_checked` is the
    raising form. The arithmetic below is the exact expression a device
    kernel evaluates, with `Int` standing in for the kernel's `UInt32`; both
    are exact for the 16-bit window, so the two agree bit for bit.

    The width-8 branch is the specialization the module docstring justifies:
    one load, no shift, no mask, and no second byte, so an uncompressed
    stream reads exactly as `bins[f * n_rows + r]` reads today.
    """
    if width == BIN_WIDTH_FALLBACK:
        return Int(buf[base + index])
    var byte = base + element_byte_offset(index, width)
    var window = Int(buf[byte]) | (Int(buf[byte + 1]) << BITS_PER_BYTE)
    return (window >> element_bit_shift(index, width)) & bin_width_mask(
        width
    )


def unpack_value_checked(
    buf: List[UInt8], base: Int, index: Int, n_elems: Int, width: Int
) raises -> Int:
    """`unpack_value` with the width, the element index, and the stream's
    whole footprint checked against the buffer.

    Checking the footprint rather than this element's window is deliberate:
    a buffer that can hold the last element can hold every earlier one, and
    the footprint is the quantity a layout has to have reserved.
    """
    check_bin_width(width)
    check_element_index(index, n_elems)
    if base < 0:
        raise Error("stream base must not be negative")
    if base + writable_footprint_end(n_elems, width) >= len(buf):
        raise Error(
            "packed stream runs past the buffer: the tail pad is missing"
        )
    return unpack_value(buf, base, index, width)


@always_inline
def pack_value(
    mut buf: List[UInt8], base: Int, index: Int, width: Int, value: Int
):
    """Write `value` into element `index` of the stream starting at `base`.

    Read-modify-write of both window bytes, so the bits belonging to
    neighbouring elements are preserved and packing the same element twice is
    idempotent. Unchecked in the same sense as `unpack_value`; in particular
    `value` is masked to `width` bits rather than validated, so a caller that
    has not checked its bin ids against the width silently stores a truncated
    id. `pack_value_checked` is the form that refuses instead.

    The width-8 branch stores the byte outright, so an uncompressed stream
    never touches the byte past its data and two such streams laid out back
    to back stay independent.
    """
    if width == BIN_WIDTH_FALLBACK:
        buf[base + index] = UInt8(value & 255)
        return
    var byte = base + element_byte_offset(index, width)
    var shift = element_bit_shift(index, width)
    var mask = bin_width_mask(width)
    var window = Int(buf[byte]) | (Int(buf[byte + 1]) << BITS_PER_BYTE)
    # Clear this element's bits, then or the new ones in. The xor form of the
    # complement keeps every intermediate inside 16 bits.
    var cleared = window ^ (window & (mask << shift))
    var updated = cleared | ((value & mask) << shift)
    buf[byte] = UInt8(updated & 255)
    buf[byte + 1] = UInt8((updated >> BITS_PER_BYTE) & 255)


def pack_value_checked(
    mut buf: List[UInt8],
    base: Int,
    index: Int,
    n_elems: Int,
    width: Int,
    value: Int,
) raises:
    """`pack_value` with the width, the element index, the window bytes, and
    the value's range all checked.

    The value check is the one that matters: storing a bin id that does not
    fit its width is the single way this module could corrupt a dataset
    without any bounds violation, so the checked path refuses it rather than
    truncating.
    """
    check_bin_width(width)
    check_element_index(index, n_elems)
    if base < 0:
        raise Error("stream base must not be negative")
    if value < 0 or value >= bins_representable(width):
        raise Error("bin id does not fit the width it is being packed at")
    if base + writable_footprint_end(n_elems, width) >= len(buf):
        raise Error(
            "packed stream runs past the buffer: the tail pad is missing"
        )
    pack_value(buf, base, index, width, value)


# --- Stream drivers -------------------------------------------------------


def pack_column(
    mut buf: List[UInt8],
    base: Int,
    src: List[UInt8],
    src_offset: Int,
    n_elems: Int,
    width: Int,
) raises:
    """Pack `n_elems` `UInt8` bins from `src[src_offset ...]` into the stream
    at `buf[base ...]`.

    One writer, ascending, which is what the overlap rule requires. Every
    element is range-checked against the width before it is written, so a
    plan whose width is too narrow for the data fails here rather than
    producing a matrix that trains on truncated bin ids.
    """
    check_bin_width(width)
    if n_elems < 0:
        raise Error("element count must not be negative")
    if src_offset < 0 or src_offset + n_elems > len(src):
        raise Error("source range is outside the bin matrix")
    if base < 0 or base + packed_stream_bytes(n_elems, width) > len(buf):
        raise Error("packed column does not fit the destination buffer")
    var limit = bins_representable(width)
    for i in range(n_elems):
        var value = Int(src[src_offset + i])
        if value >= limit:
            raise Error(
                "bin id does not fit the width it is being packed at"
            )
        pack_value(buf, base, i, width, value)


def unpack_column(
    buf: List[UInt8],
    base: Int,
    n_elems: Int,
    width: Int,
) raises -> List[UInt8]:
    """The inverse of `pack_column`: `n_elems` bins recovered as `UInt8`.

    The reference decoder. Its result is what a device kernel reading the
    same bytes with the same width must produce for every element, which is
    the property `column_roundtrips` checks and the property the integration
    seam in the handoff is written against.
    """
    check_bin_width(width)
    if n_elems < 0:
        raise Error("element count must not be negative")
    if base < 0 or base + packed_stream_bytes(n_elems, width) > len(buf):
        raise Error("packed column does not fit the source buffer")
    var out = List[UInt8](capacity=n_elems)
    for i in range(n_elems):
        out.append(UInt8(unpack_value(buf, base, i, width)))
    return out^


def column_roundtrips(
    src: List[UInt8], src_offset: Int, n_elems: Int, width: Int
) raises -> Bool:
    """Whether packing and unpacking `n_elems` bins at `width` returns them
    unchanged.

    A verification helper for callers and for a future test lane, not a
    packing path: it allocates its own buffer and throws it away.
    """
    var buf = List[UInt8](capacity=packed_stream_bytes(n_elems, width))
    buf.resize(packed_stream_bytes(n_elems, width), 0)
    pack_column(buf, 0, src, src_offset, n_elems, width)
    var back = unpack_column(buf, 0, n_elems, width)
    if len(back) != n_elems:
        return False
    for i in range(n_elems):
        if back[i] != src[src_offset + i]:
            return False
    return True


# --- Decode cost ----------------------------------------------------------
#
# What a kernel spends per element to turn packed bytes back into a bin id.
# Counted in units a profiler can weigh, never collapsed into a scalar
# "overhead" that would let a policy pick a width without measuring.


@fieldwise_init
struct DecodeCost(Copyable, Movable):
    """Per-element decode work at one width.

    `byte_loads` are loads from the packed stream; at width 8 the second
    window byte is provably never needed, so the count drops to one. `alu_ops`
    counts the shift, or, shift, and mask that compose and extract the
    window. Both are static properties of the width, not of the data.
    """

    var byte_loads: Int
    var alu_ops: Int

    def is_free(self) -> Bool:
        """Whether decoding costs no more than reading an ordinary `UInt8`
        bin, which is true only at the fallback width."""
        return self.byte_loads == 1 and self.alu_ops == 0


def decode_cost(width: Int) raises -> DecodeCost:
    """The per-element decode cost at `width`.

    At width 8 every shift is zero and the mask is the identity, so a kernel
    specialized on the width compiles the decode away entirely and the load
    is the same one the current kernel already does. At every other width the
    element may straddle, so both window bytes are loaded unconditionally
    (branching on `element_straddles` would cost more than the load it saves
    and would diverge inside a warp) and four ALU ops extract the value.
    """
    check_bin_width(width)
    if width == BIN_WIDTH_FALLBACK:
        return DecodeCost(1, 0)
    return DecodeCost(2, 4)


# --- Invariant checks -----------------------------------------------------
#
# The properties the rest of the system is entitled to assume. They are
# functions rather than comments so that a caller, a test lane, or a future
# reader can run them instead of trusting the docstring above.


def assert_byte_identity_invariant(n_elems: Int) raises:
    """At the fallback width the packed stream is the uncompressed column.

    Checks the addressing, not a buffer: element `i` must live at byte `i`
    with shift 0 and must never straddle, and the stream must need no pad.
    This is what lets the packed path and the `UInt8` path share one kernel
    and one set of formulas, and what makes `BinLayoutPlan`'s passthrough
    plan the existing `bins` buffer rather than a copy of it.
    """
    if packed_stream_bytes(n_elems, BIN_WIDTH_FALLBACK) != n_elems:
        raise Error("width 8 must need exactly one byte per element")
    for i in range(n_elems):
        if element_byte_offset(i, BIN_WIDTH_FALLBACK) != i:
            raise Error("width 8 must address element i at byte i")
        if element_bit_shift(i, BIN_WIDTH_FALLBACK) != 0:
            raise Error("width 8 must have a zero shift")
        if element_straddles(i, BIN_WIDTH_FALLBACK):
            raise Error("width 8 must never straddle a byte boundary")


def assert_straddle_invariant(n_elems: Int, width: Int) raises:
    """No element spans more than the two bytes the decode window reads.

    The bound `shift + width <= 15` is what makes the decode a fixed
    instruction sequence. If it ever failed, `unpack_value` would silently
    return an element missing its high bits.
    """
    check_bin_width(width)
    for i in range(n_elems):
        if element_bit_shift(i, width) + width > 2 * BITS_PER_BYTE:
            raise Error("packed element spans more than two bytes")


def assert_window_inside(n_elems: Int, width: Int) raises:
    """The last element's decode window stays inside `packed_stream_bytes`.

    This is the tail pad's whole justification, so it is checked against the
    same formulas the allocator uses rather than against a constant. Below
    width 8 the window is two bytes and the pad is what makes the second one
    legal; at width 8 the window is one byte and there is no pad to check.
    """
    if n_elems <= 0:
        return
    var last = element_byte_offset(n_elems - 1, width)
    var window_end = last
    if width != BIN_WIDTH_FALLBACK:
        window_end = last + 1
    # The footprint is an upper bound, exact at width 8 and occasionally one
    # byte loose below it (two 7-bit values fill 14 bits, so the last one
    # ends inside byte 1 while the data spans two bytes and the footprint
    # allows for a third). Loose is the safe direction: it over-reserves.
    if window_end > writable_footprint_end(n_elems, width):
        raise Error("the last element's window runs past the footprint")
    if window_end >= packed_stream_bytes(n_elems, width):
        raise Error("packed stream is missing its tail pad")
