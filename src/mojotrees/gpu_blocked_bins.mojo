"""The `[block][row][G bytes]` bin layout: its geometry, and the device pass
that builds it.

`gpu_binned_layout.mojo` describes a whole family of layouts and refuses to
pick one. This module implements exactly **one** member of that family on the
device, the one the histogram row loop can read: `G` consecutive features per
block, row-major inside the block, one `UInt8` per cell. It is the corner
that file calls `LAYOUT_FEATURE_BLOCKED_U8`, and `blocked_plan` below returns
the very `BinLayoutPlan` this module's arithmetic implements, so the layout
that ships and the layout that is priced are the same object rather than two
descriptions that have to be kept in step by hand.

Who owns what in this file
--------------------------
The lane split moved mid-flight, so the boundary is written down rather than
inferred. **The reader owns the address contract; a layout lane owns the
bytes.** Concretely:

- **The contract, which the histogram kernel evaluates and must not change
  without the reader changing with it:** `blocked_column_base`,
  `blocked_offset`, `blocked_block_stride`, `blocked_padded_features`,
  `blocked_n_blocks`, `blocked_bytes`, `blocked_group_valid`,
  `check_blocked_group_matches`, `blocked_is_identity`,
  `BLOCKED_STRIDE_NONE`. `gpu_active_rows._hist_accumulate_rows` computes the
  first of these inline, because a kernel cannot walk a host `List`, so a
  change to it is a change to two files at once and
  `check_blocked_matches_plan` is what catches a drift between them.
- **The producer, which a layout lane may move, rewrite, or replace outright
  provided the contract above still holds:** `_blocked_relayout_kernel`,
  `enqueue_blocked_relayout`, `blocked_relayout_host`, `blocked_plan`,
  `check_blocked_matches_plan`, `blocked_roundtrips`, `blocked_decode_host`.
  Nothing in the reader depends on *how* the buffer comes to hold these
  bytes -- host pack, device transform, or a producer that never materializes
  the feature-major copy at all.

`docs/design/BLOCKED_BINS.md` section 1 states the same contract in the form a
layout lane implements against, with the six requirements in the order they
bind.

Why this layout and not compression
-----------------------------------
Per (row, feature) the histogram row loop moves one bin byte, one Int32 row
index, and eight bytes of quantized gradient pair. The bin is one byte of
thirteen, so narrowing it to four bits removes under four percent of the
traffic. What the byte costs is not its width: it is that
`bins[feature * n_rows + row]` puts the `G` bytes one thread wants for one
row `n_rows` apart, so a thread that owns `G` feature slots issues `G`
independent scattered loads and, wherever the node's rows are sparse enough
that each load is its own memory sector, pays `G` sectors to move `G` bytes.
Blocked by `G`, those same `G` bytes are adjacent and cost one.

That is the entire mechanism, and it is worth being precise about where it
pays and where it does not, because the two regimes point opposite ways:

- **At the root**, and at any node holding most of the matrix, the active
  rows are the identity permutation or close to it, so a feature-major
  column is read *contiguously* by consecutive threads and touches exactly
  its own bytes. Blocked reads exactly the same bytes, in the same total
  count, from a different arrangement. **The layout is neutral at the root.**
  It is not a small win there; it is no win there.
- **At a deep node**, `count` rows out of `n_rows` are read in ascending
  order with average stride `n_rows / count`, so past
  `count < n_rows / sector` every read is its own sector whatever it wanted
  from it. Feature-major spends `G` such sectors per row; blocked spends
  one. **The whole of the win is here**, in the size classes that dominate a
  real tree by count.

This has a consequence for how the change may be measured that is more
important than the change: **an isolated full-width root-sized histogram
benchmark is the one instrument this layout cannot show up on.** The
concurrent CPU campaign retracted a reading this week for the opposite error
-- an isolated kernel that said 3.25-3.40x where the in-tree figure was 2.13x
and fell to 1.20x on small nodes -- and the row-tile floor was a phase-level
win that measured 22 to 36 percent slower in a real fit. Here the bias runs
the other way, which is not a reason to trust the isolated number either. The
figure that decides is the in-run one **by node size class**, and the
prediction registered here before anything was measured is: **root and large
classes flat, medium upward, small and tiny carrying the whole effect.**

The condition the win depends on, stated as a refusal
----------------------------------------------------
The `G` bytes a thread pulls are only paid for once if the threadgroup
**consumes all `G` of them**. A threadgroup consumes `feature_group` slots
(`GpuActiveRows.set_feature_group`), so:

    G  must equal  feature_group

and `check_blocked_group_matches` refuses any other pairing rather than
letting it run. The two arrangements of a mismatch are both bad and neither
is an error a measurement would catch:

- `G > feature_group`: the block pulls `G` bytes per row and uses fewer.
  Sector count per row is 1 either way, so it is **not slower** than
  feature-major -- it simply buys nothing while costing `G/feature_group`
  times the resident bytes. A benchmark of this pairing measures a null and
  attributes it to the layout.
- `G < feature_group`: the block's slots straddle blocks, so it issues
  `feature_group / G` scattered loads instead of one. Part of the win, and
  no way to tell from the outside which part.

**So at the shipping default this layout is a no-op, and that is the single
most important thing in this file.** `gpu_tiling.free_feature_group` returns
1 at `bin_cap = 256`, because three Int32 planes of 256 cells per feature slot
is 3 KiB and a second slot would double the threadgroup footprint. Every
headline shape this project benchmarks (1,000,000 x 50 at 255 bins) runs at
`feature_group = 1`, where `G = 1` **is** the feature-major layout, byte for
byte and address for address. Reaching the win therefore requires
`set_feature_group(2)` or wider, which trades resident blocks for row-side
traffic and is UNMEASURED on every device this project runs on
(`free_feature_group` says so in as many words). That trade is not this
module's to make: the group default lives in `gpu_tiling.mojo` and
`histogram_gpu.mojo`, and nothing here changes it.

The consequence for the experiment is that the layout arm must be held
against feature-major **at the same group**, never against the shipping
default, or the measurement is of the group change and the layout is along
for the ride. See `docs/design/BLOCKED_BINS.md` for the three arms.

Feature subsampling
-------------------
Correctness is unconditional under `colsample`: the address of a cell is
computed from the feature id, so an arbitrary active subset reads the right
bytes. The *win* is not unconditional. A threadgroup's slots are
`feat_ids[slot0 .. slot0 + G)`, and under subsampling those ids need not be
the `G` lanes of one block -- active features 0, 5, 9, 12 at `G = 4` sit in
four different blocks at four different lanes, so the block issues four
scattered loads exactly as feature-major would while still occupying `G`
times the bytes. `gpu_binned_layout.subsample_waste` is the quantity, and
`blocked_alignment_fraction` below reports the share of a given active set
that does land block-aligned, so a benchmark can say which regime it ran in
rather than assuming.

Exactness
---------
**This layout cannot change a histogram, by construction, and the argument is
one sentence: it moves bytes and touches nothing else.** For every
`(feature, row)` the relayout writes `dst[blocked_offset(f, r)] =
bins[f * n_rows + r]`, and the row loop reads `blocked_offset(f, r)` where it
used to read `f * n_rows + r`. The same bin byte therefore selects the same
shared cell for the same row, the same three quantized values are added to
it, and the set of `(row, feature)` visits is untouched -- the layout is not
visible to the row walk, the tiling, the feature slots, or the flush.
Accumulation is fixed-point Int32 and integer addition is associative and
commutative, so even the *order* of the adds, which does not change either,
could not have moved a bit. This is the same structural argument
`_range_hist_atomic_kernel` makes for `GROUP` and `BIN_CAP` and
`_hist_rows_step` makes for the row unroll, and it is stronger than theirs:
those reorder adds, this one does not even do that.

`check_blocked_matches_plan` and `blocked_relayout_host` are the two
executable forms of it. The first proves the closed-form address this
module's kernel computes is the address `BinLayoutPlan.byte_of` computes for
the same plan, cell by cell; the second reproduces the device pass on the
host so a test can compare a decoded blocked buffer against the dense matrix
without a GPU.

What the transform costs, and against what
------------------------------------------
One kernel launch per fit, not per tree and not per node. It reads
`n_rows * n_features` bytes and writes `n_rows * padded_features` bytes, both
streamed. At 1,000,000 x 50 with `G = 4` that is 50 MB read and 52 MB
written: **a derived bound** of about 1.3 milliseconds against the 75-85 GB/s
device copy rate this project **measured** on an Apple M4 on 2026-08-15,
which is under 0.06 percent of a 2.58 second fit. It amortizes over every
node of every tree -- roughly 6,100 node histograms in a 100-round fit -- so
the per-node share is not a quantity worth carrying in a cost model.

The cost that is *not* negligible is residency. The blocked buffer does not
replace the feature-major one: `_flag_scan_kernel`, `_scatter_kernel`,
`gpu_predict`, `gpu_sparse` and the host binning path all still index
`bins[f * n_rows + r]`, and converting them is a much larger change with a
much worse half-applied state. So both buffers stay resident and the device
pays `n_rows * (n_features + padded_features)` bytes for bins instead of
`n_rows * n_features`: 102 MB instead of 50 MB at the reference shape. On a
device where the binned matrix is a large share of the working set that is
the term that decides, and it is the reason `set_blocked_layout` allocates on
request rather than at construction.

Rejected: doing the transform on the host
-----------------------------------------
`gpu_binned_layout.pack_binned_matrix` already produces this buffer on the
host, and using it would need no new kernel. It was rejected because it
charges `n_rows * n_features` host writes and a second upload of the whole
matrix, where the device pass charges neither: the feature-major buffer is
already on the device, so the transform is device-local traffic at device
bandwidth. The host path stays as the reference implementation
(`blocked_relayout_host`), which is what a test wants and what a
non-accelerator build can still run.

Rejected: a shared-memory tiled transpose
-----------------------------------------
The kernel below reads `G` bytes per thread from `G` different columns and
writes them contiguously, which coalesces the writes and leaves the reads as
`G` separate coalesced streams. A shared-memory tile would make both sides
contiguous. It was not built because the pass is once per fit and already
under a millisecond at the reference shape by the bound above: a two-times
improvement on it is invisible next to the arm spread on this machine, and it
would be a second piece of untested address arithmetic guarding the same
invariant. If the transform ever shows up in a profile, the tiled transpose
is the change, and it changes no address this module publishes.
"""

from std.gpu import block_idx, global_idx
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from .binning import BinnedMatrix
from .gpu_bin_packing import BIN_WIDTH_FALLBACK
from .gpu_binned_layout import (
    BLOCK_ALIGN_BYTES,
    BinLayoutPlan,
    LAYOUT_MAX_INDEX,
    check_layout_support,
    plan_feature_blocked,
)
from .gpu_tiling import is_feature_group_width


# The row stride of the feature-major layout, and therefore the value of
# `bin_row_stride` that means "no blocking". It is 1 because a feature-major
# column advances one byte per row, and the blocked reader's address formula
# collapses to the feature-major one at this value rather than branching
# around it. See `blocked_column_base`.
comptime BLOCKED_STRIDE_NONE = 1


def blocked_group_valid(group: Int) -> Bool:
    """Whether `group` is a `G` this module will lay a matrix out at.

    Exactly the feature-group ladder (1, 2, 4, 8, 16), and for a reason that
    is not tidiness: `G` must equal `GpuActiveRows.feature_group` for the
    layout to buy anything (see the module docstring), and that field is a
    rung of the ladder because each rung is a separate kernel instantiation.
    A `G` off the ladder could be laid out and could never be consumed.
    """
    return is_feature_group_width(group)


def check_blocked_group(group: Int) raises:
    """Raising form of `blocked_group_valid`."""
    if not blocked_group_valid(group):
        raise Error(
            "a blocked bin layout's G must be a rung of the feature-group"
            " ladder: 1, 2, 4, 8, or 16"
        )


def check_blocked_group_matches(group: Int, feature_group: Int) raises:
    """Refuse a `G` that does not match the histogram's feature group.

    The whole mechanism is that a threadgroup consumes every one of the `G`
    bytes it pulls for a row, and a threadgroup consumes `feature_group`
    slots. A mismatch is not a smaller win; it is a different experiment
    wearing this one's label, in the direction the module docstring works
    out. Refused here rather than reported, because both directions of
    mismatch produce a *correct* histogram and would be found by nothing.
    """
    check_blocked_group(group)
    if group != feature_group:
        raise Error(
            "a blocked bin layout's G must equal the histogram feature group;"
            " a mismatch reads correct bins and measures the wrong thing"
        )


def blocked_padded_features(n_features: Int, group: Int) raises -> Int:
    """`n_features` rounded up to a whole number of blocks.

    A short final block would break the one closed-form address this module
    publishes: `blocked_column_base` divides by `G` to find a block and takes
    the remainder as a lane, which is only the plan's `lane_of` while every
    block is `G` wide. Padding is what keeps that formula a formula rather
    than a formula plus a special case for the last block, and the special
    case is the kind that is right on every shape a test picks and wrong on
    the one shape a user has.

    The pad columns are written (with zeros, by the relayout) and never read:
    a feature id reaching the row loop comes out of `feat_ids`, whose values
    are below `n_features` by `enqueue_range_histogram`'s own bound. Their
    whole cost is `(G - 1) * n_rows` bytes in the worst case, 3 MB at
    1,000,000 rows and `G = 4`.
    """
    check_blocked_group(group)
    if n_features < 1:
        raise Error("a blocked layout needs at least one feature")
    return ((n_features + group - 1) // group) * group


def blocked_n_blocks(n_features: Int, group: Int) raises -> Int:
    """How many `G`-wide blocks the padded feature axis holds."""
    return blocked_padded_features(n_features, group) // group


def blocked_block_stride(n_rows: Int, group: Int) raises -> Int:
    """Bytes from one block's base to the next: `n_rows * G` rounded up to
    `BLOCK_ALIGN_BYTES`.

    The alignment is not decoration and it is not this module's invention: it
    is `BinLayoutPlan`'s, which aligns every block base to 16 bytes so that a
    block's first row never begins mid-vector and a block's base never
    depends on the width of the block before it. Reproduced here in closed
    form because the kernel cannot walk a `List[Int]` of block offsets, and
    `check_blocked_matches_plan` is what proves the two agree rather than a
    comment claiming they do.

    At `G = 1` this is `align_up(n_rows, 16)` and **not** `n_rows`, which
    would make the blocked buffer differ from the feature-major one at the
    degenerate width. That is why `G = 1` is never uploaded: see
    `blocked_is_identity`.
    """
    check_blocked_group(group)
    if n_rows < 1:
        raise Error("a blocked layout needs at least one row")
    var span = n_rows * group
    return (
        (span + BLOCK_ALIGN_BYTES - 1) // BLOCK_ALIGN_BYTES
    ) * BLOCK_ALIGN_BYTES


def blocked_is_identity(group: Int) raises -> Bool:
    """Whether laying out at this `G` would reproduce the feature-major
    buffer, so uploading a second copy of it would buy nothing at all.

    True at `G = 1`, and only there. A one-feature block is one feature's
    column, so the arrangement is the existing one; the addresses are not
    literally identical, because `BinLayoutPlan` pads a one-block-per-feature
    plan at width 8 to 16-byte block bases only when it is not the
    passthrough plan, but the *arrangement* is, and a second buffer holding
    the same bytes in the same order can only cost residency.

    `GpuActiveRows.set_blocked_layout` uses this to refuse `G = 1` outright
    rather than allocate a copy nothing will read differently, which also
    means the flag can never be on while the reader is doing feature-major
    arithmetic.
    """
    check_blocked_group(group)
    return group == 1


def blocked_bytes(n_rows: Int, n_features: Int, group: Int) raises -> Int:
    """Bytes the blocked buffer occupies.

    `(n_blocks - 1)` whole strides plus the last block's own data, which is
    exactly `BinLayoutPlan.total_bytes`: the final block's alignment tail is
    never allocated because nothing addresses past its last row. Checked
    against `Int32.MAX`, the same bound every device byte offset in this
    backend carries.
    """
    check_blocked_group(group)
    var blocks = blocked_n_blocks(n_features, group)
    var stride = blocked_block_stride(n_rows, group)
    if stride > LAYOUT_MAX_INDEX // blocks:
        raise Error(
            "blocked bin layout is larger than an Int32 byte offset holds"
        )
    var total = (blocks - 1) * stride + n_rows * group
    if total > LAYOUT_MAX_INDEX:
        raise Error(
            "blocked bin layout is larger than an Int32 byte offset holds"
        )
    return total


def blocked_column_base(feature: Int, n_rows: Int, group: Int) raises -> Int:
    """The byte a feature's cells start at, before the row term.

        base(f) = (f / G) * block_stride + (f mod G)

    the block's aligned base plus the feature's lane inside it. This is the
    one quantity the histogram row loop hoists out of the row walk, exactly
    as it hoists `feature * n_rows` today, so the blocked reader spends one
    integer division and one remainder **per owned slot** and nothing per
    row.

    At `G = 1` the lane is always zero and the base is `f * block_stride`,
    which is `f * align_up(n_rows, 16)` and not `f * n_rows`. The degenerate
    width is therefore a different buffer from the feature-major one even
    though it is the same arrangement, which is the second reason
    `blocked_is_identity` refuses it.
    """
    check_blocked_group(group)
    if feature < 0:
        raise Error("feature index out of range")
    var stride = blocked_block_stride(n_rows, group)
    return (feature // group) * stride + (feature % group)


def blocked_offset(
    feature: Int, row: Int, n_rows: Int, group: Int
) raises -> Int:
    """The byte holding `(feature, row)`.

        offset(f, r) = base(f) + r * G

    The whole address contract, in one line, and the line the kernel
    evaluates. The row term is `r * G` rather than `r` because the block is
    row-major: one row of a block is `G` adjacent bytes, which is the entire
    point of the layout.
    """
    if row < 0 or row >= n_rows:
        raise Error("row index out of range")
    return blocked_column_base(feature, n_rows, group) + row * group


def blocked_plan(
    n_rows: Int, n_features: Int, n_bins: Int, group: Int
) raises -> BinLayoutPlan:
    """This layout as the `BinLayoutPlan` `gpu_binned_layout.mojo` prices.

    Built over the **padded** feature count, at width 8 throughout, with
    `target_block = G` and promotion on, which is the one plan in that
    family whose blocks are all exactly `G` wide. Returning a real plan
    rather than a description is what lets `layout_node_cost`,
    `layout_build_cost`, `subsample_waste` and `blocks_touched` be applied to
    the layout that actually ships, and what lets `check_blocked_matches_plan`
    compare two independently written pieces of address arithmetic instead of
    one against a comment.
    """
    check_blocked_group(group)
    var padded = blocked_padded_features(n_features, group)
    check_layout_support(n_rows, padded, n_bins)
    var width = List[Int](capacity=padded)
    width.resize(padded, BIN_WIDTH_FALLBACK)
    return plan_feature_blocked(
        n_rows, padded, n_bins, width^, group, True
    )


def check_blocked_matches_plan(
    n_rows: Int, n_features: Int, n_bins: Int, group: Int
) raises:
    """The closed-form address equals the plan's, for every cell of the
    padded matrix.

    The invariant the whole module rests on stated as a check rather than as
    a claim, and it is the check worth having: if the two ever disagree, the
    kernel reads a shifted matrix and trains on cells belonging to other rows
    with no bounds violation anywhere and no error to catch. Quadratic in the
    matrix, so it is for tests and small shapes, which is what the plan's own
    `check_passthrough_offsets` is for as well.
    """
    var plan = blocked_plan(n_rows, n_features, n_bins, group)
    var padded = blocked_padded_features(n_features, group)
    if plan.n_features != padded:
        raise Error("blocked plan does not cover the padded feature axis")
    if plan.uniform_block_size() != group:
        raise Error("blocked plan's blocks are not all G wide")
    if plan.bytes() != blocked_bytes(n_rows, n_features, group):
        raise Error("blocked plan's size does not match the closed form")
    for f in range(padded):
        for r in range(n_rows):
            if plan.byte_of(f, r) != blocked_offset(f, r, n_rows, group):
                raise Error(
                    "blocked address arithmetic disagrees with the layout"
                    " plan it claims to implement"
                )


def blocked_alignment_fraction(
    active: List[Int], n_features: Int, group: Int
) raises -> Float64:
    """The share of `active`'s features that sit in a **fully** active block.

    One over the bin-traffic waste `gpu_binned_layout.subsample_waste`
    reports, restated in the unit a benchmark wants: 1.0 means every block
    this launch touches is consumed whole, so the layout is being asked the
    question it was built for; 0.25 at `G = 4` means every touched block
    contributes one feature and the launch is paying four bytes per useful
    one while gathering exactly as many sectors as feature-major would.

    Reported, never thresholded on. A run below 1.0 is not wrong, it is a
    different regime, and a result taken in it that does not say so is the
    error this project has now made in four separate places.
    """
    check_blocked_group(group)
    var blocks = blocked_n_blocks(n_features, group)
    var live = List[Int](capacity=blocks)
    live.resize(blocks, 0)
    for i in range(len(active)):
        var f = active[i]
        if f < 0 or f >= n_features:
            raise Error("active feature index out of range")
        live[f // group] += 1
    if len(active) == 0:
        return 0.0
    var whole = 0
    for b in range(blocks):
        if live[b] == group:
            whole += live[b]
    return Float64(whole) / Float64(len(active))


def blocked_relayout_host(
    data: BinnedMatrix, group: Int
) raises -> List[UInt8]:
    """The device pass, on the host, byte for byte.

    The reference implementation and the thing a test compares against. It is
    not on the training path: `GpuActiveRows` builds the buffer with the
    kernel below out of the feature-major copy already on the device, which
    costs no host work and no second upload. This exists so that the address
    arithmetic can be checked without an accelerator, which is the half of CI
    an Apple M4 structurally cannot run.

    Pad lanes are written as zero rather than left as whatever the
    allocation held, so a buffer built here and a buffer built by the kernel
    are comparable everywhere and not only on the cells a feature id can
    name.
    """
    check_blocked_group(group)
    if len(data.bins) != data.n_rows * data.n_features:
        raise Error("binned matrix size must equal n_rows * n_features")
    var total = blocked_bytes(data.n_rows, data.n_features, group)
    var out = List[UInt8](capacity=total)
    out.resize(total, 0)
    var src = data.bins.unsafe_ptr()
    var blocks = blocked_n_blocks(data.n_features, group)
    var stride = blocked_block_stride(data.n_rows, group)
    for b in range(blocks):
        var base = b * stride
        for r in range(data.n_rows):
            var at = base + r * group
            for lane in range(group):
                var f = b * group + lane
                var v = UInt8(0)
                if f < data.n_features:
                    v = src.unsafe_load(f * data.n_rows + r)
                out[at + lane] = v
    return out^


def blocked_decode_host(
    buf: List[UInt8], feature: Int, row: Int, n_rows: Int, group: Int
) raises -> Int:
    """One cell out of a blocked buffer, addressed the way the kernel
    addresses it. The host side of the round trip `matches_dense` performs
    for the packed family."""
    var at = blocked_offset(feature, row, n_rows, group)
    if at < 0 or at >= len(buf):
        raise Error("blocked offset outside the buffer")
    return Int(buf[at])


def blocked_roundtrips(data: BinnedMatrix, group: Int) raises -> Bool:
    """Whether the blocked buffer decodes to the dense matrix, cell for cell.

    The exactness argument as a function. If this is ever false, a histogram
    built from the blocked buffer is a histogram of different data, and no
    tolerance anywhere would notice: the bins would be legal bin ids
    belonging to other rows.
    """
    var buf = blocked_relayout_host(data, group)
    for f in range(data.n_features):
        for r in range(data.n_rows):
            if (
                blocked_decode_host(buf, f, r, data.n_rows, group)
                != data.bin_at(r, f)
            ):
                return False
    return True


# --- The device pass ------------------------------------------------------


def _blocked_relayout_kernel(
    src: MutPointer[UInt8, MutAnyOrigin],
    dst: MutPointer[UInt8, MutAnyOrigin],
    n_rows: Int32,
    n_features: Int32,
    group: Int32,
    block_stride: Int32,
):
    """Write one block's rows of the blocked buffer from the feature-major
    one.

    `grid.x` covers the rows and `grid.y` is the block, so one thread owns
    one `(block, row)` pair and writes the `G` adjacent bytes that pair
    occupies. That is deliberate on both sides of the copy: the writes of a
    threadgroup cover one contiguous `threads * G` byte span, and the `G`
    reads are `G` separate streams each of which is contiguous across
    consecutive threads. The alternative -- one thread per destination byte
    -- makes the writes contiguous per thread but scatters each stream's
    reads by `n_rows`, which is the access pattern this whole layout exists
    to remove.

    Pad lanes past `n_features` are written as zero rather than skipped, so
    the buffer has no uninitialized bytes and a host-built copy and a
    device-built copy are comparable everywhere. Nothing reads them: a
    feature id reaching the histogram row loop comes from `feat_ids` and is
    below `n_features`.
    """
    var nr = Int(n_rows)
    var r = Int(global_idx.x)
    if r >= nr:
        return
    var nf = Int(n_features)
    var g = Int(group)
    var b = Int(block_idx.y)
    var at = b * Int(block_stride) + r * g
    var f0 = b * g
    for lane in range(g):
        var f = f0 + lane
        var v = UInt8(0)
        if f < nf:
            v = src[unsafe_offset = f * nr + r][0]
        dst[unsafe_offset = at + lane] = v


def enqueue_blocked_relayout[
    src_origin: MutOrigin,
    dst_origin: MutOrigin, //
](
    ctx: DeviceContext,
    src: MutPointer[UInt8, src_origin],
    dst: MutPointer[UInt8, dst_origin],
    n_rows: Int,
    n_features: Int,
    group: Int,
    threads: Int,
) raises:
    """Build the blocked buffer on the device from the feature-major one.

    Once per fit. `src` is the `bins` pointer every histogram launch already
    receives and `dst` is the buffer `GpuActiveRows` allocated for the
    layout; they are different allocations, which is required -- a launch may
    not be handed the same buffer twice.

    Enqueued, not synchronized: it is ordered before the histogram that reads
    it by the queue, exactly as `_ensure_quantized`'s pass is ordered before
    the row loop that gathers from it.
    """
    comptime if not has_accelerator():
        raise Error(
            "the blocked bin layout needs an accelerator; this build has none"
        )
    else:
        check_blocked_group(group)
        if blocked_is_identity(group):
            raise Error(
                "a blocked layout at G = 1 is the feature-major layout; do"
                " not upload a second copy of it"
            )
        if threads < 1:
            raise Error("a relayout launch needs a positive block width")
        var blocks = blocked_n_blocks(n_features, group)
        var stride = blocked_block_stride(n_rows, group)
        # Sized so the whole buffer is written even though it is bounded by
        # `blocked_bytes`, which stops one alignment tail short of
        # `blocks * stride`: the last block's threads write only its rows,
        # and its tail is the part that is never allocated.
        var row_blocks = (n_rows + threads - 1) // threads
        ctx.enqueue_function[_blocked_relayout_kernel](
            src,
            dst,
            Int32(n_rows),
            Int32(n_features),
            Int32(group),
            Int32(stride),
            grid_dim=(row_blocks, blocks),
            block_dim=threads,
        )
