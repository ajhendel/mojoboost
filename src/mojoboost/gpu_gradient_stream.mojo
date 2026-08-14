"""Row-selection metadata and single-pass gradient staging for a GPU round.

This module holds the two pieces of the start of a boosting round that are
neither the objective itself (gpu_objectives_native.mojo) nor the histogram
(histogram_gpu.mojo, gpu_active_rows.mojo), and that today force a round
onto a slower path than the hardware needs.

`GpuRowSelection` is the device image of a row sample. A sample is two
arrays, the selected row ids and their compensation multipliers, and the
GPU trainer currently has no device image of either, so `device_gradients`
in train_gpu.mojo refuses the device objective path whenever bagging or
GOSS is configured and the whole round falls back to host-side derivatives
plus a `2 * n_rows` Float32 upload. Nothing about a sample requires that.
The gradients still come from `GpuObjectiveState`, over every row, exactly
as they do without sampling; the sample only decides which rows the root
range covers (already device side, `begin_tree(bag)`) and which rows carry
a multiplier. This struct supplies the second half: the row ids and the
multipliers as device buffers, and one kernel that applies the multipliers
in place, in the same operand order as `apply_goss_scaling` in goss.mojo,
so the gradients the histogram reads already carry the compensation.

`GpuRowSelection` also owns the device-side GOSS ranking score,
`|grad * hess|`, so a trainer that wants to sample from device-resident
gradients downloads one Float32 plane instead of needing host-side
derivatives at all. That download is the only reason the score exists;
the selection rule itself (quickselect threshold, forward pass, counter
stream) stays in goss.mojo, unchanged and host side, because it is a
ranking over the whole row set and a partial-order pass, not a per-row map.

`HostGradientStage` is the other end. Where the gradients genuinely
originate on the host, which is the custom-objective contract and nothing
else, `GpuHistogramBuilder.stage_gradients` walks the two Float64 lists
three times: once for the gradient magnitude sum, once for the hessian
magnitude sum, and once to convert into the pinned Float32 staging
buffers. Those are the same three reads of the same two arrays. This
struct fuses them into one pass that accumulates both magnitude sums in
Float64 while it converts, which is where the "streaming" design in the
handoff actually pays: not on the device, where the planes have to be
materialized anyway, but on the host-origin path, where a value is
touched three times for no reason. The sums accumulate in index order over
the Float64 inputs, which is exactly what `_fixed_scale` does, so the
scale that comes out of `device_fixed_scale` is bit-identical to the one
`stage_gradients` computes today.

What is deliberately not here
-----------------------------
No objective arms. A row-restricted gradient kernel would have to repeat
the per-objective derivative chain that `_grad_hess_kernel` already owns,
and a second definition of the objectives is worse than the pass it would
save. Gradients are produced for every row by the existing kernel and the
sample is applied afterwards, which is also what the host path does
(`_fill_grad_hess` over all rows, then `apply_goss_scaling` over the
sample), so the two stay comparable term by term.

No chunked upload. True overlap of conversion with transfer needs a copy
into a sub-range of a device buffer, and the copy entry points reached
from here take whole buffers. The single fused pass is the part that does
not need it; the handoff names the sub-range copy as the follow-up.

Precision
---------
The device carries gradients, hessians, and multipliers as Float32. The
host path multiplies Float64 gradients by a Float64 multiplier and rounds
once, at staging; this path rounds the gradient and the multiplier first
and multiplies in Float32. The products therefore agree to Float32, not
bit-exactly, which is the same trade every other device stage makes. It
matters in one specific place and the handoff says so: the GOSS *ranking*
score is a function of the gradients, so a Float32 score can order two
near-equal rows differently from the Float64 one and put a different row
on the wrong side of the top-k threshold. Row bagging has no such
exposure, because a bag is drawn from a counter stream keyed by
(seed, bag index, row) and never looks at a gradient.
"""

from std.math import isfinite
from std.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from .goss import GossSelection
from .gpu_objectives_native import GradMagnitudes
from .gpu_tiling import derive_block_threads, query_device_caps


def _row_scale_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    scale: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    count: Int32,
):
    """Apply one compensation multiplier per selected row, in place.

    One thread per entry of the selection, not per row of the dataset: a
    GOSS round selects a fraction of the rows and this kernel touches
    exactly those. `apply_goss_scaling` skips an entry whose multiplier is
    1.0 and so does this, which is a bandwidth choice rather than a
    numerical one, since multiplying a finite Float32 by exactly 1.0 is the
    identity. The high-gradient rows GOSS keeps are all at 1.0, so the skip
    covers most of the selection.

    Both planes are scaled by the same multiplier and in the same operand
    order the host does (`grad[r] * s`, `hess[r] * s`), so a row's
    compensated derivatives here are the Float32 image of the host's.
    """
    var j = global_idx.x
    if j >= Int(count):
        return
    var s = scale[unsafe_offset=j][0]
    if s == 1.0:
        return
    var r = Int(rows[unsafe_offset=j][0])
    grad[unsafe_offset=r] = grad[unsafe_offset=r][0] * s
    hess[unsafe_offset=r] = hess[unsafe_offset=r][0] * s


def _importance_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    imp: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
):
    """LightGBM's GOSS ranking score `|grad * hess|`, one row per thread.

    The same expression as `goss_importance`, over every row, so a host
    sampler fed from here ranks the rows it would have ranked. Sample
    weights are already folded into both derivatives by the gradient
    kernel, so a zero-weight row scores zero and ranks last, which is the
    documented behavior of the host score.
    """
    var r = global_idx.x
    if r >= Int(n_rows):
        return
    imp[unsafe_offset=r] = abs(
        grad[unsafe_offset=r][0] * hess[unsafe_offset=r][0]
    )


struct GpuRowSelection(Movable):
    """One round's row sample, device side.

    Construct once per training session with the capacity the sampler can
    ask for (`max_selected`, in practice `n_rows`, since a bag is Binomial
    and a GOSS selection can exceed `top_k + other_k` on importance ties).
    A session that never samples constructs with `max_selected = 0` and
    pays one element per placeholder buffer.

    Per round: `select_all` for an unsampled round, or `set_selection`
    (or `from_goss`) with the sampler's output, then `apply_compensation`
    after the gradients are filled and before their magnitudes are
    reduced. That order is the host path's order, where `goss_round`
    scales the gradients and `upload_gradients` then derives the
    fixed-point scale from the scaled values.

    The selection is a device image of a host decision, not a device
    decision. Bags come from bagging.mojo and GOSS selections from
    goss.mojo, both unchanged, so both backends still sample identical
    rows from identical seeds.
    """

    var rows_dev: DeviceBuffer[DType.int32]
    """Selected row ids, ascending, in the sampler's order. The first
    `n_selected` entries are live."""
    var scale_dev: DeviceBuffer[DType.float32]
    """Compensation multiplier per entry of `rows_dev`, 1.0 for a row the
    sampler kept outright."""
    var imp_dev: DeviceBuffer[DType.float32]
    """`|grad * hess|` per row, or a one-element placeholder when the
    session did not ask for the ranking score."""
    var stage_rows: HostBuffer[DType.int32]
    var stage_scale: HostBuffer[DType.float32]
    var host_imp: HostBuffer[DType.float32]
    var n_rows: Int
    var max_selected: Int
    var n_selected: Int
    var block_threads: Int
    var has_importance: Bool
    var has_compensation: Bool
    """Whether this round's selection carried multipliers at all. A bag
    does not, so a bagged round never launches the scaling kernel."""

    def __init__(
        out self,
        ctx: DeviceContext,
        n_rows: Int,
        max_selected: Int = 0,
        with_importance: Bool = False,
    ) raises:
        """Allocate the selection buffers once for the whole session.

        `max_selected` of 0 means this session never samples, and both
        selection buffers become one-element placeholders (zero-length
        device buffers are not portable). `with_importance` allocates the
        `n_rows` ranking plane and its pinned readback buffer, which only
        a GOSS session needs.
        """
        if n_rows < 1:
            raise Error("row selection requires at least one row")
        if max_selected < 0:
            raise Error("max_selected must be nonnegative")
        if max_selected > n_rows:
            raise Error("max_selected cannot exceed n_rows")

        self.n_rows = n_rows
        self.max_selected = max_selected
        self.n_selected = 0
        self.has_importance = with_importance
        self.has_compensation = False
        self.block_threads = derive_block_threads(query_device_caps(ctx))

        var cap = max_selected if max_selected > 0 else 1
        var imp_cap = n_rows if with_importance else 1
        self.rows_dev = ctx.enqueue_create_buffer[DType.int32](cap)
        self.scale_dev = ctx.enqueue_create_buffer[DType.float32](cap)
        self.imp_dev = ctx.enqueue_create_buffer[DType.float32](imp_cap)
        self.stage_rows = ctx.enqueue_create_host_buffer[DType.int32](cap)
        self.stage_scale = ctx.enqueue_create_host_buffer[DType.float32](cap)
        self.host_imp = ctx.enqueue_create_host_buffer[DType.float32](imp_cap)

        # The copies below take whole buffers, so the tail beyond a
        # round's selection is transferred whatever it holds. No kernel
        # reads past `n_selected`, so this is hygiene rather than
        # correctness, and it is the same choice gpu_active_rows.mojo
        # makes for its staging buffer.
        var dst_rows = self.stage_rows.unsafe_ptr()
        var dst_scale = self.stage_scale.unsafe_ptr()
        for i in range(cap):
            dst_rows.unsafe_store(i, Int32(0))
            dst_scale.unsafe_store(i, Float32(1.0))

    def select_all(mut self):
        """Train this round on every row with no compensation. An empty
        selection means the same thing here that an empty bag means
        everywhere else in the library."""
        self.n_selected = 0
        self.has_compensation = False

    def is_all_rows(self) -> Bool:
        return self.n_selected == 0

    def selected(self) -> Int:
        """Rows in the current selection, 0 meaning every row."""
        return self.n_selected

    def set_selection(
        mut self,
        ctx: DeviceContext,
        rows: List[Int],
        scale: List[Float64] = [],
    ) raises:
        """Upload this round's sample.

        `rows` holds the selected row ids in the sampler's order, which is
        the order the root range is seeded in, so the compacted rows track
        the CPU grower's row list index for index. `scale` is either empty
        (no compensation, which is what row bagging wants) or one
        multiplier per entry of `rows`.

        An empty `rows` is `select_all`, so a `GossSelection.all_rows()`
        passed straight through means what it means on the host.
        """
        if len(rows) == 0:
            if len(scale) != 0:
                raise Error(
                    "a compensation multiplier without a selected row has"
                    " nothing to scale"
                )
            self.select_all()
            return
        if len(rows) > self.max_selected:
            raise Error(
                "selection is larger than the capacity this session was"
                " constructed with"
            )
        if len(scale) != 0 and len(scale) != len(rows):
            raise Error(
                "compensation multipliers must be one per selected row"
            )
        for i in range(len(rows)):
            if rows[i] < 0 or rows[i] >= self.n_rows:
                raise Error("selected row index out of range")
        for i in range(len(scale)):
            if not isfinite(scale[i]) or scale[i] < 0.0:
                raise Error(
                    "compensation multipliers must be finite and"
                    " nonnegative"
                )

        # Any copy still reading the staging buffers has to finish before
        # they are overwritten.
        ctx.synchronize()
        var dst_rows = self.stage_rows.unsafe_ptr()
        var dst_scale = self.stage_scale.unsafe_ptr()
        for i in range(len(rows)):
            dst_rows.unsafe_store(i, Int32(rows[i]))
        if len(scale) == 0:
            for i in range(len(rows)):
                dst_scale.unsafe_store(i, Float32(1.0))
        else:
            for i in range(len(scale)):
                dst_scale.unsafe_store(i, Float32(scale[i]))
        ctx.enqueue_copy(dst_buf=self.rows_dev, src_ptr=dst_rows)
        ctx.enqueue_copy(dst_buf=self.scale_dev, src_ptr=dst_scale)
        self.n_selected = len(rows)
        self.has_compensation = len(scale) > 0

    def from_goss(
        mut self, ctx: DeviceContext, selection: GossSelection
    ) raises:
        """`set_selection` from a `GossSelection`, so the sampler's output
        reaches the device without a caller unpacking it. The no-sampling
        selection (`GossSelection.all_rows()`) becomes `select_all`."""
        self.set_selection(ctx, selection.rows, selection.scale)

    def apply_compensation(
        mut self,
        ctx: DeviceContext,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises:
        """Scale the selected rows' gradients and hessians in place.

        A no-op when the selection is every row or when no multipliers
        were supplied, which is the row-bagging case: bagging changes
        which rows a tree sees, never what a row's derivatives are.

        Call after the gradient kernel and before the magnitude reduction.
        The scale the histograms quantize with then describes the values
        the histograms will actually read, which is the invariant the
        fixed-point accumulation depends on for its overflow bound.
        """
        if self.n_selected <= 0 or not self.has_compensation:
            return
        var blocks = (
            self.n_selected + self.block_threads - 1
        ) // self.block_threads
        ctx.enqueue_function[_row_scale_kernel](
            self.rows_dev.unsafe_ptr(),
            self.scale_dev.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            Int32(self.n_selected),
            grid_dim=blocks,
            block_dim=self.block_threads,
        )

    def enqueue_importance(
        mut self,
        ctx: DeviceContext,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises:
        """Fill the ranking plane from the round's uncompensated
        gradients. Does not transfer or synchronize.

        Call before `apply_compensation`: the host sampler ranks rows by
        the derivatives the objective produced, and the multiplier it
        hands back is a consequence of that ranking, not an input to it.
        """
        if not self.has_importance:
            raise Error(
                "this selection was constructed without the ranking plane;"
                " construct with with_importance=True"
            )
        var blocks = (
            self.n_rows + self.block_threads - 1
        ) // self.block_threads
        ctx.enqueue_function[_importance_kernel](
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            self.imp_dev.unsafe_ptr(),
            Int32(self.n_rows),
            grid_dim=blocks,
            block_dim=self.block_threads,
        )

    def download_importance(
        mut self, ctx: DeviceContext
    ) raises -> List[Float64]:
        """The ranking plane, host side, ready for `goss_select`.

        One `n_rows` Float32 transfer and one synchronization, against the
        `2 * n_rows` Float32 upload the host-gradient path pays every
        round, and it replaces a full host-side objective evaluation
        rather than adding to one. The values are the Float32 image of
        `goss_importance`, so a selection drawn from them can differ from
        the CPU sampler's on rows whose scores tie to within Float32; see
        the module docstring.
        """
        if not self.has_importance:
            raise Error(
                "this selection was constructed without the ranking plane;"
                " construct with with_importance=True"
            )
        ctx.enqueue_copy(
            dst_ptr=self.host_imp.unsafe_ptr(), src_buf=self.imp_dev
        )
        ctx.synchronize()
        var src = self.host_imp.unsafe_ptr()
        var out = List[Float64](capacity=self.n_rows)
        for r in range(self.n_rows):
            out.append(Float64(src.unsafe_load(r)))
        return out^


struct HostGradientStage(Movable):
    """Single-pass staging for gradients that genuinely originate on the
    host.

    The custom-objective trainers hand back two `List[Float64]` per round
    and there is no device image of the callback that produced them, so
    those values have to be converted and uploaded. `stage_gradients` in
    histogram_gpu.mojo reads both lists three times to do it. This reads
    them once.

    Not a replacement for the device objective path and not reachable
    from it: a built-in objective produces its derivatives on the device
    and nothing per row is staged at all. This is the fallback's fast
    path, which is why it lives next to the selection rather than inside
    the builder.
    """

    var stage_g: HostBuffer[DType.float32]
    var stage_h: HostBuffer[DType.float32]
    var n_rows: Int
    var staged: Bool

    def __init__(out self, ctx: DeviceContext, n_rows: Int) raises:
        if n_rows < 1:
            raise Error("gradient staging requires at least one row")
        self.n_rows = n_rows
        self.staged = False
        self.stage_g = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        self.stage_h = ctx.enqueue_create_host_buffer[DType.float32](n_rows)

    def stage(
        mut self,
        ctx: DeviceContext,
        grad: List[Float64],
        hess: List[Float64],
    ) raises -> GradMagnitudes:
        """Convert both planes into the pinned Float32 buffers and return
        the two magnitude sums, in one pass over the inputs.

        The sums accumulate `abs(values[i])` in Float64 in ascending index
        order, which is exactly what `_fixed_scale` accumulates, so
        `device_fixed_scale` applied to these totals returns the Float32
        scale `stage_gradients` would have computed, bit for bit. Feeding
        the result to a builder therefore quantizes identically; only the
        number of times the two lists were read has changed.

        Finiteness is checked by the scale, not here, so that a
        non-finite gradient produces the same error message it produces on
        every other path.
        """
        if len(grad) != self.n_rows or len(hess) != self.n_rows:
            raise Error("gradient/hessian length must equal n_rows")

        # Any copy still reading the staging buffers has to finish before
        # they are overwritten.
        ctx.synchronize()
        self.staged = False

        var dst_g = self.stage_g.unsafe_ptr()
        var dst_h = self.stage_h.unsafe_ptr()
        var src_g = grad.unsafe_ptr()
        var src_h = hess.unsafe_ptr()
        var g_total = 0.0
        var h_total = 0.0
        for r in range(self.n_rows):
            var g = src_g.unsafe_load(r)
            var h = src_h.unsafe_load(r)
            g_total += abs(g)
            h_total += abs(h)
            dst_g.unsafe_store(r, Float32(g))
            dst_h.unsafe_store(r, Float32(h))
        self.staged = True
        return GradMagnitudes(g_total, h_total)

    def upload(
        mut self,
        ctx: DeviceContext,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises:
        """Copy the staged planes into caller-owned device buffers, which
        must hold at least `n_rows` Float32 on this context. Enqueued, not
        synchronized."""
        if not self.staged:
            raise Error("call stage before upload")
        ctx.enqueue_copy(dst_buf=grad_dev, src_ptr=self.stage_g.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=hess_dev, src_ptr=self.stage_h.unsafe_ptr())
