"""Batched per-class device primitives for one softmax boosting round.

A softmax round grows one tree per class. Today the GPU trainer walks the
classes one at a time (`_train_multiclass_gpu_rounds` in train_gpu.mojo):
per class it fills a gradient plane, reduces two magnitudes, synchronizes to
read them back, grows a whole tree, and folds that tree into the device raw
scores. Every one of those steps is `n_classes` times a thing the device
could have done once, and the two that hurt most are the binned-matrix reads
(each class's root histogram re-reads the entire matrix) and the host
synchronizations (each class's magnitude readback drains the queue).

This module supplies the primitives that do those steps for several classes
at once, without changing a single number the sequential path produces.

    `_batch_softmax_grad_kernel`   one launch fills a contiguous run of
                                   classes' gradients and hessians, reading
                                   the row-major probability matrix and
                                   writing class-major derivative planes.
                                   This is the one transpose in the design;
                                   see gpu_output_planes.mojo for why the two
                                   quantities want opposite layouts.
    `_batch_abs_sum_kernel`        one launch and one 2 KiB-per-class
                                   readback reduces every batched class's
                                   gradient and hessian magnitudes, replacing
                                   `k_count` launches and `k_count` host
                                   synchronizations.
    `_batch_hist_shared_kernel`    one histogram launch over a row window
                                   every batched class shares, reading each
                                   bin byte once and scattering it into every
                                   class's partial histogram, with a single
                                   count plane for the whole batch.
    `_batch_hist_ranges_kernel`    one histogram launch over per-class row
                                   windows and per-class feature sets, for
                                   the levels below a round's root, where the
                                   classes have diverged. Shares the launch
                                   and the occupancy, not the bin reads.
    `GpuClassBatch`                the buffers those kernels read and write,
                                   allocated once per session for a batch
                                   capacity, plus the per-class fixed-point
                                   scales and the histogram extraction.
    `MulticlassRoundGuard`         host-side bookkeeping that refuses any
                                   ordering which would break the softmax
                                   contract below. No device state.

Softmax semantics, preserved exactly
------------------------------------
The contract the sequential loop keeps, and that every batched path here has
to keep, is one sentence: **every class gradient of a round derives from the
same pre-round score matrix.**

The mechanism is the probability snapshot, not the ordering of the score
updates. `refresh_softmax` materializes `prob` from `raw` once per round; the
per-class gradient kernel reads `prob` and never `raw`; so a class tree that
commits its own score update into `raw` mid-round cannot perturb any other
class's gradients. That is already true of the sequential path (which does
commit per class, immediately) and it stays true here.

What that leaves is one real barrier and one real freedom:

  * **Barrier.** No `refresh_softmax` may run until every class tree of the
    previous round has committed its update into `raw`. Otherwise the next
    round's probabilities are computed from a partially advanced score
    matrix. `MulticlassRoundGuard` is exactly this rule.
  * **Freedom.** Within a round, the per-class commits may happen in any
    order, or all at the end. Each class writes the disjoint slots
    `raw[r * n_classes + k]` for its own `k`, so concurrent commits neither
    race nor reorder any floating-point addition. Determinism does not
    require committing in class order, and this module does not impose one.

Determinism
-----------
Batching must not perturb a single bit relative to the sequential device
path. Three things guarantee that:

  * **Histogram accumulation** is fixed-point Int32 throughout, and integer
    addition is associative, so neither the batched shared-memory layout nor
    the order blocks flush in changes a cell.
  * **The fixed-point scale** of a class is derived from that class's own
    magnitude sums, and `_batch_abs_sum_kernel` uses the same `SUM_BLOCKS`
    blocks, the same `SUM_THREADS` threads, the same grid stride, and the
    same shared-memory tree reduction as the single-class
    `_abs_sum_kernel`, per class. The host then sums the partials in
    ascending block order in Float64, as the single-class path does. So a
    class's scale is the same number whether it was reduced alone or in a
    batch of a hundred.
  * **Class ordering** is fixed by `ClassBatchPlan`: batches are contiguous
    ascending runs, slots within a batch are ascending, and results are
    always collected by ascending slot. The tree of `(round, k)` is grown
    from the same gradients and stored at the same
    `round * n_classes + k` it always was, so the serialized ensemble is
    byte-identical and no prediction shape moves.

Scope
-----
Softmax multiclass only. Nothing here generalizes to arbitrary multi-output
objectives, and it should not be made to until such an objective has a real
contract: the primitives lean on three softmax-specific facts (one shared
probability matrix per round, per-class one-vs-rest derivatives of that
matrix, and one tree per class per round in a fixed slot), and a multi-output
objective that violated any of them would need a different design rather than
a wider parameter here.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite, round
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .gpu_objectives_native import (
    HESS_FLOOR,
    SUM_BLOCKS,
    SUM_THREADS,
    GpuObjectiveState,
    GradMagnitudes,
    device_fixed_scale,
)
from .gpu_output_planes import (
    MAX_BINS,
    SHARED_CLASS_CAP,
    ClassBatchPlan,
    class_major_index,
    row_major_index,
)
from .gpu_tiling import DeviceCaps, derive_block_threads, query_device_caps
from .histogram import Histogram, _zeroed_f64, _zeroed_int

# Row indices cross into the kernels as Int32, as everywhere else on the
# device path.
comptime MAX_ROWS = Int(Int32.MAX)

# Round phases the guard walks through. A round is IDLE before it opens,
# PROBS once the probability snapshot is fresh, and stays there while
# gradients are filled and trees are grown; it returns to IDLE only when
# every class of the round has committed.
comptime ROUND_IDLE = 0
comptime ROUND_OPEN = 1


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _batch_softmax_grad_kernel(
    prob: MutPointer[Float32, MutAnyOrigin],
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_classes: Int32,
    k_begin: Int32,
    k_count: Int32,
    weighted: Int32,
):
    """One-vs-rest derivatives for a contiguous run of classes, in one
    launch. `grid.x` tiles the rows and `grid.y` indexes the batch slot, so
    the whole (row, class-in-batch) plane is one launch.

    The arithmetic per (row, class) is character for character
    `gpu_objectives_native._softmax_class_kernel`: gradient `w * (p - y)`,
    hessian `w * 2 p (1 - p)` floored at `HESS_FLOOR`, with `y` the exact
    Float32 equality test against the class label. Only the addressing
    differs, and it differs on both sides: the read is row-major
    (`prob[r * n_classes + k]`, the layout the softmax reduction needs) and
    the write is class-major (`grad[slot * n_rows + r]`, the layout the
    histogram kernels need). This kernel is the transpose, and it is the only
    one in the round.
    """
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nr = Int(n_rows)
    if r >= nr:
        return
    var slot = Int(block_idx.y)
    if slot >= Int(k_count):
        return
    var k = Int(k_begin) + slot

    var w = Float32(1.0)
    if weighted != 0:
        w = weight[unsafe_offset=r][0]
    var p = prob[unsafe_offset = r * Int(n_classes) + k][0]
    var y = Float32(0.0)
    if target[unsafe_offset=r][0] == Float32(k):
        y = 1.0
    var out = slot * nr + r
    grad[unsafe_offset=out] = w * (p - y)
    var h = 2.0 * p * (1.0 - p)
    if h < HESS_FLOOR:
        h = HESS_FLOOR
    hess[unsafe_offset=out] = w * h


def _batch_abs_sum_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    k_count: Int32,
):
    """Per-threadgroup magnitude sums for every class of a batch.

    `grid.x` is the fixed `SUM_BLOCKS` grid of the single-class reduction and
    `grid.y` is the batch slot, so one launch covers the batch. Each slot's
    partials land in its own `[grad partials | hess partials]` pair at
    `slot * 2 * SUM_BLOCKS`, and each slot's block `i` accumulates exactly
    the rows `gpu_objectives_native._abs_sum_kernel` block `i` would have
    accumulated for that class alone: same `SUM_BLOCKS * SUM_THREADS` grid
    stride, same shared-memory tree reduction, same fixed trip count. The
    per-class totals are therefore bit-identical to the sequential path's,
    which is what keeps every class's fixed-point scale (and so every
    histogram derived from it) unchanged by batching.
    """
    var slot = Int(block_idx.y)
    if slot >= Int(k_count):
        return
    var tid = thread_idx.x
    var sg = stack_allocation[
        SUM_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        SUM_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()

    var acc_g = Float32(0.0)
    var acc_h = Float32(0.0)
    var nr = Int(n_rows)
    var base = slot * nr
    var r = Int(block_idx.x) * SUM_THREADS + tid
    var stride = SUM_BLOCKS * SUM_THREADS
    while r < nr:
        acc_g += abs(grad[unsafe_offset = base + r][0])
        acc_h += abs(hess[unsafe_offset = base + r][0])
        r += stride
    sg[unsafe_offset=tid] = acc_g
    sh[unsafe_offset=tid] = acc_h
    barrier()

    var active = SUM_THREADS // 2
    while active > 0:
        if tid < active:
            sg[unsafe_offset=tid] = (
                sg[unsafe_offset=tid][0] + sg[unsafe_offset = tid + active][0]
            )
            sh[unsafe_offset=tid] = (
                sh[unsafe_offset=tid][0] + sh[unsafe_offset = tid + active][0]
            )
        barrier()
        active //= 2

    if tid == 0:
        var out = slot * 2 * SUM_BLOCKS + Int(block_idx.x)
        partials[unsafe_offset=out] = sg[unsafe_offset=0][0]
        partials[unsafe_offset = out + SUM_BLOCKS] = sh[unsafe_offset=0][0]


def _zero_batch_kernel(
    buffer: MutPointer[Int32, MutAnyOrigin],
    n: Int32,
):
    """Zero a batched histogram output. The batched paths accumulate with
    atomics into their own planes, so the output is always zeroed first, the
    same rule the single-class atomic path follows."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        buffer[unsafe_offset=i] = 0


def _batch_hist_shared_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    scales: MutPointer[Float32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_bins: Int32,
    hist_size: Int32,
    cap: Int32,
    k_count: Int32,
    rows_per_tile: Int32,
    begin: Int32,
    count: Int32,
):
    """The batched histogram for a row window every class in the batch
    shares: one bin read serves `k_count` classes.

    This is `gpu_active_rows._range_hist_atomic_kernel` with the per-class
    dimension pulled inside the threadgroup. A block still owns one
    (feature, row tile) pair, so `grid.x` is the active-feature slot and
    `grid.y` is the tile, exactly as before. The difference is the inner
    body: having loaded a row's bin once, the block folds that row into every
    resident class's partial histogram, and it accumulates the bin's row
    count once rather than `k_count` times, because a count does not depend
    on the class.

    Shared memory is `k_count` gradient planes, `k_count` hessian planes, and
    one count plane, statically sized at `SHARED_CLASS_CAP * MAX_BINS`, which
    is why the caller must keep `k_count <= SHARED_CLASS_CAP`
    (`gpu_output_planes.classes_per_block` derives the runtime bound and
    clamps to the same cap).

    Each class carries its own fixed-point scale, so `scales` is
    `[g_0 .. g_{cap-1} | h_0 .. h_{cap-1}]` and the quantization of a class's
    gradient is the quantization that class would have received alone.

    The output planes are addressed by the batch *capacity*, not by
    `k_count`, so a short final batch writes the same slots a full one
    would: grad of slot `c` at `c * hist_size`, hess at
    `(cap + c) * hist_size`, and the shared counts at `2 * cap * hist_size`.
    """
    var f = Int(feat_ids[unsafe_offset = Int(block_idx.x)][0])
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var hs = Int(hist_size)
    var kc = Int(k_count)
    var n = Int(count)

    var sg = stack_allocation[
        SHARED_CLASS_CAP * MAX_BINS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh = stack_allocation[
        SHARED_CLASS_CAP * MAX_BINS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sc = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    var b = tid
    while b < nb:
        sc[unsafe_offset=b] = 0
        for c in range(kc):
            sg[unsafe_offset = c * nb + b] = 0
            sh[unsafe_offset = c * nb + b] = 0
        b += block_dim.x
    barrier()

    var tile_begin = block_idx.y * Int(rows_per_tile)
    var tile_end = tile_begin + Int(rows_per_tile)
    if tile_end > n:
        tile_end = n

    var col = f * nr
    var j = tile_begin + tid
    while j < tile_end:
        var r = Int(rows[unsafe_offset = Int(begin) + j][0])
        # The one bin read the whole batch shares. Everything below reuses
        # it; a sequential run would have issued this load once per class.
        var bin = Int(bins[unsafe_offset = col + r])
        _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        for c in range(kc):
            var gq = Int32(
                round(
                    grad[unsafe_offset = c * nr + r][0]
                    * scales[unsafe_offset=c][0]
                )
            )
            var hq = Int32(
                round(
                    hess[unsafe_offset = c * nr + r][0]
                    * scales[unsafe_offset = Int(cap) + c][0]
                )
            )
            _ = Atomic.fetch_add(sg.unsafe_offset(c * nb + bin), gq)
            _ = Atomic.fetch_add(sh.unsafe_offset(c * nb + bin), hq)
        j += block_dim.x
    barrier()

    var base = f * nb
    b = tid
    while b < nb:
        if sc[unsafe_offset=b][0] != 0:
            for c in range(kc):
                _ = Atomic.fetch_add(
                    out_hist.unsafe_offset(c * hs + base + b),
                    sg[unsafe_offset = c * nb + b][0],
                )
                _ = Atomic.fetch_add(
                    out_hist.unsafe_offset((Int(cap) + c) * hs + base + b),
                    sh[unsafe_offset = c * nb + b][0],
                )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(2 * Int(cap) * hs + base + b),
                sc[unsafe_offset=b][0],
            )
        b += block_dim.x


def _batch_hist_ranges_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    scales: MutPointer[Float32, MutAnyOrigin],
    begins: MutPointer[Int32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_bins: Int32,
    hist_size: Int32,
    cap: Int32,
    n_slots: Int32,
    rows_stride: Int32,
    rows_per_tile: Int32,
):
    """The batched histogram for classes that have diverged: each class
    brings its own row window, its own active feature set, and (when
    `rows_stride` is nonzero) its own row permutation.

    One block still owns one (feature, tile) pair of one class, so this
    kernel shares no bin read with anything: at a node below a round's root
    the classes are looking at different rows, and pretending otherwise would
    change the histograms. What it does share is the launch. A deep node
    holds few rows, its per-class grid is a handful of threadgroups, and
    `k_count` such launches leave the device mostly idle between drains;
    folding the class into `grid.x` turns them into one launch with
    `k_count` times the blocks, which is the only lever left at that depth.

    Addressing: `grid.x` is `class_slot * n_slots + feature_slot` and
    `grid.y` is the tile. `rows_stride` is `n_rows` when each class owns a
    permutation (the general case) and 0 when they happen to share one
    buffer, which lets the same kernel serve a shared-row batch whose feature
    sets differ. Each class writes its own three planes, so unlike the shared
    kernel this one carries a count plane per class.
    """
    var ns = Int(n_slots)
    var c = Int(block_idx.x) // ns
    var slot = Int(block_idx.x) - c * ns
    var n = Int(counts[unsafe_offset=c][0])
    if n <= 0:
        return

    var f = Int(feat_ids[unsafe_offset = c * ns + slot][0])
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var hs = Int(hist_size)

    var sg = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    var b = tid
    while b < nb:
        sg[unsafe_offset=b] = 0
        sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var tile_begin = block_idx.y * Int(rows_per_tile)
    var tile_end = tile_begin + Int(rows_per_tile)
    if tile_end > n:
        tile_end = n

    var col = f * nr
    var row_base = c * Int(rows_stride) + Int(begins[unsafe_offset=c][0])
    var g_scale = scales[unsafe_offset=c][0]
    var h_scale = scales[unsafe_offset = Int(cap) + c][0]
    var j = tile_begin + tid
    while j < tile_end:
        var r = Int(rows[unsafe_offset = row_base + j][0])
        var bin = Int(bins[unsafe_offset = col + r])
        var gq = Int32(round(grad[unsafe_offset = c * nr + r][0] * g_scale))
        var hq = Int32(round(hess[unsafe_offset = c * nr + r][0] * h_scale))
        _ = Atomic.fetch_add(sg.unsafe_offset(bin), gq)
        _ = Atomic.fetch_add(sh.unsafe_offset(bin), hq)
        _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        j += block_dim.x
    barrier()

    var base = f * nb
    b = tid
    while b < nb:
        if sc[unsafe_offset=b][0] != 0:
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(c * hs + base + b),
                sg[unsafe_offset=b][0],
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset((Int(cap) + c) * hs + base + b),
                sh[unsafe_offset=b][0],
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset((2 * Int(cap) + c) * hs + base + b),
                sc[unsafe_offset=b][0],
            )
        b += block_dim.x


# ---------------------------------------------------------------------------
# The round's synchronization contract
# ---------------------------------------------------------------------------


struct MulticlassRoundGuard(Copyable, Movable):
    """Host-side bookkeeping that enforces the softmax round contract.

    No device state and no allocation past two flag lists: this exists so the
    one rule that batching can silently break is checked rather than
    commented. The rule, restated from the module docstring: a round's
    probability snapshot must be taken when the raw score matrix holds every
    previous round's trees and none of this round's.

    Ordering it accepts, per round:

        open_round()                  once, only with no pending commits
        note_probs()                  once, immediately after
        note_gradients(begin, count)  any number, any batch, after note_probs
        note_tree(k)                  once per class
        note_commit(k)                once per class, after that class's tree
        close_round()                 once every class has committed

    Commit order is deliberately unconstrained: each class writes the
    disjoint slots `raw[r * n_classes + k]`, so no ordering among them is
    observable. Commit *timing* is constrained at exactly one point, the next
    `open_round`, which is where an uncommitted tree would corrupt the next
    snapshot.
    """

    var n_classes: Int
    var phase: Int
    var probs_fresh: Bool
    var tree_done: List[Bool]
    var committed: List[Bool]
    var rounds_closed: Int

    def __init__(out self, n_classes: Int) raises:
        if n_classes < 2:
            raise Error("a multiclass round needs at least two classes")
        self.n_classes = n_classes
        self.phase = ROUND_IDLE
        self.probs_fresh = False
        self.tree_done = List[Bool](capacity=n_classes)
        self.committed = List[Bool](capacity=n_classes)
        for _ in range(n_classes):
            self.tree_done.append(False)
            self.committed.append(False)
        self.rounds_closed = 0

    def open_round(mut self) raises:
        """Start a round. Refuses to open while any class of the previous
        round still owes a score-matrix commit, which is the barrier the
        whole contract reduces to."""
        if self.phase != ROUND_IDLE:
            raise Error("a multiclass round is already open")
        for k in range(self.n_classes):
            if self.tree_done[k] and not self.committed[k]:
                raise Error(
                    "a previous round's class tree has not been committed to"
                    " the raw scores; committing after the next probability"
                    " refresh would change every class's gradients"
                )
        for k in range(self.n_classes):
            self.tree_done[k] = False
            self.committed[k] = False
        self.phase = ROUND_OPEN
        self.probs_fresh = False

    def note_probs(mut self) raises:
        """Record the round's one `refresh_softmax`."""
        if self.phase != ROUND_OPEN:
            raise Error("open the round before refreshing probabilities")
        if self.probs_fresh:
            raise Error(
                "a round refreshes the softmax probabilities exactly once;"
                " a second refresh would derive later classes' gradients"
                " from a different score matrix than the earlier ones"
            )
        self.probs_fresh = True

    def note_gradients(mut self, k_begin: Int, k_count: Int) raises:
        """Record a batched gradient fill over classes
        `[k_begin, k_begin + k_count)`."""
        if self.phase != ROUND_OPEN:
            raise Error("open the round before filling gradients")
        if not self.probs_fresh:
            raise Error(
                "refresh the softmax probabilities before filling any"
                " class's gradients"
            )
        if k_count < 1:
            raise Error("a gradient batch must cover at least one class")
        if k_begin < 0 or k_begin + k_count > self.n_classes:
            raise Error("gradient batch is outside the class range")

    def note_tree(mut self, k: Int) raises:
        """Record that class `k`'s tree of this round finished growing."""
        self._check_class(k)
        if self.phase != ROUND_OPEN:
            raise Error("open the round before growing a class tree")
        if not self.probs_fresh:
            raise Error("class trees need this round's probabilities")
        if self.tree_done[k]:
            raise Error("a class grows exactly one tree per round")
        self.tree_done[k] = True

    def note_commit(mut self, k: Int) raises:
        """Record that class `k`'s tree was folded into the raw scores."""
        self._check_class(k)
        if not self.tree_done[k]:
            raise Error("a class commits its score update after its tree")
        if self.committed[k]:
            raise Error("a class commits its score update exactly once")
        self.committed[k] = True

    def close_round(mut self) raises:
        """End the round. Every class must have grown and committed; a round
        that dropped its trees for want of progress is closed with
        `abandon_round` instead."""
        if self.phase != ROUND_OPEN:
            raise Error("no multiclass round is open")
        for k in range(self.n_classes):
            if not self.tree_done[k]:
                raise Error(
                    "every class grows a tree in a round; the ensemble's"
                    " round-major layout has no slot for a missing one"
                )
            if not self.committed[k]:
                raise Error(
                    "every class tree commits its score update before the"
                    " round closes"
                )
        self.phase = ROUND_IDLE
        self.probs_fresh = False
        self.rounds_closed += 1

    def abandon_round(mut self) raises:
        """End a round whose trees were dropped because no class made
        progress, which is what both trainers do before breaking or
        resampling. The dropped trees never reached the raw scores, so
        nothing is owed; the guard just returns to idle."""
        if self.phase != ROUND_OPEN:
            raise Error("no multiclass round is open")
        for k in range(self.n_classes):
            if self.committed[k]:
                raise Error(
                    "a round with a committed class tree cannot be abandoned;"
                    " its score update is already in the raw scores"
                )
            self.tree_done[k] = False
        self.phase = ROUND_IDLE
        self.probs_fresh = False

    def pending_commits(self) -> Int:
        """Classes whose tree is grown and whose score update has not landed.
        The quantity `open_round` requires to be zero."""
        var n = 0
        for k in range(self.n_classes):
            if self.tree_done[k] and not self.committed[k]:
                n += 1
        return n

    def _check_class(self, k: Int) raises:
        if k < 0 or k >= self.n_classes:
            raise Error("class index out of range")


# ---------------------------------------------------------------------------
# The batched device buffers
# ---------------------------------------------------------------------------


struct GpuClassBatch(Movable):
    """Device buffers for one resident batch of softmax class trees.

    Constructed once per training session for a batch *capacity*, then
    driven per round: `fill_gradients` for a run of classes, `refresh_scales`
    once for the whole batch, and then one histogram enqueue per node level.
    Every buffer is sized by the capacity, and a short final batch simply
    leaves the tail slots untouched, so the plane offsets never move.

    The context is passed in rather than opened, exactly as
    `GpuObjectiveState` does it, so a batch, its objective state, and the
    histogram builder all sit on the one queue that owns them.
    """

    var ctx: DeviceContext
    var grad_dev: DeviceBuffer[DType.float32]
    """Class-major gradients, `grad[slot * n_rows + r]`."""
    var hess_dev: DeviceBuffer[DType.float32]
    """Class-major hessians, same layout."""
    var scale_dev: DeviceBuffer[DType.float32]
    """`[g_0 .. g_{cap-1} | h_0 .. h_{cap-1}]`, the per-class fixed-point
    scales the histogram kernels quantize with."""
    var part_dev: DeviceBuffer[DType.float32]
    var host_part: HostBuffer[DType.float32]
    var out_dev: DeviceBuffer[DType.int32]
    var host_out: HostBuffer[DType.int32]
    var feat_dev: DeviceBuffer[DType.int32]
    """`cap * n_features` Int32: each slot's active feature ids, so classes
    with different feature subsamples can share one launch."""
    var begin_dev: DeviceBuffer[DType.int32]
    var count_dev: DeviceBuffer[DType.int32]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var cap: Int
    var shared_counts: Bool
    var block_threads: Int
    var g_scale: List[Float64]
    var h_scale: List[Float64]
    var n_slots: Int
    var has_gradients: Bool
    var has_scales: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        caps: DeviceCaps,
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        cap: Int,
        shared_counts: Bool,
    ) raises:
        """Allocate for `cap` resident classes over a dataset shape.

        `shared_counts` picks the output layout: `2 * cap + 1` planes when
        the batch will histogram a shared row window (one count plane for the
        batch), `3 * cap` when each class brings its own rows. The choice is
        fixed at construction because it fixes the allocation; a session that
        needs both allocates the `3 * cap` form, which the shared kernel can
        also write into.
        """
        if n_rows < 1:
            raise Error("class batching requires at least one row")
        if n_rows > MAX_ROWS:
            raise Error("GPU backend supports at most 2^31 - 1 rows")
        if n_features < 1:
            raise Error("class batching requires at least one feature")
        if n_bins < 1 or n_bins > MAX_BINS:
            raise Error("GPU backend supports at most 256 bins")
        if cap < 1:
            raise Error("class batch capacity must be positive")

        self.ctx = ctx
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins
        self.cap = cap
        self.shared_counts = shared_counts
        self.block_threads = derive_block_threads(caps)
        self.n_slots = n_features
        self.has_gradients = False
        self.has_scales = False

        var hist_size = n_features * n_bins
        var planes = 3 * cap
        if shared_counts:
            planes = 2 * cap + 1
        var out_cells = planes * hist_size

        self.grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows * cap)
        self.hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows * cap)
        self.scale_dev = ctx.enqueue_create_buffer[DType.float32](2 * cap)
        self.part_dev = ctx.enqueue_create_buffer[DType.float32](
            2 * SUM_BLOCKS * cap
        )
        self.host_part = ctx.enqueue_create_host_buffer[DType.float32](
            2 * SUM_BLOCKS * cap
        )
        self.out_dev = ctx.enqueue_create_buffer[DType.int32](out_cells)
        self.host_out = ctx.enqueue_create_host_buffer[DType.int32](out_cells)
        self.feat_dev = ctx.enqueue_create_buffer[DType.int32](
            n_features * cap
        )
        self.begin_dev = ctx.enqueue_create_buffer[DType.int32](cap)
        self.count_dev = ctx.enqueue_create_buffer[DType.int32](cap)

        self.g_scale = List[Float64](capacity=cap)
        self.h_scale = List[Float64](capacity=cap)
        for _ in range(cap):
            self.g_scale.append(1.0)
            self.h_scale.append(1.0)

        # Every feature active in every slot until `set_features` narrows a
        # slot, matching `GpuHistogramBuilder`'s construction-time state.
        with self.feat_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for c in range(cap):
                for f in range(n_features):
                    dst.unsafe_store(c * n_features + f, Int32(f))

    def __init__(
        out self,
        ctx: DeviceContext,
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        cap: Int,
        shared_counts: Bool,
    ) raises:
        """Query the device's capabilities and allocate."""
        var caps = query_device_caps(ctx)
        self = Self(
            ctx, caps, n_rows, n_features, n_bins, cap, shared_counts
        )

    @staticmethod
    def for_plan(
        ctx: DeviceContext,
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        plan: ClassBatchPlan,
    ) raises -> GpuClassBatch:
        """Allocate for a plan `gpu_output_planes.plan_class_batches`
        produced, so the memory the planner budgeted is the memory the
        session takes."""
        plan.check()
        return GpuClassBatch(
            ctx,
            n_rows,
            n_features,
            n_bins,
            plan.batch_size,
            plan.shared_counts,
        )

    @always_inline
    def hist_size(self) -> Int:
        return self.n_features * self.n_bins

    @always_inline
    def grad_offset(self, slot: Int) -> Int:
        """Where slot `slot`'s gradient plane starts. The integration hook:
        an existing per-row kernel that takes a `Float32[n_rows]` gradient
        buffer reads this batch's slot by starting here, because the plane is
        class-major and therefore contiguous."""
        return class_major_index(0, slot, self.n_rows)

    def _row_blocks(self) -> Int:
        return (
            self.n_rows + self.block_threads - 1
        ) // self.block_threads

    def _check_slot(self, slot: Int) raises:
        if slot < 0 or slot >= self.cap:
            raise Error("batch slot out of range")

    def fill_gradients(
        mut self,
        mut state: GpuObjectiveState,
        k_begin: Int,
        k_count: Int,
    ) raises:
        """Fill slots `0 .. k_count-1` with classes `k_begin ..
        k_begin+k_count-1`, from the probability snapshot `state` already
        holds.

        `state.refresh_softmax` must have run for this round first; this is
        the batched form of `GpuHistogramBuilder.fill_softmax_gradients_
        device` and reads exactly what that path reads. Slot `c` always
        carries class `k_begin + c`, which is what makes results collected by
        ascending slot results in ascending class order.
        """
        if state.n_classes < 2:
            raise Error("class batching requires a multiclass objective state")
        if state.n_rows != self.n_rows:
            raise Error("objective state and class batch disagree on n_rows")
        if not state.has_raw:
            raise Error("call init_raw before filling gradients")
        if k_count < 1 or k_count > self.cap:
            raise Error("class batch size out of range")
        if k_begin < 0 or k_begin + k_count > state.n_classes:
            raise Error("class batch is outside the class range")

        self.ctx.enqueue_function[_batch_softmax_grad_kernel](
            state.prob_dev.unsafe_ptr(),
            state.target_dev.unsafe_ptr(),
            state.weight_dev.unsafe_ptr(),
            self.grad_dev.unsafe_ptr(),
            self.hess_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(state.n_classes),
            Int32(k_begin),
            Int32(k_count),
            Int32(1) if state.weighted else Int32(0),
            grid_dim=(self._row_blocks(), k_count),
            block_dim=self.block_threads,
        )
        self.has_gradients = True
        self.has_scales = False

    def magnitude_sums(
        mut self, k_count: Int
    ) raises -> List[GradMagnitudes]:
        """Every batched class's gradient and hessian magnitude sums, in
        ascending slot order, from one launch and one readback.

        The sequential path pays `k_count` launches and `k_count` host
        synchronizations here, one per class, and each drains the queue. This
        pays one of each, and the transfer is `2 KiB * k_count` regardless of
        `n_rows`. The host-side accumulation is Float64 over ascending block
        index, per class, exactly as `GpuObjectiveState.magnitude_sums` does
        it, so each class's totals are the same two numbers.
        """
        if not self.has_gradients:
            raise Error("fill the batch's gradients before reducing them")
        if k_count < 1 or k_count > self.cap:
            raise Error("class batch size out of range")
        self.ctx.enqueue_function[_batch_abs_sum_kernel](
            self.grad_dev.unsafe_ptr(),
            self.hess_dev.unsafe_ptr(),
            self.part_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(k_count),
            grid_dim=(SUM_BLOCKS, k_count),
            block_dim=SUM_THREADS,
        )
        self.ctx.enqueue_copy(
            dst_ptr=self.host_part.unsafe_ptr(), src_buf=self.part_dev
        )
        self.ctx.synchronize()

        var src = self.host_part.unsafe_ptr()
        var out = List[GradMagnitudes](capacity=k_count)
        for c in range(k_count):
            var base = c * 2 * SUM_BLOCKS
            var g_total = 0.0
            var h_total = 0.0
            for i in range(SUM_BLOCKS):
                g_total += Float64(src.unsafe_load(base + i))
                h_total += Float64(src.unsafe_load(base + SUM_BLOCKS + i))
            if not isfinite(g_total) or not isfinite(h_total):
                raise Error("gradients and hessians must be finite")
            out.append(GradMagnitudes(g_total, h_total))
        return out^

    def set_scales(mut self, mags: List[GradMagnitudes]) raises:
        """Derive and upload each batched class's fixed-point scale from its
        own magnitude sums. `device_fixed_scale` is the same function the
        single-class device path calls, so a class's scale does not depend on
        the batch it was reduced in."""
        if len(mags) < 1 or len(mags) > self.cap:
            raise Error("magnitude list length out of range")
        for c in range(len(mags)):
            self.g_scale[c] = Float64(device_fixed_scale(mags[c].grad))
            self.h_scale[c] = Float64(device_fixed_scale(mags[c].hess))
        with self.scale_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for c in range(len(mags)):
                dst.unsafe_store(c, Float32(self.g_scale[c]))
                dst.unsafe_store(self.cap + c, Float32(self.h_scale[c]))
        self.has_scales = True

    def refresh_scales(mut self, k_count: Int) raises:
        """`magnitude_sums` then `set_scales`: the round's one readback."""
        var mags = self.magnitude_sums(k_count)
        self.set_scales(mags)

    def set_features(
        mut self, slot: Int, features: List[Int]
    ) raises:
        """Give one batch slot its own active feature set.

        Feature subsampling draws once per tree with seed
        `round * n_classes + k`, so the classes of a round generally scan
        different features; carrying a feature table per slot is what lets
        them still share a launch. Every slot must list the same *number* of
        features, because the number is the launch's `grid.x` stride, and
        `n_slots` records it.
        """
        self._check_slot(slot)
        if len(features) == 0:
            raise Error("active feature set must not be empty")
        if len(features) > self.n_features:
            raise Error("active feature set is larger than n_features")
        for i in range(len(features)):
            if features[i] < 0 or features[i] >= self.n_features:
                raise Error("feature index out of range")
        if slot == 0:
            self.n_slots = len(features)
        elif len(features) != self.n_slots:
            raise Error(
                "every class in a batch must scan the same number of"
                " features; the count is the launch's feature stride"
            )
        with self.feat_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for i in range(len(features)):
                dst.unsafe_store(slot * self.n_features + i, Int32(features[i]))

    def set_features_uniform(
        mut self, features: List[Int], k_count: Int
    ) raises:
        """The same feature set in every slot, which is the case that makes a
        batch eligible to share bin reads."""
        if k_count < 1 or k_count > self.cap:
            raise Error("class batch size out of range")
        for c in range(k_count):
            self.set_features(c, features)

    def set_windows(
        mut self, begins: List[Int], counts: List[Int]
    ) raises:
        """Each slot's row window into its permutation, for the diverged
        path. `begins[c]` and `counts[c]` are the `LeafRange` of the node
        class `c` is currently building, in its own active-row buffer."""
        if len(begins) != len(counts):
            raise Error("window begins and counts must have equal length")
        if len(begins) < 1 or len(begins) > self.cap:
            raise Error("window list length out of range")
        for c in range(len(begins)):
            if begins[c] < 0 or counts[c] < 0:
                raise Error("row windows must be nonnegative")
            if begins[c] + counts[c] > self.n_rows:
                raise Error("row window runs past the row buffer")
        with self.begin_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for c in range(len(begins)):
                dst.unsafe_store(c, Int32(begins[c]))
        with self.count_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for c in range(len(counts)):
                dst.unsafe_store(c, Int32(counts[c]))

    def _zero_output(mut self) raises:
        var planes = 3 * self.cap
        if self.shared_counts:
            planes = 2 * self.cap + 1
        var cells = planes * self.hist_size()
        var blocks = (cells + self.block_threads - 1) // self.block_threads
        self.ctx.enqueue_function[_zero_batch_kernel](
            self.out_dev.unsafe_ptr(),
            Int32(cells),
            grid_dim=blocks,
            block_dim=self.block_threads,
        )

    def enqueue_shared_histogram[
        bins_origin: MutOrigin,
        rows_origin: MutOrigin, //,
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        rows: MutPointer[Int32, rows_origin],
        begin: Int,
        count: Int,
        k_count: Int,
        rows_per_tile: Int,
    ) raises:
        """Enqueue one histogram launch covering `k_count` classes over the
        single row window `[begin, begin + count)` of one shared
        permutation.

        This is the round-root case, and the only one that shares bin reads.
        The caller must have established that it applies: every class reads
        these rows (true of a round's root, because the round draws one bag
        before any class's tree) through the same feature set (true when
        feature subsampling is off, or when the caller has narrowed every
        slot to the same list). `gpu_output_planes.BatchEligibility` is where
        that decision is written down.

        Accumulation is the atomic strategy only. The tiled strategy exists
        to avoid output atomics when many row tiles contend on the same bins;
        here the contention is already spread over `k_count` output planes,
        and a batched partial buffer would multiply the tiled path's memory
        by the same `k_count` this path is trying to make affordable. Both
        strategies produce identical integer histograms anyway, so this is a
        cost choice and not a semantic one.
        """
        if not self.has_gradients:
            raise Error("fill the batch's gradients before histogramming")
        if not self.has_scales:
            raise Error("refresh the batch's scales before histogramming")
        if k_count < 1 or k_count > self.cap:
            raise Error("class batch size out of range")
        if k_count > SHARED_CLASS_CAP:
            raise Error(
                "more classes share a threadgroup than the batched shared"
                " histogram kernel allocates for; see SHARED_CLASS_CAP"
            )
        if begin < 0 or count < 0 or begin + count > self.n_rows:
            raise Error("row window runs past the row buffer")
        if rows_per_tile < 1:
            raise Error("rows per tile must be positive")

        self._zero_output()
        if count == 0:
            return
        var n_tiles = (count + rows_per_tile - 1) // rows_per_tile
        self.ctx.enqueue_function[_batch_hist_shared_kernel](
            bins,
            rows,
            self.grad_dev.unsafe_ptr(),
            self.hess_dev.unsafe_ptr(),
            self.feat_dev.unsafe_ptr(),
            self.scale_dev.unsafe_ptr(),
            self.out_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.n_bins),
            Int32(self.hist_size()),
            Int32(self.cap),
            Int32(k_count),
            Int32(rows_per_tile),
            Int32(begin),
            Int32(count),
            grid_dim=(self.n_slots, n_tiles),
            block_dim=self.block_threads,
        )

    def enqueue_ranged_histogram[
        bins_origin: MutOrigin,
        rows_origin: MutOrigin, //,
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        rows: MutPointer[Int32, rows_origin],
        k_count: Int,
        rows_stride: Int,
        max_count: Int,
        rows_per_tile: Int,
    ) raises:
        """Enqueue one histogram launch covering `k_count` classes, each over
        its own row window (set by `set_windows`) of its own permutation.

        `rows_stride` is the distance between two classes' permutations in
        `rows`: `n_rows` when each class owns one, 0 when they share a
        buffer. `max_count` is the largest window in the batch and sizes
        `grid.y`; a class with fewer rows simply has idle tiles, which is the
        price of one launch for a ragged batch and is why the class policy in
        the handoff caps the ragged span rather than batching arbitrarily.

        No bin read is shared here, by construction: the classes are looking
        at different rows. What is shared is the launch and the occupancy,
        which is the whole win at depth, where a node's own row count is far
        too small to fill the device.
        """
        if not self.has_gradients:
            raise Error("fill the batch's gradients before histogramming")
        if not self.has_scales:
            raise Error("refresh the batch's scales before histogramming")
        if self.shared_counts:
            raise Error(
                "a shared-count output has one count plane for the batch;"
                " diverged classes need their own, so allocate the batch"
                " with shared_counts False"
            )
        if k_count < 1 or k_count > self.cap:
            raise Error("class batch size out of range")
        if rows_stride != 0 and rows_stride != self.n_rows:
            raise Error("row stride must be 0 or n_rows")
        if max_count < 0:
            raise Error("the largest row window must be nonnegative")
        if rows_per_tile < 1:
            raise Error("rows per tile must be positive")

        self._zero_output()
        if max_count == 0:
            return
        var n_tiles = (max_count + rows_per_tile - 1) // rows_per_tile
        self.ctx.enqueue_function[_batch_hist_ranges_kernel](
            bins,
            rows,
            self.grad_dev.unsafe_ptr(),
            self.hess_dev.unsafe_ptr(),
            self.feat_dev.unsafe_ptr(),
            self.scale_dev.unsafe_ptr(),
            self.begin_dev.unsafe_ptr(),
            self.count_dev.unsafe_ptr(),
            self.out_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.n_bins),
            Int32(self.hist_size()),
            Int32(self.cap),
            Int32(self.n_slots),
            Int32(rows_stride),
            Int32(rows_per_tile),
            grid_dim=(self.n_slots * k_count, n_tiles),
            block_dim=self.block_threads,
        )

    def download(mut self) raises:
        """Copy the batched histogram into pinned host memory and wait. One
        synchronization for the whole batch, where the sequential path pays
        one per class."""
        self.ctx.enqueue_copy(
            dst_ptr=self.host_out.unsafe_ptr(), src_buf=self.out_dev
        )
        self.ctx.synchronize()

    def histogram_for(self, slot: Int) raises -> Histogram:
        """One batched class's histogram, as the Float64 `Histogram` the
        grower consumes. Call after `download`.

        The counts come from the batch's shared plane when there is one, so
        every class of a shared-row batch reports the same integer counts,
        which is what a shared row window means. Dequantization divides by
        that class's own scale, so the values are the values the sequential
        path would have produced.
        """
        self._check_slot(slot)
        var hist_size = self.hist_size()
        var g = _zeroed_f64(hist_size)
        var h = _zeroed_f64(hist_size)
        var c = _zeroed_int(hist_size)
        var g_inv = 1.0 / self.g_scale[slot]
        var h_inv = 1.0 / self.h_scale[slot]
        var gp = g.unsafe_ptr()
        var hp = h.unsafe_ptr()
        var cp = c.unsafe_ptr()
        var src = self.host_out.unsafe_ptr()
        var g_base = slot * hist_size
        var h_base = (self.cap + slot) * hist_size
        var c_base = (2 * self.cap + slot) * hist_size
        if self.shared_counts:
            c_base = 2 * self.cap * hist_size
        for i in range(hist_size):
            gp.unsafe_store(i, Float64(src.unsafe_load(g_base + i)) * g_inv)
            hp.unsafe_store(i, Float64(src.unsafe_load(h_base + i)) * h_inv)
            cp.unsafe_store(i, Int(src.unsafe_load(c_base + i)))
        return Histogram(g^, h^, c^, self.n_features, self.n_bins)

    def histograms(self, k_count: Int) raises -> List[Histogram]:
        """Every batched class's histogram, in ascending slot order, which is
        ascending class order. The ordering guarantee lives here: results
        leave this module already sorted, so no caller can reorder them by
        collecting them as they complete."""
        if k_count < 1 or k_count > self.cap:
            raise Error("class batch size out of range")
        var out = List[Histogram](capacity=k_count)
        for slot in range(k_count):
            out.append(self.histogram_for(slot))
        return out^

    def scale_of(self, slot: Int) raises -> Float64:
        """Slot `slot`'s gradient scale, for a caller that dequantizes
        itself (a device-side split search reads the fixed-point planes
        directly and needs the same inverse)."""
        self._check_slot(slot)
        return self.g_scale[slot]

    def hess_scale_of(self, slot: Int) raises -> Float64:
        self._check_slot(slot)
        return self.h_scale[slot]


def batch_score_index(r: Int, k: Int, n_classes: Int) -> Int:
    """Where a raw score or probability lives: row-major, unchanged and
    unchangeable. Re-exported from `gpu_output_planes` here so a caller
    reading this module does not have to be told twice which of the two
    layouts the score matrix uses."""
    return row_major_index(r, k, n_classes)


def batch_grad_index(r: Int, slot: Int, n_rows: Int) -> Int:
    """Where a batched gradient or hessian lives: class-major, so slot
    `slot`'s plane is the contiguous `Float32[n_rows]` the existing histogram
    kernels already take."""
    return class_major_index(r, slot, n_rows)
