"""Contracts for distributed GPU histogram exchange.

Single-node GPU training exists (`train_gpu.mojo`, `histogram_gpu.mojo`) and
data-parallel CPU training exists (`distributed.mojo`). Nothing joins them,
and section 9 of docs/distributed.md is explicit about why: the only
end-to-end GPU measurement in this repository is an Apple M4, where GPU
training is slower than the four-worker CPU trainer at every size tested, and
putting a network under a backend that has not been shown to be worth using
on one node is building on an unproven foundation.

This module is what that join would need, written down and computable, and
nothing more. It opens no device, allocates no buffer, launches no kernel,
imports no device context, and binds to no communication library. Every
function here is host arithmetic over plain lists, which is what makes the
contract checkable without hardware and what keeps this file compiling on the
CPU-only Linux runners that CI actually uses.

**Status. Distributed GPU training is unavailable, and
`require_distributed_gpu` refuses it.** Three separate things are missing and
they are tracked separately, because two of them are engineering and one of
them is evidence:

- `HAS_DEVICE_COLLECTIVE` is False: no reduction that takes device buffers
  exists in this build, and none is written here
- `transport_available()` is False: nothing in this repository has moved a
  byte between two processes
- `GPU_SPEEDUP_GATE_MET` is False: no benchmark on any device has shown
  single-node GPU training beating single-node CPU training, which is the
  gate docs/distributed.md section 9 sets before distributed GPU work starts

No distributed GPU performance, scaling, or feasibility claim is made here.

## The one real result in this file

The single-node GPU histogram accumulates in *fixed point*: gradients and
hessians are scaled by `2^30 / sum|values|`, rounded to Int32, and summed
with integer atomics, which is what makes the GPU histogram bit-deterministic
where a float atomic would not be (see histogram_gpu.mojo). The scale is
computed on the host from the magnitude sum of the values being uploaded.

That has a consequence for distributing it that is easy to get wrong: **two
ranks that each compute their own scale produce integer words that cannot be
added.** Word 7 on rank 0 and word 7 on rank 1 are quantities in different
units, and summing them is meaningless. A distributed GPU histogram therefore
needs the scale itself to be global, agreed by one small reduction before the
first histogram of the round (`agree_fixed_scales`).

Once the scale is global, three things follow, and they are why this design
is worth writing down rather than defaulting to downloading Float64
histograms and reducing those:

1. **The reduction is exact.** Integer addition of the scaled words has no
   rounding at all, so nothing is lost relative to the single-node GPU
   histogram, which already quantized.
2. **The reduction is order independent.** Integer addition is associative,
   so requirement 2 of docs/distributed.md section 5, ascending rank order,
   is not needed for this path. It is needed for the CPU data-parallel path
   because floating point addition is not associative. A fixed-point exchange
   is bit-identical under *any* reduction order or topology, which is a
   strictly weaker demand on a future transport.
3. **The words still fit.** Every rank's scaled magnitudes sum to at most
   `2^30 / world`-worth of the global budget by construction, so the global
   sum of the words is bounded by the same `2^30` a single node is bounded
   by, plus the rounding slack `check_fixed_point_headroom` accounts for.
   The exchange therefore stays inside Int32 at any world size, and
   `narrow_words` verifies rather than assumes it.

What this costs is one extra `Float64` reduction of two elements per boosting
round, and that reduction does need bit-identical delivery: two ranks holding
different roundings of the magnitude sum would derive different scales and be
back where they started.

## What this module deliberately does not do

**It does not reduce device buffers.** `DeviceCollective` is declared as the
seam a NCCL, RCCL, or oneCCL adapter would implement, and the only
implementation here is `UnavailableDeviceCollective`, which raises. Writing a
real one means binding to a library this build does not have and cannot test,
and a stub that pretends would be worse than an error that says so.

**It does not fuse with the builder.** `GpuHistogramBuilder.download_raw`
already copies the fixed-point planes into pinned host memory and
`histogram_from_host` converts them to Float64. The exchange belongs exactly
between those two calls, and `reduce_fixed_words` is shaped to sit there. It
is not wired in, because wiring it means editing histogram_gpu.mojo, which
this lane does not own; the patch is written out in
handoffs/remaining_09_distributed_strategies.md.

**It does not support anything but data parallelism.** `check_gpu_strategy`
refuses the other modes rather than leaving the combination undefined.
"""

from std.math import isfinite

from .collective import Collective, zeros_f64
from .distributed_strategies import (
    STRATEGY_DATA_PARALLEL,
    STRATEGY_FEATURE_PARALLEL,
    STRATEGY_SERIAL,
    STRATEGY_VOTING_PARALLEL,
    strategy_name,
)
from .distributed_transport import transport_available

# ---------------------------------------------------------------------------
# Build-level facts
# ---------------------------------------------------------------------------
#
# Two comptime names, in the same style as `HAS_BYTE_ENDPOINT` in
# distributed_transport.mojo and for the same reason: each is a fact about the
# build or about the evidence, each has exactly one definition, and each flips
# in exactly one place when the thing it names arrives.

comptime HAS_DEVICE_COLLECTIVE = False
"""Whether a reduction that takes device buffers is compiled in. False:
no NCCL, RCCL, oneCCL, or Metal-side collective is bound anywhere in this
repository, and `UnavailableDeviceCollective` is the only implementation of
the trait."""

comptime GPU_SPEEDUP_GATE_MET = False
"""Whether single-node GPU training has been measured beating single-node CPU
training on any device. False. The Apple M4 numbers in the README are losses
at every size tested, and no discrete-GPU run exists. This is the gate
docs/distributed.md section 9 sets, and it is evidence rather than code: it
flips when a benchmark says so, not when a feature lands."""

comptime FIXED_ONE = Float64(1 << 30)
"""The fixed-point budget, which must equal `_FIXED_ONE` in histogram_gpu.mojo.

Restated rather than imported because it is private there. The duplication is
deliberate and bounded: `check_fixed_point_contract` exists so a caller that
can see both can assert they agree, and the handoff asks the GPU lane to
export the constant so this copy can be deleted."""

# The largest total row count a fixed-point exchange may cover. Row indices
# cross into the kernels as Int32 (`MAX_ROWS` in histogram_gpu.mojo), and the
# rounding slack argument below needs the same bound globally rather than per
# rank.
comptime MAX_GLOBAL_ROWS = Int(Int32.MAX)


def check_fixed_point_contract(builder_fixed_one: Float64) raises:
    """Assert that this module and the GPU builder agree about the budget.

    One line for a caller that can see both constants, so the duplicated
    `FIXED_ONE` above cannot drift silently. Nothing here can perform this
    check itself: the builder's constant is private to its module.
    """
    if builder_fixed_one != FIXED_ONE:
        raise Error(
            "the GPU fixed-point budget disagrees with"
            " distributed_gpu.FIXED_ONE; the distributed exchange and the"
            " single-node histogram would quantize differently"
        )


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

comptime GATE_DEVICE_COLLECTIVE = 1
comptime GATE_TRANSPORT = 2
comptime GATE_GPU_SPEEDUP = 4
comptime GATE_DRIVER = 8
comptime GATE_VALIDATION = 16


def distributed_gpu_gates() -> Int:
    """Every gate distributed GPU training has not passed, as a mask.

    All of them at once, because they are independent: a socket landing does
    not make the GPU faster, and a fast GPU does not write a driver.
    """
    var mask = 0
    if not HAS_DEVICE_COLLECTIVE:
        mask |= GATE_DEVICE_COLLECTIVE
    if not transport_available():
        mask |= GATE_TRANSPORT
    if not GPU_SPEEDUP_GATE_MET:
        mask |= GATE_GPU_SPEEDUP
    # No trainer calls anything in this file, and no multi-process run of any
    # kind has happened. Both are unconditional today and are named rather
    # than folded into the gates above so that closing one is visible.
    mask |= GATE_DRIVER
    mask |= GATE_VALIDATION
    return mask


def gate_name(gate: Int) -> String:
    if gate == GATE_DEVICE_COLLECTIVE:
        return "a collective that takes device buffers"
    if gate == GATE_TRANSPORT:
        return "a transport that reaches another process"
    if gate == GATE_GPU_SPEEDUP:
        return "evidence that single-node GPU training is faster than CPU"
    if gate == GATE_DRIVER:
        return "a trainer that drives a distributed GPU round"
    if gate == GATE_VALIDATION:
        return "a multi-process run on real hardware"
    return "an unrecognized gate"


def distributed_gpu_available() -> Bool:
    """Whether distributed GPU training can run in this build. False."""
    return distributed_gpu_gates() == 0


def distributed_gpu_unavailable_detail() -> String:
    """Why not, in one sentence per unmet gate.

    Stated once here so that every caller refuses for the same reason with the
    same words, rather than each discovering the gap its own way.
    """
    var mask = distributed_gpu_gates()
    if mask == 0:
        return "distributed GPU training is available"
    var text = String("distributed GPU training is unavailable; missing:")
    var gates: List[Int] = [
        GATE_DEVICE_COLLECTIVE,
        GATE_TRANSPORT,
        GATE_GPU_SPEEDUP,
        GATE_DRIVER,
        GATE_VALIDATION,
    ]
    var first = True
    for i in range(len(gates)):
        if mask & gates[i] != 0:
            if first:
                text += " "
                first = False
            else:
                text += "; "
            text += gate_name(gates[i])
    return text^


def require_distributed_gpu() raises:
    """The gate. Raises, in every build today."""
    if not distributed_gpu_available():
        raise Error(distributed_gpu_unavailable_detail())


def check_gpu_strategy(strategy: Int) raises:
    """Which parallel modes a GPU exchange is defined for.

    Data parallel only. Feature parallel keeps every row on every rank and
    partitions features, which the device builder could express through
    `set_features`, but the mode's saving is in the histogram build and its
    cost is one full copy of the dataset per rank, which is the resource a GPU
    has least of. Voting parallel selects features from local rankings, so its
    per-node feature set changes, and a device builder that reuploads a
    feature table per node is a different design from the one that exists.
    Both are refused rather than left undefined.
    """
    if strategy == STRATEGY_DATA_PARALLEL:
        return
    if strategy == STRATEGY_SERIAL:
        raise Error(
            "the serial strategy is single-node; use train_gpu directly"
        )
    if (
        strategy == STRATEGY_FEATURE_PARALLEL
        or strategy == STRATEGY_VOTING_PARALLEL
    ):
        raise Error(
            String(
                "the ",
                strategy_name(strategy),
                "-parallel strategy has no GPU histogram exchange: only data"
                " parallelism reduces a fixed histogram grid every rank"
                " agrees on. See docs/DISTRIBUTED_STRATEGIES.md",
            )
        )
    raise Error(String("unknown parallel strategy code ", strategy))


# ---------------------------------------------------------------------------
# The device-collective seam
# ---------------------------------------------------------------------------

comptime DEVICE_COLLECTIVE_NONE = 0
"""No reduction is available at all."""

comptime DEVICE_COLLECTIVE_HOST_STAGED = 1
"""The histogram is downloaded, reduced by an ordinary `Collective`, and
uploaded. Every arithmetic step of this path is in this file; no run has
executed it."""

comptime DEVICE_COLLECTIVE_DEVICE_RESIDENT = 2
"""The histogram is reduced without leaving the device, by NCCL, RCCL,
oneCCL, or an equivalent. Nothing in this repository implements it."""


def device_collective_name(kind: Int) -> String:
    if kind == DEVICE_COLLECTIVE_NONE:
        return "none"
    if kind == DEVICE_COLLECTIVE_HOST_STAGED:
        return "host-staged"
    if kind == DEVICE_COLLECTIVE_DEVICE_RESIDENT:
        return "device-resident"
    return "unknown"


def resolve_device_collective(requested: Int) raises -> Int:
    """What a request for a reduction kind actually resolves to.

    `DEVICE_COLLECTIVE_DEVICE_RESIDENT` never resolves, because nothing
    implements it, and it is refused rather than downgraded: silently staging
    through the host when a caller asked for a device-resident reduction would
    hide exactly the cost the caller was trying to avoid.
    """
    if requested == DEVICE_COLLECTIVE_NONE:
        return DEVICE_COLLECTIVE_NONE
    if requested == DEVICE_COLLECTIVE_HOST_STAGED:
        return DEVICE_COLLECTIVE_HOST_STAGED
    if requested == DEVICE_COLLECTIVE_DEVICE_RESIDENT:
        if HAS_DEVICE_COLLECTIVE:
            return DEVICE_COLLECTIVE_DEVICE_RESIDENT
        raise Error(
            "no device-resident collective is compiled in: no NCCL, RCCL, or"
            " oneCCL binding exists in this repository. Ask for the"
            " host-staged kind, which downloads the fixed-point planes and"
            " reduces them with an ordinary Collective"
        )
    raise Error(String("unknown device collective kind ", requested))


trait DeviceCollective:
    """A reduction over one GPU histogram's fixed-point words.

    The seam, drawn deliberately at *host-visible integer words* rather than
    at a device buffer. Everything above it, in this file and in a future
    driver, is then hardware independent and testable on a machine with no
    GPU, which is the same layering argument `ByteEndpoint` makes in
    distributed_transport.mojo.

    A device-resident adapter cannot conform to this trait as written, and
    that is the honest state rather than an oversight: it would take a device
    buffer and a stream, and it would have to answer four questions this
    signature does not ask. Who owns the buffer during the reduction. Whether
    the caller must synchronize before handing it over, or the adapter does.
    Which queue the reduction runs on relative to the kernel that filled the
    buffer. What a failure leaves behind on the device. Writing that signature
    without a library to bind it to would be inventing answers, so the
    device-resident contract is specified in
    docs/DISTRIBUTED_STRATEGIES.md and not declared here.

    Every implementation must satisfy the same requirements a `Collective`
    does (docs/distributed.md section 5), with one relaxation that
    `reduce_fixed_words` earns: integer addition is associative, so a
    fixed-point word reduction may be performed in any order or topology and
    still deliver identical bytes to every rank.
    """

    def kind(self) -> Int:
        """One of the `DEVICE_COLLECTIVE_*` codes."""
        ...

    def world_size(self) -> Int:
        ...

    def allreduce_words(mut self, mut words: List[Int]) raises:
        """Sum one node's fixed-point planes across ranks, leaving the
        identical result in every rank's `words`."""
        ...

    def barrier(mut self) raises:
        ...


@fieldwise_init
struct UnavailableDeviceCollective(DeviceCollective, Copyable, Movable):
    """The only implementation of `DeviceCollective` in this repository.

    It refuses. That is the point: a caller that reaches a GPU exchange gets
    one clear error naming every unmet gate instead of a stub that returns
    plausible zeros, and the trait still has a conforming type so the seam
    compiles and can be pattern-matched against by whatever lands next.
    """

    var n: Int

    def kind(self) -> Int:
        return DEVICE_COLLECTIVE_NONE

    def world_size(self) -> Int:
        return self.n

    def allreduce_words(mut self, mut words: List[Int]) raises:
        _ = len(words)
        raise Error(distributed_gpu_unavailable_detail())

    def barrier(mut self) raises:
        raise Error(distributed_gpu_unavailable_detail())


# ---------------------------------------------------------------------------
# The global fixed-point scale
# ---------------------------------------------------------------------------


@fieldwise_init
struct GpuFixedScales(Copyable, Movable):
    """The two scales every rank must quantize with.

    Held as `Float64` because that is how `GpuHistogramBuilder` holds them,
    but each is the widening of a `Float32`: the kernel multiplies in
    `Float32`, so the host inverse has to be the inverse of exactly the number
    the device used. Deriving them any other way reintroduces the mismatch
    that fixed point exists to remove.
    """

    var grad: Float64
    var hess: Float64

    def describe(self) -> String:
        return String("fixed scales grad=", self.grad, " hess=", self.hess)


def fixed_scale_from_total(total: Float64) raises -> Float64:
    """The scale for a magnitude sum, matching `_fixed_scale` in
    histogram_gpu.mojo exactly.

    Same floor, same `Float32` narrowing, same rejection of a non-finite
    input. It is written out rather than imported because that function is
    private to its module; the handoff asks for it to be exported so this can
    forward instead of mirror. Any divergence between the two is a
    correctness bug, not a style difference, which is what
    `check_fixed_point_contract` and the mirror test in the handoff are for.
    """
    var magnitude = total
    if not isfinite(magnitude):
        raise Error("gradients and hessians must be finite")
    if magnitude < 0.0:
        raise Error("a magnitude sum cannot be negative")
    if magnitude < 1e-12:
        magnitude = 1e-12
    var scale = Float32(FIXED_ONE / magnitude)
    if not isfinite(Float64(scale)) or Float64(scale) <= 0.0:
        raise Error(
            "gradient/hessian magnitudes are out of range for the GPU"
            " fixed-point histogram"
        )
    return Float64(scale)


def magnitude_sum(values: List[Float64]) raises -> Float64:
    """One rank's contribution to the global magnitude sum.

    Summed in the list's own order, which is the row order, so the reduction
    that follows sees per-rank partials formed exactly as the single-node
    builder forms its whole sum.
    """
    var total = 0.0
    for i in range(len(values)):
        total += abs(values[i])
    if not isfinite(total):
        raise Error("gradients and hessians must be finite")
    return total


def agree_fixed_scales[
    C: Collective
](
    mut comm: C,
    local_grad: List[List[Float64]],
    local_hess: List[List[Float64]],
) raises -> GpuFixedScales:
    """The scales every rank will quantize with this round.

    One reduction of two elements, once per boosting round rather than once
    per node, because the gradients change once per round and the scale is a
    function of them.

    This is the one reduction on the GPU path that *does* need requirement 1
    of docs/distributed.md section 5, bit-identical delivery: two ranks
    holding different roundings of the same magnitude sum would derive
    different scales, and every integer word after that would be in a
    different unit. Everything downstream is integer and needs nothing.

    `local_grad` and `local_hess` hold this process's local ranks' arrays, one
    entry each per `comm.n_local_ranks()`, which is the shape every other
    collective in this repository takes.
    """
    var n_local = comm.n_local_ranks()
    if len(local_grad) != n_local or len(local_hess) != n_local:
        raise Error("agree_fixed_scales needs one array per local rank")
    var sums = zeros_f64(2)
    for i in range(n_local):
        sums[0] += magnitude_sum(local_grad[i])
        sums[1] += magnitude_sum(local_hess[i])
    comm.allreduce_sum_f64(sums)
    return GpuFixedScales(
        fixed_scale_from_total(sums[0]), fixed_scale_from_total(sums[1])
    )


def check_fixed_point_headroom(total_rows: Int, world_size: Int) raises:
    """Whether the reduced words are guaranteed to stay inside Int32.

    The argument, stated so it can be checked rather than trusted. With a
    global scale, the scaled magnitudes of every row on every rank sum to at
    most `FIXED_ONE`, so the exact sum of the reduced words is bounded by
    `FIXED_ONE = 2^30` regardless of world size. Rounding each row's value to
    the nearest integer adds at most a half per row, so the realized sum is
    bounded by `2^30 + total_rows / 2`, which stays inside `Int32.MAX` for
    every row count the kernels accept anyway.

    The bound is global, not per rank, which is exactly why it survives
    sharding: splitting the same rows across more ranks does not increase
    either term.
    """
    if world_size < 1:
        raise Error("world_size must be positive")
    if total_rows < 0:
        raise Error("total_rows must not be negative")
    if total_rows > MAX_GLOBAL_ROWS:
        raise Error(
            String(
                "the world holds ",
                total_rows,
                " rows and the fixed-point exchange is bounded at ",
                MAX_GLOBAL_ROWS,
            )
        )
    var bound = FIXED_ONE + Float64(total_rows) / 2.0
    if bound > Float64(MAX_GLOBAL_ROWS):
        raise Error(
            "the fixed-point headroom does not cover this row count; the"
            " reduced words could overflow Int32"
        )


# ---------------------------------------------------------------------------
# Host-staged word exchange
# ---------------------------------------------------------------------------
#
# The three functions a driver would call between `download_raw` and
# `histogram_from_host`. They are host arithmetic over lists, so they run
# anywhere, and they are the whole staged path: there is nothing else to it.


def widen_words(words: List[Int32]) -> List[Int]:
    """The device's Int32 planes as the `Int` buffer a `Collective` reduces.

    Widening before the reduction rather than after is what makes the sum
    unable to overflow *during* the reduction even if a rank contributed
    something the headroom argument did not anticipate; `narrow_words` is then
    where an impossible value is caught, once, with the whole sum in hand.
    """
    var out = List[Int](capacity=len(words))
    for i in range(len(words)):
        out.append(Int(words[i]))
    return out^


def narrow_words(values: List[Int]) raises -> List[Int32]:
    """The reduced buffer back in the device's word type.

    Checked rather than truncated. An out-of-range word here means the
    headroom argument in `check_fixed_point_headroom` did not hold, which is a
    correctness failure and not a saturating one, and the caller has to see it
    rather than train on a wrapped histogram.
    """
    var out = List[Int32](capacity=len(values))
    var lo = -Int(Int32.MAX) - 1
    var hi = Int(Int32.MAX)
    for i in range(len(values)):
        if values[i] < lo or values[i] > hi:
            raise Error(
                String(
                    "a reduced fixed-point word (",
                    values[i],
                    ") does not fit the device's Int32 histogram; the"
                    " magnitude scale and the row count are inconsistent",
                )
            )
        out.append(Int32(values[i]))
    return out^


def check_word_planes(n_words: Int, n_features: Int, n_bins: Int) raises:
    """Refuse a word buffer that is not three planes of one histogram grid.

    Cheap, local, and worth doing before the reduction rather than after: a
    mismatched buffer reaches the peers as a length disagreement, which is
    correctly detected but is reported against the wrong rank. The same
    argument `check_histogram_buffers` makes in distributed_transport.mojo.
    """
    if n_features < 1 or n_bins < 1:
        raise Error("a histogram needs at least one feature and one bin")
    var cells = n_features * n_bins
    if n_words != 3 * cells:
        raise Error(
            String(
                "a fixed-point histogram is 3 * ",
                n_features,
                " * ",
                n_bins,
                " words, and this buffer has ",
                n_words,
            )
        )


def reduce_fixed_words[
    C: Collective
](
    mut comm: C, mut words: List[Int], n_features: Int, n_bins: Int
) raises:
    """Sum one node's downloaded fixed-point planes across the world.

    One integer reduction of all three planes at once, against the Float64
    path's three. Gradients, hessians, and counts are all integers here, so
    nothing is gained by keeping counts in their own buffer the way
    `allreduce_histogram` does to make their exactness obvious. That is a
    third of the round trips, and the round trips are what a small histogram
    actually spends its time on.

    It is not, today, fewer bytes. `Collective` reduces `List[Int]`, so each
    32-bit word is staged as 64 bits and the payload is exactly the Float64
    path's `3 * cells * 8`. The halving is available and not taken: it needs a
    32-bit reduction on the transport, and `GpuExchangePlan` records both
    numbers rather than quoting the one that has not been earned.

    The caller narrows the result with `narrow_words` and hands it back to the
    builder's Float64 conversion, which divides by the *global* scales this
    module agreed. Handing it back to a builder still holding local scales
    would convert global sums with a local unit, which is the failure this
    whole module exists to prevent.
    """
    check_word_planes(len(words), n_features, n_bins)
    comm.allreduce_sum_int(words)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@fieldwise_init
struct GpuExchangePlan(Copyable, Movable):
    """What one tree node costs on a distributed GPU path.

    Computed rather than asserted, in the same spirit as `histogram_plan` in
    distributed_transport.mojo, so the comparison against the CPU path is a
    number a test can pin instead of a claim in a comment.

    `staged_payload_bytes` is what the exchange moves today, with the words
    widened to `Int` for a `Collective` that has no 32-bit reduction.
    `native_payload_bytes` is what it would move if one were added. Both are
    recorded because the difference is the single largest saving available on
    this path and it is not code that exists.
    """

    var n_features: Int
    var n_bins: Int
    var cells: Int
    var words: Int
    var reduces_per_node: Int
    var device_downloads_per_node: Int
    var device_uploads_per_node: Int
    var host_syncs_per_node: Int
    var staged_payload_bytes: Int
    var native_payload_bytes: Int
    var f64_path_payload_bytes: Int


def gpu_exchange_plan(n_features: Int, n_bins: Int) raises -> GpuExchangePlan:
    """The per-node cost of the host-staged fixed-point exchange.

    One download, one reduction, one upload, and one host synchronization per
    node. The download and the synchronization are the ones
    `GpuHistogramBuilder.download_raw` already performs for the single-node
    path, so the exchange adds the upload and the reduction and nothing else
    to a node that was already being read back.

    The upload is what a device-resident collective would remove, and it is
    also what makes the staged path a poor fit for a device whose histograms
    never needed to reach the host at all. That trade is the argument for the
    seam in `DeviceCollective`, and it is not settled by anything measured.
    """
    if n_features < 1 or n_bins < 1:
        raise Error(
            "a histogram needs at least one feature and one bin"
        )
    var cells = n_features * n_bins
    var words = 3 * cells
    return GpuExchangePlan(
        n_features,
        n_bins,
        cells,
        words,
        1,
        1,
        1,
        1,
        words * 8,
        words * 4,
        3 * cells * 8,
    )


def staged_saving_ratio(plan: GpuExchangePlan) -> Float64:
    """Float64 payload bytes over what the staged exchange actually moves.

    1.0, at every shape. Widening each 32-bit word to an `Int` for a
    `Collective` that has no 32-bit reduction gives back exactly what fixed
    point saved, and the honest number for the path that exists is the one
    that says so. The saving that remains is in round trips, one per node
    against three, which this ratio does not measure.
    """
    if plan.staged_payload_bytes == 0:
        return 0.0
    return Float64(plan.f64_path_payload_bytes) / Float64(
        plan.staged_payload_bytes
    )


def native_saving_ratio(plan: GpuExchangePlan) -> Float64:
    """The same ratio against a 32-bit reduction that does not exist yet.

    2.0, at every shape. A ratio and not a speedup: it counts payload bytes,
    ignores framing, and says nothing about latency, which is what actually
    dominates a histogram of a few hundred kilobytes. It is recorded because
    it is the size of the prize for adding one reduction op to the transport,
    and it should be read as exactly that narrow thing.
    """
    if plan.native_payload_bytes == 0:
        return 0.0
    return Float64(plan.f64_path_payload_bytes) / Float64(
        plan.native_payload_bytes
    )


# ---------------------------------------------------------------------------
# Round description
# ---------------------------------------------------------------------------


@fieldwise_init
struct GpuRoundPlan(Copyable, Movable):
    """Every collective one distributed GPU boosting round would issue.

    Two per round plus one per node: the scale agreement, the base-score or
    convergence agreement the CPU trainer already issues, and the histogram
    exchange at every node. Written down because the count is the thing a
    transport has to be able to sustain, and because a round whose collective
    count depends on local data would deadlock, so making it a computed
    function of `nodes_per_tree` is also an assertion that it does not.
    """

    var nodes_per_tree: Int
    var scale_reduces_per_round: Int
    var node_reduces_per_round: Int

    def total_reduces_per_round(self) -> Int:
        return self.scale_reduces_per_round + self.node_reduces_per_round


def gpu_round_plan(num_leaves: Int) raises -> GpuRoundPlan:
    """The collective schedule of one round at `num_leaves` leaves.

    `num_leaves` nodes are built per tree, one at the root and one per split,
    which is the same count `docs/distributed.md` section 8 uses for the CPU
    path. One scale agreement precedes them.
    """
    if num_leaves < 1:
        raise Error("num_leaves must be at least 1")
    return GpuRoundPlan(num_leaves, 1, num_leaves)
