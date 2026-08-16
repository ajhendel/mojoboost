"""Device-side split search over GPU histograms.

`split.mojo` finds a node's best split on the host, from a `Histogram` that
`histogram_gpu.mojo` has just downloaded. That download is the expensive part
of a GPU tree node: `3 * n_features * n_bins` Int32 words plus one host
synchronization, per node, whatever the node's size. A 100-feature, 256-bin
dataset pays 300 KB and a full pipeline drain to choose one (feature, bin)
pair, so the transfer is three orders of magnitude larger than the answer it
is used to compute.

This module searches the histogram where it already lives. The kernels here
consume the same fixed-point `[grad | hess | count]` Int32 buffer
`GpuHistogramBuilder` produces and emit one compact **split record**: the
winning feature, threshold or category set, gain, runner-up gain, missing
direction, both children's statistics, and both children's leaf values. That
record is the only thing that crosses back to the host, so a node costs a
fixed 136 bytes instead of a histogram.

`train_gpu.mojo` drives this per node today, under
`SPLIT_SEARCH_DEVICE`. The module is also standalone and testable on its
own: it owns a histogram buffer that a caller can upload to directly, and
`enqueue` also accepts an external device buffer for the zero-copy path.

One node, or a whole frontier
-----------------------------
Every per-node table has a slot per record: the feature set, the allow
mask, the float parameters (which carry the node's monotone bounds), and
the histogram offset. `enqueue_frontier` stages a bounded set of leaves,
issues one copy per table, and runs one scan and one reduction over all of
them; `download_frontier` is the single wait that brings every decision
back. That is one host synchronization per tree level rather than the two
per node the incremental loop pays, and it is what the record layout was
designed for. The batch changes nothing about a decision: each node reads
only its own slots, the scan order inside a node is unchanged, and the
records are the ones the same nodes searched one at a time would produce.

Float32 near ties, and the host-scan fallback
---------------------------------------------
The scan is Float32 (point 1 below), so two candidates whose exact gains
differ by less than a few ulps can come back in either order, and that is a
different *tree*, not a different last bit.

"A few ulps" is the wrong unit for that sentence and the right one is worth
stating here, because it is what `set_gain_form` exists for. The parameter
that controls how finely this scan can separate two candidates is not the row
count and not an ulp of the gain: it is the ratio `parent_score / gain`. The
shipped subtractive form resolves to about `eps * parent_score`, an absolute
floor that does not shrink as the gain does, so at a nearly pure leaf under
logistic loss -- where the ratio runs into the thousands -- two candidates a
part in ten thousand apart land on the same Float32. `GAIN_FORM_CROSS`, the
default, evaluates the same gain through an identity that never forms the
large sum, which moves the resolution to about `eps * sqrt(parent_score *
gain)`, and takes the right-hand child sums in the integer domain where they
are exact. Candidates 5 and 6 of `docs/design/ACCURACY_BUDGET.md`; the
argument is at `gpu_cross_gain` and `gpu_right_sum` and the arm is at
`GpuSplitSearcher.set_gain_form`. It is the only arm in this module that
changes a record. Every record therefore carries
`runner_gain`, the best gain of every candidate the node scored except the
winner, over every scanned feature, so the margin the decision was made by
is a number the host can see. `GpuSplitRecord.is_near_tie` tests the margin
against a relative tolerance (`SPLIT_TIE_RELATIVE`, deliberately several
ulps wide) *and* against `GpuSplitRecord.resolution_floor`, the absolute
width the scan's own arithmetic could not see past on that node.
`host_rescan_recommended` is the policy: a run that needs CPU/GPU agreement
redoes exactly those nodes with the host scan, one node at a time, and keeps
the device decision everywhere else. Tracking the runner-up costs one
compare per candidate and cannot change which candidate wins.
`frontier_margin` reports the same quantity one level up, where it decides
which leaf splits next.

Note the shape of the two widths, because a relative one alone is the wrong
test and was the only test here until this file's own account of its
resolution was read back against it. That resolution is set by
`parent_score / gain`, not by the gain, so at a nearly pure leaf the margin
below which two candidates are indistinguishable is a large multiple of
`SPLIT_TIE_RELATIVE * gain`. A parity run gated on the relative width alone
would report no near ties on exactly the nodes where a decision flips.

**Where the CPU/GPU disagreement is not.** It is not the tie-break. The
host's rule is "highest gain; among equal gains, the first in scan order",
reached by a strict `>` over an ascending walk, and every arm here
reproduces it: the serial scan and the wide scan by the same strict `>` over
an ascending candidate ordinal, the serial fold over ascending slots, and
the threadgroup fold by `block.max` on the gain then `block.min` on the slot
of the threads holding it, which is that rule split in two.
`tests/test_split_tie_parity.mojo` constructs exact ties in all three shapes
a tie can take -- two bins of one feature, two features whose winners sit at
different bins, and the two missing directions of one bin -- under both gain
forms, and `tests/test_gpu_split_tie_parity.mojo` holds the kernels to the
same answers, including a node with more feature slots than one reduce
thread owns. On identical histograms the replica and `find_best_split` also
chose the identical split on every one of twelve hundred pseudo-random nodes
across three gradient regimes and both gain forms. What is left to explain a
prediction gap is the two documented numeric differences below plus the fact
that the two backends do not read the same histogram at all: the device
reads fixed-point sums of quantized gradients, the host reads Float64 sums.
Both push near ties over, and `host_rescan_recommended` is the answer to
both; it is built and tested here and, as of this writing, has no caller.

Semantics
---------
Every decision this module makes is the one `find_best_split` would make, in
the same order:

- Features are scanned in active-slot order, bins ascending, and within a bin
  the missing-left candidate is scored before the missing-right one. Both the
  per-feature scan and the cross-feature reduction accept a new best only on
  a strictly greater gain, so the first candidate in that order wins every
  tie, exactly as on the host. Composed, the two stages select the
  lexicographically smallest (slot, bin, direction) among the maximum-gain
  candidates, which is what the host's single loop selects.
- L1 soft-thresholding, the L2 denominator, `min_child_hess`,
  `min_data_in_leaf`, the reserved missing bin and its `default_left`
  direction, the "every ordinary bin left" top threshold, monotone candidate
  rejection and output clamping, and both categorical searches (one-vs-rest
  and the sorted many-vs-many walk) are reproduced candidate for candidate
  from `split.mojo`, `gain.mojo`, `monotone.mojo`, and `categorical.mojo`.

Two deliberate numeric differences from the host path, neither of which
changes the shape of the arithmetic:

1. **Float32.** Apple GPUs have no Float64, so gains, hessian tests, leaf
   values, and the categorical sort keys are computed in Float32, as the
   histogram kernels already are. Split decisions therefore agree with the
   host to Float32 precision, not bit-exactly; a candidate pair whose gains
   differ below Float32 resolution may resolve the other way.
2. **Exact accumulation.** A child's gradient and hessian sums accumulate in
   the histogram's fixed-point Int32 and are dequantized once, at the end,
   rather than being summed from already-dequantized values as the host does.
   Integer addition is associative and the fixed-point scale bounds every
   partial sum (see `histogram_gpu._fixed_scale`), so the device sums are
   exact and reproducible, and row counts are exact integers throughout.

Both together mean the device is bit-deterministic run to run, which is the
property the GPU backend already guarantees, but is not bit-identical to the
host scan. Equivalence tests against the CPU trainer must stay
tolerance-based.

One rule here is CatBoost's, and it is the one rule that must not be
tolerance-based
--------------------------------------------------------------------------
`random_strength` adds a seeded normal to every candidate's gain, and the
whole point of it is that the host and the device pick the **same** split
under the same seed: a stochastic split rule whose two backends disagree is
worse than no stochastic split rule, because the disagreement is invisible
and looks like noise by design. The draw's key is 64-bit integer arithmetic
and is identical on both backends; the normal it feeds is Float64
transcendental arithmetic and cannot be, so it is drawn once on the host and
uploaded as a plane the kernels read. Default 0, at which not one instruction
of it executes. The section headed `random_strength` below carries the
argument, the cost, and what would break if the draw were moved onto the
device.

Layout
------
The search runs as two kernels and never allocates per node:

- `_scan_slot_kernel`, one threadgroup per (node, active feature), writes
  that feature's best candidate for that node into a per-slot record.
- `_reduce_slots_kernel`, one thread per node, folds that node's per-slot
  records into one record in ascending slot order and fills in the child
  statistics, child leaf values, the parent's leaf value, and the node's
  runner-up gain. `_reduce_slots_block_kernel` is the same fold on a
  threadgroup, which is what runs unless
  `MOJOTREES_GPU_SPLIT_PRIMITIVES=0`.

Two launches cover a whole frontier, not two per node and not one per
feature: the grid is `(widest feature slot, node)`, which is the shape
LightGBM's CUDA best-split finder uses when it runs the frontier's tasks as
one grid of `num_tasks_` blocks. There is no per-feature or per-leaf launch
left in this module to merge away.

Collective primitives, and where they are not allowed
----------------------------------------------------
The reductions here are `gpu.primitives.block` collectives (`sum`, `max`,
`min`, `prefix_sum`) rather than hand-rolled shared-memory loops, which is
portable across NVIDIA, AMD, and Apple Metal and so keeps this module's
one-source rule. A collective may replace a loop here only where
reassociation cannot move a bit:

- Every quantity accumulated along a feature's bins is fixed-point Int32,
  and integer addition is associative, so `block.sum` over a feature's
  totals and `block.prefix_sum[exclusive=True]` over the per-thread chunk
  sums return the serial walk's words exactly.
- Gains are only ever *compared* across threads, never summed, and `max`
  and `min` are associative and commutative on the values these kernels
  produce, so a tree-shaped argmax returns the serial walk's winner exactly
  once the tie-break is carried alongside it (highest gain, then lowest
  candidate ordinal inside a feature and lowest feature slot across
  features).

What is deliberately left serial: the gain arithmetic itself, and the
per-thread walk along a chunk of bins. A gain is a difference of three
Float32 quotients, and no collective reassociates one. There is no
floating-point atomic and no floating-point sum crossing a thread boundary
anywhere in this module, which is what keeps the device path
bit-deterministic run to run.

The scan is sequential within a feature because the candidate order *is* the
tie-breaking rule, and because a threshold scan is a prefix sum. Features are
the parallel dimension. That is deliberately the cheap half of the win: the
scan is `O(n_features * n_bins)` against the histogram build's
`O(n_rows * n_features)`, so removing the download is what matters, and
parallelizing within a feature (one thread per bin over a shared-memory
prefix scan) is a later refinement that cannot change any result, since the
prefix sums are exact integers.

Toward a device-side queue
--------------------------
`max_records` lets one searcher hold several leaves' records at once, and
`enqueue_pick_best` reduces a set of them to the single best-gain leaf,
tie-broken by ascending record index. The staircase this module was built
for is: the host downloads one record per node (the incremental loop); the
host downloads the records for a whole split at once (`enqueue_frontier`
plus `download_frontier`, which is now here, and which the resident grower
in `train_gpu` uses to pay one wait per split rather than two); the
frontier itself lives in `rec_i_dev`/`rec_f_dev` and the host only reads
the finished tree.

Note what the middle step does *not* become under leaf-wise growth: one
download per tree level. Leaf-wise splitting picks the best-gain leaf in
the whole frontier, so only the two new children need searching after each
split, and there is no level of siblings to batch. A level-wise grower is
what would turn `enqueue_frontier` into one wait per level; for the grower
we ship, one wait per split is the floor while the host is still deciding.

`enqueue_pick_best` is the piece that would lift that floor, and it is
built and tested but unused, because the rest of the last step is not in
this module and is larger than it looks. The device-side row partition it
was waiting on now exists: `GpuHistogramBuilder.apply_split` partitions a
parent's row range entirely on the device and stays fully enqueued when the
caller passes the left count it already has from the parent histogram. What
remains is everything else the host still does per split, none of which is
about finding a split: the leaf-value commit, the monotone output bounds
threaded down each branch, the `min_data_in_leaf` and `max_depth` shape
rules, per-node feature subsampling, and writing the tree itself. Those are
also what the CPU/GPU equivalence tests pin, so moving them is a
correctness project and not a latency patch.

Nothing in the record layout has to change for any of it, which is why the
record carries child statistics and leaf values rather than making the host
recompute them from a histogram it no longer has.

On where the device path's remaining cost actually is: on the evidence so
far it is not the scan kernel's shape. Packing feature slots a SIMD group
at a time instead of one threadgroup each, and moving the categorical sort
scratch out of threadgroup memory to lift the occupancy that allocation
caps, were both measured on an M4 at 50000 x 100 and both came back inside
noise of the one-thread-per-threadgroup launch this module still defaults
to. Whatever the per-split overhead is, those two did not touch it.

`_scan_slot_wide_kernel` is a third attempt at the same target and it is
off by default for that reason and not for any doubt about the result: it
splits one feature's bins across a threadgroup and returns the serial
kernel's record bit for bit, which `tests/test_gpu_split_search.mojo`
asserts against the serial kernel on the same histograms. It is reached
only through `MOJOTREES_GPU_SPLIT_WIDE=1`, and the two measurements above
are what it has to be read against: a run that cannot separate it from the
serial scan is the expected outcome, not a surprise. The default flips when
an interleaved benchmark resolves it and not before.
"""

from std.gpu import block_idx, thread_idx
from std.math import fma, log, sqrt
from std.memory import stack_allocation
from std.os import getenv
from std.sys import has_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.primitives import block
from max.gpu.sync import barrier

from .apple_gpu_policy import API_METAL, parse_api
from .categorical import (
    CatBitset,
    CategoricalParams,
    CategoricalSpec,
    cat_add,
    cat_empty,
)
from .gpu_runtime import (
    READBACK_MAP,
    READBACK_PINNED_ONE_SYNC,
    READBACK_PINNED_PAIR_SYNC,
    READBACK_PLAIN_ONE,
    READBACK_PLAIN_PAIR,
    env_readback_transport,
    readback_transport,
    readback_transport_name,
    require_readback_correct,
)
from .monotone import (
    MONOTONE_DECREASING,
    MONOTONE_FREE,
    MONOTONE_INCREASING,
    OutputBounds,
)
from .rng import GOLDEN, splitmix64, uniform

# `SCORE_L2` and `SCORE_COSINE` are re-exported by `split.mojo` and defined
# beside the `ExtraTreeParams.score_function` field that carries the choice.
# They are imported here rather than restated because a device kernel that
# spelled its own copy of the two codes could drift from the host's, and the
# whole contract of this module is that the two searches disagree in loop
# structure and never in what a parameter means.
from .split import SCORE_COSINE, SCORE_L2, SplitInfo, check_score_function

# `score_function_name` is the one of the four that `split.mojo` does not
# re-export, so it is taken from the module that defines all four. That
# module imports only `cegb`, `gain`, `monotone` and `rng`, so this edge adds
# no cycle; it is the same edge `split.mojo` itself has.
from .tree_parameters_extra import score_function_name

# The widest histogram the GPU backend accepts (`histogram_gpu.MAX_BINS`).
comptime MAX_SPLIT_BINS = 256

# --- Split record layout -------------------------------------------------
#
# A record is one slice of an Int32 buffer and one slice of a Float32 buffer,
# rather than one packed struct, so no value is ever bit-cast between an
# integer and a float on the device. Counts, bin ids, and the category set
# are exact integers; gains, sums, and leaf values are Float32.
#
# Two slices, one allocation. `records_dev` holds both planes end to end and
# `rec_i_dev` / `rec_f_dev` are windows onto it, so one `enqueue_copy` moves a
# whole record set. The bit-cast sentence above still holds: nothing on the
# device reads an integer word as a float or the reverse, and the only place
# the two planes are ever addressed through one pointer is the host-side
# unpack in `download_words`, where the reinterpretation is explicit and
# region-aligned. See `SPLIT_RECORD_WORDS`.

comptime IREC_FEATURE = 0
comptime IREC_BIN = 1
comptime IREC_FLAGS = 2
comptime IREC_ORDINAL = 3
comptime IREC_LEFT_COUNT = 4
comptime IREC_RIGHT_COUNT = 5
comptime IREC_TOTAL_COUNT = 6
comptime IREC_CAT0 = 7

# A category set is 256 bits held 16 bits to an Int32 word, so no bit is ever
# written into an Int32's sign position and the host needs no unsigned
# reinterpretation to read it back.
comptime CAT_WORD_BITS = 16
comptime CAT_WORDS = MAX_SPLIT_BINS // CAT_WORD_BITS

comptime SPLIT_IWORDS = IREC_CAT0 + CAT_WORDS

comptime FREC_GAIN = 0
comptime FREC_LEFT_GRAD = 1
comptime FREC_LEFT_HESS = 2
comptime FREC_RIGHT_GRAD = 3
comptime FREC_RIGHT_HESS = 4
comptime FREC_TOTAL_GRAD = 5
comptime FREC_TOTAL_HESS = 6
comptime FREC_LEFT_VALUE = 7
comptime FREC_RIGHT_VALUE = 8
comptime FREC_PARENT_VALUE = 9
comptime FREC_RUNNER_GAIN = 10
"""The best gain among every candidate this node scored *except* the
winner, across every scanned feature. The margin `gain - runner_gain` is
what a caller measures a Float32 near-tie against; see
`GpuSplitRecord.is_near_tie` and the near-tie section of the module
docstring."""

comptime SPLIT_FWORDS = 11

comptime SPLIT_RECORD_WORDS = SPLIT_IWORDS + SPLIT_FWORDS
"""One record's whole width in four-byte words: 23 integer, then 11 float,
then 136 bytes.

**The packed record layout, and it is a layout and not a sum.** `records_dev`
is one device allocation of `max_records * SPLIT_RECORD_WORDS` Int32 words in
which slot `r`'s integer words live at `r * SPLIT_IWORDS` and slot `r`'s float
words live at `max_records * SPLIT_IWORDS + r * SPLIT_FWORDS`. `rec_i_dev` and
`rec_f_dev` are `create_sub_buffer` windows onto those two regions, so every
kernel and every reader outside this struct sees exactly the buffers it saw
when they were two allocations: same element type, same length, and
`unsafe_ptr()` is the window's base.

Plane-major and not record-major, deliberately. Interleaving each record's
integer and float words would put the whole 136-byte record contiguous, which
is the layout a reader expects from the phrase "packed record"; it is the
wrong one here. `_pick_best_record_kernel` and the frontier reduction index
`rec_i` by `r * SPLIT_IWORDS` and `rec_f` by `r * SPLIT_FWORDS`, both from
their own base, and the whole point of the sub-buffer windows is that those
kernels do not change. Record-major would have changed every one of them, and
`gpu_resident_round.mojo` reads both fields directly and belongs to another
lane. Plane-major buys the single copy for nothing.

The two regions are counted in Int32 words because a Float32 is the same four
bytes, which is the same argument `tables_dev` makes for its float parameter
region: `create_sub_buffer` takes its offset and length in elements, and at
four bytes either element type addresses the same boundary."""

comptime FLAG_FOUND = 1
comptime FLAG_DEFAULT_LEFT = 2
comptime FLAG_CATEGORICAL = 4

# --- Per-node integer parameter block ------------------------------------
#
# The two node-varying integers that cannot be kernel arguments once a whole
# frontier is searched by one launch: how many feature slots this node
# scans, and where its histogram starts in the buffer the launch was given.
# One row per record.

comptime NODE_SLOTS = 0
comptime NODE_HIST_BASE = 1
"""Offset, in Int32 words, of this node's `[grad | hess | count]`
histogram inside the buffer passed to the launch. Zero for a caller that
hands over one node's histogram (`GpuHistogramBuilder.out_dev`), and
`slot * 3 * n_features * n_bins` for a caller whose leaves share one
multi-slot buffer, which is exactly the layout `GpuLeafBatcher.out_dev`
holds and `slot_cells` measures."""

comptime NODE_WORDS = 2

# Relative margin below which a node's winning gain and its runner-up are
# not distinguishable in Float32 with any confidence.
#
# Float32 carries about 1.2e-7 of relative resolution. A gain is a
# difference of three quotients, each accumulated from dequantized sums, so
# a handful of roundings separate a computed gain from the exact one; 1e-6
# is roughly eight of those, which is the conservative side of the only
# number that matters here, since being too eager costs a host rescan of
# one node and being too lax silently accepts a split the host would not
# have chosen.
comptime SPLIT_TIE_RELATIVE = Float64(1e-6)

# Float32's unit roundoff, 2^-23. Named because the near-tie test below
# derives an absolute floor from it and a relative constant cannot stand in
# for that floor; see `GpuSplitRecord.resolution_floor`.
comptime SPLIT_F32_EPS = Float64(1.1920928955078125e-7)

# How many roundings the near-tie floor allows for. A gain is a difference of
# quotients over dequantized sums, so a computed gain sits a small multiple of
# the unit roundoff away from the exact one rather than exactly one. Eight is
# the same allowance `SPLIT_TIE_RELATIVE` above already makes (1e-6 is about
# eight times 1.2e-7), kept the same here so the two halves of the test are
# conservative by the same amount.
comptime SPLIT_TIE_ROUNDINGS = Float64(8.0)

# --- Per-node float parameter block --------------------------------------
#
# The floating-point half of a node's search parameters travels as one small
# device buffer instead of as kernel arguments, which keeps both kernels at a
# launch arity comparable to the histogram kernels'.

comptime PF_G_INV = 0
comptime PF_H_INV = 1
comptime PF_LAMBDA_L2 = 2
comptime PF_LAMBDA_L1 = 3
comptime PF_MIN_CHILD_HESS = 4
comptime PF_BOUND_LO = 5
comptime PF_BOUND_HI = 6
comptime PF_CAT_SMOOTH = 7
comptime PF_CAT_L2 = 8

comptime PF_WORDS = 9


# --- Shared scalar arithmetic --------------------------------------------
#
# These are the Float32 counterparts of `gain.mojo` and `monotone.mojo`, and
# they are the only place the gain formula is written. Both the kernels and
# the host reference below call them, so the two searches can only disagree
# in loop structure, which is what the tests compare.


@always_inline
def gpu_soft_threshold_l1(s: Float32, lambda_l1: Float32) -> Float32:
    """`gain.soft_threshold_l1` in Float32."""
    if lambda_l1 <= 0.0:
        return s
    var mag = abs(s) - lambda_l1
    if mag <= 0.0:
        return Float32(0.0)
    return mag if s > 0.0 else -mag


@always_inline
def gpu_leaf_score(
    g: Float32, h: Float32, lambda_l1: Float32, lambda_l2: Float32
) -> Float32:
    """`gain.leaf_score` in Float32."""
    var t = gpu_soft_threshold_l1(g, lambda_l1)
    return t * t / (h + lambda_l2)


@always_inline
def gpu_leaf_value(
    g: Float32, h: Float32, lambda_l1: Float32, lambda_l2: Float32
) -> Float32:
    """`tree._leaf_value` for one leaf's totals, in Float32. The host clamps
    the result into the node's monotone interval; this is the raw Newton
    value, exactly as `_leaf_value` returns it."""
    return -gpu_soft_threshold_l1(g, lambda_l1) / (h + lambda_l2)


@always_inline
def gpu_clamp(value: Float32, lo: Float32, hi: Float32) -> Float32:
    """`OutputBounds.clamp` in Float32."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


@always_inline
def gpu_violates(
    sign: Int32, left_output: Float32, right_output: Float32
) -> Bool:
    """`monotone.violates` in Float32."""
    if sign == Int32(MONOTONE_INCREASING):
        return left_output > right_output
    if sign == Int32(MONOTONE_DECREASING):
        return left_output < right_output
    return False


@always_inline
def gpu_output_score(
    grad_sum: Float32, hess_sum: Float32, lambda_l2: Float32, output: Float32
) -> Float32:
    """`monotone.output_score` in Float32."""
    return -(
        2.0 * grad_sum * output + (hess_sum + lambda_l2) * output * output
    )


# --- The two gain forms ---------------------------------------------------
#
# Candidates 5 and 6 of `docs/design/ACCURACY_BUDGET.md`, which land together
# and only together. The whole argument for them is at `gpu_cross_gain` and
# `gpu_right_sum`; `GpuSplitSearcher.set_gain_form` states what the arm costs
# a caller.

comptime GAIN_FORM_SUBTRACTIVE = 0
"""The gain as this module shipped: `left_score + right_score -
parent_score`, with each candidate's right-hand sums obtained by Float32
subtraction from the node total."""

comptime GAIN_FORM_CROSS = 1
"""The cancellation-free gain, with the right-hand sums taken in the integer
fixed-point domain before dequantization. One code for both changes because
one without the other is a regression; see `set_gain_form`."""

comptime DEFAULT_GAIN_FORM = GAIN_FORM_CROSS


def gain_form_requested() -> Int:
    """`MOJOTREES_GPU_SPLIT_GAIN_FORM=subtractive`, the switch back to the
    shipped gain.

    On unless refused, which is the same posture `MOJOTREES_GPU_SPLIT_
    PRIMITIVES` and `histogram_gpu.set_scale_shape` take, and for the same
    two reasons: the default has to be the arm the analysis prefers, and an
    environment variable can only compare two arms across two processes,
    which on a machine whose device timings drift several-fold between time
    windows resolves nothing. `GpuSplitSearcher.set_gain_form` is the
    in-process handle.

    Anything other than the exact string `subtractive` leaves the default
    alone rather than raising, because this is read in a constructor that
    has no better failure mode; the setter is where a bad code is refused.
    """
    return (
        GAIN_FORM_SUBTRACTIVE if getenv(
            "MOJOTREES_GPU_SPLIT_GAIN_FORM"
        ) == "subtractive" else DEFAULT_GAIN_FORM
    )


def describe_gain_form(form: Int) -> String:
    """The arm's name, for `describe_scan` and for a test that wants to
    assert which one is live without restating the constant."""
    if form == GAIN_FORM_CROSS:
        return "cross"
    if form == GAIN_FORM_SUBTRACTIVE:
        return "subtractive"
    return "unknown"


@always_inline
def gpu_resolve_gain_form(requested: Int32, lambda_l1: Float32) -> Int32:
    """The arm a scan actually runs, which is the requested one except under
    L1.

    **The cross form is not valid at `lambda_l1 != 0` and this is the guard
    that says so.** The identity at `gpu_cross_gain` is derived from
    `GL + GR = G`. With L1 the three gradient sums entering the gain are not
    `GL`, `GR`, `G` but their soft-thresholded images `T(GL)`, `T(GR)`,
    `T(G)`, and soft thresholding is not additive: `T(GL) + T(GR)` differs
    from `T(G)` by up to `lambda_l1` whenever the two children pull the same
    way. The identity is then simply false, and the error it introduces is a
    *bias*, not a rounding.

    Measured, in a standalone NumPy model of this scan (not a mojotrees
    measurement): applying the cross form anyway at `lambda_l1 = 0.5`,
    `lambda_l2 = 1`, 200,000 rows, over each node's top 200 candidates, the
    median relative error of the computed gain runs 1.6e-06 at a
    `parent_score / gain` ratio of 0.03 up to 1.6e-04 at a ratio of 293 --
    where the shipped form is at 1.0e-05. The tell that it is a bias rather
    than noise is that the median and the p99 agree to two figures and that
    the Float32-right and Int32-right arms agree to three, which rounding
    error does not do.

    Rejected alternative: derive a second identity for the thresholded case.
    There is not one to derive. `T` is piecewise linear with a dead zone, so
    which of the three sums is in the dead zone changes the algebra, and the
    resulting expression would need the same `T(GL) + T(GR) - T(G)` residual
    that the subtraction it is trying to remove already carries. L1 keeps
    the shipped form, and the shipped form at `lambda_l1 != 0` is exactly
    what it was.
    """
    if lambda_l1 != Float32(0.0):
        return Int32(GAIN_FORM_SUBTRACTIVE)
    return requested


@always_inline
def gpu_right_sum(
    total_f: Float32,
    left_f: Float32,
    total_q: Int32,
    left_q: Int32,
    inv: Float32,
    form: Int32,
) -> Float32:
    """A candidate's right-hand child sum, by Float32 subtraction from the
    node total or by Int32 subtraction before dequantization.

    Candidate 6. `total_q` and `left_q` are the exact fixed-point sums the
    scan already holds in registers, and `inv` is the dequantizing factor.

    WHY THE INTEGER ROUTE IS THE EXACT ONE, AND WHERE THE ERROR ACTUALLY IS
    ----------------------------------------------------------------------
    `total_q - left_q` is exact: Int32 addition is associative and a parent
    cell is the exact integer sum of its children's, which is the same
    property `quantized_gradient.subtract_quantized` already relies on. What
    is *not* obvious, and what changed under candidate 3, is where the
    Float32 route loses its bits.

    It is not the subtraction. Since `fixed_point_scale_pow2` made the scale
    a power of two, `inv` is exactly representable and multiplying by it is
    an exponent adjustment, so `total_f - left_f` equals
    `inv * (fl32(total_q) - fl32(left_q))` exactly, and by Sterbenz's lemma
    that inner subtraction is itself exact whenever the left child holds
    between half and twice the node total. **The Float32 route's whole error
    is the two Int32 -> Float32 casts in front of it.** A fixed-point sum
    runs to `2^30` and Float32 carries 24 bits, so each cast rounds by up to
    `2^-24` *of the node total*, and the derived right-hand sum inherits both
    -- an absolute error set by the parent's magnitude, not by its own. The
    integer route casts once, after the subtraction, so its error is `2^-24`
    of the right child's own magnitude. On a candidate whose right child
    holds a hundredth of the node, that is a hundredfold difference.

    So candidate 3 did not make candidate 6 redundant; it moved the argument.
    The budget document derived candidate 6 from an inexact dequantization,
    and that reason is gone. The reason that remains is the cast, and it is
    the stronger one, because it does not shrink with the scale's shape.

    WHY IT IS BOUND TO THE CROSS FORM
    ---------------------------------
    Exactly the reasoning `ACCURACY_BUDGET.md` section 9 gives, and it
    survives the power-of-two scale unchanged -- if anything the power of two
    sharpens it, because with the dequantizing multiply now exact the
    anti-correlation below is exact rather than approximate. Under the
    Float32 route the derived right-hand sum carries the left's cast error
    with the opposite sign. The subtractive gain adds `GL^2/HL'` and
    `GR^2/HR'` together, so those two errors partly cancel: the shipped form
    is quietly benefiting from an error it introduces. Make the right-hand
    sum exact and the cancellation goes with it. The cross form subtracts
    where the subtractive form adds, so the same anti-correlation hurts it,
    and removing it helps.

    Measured, standalone NumPy model, median relative error of the computed
    gain over each node's top 200 candidates at `lambda_l2 = 1`, 200,000
    rows, 20 features (**not** a mojotrees measurement):

        parent/gain    sub+f32     sub+int     cross+f32   cross+int
              3.1      3.15e-07    1.77e-07    1.37e-07    1.18e-07
             27.5      1.07e-06    2.49e-06    3.93e-07    2.42e-07
              293      1.24e-05    1.34e-05    1.07e-06    7.14e-07

    Integer subtraction on top of the shipped form is worse than doing
    nothing at two of those three settings. On top of the cross form it is
    better at all three. That is why there is one arm code and not two.
    """
    if form == Int32(GAIN_FORM_CROSS):
        return (total_q - left_q).cast[DType.float32]() * inv
    return total_f - left_f


@always_inline
def gpu_cross_node_s(total_h: Float32, child_l2: Float32) -> Float32:
    """`H + 2*child_l2`, the sum of the two children's L2 denominators, as a
    node constant.

    Spelled as two adds rather than `total_h + 2.0 * child_l2` on purpose:
    a multiply feeding an add is the contractable shape
    (`docs/NUMERICS.md` section 6), and this value is consumed by both the
    cross term and the offset, so the host replica and the device kernel
    have to agree on it. Two adds cannot be contracted and there is nothing
    left for an optimizer to decide.
    """
    return total_h + child_l2 + child_l2


@always_inline
def gpu_cross_offset(
    total_g: Float32,
    total_h: Float32,
    lambda_l1: Float32,
    lambda_l2: Float32,
    child_l2: Float32,
    node_s: Float32,
) -> Float32:
    """The node constant the cross form subtracts, computed once per node.

    `gpu_cross_gain`'s first term is the gain a parent scored with
    `child_l2` would have; the node's actual parent score uses `lambda_l2`.
    The difference is

        G^2 / node_s - G^2 / (H + lambda_l2)
            = G^2 * (2*child_l2 - lambda_l2) / (node_s * (H + lambda_l2))

    which depends on nothing that varies between candidates. For the ordinal
    and one-vs-rest paths `child_l2` is `lambda_l2` and this collapses to
    the `lambda_l2 * G^2 / ((H + lambda_l2) * (H + 2*lambda_l2))` term
    `ACCURACY_BUDGET.md` section 8 gives; the many-vs-many categorical walk
    scores children at `lambda_l2 + cat_l2` against a parent at `lambda_l2`,
    which the general form covers and the section 8 form does not.

    Note what this subtraction is and is not. It is a cancelling subtract,
    but the quantity removed is bounded by `lambda_l2 / (H + lambda_l2)`
    times the parent score rather than by the parent score itself, so where
    the shipped form cancels against `P` this cancels against a term that
    vanishes with `lambda_l2` and is negligible whenever the node's hessian
    mass exceeds it. At `lambda_l2 = 0` and `cat_l2 = 0` it is exactly zero
    and the cross form has no subtraction anywhere.

    `2*child_l2 - lambda_l2` is spelled as `(child_l2 + child_l2) -
    lambda_l2` for the reason given at `gpu_cross_node_s`.
    """
    var tg = gpu_soft_threshold_l1(total_g, lambda_l1)
    var scaled = (child_l2 + child_l2) - lambda_l2
    return tg * tg * scaled / (node_s * (total_h + lambda_l2))


@always_inline
def gpu_cross_gain(
    left_g: Float32,
    left_h: Float32,
    right_g: Float32,
    right_h: Float32,
    child_l2: Float32,
    node_s: Float32,
    cross_offset: Float32,
) -> Float32:
    """The split gain in the form that does not cancel against the parent
    score. Candidate 5.

    Writing `HL' = HL + child_l2`, `HR' = HR + child_l2`, `S = HL' + HR'`,
    and using `GL + GR = G`:

        GL^2/HL' + GR^2/HR' - G^2/S  ==  (GL*HR' - GR*HL')^2 / (HL'*HR'*S)

    as an identity in exact arithmetic, for any values whatever. The node's
    parent score is taken at `lambda_l2` rather than at `child_l2` and at
    `H + lambda_l2` rather than at `S`, and `gpu_cross_offset` is exactly
    that difference. So this returns the same gain the subtractive form
    returns, not a surrogate ordering key: `best_gain` still starts at zero,
    `min_gain_to_split` still applies to it, `SPLIT_TIE_RELATIVE` still
    measures a margin against it, and the monotone branch below still
    returns a comparable number. That was worth the one extra node constant.
    A ranking-only key would have moved all four of those and bought nothing
    the identity does not already give.

    WHAT THE WIN IS, AND WHERE IT IS NOT
    ------------------------------------
    Not what it looks like. The subtraction of `parent_score` is **not**
    where the shipped form loses its bits, and a previous claim in this
    project that dropping it was a free accuracy win was wrong. Rounding is
    monotone, so subtracting a constant preserves order; and in the near-tie
    regime Sterbenz's lemma applies (`P/2 <= left_score + right_score <=
    2P`), which makes that subtraction *exactly representable*. The
    information is already gone one step earlier, in forming
    `left_score + right_score`, whose rounding error is `eps` times its own
    magnitude and therefore about `eps * P` in absolute terms -- no matter
    how small the gain it is about to become.

    That is the whole mechanism. The shipped form's absolute resolution is
    `eps * parent_score`, a floor that does not shrink as the gain does. The
    cross form never forms the large sum: its error enters through `D`,
    whose relative error is `eps * |GL*HR'| / |D|`, and since the gain is
    proportional to `D^2` this gives an absolute resolution of order
    `eps * sqrt(parent_score * gain)`.

    **Derived bound: the two resolutions differ by a factor of about
    `sqrt(parent_score / gain)`.** That is the number to reason with. It is
    one at a centered node, where the two forms are interchangeable, and it
    grows without bound as a node's gradients become one-sided -- which is
    what a nearly pure leaf under logistic or softmax loss looks like in a
    late round.

    Measured against that bound, in a standalone NumPy model of this scan
    (**not** a mojotrees measurement, and not real data): draw pairs of
    candidates in one node whose exact gains differ by a chosen relative gap
    and count how often each form ranks the better one above the worse one.
    Percent correct, at `lambda_l2 = 1`:

        parent/gain    gap 1e-6   1e-5   1e-4   1e-3     resolves at
          1  shipped      100     100    100    100      1e-6
             cross        100     100    100    100      1e-6
         30  shipped       40     100    100    100      1e-5
             cross         97     100    100    100      3e-6
        300  shipped       28      49    100    100      1e-4
             cross         67     100    100    100      1e-5
       2900  shipped       19      19     25    100      1e-3
             cross         52      75    100    100      1e-4

    Resolution ratios of 1, 3.3, 10, 10 against a bound predicting 1, 5.5,
    17, 54 on a grid whose steps are a factor of three: the bound is the
    right shape and is not tight. Note also that a form which cannot resolve
    a gap does not coin-flip. The shipped form *ties* the two candidates and
    the scan keeps its incumbent, which is why its scores sit near 19 percent
    rather than near 50; it does not fail at random, it defers to scan order.

    And the honest other end: at `parent / gain` below one this buys
    nothing, and on the median it is a few percent worse. Over each node's
    top 200 candidates at a centered node the shipped form's median relative
    error is 5.3e-08 against the cross form's 6.9e-08. There is no
    cancellation there to remove, and the cross form pays one more rounding
    for the privilege. The case for it is entirely in the one-sided regime.

    THE CONTRACTION, WHICH IS DELIBERATE AND NOT INCIDENTAL
    -------------------------------------------------------
    `GL*HR' - GR*HL'` is a product feeding a subtract, the shape
    `docs/NUMERICS.md` section 6 warns about, and this project has been
    bitten by it three times. Unfused is not expressible in this language
    and binding to a named local does not block contraction, so leaving it
    written as an infix expression would leave the result at the optimizer's
    discretion -- and this expression is evaluated both in a device kernel
    and in `reference_search` on the host, which is exactly the pair that
    must not diverge. `fma` is therefore explicit: one rounding for
    `right_g * hl`, one for the fused multiply-add, and nothing left to
    decide.

    **The exactness of this form does not depend on the fusion**, which is
    the trap worth naming. The `eps * |GL*HR'| / |D|` bound above holds for
    either contraction; fusing removes one of the two roundings inside `D`
    and so is the better of two acceptable spellings, not a load-bearing
    one. Measured in the same model, the fused and unfused arms score 97.8
    and 97.2 percent on the 3e-05 row above -- a real difference, and a
    small one.

    `left_g` and `right_g` arrive already soft-thresholded, as in
    `gpu_split_gain`. That is a formality here: `gpu_resolve_gain_form`
    refuses this arm whenever `lambda_l1` is nonzero, because the identity
    above is false under soft thresholding, and the reasoning is there.
    """
    var hl = left_h + child_l2
    var hr = right_h + child_l2
    var d = fma(left_g, hr, -(right_g * hl))
    return d * d / (hl * hr * node_s) - cross_offset


@always_inline
def gpu_cat_gain(
    left_g: Float32,
    left_h: Float32,
    right_g: Float32,
    right_h: Float32,
    lambda_l1: Float32,
    child_l2: Float32,
    parent_score: Float32,
    node_s: Float32,
    cross_offset: Float32,
    form: Int32,
) -> Float32:
    """A categorical candidate's gain under either form.

    The categorical searches never consult the monotone sign -- they scored
    `gpu_leaf_score` twice and subtracted the parent inline before this
    function existed -- so this is the whole of their gain arithmetic and
    the subtractive arm below is that inline expression, unchanged term for
    term. `left_g` and `right_g` arrive *un*-thresholded here, as they did
    inline, and `gpu_leaf_score` thresholds them; the ordinal path passes
    thresholded values to `gpu_split_gain` instead. The two conventions are
    pre-existing and are kept rather than unified, because unifying them
    would move the shipped arm's bits for no reason.
    """
    if form == Int32(GAIN_FORM_CROSS):
        return gpu_cross_gain(
            gpu_soft_threshold_l1(left_g, lambda_l1),
            left_h,
            gpu_soft_threshold_l1(right_g, lambda_l1),
            right_h,
            child_l2,
            node_s,
            cross_offset,
        )
    return (
        gpu_leaf_score(left_g, left_h, lambda_l1, child_l2)
        + gpu_leaf_score(right_g, right_h, lambda_l1, child_l2)
        - parent_score
    )


# --- score_function = Cosine ----------------------------------------------
#
# CatBoost's `score_function=Cosine`, `sum(-out * G) / sqrt(sum(out^2 * H))`
# over the children minus the same functional of the unsplit node, in the
# device's Float32. The specification is the CPU path and nothing here is an
# independent derivation: `split._cosine_out`, `split._cosine_unsplit`,
# `split._cosine_pair` and `split._cosine_score` are the four functions these
# four mirror, term for term and in the same order, and the CatBoost source
# each was read from is cited there rather than restated here.
#
# WHY THIS IS A SECOND FUNCTIONAL AND NOT A RELABELLING OF THE L2 KERNEL
# ----------------------------------------------------------------------
# Because the argmax coincidence is per parent and this module's answer is
# not consumed per parent. Substituting the free Newton step makes Cosine's
# numerator and denominator the same expression at `lambda_l2 = 0`, so the
# score collapses to `sqrt` of the L2 score and `sqrt` is strictly
# increasing -- within one node. The stock `grow_policy` is `lossguide`,
# which is a leaf-wise queue over candidates from *different* parents, and
# `sqrt(a) - sqrt(p)` does not order like `a - p` across two different `p`.
# The record this module returns carries `FREC_GAIN` into exactly that
# queue. So the identity is true, is stated at `split._cosine_pair`, and is
# not available here; scoring Cosine by relabelling the L2 kernel would be
# wrong at the default growth policy and wrong in a way no single-node test
# can see.
#
# WHAT IS AND IS NOT THE SAME NUMBER AS THE CPU'S
# ------------------------------------------------
# The candidate set, the admission guards, the scan order, the tie rule and
# the accumulation order are the same; see `gpu_cosine_gain`. The
# arithmetic is not, and the divergence is entirely the one this module
# already had before Cosine existed -- Float32 against the host's Float64,
# over a fixed-point histogram against the host's exact Float64 sums -- plus
# exactly one new elementary operation, the Float32 square root. That one
# is named because it is the only part of Cosine with no counterpart in the
# L2 gain and therefore the only part whose divergence this section
# introduces rather than inherits.

comptime GPU_COSINE_DEN_FLOOR = Float32(1.17549435082228750797e-38)
"""CatBoost's denominator seed, in the largest form Float32 can carry it.

`split._COSINE_DEN_FLOOR` is 1e-100, which is CatBoost's own
(`score_calcers.h`, `Scores.resize(splitsCount, {0, 1e-100})`). **1e-100 is
not representable in Float32 and rounds to zero**, so transcribing the
constant would silently delete the guard rather than port it, and the guard
is load-bearing: it is what turns the `0/0` a zero-gradient candidate
produces into `0`. `_cosine_out` returns 0.0 for a child of non-positive
weight and a child whose thresholded gradient is exactly zero gives a zero
numerator too, so `num` and `den` reach the ratio as `0` and `0` together.

The smallest positive normal Float32 is used instead. It is a divide-by-zero
guard and not a regularizer in either precision: adding it to any `den` a
`min_child_hess` of even 1e-9 admits is exactly a no-op under Float32
rounding, because such a `den` exceeds it by more than 2^24. The seed's
*value* therefore does not enter any candidate's score in either backend,
only its being nonzero does, and that property is what has been ported.
`min_child_hess` is what actually keeps H off the floor, on both sides."""


@always_inline
def gpu_cosine_out(g: Float32, h: Float32, lambda_l2: Float32) -> Float32:
    """`split._cosine_out` in Float32: CatBoost's `leafApprox` in our sign
    convention, with CatBoost's `CalcAverage` zero-weight guard kept.

    A child of non-positive weight emits zero and contributes nothing to
    either accumulator, rather than dividing by `lambda_l2` alone. That is
    why the Cosine branch has no divide-by-zero at `lambda_l2 = 0` where the
    L2 branch would produce an infinity, and the guard is the host's, not a
    device concession."""
    if not (h > Float32(0.0)):
        return Float32(0.0)
    return -g / (h + lambda_l2)


@always_inline
def gpu_cosine_score(num: Float32, den: Float32) -> Float32:
    """`split._cosine_score` in Float32: `Scores[i][0] / sqrt(Scores[i][1])`
    with the denominator seed folded in.

    One function for the same reason the host keeps one: the seed cannot then
    be applied twice or forgotten at one of the call sites.

    **This is the square root, and it is the one operation in the Cosine gain
    with no counterpart in the L2 gain.** IEEE-754 makes `sqrt` correctly
    rounded, so on any backend that honors the standard this is the same
    number the host replica computes; a backend compiling with a relaxed
    `sqrt` is the one place a device record could differ from the replica's
    by a last bit, and that possibility is stated at `gpu_cosine_gain`
    rather than assumed away."""
    return num / sqrt(den + GPU_COSINE_DEN_FLOOR)


@always_inline
def gpu_cosine_parent(
    total_g: Float32,
    total_h: Float32,
    lambda_l1: Float32,
    lambda_l2: Float32,
) -> Float32:
    """`split._cosine_unsplit` scored, which is the node constant the Cosine
    gain subtracts: CatBoost's `CalcScoreWithoutSplit`, the same calcer run
    over the node's own totals with an empty second child.

    Hoisted per node by every kernel, exactly as `parent_score` is, and for
    the same reason: it is constant across a node's candidates -- every
    feature's bins total to the same sums -- so subtracting it sets the zero
    point the `> best_gain` test measures from and does nothing else. The
    gradient arrives *un*-thresholded and is thresholded here, which is what
    `find_best_split` does when it passes `parent_g` to `_cosine_unsplit`."""
    var t = gpu_soft_threshold_l1(total_g, lambda_l1)
    var out = gpu_cosine_out(t, total_h, lambda_l2)
    return gpu_cosine_score(-out * t, out * out * total_h)


@always_inline
def gpu_cosine_gain(
    left_g: Float32,
    left_h: Float32,
    right_g: Float32,
    right_h: Float32,
    lambda_l2: Float32,
    parent_cos: Float32,
    sign: Int32,
    bound_lo: Float32,
    bound_hi: Float32,
    constrained: Bool,
) -> Float32:
    """One candidate's Cosine gain: `split._cosine_pair` folded into
    `split._cosine_score` and the node's unsplit score subtracted, which is
    the two lines `find_best_split` writes at each of its two scoring sites.

    `left_g` and `right_g` arrive already soft-thresholded, as they do at
    `gpu_split_gain`, and as `tl` and `tr` do on the host.

    A MONOTONE REJECTION IS 0.0 AND NOT A SENTINEL
    ----------------------------------------------
    The host carries it out of band, as `_CosineTerms.ok`, because 0.0 is a
    legitimate *numerator* there and cannot double as a rejection. By the
    time the value is a gain that ambiguity is gone: `find_best_split` writes
    `gain = 0.0` and overwrites it only when `ok`, so a rejected candidate
    scores exactly 0.0 and loses to a `best_gain` that starts at 0.0 under a
    strict `>`. Returning 0.0 here is therefore the host's rule and not a
    device shortcut, and it is the same rule `gpu_split_gain`'s constrained
    branch already follows for L2.

    Note what is subtracted and what is not: a rejection returns 0.0 flat,
    **not** `-parent_cos`. The two happen to be indistinguishable to the
    caller today -- `parent_cos` is `|G| / sqrt(H)` and therefore never
    negative, so `-parent_cos` is never positive and loses to a `best_gain`
    that starts at 0.0 exactly as 0.0 does. It is written the host's way
    anyway, because the equivalence rests on a sign argument about a
    quantity computed elsewhere and the host's spelling rests on nothing.

    WHAT IS THE SAME AS THE HOST, EXACTLY
    -------------------------------------
    The addend order in both accumulators is left child then right child,
    fixed here as it is fixed in `_cosine_pair`, so neither value moves with
    a worker count or a launch shape. The clamp is applied before the
    accumulators are built, so they are built from the output the leaf will
    actually emit, which is what CatBoost's monotone branch does. The two
    parameters `_cosine_pair` takes that this does not -- `finish` and its
    `max_delta_step` / `path_smooth` / parent-output arguments -- are refused
    for the whole device search by `device_search_eligibility`'s
    `SEARCH_EXTRA_PARAMS`, so there is no configuration in which the host
    applies them and this does not.

    WHAT IS NOT THE SAME, EXACTLY
    -----------------------------
    Three things, and they are worth separating because only one of them is
    new.

    1. Float32 against Float64. Every operation below is a Float32 operation
       and the host's is a Float64 one. Inherited: this is what every gain in
       this module already is.
    2. A fixed-point histogram against exact Float64 sums. The device's
       `left_g` was dequantized from an Int32 accumulation. Inherited, and it
       is the accumulation this module already had; `gpu_right_sum` is where
       its one avoidable cast lives and Cosine changes nothing about it.
    3. The square root, at `gpu_cosine_score`. **New**, and the only new one.
       It is correctly rounded under IEEE-754 and so agrees between the host
       replica and a conforming device; a backend that compiles `sqrt` in a
       relaxed mode would differ from the replica in the last bit, which no
       L2 record could ever do because no L2 record takes a root.

    The two accumulator adds are written as explicit `fma` for the reason
    `gpu_cross_gain` gives: a product feeding an add is the contractable
    shape (`docs/NUMERICS.md` section 6), unfused is not expressible and a
    named local does not block contraction, and this expression is evaluated
    both in the kernels and in `reference_search` on the host -- which is
    exactly the pair that must not diverge. That pins one rounding where the
    host's `+=` spelling leaves two, which is a divergence *in association*
    from the host and is recorded as one. It is strictly smaller than
    difference 1 above, which is already present in every term."""
    var left_out = gpu_cosine_out(left_g, left_h, lambda_l2)
    var right_out = gpu_cosine_out(right_g, right_h, lambda_l2)
    if constrained:
        left_out = gpu_clamp(left_out, bound_lo, bound_hi)
        right_out = gpu_clamp(right_out, bound_lo, bound_hi)
        if gpu_violates(sign, left_out, right_out):
            return Float32(0.0)
    var num = fma(-right_out, right_g, -left_out * left_g)
    var den = fma(right_out * right_out, right_h, left_out * left_out * left_h)
    return gpu_cosine_score(num, den) - parent_cos


@always_inline
def gpu_split_gain(
    left_g: Float32,
    left_h: Float32,
    right_g: Float32,
    right_h: Float32,
    lambda_l2: Float32,
    parent_score: Float32,
    sign: Int32,
    bound_lo: Float32,
    bound_hi: Float32,
    constrained: Bool,
    node_s: Float32 = Float32(0.0),
    cross_offset: Float32 = Float32(0.0),
    form: Int32 = Int32(GAIN_FORM_SUBTRACTIVE),
    score: Int32 = Int32(SCORE_L2),
    parent_cos: Float32 = Float32(0.0),
) -> Float32:
    """`split._split_gain` in Float32. `left_g` and `right_g` are already
    soft-thresholded; a candidate running against `sign` scores 0.0, which
    no caller accepts.

    The three trailing parameters select and feed the cancellation-free arm
    and default to the shipped one, so a caller that has no node constants
    to hand gets exactly the expression this function held before they
    existed.

    **The monotone-constrained branch keeps the subtraction and is not
    offered the cross arm.** Its gain is `output_score(left) +
    output_score(right) - parent_score` over *clamped* leaf outputs, which
    is a different expression that the identity does not cover: once an
    output is clamped it is no longer `-G/H'`, and the algebra that turns
    two quotients into one cross product has nothing to work with. That is
    acceptable rather than a gap. Clamping an output moves the gain by far
    more than the cancellation does, so a constrained candidate's gain was
    never resolved to `eps * parent_score` in the first place.

    `score` selects the functional, and `SCORE_L2` -- the default, and the
    value every existing caller gets without naming it -- is every line
    below. `SCORE_COSINE` leaves before any of them and takes none of the
    three trailing gain-form parameters with it: the cross form is an
    identity about `GL^2/HL' + GR^2/HR' - G^2/S` and there is no such
    identity for a ratio, so `node_s`, `cross_offset` and `form` are simply
    not part of Cosine's arithmetic. **They still matter to a Cosine scan,
    through `gpu_right_sum`**, which is where the right-hand child's sums
    come from and is upstream of this function under either functional.
    `parent_cos` is the node constant `gpu_cosine_parent` produced and is
    ignored under `SCORE_L2`, where it is not read at all.
    """
    if score == Int32(SCORE_COSINE):
        return gpu_cosine_gain(
            left_g,
            left_h,
            right_g,
            right_h,
            lambda_l2,
            parent_cos,
            sign,
            bound_lo,
            bound_hi,
            constrained,
        )
    if not constrained:
        if form == Int32(GAIN_FORM_CROSS):
            return gpu_cross_gain(
                left_g,
                left_h,
                right_g,
                right_h,
                lambda_l2,
                node_s,
                cross_offset,
            )
        return (
            left_g * left_g / (left_h + lambda_l2)
            + right_g * right_g / (right_h + lambda_l2)
            - parent_score
        )
    var left_out = gpu_clamp(
        -left_g / (left_h + lambda_l2), bound_lo, bound_hi
    )
    var right_out = gpu_clamp(
        -right_g / (right_h + lambda_l2), bound_lo, bound_hi
    )
    if gpu_violates(sign, left_out, right_out):
        return Float32(0.0)
    return (
        gpu_output_score(left_g, left_h, lambda_l2, left_out)
        + gpu_output_score(right_g, right_h, lambda_l2, right_out)
        - parent_score
    )


# --- random_strength: seeded noise on a candidate's gain ------------------
#
# CatBoost's one regularizer LightGBM has no equivalent of, on the device
# side. The rule, the formula, and the CatBoost source it was read from are
# `tree_parameters_extra.mojo`'s; this section owns only the part that has to
# survive the crossing to a Float32 accelerator, and the whole reason it is
# written the way it is:
#
#     the device and the host must pick the SAME split under the same seed.
#
# That is the feature. A stochastic split rule whose two backends disagree is
# worse than no stochastic split rule at all, because the disagreement is
# invisible and looks like noise by design. So the question this section
# answers is not "how do I draw a normal on a GPU" but "which half of the
# draw can be made bit-identical on both backends, and where does the other
# half have to live".
#
# WHAT IS BIT-IDENTICAL, AND WHAT CANNOT BE
# -----------------------------------------
# The draw is two stages:
#
#   A. key -> counter.  (seed, tree, node, feature, bin) folded through
#      splitmix64. Pure 64-bit integer arithmetic: exact and associative,
#      no rounding, no libm, no FMA to contract. This reproduces bit for bit
#      on the host, on Metal, on CUDA, at any `MOJOTREES_NUM_WORKERS`, and
#      `gpu_random_score_stream` below is the one definition of it.
#      `random_score_key_probe` runs it on the device and hands the words
#      back so a test can compare them to the host's, which
#      `tests/test_gpu_random_score_noise.mojo` does.
#
#   B. counter -> N(0, 1).  Marsaglia's polar method: `log`, `sqrt`, and a
#      rejection test, all in Float64. **This one cannot cross.** Apple GPUs
#      have no Float64 at all, and even where Float64 exists `log` is not a
#      correctly rounded operation, so a device evaluation of stage B is a
#      different number from the host's. The rejection test is what makes
#      that fatal rather than merely imprecise: `s = u*u + v*v` is accepted
#      only when `0 < s < 1`, so a pair that lands within an ulp of 1.0 can
#      be accepted on one backend and rejected on the other, and a rejected
#      pair advances the counter. The two backends then walk off onto
#      completely different draws, not draws a few ulps apart. Squeezing the
#      test into exact integers does not save it either: the host's own
#      comparison is against the *rounded* Float64 `s`, so an exact test is
#      the wrong test, not a better one.
#
# THEREFORE: STAGE B RUNS ON THE HOST, ONCE, AND THE DEVICE READS THE ANSWER
# --------------------------------------------------------------------------
# `random_score_plane` evaluates both stages on the host for one node's
# (feature, bin) candidates and returns one Float32 per candidate:
# `Float32(stdev * normal)`, a single rounding of exactly the Float64 number
# `split.find_best_split` adds. `GpuSplitSearcher.stage_random_score` uploads
# that plane and the scan kernels add `plane[slot, bin]` to the candidate's
# gain. The two backends therefore add *the same number*, and the only
# remaining difference between a noised device gain and a noised host gain is
# the Float32-versus-Float64 difference the gain itself already had. The
# noise introduces no new class of divergence, which is the strongest
# statement available on a device with no Float64.
#
# What that costs, stated plainly because it is not free: one Float32 per
# (feature, bin) candidate crossing host to device per node, which is one
# third of the histogram this module exists to stop moving, plus one host
# `log` and `sqrt` per candidate. It is paid only when `random_strength` is
# non-zero, which is never in LightGBM mode; at the default the buffer is not
# allocated, no plane is built, no byte crosses, and the kernels take the
# same instructions they took before this section existed.
#
# WHICH GAIN, AND WHICH SCORE FUNCTION
# ------------------------------------
# The noise is added to `gpu_split_gain`'s value: LightGBM's second-order
# gain `G^2/(H+lambda)`, which is CatBoost's `score_function=L2` shape, not
# CatBoost's default `Cosine`. CatBoost scales one `scoreStDev` by a
# derivative RMS, which is dimensionally a gradient and therefore pairs with
# `Cosine`; against a second-order gain (gradient^2/hessian) the same number
# is a different size, so the useful range of `random_strength` here is not
# the range CatBoost documents. That is CatBoost's own behavior under
# `score_function=L2` and it is the pairing `tree_parameters_extra.mojo`
# chose; this module reproduces the host's choice rather than making a
# second one. If a `Cosine` variant lands on the host, it lands here as
# another `gpu_split_gain` arm and the noise term below does not move.
#
# The noise is **independent of the gain form.** It is added to whatever
# `gpu_split_gain` returned, so `GAIN_FORM_CROSS` and
# `GAIN_FORM_SUBTRACTIVE` get the identical Float32 addend; the two arms'
# noised gains differ by exactly the amount their un-noised gains already
# differed by, and neither arm's noise had to be recalibrated. In particular
# the cross form's refusal of itself under `lambda_l1 != 0` (see
# `gpu_resolve_gain_form`) changes which gain the noise lands on and changes
# nothing about the noise.

comptime RANDOM_SCORE_DOMAIN = UInt64(0x52414E4453434F52)
"""Domain separator folded into the seed, ASCII "RANDSCOR", so this stream
can never coincide with `tree_parameters_extra.extra_split_stream`'s even
when both seeds are equal. Must equal
`tree_parameters_extra._RANDOM_SCORE_DOMAIN`."""

comptime OBLIVIOUS_SCORE_DOMAIN = UInt64(0x4F424C5653434F52)
"""The second domain separator, ASCII "OBLVSCOR", for a draw keyed to an
oblivious LEVEL rather than to a node. Must equal
`tree_parameters_extra._OBLIVIOUS_SCORE_DOMAIN`.

A depth and a node id are both small nonnegative integers, so one domain
would make an oblivious level at depth 0 and a leaf-wise node 0 draw the
identical value for the same (seed, tree, feature, bin). The second constant
makes the two streams disjoint as a property rather than as a convention."""

comptime RANDOM_SCORE_POLAR_MAX_TRIES = 64
"""Marsaglia's polar method rejects a pair outside the unit disc, which is
`1 - pi/4` of the plane. Sixty-four rejections in a row has probability about
3e-43; the bound exists so the loop is provably finite. Must equal
`tree_parameters_extra._POLAR_MAX_TRIES`."""


@always_inline
def gpu_random_score_stream(
    seed: Int, tree_index: Int, node: Int, feature: Int, bin: Int
) -> UInt64:
    """The counter key for one candidate's noise draw, keyed by
    (seed, tree, node, feature, bin) and by nothing else.

    **This is stage A, and it is the function that has to be identical on
    both backends.** It is pure 64-bit integer arithmetic, so it is: there is
    no rounding in it, no library call, and no multiply-add for a compiler to
    contract. It reads no counter that advances with evaluation order, so a
    candidate's key is the same value whether its feature was scanned first
    or last, whether it ran on its own threadgroup or shared one, on one host
    thread or on sixty-four device lanes.

    Byte for byte the construction `tree_parameters_extra.random_score_stream`
    uses, and it must stay that way: sign bits masked off so a negative seed
    is accepted, `node + 1` / `feature + 1` / `bin + 1` so that the lowest
    index is not the identity element of the xor, and the bin mixed in last.
    `tests/test_random_score_noise.mojo` pins the words this returns, which is
    what would catch either copy drifting; the one-line assertion that ties it
    directly to the host's copy is written out there and becomes live the
    moment the two branches meet.
    """
    return gpu_score_stream_in(
        RANDOM_SCORE_DOMAIN, seed, tree_index, node, feature, bin
    )


@always_inline
def gpu_score_stream_in(
    domain: UInt64,
    seed: Int,
    tree_index: Int,
    site: Int,
    feature: Int,
    bin: Int,
) -> UInt64:
    """Stage A itself, in `domain`. The device's copy of
    `tree_parameters_extra.score_stream_in`, and the ONE definition of the
    arithmetic on this side: both wrappers call it and differ only in the
    constant they pass.

    `site` is the term naming where in the tree the draw was taken -- a node
    id in `RANDOM_SCORE_DOMAIN`, a level depth in
    `OBLIVIOUS_SCORE_DOMAIN`. One parameter because it is one position in the
    key; the domain is what keeps the two readings of it disjoint.

    THE IRREGULARITY IS LOAD-BEARING: `tree_index` carries no `+1`, while
    `site`, `feature` and `bin` all do. A reimplementation that tidies that up
    desynchronizes the two backends silently."""
    var h = splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ domain)
    h = splitmix64(h ^ UInt64(tree_index & 0x7FFFFFFFFFFFFFFF))
    h = splitmix64(h ^ UInt64((site + 1) & 0x7FFFFFFFFFFFFFFF))
    h = splitmix64(h ^ UInt64((feature + 1) & 0x7FFFFFFFFFFFFFFF))
    return splitmix64(h ^ UInt64((bin + 1) & 0x7FFFFFFFFFFFFFFF))


@always_inline
def gpu_oblivious_score_stream(
    seed: Int, tree_index: Int, depth: Int, feature: Int, bin: Int
) -> UInt64:
    """The counter key for one `grow_policy=oblivious` LEVEL candidate's
    noise draw, keyed by (seed, tree, **depth**, feature, bin).

    Byte for byte `tree_parameters_extra.oblivious_score_stream`, and it must
    stay that way for the reason `gpu_random_score_stream` gives about its own
    twin: the cross-backend equality assertion is over these words.

    THE DEPTH TERM IS WHY THIS IS NOT CATBOOST'S MECHANISM, AND THAT IS
    DELIBERATE
    -------------------------------------------------------------------
    CatBoost redraws per level by ADVANCING a generator --
    `greedy_tensor_search.cpp:884` takes a fresh `GenRand()` inside
    `CalcScores`, which `:1199` calls inside the `curDepth` loop, while the
    standard deviation at `:1186` is drawn once per tree immediately before
    it. We reproduce the property (a candidate draws different noise at each
    depth) counter-based instead. An advancing generator is a function of
    iteration order and generator state and would diverge between the two
    backends the moment either changed the order or the worker count it
    visited depths with, silently, with both models still training. Keying on
    the depth makes the draw worker-independent by construction, which is the
    only form in which the CPU and the GPU can be asserted equal.

    **Do not "fix" this back into a running generator.**"""
    return gpu_score_stream_in(
        OBLIVIOUS_SCORE_DOMAIN, seed, tree_index, depth, feature, bin
    )


def host_standard_normal(stream: UInt64) -> Float64:
    """Stage B: a standard normal from one counter key, by Marsaglia's polar
    method. **Host only**, and deliberately so; see the section header.

    Byte for byte `tree_parameters_extra.standard_normal`, including the
    order of operations, because the number this returns is the number the
    host scan adds and any difference between the two copies is a
    backend disagreement wearing a rounding's clothes.

    On FMA, which this repository has been bitten by twice: `u * u + v * v`
    is a multiply-add and a compiler is free to contract it, and binding
    either product to a named local does not stop it. Nothing here depends on
    it *not* being contracted -- what it depends on is that this expression
    and `tree_parameters_extra.standard_normal`'s are the same source text
    compiled by the same toolchain, so they contract the same way or not at
    all. That is why this function is a copy rather than a rewrite, why the
    tests do not pin literal draw values (they would be pinning a contraction
    decision, and the standing rule is deterministic on a given toolchain,
    not identical to the past), and why the literals that *are* pinned are
    stage A's, which has no floating-point in it to contract.
    """
    var i = 0
    while i < RANDOM_SCORE_POLAR_MAX_TRIES:
        var base = stream + GOLDEN * UInt64(2 * i)
        var u = 2.0 * uniform(base) - 1.0
        var v = 2.0 * uniform(base + GOLDEN) - 1.0
        var s = u * u + v * v
        if s > 0.0 and s < 1.0:
            return u * sqrt(-2.0 * log(s) / s)
        i += 1
    return 0.0


@always_inline
def host_random_score_noise(
    stdev: Float64,
    seed: Int,
    tree_index: Int,
    node: Int,
    feature: Int,
    bin: Int,
    domain: UInt64 = RANDOM_SCORE_DOMAIN,
) -> Float32:
    """The number a candidate's gain is shifted by, as the device consumes
    it: `Float32(stdev * normal)`.

    `stdev` is CatBoost's `scoreStDev`, which on the host bundle is
    `ExtraTreeParams.random_score_stdev()` = `random_strength *
    random_score_scale`. The rounding to Float32 happens once, here, on the
    exact Float64 product the host scan adds; the device then adds that
    rounded value and the host adds the unrounded one, which is a difference
    of at most one Float32 ulp of the noise and is inside the Float32
    difference the gain already carries.

    At or below zero -- the default, and every LightGBM-mode fit -- this is
    exactly 0.0 and no stream is touched.

    `domain` selects which stream the key is taken in and `node` is read
    accordingly: `RANDOM_SCORE_DOMAIN` (the default, and the only value any
    caller passed before oblivious levels existed) reads it as a node id;
    `OBLIVIOUS_SCORE_DOMAIN` reads it as a level depth. A defaulted argument
    rather than a second function, so there is one place where stage A meets
    stage B and one rounding to Float32 for both readings.
    """
    if not (stdev > 0.0) or stdev > Float64.MAX_FINITE:
        return Float32(0.0)
    return Float32(
        stdev
        * host_standard_normal(
            gpu_score_stream_in(
                domain, seed, tree_index, node, feature, bin
            )
        )
    )


def random_score_plane(
    stdev: Float64,
    seed: Int,
    tree_index: Int,
    node: Int,
    features: List[Int],
    n_bins: Int,
    domain: UInt64 = RANDOM_SCORE_DOMAIN,
) raises -> List[Float32]:
    """One node's noise plane, `len(features) * n_bins` Float32 in
    (slot-major, bin-minor) order, which is the layout the scan kernels
    index.

    Keyed by the *global* feature id, not by the slot: reordering or
    narrowing a node's feature set permutes this plane and changes no value
    in it, which is what makes a per-node feature draw
    (`feature_fraction_bynode`) leave the noise alone.

    `node` is the node id and is required: a grower that does not pass its
    node ids would draw every node of a tree from the same stream, which is
    the refusal `ExtraTreeParams.needs_node_identity` states on the host, and
    a default of 0 standing in for a node id is exactly the failure it exists
    to prevent.

    THE ONE LINE THAT MOVES WHEN THE HOST LANE MERGES. The body calls
    `host_random_score_noise`, this module's copy of the host's draw. When
    `tree_parameters_extra.random_score_noise` is on the same branch, that
    call becomes a call to it and this module stops holding a second copy of
    stage B. Nothing else changes: the key is already the same construction,
    the layout is this function's, and the plane is the same numbers.
    """
    if n_bins < 1:
        raise Error("a noise plane needs at least one bin")
    if node < 0:
        raise Error(
            "random_strength keys its draw by node id, which must be"
            " nonnegative; a grower that cannot supply one cannot use it"
        )
    var out = List[Float32](capacity=len(features) * n_bins)
    for slot in range(len(features)):
        var f = features[slot]
        for b in range(n_bins):
            out.append(
                host_random_score_noise(
                    stdev, seed, tree_index, node, f, b, domain
                )
            )
    return out^


def oblivious_score_plane(
    stdev: Float64,
    seed: Int,
    tree_index: Int,
    depth: Int,
    features: List[Int],
    n_bins: Int,
) raises -> List[Float32]:
    """One oblivious LEVEL's noise plane: `random_score_plane` in
    `OBLIVIOUS_SCORE_DOMAIN`, with the level depth in the site position.

    The layout is the same (slot-major, bin-minor) plane the scan kernels
    index, and it is keyed by the *global* feature id for the same reason:
    reordering or narrowing the level's feature set permutes the plane and
    changes no value in it.

    One plane per level and not per leaf, because the noise is drawn on the
    level's aggregate score and the level takes one split.
    """
    if depth < 0:
        raise Error(
            "an oblivious level sits at a nonnegative depth, got ", depth
        )
    return random_score_plane(
        stdev,
        seed,
        tree_index,
        depth,
        features,
        n_bins,
        OBLIVIOUS_SCORE_DOMAIN,
    )


def _random_score_key_kernel(
    args: MutPointer[Int32, MutAnyOrigin],
    keys: MutPointer[Int32, MutAnyOrigin],
    n: Int32,
):
    """Run `gpu_random_score_stream` on the device and hand the 64-bit words
    back as (low, high) Int32 pairs.

    Two Int32 halves rather than a UInt64 buffer so the readback is the same
    four-byte element every other buffer in this module moves, and so the
    comparison a test makes is over words no float ever touched."""
    var i = Int(block_idx.x)
    if i >= Int(n):
        return
    var b = i * 5
    var key = gpu_random_score_stream(
        Int(args[unsafe_offset=b][0]),
        Int(args[unsafe_offset = b + 1][0]),
        Int(args[unsafe_offset = b + 2][0]),
        Int(args[unsafe_offset = b + 3][0]),
        Int(args[unsafe_offset = b + 4][0]),
    )
    keys[unsafe_offset = 2 * i] = (key & UInt64(0xFFFFFFFF)).cast[
        DType.int32
    ]()
    keys[unsafe_offset = 2 * i + 1] = (
        (key >> UInt64(32)) & UInt64(0xFFFFFFFF)
    ).cast[DType.int32]()


def random_score_key_probe(queries: List[Int]) raises -> List[UInt64]:
    """Evaluate stage A on the accelerator for a list of
    (seed, tree, node, feature, bin) tuples, flattened five ints to a tuple.

    This exists for one assertion and it is the assertion this lane was
    written to make: the device's key and the host's key are the same 64-bit
    word, bit for bit, for every tuple. Everything downstream of the key --
    the normal, the standard deviation, the gain it lands on -- is either
    host-computed or already covered by this module's Float32 contract, so if
    this holds and the plane is uploaded, the two backends noise the same
    candidate by the same amount.

    `tests/test_gpu_random_score_noise.mojo` is the caller. It is not on any
    training path and allocates its own context.
    """
    if len(queries) % 5 != 0:
        raise Error(
            "random_score_key_probe takes five ints per query"
            " (seed, tree, node, feature, bin), got ",
            len(queries),
        )
    var n = len(queries) // 5
    if n == 0:
        return List[UInt64]()
    comptime if not has_accelerator():
        raise Error(
            "random_score_key_probe needs an accelerator; this binary was"
            " built without one"
        )
    else:
        var ctx = DeviceContext()
        var args = ctx.enqueue_create_buffer[DType.int32](len(queries))
        var out = ctx.enqueue_create_buffer[DType.int32](2 * n)
        with args.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for i in range(len(queries)):
                dst.unsafe_store(i, Int32(queries[i]))
        ctx.enqueue_function[_random_score_key_kernel](
            args.unsafe_ptr(),
            out.unsafe_ptr(),
            Int32(n),
            grid_dim=n,
            block_dim=1,
        )
        ctx.synchronize()
        var words = List[UInt64](capacity=n)
        with out.map_to_host() as host:
            var src = host.unsafe_ptr()
            for i in range(n):
                var lo = UInt64(Int(src[unsafe_offset = 2 * i][0]) & 0xFFFFFFFF)
                var hi = UInt64(Int(src[unsafe_offset = 2 * i + 1][0]) & 0xFFFFFFFF)
                words.append(lo | (hi << UInt64(32)))
        return words^


def _oblivious_score_key_kernel(
    args: MutPointer[Int32, MutAnyOrigin],
    keys: MutPointer[Int32, MutAnyOrigin],
    n: Int32,
):
    """`_random_score_key_kernel` in `OBLIVIOUS_SCORE_DOMAIN`: run
    `gpu_oblivious_score_stream` on the device and hand the words back.

    A second kernel rather than a domain word threaded through the first,
    because the first's five-int query tuple is a contract
    `tests/test_gpu_random_score_noise.mojo` already writes against, and
    widening it would edit a passing test to make a new one compile."""
    var i = Int(block_idx.x)
    if i >= Int(n):
        return
    var b = i * 5
    var key = gpu_oblivious_score_stream(
        Int(args[unsafe_offset=b][0]),
        Int(args[unsafe_offset = b + 1][0]),
        Int(args[unsafe_offset = b + 2][0]),
        Int(args[unsafe_offset = b + 3][0]),
        Int(args[unsafe_offset = b + 4][0]),
    )
    keys[unsafe_offset = 2 * i] = (key & UInt64(0xFFFFFFFF)).cast[
        DType.int32
    ]()
    keys[unsafe_offset = 2 * i + 1] = (
        (key >> UInt64(32)) & UInt64(0xFFFFFFFF)
    ).cast[DType.int32]()


def oblivious_score_key_probe(queries: List[Int]) raises -> List[UInt64]:
    """Evaluate stage A on the accelerator for a list of
    (seed, tree, **depth**, feature, bin) tuples, flattened five ints to a
    tuple.

    `random_score_key_probe`'s twin in the level domain, and it exists for
    the same one assertion: the device's key and the host's key are the same
    64-bit word for every tuple. With that and the plane comparison, a level's
    noise is pinned end to end across the two backends.
    """
    if len(queries) % 5 != 0:
        raise Error(
            "oblivious_score_key_probe takes five ints per query"
            " (seed, tree, depth, feature, bin), got ",
            len(queries),
        )
    var n = len(queries) // 5
    if n == 0:
        return List[UInt64]()
    comptime if not has_accelerator():
        raise Error(
            "oblivious_score_key_probe needs an accelerator; this binary was"
            " built without one"
        )
    else:
        var ctx = DeviceContext()
        var args = ctx.enqueue_create_buffer[DType.int32](len(queries))
        var out = ctx.enqueue_create_buffer[DType.int32](2 * n)
        with args.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for i in range(len(queries)):
                dst.unsafe_store(i, Int32(queries[i]))
        ctx.enqueue_function[_oblivious_score_key_kernel](
            args.unsafe_ptr(),
            out.unsafe_ptr(),
            Int32(n),
            grid_dim=n,
            block_dim=1,
        )
        ctx.synchronize()
        var words = List[UInt64](capacity=n)
        with out.map_to_host() as host:
            var src = host.unsafe_ptr()
            for i in range(n):
                var lo = UInt64(Int(src[unsafe_offset = 2 * i][0]) & 0xFFFFFFFF)
                var hi = UInt64(
                    Int(src[unsafe_offset = 2 * i + 1][0]) & 0xFFFFFFFF
                )
                words.append(lo | (hi << UInt64(32)))
        return words^


# --- Collective primitives, and the switch that holds both arms -----------
#
# Mojo's `gpu.primitives.block` collectives (`sum`, `max`, `min`,
# `prefix_sum`) are supported on NVIDIA, AMD, and Apple Metal alike, so using
# them keeps this module's one-portable-source rule intact: there is still no
# per-backend code path here, and there are still no floating-point atomics.
#
# What they are allowed to replace is decided by associativity and nothing
# else. Every quantity these kernels accumulate along a feature's bins is
# fixed-point Int32, and integer addition is associative, so a tree-shaped
# `prefix_sum` or `sum` over those returns the serial walk's value bit for
# bit. The Float32 quantities are only ever *compared*, never summed across
# threads: `max` and `min` are associative and commutative on the values
# these kernels produce (no NaN, no signed zero), so a tree-shaped `max` over
# gains also returns the serial walk's value bit for bit. No collective in
# this module reassociates a floating-point sum, and none may: a gain is a
# difference of three Float32 quotients, and reassociating that would move a
# last bit and therefore, at a near tie, move a decision.

comptime NO_CANDIDATE = Int32(2147483647)
"""The identity a thread holding no candidate contributes to a `block.min`
over candidate positions. `Int32.MAX`, spelled out because every real
position is a candidate ordinal or a feature slot, both of which are far
below it."""

comptime REDUCE_SLOT_THREADS = 64
"""Threads per threadgroup in the primitive cross-feature reduction. A warp
multiple on every supported backend (it is `gpu_tiling.WARP_GRANULARITY`),
which is what `block.max` and `block.min` want; the collectives allocate
their own threadgroup scratch, one word per warp, so this kernel reserves no
shared memory of its own and raises no device floor."""


def split_primitives_requested() -> Bool:
    """`MOJOTREES_GPU_SPLIT_PRIMITIVES=0`, the switch back to the
    hand-rolled reductions.

    On unless refused, which is the opposite posture from
    `MOJOTREES_GPU_SPLIT_WIDE` and for a reason: the wide scan changes which
    kernel shape does the scanning, while the collectives change only how a
    reduction is spelled. Both arms return the same record by construction
    (integer sums and float maxima are both associative), and
    `tests/test_gpu_split_scan.mojo` asserts field for field that they do,
    so what the switch preserves is a measurement handle and an escape
    hatch, not a doubt about the answer.

    Read once, at construction, and stored on the searcher, where
    `GpuSplitSearcher.set_primitives` can override it. A benchmark that wants
    to hold both arms in one process sets the field rather than re-execing
    with a different environment, which is the same shape
    `MOJOTREES_GPU_SPLIT_RESIDENT` and `MOJOTREES_GPU_SPLIT_WIDE` already
    have: one environment variable that decides the default, one explicit
    handle that overrides it.
    """
    return getenv("MOJOTREES_GPU_SPLIT_PRIMITIVES") != "0"


def table_upload_hoisting_requested() -> Bool:
    """`MOJOTREES_GPU_SPLIT_TABLE_PACK=0`, the switch back to four separate
    per-table uploads.

    On unless refused, because the packed arm writes the device exactly the
    bytes the four-copy arm writes (`GpuSplitSearcher._copy_tables` argues
    that byte for byte) and differs only in how many times the host blocks
    to do it. Read once at construction and overridable per searcher through
    `GpuSplitSearcher.set_table_upload_hoisting`, the same one-variable /
    one-handle shape `MOJOTREES_GPU_SPLIT_PRIMITIVES` already has: an
    interleaved benchmark has to hold both arms inside one process and one
    thermal state, and re-execing with a different environment cannot do
    that on a machine whose device timings drift several-fold between time
    windows.
    """
    return getenv("MOJOTREES_GPU_SPLIT_TABLE_PACK") != "0"


def _require_readback_implemented(transport: Int) raises:
    """Refuse a transport `download_words` does not execute.

    Four of the seven rows in `gpu_runtime`'s table are reachable here: both
    pinned arms and both plain ones, which is every arm that is correct on
    Metal and cheap to reach once the record is one allocation. The other
    three are refused, and for two different reasons that are worth keeping
    apart.

    `READBACK_MAP` is refused because it is the slowest transport measured,
    at 349.47 us a trip against `plain_one`'s 124.85, so implementing it
    would add a `map_to_host` round trip per plane to a fit in exchange for
    nothing. The probe already executes it; that is where it belongs.

    The two `nosync` rows are refused because they are **wrong**, and they
    are refused twice over: `require_readback_correct` rejects them on Metal
    off the measured column, and this rejects them everywhere, because what
    makes them wrong elsewhere is unestablished rather than known to be
    false. A fit must not be the place that finds out.
    """
    if (
        transport == READBACK_PINNED_PAIR_SYNC
        or transport == READBACK_PINNED_ONE_SYNC
        or transport == READBACK_PLAIN_PAIR
        or transport == READBACK_PLAIN_ONE
    ):
        return
    if transport == READBACK_MAP:
        raise Error(
            "readback transport map is not implemented by the split searcher:",
            " it measured 349.47 us a trip against plain_one's 124.85, so it",
            " lives in probes/readback_cost.mojo and not on the split path",
        )
    raise Error(
        "readback transport ",
        readback_transport_name(transport),
        " is not implemented by the split searcher; it does not deliver the",
        " record on Metal and no backend has established that it delivers it",
        " anywhere. Reachable arms: pinned_pair_sync, pinned_one_sync,",
        " plain_pair, plain_one",
    )


# --- Kernels --------------------------------------------------------------


def _scan_slot_kernel(
    hist: MutPointer[Int32, MutAnyOrigin],
    node_tab: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    allow: MutPointer[Int32, MutAnyOrigin],
    missing: MutPointer[Int32, MutAnyOrigin],
    cat_n: MutPointer[Int32, MutAnyOrigin],
    mono: MutPointer[Int32, MutAnyOrigin],
    fparams: MutPointer[Float32, MutAnyOrigin],
    noise: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[Int32, MutAnyOrigin],
    out_f: MutPointer[Float32, MutAnyOrigin],
    n_bins: Int32,
    hist_size: Int32,
    record_base: Int32,
    feat_stride: Int32,
    min_data_in_leaf: Int32,
    constrained: Int32,
    cat_onehot_max: Int32,
    cat_max_threshold: Int32,
    cat_min_group: Int32,
    gain_form: Int32,
    noisy: Int32,
    score_function: Int32,
):
    """One threadgroup per (node, active feature slot): scan that feature's
    candidates for that node and write its best one as a per-slot record.

    The scan runs on one thread because the candidate order is the
    tie-breaking rule and a threshold scan is a prefix sum; features are the
    parallel dimension, and nodes are the second one. A launch covers
    records `[record_base, record_base + grid_dim.y)`, each with its own
    feature set, allow mask, float parameters, and histogram offset, which
    is what lets a whole frontier be scanned by one launch instead of one
    launch and one host wait per node. Every accumulation is exact
    fixed-point Int32, so the sums do not depend on either choice and a
    later per-bin parallel scan cannot change a result."""
    # Allocated at entry rather than inside the categorical branch, so the
    # threadgroup's shared allocation is unconditional and static.
    var keys = stack_allocation[
        MAX_SPLIT_BINS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var sorted_bins = stack_allocation[
        MAX_SPLIT_BINS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var slot = Int(block_idx.x)
    var record = Int(record_base) + Int(block_idx.y)
    var nt = record * NODE_WORDS
    # A launch is as wide as the widest node in the batch, so a node with a
    # narrower feature set leaves the tail slots alone. The reduction reads
    # only this node's own slots, so what those hold does not matter.
    if slot >= Int(node_tab[unsafe_offset = nt + NODE_SLOTS][0]):
        return
    var stride = Int(feat_stride)
    var table = record * stride
    var f = Int(feat_ids[unsafe_offset = table + slot][0])
    var nb = Int(n_bins)
    var hs = Int(hist_size)
    var base = Int(node_tab[unsafe_offset = nt + NODE_HIST_BASE][0]) + f * nb
    var io = (table + slot) * SPLIT_IWORDS
    var fo = (table + slot) * SPLIT_FWORDS
    var pf = record * PF_WORDS
    # This slot's row of the noise plane, one Float32 per bin, in the same
    # (record, slot) cell order every other per-slot table uses. Read only
    # when `noisy`; at the default the pointer is a one-element placeholder
    # and no lane touches it. See the `random_strength` section above for why
    # the value is uploaded rather than drawn here.
    var noise_base = (table + slot) * nb

    for i in range(SPLIT_IWORDS):
        out_i[unsafe_offset = io + i] = Int32(0)
    for i in range(SPLIT_FWORDS):
        out_f[unsafe_offset = fo + i] = Float32(0.0)
    out_i[unsafe_offset = io + IREC_FEATURE] = Int32(-1)
    out_i[unsafe_offset = io + IREC_BIN] = Int32(-1)
    out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(-1)

    var g_inv = fparams[unsafe_offset = pf + PF_G_INV][0]
    var h_inv = fparams[unsafe_offset = pf + PF_H_INV][0]
    var lambda_l2 = fparams[unsafe_offset = pf + PF_LAMBDA_L2][0]
    var lambda_l1 = fparams[unsafe_offset = pf + PF_LAMBDA_L1][0]
    var min_child_hess = fparams[unsafe_offset = pf + PF_MIN_CHILD_HESS][0]
    var bound_lo = fparams[unsafe_offset = pf + PF_BOUND_LO][0]
    var bound_hi = fparams[unsafe_offset = pf + PF_BOUND_HI][0]
    var cat_smooth = fparams[unsafe_offset = pf + PF_CAT_SMOOTH][0]
    var cat_l2 = fparams[unsafe_offset = pf + PF_CAT_L2][0]

    # Totals over this feature's bins. Every accumulated feature has the same
    # totals bit for bit, because a row contributes the same quantized value
    # to exactly one bin of each, so slot 0's copy is the one the reduction
    # takes the parent's leaf value from.
    var tg = Int32(0)
    var th = Int32(0)
    var tc = Int32(0)
    for b in range(nb):
        tg += hist[unsafe_offset = base + b][0]
        th += hist[unsafe_offset = hs + base + b][0]
        tc += hist[unsafe_offset = 2 * hs + base + b][0]
    var total_g = tg.cast[DType.float32]() * g_inv
    var total_h = th.cast[DType.float32]() * h_inv
    out_f[unsafe_offset = fo + FREC_TOTAL_GRAD] = total_g
    out_f[unsafe_offset = fo + FREC_TOTAL_HESS] = total_h
    out_i[unsafe_offset = io + IREC_TOTAL_COUNT] = tc

    # A feature the node's interaction constraints disallow is skipped before
    # any of its candidates is scored, exactly as in `find_best_split`.
    if allow[unsafe_offset = table + slot][0] == Int32(0):
        return

    var sign = Int32(MONOTONE_FREE)
    if constrained != Int32(0):
        sign = mono[unsafe_offset=f][0]
    var is_constrained = constrained != Int32(0)
    var parent_score = gpu_leaf_score(
        total_g, total_h, lambda_l1, lambda_l2
    )
    # The cross form's two node constants, hoisted here because that is what
    # they are: neither depends on the candidate. `gpu_resolve_gain_form`
    # sends L1 back to the subtractive arm, where both are ignored.
    var form = gpu_resolve_gain_form(gain_form, lambda_l1)
    var node_s = gpu_cross_node_s(total_h, lambda_l2)
    var cross_offset = gpu_cross_offset(
        total_g, total_h, lambda_l1, lambda_l2, lambda_l2, node_s
    )
    # Cosine's node constant, hoisted beside the other three because that is
    # what it is. Behind the selector, and the selector is read once per slot
    # into a value the whole scan holds constant, for the reason
    # `find_best_split` gives for reading it once per node: an L2 scan must
    # leave this kernel on exactly the instruction sequence it took before
    # the parameter existed, and the square root is the one operation here
    # that an L2 scan has never had to issue. See `gpu_cosine_parent`.
    var cosine = score_function == Int32(SCORE_COSINE)
    var parent_cos = Float32(0.0)
    if cosine:
        parent_cos = gpu_cosine_parent(
            total_g, total_h, lambda_l1, lambda_l2
        )

    var best_gain = Float32(0.0)
    # The best gain of every candidate this feature scored except the
    # winner, kept so the reduction can report the node's margin. Updated at
    # every acceptance site and nowhere else, so it costs one compare per
    # candidate and cannot change which candidate wins.
    var runner_gain = Float32(0.0)
    var best_bin = -1
    var best_ordinal = -1
    var best_default_left = False
    var best_left_g = Int32(0)
    var best_left_h = Int32(0)
    var best_left_c = Int32(0)
    var found = False
    var is_categorical = False

    var n_cat = Int(cat_n[unsafe_offset=f][0])
    if n_cat >= 2:
        # --- Category partition search ---------------------------------
        #
        # Bin 0 (missing, unseen, dropped) is never a member of a candidate
        # set, so those rows always route right and this feature's missing
        # bin plays no part.
        if n_cat <= Int(cat_onehot_max):
            # One-vs-rest: the single category goes left.
            for t in range(1, n_cat + 1):
                var lg = hist[unsafe_offset = base + t][0]
                var lh = hist[unsafe_offset = hs + base + t][0]
                var lc = hist[unsafe_offset = 2 * hs + base + t][0]
                if lc < min_data_in_leaf:
                    continue
                var lhf = lh.cast[DType.float32]() * h_inv
                if lhf < min_child_hess:
                    continue
                var rc = tc - lc
                if rc < min_data_in_leaf:
                    continue
                var rhf = gpu_right_sum(total_h, lhf, th, lh, h_inv, form)
                if rhf < min_child_hess:
                    continue
                var lgf = lg.cast[DType.float32]() * g_inv
                var rgf = gpu_right_sum(total_g, lgf, tg, lg, g_inv, form)
                var gain = gpu_cat_gain(
                    lgf,
                    lhf,
                    rgf,
                    rhf,
                    lambda_l1,
                    lambda_l2,
                    parent_score,
                    node_s,
                    cross_offset,
                    form,
                )
                if gain > best_gain:
                    runner_gain = best_gain
                    best_gain = gain
                    best_left_g = lg
                    best_left_h = lh
                    best_left_c = lc
                    found = True
                    is_categorical = True
                    for w in range(CAT_WORDS):
                        out_i[unsafe_offset = io + IREC_CAT0 + w] = Int32(0)
                    var word = io + IREC_CAT0 + (t // CAT_WORD_BITS)
                    out_i[unsafe_offset=word] = Int32(
                        1 << (t % CAT_WORD_BITS)
                    )
                elif gain > runner_gain:
                    runner_gain = gain
        else:
            # Many-vs-many over prefixes of the gradient/hessian ordering,
            # walked from both ends.
            var used = 0
            for t in range(1, n_cat + 1):
                var lc = hist[unsafe_offset = 2 * hs + base + t][0]
                if lc.cast[DType.float32]() < cat_smooth:
                    continue
                var lg = hist[unsafe_offset = base + t][0]
                var lh = hist[unsafe_offset = hs + base + t][0]
                keys[unsafe_offset=used] = (
                    lg.cast[DType.float32]()
                    * g_inv
                    / (lh.cast[DType.float32]() * h_inv + cat_smooth)
                )
                sorted_bins[unsafe_offset=used] = Int32(t)
                used += 1

            if used >= 2:
                # Stable ascending insertion sort, which is the ordering
                # `metrics._argsort` gives the host: ties keep the lower
                # category.
                for i in range(1, used):
                    var kv = keys[unsafe_offset=i][0]
                    var bv = sorted_bins[unsafe_offset=i][0]
                    var j = i - 1
                    while j >= 0 and keys[unsafe_offset=j][0] > kv:
                        keys[unsafe_offset = j + 1] = keys[unsafe_offset=j][0]
                        sorted_bins[unsafe_offset = j + 1] = sorted_bins[
                            unsafe_offset=j
                        ][0]
                        j -= 1
                    keys[unsafe_offset = j + 1] = kv
                    sorted_bins[unsafe_offset = j + 1] = bv

                var l2c = lambda_l2 + cat_l2
                # The many-vs-many walk scores children at `l2c` against a
                # parent scored at `lambda_l2`, so it needs its own pair of
                # node constants; `gpu_cross_offset` is general in exactly
                # that argument.
                var cat_s = gpu_cross_node_s(total_h, l2c)
                var cat_offset = gpu_cross_offset(
                    total_g, total_h, lambda_l1, lambda_l2, l2c, cat_s
                )
                var max_num_cat = Int(cat_max_threshold)
                if (used + 1) // 2 < max_num_cat:
                    max_num_cat = (used + 1) // 2
                var steps = used if used < max_num_cat else max_num_cat

                for d in range(2):
                    var direction = 1 if d == 0 else -1
                    var start_pos = 0 if d == 0 else used - 1
                    var pos = start_pos
                    var group = Int32(0)
                    var lg = Int32(0)
                    var lh = Int32(0)
                    var lc = Int32(0)
                    for i in range(steps):
                        var t = Int(sorted_bins[unsafe_offset=pos][0])
                        pos += direction
                        lg += hist[unsafe_offset = base + t][0]
                        lh += hist[unsafe_offset = hs + base + t][0]
                        var cnt = hist[unsafe_offset = 2 * hs + base + t][0]
                        lc += cnt
                        group += cnt

                        var lhf = lh.cast[DType.float32]() * h_inv
                        if lc < min_data_in_leaf or lhf < min_child_hess:
                            continue
                        var rc = tc - lc
                        if rc < min_data_in_leaf or rc < cat_min_group:
                            break
                        var rhf = gpu_right_sum(
                            total_h, lhf, th, lh, h_inv, form
                        )
                        if rhf < min_child_hess:
                            break
                        if group < cat_min_group:
                            continue
                        group = Int32(0)

                        var lgf = lg.cast[DType.float32]() * g_inv
                        var rgf = gpu_right_sum(
                            total_g, lgf, tg, lg, g_inv, form
                        )
                        var gain = gpu_cat_gain(
                            lgf,
                            lhf,
                            rgf,
                            rhf,
                            lambda_l1,
                            l2c,
                            parent_score,
                            cat_s,
                            cat_offset,
                            form,
                        )
                        if gain > best_gain:
                            runner_gain = best_gain
                            best_gain = gain
                            best_left_g = lg
                            best_left_h = lh
                            best_left_c = lc
                            found = True
                            is_categorical = True
                            for w in range(CAT_WORDS):
                                out_i[
                                    unsafe_offset = io + IREC_CAT0 + w
                                ] = Int32(0)
                            var p = start_pos
                            for _ in range(i + 1):
                                var m = Int(sorted_bins[unsafe_offset=p][0])
                                var word = (
                                    io + IREC_CAT0 + (m // CAT_WORD_BITS)
                                )
                                out_i[unsafe_offset=word] = out_i[
                                    unsafe_offset=word
                                ][0] | Int32(1 << (m % CAT_WORD_BITS))
                                p += direction
                        elif gain > runner_gain:
                            runner_gain = gain
    else:
        # --- Ordinal threshold scan ------------------------------------
        var missing_bin = Int(missing[unsafe_offset=f][0])
        var n_scan = missing_bin if missing_bin >= 0 else nb
        var miss_g = Int32(0)
        var miss_h = Int32(0)
        var miss_c = Int32(0)
        if missing_bin >= 0:
            miss_g = hist[unsafe_offset = base + missing_bin][0]
            miss_h = hist[unsafe_offset = hs + base + missing_bin][0]
            miss_c = hist[unsafe_offset = 2 * hs + base + missing_bin][0]

        var left_g = Int32(0)
        var left_h = Int32(0)
        var left_c = Int32(0)
        for b in range(n_scan):
            # The top threshold puts every ordinary bin left, so it is only a
            # split at all when missing rows fill the right child.
            if b == n_scan - 1 and miss_c == Int32(0):
                break
            left_g += hist[unsafe_offset = base + b][0]
            left_h += hist[unsafe_offset = hs + base + b][0]
            left_c += hist[unsafe_offset = 2 * hs + base + b][0]

            # This threshold's `random_strength` shift, read once and shared
            # by the two routing directions below, because the noise belongs
            # to the threshold and not to the direction. Sharing it is what
            # keeps LightGBM's rule intact: an exact tie between the two
            # directions still keeps `default_left`, since equal gains stay
            # equal after the same number is added to both. `split.mojo`
            # takes one draw per bin for the same reason.
            var bin_noise = Float32(0.0)
            if noisy != Int32(0):
                bin_noise = noise[unsafe_offset = noise_base + b][0]

            # Missing to the left, scored first so an exact tie keeps
            # default_left, as in LightGBM and as on the host.
            if missing_bin >= 0:
                var dl_g = left_g + miss_g
                var dl_h = left_h + miss_h
                var dl_c = left_c + miss_c
                var dl_hf = dl_h.cast[DType.float32]() * h_inv
                var dr_hf = gpu_right_sum(
                    total_h, dl_hf, th, dl_h, h_inv, form
                )
                if not (
                    dl_hf < min_child_hess
                    or dr_hf < min_child_hess
                    or dl_c < min_data_in_leaf
                    or tc - dl_c < min_data_in_leaf
                ):
                    var dl_gf = dl_g.cast[DType.float32]() * g_inv
                    var dr_gf = gpu_right_sum(
                        total_g, dl_gf, tg, dl_g, g_inv, form
                    )
                    var gain = gpu_split_gain(
                        gpu_soft_threshold_l1(dl_gf, lambda_l1),
                        dl_hf,
                        gpu_soft_threshold_l1(dr_gf, lambda_l1),
                        dr_hf,
                        lambda_l2,
                        parent_score,
                        sign,
                        bound_lo,
                        bound_hi,
                        is_constrained,
                        node_s,
                        cross_offset,
                        form,
                        score_function,
                        parent_cos,
                    )
                    if noisy != Int32(0):
                        gain += bin_noise
                    if gain > best_gain:
                        runner_gain = best_gain
                        best_gain = gain
                        best_bin = b
                        best_ordinal = 2 * b
                        best_default_left = True
                        best_left_g = dl_g
                        best_left_h = dl_h
                        best_left_c = dl_c
                        found = True
                    elif gain > runner_gain:
                        runner_gain = gain

            # Missing to the right. For a feature with no missing bin this is
            # the only candidate and the scan is exactly the ordinal one.
            if missing_bin < 0 or miss_c > Int32(0):
                var lhf = left_h.cast[DType.float32]() * h_inv
                var rhf = gpu_right_sum(
                    total_h, lhf, th, left_h, h_inv, form
                )
                if lhf < min_child_hess or rhf < min_child_hess:
                    continue
                if (
                    left_c < min_data_in_leaf
                    or tc - left_c < min_data_in_leaf
                ):
                    continue
                var lgf = left_g.cast[DType.float32]() * g_inv
                var rgf = gpu_right_sum(
                    total_g, lgf, tg, left_g, g_inv, form
                )
                var gain = gpu_split_gain(
                    gpu_soft_threshold_l1(lgf, lambda_l1),
                    lhf,
                    gpu_soft_threshold_l1(rgf, lambda_l1),
                    rhf,
                    lambda_l2,
                    parent_score,
                    sign,
                    bound_lo,
                    bound_hi,
                    is_constrained,
                    node_s,
                    cross_offset,
                    form,
                    score_function,
                    parent_cos,
                )
                if noisy != Int32(0):
                    gain += bin_noise
                if gain > best_gain:
                    runner_gain = best_gain
                    best_gain = gain
                    best_bin = b
                    best_ordinal = 2 * b + 1
                    best_default_left = False
                    best_left_g = left_g
                    best_left_h = left_h
                    best_left_c = left_c
                    found = True
                elif gain > runner_gain:
                    runner_gain = gain

    if not found:
        return

    var flags = Int32(FLAG_FOUND)
    if best_default_left:
        flags += Int32(FLAG_DEFAULT_LEFT)
    if is_categorical:
        flags += Int32(FLAG_CATEGORICAL)
    out_i[unsafe_offset = io + IREC_FEATURE] = Int32(f)
    out_i[unsafe_offset = io + IREC_BIN] = Int32(best_bin)
    out_i[unsafe_offset = io + IREC_FLAGS] = flags
    out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(best_ordinal)
    out_i[unsafe_offset = io + IREC_LEFT_COUNT] = best_left_c
    out_i[unsafe_offset = io + IREC_RIGHT_COUNT] = tc - best_left_c
    var lgf = best_left_g.cast[DType.float32]() * g_inv
    var lhf = best_left_h.cast[DType.float32]() * h_inv
    out_f[unsafe_offset = fo + FREC_GAIN] = best_gain
    out_f[unsafe_offset = fo + FREC_RUNNER_GAIN] = runner_gain
    out_f[unsafe_offset = fo + FREC_LEFT_GRAD] = lgf
    out_f[unsafe_offset = fo + FREC_LEFT_HESS] = lhf
    # The winner's child statistics take the same right-hand rule the gain
    # that selected it took, so the record cannot report a child the search
    # did not score. These feed the host's leaf values, which is a
    # leaf-channel quantity and cheap by `ACCURACY_BUDGET.md` section 2's
    # argument; the reason to make them exact is consistency, not accuracy.
    out_f[unsafe_offset = fo + FREC_RIGHT_GRAD] = gpu_right_sum(
        total_g, lgf, tg, best_left_g, g_inv, form
    )
    out_f[unsafe_offset = fo + FREC_RIGHT_HESS] = gpu_right_sum(
        total_h, lhf, th, best_left_h, h_inv, form
    )


# Threads per threadgroup in the wide ordinal scan. A warp multiple on every
# supported backend (it is `gpu_tiling.WARP_GRANULARITY`), and the width the
# shared-memory budget below is stated at.
comptime WIDE_SCAN_THREADS = 64

# Threadgroup memory `_scan_slot_wide_kernel` reserves: twelve
# `WIDE_SCAN_THREADS`-long Int32 or Float32 arrays, which at 64 threads is
# 3072 bytes, the same reservation the histogram kernels already make. The
# wide scan therefore raises no device floor, and
# `gpu_portability.MIN_SHARED_MEMORY_PER_BLOCK` covers it unchanged.
comptime WIDE_SCAN_SHARED_BYTES = 12 * WIDE_SCAN_THREADS * 4


def wide_scan_requested() -> Bool:
    """`MOJOTREES_GPU_SPLIT_WIDE=1`, the switch for the wide scan.

    Off unless asked for, which is this package's rule for a path no
    benchmark has priced rather than a doubt about the result: the wide
    kernel returns the serial kernel's records bit for bit (see
    `_scan_slot_wide_kernel`) and `tests/parallel/test_gpu_split_search.mojo`
    asserts that, so what is unmeasured is only whether it is faster. The
    scan is a small share of a split's cost on the one device this
    repository has run on -- `bench-launch-cost` prices a split's fixed
    overhead at roughly 280us -- so the honest expectation is a small win,
    and the default flips when a run says so and not before.
    """
    return getenv("MOJOTREES_GPU_SPLIT_WIDE") == "1"


def wide_scan_for(has_categorical: Bool) -> Bool:
    """Whether a searcher over this dataset scans wide: requested, and no
    categorical feature to scan.

    The categorical bar is the kernel's, not a policy: `_scan_slot_wide_kernel`
    implements the ordinal threshold scan and nothing else. Refusing per
    dataset rather than per feature keeps one kernel per launch, and a
    dataset that declares a categorical feature is one the serial kernel
    already serves correctly.
    """
    return wide_scan_requested() and not has_categorical


def _scan_slot_wide_kernel(
    hist: MutPointer[Int32, MutAnyOrigin],
    node_tab: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    allow: MutPointer[Int32, MutAnyOrigin],
    missing: MutPointer[Int32, MutAnyOrigin],
    mono: MutPointer[Int32, MutAnyOrigin],
    fparams: MutPointer[Float32, MutAnyOrigin],
    noise: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[Int32, MutAnyOrigin],
    out_f: MutPointer[Float32, MutAnyOrigin],
    n_bins: Int32,
    hist_size: Int32,
    record_base: Int32,
    feat_stride: Int32,
    min_data_in_leaf: Int32,
    constrained: Int32,
    gain_form: Int32,
    noisy: Int32,
    score_function: Int32,
):
    """`_scan_slot_kernel`'s ordinal scan, spread over a threadgroup instead
    of run on one thread, and it writes the same per-slot record.

    Same grid, `WIDE_SCAN_THREADS` threads to a threadgroup rather than one:
    `_scan_slot_kernel` puts a whole feature on a single lane because a
    threshold scan is a prefix sum, which is right about the dependency and
    wrong about it being serial. A prefix sum splits: each thread takes one
    contiguous chunk of the bins, the chunk sums are combined, and every
    thread starts its own walk from the exact sum of the chunks before it.

    Why the result is the serial one, bit for bit:

    - The running left sums are fixed-point Int32 and integer addition is
      associative, so a thread's starting sums are the ones the serial walk
      would have reached at that bin, whatever order the chunks were summed
      in. Every candidate is then scored by the same expressions over the
      same Float32 inputs.
    - The serial scan takes a candidate on a strict `>`, so its winner is
      the highest gain and, among equal gains, the earliest in scan order.
      Candidate order is the ordinal (`2 * bin` for missing-left, `2 * bin +
      1` for missing-right), which ascends with the scan, so the reduction
      below picks by gain and breaks ties on the lower ordinal and lands on
      the same candidate.
    - `runner_gain` is the second largest accepted gain counted with
      multiplicity, which is order independent, so merging each thread's top
      two is the whole-feature top two.

    Categorical features are not scanned here. Their many-vs-many search
    sorts categories by a gradient ratio and walks prefixes of that order,
    which is a different algorithm with its own scratch, and it stays in
    `_scan_slot_kernel`. `GpuSplitSearcher` refuses this kernel outright for
    a dataset that declares any categorical feature rather than branching
    per feature inside the launch, so nothing here can meet one.
    """
    # Totals reduction and chunk sums use separate arrays, so the read of the
    # totals and the write of the chunk sums need no barrier between them.
    var t_g = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var t_h = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var t_c = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_g = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_h = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_c = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var a_gain = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var a_runner = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var a_ord = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var a_lg = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var a_lh = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var a_lc = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var tid = Int(thread_idx.x)
    var slot = Int(block_idx.x)
    var record = Int(record_base) + Int(block_idx.y)
    var nt = record * NODE_WORDS
    # Every early return in this kernel is decided by the grid position, the
    # node table, or a per-feature table, never by `tid`, so a threadgroup
    # takes it whole and no barrier below is reached by a subset of it.
    if slot >= Int(node_tab[unsafe_offset = nt + NODE_SLOTS][0]):
        return
    var stride = Int(feat_stride)
    var table = record * stride
    var f = Int(feat_ids[unsafe_offset = table + slot][0])
    var nb = Int(n_bins)
    var hs = Int(hist_size)
    var base = Int(node_tab[unsafe_offset = nt + NODE_HIST_BASE][0]) + f * nb
    var io = (table + slot) * SPLIT_IWORDS
    var fo = (table + slot) * SPLIT_FWORDS
    var pf = record * PF_WORDS
    # This slot's row of the noise plane, one Float32 per bin, in the same
    # (record, slot) cell order every other per-slot table uses. Read only
    # when `noisy`; at the default the pointer is a one-element placeholder
    # and no lane touches it. See the `random_strength` section above for why
    # the value is uploaded rather than drawn here.
    var noise_base = (table + slot) * nb

    var pg = Int32(0)
    var ph = Int32(0)
    var pc = Int32(0)
    var bb = tid
    while bb < nb:
        pg += hist[unsafe_offset = base + bb][0]
        ph += hist[unsafe_offset = hs + base + bb][0]
        pc += hist[unsafe_offset = 2 * hs + base + bb][0]
        bb += WIDE_SCAN_THREADS
    t_g[unsafe_offset=tid] = pg
    t_h[unsafe_offset=tid] = ph
    t_c[unsafe_offset=tid] = pc
    barrier()
    var active = WIDE_SCAN_THREADS // 2
    while active > 0:
        if tid < active:
            t_g[unsafe_offset=tid] = (
                t_g[unsafe_offset=tid][0] + t_g[unsafe_offset = tid + active][0]
            )
            t_h[unsafe_offset=tid] = (
                t_h[unsafe_offset=tid][0] + t_h[unsafe_offset = tid + active][0]
            )
            t_c[unsafe_offset=tid] = (
                t_c[unsafe_offset=tid][0] + t_c[unsafe_offset = tid + active][0]
            )
        barrier()
        active //= 2
    var tg = t_g[unsafe_offset=0][0]
    var th = t_h[unsafe_offset=0][0]
    var tc = t_c[unsafe_offset=0][0]

    var g_inv = fparams[unsafe_offset = pf + PF_G_INV][0]
    var h_inv = fparams[unsafe_offset = pf + PF_H_INV][0]
    var lambda_l2 = fparams[unsafe_offset = pf + PF_LAMBDA_L2][0]
    var lambda_l1 = fparams[unsafe_offset = pf + PF_LAMBDA_L1][0]
    var min_child_hess = fparams[unsafe_offset = pf + PF_MIN_CHILD_HESS][0]
    var bound_lo = fparams[unsafe_offset = pf + PF_BOUND_LO][0]
    var bound_hi = fparams[unsafe_offset = pf + PF_BOUND_HI][0]
    var total_g = tg.cast[DType.float32]() * g_inv
    var total_h = th.cast[DType.float32]() * h_inv

    # The record belongs to one thread throughout: nothing else in the
    # threadgroup writes `out_i` or `out_f`, so the initial clear and the
    # final winner need no barrier between them and the scan.
    if tid == 0:
        for i in range(SPLIT_IWORDS):
            out_i[unsafe_offset = io + i] = Int32(0)
        for i in range(SPLIT_FWORDS):
            out_f[unsafe_offset = fo + i] = Float32(0.0)
        out_i[unsafe_offset = io + IREC_FEATURE] = Int32(-1)
        out_i[unsafe_offset = io + IREC_BIN] = Int32(-1)
        out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(-1)
        out_f[unsafe_offset = fo + FREC_TOTAL_GRAD] = total_g
        out_f[unsafe_offset = fo + FREC_TOTAL_HESS] = total_h
        out_i[unsafe_offset = io + IREC_TOTAL_COUNT] = tc

    # A feature the node's interaction constraints disallow is skipped before
    # any of its candidates is scored, exactly as in `find_best_split`.
    if allow[unsafe_offset = table + slot][0] == Int32(0):
        return

    var sign = Int32(MONOTONE_FREE)
    if constrained != Int32(0):
        sign = mono[unsafe_offset=f][0]
    var is_constrained = constrained != Int32(0)
    var parent_score = gpu_leaf_score(total_g, total_h, lambda_l1, lambda_l2)
    # The cross form's node constants; see `_scan_slot_kernel` for what they
    # are and `gpu_resolve_gain_form` for why L1 does not get this arm.
    var form = gpu_resolve_gain_form(gain_form, lambda_l1)
    var node_s = gpu_cross_node_s(total_h, lambda_l2)
    var cross_offset = gpu_cross_offset(
        total_g, total_h, lambda_l1, lambda_l2, lambda_l2, node_s
    )
    # Cosine's node constant, behind the selector for the reason
    # `_scan_slot_kernel` gives: an L2 scan must issue the instructions it
    # issued before this parameter existed, and the square root is the one
    # operation an L2 scan has never had to issue.
    var cosine = score_function == Int32(SCORE_COSINE)
    var parent_cos = Float32(0.0)
    if cosine:
        parent_cos = gpu_cosine_parent(
            total_g, total_h, lambda_l1, lambda_l2
        )

    var missing_bin = Int(missing[unsafe_offset=f][0])
    var n_scan = missing_bin if missing_bin >= 0 else nb
    if n_scan < 1:
        return
    var miss_g = Int32(0)
    var miss_h = Int32(0)
    var miss_c = Int32(0)
    if missing_bin >= 0:
        miss_g = hist[unsafe_offset = base + missing_bin][0]
        miss_h = hist[unsafe_offset = hs + base + missing_bin][0]
        miss_c = hist[unsafe_offset = 2 * hs + base + missing_bin][0]

    # One contiguous chunk of the scan per thread. `per` is the same for
    # every thread, so a chunk's start is its thread index times it and the
    # partition is a function of `n_scan` alone.
    var per = (n_scan + WIDE_SCAN_THREADS - 1) // WIDE_SCAN_THREADS
    var lo = tid * per
    if lo > n_scan:
        lo = n_scan
    var hi = lo + per
    if hi > n_scan:
        hi = n_scan

    var cg = Int32(0)
    var ch = Int32(0)
    var cc = Int32(0)
    for i in range(lo, hi):
        cg += hist[unsafe_offset = base + i][0]
        ch += hist[unsafe_offset = hs + base + i][0]
        cc += hist[unsafe_offset = 2 * hs + base + i][0]
    s_g[unsafe_offset=tid] = cg
    s_h[unsafe_offset=tid] = ch
    s_c[unsafe_offset=tid] = cc
    barrier()
    # Exclusive prefix over the chunk sums, summed low index first. Every
    # thread reads the same shared values in the same order, so this is one
    # integer sum with one answer.
    var left_g = Int32(0)
    var left_h = Int32(0)
    var left_c = Int32(0)
    for j in range(tid):
        left_g += s_g[unsafe_offset=j][0]
        left_h += s_h[unsafe_offset=j][0]
        left_c += s_c[unsafe_offset=j][0]

    var best_gain = Float32(0.0)
    var runner_gain = Float32(0.0)
    var best_ordinal = -1
    var best_left_g = Int32(0)
    var best_left_h = Int32(0)
    var best_left_c = Int32(0)

    for b in range(lo, hi):
        # The top threshold puts every ordinary bin left, so it is only a
        # split at all when missing rows fill the right child. Serial breaks
        # out of the loop here; the bin is the last one either way, so
        # skipping it is the same thing.
        if b == n_scan - 1 and miss_c == Int32(0):
            continue
        left_g += hist[unsafe_offset = base + b][0]
        left_h += hist[unsafe_offset = hs + base + b][0]
        left_c += hist[unsafe_offset = 2 * hs + base + b][0]

        # This threshold's `random_strength` shift, read once and shared by
        # the two routing directions; see `_scan_slot_kernel`. Keyed by bin
        # and not by scan position, which is the whole reason a wide scan can
        # carry it at all: a thread that starts in the middle of the bin
        # range reads the same number for its bins that a serial walk would
        # have reached them with.
        var bin_noise = Float32(0.0)
        if noisy != Int32(0):
            bin_noise = noise[unsafe_offset = noise_base + b][0]

        # Missing to the left, scored first so an exact tie keeps
        # default_left, as in LightGBM and as on the host.
        if missing_bin >= 0:
            var dl_g = left_g + miss_g
            var dl_h = left_h + miss_h
            var dl_c = left_c + miss_c
            var dl_hf = dl_h.cast[DType.float32]() * h_inv
            var dr_hf = gpu_right_sum(total_h, dl_hf, th, dl_h, h_inv, form)
            if not (
                dl_hf < min_child_hess
                or dr_hf < min_child_hess
                or dl_c < min_data_in_leaf
                or tc - dl_c < min_data_in_leaf
            ):
                var dl_gf = dl_g.cast[DType.float32]() * g_inv
                var dr_gf = gpu_right_sum(
                    total_g, dl_gf, tg, dl_g, g_inv, form
                )
                var gain = gpu_split_gain(
                    gpu_soft_threshold_l1(dl_gf, lambda_l1),
                    dl_hf,
                    gpu_soft_threshold_l1(dr_gf, lambda_l1),
                    dr_hf,
                    lambda_l2,
                    parent_score,
                    sign,
                    bound_lo,
                    bound_hi,
                    is_constrained,
                    node_s,
                    cross_offset,
                    form,
                    score_function,
                    parent_cos,
                )
                if noisy != Int32(0):
                    gain += bin_noise
                if gain > best_gain:
                    runner_gain = best_gain
                    best_gain = gain
                    best_ordinal = 2 * b
                    best_left_g = dl_g
                    best_left_h = dl_h
                    best_left_c = dl_c
                elif gain > runner_gain:
                    runner_gain = gain

        # Missing to the right. For a feature with no missing bin this is
        # the only candidate and the scan is exactly the ordinal one.
        if missing_bin < 0 or miss_c > Int32(0):
            var lhf = left_h.cast[DType.float32]() * h_inv
            var rhf = gpu_right_sum(total_h, lhf, th, left_h, h_inv, form)
            if lhf < min_child_hess or rhf < min_child_hess:
                continue
            if left_c < min_data_in_leaf or tc - left_c < min_data_in_leaf:
                continue
            var lgf = left_g.cast[DType.float32]() * g_inv
            var rgf = gpu_right_sum(total_g, lgf, tg, left_g, g_inv, form)
            var gain = gpu_split_gain(
                gpu_soft_threshold_l1(lgf, lambda_l1),
                lhf,
                gpu_soft_threshold_l1(rgf, lambda_l1),
                rhf,
                lambda_l2,
                parent_score,
                sign,
                bound_lo,
                bound_hi,
                is_constrained,
                node_s,
                cross_offset,
                form,
                score_function,
                parent_cos,
            )
            if noisy != Int32(0):
                gain += bin_noise
            if gain > best_gain:
                runner_gain = best_gain
                best_gain = gain
                best_ordinal = 2 * b + 1
                best_left_g = left_g
                best_left_h = left_h
                best_left_c = left_c
            elif gain > runner_gain:
                runner_gain = gain

    a_gain[unsafe_offset=tid] = best_gain
    a_runner[unsafe_offset=tid] = runner_gain
    a_ord[unsafe_offset=tid] = Int32(best_ordinal)
    a_lg[unsafe_offset=tid] = best_left_g
    a_lh[unsafe_offset=tid] = best_left_h
    a_lc[unsafe_offset=tid] = best_left_c
    barrier()
    if tid != 0:
        return

    # Winner: highest gain, ties to the lower ordinal, which is what the
    # serial scan's strict `>` over an ascending candidate order gives.
    var win = -1
    for j in range(WIDE_SCAN_THREADS):
        if a_gain[unsafe_offset=j][0] <= Float32(0.0):
            continue
        if win < 0:
            win = j
        elif a_gain[unsafe_offset=j][0] > a_gain[unsafe_offset=win][0]:
            win = j
        elif (
            a_gain[unsafe_offset=j][0] == a_gain[unsafe_offset=win][0]
            and a_ord[unsafe_offset=j][0] < a_ord[unsafe_offset=win][0]
        ):
            win = j
    if win < 0:
        return

    # Top two of the union of the per-thread top twos, which is the top two
    # of every accepted gain: the serial `runner_gain` counted with
    # multiplicity.
    var m1 = Float32(0.0)
    var m2 = Float32(0.0)
    for j in range(WIDE_SCAN_THREADS):
        var v = a_gain[unsafe_offset=j][0]
        if v > m1:
            m2 = m1
            m1 = v
        elif v > m2:
            m2 = v
        var u = a_runner[unsafe_offset=j][0]
        if u > m1:
            m2 = m1
            m1 = u
        elif u > m2:
            m2 = u

    var ordinal = Int(a_ord[unsafe_offset=win][0])
    var flags = Int32(FLAG_FOUND)
    if ordinal % 2 == 0:
        flags += Int32(FLAG_DEFAULT_LEFT)
    var won_left_c = a_lc[unsafe_offset=win][0]
    out_i[unsafe_offset = io + IREC_FEATURE] = Int32(f)
    out_i[unsafe_offset = io + IREC_BIN] = Int32(ordinal // 2)
    out_i[unsafe_offset = io + IREC_FLAGS] = flags
    out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(ordinal)
    out_i[unsafe_offset = io + IREC_LEFT_COUNT] = won_left_c
    out_i[unsafe_offset = io + IREC_RIGHT_COUNT] = tc - won_left_c
    var won_lg = a_lg[unsafe_offset=win][0]
    var won_lh = a_lh[unsafe_offset=win][0]
    var lgf = won_lg.cast[DType.float32]() * g_inv
    var lhf = won_lh.cast[DType.float32]() * h_inv
    out_f[unsafe_offset = fo + FREC_GAIN] = m1
    out_f[unsafe_offset = fo + FREC_RUNNER_GAIN] = m2
    out_f[unsafe_offset = fo + FREC_LEFT_GRAD] = lgf
    out_f[unsafe_offset = fo + FREC_LEFT_HESS] = lhf
    # The same right-hand rule the winning gain used; see `_scan_slot_kernel`.
    out_f[unsafe_offset = fo + FREC_RIGHT_GRAD] = gpu_right_sum(
        total_g, lgf, tg, won_lg, g_inv, form
    )
    out_f[unsafe_offset = fo + FREC_RIGHT_HESS] = gpu_right_sum(
        total_h, lhf, th, won_lh, h_inv, form
    )


def _scan_slot_wide_primitive_kernel(
    hist: MutPointer[Int32, MutAnyOrigin],
    node_tab: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    allow: MutPointer[Int32, MutAnyOrigin],
    missing: MutPointer[Int32, MutAnyOrigin],
    mono: MutPointer[Int32, MutAnyOrigin],
    fparams: MutPointer[Float32, MutAnyOrigin],
    noise: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[Int32, MutAnyOrigin],
    out_f: MutPointer[Float32, MutAnyOrigin],
    n_bins: Int32,
    hist_size: Int32,
    record_base: Int32,
    feat_stride: Int32,
    min_data_in_leaf: Int32,
    constrained: Int32,
    gain_form: Int32,
    noisy: Int32,
    score_function: Int32,
):
    """`_scan_slot_wide_kernel` with its four hand-rolled reductions written
    as `gpu.primitives.block` collectives, and returning its record bit for
    bit.

    The four, and why each substitution is exact:

    - The feature's totals were a strided partial sum per thread followed by
      a shared-memory halving tree. They are now `block.sum` over the same
      per-thread partials. The accumulated quantity is fixed-point Int32 and
      integer addition is associative, so the tree the collective happens to
      use returns the same word the halving tree did.
    - The running left sums across chunk boundaries were an exclusive prefix
      each thread computed by walking every lower thread's chunk sum out of
      shared memory. They are now `block.prefix_sum[exclusive=True]` over
      the same chunk sums. Int32 again, so again exact.
    - The winner across threads was a serial walk on thread 0 over a shared
      array of per-thread bests, taking the highest gain and, among equal
      gains, the lowest candidate ordinal. It is now `block.max` over the
      gains followed by `block.min` over the ordinals of the threads holding
      the maximum. `max` and `min` reassociate exactly on these values, and
      the pair (gain, ordinal) reproduces the serial rule exactly, because
      candidate ordinals ascend with the scan and are unique across threads:
      one thread and only one holds the maximum gain at the minimum ordinal,
      and it is the thread the serial walk would have stopped on.
    - The node's runner-up was a serial top-two merge over the same shared
      array. `runner_gain` is the second largest, counted with multiplicity,
      of the union of every thread's best and every thread's own runner-up.
      Removing one occurrence of the maximum is the same as excluding the
      winning thread's best, so the second largest is
      `max(max over non-winning threads of their best, max over all threads
      of their runner-up)`, which is two more `block.max` calls.

    Not a substitution: nothing here sums a Float32 across threads, and
    nothing may. A gain is a difference of three Float32 quotients, and a
    tree-shaped float sum would move its last bit, which at a near tie is a
    different split and not a different rounding.

    Everything else -- the candidate order inside a chunk, the missing-left
    before missing-right ordering, the `min_data_in_leaf` and
    `min_sum_hessian_in_leaf` gates, the monotone rejection, the top
    threshold rule -- is copied unchanged from `_scan_slot_wide_kernel`,
    because those are the semantics and not the reduction."""
    # The winning thread's fixed-point left sums, published once. Allocated
    # at entry so the threadgroup's shared reservation is unconditional and
    # static, as in the kernels above; the collectives allocate their own
    # scratch on top of this, one word per warp per reduction.
    var won = stack_allocation[
        3,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var tid = Int(thread_idx.x)
    var slot = Int(block_idx.x)
    var record = Int(record_base) + Int(block_idx.y)
    var nt = record * NODE_WORDS
    # Every early return in this kernel is decided by the grid position, the
    # node table, or a per-feature table, never by `tid`, so a threadgroup
    # takes it whole and no collective below is reached by a subset of it.
    # That is the same rule the hand-rolled barriers needed, and the
    # collectives need it for the same reason.
    if slot >= Int(node_tab[unsafe_offset = nt + NODE_SLOTS][0]):
        return
    var stride = Int(feat_stride)
    var table = record * stride
    var f = Int(feat_ids[unsafe_offset = table + slot][0])
    var nb = Int(n_bins)
    var hs = Int(hist_size)
    var base = Int(node_tab[unsafe_offset = nt + NODE_HIST_BASE][0]) + f * nb
    var io = (table + slot) * SPLIT_IWORDS
    var fo = (table + slot) * SPLIT_FWORDS
    var pf = record * PF_WORDS
    # This slot's row of the noise plane, one Float32 per bin, in the same
    # (record, slot) cell order every other per-slot table uses. Read only
    # when `noisy`; at the default the pointer is a one-element placeholder
    # and no lane touches it. See the `random_strength` section above for why
    # the value is uploaded rather than drawn here.
    var noise_base = (table + slot) * nb

    var pg = Int32(0)
    var ph = Int32(0)
    var pc = Int32(0)
    var bb = tid
    while bb < nb:
        pg += hist[unsafe_offset = base + bb][0]
        ph += hist[unsafe_offset = hs + base + bb][0]
        pc += hist[unsafe_offset = 2 * hs + base + bb][0]
        bb += WIDE_SCAN_THREADS
    var tg = block.sum[block_size=WIDE_SCAN_THREADS](pg)
    var th = block.sum[block_size=WIDE_SCAN_THREADS](ph)
    var tc = block.sum[block_size=WIDE_SCAN_THREADS](pc)

    var g_inv = fparams[unsafe_offset = pf + PF_G_INV][0]
    var h_inv = fparams[unsafe_offset = pf + PF_H_INV][0]
    var lambda_l2 = fparams[unsafe_offset = pf + PF_LAMBDA_L2][0]
    var lambda_l1 = fparams[unsafe_offset = pf + PF_LAMBDA_L1][0]
    var min_child_hess = fparams[unsafe_offset = pf + PF_MIN_CHILD_HESS][0]
    var bound_lo = fparams[unsafe_offset = pf + PF_BOUND_LO][0]
    var bound_hi = fparams[unsafe_offset = pf + PF_BOUND_HI][0]
    var total_g = tg.cast[DType.float32]() * g_inv
    var total_h = th.cast[DType.float32]() * h_inv

    # The record belongs to one thread throughout: nothing else in the
    # threadgroup writes `out_i` or `out_f`, so the initial clear and the
    # final winner need no barrier between them and the scan.
    if tid == 0:
        for i in range(SPLIT_IWORDS):
            out_i[unsafe_offset = io + i] = Int32(0)
        for i in range(SPLIT_FWORDS):
            out_f[unsafe_offset = fo + i] = Float32(0.0)
        out_i[unsafe_offset = io + IREC_FEATURE] = Int32(-1)
        out_i[unsafe_offset = io + IREC_BIN] = Int32(-1)
        out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(-1)
        out_f[unsafe_offset = fo + FREC_TOTAL_GRAD] = total_g
        out_f[unsafe_offset = fo + FREC_TOTAL_HESS] = total_h
        out_i[unsafe_offset = io + IREC_TOTAL_COUNT] = tc

    # A feature the node's interaction constraints disallow is skipped before
    # any of its candidates is scored, exactly as in `find_best_split`.
    if allow[unsafe_offset = table + slot][0] == Int32(0):
        return

    var sign = Int32(MONOTONE_FREE)
    if constrained != Int32(0):
        sign = mono[unsafe_offset=f][0]
    var is_constrained = constrained != Int32(0)
    var parent_score = gpu_leaf_score(total_g, total_h, lambda_l1, lambda_l2)
    # The cross form's node constants; see `_scan_slot_kernel` for what they
    # are and `gpu_resolve_gain_form` for why L1 does not get this arm.
    var form = gpu_resolve_gain_form(gain_form, lambda_l1)
    var node_s = gpu_cross_node_s(total_h, lambda_l2)
    var cross_offset = gpu_cross_offset(
        total_g, total_h, lambda_l1, lambda_l2, lambda_l2, node_s
    )
    # Cosine's node constant, behind the selector for the reason
    # `_scan_slot_kernel` gives: an L2 scan must issue the instructions it
    # issued before this parameter existed, and the square root is the one
    # operation an L2 scan has never had to issue.
    var cosine = score_function == Int32(SCORE_COSINE)
    var parent_cos = Float32(0.0)
    if cosine:
        parent_cos = gpu_cosine_parent(
            total_g, total_h, lambda_l1, lambda_l2
        )

    var missing_bin = Int(missing[unsafe_offset=f][0])
    var n_scan = missing_bin if missing_bin >= 0 else nb
    if n_scan < 1:
        return
    var miss_g = Int32(0)
    var miss_h = Int32(0)
    var miss_c = Int32(0)
    if missing_bin >= 0:
        miss_g = hist[unsafe_offset = base + missing_bin][0]
        miss_h = hist[unsafe_offset = hs + base + missing_bin][0]
        miss_c = hist[unsafe_offset = 2 * hs + base + missing_bin][0]

    # One contiguous chunk of the scan per thread. `per` is the same for
    # every thread, so a chunk's start is its thread index times it and the
    # partition is a function of `n_scan` alone.
    var per = (n_scan + WIDE_SCAN_THREADS - 1) // WIDE_SCAN_THREADS
    var lo = tid * per
    if lo > n_scan:
        lo = n_scan
    var hi = lo + per
    if hi > n_scan:
        hi = n_scan

    var cg = Int32(0)
    var ch = Int32(0)
    var cc = Int32(0)
    for i in range(lo, hi):
        cg += hist[unsafe_offset = base + i][0]
        ch += hist[unsafe_offset = hs + base + i][0]
        cc += hist[unsafe_offset = 2 * hs + base + i][0]
    # Exclusive prefix over the chunk sums. Fixed-point Int32, so the
    # collective's tree and the serial low-index-first walk it replaces
    # agree word for word.
    var left_g = block.prefix_sum[
        block_size=WIDE_SCAN_THREADS, exclusive=True
    ](cg)
    var left_h = block.prefix_sum[
        block_size=WIDE_SCAN_THREADS, exclusive=True
    ](ch)
    var left_c = block.prefix_sum[
        block_size=WIDE_SCAN_THREADS, exclusive=True
    ](cc)

    var best_gain = Float32(0.0)
    var runner_gain = Float32(0.0)
    var best_ordinal = -1
    var best_left_g = Int32(0)
    var best_left_h = Int32(0)
    var best_left_c = Int32(0)

    for b in range(lo, hi):
        # The top threshold puts every ordinary bin left, so it is only a
        # split at all when missing rows fill the right child. Serial breaks
        # out of the loop here; the bin is the last one either way, so
        # skipping it is the same thing.
        if b == n_scan - 1 and miss_c == Int32(0):
            continue
        left_g += hist[unsafe_offset = base + b][0]
        left_h += hist[unsafe_offset = hs + base + b][0]
        left_c += hist[unsafe_offset = 2 * hs + base + b][0]

        # This threshold's `random_strength` shift, read once and shared by
        # the two routing directions; see `_scan_slot_kernel`. Keyed by bin
        # and not by scan position, which is the whole reason a wide scan can
        # carry it at all: a thread that starts in the middle of the bin
        # range reads the same number for its bins that a serial walk would
        # have reached them with.
        var bin_noise = Float32(0.0)
        if noisy != Int32(0):
            bin_noise = noise[unsafe_offset = noise_base + b][0]

        # Missing to the left, scored first so an exact tie keeps
        # default_left, as in LightGBM and as on the host.
        if missing_bin >= 0:
            var dl_g = left_g + miss_g
            var dl_h = left_h + miss_h
            var dl_c = left_c + miss_c
            var dl_hf = dl_h.cast[DType.float32]() * h_inv
            var dr_hf = gpu_right_sum(total_h, dl_hf, th, dl_h, h_inv, form)
            if not (
                dl_hf < min_child_hess
                or dr_hf < min_child_hess
                or dl_c < min_data_in_leaf
                or tc - dl_c < min_data_in_leaf
            ):
                var dl_gf = dl_g.cast[DType.float32]() * g_inv
                var dr_gf = gpu_right_sum(
                    total_g, dl_gf, tg, dl_g, g_inv, form
                )
                var gain = gpu_split_gain(
                    gpu_soft_threshold_l1(dl_gf, lambda_l1),
                    dl_hf,
                    gpu_soft_threshold_l1(dr_gf, lambda_l1),
                    dr_hf,
                    lambda_l2,
                    parent_score,
                    sign,
                    bound_lo,
                    bound_hi,
                    is_constrained,
                    node_s,
                    cross_offset,
                    form,
                    score_function,
                    parent_cos,
                )
                if noisy != Int32(0):
                    gain += bin_noise
                if gain > best_gain:
                    runner_gain = best_gain
                    best_gain = gain
                    best_ordinal = 2 * b
                    best_left_g = dl_g
                    best_left_h = dl_h
                    best_left_c = dl_c
                elif gain > runner_gain:
                    runner_gain = gain

        # Missing to the right. For a feature with no missing bin this is
        # the only candidate and the scan is exactly the ordinal one.
        if missing_bin < 0 or miss_c > Int32(0):
            var lhf = left_h.cast[DType.float32]() * h_inv
            var rhf = gpu_right_sum(total_h, lhf, th, left_h, h_inv, form)
            if lhf < min_child_hess or rhf < min_child_hess:
                continue
            if left_c < min_data_in_leaf or tc - left_c < min_data_in_leaf:
                continue
            var lgf = left_g.cast[DType.float32]() * g_inv
            var rgf = gpu_right_sum(total_g, lgf, tg, left_g, g_inv, form)
            var gain = gpu_split_gain(
                gpu_soft_threshold_l1(lgf, lambda_l1),
                lhf,
                gpu_soft_threshold_l1(rgf, lambda_l1),
                rhf,
                lambda_l2,
                parent_score,
                sign,
                bound_lo,
                bound_hi,
                is_constrained,
                node_s,
                cross_offset,
                form,
                score_function,
                parent_cos,
            )
            if noisy != Int32(0):
                gain += bin_noise
            if gain > best_gain:
                runner_gain = best_gain
                best_gain = gain
                best_ordinal = 2 * b + 1
                best_left_g = left_g
                best_left_h = left_h
                best_left_c = left_c
            elif gain > runner_gain:
                runner_gain = gain

    # Winner: highest gain, ties to the lower ordinal. Every accepted gain
    # is strictly positive (a thread's `best_gain` starts at zero and only a
    # strictly greater candidate replaces it), so zero is a safe identity
    # for a thread that found nothing, and a maximum of zero means the whole
    # threadgroup found nothing, which is the serial kernel's `win < 0`.
    var top = block.max[block_size=WIDE_SCAN_THREADS](best_gain)
    if top <= Float32(0.0):
        return
    var my_ordinal = NO_CANDIDATE
    if best_gain == top:
        my_ordinal = Int32(best_ordinal)
    var win_ordinal = block.min[block_size=WIDE_SCAN_THREADS](my_ordinal)
    # Candidate ordinals are unique across threads, because the chunks are
    # disjoint ranges of bins, so exactly one thread satisfies both halves.
    var is_winner = best_gain == top and Int32(best_ordinal) == win_ordinal
    if is_winner:
        won[unsafe_offset=0] = best_left_g
        won[unsafe_offset=1] = best_left_h
        won[unsafe_offset=2] = best_left_c

    # `runner_gain` for the whole feature is the second largest, counted
    # with multiplicity, of every thread's best together with every thread's
    # own runner-up. One occurrence of the maximum is exactly the winning
    # thread's best, so excluding that thread from the first maximum and
    # folding in the maximum runner-up gives the serial merge's answer.
    var excluding_winner = best_gain
    if is_winner:
        excluding_winner = Float32(0.0)
    var other_best = block.max[block_size=WIDE_SCAN_THREADS](
        excluding_winner
    )
    var best_runner = block.max[block_size=WIDE_SCAN_THREADS](runner_gain)
    # The collectives fence threadgroup memory themselves, but the write to
    # `won` above is this kernel's own and is spelled out rather than
    # inferred from theirs.
    barrier()
    if tid != 0:
        return

    var m2 = other_best if other_best > best_runner else best_runner
    var ordinal = Int(win_ordinal)
    var flags = Int32(FLAG_FOUND)
    if ordinal % 2 == 0:
        flags += Int32(FLAG_DEFAULT_LEFT)
    var won_left_c = won[unsafe_offset=2][0]
    out_i[unsafe_offset = io + IREC_FEATURE] = Int32(f)
    out_i[unsafe_offset = io + IREC_BIN] = Int32(ordinal // 2)
    out_i[unsafe_offset = io + IREC_FLAGS] = flags
    out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(ordinal)
    out_i[unsafe_offset = io + IREC_LEFT_COUNT] = won_left_c
    out_i[unsafe_offset = io + IREC_RIGHT_COUNT] = tc - won_left_c
    var won_lg = won[unsafe_offset=0][0]
    var won_lh = won[unsafe_offset=1][0]
    var lgf = won_lg.cast[DType.float32]() * g_inv
    var lhf = won_lh.cast[DType.float32]() * h_inv
    out_f[unsafe_offset = fo + FREC_GAIN] = top
    out_f[unsafe_offset = fo + FREC_RUNNER_GAIN] = m2
    out_f[unsafe_offset = fo + FREC_LEFT_GRAD] = lgf
    out_f[unsafe_offset = fo + FREC_LEFT_HESS] = lhf
    # The same right-hand rule the winning gain used; see `_scan_slot_kernel`.
    out_f[unsafe_offset = fo + FREC_RIGHT_GRAD] = gpu_right_sum(
        total_g, lgf, tg, won_lg, g_inv, form
    )
    out_f[unsafe_offset = fo + FREC_RIGHT_HESS] = gpu_right_sum(
        total_h, lhf, th, won_lh, h_inv, form
    )


# --- grow_policy = oblivious: the cross-leaf reduction ---------------------
#
# See `docs/design/OBLIVIOUS.md` section B2. An oblivious tree chooses one
# (feature, threshold, missing direction) per *level* and applies it to every
# leaf of that level, so the quantity being maximized is not a leaf's gain but
#
#     score(f, b, dir) = sum over the level's leaves l of gain_l(f, b, dir)
#
# with each leaf using its own left/right sums. Gain is not additive across
# histograms, so this cannot be computed from a merged histogram and the level
# genuinely needs one histogram per leaf.
#
# **Why this is a kernel and not a launch.** The lane's own precondition is
# that the cross-leaf reduction be FUSED into the existing per-level search
# launch and never become a command buffer of its own. On Metal one
# `enqueue_function` is one command buffer and the queue is 64 deep
# (`docs/GPU_PORTABILITY.md` 6.2); the static census in
# `gpu_resident_round.oblivious_launch_census` shows depth 6 landing at 62
# buffers per tree fused and 68 standalone, so a standalone reduce puts the
# tree past the measured knee and the queue-depth argument for oblivious
# evaporates at CatBoost's own default depth. The fusion here is the cheapest
# possible one: the sum over leaves is the innermost loop of the scan that
# already runs, and the launch count is unchanged at two -- this scan, then
# `_reduce_slots_kernel` over one record instead of over a frontier.

comptime OBLIVIOUS_MAX_LEAVES = 64
"""Leaves in one oblivious level this kernel will scan, which is `2 ** 6` and
therefore CatBoost's default depth exactly.

A bound rather than a preference: the per-leaf scan state below lives in a
threadgroup allocation sized at compile time, and at fifteen words a leaf that
is 3,840 bytes, against the 3,072 `WIDE_SCAN_SHARED_BYTES` already reserves.
Depth 7 would double it and also lands at 71 command buffers per tree, over
the queue's knee whatever this kernel does, so the two limits agree about
where to stop.

Thirteen of the fifteen words are the L2 scan's and were there before Cosine.
The two Cosine adds -- `un_num` and `un_den`, a leaf's unsplit accumulator
terms -- are reserved unconditionally rather than behind the selector, because
a threadgroup allocation is sized at compile time and a kernel argument is not
a compile-time value. They are written only under `SCORE_COSINE`; an L2 level
pays 512 bytes of address space and no instruction."""


# --- score_function = Cosine, across a level -------------------------------
#
# `gpu_cosine_gain` above scores ONE node: it folds a candidate's two
# accumulator terms and takes the square root on the spot. A level cannot use
# it. CatBoost keeps `numScoreBlocks = 1` for `SymmetricTree`
# (`catboost/cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp`)
# precisely because the two accumulators are summed across every leaf of the
# level and ONE ratio is taken per candidate, so a level's score is
#
#     sum_l num_l / sqrt(sum_l den_l)   minus the same over the unsplit leaves
#
# and NOT `sum_l (num_l / sqrt(den_l))`. Gain is additive across leaves; a
# ratio is not. That is the whole of why `_scan_slot_oblivious_kernel`'s leaf
# loop cannot simply call `gpu_cosine_gain` where it calls `gpu_split_gain`,
# and it is why the two functions below exist: they return the TERMS, and the
# root is taken once, after the leaf loop closes.
#
# `split.find_best_split_shared` is the specification and these mirror its
# `den_left` / `den_right` planes statement for statement -- the same
# accumulation order (leaves ascending, innermost), the same substitution for
# an illegal leaf, the same single subtraction of the level's unsplit score at
# the end. What differs from the host is what already differed before Cosine
# and one thing more, both named at `gpu_cosine_gain`: Float32 over a
# fixed-point histogram rather than Float64 over exact sums (inherited), and
# the square root (new to Cosine, correctly rounded under IEEE-754).


@always_inline
def gpu_cosine_unsplit(
    total_g: Float32,
    total_h: Float32,
    lambda_l1: Float32,
    lambda_l2: Float32,
) -> SIMD[DType.float32, 2]:
    """`split._cosine_unsplit`'s two terms, `(num, den)`, in Float32.

    CatBoost's `CalcScoreWithoutSplit` (`leafwise_scoring.cpp`): the same
    calcer run over a leaf's own totals with an empty second child. The
    gradient arrives *un*-thresholded and is thresholded here, which is what
    `find_best_split_shared` does when it passes `parent_g` to
    `_cosine_unsplit`.

    This is the same arithmetic `gpu_cosine_parent` performs, stopped one step
    earlier. `gpu_cosine_parent` scores a node on the spot because a node's
    unsplit score is what a node's gain subtracts; a level's is the score of
    the SUM of these terms over its leaves, so the level needs them unscored
    and cannot reuse that function. Two entry points on one expression rather
    than one entry point that returns the wrong shape to one of its callers.

    The terms are needed twice and are computed once per leaf: they are the
    level's zero point, accumulated over the leaves into the constant the
    level's gain subtracts, AND they are what a leaf contributes at a
    candidate it cannot take. `find_best_split_shared` computes `pt` once per
    leaf and uses it for both, and the kernel caches it per leaf for the same
    reason."""
    var t = gpu_soft_threshold_l1(total_g, lambda_l1)
    var out = gpu_cosine_out(t, total_h, lambda_l2)
    return SIMD[DType.float32, 2](-out * t, out * out * total_h)


@always_inline
def gpu_cosine_level_terms(
    left_g: Float32,
    left_h: Float32,
    right_g: Float32,
    right_h: Float32,
    lambda_l2: Float32,
    sign: Int32,
    bound_lo: Float32,
    bound_hi: Float32,
    constrained: Bool,
    unsplit: SIMD[DType.float32, 2],
) -> SIMD[DType.float32, 2]:
    """One leaf's contribution to a level's two Cosine accumulators:
    `split._cosine_pair`, with the caller's `ok`-handling folded in.

    `left_g` and `right_g` arrive already soft-thresholded, as they do at
    `gpu_split_gain` and as `tl` and `tr` do on the host.

    THE ILLEGAL-LEAF SUBSTITUTION IS AN ARGUMENT, NOT A PARAMETER
    -------------------------------------------------------------
    The host carries the monotone rejection out of band as `_CosineTerms.ok`
    and lets each caller decide: `find_best_split` drops the candidate, and
    `find_best_split_shared` writes `cn = ct.num if ct.ok else pt.num`. Under
    a level there is exactly one right answer and it is the second, so this
    takes the leaf's unsplit terms as an argument and returns them rather than
    returning a flag a caller could get wrong. **This is where the L2 arm and
    the Cosine arm genuinely differ**: the L2 arm adds `0.0` for a leaf that
    cannot take the candidate, and 0.0 is a legitimate *numerator* here, so
    the ratio's only way to say "this leaf stays as it is" is to add its
    unsplit terms to both accumulators. The two are arithmetically the same
    rule -- `sum over legal of (child - parent)` is identically
    `(sum over legal of child + sum over illegal of parent) - sum over all of
    parent` -- and the ratio admits only the second spelling.

    The kernel applies the same substitution at the `min_data_in_leaf` /
    `min_child_hess` rejection, which does not reach this function at all: it
    is tested before the terms are formed, exactly as the host tests it before
    calling `_cosine_pair`.

    The clamp is applied before the accumulators are built, so they are built
    from the output the leaf will actually emit, which is what CatBoost's
    monotone branch does and what `_cosine_pair` does.

    The two accumulator adds are written as explicit `fma` for the reason
    `gpu_cosine_gain` gives, and they are the same two expressions it writes,
    so the per-node and the per-level arms round a candidate's terms
    identically. That pins one rounding where the host's `+=` spelling leaves
    two, which is a divergence in association from the host and is recorded as
    one; it is strictly smaller than the Float32-against-Float64 difference
    already present in every term."""
    var left_out = gpu_cosine_out(left_g, left_h, lambda_l2)
    var right_out = gpu_cosine_out(right_g, right_h, lambda_l2)
    if constrained:
        left_out = gpu_clamp(left_out, bound_lo, bound_hi)
        right_out = gpu_clamp(right_out, bound_lo, bound_hi)
        if gpu_violates(sign, left_out, right_out):
            return unsplit
    var num = fma(-right_out, right_g, -left_out * left_g)
    var den = fma(right_out * right_out, right_h, left_out * left_out * left_h)
    return SIMD[DType.float32, 2](num, den)


def _scan_slot_oblivious_kernel(
    hist: MutPointer[Int32, MutAnyOrigin],
    node_tab: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    allow: MutPointer[Int32, MutAnyOrigin],
    missing: MutPointer[Int32, MutAnyOrigin],
    cat_n: MutPointer[Int32, MutAnyOrigin],
    mono: MutPointer[Int32, MutAnyOrigin],
    fparams: MutPointer[Float32, MutAnyOrigin],
    noise: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[Int32, MutAnyOrigin],
    out_f: MutPointer[Float32, MutAnyOrigin],
    n_bins: Int32,
    hist_size: Int32,
    level_record: Int32,
    leaf_base: Int32,
    n_leaves: Int32,
    feat_stride: Int32,
    min_data_in_leaf: Int32,
    constrained: Int32,
    gain_form: Int32,
    score_function: Int32,
    noisy: Int32,
):
    """One threadgroup per active feature slot: scan that feature's candidates
    for a whole oblivious *level* and write the level's best one as a per-slot
    record, so that the existing cross-feature reduction folds a level exactly
    as it folds a node.

    The level's leaves are records `[leaf_base, leaf_base + n_leaves)`, and the
    only thing this kernel reads per leaf is that record's `NODE_HIST_BASE`.
    Everything else -- the active feature list, the allow mask, the float
    parameter block, the monotone vector -- is read from `level_record`, which
    is correct rather than merely convenient under the resident plane's
    refusals: a node's feature set is the tree's feature set, its allow mask is
    "everything", and its output interval is unbounded, so all three are
    tree-level and are staged once per tree. `_launch_child_search` relies on
    the same three facts and says so.

    The output is written where a one-node search would have written it, at
    `(level_record * feat_stride + slot)`, so `_reduce_slots_kernel` or
    `_reduce_slots_block_kernel` launched over `record_base = level_record,
    n_records = 1` folds these slots into `level_record`'s record with no
    change at all. That reduction walks slots in ascending order and accepts a
    new best only on a strictly greater gain, which is the second half of the
    tie rule below.

    **The summation order is part of the answer and is fixed here.** Float32
    addition is not associative, so a cross-leaf sum computed by a block
    collective and one computed by a loop are different numbers, and this mode's
    accuracy gate is node-identity against the CPU oblivious grower with no
    tolerance. The sum is therefore taken **serially, in ascending leaf record
    order**, which is the order a host loop over the level's frontier takes, and
    the leaf loop is the innermost of the three. That is also why features and
    not leaves are the parallel dimension, which is the same argument
    `_scan_slot_kernel` makes for its own single thread: the candidate order is
    the tie-breaking rule and a threshold scan is a prefix sum. A wide form of
    this kernel is possible and is deliberately not written, because it would
    have to agree with the CPU lane about a tree-reduction order before it could
    be bit-identical to anything, and no such agreement exists.

    **Candidate order and ties.** Bins ascend, and within a bin the
    missing-left candidate is scored before missing-right, exactly as in
    `_scan_slot_kernel`, so an exact tie keeps `default_left` as it does in
    LightGBM and on the host. A new candidate is accepted only on a strictly
    greater summed score, so the winner is the first candidate of the lowest
    bin holding the maximum, and the reduction downstream extends that to the
    first such feature.

    **A leaf that cannot take the candidate contributes exactly zero.** The
    legality tests -- `min_data_in_leaf` on both children and
    `min_child_hess` on both children -- are applied per leaf, against that
    leaf's own sums, because that is where they mean anything: an oblivious
    split is one split but it makes 2^d children and each has to be a legal
    leaf on its own. A leaf that fails adds `0.0` and the candidate stays
    available to the rest of the level. Under `SCORE_COSINE` "contributes
    nothing" is spelled differently and means the same thing; see below.

    **`score_function` selects which functional the level maximizes, and it
    changes the shape of the accumulation and not only its arithmetic.**
    `SCORE_L2` is every line this kernel had before the parameter existed:
    one accumulator, `total += gpu_split_gain(...)` over the leaves. A
    `SCORE_COSINE` level cannot be written that way, because CatBoost's
    Cosine score is a ratio and a level's ratio is not the sum of its leaves'
    ratios. CatBoost keeps `numScoreBlocks = 1` for `SymmetricTree`
    (`greedy_search_helper.cpp`) for exactly this reason: the two accumulators
    are summed across every leaf of the level and ONE square root is taken per
    candidate. So the Cosine arm carries `cos_num` and `cos_den` where the L2
    arm carries `total`, folds the leaves into both in the same ascending
    order, and calls `gpu_cosine_score` once, after the leaf loop closes --
    which is `split.find_best_split_shared`'s `den_left` / `den_right` planes,
    kernel-side. Two accumulators instead of one is the whole cost; the leaf
    loop is the same loop, in the same launch, and the launch count per level
    is unchanged at two.

    **Under Cosine an illegal leaf contributes its UNSPLIT terms, not zero,
    and that is the same rule rather than a different one.** `0.0` is a
    legitimate numerator once the sum is a ratio, so it cannot double as "this
    leaf stays as it is". The identity is
    `sum over legal of (child - parent) == (sum over legal of child + sum over
    illegal of parent) - sum over all of parent`, and the ratio admits only
    the second spelling. `find_best_split_shared` substitutes `pt.num` /
    `pt.den` at exactly the two sites this kernel substitutes `un_num[l]` /
    `un_den[l]`: the minimum-rejection, and the monotone rejection inside
    `gpu_cosine_level_terms`.

    **The level's unsplit score is subtracted once, at the end.** It is
    accumulated leaf by leaf in the setup loop below, in the same ascending
    order, and is constant across this feature's candidates -- every feature's
    bins total to the same per-leaf sums -- so subtracting it sets the zero
    point the `> best_gain` test measures from and does nothing else. Exactly
    what `level_parent` is in `find_best_split_shared`.

    **The top-threshold rule still needs no branch under Cosine**, which is
    not obvious and is the case worth checking. A candidate every leaf refuses
    accumulates precisely the level's own unsplit terms, in precisely the
    order `level_parent` was accumulated in, so it scores `level_parent -
    level_parent` -- an exact `0.0` in Float32, not a near one -- and `0.0`
    never beats a `best_gain` that starts at `0.0` under a strict `>`. The
    same holds at both minimums zero, where a full-left candidate's left child
    IS the parent and its right child is empty: `gpu_cosine_out` returns 0 for
    a child of non-positive weight, so the pair's terms reduce to the unsplit
    terms term for term. The L2 arm's argument for the same property is two
    paragraphs down and is the same argument in a different functional.

    **This is OUR rule, not CatBoost's, and the distinction was established
    2026-08-16 rather than assumed.** An earlier draft of this docstring called
    it "CatBoost's rule ... marked verify". It was then checked: **CatBoost
    applies `min_data_in_leaf` only to `Depthwise` and `Lossguide`, and
    `SymmetricTree` carries no such constraint at all.** So there is no
    CatBoost rule here to match, and the zero-contribution treatment is a
    decision this project is making because our own `min_data_in_leaf` and
    `min_child_hess` exist and have to mean something under a shared split.

    That matters for how it may be defended. It cannot be justified by "it is
    what CatBoost does"; it has to be justified on its own terms, which are:
    an oblivious split makes 2^d children, each has to be a legal leaf, and
    refusing the whole candidate because one leaf of sixty-four fails would let
    a single small leaf veto the level. It is written down in one place so the
    CPU grower matches it rather than guessing twice.

    Note what the zero does *not* do: it does not make an illegal candidate
    win. A candidate every leaf refuses scores exactly `0.0`, and `0.0` never
    beats the initial best, so a level with no admissible candidate anywhere
    writes no record and the commit reads `FLAG_FOUND` clear, exactly as a node
    with no admissible split does today.

    **The top-threshold rule is not special-cased and does not need to be.**
    `_scan_slot_kernel` breaks out of the scan at the last ordinary bin when
    the feature reserves no missing rows, because putting every row left is not
    a split. Here the candidate set has to be identical across the level, so
    the break cannot be taken per leaf; instead such a leaf fails the
    legality test on its empty right child -- zero rows is below any
    `min_data_in_leaf` of one or more, and zero hessian is below any positive
    `min_child_hess` -- and contributes zero. With both minimums at zero the
    candidate scores an exact `0.0` for that leaf, since its left child is the
    parent, and `0.0` cannot win. The two spellings therefore agree on every
    setting and this one needs no branch.

    **Categorical features are refused, not scored.** A cross-leaf category
    partition is a different search: the many-vs-many arm sorts bins by each
    node's own gradient ordering, and a level has 2^d orderings that need not
    agree, so there is no single prefix to walk. A feature with two or more
    categories is skipped here and `_launch_oblivious_search` refuses a dataset
    that has one, so the refusal is visible at the call rather than as a
    silently narrower search.

    `random_strength`: ONE DRAW PER (FEATURE, BIN), ON THE LEVEL'S AGGREGATE
    -----------------------------------------------------------------------
    Read from `catboost/private/libs/algo/tensor_search_helpers.cpp`'s
    `SetBestScore` (v1.2.10, lines 716-757), which is where CatBoost's
    `random_strength` actually lands: `scoreWoNoise = scores[binFeatureIdx]`
    is **already the level-summed score across every leaf**, returned by
    `CalculateNonPairwiseScore`, and the noise is added to that one number
    before the argmax runs over the noised instances. It is NOT a draw per
    (leaf, candidate) folded into the sum. Those are two different
    regularizers wearing one parameter name, and this kernel implements the
    first because that is the one CatBoost implements.

    So the addend enters at exactly one place: after the leaf loop has closed,
    after the Cosine ratio has been taken and `level_parent` subtracted, and
    immediately before the `> best_gain` test. **Never into `cos_num` or
    `cos_den`.** Adding it inside the accumulation would noise the numerator
    and the denominator of a ratio -- a different functional, scaled by the
    level's width, and it would look entirely correct in a diff. The Cosine
    arm exists precisely so that the level's score is expressible as one
    number at one point; that is the point at which the noise is added.

    One draw per (feature, bin), shared by the missing-left and missing-right
    candidates and by every leaf of the level, for the two reasons the
    per-node scan gives and one more: the noise belongs to the threshold and
    not to the routing (so an exact tie between the two directions still keeps
    `default_left`, since equal numbers stay equal after the same addend), and
    the level takes ONE split, so it draws once.
    `split.find_best_split_shared` takes its draw at the same granularity and
    at the same point in its own scan.

    **The key's third component is the level's DEPTH, in its own domain**,
    staged by `GpuSplitSearcher.stage_random_score_level`. A level has no
    node, which is what kept `random_strength` off this path; it has a depth,
    and the depth is exactly the term CatBoost's fresh-per-level seed is
    standing in for. `gpu_oblivious_score_stream` says why it is a counter
    term rather than an advancing generator, and `OBLIVIOUS_SCORE_DOMAIN` why
    it is a second domain rather than the node slot reused.

    **What the noise costs the top-threshold argument.** Two paragraphs above,
    this docstring states that a candidate every leaf refuses scores an exact
    `0.0` and therefore never beats a `best_gain` that starts at `0.0` under a
    strict `>`. That holds with the noise **off**, which is the default and
    every LightGBM-mode fit. With it on, `0.0 + noise` beats `0.0` whenever
    the draw is positive, so such a candidate can win a level in which nothing
    admissible was found. That is not a defect introduced here: it is what
    `_scan_slot_kernel` does at the same initial `best_gain`, and it is what
    `find_best_split_shared` does at its own `f_gain = 0.0`, so all three
    agree. It is also CatBoost's own shape -- `bestScoreInstance` starts at
    `MINIMAL_SCORE` and every candidate it compares is already noised. The
    property the two backends must share is that they share it, and they do."""
    var slot = Int(block_idx.x)
    var record = Int(level_record)
    var nt = record * NODE_WORDS
    if slot >= Int(node_tab[unsafe_offset = nt + NODE_SLOTS][0]):
        return
    var stride = Int(feat_stride)
    var table = record * stride
    var f = Int(feat_ids[unsafe_offset = table + slot][0])
    var nb = Int(n_bins)
    var hs = Int(hist_size)
    var io = (table + slot) * SPLIT_IWORDS
    var fo = (table + slot) * SPLIT_FWORDS
    var pf = record * PF_WORDS
    # This slot's row of the level's noise plane, one Float32 per bin, in the
    # same (record, slot) cell order every other per-slot table uses -- the
    # identical indexing `_scan_slot_kernel` uses, because a level's plane is
    # the level record's plane and the level record is an ordinary record.
    # Read only when `noisy`; at the default the pointer is a one-element
    # placeholder and no lane touches it.
    var noise_base = (table + slot) * nb
    var nl = Int(n_leaves)
    if nl > OBLIVIOUS_MAX_LEAVES:
        nl = OBLIVIOUS_MAX_LEAVES

    for i in range(SPLIT_IWORDS):
        out_i[unsafe_offset = io + i] = Int32(0)
    for i in range(SPLIT_FWORDS):
        out_f[unsafe_offset = fo + i] = Float32(0.0)
    out_i[unsafe_offset = io + IREC_FEATURE] = Int32(-1)
    out_i[unsafe_offset = io + IREC_BIN] = Int32(-1)
    out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(-1)

    # Per-leaf scan state, twelve words a leaf. Threadgroup rather than local
    # because `_scan_slot_kernel` already reserves two arrays this way in a
    # one-thread launch and the budget is stated once, at
    # `OBLIVIOUS_MAX_LEAVES`.
    var base_of = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var tot_g = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var tot_h = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var tot_c = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var mis_g = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var mis_h = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var mis_c = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var run_g = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var run_h = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var run_c = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var par_score = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var node_ss = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var cross_off = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    # Cosine's two per-leaf words: the leaf's unsplit accumulator terms, which
    # are both the level's zero point and what the leaf contributes at a
    # candidate it cannot take. Written only under `SCORE_COSINE`; see
    # `gpu_cosine_unsplit` and `OBLIVIOUS_MAX_LEAVES`.
    var un_num = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var un_den = stack_allocation[
        OBLIVIOUS_MAX_LEAVES,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var g_inv = fparams[unsafe_offset = pf + PF_G_INV][0]
    var h_inv = fparams[unsafe_offset = pf + PF_H_INV][0]
    var lambda_l2 = fparams[unsafe_offset = pf + PF_LAMBDA_L2][0]
    var lambda_l1 = fparams[unsafe_offset = pf + PF_LAMBDA_L1][0]
    var min_child_hess = fparams[unsafe_offset = pf + PF_MIN_CHILD_HESS][0]
    var bound_lo = fparams[unsafe_offset = pf + PF_BOUND_LO][0]
    var bound_hi = fparams[unsafe_offset = pf + PF_BOUND_HI][0]
    var form = gpu_resolve_gain_form(gain_form, lambda_l1)
    # The functional, read once per slot into a value the whole scan holds
    # constant, for the reason `_scan_slot_kernel` gives for reading it once
    # per node: an L2 level must leave this kernel on exactly the instruction
    # sequence it took before the parameter existed, and the square root is
    # the one operation an L2 scan has never had to issue.
    var cosine = score_function == Int32(SCORE_COSINE)
    # The level's own unsplit accumulators, folded over the leaves below in the
    # same ascending order every other cross-leaf sum here uses. The single
    # score of them is the constant every candidate of this feature subtracts.
    var p_num = Float32(0.0)
    var p_den = Float32(0.0)

    # The level's totals, summed over leaves in ascending record order for the
    # same reason the score is. `IREC_TOTAL_COUNT` and the two total sums are
    # what the reduction copies into the level's record and what a caller
    # reads as "how big was this level", so they are the level's numbers and
    # not any one leaf's.
    var level_g = Float32(0.0)
    var level_h = Float32(0.0)
    var level_c = Int32(0)
    for l in range(nl):
        var lnt = (Int(leaf_base) + l) * NODE_WORDS
        var lb = (
            Int(node_tab[unsafe_offset = lnt + NODE_HIST_BASE][0]) + f * nb
        )
        base_of[unsafe_offset=l] = Int32(lb)
        var tg = Int32(0)
        var th = Int32(0)
        var tc = Int32(0)
        for b in range(nb):
            tg += hist[unsafe_offset = lb + b][0]
            th += hist[unsafe_offset = hs + lb + b][0]
            tc += hist[unsafe_offset = 2 * hs + lb + b][0]
        tot_g[unsafe_offset=l] = tg
        tot_h[unsafe_offset=l] = th
        tot_c[unsafe_offset=l] = tc
        run_g[unsafe_offset=l] = Int32(0)
        run_h[unsafe_offset=l] = Int32(0)
        run_c[unsafe_offset=l] = Int32(0)
        var tgf = tg.cast[DType.float32]() * g_inv
        var thf = th.cast[DType.float32]() * h_inv
        par_score[unsafe_offset=l] = gpu_leaf_score(
            tgf, thf, lambda_l1, lambda_l2
        )
        var ns = gpu_cross_node_s(thf, lambda_l2)
        node_ss[unsafe_offset=l] = ns
        cross_off[unsafe_offset=l] = gpu_cross_offset(
            tgf, thf, lambda_l1, lambda_l2, lambda_l2, ns
        )
        # This leaf's unsplit terms, computed once and used twice, which is
        # what `find_best_split_shared` does with its `pt`: they accumulate
        # into the level's zero point here, and they are what this leaf
        # contributes at a candidate it cannot take.
        if cosine:
            var ut = gpu_cosine_unsplit(tgf, thf, lambda_l1, lambda_l2)
            un_num[unsafe_offset=l] = ut[0]
            un_den[unsafe_offset=l] = ut[1]
            p_num += ut[0]
            p_den += ut[1]
        level_g += tgf
        level_h += thf
        level_c += tc
    out_f[unsafe_offset = fo + FREC_TOTAL_GRAD] = level_g
    out_f[unsafe_offset = fo + FREC_TOTAL_HESS] = level_h
    out_i[unsafe_offset = io + IREC_TOTAL_COUNT] = level_c

    if allow[unsafe_offset = table + slot][0] == Int32(0):
        return
    # A categorical feature is skipped rather than scored; see the docstring.
    if Int(cat_n[unsafe_offset=f][0]) >= 2:
        return

    var sign = Int32(MONOTONE_FREE)
    if constrained != Int32(0):
        sign = mono[unsafe_offset=f][0]
    var is_constrained = constrained != Int32(0)

    var missing_bin = Int(missing[unsafe_offset=f][0])
    var n_scan = missing_bin if missing_bin >= 0 else nb
    if missing_bin >= 0:
        for l in range(nl):
            var lb = Int(base_of[unsafe_offset=l][0])
            mis_g[unsafe_offset=l] = hist[unsafe_offset = lb + missing_bin][0]
            mis_h[unsafe_offset=l] = hist[
                unsafe_offset = hs + lb + missing_bin
            ][0]
            mis_c[unsafe_offset=l] = hist[
                unsafe_offset = 2 * hs + lb + missing_bin
            ][0]
    else:
        for l in range(nl):
            mis_g[unsafe_offset=l] = Int32(0)
            mis_h[unsafe_offset=l] = Int32(0)
            mis_c[unsafe_offset=l] = Int32(0)

    # ONE ratio for the whole level, which is CatBoost's `numScoreBlocks = 1`,
    # taken here over the accumulators the leaf loop folded and subtracted from
    # every candidate below. `find_best_split_shared` computes `level_parent`
    # at exactly this point and for exactly this reason.
    var level_parent = Float32(0.0)
    if cosine:
        level_parent = gpu_cosine_score(p_num, p_den)

    var best_gain = Float32(0.0)
    var runner_gain = Float32(0.0)
    var best_bin = -1
    var best_ordinal = -1
    var best_default_left = False
    # The level's own left statistics at the winning candidate, summed over
    # leaves in the same ascending order. They are the level's, not a node's,
    # and the commit that applies this split derives each leaf's own children
    # from that leaf's histogram rather than from these.
    var best_left_c = Int32(0)
    var best_left_gf = Float32(0.0)
    var best_left_hf = Float32(0.0)
    var found = False

    for b in range(n_scan):
        # Each leaf's prefix advances by this bin, in record order.
        for l in range(nl):
            var lb = Int(base_of[unsafe_offset=l][0])
            run_g[unsafe_offset=l] = (
                run_g[unsafe_offset=l][0] + hist[unsafe_offset = lb + b][0]
            )
            run_h[unsafe_offset=l] = (
                run_h[unsafe_offset=l][0]
                + hist[unsafe_offset = hs + lb + b][0]
            )
            run_c[unsafe_offset=l] = (
                run_c[unsafe_offset=l][0]
                + hist[unsafe_offset = 2 * hs + lb + b][0]
            )

        # This threshold's `random_strength` shift for the whole level: read
        # once per bin, outside the direction loop and outside the leaf loop,
        # because the level takes one split and CatBoost noises the level's
        # aggregate score once. Shared by the two routing directions for the
        # reason `_scan_slot_kernel` gives -- the noise belongs to the
        # threshold, not to the direction, so an exact tie still keeps
        # `default_left`.
        var bin_noise = Float32(0.0)
        if noisy != Int32(0):
            bin_noise = noise[unsafe_offset = noise_base + b][0]

        # Missing to the left first, so an exact tie keeps `default_left`.
        for d in range(2):
            var want_default_left = d == 0
            if want_default_left and missing_bin < 0:
                continue
            if not want_default_left and missing_bin >= 0:
                # `_scan_slot_kernel` scores the missing-right candidate of a
                # feature that reserves a missing bin only when some row is
                # actually missing; with none, the two candidates are the same
                # split and scoring both would let the second win a tie the
                # first should have kept. The test is over the level, since the
                # candidate is.
                var any_missing = False
                for l in range(nl):
                    if mis_c[unsafe_offset=l][0] > Int32(0):
                        any_missing = True
                if not any_missing:
                    continue
            var total = Float32(0.0)
            # Cosine's two cross-leaf accumulators, which stand where `total`
            # stands on the L2 arm. `split.find_best_split_shared`'s
            # `acc_left`/`den_left` pair, per candidate rather than per bin
            # because this kernel scores one candidate at a time.
            var cos_num = Float32(0.0)
            var cos_den = Float32(0.0)
            var cand_c = Int32(0)
            var cand_gf = Float32(0.0)
            var cand_hf = Float32(0.0)
            for l in range(nl):
                var lg = run_g[unsafe_offset=l][0]
                var lh = run_h[unsafe_offset=l][0]
                var lc = run_c[unsafe_offset=l][0]
                if want_default_left:
                    lg += mis_g[unsafe_offset=l][0]
                    lh += mis_h[unsafe_offset=l][0]
                    lc += mis_c[unsafe_offset=l][0]
                var tg = tot_g[unsafe_offset=l][0]
                var th = tot_h[unsafe_offset=l][0]
                var tc = tot_c[unsafe_offset=l][0]
                var tgf = tg.cast[DType.float32]() * g_inv
                var thf = th.cast[DType.float32]() * h_inv
                var lhf = lh.cast[DType.float32]() * h_inv
                var rhf = gpu_right_sum(thf, lhf, th, lh, h_inv, form)
                var lgf = lg.cast[DType.float32]() * g_inv
                var rgf = gpu_right_sum(tgf, lgf, tg, lg, g_inv, form)
                # This leaf's own legality. A leaf that fails adds nothing and
                # does not disqualify the candidate for the rest of the level.
                # "Nothing" is `0.0` under L2 and the leaf's unsplit terms
                # under Cosine, which is the same rule in the two functionals'
                # own spellings; see the docstring.
                if (
                    lc < min_data_in_leaf
                    or tc - lc < min_data_in_leaf
                    or lhf < min_child_hess
                    or rhf < min_child_hess
                ):
                    if cosine:
                        cos_num += un_num[unsafe_offset=l][0]
                        cos_den += un_den[unsafe_offset=l][0]
                    continue
                if cosine:
                    var ct = gpu_cosine_level_terms(
                        gpu_soft_threshold_l1(lgf, lambda_l1),
                        lhf,
                        gpu_soft_threshold_l1(rgf, lambda_l1),
                        rhf,
                        lambda_l2,
                        sign,
                        bound_lo,
                        bound_hi,
                        is_constrained,
                        SIMD[DType.float32, 2](
                            un_num[unsafe_offset=l][0],
                            un_den[unsafe_offset=l][0],
                        ),
                    )
                    cos_num += ct[0]
                    cos_den += ct[1]
                else:
                    total += gpu_split_gain(
                        gpu_soft_threshold_l1(lgf, lambda_l1),
                        lhf,
                        gpu_soft_threshold_l1(rgf, lambda_l1),
                        rhf,
                        lambda_l2,
                        par_score[unsafe_offset=l][0],
                        sign,
                        bound_lo,
                        bound_hi,
                        is_constrained,
                        node_ss[unsafe_offset=l][0],
                        cross_off[unsafe_offset=l][0],
                        form,
                    )
                cand_c += lc
                cand_gf += lgf
                cand_hf += lhf
            # The one square root of the level, after the leaf loop has closed
            # over both accumulators. This is the line the L2 arm does not have
            # and the reason a level's Cosine score is not a sum of per-leaf
            # Cosine gains.
            if cosine:
                total = gpu_cosine_score(cos_num, cos_den) - level_parent
            # `random_strength`, HERE and nowhere earlier: on the level's one
            # aggregate score, after the cross-leaf sum, after the single
            # Cosine ratio, and immediately before the argmax -- which is
            # `SetBestScore`'s `scoreWoNoise + Normal(0, scoreStDev)` on a
            # `scores[binFeatureIdx]` that is already level-summed. Adding it
            # to `cos_num` or `cos_den` above would be a different
            # regularizer and would read as correct; see the docstring.
            if noisy != Int32(0):
                total += bin_noise
            if total > best_gain:
                runner_gain = best_gain
                best_gain = total
                best_bin = b
                best_ordinal = 2 * b + (0 if want_default_left else 1)
                best_default_left = want_default_left
                best_left_c = cand_c
                best_left_gf = cand_gf
                best_left_hf = cand_hf
                found = True
            elif total > runner_gain:
                runner_gain = total

    if not found:
        return

    var flags = Int32(FLAG_FOUND)
    if best_default_left:
        flags += Int32(FLAG_DEFAULT_LEFT)
    out_i[unsafe_offset = io + IREC_FEATURE] = Int32(f)
    out_i[unsafe_offset = io + IREC_BIN] = Int32(best_bin)
    out_i[unsafe_offset = io + IREC_FLAGS] = flags
    out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(best_ordinal)
    out_i[unsafe_offset = io + IREC_LEFT_COUNT] = best_left_c
    out_i[unsafe_offset = io + IREC_RIGHT_COUNT] = level_c - best_left_c
    out_f[unsafe_offset = fo + FREC_GAIN] = best_gain
    out_f[unsafe_offset = fo + FREC_RUNNER_GAIN] = runner_gain
    out_f[unsafe_offset = fo + FREC_LEFT_GRAD] = best_left_gf
    out_f[unsafe_offset = fo + FREC_LEFT_HESS] = best_left_hf
    out_f[unsafe_offset = fo + FREC_RIGHT_GRAD] = level_g - best_left_gf
    out_f[unsafe_offset = fo + FREC_RIGHT_HESS] = level_h - best_left_hf


def _reduce_slots_kernel(
    slot_i: MutPointer[Int32, MutAnyOrigin],
    slot_f: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[Int32, MutAnyOrigin],
    out_f: MutPointer[Float32, MutAnyOrigin],
    node_tab: MutPointer[Int32, MutAnyOrigin],
    fparams: MutPointer[Float32, MutAnyOrigin],
    record_base: Int32,
    feat_stride: Int32,
):
    """Fold one node's per-slot records into one, in ascending slot order,
    and fill in the child and parent leaf values.

    One thread per node, launched over the same records the scan covered.
    A single thread walking the slots in order, accepting a new best only on
    a strictly greater gain, is what makes the winner the first candidate of
    the first slot holding the maximum gain: the same rule, in the same
    order, as the host's one loop. No atomics are involved, so the result
    cannot depend on scheduling, and a batched frontier reduces exactly as a
    one-node launch does since no node reads another node's slots."""
    var record = Int(record_base) + Int(block_idx.x)
    var table = record * Int(feat_stride)
    var nt = record * NODE_WORDS
    var n_slots = Int(node_tab[unsafe_offset = nt + NODE_SLOTS][0])
    var pf = record * PF_WORDS
    var lambda_l2 = fparams[unsafe_offset = pf + PF_LAMBDA_L2][0]
    var lambda_l1 = fparams[unsafe_offset = pf + PF_LAMBDA_L1][0]
    var oi = record * SPLIT_IWORDS
    var of = record * SPLIT_FWORDS

    var best = -1
    var best_gain = Float32(0.0)
    # The best gain among the slots that did not win, folded together with
    # the winning slot's own runner-up below. Ties do not move it: a slot
    # whose gain equals the current best is not the winner and is a genuine
    # runner-up, which is exactly the case a near-tie test has to see.
    var runner_gain = Float32(0.0)
    for s in range(n_slots):
        var si = (table + s) * SPLIT_IWORDS
        var sf = (table + s) * SPLIT_FWORDS
        var flags = slot_i[unsafe_offset = si + IREC_FLAGS][0]
        if (flags & Int32(FLAG_FOUND)) == Int32(0):
            continue
        var gain = slot_f[unsafe_offset = sf + FREC_GAIN][0]
        if best < 0 or gain > best_gain:
            if best >= 0 and best_gain > runner_gain:
                runner_gain = best_gain
            best = s
            best_gain = gain
        elif gain > runner_gain:
            runner_gain = gain

    # The parent's totals come from this node's slot 0, which is the feature
    # the host grower already uses for leaf values (`tree_features[0]`).
    # Every accumulated feature carries the same totals bit for bit.
    var zi = table * SPLIT_IWORDS
    var zf = table * SPLIT_FWORDS
    var total_g = slot_f[unsafe_offset = zf + FREC_TOTAL_GRAD][0]
    var total_h = slot_f[unsafe_offset = zf + FREC_TOTAL_HESS][0]
    var total_c = slot_i[unsafe_offset = zi + IREC_TOTAL_COUNT][0]

    if best < 0:
        for i in range(SPLIT_IWORDS):
            out_i[unsafe_offset = oi + i] = Int32(0)
        for i in range(SPLIT_FWORDS):
            out_f[unsafe_offset = of + i] = Float32(0.0)
        out_i[unsafe_offset = oi + IREC_FEATURE] = Int32(-1)
        out_i[unsafe_offset = oi + IREC_BIN] = Int32(-1)
        out_i[unsafe_offset = oi + IREC_ORDINAL] = Int32(-1)
        out_i[unsafe_offset = oi + IREC_TOTAL_COUNT] = total_c
        out_f[unsafe_offset = of + FREC_TOTAL_GRAD] = total_g
        out_f[unsafe_offset = of + FREC_TOTAL_HESS] = total_h
        out_f[unsafe_offset = of + FREC_PARENT_VALUE] = gpu_leaf_value(
            total_g, total_h, lambda_l1, lambda_l2
        )
        return

    var bi = (table + best) * SPLIT_IWORDS
    var bf = (table + best) * SPLIT_FWORDS
    for i in range(SPLIT_IWORDS):
        out_i[unsafe_offset = oi + i] = slot_i[unsafe_offset = bi + i][0]
    for i in range(SPLIT_FWORDS):
        out_f[unsafe_offset = of + i] = slot_f[unsafe_offset = bf + i][0]

    # The winner's own totals are bit-identical to slot 0's, but the record
    # is defined to carry slot 0's, so a caller reading the parent's
    # statistics gets the same numbers whether or not a split was found.
    out_i[unsafe_offset = oi + IREC_TOTAL_COUNT] = total_c
    out_f[unsafe_offset = of + FREC_TOTAL_GRAD] = total_g
    out_f[unsafe_offset = of + FREC_TOTAL_HESS] = total_h

    # The node's runner-up is the better of the best losing feature and the
    # winning feature's own second candidate, so the margin a caller tests
    # covers both ways a decision can be close: two features that scored
    # nearly the same, and two bins of one feature that did.
    var own_runner = slot_f[unsafe_offset = bf + FREC_RUNNER_GAIN][0]
    if own_runner > runner_gain:
        runner_gain = own_runner
    out_f[unsafe_offset = of + FREC_RUNNER_GAIN] = runner_gain

    var left_g = slot_f[unsafe_offset = bf + FREC_LEFT_GRAD][0]
    var left_h = slot_f[unsafe_offset = bf + FREC_LEFT_HESS][0]
    var right_g = slot_f[unsafe_offset = bf + FREC_RIGHT_GRAD][0]
    var right_h = slot_f[unsafe_offset = bf + FREC_RIGHT_HESS][0]
    out_f[unsafe_offset = of + FREC_LEFT_VALUE] = gpu_leaf_value(
        left_g, left_h, lambda_l1, lambda_l2
    )
    out_f[unsafe_offset = of + FREC_RIGHT_VALUE] = gpu_leaf_value(
        right_g, right_h, lambda_l1, lambda_l2
    )
    out_f[unsafe_offset = of + FREC_PARENT_VALUE] = gpu_leaf_value(
        total_g, total_h, lambda_l1, lambda_l2
    )


def _reduce_slots_block_kernel(
    slot_i: MutPointer[Int32, MutAnyOrigin],
    slot_f: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[Int32, MutAnyOrigin],
    out_f: MutPointer[Float32, MutAnyOrigin],
    node_tab: MutPointer[Int32, MutAnyOrigin],
    fparams: MutPointer[Float32, MutAnyOrigin],
    record_base: Int32,
    feat_stride: Int32,
):
    """`_reduce_slots_kernel` on a threadgroup instead of on one thread, and
    it writes the same record.

    The serial fold is the one place the default path spends time
    proportional to the feature count on a single lane: one thread per node
    walking every one of that node's feature slots in order. A hundred
    features is a hundred dependent iterations while the rest of the device
    is idle. Here each thread takes a strided subset of the slots, and the
    cross-thread fold is three `gpu.primitives.block` collectives.

    Why the winner is the serial winner. The serial rule is "highest gain,
    and among equal gains the lowest slot", because it walks slots ascending
    and accepts only on a strictly greater gain. Split in two, that is a
    `block.max` over the gains and then a `block.min` over the slot indices
    of the threads holding the maximum. Both reassociate exactly: `max` and
    `min` are associative and commutative, and no gain is recomputed here,
    only compared. Within a thread the same strict `>` over its own
    ascending slots keeps the lowest of its own ties, so the pair really is
    the global lexicographic minimum among the maximum-gain slots.

    Why the runner-up is the serial runner-up. `_reduce_slots_kernel` ends
    with `runner_gain` equal to the best gain over every found slot that is
    not the winner: a slot displaced from `best` is folded in at the
    displacement, and a slot rejected against `best` is folded in on the
    spot, so every non-winning slot is compared exactly once, and a slot
    whose gain ties the winner's is a genuine runner-up rather than a
    winner. Excluding one slot is decomposable, because exactly one thread
    owns the winning slot: that thread contributes the best of its *other*
    slots, every other thread contributes its own best, and one `block.max`
    finishes it.

    No floating-point sum crosses a thread boundary here. The gains being
    reduced were computed by the scan kernel and are only compared; the leaf
    values below are computed once, by one thread, from the winning slot's
    already-reduced sums, exactly as the serial kernel computes them."""
    var tid = Int(thread_idx.x)
    var record = Int(record_base) + Int(block_idx.x)
    var table = record * Int(feat_stride)
    var nt = record * NODE_WORDS
    var n_slots = Int(node_tab[unsafe_offset = nt + NODE_SLOTS][0])
    var pf = record * PF_WORDS
    var lambda_l2 = fparams[unsafe_offset = pf + PF_LAMBDA_L2][0]
    var lambda_l1 = fparams[unsafe_offset = pf + PF_LAMBDA_L1][0]
    var oi = record * SPLIT_IWORDS
    var of = record * SPLIT_FWORDS

    # This thread's own best and second best over its strided share of the
    # slots, in ascending slot order, so a tie inside one thread keeps the
    # lower slot exactly as the serial walk does. A found slot's gain is
    # always strictly positive, so zero is a safe identity for a thread with
    # no slots at all, which is every thread past `n_slots`.
    var my_gain = Float32(0.0)
    var my_second = Float32(0.0)
    var my_slot = NO_CANDIDATE
    var s = tid
    while s < n_slots:
        var si = (table + s) * SPLIT_IWORDS
        var sf = (table + s) * SPLIT_FWORDS
        var flags = slot_i[unsafe_offset = si + IREC_FLAGS][0]
        if (flags & Int32(FLAG_FOUND)) != Int32(0):
            var gain = slot_f[unsafe_offset = sf + FREC_GAIN][0]
            if gain > my_gain:
                my_second = my_gain
                my_gain = gain
                my_slot = Int32(s)
            elif gain > my_second:
                my_second = gain
        s += REDUCE_SLOT_THREADS

    var top = block.max[block_size=REDUCE_SLOT_THREADS](my_gain)
    var mine = NO_CANDIDATE
    if my_gain == top:
        mine = my_slot
    var best = block.min[block_size=REDUCE_SLOT_THREADS](mine)
    var contribution = my_gain
    if my_slot == best:
        contribution = my_second
    var runner = block.max[block_size=REDUCE_SLOT_THREADS](contribution)

    if tid != 0:
        return

    # The parent's totals come from this node's slot 0, which is the feature
    # the host grower already uses for leaf values (`tree_features[0]`).
    # Every accumulated feature carries the same totals bit for bit.
    var zi = table * SPLIT_IWORDS
    var zf = table * SPLIT_FWORDS
    var total_g = slot_f[unsafe_offset = zf + FREC_TOTAL_GRAD][0]
    var total_h = slot_f[unsafe_offset = zf + FREC_TOTAL_HESS][0]
    var total_c = slot_i[unsafe_offset = zi + IREC_TOTAL_COUNT][0]

    if top <= Float32(0.0):
        for i in range(SPLIT_IWORDS):
            out_i[unsafe_offset = oi + i] = Int32(0)
        for i in range(SPLIT_FWORDS):
            out_f[unsafe_offset = of + i] = Float32(0.0)
        out_i[unsafe_offset = oi + IREC_FEATURE] = Int32(-1)
        out_i[unsafe_offset = oi + IREC_BIN] = Int32(-1)
        out_i[unsafe_offset = oi + IREC_ORDINAL] = Int32(-1)
        out_i[unsafe_offset = oi + IREC_TOTAL_COUNT] = total_c
        out_f[unsafe_offset = of + FREC_TOTAL_GRAD] = total_g
        out_f[unsafe_offset = of + FREC_TOTAL_HESS] = total_h
        out_f[unsafe_offset = of + FREC_PARENT_VALUE] = gpu_leaf_value(
            total_g, total_h, lambda_l1, lambda_l2
        )
        return

    var bi = (table + Int(best)) * SPLIT_IWORDS
    var bf = (table + Int(best)) * SPLIT_FWORDS
    for i in range(SPLIT_IWORDS):
        out_i[unsafe_offset = oi + i] = slot_i[unsafe_offset = bi + i][0]
    for i in range(SPLIT_FWORDS):
        out_f[unsafe_offset = of + i] = slot_f[unsafe_offset = bf + i][0]

    # The winner's own totals are bit-identical to slot 0's, but the record
    # is defined to carry slot 0's, so a caller reading the parent's
    # statistics gets the same numbers whether or not a split was found.
    out_i[unsafe_offset = oi + IREC_TOTAL_COUNT] = total_c
    out_f[unsafe_offset = of + FREC_TOTAL_GRAD] = total_g
    out_f[unsafe_offset = of + FREC_TOTAL_HESS] = total_h

    # The node's runner-up is the better of the best losing feature and the
    # winning feature's own second candidate, so the margin a caller tests
    # covers both ways a decision can be close.
    var node_runner = runner
    var own_runner = slot_f[unsafe_offset = bf + FREC_RUNNER_GAIN][0]
    if own_runner > node_runner:
        node_runner = own_runner
    out_f[unsafe_offset = of + FREC_RUNNER_GAIN] = node_runner

    var left_g = slot_f[unsafe_offset = bf + FREC_LEFT_GRAD][0]
    var left_h = slot_f[unsafe_offset = bf + FREC_LEFT_HESS][0]
    var right_g = slot_f[unsafe_offset = bf + FREC_RIGHT_GRAD][0]
    var right_h = slot_f[unsafe_offset = bf + FREC_RIGHT_HESS][0]
    out_f[unsafe_offset = of + FREC_LEFT_VALUE] = gpu_leaf_value(
        left_g, left_h, lambda_l1, lambda_l2
    )
    out_f[unsafe_offset = of + FREC_RIGHT_VALUE] = gpu_leaf_value(
        right_g, right_h, lambda_l1, lambda_l2
    )
    out_f[unsafe_offset = of + FREC_PARENT_VALUE] = gpu_leaf_value(
        total_g, total_h, lambda_l1, lambda_l2
    )


def _pick_best_record_kernel(
    rec_i: MutPointer[Int32, MutAnyOrigin],
    rec_f: MutPointer[Float32, MutAnyOrigin],
    n_records: Int32,
    out_index: Int32,
):
    """Reduce a set of finished records to the single best-gain one, ties
    going to the lower record index.

    This is the frontier selection the host currently does in `grow_tree_gpu`
    (`for i in range(len(frontier))`, strictly greater gain wins). Running it
    here is what lets a whole tree level, and eventually a whole tree, be
    grown without a host round trip per node."""
    var best = -1
    var best_gain = Float32(0.0)
    for r in range(Int(n_records)):
        var flags = rec_i[unsafe_offset = r * SPLIT_IWORDS + IREC_FLAGS][0]
        if (flags & Int32(FLAG_FOUND)) == Int32(0):
            continue
        var gain = rec_f[unsafe_offset = r * SPLIT_FWORDS + FREC_GAIN][0]
        if best < 0 or gain > best_gain:
            best = r
            best_gain = gain

    # The destination slot lives in the same buffer as the sources, so one
    # pointer pair carries both and no two kernel arguments alias.
    var oi = Int(out_index) * SPLIT_IWORDS
    var of = Int(out_index) * SPLIT_FWORDS
    if best < 0:
        for i in range(SPLIT_IWORDS):
            rec_i[unsafe_offset = oi + i] = Int32(0)
        for i in range(SPLIT_FWORDS):
            rec_f[unsafe_offset = of + i] = Float32(0.0)
        rec_i[unsafe_offset = oi + IREC_FEATURE] = Int32(-1)
        rec_i[unsafe_offset = oi + IREC_BIN] = Int32(-1)
        rec_i[unsafe_offset = oi + IREC_ORDINAL] = Int32(-1)
        return
    var bi = best * SPLIT_IWORDS
    var bf = best * SPLIT_FWORDS
    if bi == oi:
        return
    for i in range(SPLIT_IWORDS):
        rec_i[unsafe_offset = oi + i] = rec_i[unsafe_offset = bi + i][0]
    for i in range(SPLIT_FWORDS):
        rec_f[unsafe_offset = of + i] = rec_f[unsafe_offset = bf + i][0]


# --- Host-side record and parameters -------------------------------------


@fieldwise_init
struct ChildStats(Copyable, Movable):
    """One child's statistics as the device computed them: gradient and
    hessian sums dequantized once from exact fixed-point accumulation, and an
    exact row count."""

    var grad: Float64
    var hess: Float64
    var count: Int


struct GpuSplitRecord(Copyable, Movable, Writable):
    """The single value a node's search returns to the host.

    Everything the grower needs from a node: the split itself, both
    children's statistics and Newton leaf values, and the parent's leaf
    value. The host still clamps the leaf values into the node's monotone
    interval and collapses an inverted pair to its midpoint, exactly as
    `grow_tree_gpu` does today; those are host-side bookkeeping over the
    tree's own bounds, not properties of the histogram."""

    var feature: Int
    var bin: Int
    var gain: Float64
    var found: Bool
    var default_left: Bool
    var is_categorical: Bool
    var cat_bitset: CatBitset
    var left: ChildStats
    var right: ChildStats
    var total: ChildStats
    var left_value: Float64
    var right_value: Float64
    var parent_value: Float64
    var ordinal: Int
    """Position of the winning candidate within its feature's scan:
    `2 * bin` with the missing rows left, `2 * bin + 1` with them right, and
    -1 for a categorical partition or no split. Diagnostic only; the
    tie-breaking rule is the scan order itself."""
    var runner_gain: Float64
    """The best gain of every candidate this node scored except the winner,
    over every scanned feature. `gain - runner_gain` is the margin the
    decision was made by, and `is_near_tie` is the test a caller applies to
    it."""

    def __init__(out self):
        """The absence of a split, with zero statistics."""
        self.feature = -1
        self.bin = -1
        self.gain = 0.0
        self.found = False
        self.default_left = False
        self.is_categorical = False
        self.cat_bitset = cat_empty()
        self.left = ChildStats(0.0, 0.0, 0)
        self.right = ChildStats(0.0, 0.0, 0)
        self.total = ChildStats(0.0, 0.0, 0)
        self.left_value = 0.0
        self.right_value = 0.0
        self.parent_value = 0.0
        self.ordinal = -1
        self.runner_gain = 0.0

    def margin(self) -> Float64:
        """How far ahead of the runner-up the winning candidate scored.

        Zero when nothing was found and when the two best candidates scored
        exactly alike, in which case the scan order decided, which is
        deterministic but is a decision the host would have made on
        Float64 gains instead.
        """
        if not self.found:
            return 0.0
        var m = self.gain - self.runner_gain
        return m if m > 0.0 else 0.0

    def parent_score_bound(self) -> Float64:
        """An upper bound on this node's parent score, from the record.

        The parent score is `T(G)^2 / (H + lambda_l2)` and the parent leaf
        value is `-T(G) / (H + lambda_l2)`, so the score is
        `-parent_value * T(G)`. The record carries `parent_value` and the
        node's `total.grad`, but not `lambda_l1` and therefore not `T(G)`.
        Soft-thresholding shrinks toward zero without crossing it, so
        `|T(G)| <= |G|` with the same sign, which makes `-parent_value * G`
        the score itself when `lambda_l1` is zero and an upper bound on it
        otherwise. A bound is the right side to err on: this feeds a floor
        below which a decision is declared unresolvable, and overstating the
        floor costs a host rescan of one node while understating it silently
        keeps a split the host would not have chosen.
        """
        var bound = -self.parent_value * self.total.grad
        return bound if bound > 0.0 else 0.0

    def resolution_floor(self) -> Float64:
        """The absolute gain difference the device scan could not have
        resolved on this node, whichever gain form produced the record.

        This is the number `SPLIT_TIE_RELATIVE` cannot express, and leaving
        it out is what makes a purely relative near-tie test miss the regime
        it exists for. The module docstring states the two resolutions: the
        subtractive form cancels against the parent score and resolves to
        about `eps * parent_score`, an absolute floor that does not shrink
        as the gain does; `GAIN_FORM_CROSS` never forms that sum and
        resolves to about `eps * sqrt(parent_score * gain)`. A record does
        not say which form scored it, so this takes the larger of the two:
        the subtractive floor at a nearly pure leaf (`parent_score > gain`,
        the hard case) and the cross floor when the gain is the larger
        quantity.

        Written as a floor on the *margin* rather than as a relative width,
        because that is what it is. At `parent_score / gain` in the
        thousands -- the range `gpu_right_sum`'s own measured table covers,
        and where a boosted ensemble spends its late rounds -- this floor is
        orders of magnitude wider than `SPLIT_TIE_RELATIVE * gain`, so a
        parity run gated on the relative test alone would keep flipped
        splits and report no near ties at all.
        """
        var parent = self.parent_score_bound()
        var gain = abs(self.gain)
        var cross = sqrt(parent * gain)
        var scale = parent if parent > cross else cross
        return SPLIT_TIE_ROUNDINGS * SPLIT_F32_EPS * scale

    def is_near_tie(
        self,
        relative: Float64 = SPLIT_TIE_RELATIVE,
        resolution_aware: Bool = True,
    ) -> Bool:
        """Whether this node's decision is inside Float32's resolution.

        The device scans in Float32, so two candidates whose exact gains
        differ by less than a few Float32 ulps can come back in either
        order; when they do, the split chosen here can differ from the one
        the host's Float64 scan would have chosen, and the difference is a
        different tree, not a different last bit of a value.

        This is the test, and `host_rescan_recommended` is the policy built
        on it: a node that answers True is a node to redo with the host
        scan when the caller needs CPU/GPU agreement. A node with no
        runner-up (one candidate, or one feature with one admissible bin)
        is never near a tie whatever the margin.

        Two widths, and a margin inside either one answers True. `relative`
        is a fraction of the gain itself and is what this test has always
        applied. `resolution_aware` adds `resolution_floor`, the absolute
        width the scan's arithmetic could not see past on this node, which
        the relative width cannot stand in for: the two differ by orders of
        magnitude in exactly the regime where a flip happens, and a node
        whose margin is a millionth of its gain but a thousandth of its
        parent score is a coin flip that the relative test alone calls
        resolved.

        Passing `resolution_aware=False` restores the earlier test exactly,
        which is what a caller comparing the two policies wants. It is not
        the cheaper answer: both widths are a few arithmetic operations on a
        record the caller already holds, and neither reads the device.
        """
        if not self.found or self.runner_gain <= 0.0:
            return False
        var scale = abs(self.gain)
        if scale < abs(self.runner_gain):
            scale = abs(self.runner_gain)
        var width = relative * scale
        if resolution_aware:
            var floor = self.resolution_floor()
            if floor > width:
                width = floor
        return self.margin() <= width

    def to_split_info(self) -> SplitInfo:
        """The `SplitInfo` the existing growers consume, so the device path
        is a drop-in for `_search`'s return value."""
        if not self.found:
            return SplitInfo(-1, -1, 0.0, False)
        if self.is_categorical:
            return SplitInfo.categorical(
                self.feature, self.gain, self.cat_bitset
            )
        return SplitInfo(
            self.feature, self.bin, self.gain, True, self.default_left
        )

    def write_to(self, mut writer: Some[Writer]):
        if not self.found:
            writer.write("GpuSplitRecord(none)")
        elif self.is_categorical:
            writer.write(
                "GpuSplitRecord(feature=",
                self.feature,
                ", categorical, gain=",
                self.gain,
                ")",
            )
        else:
            writer.write(
                "GpuSplitRecord(feature=",
                self.feature,
                ", bin<=",
                self.bin,
                ", gain=",
                self.gain,
                ", default_left=",
                self.default_left,
                ")",
            )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def host_rescan_recommended(
    record: GpuSplitRecord,
    tie_relative: Float64 = SPLIT_TIE_RELATIVE,
    enabled: Bool = True,
    resolution_aware: Bool = True,
) raises -> Bool:
    """The conservative fallback contract, in one place.

    True when this node's decision should be redone by the host scan
    because the device could not separate its two best candidates in
    Float32. The caller's part of the contract is what it does with a True:
    download that node's histogram (`GpuHistogramBuilder.download_raw` and
    `histogram_from_host`, which the host-search path already calls for
    every node) and take `_search`'s answer for that node instead of this
    record's. Nothing else about the node changes: the row counts, the
    child statistics, and the partition are recomputed by the host scan
    from the same histogram.

    The fallback is per node, not per tree. A tie at one node says nothing
    about the next one, and a whole-tree fallback would give up the
    transfer saving on every node to fix the handful that are close.

    `enabled` is the caller's switch. A run that does not need CPU/GPU
    agreement (the default posture of the device search, whose gains are
    documented as Float32) passes False and keeps every decision on the
    device; a parity run passes True. Left as a parameter rather than read
    from the environment here, because the trainer already owns the split
    strategy resolution and one place should decide it.

    `resolution_aware` is passed straight to `is_near_tie` and defaults the
    same way, so this policy sees the absolute Float32 resolution floor as
    well as the relative width. False is the pre-floor policy, kept reachable
    because it is the arm a comparison of the two needs; it is strictly
    narrower, so it recommends a subset of the rescans.

    Raises for a nonpositive tolerance, which would silently disable the
    check.
    """
    if tie_relative <= 0.0:
        raise Error("the near-tie tolerance must be positive")
    if not enabled:
        return False
    return record.is_near_tie(tie_relative, resolution_aware)


def frontier_margin(records: List[GpuSplitRecord]) raises -> Float64:
    """How much better the best leaf of a frontier is than the next best.

    The node-level margin's counterpart at the level above: which leaf a
    leaf-wise grower splits next is also decided by comparing gains, so a
    frontier whose two best leaves are within Float32 of each other can
    grow a differently *shaped* tree than the host would, even when every
    node's own decision is unambiguous.

    Returns 0.0 when fewer than two leaves offer a split, which is the
    case where the order cannot matter. The trainer's tie-break (lowest
    frontier index wins) is unchanged and deterministic either way; this
    only reports how close the call was.
    """
    var best = 0.0
    var second = 0.0
    var seen = 0
    for i in range(len(records)):
        if not records[i].found:
            continue
        var g = records[i].gain
        seen += 1
        if seen == 1 or g > best:
            if seen > 1 and best > second:
                second = best
            best = g
        elif g > second:
            second = g
    if seen < 2:
        return 0.0
    var m = best - second
    return m if m > 0.0 else 0.0


# --- Which configurations the device search can serve ---------------------

comptime SEARCH_OK = 0
comptime SEARCH_EXTRA_PARAMS = 1
comptime SEARCH_FEATURE_BYLEVEL = 2
comptime SEARCH_TOO_MANY_BINS = 3


def device_search_eligibility(
    n_bins: Int,
    extra_active: Bool,
    feature_fraction_bylevel_active: Bool,
) raises -> Int:
    """Which reason, if any, keeps a configuration off the device split
    search.

    The kernels score from `GpuSplitParams` alone: the two lambdas, the two
    child floors, and the categorical parameters. There is nowhere in them
    to charge a gain floor, a per-feature multiplier, a CEGB cost, a
    monotone penalty, a drawn threshold, or a capped and smoothed child
    output, and nowhere to draw a per-level feature subset, so a
    configuration that asks for any of those is refused rather than served
    a tree that quietly ignores it. The host scan honors all of them and is
    the fallback for every code below.

    Scalars rather than a `TreeParams`, so this module stays free of the
    tree-parameter graph and the trainer can call it with
    `params.extra.is_active()` and
    `params.feature_fraction_bylevel != 1.0` without a new import in either
    direction. `train_gpu._check_device_search_supported` is the caller
    that should consume it; the handoff carries that patch.
    """
    if n_bins < 1:
        raise Error("split search requires at least one bin")
    if n_bins > MAX_SPLIT_BINS:
        return SEARCH_TOO_MANY_BINS
    if extra_active:
        return SEARCH_EXTRA_PARAMS
    if feature_fraction_bylevel_active:
        return SEARCH_FEATURE_BYLEVEL
    return SEARCH_OK


def device_search_reason(code: Int) -> String:
    """A sentence a caller can raise verbatim, worded as the trainer words
    its own refusals today."""
    if code == SEARCH_OK:
        return String("the device split search can serve this configuration")
    if code == SEARCH_EXTRA_PARAMS:
        return String(
            "the device split search does not implement min_gain_to_split,"
            " max_delta_step, path_smooth, extra_trees, monotone_penalty,"
            " feature_contri, or the CEGB costs; the kernel scores from"
            " GpuSplitParams alone. Use the host split scan, which is the"
            " default (MOJOTREES_GPU_SPLIT_STRATEGY=host, or"
            " split_search=SPLIT_SEARCH_HOST)"
        )
    if code == SEARCH_FEATURE_BYLEVEL:
        return String(
            "the device split search does not implement"
            " feature_fraction_bylevel; the per-node draw it stages is taken"
            " from the tree's feature set directly. Use the host split scan,"
            " which is the default"
        )
    if code == SEARCH_TOO_MANY_BINS:
        return String(
            "the device split search supports at most 256 bins, which is the"
            " widest histogram a threadgroup's shared scratch holds"
        )
    return String("unknown split search eligibility code")


@fieldwise_init
struct GpuSplitParams(Copyable, Movable):
    """The node-independent half of a search's parameters, named as in
    `TreeParams`."""

    var lambda_l2: Float64
    var lambda_l1: Float64
    var min_child_hess: Float64
    var min_data_in_leaf: Int
    var cat: CategoricalParams

    @staticmethod
    def default() -> GpuSplitParams:
        return GpuSplitParams(1.0, 0.0, 1e-3, 0, CategoricalParams.default())


def _f32_bound(value: Float64) -> Float32:
    """A monotone output bound in the device's Float32. `monotone.NO_BOUND`
    is the largest finite Float64, which is not representable, so both
    sentinels map to the largest finite Float32 and stay sentinels."""
    var limit = Float64(Float32.MAX_FINITE)
    if value >= limit:
        return Float32.MAX_FINITE
    if value <= -limit:
        return -Float32.MAX_FINITE
    return Float32(value)


def _bitset_from_words(words: List[Int32], offset: Int) -> CatBitset:
    """Reassemble a 256-bit category set from the record's 16-bit words."""
    var bitset = cat_empty()
    for b in range(1, MAX_SPLIT_BINS):
        var w = words[offset + IREC_CAT0 + b // CAT_WORD_BITS]
        if (w & Int32(1 << (b % CAT_WORD_BITS))) != Int32(0):
            cat_add(bitset, b)
    return bitset


# --- Host-side searcher ---------------------------------------------------


struct SplitNodeRequest(Copyable, Movable):
    """One leaf of a frontier, as a batched search takes it.

    Everything here is per node; everything that is per tree or per run
    (the two lambdas, the child floors, the categorical parameters, the
    monotone sign vector, the missing-bin table) stays on the searcher or
    on `GpuSplitParams`, so a batch stages only what actually varies.

    The record slot a node's answer lands in is its position in the list
    the batch was given, so a caller keeps its frontier and its records in
    the same order and reads them back by index.
    """

    var hist_slot: Int
    """Which histogram inside the buffer handed to the batch this node's
    scan reads, in units of `3 * n_features * n_bins` Int32 words. Zero for
    a single-node buffer; the pool slot for a caller holding a level's
    histograms at once."""
    var features: List[Int]
    """This node's active features, global ids in scan order. Empty leaves
    the record's current set alone, which is the tree-level set a caller
    broadcast with `set_features`; a per-node draw
    (`feature_fraction_bynode`) passes its own."""
    var allowed: List[Bool]
    """The interaction-constraint mask by global feature id, empty for
    "every feature allowed"."""
    var bounds: OutputBounds
    """The node's monotone output interval, which is the one float
    parameter that differs between the leaves of one tree."""
    var node: Int
    """This leaf's node id in the tree being grown, or -1 for "not supplied".

    Read by exactly one rule, `random_strength`, whose draw is keyed by
    (seed, tree, **node**, feature, bin); every other rule on a frontier
    request is a property of the histogram or of the feature set and does not
    care which node it belongs to. -1 is the default and is fine while the
    noise is off; with it on, a batch carrying -1 is refused rather than
    drawing every node of the tree from one stream, which is the refusal
    `ExtraTreeParams.needs_node_identity` makes on the host."""

    def __init__(
        out self,
        hist_slot: Int = 0,
        var features: List[Int] = [],
        var allowed: List[Bool] = [],
        var bounds: OutputBounds = OutputBounds.unbounded(),
        node: Int = -1,
    ):
        self.hist_slot = hist_slot
        self.features = features^
        self.allowed = allowed^
        self.bounds = bounds^
        self.node = node


def _launch_search(
    mut ctx: DeviceContext,
    mut hist: DeviceBuffer[DType.int32],
    mut node: DeviceBuffer[DType.int32],
    mut feat: DeviceBuffer[DType.int32],
    mut allow: DeviceBuffer[DType.int32],
    mut missing: DeviceBuffer[DType.int32],
    mut catn: DeviceBuffer[DType.int32],
    mut mono: DeviceBuffer[DType.int32],
    mut fparam: DeviceBuffer[DType.float32],
    mut slot_i: DeviceBuffer[DType.int32],
    mut slot_f: DeviceBuffer[DType.float32],
    mut rec_i: DeviceBuffer[DType.int32],
    mut rec_f: DeviceBuffer[DType.float32],
    n_bins: Int,
    hist_size: Int,
    feat_stride: Int,
    widest_slots: Int,
    record_base: Int,
    n_records: Int,
    min_data_in_leaf: Int,
    constrained: Bool,
    cat_onehot_max: Int,
    cat_max_threshold: Int,
    cat_min_group: Int,
    wide: Bool = False,
    primitives: Bool = True,
    gain_form: Int = DEFAULT_GAIN_FORM,
    score_function: Int = SCORE_L2,
) raises:
    """The launch without a noise plane, which is every caller that does not
    stage one.

    An overload rather than a defaulted argument because the plane is a
    `DeviceBuffer` and there is no null one to default to. A one-element
    window onto `fparam` stands in for the plane and is never dereferenced:
    `noisy` is zero, every read of that pointer is behind it, and the window
    is the same Float32 element type so nothing is reinterpreted even in
    principle. A window rather than a fresh buffer because this is a
    per-launch path and it must not allocate.

    **`gpu_resident_round`'s device-resident loop lands here, so
    `random_strength` does not reach that path.** That is a wiring gap and
    not a semantic one: the resident loop already holds the searcher, so it
    would pass `searcher.noise_dev` and `searcher.noise_stdev > 0.0` to the
    overload below and stage a plane per node the way `enqueue` does. It is
    left alone here because that file belongs to another lane this round.
    """
    var unread = fparam.create_sub_buffer[DType.float32](0, 1)
    _launch_search(
        ctx,
        hist,
        node,
        feat,
        allow,
        missing,
        catn,
        mono,
        fparam,
        unread,
        slot_i,
        slot_f,
        rec_i,
        rec_f,
        n_bins,
        hist_size,
        feat_stride,
        widest_slots,
        record_base,
        n_records,
        min_data_in_leaf,
        constrained,
        cat_onehot_max,
        cat_max_threshold,
        cat_min_group,
        wide,
        primitives,
        gain_form,
        False,
        score_function,
    )


def _launch_search(
    mut ctx: DeviceContext,
    mut hist: DeviceBuffer[DType.int32],
    mut node: DeviceBuffer[DType.int32],
    mut feat: DeviceBuffer[DType.int32],
    mut allow: DeviceBuffer[DType.int32],
    mut missing: DeviceBuffer[DType.int32],
    mut catn: DeviceBuffer[DType.int32],
    mut mono: DeviceBuffer[DType.int32],
    mut fparam: DeviceBuffer[DType.float32],
    mut noise: DeviceBuffer[DType.float32],
    mut slot_i: DeviceBuffer[DType.int32],
    mut slot_f: DeviceBuffer[DType.float32],
    mut rec_i: DeviceBuffer[DType.int32],
    mut rec_f: DeviceBuffer[DType.float32],
    n_bins: Int,
    hist_size: Int,
    feat_stride: Int,
    widest_slots: Int,
    record_base: Int,
    n_records: Int,
    min_data_in_leaf: Int,
    constrained: Bool,
    cat_onehot_max: Int,
    cat_max_threshold: Int,
    cat_min_group: Int,
    wide: Bool = False,
    primitives: Bool = True,
    gain_form: Int = DEFAULT_GAIN_FORM,
    noisy: Bool = False,
    score_function: Int = SCORE_L2,
) raises:
    """Enqueue the two kernels of one search, over `n_records` consecutive
    nodes starting at `record_base`.

    One node and a whole frontier take the same two launches; the batch is
    the grid's second dimension. Everything that varies per node (the
    feature set, the allow mask, the float parameters, the histogram
    offset) is read from a per-record slot, and everything that does not
    (the bin count, the tree's monotone vector, the categorical and
    minimum-rows parameters) stays a kernel argument.

    A free function over the context and the buffers rather than a method,
    so the histogram buffer is an ordinary argument whether it belongs to
    this searcher, to the histogram builder next to it, or to a multi-slot
    batcher holding a whole level's histograms at once.

    `wide` runs the same scan on `WIDE_SCAN_THREADS` threads per feature
    instead of one, writing the same per-slot records, so the reduction
    below is the same kernel over the same slots either way. Only
    `GpuSplitSearcher` decides it: the wide kernel scans ordinal features
    only, and the searcher is what knows whether the dataset has a
    categorical one.

    `primitives` swaps the hand-rolled shared-memory reductions for
    `gpu.primitives.block` collectives in both the wide scan and the
    cross-feature reduction. It changes no record: every reduction it
    replaces is over fixed-point Int32 sums or over Float32 maxima, both of
    which reassociate exactly. It does change the reduction's launch shape,
    which is why it is a parameter here and not a constant: the collective
    reduction runs `REDUCE_SLOT_THREADS` threads per node where the serial
    one runs a single thread, and the two launches must agree with the
    kernels they carry. The launch *count* is the same either way, two, and
    it is already the LightGBM shape: one grid over every (leaf, feature)
    task, not one launch per feature or per leaf. See
    `split_primitives_requested` for the switch and
    `GpuSplitSearcher.set_primitives` for the in-process override.

    `gain_form` reaches every scan kernel as a launch argument rather than a
    second instantiation, for the same reason `primitives` does: the arms
    have to be alternated inside one process. Unlike `primitives` and
    `wide`, **this one changes records.** See
    `GpuSplitSearcher.set_gain_form`.

    `noisy` is `random_strength`, and it is the second arm here that changes
    records. `noise` is the per-(record, slot, bin) plane the host filled
    from `random_score_plane`; when `noisy` is False it is a one-element
    placeholder no kernel reads, and every kernel takes the instruction
    sequence it took before the parameter existed.

    `score_function` is the third, and it is not an arm at all in the sense
    the other two are: it is CatBoost's parameter, `SCORE_L2` or
    `SCORE_COSINE`, and it selects which functional the scan maximizes
    rather than which spelling of one functional it uses. It reaches the
    kernels the same way `gain_form` does, as a launch argument and not a
    second instantiation. Defaulted to `SCORE_L2` so that every caller that
    does not name it -- including `gpu_resident_round`'s device-resident
    loop, which calls this directly -- makes the launch it made before the
    parameter existed. The oblivious level search deliberately does not take
    it; `_launch_oblivious_search` says why."""
    # The whole dispatch sits behind a compile-time accelerator test, so a
    # CPU-only extension build never instantiates any of these kernels and
    # never asks the backend for a GPU architecture it was not built with.
    # An accelerator machine cannot reproduce that failure, which is why it
    # is a guard here rather than a test somewhere.
    comptime if not has_accelerator():
        raise Error(
            "the device split search needs an accelerator; this binary was"
            " built without one"
        )
    else:
        if wide and primitives:
            ctx.enqueue_function[_scan_slot_wide_primitive_kernel](
                hist.unsafe_ptr(),
                node.unsafe_ptr(),
                feat.unsafe_ptr(),
                allow.unsafe_ptr(),
                missing.unsafe_ptr(),
                mono.unsafe_ptr(),
                fparam.unsafe_ptr(),
                noise.unsafe_ptr(),
                slot_i.unsafe_ptr(),
                slot_f.unsafe_ptr(),
                Int32(n_bins),
                Int32(hist_size),
                Int32(record_base),
                Int32(feat_stride),
                Int32(min_data_in_leaf),
                Int32(1) if constrained else Int32(0),
                Int32(gain_form),
                Int32(1) if noisy else Int32(0),
                Int32(score_function),
                grid_dim=(widest_slots, n_records),
                block_dim=WIDE_SCAN_THREADS,
            )
        elif wide:
            ctx.enqueue_function[_scan_slot_wide_kernel](
                hist.unsafe_ptr(),
                node.unsafe_ptr(),
                feat.unsafe_ptr(),
                allow.unsafe_ptr(),
                missing.unsafe_ptr(),
                mono.unsafe_ptr(),
                fparam.unsafe_ptr(),
                noise.unsafe_ptr(),
                slot_i.unsafe_ptr(),
                slot_f.unsafe_ptr(),
                Int32(n_bins),
                Int32(hist_size),
                Int32(record_base),
                Int32(feat_stride),
                Int32(min_data_in_leaf),
                Int32(1) if constrained else Int32(0),
                Int32(gain_form),
                Int32(1) if noisy else Int32(0),
                Int32(score_function),
                grid_dim=(widest_slots, n_records),
                block_dim=WIDE_SCAN_THREADS,
            )
        else:
            ctx.enqueue_function[_scan_slot_kernel](
                hist.unsafe_ptr(),
                node.unsafe_ptr(),
                feat.unsafe_ptr(),
                allow.unsafe_ptr(),
                missing.unsafe_ptr(),
                catn.unsafe_ptr(),
                mono.unsafe_ptr(),
                fparam.unsafe_ptr(),
                noise.unsafe_ptr(),
                slot_i.unsafe_ptr(),
                slot_f.unsafe_ptr(),
                Int32(n_bins),
                Int32(hist_size),
                Int32(record_base),
                Int32(feat_stride),
                Int32(min_data_in_leaf),
                Int32(1) if constrained else Int32(0),
                Int32(cat_onehot_max),
                Int32(cat_max_threshold),
                Int32(cat_min_group),
                Int32(gain_form),
                Int32(1) if noisy else Int32(0),
                Int32(score_function),
                grid_dim=(widest_slots, n_records),
                block_dim=1,
            )
        if primitives:
            ctx.enqueue_function[_reduce_slots_block_kernel](
                slot_i.unsafe_ptr(),
                slot_f.unsafe_ptr(),
                rec_i.unsafe_ptr(),
                rec_f.unsafe_ptr(),
                node.unsafe_ptr(),
                fparam.unsafe_ptr(),
                Int32(record_base),
                Int32(feat_stride),
                grid_dim=n_records,
                block_dim=REDUCE_SLOT_THREADS,
            )
        else:
            ctx.enqueue_function[_reduce_slots_kernel](
                slot_i.unsafe_ptr(),
                slot_f.unsafe_ptr(),
                rec_i.unsafe_ptr(),
                rec_f.unsafe_ptr(),
                node.unsafe_ptr(),
                fparam.unsafe_ptr(),
                Int32(record_base),
                Int32(feat_stride),
                grid_dim=n_records,
                block_dim=1,
            )

def _launch_oblivious_search(
    mut ctx: DeviceContext,
    mut hist: DeviceBuffer[DType.int32],
    mut node: DeviceBuffer[DType.int32],
    mut feat: DeviceBuffer[DType.int32],
    mut allow: DeviceBuffer[DType.int32],
    mut missing: DeviceBuffer[DType.int32],
    mut catn: DeviceBuffer[DType.int32],
    mut mono: DeviceBuffer[DType.int32],
    mut fparam: DeviceBuffer[DType.float32],
    mut slot_i: DeviceBuffer[DType.int32],
    mut slot_f: DeviceBuffer[DType.float32],
    mut rec_i: DeviceBuffer[DType.int32],
    mut rec_f: DeviceBuffer[DType.float32],
    n_bins: Int,
    hist_size: Int,
    feat_stride: Int,
    widest_slots: Int,
    level_record: Int,
    leaf_base: Int,
    n_leaves: Int,
    min_data_in_leaf: Int,
    constrained: Bool,
    has_categorical: Bool,
    primitives: Bool = True,
    gain_form: Int = DEFAULT_GAIN_FORM,
    score_function: Int = SCORE_L2,
) raises:
    """The level launch without a noise plane, which is every caller that does
    not stage one -- `gpu_resident_round`'s device-owned oblivious plane
    included.

    An overload rather than a defaulted argument for the reason
    `_launch_search`'s own no-noise overload gives: the plane is a
    `DeviceBuffer` and there is no null one to default to. A one-element
    window onto `fparam` stands in and is never dereferenced, since `noisy` is
    zero and every read of that pointer is behind it; a window rather than a
    fresh buffer because this is a per-launch path and it must not allocate.

    **`gpu_resident_round.grow_tree_device_oblivious` lands here, so
    `random_strength` does not reach the device-owned oblivious plane.** That
    is a wiring gap and not a semantic one -- the plane holds the searcher and
    would stage a plane per level exactly as `enqueue_oblivious_level` does --
    and it is currently unreachable anyway: `_check_device_search_supported`
    refuses `ExtraTreeParams.is_active()`, which still names `random_strength`,
    before any oblivious growth begins. Making that term's removal *earnable*
    is what this lane built; performing the removal is not this lane's.
    """
    var unread = fparam.create_sub_buffer[DType.float32](0, 1)
    _launch_oblivious_search(
        ctx,
        hist,
        node,
        feat,
        allow,
        missing,
        catn,
        mono,
        fparam,
        unread,
        slot_i,
        slot_f,
        rec_i,
        rec_f,
        n_bins,
        hist_size,
        feat_stride,
        widest_slots,
        level_record,
        leaf_base,
        n_leaves,
        min_data_in_leaf,
        constrained,
        has_categorical,
        primitives,
        gain_form,
        score_function,
        False,
    )


def _launch_oblivious_search(
    mut ctx: DeviceContext,
    mut hist: DeviceBuffer[DType.int32],
    mut node: DeviceBuffer[DType.int32],
    mut feat: DeviceBuffer[DType.int32],
    mut allow: DeviceBuffer[DType.int32],
    mut missing: DeviceBuffer[DType.int32],
    mut catn: DeviceBuffer[DType.int32],
    mut mono: DeviceBuffer[DType.int32],
    mut fparam: DeviceBuffer[DType.float32],
    mut noise: DeviceBuffer[DType.float32],
    mut slot_i: DeviceBuffer[DType.int32],
    mut slot_f: DeviceBuffer[DType.float32],
    mut rec_i: DeviceBuffer[DType.int32],
    mut rec_f: DeviceBuffer[DType.float32],
    n_bins: Int,
    hist_size: Int,
    feat_stride: Int,
    widest_slots: Int,
    level_record: Int,
    leaf_base: Int,
    n_leaves: Int,
    min_data_in_leaf: Int,
    constrained: Bool,
    has_categorical: Bool,
    primitives: Bool = True,
    gain_form: Int = DEFAULT_GAIN_FORM,
    score_function: Int = SCORE_L2,
    noisy: Bool = False,
) raises:
    """Enqueue the two kernels of one oblivious *level* search: the cross-leaf
    scan and the ordinary cross-feature reduction over the single record the
    scan wrote into.

    `noise` is the level record's `(slot, bin)` plane, filled by
    `GpuSplitSearcher.stage_random_score_level` from
    `oblivious_score_plane`; when `noisy` is False it is a one-element window
    the kernel never reads. **The level's plane is a single record's plane**,
    because the level is scored as one candidate set and elects one split, so
    it costs one `n_features * n_bins` row rather than one per leaf.

    **Two launches, which is the whole point.** A level search costs exactly
    what a node search costs, because the sum over the level's leaves is the
    innermost loop of `_scan_slot_oblivious_kernel` rather than a launch beside
    it. The census this holds open is written out in
    `gpu_resident_round.oblivious_launch_census`: at depth 6 a tree is 62
    command buffers fused and 68 with a standalone reduce, against a queue that
    is 64 deep and whose per-launch enqueue cost is measured to roughly double
    past that. Anything added here is added six times a tree and is spent at
    the point where it is most expensive.

    The reduction is `_reduce_slots_kernel` unchanged, over
    `record_base = level_record` and one record. Nothing about it is oblivious:
    it walks the level's per-feature slots in ascending order, accepts a new
    best only on a strictly greater gain, and fills in `FREC_LEFT_VALUE`,
    `FREC_RIGHT_VALUE` and `FREC_PARENT_VALUE` from the sums the scan left. For
    a level those three are the level's aggregate values and not any one leaf's,
    which is correct as a summary and is deliberately **not** what a commit
    should write onto a node: an oblivious commit derives each leaf's own
    children from that leaf's own histogram at the winning candidate. The
    record's gain, feature, bin and default direction are the level's decision
    and are exactly what a commit needs.

    `has_categorical` refuses rather than narrows. The scan skips a categorical
    feature, so a dataset with one would be searched over a strictly smaller
    candidate set than the CPU grower searches, and the accuracy gate for this
    mode is node-identity with no tolerance. A refusal at the launch is a
    reachable, named failure; a quietly narrower search is a wrong tree that
    looks right.

    `score_function` does NOT refuse, and the fact that it once did is the
    point of the parameter. **A level's Cosine score is a ratio of two
    cross-leaf accumulators with a single square root taken at the end, not a
    sum of per-leaf ratios**, and a leaf loop that could only do `total +=
    gain` therefore had nothing to add. `_scan_slot_oblivious_kernel` now
    carries the two accumulators and takes the one root, and it carries the
    illegal-leaf substitution -- unsplit terms rather than `0.0` -- that
    `split.find_best_split_shared` carries, which is the one place the two
    functionals genuinely differ rather than merely rounding differently. So
    the refusal is retired because the thing it stood in for is written, which
    is the only reason a refusal in this package may be retired.

    Range-checked with the host's own `check_score_function`, which refuses an
    unknown code rather than resolving it to `SCORE_L2`. That direction is
    load-bearing: a third selector added later and not taught to this kernel
    must fail here, because the alternative is that it silently receives an L2
    answer under its own label.

    The launch count is unchanged and that is a precondition rather than a
    happy result. The Cosine accumulation is the same leaf loop of the same
    scan kernel with a second accumulator beside the first, so a level still
    costs two command buffers and `oblivious_launch_census(6)` is still 62.
    Three hundred and sixty trees of six levels is the shape this mode will be
    run at; a correct kernel that cost a launch per leaf per level would be
    360 * 126 extra command buffers and would have to be rewritten before it
    could ship."""
    comptime if not has_accelerator():
        raise Error(
            "the device split search needs an accelerator; this binary was"
            " built without one"
        )
    else:
        if n_leaves < 1:
            raise Error("an oblivious level holds at least one leaf")
        if n_leaves > OBLIVIOUS_MAX_LEAVES:
            raise Error(
                String(
                    "an oblivious level of ",
                    n_leaves,
                    " leaves is past the ",
                    OBLIVIOUS_MAX_LEAVES,
                    " this scan reserves per-leaf state for, which is depth"
                    " 6 and CatBoost's default; depth 7 is over the measured"
                    " 64-buffer queue knee whatever this kernel does",
                )
            )
        check_score_function(score_function)
        if has_categorical:
            raise Error(
                "the oblivious cross-leaf scan does not search category"
                " partitions: the many-vs-many arm walks prefixes of each"
                " node's own gradient ordering and a level has one ordering"
                " per leaf, which need not agree, so there is no single"
                " prefix to walk"
            )
        ctx.enqueue_function[_scan_slot_oblivious_kernel](
            hist.unsafe_ptr(),
            node.unsafe_ptr(),
            feat.unsafe_ptr(),
            allow.unsafe_ptr(),
            missing.unsafe_ptr(),
            catn.unsafe_ptr(),
            mono.unsafe_ptr(),
            fparam.unsafe_ptr(),
            noise.unsafe_ptr(),
            slot_i.unsafe_ptr(),
            slot_f.unsafe_ptr(),
            Int32(n_bins),
            Int32(hist_size),
            Int32(level_record),
            Int32(leaf_base),
            Int32(n_leaves),
            Int32(feat_stride),
            Int32(min_data_in_leaf),
            Int32(1) if constrained else Int32(0),
            Int32(gain_form),
            Int32(score_function),
            Int32(1) if noisy else Int32(0),
            grid_dim=widest_slots,
            block_dim=1,
        )
        if primitives:
            ctx.enqueue_function[_reduce_slots_block_kernel](
                slot_i.unsafe_ptr(),
                slot_f.unsafe_ptr(),
                rec_i.unsafe_ptr(),
                rec_f.unsafe_ptr(),
                node.unsafe_ptr(),
                fparam.unsafe_ptr(),
                Int32(level_record),
                Int32(feat_stride),
                grid_dim=1,
                block_dim=REDUCE_SLOT_THREADS,
            )
        else:
            ctx.enqueue_function[_reduce_slots_kernel](
                slot_i.unsafe_ptr(),
                slot_f.unsafe_ptr(),
                rec_i.unsafe_ptr(),
                rec_f.unsafe_ptr(),
                node.unsafe_ptr(),
                fparam.unsafe_ptr(),
                Int32(level_record),
                Int32(feat_stride),
                grid_dim=1,
                block_dim=1,
            )


struct GpuSplitSearcher(Movable):
    """Device-resident split search for one dataset shape.

    Construct once per training session next to the histogram builder, call
    `set_features` once per tree, `set_allowed`/`set_monotone` per node, and
    `search` (or `enqueue` + `download`) per node. Nothing is allocated per
    node: every buffer is sized at construction from `n_features`, `n_bins`,
    and `max_records`.

    One node at a time, or a whole frontier
    ---------------------------------------
    Every per-node table has one slot per record: the feature set, the allow
    mask, the float parameters (which carry the node's monotone output
    bounds), and the histogram offset. That is what lets `enqueue_frontier`
    stage a bounded set of leaves, launch them all, and wait once, instead
    of waiting once per node as the node-at-a-time loop does. The two paths
    run the same two kernels over the same slots and return the same
    records; the batch only changes how many nodes one launch covers and
    how often the host blocks.

    The staging contract, which is what a caller has to respect:

    - `enqueue_frontier` writes every record's slot first and then the
      tables cross, so nothing is overwritten while a copy of it is in
      flight. `download_frontier` is the batch's single wait, and no new
      staging may begin before it (or an explicit `synchronize`).
    - `enqueue` and `search` stage one record and copy, which is the
      node-at-a-time loop's existing contract: a node's `enqueue` is
      followed by its `download` before the next node stages.

    One copy, not four
    ------------------
    The four per-node tables live at fixed offsets inside one pinned
    staging allocation and one device allocation, and `_copy_tables` moves
    the whole thing in a single `enqueue_copy`. On Metal that is the
    difference between four full-queue drains and one, because a copy there
    drains regardless of its size (**measured** by disassembly,
    `docs/GPU_PORTABILITY.md` section 6.1). **Four drains is not four waits.**
    None of the four blocks on a device answer the host needs next, and under
    section 6.1.1, withdrawn 2026-08-16, that is what separates a copy count
    from a time: the collapse this is part of **measured** 0.016 seconds at
    1,000,000 x 50 against a registered prediction of 0.64, a null under M0
    (`bench/results/session3_2026-08-16/RESULTS.md`). What one copy instead of
    four earns is three fewer ordering points, one staging lifetime instead of
    four, and a device that cannot be left holding three fresh tables and one
    stale one. `set_table_upload_hoisting` puts the four-copy arm back at run
    time for an interleaved measurement, and `_copy_tables` argues why the two
    arms leave the device holding identical bytes.
    """

    var ctx: DeviceContext
    var n_features: Int
    var n_bins: Int
    var max_records: Int
    # The histogram this searcher owns, in `GpuHistogramBuilder`'s layout, for
    # callers that stage a histogram through the host. The zero-copy path
    # passes the builder's own buffer to `enqueue` instead.
    #
    # Allocated on first use rather than at construction, because neither
    # device search path touches it: `enqueue` and `enqueue_frontier` are
    # handed the builder's buffer, and only `upload_histogram` and `search`,
    # which exist so the search is exercisable on its own, read this one.
    # At the default 255 bins and a 50-feature fit that is 153 KB of device
    # memory per searcher that a fit allocated and never addressed. Until
    # `_ensure_hist` runs it holds a single placeholder element, since a
    # zero-length device buffer is not portable.
    var hist_dev: DeviceBuffer[DType.int32]
    var hist_owned: Bool
    """Whether `hist_dev` has been sized for a `3 * n_features * n_bins`
    histogram yet. False on a searcher that has only ever been driven by the
    trainer."""
    # The one device allocation the four per-node tables live in, and the
    # four windows onto it that the kernels are handed. See `_copy_tables`
    # for why they share an allocation and `_feat_off` for the layout.
    #
    # `tables_dev` is never passed to a kernel and never read except by the
    # packed copy. It is a field rather than a local so that it outlives the
    # four sub-buffer views below, which alias its storage.
    var tables_dev: DeviceBuffer[DType.int32]
    # Per-record tables. `feat_dev` and `allow_dev` are strided by
    # `n_features` rather than by a batch's slot count, so narrowing one
    # node's feature set never moves another node's row, which is the same
    # choice `GpuLeafBatcher` makes for its item tables.
    #
    # Each is a `create_sub_buffer` window onto `tables_dev` rather than its
    # own allocation. Nothing outside this struct can tell: a sub-buffer is
    # a `DeviceBuffer` of the same element type and length, its
    # `unsafe_ptr()` is the window's base, and `enqueue_copy` into it lands
    # at the window. `gpu_resident_round` reads all four of these fields
    # directly and `_launch_search` hands all four to the kernels; both see
    # what they saw before.
    var node_dev: DeviceBuffer[DType.int32]
    var feat_dev: DeviceBuffer[DType.int32]
    var allow_dev: DeviceBuffer[DType.int32]
    var missing_dev: DeviceBuffer[DType.int32]
    var catn_dev: DeviceBuffer[DType.int32]
    var mono_dev: DeviceBuffer[DType.int32]
    var fparam_dev: DeviceBuffer[DType.float32]
    var slot_i_dev: DeviceBuffer[DType.int32]
    var slot_f_dev: DeviceBuffer[DType.float32]
    # Both record planes in one allocation, so that one `enqueue_copy` moves
    # the whole record set instead of one per plane. The layout, and why it is
    # plane-major rather than record-major, is at `SPLIT_RECORD_WORDS`.
    #
    # A field rather than a local for the same reason `tables_dev` is one: the
    # two sub-buffer views below alias its storage and must not outlive it. It
    # is declared before them for the same reason.
    var records_dev: DeviceBuffer[DType.int32]
    var rec_i_dev: DeviceBuffer[DType.int32]
    var rec_f_dev: DeviceBuffer[DType.float32]
    # Pinned staging for the per-node tables, so a node's parameters upload
    # as an ordinary one-way copy rather than through `map_to_host`, which
    # moves the buffer both ways on every use. One slot per record: see the
    # staging contract above.
    #
    # One-way, not asynchronous. On Metal `enqueue_copy` is a synchronous
    # full-queue drain in both directions (**measured** by disassembly,
    # `docs/GPU_PORTABILITY.md` section 6.1), so an upload here is an ordering
    # point and the copy below has to be counted as a drain, not as an
    # enqueue. That is why there is one buffer and not four.
    #
    # It is a drain and not a wait. Section 6.1.1, withdrawn 2026-08-16, took
    # back the step that turned each such drain into time: nothing is queued
    # behind these uploads, and draining a queue that holds nothing costs
    # nothing. So the reason for one buffer instead of four is the ordering
    # and staleness argument above, plus one staging lifetime to reason about
    # instead of four. It is not a predicted saving, and none may be quoted.
    #
    # One pinned buffer holding all four staged tables end to end, in the
    # same order and at the same offsets as `tables_dev`, so the packed
    # upload is one `enqueue_copy` from its base. The float parameter block
    # occupies the last region and is written through a Float32 view of it:
    # both element types are four bytes wide, so the region boundaries fall
    # on both alignments and an Int32 word count addresses either.
    var stage_tables: HostBuffer[DType.int32]
    var host_i: HostBuffer[DType.int32]
    """Pinned download staging, sized for the **whole** packed record set and
    not just for the integer plane, so that the packed pinned arm needs no
    allocation of its own. The pair arms use its first `max_records *
    SPLIT_IWORDS` words and leave the rest untouched.

    Read only by the two pinned arms of `download_words`. Every other arm
    lands in `plain_words` and never touches this."""
    var host_f: HostBuffer[DType.float32]
    """The float plane's pinned staging, for the pair arms only."""
    var plain_words: List[Int32]
    """The unpinned download destination, `max_records * SPLIT_RECORD_WORDS`
    words in the layout `SPLIT_RECORD_WORDS` describes.

    **This field's type is the correctness argument for the arm that ships**,
    so it is a `List` on purpose and must stay one. `docs/GPU_PORTABILITY.md`
    section 6.5.1 establishes by execution that `enqueue_copy` on Metal has
    two implementations and that the destination picks which: into memory from
    `DeviceContext.enqueue_create_host_buffer` it enqueues an asynchronous blit
    and returns, and into an arbitrary host pointer it commits, waits, and
    memcpys. A `List`'s storage comes from Mojo's heap allocator and there is
    no path by which `enqueue_create_host_buffer` could have produced it, so
    the copy into it takes the synchronous path and the drain is inside the
    copy. That is what lets the unpinned arms carry no `synchronize()`.

    Swapping this for a `HostBuffer` to save an allocation would silently make
    the shipped readback wrong, and wrong in the worst available way: the blit
    usually wins the race under a small fixture and loses it under a real
    histogram, so a test would pass and a fit would read the previous split
    record for every node. `download_words` branches on
    `ReadbackTransport.pinned_destination` rather than on an arm code, so the
    branch that omits the wait cannot be reached with a pinned destination;
    this docstring is why that indirection is there."""
    var active: List[Int]
    """The most recently broadcast feature set, and what `n_active`
    reports. A record whose own set was narrowed by `set_features(...,
    record=r)` carries its slot count in `active_len` instead."""
    var active_len: List[Int]
    """Feature slots per record, one entry per record slot."""
    var missing_bin: List[Int]
    var cat_n: List[Int]
    var mono_host: List[Int32]
    """The monotone vector this searcher last wrote into `mono_dev`, one
    entry per feature. The host's mirror of a device buffer nothing on the
    device writes, which is what lets `set_monotone` skip a repeat; see
    there for the enumeration that makes the mirror trustworthy."""
    var mono_uploaded: Bool
    """Whether `mono_host` is known to describe `mono_dev`. Set by the
    constructor, which writes the all-free vector itself, and by every
    `set_monotone` that maps. It exists so that the skip is a claim about a
    write this searcher performed rather than about a buffer's initial
    contents, which no backend promises."""
    var constrained: Bool
    var wide_scan: Bool
    """Whether the per-feature scan runs on a threadgroup rather than on one
    thread. Decided once at construction by `wide_scan_for`, because both
    inputs are fixed there: the environment switch, and whether this
    dataset declares a categorical feature, which the wide kernel does not
    scan. Reported by `describe_scan`."""
    var use_primitives: Bool
    """Whether the reductions run as `gpu.primitives.block` collectives
    rather than as the hand-rolled shared-memory loops. Read once at
    construction from `MOJOTREES_GPU_SPLIT_PRIMITIVES` and settable
    afterwards, so one process can hold both arms; see `set_primitives`.
    Unlike `wide_scan` this has no dataset precondition, because both
    reductions it replaces are exact under reassociation on every dataset:
    the collectives choose the same split as the loops, whatever the
    histogram."""
    var hoist_tables: Bool
    """Whether the four per-node tables cross in one packed copy rather than
    in four, and whether an unchanged monotone vector is re-mapped. Read once
    at construction from `MOJOTREES_GPU_SPLIT_TABLE_PACK` and settable
    afterwards, so one process can hold both arms; see
    `set_table_upload_hoisting`."""
    var readback: Int
    """Which `ReadbackTransport` `download_words` executes.

    `READBACK_PLAIN_ONE` by default, read once at construction from
    `MOJOTREES_GPU_READBACK` and settable afterwards, so one process can hold
    both arms; see `set_readback_transport`. Like `set_primitives` and
    `set_table_upload_hoisting` and unlike `set_gain_form`, this returns the
    same records whichever arm is live: it moves the same words out of the
    same device buffer and only the number of copies and the kind of
    destination differ."""
    var api_is_metal: Bool
    """Whether this searcher's context reports the Metal API.

    Read once at construction, because `ctx.api()` returns a `String` and
    `download_words` is on the per-split path. It feeds
    `require_readback_correct`, which is what keeps the two measured-wrong
    transports out of a fit; see `set_readback_transport`."""
    var gain_form_code: Int
    """Which gain form and right-hand subtraction rule the scans use.
    `GAIN_FORM_CROSS` or `GAIN_FORM_SUBTRACTIVE`, read once at construction
    from `MOJOTREES_GPU_SPLIT_GAIN_FORM` and settable afterwards. **The one
    arm on this searcher that changes a record**; see `set_gain_form`."""
    var score_function_code: Int
    """Which functional the scans maximize: `SCORE_L2` or `SCORE_COSINE`.

    `SCORE_L2` at construction and there is no environment variable for it,
    which is the difference between this and `gain_form_code`: this is
    CatBoost's `score_function` parameter arriving from a fit, not a numeric
    arm of one functional that a benchmark alternates. `set_score_function`
    is the only way it moves."""
    var noise_stdev: Float64
    """CatBoost's `scoreStDev`: `random_strength * random_score_scale`, or
    0.0 for "off", which is the default and every LightGBM-mode fit.

    A run-time field and not an environment variable, deliberately: both arms
    have to be reachable in one binary and one process, and the parameter is
    a per-tree quantity (the scale is the ensemble's derivative RMS times the
    model-size decay, which changes every iteration) rather than a session
    constant an environment variable could carry. See `set_random_score`."""
    var noise_seed: Int
    """`random_strength_seed`, the seed the noise stream is keyed from."""
    var noise_tree: Int
    """The tree index the noise stream is keyed from. One of the five key
    components, and the reason two trees of one fit do not reuse a draw."""
    var noise_node: List[Int]
    """The node id each record's staged plane was drawn for, or -1 for "no
    plane staged", and under `grow_policy=oblivious` it holds the level's
    DEPTH instead, written by `stage_random_score_level`; the two readings
    never mix, because a searcher grows one policy's trees and the two draws
    live in different domains. A launch with `noise_stdev > 0` and a -1 here
    is refused
    rather than run: a default 0 standing in for a node id would draw every
    node of the tree from the same stream, which is exactly what
    `ExtraTreeParams.needs_node_identity` refuses on the host."""
    var noise_dev: DeviceBuffer[DType.float32]
    """The noise plane, `max_records * n_features * n_bins` Float32 in
    (record, slot, bin) order -- the same cell order `feat_dev` and
    `allow_dev` use, one row of `n_bins` per cell.

    A one-element placeholder until `_ensure_noise` sizes it, as `hist_dev`
    is, because a zero-length device buffer is not portable and because a
    LightGBM-mode fit must not pay for a buffer it never reads."""
    var noise_owned: Bool
    """Whether `noise_dev` has been sized. See `_ensure_noise`."""
    var noise_stage: List[Float32]
    """Host mirror of `noise_dev`, staged per record and copied per launch.

    Ordinary heap memory rather than a pinned `HostBuffer`, and it is a field
    rather than a local for the reason section 6.5.1 of
    `docs/GPU_PORTABILITY.md` gives: a staging buffer that is overwritten
    before the copy it feeds has completed is a live race, so the staging
    must outlive the launch. It does, and the next round's staging is
    separated from this one by the same `enqueue`/`download` ordering
    contract the per-node tables already rely on."""

    def __init__(
        out self,
        n_features: Int,
        n_bins: Int,
        missing_bins: List[Int] = [],
        cats: CategoricalSpec = CategoricalSpec.none(),
        max_records: Int = 1,
    ) raises:
        """Size every buffer and upload the per-feature tables that do not
        change during training: the missing-bin table and the category
        counts.

        Opens a private `DeviceContext`. A searcher reading another owner's
        device buffer (the trainer integration reads the histogram
        builder's) must share that owner's context instead, so the two
        enqueue into one in-order queue; see the context overload below."""
        var ctx = DeviceContext()
        self = Self(ctx, n_features, n_bins, missing_bins, cats, max_records)

    def __init__(
        out self,
        ctx: DeviceContext,
        n_features: Int,
        n_bins: Int,
        missing_bins: List[Int] = [],
        cats: CategoricalSpec = CategoricalSpec.none(),
        max_records: Int = 1,
    ) raises:
        """Build on a caller-supplied context; the private-context form
        above lands here. Sharing the histogram builder's context is what
        makes `enqueue` over the builder's own buffer safe without a fence:
        one queue orders the histogram kernels before the scan."""
        if n_features < 1:
            raise Error("split search requires at least one feature")
        if n_bins < 1:
            raise Error("split search requires at least one bin")
        if n_bins > MAX_SPLIT_BINS:
            raise Error("split search supports at most 256 bins")
        if len(missing_bins) > 0 and len(missing_bins) != n_features:
            raise Error("missing_bins length must equal n_features")
        if max_records < 1:
            raise Error("max_records must be at least one")
        for f in range(n_features):
            if len(missing_bins) > 0 and missing_bins[f] >= n_bins:
                raise Error("missing bin index out of range")
            if cats.is_cat(f) and cats.n_categories(f) >= n_bins:
                raise Error(
                    "categorical feature has more categories than bins"
                )

        self.ctx = ctx
        self.n_features = n_features
        self.n_bins = n_bins
        self.max_records = max_records
        self.constrained = False
        self.missing_bin = List[Int](capacity=n_features)
        self.cat_n = List[Int](capacity=n_features)
        self.active = List[Int](capacity=n_features)
        self.active_len = List[Int](capacity=max_records)
        var any_cat = False
        for f in range(n_features):
            self.missing_bin.append(
                missing_bins[f] if len(missing_bins) > 0 else -1
            )
            self.cat_n.append(cats.n_categories(f) if cats.is_cat(f) else 0)
            if self.cat_n[f] >= 2:
                any_cat = True
            self.active.append(f)
        # Fixed here and not revisited: `cats` is a construction-time fact,
        # and narrowing a record's feature set later can only remove
        # features, never introduce a categorical one.
        self.wide_scan = wide_scan_for(any_cat)
        self.use_primitives = split_primitives_requested()
        self.hoist_tables = table_upload_hoisting_requested()
        self.gain_form_code = gain_form_requested()
        # CatBoost's `score_function`, which starts at LightGBM's functional
        # and moves only when a caller asks. No environment entry, for the
        # reason the field gives.
        self.score_function_code = SCORE_L2
        # Resolved here rather than per download: `ctx.api()` builds a String
        # and the readback is on the per-split path, where a device context's
        # API cannot change under it.
        self.api_is_metal = parse_api(ctx.api()) == API_METAL
        self.readback = env_readback_transport()
        require_readback_correct(self.readback, self.api_is_metal)
        _require_readback_implemented(self.readback)
        self.mono_host = List[Int32](capacity=n_features)
        for _ in range(n_features):
            self.mono_host.append(Int32(MONOTONE_FREE))
        self.mono_uploaded = False
        for _ in range(max_records):
            self.active_len.append(n_features)

        # `random_strength` starts off, which is LightGBM's behavior and this
        # module's default: no buffer, no plane, no arithmetic. Every field
        # here is inert until `set_random_score` is called with a positive
        # standard deviation.
        self.noise_stdev = 0.0
        self.noise_seed = 0
        self.noise_tree = 0
        self.noise_node = List[Int](capacity=max_records)
        for _ in range(max_records):
            self.noise_node.append(-1)
        self.noise_stage = List[Float32]()

        var table_cells = max_records * n_features
        # A placeholder until `_ensure_hist` sizes it, because a
        # zero-length device buffer is not portable. See the field.
        self.hist_dev = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.hist_owned = False
        # The same placeholder pattern, and the same reason, for the noise
        # plane; `_ensure_noise` sizes it the first time a positive
        # `random_strength` is set.
        self.noise_dev = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.noise_owned = False
        # The four per-node tables in one allocation, in the order
        # `_feat_off` fixes: node, features, allow mask, float parameters.
        # The float region is counted in Int32 words because a Float32 is
        # the same four bytes; `create_sub_buffer` takes its offset and
        # length in elements, and at four bytes either element type
        # addresses the same boundary.
        var node_words = max_records * NODE_WORDS
        var param_words = max_records * PF_WORDS
        var feat_off = node_words
        var allow_off = feat_off + table_cells
        var param_off = allow_off + table_cells
        self.tables_dev = self.ctx.enqueue_create_buffer[DType.int32](
            param_off + param_words
        )
        self.node_dev = self.tables_dev.create_sub_buffer[DType.int32](
            0, node_words
        )
        self.feat_dev = self.tables_dev.create_sub_buffer[DType.int32](
            feat_off, table_cells
        )
        self.allow_dev = self.tables_dev.create_sub_buffer[DType.int32](
            allow_off, table_cells
        )
        self.fparam_dev = self.tables_dev.create_sub_buffer[DType.float32](
            param_off, param_words
        )
        self.missing_dev = self.ctx.enqueue_create_buffer[DType.int32](
            n_features
        )
        self.catn_dev = self.ctx.enqueue_create_buffer[DType.int32](n_features)
        self.mono_dev = self.ctx.enqueue_create_buffer[DType.int32](n_features)
        self.slot_i_dev = self.ctx.enqueue_create_buffer[DType.int32](
            table_cells * SPLIT_IWORDS
        )
        self.slot_f_dev = self.ctx.enqueue_create_buffer[DType.float32](
            table_cells * SPLIT_FWORDS
        )
        # Both record planes end to end in one allocation, with the two
        # windows the kernels and `gpu_resident_round.mojo` already take. The
        # float window's offset and length are given in Int32 elements, which
        # is what `create_sub_buffer` wants and what four-byte elements make
        # unambiguous; the same argument `tables_dev` makes above.
        var rec_i_words = max_records * SPLIT_IWORDS
        var rec_f_words = max_records * SPLIT_FWORDS
        self.records_dev = self.ctx.enqueue_create_buffer[DType.int32](
            max_records * SPLIT_RECORD_WORDS
        )
        self.rec_i_dev = self.records_dev.create_sub_buffer[DType.int32](
            0, rec_i_words
        )
        self.rec_f_dev = self.records_dev.create_sub_buffer[DType.float32](
            rec_i_words, rec_f_words
        )
        self.stage_tables = self.ctx.enqueue_create_host_buffer[DType.int32](
            param_off + param_words
        )
        # Sized for the packed record set, not for the integer plane: the pair
        # arms use the first `rec_i_words` words and the packed pinned arm
        # uses all of it, so neither needs a second pinned allocation.
        self.host_i = self.ctx.enqueue_create_host_buffer[DType.int32](
            max_records * SPLIT_RECORD_WORDS
        )
        self.host_f = self.ctx.enqueue_create_host_buffer[DType.float32](
            rec_f_words
        )
        # Ordinary heap memory, which is the whole correctness argument for
        # the arm that ships; see the field.
        self.plain_words = List[Int32](
            capacity=max_records * SPLIT_RECORD_WORDS
        )
        for _ in range(max_records * SPLIT_RECORD_WORDS):
            self.plain_words.append(Int32(0))

        # Every record starts as "every feature, all allowed, one histogram
        # at offset zero", so a caller that never narrows a set and never
        # batches (the node-at-a-time loop, and every existing caller) sees
        # exactly the behavior it saw when these tables held one slot.
        var dst_node = self.stage_tables.unsafe_ptr()
        var dst_feat = dst_node.unsafe_offset(feat_off)
        var dst_allow = dst_node.unsafe_offset(allow_off)
        var dst_param = dst_node.unsafe_offset(param_off).unsafe_bitcast[
            Scalar[DType.float32]
        ]()
        for r in range(max_records):
            var nt = r * NODE_WORDS
            dst_node.unsafe_store(nt + NODE_SLOTS, Int32(n_features))
            dst_node.unsafe_store(nt + NODE_HIST_BASE, Int32(0))
            for f in range(n_features):
                dst_feat.unsafe_store(r * n_features + f, Int32(f))
                dst_allow.unsafe_store(r * n_features + f, Int32(1))
            for w in range(PF_WORDS):
                dst_param.unsafe_store(r * PF_WORDS + w, Float32(0.0))
        # One copy where there were three, and it carries the float
        # parameter region as well, which the three did not. That region was
        # previously left unwritten on the device until the first
        # `_copy_tables`, and no kernel could read it before then, so
        # uploading the zeros staged above changes no kernel's input and
        # only replaces an unspecified allocation with a defined one.
        self.ctx.enqueue_copy(dst_buf=self.tables_dev, src_ptr=dst_node)

        with self.missing_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(n_features):
                dst.unsafe_store(f, Int32(self.missing_bin[f]))
        with self.catn_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(n_features):
                var n_cat = cats.n_categories(f) if cats.is_cat(f) else 0
                dst.unsafe_store(f, Int32(n_cat))
        with self.mono_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(n_features):
                dst.unsafe_store(f, Int32(MONOTONE_FREE))
        # `mono_host` was filled with the same all-free vector above. The
        # mirror is true of the device from here, which is what lets
        # `set_monotone` decline to repeat it.
        self.mono_uploaded = True
        # RETAINED, and load-bearing. The table copy above uploads *out of*
        # `stage_tables`, a pinned buffer every subsequent node rewrites in
        # place through `_stage_params`. Section 6.5.1 of
        # docs/GPU_PORTABILITY.md measured that direction too: a pinned buffer
        # uploaded and then overwritten with no drain in between delivered the
        # overwritten bytes once in four repetitions and the original three
        # times, which is a live race and not a fixed ordering. Without this
        # wait the first node's staging could reach the device instead of the
        # constructor's zeros.
        #
        # This comment previously said the copy drained by itself and that the
        # wait was redundant on Metal, on the strength of section 6.1. 6.5.1
        # withdrew that for pinned memory, which is what this is.
        self.ctx.synchronize()

    def synchronize(self) raises:
        """Block until every enqueued device operation has completed."""
        self.ctx.synchronize()

    def n_active(self) -> Int:
        """How many feature slots the next search scans."""
        return len(self.active)

    def set_primitives(mut self, enabled: Bool):
        """Force the collective reductions on or off for this searcher,
        overriding `MOJOTREES_GPU_SPLIT_PRIMITIVES`.

        The handle an interleaved benchmark needs: the two arms have to be
        alternated inside one process and one thermal state to be comparable
        at all on this machine, which re-execing with a different
        environment cannot do. Safe to change between searches, because it
        selects a kernel at launch time and owns no state; the records
        either arm returns are the same records."""
        self.use_primitives = enabled

    def set_table_upload_hoisting(mut self, on: Bool):
        """Force the packed table upload on or off for this searcher,
        overriding `MOJOTREES_GPU_SPLIT_TABLE_PACK`.

        Off is what this module shipped before this lane: four
        `enqueue_copy` calls in `_copy_tables`, and a `map_to_host` in
        `set_monotone` on every call whether or not the vector moved. On is
        one `enqueue_copy` covering all four tables, and a `set_monotone`
        that maps only when the vector it is handed differs from the one it
        last wrote.

        Reachable at run time rather than through the environment, and for
        the same reason `set_primitives` and `GpuActiveRows.set_row_unroll`
        are: this machine's device timings drift several-fold between time
        windows, so only arms interleaved inside one process and one thermal
        state compare at all, and an environment-only knob would have forced
        a two-build comparison.

        It cannot change a record. Both arms leave the device holding the
        same bytes in the same four tables before either kernel is enqueued
        (`_copy_tables` argues that region by region) and the same monotone
        vector (`set_monotone` argues that one). What differs is only how
        many drains it takes to put them there.

        **What differs is not expected to be a time.** Both arms make the same
        number of round trips, which under `docs/GPU_PORTABILITY.md` section
        6.1.1 is the count that predicts time; they differ only in copies,
        which predict portability risk and ordering hazards. The thirteen-copy
        collapse this knob is part of **measured** 0.016 seconds at
        1,000,000 x 50 against a registered prediction of 0.64, a null under
        M0 (`bench/results/session3_2026-08-16/RESULTS.md`). Keep the knob for
        the A/B and for the staleness argument, not for a predicted win.

        Takes effect on the next `_copy_tables`, which is the next
        `enqueue`, `search` or `enqueue_frontier`, and on the next
        `set_monotone`.
        """
        self.hoist_tables = on

    def set_readback_transport(mut self, transport: Int) raises:
        """Which `ReadbackTransport` `download_words` executes, overriding
        `MOJOTREES_GPU_READBACK`.

        `READBACK_PLAIN_ONE` is the default and what ships:
        `pixi run probe-readback` measured it at 124.85 us a trip against
        `READBACK_PINNED_PAIR_SYNC`'s 202.14, arms interleaved in one process,
        with a bare `synchronize()` floor of 10.59. `READBACK_PINNED_PAIR_SYNC`
        is what this module did until 2026-08-16 and is the arm a window holds
        this one against; `READBACK_PLAIN_PAIR` and `READBACK_PINNED_ONE_SYNC`
        separate the two changes (the destination kind and the packing) so a
        window can attribute the difference to one of them rather than to
        both. `READBACK_MAP` and the two `nosync` rows raise here; see
        `_require_readback_implemented`.

        Reachable at run time rather than through the environment alone, and
        for the same reason `set_primitives`, `set_table_upload_hoisting` and
        `GpuActiveRows.set_row_unroll` are: this machine's device timings
        drift several-fold between time windows, so only arms interleaved
        inside one process and one thermal state compare at all.

        **It cannot change a record.** Every arm moves the same
        `max_records * 136` bytes out of the same `records_dev` and unpacks
        them into the same two lists. What differs is how many command buffers
        carry them and whether the wait is a `synchronize()` or the copy's own
        drain. This is a transfer shape in the sense `set_table_upload_
        hoisting` is, not a numeric arm in the sense `set_gain_form` is.

        Refuses an arm this backend is measured to get wrong before it can be
        stored, so a wrong transport cannot be reached by a fit even through
        this handle. Takes effect on the next `download_words`.
        """
        require_readback_correct(transport, self.api_is_metal)
        _require_readback_implemented(transport)
        self.readback = transport

    def describe_readback(self) raises -> String:
        """The live transport as `name buffers/trip`, for `describe_scan` and
        for a test that wants to assert which arm is live without restating
        the constant."""
        var row = readback_transport(self.readback)
        return String(
            readback_transport_name(self.readback),
            " buffers=",
            row.command_buffers,
        )

    def set_gain_form(mut self, form: Int) raises:
        """Which gain expression and right-hand subtraction rule this
        searcher's scans use, overriding `MOJOTREES_GPU_SPLIT_GAIN_FORM`.

        `GAIN_FORM_CROSS` is the default: the cancellation-free gain of
        `gpu_cross_gain` with the right-hand sums taken in the integer
        fixed-point domain by `gpu_right_sum`. `GAIN_FORM_SUBTRACTIVE` is
        what this module shipped. The arithmetic, the derived bound, and the
        standalone measurements are all at those two functions and nothing
        about them is restated here.

        **This is the one arm on this searcher that changes a record**, and
        it is numeric in the sense `histogram_gpu.set_scale_shape` is:
        `set_primitives`, `set_table_upload_hoisting`, and the wide scan are
        launch shapes and transfer shapes, and every one of them returns the
        same record by construction. This one changes the value of every
        gain, therefore possibly which candidate wins, therefore possibly a
        tree. A fixture taken under one arm does not describe the other and
        must not be diffed against it. It also moves the `min_child_hess`
        admission test, because the right-hand hessian that test reads is
        one of the quantities the arm changes.

        WHY THE TWO CHANGES ARE ONE CODE AND NOT TWO
        --------------------------------------------
        Because the four-way factorial has a cell that is worse than doing
        nothing, and a knob whose settings include a known regression is a
        trap rather than a measurement handle. Integer right-hand
        subtraction on top of the *subtractive* gain removes an
        anti-correlation the subtractive gain was quietly living off, and
        measures worse than the shipped arm at two of three settings; the
        table is at `gpu_right_sum`. If a later lane wants the factorial to
        study the mechanism, it should reach the two helpers directly rather
        than widen this code.

        WHAT THE DEFAULT DOES AND DOES NOT REST ON
        ------------------------------------------
        It rests on an exact algebraic identity, a derived error bound, and
        a standalone NumPy model of this scan. It does **not** rest on any
        measurement of mojotrees, and `ACCURACY_BUDGET.md` section 1 is
        explicit that no verdict in it is discharged until `bench/real_data`
        has been run with a before arm and an after arm. Nothing in this
        lane ran it. What is claimed is a mechanism and an order of
        magnitude, in the split-selection channel, which is the channel that
        does not self-correct.

        On cost: one divide fewer and two multiplies more per candidate,
        plus two node constants hoisted out of the candidate loop, and one
        Int32 subtract in place of one Float32 subtract per right-hand sum.
        **It is not more expensive, by inspection.** No time was measured,
        in either direction, and none may be quoted.

        Refuses an unknown code rather than defaulting to one, for the
        reason `set_scale_shape` gives: silently scoring candidates by an
        expression the caller did not ask for is the failure this module is
        arranged to prevent.

        Takes effect on the next launch, which is the next `enqueue`,
        `search`, or `enqueue_frontier`. It selects a kernel argument and
        owns no state, so it is safe to change between searches.
        """
        if form != GAIN_FORM_CROSS and form != GAIN_FORM_SUBTRACTIVE:
            raise Error(
                String(
                    "unknown split gain form: ",
                    String(form),
                    "; expected GAIN_FORM_CROSS or GAIN_FORM_SUBTRACTIVE",
                )
            )
        self.gain_form_code = form

    def gain_form(self) -> Int:
        """The gain arm `set_gain_form` last chose."""
        return self.gain_form_code

    def set_score_function(mut self, score: Int) raises:
        """Which functional this searcher's scans maximize: CatBoost's
        `score_function`, `SCORE_L2` or `SCORE_COSINE`.

        `SCORE_L2` is the default and is LightGBM's second-order gain, which
        is what every scan this module has ever run computed. `SCORE_COSINE`
        is `sum(-out * G) / sqrt(sum(out^2 * H))` over the children minus the
        same functional of the unsplit node; the arithmetic and the
        CatBoost source it was read from are at `gpu_cosine_gain` and at
        `split._cosine_pair`, and nothing about them is restated here.

        **This changes records, and it changes them more than
        `set_gain_form` does.** `set_gain_form` picks between two spellings
        of one functional whose exact values agree; this picks between two
        functionals. Gains under the two are not comparable, not on the same
        scale, and not in the same units, so a fixture taken under one says
        nothing about the other and `frontier_margin`, `host_rescan_
        recommended` and `min_gain_to_split` all measure something different
        under each. That is a property of CatBoost's parameter and not of
        this implementation.

        **Not waived when the two agree.** They agree on the argmax within
        one parent at `lambda_l2 = 0`, provably, and it is tempting to route
        a Cosine request to the cheaper L2 scan on that identity. It must
        not be: the record's gain feeds a leaf-wise queue that compares
        candidates from different parents, where `sqrt(a) - sqrt(p)` does not
        order like `a - p`, and this searcher cannot see the growth policy.
        The full statement is at the head of the Cosine section.

        REFUSED WITH A CATEGORICAL FEATURE, WHICH IS `find_best_split`'S OWN
        RULE
        --------------------------------------------------------------------
        A categorical feature is searched as category *partitions*, scored
        with the L2 gain by a search whose winner is the only candidate that
        reaches the fold. Allowing Cosine on a matrix that has one would put
        two different score functions inside one argmax, which is why
        `find_best_split` raises on exactly this pair, and it raises here for
        exactly that reason and with the same content. The check is over the
        construction-time category counts, as `set_random_score`'s is, and
        for the same reason: `cats` is fixed at construction and narrowing a
        record's feature set can only remove features.

        Refuses an unknown code rather than defaulting to one. That
        direction is load-bearing and not defensive tidiness: a third
        selector added later and not taught to these kernels must fail here,
        because the alternative is that it silently receives an L2 answer
        under its own label. `check_score_function` is the range check and
        it is the host's, so the two cannot drift.

        Takes effect on the next launch. It selects a kernel argument and
        owns no state, so it is safe to change between searches.
        """
        check_score_function(score)
        if score == SCORE_COSINE:
            for f in range(self.n_features):
                if self.cat_n[f] >= 2:
                    raise Error(
                        "score_function=cosine is implemented for numerical"
                        " thresholds only; this searcher was constructed with"
                        " a categorical feature, whose candidates are category"
                        " partitions scored with the L2 gain, so the two score"
                        " functions would end up inside one argmax"
                    )
        self.score_function_code = score

    def score_function(self) -> Int:
        """The functional `set_score_function` last chose."""
        return self.score_function_code

    def set_random_score(
        mut self, stdev: Float64, seed: Int = 0, tree_index: Int = 0
    ) raises:
        """Turn CatBoost's `random_strength` on for the trees that follow,
        with standard deviation `stdev` keyed from `seed` and `tree_index`.

        `stdev` is CatBoost's `scoreStDev` in full: `random_strength *
        derivativesStDevFromZero * modelSizeDecrease`, which on the host
        bundle is `ExtraTreeParams.random_score_stdev()`. It is not a session
        constant -- the last two factors are properties of the ensemble at
        this iteration -- so it is set per tree, from whoever owns the
        gradient vector, exactly as
        `tree_parameters_extra.random_score_scale_from_gradients` describes.
        Zero or negative turns the noise off, which is the default state, and
        is the value LightGBM mode always passes.

        **This is the second arm on this searcher that changes a record**,
        the first being `set_gain_form`. It changes it on purpose and by a
        seeded amount, which is the difference between a regularizer and a
        defect, and it is the reason the plane is host-computed: see the
        `random_strength` section at the top of this module for why stage B
        of the draw cannot cross to a Float32 device and what would break if
        it were made to.

        Refused with a categorical feature, matching `find_best_split`: a
        categorical candidate is a category set chosen by a partition search,
        so only that search's winner would be noised while every numerical
        feature had every candidate noised. That is a different regularizer
        with the same name, and half-applying it is worse than refusing it.

        Takes effect on the next `stage_random_score`; a record whose plane
        has not been staged since is refused at launch rather than searched
        without its noise.
        """
        if stdev != stdev or stdev > Float64.MAX_FINITE:
            raise Error(
                "random_strength needs a finite noise standard deviation"
            )
        if not (stdev > 0.0):
            self.noise_stdev = 0.0
            self.noise_seed = seed
            self.noise_tree = tree_index
            for r in range(self.max_records):
                self.noise_node[r] = -1
            return
        for f in range(self.n_features):
            if self.cat_n[f] >= 2:
                raise Error(
                    "random_strength is implemented for numerical thresholds"
                    " only; this searcher was constructed with a categorical"
                    " feature, whose candidates are category partitions and"
                    " cannot each be noised from the threshold scan"
                )
        self._ensure_noise()
        self.noise_stdev = stdev
        self.noise_seed = seed
        self.noise_tree = tree_index
        # A new tree or a new standard deviation invalidates every staged
        # plane: the draw is keyed by the tree index and scaled by the
        # deviation, so a plane held over would be the previous tree's noise.
        for r in range(self.max_records):
            self.noise_node[r] = -1

    def random_score_stdev(self) -> Float64:
        """The noise standard deviation `set_random_score` last chose. 0.0
        when `random_strength` is off, which is the default."""
        return self.noise_stdev

    def _ensure_noise(mut self) raises:
        """Size the noise plane, once, on the first positive
        `random_strength`. A LightGBM-mode searcher never reaches this and
        therefore never allocates it."""
        if self.noise_owned:
            return
        var cells = self.max_records * self.n_features * self.n_bins
        self.noise_dev = self.ctx.enqueue_create_buffer[DType.float32](cells)
        self.noise_stage = List[Float32](capacity=cells)
        for _ in range(cells):
            self.noise_stage.append(Float32(0.0))
        self.noise_owned = True

    def stage_random_score(mut self, record: Int, node: Int) raises:
        """Draw record `record`'s noise plane for node `node` and stage it.

        Called once per node, after `set_features` has fixed that record's
        active set and before the `enqueue` or `enqueue_frontier` that
        searches it. The draw is keyed by
        (`seed`, `tree_index`, `node`, global feature id, bin), so restaging
        the same node reproduces the same plane bit for bit, and a node whose
        feature set was narrowed or reordered gets the same numbers in the
        permuted positions.

        Stages only; the copy happens at launch, which is what keeps the host
        mirror alive across the transfer (see `noise_stage`).
        """
        self._check_record(record)
        if not (self.noise_stdev > 0.0):
            raise Error(
                "stage_random_score needs a positive noise standard"
                " deviation; call set_random_score first"
            )
        if node < 0:
            raise Error(
                "random_strength keys its draw by node id, which must be"
                " nonnegative"
            )
        var slots = self.active_len[record]
        var plane = random_score_plane(
            self.noise_stdev,
            self.noise_seed,
            self.noise_tree,
            node,
            self._record_features(record),
            self.n_bins,
        )
        var base = record * self.n_features * self.n_bins
        for i in range(slots * self.n_bins):
            self.noise_stage[base + i] = plane[i]
        self.noise_node[record] = node

    def stage_random_score_level(
        mut self, record: Int, depth: Int
    ) raises:
        """Draw an oblivious LEVEL's noise plane for the level at `depth` and
        stage it into `record`, which is the level record the level search
        writes into.

        `stage_random_score`'s twin, and the difference between them is the
        whole of what this lane resolved. That one keys its draw by a node id
        in `RANDOM_SCORE_DOMAIN`; a level has no node, which is the reason
        `random_strength` was refused here. It has a DEPTH, and CatBoost's own
        per-level redraw is keyed to nothing else -- `CalcScores` takes a
        fresh seed inside the `curDepth` loop -- so this keys by the depth in
        `OBLIVIOUS_SCORE_DOMAIN` and the objection dissolves.

        One plane per level, not per leaf. The level is one candidate set and
        elects one split, so the draw is per (feature, bin) and the leaf loop
        never sees it: the kernel adds it to the level's aggregate score after
        the cross-leaf sum and, under Cosine, after the single ratio.

        Called once per level, after `set_features` has fixed the level
        record's active set and before the `enqueue_oblivious_level` or
        `search_oblivious_level` that searches it. Restaging the same depth
        reproduces the same plane bit for bit.
        """
        self._check_record(record)
        if not (self.noise_stdev > 0.0):
            raise Error(
                "stage_random_score_level needs a positive noise standard"
                " deviation; call set_random_score first"
            )
        if depth < 0:
            raise Error(
                "an oblivious level sits at a nonnegative depth, got ", depth
            )
        var slots = self.active_len[record]
        var plane = oblivious_score_plane(
            self.noise_stdev,
            self.noise_seed,
            self.noise_tree,
            depth,
            self._record_features(record),
            self.n_bins,
        )
        var base = record * self.n_features * self.n_bins
        for i in range(slots * self.n_bins):
            self.noise_stage[base + i] = plane[i]
        self.noise_node[record] = depth

    def _record_features(self, record: Int) -> List[Int]:
        """Record `record`'s active feature ids, read back out of the staged
        feature table so the plane is keyed by exactly the features the scan
        will walk, in exactly the order it will walk them."""
        var src = self.stage_tables.unsafe_ptr().unsafe_offset(
            self._feat_off()
        )
        var slots = self.active_len[record]
        var out = List[Int](capacity=slots)
        for slot in range(slots):
            out.append(Int(src[unsafe_offset = record * self.n_features + slot][0]))
        return out^

    def _check_noise_staged(self, record_base: Int, n_records: Int) raises:
        """Refuse a launch whose noise plane was never drawn for one of its
        records. The alternative is a node searched against another node's
        noise, or against zeros, and neither announces itself."""
        if not (self.noise_stdev > 0.0):
            return
        for r in range(record_base, record_base + n_records):
            if self.noise_node[r] < 0:
                raise Error(
                    "random_strength is on but record ",
                    r,
                    " has no noise plane staged for it; call"
                    " stage_random_score(record, node) -- or, for an"
                    " oblivious level record,"
                    " stage_random_score_level(record, depth) -- after"
                    " set_features and before the search",
                )

    def _feat_off(self) -> Int:
        """First Int32 word of the feature table inside the packed tables.

        The layout, in Int32 words, of both `tables_dev` and
        `stage_tables`, which hold the same four regions at the same
        offsets:

        - `[0, max_records * NODE_WORDS)` the node table
        - `[_feat_off, + max_records * n_features)` the feature table
        - `[_allow_off, + max_records * n_features)` the allow mask
        - `[_param_off, + max_records * PF_WORDS)` the float parameters

        Recomputed from two fields rather than stored, because both are
        construction-time constants and three more fields to keep in step
        with them is three more ways for the layout to disagree with
        itself."""
        return self.max_records * NODE_WORDS

    def _allow_off(self) -> Int:
        """First Int32 word of the allow mask. See `_feat_off`."""
        return self._feat_off() + self.max_records * self.n_features

    def _param_off(self) -> Int:
        """First word of the float parameter block, counted in Int32 words
        because a Float32 is the same four bytes wide. See `_feat_off`."""
        return self._allow_off() + self.max_records * self.n_features

    def describe_scan(self) raises -> String:
        """One line for benchmark output and bug reports: which scan kernel
        this searcher launches, how wide its threadgroup is, whether its
        reductions are collectives or hand-rolled loops, and which readback
        transport brings the records home."""
        var reduction = String(
            " reduce=block-primitives threads=",
            REDUCE_SLOT_THREADS,
        ) if self.use_primitives else String(" reduce=serial threads=1")
        # Which upload arm is live belongs on this line for the same reason
        # the reduction does: an interleaved benchmark prints it next to the
        # timing, and a run whose arm cannot be read off its own output is a
        # run that proves nothing.
        var tables = String(
            " tables=packed copies=1"
        ) if self.hoist_tables else String(" tables=split copies=4")
        # And the gain arm, which belongs here more than either of the above
        # do: it is the only one of the three that changes what the records
        # say, so a result reported without it is unattributable.
        var gain = String(
            " gain=", describe_gain_form(self.gain_form_code)
        )
        # And which functional those gains are gains of, for a stronger form
        # of the same reason: `gain=` names two spellings of one expression,
        # this names two different expressions, and a number reported without
        # it cannot even be compared to another number from this searcher.
        var score = String(
            " score=", score_function_name(self.score_function_code)
        )
        # And the readback arm, for the same reason the upload arm is here:
        # it is the round trip that predicts this plane's time, a window
        # interleaves two settings of it, and a run whose arm cannot be read
        # off its own output proves nothing about either.
        var back = String(" readback=", self.describe_readback())
        # And `random_strength`, for the reason the gain arm is here: it is
        # the other arm that changes what the records say, and a stochastic
        # split rule that does not print itself is a result nobody can
        # attribute. Off is the default and prints as "off", not as nothing,
        # so a line that lacks the field is an old line rather than a quiet
        # one.
        var noise = String(" noise=off")
        if self.noise_stdev > 0.0:
            noise = String(
                " noise=stdev:",
                self.noise_stdev,
                " seed:",
                self.noise_seed,
                " tree:",
                self.noise_tree,
            )
        if self.wide_scan:
            return String(
                "scan=wide threads=",
                WIDE_SCAN_THREADS,
                reduction,
                tables,
                gain,
                score,
                back,
                noise,
            )
        return String(
            "scan=serial threads=1",
            reduction,
            tables,
            gain,
            score,
            back,
            noise,
        )

    def _check_record(self, record: Int) raises:
        if record < 0 or record >= self.max_records:
            raise Error("record index out of range")

    def _stage_features(
        mut self, features: List[Int], record: Int
    ) raises:
        """Write one record's feature slots and slot count into the pinned
        tables. Copies nothing: the caller decides when the tables go
        across, which is what lets a frontier stage every node first."""
        var node = self.stage_tables.unsafe_ptr()
        var dst = node.unsafe_offset(self._feat_off())
        var base = record * self.n_features
        for i in range(len(features)):
            dst.unsafe_store(base + i, Int32(features[i]))
        self.active_len[record] = len(features)
        node.unsafe_store(
            record * NODE_WORDS + NODE_SLOTS, Int32(len(features))
        )

    def _stage_allowed(
        mut self, allowed: List[Bool], record: Int
    ) raises:
        """Write one record's allow mask, translated from global feature
        ids into this record's slot order."""
        var pack = self.stage_tables.unsafe_ptr()
        var dst = pack.unsafe_offset(self._allow_off())
        var base = record * self.n_features
        var n_slots = self.active_len[record]
        var feat = pack.unsafe_offset(self._feat_off())
        for i in range(n_slots):
            var f = Int(feat.unsafe_load(base + i))
            var ok = True
            if len(allowed) > 0:
                ok = f < len(allowed) and allowed[f]
            dst.unsafe_store(base + i, Int32(1) if ok else Int32(0))
        for i in range(n_slots, self.n_features):
            dst.unsafe_store(base + i, Int32(0))

    def _stage_hist_base(mut self, record: Int, hist_base: Int32):
        """Write one record's histogram offset into the staged node table.

        The node table is the first region of `stage_tables`, so its record
        rows are at `record * NODE_WORDS` from the base with no further
        offset; `_feat_off` has the whole layout.

        A method rather than three open-coded stores, so that the writers of
        `NODE_HIST_BASE` can be enumerated by grepping for one name. The
        three are `enqueue`, `search` and `enqueue_frontier`, and the
        distinction matters because this word is the one thing in the four
        staged tables that a shipping loop changes between one split and the
        next, while the feature set, the allow mask and the parameter block
        usually do not."""
        self.stage_tables.unsafe_ptr().unsafe_store(
            record * NODE_WORDS + NODE_HIST_BASE, hist_base
        )

    def _copy_tables(mut self) raises:
        """Put the four staged per-node tables on the device: one copy when
        `hoist_tables` is on, four when it is off.

        Why this is one call and not four
        ---------------------------------
        On Metal `enqueue_copy` is a synchronous full-queue drain in both
        directions, **measured** by disassembly and recorded in
        `docs/GPU_PORTABILITY.md` section 6.1. Four calls that together move a
        few tens of kilobytes are four drains; one call that moves the same
        bytes is one. That is why the packed arm does not bother to skip
        regions that did not change: not copying a clean region saves bytes,
        and bytes were never what this was about.

        **What four drains are worth was overstated and the overstatement is
        withdrawn.** An earlier version of this docstring priced each drain at
        the ~458 microsecond per-synchronization constant and called the four
        "where the waits were". That constant is **derived**, from the
        depthwise A/B, and what that A/B removed were per-level *round trips*:
        host code blocking on a device answer it needs before it can decide
        what to enqueue next. None of these four is one. Section 6.1.1 records
        the withdrawal, on 2026-08-16, together with the data that forced it:
        collapsing thirteen copies per tree on the device-resident plane, four
        of them this method's, **measured** 0.016 seconds at 1,000,000 x 50
        against a registered prediction of 0.64
        (`bench/results/session3_2026-08-16/RESULTS.md`), which is a null under
        M0. Draining a queue that holds nothing costs nothing.

        `grow_tree_device_resident` counts sixteen copies per tree on the
        device-resident plane and attributes four of them to this method. The
        packed arm makes it one, so that count is thirteen. The shipping
        node-at-a-time and frontier loops call this once per split rather than
        once per tree, so there the same change is four copies per split down
        to one. **Read those as hazard and portability counts, not as a wait
        budget.** What one copy instead of four earns here is three fewer
        ordering points per call, one staging lifetime instead of four, and a
        device that can no longer be left holding three fresh tables and one
        stale one; that last is the argument this method is actually built
        around, and it is a correctness property rather than a speed one. The
        four-copy arm is kept as the arm an interleaved A/B holds this
        against, not because anyone predicts it will lose on a clock.

        Why one allocation and not four plus a scatter
        ----------------------------------------------
        The four tables keep their own `DeviceBuffer` handles, because
        `_launch_search` passes each to the kernels separately and
        `gpu_resident_round` reads all four fields by name. What changed is
        that the handles are `create_sub_buffer` windows onto one parent
        allocation instead of four independent ones, so a single copy of the
        parent writes all four regions.

        Three properties of that facility are **measured**, by running a
        program that asserts each one on this backend, rather than assumed
        from its signature, because all three are silent when wrong:

        1. a sub-buffer used as an `enqueue_copy` destination lands at its
           window and writes only its own length, leaving the words on
           either side alone (this is what makes the four-copy arm a valid
           reference arm rather than a second implementation);
        2. a window whose element type differs from its parent's (the
           Float32 parameter region inside an Int32 parent) aliases the same
           bytes, with offset and length counted in elements, which is
           unambiguous here only because both types are four bytes wide;
        3. the windows keep addressing the parent's storage after the struct
           holding all five handles is moved, which `GpuSplitSearcher` is:
           the trainer moves one out of its constructor and into a `List`.

        `GpuHistogramBuilder.readback_range` already depends on the source
        direction of the same facility.

        The bytes are the same bytes
        ----------------------------
        This is the claim that outranks the saving. Region by region, the
        packed copy writes exactly what the four copies wrote:

        - the node table, `[0, max_records * NODE_WORDS)`, from the same
          staged words `_stage_features` and the three `NODE_HIST_BASE`
          writers produce;
        - the feature table and the allow mask, each `max_records *
          n_features` words at `_feat_off` and `_allow_off`, from the same
          staged words `_stage_features` and `_stage_allowed` produce;
        - the float parameter block, `max_records * PF_WORDS` words at
          `_param_off`, from the same staged words `_stage_params`
          produces.

        The staging buffer is one pinned allocation laid out in that order
        and the device buffer is one allocation laid out in that order, so
        "copy the parent" is "copy all four regions", and no region can be
        left holding a previous call's contents while another is fresh.
        That last clause is the whole safety argument, and it is why there
        is no per-table dirty flag here.

        The skip that is not here, and why
        ----------------------------------
        This lane was asked to upload each table at its true scope of
        change: once per fit for the tables that are fit-constant, once per
        tree for the ones that are not. The mechanism that needs is a dirty
        flag per table, and the flag needs every mutator enumerated, because
        a table that changed without setting its flag is a stale table on
        the device, which does not crash, does not raise, and quietly grows
        a worse tree. Here is the enumeration, and here is what it decided.

        **Node table.** Host writers: `_stage_features` (the `NODE_SLOTS`
        word) and `_stage_hist_base`, whose three callers are `enqueue`,
        `search` and `enqueue_frontier`. Device writers: **one**, and this
        is the finding that settles the question.
        `GpuTreeTables.enqueue_stage_child_search` launches
        `_stage_child_search_kernel` over `GpuSplitSearcher.node_dev` and
        rewrites the two scratch records' histogram bases once per split on
        the device-resident plane; `gpu_resident_round` reaches it through
        the public field, which this module cannot intercept. So the pinned
        staging buffer is not a mirror of the node table's device contents
        at all, and no host-side flag can be evidence about them. The node
        table therefore uploads on every call, exactly as it did before.

        **Feature table.** Writers: `_stage_features`, from `set_features`.
        Called once per tree by the trainer with the tree's feature set,
        which is the fit's whole feature set unless `feature_fraction` is
        active, and per record by `enqueue_frontier` for a caller drawing a
        subset per node. Fit-constant in the common case, genuinely per tree
        under column subsampling.

        **Allow mask.** Writers: `_stage_allowed`, from `set_allowed` and
        from `set_features` (which resets it, because the mask is indexed by
        slot and a new slot order invalidates it). Fit-constant unless
        interaction constraints are in play.

        **Float parameter block.** Writers: `_stage_params`, called by
        `enqueue`, `search`, `enqueue_frontier`, and directly by
        `gpu_resident_round` for every record before each tree. This one is
        **not** fit-constant, contrary to what it looks like from its field
        list: `PF_G_INV` and `PF_H_INV` are the reciprocals of
        `GpuHistogramBuilder.g_scale` and `h_scale`, which
        `upload_gradients` recomputes every boosting round from the actual
        gradient magnitudes (`histogram_gpu._fixed_scale` ->
        `quantized_gradient.fixed_point_scale`). Only the regularization and
        categorical words in it are fit-constant. So this table changes once
        per round on every fit that is not constant-gradient, which is every
        real fit.

        What that enumeration buys, counted rather than estimated: a
        per-table skip could remove the feature and allow copies on a fit
        without column subsampling or interaction constraints, and neither
        parameter nor node copy on any fit. Two of four, in the good case,
        and one of four in the bad one. Packing removes three of four in
        every case, needs no flag to be right, and cannot go stale, so the
        two mechanisms are not additive: with one copy left there is nothing
        for a flag to remove, since the one copy is the node table's and the
        node table is the one no flag may skip.

        Rejected, and worth saying why. A content hash over each table would
        have removed the need to enumerate mutators; it was rejected because
        a hash collision produces exactly the silent staleness above, and
        because it does not help with the node table either, whose device
        contents no host-side hash can see. An explicit `mark_*_dirty` on
        each mutator is the auditable form and is what the enumeration above
        would have driven; it was rejected only because packing dominates it
        on this call graph, not because it was wrong. The one place a skip
        is provable is the monotone vector, which has no device writer at
        all, and `set_monotone` takes it. What that skip removes is a copy
        and therefore an ordering point, not a wait; under section 6.1.1 no
        copy count in this docstring converts to seconds.

        Ordering is unchanged. The copy is issued from `_launch` before
        either kernel is enqueued, into the same in-order queue, so the
        tables are on the device before the scan reads them exactly as
        before.
        """
        var base = self.stage_tables.unsafe_ptr()
        if self.hoist_tables:
            self.ctx.enqueue_copy(dst_buf=self.tables_dev, src_ptr=base)
            return
        # The arm this module shipped before the packing lane, kept
        # reachable at run time so a benchmark can interleave the two. Four
        # copies of four windows of the same staging buffer write the same
        # bytes into the same device words the one copy above writes; only
        # the number of queue drains differs.
        self.ctx.enqueue_copy(dst_buf=self.node_dev, src_ptr=base)
        self.ctx.enqueue_copy(
            dst_buf=self.feat_dev,
            src_ptr=base.unsafe_offset(self._feat_off()),
        )
        self.ctx.enqueue_copy(
            dst_buf=self.allow_dev,
            src_ptr=base.unsafe_offset(self._allow_off()),
        )
        self.ctx.enqueue_copy(
            dst_buf=self.fparam_dev,
            src_ptr=base.unsafe_offset(self._param_off()).unsafe_bitcast[
                Scalar[DType.float32]
            ](),
        )

    def set_features(
        mut self, features: List[Int], record: Int = -1
    ) raises:
        """Restrict later searches to `features` (global feature ids,
        ascending, one entry each), the same subsampled set
        `GpuHistogramBuilder.set_features` accumulated. Slot order is scan
        order, so it also fixes the cross-feature tie-breaking.

        `record` of -1, the default, applies the set to every record slot,
        which is what a tree-level or node-at-a-time caller means and what
        this method has always done. A frontier that draws a different
        subset per node (`feature_fraction_bynode`) passes the record it is
        staging instead, and only that node's slots move.

        Staged through pinned memory rather than a mapping, so it costs no
        host synchronization; the copy goes out with the node's `enqueue`.
        """
        if len(features) == 0:
            raise Error("active feature set must not be empty")
        if len(features) > self.n_features:
            raise Error("active feature set is larger than n_features")
        for i in range(len(features)):
            if features[i] < 0 or features[i] >= self.n_features:
                raise Error("feature index out of range")
        if record >= 0:
            self._check_record(record)
            self._stage_features(features, record)
            # The allow mask is indexed by slot, so a new slot order
            # invalidates it; reset this record's to "every listed feature
            # allowed" and let its `set_allowed` narrow it again.
            self._stage_allowed(List[Bool](), record)
            return
        self.active = features.copy()
        for r in range(self.max_records):
            self._stage_features(features, r)
            self._stage_allowed(List[Bool](), r)

    def set_allowed(
        mut self, allowed: List[Bool] = [], record: Int = -1
    ) raises:
        """This node's interaction-constraint allow mask, indexed by global
        feature id. Empty (the default) allows every feature; a mask shorter
        than `n_features` disallows the features past its end, exactly as
        `find_best_split` reads it.

        Indexed by slot on the device, so it is re-staged after every
        `set_features`. `record` of -1 applies the mask to every record
        slot, as the node-at-a-time loop wants; a frontier passes the record
        it is staging.

        The copy is issued by `enqueue` or `enqueue_frontier` rather than
        here, so a batch stages every node's mask before any of them
        crosses.
        """
        if record >= 0:
            self._check_record(record)
            self._stage_allowed(allowed, record)
            return
        for r in range(self.max_records):
            self._stage_allowed(allowed, r)

    def set_monotone(mut self, signs: List[Int] = []) raises:
        """This tree's active monotone constraint vector. Empty (the default)
        keeps the unconstrained scoring path, exactly as on the host.

        Called once per tree by the trainer (`train_gpu.mojo` line 1034) and
        handed the fit's constraint vector, which does not vary by tree.
        `map_to_host` moves the buffer in both directions on every use, so
        an unconditional map is one or two drains per tree spent writing the
        same words that are already there. With `hoist_tables` on, this maps
        only when the vector it is handed differs from the one this searcher
        last wrote, which makes it once per fit for every configuration
        mojotrees supports. Per tree to per fit is a reduction in ordering
        points and in places the mirror could go stale; it is **not** a
        predicted time, because nothing is queued behind those maps and
        section 6.1.1 no longer permits a copy count to be priced.

        Why a mirror here is trustworthy, where it is not for the four
        tables `_copy_tables` carries. The skip is sound exactly when the
        host's `mono_host` is what `mono_dev` holds, and that needs every
        writer of `mono_dev` enumerated. There are two, and both are in this
        file and both update the mirror:

        1. the constructor, which writes `MONOTONE_FREE` to every entry and
           fills `mono_host` with the same;
        2. this method.

        Nothing else can write it. `mono_dev` leaves this struct in exactly
        two places, `_launch_search`'s `mono` argument and
        `gpu_resident_round`'s direct read of the field, and both hand it to
        the scan kernels, which take it as `mono` and only ever load from
        it: `_scan_slot_kernel`, `_scan_slot_wide_kernel` and
        `_scan_slot_wide_primitive_kernel` contain no store through that
        pointer, and the two reduction kernels are not given it at all.
        Contrast `node_dev`, which `GpuTreeTables.enqueue_stage_child_search`
        writes on the device inside every tree of the resident plane, and
        whose host staging is therefore not a mirror of anything.

        `constrained` is set on every call whether or not the map is
        skipped, because it is a kernel argument rather than device state
        and the skip says nothing about it.
        """
        if len(signs) > 0 and len(signs) != self.n_features:
            raise Error("monotone length must equal n_features")
        for f in range(self.n_features):
            if (
                len(signs) > 0
                and self.cat_n[f] > 0
                and signs[f] != MONOTONE_FREE
            ):
                raise Error(
                    "monotonic constraints are not supported on categorical"
                    " features"
                )
        self.constrained = len(signs) > 0
        # Compared against the mirror rather than against a hash of it: the
        # words are `n_features` Int32s already in cache, the comparison is
        # the loop that would have written them, and an exact comparison
        # cannot collide. A hash could, and a collision here is the failure
        # this whole lane is written to avoid: a stale table that does not
        # crash, does not raise, and quietly produces a worse model.
        var moved = not self.mono_uploaded
        for f in range(self.n_features):
            var s = Int32(signs[f] if len(signs) > 0 else MONOTONE_FREE)
            if s != self.mono_host[f]:
                moved = True
                self.mono_host[f] = s
        if not moved and self.hoist_tables:
            return
        with self.mono_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(self.n_features):
                dst.unsafe_store(f, self.mono_host[f])
        self.mono_uploaded = True

    def _ensure_hist(mut self) raises:
        """Size the searcher's own histogram buffer, once, on first use.

        Both callers are the standalone path (`upload_histogram` and
        `search`); a searcher the trainer drives never reaches either and
        therefore never pays for the buffer. Sizing is a construction-time
        fact (`n_features` and `n_bins` never change), so this can only run
        once, and it runs before the copy or the launch that needs it."""
        if self.hist_owned:
            return
        self.hist_dev = self.ctx.enqueue_create_buffer[DType.int32](
            3 * self.n_features * self.n_bins
        )
        self.hist_owned = True

    def upload_histogram(mut self, words: List[Int32]) raises:
        """Stage a fixed-point `[grad | hess | count]` histogram into this
        searcher's own buffer. The trainer integration uses the zero-copy
        `enqueue` overload against the builder's buffer instead; this exists
        so the search is exercisable, and benchmarkable, on its own."""
        if len(words) != 3 * self.n_features * self.n_bins:
            raise Error(
                "histogram must hold 3 * n_features * n_bins Int32 words"
            )
        self._ensure_hist()
        self.ctx.enqueue_copy(
            dst_buf=self.hist_dev, src_ptr=words.unsafe_ptr()
        )
        self.ctx.synchronize()

    def upload_level_histogram(
        mut self, words: List[Int32], n_slots: Int
    ) raises:
        """Stage `n_slots` consecutive fixed-point histograms into this
        searcher's own buffer, slot 0 first.

        `upload_histogram` for an oblivious level, whose search reads one
        histogram per leaf out of one allocation with a slot stride of
        `3 * n_features * n_bins` -- the same stride `GpuLeafBatcher.out_dev`
        uses, so a caller holding a real frontier's histograms hands them
        straight to `enqueue_oblivious_level` instead. This reallocates rather
        than growing, because it exists so the level search is exercisable on
        its own and a standalone caller changes shape between calls."""
        if n_slots < 1:
            raise Error("a level holds at least one histogram slot")
        var cells = 3 * self.n_features * self.n_bins
        if len(words) != n_slots * cells:
            raise Error(
                "a level histogram must hold n_slots * 3 * n_features *"
                " n_bins Int32 words"
            )
        self.hist_dev = self.ctx.enqueue_create_buffer[DType.int32](
            n_slots * cells
        )
        self.hist_owned = True
        self.ctx.enqueue_copy(
            dst_buf=self.hist_dev, src_ptr=words.unsafe_ptr()
        )
        self.ctx.synchronize()

    def search_oblivious_level(
        mut self,
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
        leaf_slots: List[Int],
        level_record: Int = 0,
        leaf_base: Int = 1,
        score_function: Int = SCORE_L2,
    ) raises -> GpuSplitRecord:
        """Search the level staged by `upload_level_histogram` and return the
        level's record: one (feature, bin, missing direction) chosen by the
        summed gain over `leaf_slots`, in that order.

        The standalone counterpart of `search`, and it spells the field
        borrows out at the free-function boundary for the same aliasing reason
        that method does.

        `score_function` is an explicit argument here and not a searcher
        field, which is the one place this differs from `gain_form`. It is
        deliberate and it is temporary: `lane/cosine-device` owns the searcher
        field and the per-node kernels that read it, and a field on this
        searcher that only the level scan honored would answer Cosine for a
        level and L2 for a node without saying so. Named at the call, nothing
        is silent; when the two lanes meet, the default becomes the field."""
        self._check_record(level_record)
        if len(leaf_slots) < 1:
            raise Error("an oblivious level holds at least one leaf")
        if leaf_base < 0 or leaf_base + len(leaf_slots) > self.max_records:
            raise Error("the level's leaf records are outside the searcher")
        if (
            level_record >= leaf_base
            and level_record < leaf_base + len(leaf_slots)
        ):
            raise Error(
                "the level record must sit outside the leaf records it reads"
            )
        self._ensure_hist()
        var has_cat = False
        for f in range(self.n_features):
            if self.cat_n[f] >= 2:
                has_cat = True
        self._stage_params(
            params, g_scale, h_scale, OutputBounds.unbounded(), level_record
        )
        var slot_cells = 3 * self.n_features * self.n_bins
        for i in range(len(leaf_slots)):
            if leaf_slots[i] < 0:
                raise Error("a histogram slot must be nonnegative")
            self._stage_hist_base(
                leaf_base + i, Int32(leaf_slots[i] * slot_cells)
            )
        self._copy_tables()
        # The level record's own plane, and only it: the level is one
        # candidate set electing one split, so the leaf records carry no
        # noise and are never asked for one.
        self._check_noise_staged(level_record, 1)
        self._copy_noise(level_record, 1)
        _launch_oblivious_search(
            self.ctx,
            self.hist_dev,
            self.node_dev,
            self.feat_dev,
            self.allow_dev,
            self.missing_dev,
            self.catn_dev,
            self.mono_dev,
            self.fparam_dev,
            self.noise_dev,
            self.slot_i_dev,
            self.slot_f_dev,
            self.rec_i_dev,
            self.rec_f_dev,
            self.n_bins,
            self.n_features * self.n_bins,
            self.n_features,
            self.active_len[level_record],
            level_record,
            leaf_base,
            len(leaf_slots),
            params.min_data_in_leaf,
            self.constrained,
            has_cat,
            self.use_primitives,
            self.gain_form_code,
            score_function,
            self.noise_stdev > 0.0,
        )
        return self.download(level_record)

    def _stage_params(
        mut self,
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
        bounds: OutputBounds,
        record: Int,
    ) raises:
        """Write one record's float parameter block, including its monotone
        output bounds, which are the only per-node value in it. Copies
        nothing; `_copy_tables` does."""
        if g_scale <= 0.0 or h_scale <= 0.0:
            raise Error("fixed-point scales must be positive")
        var dst = self.stage_tables.unsafe_ptr().unsafe_offset(
            self._param_off()
        ).unsafe_bitcast[Scalar[DType.float32]]()
        var base = record * PF_WORDS
        dst.unsafe_store(base + PF_G_INV, Float32(1.0 / g_scale))
        dst.unsafe_store(base + PF_H_INV, Float32(1.0 / h_scale))
        dst.unsafe_store(base + PF_LAMBDA_L2, Float32(params.lambda_l2))
        dst.unsafe_store(base + PF_LAMBDA_L1, Float32(params.lambda_l1))
        dst.unsafe_store(
            base + PF_MIN_CHILD_HESS, Float32(params.min_child_hess)
        )
        dst.unsafe_store(base + PF_BOUND_LO, _f32_bound(bounds.lo))
        dst.unsafe_store(base + PF_BOUND_HI, _f32_bound(bounds.hi))
        dst.unsafe_store(
            base + PF_CAT_SMOOTH, Float32(params.cat.cat_smooth)
        )
        dst.unsafe_store(base + PF_CAT_L2, Float32(params.cat.cat_l2))

    def _launch(
        mut self,
        mut hist: DeviceBuffer[DType.int32],
        params: GpuSplitParams,
        record_base: Int,
        n_records: Int,
        widest_slots: Int,
    ) raises:
        """Copy the staged tables and launch the two kernels over
        `n_records` consecutive record slots. The one place either entry
        point reaches the device."""
        self._check_noise_staged(record_base, n_records)
        self._copy_tables()
        self._copy_noise(record_base, n_records)
        _launch_search(
            self.ctx,
            hist,
            self.node_dev,
            self.feat_dev,
            self.allow_dev,
            self.missing_dev,
            self.catn_dev,
            self.mono_dev,
            self.fparam_dev,
            self.noise_dev,
            self.slot_i_dev,
            self.slot_f_dev,
            self.rec_i_dev,
            self.rec_f_dev,
            self.n_bins,
            self.n_features * self.n_bins,
            self.n_features,
            widest_slots,
            record_base,
            n_records,
            params.min_data_in_leaf,
            self.constrained,
            params.cat.max_cat_to_onehot,
            params.cat.max_cat_threshold,
            params.cat.min_data_per_group,
            self.wide_scan,
            self.use_primitives,
            self.gain_form_code,
            self.noise_stdev > 0.0,
            self.score_function_code,
        )

    def _copy_noise(mut self, record_base: Int, n_records: Int) raises:
        """Upload the staged noise rows for `[record_base, record_base +
        n_records)`, and nothing else.

        One copy per launch, over exactly the records the launch searches, so
        a node-at-a-time loop moves one node's plane and a frontier moves the
        frontier's. It is a real transfer -- `n_features * n_bins` Float32 a
        node, a third of the histogram this module exists to stop moving --
        and it is the price of the two backends adding the same number; see
        the `random_strength` section at the top of this module. Nothing is
        copied, and this function returns immediately, when the noise is off.
        """
        if not (self.noise_stdev > 0.0):
            return
        var stride = self.n_features * self.n_bins
        var window = self.noise_dev.create_sub_buffer[DType.float32](
            record_base * stride, n_records * stride
        )
        self.ctx.enqueue_copy(
            dst_buf=window,
            src_ptr=self.noise_stage.unsafe_ptr().unsafe_offset(
                record_base * stride
            ),
        )

    def enqueue(
        mut self,
        mut hist: DeviceBuffer[DType.int32],
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
        bounds: OutputBounds = OutputBounds.unbounded(),
        record: Int = 0,
        hist_slot: Int = 0,
        node: Int = -1,
    ) raises:
        """Enqueue the scan and reduction over `hist`, writing record slot
        `record`.

        `node` is this leaf's node id, read by `random_strength` alone, which
        keys its draw by it; -1, the default, is correct whenever the noise is
        off and refused when it is on. See `set_random_score`.

        It does transfer, and on Metal it therefore drains. `_launch` copies
        the staged per-node tables across before either kernel is enqueued,
        and `enqueue_copy` on Metal is a synchronous full-queue drain
        (**measured** by disassembly, `docs/GPU_PORTABILITY.md` section 6.1).
        An earlier version of this line said "does not transfer or
        synchronize" and was wrong on both halves. Nothing about the launch
        changed; what changed is what a copy count may claim.

        That drain is **one** with `hoist_tables` on, where it was four: the
        four tables share an allocation and cross in one copy. See
        `_copy_tables`.

        It is a drain, not a round trip. No host decision here reads a device
        answer, so under section 6.1.1 this ordering point predicts no time,
        and four to one is a hazard and portability improvement rather than a
        measured saving.

        `hist` is a device buffer of `3 * n_features * n_bins` Int32 words in
        `GpuHistogramBuilder`'s `[grad | hess | count]` layout, and `g_scale`
        / `h_scale` are the fixed-point scales that histogram was accumulated
        with (`builder.g_scale`, `builder.h_scale`). `hist_slot` names which
        histogram inside a multi-slot buffer to read, for a caller holding a
        whole level's histograms at once (`GpuLeafBatcher.out_dev`, whose
        slot stride is exactly `3 * n_features * n_bins`); it stays 0 for
        the builder's single-node buffer.

        Ordering contract: this stages one record and copies the tables, so
        a node's `enqueue` is followed by its `download` (or an explicit
        `synchronize`) before the next node stages. That is the
        node-at-a-time loop, unchanged. A caller that wants a whole
        frontier without a wait per node uses `enqueue_frontier`, which
        stages every node before any table crosses."""
        self._check_record(record)
        if hist_slot < 0:
            raise Error("histogram slot must be nonnegative")
        self._stage_params(params, g_scale, h_scale, bounds, record)
        self._stage_hist_base(
            record, Int32(hist_slot * 3 * self.n_features * self.n_bins)
        )
        if self.noise_stdev > 0.0:
            self.stage_random_score(record, node)
        self._launch(hist, params, record, 1, self.active_len[record])

    def enqueue_oblivious_level(
        mut self,
        mut hist: DeviceBuffer[DType.int32],
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
        leaf_slots: List[Int],
        level_record: Int,
        leaf_base: Int,
        score_function: Int = SCORE_L2,
    ) raises:
        """Enqueue one oblivious level's search: the cross-leaf scan over the
        level's leaves and the ordinary cross-feature reduction into
        `level_record`.

        `score_function` is an explicit argument rather than a searcher field
        for the reason `search_oblivious_level` gives.

        `leaf_slots[i]` is the histogram pool slot leaf `i` of the level owns,
        in the leaf-index order the level's frontier holds them, and this
        stages it into record `leaf_base + i`'s `NODE_HIST_BASE`. Those records
        are read for that one word and for nothing else: the feature list, the
        allow mask, the monotone vector and the float parameter block all come
        from `level_record`, because under the resident plane's refusals they
        are tree-level and are staged once.

        **The order of `leaf_slots` is the summation order and therefore part
        of the answer.** `_scan_slot_oblivious_kernel` sums each candidate's
        gain over the leaves in ascending record order, and Float32 addition is
        not associative, so handing the same level's slots in a different order
        is a different tree. The order this mode fixes is ascending leaf index,
        where the leaf index of a row is the bit pattern of its split outcomes
        with the **first** level's outcome as the least significant bit --
        CatBoost's own convention, and the one
        `gpu_resident_round.OBLIVIOUS_LEAF_INDEX_RULE` states for both backends
        to hold to.

        `leaf_base` must not overlap `level_record`, since the scan reads the
        leaf records while writing the level record's slots.

        This is the standalone, testable entry point, and it copies the staged
        tables, which on Metal drains the queue. The resident plane must not
        use it for the same reason `_launch_child_search` bypasses
        `enqueue_frontier`: the per-record histogram base is written on the
        device there, and copying the host's stale mirror over it would point
        the scan at the previous level's slots. The plane calls
        `_launch_oblivious_search` directly instead."""
        self._check_record(level_record)
        if len(leaf_slots) < 1:
            raise Error("an oblivious level holds at least one leaf")
        if leaf_base < 0 or leaf_base + len(leaf_slots) > self.max_records:
            raise Error("the level's leaf records are outside the searcher")
        if (
            level_record >= leaf_base
            and level_record < leaf_base + len(leaf_slots)
        ):
            raise Error(
                "the level record must sit outside the leaf records it reads"
            )
        var has_cat = False
        for f in range(self.n_features):
            if self.cat_n[f] >= 2:
                has_cat = True
        self._stage_params(
            params, g_scale, h_scale, OutputBounds.unbounded(), level_record
        )
        var slot_cells = 3 * self.n_features * self.n_bins
        for i in range(len(leaf_slots)):
            if leaf_slots[i] < 0:
                raise Error("a histogram slot must be nonnegative")
            self._stage_hist_base(
                leaf_base + i, Int32(leaf_slots[i] * slot_cells)
            )
        self._copy_tables()
        self._check_noise_staged(level_record, 1)
        self._copy_noise(level_record, 1)
        _launch_oblivious_search(
            self.ctx,
            hist,
            self.node_dev,
            self.feat_dev,
            self.allow_dev,
            self.missing_dev,
            self.catn_dev,
            self.mono_dev,
            self.fparam_dev,
            self.noise_dev,
            self.slot_i_dev,
            self.slot_f_dev,
            self.rec_i_dev,
            self.rec_f_dev,
            self.n_bins,
            self.n_features * self.n_bins,
            self.n_features,
            self.active_len[level_record],
            level_record,
            leaf_base,
            len(leaf_slots),
            params.min_data_in_leaf,
            self.constrained,
            has_cat,
            self.use_primitives,
            self.gain_form_code,
            score_function,
            self.noise_stdev > 0.0,
        )

    def enqueue_pick_best(
        mut self, n_records: Int, record: Int = 0
    ) raises:
        """Reduce record slots `[0, n_records)` into slot `record`: the
        best-gain leaf of a frontier, ties going to the lower slot. The step
        that lets a whole tree level be selected without a host round trip
        per node.

        `record` must be outside `[0, n_records)` unless the frontier is
        being consumed, since the destination slot is overwritten in place."""
        if n_records < 1 or n_records > self.max_records:
            raise Error("n_records out of range")
        if record < 0 or record >= self.max_records:
            raise Error("record index out of range")
        self.ctx.enqueue_function[_pick_best_record_kernel](
            self.rec_i_dev.unsafe_ptr(),
            self.rec_f_dev.unsafe_ptr(),
            Int32(n_records),
            Int32(record),
            grid_dim=1,
            block_dim=1,
        )

    def download_words(
        mut self, mut words_i: List[Int32], mut words_f: List[Float32]
    ) raises:
        """Copy every record slot into two host lists. `max_records * 136`
        bytes, whatever the histogram shape; a 100-feature, 256-bin node's
        histogram is 300 KB.

        **The round trip this plane's time is made of.** `_device_search_
        resident` reads one record per split and this is that read, about a
        hundred times in a fit. `gpu_runtime`'s transport table prices every
        way of doing it and `self.readback` picks one; the default is
        `READBACK_PLAIN_ONE`, measured at 124.85 us a trip against the
        `READBACK_PINNED_PAIR_SYNC` shape this shipped until 2026-08-16 at
        202.14. Same words, same values, two command buffers instead of four.

        WHERE THE WAIT IS, AND WHY THE BRANCH IS ON THE DESTINATION
        -----------------------------------------------------------
        `docs/GPU_PORTABILITY.md` section 6.5.1: on Metal `enqueue_copy` has
        two implementations and **the destination picks one**. Into pinned
        memory from `enqueue_create_host_buffer` it enqueues a blit and
        returns, so the words are not there until something waits. Into an
        arbitrary host pointer it commits, waits, and memcpys, so the words
        are there when the call returns and a following `synchronize()` would
        be waiting on an empty queue.

        So the pinned arms below end in `self.ctx.synchronize()` and the plain
        arms do not, and that trailing call is **load-bearing** wherever it
        appears: 6.1's bullet licensing its removal is exactly what 6.5.1
        withdrew, after measuring 64 of 64 stale words behind a slow kernel
        and 0 of 64 behind a fast one. Latency-dependent, which is why it
        cannot be validated by running something small.

        The branch is therefore on `row.pinned_destination` and not on the
        arm code. Written as `if self.readback == READBACK_PLAIN_ONE`, a later
        edit that pointed a plain arm at `host_i` to save an allocation would
        compile, pass, and corrupt every split after the first. Written this
        way it cannot: the only branch that omits the wait is the one whose
        destination the table says is unpinned, and `plain_words` is a `List`
        whose storage no `enqueue_create_host_buffer` ever produced.

        The unpack is `[integer plane | float plane]`, the layout
        `SPLIT_RECORD_WORDS` fixes, and the float half is read through a
        four-byte-aligned `Float32` view of the same words.

        **Every arm leaves the queue drained**, which is the postcondition
        callers rely on and the reason dropping the `synchronize()` on the
        plain arms is a saving rather than a semantic change. Section 6.1's
        disassembly of the synchronous path is not retracted by 6.5.1: a copy
        into an arbitrary host pointer commits the queue and waits for it, so
        it drains everything enqueued before it and not merely itself. What
        6.5.1 corrected is which destinations take that path.
        """
        var n_i = self.max_records * SPLIT_IWORDS
        var n_f = self.max_records * SPLIT_FWORDS
        words_i.resize(n_i, Int32(0))
        words_f.resize(n_f, Float32(0.0))
        var row = readback_transport(self.readback)
        if row.pinned_destination:
            if row.packed_source:
                self.ctx.enqueue_copy(
                    dst_ptr=self.host_i.unsafe_ptr(), src_buf=self.records_dev
                )
            else:
                self.ctx.enqueue_copy(
                    dst_ptr=self.host_i.unsafe_ptr(), src_buf=self.rec_i_dev
                )
                self.ctx.enqueue_copy(
                    dst_ptr=self.host_f.unsafe_ptr(), src_buf=self.rec_f_dev
                )
            # Load-bearing. The copies above are asynchronous blits into
            # pinned memory; without this the reads below see the previous
            # record. Section 6.5.1, measured.
            self.ctx.synchronize()
            var src_i = self.host_i.unsafe_ptr()
            for i in range(n_i):
                words_i[i] = src_i.unsafe_load(i)
            if row.packed_source:
                # The packed pinned arm lands both planes in `host_i`, so the
                # float words are the tail of that buffer read as Float32.
                var src_p = self.host_i.unsafe_ptr().unsafe_offset(
                    n_i
                ).unsafe_bitcast[Scalar[DType.float32]]()
                for i in range(n_f):
                    words_f[i] = src_p.unsafe_load(i)
            else:
                var src_f = self.host_f.unsafe_ptr()
                for i in range(n_f):
                    words_f[i] = src_f.unsafe_load(i)
            return
        # Unpinned. Every copy below drains inside itself, so no
        # `synchronize()` follows and none may be added back as a precaution:
        # it would be a wait on an empty queue, which is what the pinned arm's
        # measured 202.14 us is mostly made of.
        var dst = self.plain_words.unsafe_ptr()
        if row.packed_source:
            self.ctx.enqueue_copy(dst_ptr=dst, src_buf=self.records_dev)
        else:
            self.ctx.enqueue_copy(dst_ptr=dst, src_buf=self.rec_i_dev)
            self.ctx.enqueue_copy(
                dst_ptr=dst.unsafe_offset(n_i).unsafe_bitcast[
                    Scalar[DType.float32]
                ](),
                src_buf=self.rec_f_dev,
            )
        for i in range(n_i):
            words_i[i] = dst.unsafe_load(i)
        var src_f = dst.unsafe_offset(n_i).unsafe_bitcast[
            Scalar[DType.float32]
        ]()
        for i in range(n_f):
            words_f[i] = src_f.unsafe_load(i)

    def download(mut self, record: Int = 0) raises -> GpuSplitRecord:
        """Copy the record buffer to the host and decode slot `record`."""
        if record < 0 or record >= self.max_records:
            raise Error("record index out of range")
        var words_i = List[Int32]()
        var words_f = List[Float32]()
        self.download_words(words_i, words_f)
        return decode_record(words_i, words_f, record)

    def search(
        mut self,
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
        bounds: OutputBounds = OutputBounds.unbounded(),
        record: Int = 0,
        node: Int = -1,
    ) raises -> GpuSplitRecord:
        """Search the histogram staged by `upload_histogram` and return its
        record.

        `node` is this leaf's node id and is read by `random_strength` alone,
        which keys its draw by it; -1, the default, is correct whenever the
        noise is off and refused when it is on. See `set_random_score`."""
        self._check_record(record)
        if self.noise_stdev > 0.0:
            self.stage_random_score(record, node)
        # Ordinarily a no-op, since `upload_histogram` has run and sized the
        # buffer; it is here so that this method never launches against the
        # placeholder, whatever order a caller uses.
        self._ensure_hist()
        self._stage_params(params, g_scale, h_scale, bounds, record)
        self._stage_hist_base(record, Int32(0))
        # Do not call `_launch(self.hist_dev, ...)`: that borrows all of
        # `self` mutably for the method receiver while also borrowing one of
        # its fields mutably as an argument, which Mojo correctly rejects as
        # aliasing.  The owned-histogram path is the one place the histogram
        # belongs to the searcher, so spell the disjoint field borrows out at
        # the free-function boundary after staging the tables.
        #
        # `wide` is passed here, which it was not until the lane merge.
        # Omitting it made `search` run the serial scan even on a searcher
        # whose `wide_scan` was set, while `enqueue` and `enqueue_frontier`
        # (both through `_launch`) honored it. The trainer only ever uses
        # those two, so no fit was ever affected; what the omission did cost
        # was a test, because `test_gpu_split_search.test_wide_scan_matches
        # _the_serial_wide_scan` reaches the wide kernel through `search`
        # and had therefore been comparing the serial scan against itself.
        self._copy_tables()
        self._copy_noise(record, 1)
        _launch_search(
            self.ctx,
            self.hist_dev,
            self.node_dev,
            self.feat_dev,
            self.allow_dev,
            self.missing_dev,
            self.catn_dev,
            self.mono_dev,
            self.fparam_dev,
            self.noise_dev,
            self.slot_i_dev,
            self.slot_f_dev,
            self.rec_i_dev,
            self.rec_f_dev,
            self.n_bins,
            self.n_features * self.n_bins,
            self.n_features,
            self.active_len[record],
            record,
            1,
            params.min_data_in_leaf,
            self.constrained,
            params.cat.max_cat_to_onehot,
            params.cat.max_cat_threshold,
            params.cat.min_data_per_group,
            wide=self.wide_scan,
            primitives=self.use_primitives,
            # `gain_form` was missing here until 2026-08-16, so this entry
            # point ran `DEFAULT_GAIN_FORM` whatever `set_gain_form` had been
            # told -- the sibling `enqueue_frontier` passed it and this one did
            # not. Spotted by the random-strength lane, which correctly left it
            # alone because fixing it moves records and needed the gain-form
            # suite run. Keyword-passed like its neighbours so a future
            # argument inserted above cannot silently rebind it.
            gain_form=self.gain_form_code,
            score_function=self.score_function_code,
            noisy=self.noise_stdev > 0.0,
        )
        return self.download(record)

    def enqueue_frontier(
        mut self,
        mut hist: DeviceBuffer[DType.int32],
        nodes: List[SplitNodeRequest],
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
    ) raises:
        """Enqueue a whole frontier's searches in one pair of launches,
        writing record slots `[0, len(nodes))`.

        This is the entry point that removes the *download* per node. Every
        node's feature set, allow mask, monotone bounds, and histogram slot
        are written into their own record's staging slot first, then the
        tables cross, then one scan covers the whole batch and one
        reduction writes every record. The host's next download is
        `download_frontier`, which is one for the level rather than one per
        leaf.

        **The download per node is the part that is worth time**, because a
        download is a round trip: the host reads a device answer and then
        decides what to enqueue next. Removing about thirty of those per tree
        **measured** 0.75 seconds at 1,000,000 x 50, resolved by a wide margin
        (`bench/results/session3_2026-08-16/RESULTS.md`). That is the count
        this entry point should be defended on.

        It does transfer, and on Metal it therefore drains. The tables cross
        in `_copy_tables` before the launches, and on Metal an `enqueue_copy`
        is a synchronous full-queue drain in both directions, **measured** by
        disassembly and recorded in `docs/GPU_PORTABILITY.md` section 6.1. An
        earlier version of this line said "does not transfer and does not
        synchronize", which was wrong on both halves; a later one called that
        drain a host wait, which section 6.1.1 withdrew on 2026-08-16. So a
        level is not free of ordering points, and it is free of round trips
        but one.

        That crossing is **one** copy with `hoist_tables` on and four with it
        off. It was four unconditionally before the table-packing lane, and
        `grow_tree_device_resident`'s per-tree copy count, which attributed
        four of its sixteen to this call, is therefore thirteen on the packed
        arm. That is a hazard and portability count. Four to one bought
        0.016 seconds across the whole thirteen-copy collapse, a null under
        M0, and no part of it may be quoted as a saving.

        `hist` holds every node's histogram: the builder's single-node
        buffer when the batch has one node, and a multi-slot buffer
        (`GpuLeafBatcher.out_dev`, slot stride `3 * n_features * n_bins`)
        when it has more. A node's `hist_slot` names its slot, and two
        nodes may name the same slot only if they really do share a
        histogram.

        What stays the caller's: which leaves are in the frontier at all,
        the depth and minimum-row rules (properties of the tree, not of a
        histogram), and the split it commits. What the batch does not
        change: the scan order inside a node, the cross-feature
        tie-breaking, and therefore every record it returns, which is
        identical to what the same nodes searched one at a time would
        return.
        """
        if len(nodes) < 1:
            raise Error("a frontier batch needs at least one node")
        if len(nodes) > self.max_records:
            raise Error(
                "the frontier is larger than the record capacity this"
                " searcher was constructed with"
            )
        var widest = 1
        var slot_cells = 3 * self.n_features * self.n_bins
        for i in range(len(nodes)):
            if nodes[i].hist_slot < 0:
                raise Error("histogram slot must be nonnegative")
            if len(nodes[i].features) > 0:
                self.set_features(nodes[i].features, record=i)
            self.set_allowed(nodes[i].allowed, record=i)
            self._stage_params(
                params, g_scale, h_scale, nodes[i].bounds, i
            )
            self._stage_hist_base(i, Int32(nodes[i].hist_slot * slot_cells))
            # After the feature set, because the plane is keyed by the global
            # ids of exactly the features this record will scan.
            if self.noise_stdev > 0.0:
                self.stage_random_score(i, nodes[i].node)
            if self.active_len[i] > widest:
                widest = self.active_len[i]
        self._launch(hist, params, 0, len(nodes), widest)

    def download_frontier(
        mut self, n_records: Int
    ) raises -> List[GpuSplitRecord]:
        """The batch's one wait: copy every record back and decode slots
        `[0, n_records)`.

        `max_records * 136` bytes and one wait, against one histogram download
        and one wait per node, which is what the node-at-a-time loop pays and
        what the whole record layout exists to avoid.

        One wait, not necessarily one `synchronize()`: under the default
        `READBACK_PLAIN_ONE` the wait is the copy's own drain. See
        `download_words`.
        """
        if n_records < 1 or n_records > self.max_records:
            raise Error("n_records out of range")
        var words_i = List[Int32]()
        var words_f = List[Float32]()
        self.download_words(words_i, words_f)
        var out = List[GpuSplitRecord](capacity=n_records)
        for r in range(n_records):
            out.append(decode_record(words_i, words_f, r))
        return out^

    def search_frontier(
        mut self,
        mut hist: DeviceBuffer[DType.int32],
        nodes: List[SplitNodeRequest],
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
    ) raises -> List[GpuSplitRecord]:
        """`enqueue_frontier` followed by `download_frontier`: a bounded
        frontier's decisions, in one launch pair and one wait."""
        self.enqueue_frontier(hist, nodes, params, g_scale, h_scale)
        return self.download_frontier(len(nodes))


def decode_record(
    words_i: List[Int32], words_f: List[Float32], record: Int
) raises -> GpuSplitRecord:
    """Decode one record slot out of a downloaded record buffer pair. Public
    because the device-side queue will download many records at once and
    decode the slots it needs."""
    var io = record * SPLIT_IWORDS
    var fo = record * SPLIT_FWORDS
    if len(words_i) < io + SPLIT_IWORDS or len(words_f) < fo + SPLIT_FWORDS:
        raise Error("record buffers are too short for that record index")
    var flags = words_i[io + IREC_FLAGS]
    var out = GpuSplitRecord()
    out.feature = Int(words_i[io + IREC_FEATURE])
    out.bin = Int(words_i[io + IREC_BIN])
    out.ordinal = Int(words_i[io + IREC_ORDINAL])
    out.found = (flags & Int32(FLAG_FOUND)) != Int32(0)
    out.default_left = (flags & Int32(FLAG_DEFAULT_LEFT)) != Int32(0)
    out.is_categorical = (flags & Int32(FLAG_CATEGORICAL)) != Int32(0)
    out.gain = Float64(words_f[fo + FREC_GAIN])
    out.left = ChildStats(
        Float64(words_f[fo + FREC_LEFT_GRAD]),
        Float64(words_f[fo + FREC_LEFT_HESS]),
        Int(words_i[io + IREC_LEFT_COUNT]),
    )
    out.right = ChildStats(
        Float64(words_f[fo + FREC_RIGHT_GRAD]),
        Float64(words_f[fo + FREC_RIGHT_HESS]),
        Int(words_i[io + IREC_RIGHT_COUNT]),
    )
    out.total = ChildStats(
        Float64(words_f[fo + FREC_TOTAL_GRAD]),
        Float64(words_f[fo + FREC_TOTAL_HESS]),
        Int(words_i[io + IREC_TOTAL_COUNT]),
    )
    out.left_value = Float64(words_f[fo + FREC_LEFT_VALUE])
    out.right_value = Float64(words_f[fo + FREC_RIGHT_VALUE])
    out.parent_value = Float64(words_f[fo + FREC_PARENT_VALUE])
    out.runner_gain = Float64(words_f[fo + FREC_RUNNER_GAIN])
    if out.is_categorical:
        out.cat_bitset = _bitset_from_words(words_i, io)
    return out^


# --- Host reference -------------------------------------------------------


def reference_search(
    hist: List[Int32],
    n_features: Int,
    n_bins: Int,
    g_scale: Float64,
    h_scale: Float64,
    params: GpuSplitParams,
    features: List[Int] = [],
    allowed: List[Bool] = [],
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
    cats: CategoricalSpec = CategoricalSpec.none(),
    bounds: OutputBounds = OutputBounds.unbounded(),
    gain_form: Int = DEFAULT_GAIN_FORM,
    noise: List[Float32] = [],
    score_function: Int = SCORE_L2,
) raises -> GpuSplitRecord:
    """The kernels' arithmetic, run on the host over the same fixed-point
    histogram.

    This is not an independent implementation of split finding: it calls the
    same Float32 helpers the kernels do and walks the candidates in the same
    order, so what the tests compare is that the two loop structures agree,
    and what they assert against hand-computed gains is the shared
    arithmetic. It is also what makes the analytical tests runnable on a
    machine with no accelerator, and it is the reference the device path is
    validated against before it replaces the host scan.

    `gain_form` selects the same arm the kernels take and defaults to the
    same one they default to, so the replica keeps replicating. Passing a
    different arm here than the searcher is running is how a test compares
    the two forms on one histogram without a device; passing a different one
    by accident is how a host/device comparison fails for a reason that is
    not about either of them.

    `noise` is one node's `random_strength` plane, `len(features) * n_bins`
    Float32 in the order `random_score_plane` builds it -- the same buffer,
    element for element, that `GpuSplitSearcher.stage_random_score` uploads.
    Empty is the default and is a strict no-op: not a zero added, no
    arithmetic executed, and the same record the replica returned before this
    parameter existed. Categorical features are refused with it, matching
    `split.find_best_split`: a categorical candidate is a category *set*
    chosen by a partition search, so only that search's winner would be
    noised while every numerical feature had every candidate noised, which is
    a different regularizer wearing the same name.

    `score_function` selects the same functional the kernels take and
    defaults to the same one they default to, so the replica keeps
    replicating; the reason it is a parameter here rather than a second
    function is the reason `gain_form` is. `SCORE_COSINE` is refused with a
    categorical feature, which is `GpuSplitSearcher.set_score_function`'s
    refusal and `find_best_split`'s before it: the category partition search
    scores with the L2 gain, so allowing the pair would put two score
    functions inside one argmax.
    """
    if len(hist) != 3 * n_features * n_bins:
        raise Error("histogram must hold 3 * n_features * n_bins Int32 words")
    if g_scale <= 0.0 or h_scale <= 0.0:
        raise Error("fixed-point scales must be positive")
    if len(missing_bins) > 0 and len(missing_bins) != n_features:
        raise Error("missing_bins length must equal n_features")
    if len(monotone) > 0 and len(monotone) != n_features:
        raise Error("monotone length must equal n_features")

    var hs = n_features * n_bins
    var g_inv = Float32(1.0 / g_scale)
    var h_inv = Float32(1.0 / h_scale)
    var lambda_l2 = Float32(params.lambda_l2)
    var lambda_l1 = Float32(params.lambda_l1)
    var min_child_hess = Float32(params.min_child_hess)
    var min_data_in_leaf = Int32(params.min_data_in_leaf)
    var bound_lo = _f32_bound(bounds.lo)
    var bound_hi = _f32_bound(bounds.hi)
    var cat_smooth = Float32(params.cat.cat_smooth)
    var cat_l2 = Float32(params.cat.cat_l2)
    var constrained = len(monotone) > 0
    var form = gpu_resolve_gain_form(Int32(gain_form), lambda_l1)
    # The functional, range-checked by the host's own check so the replica
    # cannot accept a code a kernel would refuse.
    check_score_function(score_function)
    var score_code = Int32(score_function)

    var active = features.copy()
    if len(active) == 0:
        active = List[Int](capacity=n_features)
        for f in range(n_features):
            active.append(f)

    if score_function == SCORE_COSINE:
        for slot in range(len(active)):
            var cf = active[slot]
            if cats.is_cat(cf) and cats.n_categories(cf) >= 2:
                raise Error(
                    "score_function=cosine is implemented for numerical"
                    " thresholds only; a categorical feature is searched as"
                    " category partitions scored with the L2 gain, and only"
                    " that search's winner would reach the fold, so the two"
                    " score functions would end up inside one argmax"
                )

    var noisy = len(noise) > 0
    if noisy:
        if len(noise) != len(active) * n_bins:
            raise Error(
                "a random_strength noise plane must hold one Float32 per"
                " (active feature, bin): expected ",
                len(active) * n_bins,
                ", got ",
                len(noise),
            )
        for slot in range(len(active)):
            var cf = active[slot]
            if cats.is_cat(cf) and cats.n_categories(cf) >= 2:
                raise Error(
                    "random_strength is implemented for numerical thresholds"
                    " only; a categorical feature is searched as category"
                    " partitions, and only that search's winner would reach"
                    " a per-candidate draw"
                )

    var out = GpuSplitRecord()
    var best_slot = -1
    var best_gain = Float32(0.0)
    var slot_records = List[GpuSplitRecord]()
    var slot_gains = List[Float32]()
    var slot_runners = List[Float32]()

    for slot in range(len(active)):
        var f = active[slot]
        if f < 0 or f >= n_features:
            raise Error("feature index out of range")
        var base = f * n_bins
        var rec = GpuSplitRecord()

        var tg = Int32(0)
        var th = Int32(0)
        var tc = Int32(0)
        for b in range(n_bins):
            tg += hist[base + b]
            th += hist[hs + base + b]
            tc += hist[2 * hs + base + b]
        var total_g = tg.cast[DType.float32]() * g_inv
        var total_h = th.cast[DType.float32]() * h_inv
        rec.total = ChildStats(
            Float64(total_g), Float64(total_h), Int(tc)
        )

        var permitted = True
        if len(allowed) > 0:
            permitted = f < len(allowed) and allowed[f]
        if not permitted:
            slot_records.append(rec^)
            slot_gains.append(Float32(0.0))
            slot_runners.append(Float32(0.0))
            continue

        var sign = Int32(MONOTONE_FREE)
        if constrained:
            sign = Int32(monotone[f])
        var parent_score = gpu_leaf_score(
            total_g, total_h, lambda_l1, lambda_l2
        )
        # The kernels' node constants, computed from the same totals by the
        # same two functions. See `_scan_slot_kernel`.
        var node_s = gpu_cross_node_s(total_h, lambda_l2)
        var cross_offset = gpu_cross_offset(
            total_g, total_h, lambda_l1, lambda_l2, lambda_l2, node_s
        )
        # And Cosine's, hoisted per slot behind the selector exactly as
        # `_scan_slot_kernel` hoists it.
        var cosine = score_function == SCORE_COSINE
        var parent_cos = Float32(0.0)
        if cosine:
            parent_cos = gpu_cosine_parent(
                total_g, total_h, lambda_l1, lambda_l2
            )
        var gain_here = Float32(0.0)
        # The same runner-up rule the kernel applies: the best gain among
        # every candidate this feature scored except the winner.
        var runner_here = Float32(0.0)
        var left_best_g = Int32(0)
        var left_best_h = Int32(0)
        var left_best_c = Int32(0)

        var n_cat = cats.n_categories(f) if cats.is_cat(f) else 0
        if n_cat >= 2:
            if n_cat >= n_bins:
                raise Error(
                    "categorical feature has more categories than bins"
                )
            if n_cat <= params.cat.max_cat_to_onehot:
                for t in range(1, n_cat + 1):
                    var lg = hist[base + t]
                    var lh = hist[hs + base + t]
                    var lc = hist[2 * hs + base + t]
                    if lc < min_data_in_leaf:
                        continue
                    var lhf = lh.cast[DType.float32]() * h_inv
                    if lhf < min_child_hess:
                        continue
                    var rc = tc - lc
                    if rc < min_data_in_leaf:
                        continue
                    var rhf = gpu_right_sum(
                        total_h, lhf, th, lh, h_inv, form
                    )
                    if rhf < min_child_hess:
                        continue
                    var lgf = lg.cast[DType.float32]() * g_inv
                    var rgf = gpu_right_sum(total_g, lgf, tg, lg, g_inv, form)
                    var gain = gpu_cat_gain(
                        lgf,
                        lhf,
                        rgf,
                        rhf,
                        lambda_l1,
                        lambda_l2,
                        parent_score,
                        node_s,
                        cross_offset,
                        form,
                    )
                    if gain > gain_here:
                        runner_here = gain_here
                        gain_here = gain
                        left_best_g = lg
                        left_best_h = lh
                        left_best_c = lc
                        rec.found = True
                        rec.is_categorical = True
                        rec.feature = f
                        rec.cat_bitset = cat_empty()
                        cat_add(rec.cat_bitset, t)
                    elif gain > runner_here:
                        runner_here = gain
            else:
                var keys = List[Float32]()
                var sorted_bins = List[Int]()
                for t in range(1, n_cat + 1):
                    var lc = hist[2 * hs + base + t]
                    if lc.cast[DType.float32]() < cat_smooth:
                        continue
                    var lg = hist[base + t]
                    var lh = hist[hs + base + t]
                    keys.append(
                        lg.cast[DType.float32]()
                        * g_inv
                        / (lh.cast[DType.float32]() * h_inv + cat_smooth)
                    )
                    sorted_bins.append(t)
                var used = len(sorted_bins)
                if used >= 2:
                    for i in range(1, used):
                        var kv = keys[i]
                        var bv = sorted_bins[i]
                        var j = i - 1
                        while j >= 0 and keys[j] > kv:
                            keys[j + 1] = keys[j]
                            sorted_bins[j + 1] = sorted_bins[j]
                            j -= 1
                        keys[j + 1] = kv
                        sorted_bins[j + 1] = bv

                    var l2c = lambda_l2 + cat_l2
                    # Children at `l2c` against a parent at `lambda_l2`, so
                    # this walk gets its own node constants, exactly as in
                    # `_scan_slot_kernel`.
                    var cat_s = gpu_cross_node_s(total_h, l2c)
                    var cat_offset = gpu_cross_offset(
                        total_g, total_h, lambda_l1, lambda_l2, l2c, cat_s
                    )
                    var max_num_cat = params.cat.max_cat_threshold
                    if (used + 1) // 2 < max_num_cat:
                        max_num_cat = (used + 1) // 2
                    var steps = used if used < max_num_cat else max_num_cat
                    var min_group = Int32(params.cat.min_data_per_group)

                    for d in range(2):
                        var direction = 1 if d == 0 else -1
                        var start_pos = 0 if d == 0 else used - 1
                        var pos = start_pos
                        var group = Int32(0)
                        var lg = Int32(0)
                        var lh = Int32(0)
                        var lc = Int32(0)
                        for i in range(steps):
                            var t = sorted_bins[pos]
                            pos += direction
                            lg += hist[base + t]
                            lh += hist[hs + base + t]
                            var cnt = hist[2 * hs + base + t]
                            lc += cnt
                            group += cnt

                            var lhf = lh.cast[DType.float32]() * h_inv
                            if lc < min_data_in_leaf or lhf < min_child_hess:
                                continue
                            var rc = tc - lc
                            if rc < min_data_in_leaf or rc < min_group:
                                break
                            var rhf = gpu_right_sum(
                                total_h, lhf, th, lh, h_inv, form
                            )
                            if rhf < min_child_hess:
                                break
                            if group < min_group:
                                continue
                            group = Int32(0)

                            var lgf = lg.cast[DType.float32]() * g_inv
                            var rgf = gpu_right_sum(
                                total_g, lgf, tg, lg, g_inv, form
                            )
                            var gain = gpu_cat_gain(
                                lgf,
                                lhf,
                                rgf,
                                rhf,
                                lambda_l1,
                                l2c,
                                parent_score,
                                cat_s,
                                cat_offset,
                                form,
                            )
                            if gain > gain_here:
                                runner_here = gain_here
                                gain_here = gain
                                left_best_g = lg
                                left_best_h = lh
                                left_best_c = lc
                                rec.found = True
                                rec.is_categorical = True
                                rec.feature = f
                                rec.cat_bitset = cat_empty()
                                var p = start_pos
                                for _ in range(i + 1):
                                    cat_add(rec.cat_bitset, sorted_bins[p])
                                    p += direction
                            elif gain > runner_here:
                                runner_here = gain
        else:
            var missing_bin = -1
            if len(missing_bins) > 0:
                missing_bin = missing_bins[f]
            var n_scan = missing_bin if missing_bin >= 0 else n_bins
            var miss_g = Int32(0)
            var miss_h = Int32(0)
            var miss_c = Int32(0)
            if missing_bin >= 0:
                miss_g = hist[base + missing_bin]
                miss_h = hist[hs + base + missing_bin]
                miss_c = hist[2 * hs + base + missing_bin]

            var left_g = Int32(0)
            var left_h = Int32(0)
            var left_c = Int32(0)
            for b in range(n_scan):
                if b == n_scan - 1 and miss_c == Int32(0):
                    break
                left_g += hist[base + b]
                left_h += hist[hs + base + b]
                left_c += hist[2 * hs + base + b]

                # One `random_strength` shift per threshold, shared by the
                # two routing directions; see `_scan_slot_kernel`.
                var bin_noise = Float32(0.0)
                if noisy:
                    bin_noise = noise[slot * n_bins + b]

                if missing_bin >= 0:
                    var dl_g = left_g + miss_g
                    var dl_h = left_h + miss_h
                    var dl_c = left_c + miss_c
                    var dl_hf = dl_h.cast[DType.float32]() * h_inv
                    var dr_hf = gpu_right_sum(
                        total_h, dl_hf, th, dl_h, h_inv, form
                    )
                    if not (
                        dl_hf < min_child_hess
                        or dr_hf < min_child_hess
                        or dl_c < min_data_in_leaf
                        or tc - dl_c < min_data_in_leaf
                    ):
                        var dl_gf = dl_g.cast[DType.float32]() * g_inv
                        var dr_gf = gpu_right_sum(
                            total_g, dl_gf, tg, dl_g, g_inv, form
                        )
                        var gain = gpu_split_gain(
                            gpu_soft_threshold_l1(dl_gf, lambda_l1),
                            dl_hf,
                            gpu_soft_threshold_l1(dr_gf, lambda_l1),
                            dr_hf,
                            lambda_l2,
                            parent_score,
                            sign,
                            bound_lo,
                            bound_hi,
                            constrained,
                            node_s,
                            cross_offset,
                            form,
                            score_code,
                            parent_cos,
                        )
                        if noisy:
                            gain += bin_noise
                        if gain > gain_here:
                            runner_here = gain_here
                            gain_here = gain
                            left_best_g = dl_g
                            left_best_h = dl_h
                            left_best_c = dl_c
                            rec.found = True
                            rec.feature = f
                            rec.bin = b
                            rec.ordinal = 2 * b
                            rec.default_left = True
                        elif gain > runner_here:
                            runner_here = gain

                if missing_bin < 0 or miss_c > Int32(0):
                    var lhf = left_h.cast[DType.float32]() * h_inv
                    var rhf = gpu_right_sum(
                        total_h, lhf, th, left_h, h_inv, form
                    )
                    if lhf < min_child_hess or rhf < min_child_hess:
                        continue
                    if (
                        left_c < min_data_in_leaf
                        or tc - left_c < min_data_in_leaf
                    ):
                        continue
                    var lgf = left_g.cast[DType.float32]() * g_inv
                    var rgf = gpu_right_sum(
                        total_g, lgf, tg, left_g, g_inv, form
                    )
                    var gain = gpu_split_gain(
                        gpu_soft_threshold_l1(lgf, lambda_l1),
                        lhf,
                        gpu_soft_threshold_l1(rgf, lambda_l1),
                        rhf,
                        lambda_l2,
                        parent_score,
                        sign,
                        bound_lo,
                        bound_hi,
                        constrained,
                        node_s,
                        cross_offset,
                        form,
                        score_code,
                        parent_cos,
                    )
                    if noisy:
                        gain += bin_noise
                    if gain > gain_here:
                        runner_here = gain_here
                        gain_here = gain
                        left_best_g = left_g
                        left_best_h = left_h
                        left_best_c = left_c
                        rec.found = True
                        rec.feature = f
                        rec.bin = b
                        rec.ordinal = 2 * b + 1
                        rec.default_left = False
                    elif gain > runner_here:
                        runner_here = gain

        if rec.found:
            var lgf = left_best_g.cast[DType.float32]() * g_inv
            var lhf = left_best_h.cast[DType.float32]() * h_inv
            rec.gain = Float64(gain_here)
            rec.left = ChildStats(
                Float64(lgf), Float64(lhf), Int(left_best_c)
            )
            # The same right-hand rule the winning gain used; see
            # `_scan_slot_kernel`.
            rec.right = ChildStats(
                Float64(
                    gpu_right_sum(total_g, lgf, tg, left_best_g, g_inv, form)
                ),
                Float64(
                    gpu_right_sum(total_h, lhf, th, left_best_h, h_inv, form)
                ),
                Int(tc - left_best_c),
            )
        slot_records.append(rec^)
        slot_gains.append(gain_here)
        slot_runners.append(runner_here)

    # The same fold `_reduce_slots_kernel` performs: the winner by strictly
    # greater gain in ascending slot order, and the node's runner-up as the
    # better of the best losing feature and the winner's own second
    # candidate.
    var runner = Float32(0.0)
    for slot in range(len(slot_records)):
        if not slot_records[slot].found:
            continue
        if best_slot < 0 or slot_gains[slot] > best_gain:
            if best_slot >= 0 and best_gain > runner:
                runner = best_gain
            best_slot = slot
            best_gain = slot_gains[slot]
        elif slot_gains[slot] > runner:
            runner = slot_gains[slot]

    var total = slot_records[0].total.copy()
    if best_slot >= 0:
        out = slot_records[best_slot].copy()
        if slot_runners[best_slot] > runner:
            runner = slot_runners[best_slot]
        out.runner_gain = Float64(runner)
    out.total = total.copy()
    out.parent_value = Float64(
        gpu_leaf_value(
            Float32(total.grad), Float32(total.hess), lambda_l1, lambda_l2
        )
    )
    if out.found:
        out.left_value = Float64(
            gpu_leaf_value(
                Float32(out.left.grad),
                Float32(out.left.hess),
                lambda_l1,
                lambda_l2,
            )
        )
        out.right_value = Float64(
            gpu_leaf_value(
                Float32(out.right.grad),
                Float32(out.right.hess),
                lambda_l1,
                lambda_l2,
            )
        )
    return out^
