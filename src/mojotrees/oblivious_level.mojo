"""The level engine for `grow_policy = oblivious`, in CatBoost's loop order.

What this module changes, and it is a loop order and nothing else
------------------------------------------------------------------
A level of an oblivious tree shares ONE split across every leaf on it. The
grower that shipped still produced that level's statistics **leaf by leaf**:
for each of the level's `L` leaves it called the leaf-wise histogram builder
over that leaf's row-id list, which walks every drawn feature's bin column
for that one leaf. Feature-major bins are `n_rows` bytes a column, so a
level of `L` leaves streamed the whole `n_features * n_rows` bin matrix `L`
times. At depth 6 the levels are 1, 2, 4, 8, 16 and 32 leaves wide, so one
tree read the bin matrix 63 times where it needs to read it 6.

CatBoost reads it once per level. `CalcScores` (`greedy_tensor_search.cpp`,
`CalcBestScore`) fans out **over candidate features**, and each task owns a
private `stats[leaf * bucketCount + bin]` array that every leaf of the level
accumulates into during **one contiguous pass over the documents**
(`scoring.cpp`, `TStatsIndexer::GetIndex` and `UpdateWeighted`: the index is
`BucketCount * LeafIndices[obj] + quantizedValue`). No atomics, no shared
histogram, no cross-thread reduction, and the working set is the level's
`L * n_bins` cells of one feature rather than the whole bin matrix.

`accumulate_level_stats` below is that pass. It is the only thing in this
module: the search, the leaf values, the node ids and the frontier are
untouched, because the defect was never in any of them.

The layout, which is chosen so nothing downstream has to change
---------------------------------------------------------------
The flat buffer is `[slot][feature][bin]`, and a slot's span is byte for
byte what a `Histogram` of the same shape holds: `_gh` interleaved
`(g, h)` at `2 * (f * n_bins + b)`, `_count` at `f * n_bins + b`. So
`scatter_level_stats` hands a slot to a pooled `Histogram` with a copy and
no repack, and every consumer downstream -- `find_best_split_shared`,
`subtract_histogram_into`, `_leaf_value` -- sees exactly the object it saw
before.

Per feature task the working set is `L * n_bins` cells, 24 bytes each: 196 KB
at 32 leaves and 256 bins, which is the arrangement's whole point. The flat
buffer as a whole is `L * n_features * n_bins` cells and is allocated once
per tree at the widest level the tree will reach.

Determinism
-----------
Every cell of `[slot][f][bin]` is written by exactly one task -- the task
that owns feature `f` -- and the addends arrive in ascending fold-document
order inside that task. Neither the set of addends nor their order depends
on the task count, so the buffer is identical at every
`MOJOTREES_NUM_WORKERS` and on every machine.

**It is NOT identical to what the leaf-wise builder produced**, and that is
declared rather than discovered: `histogram`'s subset builder folds
per-row-block partial sums, so its addend order is a block order, while this
one is strictly ascending. The cells differ in the last bits when they
differ at all. See `docs/design/OBLIVIOUS.md` and the level engine's entry
in the grower for the two places that can turn into a different tree (an
exact tie between two candidates' summed gains).
"""

from std.os import getenv

from .binning import BinnedMatrix
from .histogram import Histogram
from .parallel import DispatchSettings, dispatch_features_with


def level_engine_enabled() raises -> Bool:
    """`MOJOTREES_OBLIVIOUS_LEVEL_ENGINE` as a switch, defaulting to on.

    Unset means ON, which is what makes this engine *reached* rather than
    built: every `grow_policy = oblivious` CPU fit it supports takes it
    without anyone opting in. `0` restores the leaf-by-leaf builder, and is
    there so an A/B runs in one process rather than across two commits.

    Refuses a value it does not recognize rather than falling back to a
    default, which is the standing rule for every environment knob in this
    package (`phase_profile.resolve_mode` says the same thing at greater
    length): a typo that silently selects the path the caller was trying to
    leave is worse than an error.
    """
    var raw = getenv("MOJOTREES_OBLIVIOUS_LEVEL_ENGINE")
    if raw == "" or raw == "1" or raw == "on" or raw == "true":
        return True
    if raw == "0" or raw == "off" or raw == "false":
        return False
    raise Error(
        "MOJOTREES_OBLIVIOUS_LEVEL_ENGINE must be one of 1, on, true, 0,"
        " off, false; got '",
        raw,
        "'",
    )


def level_stats_ops(
    n_docs: Int, n_columns: Int, n_slots: Int, n_bins: Int
) -> Int:
    """The work estimate `accumulate_level_stats` hands the scheduler.

    One indexed bin load plus one read-modify-write of a 24-byte cell per
    (document, column), plus the zeroing pass over the slots this call
    writes, which `apple_cpu_policy.derive_accumulation_plan` counts for the
    leaf-wise builder for the same reason: at a small level the zeroing is
    the larger half and a plan that ignores it fans out a call that has
    nothing to do.
    """
    return n_docs * n_columns + n_slots * n_columns * n_bins


def accumulate_level_stats(
    mut gh: List[Float64],
    mut count: List[Int],
    data: BinnedMatrix,
    docs: List[Int],
    slots: List[Int],
    fold_grad: List[Float64],
    fold_hess: List[Float64],
    columns: List[Int],
    n_slots: Int,
    settings: DispatchSettings,
) raises:
    """One pass over `docs` per column, folding every slot of the level in.

    `docs[i]` is a row id into `data`, `slots[i]` the slot in `[0, n_slots)`
    that row belongs to, and `fold_grad[i]` / `fold_hess[i]` that row's
    gradient and hessian already gathered into fold order -- gathered, so
    that the inner loop reads them sequentially and the only indexed load
    left is the bin.

    `docs` must be **ascending**. Nothing here checks it and nothing here
    breaks without it; it is a locality contract, and it is the contract
    that makes this function worth having. Ascending row ids walk a
    feature-major bin column forwards, so the column is streamed once. The
    leaf-major order a per-leaf builder produces walks it once per leaf.

    `gh` and `count` are grown, never shrunk, and every cell of every column
    named in `columns` is written before it is read, so their contents on
    entry are irrelevant. Columns NOT named are left exactly as they were,
    which is the same contract the leaf-wise builder has under a feature
    draw: a buffer's undrawn columns are stale and no consumer may read
    them.
    """
    var n_bins = data.n_bins
    var cells = data.n_features * n_bins
    var n = len(docs)
    var n_active = len(columns)
    if n_slots <= 0 or n_active <= 0:
        return
    if len(slots) != n or len(fold_grad) != n or len(fold_hess) != n:
        raise Error(
            "the level fold's documents, slots, gradients and hessians must"
            " be the same length"
        )
    var need = n_slots * cells
    if len(gh) < 2 * need:
        gh.resize(2 * need, 0.0)
    if len(count) < need:
        count.resize(need, 0)

    var fail = List[UInt8](capacity=n_active)
    fail.resize(n_active, UInt8(0))

    var fail_p = fail.unsafe_ptr()
    var ghp = gh.unsafe_ptr()
    var cp = count.unsafe_ptr()
    var bins_p = data.bins.unsafe_ptr()
    var docs_p = docs.unsafe_ptr()
    var slots_p = slots.unsafe_ptr()
    var g_p = fold_grad.unsafe_ptr()
    var h_p = fold_hess.unsafe_ptr()
    var cols_p = columns.unsafe_ptr()
    var n_rows = data.n_rows
    var n_features = data.n_features

    def one_column(j: Int) raises {imm}:
        var f = cols_p.unsafe_load(j)
        if f < 0 or f >= n_features:
            raise Error("column out of range in the oblivious level fold")
        var col = f * n_bins

        # This column's stripe in every slot, zeroed by the task that owns
        # it. Zeroing here rather than over the whole buffer keeps it
        # parallel, keeps it to the drawn columns, and leaves the stripe
        # in cache for the pass below.
        for s in range(n_slots):
            var base = s * cells + col
            for b in range(n_bins):
                ghp.unsafe_store(2 * (base + b), 0.0)
                ghp.unsafe_store(2 * (base + b) + 1, 0.0)
                cp.unsafe_store(base + b, 0)

        var fb = bins_p.unsafe_offset(f * n_rows)
        for i in range(n):
            var cell = (
                slots_p.unsafe_load(i) * cells
                + col
                + Int(fb.unsafe_load(docs_p.unsafe_load(i)))
            )
            var gi = 2 * cell
            ghp.unsafe_store(gi, ghp.unsafe_load(gi) + g_p.unsafe_load(i))
            ghp.unsafe_store(
                gi + 1, ghp.unsafe_load(gi + 1) + h_p.unsafe_load(i)
            )
            cp.unsafe_store(cell, cp.unsafe_load(cell) + 1)

    def guarded(j: Int) {imm}:
        try:
            one_column(j)
        except:
            fail_p.unsafe_store(j, UInt8(1))

    dispatch_features_with(
        settings,
        guarded,
        n_active,
        level_stats_ops(n, n_active, n_slots, n_bins),
    )

    for j in range(n_active):
        if fail[j] != UInt8(0):
            one_column(j)
            raise Error(
                "the oblivious level accumulation failed on column ", j
            )


def copy_level_slot(
    mut hist: Histogram,
    gh: List[Float64],
    count: List[Int],
    slot: Int,
    columns: List[Int],
    whole: Bool,
) raises:
    """Copy one slot's drawn columns out of the flat level buffer into a
    pooled `Histogram`.

    `hist` is a pooled buffer whose undrawn columns hold whatever the last
    node to use it left there, and this preserves that: only the columns in
    `columns` are written, exactly as the leaf-wise builder writes only the
    ones it was given. `whole` says the caller has already established that
    `columns` is every feature in ascending order (`level_columns_are_whole`
    answers it once per tree rather than once per leaf), which turns the copy
    into one contiguous run.

    This is the only cost the flat buffer adds over writing the histograms
    in place, and it is one sequential copy of `n_features * n_bins` cells
    per leaf of the level against the `n_features * n_rows` bytes of bin
    matrix the leaf-by-leaf builder re-read for that same leaf.
    """
    var n_features = hist.n_features
    var n_bins = hist.n_bins
    var cells = n_features * n_bins
    var sbase = slot * cells
    if len(gh) < 2 * (sbase + cells) or len(count) < sbase + cells:
        raise Error("the level stats buffer is too small for slot ", slot)
    var src_gh = gh.unsafe_ptr()
    var src_c = count.unsafe_ptr()
    var dst_gh = hist._gh.unsafe_ptr()
    var dst_c = hist._count.unsafe_ptr()
    if whole:
        for i in range(cells):
            dst_gh.unsafe_store(2 * i, src_gh.unsafe_load(2 * (sbase + i)))
            dst_gh.unsafe_store(
                2 * i + 1, src_gh.unsafe_load(2 * (sbase + i) + 1)
            )
            dst_c.unsafe_store(i, src_c.unsafe_load(sbase + i))
    else:
        for j in range(len(columns)):
            var col = columns[j] * n_bins
            for b in range(n_bins):
                var src = sbase + col + b
                dst_gh.unsafe_store(2 * (col + b), src_gh.unsafe_load(2 * src))
                dst_gh.unsafe_store(
                    2 * (col + b) + 1, src_gh.unsafe_load(2 * src + 1)
                )
                dst_c.unsafe_store(col + b, src_c.unsafe_load(src))


def level_columns_are_whole(columns: List[Int], n_features: Int) -> Bool:
    """Whether `columns` is `[0, n_features)` in ascending order, which is
    what a fit without a feature draw hands the builder and what lets
    `copy_level_slot` run as one contiguous copy. Asked once per tree."""
    if len(columns) != n_features:
        return False
    for j in range(n_features):
        if columns[j] != j:
            return False
    return True
