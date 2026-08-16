"""Ordered boosting: the permutation, the fold ladder, and the memory it costs.

This is the mechanism CatBoost is named for. A row's derivative is taken from
a model that was never fitted on that row, which removes the "prediction
shift" that ordinary boosting has: in ordinary boosting every row's residual
is measured against an ensemble that has already been fitted on that row, so
the residual is optimistically small and the split search sees a target it has
partly memorized.

Verified against CatBoost `master` (August 2026). The files that settle it:

- `catboost/private/libs/algo/fold.h` -- `TFold::TBodyTail` holds the
  `BodyFinish` / `TailFinish` pair and the `Approx`, `WeightedDerivatives`,
  `SampleWeightedDerivatives` planes; `TFold::BodyTailArr` is the ladder.
- `catboost/private/libs/algo/fold.cpp` -- `SelectMinBatchSize`,
  `SelectTailSize`, `UpdateSize`, `TFold::BuildDynamicFold` (the ladder loop),
  `InitPermutationData` (the permutation).
- `catboost/libs/data/objects_grouping.cpp` -- `NCB::Shuffle`, the *block*
  permutation.
- `catboost/private/libs/algo/learn_context.cpp` -- `IsPermutationNeeded`,
  `CountLearningFolds`, `TFoldsCreationParams`, and the loop that builds
  `Folds` plus the separate `AveragingFold`.
- `catboost/private/libs/algo/train.cpp` -- `TrainOneIteration` (which fold is
  taken per tree), `UpdateLearningFold`.
- `catboost/private/libs/algo/tensor_search_helpers.cpp` --
  `CalcWeightedDerivatives` (derivatives over `[0, TailFinish)` from that
  rung's own `Approx`).
- `catboost/private/libs/algo/approx_calcer.cpp` -- `CalcApproxDeltaSimple`
  (leaf values from `[0, BodyFinish)`, applied over `[0, TailFinish)`) and
  `CalcLeafValues` (**the model's** leaf values, from the `AveragingFold`).
- `catboost/private/libs/algo/scoring.cpp::CalcScoresForSubCandidate` -- the
  split score is SUMMED over every rung, each with its own scaled `l2`.
- `catboost/private/libs/options/boosting_options.cpp` -- the option surface
  and its defaults.
- `catboost/private/libs/options/defaults_helper.h` --
  `DefaultFoldPermutationBlockSize`, `UpdateBoostingTypeOption`.

See `docs/design/CATBOOST_CATALOG.md`, the A7 note and A17.

The ladder
----------
`BuildDynamicFold` grows a geometric ladder of prefixes of the permuted row
order. Writing `b_0 < b_1 < ... < b_K = n`:

    b_0    = SelectMinBatchSize(n)
    b_{f+1} = min(ceil(b_f * fold_len_multiplier), n)

Rung `f` (there are `K` of them) has body `[0, b_f)` and tail `[0, b_{f+1})`.
The body is what its leaf values are fitted on; the tail is what its raw-score
plane covers. A row at permuted position `q` in `[b_f, b_{f+1})` therefore sits
in rung `f`'s tail and outside rung `f`'s body: rung `f`'s model has never seen
it, which is the entire point.

**The first `b_0` positions are a leak and CatBoost accepts it.** Positions
below `b_0` are inside every rung's body, so their derivative always comes from
a model that has seen them. That is the price of not keeping `n` models.

Memory, a derived bound and not a measurement
---------------------------------------------
The rungs are nested prefixes, not disjoint blocks, so the ladder does NOT cost
`K * n`. Plane `f` has length `b_{f+1}`, and the lengths below the top are
geometric with ratio `m = fold_len_multiplier`, so with `b_{K-1} < n`:

    sum_{f=0}^{K-1} b_{f+1}  =  n  +  sum_{f=1}^{K-1} b_f
                             <   n  +  n * m / (m - 1)
                             =   n * (2m - 1) / (m - 1)

which is a strict `3n` at the default `m = 2.0`, independent of `K`. The `n` on
its own line is the top rung, whose plane covers every row; the geometric tail
below it is what would be `2n` if `n` landed exactly on the ladder, and the
truncated top step is why the honest bound is 3 and not 2. At `n = 1e6` with
the defaults the exact count is `2 638 200`, or `2.64n`.
`ordered_plane_entries` computes it; the closed form above is the bound. Both
are derived, neither is measured. `K` itself is logarithmic:

    K = ceil(log_m(n / b_0)) + 1,  b_0 = SelectMinBatchSize(n)

which is 14 at `n = 1e6` with the defaults (`b_0 = 100`, `m = 2`). The count
grows with `n` and is not a constant, but it is computable from `n` alone
before the first tree, so a device can size its buffers once at fit time.

Determinism
-----------
The permutation is random and must be reproducible under `seed` at any
`MOJOTREES_NUM_WORKERS` and on any machine. **CatBoost's own permutation is
not**: `NCB::Shuffle` calls `CreateShuffledIndices`, a sequential Fisher-Yates
over one running generator, so a row's destination depends on every draw taken
before it.

Ours is counter-based, in the same family as `sampling._mvs_stream` and
`langevin._langevin_row_stream` but a different shape, because a permutation is
a global constraint and a bootstrap weight is not:

- MVS and Langevin can give each row an *independent* draw, so "row `r` reads
  `stream + r`, nothing advances" is the whole scheme.
- A permutation cannot: the draws have to be turned into a total order. So each
  *block* gets an independent 64-bit key from the same kind of stream, and the
  blocks are then sorted by `(key, block index)`. The key is a pure function of
  `(seed, permutation index, block index)`; the sort is a total order with an
  explicit tie-break, so it has exactly one answer. No sequential stream, no
  per-worker generator, and no dependence on how the sort is scheduled.

Two domain constants, for the reason `langevin.mojo` gives for having two: the
permutation stream and the per-round permutation-choice stream are both keyed
by a small integer against the same seed, so without separation permutation 3's
keys and round 3's choice would be drawn from one counter run.

`_ORDERED_BLOCK_TIE` is not a third stream, it is the statement that ties are
broken by block index. Keys are 64 bits and the block count is at most a few
tens of thousands, so the collision probability is under `m^2 / 2^65`, about
`5e-11` at `m = 40000`; the tie-break makes the *result* exact regardless, at
the cost of a negligible bias toward low block indices among colliding pairs.

Blocks, not rows
----------------
CatBoost permutes *blocks* of consecutive rows, not rows
(`NCB::Shuffle`, `permuteBlockSize != 1` arm), with
`DefaultFoldPermutationBlockSize(n) = min(256, n / 1000 + 1)`. Rows inside a
block keep their original relative order. We keep that, because it is what
makes the permuted read of a row-major matrix land in cache lines rather than
scattering, and because it is the shape a device gather wants.
"""

from std.math import ceil

from .rng import GOLDEN, splitmix64


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Separates the block-key stream from every other per-index stream in the
# package. `sampling._MVS_DOMAIN` and `langevin._LANGEVIN_ROW_DOMAIN` are the
# same device for the same reason: several mechanisms are keyed by (seed, small
# integer) and default their seed to 0, so a caller who sets one seed and turns
# two of them on would otherwise draw the same numbers twice.
comptime _ORDERED_PERM_DOMAIN = UInt64(0x0DE12ED91E125071)

# The per-round choice of which permutation a tree is grown against gets its
# OWN domain rather than sharing the permutation domain with a different index,
# because a permutation index and a round index occupy the same small integers.
# Sharing would make round 3's choice the same draw as permutation 3's key 0.
comptime _ORDERED_CHOICE_DOMAIN = UInt64(0x0DE12ED9C401CE55)


# ---------------------------------------------------------------------------
# Defaults, verified from CatBoost source
# ---------------------------------------------------------------------------

# `boosting_options.cpp`: `FoldLenMultiplier("fold_len_multiplier", 2.0)`.
# `TBoostingOptions::Validate` requires it strictly greater than 1.
comptime DEFAULT_FOLD_LEN_MULTIPLIER = 2.0

# `boosting_options.cpp`: `PermutationCount("permutation_count", 4)`, which
# `CountLearningFolds` turns into `max(1, 4 - 1) = 3` *learning* permutations
# (learn_context.cpp:48-50, :81-83). The fourth is the averaging fold, which is
# a plain single-rung fold and is not a member of this ladder.
#
# **Ours defaults to 1 and that is a divergence, stated rather than hidden.**
# Every extra permutation multiplies the raw-score planes -- `2n` Float64 each
# at the default multiplier -- and its only effect is to decorrelate the
# structure search across rounds. One permutation is the mechanism; three is
# variance reduction on top of it. `permutation_count=3` reproduces CatBoost's
# count exactly and costs exactly three times the planes.
comptime DEFAULT_ORDERED_PERMUTATION_COUNT = 1

# `boosting_options.cpp`: `PermutationBlockSize("fold_permutation_block", 0)`,
# and `defaults_helper.h`: `FoldPermutationBlockSizeNotSet = 0`. Zero means
# "resolve from the row count", which is what `resolve_block_size` does.
comptime DEFAULT_FOLD_PERMUTATION_BLOCK_SIZE = 0

# CatBoost draws its permutations from the fit's global `random_seed`, whose
# default is 0, through a sequential generator that cannot be named. Ours is a
# parameter for the same reason `langevin.DEFAULT_LANGEVIN_SEED` is, and keeps
# 0 so a port of a default CatBoost configuration reads the same.
comptime DEFAULT_ORDERED_SEED = 0

# `defaults_helper.h::DefaultFoldPermutationBlockSize` caps the block at 256.
comptime _MAX_PERMUTATION_BLOCK = 256

# `fold.cpp::SelectMinBatchSize`: `learnSampleCount > 500 ? Min<ui32>(100,
# learnSampleCount / 50) : 1`.
comptime _MIN_BATCH_ROW_THRESHOLD = 500
comptime _MIN_BATCH_CAP = 100
comptime _MIN_BATCH_DIVISOR = 50

# `defaults_helper.h::UpdateBoostingTypeOption` hard-sets Plain at or above
# this row count. It is recorded here because it is the number every summary of
# CatBoost's default gets wrong; see `catboost_auto_is_ordered` for what it
# actually decides.
comptime CATBOOST_ORDERED_ROW_THRESHOLD = 50000

# The second clause of the same test: fewer than this many iterations also
# forces Plain.
comptime CATBOOST_ORDERED_ITERATION_THRESHOLD = 500


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def min_batch_size(n_rows: Int) -> Int:
    """`fold.cpp::SelectMinBatchSize`, verbatim.

    `n > 500 ? min(100, n / 50) : 1`. It is the size of the seed prefix: the
    first `b_0` permuted positions sit inside every rung's body and therefore
    get a derivative from a model that has seen them. CatBoost pays that leak
    rather than keeping a rung per row.
    """
    if n_rows <= 0:
        return 0
    if n_rows <= _MIN_BATCH_ROW_THRESHOLD:
        return 1
    var by_ratio = n_rows // _MIN_BATCH_DIVISOR
    var size = _MIN_BATCH_CAP if _MIN_BATCH_CAP < by_ratio else by_ratio
    if size < 1:
        size = 1
    return size


def default_permutation_block_size(n_rows: Int) -> Int:
    """`defaults_helper.h::DefaultFoldPermutationBlockSize`, verbatim:
    `min(256, docCount / 1000 + 1)`.

    A dataset under 1000 rows gets block size 1, which is a plain row
    permutation; a dataset of 256000 rows or more gets the 256 cap.
    """
    if n_rows <= 0:
        return 1
    var by_ratio = n_rows // 1000 + 1
    return _MAX_PERMUTATION_BLOCK if _MAX_PERMUTATION_BLOCK < by_ratio else by_ratio


def catboost_auto_is_ordered(
    n_rows: Int, n_iterations: Int, is_gpu: Bool, is_multiclass: Bool
) -> Bool:
    """Whether CatBoost, left alone, would choose Ordered.

    **The widely repeated summary "Ordered below 50k rows on CPU, Plain above"
    is wrong, and this function is the corrected rule read off the source.**

    - `boosting_options.cpp:16` constructs `BoostingType("boosting_type",
      EBoostingType::Plain)`. That is the value a CPU fit keeps.
    - `catboost_options.cpp:802-806` is the only place Ordered is ever
      installed as a default, and its condition includes
      `TaskType == ETaskType::GPU`. **On the CPU the default is Plain at every
      row count.**
    - The same block excludes the multiclass-only metrics and
      `RMSEWithUncertainty` / `MultiLogloss` / `MultiCrossEntropy`, so those
      stay Plain even on the GPU. `docs/en/references/training-parameters/common.md`
      agrees.
    - `defaults_helper.h:33-42::UpdateBoostingTypeOption` then hard-sets Plain
      when `learnSampleCount >= 50000 || IterationCount < 500`. `TOption::SetDefault`
      does not raise `IsSetFlag` (`option.h:28-31, 80-85`), so this later test
      still sees `NotSet()` and wins over the GPU default above. **The iteration
      clause is not optional and is usually omitted from the summary.**

    So the 50000 is real, but it is a threshold for turning Ordered OFF on the
    GPU, not for turning it on at all on the CPU.
    """
    if not is_gpu:
        return False
    if is_multiclass:
        return False
    if n_rows >= CATBOOST_ORDERED_ROW_THRESHOLD:
        return False
    if n_iterations < CATBOOST_ORDERED_ITERATION_THRESHOLD:
        return False
    return True


def check_fold_len_multiplier(multiplier: Float64) raises:
    """`TBoostingOptions::Validate`: "fold len multiplier should be greater
    than 1". Written so a NaN is rejected."""
    if not (multiplier > 1.0):
        raise Error("fold_len_multiplier must be greater than 1")


def fold_ladder(n_rows: Int, multiplier: Float64) raises -> List[Int]:
    """The ladder `[b_0, b_1, ..., b_K]`, `fold.cpp::BuildDynamicFold`'s loop.

    Length `K + 1`; there are `K` rungs, rung `f` having body `[0, ladder[f])`
    and tail `[0, ladder[f + 1])`. The loop always produces at least one rung,
    matching CatBoost's `while (BodyTailArr.empty() || leftPartLen < n)`.

    `UpdateSize`'s query clamp is not carried: it exists to round a body up to
    a query boundary and mojotrees has no query grouping in this path.
    """
    check_fold_len_multiplier(multiplier)
    if n_rows <= 0:
        raise Error("ordered boosting needs at least one row")
    var ladder = List[Int]()
    var start = min_batch_size(n_rows)
    if start > n_rows:
        start = n_rows
    ladder.append(start)
    while True:
        var prev = ladder[len(ladder) - 1]
        var grown = Int(ceil(Float64(prev) * multiplier))
        # `multiplier > 1` and `prev >= 1` make this strictly larger, but a
        # multiplier one ulp above 1 at a huge `prev` could round back; the
        # floor keeps the loop finite without changing any reachable value.
        if grown <= prev:
            grown = prev + 1
        if grown > n_rows:
            grown = n_rows
        ladder.append(grown)
        if grown >= n_rows:
            break
    return ladder^


def n_rungs(ladder: List[Int]) -> Int:
    """How many models the ladder carries: one fewer than its length."""
    return len(ladder) - 1


def fold_bounds(ladder: List[Int]) -> List[Int]:
    """The device-side boundary plane: `[0, b_0, b_1, ..., b_K]`.

    `n_folds + 1` entries for `n_folds = K + 1` disjoint segments, segment `s`
    being permuted positions `[bounds[s], bounds[s + 1])`. Segment 0 is the
    seed prefix `[0, b_0)`; segment `f + 1` is the set of rows that first enter
    rung `f`'s tail. Prefix offsets rather than per-row fold ids, because every
    consumer wants a contiguous window and per-row ids would make it a gather.
    """
    var bounds = List[Int](capacity=len(ladder) + 1)
    bounds.append(0)
    for i in range(len(ladder)):
        bounds.append(ladder[i])
    return bounds^


def plane_offsets(ladder: List[Int]) -> List[Int]:
    """Prefix offsets of the per-rung raw-score planes in one flat buffer.

    `K + 1` entries: rung `f`'s plane is `[offsets[f], offsets[f + 1])` and has
    length `ladder[f + 1]`. The last entry is the total, which is the number
    `ordered_plane_entries` reports.
    """
    var offsets = List[Int](capacity=len(ladder))
    var total = 0
    offsets.append(0)
    for f in range(n_rungs(ladder)):
        total += ladder[f + 1]
        offsets.append(total)
    return offsets^


def ordered_plane_entries(
    n_rows: Int, multiplier: Float64, n_permutations: Int
) raises -> Int:
    """Exactly how many Float64 the raw-score planes occupy for one fit.

    A **derived** number, not a measurement: it is the sum of the rung tail
    lengths times the permutation count. The closed-form bound is
    `n * (2 * multiplier - 1) / (multiplier - 1) * n_permutations`, a strict
    `3 * n * n_permutations` at the default multiplier; the module docstring
    derives it. Times 8 bytes for the host byte count. On a device staging
    derivatives at Int16 it is 4 bytes per entry and at Int32 it is 8, so at
    `n = 1e6` one permutation is 10.6 MB / 21.1 MB and CatBoost's three
    learning permutations would be 31.7 MB / 63.3 MB.
    """
    if n_permutations < 1:
        raise Error("permutation_count must be positive")
    var ladder = fold_ladder(n_rows, multiplier)
    var offsets = plane_offsets(ladder)
    return offsets[len(offsets) - 1] * n_permutations


def fold_ids(ladder: List[Int], n_rows: Int) -> List[Int]:
    """Rung index per permuted position, dense.

    Position `q` reads rung `f`, the largest `f` with `ladder[f] <= q`, floored
    at 0 so the seed prefix `[0, b_0)` reads rung 0. Rung `f`'s plane has
    length `ladder[f + 1] > q` by construction, so the read is always in range.

    Dense rather than a binary search per read because it is built once per fit
    and read once per row per round, and because the device wants the boundary
    form (`fold_bounds`) instead -- this is the host's convenience, not the
    wire format.
    """
    var ids = List[Int](capacity=n_rows)
    var last = n_rungs(ladder) - 1
    var f = 0
    for q in range(n_rows):
        while f < last and ladder[f + 1] <= q:
            f += 1
        ids.append(f)
    return ids^


# ---------------------------------------------------------------------------
# The permutation
# ---------------------------------------------------------------------------


def _perm_stream(seed: Int, permutation_index: Int) -> UInt64:
    """Start of the key stream for one permutation.

    Same shape as `sampling._mvs_stream` and `langevin._langevin_row_stream`:
    mix the seed against this module's domain constant, spread the index by
    the golden-ratio increment, mix again. Sign bits are masked off so a
    negative seed is accepted without relying on signed-to-unsigned conversion.

    The result is a *start*, not a running state. Nothing advances it.
    """
    var h = splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _ORDERED_PERM_DOMAIN)
    return splitmix64(
        h ^ (UInt64(permutation_index & 0x7FFFFFFFFFFFFFFF) * GOLDEN)
    )


def _choice_stream(seed: Int, round: Int) -> UInt64:
    """Start of the stream that picks which permutation a round grows against.
    A second domain, so round `j` and permutation `j` never share a draw."""
    var h = splitmix64(
        UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _ORDERED_CHOICE_DOMAIN
    )
    return splitmix64(h ^ (UInt64(round & 0x7FFFFFFFFFFFFFFF) * GOLDEN))


def block_key(stream: UInt64, block: Int) -> UInt64:
    """The sort key belonging to one block of one permutation.

    Exposed rather than hidden inside the builder so a test can pin one block's
    key without building a permutation, and so a device lane can reproduce the
    same value from the same three integers.
    """
    return splitmix64(stream ^ (UInt64(block & 0x7FFFFFFFFFFFFFFF) * GOLDEN))


def _order_by_key(keys: List[UInt64]) -> List[Int]:
    """Block indices sorted by `(key, index)` ascending, bottom-up merge sort.

    Merge sort rather than anything in-place-and-clever because the ordering is
    the determinism claim: a stable merge over a total order has exactly one
    answer, at any worker count, on any machine, and there is no comparison
    whose result depends on how the range was cut. The tie-break on the index
    is what makes the order total when two 64-bit keys collide.
    """
    var m = len(keys)
    var src = List[Int](capacity=m)
    for i in range(m):
        src.append(i)
    if m < 2:
        return src^
    var dst = List[Int](capacity=m)
    for _ in range(m):
        dst.append(0)
    var width = 1
    while width < m:
        var lo = 0
        while lo < m:
            var mid = lo + width
            if mid > m:
                mid = m
            var hi = lo + 2 * width
            if hi > m:
                hi = m
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                var a = src[i]
                var b = src[j]
                var ka = keys[a]
                var kb = keys[b]
                # `<=` on the key with the index tie-break folded in: `a < b`
                # always holds here because the left run holds lower indices.
                if ka <= kb:
                    dst[k] = a
                    i += 1
                else:
                    dst[k] = b
                    j += 1
                k += 1
            while i < mid:
                dst[k] = src[i]
                i += 1
                k += 1
            while j < hi:
                dst[k] = src[j]
                j += 1
                k += 1
            lo += 2 * width
        for t in range(m):
            src[t] = dst[t]
        width *= 2
    return src^


def ordered_permutation(
    seed: Int, permutation_index: Int, n_rows: Int, block_size: Int
) raises -> List[Int]:
    """The permuted row order: `perm[q]` is the original row at position `q`.

    `NCB::Shuffle`'s block permutation with a deterministic order in place of
    its sequential Fisher-Yates. Rows inside a block keep their original
    relative order, and the short final block is placed wherever its key sends
    it, exactly as CatBoost's `currentIdx += blockEndIndx - blockStartIdx`
    does -- so permuted positions are NOT aligned to the block size in general.
    """
    if n_rows < 0:
        raise Error("n_rows must be nonnegative")
    if block_size < 1:
        raise Error("permutation block size must be positive")
    var perm = List[Int](capacity=n_rows)
    if n_rows == 0:
        return perm^
    var m = (n_rows + block_size - 1) // block_size
    var stream = _perm_stream(seed, permutation_index)
    var keys = List[UInt64](capacity=m)
    for b in range(m):
        keys.append(block_key(stream, b))
    var order = _order_by_key(keys)
    for i in range(m):
        var b = order[i]
        var start = b * block_size
        var stop = start + block_size
        if stop > n_rows:
            stop = n_rows
        for r in range(start, stop):
            perm.append(r)
    return perm^


def permutation_choice(seed: Int, round: Int, n_permutations: Int) raises -> Int:
    """Which permutation round `round` grows its tree against.

    CatBoost's is `Folds[Rand.GenRand() % foldCount]` (`train.cpp:208`), one
    draw off the sequential learn-progress generator, so it depends on every
    draw any earlier round took. Ours is a function of `(seed, round)` alone,
    which is also what makes a continued run draw what an uninterrupted run
    would have drawn -- the same reason every other seeded decision in
    `boosting._boost_rounds` reads the absolute round index.
    """
    if n_permutations < 1:
        raise Error("permutation_count must be positive")
    if n_permutations == 1:
        return 0
    var draw = splitmix64(_choice_stream(seed, round))
    return Int(draw % UInt64(n_permutations))


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@fieldwise_init
struct OrderedBoostingParams(Copyable, Movable):
    """CatBoost's `boosting_type=Ordered`.

    Disabled by default, so an untouched bundle changes nothing: a fit that
    never enables it grows the trees it grows today, byte for byte.

    The four knobs and where each comes from:

    - `permutation_count` -- CatBoost's `permutation_count` (default 4) after
      `CountLearningFolds` turns it into `max(1, count - 1)` learning
      permutations, so 3 at their defaults. **Ours defaults to 1**; see
      `DEFAULT_ORDERED_PERMUTATION_COUNT` for why and for what it costs to
      match them.
    - `fold_len_multiplier` -- CatBoost's `fold_len_multiplier`, default 2.0,
      validated strictly greater than 1.
    - `permutation_block_size` -- CatBoost's `fold_permutation_block`, default
      0 meaning "resolve from the row count" (`resolve_block_size`).
    - `seed` -- no CatBoost equivalent; theirs is an unnamed draw off the
      global generator. Ours is a parameter so the permutation is reproducible
      by name.

    **`has_time` is expressed as `permutation_block_size = n` rather than as a
    flag.** CatBoost's `HasTimeFlag` forces `PermutationCount = 1`
    (`catboost_options.cpp:1043`) and makes `IsPermutationNeeded` false
    (`learn_context.cpp:38-46`), which sets `FoldPermutationBlockSize =
    learnSampleCount` -- one block, identity permutation. A single block here
    does exactly that and needs no second spelling of the same state.
    """

    var enabled: Bool
    var permutation_count: Int
    var fold_len_multiplier: Float64
    var permutation_block_size: Int
    var seed: Int

    @staticmethod
    def disabled() -> OrderedBoostingParams:
        """Plain boosting: the library default and what every fit today does."""
        return OrderedBoostingParams(
            False,
            DEFAULT_ORDERED_PERMUTATION_COUNT,
            DEFAULT_FOLD_LEN_MULTIPLIER,
            DEFAULT_FOLD_PERMUTATION_BLOCK_SIZE,
            DEFAULT_ORDERED_SEED,
        )

    @staticmethod
    def enable(
        permutation_count: Int = DEFAULT_ORDERED_PERMUTATION_COUNT,
        fold_len_multiplier: Float64 = DEFAULT_FOLD_LEN_MULTIPLIER,
        permutation_block_size: Int = DEFAULT_FOLD_PERMUTATION_BLOCK_SIZE,
        seed: Int = DEFAULT_ORDERED_SEED,
    ) -> OrderedBoostingParams:
        """Ordered boosting. `validate` is what refuses a bad combination; the
        trainer calls it before the first tree."""
        return OrderedBoostingParams(
            True,
            permutation_count,
            fold_len_multiplier,
            permutation_block_size,
            seed,
        )

    def validate(self) raises:
        """CatBoost's `TBoostingOptions::Validate` rules for these fields, plus
        the block-size range they leave implicit."""
        if not self.enabled:
            return
        if self.permutation_count < 1:
            raise Error("permutation_count must be positive")
        check_fold_len_multiplier(self.fold_len_multiplier)
        if self.permutation_block_size < 0:
            raise Error("fold_permutation_block must be nonnegative")

    def resolve_block_size(self, n_rows: Int) -> Int:
        """The block size this fit will actually use: the explicit one, or
        `DefaultFoldPermutationBlockSize(n)` when it was left at 0."""
        if self.permutation_block_size > 0:
            var chosen = self.permutation_block_size
            return n_rows if chosen > n_rows and n_rows > 0 else chosen
        return default_permutation_block_size(n_rows)


# ---------------------------------------------------------------------------
# Hessian declaration
# ---------------------------------------------------------------------------


def ordered_varies_hessian(params: OrderedBoostingParams) -> Bool:
    """Whether a fit configured this way has a per-row hessian, so that
    `histogram.CONSTANT_HESSIAN` must not be declared for it.

    **False, always, and this is a checked claim rather than an omission.**
    The twin of `langevin.langevin_varies_hessian`, and it answers the same
    way for a related but distinct reason.

    Ordered boosting changes the *point at which* a row's derivatives are
    evaluated: `hess[r]` is the objective's second derivative at that row's
    ordered raw score instead of at its plain one. It introduces no per-row
    weight and no multiplier. So for the objectives whose unweighted hessian is
    the literal constant 1.0 -- `SQUARED_ERROR`, `L1`, `HUBER`, `QUANTILE` --
    the value written is still exactly 1.0, at any raw score, and the
    two-plane histogram path stays admissible. For every objective whose
    hessian already varies, `boosting.round_has_constant_hessian` already
    returns False without knowing this bundle exists.

    This is the opposite answer from `sampling.mvs_varies_hessian` and the
    Bayesian bootstrap's, and the distinction is exactly the one those two
    docstrings draw: a *weight* that multiplies the derivative is a hessian, a
    *different evaluation point* is not.

    `params` is taken and ignored on purpose. The signature is the one a later
    edit would have to change to make this vary -- adding a per-rung row weight
    to reweight the tail, say, which is the obvious next thing a reader reaches
    for -- and `check_ordered_hessian_declaration` is what would then fire.
    """
    _ = params.enabled
    return False


def check_ordered_hessian_declaration(
    params: OrderedBoostingParams, const_hessian: Bool
) raises:
    """Refuse a constant-hessian declaration beside an ordered fit that varies
    the hessian.

    Today that combination cannot arise, because `ordered_varies_hessian` is
    False by construction. The guard exists anyway and is called by the round
    loop, for the reason `langevin.check_langevin_hessian_declaration` gives:
    installing it afterwards requires noticing, and the failure it prevents --
    a constant hessian declared beside a per-rung gradient -- is silent.
    """
    if ordered_varies_hessian(params) and const_hessian:
        raise Error(
            "a fit with ordered boosting that varies the hessian must not"
            " declare a constant hessian"
        )


def check_ordered_honored(params: OrderedBoostingParams, where: String) raises:
    """Refuse an active `ordered` bundle in a trainer that does not implement
    it, rather than training a plain ensemble and calling it ordered.

    Same policy as `efb.check_bundling_honored`: a knob that is accepted and
    silently does nothing is a wrong answer to the caller who set it.
    """
    if params.enabled:
        raise Error(
            String(
                "boosting_type=ordered is not implemented by ",
                where,
                "; only the dense single-output CPU trainer honors it",
            )
        )
