"""Feature-parallel and voting-parallel strategy cores.

`distributed.mojo` implements one parallel mode, data parallel: rows are
partitioned, histograms are all-reduced, and every rank reaches the same
decision from the same global histogram. LightGBM offers two more, selected
by its `tree_learner` parameter, and section 2 of docs/distributed.md
explains why data parallel came first. This module is the arithmetic those
other two modes need, and nothing else.

What is here is deliberately narrow. These are *cores*: pure functions and
small records over a `Collective`, with no growth loop, no trainer, and no
transport. There is no third copy of the boosting round loop and no second
`Collective` implementation anywhere in this file. A grower that wants
feature-parallel split election calls `elect_split_collective` once per node
the way it already calls `find_best_split` once per node; a grower that wants
voting calls `select_top_k`, `allreduce_votes`, and `elect_voted_features`
and then reduces a smaller histogram. Both seams are one function call wide,
which is what keeps them from becoming a parallel trainer.

**Status. Neither mode is operational, and `require_strategy_operational`
refuses both.** The cores are written and the reasoning below is meant to be
checkable, but no grower in this repository calls them, no test has run them,
and nothing in the repository has moved a byte between two processes: see
`transport_available` in distributed_transport.mojo, which is False in every
build today. No parallel speedup, scaling, or communication measurement is
claimed here or anywhere else.

## The three modes, and what each one costs

**Data parallel** (implemented, in distributed.mojo). Rows partitioned, the
full `n_features * n_bins` histogram all-reduced once per node.

**Feature parallel** (cores here). Features partitioned, *rows not*: every
rank holds the whole binned matrix and every row of it, and builds histograms
only for the features it owns. That is LightGBM's own arrangement, and it is
what removes the row-assignment broadcast the textbook version of this mode
needs: once the winning split is known, every rank routes its own copy of
every row through it without being told which row went where.

Per node the communication is one all-gather of `world_size` split
candidates, `CANDIDATE_WORDS` integers each, against data parallel's
`3 * n_features * n_bins` numbers. The trade is memory: nothing about this
mode makes a dataset that does not fit on one machine fit on two. It pays
when features are the expensive axis, which is the narrow case section 2 of
docs/distributed.md describes.

**Voting parallel** (cores here). Data parallel with a filter: each rank
ranks its own features by local gain, the ranks vote, and only the elected
features have their histograms reduced. Communication drops by roughly
`n_features / n_selected` and *exactness goes with it*: a feature that no
rank ranks highly locally can still be the global winner, and when it is, the
elected set does not contain it and the tree takes a different split.
`voting_is_exact()` returns False in one place, so no caller has to decide
for itself whether this mode agrees with single-node training.

## Determinism, which is the whole reason these are cores and not a loop

Three properties, each pinned by one function here.

**The election reproduces the single-node scan order.** `FeaturePartition`
is contiguous and ascending: rank `r` owns `[r * F // W, (r + 1) * F // W)`.
So ascending rank order *is* ascending feature order, and `elect_split` scans
candidates in ascending rank order keeping one only on a strict gain
improvement, which is exactly what `find_best_split` does over features
within one rank. Given identical histograms, feature-parallel election
therefore selects the identical split a single-node scan would, ties
included. A round-robin or hashed feature partition would break that
argument, so `FeaturePartition` implements one scheme and offers no other.

**The exchange is order independent.** `allgather_candidates` writes each
rank's record into that rank's own slot of a world-sized integer vector and
reduces with `allreduce_max_int`, which is an all-gather when every other
slot is zero and every record word is non-negative. Maximum is commutative
and idempotent, so the result depends only on what the ranks wrote, never on
arrival order, on which rank answered first, or on the reduction tree. Every
encoder here is total and non-negative by construction, which is the
invariant `encode_candidate` enforces before any collective is issued.

**Gains cross as bits.** A gain travels as the two halves of its IEEE-754 bit
pattern, so the comparison every rank makes is over identical numbers rather
than over two roundings of them. Gains are non-negative and finite by
validation, and the bit pattern of a non-negative finite double is monotone in
its value, so the comparison is the same one whether it is made on the bits or
on the doubles.

## Failure semantics

The same two-phase discipline the rest of the distributed code uses: a rank
records a status rather than raising, the statuses reduce, and every rank
raises the same error naming the lowest failing rank. A run where one rank
raises while the others block in a collective it will never call is the
failure this ordering exists to prevent.

Three failures are specific to these modes and are detected rather than
assumed away:

- a rank that never wrote its slot leaves the marker word at zero, so a
  missing contributor is an error and not a silently absent candidate
- a rank that returns a split on a feature it does not own is refused, which
  is what makes the partition load bearing rather than advisory
- ranks that are electing for different tree nodes are refused, which is the
  reordering failure requirement 5 of docs/distributed.md section 5 names,
  observed from above the transport

## What these cores do not decide

Leaf values, the frontier scan, the smaller-child choice, row routing, and
the stopping rule. Under feature parallel every one of those is a local
computation on data every rank already holds, so none of them needs a message
and none of them belongs here. Under voting parallel they are the
data-parallel ones, unchanged, over the histogram that was reduced.

See docs/DISTRIBUTED_STRATEGIES.md for the design, the driver each mode still
needs, and the validation neither has had.
"""

from std.math import isfinite

from .categorical import CAT_BITSET_WORDS, CategoricalSpec, cat_empty
from .collective import (
    STATUS_LAYOUT_MISMATCH,
    STATUS_SHAPE_MISMATCH,
    STATUS_UNSUPPORTED,
    Collective,
    agree_equal_ints,
    agree_status,
    hosts_whole_world,
    zeros_f64,
    zeros_int,
)
from .distributed_transport import (
    digest_ints,
    f64_bits,
    f64_from_bits,
    transport_available,
)
from .histogram import Histogram
from .monotone import OutputBounds
from .split import SplitInfo, find_best_split
from .tree import TreeParams

# ---------------------------------------------------------------------------
# Strategy codes
# ---------------------------------------------------------------------------
#
# The names are LightGBM's `tree_learner` values, so a configuration written
# against LightGBM means here what it means there, or is refused. mojotrees has
# no `tree_learner` parameter today (docs/LIGHTGBM_PARITY.md section 11 records
# that as a difference), and adding one is a decision for whoever owns the
# parameter surface, not for this file.

comptime STRATEGY_SERIAL = 0
"""Single node. Not a distributed mode; accepted so a caller can name it."""

comptime STRATEGY_DATA_PARALLEL = 1
"""Rows partitioned, full histograms reduced. Implemented in distributed.mojo.
"""

comptime STRATEGY_FEATURE_PARALLEL = 2
"""Features partitioned, every rank holding every row. Cores here only."""

comptime STRATEGY_VOTING_PARALLEL = 3
"""Data parallel filtered by a vote over features. Cores here only."""


def strategy_name(strategy: Int) -> String:
    if strategy == STRATEGY_SERIAL:
        return "serial"
    if strategy == STRATEGY_DATA_PARALLEL:
        return "data"
    if strategy == STRATEGY_FEATURE_PARALLEL:
        return "feature"
    if strategy == STRATEGY_VOTING_PARALLEL:
        return "voting"
    return "unknown"


def parse_strategy(text: String) raises -> Int:
    """LightGBM's `tree_learner` spellings, and its aliases for them.

    Refused rather than defaulted: a misspelled learner that silently trains
    serially is the failure this exists to prevent, and it is exactly the
    failure a default would produce.
    """
    if text == "serial":
        return STRATEGY_SERIAL
    if text == "data" or text == "data_parallel":
        return STRATEGY_DATA_PARALLEL
    if text == "feature" or text == "feature_parallel":
        return STRATEGY_FEATURE_PARALLEL
    if text == "voting" or text == "voting_parallel":
        return STRATEGY_VOTING_PARALLEL
    raise Error(
        String(
            "unknown tree_learner '",
            text,
            "'; expected serial, data, feature, or voting",
        )
    )


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


@fieldwise_init
struct StrategyCapabilities(Copyable, Movable):
    """What one mode needs and what it can carry.

    Written down as a record rather than as prose because three callers ask
    the same questions of it: a trainer deciding whether a parameter bundle is
    admissible, a binding reporting what the build can do, and the Dask
    adapter mapping a fit onto a backend's declared capability names. All
    three should get the same answers from one place.

    `implemented` is the only field about this repository rather than about
    the mode. It is False for both modes in this file, and
    `require_strategy_operational` is what enforces it.

    `exact_split_search` means the mode searches every feature at every node,
    so it chooses the split a single-node scan of the same histograms would
    choose. It is a statement about the search and not about floating point:
    data parallel regroups a sum at shard boundaries and so agrees with
    single-node training only to within accumulated rounding, which
    docs/distributed.md section 6 states precisely and this field does not
    restate.
    """

    var strategy: Int
    var implemented: Bool
    var needs_every_row_on_every_rank: Bool
    var reduces_histograms: Bool
    var exact_split_search: Bool
    var supports_categorical: Bool
    var supports_monotone: Bool
    var supports_missing: Bool
    var supports_bagging: Bool
    var supports_feature_fraction: Bool
    var supports_ranking_groups: Bool

    def describe(self) -> String:
        var text = String(strategy_name(self.strategy))
        if self.implemented:
            text += ": implemented"
        else:
            text += ": cores only, no driver"
        if self.exact_split_search:
            text += ", searches every feature"
        else:
            text += ", searches a voted subset"
        return text^


def strategy_capabilities(strategy: Int) raises -> StrategyCapabilities:
    """The capability record for one mode.

    The rows that are not obvious, with their reasons:

    **Feature parallel supports bagging and feature subsampling, and data
    parallel does not.** Both draw from a seeded RNG over *global* row or
    feature indices. Under feature parallel every rank holds every row, so
    every rank reproduces the global draw exactly and no projection onto a
    shard is needed. Under data parallel a shard-local bag has to be a
    deterministic projection of a global draw, which distributed.mojo refuses
    rather than approximates.

    **Feature parallel supports categorical splits, monotone constraints, and
    missing bins, and the data-parallel prototype does not.** Not because the
    mode is cleverer: because `search_owned_features` forwards a node's search
    to the one `find_best_split` the single-node grower and the GPU grower
    already use, so whatever that function enforces is enforced here by
    construction. The data-parallel prototype refuses them because it runs a
    second growth loop that would have to reimplement them
    (docs/distributed.md section 9).

    **Voting parallel does not search every feature.** The elected feature set
    can exclude the globally best feature. That is the mode, not a defect, and
    it is why `voting_is_exact()` exists.

    **Ranking groups.** Feature parallel keeps every row on every rank, so a
    query group is never straddled and `check_group_alignment` in
    distributed.mojo has nothing to refuse. Voting parallel partitions rows
    and inherits the data-parallel constraint exactly.
    """
    if strategy == STRATEGY_SERIAL:
        return StrategyCapabilities(
            STRATEGY_SERIAL,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        )
    if strategy == STRATEGY_DATA_PARALLEL:
        return StrategyCapabilities(
            STRATEGY_DATA_PARALLEL,
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        )
    if strategy == STRATEGY_FEATURE_PARALLEL:
        return StrategyCapabilities(
            STRATEGY_FEATURE_PARALLEL,
            False,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        )
    if strategy == STRATEGY_VOTING_PARALLEL:
        return StrategyCapabilities(
            STRATEGY_VOTING_PARALLEL,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        )
    raise Error(String("unknown parallel strategy code ", strategy))


def voting_is_exact() -> Bool:
    """Whether voting parallel reaches the split data parallel would.

    False, in one place, so that no caller has to decide for itself. The
    elected feature set is chosen from local rankings, and a feature that is
    unremarkable on every shard yet best globally is not in it.
    """
    return False


# ---------------------------------------------------------------------------
# Unsupported states
# ---------------------------------------------------------------------------
#
# Enumerated as a bit mask for the same reason distributed.mojo enumerates its
# own: a mask reduces across ranks, so every rank refuses the same
# configuration with the same message, and a refusal is never a rank-local
# decision.

comptime UNSUPPORTED_NO_DRIVER = 1
comptime UNSUPPORTED_NO_TRANSPORT = 2
comptime UNSUPPORTED_WORLD_TOO_SMALL = 4
comptime UNSUPPORTED_STRATEGY_UNKNOWN = 8
comptime UNSUPPORTED_SERIAL_WORLD = 16


def strategy_unsupported_mask(
    strategy: Int,
    world_size: Int,
    multi_process: Bool,
    transport_ready: Bool,
) -> Int:
    """Every reason this build cannot run `strategy` at this world size.

    All of them at once rather than the first one found, so a caller that is
    two changes away from a working configuration learns both changes from one
    error instead of one per attempt.

    `multi_process` is what separates a world spread over processes from one
    hosted inside this one, and it is why the transport gate is conditional. A
    four-rank world hosted by `LocalCollective` is a legal, working
    configuration today, and refusing it because no socket exists would refuse
    the only distributed training that runs. `hosts_whole_world` in
    collective.mojo is the predicate a caller reads this from, and
    `require_strategy` does exactly that.
    """
    if (
        strategy != STRATEGY_SERIAL
        and strategy != STRATEGY_DATA_PARALLEL
        and strategy != STRATEGY_FEATURE_PARALLEL
        and strategy != STRATEGY_VOTING_PARALLEL
    ):
        return UNSUPPORTED_STRATEGY_UNKNOWN
    var mask = 0
    if strategy == STRATEGY_SERIAL:
        if world_size != 1:
            mask |= UNSUPPORTED_SERIAL_WORLD
        return mask
    if world_size < 2:
        mask |= UNSUPPORTED_WORLD_TOO_SMALL
    if (
        strategy == STRATEGY_FEATURE_PARALLEL
        or strategy == STRATEGY_VOTING_PARALLEL
    ):
        mask |= UNSUPPORTED_NO_DRIVER
    if multi_process and not transport_ready:
        mask |= UNSUPPORTED_NO_TRANSPORT
    return mask


def raise_strategy_unsupported(strategy: Int, mask: Int) raises:
    """Turn a mask into the one error every rank raises.

    Ordered from the most fundamental reason to the least, so a caller reads
    the one that has to be fixed first.
    """
    if mask & UNSUPPORTED_STRATEGY_UNKNOWN != 0:
        raise Error(String("unknown parallel strategy code ", strategy))
    if mask & UNSUPPORTED_NO_DRIVER != 0:
        raise Error(
            String(
                "the ",
                strategy_name(strategy),
                "-parallel strategy is not operational: this module supplies"
                " the split-election and voting cores, and no grower in this"
                " repository calls them. See docs/DISTRIBUTED_STRATEGIES.md"
                " for the driver each mode still needs. Train with"
                " train_distributed (data parallel) or on a single node",
            )
        )
    if mask & UNSUPPORTED_NO_TRANSPORT != 0:
        raise Error(
            "distributed training cannot reach another process in this build:"
            " distributed_transport.transport_available() is False because no"
            " ByteEndpoint that opens a connection exists. A world hosted in"
            " one process (LocalCollective) still runs"
        )
    if mask & UNSUPPORTED_WORLD_TOO_SMALL != 0:
        raise Error(
            "a parallel strategy needs a world size of at least 2; use the"
            " serial strategy for a single rank"
        )
    if mask & UNSUPPORTED_SERIAL_WORLD != 0:
        raise Error(
            "the serial strategy is single-node and cannot be used with a"
            " world size greater than 1"
        )


def require_strategy_operational(
    strategy: Int,
    world_size: Int,
    multi_process: Bool,
    transport_ready: Bool,
) raises:
    """The gate, as a pure function of what it is told.

    Raises for feature and voting parallel unconditionally, and for any mode
    whose world spans processes this build cannot reach. That is deliberate: a
    core that has never been driven, never run across processes, and never
    measured is not a mode a caller should be able to select by accident.
    """
    var mask = strategy_unsupported_mask(
        strategy, world_size, multi_process, transport_ready
    )
    raise_strategy_unsupported(strategy, mask)


def require_strategy[
    C: Collective
](comm: C, strategy: Int) raises:
    """The gate, against the world and the build actually in front of it.

    The two facts the pure form has to be told are read here from the only
    places that know them: whether the world spans processes from
    `hosts_whole_world`, and whether this build can reach another process from
    `transport_available`. A caller cannot then pass a `transport_ready` its
    build does not have, which is the failure mode a boolean parameter invites
    and the reason this wrapper exists rather than being left to the caller.
    """
    require_strategy_operational(
        strategy,
        comm.world_size(),
        not hosts_whole_world(comm),
        transport_available(),
    )


# ---------------------------------------------------------------------------
# Feature partitions
# ---------------------------------------------------------------------------


struct FeaturePartition(Copyable, Movable):
    """Which rank owns which features, as contiguous ascending blocks.

    Rank `r` owns `[r * F // W, (r + 1) * F // W)`, the same block rule
    `ShardPlan.contiguous` uses for rows, and for a related reason. There it
    keeps the reduction visiting rows in the dataset's own order; here it
    keeps ascending rank order equal to ascending feature order, which is what
    makes `elect_split`'s tie-break the same one `find_best_split` applies
    inside a rank. A round-robin partition would balance a ragged per-feature
    cost slightly better and would break that equality, so it is not offered:
    every feature costs the same `n_bins` cells to scan, and there is nothing
    ragged to balance.

    A rank may own zero features when the world is wider than the feature
    count. Such a rank still takes part in every collective, contributing a
    not-found candidate, exactly as an empty shard contributes a zero
    histogram in data-parallel training.
    """

    var n_features: Int
    var world_size: Int

    def __init__(out self, n_features: Int, world_size: Int) raises:
        if world_size < 1:
            raise Error("world_size must be positive")
        if n_features < 0:
            raise Error("n_features must not be negative")
        self.n_features = n_features
        self.world_size = world_size

    def _check_rank(self, rank: Int) raises:
        if rank < 0 or rank >= self.world_size:
            raise Error(
                String(
                    "rank ", rank, " is outside a world of ", self.world_size
                )
            )

    def start(self, rank: Int) raises -> Int:
        self._check_rank(rank)
        return rank * self.n_features // self.world_size

    def end(self, rank: Int) raises -> Int:
        self._check_rank(rank)
        return (rank + 1) * self.n_features // self.world_size

    def count(self, rank: Int) raises -> Int:
        return self.end(rank) - self.start(rank)

    def owns(self, rank: Int, feature: Int) raises -> Bool:
        return feature >= self.start(rank) and feature < self.end(rank)

    def owner(self, feature: Int) raises -> Int:
        """Which rank owns `feature`, or -1 when it is out of range.

        A forward scan rather than an inverted formula: the world is small,
        the blocks are ascending, and one definition of the boundary that
        cannot drift from `start` and `end` is worth more here than the
        arithmetic.
        """
        if feature < 0 or feature >= self.n_features:
            return -1
        for r in range(self.world_size):
            if feature < self.end(r):
                return r
        return -1

    def features(self, rank: Int) raises -> List[Int]:
        """The ascending feature ids `rank` owns, in the shape
        `find_best_split(features=...)` takes.

        Ascending because that function scans the list in the order it is
        given and keeps a candidate only on a strict improvement, so a
        shuffled list would break a tie differently.
        """
        var lo = self.start(rank)
        var hi = self.end(rank)
        var out = List[Int](capacity=hi - lo)
        for f in range(lo, hi):
            out.append(f)
        return out^

    def digest(self) -> UInt64:
        """The partition every rank must agree about, in one integer, so a
        disagreement is one reduced value rather than `world_size` compared
        boundaries."""
        var values: List[Int] = [self.n_features, self.world_size]
        return digest_ints(values)


def intersect_ascending(owned: List[Int], selected: List[Int]) -> List[Int]:
    """The features a rank both owns and had subsampled into this node's set.

    Both inputs are ascending, so this is a merge and not a search, and the
    result stays ascending, which `find_best_split` requires. An empty
    `selected` means every feature was selected, which is that function's own
    convention, so `owned` passes through unchanged rather than emptying it.
    """
    if len(selected) == 0:
        return owned.copy()
    var out = List[Int]()
    var i = 0
    var j = 0
    while i < len(owned) and j < len(selected):
        if owned[i] == selected[j]:
            out.append(owned[i])
            i += 1
            j += 1
        elif owned[i] < selected[j]:
            i += 1
        else:
            j += 1
    return out^


# ---------------------------------------------------------------------------
# Split candidates on the wire
# ---------------------------------------------------------------------------
#
# A candidate is `CANDIDATE_WORDS` non-negative integers. Non-negative because
# the all-gather is built on `allreduce_max_int` over slots the other ranks
# leave at zero, and a negative word would then reduce to another rank's zero
# instead of to itself. Every field that can legitimately be negative is stored
# with an offset, and every 64-bit quantity is stored as two 32-bit halves, so
# no word can reach the sign bit of an `Int`.

comptime CANDIDATE_WORDS = 8 + 2 * CAT_BITSET_WORDS

comptime _W_MARK = 0
"""1 when a rank wrote this block. Zero means a rank never contributed, which
is a broken exchange and not an absent candidate."""

comptime _W_FOUND = 1
comptime _W_FEATURE = 2
"""`feature + 1`, so the -1 of a not-found split stores as 0."""

comptime _W_BIN = 3
"""`bin + 1`, likewise; a categorical split stores 0 here."""

comptime _W_FLAGS = 4
"""Bit 0 categorical, bit 1 default_left."""

comptime _W_GAIN_LO = 5
comptime _W_GAIN_HI = 6
comptime _W_NODE = 7
"""`node + 1`, so a zero marks a block nobody wrote."""

comptime _W_CATS = 8
"""The first of `2 * CAT_BITSET_WORDS` halves of the categorical bitset."""


def encode_candidate(split: SplitInfo, node: Int) raises -> List[Int]:
    """One rank's answer for one node, as `CANDIDATE_WORDS` non-negative
    integers.

    Every validation here happens *before* the collective it feeds, and every
    one of them is a pure function of a local candidate that a correct search
    cannot produce. That ordering matters: a raise from inside this function
    is a raise on one rank while the others wait, so the only failures it may
    have are ones a correct caller cannot reach. A gain that is negative or
    not finite is such a failure; `find_best_split` returns a found split only
    on a strictly positive finite gain.
    """
    if node < 0:
        raise Error("a node id must not be negative")
    var out = zeros_int(CANDIDATE_WORDS)
    out[_W_MARK] = 1
    out[_W_NODE] = node + 1
    if not split.found:
        return out^

    if split.gain < 0.0 or not isfinite(split.gain):
        raise Error(
            "a found split must carry a non-negative finite gain; the"
            " candidate exchange compares gains as bit patterns, and a"
            " negative, infinite, or NaN gain does not order"
        )
    if split.feature < 0:
        raise Error("a found split must name a feature")
    if split.is_categorical:
        if split.bin != -1:
            raise Error("a categorical split does not carry a threshold bin")
    elif split.bin < 0:
        raise Error("a numerical split must carry a threshold bin")

    out[_W_FOUND] = 1
    out[_W_FEATURE] = split.feature + 1
    out[_W_BIN] = split.bin + 1
    var flags = 0
    if split.is_categorical:
        flags |= 1
    if split.default_left:
        flags |= 2
    out[_W_FLAGS] = flags
    var bits = f64_bits(split.gain)
    out[_W_GAIN_LO] = Int(bits % 0x1_0000_0000)
    out[_W_GAIN_HI] = Int(bits // 0x1_0000_0000)
    if split.is_categorical:
        for w in range(CAT_BITSET_WORDS):
            var word = split.cat_bitset[w]
            out[_W_CATS + 2 * w] = Int(word % 0x1_0000_0000)
            out[_W_CATS + 2 * w + 1] = Int(word // 0x1_0000_0000)
    return out^


@fieldwise_init
struct Candidate(Copyable, Movable):
    """One rank's decoded answer.

    `present` is False only for a rank that never wrote its slot, which is a
    broken exchange rather than a rank with nothing to say: a rank with
    nothing to say writes a not-found split, which is present and not found.
    """

    var rank: Int
    var node: Int
    var split: SplitInfo
    var present: Bool


def decode_candidate(
    words: List[Int], offset: Int, rank: Int
) raises -> Candidate:
    """Decode one rank's block.

    Every field is checked against the encoding's own invariants, because
    after a reduction the words are as trustworthy as the least trustworthy
    rank that contributed to them.
    """
    if offset < 0 or offset + CANDIDATE_WORDS > len(words):
        raise Error("a candidate block runs past the end of the buffer")
    for w in range(CANDIDATE_WORDS):
        if words[offset + w] < 0:
            raise Error(
                "a candidate word is negative; the exchange reduces with a"
                " maximum and every word must be non-negative"
            )
    if words[offset + _W_MARK] == 0:
        return Candidate(rank, -1, SplitInfo(-1, -1, 0.0, False), False)
    if words[offset + _W_MARK] != 1:
        raise Error("a candidate marker word must be 0 or 1")

    var node = words[offset + _W_NODE] - 1
    if node < 0:
        raise Error("a marked candidate must carry a node id")
    if words[offset + _W_FOUND] == 0:
        return Candidate(rank, node, SplitInfo(-1, -1, 0.0, False), True)

    var flags = words[offset + _W_FLAGS]
    if flags > 3:
        raise Error("a candidate carries unknown split flags")
    var is_cat = (flags & 1) != 0
    var default_left = (flags & 2) != 0
    var bits = UInt64(words[offset + _W_GAIN_HI]) * 0x1_0000_0000 + UInt64(
        words[offset + _W_GAIN_LO]
    )
    var gain = f64_from_bits(bits)
    if gain < 0.0 or not isfinite(gain):
        raise Error("a candidate carries a negative, infinite, or NaN gain")
    var feature = words[offset + _W_FEATURE] - 1
    if feature < 0:
        raise Error("a found candidate must name a feature")

    if is_cat:
        if words[offset + _W_BIN] != 0:
            raise Error("a categorical candidate must not carry a bin")
        var bitset = cat_empty()
        for w in range(CAT_BITSET_WORDS):
            var hi = UInt64(words[offset + _W_CATS + 2 * w + 1])
            var lo = UInt64(words[offset + _W_CATS + 2 * w])
            bitset[w] = hi * 0x1_0000_0000 + lo
        return Candidate(
            rank, node, SplitInfo.categorical(feature, gain, bitset), True
        )

    var bin = words[offset + _W_BIN] - 1
    if bin < 0:
        raise Error("a numerical candidate must carry a threshold bin")
    return Candidate(
        rank, node, SplitInfo(feature, bin, gain, True, default_left), True
    )


def allgather_candidates[
    C: Collective
](mut comm: C, local: List[SplitInfo], node: Int) raises -> List[Candidate]:
    """Every rank's candidate for one node, on every rank.

    `local` holds this process's local ranks' candidates, one per
    `comm.n_local_ranks()`, which is the shape `agree_status` takes and which
    degenerates to a single entry under a one-rank-per-process transport.

    One collective, and the collective is a maximum over a vector in which
    each rank wrote only its own slot. That is an all-gather built from a
    reduction `Collective` already has, rather than a new trait method every
    future transport would have to implement. It is skipped when this process
    already hosts the whole world, because then there is no peer whose slots
    could differ from the zeros this process wrote.
    """
    var world = comm.world_size()
    var n_local = comm.n_local_ranks()
    if len(local) != n_local:
        raise Error("allgather_candidates needs one candidate per local rank")

    var buf = zeros_int(world * CANDIDATE_WORDS)
    for i in range(n_local):
        var r = comm.local_rank(i)
        if r < 0 or r >= world:
            raise Error("local rank id out of range")
        var block = encode_candidate(local[i], node)
        var base = r * CANDIDATE_WORDS
        if buf[base + _W_MARK] != 0:
            raise Error(
                String(
                    "two local ranks both claim rank ",
                    r,
                    " in the candidate exchange",
                )
            )
        for w in range(CANDIDATE_WORDS):
            buf[base + w] = block[w]

    if not hosts_whole_world(comm):
        comm.allreduce_max_int(buf)

    var out = List[Candidate](capacity=world)
    for r in range(world):
        out.append(decode_candidate(buf, r * CANDIDATE_WORDS, r))
    return out^


# ---------------------------------------------------------------------------
# Election
# ---------------------------------------------------------------------------


@fieldwise_init
struct ElectedSplit(Copyable, Movable):
    """The winning candidate and who proposed it.

    `owner` is carried because it is free and because it is the only part a
    caller cannot recompute: everything else about the decision is a pure
    function of the gathered candidates, which every rank holds identically.
    It is -1 when no rank found a split, which is how a node becomes a leaf.
    """

    var split: SplitInfo
    var owner: Int

    def is_found(self) -> Bool:
        return self.split.found


def elect_split(
    candidates: List[Candidate], partition: FeaturePartition, node: Int
) raises -> ElectedSplit:
    """The global best split, chosen identically on every rank.

    Scans ascending rank order and replaces the running best only on a strict
    gain improvement. Because `FeaturePartition` gives rank `r` a contiguous
    ascending feature block, that scan visits features in ascending order, so
    this is the identical rule `find_best_split` applies within one rank and
    the identical winner a single-node scan of the same histograms would pick,
    ties included.

    Three failures are checked rather than assumed, and all three are pure
    functions of the gathered vector, so every rank raises together:

    - a rank whose block was never written, which is a contributor that did
      not contribute
    - a rank that answered for a different node, which is a collective that
      overtook its predecessor, seen from above the transport
    - a rank that returned a split on a feature it does not own, which would
      silently double-count one feature and leave another unsearched
    """
    if len(candidates) != partition.world_size:
        raise Error(
            "the candidate vector must hold one entry per rank of the"
            " partition's world"
        )
    var best = SplitInfo(-1, -1, 0.0, False)
    var owner = -1
    for r in range(len(candidates)):
        var c = candidates[r]
        if c.rank != r:
            raise Error("candidates must be in ascending rank order")
        if not c.present:
            raise Error(
                String(
                    "rank ",
                    r,
                    " contributed no candidate for node ",
                    node,
                    "; every rank answers every election",
                )
            )
        if c.node != node:
            raise Error(
                String(
                    "rank ",
                    r,
                    " answered for node ",
                    c.node,
                    " while this election is for node ",
                    node,
                )
            )
        if not c.split.found:
            continue
        if not partition.owns(r, c.split.feature):
            raise Error(
                String(
                    "rank ",
                    r,
                    " proposed a split on feature ",
                    c.split.feature,
                    ", which it does not own",
                )
            )
        # `best.gain` starts at 0.0 and a found split's gain is strictly
        # positive, so this one test is both the improvement rule and the
        # positive-gain floor, exactly as in `find_best_split`.
        if c.split.gain > best.gain:
            best = c.split.copy()
            owner = r
    return ElectedSplit(best^, owner)


def elect_split_collective[
    C: Collective
](
    mut comm: C,
    local: List[SplitInfo],
    partition: FeaturePartition,
    node: Int,
) raises -> ElectedSplit:
    """One node's whole feature-parallel exchange: gather, then elect.

    This is the entire seam a feature-parallel grower needs. It stands where
    the `find_best_split` call stands in a single-node grower, costs one
    collective, and returns a split every rank agrees on. Nothing else about
    the grower changes, which is the argument for building the mode this way
    rather than as a third growth loop.
    """
    if partition.world_size != comm.world_size():
        raise Error(
            "the feature partition and the collective describe different"
            " worlds"
        )
    var gathered = allgather_candidates(comm, local, node)
    return elect_split(gathered, partition, node)


def search_owned_features(
    hist: Histogram,
    partition: FeaturePartition,
    rank: Int,
    params: TreeParams,
    n_rows: Int,
    allowed: List[Bool] = [],
    selected: List[Int] = [],
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
    bounds: OutputBounds = OutputBounds.unbounded(),
    cats: CategoricalSpec = CategoricalSpec.none(),
    depth: Int = 0,
    node: Int = 0,
    tree_index: Int = 0,
    parent_output: Float64 = 0.0,
) raises -> SplitInfo:
    """This rank's best split over the features it owns.

    A thin forward to the one `find_best_split` the single-node grower and the
    GPU grower already use, narrowed to this rank's features. It is a forward
    and not a reimplementation on purpose: monotone constraints, categorical
    partitions, missing-value routing, the interaction allow mask, the CEGB
    cost, and the gain floor are then enforced here by construction rather
    than by a second copy that can drift. That is the difference between these
    cores and the data-parallel prototype's second growth loop, and it is why
    `strategy_capabilities` can claim those rows for this mode.

    The two guards ahead of the scan are the ones `tree._search` applies, kept
    here because a rank that skipped them would propose a split its peers
    would have refused, and the election cannot tell the difference.

    `params.extra.needs_grower_support()` is refused rather than forwarded:
    `extra_trees`, `max_delta_step`, and `path_smooth` are applied by the
    grower, and there is no feature-parallel grower to be trusted with them.
    """
    partition._check_rank(rank)
    if hist.n_features != partition.n_features:
        raise Error(
            "the histogram and the feature partition disagree about the"
            " feature count"
        )
    if params.extra.needs_grower_support():
        raise Error(
            "extra_trees, max_delta_step, and path_smooth are applied by the"
            " grower, and no feature-parallel grower exists to apply them"
        )
    if params.max_depth > 0 and depth >= params.max_depth:
        return SplitInfo(-1, -1, 0.0, False)
    if n_rows < 2 * params.min_data_in_leaf or n_rows < 2:
        return SplitInfo(-1, -1, 0.0, False)

    var owned = intersect_ascending(partition.features(rank), selected)
    if len(owned) == 0:
        # A rank that owns nothing this node may search still answers the
        # election, with a not-found candidate. Returning here also avoids
        # `find_best_split`'s empty-list convention, under which an empty
        # feature list means every feature rather than none.
        return SplitInfo(-1, -1, 0.0, False)

    return find_best_split(
        hist,
        lambda_reg=params.lambda_reg,
        min_child_hess=params.min_child_hess,
        min_data_in_leaf=params.min_data_in_leaf,
        lambda_l1=params.lambda_l1,
        allowed=allowed,
        features=owned,
        missing_bins=missing_bins,
        monotone=monotone,
        bounds=bounds,
        cats=cats,
        cat_params=params.cat,
        extra=params.extra,
        n_rows=n_rows,
        depth=depth,
        node=node,
        tree_index=tree_index,
        parent_output=parent_output,
    )


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

comptime DEFAULT_TOP_K = 20
"""LightGBM's `top_k` default, kept so a configuration written against it
means the same number of votes here."""


def check_top_k(k: Int) raises:
    if k < 1:
        raise Error("top_k must be at least 1")


def select_top_k(
    gains: List[Float64], found: List[Bool], k: Int
) raises -> List[Int]:
    """This rank's `k` best features by local gain, ascending by feature id.

    `gains[f]` is the best gain this rank found on feature `f` and `found[f]`
    whether it found one at all. Selection is by repeated maximum scan rather
    than by a sort: `k` is 20 by default against a feature count in the
    hundreds or thousands, the scan is exact, and it needs no comparator whose
    tie-break could differ from the one this file argues about everywhere
    else. Ties go to the lower feature id, which is `find_best_split`'s rule.

    The result is ascending because it is a feature id list, and every
    consumer of one here treats ascending order as part of the type.
    """
    check_top_k(k)
    if len(gains) != len(found):
        raise Error("gains and found must describe the same features")
    var n = len(gains)
    var taken = List[Bool](capacity=n)
    taken.resize(n, False)
    for _ in range(k):
        var best = -1
        var best_gain = 0.0
        for f in range(n):
            if taken[f] or not found[f] or gains[f] <= 0.0:
                continue
            if gains[f] > best_gain:
                best = f
                best_gain = gains[f]
        if best < 0:
            break
        taken[best] = True
    var out = List[Int]()
    for f in range(n):
        if taken[f]:
            out.append(f)
    return out^


def allreduce_votes[
    C: Collective
](
    mut comm: C, local_votes: List[List[Int]], n_features: Int
) raises -> List[Int]:
    """How many ranks voted for each feature.

    `local_votes[i]` is the ascending feature list local rank `i` selected.
    One integer reduction over `n_features` counts, exact at any world size
    and independent of arrival order, costing `n_features` integers against
    the `3 * n_features * n_bins` numbers a full histogram reduction costs.
    That ratio is the mode.

    LightGBM's voting is richer than a count: it aggregates the local gains as
    well and runs a second local pass over the merged candidates. This counts
    votes only and breaks ties by ascending feature id. It is a different
    selection rule from LightGBM's and therefore a different model; see
    docs/DISTRIBUTED_STRATEGIES.md.
    """
    if n_features < 0:
        raise Error("n_features must not be negative")
    if len(local_votes) != comm.n_local_ranks():
        raise Error("allreduce_votes needs one vote list per local rank")
    var counts = zeros_int(n_features)
    for i in range(len(local_votes)):
        var votes = local_votes[i]
        var previous = -1
        for j in range(len(votes)):
            var f = votes[j]
            if f < 0 or f >= n_features:
                raise Error("a vote names a feature outside the dataset")
            if f <= previous:
                raise Error("a vote list must be ascending and distinct")
            previous = f
            counts[f] += 1
    if not hosts_whole_world(comm):
        comm.allreduce_sum_int(counts)
    return counts^


def elect_voted_features(votes: List[Int], n_select: Int) raises -> List[Int]:
    """The features whose histograms will be reduced, ascending.

    By vote count descending, ties by ascending feature id, which makes the
    elected set a pure function of the reduced vote vector and therefore
    identical on every rank without another message. A feature nobody voted
    for is never elected, so the set is smaller than `n_select` when fewer
    features drew a vote. That is correct: reducing a feature no rank believes
    can split is pure cost.
    """
    if n_select < 1:
        raise Error("n_select must be at least 1")
    var n = len(votes)
    var taken = List[Bool](capacity=n)
    taken.resize(n, False)
    for _ in range(n_select):
        var best = -1
        var best_votes = 0
        for f in range(n):
            if taken[f] or votes[f] <= 0:
                continue
            if votes[f] > best_votes:
                best = f
                best_votes = votes[f]
        if best < 0:
            break
        taken[best] = True
    var out = List[Int]()
    for f in range(n):
        if taken[f]:
            out.append(f)
    return out^


struct PackedHistogram(Copyable, Movable):
    """The elected features' histogram cells, contiguous and reducible.

    Packed rather than reduced in place because a reduction over the full
    histogram buffer would move exactly the bytes this mode exists not to
    move. The packing order is the elected list's order, which is ascending,
    so unpacking needs no key beyond the same list.
    """

    var grad: List[Float64]
    var hess: List[Float64]
    var count: List[Int]
    var n_selected: Int
    var n_bins: Int

    def __init__(out self, n_selected: Int, n_bins: Int) raises:
        if n_selected < 0 or n_bins < 1:
            raise Error("a packed histogram needs at least one bin")
        var cells = n_selected * n_bins
        self.grad = zeros_f64(cells)
        self.hess = zeros_f64(cells)
        self.count = zeros_int(cells)
        self.n_selected = n_selected
        self.n_bins = n_bins

    def cells(self) -> Int:
        return self.n_selected * self.n_bins


def voting_payload_cells(n_selected: Int, n_bins: Int) raises -> Int:
    """Histogram cells one voting-parallel node reduces.

    The whole point of the mode in one number: `n_selected * n_bins` against
    data parallel's `n_features * n_bins`, at the cost of exactness.
    """
    if n_selected < 0 or n_bins < 1:
        raise Error("a voting payload needs a bin count of at least 1")
    return n_selected * n_bins


def pack_selected(
    hist: Histogram, selected: List[Int]
) raises -> PackedHistogram:
    """Copy the elected features' cells out of a local histogram."""
    var packed = PackedHistogram(len(selected), hist.n_bins)
    var n_bins = hist.n_bins
    var previous = -1
    for i in range(len(selected)):
        var f = selected[i]
        if f < 0 or f >= hist.n_features:
            raise Error("a selected feature is outside the histogram")
        if f <= previous:
            raise Error("the selected feature list must be ascending")
        previous = f
        var src = f * n_bins
        var dst = i * n_bins
        for b in range(n_bins):
            packed.grad[dst + b] = hist.grad[src + b]
            packed.hess[dst + b] = hist.hess[src + b]
            packed.count[dst + b] = hist.count[src + b]
    return packed^


def unpack_selected(
    mut hist: Histogram, selected: List[Int], packed: PackedHistogram
) raises:
    """Write reduced cells back, leaving every unelected feature at zero.

    Zero and not stale: `find_best_split` reads an unelected feature's slice
    only if it is asked to scan it, and a caller that passes the elected list
    as `features` never asks. Zeroing the rest rather than leaving it means a
    caller that forgets sees a feature with no rows in it instead of another
    node's statistics.
    """
    if packed.n_selected != len(selected) or packed.n_bins != hist.n_bins:
        raise Error(
            "the packed histogram does not match the selected feature list"
        )
    for i in range(len(hist.grad)):
        hist.grad[i] = 0.0
        hist.hess[i] = 0.0
        hist.count[i] = 0
    var n_bins = hist.n_bins
    for i in range(len(selected)):
        var f = selected[i]
        if f < 0 or f >= hist.n_features:
            raise Error("a selected feature is outside the histogram")
        var dst = f * n_bins
        var src = i * n_bins
        for b in range(n_bins):
            hist.grad[dst + b] = packed.grad[src + b]
            hist.hess[dst + b] = packed.hess[src + b]
            hist.count[dst + b] = packed.count[src + b]


def allreduce_selected[
    C: Collective
](mut comm: C, mut packed: PackedHistogram) raises:
    """Reduce one voting-parallel node's packed histogram.

    Three reductions, matching `allreduce_histogram` in distributed.mojo
    exactly so the two modes have one communication shape between them, and
    for the same reason: counts stay integers and their exactness stays
    obvious. Packing all three into one buffer is the optimization section 8
    of docs/distributed.md describes for data parallel, and it is deferred
    here for the same reason it is deferred there.
    """
    comm.allreduce_sum_f64(packed.grad)
    comm.allreduce_sum_f64(packed.hess)
    comm.allreduce_sum_int(packed.count)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@fieldwise_init
struct StrategyCostPlan(Copyable, Movable):
    """What one tree node costs on the wire under one mode.

    Computed rather than asserted, so a test can pin the comparison between
    modes against these functions instead of against a comment, exactly as
    `histogram_plan` in distributed_transport.mojo does for data parallel.
    The byte counts are payload only: framing is the transport's, and
    `HistogramPlan.framing_bytes_per_node` is where it is accounted.
    """

    var strategy: Int
    var world_size: Int
    var reduces_per_node: Int
    var f64_elements_per_node: Int
    var int_elements_per_node: Int
    var payload_bytes_per_node: Int


def strategy_cost_plan(
    strategy: Int,
    world_size: Int,
    n_features: Int,
    n_bins: Int,
    n_selected: Int = 0,
) raises -> StrategyCostPlan:
    """The per-node communication of one mode.

    Data parallel is three reductions of `n_features * n_bins`, which is what
    `histogram_plan` already counts and is repeated here only so the modes can
    be compared in one place. Feature parallel is one reduction of
    `world_size * CANDIDATE_WORDS` integers, independent of the bin count and
    of the feature count. Voting parallel is data parallel over the elected
    features, plus the one `n_features` vote reduction that elected them.

    `n_selected` is read only for voting parallel, where 0 means the
    `DEFAULT_TOP_K` LightGBM default.
    """
    if world_size < 1:
        raise Error("world_size must be positive")
    if n_features < 1 or n_bins < 1:
        raise Error("a cost plan needs at least one feature and one bin")
    if strategy == STRATEGY_DATA_PARALLEL:
        var cells = n_features * n_bins
        return StrategyCostPlan(
            strategy, world_size, 3, 2 * cells, cells, 3 * cells * 8
        )
    if strategy == STRATEGY_FEATURE_PARALLEL:
        var words = world_size * CANDIDATE_WORDS
        return StrategyCostPlan(strategy, world_size, 1, 0, words, words * 8)
    if strategy == STRATEGY_VOTING_PARALLEL:
        var selected = n_selected if n_selected > 0 else DEFAULT_TOP_K
        if selected > n_features:
            selected = n_features
        var voted_cells = selected * n_bins
        return StrategyCostPlan(
            strategy,
            world_size,
            4,
            2 * voted_cells,
            voted_cells + n_features,
            3 * voted_cells * 8 + n_features * 8,
        )
    if strategy == STRATEGY_SERIAL:
        return StrategyCostPlan(strategy, world_size, 0, 0, 0, 0)
    raise Error(String("unknown parallel strategy code ", strategy))


# ---------------------------------------------------------------------------
# Cross-rank agreement
# ---------------------------------------------------------------------------


def agree_strategy[
    C: Collective
](
    mut comm: C, strategy: Int, partition: FeaturePartition, top_k: Int
) raises:
    """Refuse a world whose ranks were configured differently.

    One collective, over the values that decide what every rank is about to
    do. A rank started with a different strategy, a different feature count,
    or a different `top_k` would search a different set and vote a different
    number of times, and the resulting model would be neither of the two
    configurations. The partition digest rides along so a field added to the
    partition later is covered without another reduction.
    """
    var digest = partition.digest()
    var values: List[Int] = [
        strategy,
        partition.n_features,
        partition.world_size,
        top_k,
        Int(digest % 0x1_0000_0000),
        Int(digest // 0x1_0000_0000),
    ]
    var names: List[String] = [
        "the parallel strategy",
        "the feature count",
        "the world size",
        "top_k",
        "the feature partition",
        "the feature partition",
    ]
    var bad = agree_equal_ints(comm, values)
    if bad >= 0:
        raise Error(
            String("distributed training: ranks disagree about ", names[bad])
        )


def strategy_statuses[
    C: Collective
](
    comm: C, strategy: Int, partition: FeaturePartition, n_features: Int
) -> List[Int]:
    """Per-local-rank status for a strategy's preconditions.

    Recorded rather than raised, so `agree_status` can turn them into one
    error every rank raises together. This is the shape every validation in
    distributed.mojo takes, and it is why a bad configuration stops the world
    instead of hanging it.
    """
    var n_local = comm.n_local_ranks()
    var statuses = zeros_int(n_local)
    for i in range(n_local):
        var r = comm.local_rank(i)
        if r < 0 or r >= comm.world_size():
            statuses[i] = STATUS_LAYOUT_MISMATCH
        elif partition.world_size != comm.world_size():
            statuses[i] = STATUS_LAYOUT_MISMATCH
        elif partition.n_features != n_features:
            statuses[i] = STATUS_SHAPE_MISMATCH
        elif (
            strategy != STRATEGY_FEATURE_PARALLEL
            and strategy != STRATEGY_VOTING_PARALLEL
        ):
            statuses[i] = STATUS_UNSUPPORTED
    return statuses^


def check_strategy_world[
    C: Collective
](
    mut comm: C,
    strategy: Int,
    partition: FeaturePartition,
    n_features: Int,
    top_k: Int,
) raises:
    """Everything that has to be true before the first election.

    The gate first, then two collectives, both once per run and neither per
    node: the status agreement and the configuration agreement. The gate is
    ahead of both because its inputs are the strategy, the world size, and two
    facts about the build, all of which are identical on every rank, so it
    raises everywhere or nowhere and no rank is left in a collective a refused
    peer will never call. The two agreements are then in the order
    `grow_tree_distributed` uses, statuses first, so a rank whose own state is
    broken is reported as broken rather than as disagreeing.

    This function is therefore the only entry point a driver needs, and it is
    the one that will start raising less as the modes become real.
    """
    require_strategy(comm, strategy)
    var statuses = strategy_statuses(comm, strategy, partition, n_features)
    agree_status(comm, statuses)
    agree_strategy(comm, strategy, partition, top_k)
