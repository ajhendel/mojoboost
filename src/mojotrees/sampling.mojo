"""Deterministic feature subsampling.

Two fractions control how many features a tree may split on, matching
LightGBM's parameters of the same names:

- `feature_fraction` (LightGBM's colsample_bytree): the share of all
  features sampled once per tree. Every node of that tree searches within
  that one set.
- `feature_fraction_bynode` (colsample_bynode): the share sampled again at
  every node, drawn from the tree's set. Applied on top of
  `feature_fraction`, so the effective per-node share is the product.

Both are sampled without replacement and are reproducible from
`feature_fraction_seed`.

Selection count follows LightGBM's `ColSampler::GetCnt`: round(total *
fraction), floored at 2 features (or at `total` when a node has fewer than
2 to choose from). The subset itself is drawn with LightGBM's selection
algorithm (Knuth's Algorithm S), which walks the candidate list once and
keeps candidate i with probability (remaining picks) / (remaining
candidates), so the result is a uniformly random subset already in
ascending feature order.

INTENTIONAL DIFFERENCE FROM LightGBM: LightGBM draws from a single
linear-congruential stream that advances as training proceeds, so a tree's
feature set depends on how many draws happened before it. mojotrees derives
an independent counter-based splitmix64 stream per (seed, tree index, node
id) instead. Selections are therefore reproducible per tree and per node
regardless of history, thread count, or which trees were grown, at the cost
of not reproducing LightGBM's exact subsets for a given seed. The
distribution and the per-selection algorithm are the same.

A second difference shows up only with interaction constraints: LightGBM
draws the per-node set from the features the branch already allows, while
these draws come from the tree's set and the allow mask is applied
afterwards, in the split search. The candidates are drawn from the same
pool either way; a node can simply end up with fewer of them here.

The prefiltered pool
--------------------
`select_tree_features` takes the pool as an argument rather than building it
from `n_features`, because `feature_pre_filter` deletes features. LightGBM's
`ColSampler` samples `Dataset::ValidFeatureIndices()` -- the features that
survived Dataset construction -- and sizes the draw by *their* count
(`GetCnt(valid_feature_indices_.size(), fraction)`,
`src/treelearner/col_sampler.hpp`), so a prefiltered fit both never draws a
dropped feature and draws a fraction of the survivors rather than of
everything. `binning.fit_bins(feature_pre_filter=True)` produces the list and
`BinnedMatrix.usable_features()` carries it to a grower. An empty pool is the
default and means nothing was filtered, so it is the draw this module has
always made, from the same counter stream.

Per-level selection
-------------------
`feature_fraction_bylevel` draws a third set once per depth, sitting between
the tree's set and the node's. It is XGBoost's `colsample_bylevel`; LightGBM
has no equivalent, so this is an extension rather than a parity row.
mojotrees grows leaf-wise, so a "level" is a node's depth in edges from the
root: every node at the same depth of the same tree shares one draw, and the
per-node draw is then taken from it. The three fractions multiply, so a
node's effective share of the feature set is `feature_fraction *
feature_fraction_bylevel * feature_fraction_bynode` (each rounded by
`selection_count` in turn). A bylevel fraction of 1.0 passes the tree's set
through untouched, which is what keeps every selection made today bit
identical.

Parameter names
---------------
`canonical_sampling_param` maps every LightGBM, XGBoost, and scikit-learn
spelling of a row or feature sampling parameter onto the mojotrees name,
including `colsample_bytree` and `colsample_bynode`, which the parity audit
records as spellings mojotrees does not yet accept. The table here is meant
to be the single place those aliases are written down, so the Python layer,
the CLI, and the C API resolve through it instead of each keeping a list.

Class-conditional row bagging
-----------------------------
`pos_bagging_fraction` / `neg_bagging_fraction` keep positive and negative
rows at different rates, LightGBM's balanced bagging for unbalanced binary
data. The draw is the one uniform bagging makes (bagging.mojo), with the
per-row keep probability chosen by the row's label instead of being shared,
so setting both fractions to the same value reproduces a uniform bag row for
row. Rows are positive when `label > 0`, so 0/1 and -1/+1 labels both work.

This matches LightGBM's `BalancedBaggingHelper` (src/boosting/bagging.hpp)
decision for decision: one draw per row whichever class it belongs to,
compared against that class's fraction, and `label > 0` as the class test.
The one condition that cannot live in `ClassBaggingParams` is LightGBM's
requirement that the dataset hold a positive row at all, since that needs the
labels; see `has_positive_rows`.

Bayesian bootstrap
------------------
CatBoost's `bootstrap_type=Bayesian` is the one row sampler here that does
not select rows at all: every row is kept and every row is given a random
weight, redrawn once per tree, controlled by `bagging_temperature`. The draw
is transcribed from `catboost/private/libs/algo/tensor_search_helpers.cpp`
(`GenerateBayessianWeight`, `GenerateRandomWeights`, `CalcWeightedData`,
`Bootstrap`); see `bayesian_bootstrap_weight` for the formula and
`BayesianBootstrapParams` for the option semantics.

**Its weights are hessians.** Because a bootstrap weight multiplies the row's
derivatives exactly as a sample weight does, a fit with this on has a per-row
hessian and must NOT declare `histogram.CONSTANT_HESSIAN`.
`bayesian_bootstrap_varies_hessian` states that, and
`check_bayesian_bootstrap_hessian_declaration` refuses the combination
outright, so a caller that wires this into a round loop cannot take the
two-plane path by omission. See `boosting.round_has_constant_hessian`.

Row sets and the GPU
--------------------
A sampled row set is an ascending, duplicate-free list of row indices, and
an empty list means "every row" everywhere in mojotrees. `contiguous_ranges`
turns such a list into half-open `[start, end)` ranges and `row_mask` into a
dense mask, which is what a device-side active-row pass consumes;
`expand_row_scale` turns a (rows, scale) pair such as GOSS produces into a
dense per-row multiplier that is zero off the sample, so a kernel that
multiplies gradients by it reproduces the sampled histogram without gathering
rows first.
"""

from std.math import log

from .bagging import DEFAULT_BAGGING_SEED
from .rng import GOLDEN, splitmix64, uniform

# LightGBM's feature_fraction_seed default.
comptime DEFAULT_FEATURE_FRACTION_SEED = 2

# No LightGBM equivalent; 1.0 leaves the tree's feature set alone.
comptime DEFAULT_FEATURE_FRACTION_BYLEVEL = 1.0

# LightGBM's pos_bagging_fraction / neg_bagging_fraction defaults, which
# together mean "no class-conditional bagging".
comptime DEFAULT_POS_BAGGING_FRACTION = 1.0
comptime DEFAULT_NEG_BAGGING_FRACTION = 1.0

# Distinguishes the once-per-tree draw from the per-node draws, which are
# tagged with node id + 1.
comptime _TREE_TAG = 0

# Separates the per-depth streams from the per-tree and per-node ones, so a
# depth can never inherit the stream a node id already owns.
comptime _LEVEL_DOMAIN = UInt64(0xA5A55A5AC3C33C3C)

# CatBoost's `bagging_temperature` default: `bootstrap_options.cpp` constructs
# `BaggingTemperature("bagging_temperature", 1.0)`. At 1.0 the weights are
# standard exponentials, mean 1.
comptime DEFAULT_BAGGING_TEMPERATURE = 1.0

# CatBoost draws the bootstrap from the fit's global `random_seed`, whose
# default is 0. mojotrees gives the bootstrap its own seed so it composes with
# the feature and bagging seeds instead of sharing one counter, and 0 is kept
# as the value so a port of a default CatBoost configuration reads the same.
comptime DEFAULT_BOOTSTRAP_SEED = 0

# `bootstrap_type` values this module implements.
comptime BOOTSTRAP_NO = 0
comptime BOOTSTRAP_BAYESIAN = 1

# CatBoost's own guard against log(0), verbatim: `GenerateBayessianWeight`
# computes `-FastLogf(rand.GenRandReal1() + 1e-100)`. Its `GenRandReal1` is
# closed on both ends, so a draw of exactly 0 is possible there; `rng.uniform`
# is half-open [0, 1) and can also return 0. The epsilon is carried across
# rather than dropped so the largest weight either library can produce is the
# same finite number, `(-log(1e-100)) ** t`, instead of an infinity here.
comptime _BAYESIAN_LOG_EPS = 1e-100

# Keeps the per-tree bootstrap stream clear of the per-tree feature stream:
# both are keyed by (seed, tree index) and a caller is free to give both
# samplers the same seed, so without a domain constant a tree's feature subset
# and its row weights would be drawn from one shared counter run.
comptime _BOOTSTRAP_DOMAIN = UInt64(0xBA9E51A4B007574B)


def _stream(seed: Int, tree_index: Int, tag: Int) -> UInt64:
    """Start of the counter stream for one selection. Sign bits are masked
    off so negative seeds are accepted (as in LightGBM) without relying on
    signed-to-unsigned conversion."""
    var h = splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF))
    h = splitmix64(h ^ UInt64(tree_index & 0x7FFFFFFFFFFFFFFF))
    return splitmix64(h ^ UInt64(tag & 0x7FFFFFFFFFFFFFFF))


def _fraction_in_range(fraction: Float64) -> Bool:
    """`check_feature_fraction`'s predicate without its name.

    The name is a `String` parameter, and every spelling a caller passes here
    -- `"feature_fraction_bylevel"` is 24 characters -- is longer than the
    small-string buffer, so materializing it is a heap allocation. The name
    is only ever read to phrase the error, so a per-node caller tests the
    range with this and calls `check_feature_fraction` only on the path that
    is about to raise. Same accepted range, same error text, no allocation on
    the path that succeeds."""
    return fraction > 0.0 and fraction <= 1.0


def check_feature_fraction(fraction: Float64, name: String) raises:
    """Validate one fraction; LightGBM's accepted range is (0, 1]."""
    if not _fraction_in_range(fraction):
        raise Error(String(name, " must be in (0, 1]"))


def check_feature_fractions(
    feature_fraction: Float64,
    feature_fraction_bynode: Float64,
    feature_fraction_bylevel: Float64 = DEFAULT_FEATURE_FRACTION_BYLEVEL,
) raises:
    """Validate every stage's fraction before growth starts.

    `select_level_features` validates its own fraction too, so the third
    argument is about failing before the first histogram rather than part way
    down a tree. It defaults to 1.0 so the two-argument callers that predate
    the per-level draw keep working unchanged.
    """
    check_feature_fraction(feature_fraction, "feature_fraction")
    check_feature_fraction(feature_fraction_bynode, "feature_fraction_bynode")
    check_feature_fraction(
        feature_fraction_bylevel, "feature_fraction_bylevel"
    )


def selection_count(total: Int, fraction: Float64) -> Int:
    """How many of `total` candidates to select, LightGBM's
    `ColSampler::GetCnt`: round(total * fraction), never fewer than 2 (or
    than `total` when that is smaller), never more than `total`."""
    if total <= 0:
        return 0
    var count = Int(Float64(total) * fraction + 0.5)
    var floor_count = 2 if total >= 2 else total
    if count < floor_count:
        count = floor_count
    if count > total:
        count = total
    return count


def sample_without_replacement(
    pool: List[Int], count: Int, stream: UInt64
) -> List[Int]:
    """Draw `count` distinct entries of `pool` (Knuth's Algorithm S, as in
    LightGBM's `Random::Sample`). The result keeps `pool`'s order, so an
    ascending pool yields an ascending subset, and no entry repeats."""
    var n = len(pool)
    if count >= n:
        return pool.copy()
    var out = List[Int]()
    if count <= 0:
        return out^
    out.reserve(count)
    var chosen = 0
    for i in range(n):
        if chosen >= count:
            break
        var prob = Float64(count - chosen) / Float64(n - i)
        if uniform(stream + UInt64(i)) < prob:
            out.append(pool[i])
            chosen += 1
    return out^


def check_feature_pool(pool: List[Int], n_features: Int) raises:
    """A prefiltered pool has to be ascending, unique, and inside the matrix.

    Ascending because `sample_without_replacement` keeps its input's order and
    every caller downstream (`select_split_features`, the growers' column
    lists) reads the result as ascending; unique because a repeated id would
    give one feature two chances in the same draw.
    """
    if len(pool) == 0:
        raise Error("feature pool must not be empty")
    if pool[0] < 0 or pool[len(pool) - 1] >= n_features:
        raise Error(
            "feature pool ids must be in [0, ", n_features, ")"
        )
    for i in range(1, len(pool)):
        if pool[i] <= pool[i - 1]:
            raise Error("feature pool must be strictly ascending")


def select_tree_features(
    n_features: Int,
    fraction: Float64,
    seed: Int,
    tree_index: Int,
    usable: List[Int] = [],
) raises -> List[Int]:
    """The ascending feature ids one tree may split on.

    `usable` is LightGBM's `Dataset::ValidFeatureIndices()`, the pool its
    `ColSampler` actually samples from (`src/treelearner/col_sampler.hpp`:
    `used_cnt_bytree_ = GetCnt(valid_feature_indices_.size(), fraction)` then
    `random_.Sample(valid_feature_indices_.size(), used_cnt_bytree_)`). Two
    things follow, and both are reproduced here: a prefiltered feature is never
    drawn, and the *count* drawn is the fraction of the surviving features
    rather than of all of them.

    An empty `usable` -- the default -- means nothing was prefiltered, so the
    pool is every feature and this is the function it has always been, down to
    the same counter stream. Passing `binning.all_features(n_features)`
    explicitly is the same draw.
    """
    check_feature_fraction(fraction, "feature_fraction")
    if n_features < 1:
        raise Error("n_features must be positive")
    var pool: List[Int]
    if len(usable) > 0:
        check_feature_pool(usable, n_features)
        pool = usable.copy()
    else:
        pool = List[Int](capacity=n_features)
        for f in range(n_features):
            pool.append(f)
    if fraction >= 1.0:
        return pool^
    return sample_without_replacement(
        pool,
        selection_count(len(pool), fraction),
        _stream(seed, tree_index, _TREE_TAG),
    )


def select_node_features(
    tree_features: List[Int],
    fraction: Float64,
    seed: Int,
    tree_index: Int,
    node: Int,
) raises -> List[Int]:
    """The ascending feature ids one node may split on, drawn from the
    tree's own set so per-node selection composes with per-tree selection.
    Returns the tree's set unchanged when the fraction is 1.0."""
    # Called once per node, so the name is built only when it will be read;
    # see `_fraction_in_range`.
    if not _fraction_in_range(fraction):
        check_feature_fraction(fraction, "feature_fraction_bynode")
    if fraction >= 1.0:
        return tree_features.copy()
    return sample_without_replacement(
        tree_features,
        selection_count(len(tree_features), fraction),
        _stream(seed, tree_index, node + 1),
    )


def _level_stream(seed: Int, tree_index: Int, depth: Int) -> UInt64:
    """Start of the counter stream for one depth's draw. It hangs off the
    tree's own stream, so a tree's level sets move with the tree, and is then
    mixed with a domain constant that keeps depths clear of the per-node tag
    space."""
    var h = _stream(seed, tree_index, _TREE_TAG) ^ _LEVEL_DOMAIN
    return splitmix64(h + UInt64(depth & 0x7FFFFFFFFFFFFFFF) * GOLDEN)


def select_level_features(
    tree_features: List[Int],
    fraction: Float64,
    seed: Int,
    tree_index: Int,
    depth: Int,
) raises -> List[Int]:
    """The ascending feature ids every node at `depth` may split on, drawn
    from the tree's own set so per-level selection composes with per-tree
    selection. Returns the tree's set unchanged when the fraction is 1.0,
    which is what keeps the default path bit identical."""
    # Called once per node, so the name is built only when it will be read;
    # see `_fraction_in_range`.
    if not _fraction_in_range(fraction):
        check_feature_fraction(fraction, "feature_fraction_bylevel")
    if depth < 0:
        raise Error("depth must be nonnegative")
    if fraction >= 1.0:
        return tree_features.copy()
    return sample_without_replacement(
        tree_features,
        selection_count(len(tree_features), fraction),
        _level_stream(seed, tree_index, depth),
    )


def select_split_features(
    tree_features: List[Int],
    fraction_bylevel: Float64,
    fraction_bynode: Float64,
    seed: Int,
    tree_index: Int,
    depth: Int,
    node: Int,
) raises -> List[Int]:
    """One node's candidate features, composing the per-level and per-node
    draws on top of the tree's set.

    This is the single call a grower needs per node: with a bylevel fraction
    of 1.0 it is exactly `select_node_features` on the tree's set, so
    substituting it for that call leaves existing models unchanged.
    """
    return select_node_features(
        select_level_features(
            tree_features, fraction_bylevel, seed, tree_index, depth
        ),
        fraction_bynode,
        seed,
        tree_index,
        node,
    )


def sampling_param_names() -> List[String]:
    """Every sampling parameter mojotrees names, in mojotrees's spelling.
    `canonical_sampling_param` maps into this list and nowhere else."""
    return [
        String("bagging_fraction"),
        String("bagging_freq"),
        String("bagging_seed"),
        String("pos_bagging_fraction"),
        String("neg_bagging_fraction"),
        String("feature_fraction"),
        String("feature_fraction_bylevel"),
        String("feature_fraction_bynode"),
        String("feature_fraction_seed"),
        String("top_rate"),
        String("other_rate"),
        String("data_sample_strategy"),
        String("bootstrap_type"),
        String("bagging_temperature"),
        String("bootstrap_seed"),
    ]


def canonical_sampling_param(name: String) raises -> String:
    """The mojotrees name for one sampling parameter spelling.

    Accepts LightGBM's own aliases, the scikit-learn spellings LightGBM's
    Python API accepts, and XGBoost's `colsample_bylevel` for the per-level
    fraction mojotrees adds. Raises on anything else rather than passing an
    unrecognized name through, so a misspelled parameter cannot be silently
    ignored by the caller that resolves through here.
    """
    if (
        name == "bagging_fraction"
        or name == "sub_row"
        or name == "subsample"
        or name == "bagging"
    ):
        return "bagging_fraction"
    if name == "bagging_freq" or name == "subsample_freq":
        return "bagging_freq"
    if name == "bagging_seed" or name == "bagging_fraction_seed":
        return "bagging_seed"
    if (
        name == "pos_bagging_fraction"
        or name == "pos_sub_row"
        or name == "pos_subsample"
        or name == "pos_bagging"
    ):
        return "pos_bagging_fraction"
    if (
        name == "neg_bagging_fraction"
        or name == "neg_sub_row"
        or name == "neg_subsample"
        or name == "neg_bagging"
    ):
        return "neg_bagging_fraction"
    if (
        name == "feature_fraction"
        or name == "sub_feature"
        or name == "colsample_bytree"
    ):
        return "feature_fraction"
    # XGBoost's name; LightGBM has no per-level fraction at all.
    if name == "feature_fraction_bylevel" or name == "colsample_bylevel":
        return "feature_fraction_bylevel"
    if (
        name == "feature_fraction_bynode"
        or name == "sub_feature_bynode"
        or name == "colsample_bynode"
    ):
        return "feature_fraction_bynode"
    if name == "feature_fraction_seed":
        return "feature_fraction_seed"
    if name == "top_rate":
        return "top_rate"
    if name == "other_rate":
        return "other_rate"
    if name == "data_sample_strategy":
        return "data_sample_strategy"
    # CatBoost's names; LightGBM has no weighting bootstrap at all. `subsample`
    # is deliberately absent: CatBoost spells row bagging that way, and it
    # already resolves to `bagging_fraction` above, which is the sampler it is.
    if name == "bootstrap_type":
        return "bootstrap_type"
    if name == "bagging_temperature":
        return "bagging_temperature"
    if name == "bootstrap_seed":
        return "bootstrap_seed"
    raise Error(String("unknown sampling parameter ", name))


def is_sampling_param(name: String) -> Bool:
    """Whether `name` is a spelling `canonical_sampling_param` accepts."""
    try:
        _ = canonical_sampling_param(name)
        return True
    except:
        return False


def canonical_data_sample_strategy(value: String) raises -> String:
    """LightGBM 4.x's `data_sample_strategy` value, which selects the row
    sampler: `bagging` for row bagging (bagging.mojo) and `goss` for
    gradient-based one-side sampling (goss.mojo). mojotrees also reaches GOSS
    through `boosting="goss"`, LightGBM 3.x's spelling, so a caller that
    accepts both must reject a disagreement rather than let one win."""
    if value == "bagging":
        return "bagging"
    if value == "goss":
        return "goss"
    raise Error(String("unknown data_sample_strategy ", value))


@fieldwise_init
struct ClassBaggingParams(Copyable, Movable):
    """Class-conditional row bagging configuration.

    `pos_fraction` and `neg_fraction` in (0, 1] are the per-row keep
    probabilities for positive and negative rows, `freq` the number of rounds
    one bag is reused for (0 disables bagging), `seed` the RNG seed. These are
    LightGBM's `pos_bagging_fraction`, `neg_bagging_fraction`, `bagging_freq`,
    and `bagging_seed`; LightGBM applies them only to binary classification
    and ignores them elsewhere, and so should any caller here.
    """

    var pos_fraction: Float64
    var neg_fraction: Float64
    var freq: Int
    var seed: Int

    @staticmethod
    def disabled() -> ClassBaggingParams:
        """No class-conditional bagging. LightGBM's defaults."""
        return ClassBaggingParams(
            DEFAULT_POS_BAGGING_FRACTION,
            DEFAULT_NEG_BAGGING_FRACTION,
            0,
            DEFAULT_BAGGING_SEED,
        )

    def enabled(self) -> Bool:
        """On only when the schedule runs and at least one class is thinned.

        LightGBM's own condition (`BaggingSampleStrategy::ResetSampleConfig`)
        is this and one thing more: the dataset must hold at least one
        positive row, or balanced bagging is skipped and plain
        `bagging_fraction` applies instead. That part needs the labels, so it
        lives in `has_positive_rows` and a caller must test both.
        """
        return self.freq > 0 and (
            self.pos_fraction < 1.0 or self.neg_fraction < 1.0
        )

    def validate(self) raises:
        """Reject out-of-range settings. The comparisons are written so that
        NaN fractions are rejected too."""
        if not (self.pos_fraction > 0.0 and self.pos_fraction <= 1.0):
            raise Error("pos_bagging_fraction must be in (0, 1]")
        if not (self.neg_fraction > 0.0 and self.neg_fraction <= 1.0):
            raise Error("neg_bagging_fraction must be in (0, 1]")
        if self.freq < 0:
            raise Error("bagging_freq must be nonnegative")


def has_positive_rows(labels: List[Float64]) -> Bool:
    """Whether any row is positive, the second half of LightGBM's condition
    for turning balanced bagging on. Hoist it out of the round loop: it is one
    pass over the labels and they do not change during training."""
    for r in range(len(labels)):
        if labels[r] > 0.0:
            return True
    return False


def _row_stream(seed: Int, bag_index: Int) -> UInt64:
    """Start of the counter stream for one bag. Deliberately the same
    function bagging.mojo draws its uniform bags from, so equal positive and
    negative fractions reproduce a uniform bag row for row; each sampler keeps
    its own copy rather than sharing one, as goss.mojo does."""
    return splitmix64(
        UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ (UInt64(bag_index) * GOLDEN)
    )


def sample_rows_by_class(
    params: ClassBaggingParams,
    labels: List[Float64],
    bag_index: Int,
    mut rows: List[Int],
) raises:
    """Draw bag number `bag_index` into `rows` (cleared first), ascending.

    A row is positive when its label is above zero and is kept with
    probability `pos_fraction`; every other row is kept with probability
    `neg_fraction`. Deterministic in (seed, bag_index, labels) alone.

    Like uniform bagging, and unlike LightGBM, an unlucky draw never empties
    the bag: a class that is present but drew nothing contributes the single
    row with the smallest draw value. That guard is the only reason a bag here
    can differ from LightGBM's schedule, and it can only fire when a class
    would otherwise vanish from the tree entirely.
    """
    params.validate()
    var n = len(labels)
    if n < 1:
        raise Error("labels must not be empty")
    if bag_index < 0:
        raise Error("bag_index must be nonnegative")

    var stream = _row_stream(params.seed, bag_index)
    var keep = List[Bool](capacity=n)
    var pos_seen = 0
    var neg_seen = 0
    var pos_kept = 0
    var neg_kept = 0
    var pos_min = 2.0
    var neg_min = 2.0
    var pos_min_row = 0
    var neg_min_row = 0
    for r in range(n):
        var u = uniform(stream + UInt64(r))
        var positive = labels[r] > 0.0
        var fraction = params.pos_fraction if positive else params.neg_fraction
        keep.append(u < fraction)
        if positive:
            pos_seen += 1
            if u < fraction:
                pos_kept += 1
            if u < pos_min:
                pos_min = u
                pos_min_row = r
        else:
            neg_seen += 1
            if u < fraction:
                neg_kept += 1
            if u < neg_min:
                neg_min = u
                neg_min_row = r

    if pos_seen > 0 and pos_kept == 0:
        keep[pos_min_row] = True
    if neg_seen > 0 and neg_kept == 0:
        keep[neg_min_row] = True

    rows.clear()
    for r in range(n):
        if keep[r]:
            rows.append(r)


def refresh_class_bag(
    mut bag: List[Int],
    params: ClassBaggingParams,
    labels: List[Float64],
    iteration: Int,
) raises:
    """Redraw `bag` if `iteration` starts a new bag, leave it alone if not.

    The schedule is uniform bagging's: bags change on the iterations where
    `iteration % freq == 0`, so the bag in force at iteration i is bag number
    `i // freq`. `bag` stays empty while class bagging is disabled, and an
    empty bag means "all rows" everywhere downstream.
    """
    if not params.enabled():
        params.validate()
        return
    if iteration % params.freq != 0:
        return
    sample_rows_by_class(params, labels, iteration // params.freq, bag)


@fieldwise_init
struct BayesianBootstrapParams(Copyable, Movable):
    """CatBoost's `bootstrap_type=Bayesian`: per-row weights drawn per tree.

    Verified against CatBoost `master` (August 2026):
    `catboost/private/libs/options/bootstrap_options.h` and `.cpp` for the
    option surface, `catboost/private/libs/algo/tensor_search_helpers.cpp` for
    the draw, and `catboost/private/libs/algo/greedy_tensor_search.cpp` for how
    often it is taken.

    What the options mean there, and here:

    - `bagging_temperature` defaults to **1.0** and must be `>= 0`
      (`TBootstrapConfig::Validate`: "Bagging temperature should be >= 0").
      Zero is not "off": it is CatBoost's early return that fills every weight
      with exactly 1, which is the same model an unbootstrapped fit builds.
      Larger temperatures stretch the weight distribution, so the sampler
      becomes a stronger regularizer as it rises.
    - `subsample` is **refused** beside Bayesian bootstrap, in CatBoost and
      here: "bayesian bootstrap doesn't support 'subsample' option". Bayesian
      bootstrap keeps every row, so there is no fraction to set. mojotrees's
      `bagging_fraction` is the row-dropping sampler (CatBoost's Bernoulli),
      and the two do not compose.
    - `sampling_frequency` defaults to `PerTree`
      (`oblivious_tree_options.cpp`), and `greedy_tensor_search.cpp` calls
      `DoBootstrap` once per tree under that default and once per level under
      `PerTreeLevel`. Only the per-tree schedule is implemented here, which is
      why the draw is keyed by tree index and not by (tree, depth).
    - `sampling_unit` defaults to `Object` (per row). CatBoost's `Group` unit
      gives every row of a query group one shared weight; that belongs with
      the ranker, which samples whole queries already, and is not implemented
      here.

    Disabled by default, so an untouched bundle draws nothing and leaves every
    fit byte for byte the fit it is today.
    """

    var enabled: Bool
    var temperature: Float64
    var seed: Int

    @staticmethod
    def disabled() -> BayesianBootstrapParams:
        """No bootstrap: every row keeps weight 1 (the library default)."""
        return BayesianBootstrapParams(
            False, DEFAULT_BAGGING_TEMPERATURE, DEFAULT_BOOTSTRAP_SEED
        )

    @staticmethod
    def enable(
        temperature: Float64 = DEFAULT_BAGGING_TEMPERATURE,
        seed: Int = DEFAULT_BOOTSTRAP_SEED,
    ) -> BayesianBootstrapParams:
        """Bayesian bootstrap at CatBoost's defaults."""
        return BayesianBootstrapParams(True, temperature, seed)

    def validate(self) raises:
        """CatBoost's `TBootstrapConfig::Validate` rule for this type. The
        comparison is written so that a NaN temperature is rejected too."""
        if not (self.temperature >= 0.0):
            raise Error("bagging_temperature must be >= 0")

    def draws_weights(self) -> Bool:
        """Whether a caller has to materialize a per-row weight buffer.

        False at `temperature == 0` even when enabled, because that is
        CatBoost's own early return to all-ones and no draw is taken; the
        weights would be a vector of exactly 1.0. It is deliberately NOT the
        test the constant-hessian exclusion uses -- see
        `bayesian_bootstrap_varies_hessian`, which is coarser on purpose.
        """
        return self.enabled and self.temperature != 0.0


def bayesian_bootstrap_varies_hessian(
    params: BayesianBootstrapParams,
) -> Bool:
    """Whether a fit configured this way has a per-row hessian, so that
    `histogram.CONSTANT_HESSIAN` must not be declared for it.

    A bootstrap weight multiplies the row's derivatives exactly as a sample
    weight does (CatBoost `CalcWeightedData`:
    `sampleWeightedDerivativesData[z] = weightedDerivativesData[z] *
    sampleWeightsData[z]`, and then `SampleWeights[i] *= learnWeights[i]` so
    the leaf denominators carry the product). Under an objective whose
    unweighted hessian is the literal 1.0, a bootstrapped round stores the
    drawn weight into `hess` instead, so a histogram builder told to rebuild
    the hessian plane from the count would rebuild the wrong plane, silently.

    `params.enabled` is the whole test, and `temperature` is deliberately not
    consulted, for the same reason `boosting.round_has_constant_hessian` tests
    `goss.enabled` rather than `goss.active`: the declaration is held for a
    whole fit (on the device it is builder state set once and cannot be
    withdrawn mid-loop), and a safety predicate that reasons about the value of
    a `Float64` knob is one edit away from being wrong. A `temperature == 0`
    fit therefore pays for the third plane and gets the same model; that is the
    cheap side of the trade.
    """
    return params.enabled


def check_bayesian_bootstrap_hessian_declaration(
    params: BayesianBootstrapParams, const_hessian: Bool
) raises:
    """Refuse a constant-hessian declaration beside an active bootstrap.

    `boosting.round_has_constant_hessian` is the predicate that makes the
    declaration, and it cannot see a bootstrap configuration: its three inputs
    are the objective, the sample weights, and the GOSS parameters, and its
    signature is a GPU-visible contract that no CPU lane may widen. So the
    guard lives here instead, on the sampler's own side, and a caller that
    wires this into a round loop calls it once per fit with whatever it
    decided. It costs one branch per fit and converts the failure mode from a
    quietly wrong hessian plane into an exception at fit setup.
    """
    if bayesian_bootstrap_varies_hessian(params) and const_hessian:
        raise Error(
            "a fit with bayesian bootstrap active must not declare a"
            " constant hessian: the bootstrap weight multiplies the row's"
            " derivatives, so the hessian is the weight and varies per row"
        )


def _bootstrap_stream(seed: Int, tree_index: Int) -> UInt64:
    """Start of the counter stream for one tree's weights.

    Same shape as `_stream` and `_level_stream`: mix the seed, spread the tree
    index by the golden-ratio increment, mix again, and separate the whole
    space from the feature sampler's with a domain constant. Sign bits are
    masked off so negative seeds are accepted without relying on
    signed-to-unsigned conversion.

    The stream is a *start*, not a running state: row r reads
    `stream + r` and nothing advances, so no row's weight depends on how many
    rows were drawn before it.
    """
    var h = splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _BOOTSTRAP_DOMAIN)
    return splitmix64(h ^ (UInt64(tree_index & 0x7FFFFFFFFFFFFFFF) * GOLDEN))


def bayesian_bootstrap_weight(
    temperature: Float64, stream: UInt64, row: Int
) -> Float64:
    """One row's weight, a transcription of CatBoost's
    `GenerateBayessianWeight`.

        const float w = -FastLogf(rand.GenRandReal1() + 1e-100);
        return powf(w, baggingTemperature);

    So the weight is an exponential(1) draw raised to the temperature. At
    `temperature == 1` it is exactly the exponential, mean 1, which is what
    makes the bootstrap a reweighting rather than a rescaling: the weights
    average to 1 and the round's total hessian mass is preserved in
    expectation.

    Two deliberate differences from CatBoost, both of them the difference the
    rest of this module already makes:

    - **The stream.** CatBoost splits the rows into blocks of 1000, gives each
      block a `TRestorableFastRng64(randSeed + blockIdx)`, advances it 10
      steps "to reduce correlation between RNGs in different threads", and
      then draws sequentially within the block, so a row's weight depends on
      its position inside its block and on the block size. mojotrees derives a
      counter-based splitmix64 stream from (seed, tree index) and indexes it by
      the row, so a row's weight depends on (seed, tree, row) and on nothing
      else: not on a block layout, not on a thread count, not on how many rows
      or trees came before. The distribution and the per-row transform are the
      same; the individual weights for a given seed are not.
    - **The width.** CatBoost's log, power, and weight are `float`; this is
      `Float64` throughout, as every other draw here is.
    """
    var u = uniform(stream + UInt64(row))
    return (-log(u + _BAYESIAN_LOG_EPS)) ** temperature


def bayesian_bootstrap_weights(
    params: BayesianBootstrapParams,
    n_rows: Int,
    tree_index: Int,
    mut weights: List[Float64],
) raises:
    """Fill `weights` with tree number `tree_index`'s draw, one entry per row.

    `weights` is resized rather than appended to, so a round loop can hand the
    same buffer back every tree and allocate once for the whole fit.

    A disabled bundle and a zero temperature both fill exactly 1.0 and take no
    draw at all, which is CatBoost's early return
    (`if (baggingTemperature == 0) { Fill(..., 1); return; }`). The rows are
    walked in order only because the buffer is; every entry is computed from
    its own row index, so any order, and any partition across workers,
    produces the identical buffer.
    """
    params.validate()
    if n_rows < 0:
        raise Error("n_rows must be nonnegative")
    if tree_index < 0:
        raise Error("tree_index must be nonnegative")
    weights.clear()
    weights.resize(n_rows, 1.0)
    if not params.draws_weights():
        return
    var stream = _bootstrap_stream(params.seed, tree_index)
    for r in range(n_rows):
        weights[r] = bayesian_bootstrap_weight(params.temperature, stream, r)


def refresh_bayesian_bootstrap(
    mut weights: List[Float64],
    params: BayesianBootstrapParams,
    base_weight: List[Float64],
    n_rows: Int,
    tree_index: Int,
) raises:
    """The round's effective per-row weight: the tree's bootstrap draw times
    the user's own `sample_weight`.

    This is CatBoost's `CalcWeightedData` tail, `SampleWeights[i] *=
    learnWeights[i]`, and it is the reason a bootstrapped fit is a *weighted*
    fit as far as everything downstream is concerned: the product is what
    multiplies the derivatives and what the leaf denominators sum. An empty
    `base_weight` is the "unweighted" convention used everywhere else here and
    leaves the draw alone.

    A disabled bundle leaves `weights` empty rather than filling it with ones,
    so a caller can pass it straight through as the fit's `sample_weight` and
    an unbootstrapped, unweighted fit stays exactly unweighted -- no vector of
    1.0s that would make `boosting.round_has_constant_hessian` refuse a fit it
    should admit, and no bits moved on the default path.
    """
    if len(base_weight) != 0 and len(base_weight) != n_rows:
        raise Error("sample_weight length must match the row count")
    if not params.enabled:
        params.validate()
        weights.clear()
        if len(base_weight) != 0:
            weights.resize(n_rows, 1.0)
            for r in range(n_rows):
                weights[r] = base_weight[r]
        return
    bayesian_bootstrap_weights(params, n_rows, tree_index, weights)
    if len(base_weight) != 0:
        for r in range(n_rows):
            weights[r] = weights[r] * base_weight[r]


def canonical_bootstrap_type(value: String) raises -> String:
    """The `bootstrap_type` value one spelling selects.

    Only CatBoost's `No` and `Bayesian` are implemented. `Bernoulli` is
    refused by name rather than as an unknown value because mojotrees already
    has it under another name -- it is row bagging, `bagging_fraction` with
    `bagging_freq`, which is the same draw CatBoost's `subsample` makes -- and
    telling a user that a real bootstrap type is unknown would be misleading.
    `MVS` and `Poisson` are refused the same way; CatBoost itself refuses
    Poisson on the CPU.
    """
    if value == "no" or value == "none":
        return "no"
    if value == "bayesian":
        return "bayesian"
    if value == "bernoulli":
        raise Error(
            "bootstrap_type 'bernoulli' is row bagging under another name;"
            " use bagging_fraction with bagging_freq"
        )
    if value == "mvs" or value == "poisson":
        raise Error(
            String("bootstrap_type '", value, "' is not implemented")
        )
    raise Error(String("unknown bootstrap_type ", value))


def check_row_set(rows: List[Int], n_rows: Int) raises:
    """Reject a row set that is not strictly ascending or not in range.

    Every sampled row set mojotrees builds is ascending and duplicate free
    (uniform bagging, class bagging, and GOSS all walk rows in order), and the
    range and mask forms below rely on it, so this is the one place the
    property is enforced instead of assumed.
    """
    if n_rows < 0:
        raise Error("n_rows must be nonnegative")
    for i in range(len(rows)):
        if rows[i] < 0 or rows[i] >= n_rows:
            raise Error("sampled row index out of range")
        if i > 0 and rows[i] <= rows[i - 1]:
            raise Error("sampled rows must be strictly ascending")


def contiguous_ranges(rows: List[Int], n_rows: Int) raises -> List[Int]:
    """The sampled rows as half-open `[start, end)` ranges, flattened into
    `[start0, end0, start1, end1, ...]`.

    An empty `rows` is the "every row" convention and yields the single range
    `[0, n_rows)`, so a caller never has to special-case unsampled training.
    A row set that is already contiguous yields exactly one range, which is
    the case a device-side active-row pass can iterate without an index
    indirection at all.
    """
    check_row_set(rows, n_rows)
    var ranges = List[Int]()
    if len(rows) == 0:
        if n_rows > 0:
            ranges.append(0)
            ranges.append(n_rows)
        return ranges^
    var start = rows[0]
    var prev = rows[0]
    for i in range(1, len(rows)):
        if rows[i] != prev + 1:
            ranges.append(start)
            ranges.append(prev + 1)
            start = rows[i]
        prev = rows[i]
    ranges.append(start)
    ranges.append(prev + 1)
    return ranges^


def ranges_row_count(ranges: List[Int]) raises -> Int:
    """How many rows a flattened range list covers."""
    if len(ranges) % 2 != 0:
        raise Error("ranges must hold start/end pairs")
    var total = 0
    for i in range(0, len(ranges), 2):
        if ranges[i + 1] < ranges[i]:
            raise Error("range end must not precede its start")
        total += ranges[i + 1] - ranges[i]
    return total


def row_mask(rows: List[Int], n_rows: Int) raises -> List[Bool]:
    """The sampled rows as a dense mask over all `n_rows` rows. An empty
    `rows` means every row, so the mask is all true."""
    check_row_set(rows, n_rows)
    var mask = List[Bool](capacity=n_rows)
    mask.resize(n_rows, len(rows) == 0)
    for i in range(len(rows)):
        mask[rows[i]] = True
    return mask^


def expand_row_scale(
    rows: List[Int], scale: List[Float64], n_rows: Int
) raises -> List[Float64]:
    """A (rows, scale) sample as a dense per-row multiplier.

    `scale[i]` belongs to `rows[i]`, which is how GOSS reports its
    small-gradient amplification. Rows outside the sample get 0.0, not 1.0, so
    multiplying gradients and hessians by this vector over every row builds
    exactly the histogram the sample builds: the amplification and the
    exclusion travel together in one buffer, which is what a device-resident
    kernel needs when it cannot afford to gather rows first.

    An empty `rows` is the "every row" convention and yields all ones. An
    empty `scale` with a non-empty `rows` means an unweighted sample, as
    uniform and class bagging produce, and yields ones on the sample.
    """
    check_row_set(rows, n_rows)
    if len(scale) != 0 and len(scale) != len(rows):
        raise Error("scale length must match the sampled row count")
    var out = List[Float64](capacity=n_rows)
    out.resize(n_rows, 1.0 if len(rows) == 0 else 0.0)
    for i in range(len(rows)):
        out[rows[i]] = scale[i] if len(scale) != 0 else 1.0
    return out^
