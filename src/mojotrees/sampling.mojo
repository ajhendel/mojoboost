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
`canonical_sampling_param` maps every LightGBM, XGBoost, scikit-learn and
CatBoost spelling of a row or feature sampling parameter onto the canonical
name of docs/PARAMETER_NAMING.md -- `subsample`, `subsample_freq`,
`colsample_bytree`, `colsample_bynode`, `colsample_bylevel`. The table here
is the single place those aliases are written down, so the Python layer, the
CLI, and the C API resolve through it instead of each keeping a list.

One row-share number, one key. `bagging_fraction`, `sub_row`, `bagging` and
`subsample` all name the share of rows a round keeps, and so does the
`subsample` that CatBoost's Bernoulli and MVS bootstraps read, so all of
them resolve to `subsample`. Which sampler consumes it is decided by
`bootstrap_type` and by `data_sample_strategy`, not by the spelling; the
Bayesian bootstrap is the one that reads no share at all, because it weights
every row rather than selecting any.

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

MVS
---
CatBoost's `bootstrap_type=MVS` (Minimal Variance Sampling) is the sampler
CatBoost actually runs on the CPU: `catboost_options.cpp` installs it whenever
`bootstrap_type` was left alone and the loss is neither multiclass-only nor
multi-regression, `task_type` is CPU, and `sampling_unit` is `Object` -- which
covers both objectives this project benchmarks. **The Bayesian bootstrap above
is CatBoost's fallback, not its default.** MVS keeps a row with probability
proportional to its gradient magnitude (capped at 1), amplifies a kept row by
the reciprocal of that probability, and drops the rest outright. See
`MvsBootstrapParams` for the option semantics, `mvs_bootstrap_weights` for the
draw, `_mvs_threshold` for the solve, and `docs/design/CATBOOST_CATALOG.md`
A11 for the source citations.

**Its weights are hessians too**, for exactly the reason the Bayesian
bootstrap's are, and more strongly: see `mvs_varies_hessian` and
`check_mvs_hessian_declaration`.

**The wire is `BootstrapParams` and `bootstrap_round`.** The two bundles are
paired into one `bootstrap_type` argument and one per-round entry point, in
`goss.goss_round`'s shape and in `goss_round`'s place in a round loop:
`boosting._boost_rounds` and `boosting.train_with_valid` call it immediately
after the gradient fill. `BootstrapParams.check_hessian_declaration` is how a
trainer proves it withdrew the constant-hessian declaration, and
`check_bootstrap_honored` is what a trainer without the call in its loop uses
instead of ignoring the bundle.

**Read `check_mvs_reg` before touching the regularizer.** MVS's lambda is
`mvs_reg` and has nothing to do with `lambda_l2`; it is the only floor under a
bare denominator, its default is derived from the data rather than being a
number, and setting it to 0 in CatBoost can silently train a tree on no rows.

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

from std.math import log, sqrt, isfinite

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
comptime BOOTSTRAP_MVS = 2

# CatBoost's `subsample` default *under MVS*, and only under MVS:
# `catboost_options.cpp` does `if (bootstrapType == EBootstrapType::MVS)
# subsample.SetDefault(0.8)`. The `TBootstrapConfig` constructor's own default
# is 0.66, which is what Bernoulli and Poisson get; reading 0.66 out of the
# header and calling it "CatBoost's subsample" is wrong by a fifth of the rows.
comptime DEFAULT_MVS_SUBSAMPLE = 0.8

# CatBoost solves the MVS threshold per block of this many rows, not once over
# the dataset: `TMvsSampler::BlockSize = 8192`, `blockParams.SetBlockSize`, and
# each block targets `SampleRate * blockSize` of its own rows. It is a
# CONSTANT there, not a thread count, which is the only reason CatBoost's own
# MVS does not move with `thread_count` -- and it is the reason ours is a
# comptime constant here rather than anything derived from
# `MOJOTREES_NUM_WORKERS`. A block size that follows the worker count would
# make every weight in the fit follow it too.
comptime MVS_BLOCK_SIZE = 8192

# `std::numeric_limits<double>::epsilon()`, the literal constant CatBoost tests
# the keep probability against in `GenSampleWeights`.
comptime _MVS_PROBABILITY_EPS = 2.220446049250313e-16

# Separates the MVS stream from the Bayesian bootstrap's: both are keyed by
# (seed, tree index), both default their seed to 0, and a caller may well set
# `bootstrap_seed` once and try both samplers, so without a second domain
# constant the two would draw the same uniforms for the same tree.
comptime _MVS_DOMAIN = UInt64(0x4D56535F5361C71E)

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
    """Every sampling parameter mojotrees names, in its canonical spelling
    (docs/PARAMETER_NAMING.md). `canonical_sampling_param` maps into this
    list and nowhere else."""
    return [
        String("subsample"),
        String("subsample_freq"),
        String("bagging_seed"),
        String("pos_bagging_fraction"),
        String("neg_bagging_fraction"),
        String("colsample_bytree"),
        String("colsample_bylevel"),
        String("colsample_bynode"),
        String("feature_fraction_seed"),
        String("top_rate"),
        String("other_rate"),
        String("data_sample_strategy"),
        String("bootstrap_type"),
        String("bagging_temperature"),
        String("bootstrap_seed"),
        String("mvs_reg"),
    ]


def canonical_sampling_param(name: String) raises -> String:
    """The canonical name for one sampling parameter spelling.

    The canonical names are docs/PARAMETER_NAMING.md's: `subsample`,
    `subsample_freq`, `colsample_bytree`, `colsample_bynode`. Every other
    vendor's spelling of the same parameter -- LightGBM's own aliases, the
    scikit-learn spellings LightGBM's Python API accepts, XGBoost's
    `colsample_bylevel`, CatBoost's `rsm` -- resolves onto it. Raises on
    anything else rather than passing an unrecognized name through, so a
    misspelled parameter cannot be silently ignored by the caller that
    resolves through here.

    The names that stay LightGBM's are the ones no other vendor has a word
    for: `pos_bagging_fraction`, `neg_bagging_fraction`, `top_rate`,
    `other_rate`, `data_sample_strategy`, and the three seeds. The names
    that stay CatBoost's are the ones only CatBoost has: `bootstrap_type`,
    `bagging_temperature`, `mvs_reg`.

    **`subsample` is one key, and which sampler reads it depends on
    `bootstrap_type`.** LightGBM's `bagging_fraction`, CatBoost's Bernoulli
    `subsample` and CatBoost's MVS `subsample` are all the same number --
    the share of rows a round keeps -- so they resolve to one name and not
    to two. That is CatBoost's own shape: it has exactly one `subsample`
    option, Bernoulli, MVS and Poisson all read it, and Bayesian refuses it
    because it weights every row instead of selecting any. A key of its own
    for MVS would have made `subsample=0.8` mean one thing under one
    bootstrap and nothing under another, which is the collision this table
    exists to prevent rather than an instance of it.

    `mvs_reg` does get its own name, because it is a second number and not
    the same one: it is MVS's regularizer, CatBoost has a word for it, and
    no other sampler has anything it could collide with. MVS's seed does
    not, for the same reason in reverse -- it is the existing
    `bootstrap_seed`, which every bootstrap here already draws from.
    """
    if (
        name == "subsample"
        # LightGBM.
        or name == "bagging_fraction"
        or name == "sub_row"
        or name == "bagging"
    ):
        return "subsample"
    if name == "subsample_freq" or name == "bagging_freq":
        return "subsample_freq"
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
        name == "colsample_bytree"
        # LightGBM; CatBoost, whose `rsm` is the same per-tree share and is
        # opaque enough that it is the alias rather than the canonical name.
        or name == "feature_fraction"
        or name == "sub_feature"
        or name == "rsm"
    ):
        return "colsample_bytree"
    # XGBoost's name; LightGBM has no per-level fraction at all.
    if name == "colsample_bylevel" or name == "feature_fraction_bylevel":
        return "colsample_bylevel"
    if (
        name == "colsample_bynode"
        # LightGBM; scikit-learn's HistGradientBoosting*.
        or name == "feature_fraction_bynode"
        or name == "sub_feature_bynode"
        or name == "max_features"
    ):
        return "colsample_bynode"
    if name == "feature_fraction_seed":
        return "feature_fraction_seed"
    if name == "top_rate":
        return "top_rate"
    if name == "other_rate":
        return "other_rate"
    if name == "data_sample_strategy":
        return "data_sample_strategy"
    # CatBoost's names; LightGBM has no weighting bootstrap at all.
    # `subsample` is deliberately not among them: CatBoost spells row bagging
    # that way, and it is already the canonical name of that sampler above.
    if name == "bootstrap_type":
        return "bootstrap_type"
    if name == "bagging_temperature":
        return "bagging_temperature"
    if name == "bootstrap_seed":
        return "bootstrap_seed"
    if name == "mvs_reg":
        return "mvs_reg"
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


@fieldwise_init
struct MvsBootstrapParams(Copyable, Movable):
    """CatBoost's `bootstrap_type=MVS`, Minimal Variance Sampling.

    Verified against CatBoost `master` (August 2026):
    `catboost/private/libs/algo/mvs.cpp` and `mvs.h` for the algorithm,
    `catboost/private/libs/options/catboost_options.cpp`
    (`SetNotSpecifiedOptionsToDefaults`) for the defaults,
    `catboost/private/libs/options/bootstrap_options.{h,cpp}` for the option
    surface, `catboost/private/libs/algo/tensor_search_helpers.cpp` for the
    dispatch, and `catboost/private/libs/algo/calc_score_cache.cpp` for what a
    zero weight does. See `docs/design/CATBOOST_CATALOG.md`, the A11 note.

    **This is CatBoost's actual CPU default**, not the Bayesian bootstrap of
    `BayesianBootstrapParams`. `SetNotSpecifiedOptionsToDefaults` installs it
    whenever the user left `bootstrap_type` alone and the loss is neither
    multiclass-only nor multi-regression and `task_type` is CPU and
    `sampling_unit` is `Object`. Binary logloss and RMSE, which are the two
    objectives this project benchmarks, satisfy every one of those, so the
    CatBoost column beside our numbers has always been an MVS fit.

    What it does, per row `i`, with `lambda` from `resolve_reg`:

        g_i  = sqrt(sum_k grad[i][k]^2 + lambda)
        p_i  = 1 if g_i > mu else g_i / mu
        w_i  = 1/p_i if uniform() < p_i else 0

    `mu` is solved so `sum_i p_i` hits `subsample * rows`. Rows above the
    threshold survive certainly at weight exactly 1; rows below it survive with
    probability proportional to their gradient magnitude and are amplified by
    `mu/g` when they do. The `1/p` factor is what makes the sampled gradient
    sums unbiased, and choosing `p` proportional to `|g|` is what minimizes
    their variance at a fixed expected sample size -- the name is descriptive.

    It is the same family as GOSS, better calibrated: GOSS keeps a fixed top
    fraction and amplifies every survivor by one shared constant, where MVS
    solves for the threshold that hits the requested rate and gives each
    survivor its own amplification.

    The options, and where ours differ:

    - `subsample` defaults to **0.8** under MVS (`DEFAULT_MVS_SUBSAMPLE`) and
      must be in `(0, 1]` (`TBootstrapConfig::Validate`: "Subsample should be
      in (0,1]"). At exactly 1.0 CatBoost fills every weight with 1 and takes
      no draw, which is the unsampled model; `samples_rows` is that test.
    - `mvs_reg` is a `TMaybe<float>` defaulting to `Nothing()`, so "unset" is a
      real state and not a magic number. Unset means the lambda is derived from
      the data every tree; see `resolve_reg`. `reg_is_set` carries the
      distinction.
    - **`bagging_temperature` is refused beside MVS here and is NOT refused by
      CatBoost.** Their `Validate` rejects it for `No` and for Bernoulli but
      the `MVS` arm of that switch checks only the sampling unit, so CatBoost
      accepts the parameter, never reads it (the MVS branch of `Bootstrap`
      does not pass it to `TMvsSampler` at all), and `Save` drops it. A knob
      that is accepted and does nothing is a silent wrong answer to the user
      who set it. `check_mvs_bagging_temperature` is the refusal.
    - `sampling_unit` must be `Object`; CatBoost's own `Validate` says so
      ("MVS bootstrap supports per object sampling only") and `Bootstrap`
      repeats it. Group sampling is not implemented here at all.
    - `sampling_frequency` defaults to `PerTree`, so the draw is keyed by tree
      index and not by (tree, level), exactly as the Bayesian bootstrap is.

    Disabled by default, so an untouched bundle draws nothing and leaves every
    fit byte for byte the fit it is today.
    """

    var enabled: Bool
    var subsample: Float64
    var reg: Float64
    var reg_is_set: Bool
    var seed: Int

    @staticmethod
    def disabled() -> MvsBootstrapParams:
        """No MVS: every row kept at weight 1 (the library default)."""
        return MvsBootstrapParams(
            False, DEFAULT_MVS_SUBSAMPLE, 0.0, False, DEFAULT_BOOTSTRAP_SEED
        )

    @staticmethod
    def enable(
        subsample: Float64 = DEFAULT_MVS_SUBSAMPLE,
        seed: Int = DEFAULT_BOOTSTRAP_SEED,
    ) -> MvsBootstrapParams:
        """MVS at CatBoost's defaults, with `mvs_reg` left unset so the lambda
        is derived from the data as CatBoost's is."""
        return MvsBootstrapParams(True, subsample, 0.0, False, seed)

    @staticmethod
    def enable_with_reg(
        reg: Float64,
        subsample: Float64 = DEFAULT_MVS_SUBSAMPLE,
        seed: Int = DEFAULT_BOOTSTRAP_SEED,
    ) -> MvsBootstrapParams:
        """MVS with an explicit `mvs_reg`. `validate` refuses exactly 0; see
        `check_mvs_reg`."""
        return MvsBootstrapParams(True, subsample, reg, True, seed)

    def validate(self) raises:
        """CatBoost's `TBootstrapConfig::Validate` rules for this type, plus
        one refusal of ours. Comparisons are written so a NaN is rejected."""
        if not (self.subsample > 0.0 and self.subsample <= 1.0):
            raise Error("subsample must be in (0, 1]")
        if self.reg_is_set:
            check_mvs_reg(self.reg)

    def samples_rows(self) -> Bool:
        """Whether any row can be dropped or reweighted.

        False at `subsample == 1.0` even when enabled, which is CatBoost's own
        `if (SampleRate == 1.0f) Fill(SampleWeights, 1.0f)` early return. It is
        deliberately NOT the test the constant-hessian exclusion uses; see
        `mvs_varies_hessian`.
        """
        return self.enabled and self.subsample != 1.0

    def resolve_reg(self, auto_lambda: Float64) -> Float64:
        """The lambda a draw actually uses: the explicit `mvs_reg` when the
        user set one, otherwise the data-derived value.

        This is `TMvsSampler::GetLambda`'s first two lines
        (`if (Lambda.Defined()) return Lambda.GetRef();`), with the derivation
        of `auto_lambda` left to `mvs_auto_lambda_from_gradients` and
        `mvs_auto_lambda_from_leaf_values` because CatBoost picks between those
        two by whether any tree exists yet.
        """
        if self.reg_is_set:
            return self.reg
        return auto_lambda


def check_mvs_reg(reg: Float64) raises:
    """Validate an explicit `mvs_reg`, and refuse exactly zero.

    CatBoost requires `mvs_reg >= 0` (`TBootstrapConfig::Validate`: "MVS
    regularization parameter should be >= 0") and accepts 0. **We refuse 0**,
    and the reason is a defect in CatBoost worth stating in full because it is
    not a crash and produces no NaN anybody ever sees:

    lambda is what floors the row statistic `g_i = sqrt(sum_k der^2 + lambda)`.
    At `lambda == 0`, a block whose every derivative is zero has every `g_i`
    zero. `CalculateThreshold` then pivots on 0, computes `sumOfSmall / pivot`
    as `0.0/0.0` = NaN, tests `NaN > sampleSize` which is **false**, falls into
    the undershoot branch, and returns `0.0 / (0.8*n - n)` = **negative zero**.
    `GetSingleProbability` then computes `0.0 / -0.0` = NaN, and the next line
    is `if (probability > std::numeric_limits<double>::epsilon())`, which is
    **also false for NaN**, so the `else` runs and writes weight 0 for every
    row. The block is dropped entirely by `SetControlNoZeroWeighted`, the tree
    trains on nothing, and the user gets a model that learned nothing with no
    exception, no warning, and no NaN left in the output to notice. The NaN is
    eaten by a comparison that swallows it.

    There is no configuration in which passing 0 is better than leaving
    `mvs_reg` unset: unset derives a positive lambda from the data every tree,
    and 0's only two outcomes are "indistinguishable from a very small lambda"
    and "silently empty tree". So this raises rather than accepting a setting
    whose best case is the default.
    """
    if not (reg >= 0.0):
        raise Error("mvs_reg must be >= 0")
    if reg == 0.0:
        raise Error(
            "mvs_reg must not be 0: it is the only floor under the MVS"
            " threshold, and a zero-derivative block at mvs_reg=0 gives every"
            " row weight 0 and trains a tree on no rows at all. Leave mvs_reg"
            " unset to derive it from the data, as CatBoost does by default"
        )


def check_mvs_bagging_temperature(
    params: MvsBootstrapParams, temperature_is_set: Bool
) raises:
    """Refuse `bagging_temperature` beside MVS. A divergence, deliberately.

    `bagging_temperature` is the Bayesian bootstrap's knob and MVS never reads
    it. CatBoost's `TBootstrapConfig::Validate` refuses it for `No` and for
    Bernoulli but not for MVS -- the `MVS` arm of that switch tests only the
    sampling unit -- so CatBoost accepts `bootstrap_type=MVS,
    bagging_temperature=5`, ignores the temperature, and drops it in `Save`.
    The user gets no error and no effect.

    `temperature_is_set` is the caller's "the user actually wrote this down"
    flag, the equivalent of CatBoost's `TOption::IsSet`. A defaulted
    temperature is not a user setting and is not refused.
    """
    if params.enabled and temperature_is_set:
        raise Error(
            "bagging_temperature belongs to bootstrap_type 'bayesian' and is"
            " never read by MVS; remove it or switch bootstrap_type"
        )


def check_bootstrap_type_exclusive(
    mvs: MvsBootstrapParams, bayesian: BayesianBootstrapParams
) raises:
    """One `bootstrap_type` at a time. CatBoost's is a single enum, so the
    combination cannot be spelled there; ours is two bundles and can be."""
    if mvs.enabled and bayesian.enabled:
        raise Error(
            "bootstrap_type selects one sampler: mvs and bayesian bootstrap"
            " cannot both be enabled"
        )


def mvs_varies_hessian(params: MvsBootstrapParams) -> Bool:
    """Whether a fit configured this way has a per-row hessian, so that
    `histogram.CONSTANT_HESSIAN` must not be declared for it.

    Identical in force and in reason to
    `bayesian_bootstrap_varies_hessian`, and it is the same line of CatBoost
    that settles it: `Bootstrap` calls `CalcWeightedData` for MVS on the same
    line it calls it for Bayesian, and that function does
    `sampleWeightedDerivativesData[z] = weightedDerivativesData[z] *
    sampleWeightsData[z]` and then `SampleWeights[i] *= learnWeights[i]`. An
    MVS weight multiplies the row's derivatives and rides in the leaf
    denominators exactly as a sample weight does, so under an objective whose
    unweighted hessian is the literal 1.0 a sampled round stores `1/p` (or 0)
    into `hess`, and a histogram builder told to rebuild the hessian plane from
    the count would rebuild the wrong plane, silently.

    MVS is in fact the *stronger* case of the two: its weights are not merely
    unequal, most of them are exactly 1 and the rest are large, and a zero
    weight is a dropped row.

    `params.enabled` is the whole test, and `subsample` is deliberately not
    consulted, for the reason `bayesian_bootstrap_varies_hessian` gives: the
    declaration is held for a whole fit, on the device it is builder state set
    once that cannot be withdrawn mid-loop, and a safety predicate that reasons
    about the value of a `Float64` knob is one edit away from being wrong. A
    `subsample == 1.0` fit therefore pays for the third plane and gets the same
    model; that is the cheap side of the trade.
    """
    return params.enabled


def check_mvs_hessian_declaration(
    params: MvsBootstrapParams, const_hessian: Bool
) raises:
    """Refuse a constant-hessian declaration beside an active MVS bootstrap.

    The twin of `check_bayesian_bootstrap_hessian_declaration`, and it exists
    for the same structural reason: `boosting.round_has_constant_hessian` is
    the predicate that makes the declaration and it cannot see a bootstrap
    configuration -- its three inputs are the objective, the sample weights and
    the GOSS parameters, and its signature is a GPU-visible contract that no
    CPU lane may widen. So the guard lives on the sampler's side, costs one
    branch per fit, and converts the failure mode from a quietly wrong hessian
    plane into an exception at fit setup.
    """
    if mvs_varies_hessian(params) and const_hessian:
        raise Error(
            "a fit with mvs bootstrap active must not declare a constant"
            " hessian: the mvs weight multiplies the row's derivatives, so the"
            " hessian is the weight and varies per row"
        )


def _mvs_stream(seed: Int, tree_index: Int) -> UInt64:
    """Start of the counter stream for one tree's keep decisions.

    Same shape as `_stream`, `_level_stream` and `_bootstrap_stream`: mix the
    seed, spread the tree index by the golden-ratio increment, mix again, and
    separate the whole space from the other per-tree streams with a domain
    constant.

    The stream is a *start*, not a running state: row `r` reads `stream + r`
    and nothing advances, so no row's keep decision depends on how many rows
    were drawn before it, on which block it landed in, or on how many blocks
    took the degenerate guard and skipped their draws entirely.
    """
    var h = splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _MVS_DOMAIN)
    return splitmix64(h ^ (UInt64(tree_index & 0x7FFFFFFFFFFFFFFF) * GOLDEN))


def _row_magnitude_squared(
    gradients: List[Float64], row: Int, n_outputs: Int
) -> Float64:
    """`sum_k der[row][k]^2` over a row-major flattened gradient buffer.

    Row-major with stride `n_outputs` (`gradients[row * n_outputs + k]`) is the
    layout `boosting` already uses for multiclass raw scores and probabilities.
    At `n_outputs == 1` this is one load and one multiply.
    """
    var acc = 0.0
    var base = row * n_outputs
    for k in range(n_outputs):
        var d = gradients[base + k]
        acc += d * d
    return acc


def mvs_auto_lambda_from_gradients(
    gradients: List[Float64], n_rows: Int, n_outputs: Int = 1
) raises -> Float64:
    """The derived `mvs_reg` for the FIRST tree of a fit, when no tree exists
    yet: the squared mean gradient magnitude.

    `TMvsSampler::GetLambda` falls to `CalculateMeanGradValue` when
    `leafValues` is empty, and returns `mean * mean`. CatBoost accumulates that
    mean in per-thread blocks and then sums the block totals, so its value
    moves with `thread_count`; ours accumulates in row order, once, so it does
    not. That is a determinism divergence in our favor and is recorded as one.

    An empty dataset yields 0, which `check_mvs_reg`'s reasoning says is the
    dangerous value -- but a zero-row fit has no block to sample and the
    threshold guard covers it.
    """
    if n_rows < 0:
        raise Error("n_rows must be nonnegative")
    if n_outputs < 1:
        raise Error("n_outputs must be positive")
    if len(gradients) < n_rows * n_outputs:
        raise Error("gradient buffer is shorter than n_rows * n_outputs")
    if n_rows == 0:
        return 0.0
    var total = 0.0
    for r in range(n_rows):
        total += sqrt(_row_magnitude_squared(gradients, r, n_outputs))
    var mean = total / Float64(n_rows)
    return mean * mean


def mvs_auto_lambda_from_leaf_values(
    leaf_values: List[Float64], n_leaves: Int, n_outputs: Int = 1
) raises -> Float64:
    """The derived `mvs_reg` for every tree AFTER the first: the squared mean
    L2 norm of the previous tree's leaf-value vectors.

    `TMvsSampler::GetLambda` prefers `CalculateLastIterMeanLeafValue` whenever
    `leafValues` is non-empty, and what it reads is `leafValues.back()`, the
    last iteration only -- `DoBootstrap` passes `ctx->LearnProgress->LeafValues`,
    which is the whole history. So the lambda a tree samples with is a property
    of the tree before it, which is why this is a separate entry point rather
    than something `resolve_reg` could compute on its own.

    `leaf_values` is row-major over leaves with stride `n_outputs`
    (`leaf_values[leaf * n_outputs + dim]`), matching the gradient layout above
    rather than CatBoost's `[dim][leaf]`; the sum is over the same numbers
    either way, but the summation ORDER differs from theirs and so may the last
    bits. Recorded, not hidden.
    """
    if n_leaves < 0:
        raise Error("n_leaves must be nonnegative")
    if n_outputs < 1:
        raise Error("n_outputs must be positive")
    if len(leaf_values) < n_leaves * n_outputs:
        raise Error("leaf value buffer is shorter than n_leaves * n_outputs")
    if n_leaves == 0:
        return 0.0
    var total = 0.0
    for leaf in range(n_leaves):
        total += sqrt(_row_magnitude_squared(leaf_values, leaf, n_outputs))
    var mean = total / Float64(n_leaves)
    return mean * mean


def _mvs_threshold(
    mut candidates: List[Float64], begin: Int, end: Int, sample_size: Float64
) -> Float64:
    """Solve for the threshold `mu` with `sum_i min(1, g_i/mu) == sample_size`.

    A transcription of `TMvsSampler::CalculateThreshold`, written as a loop
    rather than a recursion. Pivot on the first candidate, partition three ways,
    work out the sample size that pivot implies, and move into the large side
    if it overshoots or the small side if it undershoots; when a side runs out,
    solve the remaining linear equation exactly. Expected O(end - begin).

    `candidates` is permuted in place, which is what makes it O(n) and is what
    CatBoost does too (two `std::partition` passes over its own block buffer).

    **One deliberate difference, and it is the determinism one.** CatBoost
    partitions with `std::partition`, which is not stable, so the order its
    `Accumulate` sums the small side in is an implementation detail of the
    standard library, and floating-point addition is not associative. This uses
    an explicit three-way (Dutch-flag) partition, so the permutation is a fixed
    function of the input order alone -- and the input order is row order. The
    sums are therefore reproducible here in a way they are not there. This does
    NOT preserve relative order; it only has to be deterministic, and it is.

    The two divisions this can return by are the ones the A11 catalog note
    dissects. Neither is guarded here on purpose: this function transcribes the
    solve, and the caller guards the result with `_mvs_threshold_is_usable` so
    there is exactly one place a degenerate threshold is handled.
    """
    var lo = begin
    var hi = end
    var sum_small = 0.0
    var n_large = 0
    while lo < hi:
        var pivot = candidates[lo]
        # Three-way partition: [lo, lt) < pivot, [lt, gt) == pivot,
        # [gt, hi) > pivot. A NaN candidate compares false both ways and lands
        # in the middle, which carries it into the sums and out through the
        # caller's guard rather than silently vanishing.
        var lt = lo
        var i = lo
        var gt = hi
        while i < gt:
            var v = candidates[i]
            if v < pivot:
                candidates[i] = candidates[lt]
                candidates[lt] = v
                lt += 1
                i += 1
            elif v > pivot:
                gt -= 1
                candidates[i] = candidates[gt]
                candidates[gt] = v
            else:
                i += 1

        var sum_small_update = 0.0
        for j in range(lo, lt):
            sum_small_update += candidates[j]
        var n_large_update = hi - gt
        var n_middle = gt - lt
        var sum_middle = Float64(n_middle) * pivot

        var estimated = (sum_small + sum_small_update) / pivot + Float64(
            n_large + n_large_update + n_middle
        )
        if estimated > sample_size:
            # mu must be larger: everything at or below the pivot is "small".
            if gt != hi:
                sum_small += sum_middle + sum_small_update
                lo = gt
            else:
                return (
                    sum_small + sum_small_update + sum_middle
                ) / (sample_size - Float64(n_large))
        else:
            # mu must be smaller: everything at or above the pivot is "large".
            if lt != lo:
                n_large += n_large_update + n_middle
                hi = lt
            else:
                return sum_small / (
                    sample_size
                    - Float64(n_large + n_middle + n_large_update)
                )
    return 0.0


def _mvs_threshold_is_usable(threshold: Float64) -> Bool:
    """Whether a solved threshold can be divided by.

    OURS, and NOT verified from CatBoost source, because there is nothing to
    verify it against: CatBoost does not guard this. See `check_mvs_reg` for
    the full chain, and the A11 catalog note for the second shape this catches
    that refusing `mvs_reg=0` does not -- an exact tie on the boundary can give
    a `0/0` NaN threshold at a perfectly positive lambda.

    Rejecting non-finite covers NaN and both infinities; rejecting
    non-positive covers the negative zero CatBoost's undershoot branch returns,
    since `-0.0 > 0.0` is false.
    """
    return isfinite(threshold) and threshold > 0.0


@fieldwise_init
struct MvsAudit(Copyable, Movable):
    """What one tree's MVS draw actually did, so a test can prove a path was
    taken rather than assume it.

    `blocks_guarded` is the count of blocks that hit
    `_mvs_threshold_is_usable` returning False and kept every row at weight 1.
    It exists because that branch is the one with no CatBoost referent, and a
    test that claims to exercise it has to be able to show it fired. Under the
    default (`mvs_reg` unset, so lambda derived and positive) it should be 0 on
    any dataset with a nonzero gradient, and a nonzero count on a real fit is a
    signal worth chasing rather than a statistic.
    """

    var blocks: Int
    var blocks_guarded: Int
    var rows_kept: Int
    var rows_kept_certainly: Int

    @staticmethod
    def empty() -> MvsAudit:
        return MvsAudit(0, 0, 0, 0)


def mvs_bootstrap_weights(
    params: MvsBootstrapParams,
    gradients: List[Float64],
    n_rows: Int,
    tree_index: Int,
    auto_lambda: Float64,
    mut weights: List[Float64],
    mut audit: MvsAudit,
    n_outputs: Int = 1,
) raises:
    """Fill `weights` with tree number `tree_index`'s MVS draw, one per row,
    and fill `audit` with what the draw did.

    `weights` is resized rather than appended to, so a round loop can hand the
    same buffer back every tree and allocate once for the whole fit.

    A disabled bundle and `subsample == 1.0` both fill exactly 1.0 and take no
    draw at all, which is CatBoost's `if (SampleRate == 1.0f) Fill(..., 1.0f)`
    early return.

    `auto_lambda` is what `resolve_reg` uses when the user left `mvs_reg`
    unset; a caller computes it with `mvs_auto_lambda_from_leaf_values` if any
    tree exists and `mvs_auto_lambda_from_gradients` otherwise, which is the
    branch `TMvsSampler::GetLambda` takes. It is a parameter rather than
    something derived in here because the leaf-value branch needs the previous
    tree, which this module cannot see.

    **Determinism.** Every row's keep decision is `uniform(stream + row)` for a
    stream derived from `(seed, tree_index)` alone, so it is independent of
    worker count, of buffer length, and of how many draws any other row took.
    The threshold is solved per block of `MVS_BLOCK_SIZE`, a constant, over
    candidates written in row order, with a partition whose permutation is a
    fixed function of that order -- so it too is independent of worker count.
    The blocks may be computed in any order or in parallel and the buffer is
    identical.
    """
    params.validate()
    if n_rows < 0:
        raise Error("n_rows must be nonnegative")
    if tree_index < 0:
        raise Error("tree_index must be nonnegative")
    if n_outputs < 1:
        raise Error("n_outputs must be positive")
    weights.clear()
    weights.resize(n_rows, 1.0)
    audit = MvsAudit.empty()
    if not params.samples_rows():
        audit.rows_kept = n_rows
        audit.rows_kept_certainly = n_rows
        return
    if len(gradients) < n_rows * n_outputs:
        raise Error("gradient buffer is shorter than n_rows * n_outputs")

    var lam = params.resolve_reg(auto_lambda)
    if not (lam >= 0.0):
        raise Error("mvs lambda must be nonnegative and not NaN")
    var stream = _mvs_stream(params.seed, tree_index)

    var block_start = 0
    while block_start < n_rows:
        var block_end = block_start + MVS_BLOCK_SIZE
        if block_end > n_rows:
            block_end = n_rows
        var block_size = block_end - block_start
        audit.blocks += 1

        # `thresholdCandidates[idx] = sqrt(lambda + sum_k der^2)`, in row order.
        #
        # Two buffers, not one, and the second is not a copy for tidiness. The
        # solve PERMUTES its input, which destroys the row-to-magnitude
        # mapping, so CatBoost recomputes every magnitude from the derivatives
        # a second time in its weight loop -- a second `sum_k der^2` and a
        # second `sqrt` for every row of every tree. Keeping the row-ordered
        # copy costs 8 bytes per row of one block, `8 * MVS_BLOCK_SIZE` = 64 KiB
        # at the ceiling, and removes `n_rows` square roots and
        # `n_rows * n_outputs` multiply-adds per tree. Strictly less work for
        # the identical result.
        var magnitudes = List[Float64](capacity=block_size)
        for r in range(block_start, block_end):
            magnitudes.append(
                sqrt(_row_magnitude_squared(gradients, r, n_outputs) + lam)
            )
        var candidates = magnitudes.copy()

        var target = params.subsample * Float64(block_size)
        var mu = _mvs_threshold(candidates, 0, block_size, target)

        if not _mvs_threshold_is_usable(mu):
            # OURS: no information to sample on, so keep the block whole. This
            # is the limit of the method as mu falls to zero (every p rises to
            # 1), and it is the only choice that cannot produce a silently
            # empty tree. CatBoost drops every row here instead.
            audit.blocks_guarded += 1
            for r in range(block_start, block_end):
                weights[r] = 1.0
            audit.rows_kept += block_size
            audit.rows_kept_certainly += block_size
            block_start = block_end
            continue

        for r in range(block_start, block_end):
            var g = magnitudes[r - block_start]
            var p: Float64
            if g > mu:
                p = 1.0
            else:
                p = g / mu
            if p > _MVS_PROBABILITY_EPS:
                if p >= 1.0:
                    # Certain keep. CatBoost still burns a draw here; ours does
                    # not need to, because the stream is keyed by row and skipping
                    # a row's draw cannot shift any other row's.
                    weights[r] = 1.0
                    audit.rows_kept += 1
                    audit.rows_kept_certainly += 1
                elif uniform(stream + UInt64(r)) < p:
                    weights[r] = 1.0 / p
                    audit.rows_kept += 1
                else:
                    weights[r] = 0.0
            else:
                weights[r] = 0.0
        block_start = block_end


def refresh_mvs_bootstrap(
    mut weights: List[Float64],
    mut audit: MvsAudit,
    params: MvsBootstrapParams,
    gradients: List[Float64],
    base_weight: List[Float64],
    n_rows: Int,
    tree_index: Int,
    auto_lambda: Float64,
    n_outputs: Int = 1,
) raises:
    """The round's effective per-row weight: the tree's MVS draw times the
    user's own `sample_weight`.

    This is CatBoost's `CalcWeightedData` tail, `SampleWeights[i] *=
    learnWeights[i]`, and it is why a sampled fit is a *weighted* fit as far as
    everything downstream is concerned. An empty `base_weight` is the
    "unweighted" convention used everywhere else here and leaves the draw
    alone.

    A disabled bundle leaves `weights` empty rather than filling it with ones,
    exactly as `refresh_bayesian_bootstrap` does, so a caller can pass it
    straight through as the fit's `sample_weight` and an unsampled, unweighted
    fit stays exactly unweighted -- no vector of 1.0s that would make
    `boosting.round_has_constant_hessian` refuse a fit it should admit, and no
    bits moved on the default path.
    """
    if len(base_weight) != 0 and len(base_weight) != n_rows:
        raise Error("sample_weight length must match the row count")
    if not params.enabled:
        params.validate()
        audit = MvsAudit.empty()
        weights.clear()
        if len(base_weight) != 0:
            weights.resize(n_rows, 1.0)
            for r in range(n_rows):
                weights[r] = base_weight[r]
        return
    mvs_bootstrap_weights(
        params,
        gradients,
        n_rows,
        tree_index,
        auto_lambda,
        weights,
        audit,
        n_outputs,
    )
    if len(base_weight) != 0:
        for r in range(n_rows):
            weights[r] = weights[r] * base_weight[r]


def mvs_kept_rows(weights: List[Float64]) raises -> List[Int]:
    """The ascending, duplicate-free row set a set of MVS weights implies.

    A zero weight is a *dropped* row, not merely a down-weighted one: CatBoost
    sets `performRandomChoice = false` for MVS alone and
    `TCalcScoreFold::Sample` then calls `SetControlNoZeroWeighted`, which
    compacts the zero-weight rows out of the score fold so the split search
    never visits them. This is the mojotrees statement of that, in the ascending
    row-set form the rest of this module and the device path already take
    (`check_row_set`, `contiguous_ranges`, `row_mask`).

    **No timing claim attaches to this.** A sampler that visits fewer rows
    looks faster for reasons that have nothing to do with whether the sampler
    is any good, and this lane measured nothing.
    """
    var rows = List[Int]()
    for r in range(len(weights)):
        if weights[r] != 0.0:
            rows.append(r)
    return rows^


@fieldwise_init
struct BootstrapParams(Copyable, Movable):
    """`bootstrap_type` as one bundle, so a trainer takes one argument.

    CatBoost's `bootstrap_type` is a single enum and cannot name two samplers
    at once; ours is two independent bundles and could, so this pairs them and
    `validate` runs `check_bootstrap_type_exclusive` over the pair. Every
    trainer that samples rows takes this rather than the two bundles, which is
    what makes the exclusivity impossible to forget at a call site.

    The mapping from CatBoost's option surface:

    - `bootstrap_type=No` (and mojotrees's default) is `disabled()`: both
      bundles off, no draw taken, no weight vector materialized, and every fit
      byte for byte the fit it is today.
    - `bootstrap_type=MVS` is `mvs_at`, and `subsample` is its rate. CatBoost's
      CPU default for both objectives this project benchmarks, at
      `subsample=0.8` (`DEFAULT_MVS_SUBSAMPLE`).
    - `bootstrap_type=Bayesian` is `bayesian_at`, and `bagging_temperature` is
      its knob.
    - `bootstrap_type=Bernoulli` is `bagging.BaggingParams` under mojotrees's
      own name and is not here at all; `canonical_bootstrap_type` refuses the
      spelling by name rather than as an unknown value.

    **Both samplers put a per-row weight on the round's derivatives**, so a fit
    with either on has a per-row hessian. `varies_hessian` states that and
    `check_hessian_declaration` refuses a constant-hessian declaration beside
    it; a round loop must call the second one after it has decided, which is
    the whole point of the pair existing (see `mvs_varies_hessian` and
    `bayesian_bootstrap_varies_hessian` for the argument, and
    `boosting.round_has_constant_hessian` for why the guard cannot live there).
    """

    var mvs: MvsBootstrapParams
    var bayesian: BayesianBootstrapParams

    @staticmethod
    def disabled() -> BootstrapParams:
        """`bootstrap_type=No`: neither sampler, which is the library
        default and the arm every existing fit is on."""
        return BootstrapParams(
            MvsBootstrapParams.disabled(), BayesianBootstrapParams.disabled()
        )

    @staticmethod
    def mvs_at(
        subsample: Float64 = DEFAULT_MVS_SUBSAMPLE,
        seed: Int = DEFAULT_BOOTSTRAP_SEED,
    ) -> BootstrapParams:
        """`bootstrap_type=MVS` with `mvs_reg` left unset, so the lambda is
        derived from the data every tree exactly as CatBoost's is. This is
        CatBoost's own CPU default configuration."""
        return BootstrapParams(
            MvsBootstrapParams.enable(subsample, seed),
            BayesianBootstrapParams.disabled(),
        )

    @staticmethod
    def mvs_with_reg(
        reg: Float64,
        subsample: Float64 = DEFAULT_MVS_SUBSAMPLE,
        seed: Int = DEFAULT_BOOTSTRAP_SEED,
    ) -> BootstrapParams:
        """`bootstrap_type=MVS` with an explicit `mvs_reg`. `validate` refuses
        exactly 0; `check_mvs_reg` says why at length."""
        return BootstrapParams(
            MvsBootstrapParams.enable_with_reg(reg, subsample, seed),
            BayesianBootstrapParams.disabled(),
        )

    @staticmethod
    def bayesian_at(
        temperature: Float64 = DEFAULT_BAGGING_TEMPERATURE,
        seed: Int = DEFAULT_BOOTSTRAP_SEED,
    ) -> BootstrapParams:
        """`bootstrap_type=Bayesian` at `bagging_temperature`."""
        return BootstrapParams(
            MvsBootstrapParams.disabled(),
            BayesianBootstrapParams.enable(temperature, seed),
        )

    def enabled(self) -> Bool:
        """Whether any bootstrap is configured. Deliberately coarser than
        `MvsBootstrapParams.samples_rows` and
        `BayesianBootstrapParams.draws_weights`: those two answer "will this
        draw move a number", and this one answers "is a sampler configured",
        which is the question every fit-lifetime decision has to ask."""
        return self.mvs.enabled or self.bayesian.enabled

    def validate(self) raises:
        """One `bootstrap_type` at a time, and each bundle's own ranges."""
        check_bootstrap_type_exclusive(self.mvs, self.bayesian)
        self.mvs.validate()
        self.bayesian.validate()

    def varies_hessian(self) -> Bool:
        """Whether a fit configured this way has a per-row hessian, so that
        `histogram.CONSTANT_HESSIAN` must not be declared for it. The union of
        `mvs_varies_hessian` and `bayesian_bootstrap_varies_hessian`, both of
        which test `enabled` alone and for the reason they each give."""
        return mvs_varies_hessian(self.mvs) or (
            bayesian_bootstrap_varies_hessian(self.bayesian)
        )

    def check_hessian_declaration(self, const_hessian: Bool) raises:
        """Refuse a constant-hessian declaration beside an active bootstrap.

        Both twins, in one call. A round loop calls this once per fit **after**
        it has computed its declaration, so the failure mode of forgetting the
        exclusion is an exception at fit setup rather than a hessian plane
        rebuilt from the row count while the true hessian is the draw.
        """
        check_mvs_hessian_declaration(self.mvs, const_hessian)
        check_bayesian_bootstrap_hessian_declaration(
            self.bayesian, const_hessian
        )


def check_bootstrap_honored(params: BootstrapParams, where: String) raises:
    """Refuse an active bootstrap on an entry point that does not run one.

    The shape `efb.check_bundling_honored` and
    `ordered_boosting.check_ordered_honored` already take, and it is here for
    the same reason: a trainer that accepts a sampler bundle and never draws
    from it trains an unsampled model and reports a sampled one. Every trainer
    that does not thread `bootstrap_round` into its round loop calls this
    instead of ignoring the bundle.
    """
    if params.enabled():
        raise Error(
            "bootstrap_type is not implemented by ",
            where,
            ": the MVS and Bayesian draws are per-round work and this entry"
            " point's loop does not call sampling.bootstrap_round, so the fit"
            " would be unsampled. Use boosting.train, boosting.train_more or"
            " boosting.train_with_valid, or drop bootstrap_type",
        )


def apply_bootstrap_weights(
    mut grad: List[Float64],
    mut hess: List[Float64],
    weights: List[Float64],
    n_rows: Int,
    n_outputs: Int = 1,
) raises:
    """Multiply one round's derivatives by the drawn per-row weight, in place.

    This is CatBoost's `CalcWeightedData`, first line:

        sampleWeightedDerivativesData[z] =
            weightedDerivativesData[z] * sampleWeightsData[z]

    with two things worth being exact about, because getting either wrong is a
    silently wrong model rather than an error.

    **`weights` is the DRAW alone, not the draw times the user's
    `sample_weight`.** CatBoost multiplies its *already user-weighted*
    derivatives by the raw draw and only afterwards folds the user's weights
    into `SampleWeights` (`SampleWeights[i] *= learnWeights[i]`). A caller here
    has already filled `grad` and `hess` through
    `boosting._fill_grad_hess(..., sample_weight, ...)`, so the user's weight
    is in the derivatives; passing the product would square it.

    **The product CatBoost keeps in `SampleWeights` is `hess`.** Under an
    objective whose unweighted hessian is the literal 1.0, this leaves
    `hess[r] = draw[r] * sample_weight[r]`, which is exactly the vector
    CatBoost's leaf denominators sum -- so no second weight vector has to be
    carried, and the fit's leaf values and `min_child_hess` counts are the
    weighted ones by construction. It is also precisely why a bootstrapped fit
    must not declare a constant hessian.

    Every row is scaled by its own weight and reads nothing else, so any
    partition of the row range across workers produces the identical buffers.
    """
    if n_rows < 0:
        raise Error("n_rows must be nonnegative")
    if n_outputs < 1:
        raise Error("n_outputs must be positive")
    if len(weights) != n_rows:
        raise Error("bootstrap weight length must match the row count")
    var need = n_rows * n_outputs
    if len(grad) != need or len(hess) != need:
        raise Error(
            "gradient and hessian buffers must hold n_rows * n_outputs entries"
        )
    for r in range(n_rows):
        var w = weights[r]
        var base = r * n_outputs
        for k in range(n_outputs):
            grad[base + k] = grad[base + k] * w
            hess[base + k] = hess[base + k] * w


def bootstrap_round(
    mut rows: List[Int],
    mut grad: List[Float64],
    mut hess: List[Float64],
    mut weights: List[Float64],
    mut audit: MvsAudit,
    params: BootstrapParams,
    n_rows: Int,
    tree_index: Int,
    auto_lambda: Float64,
    n_outputs: Int = 1,
) raises:
    """Run one round's bootstrap: draw the per-row weights, scale the round's
    derivatives by them, and hand back the row set the split search should see.

    Deliberately the same shape as `goss.goss_round`, and it sits in the same
    place in a round loop, immediately after the gradient fill: both are row
    samplers that read this round's derivatives, rewrite them, and may replace
    the row list. A disabled bundle validates and returns having touched
    nothing at all -- not `rows`, not `grad`, not `hess`, and it leaves
    `weights` **empty** rather than filled with 1.0s, which is the convention
    `refresh_bayesian_bootstrap` and `refresh_mvs_bootstrap` already state and
    matters because a vector of ones handed on as a `sample_weight` would cost
    an unbootstrapped fit its constant-hessian specialization for nothing.

    Ordering inside a round, which is not free to move:

    1. The caller fills `grad`/`hess` from the raw scores and the user's
       `sample_weight`. Those are CatBoost's `WeightedDerivatives`.
    2. The caller computes `random_strength`'s per-tree scale from `grad` if it
       wants one. CatBoost's `CalcScoreStDev` reads `WeightedDerivatives`, and
       `Bootstrap` writes `SampleWeightedDerivatives`, so the scale is taken
       **before** this call and is not affected by the draw.
    3. This call. MVS's row magnitudes are read from `grad` as it arrives, for
       the same reason: `Bootstrap` is handed the pre-bootstrap derivatives.
    4. Growth, on the rewritten `grad`/`hess` and the returned `rows`.

    `auto_lambda` is what `MvsBootstrapParams.resolve_reg` uses when the user
    left `mvs_reg` unset, and it is a parameter because the branch
    `TMvsSampler::GetLambda` takes needs the previous tree: the caller passes
    `mvs_auto_lambda_from_gradients(grad, n_rows, n_outputs)` on the first tree
    of the ensemble and `mvs_auto_lambda_from_leaf_values(...)` on every tree
    after it. It is ignored entirely when the bundle is Bayesian or when
    `mvs_reg` was set explicitly.

    **Rows.** MVS is a row *dropper* (`SetControlNoZeroWeighted`), so its
    zero-weight rows are compacted out into an ascending row list, which is
    what makes `min_data_in_leaf` count the rows the tree was actually fitted
    on. The Bayesian bootstrap keeps every row and never touches `rows`. Two
    refusals guard the compaction rather than letting it go quietly wrong:
    `rows` must arrive empty under MVS, because an empty list means "every row"
    everywhere in mojotrees and silently intersecting a bag with a draw would
    be a third sampler nobody asked for; and a draw that kept no row at all is
    refused rather than written back as an empty list, which would invert into
    "every row" and train a full-data tree while reporting a sampled one.

    **Determinism.** Nothing here reads a worker count, a block layout, or a
    running counter: every weight is a function of `(seed, tree_index, row)`
    alone (`_mvs_stream`, `_bootstrap_stream`), the MVS threshold is solved per
    fixed-size block over candidates in row order, and the scaling is
    elementwise. The buffers are identical at any `MOJOTREES_NUM_WORKERS` and
    on any machine.
    """
    params.validate()
    if not params.enabled():
        weights.clear()
        audit = MvsAudit.empty()
        return
    if n_rows < 0:
        raise Error("n_rows must be nonnegative")
    if n_outputs < 1:
        raise Error("n_outputs must be positive")

    # The draw alone: an empty `base_weight` is the "unweighted" convention,
    # and it is the right one here because the caller's `sample_weight` is
    # already inside `grad` and `hess`. See `apply_bootstrap_weights`.
    var no_base = List[Float64]()
    if params.bayesian.enabled:
        audit = MvsAudit.empty()
        refresh_bayesian_bootstrap(
            weights, params.bayesian, no_base, n_rows, tree_index
        )
    else:
        if len(rows) != 0:
            raise Error(
                "mvs bootstrap owns the round's row set and cannot compose"
                " with another row sampler: it drops its own zero-weight rows,"
                " so a bag arriving here would be silently intersected with a"
                " draw that never saw it"
            )
        refresh_mvs_bootstrap(
            weights,
            audit,
            params.mvs,
            grad,
            no_base,
            n_rows,
            tree_index,
            auto_lambda,
            n_outputs,
        )
    apply_bootstrap_weights(grad, hess, weights, n_rows, n_outputs)
    if not params.mvs.enabled:
        return
    var kept = mvs_kept_rows(weights)
    if len(kept) == 0 and n_rows > 0:
        raise Error(
            "the mvs draw kept no rows at all. An empty row list means 'every"
            " row' everywhere in mojotrees, so this cannot be handed on;"
            " leave mvs_reg unset so the lambda is derived from the data, or"
            " raise subsample"
        )
    if len(kept) != n_rows:
        rows = kept^


def canonical_bootstrap_type(value: String) raises -> String:
    """The `bootstrap_type` value one spelling selects.

    `bootstrap_type` is CatBoost's parameter and keeps CatBoost's name
    (docs/PARAMETER_NAMING.md); no other vendor has one. Values are
    canonical lowercase here, as in `device_policy.parse_device`: CatBoost
    writes `Bayesian` and `MVS`, and the surface the user types at folds
    the case before calling in (`params._lower_ascii`, and `.lower()` on the
    Python side).

    CatBoost's `No`, `Bayesian` and `MVS` are implemented. `Bernoulli` is
    refused by name rather than as an unknown value because mojotrees already
    has it under another name -- it is row bagging, `subsample` with
    `subsample_freq`, which is the same draw CatBoost's `subsample` makes --
    and telling a user that a real bootstrap type is unknown would be
    misleading. `Poisson` is refused the same way; CatBoost itself refuses
    Poisson on the CPU.
    """
    if value == "no" or value == "none":
        return "no"
    if value == "bayesian":
        return "bayesian"
    if value == "mvs":
        return "mvs"
    if value == "bernoulli":
        raise Error(
            "bootstrap_type 'bernoulli' is row bagging under another name;"
            " use subsample with subsample_freq"
        )
    if value == "poisson":
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
