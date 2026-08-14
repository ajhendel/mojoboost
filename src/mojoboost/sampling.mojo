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
feature set depends on how many draws happened before it. mojoboost derives
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
"""

# LightGBM's feature_fraction_seed default.
comptime DEFAULT_FEATURE_FRACTION_SEED = 2

# Distinguishes the once-per-tree draw from the per-node draws, which are
# tagged with node id + 1.
comptime _TREE_TAG = 0


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    """Uniform in [0, 1) with 53 significant bits, from a counter value."""
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _stream(seed: Int, tree_index: Int, tag: Int) -> UInt64:
    """Start of the counter stream for one selection. Sign bits are masked
    off so negative seeds are accepted (as in LightGBM) without relying on
    signed-to-unsigned conversion."""
    var h = _splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF))
    h = _splitmix64(h ^ UInt64(tree_index & 0x7FFFFFFFFFFFFFFF))
    return _splitmix64(h ^ UInt64(tag & 0x7FFFFFFFFFFFFFFF))


def check_feature_fraction(fraction: Float64, name: String) raises:
    """Validate one fraction; LightGBM's accepted range is (0, 1]."""
    if not (fraction > 0.0 and fraction <= 1.0):
        raise Error(String(name, " must be in (0, 1]"))


def check_feature_fractions(
    feature_fraction: Float64, feature_fraction_bynode: Float64
) raises:
    check_feature_fraction(feature_fraction, "feature_fraction")
    check_feature_fraction(feature_fraction_bynode, "feature_fraction_bynode")


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
        if _uniform(stream + UInt64(i)) < prob:
            out.append(pool[i])
            chosen += 1
    return out^


def select_tree_features(
    n_features: Int, fraction: Float64, seed: Int, tree_index: Int
) raises -> List[Int]:
    """The ascending feature ids one tree may split on."""
    check_feature_fraction(fraction, "feature_fraction")
    if n_features < 1:
        raise Error("n_features must be positive")
    var all_features = List[Int](capacity=n_features)
    for f in range(n_features):
        all_features.append(f)
    if fraction >= 1.0:
        return all_features^
    return sample_without_replacement(
        all_features,
        selection_count(n_features, fraction),
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
    check_feature_fraction(fraction, "feature_fraction_bynode")
    if fraction >= 1.0:
        return tree_features.copy()
    return sample_without_replacement(
        tree_features,
        selection_count(len(tree_features), fraction),
        _stream(seed, tree_index, node + 1),
    )
