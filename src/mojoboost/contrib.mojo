"""Exact per-feature contributions (TreeSHAP).

`predict(..., pred_contrib=True)` in the Python API lands here. For one
example this returns one number per feature plus an expected-value column,
and those numbers sum to the raw model score. That is not an approximation
and not a split-gain heuristic: the numbers are the Shapley values of the
set function

    v(S) = E[ f(x) | x_S ]

where the conditional expectation is taken over the tree's own node covers
(see `Tree.count`), which is the "path-dependent" or "tree_path_dependent"
convention LightGBM, XGBoost, and the reference `shap` package all use for
a model without a supplied background dataset.

The algorithm is Algorithm 2 of Lundberg, Erion, and Lee, "Consistent
Individualized Feature Attribution for Tree Ensembles" (arXiv:1802.03888),
the polynomial-time recursion that carries a set of "unique paths" down the
tree and extends, unwinds, and re-extends them. It runs in O(L * D^2) per
tree for L leaves and depth D instead of the O(2^M) an enumeration over
feature subsets costs, and it returns the same numbers; `tests/
test_contrib.mojo` checks that against a subset-enumeration reference.

What makes it exact here
------------------------
Two Shapley properties do the work, and both are checked in the tests.

- *Efficiency*: the contributions plus the expected value sum to the raw
  score, for every row, exactly (up to floating-point summation). This holds
  whatever the node covers are, as long as they are positive, because
  v(all features) is the score of the path the row actually takes and
  v(no features) is the cover-weighted mean of the leaf values.
- *Symmetry and linearity*: contributions add across trees, so an ensemble
  is explained by summing its trees' contributions. The learning rate scales
  each tree's contribution exactly as it scales that tree's output.

Routing, not thresholds
-----------------------
The recursion only ever asks which child a row takes, which it gets from
`Tree.goes_left`. So numerical splits, missing-value routing, and
categorical set membership are all handled by construction, with no separate
code path: a categorical node's "hot" child is the one the row's category
routes to, exactly as in prediction. A feature split on repeatedly along one
root-to-leaf path is handled by the unwind step, which removes the earlier
occurrence before re-extending with the new fractions.

Layout
------
For a single-output model the result has `n_features + 1` entries, the last
being the expected value, which is LightGBM's layout for regression and
binary classification. For a multiclass model the result has
`n_classes * (n_features + 1)` entries in class-major blocks: entry
`k * (n_features + 1) + f` is feature f's contribution to class k's raw
score, and `k * (n_features + 1) + n_features` is class k's expected value.
That is LightGBM's multiclass width and block order.

Iteration ranges
----------------
Every entry point takes an `IterationRange` and explains exactly the
iterations it selects, with the base score belonging to iteration 0 as it
does everywhere else (see boosting.mojo). So the contributions of `[0, k)`
and `[k, n)` add up to the contributions of the whole model, feature by
feature, and each range's own contributions sum to that range's raw score.
"""

from .boosting import Booster, IterationRange, MulticlassBooster
from .model import Model, MulticlassModel
from .tree import Tree


struct _PathBuffer(Movable):
    """Scratch for the "unique path" the recursion carries down the tree.

    Four parallel arrays rather than a list of path-element structs: each
    recursive step copies its parent's path prefix, and flat arrays keep
    that a plain element-wise copy with no per-element construction.

    One buffer serves a whole tree. A call at node depth d owns the slice
    beginning at `1 + d(d+1)/2` and never writes below it, so a parent's
    path survives both of its children; `for_depth` sizes the buffer for the
    deepest node any tree in the ensemble has.
    """

    var feature_index: List[Int]
    var zero_fraction: List[Float64]
    var one_fraction: List[Float64]
    var pweight: List[Float64]

    def __init__(out self, max_depth: Int):
        # (max_depth + 2)(max_depth + 3)/2 is the triangular bound on the
        # slice offsets, with room to spare for the deepest slice itself.
        var span = max_depth + 2
        var n = span * (span + 1) // 2
        self.feature_index = List[Int](capacity=n)
        self.feature_index.resize(n, -1)
        self.zero_fraction = List[Float64](capacity=n)
        self.zero_fraction.resize(n, 0.0)
        self.one_fraction = List[Float64](capacity=n)
        self.one_fraction.resize(n, 0.0)
        self.pweight = List[Float64](capacity=n)
        self.pweight.resize(n, 0.0)


def _extend_path(
    mut path: _PathBuffer,
    base: Int,
    unique_depth: Int,
    zero_fraction: Float64,
    one_fraction: Float64,
    feature_index: Int,
):
    """Add one feature to the path, keeping the weight of every subset size.

    `pweight[i]` is the proportion of subsets of the path's features that are
    of size i and are "on" (present in the conditioning set), so extending
    has to redistribute weight between neighbouring sizes."""
    path.feature_index[base + unique_depth] = feature_index
    path.zero_fraction[base + unique_depth] = zero_fraction
    path.one_fraction[base + unique_depth] = one_fraction
    path.pweight[base + unique_depth] = 1.0 if unique_depth == 0 else 0.0
    var span = Float64(unique_depth + 1)
    for i in range(unique_depth - 1, -1, -1):
        path.pweight[base + i + 1] += (
            one_fraction * path.pweight[base + i] * Float64(i + 1) / span
        )
        path.pweight[base + i] = (
            zero_fraction
            * path.pweight[base + i]
            * Float64(unique_depth - i)
            / span
        )


def _unwind_path(
    mut path: _PathBuffer, base: Int, unique_depth: Int, path_index: Int
):
    """Undo `_extend_path` for the feature at `path_index`, then close the
    gap. This is what lets a feature be split on more than once along a
    root-to-leaf path: the earlier occurrence is removed and the node
    re-extends with the fractions the deeper split implies."""
    var one_fraction = path.one_fraction[base + path_index]
    var zero_fraction = path.zero_fraction[base + path_index]
    var next_one_portion = path.pweight[base + unique_depth]
    var span = Float64(unique_depth + 1)
    # `_extend_path` wrote size i + 1's share out of size i, so undoing it
    # runs from the top down and recovers each size's weight in place.
    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.0:
            var tmp = path.pweight[base + i]
            path.pweight[base + i] = (
                next_one_portion * span / (Float64(i + 1) * one_fraction)
            )
            next_one_portion = tmp - (
                path.pweight[base + i]
                * zero_fraction
                * Float64(unique_depth - i)
                / span
            )
        else:
            path.pweight[base + i] = (
                path.pweight[base + i]
                * span
                / (zero_fraction * Float64(unique_depth - i))
            )
    for i in range(path_index, unique_depth):
        path.feature_index[base + i] = path.feature_index[base + i + 1]
        path.zero_fraction[base + i] = path.zero_fraction[base + i + 1]
        path.one_fraction[base + i] = path.one_fraction[base + i + 1]


def _unwound_path_sum(
    path: _PathBuffer, base: Int, unique_depth: Int, path_index: Int
) -> Float64:
    """The total weight the path would carry with the feature at
    `path_index` removed, without actually removing it. This is the Shapley
    weight a leaf contributes to that feature."""
    var one_fraction = path.one_fraction[base + path_index]
    var zero_fraction = path.zero_fraction[base + path_index]
    var next_one_portion = path.pweight[base + unique_depth]
    var span = Float64(unique_depth + 1)
    var total = 0.0
    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.0:
            var tmp = next_one_portion * span / (Float64(i + 1) * one_fraction)
            total += tmp
            next_one_portion = path.pweight[base + i] - (
                tmp * zero_fraction * Float64(unique_depth - i) / span
            )
        elif zero_fraction != 0.0:
            total += (
                (path.pweight[base + i] / zero_fraction)
                * span
                / Float64(unique_depth - i)
            )
    return total


def _tree_shap(
    tree: Tree,
    bins: List[Int],
    mut phi: List[Float64],
    offset: Int,
    node: Int,
    unique_depth: Int,
    mut path: _PathBuffer,
    parent_base: Int,
    parent_zero_fraction: Float64,
    parent_one_fraction: Float64,
    parent_feature_index: Int,
    scale: Float64,
):
    """Accumulate `scale` times this subtree's contributions into
    `phi[offset ..]`. Recursion follows both children, the one the row takes
    ("hot") carrying the row's own conditioning and the other ("cold")
    carrying none."""
    # This call's slice sits just past the parent's, so the parent's path is
    # still intact when the second child runs.
    var base = parent_base + unique_depth + 1
    for i in range(unique_depth + 1):
        path.feature_index[base + i] = path.feature_index[parent_base + i]
        path.zero_fraction[base + i] = path.zero_fraction[parent_base + i]
        path.one_fraction[base + i] = path.one_fraction[parent_base + i]
        path.pweight[base + i] = path.pweight[parent_base + i]
    _extend_path(
        path,
        base,
        unique_depth,
        parent_zero_fraction,
        parent_one_fraction,
        parent_feature_index,
    )

    var split_feature = tree.feature[node]
    if split_feature < 0:
        # A leaf pays out: each feature on the path earns the leaf value
        # times the weight of the subsets that distinguish it. Index 0 holds
        # the root's placeholder feature (-1) and is skipped, which is also
        # why a depth-zero tree attributes nothing to any feature and shows
        # up entirely in the expected-value column.
        for i in range(1, unique_depth + 1):
            var w = _unwound_path_sum(path, base, unique_depth, i)
            phi[offset + path.feature_index[base + i]] += (
                w
                * (path.one_fraction[base + i] - path.zero_fraction[base + i])
                * tree.value[node]
                * scale
            )
        return

    # Which child the row takes. `goes_left` is the same routing prediction
    # uses, so numerical thresholds, missing bins, and categorical sets all
    # arrive here already resolved.
    var hot: Int
    var cold: Int
    if tree.goes_left(node, bins[split_feature]):
        hot = tree.left[node]
        cold = tree.right[node]
    else:
        hot = tree.right[node]
        cold = tree.left[node]

    var cover = tree.count[node]
    var hot_zero = tree.count[hot] / cover
    var cold_zero = tree.count[cold] / cover
    var incoming_zero = 1.0
    var incoming_one = 1.0

    # If this feature is already on the path, undo that split so this node
    # can redo it; the deeper split is the binding one.
    var depth = unique_depth
    var path_index = 0
    while path_index <= unique_depth:
        if path.feature_index[base + path_index] == split_feature:
            break
        path_index += 1
    if path_index <= unique_depth:
        incoming_zero = path.zero_fraction[base + path_index]
        incoming_one = path.one_fraction[base + path_index]
        _unwind_path(path, base, unique_depth, path_index)
        depth -= 1

    _tree_shap(
        tree,
        bins,
        phi,
        offset,
        hot,
        depth + 1,
        path,
        base,
        hot_zero * incoming_zero,
        incoming_one,
        split_feature,
        scale,
    )
    _tree_shap(
        tree,
        bins,
        phi,
        offset,
        cold,
        depth + 1,
        path,
        base,
        cold_zero * incoming_zero,
        0.0,
        split_feature,
        scale,
    )


def tree_expected_value(tree: Tree) raises -> Float64:
    """The tree's output averaged over its node covers, which is `v({})`:
    what the tree predicts when nothing about the row is known.

    Every leaf's cover fraction along its path telescopes to
    `count[leaf] / count[root]`, so this is one pass over the leaves rather
    than a traversal. A single-leaf tree returns that leaf's value."""
    if len(tree.count) != len(tree.feature) or not tree.count[0] > 0.0:
        raise Error(
            "tree carries no node counts, so it has no expected value; see"
            " Tree.check_node_counts"
        )
    var total = 0.0
    for i in range(len(tree.feature)):
        if tree.feature[i] < 0:
            total += tree.value[i] * tree.count[i]
    return total / tree.count[0]


def tree_contrib_into(
    tree: Tree,
    bins: List[Int],
    mut phi: List[Float64],
    offset: Int,
    mut path: _PathBuffer,
    scale: Float64,
):
    """Add `scale` times one tree's exact contributions into `phi[offset ..]`,
    leaving the expected-value column alone. The caller owns the path buffer
    so a batch of rows allocates it once."""
    _tree_shap(tree, bins, phi, offset, 0, 0, path, 0, 1.0, 1.0, -1, scale)


def _max_depth(trees: List[Tree], start: Int, stop: Int) -> Int:
    var best = 0
    for i in range(start, stop):
        var d = trees[i].depth()
        if d > best:
            best = d
    return best


struct ContribExplainer(Movable):
    """Exact TreeSHAP bound to one fitted ensemble: build it once, explain
    as many rows as you like.

    What it holds is a property of the model, not of any row: the path
    scratch sized for the ensemble's deepest tree, and the feature count that
    fixes the output width. Building it also checks, once, that every tree
    carries the node covers the algorithm conditions on, so a model loaded
    from a file written before covers were recorded fails here with a clear
    message rather than dividing by zero somewhere in the recursion.
    """

    var n_features: Int
    var n_classes: Int
    var path: _PathBuffer

    def __init__(
        out self, trees: List[Tree], n_features: Int, n_classes: Int
    ) raises:
        if n_features < 1:
            raise Error("contributions need at least one feature")
        for i in range(len(trees)):
            trees[i].check_node_counts()
        self.n_features = n_features
        self.n_classes = n_classes
        self.path = _PathBuffer(_max_depth(trees, 0, len(trees)))

    @staticmethod
    def for_booster(
        booster: Booster, n_features: Int
    ) raises -> ContribExplainer:
        return ContribExplainer(booster.trees, n_features, 1)

    @staticmethod
    def for_multiclass(
        booster: MulticlassBooster, n_features: Int
    ) raises -> ContribExplainer:
        return ContribExplainer(
            booster.trees, n_features, booster.n_classes
        )

    @always_inline
    def width(self) -> Int:
        """Entries one row's contributions occupy: `n_features + 1` per
        class, so `n_features + 1` for a single-output model."""
        return self.n_classes * (self.n_features + 1)

    def contrib_bins_into(
        mut self,
        booster: Booster,
        bins: List[Int],
        mut out: List[Float64],
        offset: Int,
        rng: IterationRange,
    ) raises:
        """Write one row's contributions into `out[offset ..offset + width)`,
        overwriting whatever was there.

        The expected-value column collects the base score (when the range
        includes iteration 0) plus each selected tree's own expected value,
        and the feature columns collect the same trees' Shapley values, both
        scaled by the learning rate exactly as prediction scales tree
        outputs. So the entries sum to `predict_raw_bins_range` for this row
        and range."""
        for i in range(self.n_features + 1):
            out[offset + i] = 0.0
        if rng.includes_base():
            out[offset + self.n_features] = booster.base_score
        for i in range(rng.start, rng.stop):
            out[offset + self.n_features] += (
                booster.learning_rate * tree_expected_value(booster.trees[i])
            )
            tree_contrib_into(
                booster.trees[i],
                bins,
                out,
                offset,
                self.path,
                booster.learning_rate,
            )

    def contrib_bins_multiclass_into(
        mut self,
        booster: MulticlassBooster,
        bins: List[Int],
        mut out: List[Float64],
        offset: Int,
        rng: IterationRange,
    ) raises:
        """Write one row's per-class contributions into
        `out[offset ..offset + width)`, class-major: class k occupies
        `offset + k * (n_features + 1)` onward. Each class's block sums to
        that class's raw score over `rng`."""
        var stride = self.n_features + 1
        for i in range(self.width()):
            out[offset + i] = 0.0
        for k in range(self.n_classes):
            var base = offset + k * stride
            if rng.includes_base():
                out[base + self.n_features] = booster.base_scores[k]
            for i in range(rng.start, rng.stop):
                ref tree = booster.trees[i * self.n_classes + k]
                out[base + self.n_features] += (
                    booster.learning_rate * tree_expected_value(tree)
                )
                tree_contrib_into(
                    tree, bins, out, base, self.path, booster.learning_rate
                )


def predict_contrib_bins(
    booster: Booster, bins: List[Int], n_features: Int, rng: IterationRange
) raises -> List[Float64]:
    """One example's exact contributions from its per-feature bin ids,
    length `n_features + 1` with the expected value last."""
    var explainer = ContribExplainer.for_booster(booster, n_features)
    var out = List[Float64](capacity=explainer.width())
    out.resize(explainer.width(), 0.0)
    explainer.contrib_bins_into(booster, bins, out, 0, rng)
    return out^


def predict_contrib_bins_multiclass(
    booster: MulticlassBooster,
    bins: List[Int],
    n_features: Int,
    rng: IterationRange,
) raises -> List[Float64]:
    """One example's exact per-class contributions from its bin ids, length
    `n_classes * (n_features + 1)` in class-major blocks."""
    var explainer = ContribExplainer.for_multiclass(booster, n_features)
    var out = List[Float64](capacity=explainer.width())
    out.resize(explainer.width(), 0.0)
    explainer.contrib_bins_multiclass_into(booster, bins, out, 0, rng)
    return out^


def predict_contrib(
    model: Model, row: List[Float64], rng: IterationRange
) raises -> List[Float64]:
    """Exact feature contributions for one raw example.

    Returns `n_features + 1` values: one per feature, then the expected
    value. They sum to `model.predict_raw_range(row, rng)`."""
    return predict_contrib_bins(
        model.booster, model.mapper.bin_row(row), model.mapper.n_features, rng
    )


def predict_contrib_multiclass(
    model: MulticlassModel, row: List[Float64], rng: IterationRange
) raises -> List[Float64]:
    """Exact per-class feature contributions for one raw example.

    Returns `n_classes * (n_features + 1)` values in class-major blocks;
    block k sums to `model.predict_raw_range(row, rng)[k]`."""
    return predict_contrib_bins_multiclass(
        model.booster, model.mapper.bin_row(row), model.mapper.n_features, rng
    )
