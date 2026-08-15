"""Feature importance.

Split-count importance counts how many internal nodes across an ensemble
test each feature, matching LightGBM's importance_type of "split".
Gain importance sums the recorded split gains instead, matching
importance_type of "gain".

This is the one implementation of both. The model dump reports the same
gains per node (see model_dump.mojo), so the per-feature sums a consumer
takes over a dump agree with `gain_importance` by construction rather than
by coincidence; nothing recomputes importance from a dump.

Gains are recorded during growth, and as of model format v4 they are
serialized with the tree (see serialize.mojo), so a model loaded from a
file reports the gain importance it was trained with. A model read from a
v1, v2, or v3 file carries no gains and reports zero gain importance:
those formats dropped them, and they cannot be recovered from a fitted
tree. `has_split_gains` in model_dump.mojo is the predicate that tells the
two cases apart, and a consumer should ask it before reading a zero as a
measurement.
"""

from .tree import Tree


def split_importance(
    trees: List[Tree], n_features: Int
) raises -> List[Int]:
    """Number of splits on each feature across all trees. Works on the
    trees of a Booster or a MulticlassBooster alike."""
    if n_features < 1:
        raise Error("n_features must be positive")
    var counts = List[Int](capacity=n_features)
    for _ in range(n_features):
        counts.append(0)
    for t in range(len(trees)):
        for i in range(len(trees[t].feature)):
            var f = trees[t].feature[i]
            if f < 0:
                continue
            if f >= n_features:
                raise Error("tree splits on a feature outside n_features")
            counts[f] += 1
    return counts^


def gain_importance(
    trees: List[Tree], n_features: Int
) raises -> List[Float64]:
    """Total split gain attributed to each feature across all trees.
    Works on the trees of a Booster or a MulticlassBooster alike.

    Every entry is zero for an ensemble whose gains did not survive: one
    loaded from a model file written before v4. That is an absence, not a
    measurement of zero; see the module docstring.
    """
    if n_features < 1:
        raise Error("n_features must be positive")
    var gains = List[Float64](capacity=n_features)
    for _ in range(n_features):
        gains.append(0.0)
    for t in range(len(trees)):
        for i in range(len(trees[t].feature)):
            var f = trees[t].feature[i]
            if f < 0:
                continue
            if f >= n_features:
                raise Error("tree splits on a feature outside n_features")
            gains[f] += trees[t].split_gain[i]
    return gains^
