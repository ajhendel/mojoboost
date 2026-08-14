"""Feature importance.

Split-count importance counts how many internal nodes across an ensemble
test each feature, matching LightGBM's importance_type of "split".
Gain-based importance needs per-node gains stored in the tree and can be
added once trees record them.
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
