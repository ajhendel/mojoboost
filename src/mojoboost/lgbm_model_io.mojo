"""LightGBM text model interop: read and write LightGBM's own model format.

This module is self-contained. It converts between LightGBM's line-oriented
text model (the string `Booster.save_model` writes and `Booster.model_str`
returns) and mojoboost's `Model` / `MulticlassModel`. It reads
`model.mojo`, `serialize.mojo`, and `tree.mojo` but changes nothing in them:
everything here is expressed in terms of the public `BinMapper`, `Tree`,
`Booster`, and `MulticlassBooster` structures.

Why a bin mapper has to be synthesized
--------------------------------------
A mojoboost tree routes by bin id (`Tree.threshold_bin`), a LightGBM tree by
a real-valued threshold. The two are reconciled exactly, in both directions,
by one identity: `BinMapper.bin_value` sends `v` to the first bin `b` with
`v <= edges[b]`, so

    bin(v) <= t   if and only if   v <= edges[edge_offsets[f] + t].

Reading therefore collects every distinct threshold LightGBM used on feature
`f`, sorts them, and makes exactly those the feature's bin edges; node `j`'s
`threshold_bin` is then the position of its threshold in that list. Writing
runs the identity the other way and emits
`edges[edge_offsets[f] + threshold_bin]`. Neither direction approximates:
the routing decision for every raw value is preserved exactly.

A consequence worth stating plainly: a model read from a LightGBM file has
only the bin edges its trees actually split on, not the ones LightGBM fit.
That is enough to predict identically and it is all the file carries, but it
is not the training-time binning, so such a model must not be used as if its
mapper described the training data (for example to continue training).

Missing values
--------------
LightGBM stores a per-node `missing_type` in the `decision_type` bitfield:
0 = None, 1 = Zero, 2 = NaN. mojoboost reserves a per-feature missing bin
instead and routes any row landing in it by the node's `default_left`.
`None` and `NaN` map across exactly; `Zero` (where a zero feature value is
also treated as missing) has no representation in a bin-routed tree and is
rejected. Nodes splitting the same feature must agree on `missing_type`,
since mojoboost's reservation is a property of the feature.

Base score and shrinkage
------------------------
LightGBM folds both the learning rate and the boost-from-average init score
into the stored leaf values: predicting is a plain sum over trees. mojoboost
keeps them separate as `base_score` and `learning_rate`. So a model read from
a LightGBM file gets `learning_rate = 1.0` and `base_score = 0.0` with the
leaf values taken verbatim, and a model written to one gets every leaf value
multiplied by `learning_rate`, with `base_score` added into the first
iteration's trees the way LightGBM's `Tree::AddBias` does. Predictions are
identical either way, term for term and rounding for rounding, but the round
trip is not the identity on the `(base_score, learning_rate, leaf value)`
triple, only on what it predicts.

What is rejected, and why
-------------------------
- Categorical splits (`decision_type` bit 0, `num_cat > 0`). LightGBM keeps
  category sets over raw codes in `cat_threshold`; mojoboost keeps 256-bit
  sets over bins plus a fitted code table (see categorical.mojo) that a model
  file does not carry. Until that table can be reconstructed, converting
  would silently change which categories go left.
- Linear trees (`is_linear=1`, `leaf_const`, `leaf_coeff`). mojoboost leaves
  hold a constant.
- `missing_type = Zero`, as above.
- `average_output` (random forest boosting). mojoboost sums trees.
- Objectives mojoboost does not implement, `multiclassova` and friends, a
  `binary` objective with a non-unit `sigmoid`, and files written under a
  custom objective (whose link mojoboost cannot know).

Every rejection raises with the construct named. Nothing is silently
approximated.

Node vocabulary
---------------
The names below are the shared vocabulary this module and the model
inspection / dump schema (`inspection.mojo`, task 14) both use, so the two
can be reconciled once rather than translated repeatedly:

- `node_index`  - position in a tree's flat node arrays. Parents always
  precede their children, so an ascending scan is a valid top-down pass.
- `is_leaf`     - `feature[node] < 0`.
- `split_feature` - column index split at an internal node, -1 at a leaf.
- `threshold`   - real-valued upper bound of the left branch: `<=` goes left.
- `threshold_bin` - the same split in bin space; `threshold` is
  `edges[edge_offsets[split_feature] + threshold_bin]`.
- `decision_type` - numerical `<=` versus categorical set membership.
- `default_left` - where a missing value goes; `missing_bin` is the feature's
  reserved missing bin, or -1 when it reserves none.
- `left_child` / `right_child` - `node_index` values in mojoboost. LightGBM
  instead encodes a leaf as a negative number, `~leaf` (so -1 is leaf 0), and
  a non-negative number as an internal-node index in its own separate
  numbering.
- `leaf_ordinal` - a leaf's rank among its tree's leaves in `node_index`
  order (`Tree.leaf_ordinals`). This is mojoboost's leaf numbering and is not
  LightGBM's leaf index; the writer here numbers LightGBM leaves in preorder,
  which coincides with `leaf_ordinal` only by accident.
- `value`       - node output: the leaf value at a leaf, and at an internal
  node the value it carried when it was created (LightGBM's
  `internal_value`).
- `count`       - training rows reaching the node, LightGBM's `leaf_count` /
  `internal_count`; the cover exact TreeSHAP conditions on.
- `split_gain`  - the gain recorded when the node was split.
"""

from std.math import isfinite

from .binning import BinMapper
from .boosting import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    CUSTOM,
    FAIR,
    GAMMA,
    HUBER,
    L1,
    MAPE,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    TWEEDIE,
    Booster,
    MulticlassBooster,
)
from .categorical import CategoricalSpec
from .model import Model, MulticlassModel
from .params import MULTICLASS, objective_from_name
from .ranking import LAMBDARANK
from .tree import Tree

# LightGBM's `decision_type` bitfield (see its `Tree::GetDecisionType`).
comptime _CATEGORICAL_MASK = 1
comptime _DEFAULT_LEFT_MASK = 2
comptime _MISSING_TYPE_SHIFT = 2
comptime _MISSING_TYPE_MASK = 3

comptime _MISSING_NONE = 0
comptime _MISSING_ZERO = 1
comptime _MISSING_NAN = 2

# mojoboost bins are UInt8, so a converted model may use at most this many
# bins for any one feature. A LightGBM model trained with the default
# max_bin=255 has at most 254 distinct thresholds on a feature, which fits
# with room for a reserved missing bin.
comptime _MAX_BINS = 256

# The format version this writer emits. The reader accepts v2, v3, and v4;
# they differ only in fields this converter either writes explicitly or
# rejects.
comptime _WRITE_VERSION = "v4"


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


struct _KeyVals(Copyable, Movable):
    """A LightGBM `key=value` block, in file order.

    A linear scan rather than a hash map: a header is a dozen keys and a tree
    block under twenty, and keeping insertion order makes error messages
    quote the file the way it reads.
    """

    var keys: List[String]
    var vals: List[String]

    def __init__(out self):
        self.keys = List[String]()
        self.vals = List[String]()

    def set(mut self, key: String, value: String):
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                self.vals[i] = value.copy()
                return
        self.keys.append(key.copy())
        self.vals.append(value.copy())

    def has(self, key: String) -> Bool:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return True
        return False

    def get(self, key: String, default: String) -> String:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return self.vals[i].copy()
        return default.copy()

    def require(self, key: String, where: String) raises -> String:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return self.vals[i].copy()
        raise Error(
            "LightGBM model ", where, " is missing '", key, "='"
        )

    def require_int(self, key: String, where: String) raises -> Int:
        var raw = self.require(key, where)
        try:
            return Int(raw)
        except:
            raise Error(
                "LightGBM model ",
                where,
                ": '",
                key,
                "=",
                raw,
                "' is not an integer",
            )

    def int_or(self, key: String, where: String, default: Int) raises -> Int:
        if not self.has(key):
            return default
        return self.require_int(key, where)


def _parse_ints(
    value: String, expected: Int, what: String
) raises -> List[Int]:
    """Whitespace-separated integers. `expected < 0` accepts any count."""
    var out = List[Int]()
    for token in value.split():
        try:
            out.append(Int(String(token)))
        except:
            raise Error(
                "LightGBM model: '", what, "' holds a non-integer entry '",
                String(token),
                "'",
            )
    if expected >= 0 and len(out) != expected:
        raise Error(
            "LightGBM model: '",
            what,
            "' has ",
            len(out),
            " entries, expected ",
            expected,
        )
    return out^


def _parse_floats(
    value: String, expected: Int, what: String
) raises -> List[Float64]:
    """Whitespace-separated reals. `expected < 0` accepts any count."""
    var out = List[Float64]()
    for token in value.split():
        var text = String(token)
        var parsed: Float64
        try:
            parsed = Float64(text)
        except:
            raise Error(
                "LightGBM model: '", what, "' holds a non-numeric entry '",
                text,
                "'",
            )
        out.append(parsed)
    if expected >= 0 and len(out) != expected:
        raise Error(
            "LightGBM model: '",
            what,
            "' has ",
            len(out),
            " entries, expected ",
            expected,
        )
    return out^


def _f64_text(x: Float64, what: String) raises -> String:
    """A decimal rendering of `x` that parses back to exactly `x`.

    LightGBM's format is decimal text, so the writer cannot fall back on raw
    bit patterns the way `serialize.mojo` does. Every value is therefore
    checked: if the shortest round-tripping form is not in fact
    round-tripping, the model is refused rather than written out with a
    silently perturbed threshold or leaf value.
    """
    if not isfinite(x):
        raise Error(
            "cannot write ",
            what,
            " to a LightGBM model file: it is not finite (",
            x,
            ")",
        )
    var text = String(x)
    var back: Float64
    try:
        back = Float64(text)
    except:
        raise Error(
            "cannot write ", what, ": '", text, "' does not parse back"
        )
    if back != x:
        raise Error(
            "cannot write ",
            what,
            " exactly: it renders as '",
            text,
            "', which reads back as a different value",
        )
    return text^


def _join_ints(values: List[Int]) -> String:
    var out = String("")
    for i in range(len(values)):
        if i > 0:
            out += " "
        out += String(values[i])
    return out^


def _join_floats(values: List[Float64], what: String) raises -> String:
    var out = String("")
    for i in range(len(values)):
        if i > 0:
            out += " "
        out += _f64_text(values[i], what)
    return out^


# ---------------------------------------------------------------------------
# Parsed LightGBM text
# ---------------------------------------------------------------------------


struct _LgbmTree(Copyable, Movable):
    """One `Tree=` block, already validated for shape.

    The split arrays have `num_leaves - 1` entries and the leaf arrays
    `num_leaves`, which is LightGBM's own invariant; a single-leaf tree has
    empty split arrays and one leaf value.
    """

    var num_leaves: Int
    var split_feature: List[Int]
    var threshold: List[Float64]
    var decision_type: List[Int]
    var left_child: List[Int]
    var right_child: List[Int]
    var split_gain: List[Float64]
    var internal_value: List[Float64]
    var internal_count: List[Float64]
    var leaf_value: List[Float64]
    var leaf_count: List[Float64]

    def __init__(
        out self,
        num_leaves: Int,
        var split_feature: List[Int],
        var threshold: List[Float64],
        var decision_type: List[Int],
        var left_child: List[Int],
        var right_child: List[Int],
        var split_gain: List[Float64],
        var internal_value: List[Float64],
        var internal_count: List[Float64],
        var leaf_value: List[Float64],
        var leaf_count: List[Float64],
    ):
        self.num_leaves = num_leaves
        self.split_feature = split_feature^
        self.threshold = threshold^
        self.decision_type = decision_type^
        self.left_child = left_child^
        self.right_child = right_child^
        self.split_gain = split_gain^
        self.internal_value = internal_value^
        self.internal_count = internal_count^
        self.leaf_value = leaf_value^
        self.leaf_count = leaf_count^


struct _LgbmText(Copyable, Movable):
    """A parsed LightGBM model file: its header and its tree blocks."""

    var header: _KeyVals
    var trees: List[_LgbmTree]

    def __init__(out self, var header: _KeyVals, var trees: List[_LgbmTree]):
        self.header = header^
        self.trees = trees^


def _finish_tree(block: _KeyVals, index: Int) raises -> _LgbmTree:
    """Validate and normalize one `Tree=` block."""
    var where = String("tree ") + String(index)

    if block.int_or("is_linear", where, 0) != 0:
        raise Error(
            "LightGBM ",
            where,
            " is a linear tree (is_linear=1); mojoboost leaves hold a"
            " constant, so a linear model cannot be converted",
        )
    if block.has("leaf_const") or block.has("leaf_coeff"):
        raise Error(
            "LightGBM ",
            where,
            " carries linear-model leaf coefficients; mojoboost leaves hold a"
            " constant, so a linear model cannot be converted",
        )
    var num_cat = block.int_or("num_cat", where, 0)
    if num_cat != 0 or block.has("cat_threshold") or block.has(
        "cat_boundaries"
    ):
        raise Error(
            "LightGBM ",
            where,
            " has categorical splits (num_cat=",
            num_cat,
            "); a model file does not carry the fitted category-to-bin table"
            " mojoboost routes them with, so they are not convertible yet",
        )

    var num_leaves = block.require_int("num_leaves", where)
    if num_leaves < 1:
        raise Error(
            "LightGBM ", where, " declares num_leaves=", num_leaves
        )
    var n_split = num_leaves - 1

    var split_feature = _parse_ints(
        block.get("split_feature", ""), n_split, where + " split_feature"
    )
    var threshold = _parse_floats(
        block.get("threshold", ""), n_split, where + " threshold"
    )
    var decision_type = _parse_ints(
        block.get("decision_type", _repeat_zero(n_split)),
        n_split,
        where + " decision_type",
    )
    var left_child = _parse_ints(
        block.get("left_child", ""), n_split, where + " left_child"
    )
    var right_child = _parse_ints(
        block.get("right_child", ""), n_split, where + " right_child"
    )
    var split_gain = _parse_floats(
        block.get("split_gain", _repeat_zero(n_split)),
        n_split,
        where + " split_gain",
    )
    var internal_value = _parse_floats(
        block.get("internal_value", _repeat_zero(n_split)),
        n_split,
        where + " internal_value",
    )
    var internal_count = _parse_floats(
        block.get("internal_count", _repeat_zero(n_split)),
        n_split,
        where + " internal_count",
    )
    var leaf_value = _parse_floats(
        block.require("leaf_value", where), num_leaves, where + " leaf_value"
    )
    var leaf_count = _parse_floats(
        block.get("leaf_count", _repeat_zero(num_leaves)),
        num_leaves,
        where + " leaf_count",
    )

    for j in range(n_split):
        if decision_type[j] & _CATEGORICAL_MASK != 0:
            raise Error(
                "LightGBM ",
                where,
                " node ",
                j,
                " is a categorical split; a model file does not carry the"
                " fitted category-to-bin table mojoboost routes them with, so"
                " it is not convertible yet",
            )
        var missing = (
            decision_type[j] >> _MISSING_TYPE_SHIFT
        ) & _MISSING_TYPE_MASK
        if missing == _MISSING_ZERO:
            raise Error(
                "LightGBM ",
                where,
                " node ",
                j,
                " uses missing_type='Zero', which treats a zero feature value"
                " as missing; mojoboost routes missing values by a reserved"
                " bin and cannot express that",
            )
        if missing != _MISSING_NONE and missing != _MISSING_NAN:
            raise Error(
                "LightGBM ", where, " node ", j, " has an unknown missing_type"
            )
        if not isfinite(threshold[j]):
            raise Error(
                "LightGBM ",
                where,
                " node ",
                j,
                " has a non-finite threshold",
            )

    return _LgbmTree(
        num_leaves,
        split_feature^,
        threshold^,
        decision_type^,
        left_child^,
        right_child^,
        split_gain^,
        internal_value^,
        internal_count^,
        leaf_value^,
        leaf_count^,
    )


def _repeat_zero(n: Int) -> String:
    """`n` zeros, the stand-in for an optional array LightGBM omitted."""
    var out = String("")
    for i in range(n):
        if i > 0:
            out += " "
        out += "0"
    return out^


def _parse_lgbm_text(text: String) raises -> _LgbmText:
    """Split a LightGBM model string into its header and its tree blocks.

    Everything from `end of trees` on (feature importances, the parameter
    dump, `pandas_categorical`) describes how the model was trained rather
    than what it computes, so it is read past rather than into.
    """
    var header = _KeyVals()
    var trees = List[_LgbmTree]()
    var block = _KeyVals()
    var in_tree = False
    var tree_index = 0
    var saw_magic = False

    for raw_line in text.split("\n"):
        var line = String(String(raw_line).strip())
        if line.byte_length() == 0:
            continue
        if line == "end of trees":
            break
        if line == "tree":
            saw_magic = True
            continue
        var eq = line.find("=")
        if eq < 0:
            # LightGBM writes a handful of bare flag lines in the header.
            if line == "average_output":
                raise Error(
                    "this LightGBM model averages its trees (boosting='rf');"
                    " mojoboost sums them, so a random forest cannot be"
                    " converted"
                )
            continue
        var key = String(line[byte=0:eq])
        var value = String(line[byte=eq + 1 :])
        if key == "Tree":
            if in_tree:
                trees.append(_finish_tree(block, tree_index))
                tree_index += 1
            block = _KeyVals()
            in_tree = True
            continue
        if in_tree:
            block.set(key, value)
        else:
            header.set(key, value)

    if in_tree:
        trees.append(_finish_tree(block, tree_index))

    if not saw_magic:
        raise Error(
            "not a LightGBM model file: it does not start with a 'tree' line"
        )
    return _LgbmText(header^, trees^)


def _check_version(header: _KeyVals) raises:
    var version = header.get("version", "v3")
    if version != "v2" and version != "v3" and version != "v4":
        raise Error(
            "unsupported LightGBM model format version '",
            version,
            "'; this reader handles v2, v3, and v4",
        )


# ---------------------------------------------------------------------------
# Objective mapping
# ---------------------------------------------------------------------------


struct _Objective(Copyable, Movable):
    """A parsed `objective=` line: the name and its `key:value` settings."""

    var name: String
    var keys: List[String]
    var values: List[String]

    def __init__(
        out self, var name: String, var keys: List[String],
        var values: List[String],
    ):
        self.name = name^
        self.keys = keys^
        self.values = values^

    def setting(self, key: String, default: String) -> String:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return self.values[i].copy()
        return default.copy()


def _parse_objective_line(line: String) raises -> _Objective:
    """`objective=quantile alpha:0.9` into its name and its settings."""
    var name = String("")
    var keys = List[String]()
    var values = List[String]()
    var first = True
    for token_slice in line.split():
        var token = String(token_slice)
        if first:
            name = token^
            first = False
            continue
        var colon = token.find(":")
        if colon < 0:
            continue
        keys.append(String(token[byte=0:colon]))
        values.append(String(token[byte=colon + 1 :]))
    if name.byte_length() == 0:
        raise Error(
            "this LightGBM model was written under a custom objective (its"
            " 'objective=' line is empty); mojoboost cannot know the link"
            " function, so load it with a custom-objective entry point"
            " instead"
        )
    return _Objective(name^, keys^, values^)


def lgbm_objective_code(objective_line: String) raises -> Int:
    """The mojoboost objective code for a LightGBM `objective=` line.

    Returns `MULTICLASS` (from params.mojo) for `multiclass` and `softmax`,
    which need `load_lgbm_multiclass_model` rather than `load_lgbm_model`.
    Objectives mojoboost does not implement are reported by name; see
    `objective_from_name`.
    """
    var parsed = _parse_objective_line(objective_line)
    var name = parsed.name.copy()
    if (
        name == "multiclassova"
        or name == "multiclass_ova"
        or name == "ova"
        or name == "ovr"
    ):
        raise Error(
            "objective 'multiclassova' trains one independent binary model"
            " per class; mojoboost's multiclass model is a softmax over"
            " per-class scores and cannot represent it"
        )
    if name == "lambdarank":
        # A ranking model's trees are ordinary trees and its prediction is the
        # raw score, so it converts even though `objective_from_name` refuses
        # the name (there it would mean "train this", which needs groups).
        return LAMBDARANK
    if name == "binary":
        var sigmoid = parsed.setting("sigmoid", "1")
        var value: Float64
        try:
            value = Float64(sigmoid)
        except:
            raise Error(
                "LightGBM model: 'binary sigmoid:", sigmoid, "' is not a"
                " number"
            )
        if value != 1.0:
            raise Error(
                "this LightGBM model uses sigmoid=",
                value,
                " for its binary objective; mojoboost's logistic link is"
                " fixed at sigmoid=1, so its probabilities would differ",
            )
        return BINARY_LOGISTIC
    return objective_from_name(name)


def lgbm_objective_name(objective: Int) raises -> String:
    """The LightGBM `objective=` line for a mojoboost objective code.

    The scalar parameter of `huber`, `quantile`, `fair`, and `tweedie` is not
    kept on a fitted `Booster` (it shaped the gradients, not the trees), so it
    is not written; a LightGBM reader will report its own default for it. The
    trees, and so the predictions, are unaffected.
    """
    if objective == SQUARED_ERROR:
        return String("regression")
    if objective == BINARY_LOGISTIC:
        return String("binary sigmoid:1")
    if objective == POISSON:
        return String("poisson")
    if objective == HUBER:
        return String("huber")
    if objective == QUANTILE:
        return String("quantile")
    if objective == L1:
        return String("regression_l1")
    if objective == GAMMA:
        return String("gamma")
    if objective == TWEEDIE:
        return String("tweedie")
    if objective == MAPE:
        return String("mape")
    if objective == FAIR:
        return String("fair")
    if objective == CROSS_ENTROPY:
        return String("cross_entropy")
    if objective == LAMBDARANK:
        return String("lambdarank")
    if objective == CUSTOM:
        raise Error(
            "a model trained with a custom objective cannot be written as a"
            " LightGBM model file: the file has nowhere to record the"
            " gradient callback or the link function"
        )
    raise Error(
        "no LightGBM objective name for mojoboost objective code ", objective
    )


# ---------------------------------------------------------------------------
# Reading: LightGBM text -> Model / MulticlassModel
# ---------------------------------------------------------------------------


def _collect_edges(
    trees: List[_LgbmTree],
    n_features: Int,
    mut edges: List[Float64],
    mut offsets: List[Int],
) raises:
    """Every distinct threshold used on each feature, ascending, in
    `BinMapper` layout: feature f's edges are `edges[offsets[f]:offsets[f+1]]`.

    Bucketed by feature with a counting pass so a large model costs one sort
    per feature rather than a scan per feature over every node.
    """
    var counts = List[Int](capacity=n_features)
    counts.resize(n_features, 0)
    for t in range(len(trees)):
        ref tree = trees[t]
        for j in range(len(tree.split_feature)):
            var f = tree.split_feature[j]
            if f < 0 or f >= n_features:
                raise Error(
                    "LightGBM tree ",
                    t,
                    " node ",
                    j,
                    " splits feature ",
                    f,
                    ", which is outside the model's ",
                    n_features,
                    " features",
                )
            counts[f] += 1

    var starts = List[Int](capacity=n_features + 1)
    starts.append(0)
    var total = 0
    for f in range(n_features):
        total += counts[f]
        starts.append(total)

    var flat = List[Float64](capacity=total)
    flat.resize(total, 0.0)
    var cursor = List[Int](capacity=n_features)
    for f in range(n_features):
        cursor.append(starts[f])
    for t in range(len(trees)):
        ref tree = trees[t]
        for j in range(len(tree.split_feature)):
            var f = tree.split_feature[j]
            flat[cursor[f]] = tree.threshold[j]
            cursor[f] += 1

    edges.clear()
    offsets.clear()
    offsets.append(0)
    for f in range(n_features):
        var column = List[Float64](capacity=starts[f + 1] - starts[f])
        for i in range(starts[f], starts[f + 1]):
            column.append(flat[i])
        sort(column)
        for i in range(len(column)):
            if i == 0 or column[i] != column[i - 1]:
                edges.append(column[i])
        offsets.append(len(edges))


def _collect_missing_bins(
    trees: List[_LgbmTree], n_features: Int, offsets: List[Int]
) raises -> List[Int]:
    """Each feature's reserved missing bin, or -1.

    LightGBM stores `missing_type` per node; mojoboost reserves per feature,
    so the nodes splitting one feature have to agree. They always do in a
    LightGBM-trained model, where the setting comes from the feature's bin
    mapper, and a file where they do not is malformed rather than merely
    unsupported.
    """
    var kind = List[Int](capacity=n_features)
    kind.resize(n_features, -1)
    for t in range(len(trees)):
        ref tree = trees[t]
        for j in range(len(tree.split_feature)):
            var f = tree.split_feature[j]
            var missing = (
                tree.decision_type[j] >> _MISSING_TYPE_SHIFT
            ) & _MISSING_TYPE_MASK
            if kind[f] < 0:
                kind[f] = missing
            elif kind[f] != missing:
                raise Error(
                    "LightGBM model: feature ",
                    f,
                    " is split with two different missing_type settings;"
                    " mojoboost reserves a missing bin per feature, so the"
                    " nodes on one feature must agree",
                )
    var missing_bin = List[Int](capacity=n_features)
    for f in range(n_features):
        if kind[f] == _MISSING_NAN:
            # Ordinary bins are 0..n_edges, so the reservation sits above them.
            missing_bin.append(offsets[f + 1] - offsets[f] + 1)
        else:
            missing_bin.append(-1)
    return missing_bin^


def _bins_needed(offsets: List[Int], missing_bin: List[Int]) raises -> Int:
    var n_features = len(missing_bin)
    var needed = 1
    for f in range(n_features):
        var n_edges = offsets[f + 1] - offsets[f]
        var here = n_edges + 1
        if missing_bin[f] >= 0:
            here += 1
        if here > needed:
            needed = here
    if needed > _MAX_BINS:
        raise Error(
            "this LightGBM model needs ",
            needed,
            " bins for one feature; mojoboost stores bins in a byte, so at"
            " most ",
            _MAX_BINS,
            " are available. Retrain with a smaller max_bin.",
        )
    return needed


def _edge_index(
    mapper: BinMapper, feature: Int, threshold: Float64
) raises -> Int:
    """The bin whose upper edge is exactly `threshold`.

    The edges were built from these thresholds, so the match is exact by
    construction; the search is a binary one because a feature can carry
    hundreds of them.
    """
    var lo = mapper.edge_offsets[feature]
    var hi = mapper.edge_offsets[feature + 1]
    var left = lo
    var right = hi
    while left < right:
        var mid = (left + right) // 2
        if mapper.edges[mid] < threshold:
            left = mid + 1
        else:
            right = mid
    if left >= hi or mapper.edges[left] != threshold:
        raise Error(
            "internal error converting a LightGBM model: threshold ",
            threshold,
            " on feature ",
            feature,
            " is not among the edges collected for it",
        )
    return left - lo


struct _NodeArrays(Copyable, Movable):
    """A mojoboost tree under construction, one appended node at a time."""

    var feature: List[Int]
    var threshold_bin: List[Int]
    var left: List[Int]
    var right: List[Int]
    var value: List[Float64]
    var split_gain: List[Float64]
    var default_left: List[Bool]
    var missing_bin: List[Int]
    var count: List[Float64]

    def __init__(out self):
        self.feature = List[Int]()
        self.threshold_bin = List[Int]()
        self.left = List[Int]()
        self.right = List[Int]()
        self.value = List[Float64]()
        self.split_gain = List[Float64]()
        self.default_left = List[Bool]()
        self.missing_bin = List[Int]()
        self.count = List[Float64]()

    def add(mut self) -> Int:
        """Append a leaf-shaped placeholder and return its node index."""
        var node = len(self.feature)
        self.feature.append(-1)
        self.threshold_bin.append(-1)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(0.0)
        self.split_gain.append(0.0)
        self.default_left.append(False)
        self.missing_bin.append(-1)
        self.count.append(0.0)
        return node


def _emit_node(
    mut arrays: _NodeArrays,
    src: _LgbmTree,
    mapper: BinMapper,
    mut seen_internal: List[Bool],
    mut seen_leaf: List[Bool],
    code: Int,
    tree_index: Int,
) raises -> Int:
    """Convert one LightGBM subtree, returning its mojoboost node index.

    `code` is LightGBM's child encoding: non-negative is an internal-node
    index, negative is the leaf `-code - 1`. Nodes are emitted in preorder, so
    a parent's index is always below its children's, which is the ordering
    `tree.node_bounds` and every top-down pass in mojoboost assume.
    """
    var node = arrays.add()
    if code < 0:
        var leaf = -code - 1
        if leaf < 0 or leaf >= src.num_leaves:
            raise Error(
                "LightGBM tree ", tree_index, " refers to leaf ", leaf,
                ", which does not exist",
            )
        if seen_leaf[leaf]:
            raise Error(
                "LightGBM tree ",
                tree_index,
                " reaches leaf ",
                leaf,
                " twice; its child links do not form a tree",
            )
        seen_leaf[leaf] = True
        arrays.value[node] = src.leaf_value[leaf]
        arrays.count[node] = src.leaf_count[leaf]
        return node

    if code >= src.num_leaves - 1:
        raise Error(
            "LightGBM tree ",
            tree_index,
            " refers to internal node ",
            code,
            ", which does not exist",
        )
    if seen_internal[code]:
        raise Error(
            "LightGBM tree ",
            tree_index,
            " reaches internal node ",
            code,
            " twice; its child links do not form a tree",
        )
    seen_internal[code] = True

    var feature = src.split_feature[code]
    arrays.feature[node] = feature
    arrays.threshold_bin[node] = _edge_index(
        mapper, feature, src.threshold[code]
    )
    arrays.default_left[node] = (
        src.decision_type[code] & _DEFAULT_LEFT_MASK
    ) != 0
    arrays.missing_bin[node] = mapper.missing_bin[feature]
    arrays.value[node] = src.internal_value[code]
    arrays.split_gain[node] = src.split_gain[code]
    arrays.count[node] = src.internal_count[code]

    var left = _emit_node(
        arrays,
        src,
        mapper,
        seen_internal,
        seen_leaf,
        src.left_child[code],
        tree_index,
    )
    var right = _emit_node(
        arrays,
        src,
        mapper,
        seen_internal,
        seen_leaf,
        src.right_child[code],
        tree_index,
    )
    arrays.left[node] = left
    arrays.right[node] = right
    return node


def _convert_tree(
    src: _LgbmTree, mapper: BinMapper, tree_index: Int
) raises -> Tree:
    var arrays = _NodeArrays()
    var n_internal = src.num_leaves - 1
    var seen_internal = List[Bool](capacity=n_internal)
    seen_internal.resize(n_internal, False)
    var seen_leaf = List[Bool](capacity=src.num_leaves)
    seen_leaf.resize(src.num_leaves, False)

    # A single-leaf tree has no internal nodes, so its root is leaf 0.
    var root_code = 0 if src.num_leaves > 1 else -1
    _ = _emit_node(
        arrays,
        src,
        mapper,
        seen_internal,
        seen_leaf,
        root_code,
        tree_index,
    )
    for leaf in range(src.num_leaves):
        if not seen_leaf[leaf]:
            raise Error(
                "LightGBM tree ",
                tree_index,
                " never reaches leaf ",
                leaf,
                "; its child links do not form a tree",
            )
    return Tree(
        arrays.feature^,
        arrays.threshold_bin^,
        arrays.left^,
        arrays.right^,
        arrays.value^,
        arrays.split_gain^,
        src.num_leaves,
        arrays.default_left^,
        arrays.missing_bin^,
        List[Int](),
        List[UInt64](),
        arrays.count^,
    )


def _synthesize_mapper(
    trees: List[_LgbmTree], n_features: Int
) raises -> BinMapper:
    """The bin mapper a LightGBM file implies: one edge per distinct
    threshold, and a reserved missing bin for each feature whose nodes say
    `missing_type=NaN`. See the module docstring for why this is exact for
    prediction and wrong to reuse for training."""
    var edges = List[Float64]()
    var offsets = List[Int]()
    _collect_edges(trees, n_features, edges, offsets)
    var missing_bin = _collect_missing_bins(trees, n_features, offsets)
    var n_bins = _bins_needed(offsets, missing_bin)
    return BinMapper(
        edges^,
        offsets^,
        n_features,
        n_bins,
        CategoricalSpec.all_numerical(n_features),
        missing_bin^,
    )


def _convert_trees(
    src: List[_LgbmTree], mapper: BinMapper
) raises -> List[Tree]:
    var trees = List[Tree](capacity=len(src))
    for t in range(len(src)):
        trees.append(_convert_tree(src[t], mapper, t))
    return trees^


def _n_features(header: _KeyVals) raises -> Int:
    var max_idx = header.require_int("max_feature_idx", "header")
    if max_idx < 0:
        raise Error(
            "LightGBM model header declares max_feature_idx=",
            max_idx,
            ", so the model has no features",
        )
    return max_idx + 1


def parse_lgbm_model(text: String) raises -> Model:
    """Convert a single-output LightGBM model string into a `Model`.

    The result predicts exactly what LightGBM's own predictor would, on the
    raw feature values, for every construct this module accepts. It carries
    `learning_rate = 1.0` and `base_score = 0.0` because LightGBM has already
    folded both into its leaf values; see the module docstring.
    """
    var parsed = _parse_lgbm_text(text)
    _check_version(parsed.header)

    var num_class = parsed.header.int_or("num_class", "header", 1)
    var objective = lgbm_objective_code(
        parsed.header.require("objective", "header")
    )
    if objective == MULTICLASS or num_class > 1:
        raise Error(
            "this is a multiclass LightGBM model (num_class=",
            num_class,
            "); use load_lgbm_multiclass_model",
        )
    var per_iter = parsed.header.int_or(
        "num_tree_per_iteration", "header", 1
    )
    if per_iter != 1:
        raise Error(
            "this LightGBM model grows ",
            per_iter,
            " trees per iteration but declares num_class=1; mojoboost's"
            " single-output model grows one",
        )

    var mapper = _synthesize_mapper(parsed.trees, _n_features(parsed.header))
    var trees = _convert_trees(parsed.trees, mapper)
    var booster = Booster(trees^, 0.0, 1.0, objective)
    return Model(mapper^, booster^)


def parse_lgbm_multiclass_model(text: String) raises -> MulticlassModel:
    """Convert a multiclass LightGBM model string into a `MulticlassModel`.

    LightGBM stores one tree per class per iteration in round-major order,
    which is the order `MulticlassBooster` already uses, so the trees carry
    over unpermuted. Every class's base score is 0.0 for the reason given in
    the module docstring.
    """
    var parsed = _parse_lgbm_text(text)
    _check_version(parsed.header)

    var objective = lgbm_objective_code(
        parsed.header.require("objective", "header")
    )
    if objective != MULTICLASS:
        raise Error(
            "this is a single-output LightGBM model; use load_lgbm_model"
        )
    var n_classes = parsed.header.int_or("num_class", "header", 0)
    if n_classes < 2:
        raise Error(
            "LightGBM multiclass model declares num_class=",
            n_classes,
            "; at least 2 classes are needed",
        )
    var per_iter = parsed.header.int_or(
        "num_tree_per_iteration", "header", n_classes
    )
    if per_iter != n_classes:
        raise Error(
            "LightGBM multiclass model grows ",
            per_iter,
            " trees per iteration for ",
            n_classes,
            " classes; mojoboost grows one per class",
        )
    if len(parsed.trees) % n_classes != 0:
        raise Error(
            "LightGBM multiclass model holds ",
            len(parsed.trees),
            " trees, which is not a whole number of rounds over ",
            n_classes,
            " classes",
        )

    var mapper = _synthesize_mapper(parsed.trees, _n_features(parsed.header))
    var trees = _convert_trees(parsed.trees, mapper)
    var base_scores = List[Float64](capacity=n_classes)
    base_scores.resize(n_classes, 0.0)
    var booster = MulticlassBooster(trees^, base_scores^, n_classes, 1.0)
    return MulticlassModel(mapper^, booster^)


def lgbm_text_kind(text: String) raises -> String:
    """Which loader a LightGBM model string needs, "objective" or
    "multiclass". The counterpart of `serialize.model_file_kind`."""
    var parsed = _parse_lgbm_text(text)
    _check_version(parsed.header)
    var objective = lgbm_objective_code(
        parsed.header.require("objective", "header")
    )
    if objective == MULTICLASS:
        return String("multiclass")
    if parsed.header.int_or("num_class", "header", 1) > 1:
        return String("multiclass")
    return String("objective")


def load_lgbm_model(path: String) raises -> Model:
    """Read a single-output LightGBM model file. See `parse_lgbm_model`."""
    return parse_lgbm_model(open(path, "r").read())


def load_lgbm_multiclass_model(path: String) raises -> MulticlassModel:
    """Read a multiclass LightGBM model file. See
    `parse_lgbm_multiclass_model`."""
    return parse_lgbm_multiclass_model(open(path, "r").read())


def lgbm_model_file_kind(path: String) raises -> String:
    """Which loader a LightGBM model file needs. See `lgbm_text_kind`."""
    return lgbm_text_kind(open(path, "r").read())


# ---------------------------------------------------------------------------
# Writing: Model / MulticlassModel -> LightGBM text
# ---------------------------------------------------------------------------


struct _LgbmOut(Copyable, Movable):
    """One LightGBM tree under construction, in LightGBM's own numbering."""

    var split_feature: List[Int]
    var threshold: List[Float64]
    var decision_type: List[Int]
    var left_child: List[Int]
    var right_child: List[Int]
    var split_gain: List[Float64]
    var internal_value: List[Float64]
    var internal_count: List[Int]
    var leaf_value: List[Float64]
    var leaf_count: List[Int]

    def __init__(out self):
        self.split_feature = List[Int]()
        self.threshold = List[Float64]()
        self.decision_type = List[Int]()
        self.left_child = List[Int]()
        self.right_child = List[Int]()
        self.split_gain = List[Float64]()
        self.internal_value = List[Float64]()
        self.internal_count = List[Int]()
        self.leaf_value = List[Float64]()
        self.leaf_count = List[Int]()


def _check_writable_tree(tree: Tree, tree_index: Int) raises:
    for i in range(len(tree.feature)):
        if tree.cat_offset[i] >= 0:
            raise Error(
                "tree ",
                tree_index,
                " node ",
                i,
                " splits a categorical feature by bin set; LightGBM's format"
                " stores category sets over raw codes, which mojoboost's"
                " fitted table would have to be inverted to produce, so"
                " categorical models cannot be written yet",
            )


def _emit_lgbm_node(
    mut dst: _LgbmOut,
    tree: Tree,
    mapper: BinMapper,
    node: Int,
    scale: Float64,
    bias: Float64,
    tree_index: Int,
) raises -> Int:
    """Convert one mojoboost subtree, returning LightGBM's child code.

    Preorder, so the mojoboost root becomes LightGBM internal node 0, which
    is where its predictor starts. `scale` is the learning rate and `bias` the
    base score being folded into the leaf values (see the module docstring);
    `bias` is nonzero only for an iteration-0 tree.
    """
    if tree.feature[node] < 0:
        var leaf = len(dst.leaf_value)
        dst.leaf_value.append(scale * tree.value[node] + bias)
        dst.leaf_count.append(Int(tree.count[node]))
        return -leaf - 1

    var feature = tree.feature[node]
    var bin = tree.threshold_bin[node]
    var lo = mapper.edge_offsets[feature]
    var hi = mapper.edge_offsets[feature + 1]
    if bin < 0 or lo + bin >= hi:
        raise Error(
            "tree ",
            tree_index,
            " node ",
            node,
            " splits feature ",
            feature,
            " at bin ",
            bin,
            ", which has no upper edge in the model's bin mapper, so it has"
            " no real-valued LightGBM threshold",
        )

    var decision = 0
    if tree.default_left[node]:
        decision |= _DEFAULT_LEFT_MASK
    if tree.missing_bin[node] >= 0:
        decision |= _MISSING_NAN << _MISSING_TYPE_SHIFT

    var index = len(dst.split_feature)
    dst.split_feature.append(feature)
    dst.threshold.append(mapper.edges[lo + bin])
    dst.decision_type.append(decision)
    dst.split_gain.append(tree.split_gain[node])
    dst.internal_value.append(scale * tree.value[node] + bias)
    dst.internal_count.append(Int(tree.count[node]))
    # Reserved now, patched once the children know their own codes.
    dst.left_child.append(0)
    dst.right_child.append(0)

    var left = _emit_lgbm_node(
        dst, tree, mapper, tree.left[node], scale, bias, tree_index
    )
    var right = _emit_lgbm_node(
        dst, tree, mapper, tree.right[node], scale, bias, tree_index
    )
    dst.left_child[index] = left
    dst.right_child[index] = right
    return index


def _tree_body(
    tree: Tree,
    mapper: BinMapper,
    scale: Float64,
    bias: Float64,
    tree_index: Int,
) raises -> String:
    """One tree's LightGBM block, without its `Tree=` line.

    `leaf_weight` and `internal_weight` (LightGBM's per-node hessian sums) are
    written as zeros: mojoboost keeps node covers, not hessian sums, and
    LightGBM treats both fields as optional.
    """
    _check_writable_tree(tree, tree_index)
    var built = _LgbmOut()
    _ = _emit_lgbm_node(built, tree, mapper, 0, scale, bias, tree_index)

    var n_leaves = len(built.leaf_value)
    var zeros_leaf = List[Float64](capacity=n_leaves)
    zeros_leaf.resize(n_leaves, 0.0)
    var zeros_internal = List[Float64](capacity=len(built.split_feature))
    zeros_internal.resize(len(built.split_feature), 0.0)

    var what = String("tree ") + String(tree_index)
    var body = String("")
    body += "num_leaves=" + String(n_leaves) + "\n"
    body += "num_cat=0\n"
    body += "split_feature=" + _join_ints(built.split_feature) + "\n"
    body += (
        "split_gain=" + _join_floats(built.split_gain, what + " split gain")
        + "\n"
    )
    body += (
        "threshold=" + _join_floats(built.threshold, what + " threshold")
        + "\n"
    )
    body += "decision_type=" + _join_ints(built.decision_type) + "\n"
    body += "left_child=" + _join_ints(built.left_child) + "\n"
    body += "right_child=" + _join_ints(built.right_child) + "\n"
    body += (
        "leaf_value=" + _join_floats(built.leaf_value, what + " leaf value")
        + "\n"
    )
    body += "leaf_weight=" + _join_floats(zeros_leaf, what + " leaf weight")
    body += "\n"
    body += "leaf_count=" + _join_ints(built.leaf_count) + "\n"
    body += (
        "internal_value="
        + _join_floats(built.internal_value, what + " internal value")
        + "\n"
    )
    body += (
        "internal_weight="
        + _join_floats(zeros_internal, what + " internal weight")
        + "\n"
    )
    body += "internal_count=" + _join_ints(built.internal_count) + "\n"
    body += "is_linear=0\n"
    # The leaf values above already carry the shrinkage, exactly as LightGBM's
    # own do; this field only records what was applied, and an
    # iteration-0 tree that absorbed the base score records 1 the way
    # `Tree::AddBias` does.
    if bias != 0.0:
        body += "shrinkage=1\n"
    else:
        body += "shrinkage=" + _f64_text(scale, what + " shrinkage") + "\n"
    body += "\n"
    return body^


def _feature_infos(mapper: BinMapper) raises -> String:
    """LightGBM's `feature_infos`: each feature's `[min:max]`, or `none` for
    one with no edges (never split, so never binned into the file)."""
    var out = String("")
    for f in range(mapper.n_features):
        if f > 0:
            out += " "
        var lo = mapper.edge_offsets[f]
        var hi = mapper.edge_offsets[f + 1]
        if hi <= lo:
            out += "none"
            continue
        out += "["
        out += _f64_text(mapper.edges[lo], "a feature range")
        out += ":"
        out += _f64_text(mapper.edges[hi - 1], "a feature range")
        out += "]"
    return out^


def _feature_names(n_features: Int) -> String:
    """LightGBM's default column names. mojoboost's `Model` carries no
    feature names, so the file gets the names LightGBM itself invents when a
    dataset has none."""
    var out = String("")
    for f in range(n_features):
        if f > 0:
            out += " "
        out += "Column_" + String(f)
    return out^


def _assemble(
    mapper: BinMapper,
    objective_line: String,
    num_class: Int,
    trees_per_iteration: Int,
    blocks: List[String],
) raises -> String:
    var sizes = List[Int](capacity=len(blocks))
    for i in range(len(blocks)):
        sizes.append(blocks[i].byte_length())

    var out = String("tree\n")
    out += "version=" + _WRITE_VERSION + "\n"
    out += "num_class=" + String(num_class) + "\n"
    out += "num_tree_per_iteration=" + String(trees_per_iteration) + "\n"
    out += "label_index=0\n"
    out += "max_feature_idx=" + String(mapper.n_features - 1) + "\n"
    out += "objective=" + objective_line + "\n"
    out += "feature_names=" + _feature_names(mapper.n_features) + "\n"
    out += "feature_infos=" + _feature_infos(mapper) + "\n"
    out += "tree_sizes=" + _join_ints(sizes) + "\n"
    out += "\n"
    for i in range(len(blocks)):
        out += blocks[i]
    out += "end of trees\n"
    out += "\n"
    out += "feature_importances:\n"
    out += "\n"
    out += "parameters:\n"
    out += "[objective: " + objective_line + "]\n"
    out += "\n"
    out += "end of parameters\n"
    out += "\n"
    out += "pandas_categorical:null\n"
    return out^


def dump_lgbm_model(model: Model) raises -> String:
    """Render a single-output `Model` as a LightGBM model string.

    Leaf values are multiplied by the learning rate and the base score is
    folded into tree 0, so the file predicts exactly what the model does; the
    module docstring spells out why the round trip is faithful in predictions
    rather than in fields. Raises for a categorical or custom-objective
    model.
    """
    var objective_line = lgbm_objective_name(model.booster.objective)
    ref trees = model.booster.trees
    if len(trees) == 0 and model.booster.base_score != 0.0:
        raise Error(
            "cannot write a model with no trees and a nonzero base score (",
            model.booster.base_score,
            "): LightGBM's format keeps the base score inside the first"
            " tree's leaf values, and there is no first tree",
        )
    var blocks = List[String](capacity=len(trees))
    for t in range(len(trees)):
        var bias = model.booster.base_score if t == 0 else 0.0
        var block = String("Tree=") + String(t) + "\n"
        block += _tree_body(
            trees[t], model.mapper, model.booster.learning_rate, bias, t
        )
        block += "\n"
        blocks.append(block^)
    return _assemble(model.mapper, objective_line, 1, 1, blocks)


def dump_lgbm_multiclass_model(model: MulticlassModel) raises -> String:
    """Render a `MulticlassModel` as a LightGBM model string.

    The trees keep their round-major order, which is LightGBM's own. Each
    class's base score is folded into that class's iteration-0 tree.
    """
    var n_classes = model.booster.n_classes
    var objective_line = String("multiclass num_class:") + String(n_classes)
    ref trees = model.booster.trees
    if len(trees) == 0:
        for k in range(n_classes):
            if model.booster.base_scores[k] != 0.0:
                raise Error(
                    "cannot write a multiclass model with no trees and a"
                    " nonzero base score for class ",
                    k,
                    ": LightGBM's format keeps it inside the first tree's"
                    " leaf values, and there is no first tree",
                )
    var blocks = List[String](capacity=len(trees))
    for t in range(len(trees)):
        var bias = model.booster.base_scores[t] if t < n_classes else 0.0
        var block = String("Tree=") + String(t) + "\n"
        block += _tree_body(
            trees[t], model.mapper, model.booster.learning_rate, bias, t
        )
        block += "\n"
        blocks.append(block^)
    return _assemble(
        model.mapper, objective_line, n_classes, n_classes, blocks
    )


def save_lgbm_model(model: Model, path: String) raises:
    """Write a single-output `Model` to `path` in LightGBM's text format."""
    var text = dump_lgbm_model(model)
    with open(path, "w") as f:
        f.write(text)


def save_lgbm_multiclass_model(model: MulticlassModel, path: String) raises:
    """Write a `MulticlassModel` to `path` in LightGBM's text format."""
    var text = dump_lgbm_multiclass_model(model)
    with open(path, "w") as f:
        f.write(text)
