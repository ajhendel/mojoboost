"""Structured model inspection: the dump schema, built in Mojo.

A fitted model is a tree ensemble plus the binning that feeds it, and every
consumer that wants to look inside one (a Python notebook, the C ABI, a
report generator) wants the same facts in the same shape. This module
writes that shape once, as JSON, from the public structures a fitted model
already exposes: `Model`, `MulticlassModel`, `Tree`, and `BinMapper`.
`docs/MODEL_INSPECTION_SCHEMA.md` is the schema's normative statement and
this is one of its two reference implementations; the other,
`python/mojoboost/inspection.py`, builds the identical schema by parsing
the model text, so that it works against the bindings as they are built
today.

The schema is versioned separately from the model file format.
`DUMP_FORMAT_VERSION` here is the schema's, and it is bumped only for a
change a consumer written against the previous version could not survive;
adding an optional key does not bump it. `model_format_version` in the dump
records which save format the same model serializes to, so a consumer can
tell a v1 model's absent node covers from a v3 model's present ones.

What only this side has
-----------------------
Split gains. They are recorded when a node is split and are deliberately
not serialized (see serialize.mojo), so a dump built by parsing a model
file cannot carry them. `split_gains` exposes them on their own, which is
the smaller of the two hooks a binding can offer; `dump_model` is the whole
schema, for a consumer that would rather not parse anything.

Not offered
-----------
Leaf editing. A grown tree carries invariants that an arbitrary leaf edit
falsifies without any way to detect it: `Tree.count` is the training rows
that reached a node, an internal node's `value` is what that node held when
it was created, and `split_gain` was computed from the sums the leaf held
at growth time. Changing a leaf value leaves all three describing a model
that no longer exists. Inspection is read only until those invariants can
be restated and tested completely.
"""

from std.math import isnan

from .binning import BinMapper
from .categorical import CAT_MAX_BINS, cat_pool_contains
from .model import Model, MulticlassModel
from .monotone import MonotoneConstraints
from .tree import Tree

# The inspection schema's own version, not the model file format's.
comptime DUMP_FORMAT_VERSION = 1

# The save format a model written today serializes to (see serialize.mojo).
# Recorded in the dump so a consumer knows which optional fields a model of
# this vintage can carry.
comptime MODEL_FORMAT_VERSION = 3

# Larger than any finite Float64, so a comparison against it detects the
# infinities without reaching for an `isinf`.
comptime _F64_MAX = 1.7976931348623157e308


def _json_f64(x: Float64) -> String:
    """A JSON number, or `null` for a value JSON cannot spell.

    NaN and the infinities are not JSON numbers, and a dump that emitted
    them would not parse. `null` says "no value here", which is something a
    consumer can act on.
    """
    if isnan(x):
        return String("null")
    if x > _F64_MAX or x < -_F64_MAX:
        return String("null")
    return String(x)


def _json_bool(b: Bool) -> String:
    return String("true") if b else String("false")


def _json_string(s: String) -> String:
    """A quoted JSON string. Feature names come from user data, so the
    quote, the backslash, and the three whitespace escapes are handled
    rather than assumed away."""
    var out = String("\"")
    for cp in s.codepoint_slices():
        var c = String(cp)
        if c == "\"":
            out += "\\\""
        elif c == "\\":
            out += "\\\\"
        elif c == "\n":
            out += "\\n"
        elif c == "\r":
            out += "\\r"
        elif c == "\t":
            out += "\\t"
        else:
            out += c
    out += "\""
    return out^


def _json_int_list(values: List[Int]) -> String:
    var out = String("[")
    for i in range(len(values)):
        if i > 0:
            out += ","
        out += String(values[i])
    out += "]"
    return out^


def _json_f64_list(values: List[Float64]) -> String:
    var out = String("[")
    for i in range(len(values)):
        if i > 0:
            out += ","
        out += _json_f64(values[i])
    out += "]"
    return out^


def _feature_names(names: List[String], n_features: Int) -> List[String]:
    """The names to report. A model carries none of its own, so an empty or
    mismatched list falls back to LightGBM's `Column_0`, `Column_1`, ...,
    which is what LightGBM's own `feature_name()` reports in that case."""
    if len(names) == n_features:
        return names.copy()
    var out = List[String](capacity=n_features)
    for f in range(n_features):
        out.append(String("Column_") + String(f))
    return out^


def split_gains(trees: List[Tree]) -> String:
    """Per node split gains, one JSON array per tree, in tree order.

    The one fact a dump built from a saved model cannot recover: gains are
    recorded when a node is split and are not serialized. A leaf's entry is
    0.0, since a leaf earned no gain. This is the minimal shape a binding
    has to expose for `python/mojoboost/inspection.py` to report
    `has_split_gain: true`.
    """
    var out = String("[")
    for t in range(len(trees)):
        if t > 0:
            out += ","
        out += _json_f64_list(trees[t].split_gain)
    out += "]"
    return out^


def node_depths(tree: Tree) -> List[Int]:
    """Each node's depth in edges from the root: 0 at the root, so this is
    the quantity `TreeParams.max_depth` bounds. `Tree.depth` reports the
    maximum of this list."""
    var n_nodes = len(tree.feature)
    var depths = List[Int](capacity=n_nodes)
    depths.resize(n_nodes, 0)
    var stack: List[Int] = [0]
    while len(stack) > 0:
        var node = stack.pop()
        if tree.feature[node] < 0:
            continue
        var left = tree.left[node]
        var right = tree.right[node]
        depths[left] = depths[node] + 1
        depths[right] = depths[node] + 1
        stack.append(left)
        stack.append(right)
    return depths^


def split_ordinals(tree: Tree) -> List[Int]:
    """Each internal node's rank among this tree's internal nodes, in node
    order, and -1 for every leaf. The split counterpart of
    `Tree.leaf_ordinals`, and what the dump reports as `split_index`."""
    var out = List[Int](capacity=len(tree.feature))
    var next_ordinal = 0
    for i in range(len(tree.feature)):
        if tree.feature[i] < 0:
            out.append(-1)
        else:
            out.append(next_ordinal)
            next_ordinal += 1
    return out^


def category_bins(tree: Tree, node: Int) -> List[Int]:
    """The bin ids in node `node`'s category set, ascending, or an empty
    list for a numerical node. Bin 0 is never a member, so the missing,
    unseen, and dropped categories are absent by construction."""
    var out = List[Int]()
    var offset = tree.cat_offset[node]
    if offset < 0:
        return out^
    for b in range(CAT_MAX_BINS):
        if cat_pool_contains(tree.cat_bitset, offset, b):
            out.append(b)
    return out^


def category_codes(
    mapper: BinMapper, tree: Tree, node: Int
) raises -> List[Int]:
    """The raw category codes node `node` routes left, ascending.

    The bins of a categorical split mean nothing outside the model; the
    codes they stand for are what a reader of the dump recognizes. Feature
    f's i-th kept code is bin i + 1 (see categorical.mojo).
    """
    var out = List[Int]()
    var bins = category_bins(tree, node)
    if len(bins) == 0:
        return out^
    var feature = tree.feature[node]
    var begin = mapper.cats.offsets[feature]
    var n_codes = mapper.cats.offsets[feature + 1] - begin
    for i in range(len(bins)):
        var b = bins[i]
        if b < 1 or b > n_codes:
            raise Error(
                "tree node ",
                node,
                " routes bin ",
                b,
                ", which feature ",
                feature,
                " has no category for",
            )
        out.append(mapper.cats.codes[begin + b - 1])
    return out^


def has_threshold(mapper: BinMapper, feature: Int, bin: Int) -> Bool:
    """Whether a split at `bin` of `feature` has a raw-value threshold.

    False for a categorical feature, which has no edges at all, and for a
    bin index with no upper edge, which is what a node that routes only by
    bin id leaves behind.
    """
    if mapper.cats.is_cat(feature):
        return False
    var begin = mapper.edge_offsets[feature]
    var end = mapper.edge_offsets[feature + 1]
    return bin >= 0 and bin < end - begin


def threshold_value(
    mapper: BinMapper, feature: Int, bin: Int
) raises -> Float64:
    """The largest raw value a split at `bin` sends left.

    mojoboost splits on bin ids, and a value maps to the first bin whose
    upper edge it does not exceed, so `bin <= threshold` holds exactly when
    `value <= edges[threshold]`. The bin's upper edge is therefore the exact
    real-valued boundary and not an approximation of one, which is what lets
    a consumer route raw values through the dump and land where the model
    lands. Raises when `has_threshold` is False, rather than inventing an
    infinity for a bound that does not exist.
    """
    if not has_threshold(mapper, feature, bin):
        raise Error(
            "feature ", feature, " has no bin edge at ", bin
        )
    return mapper.edges[mapper.edge_offsets[feature] + bin]


def _write_feature_infos(
    mut out: String,
    mapper: BinMapper,
    names: List[String],
    monotone: MonotoneConstraints,
):
    """One record per feature: how it is binned, how missing values are
    routed through it, and the constraint sign it was grown under."""
    out += "\"feature_infos\":["
    for f in range(mapper.n_features):
        if f > 0:
            out += ","
        var missing_bin = mapper.missing_bin[f]
        out += "{\"index\":" + String(f)
        out += ",\"name\":" + _json_string(names[f])
        out += ",\"missing_bin\":" + String(missing_bin)
        out += ",\"missing_type\":"
        out += ("\"NaN\"" if missing_bin >= 0 else "\"None\"")
        out += ",\"monotone\":"
        if len(monotone.signs) == mapper.n_features:
            out += String(monotone.signs[f])
        else:
            out += "0"
        if mapper.cats.is_cat(f):
            var begin = mapper.cats.offsets[f]
            var end = mapper.cats.offsets[f + 1]
            var codes = List[Int](capacity=end - begin)
            for i in range(begin, end):
                codes.append(mapper.cats.codes[i])
            out += ",\"type\":\"categorical\""
            out += ",\"categories\":" + _json_int_list(codes)
            out += ",\"bin_upper_bounds\":null"
            # Bin 0 collects the missing, unseen, and dropped codes.
            out += ",\"num_bin\":" + String(len(codes) + 1)
        else:
            var begin = mapper.edge_offsets[f]
            var end = mapper.edge_offsets[f + 1]
            var edges = List[Float64](capacity=end - begin)
            for i in range(begin, end):
                edges.append(mapper.edges[i])
            var n_bins = len(edges) + 1
            if missing_bin >= 0:
                n_bins += 1
            out += ",\"type\":\"numerical\""
            out += ",\"categories\":null"
            out += ",\"bin_upper_bounds\":" + _json_f64_list(edges)
            out += ",\"num_bin\":" + String(n_bins)
        out += "}"
    out += "]"


def _write_node(
    mut out: String,
    mapper: BinMapper,
    tree: Tree,
    node: Int,
    depths: List[Int],
    splits: List[Int],
    leaves: List[Int],
    names: List[String],
) raises:
    """One node and, recursively, everything below it.

    A leaf and an internal node are told apart by which keys they carry: a
    leaf has `leaf_index` and an internal node has `split_index`, so a
    consumer branches on a key's presence rather than on a type tag.
    """
    if tree.feature[node] < 0:
        out += "{\"node_index\":" + String(node)
        out += ",\"leaf_index\":" + String(leaves[node])
        out += ",\"leaf_value\":" + _json_f64(tree.value[node])
        out += ",\"leaf_count\":" + _json_f64(tree.count[node])
        out += ",\"depth\":" + String(depths[node])
        out += "}"
        return

    var feature = tree.feature[node]
    var categorical = tree.cat_offset[node] >= 0
    var bin = tree.threshold_bin[node]
    out += "{\"node_index\":" + String(node)
    out += ",\"split_index\":" + String(splits[node])
    out += ",\"split_feature\":" + String(feature)
    out += ",\"split_feature_name\":" + _json_string(names[feature])
    out += ",\"decision_type\":"
    out += ("\"==\"" if categorical else "\"<=\"")
    out += ",\"threshold\":"
    if has_threshold(mapper, feature, bin):
        out += _json_f64(threshold_value(mapper, feature, bin))
    else:
        out += "null"
    out += ",\"threshold_bin\":" + String(bin)
    out += ",\"categories\":"
    if categorical:
        out += _json_int_list(category_codes(mapper, tree, node))
    else:
        out += "null"
    out += ",\"category_bins\":"
    if categorical:
        out += _json_int_list(category_bins(tree, node))
    else:
        out += "null"
    out += ",\"default_left\":" + _json_bool(tree.default_left[node])
    out += ",\"missing_bin\":" + String(tree.missing_bin[node])
    out += ",\"missing_type\":"
    out += (
        "\"NaN\"" if tree.missing_bin[node] >= 0 else "\"None\""
    )
    out += ",\"split_gain\":" + _json_f64(tree.split_gain[node])
    out += ",\"internal_value\":" + _json_f64(tree.value[node])
    out += ",\"internal_count\":" + _json_f64(tree.count[node])
    out += ",\"depth\":" + String(depths[node])
    out += ",\"left_child\":"
    _write_node(
        out, mapper, tree, tree.left[node], depths, splits, leaves, names
    )
    out += ",\"right_child\":"
    _write_node(
        out, mapper, tree, tree.right[node], depths, splits, leaves, names
    )
    out += "}"


def _write_trees(
    mut out: String,
    mapper: BinMapper,
    trees: List[Tree],
    learning_rate: Float64,
    per_iteration: Int,
    names: List[String],
) raises:
    out += "\"tree_info\":["
    for t in range(len(trees)):
        if t > 0:
            out += ","
        ref tree = trees[t]
        var depths = node_depths(tree)
        var splits = split_ordinals(tree)
        var leaves = tree.leaf_ordinals()
        var n_cat = 0
        var max_depth = 0
        for i in range(len(tree.feature)):
            if tree.cat_offset[i] >= 0:
                n_cat += 1
            if depths[i] > max_depth:
                max_depth = depths[i]
        out += "{\"tree_index\":" + String(t)
        out += ",\"iteration\":" + String(t // per_iteration)
        out += ",\"class_id\":" + String(t % per_iteration)
        out += ",\"num_leaves\":" + String(tree.n_leaves)
        out += ",\"num_nodes\":" + String(len(tree.feature))
        out += ",\"num_cat\":" + String(n_cat)
        out += ",\"max_depth\":" + String(max_depth)
        # Leaf values are stored unshrunk; the ensemble multiplies by this
        # when it predicts. See `leaf_value_is_shrunk` in the header.
        out += ",\"shrinkage\":" + _json_f64(learning_rate)
        out += ",\"tree_structure\":"
        _write_node(out, mapper, tree, 0, depths, splits, leaves, names)
        out += "}"
    out += "]"


def _has_node_counts(trees: List[Tree]) -> Bool:
    """Whether every tree carries covers. False for a model loaded from a
    v1 or v2 file, whose nodes predate them."""
    if len(trees) == 0:
        return False
    for t in range(len(trees)):
        if not trees[t].has_node_counts():
            return False
    return True


def _write_header(
    mut out: String,
    mapper: BinMapper,
    names: List[String],
    monotone: MonotoneConstraints,
    trees: List[Tree],
    learning_rate: Float64,
    base_scores: List[Float64],
    n_classes: Int,
    per_iteration: Int,
):
    """Everything above `tree_info`, which is the same for a single-output
    and a multiclass model apart from the class count and the base scores.
    """
    out += "{\"dump_format_version\":" + String(DUMP_FORMAT_VERSION)
    out += ",\"producer\":\"mojoboost\""
    out += ",\"model_format_version\":" + String(MODEL_FORMAT_VERSION)
    out += ",\"source\":\"native\""
    out += ",\"num_class\":" + String(n_classes)
    out += ",\"num_tree_per_iteration\":" + String(per_iteration)
    out += ",\"num_iteration\":" + String(len(trees) // per_iteration)
    out += ",\"learning_rate\":" + _json_f64(learning_rate)
    out += ",\"base_score\":" + _json_f64_list(base_scores)
    out += ",\"leaf_value_is_shrunk\":false"
    out += ",\"num_feature\":" + String(mapper.n_features)
    out += ",\"max_feature_idx\":" + String(mapper.n_features - 1)
    out += ",\"num_bin\":" + String(mapper.n_bins)
    out += ",\"feature_names\":["
    for f in range(len(names)):
        if f > 0:
            out += ","
        out += _json_string(names[f])
    out += "],"
    _write_feature_infos(out, mapper, names, monotone)
    out += ",\"monotone_constraints\":"
    if len(monotone.signs) == mapper.n_features:
        out += _json_int_list(monotone.signs)
    else:
        out += "null"
    out += ",\"has_split_gain\":true"
    out += ",\"has_node_count\":" + _json_bool(_has_node_counts(trees))
    out += ","


def dump_model(
    model: Model, feature_names: List[String] = []
) raises -> String:
    """A fitted single-output model as the inspection schema, in JSON.

    `feature_names` names the features; a model carries no names of its
    own, so an empty list gives LightGBM's `Column_0`, `Column_1`, ...

    The objective is reported as its trainer code, under `objective_code`.
    The name that code stands for is the consumer's to resolve: the code is
    what the model holds, and the LightGBM spelling of it belongs with the
    API that accepts those spellings.
    """
    var names = _feature_names(feature_names, model.mapper.n_features)
    var base_scores: List[Float64] = [model.booster.base_score]
    var out = String("")
    _write_header(
        out,
        model.mapper,
        names,
        model.booster.monotone,
        model.booster.trees,
        model.booster.learning_rate,
        base_scores,
        1,
        1,
    )
    out += "\"objective_code\":" + String(model.booster.objective) + ","
    _write_trees(
        out,
        model.mapper,
        model.booster.trees,
        model.booster.learning_rate,
        1,
        names,
    )
    out += "}"
    return out^


def dump_multiclass_model(
    model: MulticlassModel, feature_names: List[String] = []
) raises -> String:
    """A fitted softmax model as the inspection schema, in JSON.

    Trees keep the ensemble's round-major order, so `tree_index` runs
    straight through and each tree reports the `iteration` and `class_id`
    it belongs to rather than making a reader recompute them. The objective
    is the softmax one by construction, so there is no code to report.
    """
    var names = _feature_names(feature_names, model.mapper.n_features)
    var out = String("")
    _write_header(
        out,
        model.mapper,
        names,
        model.booster.monotone,
        model.booster.trees,
        model.booster.learning_rate,
        model.booster.base_scores,
        model.booster.n_classes,
        model.booster.n_classes,
    )
    out += "\"objective_code\":null,"
    _write_trees(
        out,
        model.mapper,
        model.booster.trees,
        model.booster.learning_rate,
        model.booster.n_classes,
        names,
    )
    out += "}"
    return out^
