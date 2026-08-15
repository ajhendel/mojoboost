"""Structured model inspection: the dump schema as JSON.

`model_dump.mojo` builds the schema from a fitted model; this module is the
one thing that turns it into text. Nothing here decides what a dump
contains, computes a depth, converts a threshold, or reconstructs a
category set: it walks `ModelDump` and writes JSON, so a consumer outside
Mojo can have the whole schema without a parser and without a Python
runtime.

    var dump = build_dump(model)          # model_dump.mojo
    var text = dump_json(dump)            # here

`docs/MODEL_INSPECTION_SCHEMA.md` is the schema's normative statement.
`dump_model` and `dump_multiclass_model` are the two shorthands that build
and render in one call, which is what the bindings and the C ABI want.

Shape
-----
The JSON nests each tree under `tree_structure`, root first, because that
is what the schema documents and what a LightGBM reader expects.
`ModelDump` holds the same nodes as a flat table with child links; the
nesting here is a rendering of that table and adds nothing to it.

The schema's primitives (`node_depths`, `split_ordinals`, `category_bins`,
`category_codes`, `has_threshold`, `threshold_value`, the two capability
predicates, and the version constants) live in `model_dump.mojo` and are
re-exported here, so a caller that reaches for `inspection` finds them
where the schema doc says they are.

Editing
-------
Nothing here writes to a model. Writing lives in `model_editing.mojo`
(rollback, leaf outputs, shuffling, refit, bounds), which checks the
invariants a fitted tree records before every write; its
`MODEL_EDITING_SUPPORTED` and `model_editing_status_json` are re-exported
here so a consumer that asks this module "can I edit?" gets that module's
answer.
"""

from std.math import isnan

from .model import Model, MulticlassModel
from .model_dump import (
    DUMP_FORMAT_VERSION,
    MODEL_FORMAT_VERSION,
    DumpFeature,
    DumpNode,
    DumpTree,
    ModelDump,
    build_dump,
    build_multiclass_dump,
    category_bins,
    category_codes,
    dump_bin_value,
    dump_goes_left,
    dump_leaf_index,
    dump_raw_scores,
    dump_split_values,
    has_node_counts,
    has_split_gains,
    has_threshold,
    node_depths,
    node_parents,
    split_ordinals,
    threshold_value,
)
from .tree import Tree
from .model_editing import (
    MODEL_EDITING_SUPPORTED as _MODEL_EDITING_SUPPORTED,
    model_editing_status_json as _model_editing_status_json,
)

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


def split_gains(trees: List[Tree]) -> String:
    """Per node split gains, one JSON array per tree, in tree order.

    A leaf's entry is 0.0, since a leaf earned no gain. Model format v4
    serializes gains (see serialize.mojo), so a model saved and loaded by a
    current build carries its own; this stays as the smallest shape a
    binding can expose to give a dump built from an older file's *text* its
    gains back, from a live handle that still holds them. The whole dump
    carries them already, so a build that exposes the dump does not need
    this.
    """
    var out = String("[")
    for t in range(len(trees)):
        if t > 0:
            out += ","
        out += _json_f64_list(trees[t].split_gain)
    out += "]"
    return out^


def _write_feature(mut out: String, info: DumpFeature):
    out += "{\"index\":" + String(info.index)
    out += ",\"name\":" + _json_string(info.name)
    out += ",\"missing_bin\":" + String(info.missing_bin)
    out += ",\"missing_type\":"
    out += ("\"NaN\"" if info.has_missing_bin() else "\"None\"")
    out += ",\"monotone\":" + String(info.monotone)
    if info.is_categorical:
        out += ",\"type\":\"categorical\""
        out += ",\"categories\":" + _json_int_list(info.categories)
        out += ",\"bin_upper_bounds\":null"
    else:
        out += ",\"type\":\"numerical\""
        out += ",\"categories\":null"
        out += ",\"bin_upper_bounds\":" + _json_f64_list(
            info.bin_upper_bounds
        )
    out += ",\"num_bin\":" + String(info.num_bin)
    out += "}"


def _write_node(
    mut out: String,
    dump: ModelDump,
    tree: DumpTree,
    node: Int,
) raises:
    """One node and, recursively, everything below it.

    A leaf and an internal node are told apart by which keys they carry: a
    leaf has `leaf_index` and an internal node has `split_index`, so a
    consumer branches on a key's presence rather than on a type tag.
    """
    ref rec = tree.nodes[node]
    if rec.is_leaf:
        out += "{\"node_index\":" + String(rec.node_index)
        out += ",\"leaf_index\":" + String(rec.leaf_index)
        out += ",\"leaf_value\":" + _json_f64(rec.value)
        out += ",\"leaf_count\":" + _json_f64(rec.count)
        out += ",\"depth\":" + String(rec.depth)
        out += "}"
        return

    out += "{\"node_index\":" + String(rec.node_index)
    out += ",\"split_index\":" + String(rec.split_index)
    out += ",\"split_feature\":" + String(rec.split_feature)
    out += ",\"split_feature_name\":" + _json_string(
        dump.features[rec.split_feature].name
    )
    out += ",\"decision_type\":"
    out += ("\"==\"" if rec.is_categorical else "\"<=\"")
    out += ",\"threshold\":"
    if rec.has_threshold:
        out += _json_f64(rec.threshold)
    else:
        out += "null"
    out += ",\"threshold_bin\":" + String(rec.threshold_bin)
    out += ",\"categories\":"
    if rec.is_categorical:
        out += _json_int_list(rec.categories)
    else:
        out += "null"
    out += ",\"category_bins\":"
    if rec.is_categorical:
        out += _json_int_list(rec.category_bins)
    else:
        out += "null"
    out += ",\"default_left\":" + _json_bool(rec.default_left)
    out += ",\"missing_bin\":" + String(rec.missing_bin)
    out += ",\"missing_type\":"
    out += ("\"NaN\"" if rec.missing_bin >= 0 else "\"None\"")
    out += ",\"split_gain\":"
    # Null rather than a zero a consumer could mistake for a measured gain:
    # a model read back from a file has no gains at all.
    if dump.has_split_gain:
        out += _json_f64(rec.split_gain)
    else:
        out += "null"
    out += ",\"internal_value\":" + _json_f64(rec.value)
    out += ",\"internal_count\":" + _json_f64(rec.count)
    out += ",\"depth\":" + String(rec.depth)
    out += ",\"left_child\":"
    _write_node(out, dump, tree, rec.left)
    out += ",\"right_child\":"
    _write_node(out, dump, tree, rec.right)
    out += "}"


def _write_tree(mut out: String, dump: ModelDump, tree: DumpTree) raises:
    out += "{\"tree_index\":" + String(tree.tree_index)
    out += ",\"iteration\":" + String(tree.iteration)
    out += ",\"class_id\":" + String(tree.class_id)
    out += ",\"num_leaves\":" + String(tree.num_leaves)
    out += ",\"num_nodes\":" + String(tree.num_nodes())
    out += ",\"num_cat\":" + String(tree.num_cat)
    out += ",\"max_depth\":" + String(tree.max_depth)
    # Leaf values are stored unshrunk; the ensemble multiplies by this when
    # it predicts. See `leaf_value_is_shrunk` in the header.
    out += ",\"shrinkage\":" + _json_f64(tree.shrinkage)
    out += ",\"tree_structure\":"
    _write_node(out, dump, tree, 0)
    out += "}"


def dump_json(dump: ModelDump) raises -> String:
    """A built dump as the inspection schema, in JSON.

    The whole schema and nothing else: what the dump carries is
    `model_dump.mojo`'s decision, and this reports it.
    """
    var out = String("{\"dump_format_version\":")
    out += String(dump.dump_format_version)
    out += ",\"producer\":\"mojotrees\""
    out += ",\"model_format_version\":" + String(dump.model_format_version)
    out += ",\"source\":\"native\""
    out += ",\"objective_code\":"
    if dump.has_objective_code:
        out += String(dump.objective_code)
    else:
        out += "null"
    out += ",\"num_class\":" + String(dump.num_class)
    out += ",\"num_tree_per_iteration\":" + String(
        dump.num_tree_per_iteration
    )
    out += ",\"num_iteration\":" + String(dump.num_iteration)
    out += ",\"learning_rate\":" + _json_f64(dump.learning_rate)
    out += ",\"base_score\":" + _json_f64_list(dump.base_score)
    out += ",\"leaf_value_is_shrunk\":false"
    out += ",\"num_feature\":" + String(dump.num_feature)
    out += ",\"max_feature_idx\":" + String(dump.max_feature_idx())
    out += ",\"num_bin\":" + String(dump.num_bin)
    out += ",\"feature_names\":["
    for f in range(len(dump.features)):
        if f > 0:
            out += ","
        out += _json_string(dump.features[f].name)
    out += "]"
    out += ",\"feature_infos\":["
    for f in range(len(dump.features)):
        if f > 0:
            out += ","
        _write_feature(out, dump.features[f])
    out += "]"
    out += ",\"monotone_constraints\":"
    if dump.is_constrained():
        out += _json_int_list(dump.monotone_constraints)
    else:
        out += "null"
    out += ",\"has_split_gain\":" + _json_bool(dump.has_split_gain)
    out += ",\"has_node_count\":" + _json_bool(dump.has_node_count)
    out += ",\"tree_info\":["
    for t in range(len(dump.trees)):
        if t > 0:
            out += ","
        _write_tree(out, dump, dump.trees[t])
    out += "]}"
    return out^


# Whether this build can edit a fitted model in place, and the status that
# says which operations that covers. Both are `model_editing.mojo`'s: it is
# the implementation, so it is the one place the claim is made, and this
# module re-exports it so a consumer that reached the status here keeps
# reaching it. `python/mojotrees/inspection.py` mirrors the flag as
# `MODEL_EDITING_SUPPORTED`.
comptime MODEL_EDITING_SUPPORTED = _MODEL_EDITING_SUPPORTED
comptime model_editing_status_json = _model_editing_status_json


def dump_model(
    model: Model, feature_names: List[String] = []
) raises -> String:
    """A fitted single-output model as the inspection schema, in JSON.

    `feature_names` names the features; a model carries no names of its own,
    so an empty list gives LightGBM's `Column_0`, `Column_1`, ...
    """
    return dump_json(build_dump(model, feature_names))


def dump_multiclass_model(
    model: MulticlassModel, feature_names: List[String] = []
) raises -> String:
    """A fitted softmax model as the inspection schema, in JSON."""
    return dump_json(build_multiclass_dump(model, feature_names))
