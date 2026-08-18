"""Structured model inspection, from the native dump.

`src/mojotrees/model_dump.mojo` builds a fitted model's inspection schema
and `src/mojotrees/inspection.mojo` renders it as JSON. This module hands
the built dump to Python as plain Python objects, which is what lets
`python/mojotrees/inspection.py` stop rebuilding the schema by parsing
`Booster.model_to_string()`.

That parser is a second implementation of facts Mojo already knows: how a
tree is laid out, what a split bin's upper edge means, how a category set
maps back to codes, and how a row routes. It is marked in that file as the
compatibility path, with a deletion point; these functions are what it is
waiting for. `handoffs/migration_19_model_inspection.md (deleted, recover with git log --all --diff-filter=D -- handoffs/migration_19_model_inspection.md)` sections 4 and 8
specified the entry points and the deletion, and this module implements
that specification.

The suffix convention is the one every other model accessor uses
(`n_features` / `n_features_multiclass`): one entry point per model kind,
because a handle is a `Model` or a `MulticlassModel` and the Python layer
already knows which it holds.

Two things this deliberately does not do:

- It does not nest the trees. A tree crosses as a flat node table with
  `left` and `right` as indices into it, and `inspection._nested_node`
  nests them. Nesting on this side would mean building the same dict twice.
- It does not parse model text, in either direction. Whether a given model
  carries per node split gains is `has_split_gain`'s answer, computed
  natively: a model saved by a current build keeps them (format v4), and
  one read back from a v1, v2, or v3 file does not, because those formats
  dropped them and a fitted tree cannot recompute them. A node's
  `split_gain` is None in that case, never a zero a consumer could mistake
  for a measured gain.

Wave 2's dump-consuming functions rebuild the dump per call. That is right
for the one-row conformance checks the Python layer makes and wrong for a
loop over rows; if a caller ever needs per-row routing at scale, the fix
is a handle that owns a built `ModelDump`, not a faster rebuild.
"""

from std.python import Python, PythonObject

from binding_support import (
    f64_buffer,
    py_dict,
    py_f64_list,
    py_int_list,
    str_sequence,
)

from mojotrees.inspection import dump_json
from mojotrees.model import Model, MulticlassModel
from mojotrees.model_dump import (
    DUMP_FORMAT_VERSION,
    MODEL_FORMAT_VERSION,
    DumpFeature,
    DumpNode,
    DumpTree,
    ModelDump,
    build_dump,
    build_multiclass_dump,
    dump_leaf_index as mojo_dump_leaf_index,
    dump_raw_scores as mojo_dump_raw_scores,
    dump_split_values as mojo_dump_split_values,
)
from mojotrees.serialize import model_file_kind as mojo_model_file_kind


# -- the schema, as Python objects ---------------------------------------


def _py_feature(info: DumpFeature) raises -> PythonObject:
    """One feature record, already in the schema's shape."""
    var out = py_dict()
    out["index"] = PythonObject(info.index)
    out["name"] = PythonObject(info.name)
    out["missing_bin"] = PythonObject(info.missing_bin)
    if info.has_missing_bin():
        out["missing_type"] = PythonObject("NaN")
    else:
        out["missing_type"] = PythonObject("None")
    out["monotone"] = PythonObject(info.monotone)
    out["num_bin"] = PythonObject(info.num_bin)
    if info.is_categorical:
        out["type"] = PythonObject("categorical")
        out["categories"] = py_int_list(info.categories)
        out["bin_upper_bounds"] = PythonObject(None)
    else:
        out["type"] = PythonObject("numerical")
        out["categories"] = PythonObject(None)
        out["bin_upper_bounds"] = py_f64_list(info.bin_upper_bounds)
    return out^


def _py_node(
    dump: ModelDump, node: DumpNode, has_gain: Bool
) raises -> PythonObject:
    """One node of the flat node table. `left` and `right` stay indices
    into that table; the Python layer nests them."""
    var out = py_dict()
    out["node_index"] = PythonObject(node.node_index)
    out["depth"] = PythonObject(node.depth)
    out["parent"] = PythonObject(node.parent)
    out["left"] = PythonObject(node.left)
    out["right"] = PythonObject(node.right)
    out["is_leaf"] = PythonObject(node.is_leaf)
    out["leaf_index"] = PythonObject(node.leaf_index)
    out["split_index"] = PythonObject(node.split_index)
    out["value"] = PythonObject(node.value)
    out["count"] = PythonObject(node.count)
    if node.is_leaf:
        out["is_linear"] = PythonObject(node.is_linear)
        if node.is_linear:
            out["leaf_const"] = PythonObject(node.leaf_const)
            out["leaf_features"] = py_int_list(node.leaf_features)
            out["leaf_coeff"] = py_f64_list(node.leaf_coeff)
        return out^
    out["split_feature"] = PythonObject(node.split_feature)
    out["split_feature_name"] = PythonObject(
        dump.features[node.split_feature].name
    )
    if node.is_categorical:
        out["decision_type"] = PythonObject("==")
    else:
        out["decision_type"] = PythonObject("<=")
    out["threshold_bin"] = PythonObject(node.threshold_bin)
    if node.has_threshold:
        out["threshold"] = PythonObject(node.threshold)
    else:
        out["threshold"] = PythonObject(None)
    if node.is_categorical:
        out["categories"] = py_int_list(node.categories)
        out["category_bins"] = py_int_list(node.category_bins)
    else:
        out["categories"] = PythonObject(None)
        out["category_bins"] = PythonObject(None)
    out["default_left"] = PythonObject(node.default_left)
    out["missing_bin"] = PythonObject(node.missing_bin)
    if node.missing_bin >= 0:
        out["missing_type"] = PythonObject("NaN")
    else:
        out["missing_type"] = PythonObject("None")
    # Null and not a zero a consumer could mistake for a measured gain: a
    # model read back from a file carries no gains at all.
    if has_gain:
        out["split_gain"] = PythonObject(node.split_gain)
    else:
        out["split_gain"] = PythonObject(None)
    return out^


def _py_tree(dump: ModelDump, tree: DumpTree) raises -> PythonObject:
    var out = py_dict()
    out["tree_index"] = PythonObject(tree.tree_index)
    out["iteration"] = PythonObject(tree.iteration)
    out["class_id"] = PythonObject(tree.class_id)
    out["num_leaves"] = PythonObject(tree.num_leaves)
    out["num_nodes"] = PythonObject(tree.num_nodes())
    out["num_cat"] = PythonObject(tree.num_cat)
    out["max_depth"] = PythonObject(tree.max_depth)
    out["shrinkage"] = PythonObject(tree.shrinkage)
    var nodes = Python.list()
    for i in range(len(tree.nodes)):
        nodes.append(_py_node(dump, tree.nodes[i], dump.has_split_gain))
    out["nodes"] = nodes^
    return out^


def _py_dump(dump: ModelDump) raises -> PythonObject:
    """A built dump as plain Python objects. Floats cross as C doubles, so
    the payload holds the model's own bits and not a decimal of them."""
    var out = py_dict()
    out["dump_format_version"] = PythonObject(dump.dump_format_version)
    out["model_format_version"] = PythonObject(dump.model_format_version)
    if dump.has_objective_code:
        out["objective_code"] = PythonObject(dump.objective_code)
    else:
        out["objective_code"] = PythonObject(None)
    out["num_class"] = PythonObject(dump.num_class)
    out["num_tree_per_iteration"] = PythonObject(dump.num_tree_per_iteration)
    out["num_iteration"] = PythonObject(dump.num_iteration)
    out["learning_rate"] = PythonObject(dump.learning_rate)
    out["base_score"] = py_f64_list(dump.base_score)
    out["num_feature"] = PythonObject(dump.num_feature)
    out["num_bin"] = PythonObject(dump.num_bin)
    if dump.is_constrained():
        out["monotone_constraints"] = py_int_list(dump.monotone_constraints)
    else:
        out["monotone_constraints"] = PythonObject(None)
    out["has_split_gain"] = PythonObject(dump.has_split_gain)
    out["has_node_count"] = PythonObject(dump.has_node_count)
    out["linear_tree"] = PythonObject(dump.linear_tree)
    var infos = Python.list()
    for f in range(len(dump.features)):
        infos.append(_py_feature(dump.features[f]))
    out["feature_infos"] = infos^
    var trees = Python.list()
    for t in range(len(dump.trees)):
        trees.append(_py_tree(dump, dump.trees[t]))
    out["trees"] = trees^
    return out^


# -- wave 1: the dump ----------------------------------------------------


def dump_model(
    model: PythonObject,
    feature_names: PythonObject,
    n_names: PythonObject,
) raises -> PythonObject:
    """The fitted model as the inspection schema, as plain Python objects.

    `feature_names` is a sequence of `n_names` strings, and an empty one is
    what makes the builder fall back to `Column_0`, `Column_1`, ....
    `python/mojotrees/inspection.py` resolves the override, the names the
    model carries, and that fallback in that order before calling, and it
    nests the flat node tables on arrival. See
    docs/MODEL_INSPECTION_SCHEMA.md for every key.
    """
    var m = model.downcast_value_ptr[Model]()
    return _py_dump(
        build_dump(m[], str_sequence(feature_names, Int(py=n_names)))
    )


def dump_model_multiclass(
    model: PythonObject,
    feature_names: PythonObject,
    n_names: PythonObject,
) raises -> PythonObject:
    """`dump_model` for a softmax model. Its trees keep the ensemble's
    round-major order and each reports the iteration and class it belongs
    to."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    return _py_dump(
        build_multiclass_dump(
            m[], str_sequence(feature_names, Int(py=n_names))
        )
    )


# -- wave 2: the derived queries -----------------------------------------


def split_values(
    model: PythonObject, feature: PythonObject
) raises -> PythonObject:
    """Every threshold the ensemble splits `feature` at, depth first per
    tree, root first.

    The values are the split bins' upper edges, which is the exact
    boundary routing uses. Raises for a categorical feature, whose splits
    are category sets and have no value to bin.
    """
    var m = model.downcast_value_ptr[Model]()
    return py_f64_list(
        mojo_dump_split_values(build_dump(m[]), Int(py=feature))
    )


def split_values_multiclass(
    model: PythonObject, feature: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return py_f64_list(
        mojo_dump_split_values(
            build_multiclass_dump(m[]), Int(py=feature)
        )
    )


def dump_raw_scores(
    model: PythonObject, x_addr: PythonObject, n_features: PythonObject
) raises -> PythonObject:
    """Raw scores for one raw example, routed through the dump's own bin
    edges and node records rather than through the model's prediction
    path. One entry for a single-output model, one per class for a softmax
    one.

    This is the check that pins the dump to the model: the two agree or
    the dump is wrong. It is not a prediction entry point; `predict_raw`
    is.
    """
    var m = model.downcast_value_ptr[Model]()
    var nf = Int(py=n_features)
    return py_f64_list(
        mojo_dump_raw_scores(build_dump(m[]), f64_buffer(Int(py=x_addr), nf))
    )


def dump_raw_scores_multiclass(
    model: PythonObject, x_addr: PythonObject, n_features: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    var nf = Int(py=n_features)
    return py_f64_list(
        mojo_dump_raw_scores(
            build_multiclass_dump(m[]), f64_buffer(Int(py=x_addr), nf)
        )
    )


def dump_leaf_index(
    model: PythonObject,
    tree_index: PythonObject,
    x_addr: PythonObject,
    n_features: PythonObject,
) raises -> PythonObject:
    """The leaf ordinal one raw example reaches in one tree, routed through
    the dump. The same numbering `predict_leaf` reports."""
    var m = model.downcast_value_ptr[Model]()
    var nf = Int(py=n_features)
    return PythonObject(
        mojo_dump_leaf_index(
            build_dump(m[]),
            Int(py=tree_index),
            f64_buffer(Int(py=x_addr), nf),
        )
    )


def dump_leaf_index_multiclass(
    model: PythonObject,
    tree_index: PythonObject,
    x_addr: PythonObject,
    n_features: PythonObject,
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    var nf = Int(py=n_features)
    return PythonObject(
        mojo_dump_leaf_index(
            build_multiclass_dump(m[]),
            Int(py=tree_index),
            f64_buffer(Int(py=x_addr), nf),
        )
    )


def objective_code(model: PythonObject) raises -> PythonObject:
    """The trainer's objective code the model carries.

    A softmax model has no single code, which is why there is no
    multiclass counterpart: the Python layer answers `"multiclass"` from
    the handle's kind. This takes a *model handle*; the name-to-code
    resolver is `objective_code_of_name` in `objective_bindings`.
    """
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(m[].booster.objective)


# -- wave 3: JSON, for consumers outside Python --------------------------


def dump_model_json(
    model: PythonObject,
    feature_names: PythonObject,
    n_names: PythonObject,
) raises -> PythonObject:
    """The whole schema as a JSON string, for the C ABI and the CLI.

    The Python layer does not use this: JSON floats are decimal, and the
    object payload `dump_model` returns carries the model's exact bits.
    """
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(
        dump_json(
            build_dump(m[], str_sequence(feature_names, Int(py=n_names)))
        )
    )


def dump_model_json_multiclass(
    model: PythonObject,
    feature_names: PythonObject,
    n_names: PythonObject,
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(
        dump_json(
            build_multiclass_dump(
                m[], str_sequence(feature_names, Int(py=n_names))
            )
        )
    )


# -- model format metadata -----------------------------------------------


def model_file_kind(path: PythonObject) raises -> PythonObject:
    """Which loader a saved model needs, `"objective"` or `"multiclass"`.

    Reads the file header only. This is what lets a caller pick between
    `load` and `load_multiclass` instead of calling one and reading the
    exception, which is what `Booster._load_path` does today: an exception
    from a corrupt file and an exception from the wrong loader are not the
    same thing and should not be handled the same way.

    Not a rename of `file_kind`, which is bound alongside it and answers
    `"objective"`, `"multiclass"`, or `"dataset"` for a caller that does
    not yet know what it was handed. This one is for the caller that knows
    it wants a model: it refuses a prepared dataset file by name, rather
    than reporting a third kind the model loaders cannot use.
    """
    return PythonObject(mojo_model_file_kind(String(py=path)))


def model_format_versions() raises -> PythonObject:
    """The two schema versions a consumer branches on.

    Keys: `model_format_version`, the save format a model written today
    serializes to, and `dump_format_version`, the inspection schema's own.
    Both are recorded in every dump; this is how a caller reads them
    without building one.
    """
    var out = py_dict()
    out["model_format_version"] = PythonObject(MODEL_FORMAT_VERSION)
    out["dump_format_version"] = PythonObject(DUMP_FORMAT_VERSION)
    return out^
