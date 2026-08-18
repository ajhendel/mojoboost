"""Structured model inspection: the native dump, in Python shapes.

LightGBM's `Booster.dump_model()`, `Booster.trees_to_dataframe()`, and
`Booster.get_split_value_histogram()` all answer the same question in
different shapes: what does this ensemble actually contain? mojotrees
answers it once, in Mojo, and this module converts that answer into the
shapes a Python consumer wants.

    from mojotrees import MojoTreesRegressor
    from mojotrees.inspection import dump_model, trees_to_dataframe

    model = MojoTreesRegressor(n_estimators=20).fit(X, y)
    dump = dump_model(model)
    frame = trees_to_dataframe(model)

Where the dump comes from
-------------------------
`src/mojotrees/model_dump.mojo` builds the schema from the fitted model
itself: node depths and parents, leaf and split ordinals, the raw value a
split bin stands for, the category codes a categorical node routes left,
and which optional facts the model can answer at all. The bindings hand
that over as plain Python objects, floats included, and this module nests
the flat node tables into the documented `tree_structure`, names the
objective its LightGBM name, and derives the records, the DataFrame, and
the split value histogram.

That is the whole division of labor. Nothing here recomputes a depth,
converts a threshold, reconstructs a category set, or decides what the
schema contains: `docs/MODEL_INSPECTION_SCHEMA.md` is the schema and Mojo
is where it is built.

Until the native binding lands
------------------------------
The extension module does not expose the dump yet, so there is a fallback
that parses `Booster.model_to_string()` and builds the same schema in
Python. It is fenced off at the bottom of this file under one banner, and
it is what every function here falls back to when the hook is missing.
`handoffs/migration_19_model_inspection.md (deleted, recover with git log --all --diff-filter=D -- handoffs/migration_19_model_inspection.md)` names the exact binding
functions and the exact lines to delete once they exist.

Its one gap used to be the save format's: split gains were not serialized,
so a dump built from the text reported `has_split_gain: False`. Model
format v4 carries them (see `src/mojotrees/serialize.mojo`), so a model
written by a current build dumps its gains whichever path built the dump.
A model read from a v1, v2, or v3 file has none to report, and says so
through `has_split_gain` rather than through a zero.

Editing
-------
This module reads; it does not write. Leaf editing (LightGBM's
`set_leaf_output`), `rollback_one_iter`, `shuffle_models`, `refit`, and
the `lower_bound` / `upper_bound` readers live on `Booster` in basic.py
and are implemented natively in src/mojotrees/model_editing.mojo, which
keeps every invariant this module reports true across an edit: node
covers, internal node values, and split gains all describe the tree at
growth time and are unchanged by a leaf write, and a model recording
monotone constraints clamps or refuses a value that would break them.
`model_editing_support()` lists each operation with `supported` and the
rule it keeps, so a consumer can branch on it rather than probe for a
method. `leaf_outputs()` is the reading half of a leaf write.
"""

import struct as _struct

from . import _arrays

__all__ = [
    "DUMP_FORMAT_VERSION",
    "SUPPORTED_MODEL_FORMAT_VERSIONS",
    "MODEL_EDITING_SUPPORTED",
    "OBJECTIVE_NAMES",
    "dump_model",
    "parse_model_string",
    "trees_to_records",
    "trees_to_dataframe",
    "split_values",
    "get_split_value_histogram",
    "feature_importance",
    "leaf_outputs",
    "model_editing_support",
    "leaf_index_of",
    "raw_scores",
    "booster_of",
    "feature_name_of",
    "n_features_of",
    "n_iter_of",
    "objective_of",
    "best_score_of",
]

#: The schema version this module emits and `docs/MODEL_INSPECTION_SCHEMA.md`
#: describes. Bumped only for a change a consumer written against the
#: previous version could not survive; new optional keys do not bump it.
#: `src/mojotrees/model_dump.mojo` carries the same number, and a dump
#: reports it under `dump_format_version` whichever side built it.
DUMP_FORMAT_VERSION = 1

#: Model file format versions (src/mojotrees/serialize.mojo) the fallback
#: parser can read. v1 and v2 predate node covers, so a dump built from one
#: reports `has_node_count: False`; v1, v2, and v3 all predate split gains,
#: so a dump built from one reports `has_split_gain: False`. The native
#: dump reports the format the model would serialize to and reads nothing,
#: so this does not bound it.
#:
#: **v5 is here because the sections are, not because the number was
#: raised.** A v5 file is one carrying any of three optional sections, and
#: this parser handles two of them: `ctr` (`_parse_ctr`, the fitted ordered
#: target statistics) and `usable` (`_parse_usable`, the split-search pool).
#: The third, `linear` (linear leaves), is **refused by name** in
#: `parse_model_string` -- reconstructing linear leaves here would be a
#: second implementation of `src/mojotrees/linear_tree.mojo`, and skipping
#: the section would report a linear model as a constant-leaf one, which is
#: the mis-parse the old blanket refusal existed to prevent. So the number
#: covers exactly what the parser genuinely reads, and what it does not read
#: says so by name.
#:
#: Parsing a section is not the same as describing it. `parse_model_string`
#: returns a v5 CTR model's tables; `dump_model`'s fallback then refuses that
#: model, because the dump schema is sized by `mapper.n_features` and a CTR
#: tree splits past it. That refusal mirrors `ctr.check_ctr_model_support`
#: in the Mojo dump, so both sides of `dump_model` say the same thing.
SUPPORTED_MODEL_FORMAT_VERSIONS = (1, 2, 3, 4, 5)

#: Whether this build can edit a fitted model in place. Mirrors
#: `MODEL_EDITING_SUPPORTED` in src/mojotrees/model_editing.mojo (re-exported
#: by src/mojotrees/inspection.mojo), which is where the claim is made and
#: checked; `model_editing_support()` lists the operations. The Python
#: entry points are `Booster.rollback_one_iter`, `Booster.set_leaf_output`,
#: `Booster.shuffle_models`, `Booster.refit`, `Booster.lower_bound` and
#: `Booster.upper_bound`.
MODEL_EDITING_SUPPORTED = True

#: Objective code to LightGBM's canonical name for it. The codes are the
#: trainer's, declared in src/mojotrees/objective_registry.mojo and mirrored
#: by the estimators' objective tables; `test_inspection.py` checks this
#: table against those tables so the two cannot drift apart silently.
#:
#: The name and not the code is what LightGBM's `objective_` attribute
#: reports, and spelling a LightGBM parameter is this layer's job: the model
#: holds the code, and `src/mojotrees/lgbm_model_io.mojo` spells the same
#: codes for a LightGBM *file*, which needs different decorations (see the
#: handoff).
#:
#: This is the pure-Python side of `model_dump`, so it must name every code a
#: model file can carry, not every code this build can fit. It listed
#: thirteen of the twenty-one assigned codes: `MULTICLASS` and the seven
#: reserved ones were absent, and `_objective_name` reported them as
#: `objective_16`. `5` stays `regression_l1` rather than the registry's `mae`
#: on purpose, because that is the spelling a LightGBM file uses and this
#: table is read next to one.
OBJECTIVE_NAMES = {
    0: "regression",
    1: "binary",
    2: "poisson",
    3: "huber",
    4: "quantile",
    5: "regression_l1",
    6: "custom",
    7: "lambdarank",
    8: "gamma",
    9: "tweedie",
    10: "mape",
    11: "fair",
    12: "cross_entropy",
    # CatBoost's three ranking losses, its two survival losses, and
    # multi-target regression. Reserved: registered, serializable, and
    # reachable by no trainer this build connects, which is why naming them
    # here is a read path only and routes nothing. See `objective_reserved`
    # and `objective_reserved_trainer` in the registry.
    13: "query_rmse",
    14: "pair_logit",
    15: "yeti_rank",
    16: "cox",
    17: "survival_aft",
    # Negative because one iteration grows more than one tree.
    -1: "multiclass",
    -2: "multi_rmse",
    -3: "multi_rmse_with_missing",
}


# -- the model handle ----------------------------------------------------


def _booster(model):
    """The `Booster` behind an estimator or a `Booster`.

    `booster_` is read off the class before it is read off the instance:
    `NotFittedError` derives from `AttributeError`, so a `getattr` with a
    default would swallow an unfitted estimator's complaint and report it
    as the wrong kind of mistake.
    """
    if callable(getattr(model, "model_to_string", None)):
        return model
    if getattr(type(model), "booster_", None) is None:
        raise TypeError(
            "model inspection takes a fitted estimator, a mojotrees.Booster, "
            "or the text model_to_string() produces, not "
            f"{type(model).__name__}"
        )
    return model.booster_


def _hook(booster, name):
    """The extension module's entry point called `name` for this kind of
    model, or None when this build does not expose it.

    A handle is a single-output model or a softmax one, and the extension
    module takes one entry point per kind, as it does for every other model
    accessor (`n_features` / `n_features_multiclass`).
    """
    from . import _mojotrees

    if getattr(booster, "_n_classes", 0):
        name += "_multiclass"
    return getattr(_mojotrees, name, None)


def _row_buffer(row):
    """One raw example as a float64 buffer plus its address, validated the
    way every other feature matrix crossing this boundary is. The buffer is
    returned so the caller keeps it alive while the address is in flight."""
    buf, _n_rows, n_features = _arrays.column_major([list(row)], name="row")
    return buf, _arrays.addr(buf), n_features


# -- the schema ----------------------------------------------------------


def _resolve_override(feature_names, n_features):
    """A caller's `feature_names`, checked against the model it names.

    The native builder falls back to `Column_i` on a length mismatch rather
    than refusing, so the mismatch is caught here, where the caller's
    argument is still in hand.
    """
    if feature_names is None:
        return None
    names = [str(name) for name in feature_names]
    if len(names) != n_features:
        raise ValueError(
            f"feature_names has {len(names)} entries for a model with "
            f"{n_features} features"
        )
    return names


def _objective_name(code, num_class):
    """LightGBM's spelling of the objective a dump reports."""
    if code is None or num_class > 1:
        return "multiclass"
    return OBJECTIVE_NAMES.get(code, f"objective_{code}")


def _nested_node(nodes, index):
    """One node of a native flat node table, and everything below it, as
    the nested record the schema documents.

    A leaf and an internal node are told apart by which keys they carry, so
    this is where `value` and `count` become `leaf_value` / `leaf_count` or
    `internal_value` / `internal_count`, and where `left` and `right` stop
    being indices and become subtrees.
    """
    node = nodes[index]
    if node["is_leaf"]:
        leaf = {
            "node_index": node["node_index"],
            "leaf_index": node["leaf_index"],
            "leaf_value": node["value"],
            "leaf_count": node["count"],
            "depth": node["depth"],
        }
        if node.get("is_linear"):
            # LightGBM's linear_tree leaf keys: the leaf's output is
            # leaf_const + sum(leaf_coeff[j] * x[leaf_features[j]]).
            leaf["leaf_const"] = node["leaf_const"]
            leaf["leaf_features"] = list(node["leaf_features"])
            leaf["leaf_coeff"] = list(node["leaf_coeff"])
        return leaf
    return {
        "node_index": node["node_index"],
        "split_index": node["split_index"],
        "split_feature": node["split_feature"],
        "split_feature_name": node["split_feature_name"],
        "decision_type": node["decision_type"],
        "threshold": node["threshold"],
        "threshold_bin": node["threshold_bin"],
        "categories": node["categories"],
        "category_bins": node["category_bins"],
        "default_left": node["default_left"],
        "missing_bin": node["missing_bin"],
        "missing_type": node["missing_type"],
        "split_gain": node["split_gain"],
        "internal_value": node["value"],
        "internal_count": node["count"],
        "depth": node["depth"],
        "left_child": _nested_node(nodes, node["left"]),
        "right_child": _nested_node(nodes, node["right"]),
    }


def _schema_from_native(payload):
    """The documented dump, from what the binding handed over.

    Every fact here was computed in Mojo. This nests the flat node tables,
    names the objective, and restates the two keys LightGBM derives from
    others (`max_feature_idx`, `feature_names`).
    """
    infos = payload["feature_infos"]
    return {
        "dump_format_version": payload["dump_format_version"],
        "producer": "mojotrees",
        "model_format_version": payload["model_format_version"],
        "source": "native",
        "objective": _objective_name(
            payload["objective_code"], payload["num_class"]
        ),
        "objective_code": payload["objective_code"],
        "num_class": payload["num_class"],
        "num_tree_per_iteration": payload["num_tree_per_iteration"],
        "num_iteration": payload["num_iteration"],
        "learning_rate": payload["learning_rate"],
        "base_score": list(payload["base_score"]),
        "leaf_value_is_shrunk": False,
        "num_feature": payload["num_feature"],
        "max_feature_idx": payload["num_feature"] - 1,
        "num_bin": payload["num_bin"],
        "feature_names": [info["name"] for info in infos],
        "feature_infos": infos,
        "monotone_constraints": payload["monotone_constraints"],
        "has_split_gain": payload["has_split_gain"],
        "has_node_count": payload["has_node_count"],
        "linear_tree": bool(payload.get("linear_tree", False)),
        "tree_info": [
            {
                "tree_index": tree["tree_index"],
                "iteration": tree["iteration"],
                "class_id": tree["class_id"],
                "num_leaves": tree["num_leaves"],
                "num_nodes": tree["num_nodes"],
                "num_cat": tree["num_cat"],
                "max_depth": tree["max_depth"],
                "shrinkage": tree["shrinkage"],
                "tree_structure": _nested_node(tree["nodes"], 0),
            }
            for tree in payload["trees"]
        ],
    }


def _native_dump(model, feature_names):
    """The dump from the extension module, or None when this build does not
    expose it.

    The names travel with their count, the way every sequence at this
    boundary does. An empty list is what makes the builder fall back to
    `Column_0`, `Column_1`, ..., and `Booster.feature_name()` already
    reports those when the model carries no names, so the override, the
    carried names, and the fallback resolve here in that order.
    """
    booster = _booster(model)
    hook = _hook(booster, "dump_model")
    if hook is None:
        return None
    n_features = int(booster.num_feature())
    names = _resolve_override(feature_names, n_features)
    if names is None:
        names = [str(name) for name in booster.feature_name()]
        if len(names) != n_features:
            names = []
    return _schema_from_native(hook(booster._handle, names, len(names)))


def dump_model(model, feature_names=None):
    """The fitted model as the documented inspection schema.

    `model` is a fitted estimator, a `mojotrees.Booster`, or the text
    `Booster.model_to_string()` produces. `feature_names` overrides the
    names the model carries; it is how a caller names the features of a
    model read back from a file written before format v4, which is the
    version that started carrying them.

    See `docs/MODEL_INSPECTION_SCHEMA.md` for every key. The two a
    consumer should branch on: `has_split_gain` says whether per node
    gains are present, and `has_node_count` whether per node training row
    counts are.
    """
    if not isinstance(model, str):
        native = _native_dump(model, feature_names)
        if native is not None:
            return native
    return _dump_from_text(model, feature_names)


# -- derived shapes ------------------------------------------------------


#: The columns `trees_to_dataframe` produces, in order. LightGBM's names,
#: so a notebook written against LightGBM reads the same frame.
TREE_FRAME_COLUMNS = (
    "tree_index",
    "node_depth",
    "node_index",
    "left_child",
    "right_child",
    "parent_index",
    "split_feature",
    "split_gain",
    "threshold",
    "decision_type",
    "missing_direction",
    "missing_type",
    "value",
    "weight",
    "count",
)


def _node_name(tree_index, node):
    if "leaf_index" in node:
        return f"{tree_index}-L{node['leaf_index']}"
    return f"{tree_index}-S{node['split_index']}"


def trees_to_records(model, dump=None):
    """`trees_to_dataframe` without pandas: one dict per node, in the
    column order `TREE_FRAME_COLUMNS` names.

    Rows come out in depth-first order per tree, root first, which is the
    order LightGBM's frame uses.
    """
    if dump is None:
        dump = dump_model(model)
    rows = []
    for tree in dump["tree_info"]:
        index = tree["tree_index"]
        stack = [(tree["tree_structure"], None)]
        while stack:
            node, parent = stack.pop()
            leaf = "leaf_index" in node
            row = {
                "tree_index": index,
                # LightGBM counts the root as depth 1; the dump counts edges
                # from the root, so the two differ by one.
                "node_depth": node["depth"] + 1,
                "node_index": _node_name(index, node),
                "left_child": None,
                "right_child": None,
                "parent_index": parent,
                "split_feature": None,
                "split_gain": None,
                "threshold": None,
                "decision_type": None,
                "missing_direction": None,
                "missing_type": None,
                "value": node["leaf_value"] if leaf else node[
                    "internal_value"
                ],
                # mojotrees records a node's training row cover, not its
                # hessian sum, so there is no LightGBM `weight` to report.
                "weight": None,
                "count": node["leaf_count"] if leaf else node[
                    "internal_count"
                ],
            }
            if not leaf:
                name = _node_name(index, node)
                row["left_child"] = _node_name(index, node["left_child"])
                row["right_child"] = _node_name(index, node["right_child"])
                row["split_feature"] = node["split_feature_name"]
                row["split_gain"] = node["split_gain"]
                row["threshold"] = (
                    node["categories"]
                    if node["decision_type"] == "=="
                    else node["threshold"]
                )
                row["decision_type"] = node["decision_type"]
                row["missing_direction"] = (
                    "left" if node["default_left"] else "right"
                )
                row["missing_type"] = node["missing_type"]
                stack.append((node["right_child"], name))
                stack.append((node["left_child"], name))
            rows.append(row)
    return rows


def trees_to_dataframe(model, dump=None):
    """The ensemble as a pandas DataFrame, one row per node.

    LightGBM's `Booster.trees_to_dataframe()`, with LightGBM's column
    names. Two columns differ in what they can hold, both because of what
    mojotrees records rather than because of this function:

    - `split_gain` is None on every row unless the dump carries gains
      (`dump["has_split_gain"]`).
    - `weight` is always None: mojotrees records a node's training row
      cover, in `count`, and not the hessian sum LightGBM calls weight.

    pandas is not a dependency of mojotrees. `trees_to_records` returns the
    same rows as plain dicts and needs nothing installed.
    """
    try:
        import pandas
    except ImportError:
        raise ImportError(
            "trees_to_dataframe needs pandas, which mojotrees does not "
            "depend on; trees_to_records() returns the same rows as plain "
            "dicts"
        ) from None
    rows = trees_to_records(model, dump=dump)
    return pandas.DataFrame(rows, columns=list(TREE_FRAME_COLUMNS))


def _feature_index(dump, feature):
    if isinstance(feature, str):
        names = dump["feature_names"]
        if feature not in names:
            raise ValueError(
                f"no feature named {feature!r}; the model has "
                + ", ".join(repr(n) for n in names)
            )
        return names.index(feature)
    index = int(feature)
    if not 0 <= index < dump["num_feature"]:
        raise ValueError(
            f"feature index {index} is outside the model's "
            f"{dump['num_feature']} features"
        )
    return index


def split_values(model, feature, dump=None):
    """Every threshold the ensemble splits `feature` at, in tree order.

    `feature` is an index or a name. The values are the bins' upper edges,
    which is the exact boundary routing uses, so the histogram they feed
    describes where the model actually cuts. A categorical feature is
    refused: its splits are category sets and have no value to bin, which
    is what LightGBM refuses for the same reason.

    Collected in Mojo (`model_dump.dump_split_values`) when the model is a
    live handle, so the values are the ones the trees hold and not a
    re-derivation of them.
    """
    if dump is None:
        dump = dump_model(model)
    index = _feature_index(dump, feature)
    info = dump["feature_infos"][index]
    if info["type"] == "categorical":
        raise ValueError(
            f"feature {info['name']!r} is categorical, so its splits are "
            "category sets and have no value to bin; LightGBM refuses this "
            "for the same reason"
        )
    if not isinstance(model, str):
        booster = _booster(model)
        hook = _hook(booster, "split_values")
        if hook is not None:
            return list(hook(booster._handle, index))
    return _split_values_from_dump(dump, index)


def _histogram(values, bins):
    """An equal-width histogram with `numpy.histogram`'s conventions: bins
    are half open except the last, which is closed, and a single distinct
    value is given a unit-wide bin around itself.

    Written out rather than delegated so the result does not depend on
    whether numpy happens to be installed. Binning collected values into a
    plot's buckets is presentation, which is why it is here and not in
    Mojo; the values themselves come from the model.
    """
    lo = min(values)
    hi = max(values)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins)] + [hi]
    counts = [0] * bins
    for value in values:
        position = int((value - lo) / width)
        if position >= bins:
            position = bins - 1
        elif position < 0:
            position = 0
        counts[position] += 1
    return counts, edges


def get_split_value_histogram(
    model, feature, bins=None, as_frame=False, dump=None
):
    """How the ensemble's splits on one feature are distributed.

    The data behind LightGBM's `plot_split_value_histogram`, and nothing
    else: no plotting dependency is introduced here, and none is needed to
    use this.

    `bins` is the maximum number of equal-width bins. `None`, or a number
    above the count of distinct split values, gives one bin per distinct
    value, as LightGBM does.

    Returns `(counts, bin_edges)`, two lists with `len(bin_edges) ==
    len(counts) + 1`. With `as_frame=True` it returns a pandas DataFrame
    with `Count` and `SplitValue` columns instead, where `SplitValue` is
    each bin's left edge.

    This deviates from LightGBM's signature deliberately. LightGBM switches
    its return type on whether pandas can be imported, and takes `xlabel`
    and `ylabel` arguments that only a plot uses. The shape you get here is
    the one you asked for.
    """
    if dump is None:
        dump = dump_model(model)
    values = split_values(model, feature, dump=dump)
    if not values:
        info = dump["feature_infos"][_feature_index(dump, feature)]
        raise ValueError(
            f"the model never splits on {info['name']!r}, so there is no "
            "split value histogram to build"
        )
    distinct = len(set(values))
    if bins is None:
        n_bins = distinct
    else:
        n_bins = int(bins)
        if n_bins < 1:
            raise ValueError("bins must be a positive integer")
        n_bins = min(n_bins, distinct)
    counts, edges = _histogram(values, n_bins)
    if not as_frame:
        return counts, edges
    try:
        import pandas
    except ImportError:
        raise ImportError(
            "as_frame=True needs pandas, which mojotrees does not depend "
            "on; the default (counts, bin_edges) return needs nothing"
        ) from None
    return pandas.DataFrame(
        {"Count": counts, "SplitValue": edges[:-1]},
        columns=["Count", "SplitValue"],
    )


def feature_importance(model, importance_type="split"):
    """Per-feature importance, from the same place `Booster` gets it.

    `src/mojotrees/importance.mojo` is the one implementation, and this is
    a delegation to it so that the inspection surface answers the question
    without becoming a second answer. The per-feature sums of the dump's
    `split_gain` equal the `"gain"` importance by construction, which is
    the check worth writing against these two together.

    A model whose gains did not survive reports zeros for `"gain"`: it was
    read from a file written before model format v4. `dump_model(model)
    ["has_split_gain"]` is what tells that apart from a measured zero.
    """
    return booster_of(model).feature_importance(importance_type)


def leaf_outputs(model, tree_index=None, dump=None):
    """The leaf values of one tree, or of every tree, by leaf ordinal.

    Returns a list per tree, indexed by the ordinal
    `predict(pred_leaf=True)` reports, so `outputs[t][leaf]` is the value
    that leaf contributes. With `tree_index` it returns that one tree's
    list. The values are unshrunk, as they are stored: what a tree
    contributes to a raw score is `shrinkage * leaf_value` (see
    `raw_scores`).

    Read only. This is the half of LightGBM's leaf-output pair mojotrees
    offers; `model_editing_support()` says why there is no setter.
    """
    if dump is None:
        dump = dump_model(model)
    trees = dump["tree_info"]
    if tree_index is not None:
        index = int(tree_index)
        if not 0 <= index < len(trees):
            raise ValueError(
                f"tree {index} is outside the model's {len(trees)} trees"
            )
        trees = [trees[index]]
    out = []
    for tree in trees:
        values = [None] * tree["num_leaves"]
        stack = [tree["tree_structure"]]
        while stack:
            node = stack.pop()
            if "leaf_index" in node:
                values[node["leaf_index"]] = node["leaf_value"]
                continue
            stack.append(node["left_child"])
            stack.append(node["right_child"])
        out.append(values)
    return out[0] if tree_index is not None else out


def model_editing_support():
    """Whether a fitted model can be edited in place, and which operations
    that covers.

    An explicit status rather than a missing method, so a consumer asking
    "can I set a leaf value?" gets an answer to branch on. The answer is
    the native one (`model_editing.model_editing_status_json`) when the
    extension is present: `supported`, the leaf numbering (`leaf_index`
    is `ordinal`, the numbers `predict(pred_leaf=True)` reports), and one
    record per operation with `supported` and `reason`. Every mutator keeps
    the invariants a fitted tree records: routing and node covers are never
    changed by a leaf write, a monotone claim is re-verified, and every
    result stays loadable by this build.
    """
    import json

    from . import _mojotrees

    status = getattr(_mojotrees, "model_editing_status", None)
    if status is not None:
        out = json.loads(str(status()))
        out["supported"] = bool(out.get("supported", MODEL_EDITING_SUPPORTED))
        return out
    return {
        "supported": MODEL_EDITING_SUPPORTED,
        "leaf_index": "ordinal",
        "leaf_value_is_shrunk": True,
        "operations": [
            {"operation": name, "supported": True, "reason": reason}
            for name, reason in (
                ("rollback_one_iter", "drops whole iterations; DART refused"),
                ("set_leaf_output", "routing and covers unchanged; monotone claim re-verified"),
                ("shuffle_models", "whole iterations move; softmax rounds stay together"),
                ("refit", "leaf values only, tree shapes kept"),
                ("lower_bound", "sound bound from per-tree leaf extremes"),
                ("upper_bound", "sound bound from per-tree leaf extremes"),
            )
        ],
    }


# -- routing the dump ----------------------------------------------------
#
# Both of these take either a live model or a dump. A live model routes in
# Mojo, through `model_dump.dump_raw_scores` and
# `model_dump.dump_leaf_index`, which walk the dump's own bin edges and
# node records rather than the model's prediction path: that independence
# is what makes "the dump describes the model that predicts" a claim that
# can fail, and so one worth checking. A dump alone has no handle to route
# with, so it falls through to the Python walker below.
#
# Both are one-row checks, not a prediction API: `Booster.predict` is what
# scores a matrix. A model handed in on a build with no native hook has its
# dump rebuilt per call, so a caller checking many rows should build the
# dump once and pass that instead.


def leaf_index_of(source, tree_index, row):
    """The leaf ordinal one raw example reaches in a tree.

    `source` is a fitted model, a `Booster`, or a dump. The same numbering
    `predict(pred_leaf=True)` reports.
    """
    if not isinstance(source, dict):
        booster = _booster(source)
        hook = _hook(booster, "dump_leaf_index")
        if hook is not None:
            buf, address, n_features = _row_buffer(row)
            index = hook(
                booster._handle, int(tree_index), address, n_features
            )
            # The buffer stays referenced until the call has returned.
            del buf
            return int(index)
        source = dump_model(source)
    return _leaf_index_from_dump(source, tree_index, row)


def raw_scores(source, row):
    """Raw scores for one raw example: one entry for a single-output model,
    one per class for a softmax one.

    `source` is a fitted model, a `Booster`, or a dump. mojotrees stores
    unshrunk leaf values and multiplies by the shrinkage when it predicts,
    so the sum is
    `base_score[k] + sum_over_trees(shrinkage * leaf_value)`.
    """
    if not isinstance(source, dict):
        booster = _booster(source)
        hook = _hook(booster, "dump_raw_scores")
        if hook is not None:
            buf, address, n_features = _row_buffer(row)
            scores = list(hook(booster._handle, address, n_features))
            del buf
            return scores
        source = dump_model(source)
    return _raw_scores_from_dump(source, row)


# -- estimator and booster attributes ------------------------------------
#
# The six LightGBM attributes task 14 owns. Each is a one line read here so
# that wiring it onto `_Base` is a property that delegates, and so that a
# `Booster` answers the same question the same way. See
# `handoffs/migration_19_model_inspection.md (deleted, recover with git log --all --diff-filter=D -- handoffs/migration_19_model_inspection.md)` for the exact patch each one
# needs.


def booster_of(model):
    """The `mojotrees.Booster` behind whatever was handed in.

    `Booster` is already what LightGBM's `booster_` returns and what
    `_Base.booster_` gives, so this only normalizes the two entry points.
    """
    return _booster(model)


def feature_name_of(model):
    """LightGBM's `feature_name_`: the training feature names, or
    `Column_0`, `Column_1`, ... when the model carries none.

    `Booster.feature_name()` already answers this, so the estimator
    attribute is that call and not a second source of truth.
    """
    return list(booster_of(model).feature_name())


def n_features_of(model):
    """LightGBM's `n_features_`: the feature count the model was fitted on,
    read from the model rather than from a fit-time attribute."""
    return int(booster_of(model).num_feature())


def n_iter_of(model):
    """LightGBM's `n_iter_`: boosting iterations trained.

    An estimator records this at fit time, because early stopping can train
    more iterations than the model ends up holding; without that record the
    model's own iteration count is the answer.
    """
    recorded = getattr(model, "n_iter_", None)
    if recorded is not None:
        return int(recorded)
    return int(booster_of(model).current_iteration())


def objective_of(model):
    """LightGBM's `objective_`: the resolved objective name.

    Resolved, not echoed: it comes from the objective code the model
    carries, so a model read back from a file answers it too, and an
    estimator constructed with an alias (`mae`) reports the canonical name
    (`regression_l1`).
    """
    booster = booster_of(model)
    if getattr(booster, "_n_classes", 0):
        return "multiclass"
    from . import _mojotrees

    hook = getattr(_mojotrees, "objective_code", None)
    if hook is not None:
        return _objective_name(int(hook(booster._handle)), 1)
    raw = parse_model_string(booster.model_to_string())
    return _objective_name(raw["objective_code"], raw["n_classes"])


def best_score_of(model):
    """LightGBM's `best_score_`: the primary validation metric's best value.

    Purely a fit-time record. A model has no validation history, so this
    reports what `fit` recorded and raises when there was no validation
    set, which is what `hasattr(model, "best_score_")` already means today.
    """
    score = getattr(model, "best_score_", None)
    if score is None:
        raise AttributeError(
            "best_score_ is set by fit(eval_set=...); this model was fitted "
            "without a validation set, so there is no metric to report"
        )
    return float(score)


# ========================================================================
# COMPATIBILITY: the schema, rebuilt in Python from the model text
# ========================================================================
#
# Everything below this banner exists because the extension module does not
# expose the native dump yet. It parses `Booster.model_to_string()` -- the
# versioned save format in src/mojotrees/serialize.mojo -- and rebuilds the
# same schema here, which is why inspection works against the bindings as
# they are built today.
#
# It is a second implementation of facts Mojo already knows: how a tree is
# laid out, what a split bin's upper edge means, how a category set maps
# back to codes, and how a row routes. Every one of them is stated in
# src/mojotrees/model_dump.mojo, and keeping two copies is the cost of
# working today rather than after the bindings land.
#
# DELETION POINT. When `_mojotrees.dump_model` / `dump_model_multiclass`,
# `split_values*`, `dump_raw_scores*`, and `dump_leaf_index*` exist, delete
# this entire section and, with it: `parse_model_string` from `__all__`,
# the `import struct as _struct` at the top, the `_dump_from_text` call in
# `dump_model`, the `_split_values_from_dump` call in `split_values`, and
# the two `source = dump_model(source)` fallbacks in `leaf_index_of` and
# `raw_scores`. `handoffs/migration_19_model_inspection.md (deleted, recover with git log --all --diff-filter=D -- handoffs/migration_19_model_inspection.md)` lists the
# tests that change in the same commit; nothing above this banner does.


#: Bin 0 of a categorical feature: missing, unseen, or dropped. Never a
#: member of a split's category set, so those rows always go right (see
#: src/mojotrees/categorical.mojo).
_UNKNOWN_BIN = 0

#: Category codes are the ones LightGBM's `static_cast<int>` can represent.
_MAX_CATEGORY = 1 << 31

_CAT_BITSET_WORDS = 4
_CAT_MAX_BINS = 64 * _CAT_BITSET_WORDS


def _f64_from_bits(token):
    bits = int(token)
    if bits < 0 or bits > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"float bit pattern out of range: {token!r}")
    return _struct.unpack("<d", _struct.pack("<Q", bits))[0]


class _Reader:
    """A mirror of `_TokenReader` in src/mojotrees/serialize.mojo. Floats
    are stored as decimal IEEE-754 bit patterns, so reading one back is
    exact: a dump reports the same bits the trained model holds, not a
    rounded decimal of them."""

    def __init__(self, text):
        self._tokens = text.split()
        self._pos = 0

    def next(self):
        if self._pos >= len(self._tokens):
            raise ValueError("unexpected end of model text")
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def peek(self):
        if self._pos >= len(self._tokens):
            return ""
        return self._tokens[self._pos]

    def expect(self, word):
        token = self.next()
        if token != word:
            raise ValueError(f"expected {word!r} in model text, got {token!r}")

    def next_int(self):
        return int(self.next())

    def next_f64(self):
        return _f64_from_bits(self.next())

    def ints(self, n):
        return [self.next_int() for _ in range(n)]

    def floats(self, n):
        return [self.next_f64() for _ in range(n)]

    def flags(self, n):
        return [self.next_int() != 0 for _ in range(n)]


def parse_model_string(text):
    """The raw contents of mojotrees's model text, as plain Python.

    This is the parse step alone: arrays exactly as
    src/mojotrees/serialize.mojo writes them, with no interpretation
    layered on. Part of the compatibility path: it goes when the native
    dump binding lands.
    """
    reader = _Reader(text)
    magic = reader.next()
    if magic != "mojotrees":
        raise ValueError(
            f"not a mojotrees model: file starts with {magic!r}"
        )
    tag = reader.next()
    if not tag.startswith("v") or not tag[1:].isdigit():
        raise ValueError(f"unreadable model format version {tag!r}")
    version = int(tag[1:])
    if version not in SUPPORTED_MODEL_FORMAT_VERSIONS:
        raise ValueError(
            f"model format v{version} is newer than this build reads; "
            "supported versions are "
            + ", ".join(f"v{v}" for v in SUPPORTED_MODEL_FORMAT_VERSIONS)
        )

    out = {"model_format_version": version}
    out["feature_names"] = _parse_feature_names(reader)
    head = reader.next()
    if head == "multiclass":
        out["kind"] = "multiclass"
        n_classes = reader.next_int()
        out["n_classes"] = n_classes
        out["objective_code"] = None
        reader.expect("learning_rate")
        out["learning_rate"] = reader.next_f64()
        reader.expect("base_scores")
        out["base_scores"] = reader.floats(n_classes)
    elif head == "objective":
        out["kind"] = "single"
        out["n_classes"] = 1
        out["objective_code"] = reader.next_int()
        reader.expect("learning_rate")
        out["learning_rate"] = reader.next_f64()
        reader.expect("base_score")
        out["base_scores"] = [reader.next_f64()]
    else:
        raise ValueError(
            f"expected 'objective' or 'multiclass' in model text, got "
            f"{head!r}"
        )

    out["mapper"] = _parse_mapper(reader, version)
    n_features = out["mapper"]["n_features"]
    out["categorical"] = _parse_categorical(reader, n_features)
    # The two v5 sections this parser reads, in the order
    # `save_model` writes them: `ctr` and then `usable`, both between the
    # categorical tables and the monotone vector.
    out["ctr"] = _parse_ctr(reader, version)
    out["usable"] = _parse_usable(reader, version, n_features)
    out["monotone"] = _parse_monotone(reader, n_features)
    out["trees"] = _parse_trees(reader, version)
    # The one v5 section this parser does not read, refused by name. It is
    # last in the file, so reaching here with it unread means it is next.
    if reader.peek() == _LINEAR_SECTION_TAG:
        raise ValueError(
            "this model carries the v5 'linear' section (linear leaves) and "
            "the pure-Python fallback parser does not read it: rebuilding a "
            "linear leaf's features, coefficients and centering here would "
            "be a second implementation of "
            "src/mojotrees/linear_tree.mojo, and skipping the section would "
            "describe a linear model as a constant-leaf one. Use the native "
            "dump, which reads it (src/mojotrees/model_dump.mojo, "
            "_attach_linear)"
        )
    return out


#: The escapes `_escape_name` in src/mojotrees/serialize.mojo writes. A
#: feature name is one whitespace-free token in the model file, so the four
#: whitespace characters and the backslash travel escaped.
_NAME_ESCAPES = {
    "s": " ",
    "t": "\t",
    "n": "\n",
    "r": "\r",
    "\\": "\\",
}

#: The token an empty feature name travels as: an empty string is not a
#: token at all.
_EMPTY_NAME_TOKEN = "\\e"


def _unescape_name(token):
    """One feature name from its token, the inverse of `_escape_name` in
    src/mojotrees/serialize.mojo."""
    if token == _EMPTY_NAME_TOKEN:
        return ""
    out = []
    i = 0
    while i < len(token):
        char = token[i]
        if char != "\\":
            out.append(char)
            i += 1
            continue
        if i + 1 >= len(token):
            raise ValueError("corrupt feature name: token ends in an escape")
        escape = token[i + 1]
        if escape not in _NAME_ESCAPES:
            raise ValueError(
                f"corrupt feature name: unknown escape {escape!r}"
            )
        out.append(_NAME_ESCAPES[escape])
        i += 2
    return "".join(out)


def _parse_feature_names(reader):
    """The optional v4 `feature_names` section, or None.

    None is not the same as `Column_0, Column_1, ...`: it says the file
    names nothing, which is every file written before v4 and every model
    saved without names.
    """
    if reader.peek() != "feature_names":
        return None
    reader.expect("feature_names")
    n = reader.next_int()
    if n < 1:
        raise ValueError("corrupt feature_names section: nonpositive count")
    return [_unescape_name(reader.next()) for _ in range(n)]


def _parse_mapper(reader, version):
    reader.expect("mapper")
    n_features = reader.next_int()
    n_bins = reader.next_int()
    n_edges = reader.next_int()
    if n_features < 1 or n_edges < 0:
        raise ValueError("corrupt mapper header")
    edges = reader.floats(n_edges)
    offsets = reader.ints(n_features + 1)
    if offsets[n_features] != n_edges:
        raise ValueError("corrupt mapper offsets")
    if version >= 2:
        missing_bin = reader.ints(n_features)
    else:
        missing_bin = [-1] * n_features
    return {
        "n_features": n_features,
        "n_bins": n_bins,
        "edges": edges,
        "edge_offsets": offsets,
        "missing_bin": missing_bin,
    }


def _parse_categorical(reader, n_features):
    """The optional `categorical` section. Absent means every feature is
    numerical, which is what a model with no categorical feature writes."""
    if reader.peek() != "categorical":
        return {
            "is_categorical": [False] * n_features,
            "codes": [],
            "offsets": [0] * (n_features + 1),
        }
    reader.expect("categorical")
    n_flags = reader.next_int()
    n_codes = reader.next_int()
    if n_flags != n_features or n_codes < 0:
        raise ValueError("corrupt categorical header")
    flags = reader.flags(n_features)
    codes = reader.ints(n_codes)
    offsets = reader.ints(n_features + 1)
    if offsets[0] != 0 or offsets[n_features] != n_codes:
        raise ValueError("corrupt categorical offsets")
    return {"is_categorical": flags, "codes": codes, "offsets": offsets}


#: The `ctr` section's own revision, carried in the section rather than in
#: the file version. `CTR_SECTION_REVISION` in src/mojotrees/serialize.mojo,
#: and the two must move together: the section is unreadable at a revision
#: this parser does not know, so it refuses one rather than guessing.
_CTR_SECTION_REVISION = 2

#: The three v5 section tags, spelled here so the refusals below can name
#: them. `_LINEAR_SECTION_TAG` is `LINEAR_SECTION_TAG` in
#: src/mojotrees/linear_tree.mojo.
_CTR_SECTION_TAG = "ctr"
_USABLE_SECTION_TAG = "usable"
_LINEAR_SECTION_TAG = "linear"

#: A CTR bucket is a bin index stored in a byte, the same ceiling
#: `MAX_BINS` puts on the mapper. `_read_ctr` in
#: src/mojotrees/serialize.mojo checks it there for the same reason.
_MAX_BINS = 256


def _parse_ctr(reader, version):
    """The optional v5 `ctr` section: the fitted ordered target statistics.

    Absent -- every file before v5, and every v5 file whose mapper carries no
    fitted tables -- means `None`, which is what `CtrTables.none()` is on the
    Mojo side.

    A mirror of `_read_ctr` in src/mojotrees/serialize.mojo, field for field
    and check for check, including the ones that are not about reading: the
    slot tables and the per-slot bucket counts state the same thing twice,
    and a file where they disagree would map a raw value through one and size
    its statistics with the other, which scores wrong instead of failing.

    Three of `CtrColumn`'s fields are not in the file (`shift`, `norm`,
    `scale`), and neither is `predict_lut`; the Mojo reader recomputes them.
    This parser does not, because it does not score -- it reports what the
    file holds. What it must not do is silently skip the section, which would
    leave the reader positioned on the middle of the CTR arrays and read them
    as the monotone vector.
    """
    if version < 5 or reader.peek() != _CTR_SECTION_TAG:
        return None
    reader.expect(_CTR_SECTION_TAG)
    revision = reader.next_int()
    if revision != _CTR_SECTION_REVISION:
        raise ValueError(
            f"unsupported ctr section revision {revision}; this parser "
            f"reads revision {_CTR_SECTION_REVISION}"
        )
    n_base_features = reader.next_int()
    n_classes = reader.next_int()
    prior_denom = reader.next_f64()
    n_slots = reader.next_int()
    n_columns = reader.next_int()
    n_class = reader.next_int()
    n_mean = reader.next_int()
    n_counter = reader.next_int()
    if n_base_features < 1 or n_classes < 1:
        raise ValueError("corrupt ctr section: nonpositive header count")
    if n_slots < 1 or n_columns < 1:
        raise ValueError(
            "corrupt ctr section: an active section carries at least one "
            "slot and one column"
        )
    if n_class < 0 or n_mean < 0 or n_counter < 0:
        raise ValueError("corrupt ctr section: negative table length")

    source_features = reader.ints(n_slots)
    for s, feature in enumerate(source_features):
        if feature < 0 or feature >= n_base_features:
            raise ValueError("corrupt ctr section: source feature out of range")
        if s > 0 and feature <= source_features[s - 1]:
            raise ValueError(
                "corrupt ctr section: source features are not ascending"
            )
    slot_buckets = reader.ints(n_slots)
    if any(b < 1 for b in slot_buckets):
        raise ValueError("corrupt ctr section: nonpositive bucket count")

    slot_code_offsets = reader.ints(n_slots + 1)
    if slot_code_offsets[0] != 0:
        raise ValueError("corrupt ctr section: code offsets must start at 0")
    for s in range(n_slots):
        lo = slot_code_offsets[s]
        hi = slot_code_offsets[s + 1]
        if hi < lo:
            raise ValueError(
                "corrupt ctr section: code offsets are not ascending"
            )
        if hi - lo + 1 != slot_buckets[s]:
            raise ValueError(
                f"corrupt ctr section: slot {s} carries {hi - lo} category "
                f"codes but claims {slot_buckets[s]} buckets; a bucket is "
                "one code plus the reserved bucket 0"
            )
    slot_codes = reader.ints(slot_code_offsets[n_slots])
    for s in range(n_slots):
        lo = slot_code_offsets[s]
        hi = slot_code_offsets[s + 1]
        for i in range(lo + 1, hi):
            if slot_codes[i] <= slot_codes[i - 1]:
                raise ValueError(
                    "corrupt ctr section: category codes are not strictly "
                    "ascending"
                )
        if lo < hi and slot_codes[lo] < 0:
            raise ValueError("corrupt ctr section: negative category code")

    counter_denominator = reader.ints(n_slots)

    columns = []
    for _ in range(n_columns):
        slot = reader.next_int()
        if slot < 0 or slot >= n_slots:
            raise ValueError("corrupt ctr section: column slot out of range")
        source = reader.next_int()
        if source != source_features[slot]:
            raise ValueError(
                "corrupt ctr section: a column's source feature disagrees "
                "with its slot"
            )
        ctr_type = reader.next_int()
        target_border_idx = reader.next_int()
        prior_index = reader.next_int()
        prior = reader.next_f64()
        border_count = reader.next_int()
        if border_count < 1 or border_count >= _MAX_BINS:
            raise ValueError(
                f"corrupt ctr section: ctr_border_count must be in "
                f"[1, {_MAX_BINS - 1}]; a ctr bucket is stored in a byte"
            )
        columns.append(
            {
                "slot": slot,
                "source_feature": source,
                "ctr_type": ctr_type,
                "target_border_idx": target_border_idx,
                "prior_index": prior_index,
                "prior": prior,
                "ctr_border_count": border_count,
            }
        )

    class_offsets = reader.ints(n_slots + 1)
    _check_ctr_offsets(class_offsets, n_class, "class")
    class_table = reader.ints(n_class)
    mean_offsets = reader.ints(n_slots + 1)
    _check_ctr_offsets(mean_offsets, n_mean, "mean")
    mean_sums = reader.floats(n_mean)
    mean_counts = reader.ints(n_mean)
    counter_offsets = reader.ints(n_slots + 1)
    _check_ctr_offsets(counter_offsets, n_counter, "counter")
    counter_counts = reader.ints(n_counter)

    return {
        "revision": revision,
        "n_base_features": n_base_features,
        "n_classes": n_classes,
        "prior_denom": prior_denom,
        "source_features": source_features,
        "slot_buckets": slot_buckets,
        "slot_code_offsets": slot_code_offsets,
        "slot_codes": slot_codes,
        "counter_denominator": counter_denominator,
        "columns": columns,
        "class_offsets": class_offsets,
        "class_table": class_table,
        "mean_offsets": mean_offsets,
        "mean_sums": mean_sums,
        "mean_counts": mean_counts,
        "counter_offsets": counter_offsets,
        "counter_counts": counter_counts,
    }


def _check_ctr_offsets(offsets, total, name):
    """One slot-offset array as `_check_ctr_offsets` in
    src/mojotrees/serialize.mojo checks it: starts at zero, never goes
    backwards, and ends exactly at the array it indexes."""
    if not offsets or offsets[0] != 0:
        raise ValueError(f"corrupt ctr section: {name} offsets do not start at 0")
    for i in range(1, len(offsets)):
        if offsets[i] < offsets[i - 1]:
            raise ValueError(
                f"corrupt ctr section: {name} offsets are not ascending"
            )
    if offsets[-1] != total:
        raise ValueError(
            f"corrupt ctr section: {name} offsets end at {offsets[-1]}, not "
            f"at the {total} entries they index"
        )


def _parse_usable(reader, version, n_features):
    """The optional v5 `usable` section: the ascending pool a tree's split
    search may draw from, LightGBM's `used_features`.

    Absent -- every file before v5, and every v5 file whose pool was never
    narrowed -- means every feature, which is what
    `binning.all_features(n_features)` is and what the Mojo reader
    substitutes.

    Two things narrow it and both are training-time:
    `fit_bins(feature_pre_filter=True)` and CatBoost-mode CTR replacement
    through `BinMapper.drop_usable`. Nothing on the inference path reads it,
    so it does not enter the dump schema; it is parsed because the section is
    in the byte stream between `ctr` and `monotone` and a parser that skipped
    it would read its feature ids as the monotone vector.
    """
    if version < 5 or reader.peek() != _USABLE_SECTION_TAG:
        return list(range(n_features))
    reader.expect(_USABLE_SECTION_TAG)
    n = reader.next_int()
    if n < 0 or n > n_features:
        raise ValueError(
            f"corrupt usable section: {n} entries for a mapper with "
            f"{n_features} features"
        )
    pool = reader.ints(n)
    previous = -1
    for feature in pool:
        if feature < 0 or feature >= n_features:
            raise ValueError(
                "corrupt usable section: feature id out of range"
            )
        if feature <= previous:
            raise ValueError(
                "corrupt usable section: feature ids are not strictly "
                "ascending"
            )
        previous = feature
    return pool


def _parse_monotone(reader, n_features):
    """The optional `monotone` section. Absent means unconstrained."""
    if reader.peek() != "monotone":
        return None
    reader.expect("monotone")
    n = reader.next_int()
    if n != n_features:
        raise ValueError("corrupt monotone section")
    return reader.ints(n)


def _parse_trees(reader, version):
    reader.expect("trees")
    n_trees = reader.next_int()
    if n_trees < 0:
        raise ValueError("corrupt tree count")
    trees = []
    for _ in range(n_trees):
        reader.expect("tree")
        n_nodes = reader.next_int()
        n_leaves = reader.next_int()
        if n_nodes < 1 or n_leaves < 1:
            raise ValueError("corrupt tree header")
        tree = {
            "n_leaves": n_leaves,
            "feature": reader.ints(n_nodes),
            "threshold_bin": reader.ints(n_nodes),
            "left": reader.ints(n_nodes),
            "right": reader.ints(n_nodes),
            "value": reader.floats(n_nodes),
        }
        if version >= 2:
            tree["default_left"] = reader.flags(n_nodes)
            tree["missing_bin"] = reader.ints(n_nodes)
        else:
            tree["default_left"] = [False] * n_nodes
            tree["missing_bin"] = [-1] * n_nodes
        # v4 puts a presence flag in front of the covers, so a tree that
        # never had any (one loaded from a v1 or v2 file and re-saved) says
        # so instead of writing zeros that read as corrupt covers.
        if version >= 4:
            reader.expect("counts")
            has_counts = reader.next_int() != 0
        else:
            has_counts = version >= 3
        if has_counts:
            tree["count"] = reader.floats(n_nodes)
        else:
            tree["count"] = [0.0] * n_nodes
        # v4: per-node split gains, behind the same kind of flag. None, and
        # not a list of zeros, is what "this tree carries no gains" means:
        # every earlier format dropped them.
        tree["split_gain"] = None
        if version >= 4:
            reader.expect("gains")
            if reader.next_int() != 0:
                tree["split_gain"] = reader.floats(n_nodes)
        # The per-tree category sets, written only by a tree that has a
        # categorical node.
        if version >= 2 and reader.peek() == "cat":
            reader.expect("cat")
            n_words = reader.next_int()
            if n_words < 0 or n_words % _CAT_BITSET_WORDS != 0:
                raise ValueError("corrupt tree: category bitset size")
            tree["cat_offset"] = reader.ints(n_nodes)
            tree["cat_bitset"] = reader.ints(n_words)
        else:
            tree["cat_offset"] = [-1] * n_nodes
            tree["cat_bitset"] = []
        trees.append(tree)
    return trees


def _model_text(model):
    """The model text and the feature names for whatever was handed in: an
    estimator, a `Booster`, or the text itself.

    The names are the ones the `Booster` actually carries and not what
    `feature_name()` reports, which invents `Column_0, Column_1, ...` when
    there are none. The difference matters here: a file can carry real
    names (v4), and invented ones must not outrank them.
    """
    if isinstance(model, str):
        return model, None
    booster = _booster(model)
    carried = getattr(booster, "_names", None)
    return (
        booster.model_to_string(),
        None if carried is None else [str(name) for name in carried],
    )


def _native_split_gains(model):
    """Per node split gains, one list per tree, when the extension module
    exposes the small hook and not the whole dump.

    Model format v4 serializes gains, so this is only asked when the text
    is older than that: the handle still holds what the file dropped.
    """
    booster = _booster(model)
    hook = _hook(booster, "split_gains")
    if hook is None:
        return None
    return hook(booster._handle)


def _text_gains(raw):
    """Per node gains for every tree, or None when the text carries none.

    A v4 file records gains per tree, and a tree that recorded none (every
    leaf, and every tree from a model loaded out of an older file) writes
    the flag rather than an array. Zeros stand in for those, so the shape
    is one list per tree either way, exactly as the hook's is.
    """
    trees = raw["trees"]
    if not any(tree["split_gain"] is not None for tree in trees):
        return None
    return [
        tree["split_gain"]
        if tree["split_gain"] is not None
        else [0.0] * len(tree["feature"])
        for tree in trees
    ]


def _dump_from_text(model, feature_names=None):
    """The documented schema, rebuilt from the model text."""
    text, carried_names = _model_text(model)
    raw = parse_model_string(text)
    # Parsing the `ctr` section is not describing it. Every field of this
    # schema is sized by `mapper.n_features`, which counts base features
    # only, while a CTR tree splits on ids up to `n_total_features() - 1`:
    # `_feature_infos` would describe none of those columns and `node_at`
    # would index `infos` past its end. The Mojo dump refuses the same model
    # for the same reason (`ctr.check_ctr_model_support`, catalog A19), and
    # the two sides of `dump_model` have to agree about what a v5 file can be
    # turned into.
    if raw["ctr"] is not None:
        raise ValueError(
            "a model carrying ctr columns cannot be described by this "
            "schema: it is sized by mapper.n_features, which counts base "
            "features only, while a ctr tree splits on ids up to "
            "n_total_features() - 1 (catalog A19). The model itself saves, "
            "loads and predicts, and parse_model_string returns its fitted "
            "tables; this is the dump schema's gap"
        )
    source = "model_to_string"
    # The text is the first source asked, because from v4 on it carries the
    # gains itself. The hook is what gives a model written in an older
    # format its gains back, and only a live handle still has them.
    gains = _text_gains(raw)
    if gains is None and not isinstance(model, str):
        native = _native_split_gains(model)
        if native is not None:
            source = "model_to_string+split_gains"
            gains = [list(tree) for tree in native]
            if len(gains) != len(raw["trees"]):
                raise ValueError(
                    "the split_gains hook returned "
                    f"{len(gains)} trees for a model with "
                    f"{len(raw['trees'])}"
                )

    n_features = raw["mapper"]["n_features"]
    names = _resolve_names(
        feature_names, carried_names, raw["feature_names"], n_features
    )
    infos = _feature_infos(raw, names)
    n_classes = raw["n_classes"]
    n_trees = len(raw["trees"])
    per_iteration = n_classes if raw["kind"] == "multiclass" else 1
    has_count = all(
        all(c > 0.0 for c in tree["count"]) for tree in raw["trees"]
    ) and bool(raw["trees"])

    trees = []
    for index, tree in enumerate(raw["trees"]):
        tree_gains = None if gains is None else gains[index]
        trees.append(
            _tree_info(
                tree,
                index,
                per_iteration,
                raw["learning_rate"],
                infos,
                tree_gains,
            )
        )

    return {
        "dump_format_version": DUMP_FORMAT_VERSION,
        "producer": "mojotrees",
        "model_format_version": raw["model_format_version"],
        "source": source,
        "objective": _objective_name(raw["objective_code"], n_classes),
        "objective_code": raw["objective_code"],
        "num_class": n_classes,
        "num_tree_per_iteration": per_iteration,
        "num_iteration": n_trees // per_iteration if per_iteration else 0,
        "learning_rate": raw["learning_rate"],
        "base_score": list(raw["base_scores"]),
        "leaf_value_is_shrunk": False,
        "num_feature": n_features,
        "max_feature_idx": n_features - 1,
        "num_bin": raw["mapper"]["n_bins"],
        "feature_names": list(names),
        "feature_infos": infos,
        "monotone_constraints": raw["monotone"],
        "has_split_gain": gains is not None,
        "has_node_count": has_count,
        # False is a fact here rather than an assumption. The text fallback
        # reads v1 through v5, and `parse_model_string` refuses the `linear`
        # section by name, so any model that reaches this line has constant
        # leaves. A linear model is dumped natively.
        "linear_tree": False,
        "tree_info": trees,
    }


def _resolve_names(override, carried, from_file, n_features):
    """The names to report, most authoritative first.

    A caller's override wins; then the names the live `Booster` carries,
    which are the training set's; then the names the file carries, which is
    all a model read back from disk has (v4 and later). `Column_i` is the
    last resort and is not a name the model claims.
    """
    if override is not None:
        return _resolve_override(override, n_features)
    for names in (carried, from_file):
        if names is not None and len(names) == n_features:
            return [str(name) for name in names]
    return [f"Column_{i}" for i in range(n_features)]


def _feature_infos(raw, names):
    """One record per feature: how it is binned, and how missing values and
    unseen categories are routed through it."""
    mapper = raw["mapper"]
    cats = raw["categorical"]
    monotone = raw["monotone"]
    infos = []
    for f in range(mapper["n_features"]):
        missing_bin = mapper["missing_bin"][f]
        info = {
            "index": f,
            "name": names[f],
            "missing_bin": missing_bin,
            "missing_type": "NaN" if missing_bin >= 0 else "None",
            "monotone": 0 if monotone is None else monotone[f],
        }
        if cats["is_categorical"][f]:
            begin = cats["offsets"][f]
            end = cats["offsets"][f + 1]
            codes = cats["codes"][begin:end]
            info["type"] = "categorical"
            info["categories"] = codes
            info["bin_upper_bounds"] = None
            # Bin 0 collects missing, unseen, and dropped codes; category
            # `codes[i]` is bin `i + 1`.
            info["num_bin"] = len(codes) + 1
        else:
            begin = mapper["edge_offsets"][f]
            end = mapper["edge_offsets"][f + 1]
            edges = mapper["edges"][begin:end]
            info["type"] = "numerical"
            info["categories"] = None
            info["bin_upper_bounds"] = edges
            info["num_bin"] = len(edges) + 1 + (1 if missing_bin >= 0 else 0)
        infos.append(info)
    return infos


def _node_depths(tree):
    """Depth in edges from the root, one entry per node. The root is 0, so
    this is the quantity `max_depth` bounds."""
    n_nodes = len(tree["feature"])
    depths = [0] * n_nodes
    parents = [-1] * n_nodes
    stack = [0]
    while stack:
        node = stack.pop()
        if tree["feature"][node] < 0:
            continue
        for child in (tree["left"][node], tree["right"][node]):
            depths[child] = depths[node] + 1
            parents[child] = node
            stack.append(child)
    return depths, parents


def _split_ordinals(tree):
    """Per node: its rank among this tree's internal nodes, and its rank
    among the leaves, in node-array order. The leaf rank is mojotrees's own
    leaf ordinal, the one `predict(pred_leaf=True)` reports (see
    `Tree.leaf_ordinals` in src/mojotrees/tree.mojo)."""
    splits = []
    leaves = []
    n_split = 0
    n_leaf = 0
    for feature in tree["feature"]:
        if feature < 0:
            splits.append(-1)
            leaves.append(n_leaf)
            n_leaf += 1
        else:
            splits.append(n_split)
            leaves.append(-1)
            n_split += 1
    return splits, leaves


def _category_bins(tree, node):
    """The bin ids node `node`'s category set holds, ascending."""
    offset = tree["cat_offset"][node]
    if offset < 0:
        return None
    words = tree["cat_bitset"][offset : offset + _CAT_BITSET_WORDS]
    return [
        b
        for b in range(_CAT_MAX_BINS)
        if (words[b >> 6] >> (b & 63)) & 1
    ]


def _threshold_value(info, threshold_bin):
    """The largest raw value node routing sends left, or None when the split
    bin has no upper edge."""
    edges = info["bin_upper_bounds"]
    if edges is None or threshold_bin < 0 or threshold_bin >= len(edges):
        return None
    return edges[threshold_bin]


def _tree_info(tree, index, per_iteration, learning_rate, infos, gains):
    depths, _parents = _node_depths(tree)
    splits, leaves = _split_ordinals(tree)
    n_cat = sum(1 for offset in tree["cat_offset"] if offset >= 0)

    def node_at(node):
        feature = tree["feature"][node]
        if feature < 0:
            return {
                "node_index": node,
                "leaf_index": leaves[node],
                "leaf_value": tree["value"][node],
                "leaf_count": tree["count"][node],
                "depth": depths[node],
            }
        info = infos[feature]
        bins = _category_bins(tree, node)
        categorical = bins is not None
        codes = None
        if categorical and info["categories"] is not None:
            codes = [
                info["categories"][b - 1]
                for b in bins
                if 1 <= b <= len(info["categories"])
            ]
        return {
            "node_index": node,
            "split_index": splits[node],
            "split_feature": feature,
            "split_feature_name": info["name"],
            "decision_type": "==" if categorical else "<=",
            "threshold": (
                None if categorical
                else _threshold_value(info, tree["threshold_bin"][node])
            ),
            "threshold_bin": tree["threshold_bin"][node],
            "categories": codes,
            "category_bins": bins,
            "default_left": tree["default_left"][node],
            "missing_bin": tree["missing_bin"][node],
            "missing_type": (
                "NaN" if tree["missing_bin"][node] >= 0 else "None"
            ),
            "split_gain": None if gains is None else gains[node],
            "internal_value": tree["value"][node],
            "internal_count": tree["count"][node],
            "depth": depths[node],
            "left_child": node_at(tree["left"][node]),
            "right_child": node_at(tree["right"][node]),
        }

    return {
        "tree_index": index,
        "iteration": index // per_iteration,
        "class_id": index % per_iteration,
        "num_leaves": tree["n_leaves"],
        "num_nodes": len(tree["feature"]),
        "num_cat": n_cat,
        "max_depth": max(depths) if depths else 0,
        "shrinkage": learning_rate,
        "tree_structure": node_at(0),
    }


def _bin_value(info, value):
    """The bin a raw feature value lands in. A mirror of
    `BinMapper.bin_value` in src/mojotrees/binning.mojo, and of
    `model_dump.dump_bin_value`, from the dump's feature record."""
    if info["type"] == "categorical":
        codes = info["categories"]
        if value != value or value < 0.0 or value >= float(_MAX_CATEGORY):
            return _UNKNOWN_BIN
        code = int(value)
        lo, hi = 0, len(codes)
        while lo < hi:
            mid = (lo + hi) // 2
            if codes[mid] < code:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(codes) and codes[lo] == code:
            return lo + 1
        return _UNKNOWN_BIN
    if value != value:
        if info["missing_bin"] >= 0:
            return info["missing_bin"]
        # No reserved bin: NaN bins as 0.0, as LightGBM does for a feature
        # whose missing_type is None.
        value = 0.0
    edges = info["bin_upper_bounds"]
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if value <= edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _goes_left(node, bin_id):
    """A mirror of `Tree.goes_left`, and of `model_dump.dump_goes_left`:
    category membership first, then the node's missing direction, then the
    threshold."""
    if node["category_bins"] is not None:
        return bin_id in node["category_bins"]
    if bin_id == node["missing_bin"]:
        return node["default_left"]
    return bin_id <= node["threshold_bin"]


def _walk(dump, tree_index, row):
    node = dump["tree_info"][tree_index]["tree_structure"]
    while "leaf_index" not in node:
        info = dump["feature_infos"][node["split_feature"]]
        bin_id = _bin_value(info, float(row[node["split_feature"]]))
        node = node["left_child"] if _goes_left(node, bin_id) else (
            node["right_child"]
        )
    return node


def _leaf_index_from_dump(dump, tree_index, row):
    return _walk(dump, tree_index, row)["leaf_index"]


def _raw_scores_from_dump(dump, row):
    scores = list(dump["base_score"])
    per_iteration = dump["num_tree_per_iteration"]
    for index, tree in enumerate(dump["tree_info"]):
        leaf = _walk(dump, index, row)
        scores[index % per_iteration] += (
            tree["shrinkage"] * leaf["leaf_value"]
        )
    return scores


def _split_values_from_dump(dump, index):
    """Every threshold the ensemble splits feature `index` at, walked out of
    the dump. `model_dump.dump_split_values` is the same walk in Mojo, in
    the same order."""
    values = []
    for tree in dump["tree_info"]:
        stack = [tree["tree_structure"]]
        while stack:
            node = stack.pop()
            if "leaf_index" in node:
                continue
            if node["split_feature"] == index:
                if node["threshold"] is not None:
                    values.append(node["threshold"])
            stack.append(node["right_child"])
            stack.append(node["left_child"])
    return values
