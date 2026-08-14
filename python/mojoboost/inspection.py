"""Structured model inspection: a stable, versioned dump of a fitted model.

LightGBM's `Booster.dump_model()`, `Booster.trees_to_dataframe()`, and
`Booster.get_split_value_histogram()` all answer the same question in
different shapes: what does this ensemble actually contain? This module
answers it once, into a documented schema, and derives the other shapes
from that. `docs/MODEL_INSPECTION_SCHEMA.md` is the schema's normative
statement; this module is its reference implementation.

    from mojoboost import MojoBoostRegressor
    from mojoboost.inspection import dump_model, trees_to_dataframe

    model = MojoBoostRegressor(n_estimators=20).fit(X, y)
    dump = dump_model(model)
    frame = trees_to_dataframe(model)

Where the dump comes from
-------------------------
Today it is read from the model itself, through
`Booster.model_to_string()`: that text is mojoboost's versioned save format
(see src/mojoboost/serialize.mojo), it is exact (floats travel as IEEE-754
bit patterns), and it needs no new native entry point, so inspection works
against the extension module as it is built today.

That source has one gap, and it is the format's, not this module's: split
gains are not serialized, so a dump read this way carries
`split_gain: None` on every node and reports `has_split_gain: False`.
`src/mojoboost/inspection.mojo` builds the same schema from the in-memory
`Model`, where the gains are still present, and exposes them on their own
as well. When a binding exposes that smaller hook as
`_mojoboost.split_gains`, this module picks it up automatically and
`has_split_gain` becomes True; nothing else about the schema changes.
`handoffs/task14_inspection.md` states the exact binding signature.

Not offered
-----------
Leaf editing (LightGBM's `set_leaf_output`). A fitted mojoboost tree is
consistent with a set of invariants this module cannot restate after an
arbitrary leaf edit: node covers are the training rows that reached a node
and are what exact TreeSHAP conditions on, internal node values are the
values those nodes held when they were created, and split gains were
computed from the sums a leaf held at growth time. An edited leaf value
falsifies its ancestors' internal values and gains while leaving them in
place, and no check here could tell an intentional edit from a corrupt one.
Until those invariants can be stated and tested completely, the dump is
read only.
"""

import struct as _struct

__all__ = [
    "DUMP_FORMAT_VERSION",
    "SUPPORTED_MODEL_FORMAT_VERSIONS",
    "OBJECTIVE_NAMES",
    "dump_model",
    "parse_model_string",
    "trees_to_records",
    "trees_to_dataframe",
    "split_values",
    "get_split_value_histogram",
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
DUMP_FORMAT_VERSION = 1

#: Model file format versions (src/mojoboost/serialize.mojo) this module can
#: read. v1 and v2 predate node covers, so a dump built from one reports
#: `has_node_count: False`.
SUPPORTED_MODEL_FORMAT_VERSIONS = (1, 2, 3)

#: Objective code (as written by `save_model`) to LightGBM's canonical name
#: for it. The codes are the trainer's, declared in
#: src/mojoboost/boosting.mojo and mirrored by the estimators' objective
#: tables; `test_inspection.py` checks this table against those tables so
#: the two cannot drift apart silently.
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
}

#: Bin 0 of a categorical feature: missing, unseen, or dropped. Never a
#: member of a split's category set, so those rows always go right (see
#: src/mojoboost/categorical.mojo).
_UNKNOWN_BIN = 0

#: Category codes are the ones LightGBM's `static_cast<int>` can represent.
_MAX_CATEGORY = 1 << 31

_CAT_BITSET_WORDS = 4
_CAT_MAX_BINS = 64 * _CAT_BITSET_WORDS


# -- the model file's token stream ---------------------------------------
#
# A mirror of `_TokenReader` in src/mojoboost/serialize.mojo. Floats are
# stored as decimal IEEE-754 bit patterns, so reading one back is exact:
# a dump reports the same bits the trained model holds, not a rounded
# decimal of them.


def _f64_from_bits(token):
    bits = int(token)
    if bits < 0 or bits > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"float bit pattern out of range: {token!r}")
    return _struct.unpack("<d", _struct.pack("<Q", bits))[0]


class _Reader:
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
    """The raw contents of mojoboost's model text, as plain Python.

    This is the parse step alone: arrays exactly as
    src/mojoboost/serialize.mojo writes them, with no interpretation
    layered on. `dump_model` builds the documented schema from it.
    """
    reader = _Reader(text)
    magic = reader.next()
    if magic != "mojoboost":
        raise ValueError(
            f"not a mojoboost model: file starts with {magic!r}"
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
    out["monotone"] = _parse_monotone(reader, n_features)
    out["trees"] = _parse_trees(reader, version)
    return out


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
        if version >= 3:
            tree["count"] = reader.floats(n_nodes)
        else:
            tree["count"] = [0.0] * n_nodes
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


# -- the schema ----------------------------------------------------------


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
            "model inspection takes a fitted estimator, a mojoboost.Booster, "
            "or the text model_to_string() produces, not "
            f"{type(model).__name__}"
        )
    return model.booster_


def _model_text(model):
    """The model text and the feature names for whatever was handed in: an
    estimator, a `Booster`, or the text itself."""
    if isinstance(model, str):
        return model, None
    booster = _booster(model)
    return booster.model_to_string(), list(booster.feature_name())


def _native_split_gains(model):
    """Per node split gains, one list per tree, when the extension module
    exposes them.

    Gains are recorded during growth and are not serialized, so this is the
    one thing the model text cannot supply.
    `handoffs/task14_inspection.md` specifies the binding. Until it exists
    this returns None and the dump reports `has_split_gain: False`.

    A handle is a single-output model or a softmax one, and the extension
    module takes one entry point per kind, as it does for every other model
    accessor.
    """
    from . import _mojoboost

    booster = _booster(model)
    name = (
        "split_gains_multiclass"
        if getattr(booster, "_n_classes", 0)
        else "split_gains"
    )
    hook = getattr(_mojoboost, name, None)
    if hook is None:
        return None
    return hook(booster._handle)


def dump_model(model, feature_names=None):
    """The fitted model as the documented inspection schema.

    `model` is a fitted estimator, a `mojoboost.Booster`, or the text
    `Booster.model_to_string()` produces. `feature_names` overrides the
    names the model carries, and is how a caller names the features of a
    model read back from a file, which carries none.

    See `docs/MODEL_INSPECTION_SCHEMA.md` for every key. The two a
    consumer should branch on: `has_split_gain` says whether per node
    gains are present, and `has_node_count` whether per node training row
    counts are.
    """
    text, carried_names = _model_text(model)
    raw = parse_model_string(text)
    source = "model_to_string"
    gains = None
    if not isinstance(model, str):
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
    names = _resolve_names(feature_names, carried_names, n_features)
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
        "producer": "mojoboost",
        "model_format_version": raw["model_format_version"],
        "source": source,
        "objective": _objective_name(raw),
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
        "tree_info": trees,
    }


def _resolve_names(override, carried, n_features):
    if override is not None:
        names = [str(name) for name in override]
        if len(names) != n_features:
            raise ValueError(
                f"feature_names has {len(names)} entries for a model with "
                f"{n_features} features"
            )
        return names
    if carried is not None and len(carried) == n_features:
        return [str(name) for name in carried]
    return [f"Column_{i}" for i in range(n_features)]


def _objective_name(raw):
    if raw["kind"] == "multiclass":
        return "multiclass"
    code = raw["objective_code"]
    return OBJECTIVE_NAMES.get(code, f"objective_{code}")


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
    among the leaves, in node-array order. The leaf rank is mojoboost's own
    leaf ordinal, the one `predict(pred_leaf=True)` reports (see
    `Tree.leaf_ordinals` in src/mojoboost/tree.mojo)."""
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
    bin has no upper edge.

    mojoboost splits on bin ids, and a value maps to the first bin whose
    upper edge it does not exceed, so `bin <= threshold_bin` holds exactly
    when `value <= edges[threshold_bin]`. The bin's upper edge is therefore
    the exact real-valued boundary, not an approximation of one.
    """
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


# -- routing, from the dump alone ----------------------------------------


def _bin_value(info, value):
    """The bin a raw feature value lands in. A mirror of
    `BinMapper.bin_value` in src/mojoboost/binning.mojo, from the dump's
    feature record rather than from the model."""
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
    """A mirror of `Tree.goes_left`: category membership first, then the
    node's missing direction, then the threshold."""
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


def leaf_index_of(dump, tree_index, row):
    """The leaf ordinal one raw example reaches in a tree of the dump.

    The same numbering `predict(pred_leaf=True)` reports, computed from the
    dump alone, which is what makes the dump checkable against the model.
    """
    return _walk(dump, tree_index, row)["leaf_index"]


def raw_scores(dump, row):
    """Raw scores for one raw example, from the dump alone: one entry for a
    single-output model, one per class for a softmax one.

    This is where the dump's leaf values are pinned to the model's own
    arithmetic. mojoboost stores unshrunk leaf values and multiplies by the
    shrinkage when it predicts, so the sum is
    `base_score[k] + sum_over_trees(shrinkage * leaf_value)`.
    """
    scores = list(dump["base_score"])
    per_iteration = dump["num_tree_per_iteration"]
    for index, tree in enumerate(dump["tree_info"]):
        leaf = _walk(dump, index, row)
        scores[index % per_iteration] += (
            tree["shrinkage"] * leaf["leaf_value"]
        )
    return scores


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
                # mojoboost records a node's training row cover, not its
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
    mojoboost records rather than because of this function:

    - `split_gain` is None on every row unless the dump carries gains
      (`dump["has_split_gain"]`).
    - `weight` is always None: mojoboost records a node's training row
      cover, in `count`, and not the hessian sum LightGBM calls weight.

    pandas is not a dependency of mojoboost. `trees_to_records` returns the
    same rows as plain dicts and needs nothing installed.
    """
    try:
        import pandas
    except ImportError:
        raise ImportError(
            "trees_to_dataframe needs pandas, which mojoboost does not "
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
    which is the exact boundary routing uses (see `_threshold_value`), so
    the histogram they feed describes where the model actually cuts.
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


def _histogram(values, bins):
    """An equal-width histogram with `numpy.histogram`'s conventions: bins
    are half open except the last, which is closed, and a single distinct
    value is given a unit-wide bin around itself.

    Written out rather than delegated so the result does not depend on
    whether numpy happens to be installed.
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
            "as_frame=True needs pandas, which mojoboost does not depend "
            "on; the default (counts, bin_edges) return needs nothing"
        ) from None
    return pandas.DataFrame(
        {"Count": counts, "SplitValue": edges[:-1]},
        columns=["Count", "SplitValue"],
    )


# -- estimator and booster attributes ------------------------------------
#
# The six LightGBM attributes task 14 owns. Each is a one line read here so
# that wiring it onto `_Base` is a property that delegates, and so that a
# `Booster` answers the same question the same way. See
# `handoffs/task14_inspection.md` for the exact patch each one needs.


def booster_of(model):
    """The `mojoboost.Booster` behind whatever was handed in.

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
    text, _ = _model_text(booster)
    return _objective_name(parse_model_string(text))


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
