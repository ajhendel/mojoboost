"""The inspection dump as Mojo data: one schema, built from the model.

A fitted model is a tree ensemble plus the binning that feeds it, and every
consumer that looks inside one wants the same facts. This module states
those facts once, as Mojo structures built from the in-memory `Model`,
`MulticlassModel`, `Tree`, and `BinMapper`, and nothing here reads or
writes text. `inspection.mojo` renders these structures as JSON,
`bindings/_mojoboost.mojo` hands them to Python, and
`docs/MODEL_INSPECTION_SCHEMA.md` is the normative statement both answer
to.

What lives here, and why it is not in a consumer
------------------------------------------------
Everything that has to know how a mojoboost tree is laid out: node depths
and parents, leaf ordinals and split ordinals, the bin id a split compares
against and the raw value that bin id stands for, the category codes a
categorical node routes left, and which optional facts this particular
model can answer at all. A consumer that recomputed any of it would be
restating this file's invariants in a second language, and the two would
drift the first time a tree gained a field.

The dump interpreter
--------------------
`dump_raw_scores` and `dump_leaf_index` route a raw example through the
dump's own structures: they bin with the dump's `bin_upper_bounds` and
route with the dump's node records, and they never touch `BinMapper` or
`Tree`. That independence is the entire point. It is what makes "walking
the dump lands where the model lands" a claim that can fail, and so a
claim worth testing: a dump whose thresholds were converted wrongly would
score differently from `Model.predict_raw`, where a wrapper around the
model's own prediction path could not tell.

Capability flags
----------------
`has_split_gain` and `has_node_count` say what this dump can answer. Both
are read off the trees rather than assumed: a model loaded from a file has
no gains (they are recorded during growth and deliberately not
serialized), and one loaded from a v1 or v2 file has no node covers. A
consumer branches on the flags; it never has to know where the model came
from.

Not offered
-----------
Leaf editing. A grown tree carries invariants an arbitrary leaf edit
falsifies with nothing left to detect it: `count` is the training rows
that reached a node, an internal node's `value` is what it held when it
was created, and `split_gain` was computed from the sums a leaf held at
growth time. Inspection is read only until those can be restated after an
edit.
"""

from std.math import isnan

from .binning import BinMapper
from .categorical import CAT_MAX_BINS, UNKNOWN_BIN, cat_pool_contains
from .model import Model, MulticlassModel
from .monotone import MonotoneConstraints
from .tree import Tree

# The inspection schema's own version, not the model file format's. Bumped
# only for a change a consumer written against the previous version could
# not survive; adding an optional field does not bump it.
comptime DUMP_FORMAT_VERSION = 1

# The save format a model written today serializes to (see serialize.mojo).
# Recorded in the dump so a consumer knows which optional facts a model of
# this vintage can carry at all.
comptime MODEL_FORMAT_VERSION = 3

# Codes a categorical feature can represent, mirroring the private
# `_MAX_CATEGORY` in categorical.mojo. The dump interpreter reproduces that
# module's binning rule rather than calling into it, for the reason the
# header gives.
comptime _MAX_CATEGORY = 1 << 31


struct DumpFeature(Copyable, Movable):
    """How one feature is binned, and how missing and unseen values route
    through it.

    A numerical feature with `k` entries in `bin_upper_bounds` uses bins
    `0..k`, plus a missing bin at `k + 1` when `missing_bin >= 0`. A
    categorical feature carries no bounds: bin 0 collects the missing,
    unseen, and dropped codes, and `categories[i]` is bin `i + 1`.
    """

    var index: Int
    var name: String
    var is_categorical: Bool
    var bin_upper_bounds: List[Float64]
    var categories: List[Int]
    var num_bin: Int
    var missing_bin: Int
    var monotone: Int

    def __init__(out self, index: Int, var name: String):
        """An unbinned numerical feature. `_build_features` fills the rest
        from the fitted mapper."""
        self.index = index
        self.name = name^
        self.is_categorical = False
        self.bin_upper_bounds = List[Float64]()
        self.categories = List[Int]()
        self.num_bin = 1
        self.missing_bin = -1
        self.monotone = 0

    def has_missing_bin(self) -> Bool:
        """Whether this feature reserves a bin for missing values, which is
        what the schema reports as `missing_type: "NaN"`."""
        return self.missing_bin >= 0


struct DumpNode(Copyable, Movable):
    """One node of a dumped tree, leaf or internal.

    `is_leaf` tells the two apart. A leaf carries `leaf_index` (its ordinal
    among this tree's leaves, the numbering `predict(pred_leaf=True)`
    reports) and `value` / `count`; an internal node carries `split_index`
    (its ordinal among this tree's internal nodes) and its routing.

    `node_index` is the index into the tree's flat arrays, and `left`,
    `right`, and `parent` are the same kind of index, with -1 where there
    is none. They are stable for a given model, which is what makes a node
    addressable, but they number internal nodes and leaves together and so
    are an implementation detail of the layout: prefer `leaf_index` when
    you mean a leaf.
    """

    var node_index: Int
    var depth: Int
    var parent: Int
    var left: Int
    var right: Int
    var is_leaf: Bool
    var leaf_index: Int
    var split_index: Int
    var split_feature: Int
    var is_categorical: Bool
    var threshold_bin: Int
    var has_threshold: Bool
    var threshold: Float64
    var category_bins: List[Int]
    var categories: List[Int]
    var default_left: Bool
    var missing_bin: Int
    var split_gain: Float64
    var value: Float64
    var count: Float64

    def __init__(out self, node_index: Int, depth: Int, parent: Int):
        """A leaf-shaped record that routes nothing. `_node_records` fills
        in the split fields for a node that has one, so every field a leaf
        does not use has one obvious value rather than a stale one."""
        self.node_index = node_index
        self.depth = depth
        self.parent = parent
        self.left = -1
        self.right = -1
        self.is_leaf = True
        self.leaf_index = -1
        self.split_index = -1
        self.split_feature = -1
        self.is_categorical = False
        self.threshold_bin = -1
        self.has_threshold = False
        self.threshold = 0.0
        self.category_bins = List[Int]()
        self.categories = List[Int]()
        self.default_left = False
        self.missing_bin = -1
        self.split_gain = 0.0
        self.value = 0.0
        self.count = 0.0


struct DumpTree(Copyable, Movable):
    """One tree of the ensemble, as a flat node table plus the facts about
    the tree as a whole.

    `nodes[i].node_index` is `i`: the table is in node-array order, which is
    the order serialization writes and the order leaf ordinals are assigned
    in. Node 0 is the root. A consumer that wants a nested structure
    follows `left` and `right`; one that wants a table reads the list.
    """

    var tree_index: Int
    var iteration: Int
    var class_id: Int
    var num_leaves: Int
    var num_cat: Int
    var max_depth: Int
    var shrinkage: Float64
    var nodes: List[DumpNode]

    def __init__(
        out self,
        tree_index: Int,
        iteration: Int,
        class_id: Int,
        num_leaves: Int,
        shrinkage: Float64,
        var nodes: List[DumpNode],
    ):
        """`num_cat` and `max_depth` are counted off the node table rather
        than passed in, so they cannot disagree with it."""
        self.tree_index = tree_index
        self.iteration = iteration
        self.class_id = class_id
        self.num_leaves = num_leaves
        self.shrinkage = shrinkage
        self.num_cat = 0
        self.max_depth = 0
        for i in range(len(nodes)):
            if nodes[i].is_categorical:
                self.num_cat += 1
            if nodes[i].depth > self.max_depth:
                self.max_depth = nodes[i].depth
        self.nodes = nodes^

    def num_nodes(self) -> Int:
        return len(self.nodes)


struct ModelDump(Copyable, Movable):
    """A fitted model as the inspection schema.

    Built by `build_dump` or `build_multiclass_dump`; rendered as JSON by
    `inspection.mojo`; handed to Python by the bindings. The two capability
    flags are the ones a consumer must branch on before using gains or
    covers.
    """

    var dump_format_version: Int
    var model_format_version: Int
    var objective_code: Int
    var has_objective_code: Bool
    var num_class: Int
    var num_tree_per_iteration: Int
    var num_iteration: Int
    var learning_rate: Float64
    var base_score: List[Float64]
    var num_feature: Int
    var num_bin: Int
    var monotone_constraints: List[Int]
    var has_split_gain: Bool
    var has_node_count: Bool
    var features: List[DumpFeature]
    var trees: List[DumpTree]

    def __init__(out self, num_feature: Int, num_bin: Int):
        """An empty dump of a model with this shape. `_build` fills it; no
        other caller should construct one directly."""
        self.dump_format_version = DUMP_FORMAT_VERSION
        self.model_format_version = MODEL_FORMAT_VERSION
        self.objective_code = -1
        self.has_objective_code = False
        self.num_class = 1
        self.num_tree_per_iteration = 1
        self.num_iteration = 0
        self.learning_rate = 0.0
        self.base_score = List[Float64]()
        self.num_feature = num_feature
        self.num_bin = num_bin
        self.monotone_constraints = List[Int]()
        self.has_split_gain = False
        self.has_node_count = False
        self.features = List[DumpFeature]()
        self.trees = List[DumpTree]()

    def max_feature_idx(self) -> Int:
        """`num_feature - 1`, the way LightGBM reports the same fact."""
        return self.num_feature - 1

    def is_constrained(self) -> Bool:
        """Whether the model was grown under monotonic constraints, which
        is when `monotone_constraints` has an entry per feature rather than
        being empty."""
        return len(self.monotone_constraints) == self.num_feature

    def feature_index(self, name: String) raises -> Int:
        """The index of the feature called `name`.

        Names are the dump's own (`Column_0`, `Column_1`, ... when the model
        carries none), so this is the lookup a caller naming a feature
        needs, and it raises rather than guessing when no feature answers to
        the name.
        """
        for f in range(len(self.features)):
            if self.features[f].name == name:
                return f
        raise Error("no feature named '", name, "' in this model")


def feature_names_or_default(
    names: List[String], n_features: Int
) -> List[String]:
    """The names to report. A model carries none of its own, so an empty or
    mismatched list falls back to LightGBM's `Column_0`, `Column_1`, ...,
    which is what LightGBM's `feature_name()` reports in that case."""
    if len(names) == n_features:
        return names.copy()
    var out = List[String](capacity=n_features)
    for f in range(n_features):
        out.append(String("Column_") + String(f))
    return out^


def node_depths(tree: Tree) -> List[Int]:
    """Each node's depth in edges from the root: 0 at the root, so this is
    the quantity `TreeParams.max_depth` bounds. `Tree.depth` reports the
    maximum of this list."""
    var n_nodes = len(tree.feature)
    var depths = List[Int](capacity=n_nodes)
    depths.resize(n_nodes, 0)
    if n_nodes == 0:
        return depths^
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


def node_parents(tree: Tree) -> List[Int]:
    """Each node's parent, and -1 for the root. Growth records children on
    their parent and not the other way round, so a consumer that wants to
    walk upward (a decision path, a frame with a parent column) needs this
    built once rather than searched for per node."""
    var n_nodes = len(tree.feature)
    var parents = List[Int](capacity=n_nodes)
    parents.resize(n_nodes, -1)
    if n_nodes == 0:
        return parents^
    for node in range(n_nodes):
        if tree.feature[node] < 0:
            continue
        parents[tree.left[node]] = node
        parents[tree.right[node]] = node
    return parents^


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
        raise Error("feature ", feature, " has no bin edge at ", bin)
    return mapper.edges[mapper.edge_offsets[feature] + bin]


def has_split_gains(trees: List[Tree]) -> Bool:
    """Whether this ensemble's gains survived to be reported.

    Gains are recorded when a node is split and are not serialized, so a
    model read back from a file carries zeros. A split is only ever taken
    for a positive gain, so one positive gain anywhere settles it, and an
    ensemble that never split has no gain to report either way.
    """
    for t in range(len(trees)):
        for i in range(len(trees[t].split_gain)):
            if trees[t].split_gain[i] > 0.0:
                return True
    return False


def has_node_counts(trees: List[Tree]) -> Bool:
    """Whether every tree carries covers. False for a model loaded from a
    v1 or v2 file, whose nodes predate them, and for an empty ensemble,
    which has no cover to report."""
    if len(trees) == 0:
        return False
    for t in range(len(trees)):
        if not trees[t].has_node_counts():
            return False
    return True


def _build_features(
    mapper: BinMapper, names: List[String], monotone: MonotoneConstraints
) raises -> List[DumpFeature]:
    """One record per feature: how it is binned, how missing values route
    through it, and the constraint sign it was grown under."""
    var out = List[DumpFeature](capacity=mapper.n_features)
    var constrained = len(monotone.signs) == mapper.n_features
    for f in range(mapper.n_features):
        var info = DumpFeature(f, names[f].copy())
        info.missing_bin = mapper.missing_bin[f]
        info.monotone = monotone.signs[f] if constrained else 0
        if mapper.cats.is_cat(f):
            var begin = mapper.cats.offsets[f]
            var end = mapper.cats.offsets[f + 1]
            info.is_categorical = True
            for i in range(begin, end):
                info.categories.append(mapper.cats.codes[i])
            # Bin 0 collects the missing, unseen, and dropped codes.
            info.num_bin = len(info.categories) + 1
        else:
            var begin = mapper.edge_offsets[f]
            var end = mapper.edge_offsets[f + 1]
            for i in range(begin, end):
                info.bin_upper_bounds.append(mapper.edges[i])
            info.num_bin = len(info.bin_upper_bounds) + 1
            if info.missing_bin >= 0:
                info.num_bin += 1
        out.append(info^)
    return out^


def _node_records(mapper: BinMapper, tree: Tree) raises -> List[DumpNode]:
    """One record per node of one tree, in node-array order."""
    var n_nodes = len(tree.feature)
    var out = List[DumpNode](capacity=n_nodes)
    if n_nodes == 0:
        return out^
    var depths = node_depths(tree)
    var parents = node_parents(tree)
    var splits = split_ordinals(tree)
    var leaves = tree.leaf_ordinals()
    for node in range(n_nodes):
        var rec = DumpNode(node, depths[node], parents[node])
        rec.value = tree.value[node]
        rec.count = tree.count[node]
        if tree.feature[node] < 0:
            rec.leaf_index = leaves[node]
            out.append(rec^)
            continue
        var feature = tree.feature[node]
        var bin = tree.threshold_bin[node]
        rec.is_leaf = False
        rec.left = tree.left[node]
        rec.right = tree.right[node]
        rec.split_index = splits[node]
        rec.split_feature = feature
        rec.threshold_bin = bin
        rec.default_left = tree.default_left[node]
        rec.missing_bin = tree.missing_bin[node]
        rec.split_gain = tree.split_gain[node]
        if tree.cat_offset[node] >= 0:
            rec.is_categorical = True
            rec.category_bins = category_bins(tree, node)
            rec.categories = category_codes(mapper, tree, node)
        elif has_threshold(mapper, feature, bin):
            rec.has_threshold = True
            rec.threshold = threshold_value(mapper, feature, bin)
        out.append(rec^)
    return out^


def _build(
    mapper: BinMapper,
    trees: List[Tree],
    monotone: MonotoneConstraints,
    learning_rate: Float64,
    base_scores: List[Float64],
    n_classes: Int,
    per_iteration: Int,
    objective_code: Int,
    has_objective_code: Bool,
    feature_names: List[String],
) raises -> ModelDump:
    """The shared body of the two builders. Everything a single-output and a
    softmax model do not share is an argument, so there is one description
    of the schema and not two."""
    if per_iteration < 1:
        raise Error(
            "a model grows at least one tree per iteration, not ",
            per_iteration,
        )
    if len(trees) % per_iteration != 0:
        raise Error(
            "an ensemble of ",
            len(trees),
            " trees does not divide into iterations of ",
            per_iteration,
        )
    var names = feature_names_or_default(feature_names, mapper.n_features)
    var dump = ModelDump(mapper.n_features, mapper.n_bins)
    dump.objective_code = objective_code
    dump.has_objective_code = has_objective_code
    dump.num_class = n_classes
    dump.num_tree_per_iteration = per_iteration
    dump.num_iteration = len(trees) // per_iteration
    dump.learning_rate = learning_rate
    dump.base_score = base_scores.copy()
    if len(monotone.signs) == mapper.n_features:
        dump.monotone_constraints = monotone.signs.copy()
    dump.has_split_gain = has_split_gains(trees)
    dump.has_node_count = has_node_counts(trees)
    dump.features = _build_features(mapper, names, monotone)
    for t in range(len(trees)):
        dump.trees.append(
            DumpTree(
                t,
                t // per_iteration,
                t % per_iteration,
                trees[t].n_leaves,
                learning_rate,
                _node_records(mapper, trees[t]),
            )
        )
    return dump^


def build_dump(
    model: Model, feature_names: List[String] = []
) raises -> ModelDump:
    """A fitted single-output model as the inspection schema.

    `feature_names` names the features; a model carries no names of its own,
    so an empty list gives LightGBM's `Column_0`, `Column_1`, ...

    The objective travels as the trainer's code, in `objective_code`. The
    name that code stands for is the consumer's to resolve: the code is what
    the model holds, and the LightGBM spelling of it belongs with the API
    that accepts those spellings.
    """
    var base_scores: List[Float64] = [model.booster.base_score]
    return _build(
        model.mapper,
        model.booster.trees,
        model.booster.monotone,
        model.booster.learning_rate,
        base_scores,
        1,
        1,
        model.booster.objective,
        True,
        feature_names,
    )


def build_multiclass_dump(
    model: MulticlassModel, feature_names: List[String] = []
) raises -> ModelDump:
    """A fitted softmax model as the inspection schema.

    Trees keep the ensemble's round-major order, so `tree_index` runs
    straight through and each tree reports the `iteration` and `class_id` it
    belongs to rather than making a reader recompute them. The objective is
    the softmax one by construction, so there is no code to report and
    `has_objective_code` is False.
    """
    return _build(
        model.mapper,
        model.booster.trees,
        model.booster.monotone,
        model.booster.learning_rate,
        model.booster.base_scores,
        model.booster.n_classes,
        model.booster.n_classes,
        -1,
        False,
        feature_names,
    )


# -- the dump interpreter ------------------------------------------------
#
# Routing a raw example through the dump alone, with no `BinMapper` and no
# `Tree`. Deliberately a second implementation of binning and routing: see
# the module header for why a wrapper around the model's own prediction
# path would be worth nothing here.


def _categorical_bin(feature: DumpFeature, value: Float64) -> Int:
    """The bin a raw value of a categorical feature lands in.

    `UNKNOWN_BIN` for a missing value (negative or NaN), for a code the
    fitted table does not keep, and for a value outside the representable
    range. Non-integral values truncate toward zero, matching LightGBM's
    `static_cast<int>`.
    """
    # `not (value >= 0.0)` also rejects NaN.
    if not (value >= 0.0) or value >= Float64(_MAX_CATEGORY):
        return UNKNOWN_BIN
    var code = Int(value)
    var lo = 0
    var hi = len(feature.categories)
    while lo < hi:
        var mid = (lo + hi) // 2
        if feature.categories[mid] < code:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(feature.categories) and feature.categories[lo] == code:
        return lo + 1
    return UNKNOWN_BIN


def _numerical_bin(feature: DumpFeature, value: Float64) -> Int:
    """The bin a raw value of a numerical feature lands in: the first bin
    whose upper edge the value does not exceed, and the last bin for a value
    above every edge."""
    var v = value
    if isnan(v):
        if feature.missing_bin >= 0:
            return feature.missing_bin
        # No reserved bin: NaN bins as 0.0, as LightGBM does for a feature
        # whose missing_type is None.
        v = 0.0
    var lo = 0
    var hi = len(feature.bin_upper_bounds)
    while lo < hi:
        var mid = (lo + hi) // 2
        if v <= feature.bin_upper_bounds[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def dump_bin_value(feature: DumpFeature, value: Float64) -> Int:
    """The bin id a raw feature value lands in, from the dump's own record
    of the feature. The counterpart of `BinMapper.bin_value`, computed from
    the schema rather than from the fitted mapper."""
    if feature.is_categorical:
        return _categorical_bin(feature, value)
    return _numerical_bin(feature, value)


def dump_goes_left(node: DumpNode, bin: Int) -> Bool:
    """Whether a row in `bin` of `node`'s split feature goes left.

    The counterpart of `Tree.goes_left`, in the schema's terms: category
    membership first (bin 0 is never a member, so missing, unseen, and
    dropped categories go right), then the node's missing direction, then
    the threshold.
    """
    if node.is_categorical:
        for i in range(len(node.category_bins)):
            if node.category_bins[i] == bin:
                return True
        return False
    if bin == node.missing_bin:
        return node.default_left
    return bin <= node.threshold_bin


def dump_leaf_node(
    dump: ModelDump, tree_index: Int, row: List[Float64]
) raises -> Int:
    """The node index of the leaf one raw example reaches in a tree of the
    dump."""
    if tree_index < 0 or tree_index >= len(dump.trees):
        raise Error(
            "tree ",
            tree_index,
            " is outside an ensemble of ",
            len(dump.trees),
            " trees",
        )
    if len(row) != dump.num_feature:
        raise Error(
            "row has ",
            len(row),
            " values for a model with ",
            dump.num_feature,
            " features",
        )
    ref tree = dump.trees[tree_index]
    if len(tree.nodes) == 0:
        raise Error("tree ", tree_index, " has no nodes")
    var node = 0
    while not tree.nodes[node].is_leaf:
        var feature = tree.nodes[node].split_feature
        var bin = dump_bin_value(dump.features[feature], row[feature])
        if dump_goes_left(tree.nodes[node], bin):
            node = tree.nodes[node].left
        else:
            node = tree.nodes[node].right
    return node


def dump_leaf_index(
    dump: ModelDump, tree_index: Int, row: List[Float64]
) raises -> Int:
    """The leaf ordinal one raw example reaches in a tree of the dump.

    The numbering `predict(pred_leaf=True)` reports, arrived at from the
    dump alone, which is what makes the dump checkable against the model.
    """
    return dump.trees[tree_index].nodes[
        dump_leaf_node(dump, tree_index, row)
    ].leaf_index


def dump_raw_scores(
    dump: ModelDump, row: List[Float64]
) raises -> List[Float64]:
    """Raw scores for one raw example, from the dump alone: one entry for a
    single-output model, one per class for a softmax one.

    This is where the dump's leaf values are pinned to the model's own
    arithmetic. mojoboost stores unshrunk leaf values and multiplies by the
    shrinkage when it predicts, so the sum is
    `base_score[k] + sum over trees of (shrinkage * leaf_value)`.
    """
    var scores = dump.base_score.copy()
    var per_iteration = dump.num_tree_per_iteration
    for t in range(len(dump.trees)):
        var node = dump_leaf_node(dump, t, row)
        scores[t % per_iteration] += (
            dump.trees[t].shrinkage * dump.trees[t].nodes[node].value
        )
    return scores^


def dump_split_values(
    dump: ModelDump, feature: Int
) raises -> List[Float64]:
    """Every threshold the ensemble splits `feature` at, tree by tree and
    depth first within a tree, root first.

    The values are the split bins' upper edges, which is the exact boundary
    routing uses (see `threshold_value`), so a histogram built from them
    describes where the model actually cuts. A categorical feature is
    refused: its splits are category sets and have no value to bin, which is
    what LightGBM refuses for the same reason.
    """
    if feature < 0 or feature >= dump.num_feature:
        raise Error(
            "feature index ",
            feature,
            " is outside the model's ",
            dump.num_feature,
            " features",
        )
    if dump.features[feature].is_categorical:
        raise Error(
            "feature '",
            dump.features[feature].name,
            "' is categorical, so its splits are category sets and have no"
            " value to bin",
        )
    var out = List[Float64]()
    for t in range(len(dump.trees)):
        ref tree = dump.trees[t]
        if len(tree.nodes) == 0:
            continue
        var stack: List[Int] = [0]
        while len(stack) > 0:
            var node = stack.pop()
            if tree.nodes[node].is_leaf:
                continue
            if (
                tree.nodes[node].split_feature == feature
                and tree.nodes[node].has_threshold
            ):
                out.append(tree.nodes[node].threshold)
            stack.append(tree.nodes[node].right)
            stack.append(tree.nodes[node].left)
    return out^
