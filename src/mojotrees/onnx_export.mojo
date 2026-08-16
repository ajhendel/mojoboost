"""ONNX export for the tree ensemble: the arithmetic, and the refusals.

What this module is
-------------------
`ai.onnx.ml.TreeEnsembleRegressor` (ml opset 3) describes a tree ensemble as
a bundle of parallel arrays. This module turns a fitted `Model` or
`MulticlassModel` into exactly those arrays -- an `OnnxPlan` -- and refuses,
by name, every model whose behavior the operator cannot reproduce.

The protobuf itself is written by `python/mojotrees/onnx_export.py`, which
reads the plan's text form and calls `onnx.helper`. The split is deliberate:
everything that can be silently wrong is here, under test, with no
dependencies; the part that needs a third-party library is a transcription
with no arithmetic in it.

`docs/design/MODEL_EXPORT.md` is the normative statement of the boundary and
of what the exactness claim covers. This docstring states the three
conversions that are easy to get wrong.

1. The threshold. mojotrees splits on bin ids: `bin(x) <= threshold_bin`.
   `BinMapper.bin_value` defines `bin(x)` as the first `b` with
   `x <= edges[b]`, and edges ascend strictly, so
   `bin(x) <= t  <=>  x <= edges[t]` in both directions with no tie case.
   The conversion is an equality, and `nodes_values_as_tensor` carries the
   `Float64` edge without narrowing.

2. NaN. A feature with a reserved missing bin routes NaN by the node's
   `default_left`. A feature *without* one bins NaN as the value `0.0` and
   then takes the ordinary threshold test, which is the case an exporter
   forgets. Either way the destination is a per-node constant, so it converts
   exactly to `nodes_missing_value_tracks_true`.

3. The learning rate. `Tree.value` is stored **unscaled**; every predict path
   is `base_score + sum learning_rate * value` (boosting.mojo). An exporter
   that copies `Tree.value` into `target_weights` produces a graph that
   loads, validates, and scores wrong by a factor of `learning_rate` on every
   row, and nothing in the ONNX ecosystem would catch it. The multiply
   happens once, here.

What is claimed, and what is not
--------------------------------
Claimed: for an accepted model, every row reaches the same leaf of every tree
in the exported graph as in `Model.predict_raw`, and each tree's contribution
is the same `Float64`. Not claimed: the final score, because
`TreeEnsembleRegressor` returns `tensor(float)` in every version of the
operator and ONNX does not specify the summation order of
`aggregate_function=SUM`.
"""

from std.memory import bitcast

from .binning import BinMapper
from .linear_tree import LinearEnsemble
from .model import Model, MulticlassModel
from .objective_registry import (
    CUSTOM,
    LINK_EXP,
    LINK_SIGMOID,
    LINK_SOFTMAX,
    objective_link,
)
from .tree import Tree


comptime ONNX_PLAN_MAGIC = "mojotrees-onnx-plan"
"""First token of a plan file. A plan is not a model and not a prepared
table, so it says so before anything else does."""

comptime ONNX_PLAN_VERSION = 1
"""The plan text's own version, independent of the model format version. It
bumps when a field is added to the plan, which is a change to the contract
between this module and `python/mojotrees/onnx_export.py` and to nothing
else."""

comptime ONNX_ML_OPSET = 3
"""The `ai.onnx.ml` opset this plan targets. 3 is the version that added
`nodes_values_as_tensor`, `target_weights_as_tensor`, `base_values_as_tensor`
(so thresholds, leaf weights, and base values travel as `double`) and
`nodes_missing_value_tracks_true`. Opset 5 deprecates
`TreeEnsembleRegressor` in favor of `TreeEnsemble`; see
docs/design/MODEL_EXPORT.md section 5 for what taking it would buy."""

# `nodes_modes` as this plan carries it. The strings the operator wants are
# the Python half's business; an integer is what survives a token stream
# without an escaping question.
comptime MODE_BRANCH_LEQ = 0
comptime MODE_LEAF = 1

# `post_transform`, plus one value that is not an ONNX post-transform at all.
comptime POST_NONE = 0
comptime POST_LOGISTIC = 1
comptime POST_SOFTMAX = 2
comptime POST_EXP = 3
"""`LINK_EXP` (poisson, gamma, tweedie) has no `post_transform` spelling.
The plan records it so the Python half appends an explicit `Exp` node after
the ensemble rather than dropping the link, which would return the raw score
under a name that promises the mean."""

comptime MULTICLASS_OBJECTIVE = -1
"""`objective_registry.MULTICLASS`, restated so this module's refusal check
can be handed an objective code for a booster that carries none. It is not
imported because the multiclass path never asks for its link: the transform
is decided by which entry point was called."""

comptime _F64_POS_INF_BITS = UInt64(0x7FF0000000000000)


def _pos_inf() -> Float64:
    """`+inf` without a math import, by its bit pattern. Used for the
    degenerate threshold that sends every non-missing row left."""
    return bitcast[DType.float64, 1](
        SIMD[DType.uint64, 1](_F64_POS_INF_BITS)
    )


def _f64_to_token(x: Float64) -> String:
    """A float as its raw IEEE-754 bits in decimal, the same convention
    serialize.mojo uses. A threshold written as a decimal literal is a
    threshold that can move a row by a whole leaf."""
    return String(x.to_bits())


struct OnnxPlan(Copyable, Movable):
    """One `ai.onnx.ml.TreeEnsembleRegressor` node, as parallel arrays.

    Field names are the operator's attribute names, so a reader can check
    this against the ONNX operator reference without a translation table.
    `nodes_*` has one entry per node of every tree, concatenated in tree
    order; `target_*` has one entry per leaf.

    `nodes_nodeids` is the node's index inside its own tree, which is what
    the operator requires (ids restart at 0 per tree). mojotrees numbers
    nodes exactly that way already, so the array is an identity per tree and
    is written out anyway because the operator reads it rather than assuming
    it.
    """

    var n_features: Int
    var n_targets: Int
    var post_transform: Int
    var base_values: List[Float64]

    var nodes_treeids: List[Int]
    var nodes_nodeids: List[Int]
    var nodes_featureids: List[Int]
    var nodes_modes: List[Int]
    var nodes_values: List[Float64]
    var nodes_truenodeids: List[Int]
    var nodes_falsenodeids: List[Int]
    var nodes_missing_value_tracks_true: List[Int]

    var target_treeids: List[Int]
    var target_nodeids: List[Int]
    var target_ids: List[Int]
    var target_weights: List[Float64]

    def __init__(out self, n_features: Int, n_targets: Int, post: Int):
        self.n_features = n_features
        self.n_targets = n_targets
        self.post_transform = post
        self.base_values = List[Float64]()
        self.nodes_treeids = List[Int]()
        self.nodes_nodeids = List[Int]()
        self.nodes_featureids = List[Int]()
        self.nodes_modes = List[Int]()
        self.nodes_values = List[Float64]()
        self.nodes_truenodeids = List[Int]()
        self.nodes_falsenodeids = List[Int]()
        self.nodes_missing_value_tracks_true = List[Int]()
        self.target_treeids = List[Int]()
        self.target_nodeids = List[Int]()
        self.target_ids = List[Int]()
        self.target_weights = List[Float64]()

    def n_nodes(self) -> Int:
        return len(self.nodes_treeids)

    def n_leaves(self) -> Int:
        return len(self.target_treeids)


def onnx_threshold(mapper: BinMapper, feature: Int, bin: Int) raises -> Float64:
    """The raw-value threshold that reproduces `bin(x) <= bin` exactly.

    `bin` equal to the feature's edge count is the degenerate split that
    sends every non-missing row left; it exports as `+inf`, which is exact
    (`x <= +inf` for every finite `x`, and for `+inf` itself, and the last
    ordinary bin is where `bin_value` puts both). Anything above that cannot
    have come from a grower and is refused rather than guessed at.
    """
    if feature < 0 or feature >= mapper.n_features:
        raise Error("onnx export: feature id ", feature, " out of range")
    var lo = mapper.edge_offsets[feature]
    var n_edges = mapper.edge_offsets[feature + 1] - lo
    if bin < 0 or bin > n_edges:
        raise Error(
            "onnx export: threshold bin ",
            bin,
            " on feature ",
            feature,
            " is outside its ",
            n_edges,
            " edges",
        )
    if bin == n_edges:
        return _pos_inf()
    return mapper.edges[lo + bin]


def onnx_nan_goes_left(
    mapper: BinMapper, tree: Tree, node: Int
) raises -> Bool:
    """Where a NaN on node `node`'s split feature actually goes.

    Computed rather than read off `default_left`, because `default_left` is
    only half the answer. A feature with no reserved missing bin bins NaN as
    the value `0.0` (binning.mojo, LightGBM's `missing_type=None`) and then
    takes the ordinary threshold test, so its NaN direction is whatever that
    test says and has nothing to do with `default_left`. Binning the NaN and
    asking `Tree.goes_left` reproduces `Model.predict` by construction, which
    is the only way to be sure this is right.
    """
    var f = tree.feature[node]
    var reserved = mapper.missing_bin[f]
    var nan_bin = reserved if reserved >= 0 else mapper.bin_value(f, 0.0)
    return tree.goes_left(node, nan_bin)


def onnx_refusals(
    mapper: BinMapper,
    trees: List[Tree],
    objective: Int,
    linear: LinearEnsemble,
    raw_score: Bool,
) raises -> List[String]:
    """Every reason this model cannot be exported exactly, all at once.

    All of them, not the first: a caller that has to fix three things learns
    about three things. An empty list is the only thing that permits an
    export. See docs/design/MODEL_EXPORT.md section 3 for why each one is a
    refusal rather than an approximation.
    """
    var out = List[String]()

    if mapper.cats.any_categorical():
        out.append(
            String(
                "categorical features: ai.onnx.ml opset 3 has no"
                " set-membership split mode, mojotrees truncates a"
                " categorical input toward zero, and every unseen or"
                " out-of-range value collapses to bin 0. None of the three"
                " survives a threshold comparison"
            )
        )
    else:
        # Belt and braces. A categorical node on a mapper that declares no
        # categorical feature would be a corrupt model, and exporting it as
        # a numerical threshold would be silent.
        for t in range(len(trees)):
            ref tree = trees[t]
            var found = False
            for i in range(len(tree.cat_offset)):
                if tree.cat_offset[i] >= 0:
                    found = True
                    break
            if found:
                out.append(
                    String(
                        "tree "
                        + String(t)
                        + " has a category-set node but the mapper declares"
                        " no categorical feature; the model is inconsistent"
                    )
                )
                break

    if linear.is_active():
        out.append(
            String(
                "linear leaves: TreeEnsembleRegressor leaves are constants,"
                " and there is no ONNX tree operator with affine leaves"
            )
        )

    if objective == CUSTOM and not raw_score:
        out.append(
            String(
                "the CUSTOM objective has no link the model holds, so the"
                " graph cannot apply one; export with raw_score=True, which"
                " is what Booster.response does for CUSTOM anyway"
            )
        )

    # Structural checks, so a corrupt tree is named here rather than raising
    # from the middle of the conversion with no context.
    for t in range(len(trees)):
        ref tree = trees[t]
        for i in range(len(tree.feature)):
            var f = tree.feature[i]
            if f < 0:
                continue
            if f >= mapper.n_features:
                out.append(
                    String(
                        "tree "
                        + String(t)
                        + " node "
                        + String(i)
                        + " splits on feature "
                        + String(f)
                        + ", which the mapper does not have"
                    )
                )
                continue
            var lo = mapper.edge_offsets[f]
            var n_edges = mapper.edge_offsets[f + 1] - lo
            var b = tree.threshold_bin[i]
            if b < 0 or b > n_edges:
                out.append(
                    String(
                        "tree "
                        + String(t)
                        + " node "
                        + String(i)
                        + " has threshold bin "
                        + String(b)
                        + " outside the "
                        + String(n_edges)
                        + " edges of feature "
                        + String(f)
                    )
                )
    return out^


def _check_exportable(
    mapper: BinMapper,
    trees: List[Tree],
    objective: Int,
    linear: LinearEnsemble,
    raw_score: Bool,
) raises:
    var reasons = onnx_refusals(mapper, trees, objective, linear, raw_score)
    if len(reasons) == 0:
        return
    var msg = String(
        "this model cannot be exported to ONNX without changing its"
        " predictions, so it is not exported. Reasons:"
    )
    for i in range(len(reasons)):
        msg += "\n  - " + reasons[i]
    raise Error(msg)


def _post_transform_for(objective: Int, raw_score: Bool) raises -> Int:
    """The `post_transform` the plan records, or `POST_EXP` for the link ONNX
    has no name for."""
    if raw_score:
        return POST_NONE
    var link = objective_link(objective)
    if link == LINK_SIGMOID:
        return POST_LOGISTIC
    if link == LINK_SOFTMAX:
        return POST_SOFTMAX
    if link == LINK_EXP:
        return POST_EXP
    return POST_NONE


def _append_tree(
    mut plan: OnnxPlan,
    mapper: BinMapper,
    tree: Tree,
    tree_id: Int,
    target: Int,
    learning_rate: Float64,
) raises:
    """One tree's nodes and leaves, appended to the plan.

    `target` is the output index every leaf of this tree contributes to: 0
    for a single-output model, the class for a multiclass one.
    """
    var n_nodes = len(tree.feature)
    for i in range(n_nodes):
        plan.nodes_treeids.append(tree_id)
        plan.nodes_nodeids.append(i)
        if tree.feature[i] < 0:
            # A leaf. The operator still wants a full row, and the values in
            # it are never read for a LEAF node.
            plan.nodes_featureids.append(0)
            plan.nodes_modes.append(MODE_LEAF)
            plan.nodes_values.append(0.0)
            plan.nodes_truenodeids.append(0)
            plan.nodes_falsenodeids.append(0)
            plan.nodes_missing_value_tracks_true.append(0)
            plan.target_treeids.append(tree_id)
            plan.target_nodeids.append(i)
            plan.target_ids.append(target)
            # The one multiply. See the module docstring.
            plan.target_weights.append(learning_rate * tree.value[i])
            continue
        var f = tree.feature[i]
        plan.nodes_featureids.append(f)
        plan.nodes_modes.append(MODE_BRANCH_LEQ)
        plan.nodes_values.append(
            onnx_threshold(mapper, f, tree.threshold_bin[i])
        )
        # The true branch is the left child, which is what makes
        # `nodes_missing_value_tracks_true` mean "NaN goes left".
        plan.nodes_truenodeids.append(tree.left[i])
        plan.nodes_falsenodeids.append(tree.right[i])
        plan.nodes_missing_value_tracks_true.append(
            1 if onnx_nan_goes_left(mapper, tree, i) else 0
        )


def onnx_plan(model: Model, raw_score: Bool = False) raises -> OnnxPlan:
    """The `TreeEnsembleRegressor` arrays for a single-output model.

    Raises, naming every reason, when the model holds anything the operator
    cannot reproduce. `raw_score=True` exports the ensemble with no response
    transform, which is the mode with the exactness claim.
    """
    _check_exportable(
        model.mapper,
        model.booster.trees,
        model.booster.objective,
        model.booster.linear,
        raw_score,
    )
    var plan = OnnxPlan(
        model.mapper.n_features,
        1,
        _post_transform_for(model.booster.objective, raw_score),
    )
    plan.base_values.append(model.booster.base_score)
    for t in range(len(model.booster.trees)):
        _append_tree(
            plan,
            model.mapper,
            model.booster.trees[t],
            t,
            0,
            model.booster.learning_rate,
        )
    return plan^


def onnx_plan_multiclass(
    model: MulticlassModel, raw_score: Bool = False
) raises -> OnnxPlan:
    """The `TreeEnsembleRegressor` arrays for a softmax multiclass model.

    `MulticlassBooster` holds trees round-major, `trees[i * n_classes + k]`,
    so tree `j` contributes to target `j % n_classes`. `n_targets` is the
    class count and `base_values` is the per-class base score, which is what
    makes one operator enough for the whole model.

    The objective is passed as `MULTICLASS` rather than read off the booster
    because a `MulticlassBooster` has no objective field, and a multiclass
    model file records none either (see the A17 note in
    docs/design/CATBOOST_CATALOG.md, hazard H2).
    """
    _check_exportable(
        model.mapper,
        model.booster.trees,
        MULTICLASS_OBJECTIVE,
        model.booster.linear,
        raw_score,
    )
    var n_classes = model.booster.n_classes
    if len(model.booster.trees) % n_classes != 0:
        raise Error(
            "onnx export: tree count is not a multiple of the class count"
        )
    var plan = OnnxPlan(
        model.mapper.n_features,
        n_classes,
        POST_NONE if raw_score else POST_SOFTMAX,
    )
    for k in range(n_classes):
        plan.base_values.append(model.booster.base_scores[k])
    for t in range(len(model.booster.trees)):
        _append_tree(
            plan,
            model.mapper,
            model.booster.trees[t],
            t,
            t % n_classes,
            model.booster.learning_rate,
        )
    return plan^


def _write_int_list(mut out: String, name: String, values: List[Int]):
    out += name + " " + String(len(values)) + "\n"
    for i in range(len(values)):
        out += String(values[i]) + " "
    out += "\n"


def _write_f64_list(mut out: String, name: String, values: List[Float64]):
    out += name + " " + String(len(values)) + "\n"
    for i in range(len(values)):
        out += _f64_to_token(values[i]) + " "
    out += "\n"


def onnx_plan_text(plan: OnnxPlan) -> String:
    """The plan as a whitespace-separated token stream.

    Same conventions as the model format: a magic first, a version second,
    floats as their raw IEEE-754 bits in decimal so a threshold cannot be
    rounded on the way to the protobuf. Sections are written in a fixed
    order and every array carries its own length, so the reader never has to
    infer one. Nothing here iterates a map, because nothing in this package
    is one; the byte stream is a function of the model alone and does not
    depend on `MOJOTREES_NUM_WORKERS` or on the machine.
    """
    var out = String("")
    out += ONNX_PLAN_MAGIC + " " + String(ONNX_PLAN_VERSION) + "\n"
    out += "opset " + String(ONNX_ML_OPSET) + "\n"
    out += "n_features " + String(plan.n_features) + "\n"
    out += "n_targets " + String(plan.n_targets) + "\n"
    out += "post_transform " + String(plan.post_transform) + "\n"
    _write_f64_list(out, "base_values", plan.base_values)
    _write_int_list(out, "nodes_treeids", plan.nodes_treeids)
    _write_int_list(out, "nodes_nodeids", plan.nodes_nodeids)
    _write_int_list(out, "nodes_featureids", plan.nodes_featureids)
    _write_int_list(out, "nodes_modes", plan.nodes_modes)
    _write_f64_list(out, "nodes_values", plan.nodes_values)
    _write_int_list(out, "nodes_truenodeids", plan.nodes_truenodeids)
    _write_int_list(out, "nodes_falsenodeids", plan.nodes_falsenodeids)
    _write_int_list(
        out,
        "nodes_missing_value_tracks_true",
        plan.nodes_missing_value_tracks_true,
    )
    _write_int_list(out, "target_treeids", plan.target_treeids)
    _write_int_list(out, "target_nodeids", plan.target_nodeids)
    _write_int_list(out, "target_ids", plan.target_ids)
    _write_f64_list(out, "target_weights", plan.target_weights)
    return out^


def save_onnx_plan(plan: OnnxPlan, path: String) raises:
    """Write a plan to `path`, for `python/mojotrees/onnx_export.py` to turn
    into a `ModelProto`."""
    var text = onnx_plan_text(plan)
    with open(path, "w") as f:
        f.write(text)
