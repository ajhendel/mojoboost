"""Editing a fitted model: rollback, bounds, leaf outputs, order, refit.

LightGBM lets a caller reach into a trained `Booster` and change it:
`rollback_one_iter`, `set_leaf_output`, `shuffle_models`, `refit`, and the
two prediction bounds `lower_bound` and `upper_bound`. mojotrees has had
none of them, and `inspection.mojo` says so with a status rather than an
`AttributeError` (`MODEL_EDITING_SUPPORTED = False`). This module is that
status turned into an implementation, and it is deliberately narrower than
LightGBM's: every operation here is one that leaves the model **loadable**
and **internally consistent**, and every operation that would not is
refused by name (see `editing_capabilities`).

What "consistent" means here
----------------------------
A fitted mojotrees model is a set of claims. An edit is safe exactly when
it leaves every claim either true or explicitly retracted:

1. **Routing.** `feature`, `threshold_bin`, `default_left`, `missing_bin`,
   `cat_offset`, `cat_bitset`, `left`, `right` decide which leaf a row
   reaches. Nothing here changes any of them, so every row lands where it
   landed before. That is what makes leaf editing and refit safe at all.
2. **Covers.** `Tree.count[i]` is how many training rows reached node `i`.
   Because routing never changes, a leaf-value edit leaves every cover
   exactly as true as it was. `refit` may recompute them from the refit
   data, and does so all-or-nothing (§ `RefitParams.recount`) so a model
   never carries covers from two different datasets.
3. **Internal values.** An internal node keeps the value it held when it
   was created. Nothing here writes one; `set_leaf_output` refuses a
   non-leaf node. This matters twice: `node_bounds` recovers the monotone
   interval chain from those values, and the dump reports them as
   `internal_value`.
4. **The monotone claim.** `Booster.monotone` asserts that predictions are
   monotone in the constrained features. A leaf edit can falsify it, so
   every write goes through `check_monotone_claim`, which re-derives the
   whole interval chain and verifies every leaf afterwards. A write that
   would break the claim is either clamped into the leaf's interval
   (`LEAF_EDIT_CLAMP`, the default) or refused (`LEAF_EDIT_REJECT`); it is
   never silently accepted.
5. **Split gains.** A gain was computed from the gradient sums a leaf held
   at growth time. Editing a leaf value, or refitting, does not rescore any
   split: gains keep describing the fit that grew the tree, which is also
   what LightGBM's `refit` does. That is provenance, not inconsistency, and
   `clear_split_gains` is the explicit way to retract it. Split-count
   importance is structural and survives every operation here except
   rollback; gain importance survives arithmetically and goes stale in
   meaning after a refit.
6. **Loadability.** `check_tree_serializable` restates the model reader's
   own admission rules (see `_read_trees` in serialize.mojo). Every
   mutating entry point here checks them before it returns, so an edit
   cannot produce a model that this build can hold but not read back.

Base scores
-----------
`Booster.base_score` (and `MulticlassBooster.base_scores`) is model state,
not a tree, and it belongs to iteration 0 (see `IterationRange`). Nothing
in this module changes it: rollback, shuffling, and refit all leave it
alone, so an ensemble rolled back to zero iterations predicts exactly the
base score, which is what a zero-iteration ensemble has always predicted.
A model trained from an `init_score` carries a base score of 0 and the
offset stays the caller's; `refit` therefore takes the same `init_score`
argument `train_more` does, for the same reason.

Ordering
--------
Trees are stored round-major. A single-output ensemble holds one tree per
iteration; a softmax ensemble holds `n_classes` per iteration, and
`trees[i * n_classes + k]` **is** class `k`'s tree of round `i`. Any
reordering that broke that indexing would silently change which class a
tree scores, so `shuffle_iterations_multiclass` permutes whole iteration
blocks and never reorders within one. Rollback likewise removes whole
iterations.

Early stopping and continued training
-------------------------------------
An ensemble carries no early-stopping metadata; the Python layer holds it
(`best_iteration_`). Rollback and shuffling invalidate it -- rollback
changes what an iteration index reaches, shuffling changes what it means --
and refit does not change the iteration count but does make the recorded
best *score* stale. The handoff carries the exact Python invalidation
patch. Continued training (`train_more`) stays valid after any operation
here: it recomputes the raw scores from whatever trees the ensemble holds,
so it resumes from the edited model rather than from the model as trained.
It is no longer true afterwards that "40 rounds then 60 more equals 100 in
one call": that claim is about an unedited ensemble.

Boosting modes
--------------
A `Booster` records no boosting mode, so the caller states it. It matters
for exactly one operation:

- **GBDT** (`EDIT_MODE_GBDT`): the ensemble is a sum, so dropping the last
  iteration leaves the model the first `n-1` rounds produced.
- **Random forest** (`EDIT_MODE_RF`): `learning_rate` is `1 / K` and the
  ensemble is an average, so dropping a tree also has to rescale the rate,
  which changes what every remaining tree contributes. That is the correct
  forest of `K - 1` trees, and it is a different arithmetic from GBDT
  rollback, so it is opt-in.
- **DART** (`EDIT_MODE_DART`): dropped iterations were rescaled in place
  when later rounds added trees, so the first `n-1` trees of a DART model
  are not the model DART would have produced at round `n-1`. Rollback is
  refused, as it is in LightGBM.

`boosting_rf.is_forest` is a structural test, not a label, so nothing here
guesses the mode from the ensemble.
"""

from std.math import exp

from .binning import BinMapper, BinnedMatrix
from .boosting import (
    Booster,
    CUSTOM,
    IterationRange,
    MulticlassBooster,
    _check_sample_weight,
    _fill_softmax_grad_hess,
    _renew_leaf_values,
    _same_signs,
    _softmax_inplace,
    fill_grad_hess,
    objective_renews_leaves,
    renewal_alpha,
    renewal_weights,
)
from .categorical import CAT_BITSET_WORDS
from .gain import soft_threshold_l1
from .model import Model, MulticlassModel
from .monotone import MonotoneConstraints, OutputBounds
from .ranking import LAMBDARANK
from .sampling import _splitmix64, _uniform
from .tree import Tree, TreeParams, node_bounds
from .tree_parameters_extra import finish_leaf_output
from .trainset import Dataset, _int_labels

# Whether this build can edit a fitted model in place. True, and the one
# place that decides it: `editing_capabilities` says which operations that
# covers and which it does not, and `python/mojotrees/inspection.py` mirrors
# it as `MODEL_EDITING_SUPPORTED`. It is not a flag: it is a claim about the
# invariants listed in the module docstring, each of which is checked here.
comptime MODEL_EDITING_SUPPORTED = True

# The boosting mode a caller states when it matters. `EDIT_MODE_UNKNOWN` is
# what a model file, a pickle, or an estimator handle leaves a caller with:
# operations that depend on the mode refuse it rather than assume GBDT.
comptime EDIT_MODE_GBDT = 0
comptime EDIT_MODE_RF = 1
comptime EDIT_MODE_DART = 2
comptime EDIT_MODE_UNKNOWN = -1

# What a leaf write does when the model records monotonic constraints and
# the requested value would break them.
comptime LEAF_EDIT_CLAMP = 0
comptime LEAF_EDIT_REJECT = 1

# Domain separator for the shuffle stream, so a shuffle seed cannot collide
# with the bagging or feature-sampling stream of the same integer seed.
comptime _SHUFFLE_DOMAIN = UInt64(0xC3D2_E1F0_5A69_7B84)

# Larger than any finite Float64, so a comparison against it detects the
# infinities without an `isinf`. Same constant inspection.mojo uses.
comptime _F64_MAX = 1.7976931348623157e308


# -- value hygiene -------------------------------------------------------


def _is_finite(x: Float64) -> Bool:
    """False for NaN and for either infinity.

    NaN fails every comparison including with itself, which is what the
    first test uses; the infinities are caught by magnitude.
    """
    if not (x == x):
        return False
    return x <= _F64_MAX and x >= -_F64_MAX


def check_finite(value: Float64, what: String) raises:
    """Refuse a non-finite value before it reaches a model.

    A NaN or infinite leaf value survives serialization (floats travel as
    raw bit patterns) and poisons every prediction that reaches the leaf,
    while the JSON dump can only report it as `null`. There is no reading of
    such a model that is more useful than the refusal.
    """
    if not _is_finite(value):
        raise Error(
            what,
            " must be finite; got ",
            value,
            ". A non-finite leaf value round-trips through the model file"
            " and makes every prediction that reaches it non-finite",
        )


# -- structural and serialization admission ------------------------------


def check_tree_structure(tree: Tree) raises:
    """Raise unless `tree` is a well-formed tree.

    The checks a grower satisfies by construction and an edited or loaded
    tree has to be held to: parallel arrays of one length, a root, children
    that exist and lie below their parent, a leaf count that matches the
    leaves, and a categorical pool whose offsets address whole sets.

    The "children lie below their parent" rule is not decoration.
    `node_bounds` propagates the monotone interval chain in one ascending
    pass over node ids, which is correct exactly because every grower
    appends a child after its parent and serialization preserves array
    order. A tree that violated it would get silently wrong intervals, and
    the monotone claim would be checked against them.
    """
    var n_nodes = len(tree.feature)
    if n_nodes < 1:
        raise Error("tree has no nodes")
    if (
        len(tree.threshold_bin) != n_nodes
        or len(tree.left) != n_nodes
        or len(tree.right) != n_nodes
        or len(tree.value) != n_nodes
        or len(tree.split_gain) != n_nodes
        or len(tree.default_left) != n_nodes
        or len(tree.missing_bin) != n_nodes
        or len(tree.cat_offset) != n_nodes
        or len(tree.count) != n_nodes
    ):
        raise Error(
            "tree arrays disagree on the node count: every per-node array"
            " must be as long as `feature`"
        )
    if len(tree.cat_bitset) % CAT_BITSET_WORDS != 0:
        raise Error(
            "tree category pool holds ",
            len(tree.cat_bitset),
            " words, which is not a whole number of ",
            CAT_BITSET_WORDS,
            "-word sets",
        )
    var leaves = 0
    for i in range(n_nodes):
        if tree.feature[i] < 0:
            leaves += 1
            continue
        var left = tree.left[i]
        var right = tree.right[i]
        if left <= i or right <= i or left >= n_nodes or right >= n_nodes:
            raise Error(
                "tree node ",
                i,
                " has children (",
                left,
                ", ",
                right,
                ") that are not later nodes of a ",
                n_nodes,
                "-node tree; the monotone interval chain is recovered in one"
                " ascending pass and needs children after their parent",
            )
        if left == right:
            raise Error("tree node ", i, " has one node as both children")
        var off = tree.cat_offset[i]
        if off >= 0:
            if off % CAT_BITSET_WORDS != 0:
                raise Error(
                    "tree node ",
                    i,
                    " has category set offset ",
                    off,
                    ", which does not start a set",
                )
            if off + CAT_BITSET_WORDS > len(tree.cat_bitset):
                raise Error(
                    "tree node ",
                    i,
                    " has category set offset ",
                    off,
                    " past the end of a ",
                    len(tree.cat_bitset),
                    "-word pool",
                )
    if leaves != tree.n_leaves:
        raise Error(
            "tree records ",
            tree.n_leaves,
            " leaves but holds ",
            leaves,
        )


def check_tree_serializable(tree: Tree, n_features: Int, n_bins: Int) raises:
    """Raise unless `tree` is one the model reader would accept back.

    This restates `_read_trees` in serialize.mojo: the reader's admission
    rules are what "loadable" means, and an editing API that could produce a
    model this build holds but cannot read would be exactly the failure this
    module exists to prevent. Structure is checked first, then the two
    dataset-relative facts a tree alone cannot settle (a split feature has
    to exist, a missing bin has to be a bin) and the covers.

    Covers are the sharp edge. The writer decides whether to write them from
    `Tree.has_node_counts`, which looks at the root alone, while the reader
    requires **every** cover to be positive. A tree with a positive root
    cover and a zero anywhere else is therefore written and then rejected,
    which is why `refit` recomputes covers all-or-nothing and why this
    checks all of them.
    """
    check_tree_structure(tree)
    if tree.n_leaves < 1:
        raise Error("tree records a nonpositive leaf count")
    var n_nodes = len(tree.feature)
    for i in range(n_nodes):
        if tree.feature[i] >= n_features:
            raise Error(
                "tree node ",
                i,
                " splits on feature ",
                tree.feature[i],
                " in a ",
                n_features,
                "-feature model",
            )
        var mb = tree.missing_bin[i]
        if mb < -1 or mb >= n_bins:
            raise Error(
                "tree node ",
                i,
                " routes missing values to bin ",
                mb,
                ", outside the ",
                n_bins,
                " bins the model was fitted with",
            )
        if not _is_finite(tree.value[i]):
            raise Error(
                "tree node ", i, " holds a non-finite value: ", tree.value[i]
            )
    if tree.has_node_counts():
        for i in range(n_nodes):
            if not tree.count[i] > 0.0:
                raise Error(
                    "tree node ",
                    i,
                    " has a nonpositive cover (",
                    tree.count[i],
                    ") while the tree reports covers; the model reader"
                    " refuses such a file",
                )


def check_booster_serializable(model: Model) raises:
    """Raise unless every tree of a fitted single-output model would load
    back. The mapper supplies the two facts a tree alone cannot check."""
    for t in range(len(model.booster.trees)):
        check_tree_serializable(
            model.booster.trees[t],
            model.mapper.n_features,
            model.mapper.n_bins,
        )


def check_multiclass_serializable(model: MulticlassModel) raises:
    """The softmax counterpart of `check_booster_serializable`, plus the one
    whole-ensemble rule the multiclass reader enforces: the tree count has
    to divide by the class count, or the round-major indexing that assigns a
    tree to a class does not exist."""
    if model.booster.n_classes < 2:
        raise Error(
            "a softmax model needs at least two classes; this one records ",
            model.booster.n_classes,
        )
    if len(model.booster.trees) % model.booster.n_classes != 0:
        raise Error(
            "softmax ensemble holds ",
            len(model.booster.trees),
            " trees, which is not a whole number of ",
            model.booster.n_classes,
            "-tree iterations",
        )
    if len(model.booster.base_scores) != model.booster.n_classes:
        raise Error(
            "softmax ensemble records ",
            len(model.booster.base_scores),
            " base scores for ",
            model.booster.n_classes,
            " classes",
        )
    for t in range(len(model.booster.trees)):
        check_tree_serializable(
            model.booster.trees[t],
            model.mapper.n_features,
            model.mapper.n_bins,
        )


# -- the monotone claim --------------------------------------------------


def monotone_claim_holds(tree: Tree, signs: List[Int]) -> Bool:
    """Whether every leaf of `tree` still lies in its monotone interval.

    `node_bounds` derives the intervals from the internal node values, which
    editing never touches, so this is a complete check of the claim
    `Booster.monotone` makes about one tree: the proof in monotone.mojo
    needs exactly "every leaf value lies in its own interval".

    Note what it is *not*: a per-leaf check. Changing one leaf's value moves
    the midpoint its parent divides at, which moves its **sibling** subtree's
    interval. That is why a write clamps against the pre-edit intervals and
    then re-derives and re-checks the whole tree.
    """
    if len(signs) == 0:
        return True
    var bounds = node_bounds(tree, signs)
    if len(bounds) == 0:
        return True
    for i in range(len(tree.feature)):
        if tree.feature[i] >= 0:
            continue
        if tree.value[i] < bounds[i].lo or tree.value[i] > bounds[i].hi:
            return False
    return True


def check_monotone_claim(booster: Booster) raises:
    """Raise unless every tree of `booster` still satisfies the monotonic
    constraints the ensemble records."""
    var signs = booster.monotone.active_signs()
    if len(signs) == 0:
        return
    for t in range(len(booster.trees)):
        if not monotone_claim_holds(booster.trees[t], signs):
            raise Error(
                "tree ",
                t,
                " no longer satisfies the monotonic constraints the ensemble"
                " records; the model would claim a monotonicity it does not"
                " have",
            )


def check_monotone_claim_multiclass(booster: MulticlassBooster) raises:
    """The softmax counterpart. Constraints apply per class tree, so this is
    the same per-tree check over every tree; softmax probabilities were
    never claimed monotone (see monotone.mojo)."""
    var signs = booster.monotone.active_signs()
    if len(signs) == 0:
        return
    for t in range(len(booster.trees)):
        if not monotone_claim_holds(booster.trees[t], signs):
            raise Error(
                "class tree ",
                t,
                " no longer satisfies the monotonic constraints the ensemble"
                " records",
            )


# -- reading leaf outputs ------------------------------------------------


def leaf_node_index(tree: Tree, leaf_ordinal: Int) raises -> Int:
    """The node id of the leaf with ordinal `leaf_ordinal`.

    Leaf ordinals are mojotrees's own numbering (`Tree.leaf_ordinals`): a
    leaf's rank among the tree's leaves in node order. It is what
    `predict(pred_leaf=True)` reports and what every entry point here
    addresses by, because node ids are an implementation detail. It is **not**
    LightGBM's leaf id, and the two agree only by coincidence.
    """
    if leaf_ordinal < 0:
        raise Error("leaf ordinal must not be negative; got ", leaf_ordinal)
    var seen = 0
    for i in range(len(tree.feature)):
        if tree.feature[i] >= 0:
            continue
        if seen == leaf_ordinal:
            return i
        seen += 1
    raise Error(
        "leaf ordinal ",
        leaf_ordinal,
        " is outside a tree with ",
        seen,
        " leaves",
    )


def leaf_outputs(tree: Tree) -> List[Float64]:
    """Every leaf's stored value, in leaf-ordinal order.

    These are **unshrunk**: mojotrees stores the value a leaf was fitted at
    and multiplies by the ensemble's `learning_rate` when it predicts (the
    dump reports the rate as `shrinkage` and says the values are unshrunk).
    LightGBM folds shrinkage into the stored leaf value, so a number read
    here is the LightGBM number divided by the learning rate. Use
    `leaf_outputs_shrunk` for the contribution scale.
    """
    var out = List[Float64](capacity=tree.n_leaves)
    for i in range(len(tree.feature)):
        if tree.feature[i] < 0:
            out.append(tree.value[i])
    return out^


def leaf_outputs_shrunk(
    tree: Tree, learning_rate: Float64
) -> List[Float64]:
    """Every leaf's contribution to a raw score, in leaf-ordinal order:
    `learning_rate * value`. This is the scale LightGBM stores leaf values
    at, so it is what a LightGBM-shaped consumer wants."""
    var out = List[Float64](capacity=tree.n_leaves)
    for i in range(len(tree.feature)):
        if tree.feature[i] < 0:
            out.append(learning_rate * tree.value[i])
    return out^


def _check_tree_index(n_trees: Int, tree_index: Int) raises:
    if tree_index < 0 or tree_index >= n_trees:
        raise Error(
            "tree ",
            tree_index,
            " is outside an ensemble of ",
            n_trees,
            " trees",
        )


def get_leaf_output(
    booster: Booster, tree_index: Int, leaf_ordinal: Int
) raises -> Float64:
    """One leaf's stored, unshrunk value. LightGBM's `get_leaf_output`, on
    mojotrees's leaf numbering and mojotrees's value scale (see
    `leaf_outputs` for both differences)."""
    _check_tree_index(len(booster.trees), tree_index)
    ref tree = booster.trees[tree_index]
    return tree.value[leaf_node_index(tree, leaf_ordinal)]


def get_leaf_output_shrunk(
    booster: Booster, tree_index: Int, leaf_ordinal: Int
) raises -> Float64:
    """One leaf's contribution to a raw score: `learning_rate * value`."""
    return booster.learning_rate * get_leaf_output(
        booster, tree_index, leaf_ordinal
    )


def multiclass_tree_index(
    booster: MulticlassBooster, iteration: Int, class_id: Int
) raises -> Int:
    """The flat index of class `class_id`'s tree in boosting round
    `iteration`. The ensemble is round-major, so this is
    `iteration * n_classes + class_id`; it is a function rather than the
    arithmetic spelled out at each call site because that indexing is the
    invariant every reordering here has to preserve."""
    if class_id < 0 or class_id >= booster.n_classes:
        raise Error(
            "class ",
            class_id,
            " is outside a ",
            booster.n_classes,
            "-class model",
        )
    var rounds = booster.n_iterations()
    if iteration < 0 or iteration >= rounds:
        raise Error(
            "iteration ",
            iteration,
            " is outside an ensemble of ",
            rounds,
            " iterations",
        )
    return iteration * booster.n_classes + class_id


def get_leaf_output_multiclass(
    booster: MulticlassBooster,
    iteration: Int,
    class_id: Int,
    leaf_ordinal: Int,
) raises -> Float64:
    """One leaf's stored, unshrunk value, addressed by (round, class, leaf)
    rather than by a flat tree index, so a caller cannot reach the wrong
    class's tree by arithmetic."""
    var t = multiclass_tree_index(booster, iteration, class_id)
    ref tree = booster.trees[t]
    return tree.value[leaf_node_index(tree, leaf_ordinal)]


# -- writing leaf outputs ------------------------------------------------


def _write_leaf(
    mut tree: Tree,
    node: Int,
    value: Float64,
    signs: List[Int],
    policy: Int,
) raises -> Float64:
    """Write one leaf and leave the tree consistent, or leave it untouched.

    The two-step check the module docstring describes: clamp into the
    interval the leaf has *before* the write, then re-derive the whole
    interval chain and verify every leaf, because moving one leaf moves its
    sibling subtree's intervals. A tree that fails the second check is
    restored and the write is refused; there is no state in which a partial
    edit is visible.
    """
    if policy != LEAF_EDIT_CLAMP and policy != LEAF_EDIT_REJECT:
        # Checked up front rather than where a clamp would happen: a bad
        # policy on an unconstrained model, or on a value that happens to
        # land inside its interval, would otherwise pass unnoticed and be
        # discovered only by the one write that needed it.
        raise Error(
            "unknown leaf edit policy ",
            policy,
            "; use LEAF_EDIT_CLAMP or LEAF_EDIT_REJECT",
        )
    if tree.feature[node] >= 0:
        raise Error(
            "node ",
            node,
            " is an internal node. Its value is the value it held when it"
            " was created, which the monotone interval chain and the dump's"
            " `internal_value` are derived from; only leaves are writable",
        )
    check_finite(value, "leaf value")
    var stored = value
    if len(signs) > 0:
        var bounds = node_bounds(tree, signs)
        if len(bounds) > 0:
            var clamped = bounds[node].clamp(value)
            if clamped != value:
                if policy == LEAF_EDIT_REJECT:
                    raise Error(
                        "leaf value ",
                        value,
                        " lies outside node ",
                        node,
                        "'s monotone interval [",
                        bounds[node].lo,
                        ", ",
                        bounds[node].hi,
                        "]; the ensemble records monotonic constraints that"
                        " this write would falsify",
                    )
                stored = clamped
    var previous = tree.value[node]
    tree.value[node] = stored
    if len(signs) > 0 and not monotone_claim_holds(tree, signs):
        tree.value[node] = previous
        raise Error(
            "writing node ",
            node,
            " would break the tree's monotonic constraints through a"
            " sibling subtree: a leaf's value sets the midpoint its parent"
            " divides at, so the write was refused and the tree is"
            " unchanged",
        )
    return stored


def set_leaf_output(
    mut booster: Booster,
    tree_index: Int,
    leaf_ordinal: Int,
    value: Float64,
    policy: Int = LEAF_EDIT_CLAMP,
) raises -> Float64:
    """Set one leaf's stored, unshrunk value and return what was stored.

    LightGBM's `set_leaf_output`, with three differences a caller has to
    know:

    - the leaf is addressed by mojotrees's leaf ordinal, not LightGBM's leaf
      id (see `leaf_node_index`);
    - the value is unshrunk, so it is multiplied by the ensemble's
      `learning_rate` when it predicts (LightGBM's stored value is already
      shrunk);
    - a model that records monotonic constraints clamps the value into the
      leaf's interval by default, or refuses it with `LEAF_EDIT_REJECT`,
      because the alternative is a model that claims a monotonicity it does
      not have. The returned value is what was actually stored, which is the
      only way a caller learns a clamp happened.

    What the write leaves alone, and why that is right: routing (so every
    row still reaches this leaf), node covers (so exact feature
    contributions stay conditioned on the true training weights), internal
    node values, and split gains (which describe the fit that grew the
    tree, not its current outputs). `clear_split_gains` retracts the last of
    those explicitly if a caller wants the honest "no gains here" state.
    """
    _check_tree_index(len(booster.trees), tree_index)
    var signs = booster.monotone.active_signs()
    var node = leaf_node_index(booster.trees[tree_index], leaf_ordinal)
    return _write_leaf(booster.trees[tree_index], node, value, signs, policy)


def set_leaf_output_multiclass(
    mut booster: MulticlassBooster,
    iteration: Int,
    class_id: Int,
    leaf_ordinal: Int,
    value: Float64,
    policy: Int = LEAF_EDIT_CLAMP,
) raises -> Float64:
    """Set one leaf of class `class_id`'s tree in round `iteration`.

    Addressed by round and class rather than by a flat tree index: the
    round-major layout is what makes a tree belong to a class, and an
    off-by-one in a caller's own arithmetic would move a leaf value from one
    class's score to another's with nothing to catch it.
    """
    var t = multiclass_tree_index(booster, iteration, class_id)
    var signs = booster.monotone.active_signs()
    var node = leaf_node_index(booster.trees[t], leaf_ordinal)
    return _write_leaf(booster.trees[t], node, value, signs, policy)


def set_leaf_outputs(
    mut booster: Booster,
    tree_index: Int,
    values: List[Float64],
    policy: Int = LEAF_EDIT_CLAMP,
) raises -> List[Float64]:
    """Rewrite every leaf of one tree, in leaf-ordinal order, and return
    what was stored.

    A batch rather than a loop of single writes because the two differ under
    constraints: each write re-derives the interval chain from the values
    written so far, so a leaf clamped early can move a later leaf's
    interval. Writing the whole tree in ordinal order makes that sequence
    explicit and reproducible instead of a property of the caller's loop.
    """
    _check_tree_index(len(booster.trees), tree_index)
    if len(values) != booster.trees[tree_index].n_leaves:
        raise Error(
            "tree ",
            tree_index,
            " has ",
            booster.trees[tree_index].n_leaves,
            " leaves but ",
            len(values),
            " values were given",
        )
    var signs = booster.monotone.active_signs()
    var stored = List[Float64](capacity=len(values))
    var ordinal = 0
    for node in range(len(booster.trees[tree_index].feature)):
        if booster.trees[tree_index].feature[node] >= 0:
            continue
        var written = _write_leaf(
            booster.trees[tree_index], node, values[ordinal], signs, policy
        )
        stored.append(written)
        ordinal += 1
    return stored^


def clear_split_gains(mut tree: Tree):
    """Zero every recorded split gain on one tree.

    The explicit retraction of provenance. A gain says what a split earned
    from the gradient sums its node held at growth time; after a refit, or
    after enough leaf edits, that is still a true statement about the fit
    that grew the tree but no longer describes the model in hand. Zeroing
    them makes `has_split_gains` report `False`, which is the schema's way
    of saying "this model has no gains to report" rather than reporting a
    zero a consumer could read as a measurement.

    Prediction is unaffected: nothing routes or scores on a gain.
    """
    for i in range(len(tree.split_gain)):
        tree.split_gain[i] = 0.0


def clear_split_gains_booster(mut booster: Booster):
    """`clear_split_gains` over a whole single-output ensemble. Partial
    clearing would leave `has_split_gains` true while most gains read zero,
    which is the one state a consumer cannot interpret."""
    for t in range(len(booster.trees)):
        clear_split_gains(booster.trees[t])


def clear_split_gains_multiclass(mut booster: MulticlassBooster):
    """`clear_split_gains` over a whole softmax ensemble."""
    for t in range(len(booster.trees)):
        clear_split_gains(booster.trees[t])


# -- rollback ------------------------------------------------------------


def _check_mode(mode: Int) raises:
    if (
        mode != EDIT_MODE_GBDT
        and mode != EDIT_MODE_RF
        and mode != EDIT_MODE_DART
        and mode != EDIT_MODE_UNKNOWN
    ):
        raise Error("unknown boosting mode ", mode)


def _forest_rate(n_trees: Int) -> Float64:
    """The shrinkage a forest of `n_trees` is stored with, `1 / K`. Kept in
    step with `boosting_rf._forest_rate` by construction: an empty forest
    keeps 1.0, since the factor multiplies nothing."""
    if n_trees <= 0:
        return 1.0
    return 1.0 / Float64(n_trees)


def _reject_rollback_mode(mode: Int) raises:
    if mode == EDIT_MODE_DART:
        raise Error(
            "rollback is not defined for a DART model: a dropped iteration's"
            " trees were rescaled in place when later rounds added theirs, so"
            " the first n-1 trees are not the model DART would have produced"
            " at round n-1. LightGBM refuses this for the same reason"
        )
    if mode == EDIT_MODE_UNKNOWN:
        raise Error(
            "rollback needs the boosting mode: a fitted ensemble records"
            " none, and the arithmetic differs (a GBDT sum drops a tree, a"
            " forest average also rescales its rate, a DART model cannot be"
            " rolled back at all). Pass EDIT_MODE_GBDT or EDIT_MODE_RF"
        )


def rollback(
    mut booster: Booster, n_iterations: Int, mode: Int = EDIT_MODE_GBDT
) raises -> Int:
    """Drop the last `n_iterations` iterations and return how many remain.

    A single-output ensemble holds one tree per iteration, so this drops
    that many trees off the end. The base score is untouched: it belongs to
    iteration 0 and an ensemble rolled back to nothing predicts it alone,
    which is what a zero-iteration ensemble has always predicted.

    `mode` decides what happens to the shrinkage factor. GBDT keeps it: the
    ensemble is a sum, so the remaining trees contribute exactly what they
    did. A random forest is an average, so its rate is `1 / K` and dropping
    trees rescales it to `1 / (K - dropped)`, which changes every remaining
    tree's contribution -- that is the correct forest of the remaining
    trees, and it is a different operation from GBDT rollback, so it has to
    be asked for. DART is refused.

    Everything the ensemble still holds stays true: the trees that remain
    are unedited, their covers and gains describe the fits that grew them,
    and the monotone claim survives because dropping trees from a sum of
    monotone trees leaves a sum of monotone trees. What does not survive is
    the caller's early-stopping metadata and any `IterationRange` built
    against the old length; `clamp_range` is the check for the latter.
    """
    _check_mode(mode)
    _reject_rollback_mode(mode)
    if n_iterations < 0:
        raise Error(
            "rollback count must not be negative; got ", n_iterations
        )
    var held = len(booster.trees)
    if n_iterations > held:
        raise Error(
            "cannot roll back ",
            n_iterations,
            " iterations from an ensemble of ",
            held,
        )
    if mode == EDIT_MODE_RF and booster.learning_rate != _forest_rate(held):
        raise Error(
            "this ensemble does not carry a forest's 1/K shrinkage (",
            booster.learning_rate,
            " for ",
            held,
            " trees), so rolling it back as a forest would rescale trees"
            " that were never averaged. Use EDIT_MODE_GBDT if it is a"
            " gradient-boosted model",
        )
    for _ in range(n_iterations):
        _ = booster.trees.pop()
    if mode == EDIT_MODE_RF:
        booster.learning_rate = _forest_rate(len(booster.trees))
    return len(booster.trees)


def rollback_one_iter(
    mut booster: Booster, mode: Int = EDIT_MODE_GBDT
) raises -> Int:
    """Drop the last boosting iteration and return how many remain.

    LightGBM's `rollback_one_iter`. Rolling back an ensemble that holds no
    iterations is refused rather than treated as a no-op: a caller unwinding
    a loop is asking for a specific model, and silently handing back the
    base-score model would hide the miscount.
    """
    if len(booster.trees) == 0:
        raise Error("this ensemble holds no iterations to roll back")
    return rollback(booster, 1, mode)


def rollback_to(
    mut booster: Booster, n_iterations: Int, mode: Int = EDIT_MODE_GBDT
) raises -> Int:
    """Truncate to the first `n_iterations` iterations, the shape early
    stopping wants: it knows the round it chose, not how many to unwind."""
    if n_iterations < 0:
        raise Error("iteration count must not be negative")
    var held = len(booster.trees)
    if n_iterations > held:
        raise Error(
            "cannot truncate to ",
            n_iterations,
            " iterations: the ensemble holds ",
            held,
        )
    return rollback(booster, held - n_iterations, mode)


def rollback_multiclass(
    mut booster: MulticlassBooster,
    n_iterations: Int,
    mode: Int = EDIT_MODE_GBDT,
) raises -> Int:
    """Drop the last `n_iterations` softmax rounds and return how many
    remain. One round is one tree per class, so this drops whole blocks of
    `n_classes` trees off the end; dropping any other number would shift
    which class every remaining tree scores."""
    _check_mode(mode)
    _reject_rollback_mode(mode)
    if mode == EDIT_MODE_RF:
        raise Error(
            "random-forest rollback has no softmax path: the forest trainer"
            " is single-output, so a multiclass ensemble carrying a 1/K rate"
            " is not a forest"
        )
    if n_iterations < 0:
        raise Error(
            "rollback count must not be negative; got ", n_iterations
        )
    if len(booster.trees) % booster.n_classes != 0:
        raise Error(
            "softmax ensemble holds ",
            len(booster.trees),
            " trees, which is not a whole number of ",
            booster.n_classes,
            "-tree rounds; it cannot be rolled back safely",
        )
    var rounds = booster.n_iterations()
    if n_iterations > rounds:
        raise Error(
            "cannot roll back ",
            n_iterations,
            " rounds from an ensemble of ",
            rounds,
        )
    for _ in range(n_iterations * booster.n_classes):
        _ = booster.trees.pop()
    return booster.n_iterations()


def rollback_one_iter_multiclass(
    mut booster: MulticlassBooster, mode: Int = EDIT_MODE_GBDT
) raises -> Int:
    """Drop the last softmax round (one tree per class) and return how many
    remain."""
    if booster.n_iterations() == 0:
        raise Error("this ensemble holds no rounds to roll back")
    return rollback_multiclass(booster, 1, mode)


def rollback_to_multiclass(
    mut booster: MulticlassBooster,
    n_iterations: Int,
    mode: Int = EDIT_MODE_GBDT,
) raises -> Int:
    """Truncate a softmax ensemble to its first `n_iterations` rounds."""
    if n_iterations < 0:
        raise Error("iteration count must not be negative")
    var rounds = booster.n_iterations()
    if n_iterations > rounds:
        raise Error(
            "cannot truncate to ",
            n_iterations,
            " rounds: the ensemble holds ",
            rounds,
        )
    return rollback_multiclass(booster, rounds - n_iterations, mode)


def clamp_range(n_iterations: Int, rng: IterationRange) raises:
    """Raise unless `rng` still addresses iterations this ensemble holds.

    `IterationRange.slice` and `.clamp` build a range against a length and
    clamp out-of-range bounds silently, which is right at the prediction
    boundary (LightGBM clamps there too). It is wrong after an edit: a range
    built before a rollback names iterations that no longer exist, and
    quietly shortening it would return a prediction from a different model
    than the caller asked for.
    """
    if rng.start < 0 or rng.stop < rng.start:
        raise Error(
            "iteration range [", rng.start, ", ", rng.stop, ") is malformed"
        )
    if rng.stop > n_iterations:
        raise Error(
            "iteration range [",
            rng.start,
            ", ",
            rng.stop,
            ") reaches past an ensemble of ",
            n_iterations,
            " iterations; it was built against a longer model",
        )


# -- prediction bounds ---------------------------------------------------


@fieldwise_init
struct ScoreBounds(Copyable, Movable, Writable):
    """A closed interval a prediction is guaranteed to lie in.

    Sound, not tight. The interval is built by taking each tree's smallest
    and largest leaf value independently, and no single row need reach the
    extreme leaf of every tree at once, so the endpoints are attainable only
    by coincidence. That is exactly what LightGBM's `lower_bound` and
    `upper_bound` report, and it is the only bound obtainable without
    searching the joint feasible set of leaf combinations.
    """

    var lower: Float64
    var upper: Float64

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ScoreBounds(", self.lower, ", ", self.upper, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def width(self) -> Float64:
        return self.upper - self.lower


def tree_leaf_bounds(tree: Tree) raises -> ScoreBounds:
    """The smallest and largest **unshrunk** value any leaf of `tree`
    emits."""
    var lo = 0.0
    var hi = 0.0
    var seen = False
    for i in range(len(tree.feature)):
        if tree.feature[i] >= 0:
            continue
        if not seen:
            lo = tree.value[i]
            hi = tree.value[i]
            seen = True
            continue
        if tree.value[i] < lo:
            lo = tree.value[i]
        if tree.value[i] > hi:
            hi = tree.value[i]
    if not seen:
        raise Error("tree has no leaves")
    return ScoreBounds(lo, hi)


def raw_score_bounds_range(
    booster: Booster, rng: IterationRange
) raises -> ScoreBounds:
    """Bounds on the raw score of the boosting iterations in `rng` alone.

    Built exactly the way `predict_raw_bins_range` sums: the base score is
    added only when the range starts at 0, because it belongs to iteration 0
    (see `IterationRange`). So the bounds of `[0, k)` and `[k, n)` add up to
    a valid -- if looser -- bound on the full model, and the bounds of a full
    range are the bounds `raw_score_bounds` reports.
    """
    clamp_range(booster.n_iterations(), rng)
    var lo = booster.base_score if rng.includes_base() else 0.0
    var hi = lo
    for i in range(rng.start, rng.stop):
        var leaf = tree_leaf_bounds(booster.trees[i])
        lo += booster.learning_rate * leaf.lower
        hi += booster.learning_rate * leaf.upper
    return ScoreBounds(lo, hi)


def raw_score_bounds(booster: Booster) raises -> ScoreBounds:
    """Bounds on the raw score of the whole ensemble: LightGBM's
    `lower_bound` and `upper_bound` in one call.

    The base score is inside them. LightGBM folds its own base score into
    the first iteration's leaf values, so summing per-tree bounds includes
    it there too; mojotrees keeps it apart, so it is added here explicitly
    and the two agree. An ensemble with no trees has both bounds equal to
    the base score, which is the value it predicts.
    """
    return raw_score_bounds_range(
        booster,
        IterationRange.slice(booster.n_iterations(), 0, booster.n_iterations()),
    )


def response_bounds(booster: Booster) raises -> ScoreBounds:
    """Bounds on the response-scale prediction.

    Every inverse link the objective registry uses is nondecreasing -- the
    logistic sigmoid, `exp` for poisson, gamma, and tweedie, and the
    identity everywhere else -- so mapping the raw endpoints through
    `Booster.response` maps the interval to the interval. That is why the
    same call covers every objective and why a CUSTOM model returns its raw
    bounds: the framework does not know the caller's link, and
    `Booster.response` is the one place that decision is recorded.
    """
    var raw = raw_score_bounds(booster)
    return ScoreBounds(
        booster.response(raw.lower), booster.response(raw.upper)
    )


def raw_score_bounds_multiclass_range(
    booster: MulticlassBooster, rng: IterationRange
) raises -> List[ScoreBounds]:
    """Per-class bounds on the raw softmax scores of the iterations in
    `rng`, one entry per class.

    Per class, not one aggregate. Summing every tree's extremes across all
    classes -- which is what a literal reading of LightGBM's single-value
    bound would do on a multiclass model -- adds up numbers that never enter
    the same score, and the result bounds nothing. Class `k`'s score is the
    sum over rounds of `trees[i * n_classes + k]`, and that is what this
    bounds.
    """
    clamp_range(booster.n_iterations(), rng)
    var out = List[ScoreBounds](capacity=booster.n_classes)
    for k in range(booster.n_classes):
        var base = booster.base_scores[k] if rng.includes_base() else 0.0
        var lo = base
        var hi = base
        for i in range(rng.start, rng.stop):
            var leaf = tree_leaf_bounds(
                booster.trees[i * booster.n_classes + k]
            )
            lo += booster.learning_rate * leaf.lower
            hi += booster.learning_rate * leaf.upper
        out.append(ScoreBounds(lo, hi))
    return out^


def raw_score_bounds_multiclass(
    booster: MulticlassBooster,
) raises -> List[ScoreBounds]:
    """Per-class bounds on the raw softmax scores of the whole ensemble."""
    return raw_score_bounds_multiclass_range(
        booster,
        IterationRange.slice(
            booster.n_iterations(), 0, booster.n_iterations()
        ),
    )


def probability_bounds_multiclass(
    booster: MulticlassBooster,
) raises -> List[ScoreBounds]:
    """Per-class bounds on the softmax probabilities.

    A class's probability is not a function of its own raw score alone, so
    this cannot be the response map of the raw bounds. It is instead the
    range of the softmax over the box the per-class raw bounds describe:

        p_k = exp(z_k) / sum_j exp(z_j)

    is increasing in `z_k` and decreasing in every other `z_j`, so over a box
    it is largest at `z_k = hi_k` with every other class at its `lo`, and
    smallest at `z_k = lo_k` with every other class at its `hi`. Both
    endpoints are computed that way, with the exponentials shifted by their
    own maximum so a wide raw range does not overflow.

    The result is sound and loose twice over: the per-class raw bounds are
    themselves not attained, and the box is larger than the set of score
    vectors a single row can actually produce (one row feeds every class).
    It is a guarantee about what the model cannot output, not a description
    of what it does output.
    """
    var raw = raw_score_bounds_multiclass(booster)
    var k_n = booster.n_classes
    var out = List[ScoreBounds](capacity=k_n)
    for k in range(k_n):
        out.append(
            ScoreBounds(
                _softmax_at(raw, k, False), _softmax_at(raw, k, True)
            )
        )
    return out^


def _softmax_at(
    bounds: List[ScoreBounds], k: Int, upper: Bool
) -> Float64:
    """Class `k`'s softmax value at the corner of the box that maximizes it
    (`upper`) or minimizes it. See `probability_bounds_multiclass` for why
    those two corners are the extremes."""
    var n = len(bounds)
    var z = List[Float64](capacity=n)
    for j in range(n):
        if j == k:
            z.append(bounds[j].upper if upper else bounds[j].lower)
        else:
            z.append(bounds[j].lower if upper else bounds[j].upper)
    var m = z[0]
    for j in range(1, n):
        if z[j] > m:
            m = z[j]
    var total = 0.0
    for j in range(n):
        total += exp(z[j] - m)
    return exp(z[k] - m) / total


# -- reordering ----------------------------------------------------------


def _check_permutation(order: List[Int], start: Int, stop: Int) raises:
    """Raise unless `order` is a permutation of `[start, stop)`.

    Checked rather than assumed because the failure is silent: a repeated
    index duplicates a tree and drops another, which changes the model's
    predictions while leaving every structural check happy.
    """
    var n = stop - start
    if len(order) != n:
        raise Error(
            "order has ",
            len(order),
            " entries for a range of ",
            n,
            " iterations",
        )
    var seen = List[Bool](capacity=n)
    seen.resize(n, False)
    for i in range(n):
        var v = order[i]
        if v < start or v >= stop:
            raise Error(
                "order entry ",
                v,
                " is outside the range [",
                start,
                ", ",
                stop,
                ")",
            )
        if seen[v - start]:
            raise Error("order repeats iteration ", v)
        seen[v - start] = True


def permute_iterations(
    mut booster: Booster, order: List[Int], rng: IterationRange
) raises:
    """Reorder the iterations in `rng` to the explicit order given.

    The deterministic core `shuffle_iterations` draws for. Reordering is
    sound for a gradient-boosted ensemble because prediction sums the trees,
    and a sum does not care about order -- but only in exact arithmetic.
    Floating-point addition is not associative, so a reordered model's
    predictions can differ from the original's in the last ulp. Two facts
    that do not survive at all:

    - an iteration index no longer names the round that grew that tree, so
      `best_iteration` and any recorded per-iteration history are stale;
    - `predict(..., num_iteration=k)` no longer means "the first k rounds",
      because the first k slots no longer hold them.

    Neither is a model inconsistency, which is why this is offered; both are
    caller state, which is why the handoff carries the invalidation patch.
    """
    clamp_range(booster.n_iterations(), rng)
    _check_permutation(order, rng.start, rng.stop)
    var reordered = List[Tree](capacity=len(booster.trees))
    for i in range(rng.start):
        reordered.append(booster.trees[i].copy())
    for i in range(len(order)):
        reordered.append(booster.trees[order[i]].copy())
    for i in range(rng.stop, len(booster.trees)):
        reordered.append(booster.trees[i].copy())
    booster.trees = reordered^


def _shuffled_order(start: Int, stop: Int, seed: Int) -> List[Int]:
    """A seeded permutation of `[start, stop)`, by Fisher-Yates over the
    same splitmix64 stream the row and feature samplers use, in its own
    domain so a shuffle seed cannot reproduce a bagging draw."""
    var order = List[Int](capacity=stop - start)
    for i in range(start, stop):
        order.append(i)
    var stream = _splitmix64(
        UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _SHUFFLE_DOMAIN
    )
    var n = len(order)
    for i in range(n - 1, 0, -1):
        var j = Int(_uniform(stream + UInt64(i)) * Float64(i + 1))
        if j > i:
            j = i
        var tmp = order[i]
        order[i] = order[j]
        order[j] = tmp
    return order^


def shuffle_iterations(
    mut booster: Booster, seed: Int, rng: IterationRange
) raises:
    """Shuffle the iterations in `rng`, LightGBM's `shuffle_models`.

    Seeded and reproducible: the same seed and the same range give the same
    order on any platform, because the draw is the same counter-based
    splitmix64 stream the samplers use. See `permute_iterations` for what
    the reordering costs.
    """
    clamp_range(booster.n_iterations(), rng)
    if rng.n_iterations() < 2:
        return
    permute_iterations(
        booster, _shuffled_order(rng.start, rng.stop, seed), rng
    )


def permute_iterations_multiclass(
    mut booster: MulticlassBooster, order: List[Int], rng: IterationRange
) raises:
    """Reorder whole softmax rounds, keeping each round's per-class trees
    together and in class order.

    This is the operation, not an optimization of it. `trees[i * n_classes
    + k]` is what makes a tree class `k`'s; a permutation that moved
    individual trees would leave a structurally valid ensemble in which
    every class scores some other class's trees, and nothing downstream
    could detect it.
    """
    clamp_range(booster.n_iterations(), rng)
    _check_permutation(order, rng.start, rng.stop)
    var k_n = booster.n_classes
    var reordered = List[Tree](capacity=len(booster.trees))
    for i in range(rng.start * k_n):
        reordered.append(booster.trees[i].copy())
    for i in range(len(order)):
        for k in range(k_n):
            reordered.append(booster.trees[order[i] * k_n + k].copy())
    for i in range(rng.stop * k_n, len(booster.trees)):
        reordered.append(booster.trees[i].copy())
    booster.trees = reordered^


def shuffle_iterations_multiclass(
    mut booster: MulticlassBooster, seed: Int, rng: IterationRange
) raises:
    """Shuffle whole softmax rounds. The multiclass `shuffle_models`."""
    clamp_range(booster.n_iterations(), rng)
    if rng.n_iterations() < 2:
        return
    permute_iterations_multiclass(
        booster, _shuffled_order(rng.start, rng.stop, seed), rng
    )


# -- refit ---------------------------------------------------------------


@fieldwise_init
struct RefitParams(Copyable, Movable):
    """How a refit rebuilds leaf values.

    `tree` is the ordinary tree-parameter bundle, and refit reads exactly
    the parts of it that describe a **leaf**: `lambda_reg`, `lambda_l1`, and
    `extra.max_delta_step` / `extra.path_smooth`. The growth budget
    (`num_leaves`, `max_depth`, the samplers, the split rules) is ignored
    on purpose -- refit never grows or prunes anything, so a parameter that
    only shapes which splits are taken has nothing to act on. Reusing
    `TreeParams` rather than declaring a second bundle keeps the leaf
    formula identical to the one the grower used.

    `decay_rate` is LightGBM's `refit_decay_rate`: the new leaf value is
    `decay_rate * old + (1 - decay_rate) * fresh`. At 1.0 the model does not
    move, at 0.0 it is refit outright, and LightGBM's default is 0.9. Both
    values live on the unshrunk scale, and shrinkage is a single positive
    factor shared by every tree, so blending unshrunk values is the same
    blend LightGBM performs on shrunk ones.

    `min_leaf_rows` leaves a leaf at its old value when fewer than that many
    refit rows reach it. LightGBM applies no such floor; the default of 1
    here only expresses that a leaf **no** row reaches has nothing to be refit
    from, and raising it trades faithfulness to the new data for stability.

    `recount` recomputes `Tree.count` from the refit data. It is
    all-or-nothing per tree: covers from two datasets in one tree would make
    exact feature contributions condition on a background that never
    existed, and a single zero cover would produce a model the reader
    refuses. A tree in which some node draws no refit row therefore keeps
    every one of its original covers, and `RefitReport` says so.
    """

    var decay_rate: Float64
    var tree: TreeParams
    var min_leaf_rows: Int
    var recount: Bool

    @staticmethod
    def default() -> RefitParams:
        """LightGBM's `refit_decay_rate` of 0.9, default tree parameters,
        covers recomputed, and no floor beyond "a leaf needs a row"."""
        return RefitParams(0.9, TreeParams.default(), 1, True)

    def check(self) raises:
        if not (self.decay_rate >= 0.0 and self.decay_rate <= 1.0):
            raise Error(
                "refit decay_rate must be in [0, 1]; got ", self.decay_rate
            )
        if self.min_leaf_rows < 0:
            raise Error("refit min_leaf_rows must not be negative")


@fieldwise_init
struct RefitReport(Copyable, Movable, Writable):
    """What a refit actually did.

    Returned rather than logged because two of these are decisions the
    caller has to be able to see: a leaf left at its old value did not
    follow the new data, and a tree whose covers were not recomputed still
    describes the original training rows.
    """

    var n_trees: Int
    var n_leaves_updated: Int
    var n_leaves_kept: Int
    var n_leaves_clamped: Int
    var n_trees_recounted: Int

    @staticmethod
    def empty() -> RefitReport:
        return RefitReport(0, 0, 0, 0, 0)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "RefitReport(trees=",
            self.n_trees,
            ", leaves_updated=",
            self.n_leaves_updated,
            ", leaves_kept=",
            self.n_leaves_kept,
            ", leaves_clamped=",
            self.n_leaves_clamped,
            ", trees_recounted=",
            self.n_trees_recounted,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _check_refit_objective(objective: Int) raises:
    """Refuse the two objectives whose gradients a refit cannot produce.

    CUSTOM's gradients come from a caller-supplied callable that the fitted
    ensemble does not carry, and LAMBDARANK's come from query groups and a
    within-group ranking that a plain (data, target) pair does not describe.
    Both would have to guess, and a guessed gradient produces leaf values
    that look ordinary and are wrong.
    """
    if objective == CUSTOM:
        raise Error(
            "refit is not available for a custom-objective model: its"
            " gradients come from a caller-supplied callable that the fitted"
            " ensemble does not carry"
        )
    if objective == LAMBDARANK:
        raise Error(
            "refit is not available for a ranking model: LambdaRank"
            " gradients are computed within a query group, which a (data,"
            " target) pair does not describe"
        )


def _parent_outputs(tree: Tree) -> List[Float64]:
    """Each node's parent's stored value, 0.0 at the root.

    `path_smooth` shrinks a leaf toward its parent's finished output, and
    refit rewrites leaves only, so every parent is an internal node still
    holding the value it was grown with. Built once per tree, before any
    leaf moves, which is also what makes it order-independent.
    """
    var n_nodes = len(tree.feature)
    var out = List[Float64](capacity=n_nodes)
    out.resize(n_nodes, 0.0)
    for node in range(n_nodes):
        if tree.feature[node] < 0:
            continue
        out[tree.left[node]] = tree.value[node]
        out[tree.right[node]] = tree.value[node]
    return out^


def _refit_tree_newton(
    mut tree: Tree,
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: RefitParams,
    signs: List[Int],
    mut report: RefitReport,
) raises:
    """Rebuild one tree's leaf values from gradients on the refit data.

    The leaf rule is the grower's own -- the L1-soft-thresholded Newton step,
    then `max_delta_step` and `path_smooth` through `finish_leaf_output` --
    applied to the gradient and hessian sums of whatever refit rows reach
    each leaf. Routing is untouched, so "whatever reaches each leaf" is
    decided by the tree the model already has.

    The monotone clamp is free of the drift `set_leaf_output` has to guard
    against: the intervals are derived once, before any leaf moves, from the
    internal node values that a refit never writes, and both the old value
    (which lay in its interval) and the clamped fresh value lie in the
    interval, so their convex combination does too.
    """
    var n_nodes = len(tree.feature)
    var g_sum = List[Float64](capacity=n_nodes)
    var h_sum = List[Float64](capacity=n_nodes)
    var rows_in = List[Int](capacity=n_nodes)
    var visits = List[Float64](capacity=n_nodes)
    g_sum.resize(n_nodes, 0.0)
    h_sum.resize(n_nodes, 0.0)
    rows_in.resize(n_nodes, 0)
    visits.resize(n_nodes, 0.0)

    for r in range(data.n_rows):
        var node = 0
        visits[0] += 1.0
        while tree.feature[node] >= 0:
            if tree.goes_left(node, data.bin_at(r, tree.feature[node])):
                node = tree.left[node]
            else:
                node = tree.right[node]
            visits[node] += 1.0
        g_sum[node] += grad[r]
        h_sum[node] += hess[r]
        rows_in[node] += 1

    var bounds = node_bounds(tree, signs)
    var parents = _parent_outputs(tree)
    var finish = params.tree.extra.needs_leaf_finish()
    var decay = params.decay_rate
    var floor = params.min_leaf_rows
    for node in range(n_nodes):
        if tree.feature[node] >= 0:
            continue
        if rows_in[node] < floor or rows_in[node] == 0:
            report.n_leaves_kept += 1
            continue
        var fresh = -soft_threshold_l1(
            g_sum[node], params.tree.lambda_l1
        ) / (h_sum[node] + params.tree.lambda_reg)
        if finish:
            fresh = finish_leaf_output(
                fresh,
                params.tree.extra.max_delta_step,
                params.tree.extra.path_smooth,
                rows_in[node],
                parents[node],
            )
        var blended = decay * tree.value[node] + (1.0 - decay) * fresh
        if len(bounds) > 0:
            var clamped = bounds[node].clamp(blended)
            if clamped != blended:
                report.n_leaves_clamped += 1
            blended = clamped
        check_finite(blended, "refit leaf value")
        tree.value[node] = blended
        report.n_leaves_updated += 1

    _apply_recount(tree, visits, params, report)


def _refit_tree_renewed(
    mut tree: Tree,
    data: BinnedMatrix,
    target: List[Float64],
    raw: List[Float64],
    renew_weights: List[Float64],
    renew_a: Float64,
    params: RefitParams,
    signs: List[Int],
    mut report: RefitReport,
) raises:
    """Rebuild one tree's leaf values for an objective whose leaves are
    renewed rather than fitted by Newton step (QUANTILE, L1, MAPE).

    Those objectives replace a grown leaf's value with a percentile of the
    residuals of the rows in it, and refitting them by the Newton rule would
    fit a different model from the one training produced. This calls the
    trainer's own `_renew_leaf_values`, then blends the result with the old
    value at the decay rate. Blending after the renewal's own clamp is
    sound: both endpoints already lie in the leaf's monotone interval, so
    every convex combination does too, and the claim survives without a
    second clamp.

    Leaves that no refit row reaches are left alone by the renewal itself,
    so blending them is the identity; they are counted as kept.
    """
    var before = leaf_outputs(tree)
    _renew_leaf_values(
        tree,
        data,
        target,
        raw,
        renew_weights,
        renew_a,
        List[Int](),
        signs,
        params.tree.extra,
    )
    var decay = params.decay_rate
    var ordinal = 0
    var visits = List[Float64](capacity=len(tree.feature))
    visits.resize(len(tree.feature), 0.0)
    for r in range(data.n_rows):
        var node = 0
        visits[0] += 1.0
        while tree.feature[node] >= 0:
            if tree.goes_left(node, data.bin_at(r, tree.feature[node])):
                node = tree.left[node]
            else:
                node = tree.right[node]
            visits[node] += 1.0
    for node in range(len(tree.feature)):
        if tree.feature[node] >= 0:
            continue
        var renewed = tree.value[node]
        var blended = decay * before[ordinal] + (1.0 - decay) * renewed
        check_finite(blended, "refit leaf value")
        tree.value[node] = blended
        if renewed == before[ordinal]:
            report.n_leaves_kept += 1
        else:
            report.n_leaves_updated += 1
        ordinal += 1
    _apply_recount(tree, visits, params, report)


def _apply_recount(
    mut tree: Tree,
    visits: List[Float64],
    params: RefitParams,
    mut report: RefitReport,
):
    """Replace a tree's node covers with the refit row counts, or leave
    every one of them alone.

    All-or-nothing, for two reasons stated in `RefitParams`: mixed covers
    describe a background dataset that never existed, and a single zero
    cover produces a model the reader refuses to load.
    """
    if not params.recount:
        return
    for i in range(len(visits)):
        if not visits[i] > 0.0:
            return
    for i in range(len(visits)):
        tree.count[i] = visits[i]
    report.n_trees_recounted += 1


def refit(
    mut booster: Booster,
    data: BinnedMatrix,
    target: List[Float64],
    params: RefitParams = RefitParams.default(),
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    init_score: List[Float64] = [],
) raises -> RefitReport:
    """Rebuild every leaf value from new data, keeping every tree's shape.

    LightGBM's `Booster.refit`. The ensemble is walked in training order:
    at tree `t` the gradients are taken at the raw scores the already-refit
    trees `0..t-1` produce on the refit data, tree `t`'s leaves are rebuilt
    from the rows that reach them, and the raw scores are advanced by the
    new tree. That is the sequence training itself follows, which is what
    makes a refit at `decay_rate = 0` the model this structure would have
    been fitted with.

    What it changes: leaf values, and (all-or-nothing per tree) node covers.
    What it does not: routing, internal node values, split gains, the base
    score, the learning rate, the objective, the monotone claim, the number
    of iterations, or the class-to-tree mapping. Split gains keep describing
    the fit that grew the trees, as they do in LightGBM;
    `clear_split_gains_booster` retracts them if that provenance is not
    wanted.

    `init_score` is the offset the ensemble was trained under, if any. It is
    training state that the model does not carry -- a booster trained from an
    init score records a base score of 0 -- so a refit has to be handed it
    again for the same reason `train_more` does.
    """
    params.check()
    _check_refit_objective(booster.objective)
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if len(init_score) != 0 and len(init_score) != data.n_rows:
        raise Error("init_score length must equal n_rows")
    _check_sample_weight(sample_weight, data.n_rows)
    if not _same_signs(params.tree.monotone.signs, booster.monotone.signs):
        raise Error(
            "refit cannot change monotone_constraints: the ensemble records"
            " the constraints all of its trees satisfy"
        )
    for t in range(len(booster.trees)):
        check_tree_structure(booster.trees[t])
    # Checked before as well as after: the leaf blend is only guaranteed to
    # stay inside a leaf's monotone interval because both endpoints already
    # are, and the old value is one of them.
    check_monotone_claim(booster)

    var n = data.n_rows
    var has_init = len(init_score) == n
    var raw = List[Float64](capacity=n)
    for r in range(n):
        var s = booster.base_score
        if has_init:
            s += init_score[r]
        raw.append(s)

    var signs = booster.monotone.active_signs()
    var renews = objective_renews_leaves(booster.objective)
    var renew_w = renewal_weights(booster.objective, target, sample_weight)
    var renew_a = renewal_alpha(booster.objective, alpha)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var report = RefitReport.empty()
    report.n_trees = len(booster.trees)

    for t in range(len(booster.trees)):
        if renews:
            _refit_tree_renewed(
                booster.trees[t],
                data,
                target,
                raw,
                renew_w,
                renew_a,
                params,
                signs,
                report,
            )
        else:
            fill_grad_hess(
                raw,
                target,
                booster.objective,
                sample_weight,
                alpha,
                grad,
                hess,
            )
            _refit_tree_newton(
                booster.trees[t], data, grad, hess, params, signs, report
            )
        for r in range(n):
            raw[r] += booster.learning_rate * booster.trees[t].predict_row(
                data, r
            )

    check_monotone_claim(booster)
    return report^


def refit_multiclass(
    mut booster: MulticlassBooster,
    data: BinnedMatrix,
    labels: List[Int],
    params: RefitParams = RefitParams.default(),
    sample_weight: List[Float64] = [],
) raises -> RefitReport:
    """Rebuild every leaf value of a softmax ensemble from new data.

    The round structure is the trainer's: the class probabilities are taken
    once per round from the current raw scores, then each class's tree is
    refit from its own one-vs-rest gradients and that class's raw scores are
    advanced. Refitting a class from probabilities recomputed after its
    sibling classes had already moved would fit a round that training never
    performed.

    Softmax has no leaf-renewal rule and no `alpha`, and `init_score` is not
    a multiclass concept (one offset per row cannot say what each class
    starts from), so neither appears here.
    """
    params.check()
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    _check_sample_weight(sample_weight, data.n_rows)
    if not _same_signs(params.tree.monotone.signs, booster.monotone.signs):
        raise Error(
            "refit cannot change monotone_constraints: the ensemble records"
            " the constraints all of its trees satisfy"
        )
    if len(booster.trees) % booster.n_classes != 0:
        raise Error(
            "softmax ensemble holds ",
            len(booster.trees),
            " trees, which is not a whole number of ",
            booster.n_classes,
            "-tree rounds",
        )
    for t in range(len(booster.trees)):
        check_tree_structure(booster.trees[t])
    check_monotone_claim_multiclass(booster)
    for r in range(len(labels)):
        if labels[r] < 0 or labels[r] >= booster.n_classes:
            raise Error(
                "label ",
                labels[r],
                " is outside a ",
                booster.n_classes,
                "-class model",
            )

    var n = data.n_rows
    var k_n = booster.n_classes
    var raw = List[Float64](capacity=n * k_n)
    for r in range(n):
        for k in range(k_n):
            raw.append(booster.base_scores[k])
    var prob = List[Float64](capacity=n * k_n)
    prob.resize(n * k_n, 0.0)

    var signs = booster.monotone.active_signs()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var report = RefitReport.empty()
    report.n_trees = len(booster.trees)

    for i in range(booster.n_iterations()):
        for r in range(n):
            for k in range(k_n):
                prob[r * k_n + k] = raw[r * k_n + k]
            _softmax_inplace(prob, r * k_n, k_n)
        for k in range(k_n):
            _fill_softmax_grad_hess(
                prob, labels, k, k_n, sample_weight, grad, hess
            )
            var t = i * k_n + k
            _refit_tree_newton(
                booster.trees[t], data, grad, hess, params, signs, report
            )
            for r in range(n):
                raw[r * k_n + k] += (
                    booster.learning_rate
                    * booster.trees[t].predict_row(data, r)
                )

    check_monotone_claim_multiclass(booster)
    return report^


def _check_refit_dataset(mapper: BinMapper, dataset: Dataset) raises:
    """The two facts a refit dataset has to satisfy before any leaf moves.

    The binning check is `update_dataset`'s, and for the same reason: a bin
    index has to mean to the refit rows what it meant to the rows the trees
    were grown on, or every row routes by a different rule than the tree
    encodes. The sparse refusal is likewise the continued-training one: the
    refit walk reads a dense binned matrix, and densifying is what the
    sparse path exists to avoid.
    """
    if dataset.is_sparse:
        raise Error(
            "refit has no sparse path: it walks a dense binned matrix, and"
            " densifying one is what the sparse representation exists to"
            " avoid"
        )
    if not mapper.matches(dataset.mapper):
        raise Error(
            "refit needs data binned by the mapper the model was trained"
            " under: this dataset is binned differently"
        )
    if not dataset.has_label():
        raise Error("refit needs a labelled dataset")


def refit_dataset(
    mut model: Model,
    dataset: Dataset,
    params: RefitParams = RefitParams.default(),
    alpha: Float64 = 0.9,
) raises -> RefitReport:
    """Refit a fitted single-output model from a `Dataset`.

    The production entry point: the label, the weights, and the init score
    come off the dataset rather than being passed again, which is the same
    contract `update_dataset` follows, and the binning is checked against
    the model's own mapper before anything is written.
    """
    _check_refit_dataset(model.mapper, dataset)
    return refit(
        model.booster,
        dataset.data,
        dataset.label,
        params,
        dataset.weight,
        alpha,
        dataset.init_score,
    )


def refit_dataset_multiclass(
    mut model: MulticlassModel,
    dataset: Dataset,
    params: RefitParams = RefitParams.default(),
) raises -> RefitReport:
    """Refit a fitted softmax model from a `Dataset`. The class count is the
    model's, and the dataset's labels are checked against it: a dataset
    whose labels have outgrown the ensemble's classes has no tree sequence
    to be refit into."""
    _check_refit_dataset(model.mapper, dataset)
    if len(dataset.init_score) != 0:
        raise Error(
            "init_score is not a multiclass concept: one offset per row"
            " cannot say what each class starts from"
        )
    return refit_multiclass(
        model.booster,
        dataset.data,
        _int_labels(dataset.label, model.booster.n_classes),
        params,
        dataset.weight,
    )


# -- what is offered, and what is refused --------------------------------


@fieldwise_init
struct EditingCapability(Copyable, Movable, Writable):
    """One editing operation and whether this build performs it.

    A status rather than a missing function: a consumer asking "can I do
    this here?" gets an answer it can branch on, and a refusal carries the
    reason it is a refusal rather than a gap.
    """

    var operation: String
    var supported: Bool
    var reason: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "EditingCapability(",
            self.operation,
            ", supported=",
            "true" if self.supported else "false",
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def editing_capabilities() -> List[EditingCapability]:
    """Every editing operation this build has an answer for.

    The single table. `MODEL_EDITING_SUPPORTED` says editing exists at all;
    this says which operations that covers, and each refusal names the
    invariant that makes it a refusal rather than an omission. The JSON
    renderer and the Python mirror both read this, so a consumer cannot be
    told two different things.
    """
    var out = List[EditingCapability]()
    out.append(
        EditingCapability(
            String("rollback_one_iter"),
            True,
            String(
                "drops whole iterations off the end; the base score belongs"
                " to iteration 0 and is untouched. A forest rescales its 1/K"
                " rate, and DART is refused"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("lower_bound"),
            True,
            String(
                "base score plus the shrunk sum of each tree's smallest leaf;"
                " sound, not attained. Per class for a softmax model"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("upper_bound"),
            True,
            String(
                "base score plus the shrunk sum of each tree's largest leaf;"
                " sound, not attained. Per class for a softmax model"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("get_leaf_output"),
            True,
            String(
                "the stored, unshrunk value, addressed by mojotrees's leaf"
                " ordinal rather than LightGBM's leaf id"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("set_leaf_output"),
            True,
            String(
                "routing, covers, internal values, and gains all survive a"
                " leaf write unchanged and still true; a model recording"
                " monotonic constraints clamps or refuses the value"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("shuffle_models"),
            True,
            String(
                "permutes whole iterations, and whole per-class blocks for a"
                " softmax model; prediction is a sum, so only the last ulp"
                " and the caller's iteration metadata move"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("refit"),
            True,
            String(
                "rebuilds leaf values, and covers all-or-nothing, from new"
                " data on the existing structure; refused for custom"
                " objectives and for ranking, whose gradients the ensemble"
                " does not carry"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("set_internal_value"),
            False,
            String(
                "an internal node's value is the value it held when it was"
                " created; the monotone interval chain and the dump's"
                " internal_value are derived from it, and nothing could tell"
                " an edit from a corruption"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("set_split_feature"),
            False,
            String(
                "changing routing falsifies every node cover below the node,"
                " which exact feature contributions condition on, and no"
                " recomputation is possible without the training rows"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("set_threshold"),
            False,
            String(
                "same as set_split_feature: routing decides which rows"
                " reached a node, and the covers that record it cannot be"
                " rebuilt from the model"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("set_split_gain"),
            False,
            String(
                "a gain is what a split earned from gradient sums the tree no"
                " longer holds; it can be retracted with clear_split_gains,"
                " not rewritten"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("set_node_count"),
            False,
            String(
                "a cover is a training row count, not a knob; refit"
                " recomputes covers from data, all-or-nothing per tree"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("add_tree"),
            False,
            String(
                "an appended tree has no cover, no gain, and no fitted leaf"
                " values; continued training through train_more is the"
                " supported way to add iterations"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("set_learning_rate"),
            False,
            String(
                "one rate shrinks every tree, so changing it rescales the"
                " whole ensemble at once; the only rate change offered is the"
                " 1/K rescale a forest rollback performs"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("set_objective"),
            False,
            String(
                "the objective selects the inverse link and the gradient"
                " family the leaf values were fitted under; changing it"
                " leaves every leaf value meaningless"
            ),
        )
    )
    out.append(
        EditingCapability(
            String("set_base_score"),
            False,
            String(
                "the base score is the objective's own starting point for the"
                " fit the trees corrected; shifting it afterwards moves every"
                " prediction by a constant the trees never saw"
            ),
        )
    )
    return out^


def editing_capability(operation: String) raises -> EditingCapability:
    """One operation's entry from `editing_capabilities`, by name. Raises
    for a name the table does not carry, so a typo is not read as an
    unsupported operation."""
    var table = editing_capabilities()
    for i in range(len(table)):
        if table[i].operation == operation:
            return table[i].copy()
    raise Error(
        "'",
        operation,
        "' is not a model editing operation this build has an answer for",
    )


def check_editing_supported(operation: String) raises:
    """Raise unless `operation` is one this build performs, with the reason
    it does not. The refusal a binding calls before it dispatches, so an
    unsupported operation fails the same way whatever reaches it."""
    var entry = editing_capability(operation)
    if not entry.supported:
        raise Error(
            "model editing operation '",
            operation,
            "' is not supported: ",
            entry.reason,
        )


def _json_string(s: String) -> String:
    """A quoted JSON string. Reasons are this module's own text, but the
    escaping is done rather than assumed, so a reason can be reworded
    without breaking the renderer."""
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


def model_editing_status_json() -> String:
    """Whether a fitted model can be edited here, and operation by
    operation, what that covers.

    One rendering of `editing_capabilities`, so the C ABI, the bindings, and
    the Python mirror all read the same table. This supersedes the hardcoded
    "not supported" status that `inspection.mojo` carried while there was
    nothing to report.
    """
    var table = editing_capabilities()
    var out = String("{\"supported\":")
    out += "true" if MODEL_EDITING_SUPPORTED else "false"
    out += ",\"leaf_index\":\"ordinal\""
    out += ",\"leaf_value_is_shrunk\":false"
    out += ",\"operations\":["
    for i in range(len(table)):
        if i > 0:
            out += ","
        out += "{\"operation\":" + _json_string(table[i].operation)
        out += ",\"supported\":"
        out += "true" if table[i].supported else "false"
        out += ",\"reason\":" + _json_string(table[i].reason)
        out += "}"
    out += "]}"
    return out^
