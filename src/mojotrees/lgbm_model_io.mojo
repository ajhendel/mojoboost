"""LightGBM text model interop: read and write LightGBM's own model format.

EXPERIMENTAL. Everything public here is quarantined behind that word until
differential fixtures against a real LightGBM build have been run; see
`LGBM_INTEROP_STATUS` and the "Experimental status" section below. Nothing
in this module has been checked against a file LightGBM actually wrote.

**This is not mojotrees's model format.** `serialize.mojo` owns persistence:
its versioned text stores floats as raw bit patterns, so a native round trip
is bit-exact and lossless, and it is the only format `save_model` /
`load_model` speak. LightGBM's format is a decimal text interchange with
strictly less in it (no split gains in the shapes mojotrees keeps, no
hessian sums, no category-to-bin table), so it is an *import and export*
path and never a persistence one. `import_lgbm_file` therefore lands its
result in the native format: the conversion's product is a mojotrees model
file, not a LightGBM one.

This module converts between LightGBM's line-oriented text model (the string
`Booster.save_model` writes and `Booster.model_str` returns) and mojotrees's
`Model` / `MulticlassModel`. It reads `binning.mojo`, `categorical.mojo`,
`model.mojo`, `objective_registry.mojo`, `serialize.mojo`, and `tree.mojo`
but changes nothing in them: everything here is expressed in terms of the
public `BinMapper`, `CategoricalSpec`, `Tree`, `Booster`, and
`MulticlassBooster` structures, and every objective fact comes from the
registry rather than from a second table kept here.

Why a bin mapper has to be synthesized
--------------------------------------
A mojotrees tree routes by bin id (`Tree.threshold_bin`), a LightGBM tree by
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
0 = None, 1 = Zero, 2 = NaN. mojotrees reserves a per-feature missing bin
instead and routes any row landing in it by the node's `default_left`.
`None` and `NaN` map across exactly; `Zero` (where a zero feature value is
also treated as missing) has no representation in a bin-routed tree and is
rejected. Nodes splitting the same feature must agree on `missing_type`,
since mojotrees's reservation is a property of the feature.

Base score and shrinkage
------------------------
LightGBM folds both the learning rate and the boost-from-average init score
into the stored leaf values: predicting is a plain sum over trees. mojotrees
keeps them separate as `base_score` and `learning_rate`. So a model read from
a LightGBM file gets `learning_rate = 1.0` and `base_score = 0.0` with the
leaf values taken verbatim, and a model written to one gets every leaf value
multiplied by `learning_rate`, with `base_score` added into the first
iteration's trees the way LightGBM's `Tree::AddBias` does.

How exact that is depends on the learning rate. A model with
`learning_rate = 1.0` and `base_score = 0.0`, which is exactly what reading a
LightGBM file produces, round-trips bit-exactly: written out and read back it
predicts the identical `Float64` for every row, and re-dumping reproduces the
file byte for byte. Under any other learning rate the two agree to a few
units in the last place instead. Prediction accumulates
`score += learning_rate * leaf_value`, which the compiler may fuse into a
single rounded multiply-add, while the file forces `learning_rate *
leaf_value` to be rounded on its own before it can be stored. No text format
avoids that, because the leaf value is the only place the shrinkage can go.
So the round trip is not the identity on the
`(base_score, learning_rate, leaf value)` triple, and on predictions it is
exact only once the rate has been folded in.

What is rejected, and why
-------------------------
- Linear trees (`is_linear=1`, `leaf_const`, `leaf_coeff`). mojotrees leaves
  hold a constant.
- `missing_type = Zero` on a numerical split, as above.
- A feature split numerically in one node and by category set in another.
  mojotrees's `is_categorical` is a property of the feature.
- A feature needing more than 256 bins, or more than
  `CAT_MAX_BINS - 1` categories: mojotrees bins are a byte and its category
  sets are 256 bits.
- `average_output` (random forest boosting). mojotrees sums trees.
- Objectives mojotrees does not implement, `multiclassova` and friends, a
  `binary` objective with a non-unit `sigmoid`, and files written under a
  custom objective (whose link mojotrees cannot know). Which names those are,
  and the reason each is refused, comes from `objective_registry.mojo`; this
  module holds no second list of them.

Every rejection raises with the construct named. Nothing is silently
approximated.

Categorical splits
------------------
These convert exactly, in both directions, and the reason is worth stating
because a model file does not carry LightGBM's fitted category-to-bin table.
It does not have to. LightGBM's `cat_threshold` is a bitset over *raw*
category codes (`Tree::CategoricalDecision` tests `static_cast<int>(fval)`
against it directly), and its routing rule is: a code whose bit is set goes
left, and everything else -- an unset bit, a negative value, a NaN, a code
past the bitset -- goes right. mojotrees's rule is: bin 0 of a categorical
feature is never a set member, and `CategoricalSpec.bin_of` sends every
missing, unseen, or untabulated value to bin 0, so those rows go right.

So the table only has to contain the codes some split mentions. Take the
union of every code appearing in any categorical split on a feature, sort it,
and let it be that feature's `CategoricalSpec` table; map each split's code
set to the corresponding bins. A code in the table lands in the same branch
it did in LightGBM by construction, and a code outside the table has its bit
unset in *every* split (that is what being outside the union means), so
LightGBM sent it right and bin 0 sends it right too. The two agree on every
possible input, not merely on the training data.

The union is then widened with whatever `feature_infos` lists for that
feature, when the header carries a category list there and the widened table
still fits in `CAT_MAX_BINS - 1` entries. That is LightGBM's own fitted
category list, so the widened table is the one LightGBM fit rather than a
projection of it; widening cannot change any routing decision, because the
codes it adds are exactly the ones no split mentions. `LgbmImportReport`
records which of the two tables each conversion used.

Experimental status
-------------------
`LGBM_INTEROP_STATUS` is the one place that says how far this has been
checked, and the answer today is: hand-written fixtures only. No file
LightGBM wrote has been read, and no file this writer produced has been fed
to LightGBM. Until that changes, this is an experiment with a narrow surface
(`import_lgbm_file`, `export_lgbm_file`, `lgbm_unsupported_reason`, and the
lower-level parse/dump pairs they are built from), and the public Python and
C surfaces are expected to carry the same word.

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
- `left_child` / `right_child` - `node_index` values in mojotrees. LightGBM
  instead encodes a leaf as a negative number, `~leaf` (so -1 is leaf 0), and
  a non-negative number as an internal-node index in its own separate
  numbering.
- `leaf_ordinal` - a leaf's rank among its tree's leaves in `node_index`
  order (`Tree.leaf_ordinals`). This is mojotrees's leaf numbering and is not
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
# Only the three objective codes this module decorates differently from the
# registry's canonical name, plus the two ensembles. The rest of the
# objective space is never spelled here: `objective_code_from_name` and
# `objective_canonical_name` own it, which is what keeps this module from
# being a second objective table.
from .boosting import (
    BINARY_LOGISTIC,
    CUSTOM,
    L1,
    Booster,
    MulticlassBooster,
)
from .categorical import (
    CAT_BITSET_WORDS,
    CAT_MAX_BINS,
    UNKNOWN_BIN,
    CategoricalSpec,
    cat_pool_contains,
)
from .model import Model, MulticlassModel
from .objective_registry import (
    objective_canonical_name,
    objective_code_from_name,
    objective_param_name,
)
from .params import MULTICLASS
from .serialize import (
    load_feature_names,
    load_model,
    load_multiclass_model,
    model_file_kind,
    save_model,
    save_multiclass_model,
)
from .tree import Tree

# LightGBM's `decision_type` bitfield (see its `Tree::GetDecisionType`).
comptime _CATEGORICAL_MASK = 1
comptime _DEFAULT_LEFT_MASK = 2
comptime _MISSING_TYPE_SHIFT = 2
comptime _MISSING_TYPE_MASK = 3

comptime _MISSING_NONE = 0
comptime _MISSING_ZERO = 1
comptime _MISSING_NAN = 2

# mojotrees bins are UInt8, so a converted model may use at most this many
# bins for any one feature. A LightGBM model trained with the default
# max_bin=255 has at most 254 distinct thresholds on a feature, which fits
# with room for a reserved missing bin.
comptime _MAX_BINS = 256

# The format version this writer emits. The reader accepts v2, v3, and v4;
# they differ only in fields this converter either writes explicitly or
# rejects.
comptime _WRITE_VERSION = "v4"

# LightGBM packs a categorical split's code set into 32-bit words.
comptime _CAT_WORD_BITS = 32

# The widest code set this converter handles, in 32-bit words, read and
# written alike. LightGBM's bitset is as wide as the largest code plus one, so
# a model whose categories are sparse integers in the billions would otherwise
# render a file of hundreds of megabytes for a single split, and would decode
# into a loop over billions of bit positions. Two million codes is far past
# any real categorical feature, and the bound is what keeps the refusal a
# message rather than a disk-filling surprise. It also bounds every code this
# module produces well under the range `CategoricalSpec.bin_of` accepts.
comptime _MAX_CAT_WORDS = 1 << 16

# How far this interop has actually been checked. Read by
# `lgbm_interop_status`, which the bindings and the Python facade are asked
# to surface verbatim rather than paraphrase (see
# handoffs/connect_16_lgbm_interop.md).
comptime LGBM_INTEROP_STATUS = String(
    "experimental: converted against hand-written fixtures only. No file"
    " written by a real LightGBM build has been read, and no file produced"
    " here has been read back by LightGBM. mojotrees's own format"
    " (serialize.mojo) remains the only supported persistence format."
)


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

    `cat_boundaries` has `num_cat + 1` entries and slices `cat_threshold`
    into one code bitset per categorical split; a categorical node's
    `threshold` is its index into `cat_boundaries` rather than a real
    threshold. Every entry of `cat_threshold` is one 32-bit word, so bit `b`
    of word `w` of a split's slice stands for category code
    `32 * w + b`.
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
    var num_cat: Int
    var cat_boundaries: List[Int]
    var cat_threshold: List[Int]

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
        num_cat: Int,
        var cat_boundaries: List[Int],
        var cat_threshold: List[Int],
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
        self.num_cat = num_cat
        self.cat_boundaries = cat_boundaries^
        self.cat_threshold = cat_threshold^

    def is_categorical(self, node: Int) -> Bool:
        """Whether LightGBM internal node `node` routes by category set."""
        return self.decision_type[node] & _CATEGORICAL_MASK != 0

    def cat_index(self, node: Int, where: String) raises -> Int:
        """The `cat_boundaries` slot a categorical node's `threshold` names.

        LightGBM stores the index as a double, so a file whose value is not a
        whole number in range is malformed rather than merely unsupported.
        """
        var raw = self.threshold[node]
        var index = Int(raw)
        if Float64(index) != raw or index < 0 or index >= self.num_cat:
            raise Error(
                "LightGBM ",
                where,
                " node ",
                node,
                " is a categorical split whose threshold (",
                raw,
                ") is not one of its ",
                self.num_cat,
                " category sets",
            )
        return index

    def cat_codes(self, node: Int, where: String) raises -> List[Int]:
        """The raw category codes a categorical node sends left, ascending.

        This is the decode half of LightGBM's `Common::FindInBitset`: the set
        is a run of 32-bit words, and code `32 * w + b` is a member exactly
        when bit `b` of word `w` is set.
        """
        var index = self.cat_index(node, where)
        var lo = self.cat_boundaries[index]
        var hi = self.cat_boundaries[index + 1]
        if hi - lo > _MAX_CAT_WORDS:
            raise Error(
                "LightGBM ",
                where,
                " node ",
                node,
                " carries a category set ",
                hi - lo,
                " words wide, so its largest code is past the ",
                _MAX_CAT_WORDS * _CAT_WORD_BITS,
                " this converter handles",
            )
        var out = List[Int]()
        for w in range(lo, hi):
            var word = self.cat_threshold[w]
            if word == 0:
                continue
            for b in range(_CAT_WORD_BITS):
                if (word >> b) & 1 != 0:
                    out.append((w - lo) * _CAT_WORD_BITS + b)
        if len(out) == 0:
            raise Error(
                "LightGBM ",
                where,
                " node ",
                node,
                " is a categorical split with an empty category set, which"
                " sends every row right and cannot have been a split",
            )
        return out^


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
            " is a linear tree (is_linear=1); mojotrees leaves hold a"
            " constant, so a linear model cannot be converted",
        )
    if block.has("leaf_const") or block.has("leaf_coeff"):
        raise Error(
            "LightGBM ",
            where,
            " carries linear-model leaf coefficients; mojotrees leaves hold a"
            " constant, so a linear model cannot be converted",
        )
    var num_cat = block.int_or("num_cat", where, 0)
    if num_cat < 0:
        raise Error("LightGBM ", where, " declares num_cat=", num_cat)
    var cat_boundaries = List[Int]()
    var cat_threshold = List[Int]()
    if num_cat > 0:
        cat_boundaries = _parse_ints(
            block.require("cat_boundaries", where),
            num_cat + 1,
            where + " cat_boundaries",
        )
        cat_threshold = _parse_ints(
            block.get("cat_threshold", ""), -1, where + " cat_threshold"
        )
        if cat_boundaries[0] != 0:
            raise Error(
                "LightGBM ",
                where,
                " has cat_boundaries starting at ",
                cat_boundaries[0],
                " rather than 0",
            )
        for c in range(num_cat):
            if cat_boundaries[c + 1] <= cat_boundaries[c]:
                raise Error(
                    "LightGBM ",
                    where,
                    " category set ",
                    c,
                    " spans no words; cat_boundaries must ascend",
                )
        if cat_boundaries[num_cat] != len(cat_threshold):
            raise Error(
                "LightGBM ",
                where,
                " declares ",
                cat_boundaries[num_cat],
                " cat_threshold words but carries ",
                len(cat_threshold),
            )
        for w in range(len(cat_threshold)):
            if cat_threshold[w] < 0 or cat_threshold[w] > 0xFFFFFFFF:
                raise Error(
                    "LightGBM ",
                    where,
                    " cat_threshold word ",
                    w,
                    " (",
                    cat_threshold[w],
                    ") is not a 32-bit bitset word",
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

    var saw_categorical = False
    for j in range(n_split):
        var missing = (
            decision_type[j] >> _MISSING_TYPE_SHIFT
        ) & _MISSING_TYPE_MASK
        # The field is two bits and LightGBM defines three values, so the
        # fourth is a file this reader does not understand.
        if missing > _MISSING_NAN:
            raise Error(
                "LightGBM ", where, " node ", j, " has an unknown missing_type"
            )
        if decision_type[j] & _CATEGORICAL_MASK != 0:
            # `Tree::CategoricalDecision` never reads missing_type or
            # default_left: a code with its bit set goes left and everything
            # else, missing included, goes right. So both bits are ignored
            # here rather than rejected, which is also why 'Zero' is not an
            # error on a categorical node the way it is on a numerical one.
            saw_categorical = True
            continue
        if missing == _MISSING_ZERO:
            raise Error(
                "LightGBM ",
                where,
                " node ",
                j,
                " uses missing_type='Zero', which treats a zero feature value"
                " as missing; mojotrees routes missing values by a reserved"
                " bin and cannot express that",
            )
        if not isfinite(threshold[j]):
            raise Error(
                "LightGBM ",
                where,
                " node ",
                j,
                " has a non-finite threshold",
            )

    if saw_categorical and num_cat == 0:
        raise Error(
            "LightGBM ",
            where,
            " has a node marked categorical but declares num_cat=0, so its"
            " category sets are missing",
        )

    var tree = _LgbmTree(
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
        num_cat,
        cat_boundaries^,
        cat_threshold^,
    )
    # Decoding every set once here means a malformed `cat_boundaries` index or
    # an empty set is reported against the tree it came from, before any
    # feature-wide table is built out of it.
    for j in range(n_split):
        if tree.is_categorical(j):
            _ = tree.cat_codes(j, where)
    return tree^


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
                    " mojotrees sums them, so a random forest cannot be"
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
            " 'objective=' line is empty); mojotrees cannot know the link"
            " function, so load it with a custom-objective entry point"
            " instead"
        )
    return _Objective(name^, keys^, values^)


def lgbm_objective_code(objective_line: String) raises -> Int:
    """The mojotrees objective code for a LightGBM `objective=` line.

    Which names resolve, which alias means which code, and why an objective
    mojotrees has not implemented is refused all come from
    `objective_registry.objective_code_from_name`. This function adds only
    what is specific to a *file*: `lambdarank` converts here (a fitted
    ranking model is trees whose prediction is the raw score, so unlike a
    parameter string it needs no query groups), a `binary` objective must
    carry the unit sigmoid mojotrees's link is fixed at, and `custom` cannot
    convert at all.

    Returns `MULTICLASS` for `multiclass` and `softmax`, which need
    `load_lgbm_multiclass_model` rather than `load_lgbm_model`.
    """
    var parsed = _parse_objective_line(objective_line)
    var name = parsed.name.copy()
    var code = objective_code_from_name(name)
    if code == CUSTOM:
        raise Error(
            "this LightGBM model declares objective='",
            name,
            "'; mojotrees cannot know a custom objective's link function, so"
            " load it with a custom-objective entry point instead",
        )
    if code == BINARY_LOGISTIC:
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
                " for its binary objective; mojotrees's logistic link is"
                " fixed at sigmoid=1, so its probabilities would differ",
            )
    return code


def lgbm_objective_name(objective: Int) raises -> String:
    """The LightGBM `objective=` line for a mojotrees objective code.

    `objective_registry.objective_canonical_name` is the name; this function
    only decorates it for a reader of a model *file*, which is a different
    audience from an error message. Two decorations exist and both are
    deliberate:

    - `L1` writes as `regression_l1`, not as the registry's `mae`. Both
      resolve back to `L1` in LightGBM and in `lgbm_objective_code`, but
      `regression_l1` is the spelling LightGBM's own writer emits.
    - `BINARY_LOGISTIC` writes `binary sigmoid:1`, the setting the reader
      above insists on, so a file this writer produced reads back without
      depending on LightGBM's default.

    The scalar parameter of `huber`, `quantile`, `fair`, and `tweedie` is not
    kept on a fitted `Booster` (it shaped the gradients, not the trees), so it
    cannot be written; `lgbm_dropped_objective_param` names the one that goes
    missing, and `LgbmExportReport` carries it. A LightGBM reader will report
    its own default for it. The trees, and so the predictions, are unaffected:
    no objective's inverse link reads its scalar.
    """
    if objective == MULTICLASS:
        raise Error(
            "a softmax model's objective line carries its class count; use"
            " dump_lgbm_multiclass_model, which writes it"
        )
    if objective == CUSTOM:
        raise Error(
            "a model trained with a custom objective cannot be written as a"
            " LightGBM model file: the file has nowhere to record the"
            " gradient callback or the link function"
        )
    if objective == L1:
        return String("regression_l1")
    if objective == BINARY_LOGISTIC:
        return String("binary sigmoid:1")
    return objective_canonical_name(objective)


def lgbm_dropped_objective_param(objective: Int) -> String:
    """The LightGBM objective setting a written file cannot carry, or an
    empty string when this objective reads none.

    `objective_registry.objective_param_name` is the source: `alpha` for
    huber and quantile, `fair_c` for fair, `tweedie_variance_power` for
    tweedie. A fitted `Booster` does not keep the value, and writing the
    registry's *default* in its place would put a number in the file that the
    model was very likely not trained with, so the setting is omitted and
    named here instead.
    """
    return objective_param_name(objective)


# ---------------------------------------------------------------------------
# Reading: LightGBM text -> Model / MulticlassModel
# ---------------------------------------------------------------------------


def _check_feature_indices(
    trees: List[_LgbmTree], n_features: Int
) raises:
    """Every split's feature index is one this model declares.

    Run before anything buckets by feature, so an out-of-range index is
    reported as the malformed file it is rather than as an out-of-bounds
    access inside a counting pass.
    """
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


def _collect_edges(
    trees: List[_LgbmTree],
    n_features: Int,
    mut edges: List[Float64],
    mut offsets: List[Int],
) raises:
    """Every distinct threshold used on each feature, ascending, in
    `BinMapper` layout: feature f's edges are `edges[offsets[f]:offsets[f+1]]`.

    Categorical nodes contribute nothing: their `threshold` is a
    `cat_boundaries` index, not a value, and a categorical feature carries no
    edges at all (see `BinMapper`).

    Bucketed by feature with a counting pass so a large model costs one sort
    per feature rather than a scan per feature over every node.
    """
    var counts = List[Int](capacity=n_features)
    counts.resize(n_features, 0)
    for t in range(len(trees)):
        ref tree = trees[t]
        for j in range(len(tree.split_feature)):
            if tree.is_categorical(j):
                continue
            counts[tree.split_feature[j]] += 1

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
            if tree.is_categorical(j):
                continue
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

    LightGBM stores `missing_type` per node; mojotrees reserves per feature,
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
            if tree.is_categorical(j):
                # A categorical feature's missing rows land in bin 0, which is
                # never a set member, so it reserves nothing and its nodes'
                # missing_type bits carry no meaning to agree about.
                continue
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
                    " mojotrees reserves a missing bin per feature, so the"
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


def _sorted_distinct(var values: List[Int]) -> List[Int]:
    """`values` sorted with duplicates dropped, which is the shape
    `CategoricalSpec` requires of a feature's code table."""
    sort(values)
    var out = List[Int](capacity=len(values))
    for i in range(len(values)):
        if i == 0 or values[i] != values[i - 1]:
            out.append(values[i])
    return out^


def _feature_info_tokens(header: _KeyVals, n_features: Int) -> List[String]:
    """The header's `feature_infos` entries, one per feature.

    An empty list means the header carries none, or carries the wrong number
    of them. Both are treated as "no information" rather than as an error:
    `feature_infos` only ever widens a category table here (see the module
    docstring), so a missing or malformed one costs nothing.
    """
    var raw = header.get("feature_infos", "")
    var out = List[String]()
    for token in raw.split():
        out.append(String(token))
    if len(out) != n_features:
        return List[String]()
    return out^


def _feature_info_codes(token: String) -> List[Int]:
    """A `feature_infos` entry read as LightGBM's category list, `0:2:5`.

    Empty for anything that is not one: `none` for an unused feature, and
    `[min:max]` for a numerical one, are both LightGBM's own spellings, and
    so is a list whose entries are not codes a `CategoricalSpec` can hold.
    """
    var out = List[Int]()
    if token == "none" or token.byte_length() == 0:
        return out^
    if String(token[byte=0:1]) == "[":
        return out^
    for part in token.split(":"):
        var text = String(part)
        var code: Int
        try:
            code = Int(text)
        except:
            return List[Int]()
        if code < 0 or code >= _MAX_CAT_WORDS * _CAT_WORD_BITS:
            return List[Int]()
        out.append(code)
    return out^


struct _CategoryTables(Copyable, Movable):
    """Which features a LightGBM model splits categorically, and the code
    table each of them converts through.

    `widened[f]` records that `feature_infos` contributed codes no split
    mentions, which is the difference between the table LightGBM fit and the
    projection of it the trees alone imply. Nothing routes differently either
    way; it is reported so a caller can say which one it got.
    """

    var spec: CategoricalSpec
    var widened: List[Bool]
    var n_categorical: Int

    def __init__(
        out self,
        var spec: CategoricalSpec,
        var widened: List[Bool],
        n_categorical: Int,
    ):
        self.spec = spec^
        self.widened = widened^
        self.n_categorical = n_categorical


def _collect_categories(
    trees: List[_LgbmTree], n_features: Int, header: _KeyVals
) raises -> _CategoryTables:
    """The `CategoricalSpec` a LightGBM model implies.

    Feature f's table is the union of every code any categorical split on f
    sends left, widened by `feature_infos` when that fits. The module
    docstring argues why the union alone already routes every possible input
    the way LightGBM does.
    """
    var is_cat = List[Bool](capacity=n_features)
    is_cat.resize(n_features, False)
    var is_num = List[Bool](capacity=n_features)
    is_num.resize(n_features, False)
    var counts = List[Int](capacity=n_features)
    counts.resize(n_features, 0)

    for t in range(len(trees)):
        ref tree = trees[t]
        var where = String("tree ") + String(t)
        for j in range(len(tree.split_feature)):
            var f = tree.split_feature[j]
            if tree.is_categorical(j):
                is_cat[f] = True
                counts[f] += len(tree.cat_codes(j, where))
            else:
                is_num[f] = True

    var n_categorical = 0
    for f in range(n_features):
        if is_cat[f] and is_num[f]:
            raise Error(
                "LightGBM model: feature ",
                f,
                " is split by category set in one node and by a numerical"
                " threshold in another; mojotrees decides that per feature,"
                " so the two cannot both hold",
            )
        if is_cat[f]:
            n_categorical += 1

    var widened = List[Bool](capacity=n_features)
    widened.resize(n_features, False)
    if n_categorical == 0:
        return _CategoryTables(
            CategoricalSpec.all_numerical(n_features), widened^, 0
        )

    var starts = List[Int](capacity=n_features + 1)
    starts.append(0)
    var total = 0
    for f in range(n_features):
        total += counts[f]
        starts.append(total)
    var flat = List[Int](capacity=total)
    flat.resize(total, 0)
    var cursor = List[Int](capacity=n_features)
    for f in range(n_features):
        cursor.append(starts[f])
    for t in range(len(trees)):
        ref tree = trees[t]
        var where = String("tree ") + String(t)
        for j in range(len(tree.split_feature)):
            if not tree.is_categorical(j):
                continue
            var f = tree.split_feature[j]
            var codes = tree.cat_codes(j, where)
            for i in range(len(codes)):
                flat[cursor[f]] = codes[i]
                cursor[f] += 1

    var infos = _feature_info_tokens(header, n_features)
    var codes = List[Int]()
    var offsets = List[Int](capacity=n_features + 1)
    offsets.append(0)
    for f in range(n_features):
        if not is_cat[f]:
            offsets.append(len(codes))
            continue
        var column = List[Int](capacity=starts[f + 1] - starts[f])
        for i in range(starts[f], starts[f + 1]):
            column.append(flat[i])
        var table = _sorted_distinct(column^)
        if len(infos) == n_features:
            var extra = _feature_info_codes(infos[f])
            if len(extra) > 0:
                var merged = table.copy()
                for i in range(len(extra)):
                    merged.append(extra[i])
                var widened_table = _sorted_distinct(merged^)
                if (
                    len(widened_table) > len(table)
                    and len(widened_table) <= CAT_MAX_BINS - 1
                ):
                    table = widened_table^
                    widened[f] = True
        if len(table) > CAT_MAX_BINS - 1:
            raise Error(
                "LightGBM model: feature ",
                f,
                " splits on ",
                len(table),
                " distinct categories; mojotrees's category sets are ",
                CAT_MAX_BINS,
                " bits, one of which is reserved for missing and unseen"
                " codes, so at most ",
                CAT_MAX_BINS - 1,
                " fit. Retrain with a smaller max_cat_threshold, or bucket"
                " the rare categories.",
            )
        for i in range(len(table)):
            codes.append(table[i])
        offsets.append(len(codes))

    return _CategoryTables(
        CategoricalSpec(is_cat^, codes^, offsets^), widened^, n_categorical
    )


def _cat_bin_of(
    cats: CategoricalSpec, feature: Int, code: Int, where: String
) raises -> Int:
    """The bin a category code holds in a converted table.

    The table was built from these codes, so the lookup cannot miss; it goes
    through `CategoricalSpec.bin_of` rather than around it so that the bin a
    split routes on and the bin a raw value maps to come from one function.
    """
    var bin = cats.bin_of(feature, Float64(code))
    if bin == UNKNOWN_BIN:
        raise Error(
            "internal error converting a LightGBM model: ",
            where,
            " sends category ",
            code,
            " of feature ",
            feature,
            " left, but that code is not in the table collected for it",
        )
    return bin


def _bins_needed(
    offsets: List[Int], missing_bin: List[Int], cats: CategoricalSpec
) raises -> Int:
    """The `n_bins` a converted mapper needs: the widest feature wins.

    A categorical feature uses one bin per kept category plus bin 0, which
    `CategoricalSpec` reserves for missing, unseen, and dropped codes.
    """
    var n_features = len(missing_bin)
    var needed = 1
    for f in range(n_features):
        var here: Int
        if cats.is_cat(f):
            here = cats.n_categories(f) + 1
        else:
            here = offsets[f + 1] - offsets[f] + 1
            if missing_bin[f] >= 0:
                here += 1
        if here > needed:
            needed = here
    if needed > _MAX_BINS:
        raise Error(
            "this LightGBM model needs ",
            needed,
            " bins for one feature; mojotrees stores bins in a byte, so at"
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
    """A mojotrees tree under construction, one appended node at a time."""

    var feature: List[Int]
    var threshold_bin: List[Int]
    var left: List[Int]
    var right: List[Int]
    var value: List[Float64]
    var split_gain: List[Float64]
    var default_left: List[Bool]
    var missing_bin: List[Int]
    var cat_offset: List[Int]
    var cat_bitset: List[UInt64]
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
        self.cat_offset = List[Int]()
        self.cat_bitset = List[UInt64]()
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
        self.cat_offset.append(-1)
        self.count.append(0.0)
        return node

    def set_categorical(mut self, node: Int, bins: List[Int]):
        """Record node `node` as routing by the bin set `bins`.

        The same layout `Tree._set_split` writes: no threshold, no default
        direction, no missing bin, and a `CAT_BITSET_WORDS`-word set appended
        to the tree's flat pool. Bin 0 is never written, so a missing or
        unseen code still falls right (see `Tree.goes_left`).
        """
        self.threshold_bin[node] = -1
        self.default_left[node] = False
        self.missing_bin[node] = -1
        self.cat_offset[node] = len(self.cat_bitset)
        for _ in range(CAT_BITSET_WORDS):
            self.cat_bitset.append(UInt64(0))
        for i in range(len(bins)):
            var bin = bins[i]
            var word = self.cat_offset[node] + (bin >> 6)
            self.cat_bitset[word] |= UInt64(1) << UInt64(bin & 63)

    def into_tree(mut self, n_leaves: Int) raises -> Tree:
        """Hand the accumulated arrays to a `Tree`, leaving this builder
        empty.

        Swapped out rather than transferred field by field: the ownership
        checker will not let a struct it still has to destroy be emptied one
        field at a time, and this is the same move `Tree.take_hist` makes for
        the same reason.
        """
        var feature = List[Int]()
        var threshold_bin = List[Int]()
        var left = List[Int]()
        var right = List[Int]()
        var value = List[Float64]()
        var split_gain = List[Float64]()
        var default_left = List[Bool]()
        var missing_bin = List[Int]()
        var cat_offset = List[Int]()
        var cat_bitset = List[UInt64]()
        var count = List[Float64]()
        swap(feature, self.feature)
        swap(threshold_bin, self.threshold_bin)
        swap(left, self.left)
        swap(right, self.right)
        swap(value, self.value)
        swap(split_gain, self.split_gain)
        swap(default_left, self.default_left)
        swap(missing_bin, self.missing_bin)
        swap(cat_offset, self.cat_offset)
        swap(cat_bitset, self.cat_bitset)
        swap(count, self.count)
        return Tree(
            feature^,
            threshold_bin^,
            left^,
            right^,
            value^,
            split_gain^,
            n_leaves,
            default_left^,
            missing_bin^,
            cat_offset^,
            cat_bitset^,
            count^,
        )


def _emit_node(
    mut arrays: _NodeArrays,
    src: _LgbmTree,
    mapper: BinMapper,
    mut seen_internal: List[Bool],
    mut seen_leaf: List[Bool],
    code: Int,
    tree_index: Int,
) raises -> Int:
    """Convert one LightGBM subtree, returning its mojotrees node index.

    `code` is LightGBM's child encoding: non-negative is an internal-node
    index, negative is the leaf `-code - 1`. Nodes are emitted in preorder, so
    a parent's index is always below its children's, which is the ordering
    `tree.node_bounds` and every top-down pass in mojotrees assume.
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
    if src.is_categorical(code):
        var where = String("tree ") + String(tree_index)
        var codes = src.cat_codes(code, where)
        var bins = List[Int](capacity=len(codes))
        for i in range(len(codes)):
            bins.append(
                _cat_bin_of(mapper.cats, feature, codes[i], where)
            )
        arrays.set_categorical(node, bins)
    else:
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
    return arrays.into_tree(src.num_leaves)


struct _Mapping(Copyable, Movable):
    """The `BinMapper` a LightGBM file implies, plus what building it found.

    The two travel together because the report needs facts only this step
    knows -- how many features turned out categorical, whether any table came
    from `feature_infos` -- and recomputing them from the finished mapper
    would be a second derivation of the same thing.
    """

    var mapper: BinMapper
    var cats: _CategoryTables

    def __init__(out self, var mapper: BinMapper, var cats: _CategoryTables):
        self.mapper = mapper^
        self.cats = cats^


def _synthesize_mapping(
    trees: List[_LgbmTree], n_features: Int, header: _KeyVals
) raises -> _Mapping:
    """The bin mapper a LightGBM file implies: one edge per distinct
    threshold on a numerical feature, a reserved missing bin for each feature
    whose nodes say `missing_type=NaN`, and a category table for each feature
    its trees split by set. See the module docstring for why this is exact
    for prediction and wrong to reuse for training."""
    _check_feature_indices(trees, n_features)
    var cats = _collect_categories(trees, n_features, header)
    var edges = List[Float64]()
    var offsets = List[Int]()
    _collect_edges(trees, n_features, edges, offsets)
    var missing_bin = _collect_missing_bins(trees, n_features, offsets)
    var n_bins = _bins_needed(offsets, missing_bin, cats.spec)
    var mapper = BinMapper(
        edges^,
        offsets^,
        n_features,
        n_bins,
        cats.spec.copy(),
        missing_bin^,
    )
    return _Mapping(mapper^, cats^)


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


struct LgbmImportReport(Copyable, Movable, Writable):
    """What one LightGBM-to-mojotrees conversion did.

    Every field is a fact the conversion established and the finished model
    no longer records, which is the whole reason it is returned rather than
    recomputed: nothing downstream can tell a synthesized bin mapper from a
    fitted one, or a widened category table from a projected one, by looking
    at the model.
    """

    var n_features: Int
    var n_trees: Int
    var n_classes: Int
    var objective: Int
    var objective_line: String
    var format_version: String
    var n_categorical_features: Int
    var n_widened_tables: Int
    var n_edges: Int
    var n_missing_reservations: Int
    var has_node_counts: Bool
    var feature_names: List[String]

    def __init__(
        out self,
        n_features: Int,
        n_trees: Int,
        n_classes: Int,
        objective: Int,
        var objective_line: String,
        var format_version: String,
        n_categorical_features: Int,
        n_widened_tables: Int,
        n_edges: Int,
        n_missing_reservations: Int,
        has_node_counts: Bool,
        var feature_names: List[String],
    ):
        self.n_features = n_features
        self.n_trees = n_trees
        self.n_classes = n_classes
        self.objective = objective
        self.objective_line = objective_line^
        self.format_version = format_version^
        self.n_categorical_features = n_categorical_features
        self.n_widened_tables = n_widened_tables
        self.n_edges = n_edges
        self.n_missing_reservations = n_missing_reservations
        self.has_node_counts = has_node_counts
        self.feature_names = feature_names^

    def mapper_is_training_binning(self) -> Bool:
        """Always False, and a method rather than a comment so a caller can
        assert on it.

        A converted mapper holds the edges the trees split on, which is all a
        LightGBM model file carries. It routes every raw value exactly the
        way LightGBM does, and it is *not* the binning LightGBM fit, so it
        must not be handed to `boosting.train_more` or to anything else that
        reads a mapper as a description of the training data.
        `BinMapper.matches` will refuse to pair it with a freshly fit mapper,
        which is the enforcement; this is the announcement.
        """
        return False

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "LgbmImportReport(n_features=",
            self.n_features,
            ", n_trees=",
            self.n_trees,
            ", n_classes=",
            self.n_classes,
            ", objective=",
            self.objective_line,
            ", version=",
            self.format_version,
            ", categorical_features=",
            self.n_categorical_features,
            ", widened_tables=",
            self.n_widened_tables,
            ", edges=",
            self.n_edges,
            ", missing_reservations=",
            self.n_missing_reservations,
            ", node_counts=",
            self.has_node_counts,
            ", feature_names=",
            len(self.feature_names),
            ", training_binning=False)",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct LgbmImport(Copyable, Movable):
    """A converted single-output model and the report of its conversion."""

    var model: Model
    var report: LgbmImportReport

    def __init__(out self, var model: Model, var report: LgbmImportReport):
        self.model = model^
        self.report = report^


struct LgbmMulticlassImport(Copyable, Movable):
    """A converted softmax model and the report of its conversion."""

    var model: MulticlassModel
    var report: LgbmImportReport

    def __init__(
        out self, var model: MulticlassModel, var report: LgbmImportReport
    ):
        self.model = model^
        self.report = report^


def _all_have_node_counts(trees: List[Tree]) -> Bool:
    """Whether every converted tree carries covers.

    LightGBM's `leaf_count` and `internal_count` are optional in its format,
    and `_finish_tree` substitutes zeros for an omitted one. A model whose
    counts are zeros predicts identically and cannot answer for feature
    contributions (`Tree.check_node_counts` says so), which is a difference
    worth reporting at conversion time rather than at the first
    `predict_contrib`.
    """
    for t in range(len(trees)):
        if not trees[t].has_node_counts():
            return False
    return True


def _import_report(
    mapping: _Mapping,
    header: _KeyVals,
    n_trees: Int,
    n_classes: Int,
    objective: Int,
    has_node_counts: Bool,
) raises -> LgbmImportReport:
    var widened = 0
    for f in range(len(mapping.cats.widened)):
        if mapping.cats.widened[f]:
            widened += 1
    var reservations = 0
    for f in range(mapping.mapper.n_features):
        if mapping.mapper.missing_bin[f] >= 0:
            reservations += 1
    # LightGBM's `feature_names` line is space-separated, so a name with a
    # space in it would arrive as two names and silently shift every column
    # after it. A count that does not match is therefore dropped whole
    # rather than trusted in part, which is also what `save_model` demands:
    # one name per feature or none at all.
    var names = List[String]()
    for token in header.get("feature_names", "").split():
        names.append(String(token))
    if len(names) != mapping.mapper.n_features:
        names = List[String]()
    return LgbmImportReport(
        mapping.mapper.n_features,
        n_trees,
        n_classes,
        objective,
        header.get("objective", ""),
        header.get("version", "v3"),
        mapping.cats.n_categorical,
        widened,
        len(mapping.mapper.edges),
        reservations,
        has_node_counts,
        names^,
    )


def parse_lgbm_model(text: String) raises -> Model:
    """Convert a single-output LightGBM model string into a `Model`.

    The result predicts exactly what LightGBM's own predictor would, on the
    raw feature values, for every construct this module accepts. It carries
    `learning_rate = 1.0` and `base_score = 0.0` because LightGBM has already
    folded both into its leaf values; see the module docstring.

    EXPERIMENTAL; `import_lgbm_model` returns the same model together with
    the report of what the conversion had to synthesize.
    """
    var imported = import_lgbm_model(text)
    return imported.model.copy()


def import_lgbm_model(text: String) raises -> LgbmImport:
    """`parse_lgbm_model` with its conversion report. EXPERIMENTAL."""
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
            " trees per iteration but declares num_class=1; mojotrees's"
            " single-output model grows one",
        )

    var mapping = _synthesize_mapping(
        parsed.trees, _n_features(parsed.header), parsed.header
    )
    var trees = _convert_trees(parsed.trees, mapping.mapper)
    var n_trees = len(trees)
    var report = _import_report(
        mapping,
        parsed.header,
        n_trees,
        1,
        objective,
        _all_have_node_counts(trees),
    )
    var booster = Booster(trees^, 0.0, 1.0, objective)
    var mapper = mapping.mapper.copy()
    return LgbmImport(Model(mapper^, booster^), report^)


def parse_lgbm_multiclass_model(text: String) raises -> MulticlassModel:
    """Convert a multiclass LightGBM model string into a `MulticlassModel`.

    LightGBM stores one tree per class per iteration in round-major order,
    which is the order `MulticlassBooster` already uses, so the trees carry
    over unpermuted. Every class's base score is 0.0 for the reason given in
    the module docstring.

    EXPERIMENTAL; `import_lgbm_multiclass_model` returns the same model
    together with the report of what the conversion had to synthesize.
    """
    var imported = import_lgbm_multiclass_model(text)
    return imported.model.copy()


def import_lgbm_multiclass_model(
    text: String,
) raises -> LgbmMulticlassImport:
    """`parse_lgbm_multiclass_model` with its conversion report.
    EXPERIMENTAL."""
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
            " classes; mojotrees grows one per class",
        )
    if len(parsed.trees) % n_classes != 0:
        raise Error(
            "LightGBM multiclass model holds ",
            len(parsed.trees),
            " trees, which is not a whole number of rounds over ",
            n_classes,
            " classes",
        )

    var mapping = _synthesize_mapping(
        parsed.trees, _n_features(parsed.header), parsed.header
    )
    var trees = _convert_trees(parsed.trees, mapping.mapper)
    var n_trees = len(trees)
    var report = _import_report(
        mapping,
        parsed.header,
        n_trees,
        n_classes,
        objective,
        _all_have_node_counts(trees),
    )
    var base_scores = List[Float64](capacity=n_classes)
    base_scores.resize(n_classes, 0.0)
    var booster = MulticlassBooster(trees^, base_scores^, n_classes, 1.0)
    var mapper = mapping.mapper.copy()
    return LgbmMulticlassImport(MulticlassModel(mapper^, booster^), report^)


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
    var cat_boundaries: List[Int]
    var cat_threshold: List[Int]

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
        # LightGBM's `cat_boundaries` always opens at 0 and gains one entry
        # per category set, so `num_cat` is its length minus one.
        self.cat_boundaries = List[Int]()
        self.cat_boundaries.append(0)
        self.cat_threshold = List[Int]()

    def num_cat(self) -> Int:
        return len(self.cat_boundaries) - 1

    def add_category_set(mut self, codes: List[Int]) raises -> Int:
        """Append one split's raw-code bitset and return its `cat_boundaries`
        index, which is what LightGBM stores in that node's `threshold`.

        `codes` must be ascending. The bitset runs from code 0 to the largest
        code, because that is the width LightGBM's `FindInBitset` reads.
        """
        var largest = codes[len(codes) - 1]
        var n_words = largest // _CAT_WORD_BITS + 1
        if n_words > _MAX_CAT_WORDS:
            raise Error(
                "cannot write a category set containing code ",
                largest,
                ": LightGBM's format stores it as a bitset that wide, and"
                " this converter stops at ",
                _MAX_CAT_WORDS * _CAT_WORD_BITS,
                " codes",
            )
        var index = self.num_cat()
        var base = len(self.cat_threshold)
        for _ in range(n_words):
            self.cat_threshold.append(0)
        for i in range(len(codes)):
            var code = codes[i]
            var word = base + code // _CAT_WORD_BITS
            self.cat_threshold[word] |= 1 << (code % _CAT_WORD_BITS)
        self.cat_boundaries.append(len(self.cat_threshold))
        return index


def _check_writable_tree(
    tree: Tree, mapper: BinMapper, tree_index: Int
) raises:
    """Refuse a tree LightGBM's format cannot hold before any of it is
    rendered, so a partial file is never written."""
    for i in range(len(tree.feature)):
        if tree.cat_offset[i] < 0:
            continue
        var feature = tree.feature[i]
        if not mapper.cats.is_cat(feature):
            raise Error(
                "tree ",
                tree_index,
                " node ",
                i,
                " routes feature ",
                feature,
                " by category set, but the model's bin mapper has no"
                " category table for it, so its bins cannot be turned back"
                " into the raw codes LightGBM's format stores",
            )


def _node_category_codes(
    tree: Tree, mapper: BinMapper, node: Int, tree_index: Int
) raises -> List[Int]:
    """The raw category codes a mojotrees categorical node sends left.

    The inverse of the read direction: a set member is a bin, and bin `b` of
    feature `f` is the code `cats.codes[cats.offsets[f] + b - 1]` (bin 0 is
    the reserved unknown bin, never a member). Membership is tested with
    `cat_pool_contains`, the same function `Tree.goes_left` routes with, so
    what is written is what the model does rather than a second reading of
    the bitset layout.
    """
    var feature = tree.feature[node]
    var offset = tree.cat_offset[node]
    var begin = mapper.cats.offsets[feature]
    var n_categories = mapper.cats.n_categories(feature)
    var codes = List[Int]()
    for bin in range(1, CAT_MAX_BINS):
        if not cat_pool_contains(tree.cat_bitset, offset, bin):
            continue
        if bin > n_categories:
            raise Error(
                "tree ",
                tree_index,
                " node ",
                node,
                " sends bin ",
                bin,
                " of feature ",
                feature,
                " left, but the model's category table for that feature holds"
                " only ",
                n_categories,
                " categories",
            )
        codes.append(mapper.cats.codes[begin + bin - 1])
    if len(codes) == 0:
        raise Error(
            "tree ",
            tree_index,
            " node ",
            node,
            " routes feature ",
            tree.feature[node],
            " by an empty category set, which sends every row right; LightGBM"
            " has no such split",
        )
    return codes^


def _emit_lgbm_node(
    mut dst: _LgbmOut,
    tree: Tree,
    mapper: BinMapper,
    node: Int,
    scale: Float64,
    bias: Float64,
    tree_index: Int,
) raises -> Int:
    """Convert one mojotrees subtree, returning LightGBM's child code.

    Preorder, so the mojotrees root becomes LightGBM internal node 0, which
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
    var decision = 0
    var threshold: Float64

    if tree.cat_offset[node] >= 0:
        # A category set, written over raw codes. `default_left` and
        # `missing_type` are left clear: LightGBM's `CategoricalDecision`
        # reads neither, and mojotrees's categorical nodes carry neither.
        var codes = _node_category_codes(tree, mapper, node, tree_index)
        decision |= _CATEGORICAL_MASK
        threshold = Float64(dst.add_category_set(codes))
    else:
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
                ", which has no upper edge in the model's bin mapper, so it"
                " has no real-valued LightGBM threshold",
            )
        threshold = mapper.edges[lo + bin]
        if tree.default_left[node]:
            decision |= _DEFAULT_LEFT_MASK
        if tree.missing_bin[node] >= 0:
            decision |= _MISSING_NAN << _MISSING_TYPE_SHIFT

    var index = len(dst.split_feature)
    dst.split_feature.append(feature)
    dst.threshold.append(threshold)
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
    written as zeros: mojotrees keeps node covers, not hessian sums, and
    LightGBM treats both fields as optional.
    """
    _check_writable_tree(tree, mapper, tree_index)
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
    body += "num_cat=" + String(built.num_cat()) + "\n"
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
    # LightGBM writes the two category arrays only for a tree that has any,
    # and writes them here, between `internal_count` and `is_linear`.
    if built.num_cat() > 0:
        body += (
            "cat_boundaries=" + _join_ints(built.cat_boundaries) + "\n"
        )
        body += "cat_threshold=" + _join_ints(built.cat_threshold) + "\n"
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
    """LightGBM's `feature_infos`, in LightGBM's two spellings.

    A categorical feature's entry is its category codes joined by `:`, which
    is what `BinMapper::bin_info_string` writes for a categorical bin mapper
    and what `_feature_info_codes` reads back. A numerical feature's is
    `[min:max]`, and one with no edges (never split, so never binned into the
    file) is `none`.
    """
    var out = String("")
    for f in range(mapper.n_features):
        if f > 0:
            out += " "
        if mapper.cats.is_cat(f):
            var begin = mapper.cats.offsets[f]
            var end = mapper.cats.offsets[f + 1]
            if end <= begin:
                out += "none"
                continue
            for i in range(begin, end):
                if i > begin:
                    out += ":"
                out += String(mapper.cats.codes[i])
            continue
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


def _feature_names(n_features: Int, names: List[String]) raises -> String:
    """LightGBM's `feature_names` line.

    `names` are the ones a native model file carries (`serialize.mojo`'s v4
    header, read back by `load_feature_names`); a `Model` itself holds none,
    so they arrive as a parameter here exactly as they do in `save_model`.
    An empty list means the model was saved without them, and the file gets
    the `Column_0`, `Column_1`, ... names LightGBM itself invents for a
    dataset that has none.

    A name with whitespace in it is refused rather than written: the line is
    space-separated, so writing one would silently produce a file with the
    wrong number of names.
    """
    var out = String("")
    for f in range(n_features):
        if f > 0:
            out += " "
        if len(names) == 0:
            out += "Column_" + String(f)
            continue
        var name = names[f]
        if name.byte_length() == 0 or len(name.split()) != 1:
            raise Error(
                "cannot write feature name '",
                name,
                "' to a LightGBM model file: its feature_names line is"
                " space-separated, so a name cannot be empty or contain"
                " whitespace",
            )
        out += name
    return out^


def _assemble(
    mapper: BinMapper,
    objective_line: String,
    num_class: Int,
    trees_per_iteration: Int,
    blocks: List[String],
    names: List[String],
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
    out += (
        "feature_names=" + _feature_names(mapper.n_features, names) + "\n"
    )
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


def _check_export_names(names: List[String], n_features: Int) raises:
    """Feature names are all of them or none of them.

    The same rule `serialize._check_feature_names` enforces, and stated the
    same way: a list of the wrong length is refused rather than padded,
    because a name attached to the wrong column is worse than no name.
    """
    if len(names) != 0 and len(names) != n_features:
        raise Error(
            "feature_names has ",
            len(names),
            " entries for a model with ",
            n_features,
            " features; pass one per feature or none at all",
        )


def _n_categorical(mapper: BinMapper) -> Int:
    var n = 0
    for f in range(mapper.n_features):
        if mapper.cats.is_cat(f):
            n += 1
    return n


struct LgbmExportReport(Copyable, Movable, Writable):
    """What one mojotrees-to-LightGBM conversion had to leave behind.

    Nothing here changes what the file predicts. It is the list of things a
    caller would otherwise have to know from the module docstring:
    which objective setting could not be written, whether the shrinkage and
    the base score had to be folded into leaf values, and therefore whether
    the file's predictions are bit-identical to the model's or merely equal
    to within a few units in the last place.
    """

    var n_trees: Int
    var n_classes: Int
    var objective_line: String
    var dropped_objective_param: String
    var n_categorical_features: Int
    var shrinkage_folded: Bool
    var base_score_folded: Bool
    var n_feature_names: Int

    def __init__(
        out self,
        n_trees: Int,
        n_classes: Int,
        var objective_line: String,
        var dropped_objective_param: String,
        n_categorical_features: Int,
        shrinkage_folded: Bool,
        base_score_folded: Bool,
        n_feature_names: Int,
    ):
        self.n_trees = n_trees
        self.n_classes = n_classes
        self.objective_line = objective_line^
        self.dropped_objective_param = dropped_objective_param^
        self.n_categorical_features = n_categorical_features
        self.shrinkage_folded = shrinkage_folded
        self.base_score_folded = base_score_folded
        self.n_feature_names = n_feature_names

    def predictions_bit_exact(self) -> Bool:
        """Whether reading this file back reproduces the model's predictions
        bit for bit.

        True exactly when nothing had to be folded: prediction accumulates
        `score += learning_rate * leaf_value`, which may become one rounded
        multiply-add, while the file must round `learning_rate * leaf_value`
        on its own before it can store it. A model already at
        `learning_rate = 1.0` with a zero base score -- which is what reading
        a LightGBM file produces -- has nothing to fold and round-trips
        exactly.
        """
        return not self.shrinkage_folded and not self.base_score_folded

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "LgbmExportReport(n_trees=",
            self.n_trees,
            ", n_classes=",
            self.n_classes,
            ", objective=",
            self.objective_line,
            ", dropped_param=",
            self.dropped_objective_param,
            ", categorical_features=",
            self.n_categorical_features,
            ", feature_names=",
            self.n_feature_names,
            ", bit_exact=",
            self.predictions_bit_exact(),
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct LgbmExport(Copyable, Movable):
    """A rendered LightGBM model string and the report of its conversion."""

    var text: String
    var report: LgbmExportReport

    def __init__(out self, var text: String, var report: LgbmExportReport):
        self.text = text^
        self.report = report^


def dump_lgbm_model(
    model: Model, feature_names: List[String] = []
) raises -> String:
    """Render a single-output `Model` as a LightGBM model string.

    Leaf values are multiplied by the learning rate and the base score is
    folded into tree 0, so the file predicts what the model does: exactly,
    when the learning rate is already 1.0, and otherwise to within a few
    units in the last place. The module docstring explains why that last
    rounding is unavoidable. Raises for a custom-objective model, and for a
    categorical node whose feature has no category table in the model's
    mapper.

    `feature_names` carries exactly what `serialize.save_model` takes under
    that name, and for the same reason: a `Model` holds no names of its own.
    An empty list writes LightGBM's own `Column_i` defaults.

    EXPERIMENTAL; `export_lgbm_model` returns the same text together with
    the report of what the conversion left behind.
    """
    var exported = export_lgbm_model(model, feature_names)
    return exported.text.copy()


def export_lgbm_model(
    model: Model, feature_names: List[String] = []
) raises -> LgbmExport:
    """`dump_lgbm_model` with its conversion report. EXPERIMENTAL."""
    _check_export_names(feature_names, model.mapper.n_features)
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
    var text = _assemble(
        model.mapper, objective_line, 1, 1, blocks, feature_names
    )
    var report = LgbmExportReport(
        len(trees),
        1,
        objective_line.copy(),
        lgbm_dropped_objective_param(model.booster.objective),
        _n_categorical(model.mapper),
        model.booster.learning_rate != 1.0,
        model.booster.base_score != 0.0,
        len(feature_names),
    )
    return LgbmExport(text^, report^)


def dump_lgbm_multiclass_model(
    model: MulticlassModel, feature_names: List[String] = []
) raises -> String:
    """Render a `MulticlassModel` as a LightGBM model string.

    The trees keep their round-major order, which is LightGBM's own. Each
    class's base score is folded into that class's iteration-0 tree.

    `feature_names` behaves exactly as it does in `dump_lgbm_model`.

    EXPERIMENTAL; `export_lgbm_multiclass_model` returns the same text
    together with the report of what the conversion left behind.
    """
    var exported = export_lgbm_multiclass_model(model, feature_names)
    return exported.text.copy()


def export_lgbm_multiclass_model(
    model: MulticlassModel, feature_names: List[String] = []
) raises -> LgbmExport:
    """`dump_lgbm_multiclass_model` with its conversion report.
    EXPERIMENTAL."""
    _check_export_names(feature_names, model.mapper.n_features)
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
    var text = _assemble(
        model.mapper,
        objective_line,
        n_classes,
        n_classes,
        blocks,
        feature_names,
    )
    var base_folded = False
    for k in range(n_classes):
        if model.booster.base_scores[k] != 0.0:
            base_folded = True
    var report = LgbmExportReport(
        len(trees),
        n_classes,
        objective_line.copy(),
        # Softmax reads no scalar parameter, so nothing is dropped.
        String(""),
        _n_categorical(model.mapper),
        model.booster.learning_rate != 1.0,
        base_folded,
        len(feature_names),
    )
    return LgbmExport(text^, report^)


def save_lgbm_model(
    model: Model, path: String, feature_names: List[String] = []
) raises:
    """Write a single-output `Model` to `path` in LightGBM's text format."""
    var text = dump_lgbm_model(model, feature_names)
    with open(path, "w") as f:
        f.write(text)


def save_lgbm_multiclass_model(
    model: MulticlassModel, path: String, feature_names: List[String] = []
) raises:
    """Write a `MulticlassModel` to `path` in LightGBM's text format."""
    var text = dump_lgbm_multiclass_model(model, feature_names)
    with open(path, "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# The narrow experimental surface
# ---------------------------------------------------------------------------
#
# Everything above converts between a LightGBM string and a live `Model`. The
# four entry points below are what a caller outside Mojo is meant to reach
# for, and all four are file-to-file, deliberately:
#
#   * they need no model handle to cross a language boundary, so the Python
#     and C facades stay thin;
#   * the *product* of an import is a mojotrees model file, which keeps
#     serialize.mojo the format of record and puts the imported model through
#     native serialization -- category tables, node covers, missing-bin
#     reservations and all -- before anything else can use it;
#   * an unsupported file is answered with a sentence rather than an
#     exception, which is what a caller offering "can I load this?" needs.


def lgbm_interop_status() -> String:
    """How far this interop has been validated. See `LGBM_INTEROP_STATUS`.

    Callers surfacing LightGBM interop are asked to show this verbatim rather
    than paraphrase it: it is the one sentence that says the feature is an
    experiment, and it changes when the differential fixtures run.
    """
    return LGBM_INTEROP_STATUS.copy()


def lgbm_unsupported_reason(text: String) -> String:
    """Why this LightGBM model string cannot be converted, or an empty string
    when it can.

    Never raises, which is the point: it is the query a caller makes *before*
    committing to a conversion, and it is answered by running the conversion
    and reporting what it said. Running it is what makes the answer
    trustworthy -- there is no second list of supported constructs here to
    drift out of step with the converter.
    """
    try:
        var kind = lgbm_text_kind(text)
        if kind == "multiclass":
            _ = import_lgbm_multiclass_model(text)
        else:
            _ = import_lgbm_model(text)
    except e:
        return String(e)
    return String("")


def lgbm_file_unsupported_reason(path: String) -> String:
    """`lgbm_unsupported_reason` for a file, including an unreadable one."""
    try:
        return lgbm_unsupported_reason(open(path, "r").read())
    except e:
        return String(e)


def import_lgbm_file(
    lgbm_path: String, model_path: String
) raises -> LgbmImportReport:
    """Convert a LightGBM model file into a mojotrees model file.

    EXPERIMENTAL. This is the supported shape of an import: the result is
    written in mojotrees's own versioned format (serialize.mojo), which is
    the format `load_model` reads and the only one this project persists
    models in. LightGBM's format is an interchange, never a substitute for
    it.

    Passing the converted model through `save_model` on the way out is not
    incidental. It is what checks that everything the conversion produced --
    the synthesized bin edges, the category tables, the per-node missing
    reservations, the node covers -- is state the native format actually
    carries, rather than something that lives only for as long as the
    converting process does.

    Which of the two loaders the file needs is decided from the file, the
    same way `serialize.model_file_kind` decides it for a native one.

    The LightGBM header's `feature_names` cross into the native file's own
    feature-name header when there is one name per feature; anything else is
    dropped whole, since a name on the wrong column is worse than no name.
    """
    var text = open(lgbm_path, "r").read()
    if lgbm_text_kind(text) == "multiclass":
        var multi = import_lgbm_multiclass_model(text)
        save_multiclass_model(
            multi.model, model_path, multi.report.feature_names
        )
        return multi.report.copy()
    var single = import_lgbm_model(text)
    save_model(single.model, model_path, single.report.feature_names)
    return single.report.copy()


def export_lgbm_file(
    model_path: String, lgbm_path: String
) raises -> LgbmExportReport:
    """Convert a mojotrees model file into a LightGBM model file.

    EXPERIMENTAL. The reverse of `import_lgbm_file`, and the reason it takes
    a native path rather than a live model is the same: the native file is
    where a mojotrees model lives, so an export is a conversion between two
    files and needs nothing else. `serialize.model_file_kind` picks the
    loader.

    The returned report says what could not be carried across, including
    whether the LightGBM file's predictions are bit-identical to the native
    model's or equal only to within a few units in the last place.
    """
    # The names live in the file's header, not on the loaded `Model`, so
    # they are read from the same path rather than reconstructed.
    var names = load_feature_names(model_path)
    if model_file_kind(model_path) == "multiclass":
        var multi = load_multiclass_model(model_path)
        var multi_out = export_lgbm_multiclass_model(multi, names)
        with open(lgbm_path, "w") as f:
            f.write(multi_out.text)
        return multi_out.report.copy()
    var single = load_model(model_path)
    var single_out = export_lgbm_model(single, names)
    with open(lgbm_path, "w") as f:
        f.write(single_out.text)
    return single_out.report.copy()
