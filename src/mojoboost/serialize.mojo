"""Model serialization.

Saves and loads fitted models (`Model` and `MulticlassModel`) as a
plain-text token stream. Floats are stored as their raw IEEE-754 bit
patterns (decimal UInt64), so save/load round-trips are bit-exact and
the format has no locale or precision pitfalls. The format is
versioned; the token after the version distinguishes single-output
files ("objective") from multiclass files ("multiclass").

Version history
---------------
- v1: mapper edges and offsets, per-node feature/threshold/children/value.
- v2: adds missing-value routing. The mapper gains its per-feature missing
  bins, and every tree gains per-node `default_left` and `missing_bin`
  arrays, so a reloaded model routes missing values exactly as the trained
  one did. v1 files still load: they describe a model trained without
  missing support, so their mapper reserves no missing bin and none of their
  nodes routes anything.
- v2 also carries an optional `monotone` section between the mapper and the
  trees, holding the monotonic constraint vector the model was trained under
  (see monotone.mojo). It is written only when there is a vector to write, so
  a model trained without constraints serializes to exactly the bytes it did
  before the section existed, and a file without the section loads as
  unconstrained.

Training-time knobs that only shaped which trees were grown (num_leaves,
regularization, interaction constraints, subsampling) are deliberately absent:
they cannot be checked against a loaded model and are not needed to evaluate
it. Monotonic constraints are the exception because they are a property the
trees satisfy, which a consumer may need to know and cannot recover.
"""

from std.memory import bitcast

from .binning import BinMapper, no_missing_bins
from .categorical import CategoricalSpec
from .boosting import Booster, MulticlassBooster
from .monotone import MonotoneConstraints
from .model import Model, MulticlassModel
from .tree import Tree

comptime _MAGIC = "mojoboost"
comptime _VERSION = "v2"


def _f64_to_token(x: Float64) -> String:
    return String(x.to_bits())


def _parse_u64(token: String) raises -> UInt64:
    if token.byte_length() == 0:
        raise Error("empty token where integer expected")
    var out: UInt64 = 0
    for b in token.as_bytes():
        if b < 48 or b > 57:
            raise Error("invalid digit in integer token")
        out = out * 10 + UInt64(Int(b) - 48)
    return out


def _parse_f64(token: String) raises -> Float64:
    return bitcast[DType.float64, 1](
        SIMD[DType.uint64, 1](_parse_u64(token))
    )


struct _TokenReader:
    var tokens: List[String]
    var pos: Int

    def __init__(out self, content: String):
        self.tokens = List[String]()
        for tok in content.split():
            self.tokens.append(String(tok))
        self.pos = 0

    def next(mut self) raises -> String:
        if self.pos >= len(self.tokens):
            raise Error("unexpected end of model file")
        var tok = self.tokens[self.pos].copy()
        self.pos += 1
        return tok^

    def peek(self) -> String:
        """The next token without consuming it, or an empty string at end of
        input. Optional sections are recognized with this."""
        if self.pos >= len(self.tokens):
            return String("")
        return self.tokens[self.pos].copy()

    def next_int(mut self) raises -> Int:
        return Int(self.next())

    def next_f64(mut self) raises -> Float64:
        return _parse_f64(self.next())


def _write_mapper(mut out: String, mapper: BinMapper):
    out += "mapper "
    out += String(mapper.n_features) + " "
    out += String(mapper.n_bins) + " "
    out += String(len(mapper.edges)) + "\n"
    for i in range(len(mapper.edges)):
        out += _f64_to_token(mapper.edges[i]) + " "
    out += "\n"
    for i in range(len(mapper.edge_offsets)):
        out += String(mapper.edge_offsets[i]) + " "
    out += "\n"
    # v2: the bin reserved for missing values of each feature, -1 for none.
    for f in range(mapper.n_features):
        out += String(mapper.missing_bin[f]) + " "
    out += "\n"


def _write_trees(mut out: String, trees: List[Tree]):
    out += "trees " + String(len(trees)) + "\n"
    for t in range(len(trees)):
        ref tree = trees[t]
        var n_nodes = len(tree.feature)
        out += "tree " + String(n_nodes) + " " + String(tree.n_leaves) + "\n"
        for i in range(n_nodes):
            out += String(tree.feature[i]) + " "
        out += "\n"
        for i in range(n_nodes):
            out += String(tree.threshold_bin[i]) + " "
        out += "\n"
        for i in range(n_nodes):
            out += String(tree.left[i]) + " "
        out += "\n"
        for i in range(n_nodes):
            out += String(tree.right[i]) + " "
        out += "\n"
        for i in range(n_nodes):
            out += _f64_to_token(tree.value[i]) + " "
        out += "\n"
        # v2: missing-value routing, one entry per node.
        for i in range(n_nodes):
            out += ("1 " if tree.default_left[i] else "0 ")
        out += "\n"
        for i in range(n_nodes):
            out += String(tree.missing_bin[i]) + " "
        out += "\n"


def _write_monotone(mut out: String, monotone: MonotoneConstraints):
    """Write the monotonic constraint vector, or nothing at all when the model
    carries none. Skipping the section keeps unconstrained models' files
    unchanged."""
    if len(monotone.signs) == 0:
        return
    out += "monotone " + String(len(monotone.signs))
    for f in range(len(monotone.signs)):
        out += " " + String(monotone.signs[f])
    out += "\n"


def _read_monotone(
    mut r: _TokenReader, n_features: Int
) raises -> MonotoneConstraints:
    """Read the optional monotonic constraint section. A file without it
    describes an unconstrained model."""
    if r.peek() != "monotone":
        return MonotoneConstraints()
    _ = r.next()
    var n = r.next_int()
    if n != n_features:
        raise Error(
            "corrupt model file: monotone section has ",
            n,
            " entries for ",
            n_features,
            " features",
        )
    var signs = List[Int](capacity=n)
    for _ in range(n):
        signs.append(r.next_int())
    return MonotoneConstraints.from_signs(signs, n_features)


def save_model(model: Model, path: String) raises:
    """Write a fitted model to `path` in the mojoboost v1 text format."""
    var out = String("")
    out += _MAGIC
    out += " "
    out += _VERSION
    out += "\n"

    out += "objective " + String(model.booster.objective) + "\n"
    out += (
        "learning_rate " + _f64_to_token(model.booster.learning_rate) + "\n"
    )
    out += "base_score " + _f64_to_token(model.booster.base_score) + "\n"

    _write_mapper(out, model.mapper)
    _write_monotone(out, model.booster.monotone)
    _write_trees(out, model.booster.trees)

    with open(path, "w") as f:
        f.write(out)


def save_multiclass_model(model: MulticlassModel, path: String) raises:
    """Write a fitted multiclass model to `path` in the mojoboost v1
    text format. Trees keep their round-major order, one per class per
    round."""
    var out = String("")
    out += _MAGIC
    out += " "
    out += _VERSION
    out += "\n"

    out += "multiclass " + String(model.booster.n_classes) + "\n"
    out += (
        "learning_rate " + _f64_to_token(model.booster.learning_rate) + "\n"
    )
    out += "base_scores"
    for k in range(model.booster.n_classes):
        out += " " + _f64_to_token(model.booster.base_scores[k])
    out += "\n"

    _write_mapper(out, model.mapper)
    _write_monotone(out, model.booster.monotone)
    _write_trees(out, model.booster.trees)

    with open(path, "w") as f:
        f.write(out)


def _read_mapper(mut r: _TokenReader, version: Int) raises -> BinMapper:
    if r.next() != "mapper":
        raise Error("expected 'mapper'")
    var n_features = r.next_int()
    var n_bins = r.next_int()
    var n_edges = r.next_int()
    if n_features < 1 or n_edges < 0:
        raise Error("corrupt mapper header")
    var edges = List[Float64](capacity=n_edges)
    for _ in range(n_edges):
        edges.append(r.next_f64())
    var offsets = List[Int](capacity=n_features + 1)
    for _ in range(n_features + 1):
        offsets.append(r.next_int())
    if offsets[n_features] != n_edges:
        raise Error("corrupt mapper offsets")
    # A v1 mapper predates missing-value support and reserves no bins.
    var missing_bin = no_missing_bins(n_features)
    if version >= 2:
        for f in range(n_features):
            var mb = r.next_int()
            if mb < -1 or mb >= n_bins:
                raise Error("corrupt mapper: missing bin out of range")
            missing_bin[f] = mb
    return BinMapper(
        edges^, offsets^, n_features, n_bins, CategoricalSpec.none(),
        missing_bin^,
    )


def _read_trees(
    mut r: _TokenReader, n_features: Int, version: Int, n_bins: Int
) raises -> List[Tree]:
    if r.next() != "trees":
        raise Error("expected 'trees'")
    var n_trees = r.next_int()
    if n_trees < 0:
        raise Error("corrupt tree count")
    var trees = List[Tree](capacity=n_trees)
    for _ in range(n_trees):
        if r.next() != "tree":
            raise Error("expected 'tree'")
        var n_nodes = r.next_int()
        var n_leaves = r.next_int()
        if n_nodes < 1 or n_leaves < 1:
            raise Error("corrupt tree header")
        var feature = List[Int](capacity=n_nodes)
        var threshold = List[Int](capacity=n_nodes)
        var left = List[Int](capacity=n_nodes)
        var right = List[Int](capacity=n_nodes)
        var value = List[Float64](capacity=n_nodes)
        for _ in range(n_nodes):
            feature.append(r.next_int())
        for _ in range(n_nodes):
            threshold.append(r.next_int())
        for _ in range(n_nodes):
            left.append(r.next_int())
        for _ in range(n_nodes):
            right.append(r.next_int())
        for _ in range(n_nodes):
            value.append(r.next_f64())
        # v1 nodes route no missing values, so they take the defaults.
        var default_left = List[Bool](capacity=n_nodes)
        var missing_bin = List[Int](capacity=n_nodes)
        if version >= 2:
            for _ in range(n_nodes):
                default_left.append(r.next_int() != 0)
            for _ in range(n_nodes):
                var mb = r.next_int()
                if mb < -1 or mb >= n_bins:
                    raise Error("corrupt tree: missing bin out of range")
                missing_bin.append(mb)
        else:
            default_left.resize(n_nodes, False)
            missing_bin.resize(n_nodes, -1)
        for i in range(n_nodes):
            if feature[i] >= n_features:
                raise Error("corrupt tree: feature index out of range")
            if feature[i] >= 0 and (
                left[i] < 0
                or left[i] >= n_nodes
                or right[i] < 0
                or right[i] >= n_nodes
            ):
                raise Error("corrupt tree: child index out of range")
        # Split gains are a training artifact and are not serialized;
        # loaded trees carry zero gains.
        var split_gain = List[Float64](capacity=n_nodes)
        for _ in range(n_nodes):
            split_gain.append(0.0)
        trees.append(
            Tree(
                feature^, threshold^, left^, right^, value^, split_gain^,
                n_leaves, default_left^, missing_bin^,
            )
        )
    return trees^


def _read_version(mut r: _TokenReader) raises -> Int:
    """Check the magic and return the format version as an integer. Both v1
    and the current v2 are readable; v1 files simply carry no missing-value
    routing."""
    if r.next() != _MAGIC:
        raise Error("not a mojoboost model file")
    var token = r.next()
    if token == "v1":
        return 1
    if token == "v2":
        return 2
    raise Error("unsupported model format version")


def _read_kind(mut r: _TokenReader) raises -> String:
    """The token after the version, either "objective" or "multiclass"."""
    var kind = r.next()
    if kind != "objective" and kind != "multiclass":
        raise Error("corrupt model file: unknown model kind")
    return kind^


def load_model(path: String) raises -> Model:
    """Load a model saved by `save_model`."""
    var content = open(path, "r").read()
    var r = _TokenReader(content)

    var version = _read_version(r)
    if _read_kind(r) != "objective":
        raise Error(
            "this is a multiclass model file; use load_multiclass_model"
        )
    var objective = r.next_int()
    if r.next() != "learning_rate":
        raise Error("expected 'learning_rate'")
    var learning_rate = r.next_f64()
    if r.next() != "base_score":
        raise Error("expected 'base_score'")
    var base_score = r.next_f64()

    var mapper = _read_mapper(r, version)
    var monotone = _read_monotone(r, mapper.n_features)
    var trees = _read_trees(r, mapper.n_features, version, mapper.n_bins)
    var booster = Booster(
        trees^, base_score, learning_rate, objective, monotone^
    )
    return Model(mapper^, booster^)


def load_multiclass_model(path: String) raises -> MulticlassModel:
    """Load a model saved by `save_multiclass_model`."""
    var content = open(path, "r").read()
    var r = _TokenReader(content)

    var version = _read_version(r)
    if _read_kind(r) != "multiclass":
        raise Error(
            "this is a single-output model file; use load_model"
        )
    var n_classes = r.next_int()
    if n_classes < 2:
        raise Error("corrupt model file: n_classes must be at least 2")
    if r.next() != "learning_rate":
        raise Error("expected 'learning_rate'")
    var learning_rate = r.next_f64()
    if r.next() != "base_scores":
        raise Error("expected 'base_scores'")
    var base_scores = List[Float64](capacity=n_classes)
    for _ in range(n_classes):
        base_scores.append(r.next_f64())

    var mapper = _read_mapper(r, version)
    var monotone = _read_monotone(r, mapper.n_features)
    var trees = _read_trees(r, mapper.n_features, version, mapper.n_bins)
    if len(trees) % n_classes != 0:
        raise Error("corrupt model file: tree count not divisible by classes")
    var booster = MulticlassBooster(
        trees^, base_scores^, n_classes, learning_rate, monotone^
    )
    return MulticlassModel(mapper^, booster^)
