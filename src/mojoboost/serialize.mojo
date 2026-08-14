"""Model serialization.

Saves and loads a fitted `Model` (BinMapper + Booster) as a plain-text
token stream. Floats are stored as their raw IEEE-754 bit patterns
(decimal UInt64), so save/load round-trips are bit-exact and the format
has no locale or precision pitfalls. The format is versioned; multiclass
ensembles are not covered yet.
"""

from std.memory import bitcast

from .binning import BinMapper
from .boosting import Booster
from .model import Model
from .tree import Tree

comptime _MAGIC = "mojoboost"
comptime _VERSION = "v1"


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

    def next_int(mut self) raises -> Int:
        return Int(self.next())

    def next_f64(mut self) raises -> Float64:
        return _parse_f64(self.next())


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

    out += "mapper "
    out += String(model.mapper.n_features) + " "
    out += String(model.mapper.n_bins) + " "
    out += String(len(model.mapper.edges)) + "\n"
    for i in range(len(model.mapper.edges)):
        out += _f64_to_token(model.mapper.edges[i]) + " "
    out += "\n"
    for i in range(len(model.mapper.edge_offsets)):
        out += String(model.mapper.edge_offsets[i]) + " "
    out += "\n"

    out += "trees " + String(len(model.booster.trees)) + "\n"
    for t in range(len(model.booster.trees)):
        ref tree = model.booster.trees[t]
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

    with open(path, "w") as f:
        f.write(out)


def load_model(path: String) raises -> Model:
    """Load a model saved by `save_model`."""
    var content = open(path, "r").read()
    var r = _TokenReader(content)

    if r.next() != _MAGIC:
        raise Error("not a mojoboost model file")
    if r.next() != _VERSION:
        raise Error("unsupported model format version")

    if r.next() != "objective":
        raise Error("expected 'objective'")
    var objective = r.next_int()
    if r.next() != "learning_rate":
        raise Error("expected 'learning_rate'")
    var learning_rate = r.next_f64()
    if r.next() != "base_score":
        raise Error("expected 'base_score'")
    var base_score = r.next_f64()

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
        trees.append(
            Tree(feature^, threshold^, left^, right^, value^, n_leaves)
        )

    var mapper = BinMapper(edges^, offsets^, n_features, n_bins)
    var booster = Booster(trees^, base_score, learning_rate, objective)
    return Model(mapper^, booster^)
