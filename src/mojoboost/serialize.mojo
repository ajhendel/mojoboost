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
- v2 likewise carries optional categorical sections, written only when the
  model actually has categorical features (see categorical.mojo). A
  `categorical` section after the mapper holds the per-feature flags and the
  fitted category tables, without which a loaded model could not bin a raw
  category code. Each tree that has a categorical node then carries a `cat`
  section holding its per-node set offsets and its bitset pool. A model with
  no categorical features writes neither section and so serializes to exactly
  the bytes it did before they existed; a file without them loads as fully
  numerical.
- v3: adds per-node covers, the training row counts every grower already had
  and now records (see `Tree.count`). They are the background weighting exact
  feature contributions condition on (see contrib.mojo), which cannot be
  recovered from a fitted tree, so they have to travel with it. Unlike the v2
  additions this section is unconditional: every tree has covers, so every v3
  tree writes them. v1 and v2 files still load and predict exactly as before;
  their trees simply carry no covers, and asking such a model for feature
  contributions raises rather than guessing at them.
- v4: adds per-node split gains (`Tree.split_gain`), and gives covers a
  presence flag. Gains are what a node's split earned when it was taken, the
  one fact model inspection and gain importance need and the one fact a
  fitted tree cannot recompute: the gradient sums it was scored from are gone
  by the time the tree exists. Before v4 they were dropped, so every loaded
  model reported zero gain importance and every dump reported
  `has_split_gain: false`. The flag on covers fixes a round trip v3 could not
  make: a model loaded from a v1 or v2 file has no covers, and writing its
  zeros as if they were covers produced a file that v3's own reader rejected.
  v4 records "no covers here" instead, so re-saving an old model is
  lossless in the only sense available to it.
- v4 also carries an optional `feature_names` section, written only when a
  caller passes names to `save_model`. A `Model` holds no names, so they are
  the writer's to supply and the reader's to hand back
  (`load_feature_names`); a file written without them loads exactly as
  before, and a consumer falls back to `Column_0`, `Column_1`, ... as it
  always did. Names are stored one whitespace-free token each, with the five
  characters that would break the token stream escaped (see `_escape_name`).

v1, v2, v3, and v4 files all load and predict identically. What an older
file cannot carry shows up as an absence a consumer can test for (no gains,
no covers, no names) rather than as a wrong number.

Training-time knobs that only shaped which trees were grown (num_leaves,
regularization, interaction constraints, subsampling) are deliberately absent:
they cannot be checked against a loaded model and are not needed to evaluate
it. Monotonic constraints are the exception because they are a property the
trees satisfy, which a consumer may need to know and cannot recover; so, as
of v4, are split gains, which are a property of how the tree was grown that
nothing downstream can rederive.
"""

from std.memory import bitcast

from .binning import BinMapper, no_missing_bins
from .categorical import CAT_BITSET_WORDS, CategoricalSpec
from .boosting import Booster, MulticlassBooster
from .monotone import MonotoneConstraints
from .model import Model, MulticlassModel
from .tree import Tree

comptime _MAGIC = "mojoboost"
comptime _VERSION = "v4"

# The version this build writes, as the integer `_read_version` returns.
# `MODEL_FORMAT_VERSION` in model_dump.mojo reports this number to a dump
# consumer and has to track it.
comptime CURRENT_FORMAT_VERSION = 4


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


# The bytes a feature name may not carry literally: the file is a
# whitespace-separated token stream, so a name holding one of the four
# whitespace bytes would be read back as two names, and the backslash is
# what escapes them. `_BYTE_DEL` is not escaped, only refused, along with
# every other control byte (see `_escape_name`).
comptime _BYTE_TAB = 9
comptime _BYTE_LF = 10
comptime _BYTE_CR = 13
comptime _BYTE_SPACE = 32
comptime _BYTE_BACKSLASH = 92
comptime _BYTE_DEL = 127

# An empty name is not a token at all, so it travels as this one.
comptime _EMPTY_NAME_TOKEN = "\\e"


def _name_bytes(name: String) -> List[UInt8]:
    """A name's bytes, materialized so they can be indexed. The escape
    codec has to look at one byte and then at the next, which a `Span`
    walked by iteration cannot do."""
    var out = List[UInt8](capacity=name.byte_length())
    for b in name.as_bytes():
        out.append(b)
    return out^


def _escape_name(name: String) raises -> String:
    """One feature name as a single whitespace-free token.

    Tab, newline, carriage return, space, and the backslash itself become
    two-character escapes; every other byte is copied through, so a name in
    any UTF-8 script survives untouched (no multi-byte sequence contains a
    byte below 128, so scanning bytes cannot cut one in half). Remaining
    control bytes are refused rather than escaped: they are not whitespace,
    so they would survive tokenization, but a name holding one is far more
    likely to be a bug at the caller than an intended name, and a file is
    the wrong place to discover it.
    """
    if name.byte_length() == 0:
        return String(_EMPTY_NAME_TOKEN)
    var bytes = _name_bytes(name)
    var out = String("")
    var start = 0
    for i in range(len(bytes)):
        var b = Int(bytes[i])
        var escape = String("")
        if b == _BYTE_SPACE:
            escape = String("\\s")
        elif b == _BYTE_TAB:
            escape = String("\\t")
        elif b == _BYTE_LF:
            escape = String("\\n")
        elif b == _BYTE_CR:
            escape = String("\\r")
        elif b == _BYTE_BACKSLASH:
            escape = String("\\\\")
        elif b < _BYTE_SPACE or b == _BYTE_DEL:
            raise Error(
                "feature name contains control byte ",
                b,
                ", which the model file format cannot carry",
            )
        else:
            continue
        out += String(name[byte=start:i])
        out += escape
        start = i + 1
    out += String(name[byte=start:])
    return out^


def _unescape_name(token: String) raises -> String:
    """The name an escaped token stands for, the exact inverse of
    `_escape_name`."""
    if token == _EMPTY_NAME_TOKEN:
        return String("")
    var bytes = _name_bytes(token)
    var out = String("")
    var start = 0
    var i = 0
    while i < len(bytes):
        if Int(bytes[i]) != _BYTE_BACKSLASH:
            i += 1
            continue
        if i + 1 >= len(bytes):
            raise Error("corrupt feature name: token ends in an escape")
        out += String(token[byte=start:i])
        var code = Int(bytes[i + 1])
        if code == 115:  # s
            out += " "
        elif code == 116:  # t
            out += "\t"
        elif code == 110:  # n
            out += "\n"
        elif code == 114:  # r
            out += "\r"
        elif code == _BYTE_BACKSLASH:
            out += "\\"
        else:
            raise Error(
                "corrupt feature name: unknown escape byte ", code
            )
        i += 2
        start = i
    out += String(token[byte=start:])
    return out^


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


def _write_feature_names(mut out: String, names: List[String]) raises:
    """v4: the optional `feature_names` section, or nothing at all when the
    caller passed no names. Skipping it keeps an unnamed model's file
    byte-identical to one written before the section existed."""
    if len(names) == 0:
        return
    out += "feature_names " + String(len(names)) + "\n"
    for f in range(len(names)):
        out += _escape_name(names[f]) + " "
    out += "\n"


def _read_feature_names(mut r: _TokenReader) raises -> List[String]:
    """v4: the optional `feature_names` section. Absent (every v1, v2, and
    v3 file, and any model saved without names) means the model carries no
    names, which is what `Column_0`, `Column_1`, ... is for."""
    if r.peek() != "feature_names":
        return List[String]()
    _ = r.next()
    var n = r.next_int()
    if n < 1:
        raise Error("corrupt feature_names section: nonpositive count")
    var names = List[String](capacity=n)
    for _ in range(n):
        names.append(_unescape_name(r.next()))
    return names^


def _check_feature_names(names: List[String], n_features: Int) raises:
    """A names section has to name this model's features. A file whose
    names and mapper disagree is corrupt, and guessing which one is right
    would put wrong names on a dump."""
    if len(names) != 0 and len(names) != n_features:
        raise Error(
            "corrupt model file: feature_names has ",
            len(names),
            " entries for ",
            n_features,
            " features",
        )


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


def _has_split_gains(tree: Tree) -> Bool:
    """Whether this tree carries the gains its splits earned.

    The per-tree form of `has_split_gains` in model_dump.mojo, stated the
    same way: a split is only ever taken for a positive gain, so one
    positive gain settles it. False for a tree grown with no splits at all,
    and for one loaded from a file written before v4, whose gains are all
    zero because the format dropped them. Kept here rather than imported so
    the writer depends on nothing but `Tree`.
    """
    if len(tree.split_gain) != len(tree.feature):
        return False
    for i in range(len(tree.split_gain)):
        if tree.split_gain[i] > 0.0:
            return True
    return False


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
        # v3: node covers, one per node. Stored as raw bit patterns like
        # every other float so a reloaded tree explains bit-identically to
        # the trained one. v4 puts a presence flag in front of them: a tree
        # loaded from a v1 or v2 file has none, and writing its zeros as
        # covers produced a file the reader then rejected.
        var covers = tree.has_node_counts()
        out += "counts " + ("1\n" if covers else "0\n")
        if covers:
            for i in range(n_nodes):
                out += _f64_to_token(tree.count[i]) + " "
            out += "\n"
        # v4: per-node split gains, behind the same kind of flag. A leaf's
        # gain is 0.0 and so is every node's on a tree that came from a file
        # written before v4, which is what "no gains here" records.
        var gains = _has_split_gains(tree)
        out += "gains " + ("1\n" if gains else "0\n")
        if gains:
            for i in range(n_nodes):
                out += _f64_to_token(tree.split_gain[i]) + " "
            out += "\n"
        # v2: category sets, written only for a tree that has a categorical
        # node so purely numerical trees keep their original bytes.
        if len(tree.cat_bitset) > 0:
            out += "cat " + String(len(tree.cat_bitset)) + "\n"
            for i in range(n_nodes):
                out += String(tree.cat_offset[i]) + " "
            out += "\n"
            for i in range(len(tree.cat_bitset)):
                out += String(tree.cat_bitset[i]) + " "
            out += "\n"


def _write_categorical(mut out: String, cats: CategoricalSpec):
    """Write the fitted category tables, or nothing at all when no feature is
    categorical. Skipping the section keeps numerical models' files
    unchanged."""
    if not cats.any_categorical():
        return
    out += (
        "categorical "
        + String(len(cats.is_categorical))
        + " "
        + String(len(cats.codes))
        + "\n"
    )
    for f in range(len(cats.is_categorical)):
        out += ("1 " if cats.is_categorical[f] else "0 ")
    out += "\n"
    for i in range(len(cats.codes)):
        out += String(cats.codes[i]) + " "
    out += "\n"
    for i in range(len(cats.offsets)):
        out += String(cats.offsets[i]) + " "
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


def save_model(
    model: Model, path: String, feature_names: List[String] = []
) raises:
    """Write a fitted model to `path` in the current mojoboost text format.

    `feature_names` is optional and travels with the model when given, so a
    consumer that loads the file can name its features instead of falling
    back to `Column_0`, `Column_1`, ... A `Model` carries no names of its
    own, which is why they are a parameter here rather than a field there;
    passing a list of the wrong length is refused rather than silently
    dropped, since a wrong name is worse than no name.
    """
    _check_feature_names(feature_names, model.mapper.n_features)
    var out = String("")
    out += _MAGIC
    out += " "
    out += _VERSION
    out += "\n"

    _write_feature_names(out, feature_names)
    out += "objective " + String(model.booster.objective) + "\n"
    out += (
        "learning_rate " + _f64_to_token(model.booster.learning_rate) + "\n"
    )
    out += "base_score " + _f64_to_token(model.booster.base_score) + "\n"

    _write_mapper(out, model.mapper)
    _write_categorical(out, model.mapper.cats)
    _write_monotone(out, model.booster.monotone)
    _write_trees(out, model.booster.trees)

    with open(path, "w") as f:
        f.write(out)


def save_multiclass_model(
    model: MulticlassModel, path: String, feature_names: List[String] = []
) raises:
    """Write a fitted multiclass model to `path` in the current mojoboost
    text format. Trees keep their round-major order, one per class per
    round, which is what recovers the iteration a tree belongs to.
    `feature_names` behaves exactly as it does in `save_model`."""
    _check_feature_names(feature_names, model.mapper.n_features)
    var out = String("")
    out += _MAGIC
    out += " "
    out += _VERSION
    out += "\n"

    _write_feature_names(out, feature_names)
    out += "multiclass " + String(model.booster.n_classes) + "\n"
    out += (
        "learning_rate " + _f64_to_token(model.booster.learning_rate) + "\n"
    )
    out += "base_scores"
    for k in range(model.booster.n_classes):
        out += " " + _f64_to_token(model.booster.base_scores[k])
    out += "\n"

    _write_mapper(out, model.mapper)
    _write_categorical(out, model.mapper.cats)
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
    var cats = _read_categorical(r, n_features, n_bins)
    return BinMapper(
        edges^, offsets^, n_features, n_bins, cats^, missing_bin^,
    )


def _read_categorical(
    mut r: _TokenReader, n_features: Int, n_bins: Int
) raises -> CategoricalSpec:
    """Read the optional `categorical` section. Absent (v1 files, and any
    model with no categorical feature) means every feature is numerical."""
    if r.peek() != "categorical":
        return CategoricalSpec.all_numerical(n_features)
    _ = r.next()
    var n_flags = r.next_int()
    var n_codes = r.next_int()
    if n_flags != n_features or n_codes < 0:
        raise Error("corrupt categorical header")
    var flags = List[Bool](capacity=n_features)
    for _ in range(n_features):
        flags.append(r.next_int() != 0)
    var codes = List[Int](capacity=n_codes)
    for _ in range(n_codes):
        codes.append(r.next_int())
    var offsets = List[Int](capacity=n_features + 1)
    for _ in range(n_features + 1):
        offsets.append(r.next_int())
    if offsets[0] != 0 or offsets[n_features] != n_codes:
        raise Error("corrupt categorical offsets")
    for f in range(n_features):
        if offsets[f + 1] < offsets[f]:
            raise Error("corrupt categorical offsets")
        var n_cat = offsets[f + 1] - offsets[f]
        if n_cat >= n_bins:
            raise Error("corrupt categorical: more categories than bins")
        if n_cat > 0 and not flags[f]:
            raise Error("corrupt categorical: table on a numerical feature")
        # Category codes must be ascending within a feature for `bin_of`'s
        # binary search to be correct.
        for i in range(offsets[f] + 1, offsets[f + 1]):
            if codes[i] <= codes[i - 1]:
                raise Error("corrupt categorical: codes are not ascending")
    return CategoricalSpec(flags^, codes^, offsets^)


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
        # v3: node covers. A v1 or v2 tree has none, which leaves `count`
        # empty and makes the loaded model refuse to produce feature
        # contributions (see contrib.mojo) while predicting as it always did.
        # v4 writes a presence flag first, so a tree that never had covers
        # says so instead of writing zeros its own reader would reject.
        var count = List[Float64](capacity=n_nodes)
        var has_counts = version == 3
        if version >= 4:
            if r.next() != "counts":
                raise Error("expected 'counts'")
            has_counts = r.next_int() != 0
        if has_counts:
            for _ in range(n_nodes):
                var c = r.next_f64()
                if not c > 0.0:
                    raise Error(
                        "corrupt tree: node cover must be positive"
                    )
                count.append(c)
        # v4: per-node split gains, behind the same kind of flag. Unlike a
        # cover a gain is not checked for sign: a leaf's is 0.0, and the
        # arithmetic that produces one can land at or just below zero. NaN
        # is refused, because it would poison every gain importance sum a
        # consumer takes over these.
        var split_gain = List[Float64](capacity=n_nodes)
        if version >= 4:
            if r.next() != "gains":
                raise Error("expected 'gains'")
            if r.next_int() != 0:
                for _ in range(n_nodes):
                    var g = r.next_f64()
                    if g != g:
                        raise Error("corrupt tree: split gain is NaN")
                    split_gain.append(g)
        if len(split_gain) != n_nodes:
            # Every tree written before v4, and every v4 tree that recorded
            # no gain: the nodes exist, so the array has to, and zero is the
            # value `has_split_gains` reads as "this model has none".
            split_gain = List[Float64](capacity=n_nodes)
            split_gain.resize(n_nodes, 0.0)
        # v2: the optional per-tree category sets. Absent means every node of
        # this tree splits numerically.
        var cat_offset = List[Int](capacity=n_nodes)
        var cat_bitset = List[UInt64]()
        if version >= 2 and r.peek() == "cat":
            _ = r.next()
            var n_words = r.next_int()
            if n_words < 0 or n_words % CAT_BITSET_WORDS != 0:
                raise Error("corrupt tree: category bitset size")
            for _ in range(n_nodes):
                var off = r.next_int()
                if off < -1 or off > n_words - CAT_BITSET_WORDS:
                    raise Error("corrupt tree: category set offset")
                if off >= 0 and off % CAT_BITSET_WORDS != 0:
                    raise Error("corrupt tree: category set offset")
                cat_offset.append(off)
            for _ in range(n_words):
                cat_bitset.append(_parse_u64(r.next()))
        else:
            cat_offset.resize(n_nodes, -1)
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
            Tree(
                feature^, threshold^, left^, right^, value^, split_gain^,
                n_leaves, default_left^, missing_bin^, cat_offset^,
                cat_bitset^, count^,
            )
        )
    return trees^


def _read_version(mut r: _TokenReader) raises -> Int:
    """Check the magic and return the format version as an integer. v1, v2,
    v3, and the current v4 are all readable: v1 files carry no missing-value
    routing, neither v1 nor v2 carries node covers, and none before v4
    carries split gains or feature names."""
    if r.next() != _MAGIC:
        raise Error("not a mojoboost model file")
    var token = r.next()
    if token == "v1":
        return 1
    if token == "v2":
        return 2
    if token == "v3":
        return 3
    if token == "v4":
        return 4
    raise Error("unsupported model format version")


def _read_kind(mut r: _TokenReader) raises -> String:
    """The token after the version, either "objective" or "multiclass"."""
    var kind = r.next()
    if kind != "objective" and kind != "multiclass":
        raise Error("corrupt model file: unknown model kind")
    return kind^


def model_file_kind(path: String) raises -> String:
    """Which loader a saved model needs, "objective" or "multiclass".

    Reads only the file header, so a caller holding a path but not the
    training history can dispatch between `load_model` and
    `load_multiclass_model`. Raises the same errors those two do for a file
    that is not a mojoboost model.
    """
    var content = open(path, "r").read()
    var r = _TokenReader(content)
    _ = _read_version(r)
    _ = _read_feature_names(r)
    return _read_kind(r)


def load_feature_names(path: String) raises -> List[String]:
    """The feature names a saved model carries, or an empty list.

    Empty is the honest answer for every file written before v4 and for any
    model saved without names: the model names nothing, and a consumer
    should fall back to `Column_0`, `Column_1`, ... rather than invent
    names. Reads only the file header, so it costs nothing next to loading
    the model.
    """
    var content = open(path, "r").read()
    var r = _TokenReader(content)
    _ = _read_version(r)
    return _read_feature_names(r)


def load_model(path: String) raises -> Model:
    """Load a model saved by `save_model`.

    Feature names, if the file carries any, are not part of a `Model`; ask
    `load_feature_names` for them. They are still checked here, so a file
    whose names and mapper disagree is refused by the loader rather than by
    whatever reads the names later.
    """
    var content = open(path, "r").read()
    var r = _TokenReader(content)

    var version = _read_version(r)
    var names = _read_feature_names(r)
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
    _check_feature_names(names, mapper.n_features)
    var monotone = _read_monotone(r, mapper.n_features)
    var trees = _read_trees(r, mapper.n_features, version, mapper.n_bins)
    var booster = Booster(
        trees^, base_score, learning_rate, objective, monotone^
    )
    return Model(mapper^, booster^)


def load_multiclass_model(path: String) raises -> MulticlassModel:
    """Load a model saved by `save_multiclass_model`. Feature names are read
    and checked exactly as `load_model` does with them."""
    var content = open(path, "r").read()
    var r = _TokenReader(content)

    var version = _read_version(r)
    var names = _read_feature_names(r)
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
    _check_feature_names(names, mapper.n_features)
    var monotone = _read_monotone(r, mapper.n_features)
    var trees = _read_trees(r, mapper.n_features, version, mapper.n_bins)
    if len(trees) % n_classes != 0:
        raise Error("corrupt model file: tree count not divisible by classes")
    var booster = MulticlassBooster(
        trees^, base_scores^, n_classes, learning_rate, monotone^
    )
    return MulticlassModel(mapper^, booster^)
