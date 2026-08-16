"""Model and prepared-table serialization.

Saves and loads fitted models (`Model` and `MulticlassModel`) as a
plain-text token stream. Floats are stored as their raw IEEE-754 bit
patterns (decimal UInt64), so save/load round-trips are bit-exact and
the format has no locale or precision pitfalls. The format is
versioned; the token after the version distinguishes single-output
files ("objective") from multiclass files ("multiclass").

Prepared tables
---------------
`save_dataset` and `load_dataset` write and read a binned
`trainset.Dataset`: the binning, the matrix it produced, and the columns
that describe its rows. That is a **different kind of file** from a model,
with its own magic (`mojotrees-dataset`) and its own version (`d1`), so
neither loader can be handed the other's file and get partway through it;
`file_kind` names all three kinds and `model_file_kind` refuses the third
with the reason. They live here rather than in `trainset.mojo` because the
mapper codec lives here, and a bin edge written two ways is a bin edge that
can disagree with itself: a prepared table carries the model format's mapper
section verbatim, and records which revision of it the file holds.

What a prepared table does not carry is trees, so loading one cannot produce
something predictable, and the raw matrix, so a loaded table cannot be
`subset`. See the section at the end of this module.

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

- v5: two optional sections, either of which makes a file declare v5 and
  neither of which is implied by it. `linear` (linear_tree.mojo) sits after
  the trees. `ctr` (ctr_columns.mojo) sits inside the mapper, right after
  `categorical`, and holds the fitted ordered-target-statistic tables: the
  slot layout, the produced column list, and the class / mean / counter
  counts a model scores its CTR columns from. Those counts are read off the
  target, so a model that lost them would keep every tree referencing a CTR
  column and bin that column as if the feature were absent -- which is why
  the section exists and why writing one used to be refused outright.

  A model with neither section still writes v4. A v4 file read by this build
  loads to `CtrTables.none()` and an inactive linear sidecar, which is what
  their absence means. A v5 file read by a v4 build is *refused*: the reader
  there accepted the `v5` token (linear trees already used it) but then hits
  `ctr` where it expects `trees` and raises `expected 'trees'`, so it stops
  rather than mis-parsing. A build older than that refuses on the version
  token itself.

  What the `ctr` section deliberately does not carry is `predict_lut` and
  `predict_lut_offsets`. They are derived from the counts, and
  `ctr_columns.rebuild_ctr_predict_lut` rebuilds them on load, so the file
  cannot hold a lookup that disagrees with the table it was built from --
  the same reason `read_linear_section` recovers a linear leaf's intercept
  from `Tree.value` instead of storing it twice.

v1 through v5 files all load and predict identically. What an older file
cannot carry shows up as an absence a consumer can test for (no gains, no
covers, no names, no CTR tables) rather than as a wrong number.

Training-time knobs that only shaped which trees were grown (num_leaves,
regularization, interaction constraints, subsampling) are deliberately absent:
they cannot be checked against a loaded model and are not needed to evaluate
it. Monotonic constraints are the exception because they are a property the
trees satisfy, which a consumer may need to know and cannot recover; so, as
of v4, are split gains, which are a property of how the tree was grown that
nothing downstream can rederive.

**`BinMapper.usable` is in that absent set, and it is the one absence that
does not announce itself.** Every other one shows up as a value a consumer can
test for -- no gains, no covers, no names, no CTR tables. `usable` instead
comes back FULL: `BinMapper.__init__` treats an empty list as "nothing was
prefiltered" and fills in every feature, so a mapper whose pool had columns
removed loads with a pool that silently has them back. Two things remove
columns from it today: `feature_pre_filter`, and CatBoost-mode CTR replacement
through `BinMapper.drop_usable`, which drops a categorical column whose CTR
columns stand in for it.

This is INERT as the code stands, and the reason is worth stating so the next
person can check whether it still holds rather than trust this sentence.
`usable` is read at fit time only -- it is the pool
`sampling.select_tree_features` draws a tree's features from, reached through
`BinnedMatrix.usable_features` -- and a loaded model never grows a tree. So no
prediction, dump, or contribution path can observe the reset.

**It stops being inert the moment anything on the load path reads `usable`,
and that is the note this paragraph exists for.** A model whose feature pool
silently resets on load is harmless until one caller reads it, and the caller
who adds that read is the one who needs to know. If you are adding it: write
the section, do not work around the reset. `is_usable` and `usable_features`
are the two accessors to grep for.
"""

from std.memory import bitcast

from .binning import MAX_BINS, BinMapper, BinnedMatrix, no_missing_bins
# The CTR state and its two writer guards. A fitted CTR table is MODEL STATE
# -- it is built from the target -- so a model that lost it keeps every tree
# referencing its CTR columns and bins those columns as if the feature were
# absent: a wrong answer that looks right. `_write_ctr` / `_read_ctr` below
# are the v5 section that carries it for a *model*;
# `check_ctr_serializable` guards that path against the one part of the
# struct the file leaves out, and `check_ctr_dataset_serializable` still
# refuses at the *prepared table* writer, which is not wired.
from .ctr_columns import (
    CtrColumn,
    CtrTables,
    check_ctr_dataset_serializable,
    check_ctr_serializable,
    rebuild_ctr_predict_lut,
)
from .categorical import CAT_BITSET_WORDS, CategoricalSpec
from .boosting import Booster, MulticlassBooster
from .linear_tree import (
    LINEAR_MODEL_FORMAT_VERSION,
    LinearEnsemble,
    linear_section_text,
    read_linear_section,
)
from .monotone import MonotoneConstraints
from .model import Model, MulticlassModel
from .sparse import SparseBinnedMatrix
from .trainset import Dataset
from .tree import Tree
from .validation import check_loaded_tree, check_tree_count, check_tree_header

comptime _MAGIC = "mojotrees"
comptime _VERSION = "v4"
# v5 is declared by a model that carries either of the two optional sections
# an older reader cannot survive: `linear` (linear_tree.mojo), after the
# trees, and `ctr` (ctr_columns.mojo), inside the mapper. Neither is implied
# by the version -- both are recognized by their own opening token, so a v5
# file may hold one, the other, or both. A model with neither still writes
# v4, byte for byte what it wrote before; see
# `linear_tree.linear_model_format_version` and `_model_version`.
comptime _VERSION_5 = "v5"

# Prepared tables are a different kind of file and say so in their first
# token, so no loader can be handed one and start reading it as a model.
# The header then carries the *model* format version, because a prepared
# table reuses the mapper section verbatim and has to record which revision
# of it the file holds.
comptime _DATASET_MAGIC = "mojotrees-dataset"
comptime _DATASET_VERSION = "d1"

# The highest version this build writes and reads, as the integer
# `_read_version` returns. A file declares this only when it carries a
# section that needs it (`_model_version`); the base version below is what a
# model with neither optional section declares, and it is the number
# `MODEL_FORMAT_VERSION` in model_dump.mojo reports to a dump consumer.
# `linear_tree.LINEAR_MODEL_FORMAT_VERSION` is the same 5 named from the
# other side.
comptime CURRENT_FORMAT_VERSION = 5
comptime _BASE_FORMAT_VERSION = 4


def _f64_to_token(x: Float64) -> String:
    return String(x.to_bits())


def _parse_u64(token: String) raises -> UInt64:
    """A u64 from a decimal token, refusing anything that would not fit.

    Every float in the file is stored as its u64 bit pattern, so the whole
    unsigned range is legitimate and the check has to be exact rather than a
    conservative digit cap. Without it a long digit run wraps the
    accumulator silently and `_parse_f64` turns the wrapped value into an
    arbitrary Float64: a corrupt file would load as a valid-looking model
    rather than as an error.
    """
    if token.byte_length() == 0:
        raise Error("empty token where integer expected")
    comptime _U64_MAX = ~UInt64(0)
    var out: UInt64 = 0
    for b in token.as_bytes():
        if b < 48 or b > 57:
            raise Error("invalid digit in integer token")
        var digit = UInt64(Int(b) - 48)
        if out > (_U64_MAX - digit) // 10:
            raise Error(
                "integer token does not fit in 64 bits: " + String(token)
            )
        out = out * 10 + digit
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


def _has_node_covers(tree: Tree) -> Bool:
    """Whether this tree's covers are all there and all usable.

    Stricter than `Tree.has_node_counts`, which asks the root only, and
    deliberately so: the reader refuses a nonpositive cover, so writing a
    partial set would produce a file this module could not read back.
    Nothing can use a partial set either, since `Tree.check_node_counts`
    (and so contrib.mojo) requires every node's. A tree that has some but
    not all is recorded as having none, which is what its consumers already
    treat it as.
    """
    if len(tree.count) != len(tree.feature):
        return False
    for i in range(len(tree.count)):
        if not tree.count[i] > 0.0:
            return False
    return True


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
        var covers = _has_node_covers(tree)
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


# The `ctr` section's own revision, carried in the section rather than in the
# file version for the reason `linear_tree.LINEAR_SECTION_REVISION` gives: a
# later change to what a CTR table stores does not need another model-format
# bump, and a reader that meets a revision it does not know refuses by number
# instead of reading the wrong fields.
comptime _CTR_SECTION_TAG = "ctr"
comptime CTR_SECTION_REVISION = 2
"""Revision 2 adds `slot_codes` / `slot_code_offsets`, 2026-08-16.

**Why this had to be a revision and not an addition.** A CTR bucket used to be
a binned category id, which the reader could recover from the mapper's own
`categorical` section. It is now an index into the source column's COMPLETE,
un-truncated code table, and nothing else in the file carries that table: a
reader that skipped it would map every raw value to bucket 0 and score every
row from the pure prior. That is the failure mode this project cares about
most, wrong rather than failed, so `_read_ctr` refuses a revision it does not
recognize by number instead of parsing on.

No file in the wild carries revision 1 with an active section: the bundle that
produces one was unreachable from every Python entry point until 2026-08-16
(catalog A36 blocker 2), so there is nothing to migrate."""


def _write_ctr(mut out: String, tables: CtrTables) raises:
    """v5: the optional `ctr` section, or nothing at all when the mapper
    carries no fitted tables.

    Skipping it is what keeps a model without CTRs writing exactly the bytes
    it wrote before the section existed, and what lets `_model_version` keep
    declaring v4 for one.

    Field order is the struct's, counts before arrays, one array per line,
    which is `_write_categorical`'s shape. Every float goes through
    `_f64_to_token`, so it travels as its IEEE-754 bit pattern and comes back
    as the same Float64 -- there is no decimal rounding anywhere in this
    format to lose a prior or a target-mean sum to.

    Three of `CtrColumn`'s ten fields are not written. `shift`, `norm` and
    `scale` are what `CtrColumn.__init__` computes from `prior` and
    `ctr_border_count` (`CalcNormalization`, `online_ctr.cpp:102`), so the
    reader runs the same constructor on the same two inputs and gets the same
    bits. `predict_lut` is left out for the same reason at table scope; see
    `ctr_columns.rebuild_ctr_predict_lut`.
    """
    if not tables.is_active():
        return
    # The lookup this section omits has to be the one the counts imply, or the
    # round trip changes the model. Checked here rather than trusted.
    check_ctr_serializable(tables)
    var n_slots = tables.n_slots()
    var n_columns = tables.n_columns()
    out += (
        _CTR_SECTION_TAG
        + " "
        + String(CTR_SECTION_REVISION)
        + " "
        + String(tables.n_base_features)
        + " "
        + String(tables.n_classes)
        + " "
        + _f64_to_token(tables.prior_denom)
        + " "
        + String(n_slots)
        + " "
        + String(n_columns)
        + " "
        + String(len(tables.class_table))
        + " "
        + String(len(tables.mean_counts))
        + " "
        + String(len(tables.counter_counts))
        + "\n"
    )
    _write_int_list(out, tables.source_features)
    _write_int_list(out, tables.slot_buckets)
    # The bucket tables. Revision 2, and the section is unreadable without
    # them: inference maps a raw category code to a bucket through exactly
    # these lists (`CtrTables.bucket_of`), and a model that lost them would
    # score every row from bucket 0.
    _write_int_list(out, tables.slot_code_offsets)
    _write_int_list(out, tables.slot_codes)
    _write_int_list(out, tables.counter_denominator)
    for c in range(n_columns):
        ref col = tables.columns[c]
        out += String(col.slot) + " "
        out += String(col.source_feature) + " "
        out += String(col.ctr_type) + " "
        out += String(col.target_border_idx) + " "
        out += String(col.prior_index) + " "
        out += _f64_to_token(col.prior) + " "
        out += String(col.ctr_border_count) + "\n"
    _write_int_list(out, tables.class_offsets)
    _write_int_list(out, tables.class_table)
    _write_int_list(out, tables.mean_offsets)
    for i in range(len(tables.mean_sums)):
        out += _f64_to_token(tables.mean_sums[i]) + " "
    out += "\n"
    _write_int_list(out, tables.mean_counts)
    _write_int_list(out, tables.counter_offsets)
    _write_int_list(out, tables.counter_counts)


def _check_ctr_offsets(
    offsets: List[Int], total: Int, name: String
) raises:
    """One slot-offset array as `_read_categorical` checks its own: starts at
    zero, never goes backwards, and ends exactly at the array it indexes.

    Without this a corrupt file reaches `ctr_predict_bin`, which slices
    `table[offsets[s] : offsets[s + 1]]`, with offsets that name rows the file
    does not contain.
    """
    if len(offsets) == 0 or offsets[0] != 0:
        raise Error("corrupt ctr section: ", name, " offsets do not start at 0")
    for i in range(1, len(offsets)):
        if offsets[i] < offsets[i - 1]:
            raise Error(
                "corrupt ctr section: ", name, " offsets are not ascending"
            )
    if offsets[len(offsets) - 1] != total:
        raise Error(
            "corrupt ctr section: ",
            name,
            " offsets end at ",
            offsets[len(offsets) - 1],
            " for a table of ",
            total,
        )


def _read_ctr(mut r: _TokenReader, version: Int) raises -> CtrTables:
    """Read the optional `ctr` section. Absent -- every file before v5, and
    every v5 model whose mapper carries no fitted tables -- means
    `CtrTables.none()`, which is what a mapper without CTR columns holds.

    Read inside `_read_mapper`, so the tables arrive attached to the mapper
    that produced them and no caller can forget to reunite the two.
    `BinMapper.attach_ctr` then re-checks the two facts that relate them: the
    base feature count, and that every CTR column's buckets fit `n_bins`.
    """
    if version < CURRENT_FORMAT_VERSION or r.peek() != _CTR_SECTION_TAG:
        return CtrTables.none()
    _ = r.next()
    var revision = r.next_int()
    if revision != CTR_SECTION_REVISION:
        raise Error(
            "unsupported ctr section revision ",
            revision,
            "; this build reads revision ",
            CTR_SECTION_REVISION,
        )
    var tables = CtrTables()
    tables.active = True
    tables.n_base_features = r.next_int()
    tables.n_classes = r.next_int()
    tables.prior_denom = r.next_f64()
    var n_slots = r.next_int()
    var n_columns = r.next_int()
    var n_class = r.next_int()
    var n_mean = r.next_int()
    var n_counter = r.next_int()
    if tables.n_base_features < 1 or tables.n_classes < 1:
        raise Error("corrupt ctr section: nonpositive header count")
    if n_slots < 1 or n_columns < 1:
        raise Error(
            "corrupt ctr section: an active section carries at least one slot"
            " and one column"
        )
    if n_class < 0 or n_mean < 0 or n_counter < 0:
        raise Error("corrupt ctr section: negative table length")

    tables.source_features = _read_int_list(r, n_slots)
    for s in range(n_slots):
        var f = tables.source_features[s]
        if f < 0 or f >= tables.n_base_features:
            raise Error("corrupt ctr section: source feature out of range")
        # `ctr_source_features` emits them ascending and `ctr_slot_columns`
        # reads slots by position, so a file that lost the order would read a
        # different column's category codes into a slot's table.
        if s > 0 and f <= tables.source_features[s - 1]:
            raise Error(
                "corrupt ctr section: source features are not ascending"
            )
    tables.slot_buckets = _read_int_list(r, n_slots)
    for s in range(n_slots):
        if tables.slot_buckets[s] < 1:
            raise Error("corrupt ctr section: nonpositive bucket count")

    # The bucket tables, revision 2. Checked against `slot_buckets` rather
    # than trusted: the two say the same thing twice, and a file where they
    # disagree would map raw values through one and size the statistics with
    # the other, which scores wrong instead of failing.
    tables.slot_code_offsets = _read_int_list(r, n_slots + 1)
    if tables.slot_code_offsets[0] != 0:
        raise Error("corrupt ctr section: code offsets must start at 0")
    for s in range(n_slots):
        var lo = tables.slot_code_offsets[s]
        var hi = tables.slot_code_offsets[s + 1]
        if hi < lo:
            raise Error("corrupt ctr section: code offsets are not ascending")
        if hi - lo + 1 != tables.slot_buckets[s]:
            raise Error(
                "corrupt ctr section: slot ",
                s,
                " carries ",
                hi - lo,
                " category codes but claims ",
                tables.slot_buckets[s],
                " buckets; a bucket is one code plus the reserved bucket 0",
            )
    tables.slot_codes = _read_int_list(
        r, tables.slot_code_offsets[n_slots]
    )
    for s in range(n_slots):
        # `CtrTables.bucket_of` binary searches these, so an unsorted or
        # duplicated table would silently resolve a raw value to the wrong
        # statistic rather than fail.
        for i in range(
            tables.slot_code_offsets[s] + 1, tables.slot_code_offsets[s + 1]
        ):
            if tables.slot_codes[i] <= tables.slot_codes[i - 1]:
                raise Error(
                    "corrupt ctr section: category codes are not strictly"
                    " ascending"
                )
        if tables.slot_code_offsets[s] < tables.slot_code_offsets[s + 1]:
            if tables.slot_codes[tables.slot_code_offsets[s]] < 0:
                raise Error("corrupt ctr section: negative category code")

    tables.counter_denominator = _read_int_list(r, n_slots)

    for _ in range(n_columns):
        var slot = r.next_int()
        if slot < 0 or slot >= n_slots:
            raise Error("corrupt ctr section: column slot out of range")
        var source = r.next_int()
        if source != tables.source_features[slot]:
            raise Error(
                "corrupt ctr section: a column's source feature disagrees with"
                " its slot"
            )
        var ctr_type = r.next_int()
        var target_border_idx = r.next_int()
        var prior_index = r.next_int()
        var prior = r.next_f64()
        var border_count = r.next_int()
        # `n_buckets` is `border_count + 1` and every one of them is a bin
        # index stored in a byte, exactly as the mapper's ceiling above. The
        # lookup is materialized as `UInt8` before `attach_ctr` gets a chance
        # to compare it against `n_bins`, so the byte bound is checked here.
        if border_count < 1 or border_count >= MAX_BINS:
            raise Error(
                "corrupt ctr section: ctr_border_count must be in [1, ",
                MAX_BINS - 1,
                "]; a ctr bucket is stored in a byte",
            )
        tables.columns.append(
            CtrColumn(
                slot,
                source,
                ctr_type,
                target_border_idx,
                prior_index,
                prior,
                border_count,
            )
        )

    tables.class_offsets = _read_int_list(r, n_slots + 1)
    _check_ctr_offsets(tables.class_offsets, n_class, "class")
    tables.class_table = _read_int_list(r, n_class)
    tables.mean_offsets = _read_int_list(r, n_slots + 1)
    _check_ctr_offsets(tables.mean_offsets, n_mean, "mean")
    var sums = List[Float64](capacity=n_mean)
    for _ in range(n_mean):
        sums.append(r.next_f64())
    tables.mean_sums = sums^
    tables.mean_counts = _read_int_list(r, n_mean)
    tables.counter_offsets = _read_int_list(r, n_slots + 1)
    _check_ctr_offsets(tables.counter_offsets, n_counter, "counter")
    tables.counter_counts = _read_int_list(r, n_counter)

    # Derived state, rebuilt rather than read. This is also the last
    # structural check the section gets: `ctr_predict_bin` raises on an
    # unknown CTR type, on a target border index the type does not have, and
    # on a bucket the slot's table is too short for.
    rebuild_ctr_predict_lut(tables)
    return tables^


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
    """Write a fitted model to `path` in the current mojotrees text format.

    `feature_names` is optional and travels with the model when given, so a
    consumer that loads the file can name its features instead of falling
    back to `Column_0`, `Column_1`, ... A `Model` carries no names of its
    own, which is why they are a parameter here rather than a field there;
    passing a list of the wrong length is refused rather than silently
    dropped, since a wrong name is worse than no name.
    """
    check_ctr_serializable(model.mapper.ctr)
    _check_feature_names(feature_names, model.mapper.n_features)
    var out = String("")
    out += _MAGIC
    out += " "
    out += _model_version(model.booster.linear, model.mapper.ctr)
    out += "\n"

    _write_feature_names(out, feature_names)
    out += "objective " + String(model.booster.objective) + "\n"
    out += (
        "learning_rate " + _f64_to_token(model.booster.learning_rate) + "\n"
    )
    out += "base_score " + _f64_to_token(model.booster.base_score) + "\n"

    _write_mapper(out, model.mapper)
    _write_categorical(out, model.mapper.cats)
    _write_ctr(out, model.mapper.ctr)
    _write_monotone(out, model.booster.monotone)
    _write_trees(out, model.booster.trees)
    out += linear_section_text(model.booster.linear)

    with open(path, "w") as f:
        f.write(out)


def _model_version(linear: LinearEnsemble, ctr: CtrTables) -> String:
    """The version token a model file declares: v5 when it carries either
    optional v5 section, v4 when it carries neither.

    Keeping the bump conditional is the point, and it is the same point
    `linear_tree.linear_model_format_version` makes: a model with constant
    leaves and no CTR tables stays readable by a build that knows about
    neither, so the version is a statement about the file rather than about
    the binary that wrote it.
    """
    if linear.is_active() or ctr.is_active():
        return String(_VERSION_5)
    return String(_VERSION)


def _mapper_section_version(mapper: BinMapper) -> Int:
    """The revision of the mapper section a prepared table holds.

    A prepared table reuses the model format's mapper section verbatim and
    records which revision of it the file carries, so the number has to
    follow what was actually written and not what this build is capable of
    writing. A mapper with fitted CTR tables would need v5; every other
    mapper writes the v4 section unchanged, and a build that predates v5
    keeps reading the tables it writes today.

    `save_dataset` refuses a CTR mapper outright
    (`check_ctr_dataset_serializable`), so the v5 arm is unreachable from
    there right now. It is written as the condition rather than as the
    constant 4 so that wiring the dataset path is one change and not two.
    """
    if mapper.has_ctr():
        return CURRENT_FORMAT_VERSION
    return _BASE_FORMAT_VERSION


def save_multiclass_model(
    model: MulticlassModel, path: String, feature_names: List[String] = []
) raises:
    """Write a fitted multiclass model to `path` in the current mojotrees
    text format. Trees keep their round-major order, one per class per
    round, which is what recovers the iteration a tree belongs to.
    `feature_names` behaves exactly as it does in `save_model`."""
    check_ctr_serializable(model.mapper.ctr)
    _check_feature_names(feature_names, model.mapper.n_features)
    var out = String("")
    out += _MAGIC
    out += " "
    out += _model_version(model.booster.linear, model.mapper.ctr)
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
    _write_ctr(out, model.mapper.ctr)
    _write_monotone(out, model.booster.monotone)
    _write_trees(out, model.booster.trees)
    out += linear_section_text(model.booster.linear)

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
    # The byte ceiling, checked on the way in as every binner checks it on
    # the way out. Without it a file can declare a bin count no
    # `BinnedMatrix` can hold, and the two ways to bin a value stop
    # agreeing: `BinMapper.transform` narrows to `UInt8` and wraps modulo
    # 256, while `BinMapper.bin_value` returns the true index. Prediction
    # would then depend on which path a caller took, by a whole leaf. One
    # bin is legal and means a single-bin binning, which is what a converted
    # model whose trees hold no threshold has.
    if n_bins < 1 or n_bins > MAX_BINS:
        raise Error(
            "corrupt mapper: n_bins must be in [1, ",
            MAX_BINS,
            "]; a bin index is stored in a byte",
        )
    var edges = List[Float64](capacity=n_edges)
    for _ in range(n_edges):
        edges.append(r.next_f64())
    var offsets = List[Int](capacity=n_features + 1)
    for _ in range(n_features + 1):
        offsets.append(r.next_int())
    if offsets[0] != 0 or offsets[n_features] != n_edges:
        raise Error("corrupt mapper offsets")
    # Each feature's slice has to be non-negative and to fit the declared
    # bins: k edges give ordinary bins 0..k, so k may not exceed
    # `n_bins - 1`. A feature that also reserves a missing bin uses one
    # fewer, and the reservation is range-checked against `n_bins` below;
    # the loose bound here is enough to keep every producible bin index
    # inside the byte, which is what the ceiling is for.
    for f in range(n_features):
        var width = offsets[f + 1] - offsets[f]
        if width < 0:
            raise Error("corrupt mapper offsets")
        if width > n_bins - 1:
            raise Error(
                "corrupt mapper: feature ",
                f,
                " has ",
                width,
                " edges, more than its ",
                n_bins,
                " bins allow",
            )
    # A v1 mapper predates missing-value support and reserves no bins.
    var missing_bin = no_missing_bins(n_features)
    if version >= 2:
        for f in range(n_features):
            var mb = r.next_int()
            if mb < -1 or mb >= n_bins:
                raise Error("corrupt mapper: missing bin out of range")
            missing_bin[f] = mb
    var cats = _read_categorical(r, n_features, n_bins)
    var mapper = BinMapper(
        edges^, offsets^, n_features, n_bins, cats^, missing_bin^,
    )
    # v5: the fitted CTR tables, which are part of the mapper and are read
    # here so that no loader can reconstruct one without the other.
    # `attach_ctr` is what relates the two: it checks that the tables were
    # planned for this feature count and that their buckets fit these bins.
    var ctr = _read_ctr(r, version)
    mapper.attach_ctr(ctr^)
    return mapper^


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
    check_tree_count(n_trees)
    var trees = List[Tree](capacity=n_trees)
    # The ensemble's running node total, threaded through
    # `check_loaded_tree` so the per-model ceiling is enforced as the file
    # is read and not after it has been allocated.
    var total_nodes = 0
    for _ in range(n_trees):
        if r.next() != "tree":
            raise Error("expected 'tree'")
        var n_nodes = r.next_int()
        var n_leaves = r.next_int()
        check_tree_header(n_nodes, n_leaves)
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
            var zeros = 0
            for _ in range(n_nodes):
                var c = r.next_f64()
                if c == 0.0:
                    zeros += 1
                elif not c > 0.0:
                    raise Error(
                        "corrupt tree: node cover must be positive"
                    )
                count.append(c)
            if zeros == n_nodes:
                # A whole block of zeros is what v3 wrote for a tree that
                # had no covers to write, since v3 had no way to say so:
                # re-saving a v1 or v2 model produced exactly this, and the
                # file then failed to load at all. Read it as the absence it
                # is. A tree with some covers and not others is still
                # refused, because nothing can use a partial set.
                count = List[Float64]()
            elif zeros > 0:
                raise Error("corrupt tree: node cover must be positive")
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
        # Topology (feature and child indices in range, children strictly
        # after their parent so every walk terminates, each node a child
        # exactly once, leaf count matching the header) and the running
        # node ceiling: validation.mojo's rules, which every loader shares.
        total_nodes = check_loaded_tree(
            feature, left, right, n_nodes, n_leaves, n_features, total_nodes
        )
        trees.append(
            Tree(
                feature^, threshold^, left^, right^, value^, split_gain^,
                n_leaves, default_left^, missing_bin^, cat_offset^,
                cat_bitset^, count^,
            )
        )
    return trees^


def _read_version(mut r: _TokenReader) raises -> Int:
    """Check the magic and return the format version as an integer.

    v1 through v5 are all readable: v1 files carry no missing-value routing,
    neither v1 nor v2 carries node covers, none before v4 carries split gains
    or feature names, and none before v5 carries linear leaves or CTR tables.
    An unknown token is refused here rather than read as the nearest known
    version, which is what stops a file written by a newer build from being
    partly parsed by this one.
    """
    if r.next() != _MAGIC:
        raise Error("not a mojotrees model file")
    var token = r.next()
    if token == "v1":
        return 1
    if token == "v2":
        return 2
    if token == "v3":
        return 3
    if token == "v4":
        return _BASE_FORMAT_VERSION
    if token == _VERSION_5:
        # The same 5 `linear_tree` names from its side; asserted by use so the
        # two constants cannot drift apart unnoticed.
        return LINEAR_MODEL_FORMAT_VERSION
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
    that is not a mojotrees model, and names the case where it is a
    mojotrees file of the other kind: a prepared table is not a model, and
    "not a mojotrees model file" would be a misleading way to say so.
    """
    var content = open(path, "r").read()
    var r = _TokenReader(content)
    if r.peek() == _DATASET_MAGIC:
        raise Error(
            "this is a prepared dataset file, not a model; read it with"
            " load_dataset"
        )
    _ = _read_version(r)
    _ = _read_feature_names(r)
    return _read_kind(r)


def file_kind(path: String) raises -> String:
    """What a mojotrees file holds: "objective", "multiclass", or
    "dataset".

    `model_file_kind` answers the first two and refuses the third, which is
    what a caller dispatching between the two model loaders wants. This one
    answers all three, for a caller that does not yet know which kind of
    thing it was handed. Reads only the header either way.
    """
    var content = open(path, "r").read()
    var r = _TokenReader(content)
    if r.peek() == _DATASET_MAGIC:
        return String("dataset")
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
    # A tree splits on a column of the *binned* matrix, which is
    # `n_total_features()` wide: base features plus the CTR columns
    # `BinMapper.bin_row` appends. Equal to `n_features` for every model
    # without CTR tables, so this widens the topology check only where the
    # wider ids are real; feature names and monotone signs stay base-width,
    # because a CTR column has neither.
    var trees = _read_trees(
        r, mapper.n_total_features(), version, mapper.n_bins
    )
    var linear = _read_linear(r, version, trees, mapper.n_features)
    var booster = Booster(
        trees^, base_score, learning_rate, objective, monotone^, linear^
    )
    return Model(mapper^, booster^)


def _read_linear(
    mut r: _TokenReader, version: Int, trees: List[Tree], n_features: Int
) raises -> LinearEnsemble:
    """The optional `linear` section (linear_tree.mojo), present only in a
    v5 file; a file without it leaves the reader where it was and yields
    the inactive sidecar."""
    if version < LINEAR_MODEL_FORMAT_VERSION:
        return LinearEnsemble.inactive()
    var result = read_linear_section(r.tokens, r.pos, trees, n_features)
    r.pos = result.next_pos
    return result.linear.copy()


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
    # The binned width, exactly as in `load_model`; see the note there.
    var trees = _read_trees(
        r, mapper.n_total_features(), version, mapper.n_bins
    )
    if len(trees) % n_classes != 0:
        raise Error("corrupt model file: tree count not divisible by classes")
    var linear = _read_linear(r, version, trees, mapper.n_features)
    var booster = MulticlassBooster(
        trees^, base_scores^, n_classes, learning_rate, monotone^, linear^
    )
    return MulticlassModel(mapper^, booster^)


# -- prepared tables -----------------------------------------------------
#
# A prepared table is a `Dataset`: a fitted binning, the matrix it produced,
# and the columns that describe its rows. Binning is the expensive part of
# starting a run, so writing one out is what lets a second process, a second
# machine, or tomorrow's run skip it.
#
# It is deliberately not a model file and shares no loader with one. A model
# carries trees and predicts; a prepared table carries data and cannot. The
# first token differs, so neither loader can be fed the other's file and get
# partway through it. What the two do share is the mapper section, byte for
# byte and codec for codec, because a bin edge written two ways is a bin
# edge that can disagree with itself.
#
# What a prepared table does not carry is the raw matrix. A table read back
# therefore cannot be `subset`, since bins cannot be refitted from bins, and
# `Dataset.has_raw()` says so rather than the caller finding out from a
# failed subset. `borrowed_binning` travels with the file because it is the
# leakage-relevant fact about the edges: whether they were fitted on these
# rows or inherited from a reference.
#
# The format is the module's text token stream, so a dense table costs one
# decimal token per cell. That is large, and it is the same trade the model
# format already makes: exact round trips and no locale or precision
# pitfalls, at the price of size.


def _write_u8_list(mut out: String, values: List[UInt8]):
    for i in range(len(values)):
        out += String(Int(values[i])) + " "
    out += "\n"


def _write_int_list(mut out: String, values: List[Int]):
    for i in range(len(values)):
        out += String(values[i]) + " "
    out += "\n"


def _read_u8_list(mut r: _TokenReader, n: Int) raises -> List[UInt8]:
    var out = List[UInt8](capacity=n)
    for _ in range(n):
        var v = r.next_int()
        if v < 0 or v > 255:
            raise Error("corrupt prepared dataset: bin value out of range")
        out.append(UInt8(v))
    return out^


def _read_int_list(mut r: _TokenReader, n: Int) raises -> List[Int]:
    var out = List[Int](capacity=n)
    for _ in range(n):
        out.append(r.next_int())
    return out^


def _write_f64_column(mut out: String, name: String, values: List[Float64]):
    """One of a dataset's optional per-row columns. A column the dataset
    does not have is written with a length of zero rather than skipped, so
    the reader never has to guess which of the three is missing."""
    out += "column " + name + " " + String(len(values)) + "\n"
    for i in range(len(values)):
        out += _f64_to_token(values[i]) + " "
    out += "\n"


def _read_f64_column(
    mut r: _TokenReader, name: String
) raises -> List[Float64]:
    if r.next() != "column":
        raise Error("expected 'column'")
    var got = r.next()
    if got != name:
        raise Error(
            "corrupt prepared dataset: expected the ",
            name,
            " column, found ",
            got,
        )
    var n = r.next_int()
    if n < 0:
        raise Error("corrupt prepared dataset: negative column length")
    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(r.next_f64())
    return out^


def save_dataset(dataset: Dataset, path: String) raises:
    """Write a binned `Dataset` to `path` as a prepared table.

    Everything needed to train on the data again without seeing it again:
    the fitted mapper (the same section a model file carries, categorical
    tables and missing-bin reservations included), the binned matrix in
    whichever representation the dataset holds, the binning parameters, the
    feature names, the categorical declaration, and the label, weight,
    init-score, and group columns.

    The raw matrix is not written. A table read back is a binning, not the
    values it was fitted from, so `Dataset.has_raw()` is false on it and
    `subset` raises. Use `Dataset.subset` before saving, not after loading.
    """
    # A model carries its CTR tables as of v5; a prepared table still cannot,
    # and the blocker is the matrix rather than the tables. The guard names
    # it, so the refusal lands on the writer instead of on whoever later
    # tries to read the file back.
    check_ctr_dataset_serializable(dataset.mapper.ctr)
    var out = String("")
    out += _DATASET_MAGIC + " " + _DATASET_VERSION + " "
    out += String(_mapper_section_version(dataset.mapper)) + "\n"
    out += "dataset "
    out += String(dataset.n_rows) + " "
    out += String(dataset.n_features) + " "
    out += String(dataset.max_bin) + " "
    out += String(1 if dataset.use_missing else 0) + " "
    out += String(1 if dataset.is_sparse else 0) + " "
    out += String(1 if dataset.borrowed_binning else 0) + "\n"

    _write_feature_names(out, dataset.feature_names)
    out += "categorical_features "
    out += String(len(dataset.categorical_features)) + "\n"
    _write_int_list(out, dataset.categorical_features)

    _write_mapper(out, dataset.mapper)
    _write_categorical(out, dataset.mapper.cats)

    if dataset.is_sparse:
        out += "sparse "
        out += String(dataset.sparse_data.nnz()) + " "
        out += String(dataset.sparse_data.n_bins) + "\n"
        _write_int_list(out, dataset.sparse_data.row_index)
        _write_u8_list(out, dataset.sparse_data.bin)
        _write_int_list(out, dataset.sparse_data.col_offsets)
        _write_u8_list(out, dataset.sparse_data.default_bin)
        _write_int_list(out, dataset.sparse_data.missing_bin)
    else:
        out += "bins "
        out += String(len(dataset.data.bins)) + " "
        out += String(dataset.data.n_bins) + "\n"
        _write_u8_list(out, dataset.data.bins)
        _write_int_list(out, dataset.data.missing_bin)

    _write_f64_column(out, "label", dataset.label)
    _write_f64_column(out, "weight", dataset.weight)
    _write_f64_column(out, "init_score", dataset.init_score)
    out += "group " + String(len(dataset.group)) + "\n"
    _write_int_list(out, dataset.group)

    with open(path, "w") as f:
        f.write(out)


def load_dataset(path: String) raises -> Dataset:
    """Read a prepared table written by `save_dataset`.

    The mapper and the matrix are checked against each other by
    `Dataset.from_binned_dense` / `from_binned_sparse`, and every column is
    checked against the row count, so a truncated or edited file is refused
    here rather than producing bin indices that mean nothing.

    A model file is refused with the reason, not with "not a mojotrees
    dataset file": it is a mojotrees file, of the other kind.
    """
    var content = open(path, "r").read()
    var r = _TokenReader(content)
    var magic = r.next()
    if magic == _MAGIC:
        raise Error(
            "this is a model file, not a prepared dataset; read it with"
            " load_model or load_multiclass_model"
        )
    if magic != _DATASET_MAGIC:
        raise Error("not a mojotrees prepared dataset file")
    if r.next() != _DATASET_VERSION:
        raise Error("unsupported prepared dataset format version")
    # The mapper section's revision, which is a model-format version because
    # the section is the model format's. `save_dataset` writes the revision
    # the section it produced actually needs (`_mapper_section_version`), so
    # a table written today still declares 4 and an older build still reads
    # it; the upper bound moves with `CURRENT_FORMAT_VERSION` so that a
    # future v5 table is not rejected by the build that can read it.
    var mapper_version = r.next_int()
    if mapper_version < 2 or mapper_version > CURRENT_FORMAT_VERSION:
        raise Error(
            "prepared dataset carries a mapper section this build cannot"
            " read"
        )

    if r.next() != "dataset":
        raise Error("expected 'dataset'")
    var n_rows = r.next_int()
    var n_features = r.next_int()
    var max_bin = r.next_int()
    var use_missing = r.next_int() != 0
    var is_sparse = r.next_int() != 0
    var borrowed_binning = r.next_int() != 0
    if n_rows < 1 or n_features < 1:
        raise Error("corrupt prepared dataset: nonpositive shape")

    var names = _read_feature_names(r)
    if r.next() != "categorical_features":
        raise Error("expected 'categorical_features'")
    var n_categorical = r.next_int()
    if n_categorical < 0 or n_categorical > n_features:
        raise Error("corrupt prepared dataset: categorical count out of range")
    var categorical = _read_int_list(r, n_categorical)

    var mapper = _read_mapper(r, mapper_version)
    _check_feature_names(names, mapper.n_features)
    if mapper.n_features != n_features:
        raise Error(
            "corrupt prepared dataset: the mapper and the header disagree on"
            " the feature count"
        )

    var data = BinnedMatrix(List[UInt8](), 0, n_features, 0)
    var sparse_data = SparseBinnedMatrix(
        List[Int](), List[UInt8](), List[Int](), List[UInt8](), 0, 0, 0
    )
    if is_sparse:
        if r.next() != "sparse":
            raise Error("expected 'sparse'")
        var nnz = r.next_int()
        var n_bins = r.next_int()
        if nnz < 0:
            raise Error("corrupt prepared dataset: negative entry count")
        if n_bins != mapper.n_bins:
            raise Error(
                "corrupt prepared dataset: the matrix and the mapper"
                " disagree on n_bins"
            )
        var row_index = _read_int_list(r, nnz)
        var bin = _read_u8_list(r, nnz)
        var col_offsets = _read_int_list(r, n_features + 1)
        var default_bin = _read_u8_list(r, n_features)
        var missing_bin = _read_int_list(r, n_features)
        sparse_data = SparseBinnedMatrix(
            row_index^,
            bin^,
            col_offsets^,
            default_bin^,
            n_rows,
            n_features,
            n_bins,
            mapper.cats.copy(),
            missing_bin^,
        )
    else:
        if r.next() != "bins":
            raise Error("expected 'bins'")
        var n_cells = r.next_int()
        var n_bins = r.next_int()
        if n_cells != n_rows * n_features:
            raise Error(
                "corrupt prepared dataset: the matrix does not have one bin"
                " per cell"
            )
        if n_bins != mapper.n_bins:
            raise Error(
                "corrupt prepared dataset: the matrix and the mapper"
                " disagree on n_bins"
            )
        var bins = _read_u8_list(r, n_cells)
        var missing_bin = _read_int_list(r, n_features)
        data = BinnedMatrix(
            bins^,
            n_rows,
            n_features,
            n_bins,
            mapper.cats.copy(),
            missing_bin^,
        )

    var label = _read_f64_column(r, "label")
    var weight = _read_f64_column(r, "weight")
    var init_score = _read_f64_column(r, "init_score")
    if r.next() != "group":
        raise Error("expected 'group'")
    var n_queries = r.next_int()
    if n_queries < 0:
        raise Error("corrupt prepared dataset: negative query count")
    var group = _read_int_list(r, n_queries)

    if is_sparse:
        return Dataset.from_binned_sparse(
            mapper^,
            sparse_data^,
            label^,
            weight^,
            group^,
            init_score^,
            names^,
            categorical^,
            max_bin,
            use_missing,
            borrowed_binning,
        )
    return Dataset.from_binned_dense(
        mapper^,
        data^,
        label^,
        weight^,
        group^,
        init_score^,
        names^,
        categorical^,
        max_bin,
        use_missing,
        borrowed_binning,
    )
