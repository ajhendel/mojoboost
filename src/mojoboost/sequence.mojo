"""Bounded-memory batch input: the chunk protocol every streaming path uses.

This is the seam between a caller that holds more rows than it wants resident
and the binning every trainer starts from. A `Sequence` hands out `RawChunk`s
in ascending row order; nothing here ever holds the whole matrix, and nothing
here decides to. The one function that does materialize
(`materialize_dense`) takes an explicit byte budget and raises when the
matrix would exceed it, so a caller that ends up with the whole dataset in
memory asked for it in writing.

What a chunk is
---------------
A `RawChunk` is a contiguous block of rows in the caller's own
representation: dense column-major *within the chunk*
(`values[f * n_rows + r]`, the layout `binning.fit_bins` and
`BinMapper.transform` already read) or a chunk-local `CscMatrix` whose row
indices are relative to the chunk. It carries the row fields that belong to
its rows (label, weight, init score, query ids) and `row_id_base`, the global
id of its first row.

Row identifiers are the driver's, not the source's. A pass assigns global ids
0, 1, 2, ... in delivery order and checks each chunk's `row_id_base` against
the running total, so a source that skips, repeats, or reorders rows is
rejected at the chunk that does it rather than producing a dataset whose rows
mean nothing in particular. That check is what makes "covers every row
exactly once" a property the code enforces instead of a promise the docstring
makes.

Schema negotiation
------------------
The source declares one `ChunkSchema`, and every delivered chunk is checked
against it by `require_chunk_schema`: the feature count, the representation
(dense or sparse), and which optional row fields it brought. A source whose
third chunk stops carrying weights is a source that would silently train a
differently weighted model, so it raises. Feature names and the categorical
declaration are the caller's policy rather than any chunk's, so they are
compared schema against schema by `ChunkSchema.require_compatible` when there
really are two (a cache manifest's and a caller's). `ChunkSchema.fingerprint`
folds all of it into one integer, which is what a cache manifest stores and
re-checks on reopen.

Repeatable iteration
--------------------
Deterministic multi-pass binning (see external_memory.mojo) reads the source
more than once, so a `Sequence` says whether it can be read again
(`is_repeatable`) and, if it can, how (`rewind`). The two in-memory sources
here are trivially repeatable. A source that can only be drained once is not
rejected: `external_memory.spill_source` copies it into a raw cache in a
single pass, and the cache is a repeatable sequence over the same rows. That
is the only way a one-shot source reaches the binner, and it is one extra
pass rather than an approximation.

Cancellation
------------
`CancelToken` is cooperative and single-threaded. Every driver in this module
polls it at chunk boundaries and raises `SEQ_CANCELLED` when it is set, so a
cancelled pass stops within one chunk of the request rather than at the end
of the data. What it deliberately is not: a flag another thread can set. The
token is passed `mut` down one call stack, which is exactly what a
caller-driven loop or a chunk callback can use and exactly what a background
canceller cannot. A cross-thread token needs an atomic flag; nothing here
pretends to have one, and `docs/EXTERNAL_MEMORY.md` records what it would
take.

What this module does not do
----------------------------
It does not parse. Chunks arrive as Float64 values, and where they came from
(a CSV reader, an Arrow batch, a NumPy view) is the caller's business. It
does not bin, cache, or checksum; that is external_memory.mojo. And it does
not convert between dense and sparse: a sparse source stays sparse all the
way into `sparse.fit_bins_csc`, exactly as `raw_data.RawData` keeps it.
"""

from .raw_data import RawData
from .sparse import CscMatrix, CsrMatrix


comptime SEQ_OK = 0
"""No failure."""

comptime SEQ_CANCELLED = 1
"""The pass was cancelled between chunks."""

comptime SEQ_SCHEMA_MISMATCH = 2
"""A chunk disagreed with the schema the pass negotiated."""

comptime SEQ_ROW_ORDER = 3
"""A chunk's `row_id_base` did not continue the pass's row order."""

comptime SEQ_NOT_REPEATABLE = 4
"""A second pass was asked of a source that can only be drained once."""

comptime SEQ_BUDGET = 5
"""A materialization would have exceeded its byte budget."""


def sequence_status_message(code: Int) -> String:
    """Text for a status code, so the two modules that raise these produce
    one wording rather than two."""
    if code == SEQ_OK:
        return "no failure"
    if code == SEQ_CANCELLED:
        return "the pass was cancelled"
    if code == SEQ_SCHEMA_MISMATCH:
        return "a chunk does not match the schema of the pass"
    if code == SEQ_ROW_ORDER:
        return "chunks must cover the rows in order, exactly once"
    if code == SEQ_NOT_REPEATABLE:
        return "this source cannot be read a second time"
    if code == SEQ_BUDGET:
        return "the data would exceed the memory budget it was given"
    return "unrecognized sequence failure"


struct CancelToken(Copyable, Movable):
    """A cooperative stop request, polled at chunk boundaries.

    Single-threaded by construction: the token travels `mut` down one call
    stack, so the only code that can set it is code the pass itself calls
    (a chunk callback, a progress hook) or code that runs between passes.
    `polls` counts the checks, which is how a caller sees that a long pass is
    in fact checking rather than ignoring the token.
    """

    var cancelled: Bool
    var polls: Int

    def __init__(out self):
        self.cancelled = False
        self.polls = 0

    @staticmethod
    def none() -> CancelToken:
        """A token nobody will set, for a caller that has nothing to cancel
        from."""
        return CancelToken()

    def cancel(mut self):
        """Ask the running pass to stop at its next chunk boundary."""
        self.cancelled = True

    def poll(mut self) -> Bool:
        """Whether a stop has been requested, counting the check."""
        self.polls += 1
        return self.cancelled

    def check(mut self) raises:
        """Raise if a stop has been requested. What every driver calls."""
        if self.poll():
            raise Error(sequence_status_message(SEQ_CANCELLED))


def _fnv1a(var h: UInt64, byte: UInt64) -> UInt64:
    """One FNV-1a step. Wrapping multiply is the algorithm, not an
    accident."""
    return (h ^ byte) * 1099511628211


comptime FNV_OFFSET: UInt64 = 14695981039346656037
"""FNV-1a 64-bit offset basis, the seed every fingerprint here starts from."""


def fnv1a_text(var h: UInt64, text: String) -> UInt64:
    """Fold a string's bytes into a running FNV-1a hash."""
    var out = h
    for b in text.as_bytes():
        out = _fnv1a(out, UInt64(Int(b)))
    return out


def fnv1a_int(var h: UInt64, value: Int) -> UInt64:
    """Fold an integer into a running FNV-1a hash, eight bytes, low first.

    Written out rather than reinterpreted so the fingerprint does not depend
    on the host's byte order: a cache written on one machine has to be
    recognized on another.
    """
    var out = h
    var v = UInt64(value)
    for _ in range(8):
        out = _fnv1a(out, v & 255)
        v = v >> 8
    return out


def fnv1a_f64(var h: UInt64, value: Float64) -> UInt64:
    """Fold a float's IEEE-754 bit pattern into a running FNV-1a hash. Bits,
    not digits, so the fold is exact and locale-free."""
    var out = h
    var v = value.to_bits()
    for _ in range(8):
        out = _fnv1a(out, v & 255)
        v = v >> 8
    return out


struct ChunkSchema(Copyable, Movable, Writable):
    """What every chunk of one source has to agree about.

    Feature names are optional (an empty list means the source has none) and
    are compared only when both sides carry them, so a cache written without
    names still reopens against a caller that has them. Everything else is
    compared unconditionally, because each of them changes what the fitted
    binning means.
    """

    var n_features: Int
    var feature_names: List[String]
    var categorical_features: List[Int]
    var is_sparse: Bool
    var has_label: Bool
    var has_weight: Bool
    var has_init_score: Bool
    var has_query_ids: Bool

    def __init__(
        out self,
        n_features: Int,
        var feature_names: List[String] = [],
        var categorical_features: List[Int] = [],
        is_sparse: Bool = False,
        has_label: Bool = True,
        has_weight: Bool = False,
        has_init_score: Bool = False,
        has_query_ids: Bool = False,
    ) raises:
        if n_features < 1:
            raise Error("a schema needs at least one feature")
        if len(feature_names) != 0 and len(feature_names) != n_features:
            raise Error("feature_name must have one name per feature")
        for i in range(len(categorical_features)):
            var f = categorical_features[i]
            if f < 0 or f >= n_features:
                raise Error("categorical feature index out of range")
            for j in range(i):
                if categorical_features[j] == f:
                    raise Error("categorical feature index listed twice")
        self.n_features = n_features
        self.feature_names = feature_names^
        self.categorical_features = categorical_features^
        self.is_sparse = is_sparse
        self.has_label = has_label
        self.has_weight = has_weight
        self.has_init_score = has_init_score
        self.has_query_ids = has_query_ids

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ChunkSchema(n_features=",
            self.n_features,
            ", sparse=",
            self.is_sparse,
            ", categorical=",
            len(self.categorical_features),
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def is_categorical(self, feature: Int) -> Bool:
        """Whether the source declared this feature categorical."""
        for i in range(len(self.categorical_features)):
            if self.categorical_features[i] == feature:
                return True
        return False

    def agrees_with(self, other: ChunkSchema) -> Bool:
        """Whether two schemas describe the same source. Names are compared
        only when both sides have them."""
        if self.n_features != other.n_features:
            return False
        if self.is_sparse != other.is_sparse:
            return False
        if self.has_label != other.has_label:
            return False
        if self.has_weight != other.has_weight:
            return False
        if self.has_init_score != other.has_init_score:
            return False
        if self.has_query_ids != other.has_query_ids:
            return False
        if len(self.categorical_features) != len(other.categorical_features):
            return False
        for i in range(len(self.categorical_features)):
            if not other.is_categorical(self.categorical_features[i]):
                return False
        if len(self.feature_names) != 0 and len(other.feature_names) != 0:
            for i in range(len(self.feature_names)):
                if self.feature_names[i] != other.feature_names[i]:
                    return False
        return True

    def require_compatible(self, other: ChunkSchema) raises:
        """`agrees_with`, but naming the first thing that differs.

        A pass calls this on every chunk after the first. The messages are
        specific because the caller has to fix the source, and "schema
        mismatch" does not say which array to look at.
        """
        if self.n_features != other.n_features:
            raise Error(
                "every chunk must carry the same number of features"
            )
        if self.is_sparse != other.is_sparse:
            raise Error(
                "a source is dense or sparse for its whole length; chunks"
                " cannot change representation"
            )
        if self.has_label != other.has_label:
            raise Error("chunks disagree about whether rows carry a label")
        if self.has_weight != other.has_weight:
            raise Error("chunks disagree about whether rows carry a weight")
        if self.has_init_score != other.has_init_score:
            raise Error(
                "chunks disagree about whether rows carry an init score"
            )
        if self.has_query_ids != other.has_query_ids:
            raise Error("chunks disagree about whether rows carry a query id")
        if len(self.categorical_features) != len(other.categorical_features):
            raise Error(
                "the categorical declaration must be the same for every"
                " chunk: it decides how a column is binned"
            )
        for i in range(len(self.categorical_features)):
            if not other.is_categorical(self.categorical_features[i]):
                raise Error(
                    "the categorical declaration must be the same for every"
                    " chunk: it decides how a column is binned"
                )
        if len(self.feature_names) != 0 and len(other.feature_names) != 0:
            for i in range(len(self.feature_names)):
                if self.feature_names[i] != other.feature_names[i]:
                    raise Error("chunks disagree about the feature names")

    def fingerprint(self) -> UInt64:
        """One integer over every fact `require_compatible` compares.

        A cache manifest stores this and re-checks it on reopen, so a cache
        built from one source cannot be handed to a caller describing
        another. Names are folded in when present, which means a source that
        gains names produces a different fingerprint; that is the intended
        answer, because the names travel into the dataset.
        """
        var h = FNV_OFFSET
        h = fnv1a_text(h, "mojoboost.chunk_schema.v1")
        h = fnv1a_int(h, self.n_features)
        h = fnv1a_int(h, 1 if self.is_sparse else 0)
        h = fnv1a_int(h, 1 if self.has_label else 0)
        h = fnv1a_int(h, 1 if self.has_weight else 0)
        h = fnv1a_int(h, 1 if self.has_init_score else 0)
        h = fnv1a_int(h, 1 if self.has_query_ids else 0)
        h = fnv1a_int(h, len(self.categorical_features))
        for f in range(self.n_features):
            if self.is_categorical(f):
                h = fnv1a_int(h, f)
        h = fnv1a_int(h, len(self.feature_names))
        for i in range(len(self.feature_names)):
            h = fnv1a_text(h, self.feature_names[i])
            h = fnv1a_int(h, 0)
        return h


def _empty_csc(n_rows: Int, n_features: Int) -> CscMatrix:
    """A structurally valid CSC with no stored entries, which is what a dense
    chunk's sparse field holds. The same shape `raw_data._empty_csc` builds,
    written again here rather than imported because that one is private to a
    file this lane does not own."""
    var offsets = List[Int](capacity=n_features + 1)
    offsets.resize(n_features + 1, 0)
    return CscMatrix(
        List[Int](), List[Float64](), offsets^, n_rows, n_features
    )


struct RawChunk(Copyable, Movable, Writable):
    """A contiguous block of rows in the caller's own representation.

    Dense chunks hold `values[f * n_rows + r]`, the layout the binner reads,
    so a chunk goes straight into `BinMapper.transform` with no repacking.
    Sparse chunks hold a chunk-local `CscMatrix`: row indices are relative to
    the chunk, and `row_id_base` is what makes them global.

    The row fields are the rows', not the dataset's: a chunk carries the
    labels of its own rows or none at all, and a pass concatenates them.
    """

    var values: List[Float64]
    var csc: CscMatrix
    var is_sparse: Bool
    var n_rows: Int
    var n_features: Int
    var row_id_base: Int
    var label: List[Float64]
    var weight: List[Float64]
    var init_score: List[Float64]
    var query_ids: List[Int]

    def __init__(
        out self,
        var values: List[Float64],
        var csc: CscMatrix,
        is_sparse: Bool,
        n_rows: Int,
        n_features: Int,
        row_id_base: Int,
        var label: List[Float64],
        var weight: List[Float64],
        var init_score: List[Float64],
        var query_ids: List[Int],
    ):
        self.values = values^
        self.csc = csc^
        self.is_sparse = is_sparse
        self.n_rows = n_rows
        self.n_features = n_features
        self.row_id_base = row_id_base
        self.label = label^
        self.weight = weight^
        self.init_score = init_score^
        self.query_ids = query_ids^

    @staticmethod
    def dense(
        var values: List[Float64],
        n_rows: Int,
        n_features: Int,
        row_id_base: Int = 0,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var init_score: List[Float64] = [],
        var query_ids: List[Int] = [],
    ) raises -> RawChunk:
        """A dense chunk, column-major within the chunk."""
        if n_rows < 1 or n_features < 1:
            raise Error("a chunk must have positive dimensions")
        if len(values) != n_rows * n_features:
            raise Error("chunk values length must equal n_rows * n_features")
        if row_id_base < 0:
            raise Error("row_id_base cannot be negative")
        var out = RawChunk(
            values^,
            _empty_csc(n_rows, n_features),
            False,
            n_rows,
            n_features,
            row_id_base,
            label^,
            weight^,
            init_score^,
            query_ids^,
        )
        out.check_fields()
        return out^

    @staticmethod
    def sparse(
        var csc: CscMatrix,
        row_id_base: Int = 0,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var init_score: List[Float64] = [],
        var query_ids: List[Int] = [],
    ) raises -> RawChunk:
        """A sparse chunk. Row indices are chunk-local and validated as
        CSC."""
        csc.validate()
        if csc.n_rows < 1 or csc.n_features < 1:
            raise Error("a chunk must have positive dimensions")
        if row_id_base < 0:
            raise Error("row_id_base cannot be negative")
        var n_rows = csc.n_rows
        var n_features = csc.n_features
        var out = RawChunk(
            List[Float64](),
            csc^,
            True,
            n_rows,
            n_features,
            row_id_base,
            label^,
            weight^,
            init_score^,
            query_ids^,
        )
        out.check_fields()
        return out^

    @staticmethod
    def from_csr(
        csr: CsrMatrix,
        row_id_base: Int = 0,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var init_score: List[Float64] = [],
        var query_ids: List[Int] = [],
    ) raises -> RawChunk:
        """A row-oriented batch, transposed once into the feature-oriented
        layout the binner reads. O(nnz + n_features) and no densification,
        the same conversion `raw_data.RawData.from_csr` makes."""
        return RawChunk.sparse(
            csr.to_csc(), row_id_base, label^, weight^, init_score^, query_ids^
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "RawChunk(rows=[",
            self.row_id_base,
            ", ",
            self.row_id_base + self.n_rows,
            "), n_features=",
            self.n_features,
            ", sparse=",
            self.is_sparse,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def check_fields(self) raises:
        """Every present row field has one entry per row of this chunk."""
        if len(self.label) != 0 and len(self.label) != self.n_rows:
            raise Error("a chunk's label must have one entry per chunk row")
        if len(self.weight) != 0 and len(self.weight) != self.n_rows:
            raise Error("a chunk's weight must have one entry per chunk row")
        if len(self.init_score) != 0 and len(self.init_score) != self.n_rows:
            raise Error(
                "a chunk's init_score must have one entry per chunk row"
            )
        if len(self.query_ids) != 0 and len(self.query_ids) != self.n_rows:
            raise Error(
                "a chunk's query_ids must have one entry per chunk row"
            )

    def schema(
        self,
        var feature_names: List[String] = [],
        var categorical_features: List[Int] = [],
    ) raises -> ChunkSchema:
        """The schema this chunk implies. Names and the categorical
        declaration are the caller's: a chunk carries values, not policy."""
        return ChunkSchema(
            self.n_features,
            feature_names^,
            categorical_features^,
            self.is_sparse,
            len(self.label) != 0,
            len(self.weight) != 0,
            len(self.init_score) != 0,
            len(self.query_ids) != 0,
        )

    def nnz(self) -> Int:
        """Stored entries for a sparse chunk, every cell for a dense one."""
        if self.is_sparse:
            return self.csc.nnz()
        return self.n_rows * self.n_features

    def value_at(self, row: Int, feature: Int) -> Float64:
        """One cell, chunk-local row index. An absent sparse entry is 0.0,
        which is what it is."""
        if self.is_sparse:
            return self.csc.lookup(row, feature)
        return self.values[feature * self.n_rows + row]

    def row(self, r: Int) raises -> List[Float64]:
        """One row as `n_features` raw values, chunk-local index."""
        if r < 0 or r >= self.n_rows:
            raise Error("chunk row index out of range")
        if self.is_sparse:
            return self.csc.row(r)
        var out = List[Float64](capacity=self.n_features)
        for f in range(self.n_features):
            out.append(self.values[f * self.n_rows + r])
        return out^

    def write_column_into(
        self, feature: Int, mut dest: List[Float64], dest_base: Int
    ) raises:
        """Write this chunk's column `feature` into `dest` starting at
        `dest_base`, one entry per chunk row.

        `dest` is preallocated by the caller (a block gather knows the global
        row count before it starts, see `gather_dense_block`), so this writes
        by index and never grows anything. Absent sparse entries are written
        as 0.0, matching `CscMatrix.lookup` and LightGBM's default
        `zero_as_missing=false`.
        """
        if feature < 0 or feature >= self.n_features:
            raise Error("feature index out of range")
        if dest_base < 0 or dest_base + self.n_rows > len(dest):
            raise Error("column destination range out of bounds")
        if not self.is_sparse:
            var src = feature * self.n_rows
            for r in range(self.n_rows):
                dest[dest_base + r] = self.values[src + r]
            return
        for r in range(self.n_rows):
            dest[dest_base + r] = 0.0
        var lo = self.csc.col_offsets[feature]
        var hi = self.csc.col_offsets[feature + 1]
        for e in range(lo, hi):
            dest[dest_base + self.csc.row_index[e]] = self.csc.values[e]

    def block_csc(self, f_start: Int, f_end: Int) raises -> CscMatrix:
        """This chunk's features `[f_start, f_end)` as their own CSC matrix,
        row indices still chunk-local.

        Column selection on a CSC is a copy of the selected columns' entry
        ranges and nothing else, so this costs the block's stored entries and
        never touches the columns that were left out.
        """
        if not self.is_sparse:
            raise Error("block_csc is for sparse chunks; this one is dense")
        if f_start < 0 or f_end > self.n_features or f_start >= f_end:
            raise Error("feature block out of range")
        var width = f_end - f_start
        var row_index = List[Int]()
        var values = List[Float64]()
        var offsets = List[Int](capacity=width + 1)
        offsets.append(0)
        for f in range(f_start, f_end):
            var lo = self.csc.col_offsets[f]
            var hi = self.csc.col_offsets[f + 1]
            for e in range(lo, hi):
                row_index.append(self.csc.row_index[e])
                values.append(self.csc.values[e])
            offsets.append(len(values))
        return CscMatrix(
            row_index^, values^, offsets^, self.n_rows, width
        )


def require_chunk_schema(schema: ChunkSchema, chunk: RawChunk) raises:
    """Check one delivered chunk against the schema the pass negotiated.

    Only the facts a chunk actually determines are compared: how many
    features it carries, whether it is dense or sparse, and which optional
    row fields it brought. Feature names and the categorical declaration are
    the caller's policy rather than the chunk's, so comparing the pass's copy
    of them against itself would be a check that cannot fail;
    `ChunkSchema.require_compatible` is where those are compared, against a
    second schema (a cache manifest's, a caller's) that could really differ.
    """
    var head = sequence_status_message(SEQ_SCHEMA_MISMATCH) + ": "
    if chunk.n_features != schema.n_features:
        raise Error(
            head + "every chunk must carry the same number of features"
        )
    if chunk.is_sparse != schema.is_sparse:
        raise Error(
            head
            + "a source is dense or sparse for its whole length; chunks"
            " cannot change representation"
        )
    if (len(chunk.label) != 0) != schema.has_label:
        raise Error(
            head + "chunks disagree about whether rows carry a label"
        )
    if (len(chunk.weight) != 0) != schema.has_weight:
        raise Error(
            head + "chunks disagree about whether rows carry a weight"
        )
    if (len(chunk.init_score) != 0) != schema.has_init_score:
        raise Error(
            head + "chunks disagree about whether rows carry an init score"
        )
    if (len(chunk.query_ids) != 0) != schema.has_query_ids:
        raise Error(
            head + "chunks disagree about whether rows carry a query id"
        )
    chunk.check_fields()


struct RowIdRange(Copyable, Movable, Writable):
    """The global rows one chunk covered: `[base, base + count)`.

    A pass records one of these per chunk. Together they are the row identity
    of the dataset: chunk `i` owns global rows `ranges[i]`, and every later
    stage (a cache file, a materialized matrix, a bagging mask) means the
    same rows by the same numbers.
    """

    var base: Int
    var count: Int

    def __init__(out self, base: Int, count: Int):
        self.base = base
        self.count = count

    def write_to(self, mut writer: Some[Writer]):
        writer.write("rows[", self.base, ", ", self.base + self.count, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def contains(self, row_id: Int) -> Bool:
        return row_id >= self.base and row_id < self.base + self.count


def check_row_coverage(ranges: List[RowIdRange], n_rows: Int) raises:
    """The ranges cover `[0, n_rows)` in order, exactly once, with no gaps.

    This is the property row identifiers exist to make checkable, and it is
    checked rather than assumed: a cache whose chunks overlap would train on
    some rows twice, and one with a gap would silently drop rows, and neither
    shows up anywhere else.
    """
    var expected = 0
    for i in range(len(ranges)):
        if ranges[i].count < 1:
            raise Error("a chunk must cover at least one row")
        if ranges[i].base != expected:
            raise Error(sequence_status_message(SEQ_ROW_ORDER))
        expected += ranges[i].count
    if expected != n_rows:
        raise Error(sequence_status_message(SEQ_ROW_ORDER))


struct SequenceStats(Copyable, Movable, Writable):
    """What a pass saw. Counting is free and it is the only way a caller
    learns that a source it thought was 10 million rows delivered 9."""

    var passes: Int
    var chunks: Int
    var rows: Int
    var stored_values: Int
    var ranges: List[RowIdRange]

    def __init__(out self):
        self.passes = 0
        self.chunks = 0
        self.rows = 0
        self.stored_values = 0
        self.ranges = List[RowIdRange]()

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "SequenceStats(passes=",
            self.passes,
            ", chunks=",
            self.chunks,
            ", rows=",
            self.rows,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def begin_pass(mut self):
        """Start a pass: the row counters reset, the pass counter does not."""
        self.passes += 1
        self.chunks = 0
        self.rows = 0
        self.stored_values = 0
        self.ranges = List[RowIdRange]()

    def observe(mut self, chunk: RawChunk) raises:
        """Record one delivered chunk and check that it continues the row
        order this pass has seen so far."""
        if chunk.row_id_base != self.rows:
            raise Error(sequence_status_message(SEQ_ROW_ORDER))
        self.chunks += 1
        self.ranges.append(RowIdRange(chunk.row_id_base, chunk.n_rows))
        self.rows += chunk.n_rows
        self.stored_values += chunk.nnz()


trait Sequence:
    """A source of raw rows, delivered in ascending row order as chunks.

    The contract, in full:

    - chunks are delivered in ascending global row order and cover the rows
      exactly once; `row_id_base` is the global id of a chunk's first row
    - every chunk agrees with `schema()`
    - `has_next` is False exactly when the source is drained
    - `rewind` restarts the source from row 0, and is allowed to raise on a
      source that cannot do it (`is_repeatable` returns False for those)
    - `n_rows_hint` is a hint: 0 means "unknown", and a wrong non-zero hint
      changes nothing but a preallocation

    Nothing in the contract permits a chunk to be produced by reading the
    whole source: an implementation that materializes everything and slices
    it satisfies the letter and defeats the purpose, so the two in-memory
    implementations here are the exception (they were handed a resident
    matrix) and every other implementation should stream.
    """

    def schema(self) -> ChunkSchema:
        """The schema every chunk of this source agrees with."""
        ...

    def n_rows_hint(self) -> Int:
        """Rows the source expects to deliver, or 0 when it does not know."""
        ...

    def is_repeatable(self) -> Bool:
        """Whether `rewind` works. A one-shot source returns False and is
        spilled to a cache before multi-pass binning reads it."""
        ...

    def rewind(mut self) raises:
        """Restart from row 0. Raises on a source that cannot."""
        ...

    def has_next(self) -> Bool:
        """Whether another chunk is available."""
        ...

    def next_chunk(mut self) raises -> RawChunk:
        """The next chunk. Raises when the source is drained."""
        ...


struct ChunkPlan(Copyable, Movable, Writable):
    """How a resident matrix of `n_rows` rows is cut into chunks.

    Row blocks are `chunk_rows` long except the last, which takes the
    remainder, so the plan is a pure function of `(n_rows, chunk_rows)` and
    two runs cut the same rows the same way. That matters more than it looks:
    bin construction reads the source once per feature block, and a source
    that cut differently on the second pass would gather a column out of
    order.
    """

    var n_rows: Int
    var chunk_rows: Int
    var n_chunks: Int

    def __init__(out self, n_rows: Int, chunk_rows: Int) raises:
        if n_rows < 1:
            raise Error("a chunk plan needs at least one row")
        if chunk_rows < 1:
            raise Error("chunk_rows must be positive")
        self.n_rows = n_rows
        self.chunk_rows = chunk_rows
        self.n_chunks = (n_rows + chunk_rows - 1) // chunk_rows

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ChunkPlan(n_rows=",
            self.n_rows,
            ", chunk_rows=",
            self.chunk_rows,
            ", n_chunks=",
            self.n_chunks,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def start(self, i: Int) -> Int:
        return i * self.chunk_rows

    def count(self, i: Int) -> Int:
        var remaining = self.n_rows - i * self.chunk_rows
        if remaining < self.chunk_rows:
            return remaining
        return self.chunk_rows


def _slice_f64(src: List[Float64], start: Int, count: Int) -> List[Float64]:
    """`src[start : start + count]`, or an empty list when `src` is empty,
    which is how an absent optional column stays absent."""
    if len(src) == 0:
        return List[Float64]()
    var out = List[Float64](capacity=count)
    for i in range(count):
        out.append(src[start + i])
    return out^


def _slice_int(src: List[Int], start: Int, count: Int) -> List[Int]:
    """The integer counterpart of `_slice_f64`."""
    if len(src) == 0:
        return List[Int]()
    var out = List[Int](capacity=count)
    for i in range(count):
        out.append(src[start + i])
    return out^


struct MemorySequence(Sequence, Copyable, Movable):
    """A resident dense column-major matrix, handed out in row blocks.

    The reference implementation of the protocol and the adapter every
    existing dense caller reaches the streaming path through: a caller that
    already has `features[f * n_rows + r]` (which is what
    `trainset.Dataset`, `raw_data.RawData.dense`, and the Python array
    conversion all produce) wraps it here and gets chunking, row
    identifiers, schema negotiation, and repeatability without copying the
    matrix into another representation.

    It is repeatable because it never consumed anything; `rewind` is an
    assignment. It is also the one implementation that legitimately holds
    the whole matrix, because it was given one.
    """

    var values: List[Float64]
    var n_rows: Int
    var n_features: Int
    var label: List[Float64]
    var weight: List[Float64]
    var init_score: List[Float64]
    var query_ids: List[Int]
    var plan: ChunkPlan
    var next_index: Int
    var _schema: ChunkSchema

    def __init__(
        out self,
        var values: List[Float64],
        n_rows: Int,
        n_features: Int,
        chunk_rows: Int,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var init_score: List[Float64] = [],
        var query_ids: List[Int] = [],
        var feature_names: List[String] = [],
        var categorical_features: List[Int] = [],
    ) raises:
        if len(values) != n_rows * n_features:
            raise Error("features length must equal n_rows * n_features")
        if len(label) != 0 and len(label) != n_rows:
            raise Error("label length must equal n_rows")
        if len(weight) != 0 and len(weight) != n_rows:
            raise Error("weight length must equal n_rows")
        if len(init_score) != 0 and len(init_score) != n_rows:
            raise Error("init_score length must equal n_rows")
        if len(query_ids) != 0 and len(query_ids) != n_rows:
            raise Error("query_ids length must equal n_rows")
        self._schema = ChunkSchema(
            n_features,
            feature_names^,
            categorical_features^,
            False,
            len(label) != 0,
            len(weight) != 0,
            len(init_score) != 0,
            len(query_ids) != 0,
        )
        self.plan = ChunkPlan(n_rows, chunk_rows)
        self.values = values^
        self.n_rows = n_rows
        self.n_features = n_features
        self.label = label^
        self.weight = weight^
        self.init_score = init_score^
        self.query_ids = query_ids^
        self.next_index = 0

    def schema(self) -> ChunkSchema:
        return self._schema.copy()

    def n_rows_hint(self) -> Int:
        return self.n_rows

    def is_repeatable(self) -> Bool:
        return True

    def rewind(mut self) raises:
        self.next_index = 0

    def has_next(self) -> Bool:
        return self.next_index < self.plan.n_chunks

    def next_chunk(mut self) raises -> RawChunk:
        if not self.has_next():
            raise Error("the sequence is drained; call rewind to read again")
        var i = self.next_index
        self.next_index += 1
        var start = self.plan.start(i)
        var count = self.plan.count(i)
        var block = List[Float64](capacity=count * self.n_features)
        block.resize(count * self.n_features, 0.0)
        for f in range(self.n_features):
            var src = f * self.n_rows + start
            var dst = f * count
            for r in range(count):
                block[dst + r] = self.values[src + r]
        return RawChunk.dense(
            block^,
            count,
            self.n_features,
            start,
            _slice_f64(self.label, start, count),
            _slice_f64(self.weight, start, count),
            _slice_f64(self.init_score, start, count),
            _slice_int(self.query_ids, start, count),
        )


struct CscSequence(Sequence, Copyable, Movable):
    """A resident CSC matrix, handed out in row blocks and kept sparse.

    The sparse counterpart of `MemorySequence`, and the reason chunking does
    not force a representation change: a chunk of a CSC matrix is the entries
    whose row index falls in the block, renumbered to the block, which costs
    the block's stored entries and never allocates a dense cell. The same
    argument `raw_data.RawData.subset` makes for row selection, applied to a
    contiguous ascending selection, so the result is canonical CSC without a
    sort.
    """

    var csc: CscMatrix
    var label: List[Float64]
    var weight: List[Float64]
    var init_score: List[Float64]
    var query_ids: List[Int]
    var plan: ChunkPlan
    var next_index: Int
    var _schema: ChunkSchema

    def __init__(
        out self,
        var csc: CscMatrix,
        chunk_rows: Int,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var init_score: List[Float64] = [],
        var query_ids: List[Int] = [],
        var feature_names: List[String] = [],
        var categorical_features: List[Int] = [],
    ) raises:
        csc.validate()
        var n_rows = csc.n_rows
        if len(label) != 0 and len(label) != n_rows:
            raise Error("label length must equal n_rows")
        if len(weight) != 0 and len(weight) != n_rows:
            raise Error("weight length must equal n_rows")
        if len(init_score) != 0 and len(init_score) != n_rows:
            raise Error("init_score length must equal n_rows")
        if len(query_ids) != 0 and len(query_ids) != n_rows:
            raise Error("query_ids length must equal n_rows")
        self._schema = ChunkSchema(
            csc.n_features,
            feature_names^,
            categorical_features^,
            True,
            len(label) != 0,
            len(weight) != 0,
            len(init_score) != 0,
            len(query_ids) != 0,
        )
        self.plan = ChunkPlan(n_rows, chunk_rows)
        self.csc = csc^
        self.label = label^
        self.weight = weight^
        self.init_score = init_score^
        self.query_ids = query_ids^
        self.next_index = 0

    def schema(self) -> ChunkSchema:
        return self._schema.copy()

    def n_rows_hint(self) -> Int:
        return self.plan.n_rows

    def is_repeatable(self) -> Bool:
        return True

    def rewind(mut self) raises:
        self.next_index = 0

    def has_next(self) -> Bool:
        return self.next_index < self.plan.n_chunks

    def next_chunk(mut self) raises -> RawChunk:
        if not self.has_next():
            raise Error("the sequence is drained; call rewind to read again")
        var i = self.next_index
        self.next_index += 1
        var start = self.plan.start(i)
        var count = self.plan.count(i)
        var n_features = self.csc.n_features
        var row_index = List[Int]()
        var values = List[Float64]()
        var offsets = List[Int](capacity=n_features + 1)
        offsets.append(0)
        for f in range(n_features):
            var lo = self.csc.col_offsets[f]
            var hi = self.csc.col_offsets[f + 1]
            for e in range(lo, hi):
                var r = self.csc.row_index[e]
                # A column's row indices ascend, so the block's entries are
                # contiguous and already in order once renumbered.
                if r >= start and r < start + count:
                    row_index.append(r - start)
                    values.append(self.csc.values[e])
            offsets.append(len(values))
        return RawChunk.sparse(
            CscMatrix(row_index^, values^, offsets^, count, n_features),
            start,
            _slice_f64(self.label, start, count),
            _slice_f64(self.weight, start, count),
            _slice_f64(self.init_score, start, count),
            _slice_int(self.query_ids, start, count),
        )


struct RowFields(Copyable, Movable):
    """The per-row columns a pass concatenates out of the chunks.

    One of these is what a dataset needs besides the matrix, and every one of
    them is `n_rows` long or empty. They are gathered in a pass of their own
    (`gather_row_fields`) because bin construction does not need them and
    reading them once is cheaper than carrying them through every block pass.
    """

    var label: List[Float64]
    var weight: List[Float64]
    var init_score: List[Float64]
    var query_ids: List[Int]
    var n_rows: Int

    def __init__(out self):
        self.label = List[Float64]()
        self.weight = List[Float64]()
        self.init_score = List[Float64]()
        self.query_ids = List[Int]()
        self.n_rows = 0

    def append_chunk(mut self, chunk: RawChunk) raises:
        """Concatenate one chunk's row fields, in row order."""
        chunk.check_fields()
        for i in range(len(chunk.label)):
            self.label.append(chunk.label[i])
        for i in range(len(chunk.weight)):
            self.weight.append(chunk.weight[i])
        for i in range(len(chunk.init_score)):
            self.init_score.append(chunk.init_score[i])
        for i in range(len(chunk.query_ids)):
            self.query_ids.append(chunk.query_ids[i])
        self.n_rows += chunk.n_rows

    def check(self, schema: ChunkSchema) raises:
        """Every field the schema said would be there is there, at full
        length. A source that carried labels for its first chunk and stopped
        is caught by `require_compatible`; this catches the rest."""
        if schema.has_label and len(self.label) != self.n_rows:
            raise Error("the source delivered a label for only some rows")
        if schema.has_weight and len(self.weight) != self.n_rows:
            raise Error("the source delivered a weight for only some rows")
        if schema.has_init_score and len(self.init_score) != self.n_rows:
            raise Error(
                "the source delivered an init score for only some rows"
            )
        if schema.has_query_ids and len(self.query_ids) != self.n_rows:
            raise Error("the source delivered a query id for only some rows")


def memory_sequence_from_raw(
    var raw: RawData,
    chunk_rows: Int,
    var label: List[Float64] = [],
    var weight: List[Float64] = [],
    var init_score: List[Float64] = [],
    var query_ids: List[Int] = [],
    var feature_names: List[String] = [],
    var categorical_features: List[Int] = [],
) raises -> MemorySequence:
    """Stream a dense `raw_data.RawData` without repacking it.

    `RawData` is the type every existing dense caller already builds
    (`Dataset.from_raw`, the sparse constructors, the Python array
    conversion), and its dense buffer is the same column-major layout a
    `MemorySequence` chunks, so this is a move rather than a copy. It raises
    on sparse input rather than densifying it, exactly as
    `RawData.transform_dense` does; `csc_sequence_from_raw` is the other
    half.
    """
    if raw.is_empty():
        raise Error("this RawData holds no matrix")
    if raw.is_sparse:
        raise Error(
            "sparse input streams as a CscSequence; call"
            " csc_sequence_from_raw"
        )
    var n_rows = raw.n_rows
    var n_features = raw.n_features
    return MemorySequence(
        raw.values^,
        n_rows,
        n_features,
        chunk_rows,
        label^,
        weight^,
        init_score^,
        query_ids^,
        feature_names^,
        categorical_features^,
    )


def csc_sequence_from_raw(
    var raw: RawData,
    chunk_rows: Int,
    var label: List[Float64] = [],
    var weight: List[Float64] = [],
    var init_score: List[Float64] = [],
    var query_ids: List[Int] = [],
    var feature_names: List[String] = [],
    var categorical_features: List[Int] = [],
) raises -> CscSequence:
    """Stream a sparse `raw_data.RawData`, keeping it sparse. Raises on dense
    input, as `RawData.transform_sparse` does."""
    if raw.is_empty():
        raise Error("this RawData holds no matrix")
    if not raw.is_sparse:
        raise Error(
            "dense input streams as a MemorySequence; call"
            " memory_sequence_from_raw"
        )
    return CscSequence(
        raw.csc^,
        chunk_rows,
        label^,
        weight^,
        init_score^,
        query_ids^,
        feature_names^,
        categorical_features^,
    )


comptime CATEGORY_KEY_STRIDE = 4294967296
"""`1 << 32`, the stride that packs (column slot, category code) into one
integer key. Category codes are already required to be below `2 ** 31` by
`categorical.fit_categorical_spec`, so the packing is lossless for every
code the binner would accept and the pack itself is where a larger one is
caught."""

comptime MAX_CATEGORY_CODE = 2147483648
"""`1 << 31`, the exclusive ceiling `categorical._MAX_CATEGORY` puts on a
category code. Repeated here rather than imported because that one is
private to a file this lane does not own; the two must stay in step, and the
handoff carries the patch that would share one."""


struct CategoryTally(Copyable, Movable, Writable):
    """Distinct category codes of the declared categorical columns, counted
    while a pass is already reading the data.

    It fits nothing. `binning.fit_bins` fits the category tables, and a
    second table here could disagree with that one. What this answers is the
    question worth answering *before* the expensive passes: how many distinct
    codes each declared column really holds, so a column with a million codes
    and a `max_bin` of 255 is reported as about to lose almost all of them to
    the unknown bin rather than discovered afterwards.

    Distinct codes are held as one sorted list of `slot * 2**32 + code` keys,
    deduplicated after every chunk. Memory is the total cardinality of the
    declared columns plus one chunk's codes, which is the bound a tally can
    have: a column whose cardinality is unbounded is exactly the case this
    exists to report.
    """

    var features: List[Int]
    var keys: List[Int]
    var max_code: List[Int]
    var missing_rows: List[Int]
    var keep: Int

    def __init__(out self, categorical_features: List[Int], keep: Int) raises:
        if keep < 1:
            raise Error("keep must be positive")
        self.features = categorical_features.copy()
        self.keys = List[Int]()
        self.max_code = List[Int]()
        self.missing_rows = List[Int]()
        for _ in range(len(self.features)):
            self.max_code.append(-1)
            self.missing_rows.append(0)
        self.keep = keep

    @staticmethod
    def none() raises -> CategoryTally:
        """A tally of nothing, for a source with no declared categoricals.
        Every method is then a no-op, which is what lets the census pass take
        one unconditionally."""
        return CategoryTally(List[Int](), 1)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "CategoryTally(features=",
            len(self.features),
            ", keep=",
            self.keep,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def observe(mut self, chunk: RawChunk) raises:
        """Fold one chunk's declared categorical columns into the tally.

        Missing is `not (v >= 0.0)`, which also rejects `NaN`, matching
        `CategoricalSpec.bin_of` and both spec fitters. An absent sparse entry
        is the value 0.0, so it is category code 0, matching
        `sparse._distinct_codes_and_counts_csc`.
        """
        if len(self.features) == 0:
            return
        var fresh = List[Int]()
        for i in range(len(self.features)):
            var f = self.features[i]
            if f < 0 or f >= chunk.n_features:
                raise Error("categorical feature index out of range")
            var slot_key = i * CATEGORY_KEY_STRIDE
            if chunk.is_sparse:
                var lo = chunk.csc.col_offsets[f]
                var hi = chunk.csc.col_offsets[f + 1]
                for e in range(lo, hi):
                    var v = chunk.csc.values[e]
                    if not (v >= 0.0):
                        self.missing_rows[i] += 1
                        continue
                    var code = self._code_of(v)
                    if code > self.max_code[i]:
                        self.max_code[i] = code
                    fresh.append(slot_key + code)
                if hi - lo < chunk.n_rows:
                    # The rows with no stored entry hold 0.0, which is code 0.
                    if self.max_code[i] < 0:
                        self.max_code[i] = 0
                    fresh.append(slot_key)
                continue
            var col = f * chunk.n_rows
            for r in range(chunk.n_rows):
                var v = chunk.values[col + r]
                if not (v >= 0.0):
                    self.missing_rows[i] += 1
                    continue
                var code = self._code_of(v)
                if code > self.max_code[i]:
                    self.max_code[i] = code
                fresh.append(slot_key + code)
        if len(fresh) == 0:
            return
        for j in range(len(self.keys)):
            fresh.append(self.keys[j])
        sort(fresh)
        var merged = List[Int]()
        for j in range(len(fresh)):
            if j == 0 or fresh[j] != fresh[j - 1]:
                merged.append(fresh[j])
        self.keys = merged^

    def _code_of(self, v: Float64) raises -> Int:
        """A raw value as a category code, refusing what the binner would
        refuse."""
        if v >= Float64(MAX_CATEGORY_CODE):
            raise Error(
                "categorical feature values must be below 2^31; use smaller"
                " integer codes"
            )
        return Int(v)

    def distinct(self, i: Int) -> Int:
        """Distinct codes seen so far in the tally's `i`-th column."""
        var lo = i * CATEGORY_KEY_STRIDE
        var hi = lo + CATEGORY_KEY_STRIDE
        var n = 0
        for j in range(len(self.keys)):
            if self.keys[j] >= lo and self.keys[j] < hi:
                n += 1
        return n

    def cap_exceeded(self, i: Int) -> Bool:
        """Whether the `i`-th column has more distinct codes than the binning
        can keep, so the rest fall into the unknown bin."""
        return self.distinct(i) > self.keep

    def any_cap_exceeded(self) -> Bool:
        for i in range(len(self.features)):
            if self.cap_exceeded(i):
                return True
        return False

    def report(self) -> String:
        """One line per declared categorical column. Empty when there are
        none, so a caller can print it unconditionally."""
        var out = String("")
        for i in range(len(self.features)):
            out += "feature " + String(self.features[i])
            out += ": " + String(self.distinct(i)) + " distinct codes"
            out += ", max " + String(self.max_code[i])
            out += ", " + String(self.missing_rows[i]) + " missing"
            if self.cap_exceeded(i):
                out += " (over the cap of " + String(self.keep) + ")"
            out += "\n"
        return out^


def gather_row_fields[S: Sequence & Movable](
    mut src: S,
    mut tally: CategoryTally,
    mut cancel: CancelToken,
    mut stats: SequenceStats,
) raises -> RowFields:
    """One pass over the source, keeping only the row fields, the row
    identity, and the category tally.

    This is the census pass. It is what tells the multi-pass binner how many
    rows there are before it allocates a feature block, it is where the row
    ranges every later stage means come from, and it is the one pass that is
    already touching every value, so the categorical tally rides along rather
    than costing a pass of its own. Pass `CategoryTally.none()` when there is
    nothing to tally.
    """
    src.rewind()
    stats.begin_pass()
    var schema = src.schema()
    var fields = RowFields()
    while src.has_next():
        cancel.check()
        var chunk = src.next_chunk()
        require_chunk_schema(schema, chunk)
        stats.observe(chunk)
        tally.observe(chunk)
        fields.append_chunk(chunk)
    cancel.check()
    fields.check(schema)
    check_row_coverage(stats.ranges, fields.n_rows)
    return fields^


def gather_dense_block[S: Sequence & Movable](
    mut src: S,
    f_start: Int,
    f_end: Int,
    n_rows: Int,
    mut cancel: CancelToken,
    mut stats: SequenceStats,
) raises -> List[Float64]:
    """One pass over the source, keeping only features `[f_start, f_end)`.

    Returns the block as a dense column-major matrix of `n_rows` rows, the
    exact layout `binning.fit_bins` takes, so the block goes into the
    unmodified binner rather than into a second copy of the quantile code.
    That is what makes the streaming edges bit-identical to the resident
    ones: they are computed by the same function on the same numbers.

    Memory is `(f_end - f_start) * n_rows * 8` bytes plus one chunk, which is
    the whole point: the caller picks the block width from a budget and pays
    one source pass per block. A sparse source is written out dense here,
    because a dense block of a sparse matrix is what a caller asked for by
    calling this; `gather_sparse_block` is the one that stays sparse.
    """
    if f_start < 0 or f_start >= f_end:
        raise Error("feature block out of range")
    if n_rows < 1:
        raise Error("n_rows must be positive")
    var width = f_end - f_start
    var block = List[Float64](capacity=width * n_rows)
    block.resize(width * n_rows, 0.0)
    src.rewind()
    stats.begin_pass()
    var schema = src.schema()
    if f_end > schema.n_features:
        raise Error("feature block out of range")
    while src.has_next():
        cancel.check()
        var chunk = src.next_chunk()
        require_chunk_schema(schema, chunk)
        stats.observe(chunk)
        if chunk.row_id_base + chunk.n_rows > n_rows:
            raise Error(
                "the source delivered more rows than the census pass counted"
            )
        for f in range(f_start, f_end):
            chunk.write_column_into(
                f, block, (f - f_start) * n_rows + chunk.row_id_base
            )
    cancel.check()
    if stats.rows != n_rows:
        raise Error(
            "the source delivered a different number of rows than the census"
            " pass counted; it is not repeatable"
        )
    check_row_coverage(stats.ranges, n_rows)
    return block^


def gather_sparse_block[S: Sequence & Movable](
    mut src: S,
    f_start: Int,
    f_end: Int,
    n_rows: Int,
    mut cancel: CancelToken,
    mut stats: SequenceStats,
) raises -> CscMatrix:
    """`gather_dense_block` for a sparse source, keeping it sparse.

    The block is assembled by holding each chunk's block-CSC and then walking
    features outer, chunks inner, which is the order that makes the output's
    row indices ascend within a column without a sort. Memory is the block's
    stored entries, twice over while the output is being built, and never a
    dense cell.
    """
    if f_start < 0 or f_start >= f_end:
        raise Error("feature block out of range")
    if n_rows < 1:
        raise Error("n_rows must be positive")
    var width = f_end - f_start
    src.rewind()
    stats.begin_pass()
    var schema = src.schema()
    if not schema.is_sparse:
        raise Error("gather_sparse_block is for sparse sources")
    if f_end > schema.n_features:
        raise Error("feature block out of range")
    var pieces = List[CscMatrix]()
    var bases = List[Int]()
    while src.has_next():
        cancel.check()
        var chunk = src.next_chunk()
        require_chunk_schema(schema, chunk)
        stats.observe(chunk)
        if chunk.row_id_base + chunk.n_rows > n_rows:
            raise Error(
                "the source delivered more rows than the census pass counted"
            )
        bases.append(chunk.row_id_base)
        pieces.append(chunk.block_csc(f_start, f_end))
    cancel.check()
    if stats.rows != n_rows:
        raise Error(
            "the source delivered a different number of rows than the census"
            " pass counted; it is not repeatable"
        )
    check_row_coverage(stats.ranges, n_rows)

    var row_index = List[Int]()
    var values = List[Float64]()
    var offsets = List[Int](capacity=width + 1)
    offsets.append(0)
    for f in range(width):
        for c in range(len(pieces)):
            var lo = pieces[c].col_offsets[f]
            var hi = pieces[c].col_offsets[f + 1]
            for e in range(lo, hi):
                row_index.append(bases[c] + pieces[c].row_index[e])
                values.append(pieces[c].values[e])
        offsets.append(len(values))
    return CscMatrix(row_index^, values^, offsets^, n_rows, width)


def materialize_dense[S: Sequence & Movable](
    mut src: S,
    n_rows: Int,
    max_bytes: Int,
    mut cancel: CancelToken,
    mut stats: SequenceStats,
) raises -> List[Float64]:
    """The whole matrix, dense column-major, if the caller says it fits.

    The escape hatch, deliberately awkward. `max_bytes` is not a hint: the
    check runs before anything is allocated, and a matrix that would exceed
    it raises rather than being loaded, so no code path here can arrive at a
    resident copy of a dataset without a caller having named a number that
    permitted it. Pass a budget you are willing to spend, not `n_rows *
    n_features * 8`.
    """
    var schema = src.schema()
    var bytes = n_rows * schema.n_features * 8
    if max_bytes < 1:
        raise Error("max_bytes must be positive")
    if bytes > max_bytes:
        raise Error(
            sequence_status_message(SEQ_BUDGET)
            + ": the dense matrix needs "
            + String(bytes)
            + " bytes and the budget is "
            + String(max_bytes)
        )
    return gather_dense_block(
        src, 0, schema.n_features, n_rows, cancel, stats
    )


def feature_block_width(
    n_features: Int, n_rows: Int, budget_bytes: Int
) raises -> Int:
    """How many features one bin-construction pass may hold at once.

    The block is `width * n_rows * 8` bytes, so the width is the budget
    divided by a column, clamped into `[1, n_features]`. Width 1 is always
    allowed however small the budget: one column is the floor of exact
    quantile binning, and refusing to fit at all would be worse than
    admitting the floor. The number of source passes bin construction costs
    is `ceil(n_features / width)`, which is the trade the caller is making.
    """
    if n_features < 1:
        raise Error("n_features must be positive")
    if n_rows < 1:
        raise Error("n_rows must be positive")
    if budget_bytes < 1:
        raise Error("budget_bytes must be positive")
    var column_bytes = n_rows * 8
    var width = budget_bytes // column_bytes
    if width < 1:
        width = 1
    if width > n_features:
        width = n_features
    return width
