"""External-memory dataset construction: bin a source larger than memory.

This is the streaming counterpart of `trainset.Dataset`. It takes a
`sequence.Sequence` and produces an `ExternalDataset`: a fitted `BinMapper`
plus a directory of binned chunk files, together with a manifest that says
what they are, how many rows each covers, and what their bytes hash to. No
step of it holds the raw matrix. The one operation that produces a resident
matrix, `materialize_binned`, takes a byte budget and raises rather than
exceeding it, so a caller cannot arrive at a loaded dataset by accident.

The passes, and why there are several
-------------------------------------
Bin edges are global quantiles. A quantile of a column needs the whole
column, so exact binning of a dataset that does not fit is a question of
which whole thing you are willing to hold. This module holds *a block of
columns*:

1. **Census.** One pass, `sequence.gather_row_fields`. Counts the rows,
   fixes the row identity (chunk `i` owns global rows `[base, base+count)`),
   and concatenates the label, weight, init score, and query ids, which are
   `n_rows` long each and are the caller's data regardless. It also tallies
   the declared categorical columns (`CategoryTally`), so a column with more
   distinct codes than the binning can keep is reported now rather than after
   the expensive passes.

2. **Bin construction**, `ceil(n_features / block_width)` passes. Each pass
   gathers one block of columns for every row and hands it to the *unmodified*
   `binning.fit_bins` (or `sparse.fit_bins_csc`). The edges are therefore
   bit-identical to the resident path by construction rather than by
   agreement: it is the same function on the same numbers, and the only thing
   this module contributes is the order the numbers were collected in.
   `block_width` comes from `sequence.feature_block_width` and a byte budget,
   so the memory ceiling is a parameter and the number of passes is what it
   buys.

3. **Transform.** One pass. Each chunk is binned by the fitted mapper and
   written to its own cache file with a checksum. Binning is per element, so
   a chunk binned on its own equals its slice of the whole matrix binned at
   once, exactly as in `sparse.transform_csc`'s argument for absent entries.

Total: `2 + ceil(n_features / block_width)` reads of the source. Fewer is
possible only by giving up exact quantiles, which this module does not do:
there is no sampled or sketched binner here, and `subsample_for_bin` remains
unimplemented (see `docs/LIGHTGBM_PARITY.md`).

Sources that can only be read once
----------------------------------
A one-shot source is spilled to a raw cache first (`spill_source`), and the
cache is a repeatable `Sequence` over the same chunks. That is one extra pass
and one extra copy on disk, and it is the only way such a source reaches
exact binning. It is never done silently: `build_external_dataset` raises on
a non-repeatable source and names `spill_source` as the fix.

Determinism
-----------
Two builds of the same source with the same parameters produce the same bin
edges, the same chunk boundaries, the same row identifiers, and the same
checksums. Nothing here samples, and nothing here depends on thread count:
the parallelism is inside `fit_bins` and `BinMapper.transform`, both of which
are already documented as bit-identical to their serial paths. The chunk cut
is `sequence.ChunkPlan`, a pure function of the row count and the chunk size.
What is *not* invariant is the chunk size itself: the same data cut into
different chunks produces the same bins and the same model, but a different
set of cache files, and the manifest records the cut so a reopened cache
cannot be mistaken for one built differently.

Cache files
-----------
One file per chunk plus one manifest and one row-field file, all of them
whitespace-separated token streams in the style of `serialize.mojo`: floats
as their IEEE-754 bit patterns, so a round trip is exact and locale-free.
Text costs about four bytes per bin where one would do; that is a deliberate
trade for using the one file primitive this repository has proven
(`open(path, "w")` and `open(path, "r").read()`), and a binary block format
is recorded as future work in `docs/EXTERNAL_MEMORY.md` rather than invented
here. One file per chunk, rather than one file with offsets, is what keeps
every read bounded: reopening chunk 4000 reads chunk 4000.

Cleanup
-------
`ExternalDataset.discard` truncates every file the cache owns and returns
their paths. It truncates rather than unlinks because this module writes
files through `open`, the only filesystem primitive already in use here, and
inventing an unlink path that nothing has run would be a claim rather than a
feature. A truncated cache cannot be read back as data (its manifest is gone
and its chunk files are empty), and the returned paths are what a caller's
own cleanup removes. `docs/EXTERNAL_MEMORY.md` records what changing this to
a real unlink would take.

What can be trained from one
----------------------------
`ExternalCapabilities` is the answer, in code. In short: everything the CPU
and GPU trainers do is supported *through* a materialized binned matrix,
which is `n_rows * n_features` bytes rather than eight times that, and
nothing is supported *without* materializing, because every histogram builder
in the tree takes a whole `BinnedMatrix`. That is a real limit and it is
stated as one; the chunk files, row identifiers, and per-chunk checksums are
exactly the substrate a chunked histogram accumulator would need, and none of
them is a claim that one exists.
"""

from std.memory import bitcast

from .bagging import BaggingParams
from .binning import BinMapper, BinnedMatrix, fit_bins, no_missing_bins
from .boosting import (
    Booster,
    BoosterParams,
    train,
    train_more,
    train_multiclass,
    train_multiclass_more,
)
from .boosting_sparse import train_multiclass_sparse, train_sparse
from .categorical import CategoricalSpec
from .device import CPU_DEVICE, GPU_DEVICE, resolve_device
from .goss import GossParams
from .model import Model, MulticlassModel
from .ranking import (
    RankGroups,
    RankerParams,
    groups_from_query_ids,
    train_ranker,
)
from .raw_data import RawData
from .sequence import (
    FNV_OFFSET,
    SEQ_NOT_REPEATABLE,
    CancelToken,
    CategoryTally,
    ChunkSchema,
    RawChunk,
    RowFields,
    RowIdRange,
    Sequence,
    SequenceStats,
    check_row_coverage,
    csc_sequence_from_raw,
    feature_block_width,
    fnv1a_f64,
    fnv1a_int,
    fnv1a_text,
    gather_dense_block,
    gather_row_fields,
    gather_sparse_block,
    memory_sequence_from_raw,
    sequence_status_message,
)
from .sparse import (
    CscMatrix,
    SparseBinnedMatrix,
    default_bins,
    fit_bins_csc,
    transform_csc,
)
from .train_gpu import train_gpu


comptime EXT_MAGIC = "mojotrees_extmem"
"""First token of every file this module writes."""

comptime EXT_VERSION = 1
"""Cache format version.

v1: manifest, one binned chunk file per chunk, one row-field file, and a raw
spill format for one-shot sources. Every file is a token stream. A cache
written by a different version is refused rather than guessed at; a cache is
a derived artifact, so rebuilding it is always available and reading it
wrongly never is.
"""

comptime DEFAULT_CHUNK_ROWS = 65536
"""Rows per chunk when a caller does not choose.

Large enough that per-chunk overhead is noise and small enough that a chunk
of a wide matrix is tens of megabytes rather than hundreds.
"""

comptime DEFAULT_BIN_MEMORY_BUDGET = 268435456
"""Bytes one bin-construction pass may hold, 256 MiB by default.

This is the dial that trades memory for passes: the block is
`width * n_rows * 8` bytes, so a bigger budget bins more columns per read of
the source. It bounds the block only; the chunk being read and the fitted
mapper are on top of it.
"""


def _u64_token(x: UInt64) -> String:
    """A UInt64 as decimal. Checksums and fingerprints are written this
    way."""
    return String(x)


def _parse_u64(token: String) raises -> UInt64:
    """Decimal digits to UInt64, refusing anything else. The same reader
    `serialize.mojo` uses, written again here because that one is private to
    a file this lane does not own."""
    if token.byte_length() == 0:
        raise Error("empty token where an integer was expected")
    comptime _U64_MAX = ~UInt64(0)
    var out: UInt64 = 0
    for b in token.as_bytes():
        if b < 48 or b > 57:
            raise Error("invalid digit in an integer token")
        # Floats are stored as bit patterns, so the whole unsigned range is
        # legitimate and this has to be exact. Without it a long digit run
        # wraps and a corrupt cache reads back as plausible numbers.
        var digit = UInt64(Int(b) - 48)
        if out > (_U64_MAX - digit) // 10:
            raise Error(
                "integer token does not fit in 64 bits: " + String(token)
            )
        out = out * 10 + digit
    return out


def _f64_token(x: Float64) -> String:
    """A float as its IEEE-754 bit pattern in decimal, so a cache round trip
    is exact."""
    return String(x.to_bits())


def _parse_f64(token: String) raises -> Float64:
    return bitcast[DType.float64, 1](
        SIMD[DType.uint64, 1](_parse_u64(token))
    )


struct _TokenReader:
    """Whitespace-separated tokens out of a file's contents."""

    var tokens: List[String]
    var pos: Int

    def __init__(out self, content: String):
        self.tokens = List[String]()
        for tok in content.split():
            self.tokens.append(String(tok))
        self.pos = 0

    def next(mut self) raises -> String:
        if self.pos >= len(self.tokens):
            raise Error("unexpected end of an external-memory cache file")
        var tok = self.tokens[self.pos].copy()
        self.pos += 1
        return tok^

    def expect(mut self, word: String) raises:
        var tok = self.next()
        if tok != word:
            raise Error(
                "expected '" + word + "' in the cache file, found '" + tok
                + "'"
            )

    def next_int(mut self) raises -> Int:
        return Int(self.next())

    def next_u64(mut self) raises -> UInt64:
        return _parse_u64(self.next())

    def next_f64(mut self) raises -> Float64:
        return _parse_f64(self.next())

    def at_end(self) -> Bool:
        return self.pos >= len(self.tokens)


def checksum_text(text: String) -> UInt64:
    """FNV-1a 64 over a file's bytes.

    Not a cryptographic hash and not offered as one: it detects a truncated
    write, a partially flushed file, a cache reused after the data behind it
    changed, and a chunk file swapped for another chunk's. It does not detect
    an adversary, and `docs/EXTERNAL_MEMORY.md` says so.
    """
    return fnv1a_text(FNV_OFFSET, text)


def mapper_fingerprint(mapper: BinMapper) raises -> UInt64:
    """One integer over everything `BinMapper.matches` compares.

    A manifest stores this so that reopening a cache against a mapper (a
    reference dataset's, a model's) is one comparison rather than a walk.
    `BinMapper.matches` remains the authority for continued training, where
    the answer has to be exact rather than probabilistic; this is the cheap
    screen in front of it.
    """
    var h = FNV_OFFSET
    h = fnv1a_text(h, "mojotrees.bin_mapper.v1")
    h = fnv1a_int(h, mapper.n_features)
    h = fnv1a_int(h, mapper.n_bins)
    h = fnv1a_int(h, len(mapper.edges))
    for i in range(len(mapper.edges)):
        h = fnv1a_f64(h, mapper.edges[i])
    for i in range(len(mapper.edge_offsets)):
        h = fnv1a_int(h, mapper.edge_offsets[i])
    for i in range(len(mapper.missing_bin)):
        h = fnv1a_int(h, mapper.missing_bin[i])
    for f in range(mapper.n_features):
        h = fnv1a_int(h, 1 if mapper.cats.is_cat(f) else 0)
    for i in range(len(mapper.cats.codes)):
        h = fnv1a_int(h, mapper.cats.codes[i])
    for i in range(len(mapper.cats.offsets)):
        h = fnv1a_int(h, mapper.cats.offsets[i])
    return h


struct CacheLayout(Copyable, Movable, Writable):
    """Where a cache's files live, and nothing else.

    Paths are derived from the directory and prefix at open time rather than
    stored in the manifest, so a cache directory can be moved or renamed and
    still open. What the manifest stores instead is the count and the
    checksums, which is what actually has to survive the move.
    """

    var directory: String
    var prefix: String

    def __init__(out self, var directory: String, var prefix: String) raises:
        if len(prefix.as_bytes()) == 0:
            raise Error("a cache prefix cannot be empty")
        for b in prefix.as_bytes():
            if b <= 32 or b == 47 or b == 92 or b >= 127:
                raise Error(
                    "a cache prefix must be printable ASCII with no"
                    " whitespace and no path separator"
                )
        self.directory = directory^
        self.prefix = prefix^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CacheLayout(", self.base(), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def base(self) -> String:
        """The common stem of every file, `directory/prefix` or just
        `prefix` when the directory is empty (the working directory)."""
        if len(self.directory.as_bytes()) == 0:
            return self.prefix.copy()
        return self.directory + "/" + self.prefix

    def manifest_path(self) -> String:
        return self.base() + ".manifest.mbx"

    def fields_path(self) -> String:
        return self.base() + ".fields.mbx"

    def chunk_path(self, index: Int) -> String:
        return self.base() + ".bins." + String(index) + ".mbx"

    def raw_chunk_path(self, index: Int) -> String:
        return self.base() + ".raw." + String(index) + ".mbx"


def _write_file(path: String, content: String) raises -> UInt64:
    """Write one cache file and return its checksum. Every write in this
    module goes through here, so no file can be written without one."""
    with open(path, "w") as f:
        f.write(content)
    return checksum_text(content)


def _read_file(path: String, expected: UInt64) raises -> String:
    """Read one cache file and check it against the checksum the manifest
    recorded. A mismatch raises here rather than surfacing as strange bins
    twenty minutes into a run."""
    var content = open(path, "r").read()
    var got = checksum_text(content)
    if got != expected:
        raise Error(
            "cache file " + path + " does not match its recorded checksum;"
            " the cache is stale or truncated and should be rebuilt"
        )
    return content^


struct ExternalMemoryParams(Copyable, Movable, Writable):
    """Everything a build decides, in one place.

    Deliberately absent: any source-file parameter. There is no `two_round`,
    no `header`, no `label_column`, no `data=` path. LightGBM spells external
    memory as text-parsing configuration because its dataset is built by its
    own parser; mojotrees's parsing lives in the caller, and copying the
    parameter names without the parser behind them would be spelling for its
    own sake. What is here is what changes the fitted binning
    (`max_bin`, `use_missing`, the categorical declaration) and what changes
    the memory ceiling (`chunk_rows`, `bin_memory_budget`).
    """

    var max_bin: Int
    var use_missing: Bool
    var categorical_features: List[Int]
    var feature_names: List[String]
    var chunk_rows: Int
    var bin_memory_budget: Int
    var verify_on_open: Bool

    def __init__(
        out self,
        max_bin: Int = 255,
        use_missing: Bool = True,
        var categorical_features: List[Int] = [],
        var feature_names: List[String] = [],
        chunk_rows: Int = DEFAULT_CHUNK_ROWS,
        bin_memory_budget: Int = DEFAULT_BIN_MEMORY_BUDGET,
        verify_on_open: Bool = True,
    ) raises:
        if max_bin < 2 or max_bin > 256:
            raise Error("max_bin must be in [2, 256]")
        if chunk_rows < 1:
            raise Error("chunk_rows must be positive")
        if bin_memory_budget < 1:
            raise Error("bin_memory_budget must be positive")
        self.max_bin = max_bin
        self.use_missing = use_missing
        self.categorical_features = categorical_features^
        self.feature_names = feature_names^
        self.chunk_rows = chunk_rows
        self.bin_memory_budget = bin_memory_budget
        self.verify_on_open = verify_on_open

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ExternalMemoryParams(max_bin=",
            self.max_bin,
            ", chunk_rows=",
            self.chunk_rows,
            ", bin_memory_budget=",
            self.bin_memory_budget,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def check_against(self, schema: ChunkSchema) raises:
        """The declared categoricals and names have to fit the source.

        Checked before the first pass, because the alternative is discovering
        an out-of-range feature index after reading the data several times.
        """
        for i in range(len(self.categorical_features)):
            var f = self.categorical_features[i]
            if f < 0 or f >= schema.n_features:
                raise Error("categorical feature index out of range")
            for j in range(i):
                if self.categorical_features[j] == f:
                    raise Error("categorical feature index listed twice")
        if (
            len(self.feature_names) != 0
            and len(self.feature_names) != schema.n_features
        ):
            raise Error("feature_name must have one name per feature")
        for i in range(len(self.feature_names)):
            for b in self.feature_names[i].as_bytes():
                if b <= 32 or b >= 127:
                    raise Error(
                        "an external-memory cache stores feature names as"
                        " whitespace-free tokens; this one has a space or a"
                        " control byte"
                    )


struct ExternalChunkRecord(Copyable, Movable, Writable):
    """One row of the manifest's chunk table."""

    var index: Int
    var base: Int
    var count: Int
    var stored: Int
    var checksum: UInt64

    def __init__(
        out self,
        index: Int,
        base: Int,
        count: Int,
        stored: Int,
        checksum: UInt64,
    ):
        self.index = index
        self.base = base
        self.count = count
        self.stored = stored
        self.checksum = checksum

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "chunk ",
            self.index,
            " rows[",
            self.base,
            ", ",
            self.base + self.count,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def range(self) -> RowIdRange:
        return RowIdRange(self.base, self.count)


struct ExternalManifest(Copyable, Movable, Writable):
    """What a cache is, written next to it.

    Everything here is either a fact a reader must check (the format version,
    the fingerprints, the row identity) or a fact a reader cannot recompute
    without the source (the chunk cut, the checksums). Paths are absent on
    purpose; see `CacheLayout`.
    """

    var version: Int
    var schema_fingerprint: UInt64
    var mapper_fingerprint: UInt64
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var max_bin: Int
    var use_missing: Bool
    var is_sparse: Bool
    var chunk_rows: Int
    var categorical_features: List[Int]
    var feature_names: List[String]
    var fields_checksum: UInt64
    var chunks: List[ExternalChunkRecord]

    def __init__(
        out self,
        schema_fingerprint: UInt64,
        mapper_fingerprint: UInt64,
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        max_bin: Int,
        use_missing: Bool,
        is_sparse: Bool,
        chunk_rows: Int,
        var categorical_features: List[Int],
        var feature_names: List[String],
        fields_checksum: UInt64,
        var chunks: List[ExternalChunkRecord],
    ):
        self.version = EXT_VERSION
        self.schema_fingerprint = schema_fingerprint
        self.mapper_fingerprint = mapper_fingerprint
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins
        self.max_bin = max_bin
        self.use_missing = use_missing
        self.is_sparse = is_sparse
        self.chunk_rows = chunk_rows
        self.categorical_features = categorical_features^
        self.feature_names = feature_names^
        self.fields_checksum = fields_checksum
        self.chunks = chunks^

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ExternalManifest(n_rows=",
            self.n_rows,
            ", n_features=",
            self.n_features,
            ", chunks=",
            len(self.chunks),
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def ranges(self) -> List[RowIdRange]:
        var out = List[RowIdRange]()
        for i in range(len(self.chunks)):
            out.append(self.chunks[i].range())
        return out^

    def check(self) raises:
        """Internal consistency: the chunk table covers every row exactly
        once, in order, and the counts agree with the header."""
        if self.version != EXT_VERSION:
            raise Error(
                "this cache was written by external-memory format v"
                + String(self.version)
                + "; this build writes v"
                + String(EXT_VERSION)
                + ". Rebuild the cache"
            )
        if self.n_rows < 1 or self.n_features < 1:
            raise Error("a cache must describe a matrix with positive size")
        for i in range(len(self.chunks)):
            if self.chunks[i].index != i:
                raise Error("the chunk table is out of order")
        check_row_coverage(self.ranges(), self.n_rows)

    def to_text(self) raises -> String:
        """The manifest as a token stream."""
        var out = String(EXT_MAGIC)
        out += " v" + String(self.version) + "\n"
        out += "schema_fingerprint " + _u64_token(self.schema_fingerprint)
        out += "\n"
        out += "mapper_fingerprint " + _u64_token(self.mapper_fingerprint)
        out += "\n"
        out += "shape " + String(self.n_rows) + " " + String(self.n_features)
        out += " " + String(self.n_bins) + "\n"
        out += "binning " + String(self.max_bin)
        out += " " + String(1 if self.use_missing else 0)
        out += " " + String(1 if self.is_sparse else 0)
        out += " " + String(self.chunk_rows) + "\n"
        out += "categorical " + String(len(self.categorical_features))
        for i in range(len(self.categorical_features)):
            out += " " + String(self.categorical_features[i])
        out += "\n"
        out += "feature_names " + String(len(self.feature_names))
        for i in range(len(self.feature_names)):
            out += " " + self.feature_names[i]
        out += "\n"
        out += "fields " + _u64_token(self.fields_checksum) + "\n"
        out += "chunks " + String(len(self.chunks)) + "\n"
        for i in range(len(self.chunks)):
            out += String(self.chunks[i].index)
            out += " " + String(self.chunks[i].base)
            out += " " + String(self.chunks[i].count)
            out += " " + String(self.chunks[i].stored)
            out += " " + _u64_token(self.chunks[i].checksum) + "\n"
        out += "end\n"
        return out^

    @staticmethod
    def from_text(content: String) raises -> ExternalManifest:
        """Parse a manifest, refusing anything this build did not write."""
        var r = _TokenReader(content)
        r.expect(EXT_MAGIC)
        var version_tok = r.next()
        if version_tok != "v" + String(EXT_VERSION):
            raise Error(
                "unrecognized external-memory cache version '" + version_tok
                + "'; rebuild the cache"
            )
        r.expect("schema_fingerprint")
        var schema_fp = r.next_u64()
        r.expect("mapper_fingerprint")
        var mapper_fp = r.next_u64()
        r.expect("shape")
        var n_rows = r.next_int()
        var n_features = r.next_int()
        var n_bins = r.next_int()
        r.expect("binning")
        var max_bin = r.next_int()
        var use_missing = r.next_int() != 0
        var is_sparse = r.next_int() != 0
        var chunk_rows = r.next_int()
        r.expect("categorical")
        var n_cat = r.next_int()
        var cats = List[Int](capacity=n_cat)
        for _ in range(n_cat):
            cats.append(r.next_int())
        r.expect("feature_names")
        var n_names = r.next_int()
        var names = List[String](capacity=n_names)
        for _ in range(n_names):
            names.append(r.next())
        r.expect("fields")
        var fields_checksum = r.next_u64()
        r.expect("chunks")
        var n_chunks = r.next_int()
        var chunks = List[ExternalChunkRecord](capacity=n_chunks)
        for _ in range(n_chunks):
            var index = r.next_int()
            var base = r.next_int()
            var count = r.next_int()
            var stored = r.next_int()
            var checksum = r.next_u64()
            chunks.append(
                ExternalChunkRecord(index, base, count, stored, checksum)
            )
        r.expect("end")
        var out = ExternalManifest(
            schema_fp,
            mapper_fp,
            n_rows,
            n_features,
            n_bins,
            max_bin,
            use_missing,
            is_sparse,
            chunk_rows,
            cats^,
            names^,
            fields_checksum,
            chunks^,
        )
        out.check()
        return out^


def _fields_to_text(fields: RowFields) raises -> String:
    """The row fields as a token stream, floats as bit patterns."""
    var out = String(EXT_MAGIC)
    out += " fields v" + String(EXT_VERSION) + "\n"
    out += "n_rows " + String(fields.n_rows) + "\n"
    out += "label " + String(len(fields.label))
    for i in range(len(fields.label)):
        out += " " + _f64_token(fields.label[i])
    out += "\n"
    out += "weight " + String(len(fields.weight))
    for i in range(len(fields.weight)):
        out += " " + _f64_token(fields.weight[i])
    out += "\n"
    out += "init_score " + String(len(fields.init_score))
    for i in range(len(fields.init_score)):
        out += " " + _f64_token(fields.init_score[i])
    out += "\n"
    out += "query_ids " + String(len(fields.query_ids))
    for i in range(len(fields.query_ids)):
        out += " " + String(fields.query_ids[i])
    out += "\nend\n"
    return out^


def _fields_from_text(content: String) raises -> RowFields:
    var r = _TokenReader(content)
    r.expect(EXT_MAGIC)
    r.expect("fields")
    r.expect("v" + String(EXT_VERSION))
    r.expect("n_rows")
    var n_rows = r.next_int()
    var out = RowFields()
    out.n_rows = n_rows
    r.expect("label")
    var n = r.next_int()
    for _ in range(n):
        out.label.append(r.next_f64())
    r.expect("weight")
    n = r.next_int()
    for _ in range(n):
        out.weight.append(r.next_f64())
    r.expect("init_score")
    n = r.next_int()
    for _ in range(n):
        out.init_score.append(r.next_f64())
    r.expect("query_ids")
    n = r.next_int()
    for _ in range(n):
        out.query_ids.append(r.next_int())
    r.expect("end")
    return out^


def _binned_chunk_to_text(
    bins: BinnedMatrix, base: Int, index: Int
) raises -> String:
    """One dense binned chunk as a token stream.

    Only the bins travel. The category tables and the missing-bin
    reservations are the mapper's, held once in the manifest's fingerprint
    and rebuilt from the mapper when a chunk is read, so a cache cannot hold
    a chunk whose idea of the binning differs from the dataset's.
    """
    var out = String(EXT_MAGIC)
    out += " chunk v" + String(EXT_VERSION) + "\n"
    out += "index " + String(index) + "\n"
    out += "base " + String(base) + "\n"
    out += "dense " + String(bins.n_rows) + " " + String(bins.n_features)
    out += " " + String(bins.n_bins) + "\n"
    out += "bins"
    for i in range(len(bins.bins)):
        out += " " + String(Int(bins.bins[i]))
    out += "\nend\n"
    return out^


def _sparse_chunk_to_text(
    bins: SparseBinnedMatrix, base: Int, index: Int
) raises -> String:
    """One sparse binned chunk as a token stream. Row indices stay
    chunk-local; `base` is what makes them global."""
    var out = String(EXT_MAGIC)
    out += " chunk v" + String(EXT_VERSION) + "\n"
    out += "index " + String(index) + "\n"
    out += "base " + String(base) + "\n"
    out += "sparse " + String(bins.n_rows) + " " + String(bins.n_features)
    out += " " + String(bins.n_bins) + "\n"
    out += "offsets"
    for i in range(len(bins.col_offsets)):
        out += " " + String(bins.col_offsets[i])
    out += "\nrow_index"
    for i in range(len(bins.row_index)):
        out += " " + String(bins.row_index[i])
    out += "\nbins"
    for i in range(len(bins.bin)):
        out += " " + String(Int(bins.bin[i]))
    out += "\nend\n"
    return out^


@fieldwise_init
struct ChunkHeader(Copyable, Movable):
    """The first few tokens of a chunk file, read before its payload."""

    var index: Int
    var base: Int
    var is_sparse: Bool
    var n_rows: Int
    var n_features: Int
    var n_bins: Int


def _read_chunk_header(mut r: _TokenReader) raises -> ChunkHeader:
    r.expect(EXT_MAGIC)
    r.expect("chunk")
    r.expect("v" + String(EXT_VERSION))
    r.expect("index")
    var index = r.next_int()
    r.expect("base")
    var base = r.next_int()
    var kind = r.next()
    var is_sparse: Bool
    if kind == "dense":
        is_sparse = False
    elif kind == "sparse":
        is_sparse = True
    else:
        raise Error("a chunk file must say 'dense' or 'sparse'")
    var n_rows = r.next_int()
    var n_features = r.next_int()
    var n_bins = r.next_int()
    return ChunkHeader(index, base, is_sparse, n_rows, n_features, n_bins)


def _dense_chunk_from_text(
    content: String, mapper: BinMapper, expect_index: Int, expect_base: Int
) raises -> BinnedMatrix:
    var r = _TokenReader(content)
    var head = _read_chunk_header(r)
    if head.is_sparse:
        raise Error("this chunk is sparse; read it with sparse_binned_chunk")
    if head.index != expect_index or head.base != expect_base:
        raise Error(
            "a chunk file does not describe the chunk the manifest expected"
        )
    if head.n_features != mapper.n_features:
        raise Error("a chunk's feature count does not match the mapper")
    r.expect("bins")
    var n = head.n_rows * head.n_features
    var bins = List[UInt8](capacity=n)
    for _ in range(n):
        var v = r.next_int()
        if v < 0 or v > 255:
            raise Error("a bin index in the cache is out of range")
        bins.append(UInt8(v))
    r.expect("end")
    return BinnedMatrix(
        bins^,
        head.n_rows,
        head.n_features,
        head.n_bins,
        mapper.cats.copy(),
        mapper.missing_bin.copy(),
    )


def _sparse_chunk_from_text(
    content: String, mapper: BinMapper, expect_index: Int, expect_base: Int
) raises -> SparseBinnedMatrix:
    var r = _TokenReader(content)
    var head = _read_chunk_header(r)
    if not head.is_sparse:
        raise Error("this chunk is dense; read it with binned_chunk")
    if head.index != expect_index or head.base != expect_base:
        raise Error(
            "a chunk file does not describe the chunk the manifest expected"
        )
    if head.n_features != mapper.n_features:
        raise Error("a chunk's feature count does not match the mapper")
    r.expect("offsets")
    var offsets = List[Int](capacity=head.n_features + 1)
    for _ in range(head.n_features + 1):
        offsets.append(r.next_int())
    var nnz = offsets[head.n_features]
    r.expect("row_index")
    var row_index = List[Int](capacity=nnz)
    for _ in range(nnz):
        row_index.append(r.next_int())
    r.expect("bins")
    var bins = List[UInt8](capacity=nnz)
    for _ in range(nnz):
        var v = r.next_int()
        if v < 0 or v > 255:
            raise Error("a bin index in the cache is out of range")
        bins.append(UInt8(v))
    r.expect("end")
    var out = SparseBinnedMatrix(
        row_index^,
        bins^,
        offsets^,
        default_bins(mapper),
        head.n_rows,
        head.n_features,
        head.n_bins,
        mapper.cats.copy(),
        mapper.missing_bin.copy(),
    )
    out.validate()
    return out^


def _raw_chunk_to_text(chunk: RawChunk) raises -> String:
    """One raw chunk as a token stream, for the spill a one-shot source needs.

    Values are bit patterns, so a spilled source bins to exactly the bins the
    original would have: the spill is a copy, not a rounding.
    """
    var out = String(EXT_MAGIC)
    out += " raw v" + String(EXT_VERSION) + "\n"
    out += "base " + String(chunk.row_id_base) + "\n"
    out += "shape " + String(chunk.n_rows) + " " + String(chunk.n_features)
    out += " " + String(1 if chunk.is_sparse else 0) + "\n"
    if chunk.is_sparse:
        out += "offsets"
        for i in range(len(chunk.csc.col_offsets)):
            out += " " + String(chunk.csc.col_offsets[i])
        out += "\nrow_index"
        for i in range(len(chunk.csc.row_index)):
            out += " " + String(chunk.csc.row_index[i])
        out += "\nvalues"
        for i in range(len(chunk.csc.values)):
            out += " " + _f64_token(chunk.csc.values[i])
        out += "\n"
    else:
        out += "values"
        for i in range(len(chunk.values)):
            out += " " + _f64_token(chunk.values[i])
        out += "\n"
    out += "label " + String(len(chunk.label))
    for i in range(len(chunk.label)):
        out += " " + _f64_token(chunk.label[i])
    out += "\nweight " + String(len(chunk.weight))
    for i in range(len(chunk.weight)):
        out += " " + _f64_token(chunk.weight[i])
    out += "\ninit_score " + String(len(chunk.init_score))
    for i in range(len(chunk.init_score)):
        out += " " + _f64_token(chunk.init_score[i])
    out += "\nquery_ids " + String(len(chunk.query_ids))
    for i in range(len(chunk.query_ids)):
        out += " " + String(chunk.query_ids[i])
    out += "\nend\n"
    return out^


def _raw_chunk_from_text(content: String) raises -> RawChunk:
    var r = _TokenReader(content)
    r.expect(EXT_MAGIC)
    r.expect("raw")
    r.expect("v" + String(EXT_VERSION))
    r.expect("base")
    var base = r.next_int()
    r.expect("shape")
    var n_rows = r.next_int()
    var n_features = r.next_int()
    var is_sparse = r.next_int() != 0

    var csc = CscMatrix(
        List[Int](), List[Float64](), List[Int](), n_rows, n_features
    )
    var values = List[Float64]()
    if is_sparse:
        r.expect("offsets")
        var offsets = List[Int](capacity=n_features + 1)
        for _ in range(n_features + 1):
            offsets.append(r.next_int())
        var nnz = offsets[n_features]
        r.expect("row_index")
        var row_index = List[Int](capacity=nnz)
        for _ in range(nnz):
            row_index.append(r.next_int())
        r.expect("values")
        var vals = List[Float64](capacity=nnz)
        for _ in range(nnz):
            vals.append(r.next_f64())
        csc = CscMatrix(row_index^, vals^, offsets^, n_rows, n_features)
    else:
        r.expect("values")
        values = List[Float64](capacity=n_rows * n_features)
        for _ in range(n_rows * n_features):
            values.append(r.next_f64())

    r.expect("label")
    var n = r.next_int()
    var label = List[Float64](capacity=n)
    for _ in range(n):
        label.append(r.next_f64())
    r.expect("weight")
    n = r.next_int()
    var weight = List[Float64](capacity=n)
    for _ in range(n):
        weight.append(r.next_f64())
    r.expect("init_score")
    n = r.next_int()
    var init_score = List[Float64](capacity=n)
    for _ in range(n):
        init_score.append(r.next_f64())
    r.expect("query_ids")
    n = r.next_int()
    var query_ids = List[Int](capacity=n)
    for _ in range(n):
        query_ids.append(r.next_int())
    r.expect("end")

    if is_sparse:
        return RawChunk.sparse(
            csc^, base, label^, weight^, init_score^, query_ids^
        )
    return RawChunk.dense(
        values^,
        n_rows,
        n_features,
        base,
        label^,
        weight^,
        init_score^,
        query_ids^,
    )


struct RawCacheSequence(Sequence, Copyable, Movable):
    """A spilled source, replayed from its raw chunk files.

    This is what makes a one-shot source binnable: `spill_source` drains it
    once into these files, and every later pass reads them instead. It is a
    `Sequence` like any other, so the multi-pass binner does not know or care
    which kind of source it is reading.

    Repeatable because a file is: `rewind` sets the index back to zero and
    the chunks are read again, byte for byte, which is what "deterministic
    multi-pass" needs from a source it reads `2 + blocks` times.
    """

    var layout: CacheLayout
    var checksums: List[UInt64]
    var bases: List[Int]
    var counts: List[Int]
    var next_index: Int
    var _schema: ChunkSchema

    def __init__(
        out self,
        var layout: CacheLayout,
        var checksums: List[UInt64],
        var bases: List[Int],
        var counts: List[Int],
        var schema: ChunkSchema,
    ):
        self.layout = layout^
        self.checksums = checksums^
        self.bases = bases^
        self.counts = counts^
        self.next_index = 0
        self._schema = schema^

    def schema(self) -> ChunkSchema:
        return self._schema.copy()

    def n_rows_hint(self) -> Int:
        var n = 0
        for i in range(len(self.counts)):
            n += self.counts[i]
        return n

    def is_repeatable(self) -> Bool:
        return True

    def rewind(mut self) raises:
        self.next_index = 0

    def has_next(self) -> Bool:
        return self.next_index < len(self.checksums)

    def next_chunk(mut self) raises -> RawChunk:
        if not self.has_next():
            raise Error("the sequence is drained; call rewind to read again")
        var i = self.next_index
        self.next_index += 1
        var content = _read_file(
            self.layout.raw_chunk_path(i), self.checksums[i]
        )
        var chunk = _raw_chunk_from_text(content)
        if chunk.row_id_base != self.bases[i]:
            raise Error("a spilled chunk does not start where it should")
        if chunk.n_rows != self.counts[i]:
            raise Error("a spilled chunk does not hold the rows it should")
        return chunk^

    def paths(self) -> List[String]:
        """Every file this spill owns, for cleanup."""
        var out = List[String]()
        for i in range(len(self.checksums)):
            out.append(self.layout.raw_chunk_path(i))
        return out^


def spill_source[S: Sequence & Movable](
    mut src: S, var layout: CacheLayout, mut cancel: CancelToken
) raises -> RawCacheSequence:
    """Drain a source once into raw chunk files and return a repeatable
    sequence over them.

    The only way a one-shot source reaches exact multi-pass binning, and it
    is explicit: the caller decides to pay one copy of the raw data on disk
    rather than accept a sampled binning. A source that is already repeatable
    does not need this and should not use it.
    """
    src.rewind()
    var schema = src.schema()
    var checksums = List[UInt64]()
    var bases = List[Int]()
    var counts = List[Int]()
    var next_row = 0
    var index = 0
    while src.has_next():
        cancel.check()
        var chunk = src.next_chunk()
        if chunk.row_id_base != next_row:
            raise Error(
                "chunks must cover the rows in order, exactly once"
            )
        var text = _raw_chunk_to_text(chunk)
        checksums.append(_write_file(layout.raw_chunk_path(index), text))
        bases.append(chunk.row_id_base)
        counts.append(chunk.n_rows)
        next_row += chunk.n_rows
        index += 1
    cancel.check()
    if index == 0:
        raise Error("the source delivered no rows")
    return RawCacheSequence(
        layout^, checksums^, bases^, counts^, schema^
    )


def _merge_block_mappers(
    blocks: List[BinMapper], n_features: Int, max_bin: Int
) raises -> BinMapper:
    """Concatenate per-block mappers into the mapper for the whole matrix.

    Sound because binning is per feature and nothing else: `fit_bins` fits
    feature f's edges, missing reservation, and category table from column f
    alone, so a block's answer for its columns is the whole matrix's answer
    for those columns. The concatenation restores the feature order, and the
    result is the mapper the resident path would have produced.
    """
    var edges = List[Float64]()
    var offsets = List[Int](capacity=n_features + 1)
    offsets.append(0)
    var missing = no_missing_bins(n_features)
    var flags = List[Bool](capacity=n_features)
    for _ in range(n_features):
        flags.append(False)
    var codes = List[Int]()
    var cat_offsets = List[Int](capacity=n_features + 1)
    cat_offsets.append(0)

    var f = 0
    for b in range(len(blocks)):
        ref m = blocks[b]
        if m.n_bins != max_bin:
            raise Error("a bin-construction block used a different max_bin")
        for local in range(m.n_features):
            if f >= n_features:
                raise Error(
                    "the bin-construction blocks cover too many features"
                )
            var lo = m.edge_offsets[local]
            var hi = m.edge_offsets[local + 1]
            for e in range(lo, hi):
                edges.append(m.edges[e])
            offsets.append(len(edges))
            missing[f] = m.missing_bin[local]
            flags[f] = m.cats.is_cat(local)
            var clo = m.cats.offsets[local]
            var chi = m.cats.offsets[local + 1]
            for c in range(clo, chi):
                codes.append(m.cats.codes[c])
            cat_offsets.append(len(codes))
            f += 1
    if f != n_features:
        raise Error("the bin-construction blocks did not cover every feature")
    return BinMapper(
        edges^,
        offsets^,
        n_features,
        max_bin,
        CategoricalSpec(flags^, codes^, cat_offsets^),
        missing^,
    )


def _block_categoricals(
    categorical_features: List[Int], f_start: Int, f_end: Int
) -> List[Int]:
    """The declared categoricals of one block, renumbered to the block."""
    var out = List[Int]()
    for i in range(len(categorical_features)):
        var f = categorical_features[i]
        if f >= f_start and f < f_end:
            out.append(f - f_start)
    return out^


def fit_mapper_external[S: Sequence & Movable](
    mut src: S,
    n_rows: Int,
    params: ExternalMemoryParams,
    mut cancel: CancelToken,
    mut stats: SequenceStats,
) raises -> BinMapper:
    """Fit the bin mapper of a source that does not fit in memory.

    One pass per block of columns, each block handed to the resident binner.
    The edges, category tables, and missing reservations are therefore the
    ones `binning.fit_bins` (or `sparse.fit_bins_csc`) would have produced on
    the whole matrix, bit for bit, because they *are* what those functions
    produced, on the columns they were given.

    `params.bin_memory_budget` sets the block width and so the number of
    passes; `sequence.feature_block_width` is the arithmetic.
    """
    var schema = src.schema()
    params.check_against(schema)
    if not src.is_repeatable():
        raise Error(
            sequence_status_message(SEQ_NOT_REPEATABLE)
            + ": bin construction reads the source once per feature block."
            " Spill it to a raw cache with spill_source first"
        )
    var n_features = schema.n_features
    var width = feature_block_width(
        n_features, n_rows, params.bin_memory_budget
    )
    var blocks = List[BinMapper]()
    var f_start = 0
    while f_start < n_features:
        cancel.check()
        var f_end = f_start + width
        if f_end > n_features:
            f_end = n_features
        var block_cats = _block_categoricals(
            params.categorical_features, f_start, f_end
        )
        if schema.is_sparse:
            var block = gather_sparse_block(
                src, f_start, f_end, n_rows, cancel, stats
            )
            blocks.append(
                fit_bins_csc(
                    block,
                    params.max_bin,
                    block_cats,
                    params.use_missing,
                )
            )
        else:
            var block = gather_dense_block(
                src, f_start, f_end, n_rows, cancel, stats
            )
            blocks.append(
                fit_bins(
                    block,
                    n_rows,
                    f_end - f_start,
                    params.max_bin,
                    block_cats,
                    params.use_missing,
                )
            )
        f_start = f_end
    return _merge_block_mappers(blocks, n_features, params.max_bin)


struct ExternalDataset(Copyable, Movable, Writable):
    """A binned dataset that lives in a cache directory.

    Holds the fitted mapper and the manifest; the bins themselves are on
    disk, one file per chunk. Every accessor either reads one chunk (bounded)
    or is explicitly a materialization with a byte budget.

    Immutable, for the same reason `trainset.Dataset` is: the bin edges were
    fitted from the data and the categorical declaration it was built with,
    so changing either afterwards would leave the cached bins describing data
    the dataset no longer holds. Build another cache.
    """

    var layout: CacheLayout
    var manifest: ExternalManifest
    var mapper: BinMapper
    var categorical_report: String
    var discarded: Bool

    def __init__(
        out self,
        var layout: CacheLayout,
        var manifest: ExternalManifest,
        var mapper: BinMapper,
        var categorical_report: String = String(""),
    ) raises:
        """`categorical_report` is `CategoryTally.report` from the build's
        census pass. It is a diagnosis rather than data, so it is not written
        to the manifest and a reopened cache carries an empty one; rebuild or
        re-tally to get it back."""
        manifest.check()
        if mapper.n_features != manifest.n_features:
            raise Error("the mapper does not describe this cache's features")
        self.layout = layout^
        self.manifest = manifest^
        self.mapper = mapper^
        self.categorical_report = categorical_report^
        self.discarded = False

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ExternalDataset(n_rows=",
            self.manifest.n_rows,
            ", n_features=",
            self.manifest.n_features,
            ", chunks=",
            len(self.manifest.chunks),
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def num_data(self) -> Int:
        """Rows in the dataset. The counterpart of `Dataset.num_data`."""
        return self.manifest.n_rows

    def num_feature(self) -> Int:
        """Features in the dataset."""
        return self.manifest.n_features

    def num_bin(self) -> Int:
        """Bins the binning reserved per feature, the effective `max_bin`."""
        return self.mapper.n_bins

    def num_chunk(self) -> Int:
        """Cache chunks. Not a LightGBM concept; it is the unit every bounded
        read here works in."""
        return len(self.manifest.chunks)

    def is_sparse(self) -> Bool:
        return self.manifest.is_sparse

    def row_range(self, index: Int) raises -> RowIdRange:
        """The global rows chunk `index` covers."""
        if index < 0 or index >= len(self.manifest.chunks):
            raise Error("chunk index out of range")
        return self.manifest.chunks[index].range()

    def chunk_of_row(self, row_id: Int) raises -> Int:
        """Which chunk holds a global row. Binary search over the chunk
        table, which is ascending and gapless by `ExternalManifest.check`."""
        if row_id < 0 or row_id >= self.manifest.n_rows:
            raise Error("row id out of range")
        var lo = 0
        var hi = len(self.manifest.chunks)
        while lo + 1 < hi:
            var mid = (lo + hi) // 2
            if self.manifest.chunks[mid].base <= row_id:
                lo = mid
            else:
                hi = mid
        return lo

    def _check_live(self) raises:
        if self.discarded:
            raise Error(
                "this external dataset's cache was discarded; rebuild it"
            )

    def binned_chunk(self, index: Int) raises -> BinnedMatrix:
        """One chunk's bins, as a `BinnedMatrix` of that chunk's rows.

        Bounded: this reads one file. The matrix carries the dataset's
        category tables and missing reservations, taken from the mapper, so a
        chunk is a valid input to anything that reads a `BinnedMatrix` and
        knows it is looking at a slice of the rows.
        """
        self._check_live()
        if self.manifest.is_sparse:
            raise Error(
                "this cache is sparse; read it with sparse_binned_chunk"
            )
        if index < 0 or index >= len(self.manifest.chunks):
            raise Error("chunk index out of range")
        ref rec = self.manifest.chunks[index]
        var content = _read_file(
            self.layout.chunk_path(index), rec.checksum
        )
        return _dense_chunk_from_text(
            content, self.mapper, index, rec.base
        )

    def sparse_binned_chunk(self, index: Int) raises -> SparseBinnedMatrix:
        """One chunk's bins, kept sparse. Row indices are chunk-local; the
        chunk's `row_range` is what makes them global."""
        self._check_live()
        if not self.manifest.is_sparse:
            raise Error("this cache is dense; read it with binned_chunk")
        if index < 0 or index >= len(self.manifest.chunks):
            raise Error("chunk index out of range")
        ref rec = self.manifest.chunks[index]
        var content = _read_file(
            self.layout.chunk_path(index), rec.checksum
        )
        return _sparse_chunk_from_text(
            content, self.mapper, index, rec.base
        )

    def row_fields(self) raises -> RowFields:
        """The label, weight, init score, and query ids, all `n_rows` long.

        These are read whole rather than by chunk, and that is not a
        contradiction: they are one to four Float64 columns, so they are
        `n_rows * 8` bytes each where the matrix is `n_rows * n_features * 8`.
        A dataset whose labels do not fit has no trainer here that could use
        them.
        """
        self._check_live()
        var content = _read_file(
            self.layout.fields_path(), self.manifest.fields_checksum
        )
        var fields = _fields_from_text(content)
        if fields.n_rows != self.manifest.n_rows:
            raise Error("the cached row fields do not match the row count")
        return fields^

    def verify(self) raises:
        """Read every file and check it against its recorded checksum.

        Costs a full read of the cache and nothing else; it changes nothing.
        What it catches is a cache that was truncated, partially written, or
        edited between the build and the run.
        """
        self._check_live()
        self.manifest.check()
        _ = _read_file(
            self.layout.fields_path(), self.manifest.fields_checksum
        )
        for i in range(len(self.manifest.chunks)):
            _ = _read_file(
                self.layout.chunk_path(i), self.manifest.chunks[i].checksum
            )

    def materialize_binned(self, max_bytes: Int) raises -> BinnedMatrix:
        """The whole binned matrix, if the caller says it fits.

        This is the bridge to every trainer in the tree, all of which take a
        whole `BinnedMatrix`. It is `n_rows * n_features` bytes, an eighth of
        the raw float64 matrix, which is why an external-memory build can
        train data a resident build cannot even read. It is still the whole
        matrix, so it is behind a budget the caller names.
        """
        self._check_live()
        if self.manifest.is_sparse:
            raise Error(
                "this cache is sparse; materialize it with"
                " materialize_sparse_binned"
            )
        var n_rows = self.manifest.n_rows
        var n_features = self.manifest.n_features
        var bytes = n_rows * n_features
        if max_bytes < 1:
            raise Error("max_bytes must be positive")
        if bytes > max_bytes:
            raise Error(
                "the binned matrix needs "
                + String(bytes)
                + " bytes and the budget is "
                + String(max_bytes)
                + "; raise the budget or train from the chunks"
            )
        var bins = List[UInt8](capacity=bytes)
        bins.resize(bytes, 0)
        for i in range(len(self.manifest.chunks)):
            var base = self.manifest.chunks[i].base
            var chunk = self.binned_chunk(i)
            for f in range(n_features):
                var src = f * chunk.n_rows
                var dst = f * n_rows + base
                for r in range(chunk.n_rows):
                    bins[dst + r] = chunk.bins[src + r]
        return BinnedMatrix(
            bins^,
            n_rows,
            n_features,
            self.mapper.n_bins,
            self.mapper.cats.copy(),
            self.mapper.missing_bin.copy(),
        )

    def materialize_sparse_binned(
        self, max_bytes: Int
    ) raises -> SparseBinnedMatrix:
        """The whole binned matrix, kept sparse, if the caller says it fits.

        The budget is checked against the stored entries the manifest already
        counted, so a cache whose chunks are dense in practice is refused
        before anything is read rather than after.
        """
        self._check_live()
        if not self.manifest.is_sparse:
            raise Error("this cache is dense; use materialize_binned")
        var stored = 0
        for i in range(len(self.manifest.chunks)):
            stored += self.manifest.chunks[i].stored
        # One Int row index and one UInt8 bin per stored entry.
        var bytes = stored * 9
        if max_bytes < 1:
            raise Error("max_bytes must be positive")
        if bytes > max_bytes:
            raise Error(
                "the binned sparse matrix needs about "
                + String(bytes)
                + " bytes and the budget is "
                + String(max_bytes)
            )
        var n_features = self.manifest.n_features
        var pieces = List[SparseBinnedMatrix]()
        var bases = List[Int]()
        for i in range(len(self.manifest.chunks)):
            bases.append(self.manifest.chunks[i].base)
            pieces.append(self.sparse_binned_chunk(i))
        var row_index = List[Int]()
        var bins = List[UInt8]()
        var offsets = List[Int](capacity=n_features + 1)
        offsets.append(0)
        # Features outer, chunks inner: chunks ascend in row order, so each
        # column's row indices come out ascending without a sort.
        for f in range(n_features):
            for c in range(len(pieces)):
                var lo = pieces[c].col_offsets[f]
                var hi = pieces[c].col_offsets[f + 1]
                for e in range(lo, hi):
                    row_index.append(bases[c] + pieces[c].row_index[e])
                    bins.append(pieces[c].bin[e])
            offsets.append(len(bins))
        var out = SparseBinnedMatrix(
            row_index^,
            bins^,
            offsets^,
            default_bins(self.mapper),
            self.manifest.n_rows,
            n_features,
            self.mapper.n_bins,
            self.mapper.cats.copy(),
            self.mapper.missing_bin.copy(),
        )
        out.validate()
        return out^

    def paths(self) -> List[String]:
        """Every file this cache owns, manifest first."""
        var out = List[String]()
        out.append(self.layout.manifest_path())
        out.append(self.layout.fields_path())
        for i in range(len(self.manifest.chunks)):
            out.append(self.layout.chunk_path(i))
        return out^

    def discard(mut self) raises -> List[String]:
        """Empty every file this cache owns and return their paths.

        Truncation, not removal: `open(path, "w")` is the one filesystem
        primitive this repository has proven, and inventing an unlink path
        nothing has run would be a claim rather than a feature. A truncated
        cache cannot be read back as data, because its manifest is gone and
        every chunk file is empty, and the caller's own cleanup removes the
        paths this returns. Calling it twice is not an error; reading the
        dataset afterwards is.
        """
        self._check_live()
        var removed = self.paths()
        for i in range(len(removed)):
            _ = _write_file(removed[i], String(""))
        self.discarded = True
        return removed^


def build_external_dataset[S: Sequence & Movable](
    mut src: S,
    var layout: CacheLayout,
    params: ExternalMemoryParams,
    mut cancel: CancelToken,
) raises -> ExternalDataset:
    """Bin a source into a cache directory and return the dataset.

    The whole build, in the order the module docstring describes: census,
    then one pass per feature block, then one transform pass. A source that
    cannot be read twice is refused by name rather than binned from a sample.
    """
    if not src.is_repeatable():
        raise Error(
            sequence_status_message(SEQ_NOT_REPEATABLE)
            + ": an external-memory build reads its source several times."
            " Spill it to a raw cache with spill_source and build from that"
        )
    var schema = src.schema()
    params.check_against(schema)

    var stats = SequenceStats()
    var tally = CategoryTally(
        params.categorical_features, params.max_bin - 1
    )
    var fields = gather_row_fields(src, tally, cancel, stats)
    var n_rows = fields.n_rows
    if n_rows < 1:
        raise Error("the source delivered no rows")
    var ranges = stats.ranges.copy()

    var mapper = fit_mapper_external(src, n_rows, params, cancel, stats)

    # Transform pass. One chunk in memory at a time, one file out per chunk.
    src.rewind()
    stats.begin_pass()
    var records = List[ExternalChunkRecord]()
    var index = 0
    while src.has_next():
        cancel.check()
        var chunk = src.next_chunk()
        stats.observe(chunk)
        var text: String
        var stored: Int
        if chunk.is_sparse:
            var binned = transform_csc(mapper, chunk.csc)
            stored = binned.nnz()
            text = _sparse_chunk_to_text(binned, chunk.row_id_base, index)
        else:
            var binned = mapper.transform(chunk.values, chunk.n_rows)
            stored = chunk.n_rows * chunk.n_features
            text = _binned_chunk_to_text(binned, chunk.row_id_base, index)
        var checksum = _write_file(layout.chunk_path(index), text)
        records.append(
            ExternalChunkRecord(
                index, chunk.row_id_base, chunk.n_rows, stored, checksum
            )
        )
        index += 1
    cancel.check()
    if stats.rows != n_rows:
        raise Error(
            "the transform pass saw a different number of rows than the"
            " census pass; the source is not repeatable"
        )
    if len(records) != len(ranges):
        raise Error(
            "the transform pass cut the source differently from the census"
            " pass; the source is not deterministic"
        )
    for i in range(len(records)):
        if (
            records[i].base != ranges[i].base
            or records[i].count != ranges[i].count
        ):
            raise Error(
                "the transform pass cut the source differently from the"
                " census pass; the source is not deterministic"
            )

    var fields_checksum = _write_file(
        layout.fields_path(), _fields_to_text(fields)
    )
    var manifest = ExternalManifest(
        schema.fingerprint(),
        mapper_fingerprint(mapper),
        n_rows,
        schema.n_features,
        mapper.n_bins,
        params.max_bin,
        params.use_missing,
        schema.is_sparse,
        params.chunk_rows,
        params.categorical_features.copy(),
        params.feature_names.copy(),
        fields_checksum,
        records^,
    )
    manifest.check()
    _ = _write_file(layout.manifest_path(), manifest.to_text())
    return ExternalDataset(layout^, manifest^, mapper^, tally.report())


def build_external_dataset_from_raw(
    var raw: RawData,
    var layout: CacheLayout,
    params: ExternalMemoryParams,
    mut cancel: CancelToken,
    var label: List[Float64] = [],
    var weight: List[Float64] = [],
    var init_score: List[Float64] = [],
    var query_ids: List[Int] = [],
) raises -> ExternalDataset:
    """Build a cache from the ingestion type the rest of the tree already
    uses.

    `raw_data.RawData` is what every existing caller holds before binning, so
    this is the door between the resident world and this one: the same matrix
    that would have gone to `Dataset` goes to a cache instead, dense or
    sparse, without changing representation. It is not the interesting case
    for external memory (the matrix is already resident, so nothing is
    saved), and it is the case that makes the streaming path testable against
    the resident one: the same input, two routes, one answer.

    The chunk size is `params.chunk_rows`, which is what the manifest records
    and what a rebuild has to repeat to produce the same cache files.
    """
    if raw.is_sparse:
        var src = csc_sequence_from_raw(
            raw^,
            params.chunk_rows,
            label^,
            weight^,
            init_score^,
            query_ids^,
            params.feature_names.copy(),
            params.categorical_features.copy(),
        )
        return build_external_dataset(src, layout^, params, cancel)
    var dense_src = memory_sequence_from_raw(
        raw^,
        params.chunk_rows,
        label^,
        weight^,
        init_score^,
        query_ids^,
        params.feature_names.copy(),
        params.categorical_features.copy(),
    )
    return build_external_dataset(dense_src, layout^, params, cancel)


def open_external_dataset(
    var layout: CacheLayout, mapper: BinMapper, verify: Bool = True
) raises -> ExternalDataset:
    """Reopen a cache built earlier, against the mapper it was built with.

    The mapper is the caller's because a cache does not store one: the bins
    are only meaningful under the binning that produced them, and a model
    already carries that binning (`Model.mapper`). Handing the wrong one is
    caught by the fingerprint rather than discovered as wrong predictions.
    """
    var content = open(layout.manifest_path(), "r").read()
    var manifest = ExternalManifest.from_text(content)
    if manifest.mapper_fingerprint != mapper_fingerprint(mapper):
        raise Error(
            "this cache was built with a different binning than the mapper"
            " given; bins would mean different things to the two"
        )
    var out = ExternalDataset(layout^, manifest^, mapper.copy())
    if verify:
        out.verify()
    return out^


struct ExternalCapabilities(Copyable, Movable, Writable):
    """What can be trained from an external-memory dataset, and how.

    Written as a struct rather than a table in a document because a caller
    has to be able to ask. Every `True` here is reachable from this module,
    either through a trainer defined here or by handing the result of
    `materialize_binned` to the function that already implements it
    (`efb.fit_bundles` is the second kind, and is not called here). Every
    `False` raises with a message that says what would have to exist.

    The dividing line is one fact: every histogram builder in the tree takes
    a whole `BinnedMatrix` or `SparseBinnedMatrix`. So everything is
    supported through `materialize_binned`, at `n_rows * n_features` bytes,
    and nothing is supported without it. The chunk files, the row
    identifiers, and the per-chunk checksums are the substrate a chunked
    accumulator would need; none of them is a claim that one exists.
    """

    var streaming_histograms: Bool
    var materialized_cpu_training: Bool
    var materialized_gpu_training: Bool
    var multiclass: Bool
    var ranking_groups: Bool
    var bagging: Bool
    var goss: Bool
    var efb: Bool
    var init_score: Bool
    var continued_training: Bool

    @staticmethod
    def current() -> ExternalCapabilities:
        """The capabilities of this build. Not aspirational."""
        return ExternalCapabilities(
            streaming_histograms=False,
            materialized_cpu_training=True,
            materialized_gpu_training=True,
            multiclass=True,
            ranking_groups=True,
            bagging=True,
            goss=True,
            efb=True,
            init_score=True,
            continued_training=True,
        )

    def __init__(
        out self,
        streaming_histograms: Bool,
        materialized_cpu_training: Bool,
        materialized_gpu_training: Bool,
        multiclass: Bool,
        ranking_groups: Bool,
        bagging: Bool,
        goss: Bool,
        efb: Bool,
        init_score: Bool,
        continued_training: Bool,
    ):
        self.streaming_histograms = streaming_histograms
        self.materialized_cpu_training = materialized_cpu_training
        self.materialized_gpu_training = materialized_gpu_training
        self.multiclass = multiclass
        self.ranking_groups = ranking_groups
        self.bagging = bagging
        self.goss = goss
        self.efb = efb
        self.init_score = init_score
        self.continued_training = continued_training

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ExternalCapabilities(streaming_histograms=",
            self.streaming_histograms,
            ", materialized=",
            self.materialized_cpu_training,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def report(self) -> String:
        """One line per capability, for a caller that wants to print it."""
        var out = String("external-memory capabilities\n")
        out += "  streaming histograms: no (every histogram builder takes a"
        out += " whole binned matrix)\n"
        out += "  CPU training, materialized: yes\n"
        out += "  sparse CPU training, materialized: yes (the stored entries"
        out += " rather than every cell)\n"
        out += "  GPU training, materialized: yes (device transfer is of the"
        out += " materialized matrix, not of chunks)\n"
        out += "  multiclass, materialized: yes\n"
        out += "  ranking groups: yes (query ids are assembled globally, so"
        out += " a query may straddle chunks)\n"
        out += "  bagging and GOSS: yes, over materialized row indices\n"
        out += "  EFB: yes, by calling efb.fit_bundles (or"
        out += " fit_bundles_dense) on a materialized matrix\n"
        out += "  init score: yes (CPU paths only, as elsewhere)\n"
        out += "  continued training: yes, under the same binning\n"
        return out^


def check_external_supported(
    dataset: ExternalDataset, streaming: Bool
) raises:
    """Raise when a caller asks for something an external dataset cannot do.

    `streaming=True` means "without materializing", which is the one thing
    this build refuses. The message names what would have to exist rather
    than saying no.
    """
    dataset.manifest.check()
    if streaming:
        raise Error(
            "training directly from cache chunks is not implemented: every"
            " histogram builder takes a whole binned matrix. Use"
            " materialize_binned with a byte budget, or add a chunked"
            " accumulator over binned_chunk"
        )


def _check_labels(label: List[Float64], n_rows: Int) raises:
    """One label per row. The same rule `trainset._check_labels` applies,
    written again here because that one is private to a file this lane does
    not own; the handoff carries the patch that would share it."""
    if len(label) != n_rows:
        raise Error("a dataset needs one label per row to train on")


def _int_labels(label: List[Float64], n_classes: Int) raises -> List[Int]:
    """Class codes from a float64 label column, matching
    `trainset._int_labels`: whole numbers in `[0, n_classes)` only."""
    var out = List[Int](capacity=len(label))
    for r in range(len(label)):
        var v = label[r]
        if v != Float64(Int(v)):
            raise Error("class labels must be whole numbers")
        var code = Int(v)
        if code < 0 or code >= n_classes:
            raise Error("class label out of range")
        out.append(code)
    return out^


def _relevance_labels(label: List[Float64]) raises -> List[Int]:
    """Graded relevances from a float64 label column, matching
    `trainset._relevance_labels`."""
    var out = List[Int](capacity=len(label))
    for r in range(len(label)):
        var v = label[r]
        if v != Float64(Int(v)):
            raise Error("relevance labels must be whole numbers")
        out.append(Int(v))
    return out^


def train_external(
    dataset: ExternalDataset,
    objective: Int,
    params: BoosterParams,
    max_bytes: Int,
    alpha: Float64 = 0.9,
    device: Int = CPU_DEVICE,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> Model:
    """Train a single-output model from an external-memory dataset.

    The counterpart of `trainset.train_dataset`, and deliberately the same
    shape: the binning is the dataset's, so `max_bin`, `use_missing`, and the
    categorical declaration come from it and are not passed again. The one
    argument it adds is `max_bytes`, the budget the binned matrix has to fit
    in, because that is the step this path has and the resident one does not.

    Device resolution, the GPU's refusal of `init_score`, and the returned
    model's mapper are all exactly `train_dataset`'s, because they are the
    same call underneath.
    """
    var data = dataset.materialize_binned(max_bytes)
    var fields = dataset.row_fields()
    _check_labels(fields.label, data.n_rows)
    var backend = resolve_device(device, data.n_rows, data.n_features, 1)
    var booster: Booster
    if backend == GPU_DEVICE:
        if len(fields.init_score) != 0:
            raise Error("init_score is a CPU training path; use device='cpu'")
        booster = train_gpu(
            data,
            fields.label,
            objective,
            params,
            fields.weight,
            alpha,
            bagging,
            goss,
        )
    else:
        booster = train(
            data,
            fields.label,
            objective,
            params,
            fields.weight,
            alpha,
            bagging,
            goss,
            fields.init_score,
        )
    return Model(dataset.mapper.copy(), booster^)


def train_external_multiclass(
    dataset: ExternalDataset,
    n_classes: Int,
    params: BoosterParams,
    max_bytes: Int,
    device: Int = CPU_DEVICE,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MulticlassModel:
    """Train a softmax model from an external-memory dataset. CPU only, as
    `trainset.train_dataset_multiclass` is."""
    var data = dataset.materialize_binned(max_bytes)
    var fields = dataset.row_fields()
    _check_labels(fields.label, data.n_rows)
    if len(fields.init_score) != 0:
        raise Error(
            "init_score is not supported for multiclass training: one offset"
            " per row cannot say what each class starts from"
        )
    _ = resolve_device(device, data.n_rows, data.n_features, n_classes)
    var booster = train_multiclass(
        data,
        _int_labels(fields.label, n_classes),
        n_classes,
        params,
        fields.weight,
        bagging,
        goss,
    )
    return MulticlassModel(dataset.mapper.copy(), booster^)


def train_external_sparse(
    dataset: ExternalDataset,
    objective: Int,
    params: BoosterParams,
    max_bytes: Int,
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> Model:
    """Train a single-output model from a sparse external-memory dataset,
    without densifying it.

    The sparse counterpart of `train_external`, and the reason a sparse
    source is worth keeping sparse all the way through: the materialized form
    is the stored entries, not `n_rows * n_features` cells, so a sparse cache
    trains within a budget a dense one could not. CPU only, as
    `boosting_sparse.train_sparse` is; the GPU sparse path is reached through
    its own trainers and is not wired here.
    """
    var data = dataset.materialize_sparse_binned(max_bytes)
    var fields = dataset.row_fields()
    _check_labels(fields.label, data.n_rows)
    var booster = train_sparse(
        data,
        fields.label,
        objective,
        params,
        fields.weight,
        alpha,
        bagging,
        goss,
        fields.init_score,
    )
    return Model(dataset.mapper.copy(), booster^)


def train_external_sparse_multiclass(
    dataset: ExternalDataset,
    n_classes: Int,
    params: BoosterParams,
    max_bytes: Int,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MulticlassModel:
    """The softmax counterpart of `train_external_sparse`. CPU only, and
    `init_score` is refused for the reason it is refused everywhere else: one
    offset per row cannot say what each class starts from."""
    var data = dataset.materialize_sparse_binned(max_bytes)
    var fields = dataset.row_fields()
    _check_labels(fields.label, data.n_rows)
    if len(fields.init_score) != 0:
        raise Error(
            "init_score is not supported for multiclass training: one offset"
            " per row cannot say what each class starts from"
        )
    var booster = train_multiclass_sparse(
        data,
        _int_labels(fields.label, n_classes),
        n_classes,
        params,
        fields.weight,
        bagging,
        goss,
    )
    return MulticlassModel(dataset.mapper.copy(), booster^)


def external_groups(dataset: ExternalDataset) raises -> RankGroups:
    """Query boundaries for a ranking dataset, from the cached query ids.

    Groups are assembled from the whole column rather than per chunk, so a
    query whose rows straddle a chunk boundary is one query and not two.
    `ranking.groups_from_query_ids` still enforces contiguity, which is the
    property that actually matters: a query's rows have to be consecutive in
    the source's row order, and chunking never reorders rows.
    """
    var fields = dataset.row_fields()
    if len(fields.query_ids) == 0:
        raise Error(
            "a ranking dataset needs query ids: the source must carry one"
            " per row so the groups can be assembled across chunks"
        )
    return groups_from_query_ids(fields.query_ids)


def train_external_ranker(
    dataset: ExternalDataset,
    params: BoosterParams,
    max_bytes: Int,
    rank_params: RankerParams = RankerParams.default(),
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> Model:
    """Train a LambdaRank model from an external-memory dataset. CPU only,
    as `ranking.fit_ranker` is."""
    var data = dataset.materialize_binned(max_bytes)
    var fields = dataset.row_fields()
    _check_labels(fields.label, data.n_rows)
    if len(fields.init_score) != 0:
        raise Error(
            "init_score is not supported for ranking: lambdas are computed"
            " within a query and start from a score of 0"
        )
    var groups = external_groups(dataset)
    var booster = train_ranker(
        data,
        _relevance_labels(fields.label),
        groups,
        params,
        rank_params,
        fields.weight,
        bagging,
    )
    return Model(dataset.mapper.copy(), booster^)


def update_external(
    mut model: Model,
    dataset: ExternalDataset,
    params: BoosterParams,
    max_bytes: Int,
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> Int:
    """Append more rounds to `model` from an external-memory dataset.

    The binning check is `trainset.update_dataset`'s and for the same reason:
    a bin index has to mean one thing to the trees already in the model and
    to the ones about to be grown. The cache's mapper fingerprint is a cheap
    screen, and `BinMapper.matches` is the answer.

    Continued training runs on the CPU, as it does from a resident dataset.
    """
    if not model.mapper.matches(dataset.mapper):
        raise Error(
            "continued training needs the dataset the model was trained on:"
            " this one is binned differently"
        )
    var data = dataset.materialize_binned(max_bytes)
    var fields = dataset.row_fields()
    _check_labels(fields.label, data.n_rows)
    return train_more(
        model.booster,
        data,
        fields.label,
        params,
        fields.weight,
        alpha,
        bagging,
        goss,
        fields.init_score,
    )


def update_external_multiclass(
    mut model: MulticlassModel,
    dataset: ExternalDataset,
    params: BoosterParams,
    max_bytes: Int,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> Int:
    """The multiclass counterpart of `update_external`, with the same binning
    requirement and the same CPU-only rule."""
    if not model.mapper.matches(dataset.mapper):
        raise Error(
            "continued training needs the dataset the model was trained on:"
            " this one is binned differently"
        )
    var data = dataset.materialize_binned(max_bytes)
    var fields = dataset.row_fields()
    _check_labels(fields.label, data.n_rows)
    if len(fields.init_score) != 0:
        raise Error(
            "init_score is not supported for multiclass training: one offset"
            " per row cannot say what each class starts from"
        )
    return train_multiclass_more(
        model.booster,
        data,
        _int_labels(fields.label, model.booster.n_classes),
        params,
        fields.weight,
        bagging,
        goss,
    )
