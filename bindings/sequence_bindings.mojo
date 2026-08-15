"""Chunked `Dataset` construction: LightGBM's `Sequence` path.

`dataset_create` takes one column-major matrix. LightGBM's `lgb.Sequence`
protocol hands a `Dataset` its rows a batch at a time, so the caller never
holds the whole raw matrix in its own memory, and this module is that door:

    acc = dataset_chunks_begin(n_features)
    dataset_chunks_push(acc, x_addr, n_rows, label_addr, weight_addr,
                        init_score_addr)      # once per batch, column-major
    ds  = dataset_chunks_finish(acc, params)  # the same Dataset
                                              # dataset_create returns

Each pushed batch is copied out of the caller's buffer as it arrives, so
the caller may free or reuse the batch immediately. Rows are appended in
push order, which is the row identity `sequence.check_row_coverage`
enforces natively: batch `i` owns the rows after batch `i-1`. `finish`
assembles the column-major matrix the binner reads and hands it to the
`Dataset` constructor with the same `params` keys `dataset_create` reads
(`group_addr` with `n_groups`, `feature_names` with `n_names`,
`categorical_addr` with `categorical_len`, `max_bin`, `use_missing`,
`keep_raw`). The per-row columns come from the pushes when the batches
carry them; when no push carried one, `finish` reads it from `params`
(`label_addr`, `weight_addr`, `init_score_addr`, `n_rows` long) exactly as
`dataset_create` does, so a caller with the label in a separate array and
the features in batches has one path.

Batches are column-major within the batch (`x[r + f * n_rows]`), which is
what every converter in `python/mojotrees/_arrays.py`, `_arrow.py`, and
`_polars.py` produces for one batch, so no batch is transposed twice: it is
converted once and appended column by column here.

The peak here is the accumulated float64 matrix plus one batch, which is
what `Dataset` needs to bin exactly (global quantiles). A build whose raw
matrix does not fit goes through `external_memory.mojo`, whose cache holds
uint8 bins and whose entry point is native; this module is the Python
side's row-at-a-time ingestion, not that.
"""

from std.python import PythonObject

from binding_support import f64_buffer, flag, int_buffer_from_f64, str_sequence

from mojotrees.trainset import Dataset


struct ChunkAccumulator(Movable, Writable):
    """Rows pushed so far, one column-major block per feature.

    `columns[f]` is feature `f` over every pushed row, so `finish` is a
    concatenation of `n_features` lists rather than a transpose of the
    whole matrix. Optional columns are all-or-nothing: the first push says
    whether label, weight, and init_score are present, and every later push
    must agree, so a batch source that drops its label halfway is refused
    rather than zero-filled.
    """

    var n_features: Int
    var n_rows: Int
    var n_batches: Int
    var columns: List[List[Float64]]
    var label: List[Float64]
    var weight: List[Float64]
    var init_score: List[Float64]
    var has_label: Bool
    var has_weight: Bool
    var has_init_score: Bool

    def __init__(out self, n_features: Int) raises:
        if n_features < 1:
            raise Error("a dataset needs at least one feature")
        self.n_features = n_features
        self.n_rows = 0
        self.n_batches = 0
        self.columns = List[List[Float64]]()
        for _ in range(n_features):
            self.columns.append(List[Float64]())
        self.label = List[Float64]()
        self.weight = List[Float64]()
        self.init_score = List[Float64]()
        self.has_label = False
        self.has_weight = False
        self.has_init_score = False

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ChunkAccumulator(n_rows=",
            self.n_rows,
            ", n_features=",
            self.n_features,
            ", n_batches=",
            self.n_batches,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def push(
        mut self,
        x_addr: Int,
        n_rows: Int,
        label_addr: Int,
        weight_addr: Int,
        init_score_addr: Int,
    ) raises:
        if n_rows < 1:
            raise Error("a batch must hold at least one row")
        if self.n_batches == 0:
            self.has_label = label_addr != 0
            self.has_weight = weight_addr != 0
            self.has_init_score = init_score_addr != 0
        else:
            if self.has_label != (label_addr != 0):
                raise Error(
                    "every batch must carry a label or none may; batch ",
                    self.n_batches,
                    " disagrees with the first",
                )
            if self.has_weight != (weight_addr != 0):
                raise Error(
                    "every batch must carry weights or none may; batch ",
                    self.n_batches,
                    " disagrees with the first",
                )
            if self.has_init_score != (init_score_addr != 0):
                raise Error(
                    "every batch must carry init_score or none may; batch ",
                    self.n_batches,
                    " disagrees with the first",
                )
        var block = f64_buffer(x_addr, n_rows * self.n_features)
        for f in range(self.n_features):
            ref column = self.columns[f]
            column.reserve(self.n_rows + n_rows)
            for r in range(n_rows):
                column.append(block[f * n_rows + r])
        if self.has_label:
            self.label.extend(f64_buffer(label_addr, n_rows))
        if self.has_weight:
            self.weight.extend(f64_buffer(weight_addr, n_rows))
        if self.has_init_score:
            self.init_score.extend(f64_buffer(init_score_addr, n_rows))
        self.n_rows += n_rows
        self.n_batches += 1

    def matrix(self) raises -> List[Float64]:
        """The pushed rows as one column-major matrix,
        `values[f * n_rows + r]`, which is what `Dataset` bins."""
        if self.n_rows < 1:
            raise Error("no rows were pushed")
        var out = List[Float64](capacity=self.n_rows * self.n_features)
        for f in range(self.n_features):
            out.extend(self.columns[f].copy())
        return out^


def _group_counts(params: PythonObject) raises -> List[Int]:
    if Int(py=params["group_addr"]) == 0:
        return List[Int]()
    return int_buffer_from_f64(
        Int(py=params["group_addr"]), Int(py=params["n_groups"])
    )


def _categorical(params: PythonObject) raises -> List[Int]:
    var addr = Int(py=params["categorical_addr"])
    if addr == 0:
        return List[Int]()
    return int_buffer_from_f64(addr, Int(py=params["categorical_len"]))


def _feature_names(params: PythonObject) raises -> List[String]:
    var n_names = Int(py=params["n_names"])
    if n_names == 0:
        return List[String]()
    return str_sequence(params["feature_names"], n_names)


def dataset_chunks_begin(n_features: PythonObject) raises -> PythonObject:
    """Start accumulating batches of `n_features` columns."""
    var acc = ChunkAccumulator(Int(py=n_features))
    return PythonObject(alloc=acc^)


def dataset_chunks_push(
    acc: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    label_addr: PythonObject,
    weight_addr: PythonObject,
    init_score_addr: PythonObject,
) raises -> PythonObject:
    """Append one column-major float64 batch (`x[f * n_rows + r]`) and its
    optional per-row columns (address 0 for absent). Returns the number of
    rows accumulated so far."""
    var a = acc.downcast_value_ptr[ChunkAccumulator]()
    a[].push(
        Int(py=x_addr),
        Int(py=n_rows),
        Int(py=label_addr),
        Int(py=weight_addr),
        Int(py=init_score_addr),
    )
    return PythonObject(a[].n_rows)


def dataset_chunks_num_data(acc: PythonObject) raises -> PythonObject:
    var a = acc.downcast_value_ptr[ChunkAccumulator]()
    return PythonObject(a[].n_rows)


def dataset_chunks_finish(
    acc: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Bin the accumulated rows into a `Dataset`, exactly as
    `dataset_create` would bin the same rows given at once. The accumulator
    is not reusable afterwards; drop it."""
    var a = acc.downcast_value_ptr[ChunkAccumulator]()
    var features = a[].matrix()
    var n_rows = a[].n_rows
    var label = a[].label.copy()
    if not a[].has_label and Int(py=params["label_addr"]) != 0:
        label = f64_buffer(Int(py=params["label_addr"]), n_rows)
    var weight = a[].weight.copy()
    if not a[].has_weight and Int(py=params["weight_addr"]) != 0:
        weight = f64_buffer(Int(py=params["weight_addr"]), n_rows)
    var init_score = a[].init_score.copy()
    if not a[].has_init_score and Int(py=params["init_score_addr"]) != 0:
        init_score = f64_buffer(Int(py=params["init_score_addr"]), n_rows)
    var dataset = Dataset(
        features,
        n_rows,
        a[].n_features,
        label^,
        weight^,
        _group_counts(params),
        init_score^,
        _feature_names(params),
        _categorical(params),
        Int(py=params["max_bin"]),
        flag(params["use_missing"], "use_missing"),
        flag(params["keep_raw"], "keep_raw"),
    )
    return PythonObject(alloc=dataset^)
