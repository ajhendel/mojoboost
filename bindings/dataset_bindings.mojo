"""`Dataset` construction beyond the dense case, and its prepared data.

A `Dataset` handle (see `src/mojotrees/trainset.mojo`) owns a binned
feature matrix, dense or sparse, the fitted `BinMapper` that produced it,
and the columns that describe its rows. `bindings/_mojotrees.mojo` builds
the dense case (`dataset_create`) and exposes its shape
(`dataset_num_data`, `dataset_num_feature`, `dataset_num_bin`); this
module carries the three constructors that shape has no room for and the
reads that answer from the constructed object.

Why the reads matter: `python/mojotrees/basic.py` keeps its own copy of
the label, the weights, the groups, the init scores, the names, and the
categorical declaration, and answers `get_field` from those. That is
correct for a dataset built in this process and cannot be right for one
that was not, and it says nothing about what the *binning* did with them.

Why the constructors matter: three real capabilities land in
`trainset.mojo` and are unreachable from Python without them. A sparse
dataset that stays sparse; `reference=`, whose absence today means a
validation set is silently binned over its own rows and then scored as if
it were not; and `subset`, which is what would let cross-validation build
folds natively. `handoffs/connect_12_dataset_cv.md` section 6.2 asked for
exactly these, and section 6.3 has the Python half.

No binning rule, no validation rule, and no fold policy is written here.
Every constructor forwards to the `Dataset` static of the same name, and
every buffer address travels inward: a column leaves as a Python list, or
is written into a buffer the caller preallocated and sized.
"""

from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from binding_support import (
    csc_from_params,
    f64_buffer,
    f64_view,
    flag,
    index_within,
    int_buffer,
    int_buffer_from_f64,
    py_dict,
    py_f64_list,
    py_int_list,
    py_str_list,
    str_sequence,
    write_f64_buffer,
)

from mojotrees.efb import feature_bin_count
from mojotrees.trainset import Dataset


def _field_values(d: Dataset, field: String) raises -> List[Float64]:
    """One of the dataset's optional columns as float64.

    `group` holds per-query row counts, which are integers; they cross as
    float64 like every other integer column at this boundary (see
    `_int_list_from_f64` in `_mojotrees.mojo`), so one accessor serves all
    four fields and a caller does not need a second buffer protocol for
    one of them.

    An absent column is an empty list, which is what the `Dataset`
    constructor stores for one that was never given. Raises for a field
    name a `Dataset` does not hold, so a typo is not read as absence.
    """
    if field == "label":
        return d.label.copy()
    if field == "weight":
        return d.weight.copy()
    if field == "init_score":
        return d.init_score.copy()
    if field == "group":
        var out = List[Float64](capacity=len(d.group))
        for i in range(len(d.group)):
            out.append(Float64(d.group[i]))
        return out^
    raise Error(
        "unknown dataset field '",
        field,
        "'; a Dataset holds 'label', 'weight', 'init_score', and 'group'",
    )


def _optional_column(
    params: PythonObject, key: String, n_rows: Int
) raises -> List[Float64]:
    """One optional per-row column from the params mapping, empty when its
    address is 0. The same convention `dataset_create` reads."""
    var addr = Int(py=params[key])
    if addr == 0:
        return List[Float64]()
    return f64_buffer(addr, n_rows)


def _group_counts(params: PythonObject) raises -> List[Int]:
    """Per-query row counts (LightGBM's `group`), empty when absent. They
    travel as float64 like every other column at this boundary."""
    if Int(py=params["group_addr"]) == 0:
        return List[Int]()
    return int_buffer_from_f64(
        Int(py=params["group_addr"]), Int(py=params["n_groups"])
    )


def _categorical(params: PythonObject) raises -> List[Int]:
    """Declared categorical feature indices, empty when absent. The binner
    validates them again, so the rules in categorical.mojo stay the only
    ones."""
    var addr = Int(py=params["categorical_addr"])
    if addr == 0:
        return List[Int]()
    return int_buffer_from_f64(addr, Int(py=params["categorical_len"]))


def _feature_names(params: PythonObject) raises -> List[String]:
    var n_names = Int(py=params["n_names"])
    if n_names == 0:
        return List[String]()
    return str_sequence(params["feature_names"], n_names)


def dataset_create_csc(params: PythonObject) raises -> PythonObject:
    """Bin a sparse CSC matrix into a `Dataset` that stays sparse.

    Reads the six sparse keys `_arrays.SparseBuffers.params()` emits, plus
    the same optional columns and binning configuration `dataset_create`
    reads: `label_addr`, `weight_addr`, `init_score_addr`, `group_addr`
    with `n_groups`, `feature_names` with `n_names`, `categorical_addr`
    with `categorical_len`, `max_bin`, `use_missing`, and `keep_raw`.

    The binned matrix is a `SparseBinnedMatrix`, so training never
    allocates `n_rows * n_features` of anything: that is the whole reason
    a sparse dataset exists, and it is why this is a separate constructor
    rather than a densifying branch of the dense one.
    """
    var n_rows = Int(py=params["n_rows"])
    var csc = csc_from_params(params)
    var label = _optional_column(params, "label_addr", n_rows)
    var weight = _optional_column(params, "weight_addr", n_rows)
    var group = _group_counts(params)
    var init_score = _optional_column(params, "init_score_addr", n_rows)
    var names = _feature_names(params)
    var categorical = _categorical(params)
    var max_bin = Int(py=params["max_bin"])
    var use_missing = flag(params["use_missing"], "use_missing")
    var keep_raw = flag(params["keep_raw"], "keep_raw")
    # Sparse binning is unbounded work over the caller's matrix and touches
    # no Python object once the six sparse buffers have been copied above,
    # so the interpreter lock goes back to the rest of the process for it.
    # See the rule at the top of `bindings/_mojotrees.mojo`.
    var dataset: Dataset
    with GILReleased(Python()):
        dataset = Dataset.from_csc(
            csc^,
            label^,
            weight^,
            group^,
            init_score^,
            names^,
            categorical^,
            max_bin,
            use_missing,
            keep_raw,
        )
    return PythonObject(alloc=dataset^)


def dataset_create_reference(
    reference: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Bin a dense matrix by another dataset's mapper: LightGBM's
    `reference=`.

    The bin indices in the result mean what they mean in the reference,
    which is what a validation set needs and what continued training
    requires. The feature count, the binning parameters, the feature
    names, and the categorical declaration are the reference's, because
    they describe the columns rather than the rows, and are deliberately
    not read from `params`; only `label_addr`, `weight_addr`,
    `init_score_addr`, `group_addr`/`n_groups`, and `keep_raw` are.

    This is the wrong constructor for a cross-validation fold, whose
    held-out rows must not have shaped the binning. `dataset_subset` with
    `shared_binning=0` is that one.

    The matrix is borrowed rather than copied, on the contract
    `binding_support.f64_view` states: `basic.Dataset` holds the array it took
    the address of for the whole synchronous call, and the reference's mapper
    only reads it. `keep_raw` still copies, because that is the caller asking
    the dataset to outlive their buffer.
    """
    var ref_dataset = reference.downcast_value_ptr[Dataset]()
    var nr = Int(py=n_rows)
    var nf = ref_dataset[].num_feature()
    var features = f64_view(Int(py=x_addr), nr * nf)
    var label = _optional_column(params, "label_addr", nr)
    var weight = _optional_column(params, "weight_addr", nr)
    var group = _group_counts(params)
    var init_score = _optional_column(params, "init_score_addr", nr)
    var keep_raw = flag(params["keep_raw"], "keep_raw")
    # Released on `dataset_create_csc`'s terms. `ref_dataset[]` reads Mojo
    # memory inside a Python object the caller's frame keeps alive, which
    # touches no refcount and is safe with the lock down.
    var dataset: Dataset
    with GILReleased(Python()):
        dataset = Dataset.from_reference_dense(
            ref_dataset[],
            features,
            nr,
            label^,
            weight^,
            group^,
            init_score^,
            keep_raw,
        )
    return PythonObject(alloc=dataset^)


def dataset_subset(
    dataset: PythonObject,
    rows_addr: PythonObject,
    n_rows: PythonObject,
    shared_binning: PythonObject,
) raises -> PythonObject:
    """The named rows as their own dataset.

    `rows_addr` is an int64 buffer of `n_rows` strictly ascending row
    indices. `shared_binning` picks between the two constructors, and the
    difference is not a detail:

    - 0: the subset is **binned over its own rows**, so the rows left out
      had no say in the edges. This is what a cross-validation fold or a
      held-out split needs.
    - 1: the subset is binned by this dataset's mapper, LightGBM's
      `Dataset.subset`, so its bin indices mean what the whole's mean and
      a model trained on the whole can score it.

    Both need the source to have been built with `keep_raw=1`: bins cannot
    be refitted from bins, and a dataset that dropped its raw matrix says
    so rather than returning an empty one. The subset keeps its own raw
    matrix when it is binned over its own rows, so a fold can be subset
    again.
    """
    var d = dataset.downcast_value_ptr[Dataset]()
    var rows = int_buffer(Int(py=rows_addr), Int(py=n_rows))
    # Both arms re-bin, which is the same unbounded work `dataset_create`
    # does, so both release. `rows` was copied out of the caller's buffer
    # above.
    if flag(shared_binning, "shared_binning"):
        var shared: Dataset
        with GILReleased(Python()):
            shared = d[].subset_shared_binning(rows)
        return PythonObject(alloc=shared^)
    var own: Dataset
    with GILReleased(Python()):
        own = d[].subset(rows)
    return PythonObject(alloc=own^)


def dataset_metadata(dataset: PythonObject) raises -> PythonObject:
    """Everything about the constructed dataset that is a scalar, in one
    call.

    Keys: `num_data`, `num_feature`, `num_bin` (the effective `max_bin`
    the binning reserved), `max_bin` (the one that was requested),
    `use_missing`, `n_groups`, `n_categorical`, `has_names`, the lengths
    of the optional columns (`n_label`, `n_weight`, `n_init_score`), and
    the three facts about how the dataset is stored: `is_sparse`, `nnz`
    (stored entries in the binned matrix: every cell when dense), and
    `has_raw` (whether the raw input was retained, which is what
    `dataset_subset` needs).

    One call rather than fourteen, because the Python `Dataset` builds its
    repr and its `get_field` answers from all of them at once. This is
    also the answer to §6.2(e) of `handoffs/connect_12_dataset_cv.md`,
    which asked for `dataset_is_sparse`, `dataset_nnz`, and
    `dataset_has_raw` as three entry points: they are three keys here
    instead, so there is one way to ask rather than two.
    """
    var d = dataset.downcast_value_ptr[Dataset]()
    var out = py_dict()
    out["num_data"] = PythonObject(d[].num_data())
    out["num_feature"] = PythonObject(d[].num_feature())
    out["num_bin"] = PythonObject(d[].num_bin())
    out["is_sparse"] = PythonObject(d[].is_sparse)
    out["nnz"] = PythonObject(d[].nnz())
    out["has_raw"] = PythonObject(d[].has_raw())
    out["max_bin"] = PythonObject(d[].max_bin)
    out["use_missing"] = PythonObject(d[].use_missing)
    out["n_groups"] = PythonObject(len(d[].group))
    out["n_categorical"] = PythonObject(len(d[].categorical_features))
    out["has_names"] = PythonObject(len(d[].feature_names) != 0)
    out["n_label"] = PythonObject(len(d[].label))
    out["n_weight"] = PythonObject(len(d[].weight))
    out["n_init_score"] = PythonObject(len(d[].init_score))
    return out^


def dataset_feature_names(dataset: PythonObject) raises -> PythonObject:
    """The names the dataset carries, or an empty list when it carries
    none.

    Empty rather than `Column_0`, `Column_1`, ...: the default naming is
    the caller's convention (`Dataset.feature_name` in basic.py, and
    `feature_names_or_default` in model_dump.mojo for a dump), and a
    dataset that was given no names should be able to say so.
    """
    var d = dataset.downcast_value_ptr[Dataset]()
    return py_str_list(d[].feature_names)


def dataset_categorical_features(dataset: PythonObject) raises -> PythonObject:
    """The feature indices the dataset was told to treat as categorical,
    in the order they were declared."""
    var d = dataset.downcast_value_ptr[Dataset]()
    return py_int_list(d[].categorical_features)


def dataset_field(
    dataset: PythonObject, field: PythonObject
) raises -> PythonObject:
    """One optional column as a Python list of floats: `label`, `weight`,
    `init_score`, or `group`. Empty when the dataset has none."""
    var d = dataset.downcast_value_ptr[Dataset]()
    return py_f64_list(_field_values(d[], String(py=field)))


def dataset_field_length(
    dataset: PythonObject, field: PythonObject
) raises -> PythonObject:
    """How many values `dataset_copy_field` would write for this field, so
    a caller can size a buffer before asking for one."""
    var d = dataset.downcast_value_ptr[Dataset]()
    return PythonObject(len(_field_values(d[], String(py=field))))


def dataset_copy_field(
    dataset: PythonObject,
    field: PythonObject,
    out_addr: PythonObject,
    capacity: PythonObject,
) raises -> PythonObject:
    """Write one optional column into a float64 buffer the caller
    preallocated, and return how many values were written.

    `capacity` is the buffer's length in values, and a column longer than
    it is refused rather than truncated: a short write here would be a
    memory error in the caller's process. This is the same preallocated
    output convention `feature_importance` uses.
    """
    var d = dataset.downcast_value_ptr[Dataset]()
    var values = _field_values(d[], String(py=field))
    write_f64_buffer(values, Int(py=out_addr), Int(py=capacity))
    return PythonObject(len(values))


def dataset_feature_num_bin(
    dataset: PythonObject, feature: PythonObject
) raises -> PythonObject:
    """How many bins one feature can take under the fitted binning.

    Read from the mapper rather than counted off the data, because a bin
    no training row happened to use is still a bin an unseen row can land
    in. `feature_bin_count` in efb.mojo is that read; this does not repeat
    it.
    """
    var d = dataset.downcast_value_ptr[Dataset]()
    var f = index_within(feature, d[].num_feature(), "feature")
    return PythonObject(feature_bin_count(d[].mapper, f))


def dataset_bin_upper_bounds(
    dataset: PythonObject, feature: PythonObject
) raises -> PythonObject:
    """The fitted bin edges of one numerical feature, ascending.

    A feature with `k` edges uses bins `0..k`: a value `v` lands in the
    first bin whose edge satisfies `v <= edge`, and a value above every
    edge lands in the last. That is the layout `BinMapper` documents and
    this reads; it does not re-derive it.

    Raises for a categorical feature, which carries no edges at all: its
    bins are category codes, and `dataset_categorical_features` says which
    features those are.
    """
    var d = dataset.downcast_value_ptr[Dataset]()
    var f = index_within(feature, d[].num_feature(), "feature")
    ref mapper = d[].mapper
    if mapper.cats.is_cat(f):
        raise Error(
            "feature ",
            f,
            " is categorical, so it has no bin edges; its bins are category"
            " codes",
        )
    var start = mapper.edge_offsets[f]
    var end = mapper.edge_offsets[f + 1]
    var out = List[Float64](capacity=end - start)
    for i in range(start, end):
        out.append(mapper.edges[i])
    return py_f64_list(out)


def dataset_missing_bins(dataset: PythonObject) raises -> PythonObject:
    """The bin each feature reserves for missing values, or -1 where none
    is reserved. One entry per feature, in feature order."""
    var d = dataset.downcast_value_ptr[Dataset]()
    return py_int_list(d[].mapper.missing_bin)
